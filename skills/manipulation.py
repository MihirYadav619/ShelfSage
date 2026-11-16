"""Table manipulation skills (Phase 5): drag a book to the table edge, then pick it up.

Book poses come from simulator ground truth for now; the YOLO detector replaces that later
(the skills only need the book's position and yaw).

Drag to the edge ("pull, then grasp"), for a book lying inside the table. Two ways to pull:

  method="base" (default): the arm only presses, the base pulls (whole-body). The arm presses
  0.50 m ahead of its mount (the middle of its reach) and stays still while the base backs away
  slowly (4 cm/s) by the whole distance, so one stroke does it, however far in the book lies.
  A short second stroke corrects what the fingertips slipped.

  method="arm": the base stands still, the arm pulls, in strokes:
    1. park in line with the book, close enough that the press point is ~0.58 m ahead of the arm
       mount (3-50 cm from the edge; the far books need the base right at the table)
    2. closed fingertips above the top cover, 4 cm in from the spine, gripper tilted back towards
       the robot: -30 deg if the IK reaches both ends of the stroke, else -15 or 0 deg (the
       tilted hand reaches close to the base, the vertical one far out)
    3. base station keeping on; guarded move down until the fingers press with 10 N
    4. pull straight towards the robot (at most 25 cm); the book follows because the rubber pads
       (mu 1.5) grip harder than the paper slides on the wood (mu 0.4): 1.5 F > 0.4 (F + m g)
    5. lift off, station keeping off, measure the book again; repeat until the spine overhangs
       the edge by 3 cm (+-1 cm). A stroke that hits the workspace boundary aborts safely; the
       next one re-parks and starts from a fresh IK solution.

Pick from the edge (as in sim/demo_pick_place.py): normal parking, pre-grasp, straight in around
the overhanging spine, close the fingers (friction grasp; robot.grasp_mode "weld" adds the attach
weld), lift, back away.
"""
import json
import math
from pathlib import Path

import numpy as np

from skills.result import SkillResult

ROOT = Path(__file__).resolve().parent.parent
META = json.loads((ROOT / "scene/library_meta.json").read_text())
BD = META["book_dims"]
UP = np.array([0.0, 0.0, 1.0])

STROKE_START = 0.58        # arm mount -> press point at the start of a stroke (m, horizontal)
MIN_CLEARANCE = 0.03       # closest the chassis front gets to the table edge (its top is 10 cm
                           # below the table top, so it may not go under it)
PRESS_IN = 0.04            # press point: this far in from the spine (between the head/tail ribs)
PRESS_FORCE = 10.0         # N
PRESS_TILTS = [math.radians(a) for a in (-30, -15, 0)]   # gripper leans back towards the robot
MAX_STROKE = 0.25
SLIP = 0.005               # the fingertips creep ~5 mm per stroke: pull that much further
MAX_STROKES = 5
BASE_PRESS_AT = 0.50       # method "base": arm mount -> press point (m, horizontal)
BASE_PULL_SPEED = 0.04     # m/s, the base backing away with the book
MAX_BASE_STROKE = 0.60     # longest single pull of the base
BASE_SLIP = 0.04           # a long base pull leaves the book ~4 % short (fingertip creep + the base
                           # stopping ~5 mm early under the load): pull that much further
OVERHANG_TOL = 0.01
W_OPEN = 0.06
LIFT, BACK = 0.10, 0.16
# carry pose for driving (book standing above the front of the chassis): 0.60 m ahead of the base
# centre, 1.20 m up. The old 0.45 m / 1.15 m put joint 5 at its limit, and moving on to the shelf
# then swung the wrist by 5 rad, which turned the book in a friction grasp. Here every joint keeps
# >= 0.34 rad from its limits and the shelf pre-place is 0.5 rad (row B) / 2 rad (row A) away.
CARRY_FWD, CARRY_Z = 0.60, 1.20


def carry_target(robot):
    """Carry pose: TCP position, approach (forward) and finger closing (left) axes, world frame."""
    x, y, yaw = robot.get_pose()
    f = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    return np.array([x, y, 0.0]) + CARRY_FWD * f + np.array([0, 0, CARRY_Z]), f, np.array([-f[1], f[0], 0.0])



# ---------------------------------------------------------------------------- geometry helpers
def table_frame(table):
    """(fwd, lateral, edge) of a table seen from its parking side: fwd = robot heading at the
    parking poses, lateral = robot's left, edge = distance of the robot-side edge along fwd."""
    t = META["tables"][table]
    yaw = t["parking"][0][2]
    fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    left = np.array([-fwd[1], fwd[0], 0.0])
    half_along = abs(fwd[0]) * t["half_size"][0] + abs(fwd[1]) * t["half_size"][1]
    edge = float(np.array([*t["pos"], 0.0]) @ fwd) - half_along
    return fwd, left, edge, yaw


def table_parking(robot, table, lateral, clearance):
    """Base pose facing the table, `clearance` m between the chassis front and the edge, with the
    base centre at `lateral` (world coordinate along the edge)."""
    fwd, left, edge, yaw = table_frame(table)
    s = edge - clearance - robot.meta["chassis_half"][0]
    p = s * fwd[:2] + lateral * left[:2]
    return float(p[0]), float(p[1]), yaw


def book_state(robot, book, table=None):
    """Pose of a lying book: centre, spine direction (unit, world), yaw, flatness. From the wrist
    camera if a Perception is attached (robot.perception, skills/perception.py; the arm moves to
    look, None if the book is not seen), else ground truth."""
    if getattr(robot, "perception", None) is not None and table is not None:
        return robot.perception.table_book(robot, book, table)
    b = robot.d.body(f"book_{book}")
    R = b.xmat.reshape(3, 3)
    return b.xpos.copy(), -R[:, 1], math.atan2(R[1, 0], R[0, 0]), float(R[2, 2])


def spine_overhang(robot, book, table, c=None):
    """How far the spine sticks out past the robot-side edge (m; negative = inside the table).
    c: the book centre if already measured."""
    fwd, _, edge, _ = table_frame(table)
    if c is None:
        c = book_state(robot, book)[0]
    return edge - (c @ fwd - BD["width"] / 2)


def footprint(robot, body, fwd, left):
    """(lateral min, lateral max, depth min, depth max) of a lying book's corners in the table
    frame (depth along fwd, lateral along left)."""
    b = robot.d.body(body)
    R = b.xmat.reshape(3, 3)
    pts = [b.xpos + sx * R[:, 0] * BD["length"] / 2 + sy * R[:, 1] * BD["width"] / 2
           for sx in (-1, 1) for sy in (-1, 1)]
    lat, dep = [p @ left for p in pts], [p @ fwd for p in pts]
    return min(lat), max(lat), min(dep), max(dep)


def drag_blockers(robot, book, table, margin=0.01):
    """Other books lying in the lane the book sweeps on its way to the edge."""
    fwd, left, edge, _ = table_frame(table)
    l0, l1, d0, _ = footprint(robot, f"book_{book}", fwd, left)
    top = META["tables"][table]["top_z"]
    out = []
    for name, info in META["books"].items():
        if name == book:
            continue
        b = robot.d.body(info["body"])
        if abs(b.xpos[2] - top - BD["rib_height"] - BD["thickness"] / 2) > 0.02:
            continue                                    # not lying on a table
        m0, m1, e0, e1 = footprint(robot, info["body"], fwd, left)
        if m1 > l0 - margin and m0 < l1 + margin and e1 > edge - 0.1 and e0 < d0:
            out.append(info["title"])
    return out


# ---------------------------------------------------------------------------- skills
def drag_to_edge(robot, book, table, say=print, log=None, method="base"):
    """Pull a book lying inside `table` towards the robot until its spine overhangs the edge by
    the table overhang (3 cm). method "base": the arm presses, the base pulls; "arm": the base
    stands, the arm pulls in strokes of at most 25 cm. Leaves the robot near the table, arm up."""
    if method not in ("base", "arm"):
        raise ValueError(f"unknown drag method {method}")
    log = [] if log is None else log
    fwd, left, edge, _ = table_frame(table)
    st = book_state(robot, book, table)
    if st is None:
        return SkillResult(False, "the camera does not see the book on the table")
    c, spine, _, flat = st
    if flat < 0.98 or spine @ fwd > -0.9:
        return SkillResult(False, "book is not lying flat with its spine towards the robot")
    target = BD["table_overhang"]
    if spine_overhang(robot, book, table, c) > target - OVERHANG_TOL:
        return SkillResult(True, "already at the edge", {"strokes": 0})
    blockers = drag_blockers(robot, book, table)
    if blockers:
        log.append((f"lane to the edge is free (blocked by {', '.join(blockers)})", False))
        return SkillResult(False, f"blocked by {', '.join(blockers)}", {"blockers": blockers})
    lateral = float(c @ left)
    mount_behind_front = robot.meta["chassis_half"][0] - robot.meta["mount_x"]
    clearance_now = None

    strokes = 0
    for strokes in range(1, MAX_STROKES + 1):
        st = book_state(robot, book, table) if strokes > 1 else st
        if st is None:
            return SkillResult(False, "lost sight of the book while dragging it", {"strokes": strokes})
        c, spine, _, _ = st
        remaining = target - spine_overhang(robot, book, table, c)
        if remaining < OVERHANG_TOL:
            strokes -= 1
            break
        press = c + spine * (BD["width"] / 2 - PRESS_IN)
        press[2] = c[2] + BD["thickness"] / 2
        by_base = method == "base"
        stroke = (min(remaining * (1 + BASE_SLIP) + SLIP, MAX_BASE_STROKE) if by_base
                  else min(remaining + SLIP, MAX_STROKE))

        # 1. park so that the press point is STROKE_START (BASE_PRESS_AT) ahead of the arm mount
        clearance = float(np.clip((BASE_PRESS_AT if by_base else STROKE_START) - (press @ fwd - edge)
                                  - mount_behind_front,
                                  MIN_CLEARANCE, META["robot"]["park_clearance"]["table"]))
        if clearance_now is None or abs(clearance - clearance_now) > 0.01:
            say(f"park {clearance * 100:.0f} cm from Table {table}, in line with the book")
            robot.stow()
            res = robot.go_to(*table_parking(robot, table, lateral, clearance), tol_xy=0.01, tol_yaw=0.01)
            log.append((f"stroke {strokes}: park {clearance * 100:.0f} cm from the table", res.success))
            if not res.success:
                return SkillResult(False, f"drive failed: {res.message} {res.collisions}")
            clearance_now = clearance
            robot.home(2.0)
            robot.set_gripper_width(0.0, 0.5)
            if getattr(robot, "perception", None) is not None:
                # look again from close by: from the far parking a mid-table book is only partly
                # in view (centre off by a few cm), from here the camera gets above it
                st = book_state(robot, book, table)
                if st is None:
                    return SkillResult(False, "lost sight of the book", {"strokes": strokes})
                c, spine, _, _ = st
                remaining = target - spine_overhang(robot, book, table, c)
                press = c + spine * (BD["width"] / 2 - PRESS_IN)
                press[2] = c[2] + BD["thickness"] / 2
                stroke = (min(remaining * (1 + BASE_SLIP) + SLIP, MAX_BASE_STROKE) if by_base
                          else min(remaining + SLIP, MAX_STROKE))

        # 2. tilt: the first one whose IK reaches both ends of the stroke (base: the arm does not
        #    move along the stroke, so above the book and pressed)
        for tilt in PRESS_TILTS:
            z_ax = math.sin(tilt) * fwd - math.cos(tilt) * UP
            end = press if by_base else press - stroke * fwd
            ends = [robot.ik(p, z_axis=z_ax, y_axis=left, n_random=4, allow=(f"book_{book}",))
                    for p in (press + 0.04 * UP, end)]
            if all(e is not None and e.success for e in ends):
                break
        else:
            log.append((f"stroke {strokes}: reachable tilt", False))
            return SkillResult(False, "stroke not reachable at any tilt", {"strokes": strokes})
        say(f"stroke {strokes}: press on the cover (tilt {math.degrees(tilt):.0f} deg), "
            f"{'base' if by_base else 'arm'} pulls {stroke * 100:.0f} cm")
        r = robot.move_to(press + 0.04 * UP, z_axis=z_ax, y_axis=left, duration=2.0)
        if r is None or not r.success:
            log.append((f"stroke {strokes}: above the book", False))
            return SkillResult(False, "cannot reach above the book", {"strokes": strokes})

        # 3-5. press, pull, lift
        robot.hold_base(True)
        g = robot.guarded_move(-UP, PRESS_FORCE, z_axis=z_ax, y_axis=left, max_dist=0.06,
                               body=f"book_{book}")
        log.append((f"stroke {strokes}: press {g.force:.0f} N", g.success))
        if not g.success:
            robot.hold_base(False)
            return SkillResult(False, "no contact with the book", {"strokes": strokes})
        if by_base:
            # the arm keeps its joint targets, so the pressing fingertips move with the base
            robot.hold_base(False)
            x, y, yaw = robot.get_pose()
            goal = np.array([x, y]) - stroke * fwd[:2]
            res = robot.go_to(goal[0], goal[1], yaw, tol_xy=0.005, tol_yaw=0.005,
                              v_max=BASE_PULL_SPEED, a_max=0.1, timeout=60.0)
            hits = [h for h in res.collisions if f"book_{book}" not in h and
                    "left_finger <-> panda_right_finger" not in h]
            log.append((f"stroke {strokes}: base pulled {stroke * 100:.0f} cm "
                        f"(error {res.final_error_xy * 1000:.0f} mm)", not hits and res.message != "timeout"))
            if hits:
                return SkillResult(False, f"collision while pulling: {hits}", {"strokes": strokes})
            clearance_now += stroke
            robot.hold_base(True)
        else:
            p = robot.fk_world()[:3, 3]
            lin = robot.move_linear(p - stroke * fwd, z_axis=z_ax, y_axis=left, speed=0.04)
            if lin.aborted:
                say("  stopped at the edge of the workspace, will press again")
        robot.move_linear(robot.fk_world()[:3, 3] + 0.04 * UP, z_axis=z_ax, y_axis=left, speed=0.04)
        robot.hold_base(False)

    robot.set_gripper_width(W_OPEN, 0.3)   # closed fingers touch each other (a self-contact)
    st = book_state(robot, book, table) if strokes > 0 else st
    if st is None:
        return SkillResult(False, "lost sight of the book after dragging it", {"strokes": strokes})
    c, _, yaw, flat = st
    over = spine_overhang(robot, book, table, c)
    ok = abs(over - target) < OVERHANG_TOL and flat > 0.98
    log.append((f"dragged to the edge: overhang {over * 100:.1f} cm in {strokes} strokes, "
                f"yaw {math.degrees(yaw):.1f} deg", ok))
    return SkillResult(ok, "at the edge" if ok else "edge not reached",
                       {"overhang": over, "strokes": strokes, "yaw": yaw})


def pick_from_table_edge(robot, book, table, say=print, log=None):
    """Grasp a book whose spine overhangs the robot-side edge: drive to the normal parking in
    line with it, fingers around the spine, close (robot.grasp), lift 10 cm, back away 16 cm."""
    log = [] if log is None else log
    fwd, left, _, _ = table_frame(table)
    st = book_state(robot, book, table)
    if st is None:
        return SkillResult(False, "the camera does not see the book on the table")
    c, spine, _, _ = st
    over = spine_overhang(robot, book, table, c)
    if over < BD["table_overhang"] - OVERHANG_TOL:
        return SkillResult(False, f"spine only {over * 100:.1f} cm past the edge, drag it first")

    def check(name, res):
        ok = bool(res) if isinstance(res, (bool, np.bool_)) else (
            res is not None and getattr(res, "success", True))
        log.append((name, ok))
        return ok

    say(f"back off to the normal parking at Table {table}")
    robot.stow()
    res = robot.go_to(*table_parking(robot, table, float(c @ left), META["robot"]["park_clearance"]["table"]),
                      tol_xy=0.01, tol_yaw=0.01)
    if not check("drive to the grasp parking", res):
        return SkillResult(False, f"drive failed: {res.message} {res.collisions}")
    if getattr(robot, "perception", None) is not None:
        # look again from here: in line with the book it is in the middle of the view (from the
        # table's middle parking a book on a side spot was cut by the image border, its centre
        # 6 cm off along its length, and the grasp held it so far off-centre that it hit the shelf)
        st = book_state(robot, book, table)
        if st is None:
            return SkillResult(False, "lost sight of the book")
        c, spine, _, _ = st
        park = table_parking(robot, table, float(c @ left), META["robot"]["park_clearance"]["table"])
        if math.hypot(park[0] - robot.get_pose()[0], park[1] - robot.get_pose()[1]) > 0.01:
            # the first estimate put the base a few cm off the book's line; off-line, the IK chose
            # a twisted posture for the pre-grasp and the move there knocked the book off
            robot.home(1.5)
            res = robot.go_to(*park, tol_xy=0.01, tol_yaw=0.01)
            if not check("re-park in line with the book", res):
                return SkillResult(False, f"drive failed: {res.message} {res.collisions}")

    g = c + spine * (BD["width"] / 2 - BD["grip_in"]["table"])
    say("pre-grasp, straight in around the spine, close the fingers")
    robot.home(2.0)
    robot.set_gripper_width(W_OPEN, 0.3)
    if not check("pre-grasp", robot.move_to(g - 0.10 * fwd, z_axis=fwd, y_axis=UP, duration=2.5)):
        return SkillResult(False, "pre-grasp not reachable")
    if not check("approach", robot.move_linear(g, z_axis=fwd, y_axis=UP, speed=0.04)):
        return SkillResult(False, "approach failed")
    width, force = robot.grasp(book)
    if not check(f"grasp: width {width * 100:.1f} cm, squeeze {force:.0f} N", 0.03 < width < 0.045 and force > 20):
        return SkillResult(False, "fingers did not close on the book")
    say("lift and back away")
    ok = check("lift", robot.move_linear(g + LIFT * UP, z_axis=fwd, y_axis=UP, speed=0.05))
    ok &= check("back off", robot.move_linear(g + LIFT * UP - BACK * fwd, z_axis=fwd, y_axis=UP, speed=0.06))
    lifted = robot.d.body(f"book_{book}").xpos[2] - c[2]
    ok &= check(f"book lifted {lifted * 100:.1f} cm", lifted > LIFT - 0.02)
    return SkillResult(ok, "holding the book" if ok else "pick failed", {"lifted": lifted})


# ---------------------------------------------------------------------------- shelf
SHELF_PRE = 0.15           # pre-grasp / pre-place / pull-out: this far out of the shelf (m)
SHELF_BACK = 0.10          # the base backs off this far before swinging the book to / from the carry
                           # pose: 15 cm out, the back edge of the book (15 cm wide) is right at the
                           # neighbours' spines (they stick out 3 cm) and the swing hit them; the
                           # arm cannot reach 20 cm out at row A, so the base makes the room
SHELF_UP = 0.03            # book lifted off the board while sliding in / out
W_OPEN_SHELF = 0.05
SET_DOWN_FORCE = 2.5       # N on the surface under the book (its weight is 3.3 N)
PLACE_CLEARANCE = 0.006    # m: placing in a shelf slot, aim this far off the divider / left neighbour
SLOT_TOL = 0.03            # a slot is taken if a book's centre is within this of it (along the shelf)


def _check(log):
    def check(name, res):
        ok = bool(res) if isinstance(res, (bool, np.bool_)) else (
            res is not None and getattr(res, "success", True))
        log.append((name, ok))
        return ok
    return check


def shelf_axes(shelf):
    """(fwd, side) of a shelf seen from its parking poses: fwd into the shelf, side along it
    (towards higher slot numbers)."""
    yaw = META["shelves"][shelf]["parking"]["1"][2]
    fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    return fwd, np.array([fwd[1], -fwd[0], 0.0])


def locate_book(robot, book):
    """Where a book is now (ground truth; the detector later): ("table", "B"), ("shelf", "S2-A3"),
    ("held", None) or ("unknown", None)."""
    if robot.holding == book:
        return ("held", None)
    p = robot.d.body(f"book_{book}").xpos
    for name, t in META["tables"].items():
        hx, hy = t["half_size"]
        if abs(p[0] - t["pos"][0]) < hx and abs(p[1] - t["pos"][1]) < hy and abs(p[2] - t["top_z"]) < 0.06:
            return ("table", name)
    for s in META["shelves"].values():
        for comp, c in s["compartments"].items():
            xs = [q[0] for q in c["positions"]]
            if (min(xs) - 0.05 < p[0] < max(xs) + 0.05 and abs(p[1] - c["surface"][1]) < 0.2
                    and c["surface"][2] < p[2] < c["surface"][2] + BD["length"]):
                return ("shelf", comp)
    return ("unknown", None)


def books_in(robot, comp):
    return [b for b in META["books"] if locate_book(robot, b) == ("shelf", comp)]


def free_slot(robot, comp, ignore=None):
    """Index of the leftmost free slot of a compartment (books packed from the left), or None."""
    shelf = comp.split("-")[0]
    slots = META["shelves"][shelf]["compartments"][comp]["positions"]
    xs = [robot.d.body(f"book_{b}").xpos[0] for b in books_in(robot, comp) if b != ignore]
    for k, s in enumerate(slots):
        if all(abs(s[0] - x) > SLOT_TOL for x in xs):
            return k
    return None


def to_carry_pose(robot, book, log):
    """Book standing above the front of the chassis, for driving."""
    p, f, left = carry_target(robot)
    return _check(log)("carry pose", robot.move_to(p, z_axis=f, y_axis=left, duration=2.5,
                                                   allow=(f"book_{book}",)))


def in_hand_correction(robot, book, tcp_nominal, book_nominal):
    """Horizontal shift of a TCP target that puts the book at book_nominal, given where it really
    sits in the fingers now (it slips a little in a friction grasp; ground truth now, the wrist
    camera later). Call it with the hand already in the placing orientation."""
    now = robot.d.body(f"book_{book}").xpos - robot.d.site_xpos[robot.tcp]
    c = (np.asarray(book_nominal) - tcp_nominal) - now
    c[2] = 0.0                                   # height: the guarded set-down takes care of it
    return c


def set_down(robot, book, z_axis, y_axis, max_dist):
    """Lower the held book until the surface carries SET_DOWN_FORCE of it (wherever it sits in the
    fingers; a fixed height failed after a few mm of slip)."""
    return robot.guarded_move(-UP, SET_DOWN_FORCE, z_axis=z_axis, y_axis=y_axis, speed=0.02,
                              max_dist=max_dist, sense=lambda: robot.object_contact_force(f"book_{book}"))


def _base_shift(robot, goal, book, log, name):
    """Move the base (arm still, book held) to goal; contacts with the held book are fine."""
    res = robot.go_to(*goal, tol_xy=0.005, tol_yaw=0.005, v_max=0.1, a_max=0.2)
    hits = [h for h in res.collisions if f"book_{book}" not in h]
    return _check(log)(name, not hits and res.message != "timeout")


def pick_from_shelf(robot, book, comp, say=print, log=None):
    """Take a standing book out of a shelf compartment, robot at the compartment's column parking:
    pre-grasp 15 cm out, straight in around the protruding spine, close, lift 3 cm off the board,
    pull out, base 10 cm back, carry pose."""
    log = [] if log is None else log
    check = _check(log)
    fwd, side = shelf_axes(comp.split("-")[0])
    b = robot.d.body(f"book_{book}")
    if getattr(robot, "perception", None) is not None:
        centre = robot.perception.shelf_book(robot, book, comp)
        if centre is None:
            return SkillResult(False, f"the camera does not see the book in {comp}")
        g = centre - fwd * (BD["width"] / 2 - BD["grip_in"]["shelf"])     # spine faces the room
    else:
        g = b.xpos - b.xmat.reshape(3, 3)[:, 1] * (BD["width"] / 2 - BD["grip_in"]["shelf"])
    say(f"take {META['books'][book]['title']} out of {comp}")
    # no home() here: at the shelf parking the unfolded home pose reaches into the shelf, hits it
    # and turned the base by 17 deg; from the stow pose straight to the pre-grasp is clear
    robot.set_gripper_width(W_OPEN_SHELF, 0.3)
    pose0 = robot.true_pose()
    if not check("shelf pre-grasp", robot.move_to(g - SHELF_PRE * fwd, z_axis=fwd, y_axis=side, duration=2.5)):
        return SkillResult(False, "shelf pre-grasp not reachable")
    if not check("approach", robot.move_linear(g, z_axis=fwd, y_axis=side, speed=0.04)):
        return SkillResult(False, "approach failed")
    width, force = robot.grasp(book)
    if not check(f"grasp: width {width * 100:.1f} cm, squeeze {force:.0f} N", 0.03 < width < 0.045 and force > 20):
        return SkillResult(False, "fingers did not close on the book")
    z0 = float(b.xpos[2])
    check("lift off the board", robot.move_linear(g + SHELF_UP * UP, z_axis=fwd, y_axis=side, speed=0.02))
    check("pull out", robot.move_linear(g + SHELF_UP * UP - SHELF_PRE * fwd, z_axis=fwd, y_axis=side, speed=0.05))
    out = bool((b.xpos - g) @ fwd < 0.0 and b.xpos[2] > z0 + 0.01)
    if not check("book out of the shelf", out):
        return SkillResult(False, "the book did not come out with the fingers")
    x, y, yaw = robot.get_pose()
    _base_shift(robot, (x - SHELF_BACK * fwd[0], y - SHELF_BACK * fwd[1], yaw), book, log, "base backs off")
    pose0 = robot.true_pose()
    to_carry_pose(robot, book, log)
    moved = math.hypot(robot.true_pose()[0] - pose0[0], robot.true_pose()[1] - pose0[1])
    check(f"base stayed put ({moved * 100:.1f} cm)", moved < 0.02)
    return SkillResult(True, "holding the book")


def place_in_shelf(robot, book, comp, say=print, log=None):
    """Put the held book into the leftmost free slot of a compartment, robot at the column
    parking: base 10 cm back, swing to the pre-place, base forward again (book 15 cm out), correct
    for where the book sits in the fingers, slide in 3 cm above the board, guarded set-down, let
    go, withdraw, stow."""
    log = [] if log is None else log
    check = _check(log)
    fwd, side = shelf_axes(comp.split("-")[0])
    k = free_slot(robot, comp, ignore=book)
    if k is None:
        return SkillResult(False, f"compartment {comp} is full")
    pos = np.array(META["shelves"][comp.split("-")[0]]["compartments"][comp]["positions"][k])
    target = pos + PLACE_CLEARANCE * side            # off the divider / the left neighbour
    gs = target - fwd * (BD["width"] / 2 - BD["grip_in"]["shelf"])
    say(f"place {META['books'][book]['title']} in {comp}, slot {k + 1}")
    park = robot.get_pose()
    back = (park[0] - SHELF_BACK * fwd[0], park[1] - SHELF_BACK * fwd[1], park[2])
    _base_shift(robot, back, book, log, "base backs off")
    if not check("shelf pre-place", robot.move_to(gs + SHELF_UP * UP - (SHELF_PRE + SHELF_BACK) * fwd,
                                                   z_axis=fwd, y_axis=side, duration=2.5, allow=(f"book_{book}",))):
        return SkillResult(False, "pre-place not reachable")
    _base_shift(robot, park, book, log, "base forward to the shelf")
    shift = in_hand_correction(robot, book, gs, target)
    say(f"  book sits {np.linalg.norm(shift) * 100:.1f} cm off in the hand, correcting")
    gs = gs + shift
    check("insert", robot.move_linear(gs + SHELF_UP * UP, z_axis=fwd, y_axis=side, speed=0.04))
    g = set_down(robot, book, fwd, side, SHELF_UP + 0.02)
    check(f"set down ({g.force:.1f} N on the board)", g)
    robot.release(book)
    robot.set_gripper_width(W_OPEN_SHELF, 0.5)
    tcp = robot.fk_world()[:3, 3]
    check("withdraw", robot.move_linear(tcp - SHELF_PRE * fwd, z_axis=fwd, y_axis=side, speed=0.06))
    robot.stow()
    robot.wait(0.5)
    return SkillResult(True, f"placed in {comp}, slot {k + 1}", {"slot": k})
