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

from skills.interface import SkillResult

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


def book_state(robot, book):
    """Ground-truth pose of a lying book: centre, spine direction (unit, world), yaw, flatness."""
    b = robot.d.body(f"book_{book}")
    R = b.xmat.reshape(3, 3)
    return b.xpos.copy(), -R[:, 1], math.atan2(R[1, 0], R[0, 0]), float(R[2, 2])


def spine_overhang(robot, book, table):
    """How far the spine sticks out past the robot-side edge (m; negative = inside the table)."""
    fwd, _, edge, _ = table_frame(table)
    c, _, _, _ = book_state(robot, book)
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
    c, spine, _, flat = book_state(robot, book)
    if flat < 0.98 or spine @ fwd > -0.9:
        return SkillResult(False, "book is not lying flat with its spine towards the robot")
    target = BD["table_overhang"]
    if spine_overhang(robot, book, table) > target - OVERHANG_TOL:
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
        c, spine, _, _ = book_state(robot, book)
        remaining = target - spine_overhang(robot, book, table)
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
    over = spine_overhang(robot, book, table)
    _, _, yaw, flat = book_state(robot, book)
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
    over = spine_overhang(robot, book, table)
    if over < BD["table_overhang"] - OVERHANG_TOL:
        return SkillResult(False, f"spine only {over * 100:.1f} cm past the edge, drag it first")
    c, spine, _, _ = book_state(robot, book)

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
