"""Table manipulation skills (Phase 5): drag a book to the table edge, then pick it up.

Book poses come from the wrist camera + detector if robot.perception is set (skills/perception.py),
else from simulator ground truth. Other books on the table (lane check, push aside) are ground
truth for now.

Drag to the edge ("pull, then grasp"), for a book lying inside the table, anywhere on it, its spine
turned up to 60 deg away from the robot. Two ways to pull:

  method="base" (default): the arm only presses, the base pulls (whole-body), in one stroke:
    1. books in the lane to the edge are pushed sideways out of it first (clear_lane: press on the
       cover's middle, the base drives along the edge; a book in the way of that push is pushed
       on first)
    2. the base parks turned half as much as the book, in line with it, the press point 0.50 m
       ahead of the arm mount (farther for books at the far side of the table: the chassis stops
       3 cm from the edge); if that is out of reach, square to the table
    3. closed fingertips press the cover 4 cm in from the spine with 20 N, kept while settling
    4. the arm keeps its joint targets and the base drives the hand (press_drive) along a straight
       line to where the press point is when the book lies straight at the edge, turning on the
       way by the book's angle: the book slides and turns with the hand. The base is PI
       controlled (the load turned a P-controlled base by 4 deg, 5 cm at the hand) and drives on
       until its own FK puts the hand on the goal.
  The book lags the hand by ~1.5 cm (straight) / a few deg (turning), compensated; it ends 2-5 cm
  past the edge and within 7 deg of straight, almost always after one stroke (else a short
  second one).

  method="arm": the base stands still, the arm pulls, in strokes:
    1. park in line with the book, close enough that the press point is ~0.58 m ahead of the arm
       mount (3-50 cm from the edge; the far books need the base right at the table)
    2. closed fingertips above the top cover, 4 cm in from the spine, gripper tilted back towards
       the robot: -30 deg if the IK reaches both ends of the stroke, else -15 or 0 deg (the
       tilted hand reaches close to the base, the vertical one far out)
    3. base station keeping on; guarded move down until the fingers press with 20 N
    4. pull straight towards the robot (at most 25 cm); the book follows because the rubber pads
       (mu 1.5) grip harder than the paper slides on the wood (mu 0.4): 1.5 F > 0.4 (F + m g)
    5. lift off, station keeping off, measure the book again; repeat until the spine overhangs
       the edge by 3 cm (+-1 cm). A stroke that hits the workspace boundary aborts safely; the
       next one re-parks and starts from a fresh IK solution.

Pick from the edge (as in sim/demo_pick_place.py): parking square to the spine at the normal
distance, pre-grasp, straight in around the overhanging spine, close the fingers (friction grasp;
robot.grasp_mode "weld" adds the attach weld), lift, back away.
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
PRESS_FORCE = 20.0         # N; at 10 N the fingertips crept ~35 % of a long base pull (the soft
                           # contact kept breaking), at 20 N about 2 %, so one pull does it
PRESS_TILTS = [math.radians(a) for a in (-30, -15, 0)]   # gripper leans back towards the robot
MAX_STROKE = 0.25
SLIP = 0.005               # the fingertips creep ~5 mm per stroke: pull that much further
MAX_STROKES = 5
BASE_PRESS_AT = 0.50       # method "base": arm mount -> press point (m, horizontal)
BASE_PULL_SPEED = 0.04     # m/s, the base backing away with the book
MAX_BASE_STROKE = 0.75     # longest single pull of the base (more than the table is deep)
BASE_LAG = 0.015           # the book lags the hand by 0.5-2.5 cm whatever the pull length (the
                           # soft fingertip contact): pull that much further (not when the stroke
                           # also turns the book: then it follows the hand and ended 1.3 cm too far)
MAX_YAW = math.radians(60) # method "base": books turned up to this far are turned straight while
                           # pulled; beyond, the spine is not the side facing the robot
YAW_TOL = math.radians(7)  # straight enough to grasp (the grasp parks square to the spine, and
                           # in the hand the book is square to the hand whatever it was on the table)
ROT_LAG = math.radians(4)  # the book lags the hand's turn by 1-9 deg (the soft fingertip contact
                           # takes up the first degrees): turn that much further
OVERHANG_TOL = 0.01        # method "arm": stroke until the overhang is 3 +- 1 cm
# method "base": aim at 3.5 cm and take 2..5 cm: the book lags the hand by 0-2 cm in a long pull
# (it varies with the arm posture), and a second stroke of 1-2 cm is not worth it; the grasp needs
# >= 2 cm under the spine for the lower finger, and the book's centre stays on the table up to 7.5
OVERHANG_AIM, OVERHANG_MIN, OVERHANG_MAX = 0.035, 0.02, 0.05
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



# ---------------------------------------------------------------------------- whole-body press
def tcp_xy(robot):
    """Where the hand really is (xy): own FK of the measured joints on the estimated base pose.
    Under the sideways load of a push the hand lags the base by ~5 cm (base yaw, joint tracking),
    so the commanded pose is not good enough to know how far the book went."""
    return robot.fk_world(robot.arm_q())[:2, 3].copy()


def press_drive(robot, goal_xy, dyaw=0.0, body=None, speed=BASE_PULL_SPEED, tol=0.005, tries=3):
    """The arm keeps its joint targets (pressing a book), the base drives so that the hand goes
    along a straight line to `goal_xy` while the base (and hand) turns by `dyaw`. The base follows
    the path with feedforward + P control; then, measuring the hand (tcp_xy), it drives on until
    the hand is within `tol` of the goal (at most `tries` rounds). Returns (error m, collisions)."""
    from sim.robot import CONTROL_DT, rot2, wrap
    n_ctrl = int(round(CONTROL_DT / robot.m.opt.timestep))
    goal_xy = np.asarray(goal_xy, float)[:2]
    x, y, yaw0 = robot.get_pose()
    yaw1 = yaw0 + dyaw
    hits = set()
    for k in range(tries):
        x, y, yaw = robot.get_pose()
        h0 = tcp_xy(robot)
        off = rot2(yaw).T @ (h0 - (x, y))                   # hand in the base frame
        dist = float(np.linalg.norm(goal_xy - h0))
        turn = wrap(yaw1 - yaw)
        if dist < tol and abs(turn) < 0.01:
            break
        T = max(1.0, 1.5 * (dist + 0.3 * abs(turn)) / speed)   # smoothstep peaks at 1.5x the mean

        def pose(t):
            sm = min(t / T, 1.0)
            sm = sm * sm * (3 - 2 * sm)
            yw = yaw + sm * turn
            h = h0 + sm * (goal_xy - h0)
            return np.append(h - rot2(yw) @ off, yw)

        # PI: the load on the hand (0.7 m ahead of the base centre) turns the base, and a P loop
        # on the velocity-controlled base leaves a steady yaw error (4 deg = 5 cm at the hand)
        t, i_xy, i_yaw = 0.0, np.zeros(2), 0.0
        while t < T + 1.0:
            want, nxt = pose(t), pose(t + CONTROL_DT)
            ff = (nxt - want) / CONTROL_DT
            px, py, pyaw = robot.get_pose()
            e_xy, e_yaw = want[:2] - (px, py), wrap(want[2] - pyaw)
            i_xy = np.clip(i_xy + e_xy * CONTROL_DT, -0.05, 0.05)
            i_yaw = float(np.clip(i_yaw + e_yaw * CONTROL_DT, -0.1, 0.1))
            v = ff[:2] + 3.0 * e_xy + 4.0 * i_xy
            w = ff[2] + 4.0 * e_yaw + 8.0 * i_yaw
            robot.send_cmd_vel(*(rot2(pyaw).T @ v), float(w))
            robot.step(n_ctrl)
            t += CONTROL_DT
            hits |= {h for h in robot.robot_contacts()
                     if (body is None or body not in h) and "left_finger <-> panda_right_finger" not in h}
        robot.stop()
        robot.wait(0.3)
    return float(np.linalg.norm(goal_xy - tcp_xy(robot))), sorted(hits)


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


def books_on_table(robot, table, exclude=()):
    """Ids of the books lying on `table` (ground truth)."""
    t = META["tables"][table]
    out = []
    for name, info in META["books"].items():
        if name in exclude:
            continue
        p = robot.d.body(info["body"]).xpos
        if (abs(p[2] - t["top_z"] - BD["rib_height"] - BD["thickness"] / 2) < 0.02
                and abs(p[0] - t["pos"][0]) < t["half_size"][0] + 0.05
                and abs(p[1] - t["pos"][1]) < t["half_size"][1] + 0.05):
            out.append(name)
    return out


def lane_blockers(robot, book, table, margin=0.01):
    """Ids of the other books lying in the lane the book sweeps on its way to the edge."""
    fwd, left, edge, _ = table_frame(table)
    l0, l1, d0, _ = footprint(robot, f"book_{book}", fwd, left)
    out = []
    for name in books_on_table(robot, table, exclude=(book,)):
        m0, m1, e0, e1 = footprint(robot, f"book_{name}", fwd, left)
        if m1 > l0 - margin and m0 < l1 + margin and e1 > edge - 0.1 and e0 < d0:
            out.append(name)
    return out


def drag_blockers(robot, book, table, margin=0.01):
    """Titles of the books in the lane (see lane_blockers)."""
    return [META["books"][b]["title"] for b in lane_blockers(robot, book, table, margin)]


SIDE_GAP = 0.03            # a pushed-aside book ends this far beside the lane
SIDE_CLEAR = 0.02          # ... and this far from other books and the table's side edges


def side_pushes(robot, body, table, keep, direction, fixed, depth=2):
    """Pushes along the table edge that move book `body` out of the lateral band `keep` (lo, hi)
    towards `direction` (-1 right, +1 left): [(book, shift), ...] in order, or None. A book in the
    way of the push is pushed on first, the same way (up to `depth` levels); books in `fixed`
    (the target) may not move."""
    fwd, left, _, _ = table_frame(table)
    t = META["tables"][table]
    centre_lat = float(np.array([*t["pos"], 0.0]) @ left)
    half_lat = abs(left[0]) * t["half_size"][0] + abs(left[1]) * t["half_size"][1]
    m0, m1, e0, e1 = footprint(robot, f"book_{body}", fwd, left)
    sh = (keep[0] - SIDE_GAP - m1) if direction < 0 else (keep[1] + SIDE_GAP - m0)
    lo, hi = min(m0, m0 + sh), max(m1, m1 + sh)               # swept along the edge
    if lo < centre_lat - half_lat + SIDE_CLEAR or hi > centre_lat + half_lat - SIDE_CLEAR:
        return None
    plan = []
    in_way = []
    for b in books_on_table(robot, table, exclude=(body,)):
        o0, o1, f0, f1 = footprint(robot, f"book_{b}", fwd, left)
        if o1 > lo - SIDE_CLEAR and o0 < hi + SIDE_CLEAR and f1 > e0 - SIDE_CLEAR and f0 < e1 + SIDE_CLEAR:
            in_way.append((o0 if direction > 0 else -o1, b))
    for _, b in sorted(in_way, reverse=True):                 # the farthest one goes first
        if b in fixed or depth == 0:
            return None
        sub = side_pushes(robot, b, table, (lo, hi), direction, fixed | {body}, depth - 1)
        if sub is None:
            return None
        plan += sub
    return plan + [(body, sh)]


def side_shift(robot, blocker, book, table):
    """Pushes that move `blocker` out of the lane of `book`: the cheaper way (less pushing) of
    left and right, pushing on books in the way first if needed; None if neither works."""
    fwd, left, _, _ = table_frame(table)
    l0, l1, _, _ = footprint(robot, f"book_{book}", fwd, left)
    plans = [p for p in (side_pushes(robot, blocker, table, (l0, l1), d, {book}) for d in (-1, 1))
             if p is not None]
    return min(plans, key=lambda p: (len(p), sum(abs(x) for _, x in p))) if plans else None


def push_aside(robot, blocker, table, shift, say=print, log=None):
    """Press on a lying book and push it `shift` m along the table edge (+ = robot's left): park
    facing the book, press its cover in the middle with 20 N, the base drives sideways."""
    log = [] if log is None else log
    fwd, left, edge, _ = table_frame(table)
    b = robot.d.body(f"book_{blocker}")
    c = b.xpos.copy()
    press = c.copy()
    press[2] = c[2] + BD["thickness"] / 2
    mount_behind_front = robot.meta["chassis_half"][0] - robot.meta["mount_x"]
    clearance = float(np.clip(BASE_PRESS_AT - (press @ fwd - edge) - mount_behind_front,
                              MIN_CLEARANCE, META["robot"]["park_clearance"]["table"]))
    name = META["books"][blocker]["title"]
    say(f"push {name} {abs(shift) * 100:.0f} cm to the {'left' if shift > 0 else 'right'}, out of the way")
    robot.stow()
    res = robot.go_to(*table_parking(robot, table, float(c @ left), clearance), tol_xy=0.01, tol_yaw=0.01)
    log.append((f"park facing {name}", res.success))
    if not res.success:
        return SkillResult(False, f"drive failed: {res.message} {res.collisions}")
    robot.home(2.0)
    robot.set_gripper_width(0.0, 0.5)
    for tilt in PRESS_TILTS:
        z_ax = math.sin(tilt) * fwd - math.cos(tilt) * UP
        ik = robot.ik(press + 0.04 * UP, z_axis=z_ax, y_axis=left, n_random=4)
        if ik is not None and ik.success:
            break
    else:
        log.append((f"reach {name}", False))
        return SkillResult(False, f"cannot reach {name} to push it aside")
    r = robot.move_to(press + 0.04 * UP, z_axis=z_ax, y_axis=left, duration=2.0)
    if r is None or not r.success:
        log.append((f"above {name}", False))
        return SkillResult(False, f"cannot reach above {name}")
    robot.hold_base(True)
    g = robot.guarded_move(-UP, PRESS_FORCE, z_axis=z_ax, y_axis=left, max_dist=0.06,
                           body=f"book_{blocker}", settle=0.5, hold=True)
    log.append((f"press on {name}: {g.force:.0f} N", g.success))
    robot.hold_base(False)
    if not g.success:
        return SkillResult(False, f"no contact with {name}")
    _, hits = press_drive(robot, tcp_xy(robot) + shift * left[:2], body=f"book_{blocker}")
    robot.hold_base(True)
    robot.move_linear(robot.fk_world()[:3, 3] + 0.06 * UP, z_axis=z_ax, y_axis=left, speed=0.04)
    robot.hold_base(False)
    robot.set_gripper_width(W_OPEN, 0.3)
    moved = float((b.xpos - c) @ left)
    ok = not hits and abs(moved - shift) < 0.03
    log.append((f"pushed {name} {moved * 100:.0f} cm (asked {shift * 100:.0f})", ok))
    if hits:
        return SkillResult(False, f"collision while pushing {name}: {hits}")
    return SkillResult(ok, "pushed aside" if ok else f"{name} moved {moved * 100:.0f} cm only")


def clear_lane(robot, book, table, say=print, log=None):
    """Push every book lying in the lane of `book` sideways out of it (nearest first; a book in
    the way of that push is pushed on first)."""
    log = [] if log is None else log
    fwd, _, _, _ = table_frame(table)
    for _ in range(4):
        blockers = lane_blockers(robot, book, table)
        if not blockers:
            return SkillResult(True, "lane free")
        blocker = min(blockers, key=lambda b: robot.d.body(f"book_{b}").xpos @ fwd)
        plan = side_shift(robot, blocker, book, table)
        name = META["books"][blocker]["title"]
        if plan is None:
            log.append((f"room to push {name} aside", False))
            return SkillResult(False, f"no room to push {name} out of the lane", {"blockers": blockers})
        for b, shift in plan:
            res = push_aside(robot, b, table, shift, say=say, log=log)
            if not res.success:
                return res
    return SkillResult(not lane_blockers(robot, book, table), "lane free")


# ---------------------------------------------------------------------------- skills
def book_yaw(spine, fwd):
    """Signed angle (rad, + = counter-clockwise) from "spine facing the robot" (-fwd) to `spine`."""
    return math.atan2(fwd[1] * spine[0] - fwd[0] * spine[1], -float(spine @ fwd))


def base_drag_plan(robot, table, c, spine, target, k=0.5):
    """One whole-body stroke for a book lying at centre c with its spine along `spine`: the base
    parks turned k (0.5) times as much as the book, the press point 0.5 m ahead of the
    arm mount (farther if the chassis would get closer than 3 cm to the edge, nearer if it would be
    farther than the normal parking); then the hand drags the press point to where it is when the
    book lies straight with its spine `target` past the edge, while the base turns back square to
    the table and on by as much, so the book turns with it. Returns (base pose, press point, goal xy, turn,
    base forward axis, base left axis)."""
    fwd, left, edge, tyaw = table_frame(table)
    theta = book_yaw(spine, fwd)
    yb = tyaw + k * theta          # turned half as much as the book, half the other way at the end:
                                   # a base turned the full 40 deg had to stay so far from the edge
                                   # (chassis corner) that the book was out of reach
    bf = np.array([math.cos(yb), math.sin(yb), 0.0])
    bl = np.array([-bf[1], bf[0], 0.0])
    press = c + spine * (BD["width"] / 2 - PRESS_IN)
    press[2] = c[2] + BD["thickness"] / 2
    hx, hy = robot.meta["chassis_half"]
    mx = robot.meta["mount_x"]

    def base_at(dist):
        b = press - (mx + dist) * bf
        b[2] = 0.0
        return b

    dist = BASE_PRESS_AT
    front = max((base_at(dist) + hx * bf + sg * hy * bl) @ fwd for sg in (-1, 1))
    too_close = front - (edge - MIN_CLEARANCE)
    too_far = (edge - META["robot"]["park_clearance"]["table"]) - front
    if too_close > 0:
        dist += too_close / (bf @ fwd)
    elif too_far > 0:
        dist -= too_far / (bf @ fwd)
    b = base_at(dist)
    c_f = float(c @ left) * left + (edge - target + BD["width"] / 2) * fwd
    goal = c_f - fwd * (BD["width"] / 2 - PRESS_IN)
    return (float(b[0]), float(b[1]), yb), press, goal[:2], -theta, bf, bl


def drag_to_edge(robot, book, table, say=print, log=None, method="base", clear=True):
    """Pull a book lying inside `table` towards the robot until its spine overhangs the edge by
    the table overhang (3 cm). method "base": the arm presses, the base pulls (and turns a book
    lying askew straight on the way); "arm": the base stands, the arm pulls in strokes of at most
    25 cm. Books in the lane are pushed aside first (clear=True). Leaves the robot near the table,
    arm up."""
    if method not in ("base", "arm"):
        raise ValueError(f"unknown drag method {method}")
    by_base = method == "base"
    log = [] if log is None else log
    fwd, left, edge, _ = table_frame(table)
    st = book_state(robot, book, table)
    if st is None:
        return SkillResult(False, "the camera does not see the book on the table")
    c, spine, _, flat = st
    max_yaw = MAX_YAW if by_base else math.acos(0.9)
    if flat < 0.98 or abs(book_yaw(spine, fwd)) > max_yaw:
        return SkillResult(False, f"book is not lying flat with its spine towards the robot "
                                  f"(turned {math.degrees(book_yaw(spine, fwd)):.0f} deg)")
    target = OVERHANG_AIM if by_base else BD["table_overhang"]
    lo = OVERHANG_MIN if by_base else target - OVERHANG_TOL
    hi = OVERHANG_MAX if by_base else target + OVERHANG_TOL
    if spine_overhang(robot, book, table, c) > lo and abs(book_yaw(spine, fwd)) < YAW_TOL:
        return SkillResult(True, "already at the edge", {"strokes": 0})
    blockers = drag_blockers(robot, book, table)
    if blockers:
        if not clear:
            log.append((f"lane to the edge is free (blocked by {', '.join(blockers)})", False))
            return SkillResult(False, f"blocked by {', '.join(blockers)}", {"blockers": blockers})
        res = clear_lane(robot, book, table, say=say, log=log)
        if not res.success:
            return SkillResult(False, f"blocked by {', '.join(blockers)}: {res.message}", {"blockers": blockers})
        st = book_state(robot, book, table)
        if st is None:
            return SkillResult(False, "lost sight of the book")
        c, spine, _, _ = st
    lateral = float(c @ left)
    mount_behind_front = robot.meta["chassis_half"][0] - robot.meta["mount_x"]
    clearance_now = None
    k_turn = 0.5

    strokes = 0
    for strokes in range(1, MAX_STROKES + 1):
        st = book_state(robot, book, table) if strokes > 1 else st
        if st is None:
            return SkillResult(False, "lost sight of the book while dragging it", {"strokes": strokes})
        c, spine, _, _ = st
        remaining = target - spine_overhang(robot, book, table, c)
        if spine_overhang(robot, book, table, c) > lo and (not by_base or abs(book_yaw(spine, fwd)) < YAW_TOL):
            strokes -= 1
            break

        if by_base:
            # 1. park turned like the book, in line with it
            park, press, goal, turn, bf, bl = base_drag_plan(robot, table, c, spine, target, k_turn)
            here = robot.get_pose()
            if (strokes == 1 or math.hypot(park[0] - here[0], park[1] - here[1]) > 0.01
                    or abs(park[2] - here[2]) > 0.02):
                say(f"park in line with the book{f', turned {math.degrees(-turn * k_turn):.0f} deg' if abs(turn) > YAW_TOL else ''}")
                robot.stow()
                res = robot.go_to(*park, tol_xy=0.01, tol_yaw=0.01)
                log.append((f"stroke {strokes}: park in line with the book, turned {math.degrees(-turn * k_turn):.0f} deg",
                            res.success))
                if not res.success:
                    return SkillResult(False, f"drive failed: {res.message} {res.collisions}")
                robot.home(2.0)
                robot.set_gripper_width(0.0, 0.5)
                if getattr(robot, "perception", None) is not None:
                    # look again from close by: from the far parking a mid-table book is only
                    # partly in view (centre off by a few cm), from here the camera gets above it
                    st = book_state(robot, book, table)
                    if st is None:
                        return SkillResult(False, "lost sight of the book", {"strokes": strokes})
                    c, spine, _, _ = st
                    # (the base stays; the turn is what the book still needs, whatever the base yaw)
                    _, press, goal, turn, _, _ = base_drag_plan(robot, table, c, spine, target, k_turn)
            # the hand keeps its orientation relative to the base, so the book turns with the base
            x, y, yaw = robot.get_pose()
            bf = np.array([math.cos(yaw), math.sin(yaw), 0.0])
            bl = np.array([-bf[1], bf[0], 0.0])
            way = goal - press[:2]
            if np.linalg.norm(way) > MAX_BASE_STROKE:
                way *= MAX_BASE_STROKE / np.linalg.norm(way)
            lag = BASE_LAG if abs(turn) < math.radians(1) else 0.0
            goal_cmd = press[:2] + way * (1 + lag / max(np.linalg.norm(way), 1e-6))
            z_axes = [math.sin(t) * bf - math.cos(t) * UP for t in PRESS_TILTS]
            y_ax = bl
        else:
            press = c + spine * (BD["width"] / 2 - PRESS_IN)
            press[2] = c[2] + BD["thickness"] / 2
            stroke = min(remaining + SLIP, MAX_STROKE)
            # 1. park so that the press point is STROKE_START ahead of the arm mount
            clearance = float(np.clip(STROKE_START - (press @ fwd - edge) - mount_behind_front,
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
            z_axes = [math.sin(t) * fwd - math.cos(t) * UP for t in PRESS_TILTS]
            y_ax = left

        # 2. tilt: the first one whose IK reaches both ends of the stroke (base: the arm does not
        #    move along the stroke, so above the book and pressed)
        for z_ax in z_axes:
            end = press if by_base else press - stroke * fwd
            ends = [robot.ik(p, z_axis=z_ax, y_axis=y_ax, n_random=4, allow=(f"book_{book}",))
                    for p in (press + 0.04 * UP, end)]
            if all(e is not None and e.success for e in ends):
                break
        else:
            log.append((f"stroke {strokes}: reachable tilt", False))
            if by_base and k_turn > 0 and abs(turn) > YAW_TOL:
                # a far book lying askew: the turned chassis keeps too far from the edge; park
                # square to the table (the chassis 3 cm from it) and turn the whole angle on the way
                k_turn = 0.0
                continue
            return SkillResult(False, "stroke not reachable at any tilt", {"strokes": strokes})
        r = robot.move_to(press + 0.04 * UP, z_axis=z_ax, y_axis=y_ax, duration=2.0)
        if r is None or not r.success:
            log.append((f"stroke {strokes}: above the book", False))
            return SkillResult(False, "cannot reach above the book", {"strokes": strokes})

        # 3-5. press, pull, lift
        robot.hold_base(True)
        g = robot.guarded_move(-UP, PRESS_FORCE, z_axis=z_ax, y_axis=y_ax, max_dist=0.06,
                               body=f"book_{book}", settle=0.5, hold=True)
        log.append((f"stroke {strokes}: press {g.force:.0f} N", g.success))
        if not g.success:
            robot.hold_base(False)
            return SkillResult(False, "no contact with the book", {"strokes": strokes})
        if by_base:
            # the arm keeps its joint targets, so the pressing fingertips move (and turn) with the base
            robot.hold_base(False)
            say(f"stroke {strokes}: press on the cover, the base pulls {np.linalg.norm(way) * 100:.0f} cm"
                + (f" and turns {math.degrees(turn):.0f} deg" if abs(turn) > YAW_TOL else ""))
            turn_cmd = turn + math.copysign(ROT_LAG, turn) if abs(turn) > math.radians(1) else turn
            err, hits = press_drive(robot, goal_cmd, turn_cmd, body=f"book_{book}")
            log.append((f"stroke {strokes}: base pulled {np.linalg.norm(way) * 100:.0f} cm, turned "
                        f"{math.degrees(turn):.0f} deg (hand error {err * 1000:.0f} mm)", not hits))
            if hits:
                return SkillResult(False, f"collision while pulling: {hits}", {"strokes": strokes})
            robot.hold_base(True)
        else:
            say(f"stroke {strokes}: press on the cover, the arm pulls {stroke * 100:.0f} cm")
            p = robot.fk_world()[:3, 3]
            lin = robot.move_linear(p - stroke * fwd, z_axis=z_ax, y_axis=y_ax, speed=0.04)
            if lin.aborted:
                say("  stopped at the edge of the workspace, will press again")
        robot.move_linear(robot.fk_world()[:3, 3] + 0.04 * UP, z_axis=z_ax, y_axis=y_ax, speed=0.04)
        robot.hold_base(False)

    robot.set_gripper_width(W_OPEN, 0.3)   # closed fingers touch each other (a self-contact)
    st = book_state(robot, book, table) if strokes > 0 else st
    if st is None:
        return SkillResult(False, "lost sight of the book after dragging it", {"strokes": strokes})
    c, spine, yaw, flat = st
    over = spine_overhang(robot, book, table, c)
    turned = book_yaw(spine, fwd)
    ok = lo < over < hi and flat > 0.98 and abs(turned) < YAW_TOL
    log.append((f"dragged to the edge: overhang {over * 100:.1f} cm in {strokes} strokes, "
                f"turned {math.degrees(turned):.1f} deg", ok))
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
    if over < OVERHANG_MIN:
        return SkillResult(False, f"spine only {over * 100:.1f} cm past the edge, drag it first")

    def check(name, res):
        ok = bool(res) if isinstance(res, (bool, np.bool_)) else (
            res is not None and getattr(res, "success", True))
        log.append((name, ok))
        return ok

    def parking(c, spine):
        # square to the spine and straight behind the grasp point, as far from it as at the normal
        # parking: the drag leaves a book up to 5 deg off; with the base square to the table and
        # the hand turned, or the base turned but the grasp point 5 cm off its axis, the IK took a
        # posture near the joint limits and the lift left the workspace
        g = c + spine * (BD["width"] / 2 - BD["grip_in"]["table"])
        x, y, yaw = table_parking(robot, table, float(c @ left), META["robot"]["park_clearance"]["table"])
        dist = float((g - np.array([x, y, 0.0])) @ fwd)
        a = -spine * np.array([1.0, 1.0, 0.0])
        a /= np.linalg.norm(a)
        b = g - dist * a
        return float(b[0]), float(b[1]), yaw + book_yaw(spine, fwd)

    say(f"back off to the normal parking at Table {table}")
    robot.stow()
    res = robot.go_to(*parking(c, spine), tol_xy=0.01, tol_yaw=0.01)
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
        park = parking(c, spine)
        if (math.hypot(park[0] - robot.get_pose()[0], park[1] - robot.get_pose()[1]) > 0.01
                or abs(park[2] - robot.get_pose()[2]) > 0.02):
            # the first estimate put the base a few cm off the book's line; off-line, the IK chose
            # a twisted posture for the pre-grasp and the move there knocked the book off
            robot.home(1.5)
            res = robot.go_to(*park, tol_xy=0.01, tol_yaw=0.01)
            if not check("re-park in line with the book", res):
                return SkillResult(False, f"drive failed: {res.message} {res.collisions}")

    g = c + spine * (BD["width"] / 2 - BD["grip_in"]["table"])
    a = -spine * np.array([1.0, 1.0, 0.0])               # approach square to the spine (a book
    a /= np.linalg.norm(a)                               # turned by the drag may be a few deg off)
    say("pre-grasp, straight in around the spine, close the fingers")
    robot.home(2.0)
    robot.set_gripper_width(W_OPEN, 0.3)
    # reject_margin: with the base a few deg off square, the IK otherwise found a twisted posture
    # (joint 5 near its limit) from which the lift left the workspace
    if not check("pre-grasp", robot.move_to(g - 0.10 * a, z_axis=a, y_axis=UP, duration=2.5,
                                            reject_margin=0.4)):
        return SkillResult(False, "pre-grasp not reachable")
    if not check("approach", robot.move_linear(g, z_axis=a, y_axis=UP, speed=0.04)):
        return SkillResult(False, "approach failed")
    width, force = robot.grasp(book)
    if not check(f"grasp: width {width * 100:.1f} cm, squeeze {force:.0f} N", 0.03 < width < 0.045 and force > 20):
        return SkillResult(False, "fingers did not close on the book")
    say("lift and back away")
    ok = check("lift", robot.move_linear(g + LIFT * UP, z_axis=a, y_axis=UP, speed=0.05))
    ok &= check("back off", robot.move_linear(g + LIFT * UP - BACK * a, z_axis=a, y_axis=UP, speed=0.06))
    lifted = robot.d.body(f"book_{book}").xpos[2] - c[2]
    ok &= check(f"book lifted {lifted * 100:.1f} cm", lifted > LIFT - 0.02)
    return SkillResult(ok, "holding the book" if ok else "pick failed", {"lifted": lifted})


def free_edge_spot(robot, table, near_lateral=None):
    """Index of a free edge spot of `table` (no book lying within 15 cm of it), the one nearest
    `near_lateral` (world coordinate along the edge) if given; None if all are taken."""
    fwd, left, _, _ = table_frame(table)
    spots = [np.array(p) for p in META["tables"][table]["edge_spots"]]
    taken = [robot.d.body(f"book_{b}").xpos[:2] for b in books_on_table(robot, table)]
    free = [k for k, sp in enumerate(spots) if all(np.linalg.norm(sp[:2] - q) > 0.15 for q in taken)]
    if not free:
        return None
    if near_lateral is None:
        return free[0]
    return min(free, key=lambda k: abs(float(spots[k] @ left) - near_lateral))


def place_on_table(robot, book, table, spot=None, say=print, log=None):
    """Lay the held book on `table` at an edge spot, spine towards the robot and 3 cm past the
    edge (where the table grasp picks books up): the reverse of pick_from_table_edge. Park in
    line with the spot, swing the book from the carry pose to lying, 10 cm above and 16 cm back,
    correct for where it sits in the fingers, forward and down until the table carries it,
    let go, slide the fingers straight back off the spine, stow."""
    log = [] if log is None else log
    check = _check(log)
    fwd, left, _, _ = table_frame(table)
    if spot is None:
        x, y, _ = robot.get_pose()
        spot = free_edge_spot(robot, table, near_lateral=float(np.array([x, y, 0.0]) @ left))
        if spot is None:
            return SkillResult(False, f"no free edge spot on Table {table}")
    target = np.array(META["tables"][table]["edge_spots"][spot])        # book centre when placed
    g = target - fwd * (BD["width"] / 2 - BD["grip_in"]["table"])       # TCP: around the spine
    say(f"lay {META['books'][book]['title']} on Table {table}, spot {spot + 1}")
    res = robot.go_to(*table_parking(robot, table, float(target @ left), META["robot"]["park_clearance"]["table"]),
                      tol_xy=0.01, tol_yaw=0.01, v_max=0.2, a_max=0.3)
    hits = [h for h in res.collisions if f"book_{book}" not in h]
    if not check("park in line with the spot", not hits and res.message != "timeout"):
        return SkillResult(False, f"drive failed: {res.message} {hits}")
    # The fingers close vertically. Which way up the book ends is fixed by joint 7's range: the
    # shelf grasp is reachable with one roll of the hand only, and from it only the turn that
    # lays a shelf book cover-down is reachable at the table (the other needs joints 4/5 at their
    # limits, in a different wrist family at every point of the move). Spine towards the robot
    # and over the edge as for a pick; the cover-up way would need a regrasp.
    R = None
    above = g + LIFT * UP - BACK * fwd
    if not check("pre-place (book lying, above and back)",
                 robot.move_to(above, z_axis=fwd, y_axis=UP, duration=3.0, allow=(f"book_{book}",))):
        return SkillResult(False, "pre-place not reachable")
    shift = in_hand_correction(robot, book, g, target)
    say(f"  book sits {np.linalg.norm(shift) * 100:.1f} cm off in the hand, correcting")
    g = g + shift
    check("forward over the table", robot.move_linear(g + LIFT * UP, z_axis=fwd, y_axis=UP, speed=0.05))
    gd = set_down(robot, book, fwd, UP, LIFT + 0.03)
    check(f"set down ({gd.force:.1f} N on the table)", gd)
    robot.release(book)
    robot.set_gripper_width(W_OPEN, 0.5)
    tcp = robot.fk_world()[:3, 3]
    check("slide the fingers back off the spine",
          robot.move_linear(tcp - BACK * fwd, z_axis=fwd, y_axis=UP, speed=0.05))
    robot.move_linear(robot.fk_world()[:3, 3] + 0.10 * UP, z_axis=fwd, y_axis=UP, speed=0.05)
    robot.stow()
    robot.wait(0.5)
    b = robot.d.body(f"book_{book}")
    Rb = b.xmat.reshape(3, 3)
    flat = abs(float(Rb[2, 2]))                                   # either cover up
    off = float(np.linalg.norm(b.xpos[:2] - target[:2]))
    ok = flat > 0.98 and off < 0.03 and abs(b.xpos[2] - target[2]) < 0.02
    check(f"book lies flat ({'cover up' if Rb[2, 2] > 0 else 'cover down'}), {off * 100:.1f} cm from the spot", ok)
    return SkillResult(ok, f"laid on Table {table}, spot {spot + 1}" if ok else "the book did not end where it should",
                       {"spot": spot, "off": off})


# ---------------------------------------------------------------------------- shelf
SHELF_PRE = 0.15           # pre-grasp / pre-place / pull-out: this far out of the shelf (m)
SHELF_BACK = 0.15          # the base backs off this far before swinging the book to / from the carry
                           # pose (the swing is checked only at its end). 0.10: placing in row A,
                           # the book hit a book standing in row B above (Phase 7 test); 0.15: 3.7 cm
                           # clear of it and the chassis 3.2 cm clear of the chairs behind (0.20 hit
                           # them and the base timed out); 0.30: pre-place out of reach
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
                              max_dist=max_dist, sense=lambda: robot.object_contact_force(f"book_{book}"),
                              back_off=True)


def _base_shift(robot, goal, book, log, name):
    """Move the base (arm still, book held) to goal; contacts with the held book are fine."""
    res = robot.go_to(*goal, tol_xy=0.005, tol_yaw=0.005, v_max=0.1, a_max=0.2)
    hits = [h for h in res.collisions if f"book_{book}" not in h]
    return _check(log)(name, not hits and res.message != "timeout")


TOUCH_FORCE = 0.05         # N: a fingertip pushing on the spine (contact normal along the approach;
                           # a standing book slides away under 0.1-0.3 N, so any touch counts)
TOUCH_SHIFT = 0.006        # m: shift towards the finger that touched
TOUCH_TRIES = 2


def frontal_touch(robot, book, axis):
    """Which finger presses on `book` head-on (contact normal along the approach `axis`): the world
    direction (unit, across the gripper) from the hand's centre to that finger, or None if neither
    (simulated fingertip force sensing, like finger_force)."""
    m, d = robot.m, robot.d
    bid = m.body(f"book_{book}").id
    tcp = d.site_xpos[robot.tcp]
    yax = d.site_xmat[robot.tcp].reshape(3, 3)[:, 1]
    f6 = np.zeros(6)
    best, side = 0.0, None
    for i, c in enumerate(d.contact[:d.ncon]):
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if bid not in (b1, b2):
            continue
        fb = b2 if b1 == bid else b1
        if fb not in robot.finger_bodies:
            continue
        mujoco_contact_force(m, d, i, f6)
        normal = c.frame[:3]
        if abs(normal @ axis) < 0.7 or f6[0] < TOUCH_FORCE:
            continue
        if f6[0] > best:
            best, side = f6[0], (1.0 if (d.xpos[fb] - tcp) @ yax > 0 else -1.0) * yax
    return side


def mujoco_contact_force(m, d, i, f6):
    import mujoco
    mujoco.mj_contactForce(m, d, i, f6)


def approach_feeling(robot, g, fwd, side, book, step=0.005):
    """Straight in to the grasp point in 5 mm steps, stopping if a fingertip runs into the book
    head-on. Returns (last LinearMoveResult, None or the direction of the finger that touched)."""
    p0 = robot.fk_world()[:3, 3]
    n = max(1, int(math.ceil(float((g - p0) @ fwd) / step)))
    res = None
    for k in range(1, n + 1):
        res = robot.move_linear(p0 + (g - p0) * k / n, z_axis=fwd, y_axis=side, speed=0.04,
                                settle=0.3 if k == n else 0.0)
        touched = frontal_touch(robot, book, fwd)
        if touched is not None:
            return res, touched
        if not res.success and res.aborted:
            return res, None
    return res, None


def pick_from_shelf(robot, book, comp, say=print, log=None):
    """Take a standing book out of a shelf compartment, robot at the compartment's column parking:
    pre-grasp 15 cm out, straight in around the protruding spine, close, lift 3 cm off the board,
    pull out, base 15 cm back, carry pose."""
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
    for attempt in range(TOUCH_TRIES + 1):
        res, touched = approach_feeling(robot, g, fwd, side, book)
        if touched is None:
            break
        # a fingertip ran into the spine: the book lies further towards that finger (the pose
        # from the camera was a few mm off; the fingers clear a 4 cm book by only 5 mm a side)
        say(f"  a fingertip touched the spine, shift {TOUCH_SHIFT * 1000:.0f} mm towards it and try again")
        log.append((f"approach: fingertip on the spine (try {attempt + 1})", attempt < TOUCH_TRIES))
        robot.move_linear(g - SHELF_PRE * fwd, z_axis=fwd, y_axis=side, speed=0.04)
        g = g + TOUCH_SHIFT * touched
        robot.move_linear(g - SHELF_PRE * fwd, z_axis=fwd, y_axis=side, speed=0.04)
    if not check("approach", res is not None and res.success and touched is None):
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
    parking: base 15 cm back, swing to the pre-place, base forward again (book 15 cm out), correct
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
