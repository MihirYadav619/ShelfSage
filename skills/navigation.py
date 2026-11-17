"""NAVIGATE: drive to the parking pose of a table or shelf column with A* on the known map,
watching the LiDAR for obstacles that are not in the map (Phase 6).

    1. turn on the spot to the goal heading (every parking pose faces +y; the base is holonomic
       and keeps that heading while driving)
    2. plan with A* in the configuration space for that heading (control/planner.py), arm stowed,
       or in the carry pose with a book (a longer footprint above 0.95 m). A start or goal right
       next to furniture (a drag parks the chassis 3 cm from a table) is inside the margin: the
       path then starts / ends with a short straight move from / to the nearest free point,
       backing straight away from the furniture
    3. follow the path (each straight segment at up to 0.5 m/s, stopping at every corner); a
       touch between the held book and anything (a shelf book sticking out) is a collision; every
       0.1 s the end points of both LiDARs (front, and rear: the base also drives backwards)
       that lie > 15 cm from anything in the map (by the robot's own pose estimate)
       are marked as obstacles. If they block the rest of the path, stop and plan again around
       them; if nothing gets through, NAVIGATE fails with the reason (for the planner to replan)
    4. final approach to the exact parking pose with go_to

The pose is robot.get_pose(): MCL with pose_source "mcl", else odometry / ground truth.

Targets: "Table B" / "Table_B" / "B" (the middle parking spot), a compartment "S2-A3" (its
column's parking pose), or a shelf "S2" (column 1).
"""
import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
META = json.loads((ROOT / "scene/library_meta.json").read_text())

V_MAX, A_MAX = 0.5, 0.5          # m/s, m/s^2 along the path
OBSTACLE_GAP = 0.15              # LiDAR point this far from the map = an obstacle (m)
OBSTACLE_RANGE = 3.0             # only obstacles this close are marked (m)
MAX_REPLANS = 5


def parse_target(target):
    """("table", "B") / ("shelf", "S2", "S2-A3" or None), or None if not understood."""
    t = target.strip()
    m = re.fullmatch(r"(?:shelf[ _]?)?(S\d)-([A-Z])(\d)", t, re.IGNORECASE)
    if m:
        shelf = m.group(1).upper()
        return ("shelf", shelf, f"{shelf}-{m.group(2).upper()}{m.group(3)}")
    m = re.fullmatch(r"(?:shelf[ _]?)?(S\d)", t, re.IGNORECASE)
    if m:
        return ("shelf", m.group(1).upper(), None)
    m = re.fullmatch(r"(?:table[ _]?)?([A-F])", t, re.IGNORECASE)
    if m:
        return ("table", m.group(1).upper())
    return None


def parking_pose(target):
    """Parking pose (x, y, yaw) for a target string, or None."""
    p = parse_target(target)
    if p is None:
        return None
    if p[0] == "table":
        if p[1] not in META["tables"]:
            return None
        return tuple(META["tables"][p[1]]["parking"][1])
    shelf, comp = p[1], p[2]
    if shelf not in META["shelves"]:
        return None
    col = comp.split("-")[1][1:] if comp else "1"
    if comp and comp not in META["shelves"][shelf]["compartments"]:
        return None
    return tuple(META["shelves"][shelf]["parking"][col])


# ---------------------------------------------------------------------------- map
def navmap(robot):
    """The robot's NavMap (built once from the model: static furniture only)."""
    if getattr(robot, "navmap", None) is None:
        from control.occupancy import exact_distance_grid
        from control.planner import NavMap, library_bounds
        b = library_bounds(META)
        robot.navmap = NavMap(robot.m, robot.d, b, exclude_root=robot.base_body)
        lz = robot.d.site_xpos[robot.lidar_site][2]
        robot.navmap.lidar_map = exact_distance_grid(robot.m, robot.d, lz - 0.05, lz + 0.05, res=0.02,
                                                     bounds=b, exclude_root=robot.base_body)
    return robot.navmap


def lidar_obstacles(robot, nm):
    """World points of LiDAR returns (front and rear LiDAR) that are not explained by the map
    (pose = the robot's own estimate)."""
    x, y, yaw = robot.get_pose()
    c, s = math.cos(yaw), math.sin(yaw)
    lm = nm.lidar_map
    ny, nx = lm.field.shape
    out = []
    for rear in ((False, True) if robot.lidar_rear_site is not None else (False,)):
        ang, rng = robot.lidar_scan(rear=rear)
        ok = np.isfinite(rng) & (rng < OBSTACLE_RANGE)
        if not ok.any():
            continue
        lx, ly = robot.lidar_offset(rear=rear)
        ex = lx + rng[ok] * np.cos(ang[ok])
        ey = ly + rng[ok] * np.sin(ang[ok])
        pts = np.stack([x + c * ex - s * ey, y + s * ex + c * ey], axis=1)
        cells = lm.to_cell(pts)
        inside = (cells[:, 0] >= 0) & (cells[:, 0] < nx) & (cells[:, 1] >= 0) & (cells[:, 1] < ny)
        d = np.full(len(pts), np.inf)
        d[inside] = lm.field[cells[inside, 1], cells[inside, 0]]
        out.append(pts[d > OBSTACLE_GAP])
    return np.concatenate(out) if out else np.zeros((0, 2))


# ---------------------------------------------------------------------------- driving
def follow(robot, path, nm, yaw, holding, log, timeout=120.0):
    """Drive along the polyline `path` (world xy) at heading yaw. Returns "arrived", "blocked"
    (an obstacle not in the map is in the way; it is now in nm.dynamic) or "collision"/"timeout"."""
    from sim.robot import CONTROL_DT, rot2
    pts = np.array(path, float)
    seg = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    n_ctrl = int(round(CONTROL_DT / robot.m.opt.timestep))
    v_cmd = np.zeros(2)
    t, k = 0.0, 0
    hits = set()

    def point_at(s):
        s = min(max(s, 0.0), total)
        i = min(int(np.searchsorted(cum, s, side="right")) - 1, len(seg) - 1)
        return pts[i] + (s - cum[i]) / max(seg_len[i], 1e-9) * seg[i]

    def book_contacts():
        """Things outside the robot that the held book touches (a shelf book, the shelf)."""
        if holding is None:
            return set()
        m, d = robot.m, robot.d
        bid = m.body(f"book_{holding}").id
        out = set()
        for c in d.contact[:d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if bid in (b1, b2):
                o = b2 if b1 == bid else b1
                if o not in robot.robot_bodies:
                    out.add(f"book_{holding} <-> {m.body(o).name}")
        return out

    # segment by segment, slowing to a stop at every corner: with pure pursuit at 0.5 m/s the
    # robot overshot a 90 deg corner by 11 cm and the held book hit a book in the shelf
    s_done, i = 0.0, 0
    while t < timeout:
        x, y, pyaw = robot.get_pose()
        p = np.array([x, y])
        if seg_len[i] < 1e-3:
            if i == len(seg) - 1:
                break
            i += 1
            continue
        u = seg[i] / seg_len[i]
        along = float((p - pts[i]) @ u)
        to_end = seg_len[i] - along
        if to_end < 0.02:
            if i == len(seg) - 1:
                break
            i += 1
            continue
        s_done = cum[i] + max(along, 0.0)
        cross = (p - pts[i]) - along * u
        speed = min(V_MAX, math.sqrt(2 * A_MAX * max(to_end, 0.0)) + 0.02)
        v_des = u * speed - 2.0 * cross
        dv = v_des - v_cmd
        lim = A_MAX * CONTROL_DT
        if np.linalg.norm(dv) > lim:
            dv *= lim / np.linalg.norm(dv)
        v_cmd = v_cmd + dv
        e_yaw = (yaw - pyaw + math.pi) % (2 * math.pi) - math.pi
        vb = rot2(pyaw).T @ v_cmd
        robot.send_cmd_vel(float(vb[0]), float(vb[1]), float(np.clip(2.0 * e_yaw, -0.5, 0.5)))
        robot.step(n_ctrl)
        t += CONTROL_DT
        k += 1
        hits |= {h for h in robot.robot_contacts() if holding is None or f"book_{holding}" not in h}
        hits |= book_contacts()
        if hits:
            robot.stop()
            return "collision", sorted(hits)
        if k % 5 == 0:                                   # 10 Hz: look for obstacles
            obs = lidar_obstacles(robot, nm)
            if len(obs) and nm.add_obstacle_points(obs):
                # blocked by the new obstacles: cells free in the static C-space, not any more
                cs_dyn, _ = nm.cspace(yaw, holding is not None)
                blocked = cs_dyn & ~nm.cspace(yaw, holding is not None, dynamic=False)[0]
                rest = [p] + [point_at(s) for s in np.arange(s_done + 0.05, total, 0.05)] + [pts[-1]]
                if any(not nm.line_free(blocked, a, b) for a, b in zip(rest[:-1], rest[1:])):
                    robot.stop()
                    robot.wait(0.3)
                    log.append((f"obstacle in the path at ({obs[:, 0].mean():.2f}, {obs[:, 1].mean():.2f})", True))
                    return "blocked", []
    robot.stop()
    return ("arrived" if t < timeout else "timeout"), []


def navigate(robot, goal, say=print, log=None, holding=None):
    """Drive to `goal` (x, y, yaw). The arm is stowed unless a book is held (then it stays in the
    carry pose). The held book touching the fingers is fine, anything else is a collision.
    Returns (ok, message)."""
    log = [] if log is None else log
    nm = navmap(robot)
    nm.clear_dynamic()
    if holding is None:
        robot.stow()
    gx, gy, yaw = goal
    heading = np.array([math.cos(yaw), math.sin(yaw)])
    x, y, pyaw = robot.get_pose()
    if abs((yaw - pyaw + math.pi) % (2 * math.pi) - math.pi) > 0.02:
        res = robot.go_to(x, y, yaw, tol_xy=0.02, tol_yaw=0.02)
        log.append(("turn to the goal heading", res.success))
    for attempt in range(MAX_REPLANS + 1):
        cs, _ = nm.cspace(yaw, holding is not None)
        x, y, _ = robot.get_pose()
        start, goal_xy = np.array([x, y]), np.array([gx, gy])
        s_free = nm.nearest_free(cs, start, prefer=-heading)
        g_free = nm.nearest_free(cs, goal_xy, prefer=-heading)
        if s_free is None or g_free is None:
            return False, "no free space around the start or the goal"
        mid = nm.plan(s_free, g_free, yaw, holding is not None)
        if mid is None:
            msg = ("no path: blocked by an obstacle that is not in the map" if nm.dynamic.any()
                   else "no path to the goal")
            log.append((f"plan a path ({msg})", False))
            return False, msg
        path = ([start] if np.linalg.norm(s_free - start) > 0.01 else []) + list(mid) + \
               ([goal_xy] if np.linalg.norm(g_free - goal_xy) > 0.01 else [])
        if len(path) < 2:
            path = [start, goal_xy]
        robot.nav_path = [np.asarray(q, float) for q in path]      # for tests and viewers
        say(f"path with {len(path) - 2} turns, {sum(np.linalg.norm(np.diff(path, axis=0), axis=1)):.1f} m"
            + (f" (replanned {attempt}x)" if attempt else ""))
        result, hits = follow(robot, path, nm, yaw, holding, log)
        log.append((f"drive along the path ({result})", result in ("arrived", "blocked")))
        if result == "arrived":
            res = robot.go_to(gx, gy, yaw, tol_xy=0.02, tol_yaw=0.02)
            hits = [c for c in res.collisions if holding is None or f"book_{holding}" not in c]
            ok = not hits and res.message != "timeout"
            log.append((f"park at ({gx:.2f}, {gy:.2f})", ok))
            return (True, "arrived") if ok else (False, f"parking failed: {res.message} {hits}")
        if result != "blocked":
            return False, f"drive failed: {result} {hits}"
        say("obstacle in the way: stop and plan around it")
    return False, "still blocked after replanning"


def near(robot, goal, tol=0.05):
    x, y, yaw = robot.get_pose()
    return math.hypot(x - goal[0], y - goal[1]) < tol and abs(
        (yaw - goal[2] + math.pi) % (2 * math.pi) - math.pi) < 0.05
