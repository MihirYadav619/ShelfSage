"""Phase 6 done-when test: navigation on the MCL pose.

    1. tour: from the start pose to every table (middle parking) and to two columns of every
       shelf, with NAVIGATE (A* + pure pursuit) and the pose from MCL only (odometry + LiDAR vs
       the map); no collisions; parked within 2 cm of the parking pose (true pose)
    2. at every parking pose the arm reaches the work poses with the MCL pose: table edge grasp
       (pre-grasp, grasp), shelf grasp in rows A and B (pre-grasp, grasp)
    3. localization error against ground truth, logged at every MCL update
       (sim/output/phase6/mcl_log.csv): at the parking poses < 1.5 cm, on the way < 3 cm
    4. obstacles that are not in the map: a cart standing in the gap between tables A and B,
       found on the way from Table B to Table E; a box dropped onto the path 1.2 m ahead of the
       driving robot (start -> Table E). Both detected, driven around, no contact, goal reached
Plots the paths and obstacles to sim/output/phase6/paths.png.

Usage (project root, venv active):
    python sim/test_phase6.py
"""
import csv
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sim.robot import Robot  # noqa: E402
from skills import navigation as nav  # noqa: E402
from skills.manipulation import BD, META, UP, shelf_axes, table_frame  # noqa: E402

OUT = ROOT / "sim/output/phase6"
TOUR = ["Table A", "Table B", "Table C", "Table F", "Table E", "Table D", "S1-A1", "S1-A4",
        "S2-A2", "S2-A3", "S3-A1", "S3-A4"]


def put(model, data, body, xy):
    adr = model.jnt_qposadr[model.body(body).jntadr[0]]
    data.qpos[adr:adr + 2] = xy
    data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
    data.qvel[model.body(body).dofadr[0]:model.body(body).dofadr[0] + 6] = 0
    mujoco.mj_forward(model, data)


def point_ahead(path, here, dist):
    """The point `dist` metres further along the polyline `path` than the point nearest `here`."""
    pts = [np.asarray(p, float) for p in path]
    best, at = np.inf, (0, 0.0)
    for i in range(len(pts) - 1):
        seg = pts[i + 1] - pts[i]
        L = float(np.linalg.norm(seg))
        u = float(np.clip((here - pts[i]) @ seg / max(L * L, 1e-12), 0, 1))
        d = float(np.linalg.norm(here - (pts[i] + u * seg)))
        if d < best:
            best, at = d, (i, u)
    i, u = at
    p = pts[i] + u * (pts[i + 1] - pts[i])
    left = dist
    for q in pts[i + 1:]:
        L = float(np.linalg.norm(q - p))
        if L >= left:
            return p + (q - p) * left / L
        left -= L
        p = q
    return pts[-1]


def reach_checks(robot, target):
    """IK (with the MCL pose) for the work poses at this parking pose. Returns [(name, ok)]."""
    books = tuple(f"book_{b}" for b in META["books"])
    out = []
    p = nav.parse_target(target)
    if p[0] == "table":
        fwd, _, edge, _ = table_frame(p[1])
        spot = np.array(META["tables"][p[1]]["edge_spots"][1])
        g = spot - fwd * (BD["width"] / 2 - BD["grip_in"]["table"])
        for name, q in (("pre-grasp", g - 0.10 * fwd), ("grasp", g)):
            r = robot.ik(q, z_axis=fwd, y_axis=UP, allow=books, reject_margin=0.4)
            out.append((f"{target}: table {name}", r is not None and r.success))
    else:
        shelf, comp = p[1], p[2]
        fwd, side = shelf_axes(shelf)
        col = comp.split("-")[1][1:]
        for row in ("A", "B"):
            c = f"{shelf}-{row}{col}"
            pos = np.array(META["shelves"][shelf]["compartments"][c]["positions"][0])
            gs = pos - fwd * (BD["width"] / 2 - BD["grip_in"]["shelf"])
            for name, q in (("pre-grasp", gs - 0.15 * fwd), ("grasp", gs)):
                r = robot.ik(q, z_axis=fwd, y_axis=side, allow=books)
                out.append((f"{c}: shelf {name}", r is not None and r.success))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(ROOT / "scene/library.xml"))
    data = mujoco.MjData(model)
    robot = Robot(model, data, pose_source="mcl")
    checks, paths, lines = [], [], []
    t0 = time.time()

    # ---------------------------------------------------------------- 1-3. tour
    park_err, loc_err, collisions_ok, reach = [], [], True, []
    for target in TOUR:
        goal = nav.parking_pose(target)
        log = []
        ok, msg = nav.navigate(robot, goal, say=lambda s: None, log=log)
        paths.append(robot.nav_path)
        tx, ty, tyaw = robot.true_pose()
        ex, ey, eyaw = robot.get_pose()
        pe = math.hypot(tx - goal[0], ty - goal[1])
        le = math.hypot(tx - ex, ty - ey)
        park_err.append(pe)
        loc_err.append(le)
        collisions_ok &= ok
        r = reach_checks(robot, target)
        reach += r
        lines.append(f"  {target:8s} {'ok ' if ok else 'XX '}{msg:10s} parked {pe * 100:4.1f} cm off, "
                     f"MCL error {le * 1000:4.1f} mm / {math.degrees(abs(tyaw - eyaw)):.2f} deg, "
                     f"reach {sum(x for _, x in r)}/{len(r)}")
    print("tour (MCL pose only):")
    print("\n".join(lines))
    checks.append(("every table and shelf reached, no collisions", collisions_ok))
    checks.append((f"parked within 2 cm of the parking pose (max {max(park_err) * 100:.1f} cm)", max(park_err) < 0.02))
    checks.append((f"arm reaches the work poses from every parking pose ({sum(x for _, x in reach)}/{len(reach)})",
                   all(x for _, x in reach)))
    for name, ok in reach:
        if not ok:
            print("   XX", name)

    log_rows = robot.mcl_log
    errs = np.array([math.hypot(t[0] - e[0], t[1] - e[1]) for _, t, e, _ in log_rows])
    yerrs = np.array([abs((t[2] - e[2] + math.pi) % (2 * math.pi) - math.pi) for _, t, e, _ in log_rows])
    with open(OUT / "mcl_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "true_x", "true_y", "true_yaw", "mcl_x", "mcl_y", "mcl_yaw", "sd_x", "sd_y", "sd_yaw", "err_xy"])
        for (t, tp, ep, sd), e in zip(log_rows, errs):
            w.writerow([f"{t:.2f}", *[f"{v:.4f}" for v in (*tp, *ep, *sd)], f"{e:.4f}"])
    odom_err = math.hypot(robot.odom[0] - robot.true_pose()[0], robot.odom[1] - robot.true_pose()[1])
    print(f"MCL: {len(errs)} updates over {data.time:.0f} s; error mean {errs.mean() * 1000:.1f} mm, "
          f"max {errs.max() * 1000:.1f} mm, yaw max {math.degrees(yerrs.max()):.2f} deg "
          f"(odometry alone is now {odom_err * 100:.1f} cm off)")
    checks.append((f"MCL error at the parking poses < 1.5 cm (max {max(loc_err) * 1000:.1f} mm)", max(loc_err) < 0.015))
    checks.append((f"MCL error on the way < 3 cm (max {errs.max() * 1000:.1f} mm), logged", errs.max() < 0.03))

    # ---------------------------------------------------------------- 4a. cart in the gap
    nav.navigate(robot, nav.parking_pose("Table B"), say=lambda s: None)
    put(model, data, "obstacle_cart", (-1.25, -2.0))
    cart0 = data.body("obstacle_cart").xpos.copy()
    log = []
    ok, msg = nav.navigate(robot, nav.parking_pose("Table E"), say=lambda s: None, log=log)
    paths.append(robot.nav_path)
    seen = [s for s, _ in log if s.startswith("obstacle in the path")]
    moved = float(np.linalg.norm(data.body("obstacle_cart").xpos - cart0))
    print(f"cart in the gap A|B, Table B -> Table E: {msg}; {seen[0] if seen else 'not seen'}; "
          f"cart moved {moved * 100:.1f} cm")
    checks.append(("cart in the path: detected, driven around, goal reached, untouched",
                   ok and bool(seen) and moved < 0.01))
    cart_xy = data.body("obstacle_cart").xpos[:2].copy()
    put(model, data, "obstacle_cart", META["test_obstacles"][1]["park"])

    # ---------------------------------------------------------------- 4b. box dropped ahead
    nav.navigate(robot, (0.0, -3.9, math.pi / 2), say=lambda s: None)
    state = {"dropped": None, "start": np.array(robot.true_pose()[:2])}
    orig_step = robot.step

    def step(n=1):
        orig_step(n)
        if state["dropped"] is None and getattr(robot, "nav_path", None) is not None:
            here = np.array(robot.true_pose()[:2])
            if np.linalg.norm(here - state["start"]) > 0.6:
                drop = point_ahead(robot.nav_path, here, 1.2)      # on the path, 1.2 m further on
                put(model, data, "obstacle_box", drop)
                state["dropped"] = drop

    robot.step = step
    log = []
    ok, msg = nav.navigate(robot, nav.parking_pose("Table E"), say=lambda s: None, log=log)
    robot.step = orig_step
    paths.append(robot.nav_path)
    seen = [s for s, _ in log if s.startswith("obstacle in the path")]
    box_moved = 0.0 if state["dropped"] is None else float(np.linalg.norm(data.body("obstacle_box").xpos[:2] - state["dropped"]))
    print(f"box dropped at {None if state['dropped'] is None else np.round(state['dropped'], 2)} while driving start -> "
          f"Table E: {msg}; {seen[0] if seen else 'not seen'}; box moved {box_moved * 100:.1f} cm")
    checks.append(("box dropped onto the path while driving: detected, driven around, goal reached, untouched",
                   ok and state["dropped"] is not None and bool(seen) and box_moved < 0.01))
    box_xy = None if state["dropped"] is None else state["dropped"]

    # ---------------------------------------------------------------- plot
    try:
        import imageio
        nm = nav.navmap(robot)
        cs, _ = nm.cspace(math.pi / 2, False, dynamic=False)
        img = np.full(cs.shape + (3,), 255, np.uint8)
        img[cs] = (215, 215, 215)
        img[nm.low.occ] = (60, 60, 60)
        for xy, half, col in ((cart_xy, (0.25, 0.40), (240, 150, 30)), (box_xy, (0.2, 0.2), (240, 150, 30))):
            if xy is None:
                continue
            c0 = nm.grid.to_cell(np.array(xy) - half)[0]
            c1 = nm.grid.to_cell(np.array(xy) + half)[0]
            img[c0[1]:c1[1] + 1, c0[0]:c1[0] + 1] = col
        colors = [(30, 90, 220)] * len(TOUR) + [(220, 30, 30), (200, 30, 160)]
        for path, col in zip(paths, colors):
            if path is None:
                continue
            for a, b in zip(path[:-1], path[1:]):
                for q in np.linspace(a, b, 200):
                    c = nm.grid.to_cell(q)[0]
                    img[c[1], c[0]] = col
        imageio.imwrite(OUT / "paths.png", np.repeat(np.repeat(img[::-1], 4, 0), 4, 1))
    except Exception as e:                                   # plotting is a bonus
        print("plot failed:", e)

    print(f"\nwall time {time.time() - t0:.0f} s, simulated {data.time:.0f} s")
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("RESULT:", "all passed" if all(ok for _, ok in checks) else "FAILED")


if __name__ == "__main__":
    main()
