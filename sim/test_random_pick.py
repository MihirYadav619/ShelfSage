"""Table pick from random book poses: how often does GRASP work when a book is not at one of the
scene's fixed spots?

Each trial starts from the reset scene, moves one table book to a random pose on its own table,
then runs NAVIGATE(Table X) -> GRASP(book) and checks that the robot holds the book at the carry
pose and no other book moved. The pose is drawn from
    depth     spine this far in from the robot-side edge, uniform in [--min-depth, --max-depth]
              (negative = overhanging the edge)
    yaw       turned by up to +-YAW deg from "spine towards the robot"
    lateral   anywhere along the table that keeps the whole book on it
and is redrawn while the book would overlap another book on that table (30 cm between centres)
or stick out past the table (more than the 3 cm overhang on the robot side). The other books on
the table stay where they are, so some poses have a book in the drag lane. --blocker also puts a
second book (another table's) in front of the target, across its lane to the edge; GRASP must
push it aside first. Books pushed aside may move (they must stay on the table), no other book may.

Usage (project root, venv active):
    python sim/test_random_pick.py                     # 24 trials, yaw 0
    python sim/test_random_pick.py --n 30 --yaw 30     # books turned up to +-30 deg
    python sim/test_random_pick.py --seed 5 --perception
    python sim/test_random_pick.py --max-depth 0.70    # far books too
    python sim/test_random_pick.py --blocker           # a book in the way of every target
"""
import argparse
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TABLE_BOOKS = ["linalg", "calculus", "ml", "dl", "robotics", "control", "signals", "python",
               "algo", "db", "thermo"]
MIN_GAP = 0.30          # m between book centres (a book's half diagonal is 13 cm)
EDGE_MARGIN = 0.02      # corners this far inside the table's other edges


def corners(c, yaw, BD):
    a = np.array([math.cos(yaw), math.sin(yaw)])           # length axis
    s = np.array([-a[1], a[0]])                             # width axis (book frame +y)
    return [c + sx * a * BD["length"] / 2 + sy * s * BD["width"] / 2 for sx in (-1, 1) for sy in (-1, 1)]


def sample_pose(rng, book, args, META, others):
    """Random (centre xy, yaw) for `book` on its table, or None after many tries."""
    BD = META["book_dims"]
    table = META["books"][book]["location"].split(":")[1]
    t = META["tables"][table]
    cx, cy = t["pos"]
    hx, hy = t["half_size"]
    edge = cy - hy                                          # all tables: robot side at -y
    for _ in range(2000):
        yaw = math.radians(rng.uniform(-args.yaw, args.yaw))
        depth = rng.uniform(args.min_depth, args.max_depth)
        spine = np.array([math.sin(yaw), -math.cos(yaw)])   # book frame -y turned by yaw
        lat = rng.uniform(cx - hx, cx + hx)
        mid = np.array([lat, edge + depth])                 # middle of the spine
        c = mid - spine * BD["width"] / 2
        ps = corners(c, yaw, BD)
        if any(abs(p[0] - cx) > hx - EDGE_MARGIN or p[1] > cy + hy - EDGE_MARGIN
               or p[1] < edge - BD["table_overhang"] for p in ps):
            continue
        if any(np.linalg.norm(c - o) < MIN_GAP for o in others):
            continue
        return table, c, yaw, depth
    return None


def blocker_pose(rng, book, table, c, yaw, META, others):
    """Random pose for a second book in front of the target (overlapping its lane), or None."""
    BD = META["book_dims"]
    t = META["tables"][table]
    cx, cy = t["pos"]
    hx, hy = t["half_size"]
    edge = cy - hy
    spine_t = min(p[1] for p in corners(c, yaw, BD))          # target's corner nearest the edge
    for _ in range(2000):
        byaw = math.radians(rng.uniform(-10, 10))
        bc = np.array([c[0] + rng.uniform(-0.12, 0.12),
                       rng.uniform(edge - BD["table_overhang"], spine_t - 0.03) + BD["width"] / 2])
        ps = corners(bc, byaw, BD)
        if any(abs(p[0] - cx) > hx - EDGE_MARGIN for p in ps):
            continue
        if min(p[1] for p in ps) < edge - BD["table_overhang"] or max(p[1] for p in ps) > spine_t - 0.03:
            continue
        if any(np.linalg.norm(bc - o) < MIN_GAP for o in others):
            continue
        return bc, byaw
    return None


def put_book(model, data, book, c, yaw):
    import mujoco
    adr = model.jnt_qposadr[model.body(f"book_{book}").jntadr[0]]
    data.qpos[adr:adr + 2] = c[:2]
    data.qpos[adr + 2] = 0.781
    data.qpos[adr + 3:adr + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    mujoco.mj_forward(model, data)


def trial(job):
    import mujoco
    from sim.robot import Robot
    from skills import interface as sk
    from skills.manipulation import META, books_on_table, lane_blockers

    book, table, c, yaw, depth, perception, extra = job[:7]
    mcl = len(job) > 7 and job[7]
    model = mujoco.MjModel.from_xml_path(str(ROOT / "scene/library.xml"))
    data = mujoco.MjData(model)
    robot = Robot(model, data, pose_source="mcl" if mcl else "true")   # resets to the home keyframe
    put_book(model, data, book, c, yaw)
    if extra is not None:
        put_book(model, data, *extra)
    if perception:
        from skills.perception import Perception
        robot.perception = Perception(say=lambda s: None)
    blocked = lane_blockers(robot, book, table)
    others = [b for b in META["books"] if b != book and b not in blocked]
    p0 = {b: data.body(f"book_{b}").xpos.copy() for b in others}
    lines, log = [], []
    sk.bind(robot, say=lambda s: None, log=log)
    t0 = time.time()
    ok, failed_at = True, ""
    for name, fn, arg in (("NAVIGATE", sk.navigate, f"Table {table}"), ("GRASP", sk.grasp, book)):
        res = fn(arg)
        lines.append(f"      {'ok' if res.success else 'XX'} {name}({arg}): {res.message}")
        if not res.success:
            ok, failed_at = False, f"{name}: {res.message}"
            break
    strokes = [s for s, _ in log if s.startswith("dragged to the edge")]
    titles = {info["title"]: b for b, info in META["books"].items()}
    pushed = {b for s, _ in log if s.startswith("pushed ")
              for t, b in titles.items() if s.startswith(f"pushed {t} ")}
    moved = max(float(np.linalg.norm(data.body(f"book_{b}").xpos - p)) for b, p in p0.items()
                if b not in pushed)
    if ok and moved > 0.01:
        ok, failed_at = False, f"other book moved {moved * 100:.1f} cm"
    on_table = books_on_table(robot, table)
    if ok and any(b not in on_table for b in pushed | set(blocked)):
        ok, failed_at = False, "a pushed book fell off the table"
    blocked = [META["books"][b]["title"] for b in blocked]
    if len(pushed) > len(blocked):
        blocked.append(f"{len(pushed)} books pushed")
    return {"book": book, "table": table, "depth": depth, "yaw": yaw, "lat": float(c[0]),
            "blocked": blocked, "ok": ok, "failed_at": failed_at, "lines": lines,
            "drag": strokes[-1] if strokes else "", "log": log, "sim": data.time, "wall": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--yaw", type=float, default=0.0, help="max yaw off 'spine to the robot', deg")
    ap.add_argument("--min-depth", type=float, default=-0.03)
    ap.add_argument("--max-depth", type=float, default=0.60)
    ap.add_argument("--perception", action="store_true")
    ap.add_argument("--mcl", action="store_true", help="robot pose from MCL, not ground truth")
    ap.add_argument("--blocker", action="store_true", help="a second book in front of every target")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("-v", "--verbose", action="store_true", help="print every step of failed trials")
    args = ap.parse_args()

    import mujoco
    from skills.manipulation import META
    model = mujoco.MjModel.from_xml_path(str(ROOT / "scene/library.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    rng = np.random.default_rng(args.seed)
    jobs = []
    for _ in range(args.n):
        book = TABLE_BOOKS[rng.integers(len(TABLE_BOOKS))]
        table = META["books"][book]["location"].split(":")[1]
        others = [data.body(f"book_{b}").xpos[:2].copy() for b in TABLE_BOOKS
                  if b != book and META["books"][b]["location"] == f"table:{table}"]
        s = sample_pose(rng, book, args, META, others)
        if s is None:
            continue
        extra = None
        if args.blocker:
            second = [b for b in TABLE_BOOKS if b != book and META["books"][b]["location"] != f"table:{table}"]
            b2 = second[rng.integers(len(second))]
            bp = blocker_pose(rng, book, table, s[1], s[2], META, others)
            if bp is None:
                continue
            extra = (b2, bp[0], bp[1])
        jobs.append((book, s[0], s[1], s[2], s[3], args.perception, extra, args.mcl))

    print(f"Random table picks: {len(jobs)} trials{' with a blocker' if args.blocker else ''}, "
          f"seed {args.seed}, yaw +-{args.yaw:.0f} deg, "
          f"depth {args.min_depth * 100:.0f}..{args.max_depth * 100:.0f} cm, book poses from "
          f"{'the detector' if args.perception else 'ground truth'}")
    workers = args.workers or (4 if args.perception else 6)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(trial, jobs))

    print(f"\n  {'':4s} {'book':9s} tbl {'depth':>6s} {'yaw':>6s} {'x':>6s}  drag / failure")
    for r in sorted(results, key=lambda r: r["depth"]):
        note = r["drag"].replace("dragged to the edge: ", "") if r["ok"] else r["failed_at"]
        if r["blocked"]:
            note += f"   [lane: {', '.join(r['blocked'])}]"
        print(f"  {'PASS' if r['ok'] else 'FAIL'} {r['book']:9s}  {r['table']}  {r['depth'] * 100:4.0f}cm "
              f"{math.degrees(r['yaw']):5.0f}d {r['lat']:6.2f}  {note}")
        if args.verbose and not r["ok"]:
            for s, good in r["log"]:
                print(f"           {'ok' if good else 'XX'} {s}")

    def rate(rs):
        return f"{sum(r['ok'] for r in rs)}/{len(rs)}" if rs else "-"

    free = [r for r in results if not r["blocked"]]
    print(f"\nall: {rate(results)}   lane free: {rate(free)}   lane blocked: "
          f"{rate([r for r in results if r['blocked']])}")
    for lo, hi in ((-1, 0.0), (0.0, 0.2), (0.2, 0.4), (0.4, 0.55), (0.55, 9)):
        rs = [r for r in free if lo <= r["depth"] < hi]
        if rs:
            print(f"  lane free, depth {max(lo, -0.03) * 100:3.0f}..{min(hi, 0.8) * 100:3.0f} cm: {rate(rs)}")
    if args.yaw:
        for lo, hi in ((0, 10), (10, 20), (20, 90)):
            rs = [r for r in free if lo <= abs(math.degrees(r["yaw"])) < hi]
            if rs:
                print(f"  lane free, |yaw| {lo}..{min(hi, args.yaw):.0f} deg: {rate(rs)}")
    print(f"sim time per trial: {np.mean([r['sim'] for r in results]):.0f} s")


if __name__ == "__main__":
    main()
