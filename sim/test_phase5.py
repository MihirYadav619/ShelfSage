"""Phase 5 done-when test: a book goes from a table to shelf compartment S2-A3, at least 8 of 10
times, using only the skills (ground-truth book poses, friction grasp).

Every trial starts from the reset scene and runs
    NAVIGATE(Table X) -> GRASP(book) -> NAVIGATE(S2-A3) -> PLACE(book, S2-A3) -> VERIFY(book, S2-A3)
for one of 10 books on 6 tables (Calculus and Databases lie inside their tables, so GRASP drags
them to the edge first). Trial 11 puts two books into S2-A3 one after the other (next free
slot), trial 12 takes Physics out of shelf compartment S2-A1 and puts it into S2-A3. Trials run
in parallel processes.

Usage (project root, venv active):
    python sim/test_phase5.py              # all trials
    python sim/test_phase5.py dl calculus  # only these books
    python sim/test_phase5.py --weld       # weld grasp instead of friction
    python sim/test_phase5.py --perception # book poses from the wrist camera + YOLO detector
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

COMP = "S2-A3"
BOOKS = ["ml", "dl", "robotics", "linalg", "calculus", "signals", "python", "algo", "db", "thermo"]


def trial(books, weld=False, perception=False):
    import mujoco
    import numpy as np
    from sim.robot import Robot
    from skills import interface as sk
    from skills.manipulation import META

    model = mujoco.MjModel.from_xml_path(str(ROOT / "scene/library.xml"))
    data = mujoco.MjData(model)
    robot = Robot(model, data, pose_source="true", grasp_mode="weld" if weld else "friction")
    if perception:                               # book poses from the wrist camera + detector
        from skills.perception import Perception
        robot.perception = Perception(say=lambda s: None)
    others = [b for b in META["books"] if b not in books]
    p0 = {b: data.body(f"book_{b}").xpos.copy() for b in others}
    lines, log = [], []
    sk.bind(robot, say=lambda s: None, log=log)
    t0 = time.time()
    ok_all, failed_at = True, ""
    for book in books:
        kind, where = META["books"][book]["location"].split(":")
        source = f"Table {where}" if kind == "table" else where
        for name, fn, args in (("NAVIGATE", sk.navigate, (source,)), ("GRASP", sk.grasp, (book,)),
                               ("NAVIGATE", sk.navigate, (COMP,)), ("PLACE", sk.place, (book, COMP)),
                               ("VERIFY", sk.verify, (book, COMP))):
            res = fn(*args)
            lines.append(f"      {'ok' if res.success else 'XX'} {name}({', '.join(args)}): {res.message}")
            if not res.success:
                ok_all, failed_at = False, f"{name} {book}"
                break
        if not ok_all:
            break
    moved = max(float(np.linalg.norm(data.body(f"book_{b}").xpos - p)) for b, p in p0.items())
    return {"books": books, "ok": ok_all and moved < 0.01, "failed_at": failed_at or
            ("" if moved < 0.01 else f"other book moved {moved * 100:.1f} cm"),
            "moved": moved, "lines": lines, "sim_time": data.time, "wall": time.time() - t0}


def main():
    weld = "--weld" in sys.argv
    perception = "--perception" in sys.argv
    picked = [a for a in sys.argv[1:] if not a.startswith("--")]
    jobs = [[b] for b in (picked or BOOKS)]
    if not picked:
        jobs.append(["ml", "robotics"])          # two books into the same compartment
        jobs.append(["physics"])                 # shelf -> shelf
    print(f"Phase 5 test: {len(jobs)} trials into {COMP}, {'weld' if weld else 'friction'} grasp, "
          f"book poses from {'the detector' if perception else 'ground truth'}")
    with ProcessPoolExecutor(max_workers=4 if perception else 6) as ex:
        results = list(ex.map(trial, jobs, [weld] * len(jobs), [perception] * len(jobs)))
    for r in results:
        print(f"  {'PASS' if r['ok'] else 'FAIL'}  {' + '.join(r['books']):18s} sim {r['sim_time']:5.0f} s"
              f"{'' if r['ok'] else '   (' + r['failed_at'] + ')'}")
        for line in r["lines"]:
            print(line)
    single = [r for r in results if len(r["books"]) == 1 and r["books"][0] in BOOKS]
    n_ok = sum(r["ok"] for r in single)
    print(f"\nsingle-book trials: {n_ok}/{len(single)} ok (done when >= 8/10)")
    extra = [r for r in results if r not in single]
    for r in extra:
        print(f"{' + '.join(r['books'])}: {'ok' if r['ok'] else 'FAILED'}")
    print("RESULT:", "passed" if n_ok >= 0.8 * len(single) and all(r["ok"] for r in extra) else "failed")


if __name__ == "__main__":
    main()
