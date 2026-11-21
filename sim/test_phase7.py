"""Phase 7 done-when test: fixed commands run end to end with the rule-based baseline planner,
from navigation to placement (command -> parser -> plan -> validator -> executor -> skills).

1. Planner checks (no physics, instant): every template parses to the expected plan, vague
   targets resolve as documented, commands outside the templates are rejected with a reason, and
   the validator rejects bad plans (unknown action, wrong arguments, GRASP before NAVIGATE, PLACE
   without the book, a full compartment, a table without a free edge spot, a VERIFY that would
   fail).
2. Ten commands, each from the reset scene, robot pose from MCL: table -> shelf, shelf -> table,
   table -> table (with a drag), shelf -> shelf, two books, a whole shelf ("shelf S1"), "the
   shelf", a lookup and a check. Every step must succeed, none of the skills' own sub-step checks
   may fail (e.g. a base move that timed out), and no other book may move more than 1 cm (books
   the drag pushes out of the lane are allowed). Logs: sim/output/phase7/.

Usage (project root, venv active):
    python sim/test_phase7.py                 # both parts
    python sim/test_phase7.py --planner-only  # part 1 only
    python sim/test_phase7.py --perception    # book poses from the wrist camera + detector
    python sim/test_phase7.py --truth         # ground-truth robot pose instead of MCL
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "sim/output/phase7"

COMMANDS = [
    "Put Machine Learning in S2-A3",               # table -> shelf
    "Move Physics to Table A",                     # shelf -> table
    "Place Calculus on Table C",                   # table -> table, dragged to the edge first
    "Move Kinematics to S3-B2",                    # shelf -> shelf (middle book of three)
    "Put Deep Learning and Robotics into S2-A3",   # two books, one after the other
    "Bring Databases to Table D",                  # table -> table
    "Put Thermodynamics on shelf S1",              # a shelf: its first compartment with room
    "Return Linear Algebra to the shelf",          # any shelf: the nearest compartment with room
    "Where is Statistics?",                        # lookup
    "Check that Physics is in S2-A1",              # check
]


# ---------------------------------------------------------------------------- 1. planner checks
def planner_checks(world):
    from planner import baseline as B
    from planner import plan as P
    A = P.Action
    checks = []

    def parsed(cmd, plan_text=None, contains=None, ok=True):
        pr = B.parse(cmd, world)
        got = P.to_text(pr.plan) if pr.ok else None
        good = pr.ok == ok and (plan_text is None or got == plan_text) and \
            (contains is None or (got is not None and contains in got))
        if good and pr.ok:
            good = P.validate(pr.plan, world).ok
        checks.append((f"parse '{cmd}' -> {got or 'rejected: ' + pr.reason}", good))

    parsed("Put Machine Learning in S2-A3",
           "FIND_BOOK(Machine Learning) -> NAVIGATE(Table B) -> GRASP(Machine Learning) -> "
           "NAVIGATE(S2-A3) -> PLACE(Machine Learning, S2-A3) -> VERIFY(Machine Learning, S2-A3)")
    parsed("please put the ml book into shelf s2-a3.", contains="PLACE(Machine Learning, S2-A3)")
    parsed("Put Signals and Systems in S2-A3", contains="GRASP(Signals and Systems)")
    parsed("Put Deep Learning and Robotics into S2-A3", contains="PLACE(Robotics, S2-A3)")
    parsed("Put Thermodynamics on shelf S1", contains="PLACE(Thermodynamics, S1-A1)")
    parsed("Return Linear Algebra to the shelf", contains="PLACE(Linear Algebra, S1-A3)")
    parsed("Put Physics back on the shelf", "VERIFY(Physics, S2-A1)")
    parsed("Where is Statistics?", "FIND_BOOK(Statistics)")
    parsed("Check that Physics is in S2-A1", "VERIFY(Physics, S2-A1)")
    parsed("Take the book closest to Table D and place it in the empty shelf nearest to the window", ok=False)
    parsed("Put the machine learning textbook in S2-A3", ok=False)
    parsed("Put Machine Learning in S9-A1", ok=False)
    parsed("Dance", ok=False)

    def rejected(name, plan, expect):
        v = P.validate(plan, world)
        checks.append((f"validator rejects {name}: {v}", not v.ok and expect in str(v)))

    rejected("an unknown action", [A("FLY", ["Table B"])], "unknown action")
    rejected("a missing argument", [A("PLACE", ["Physics"])], "argument")
    rejected("an unknown book", [A("FIND_BOOK", ["Harry Potter"])], "no book called")
    rejected("an unknown target", [A("NAVIGATE", ["Table Z"])], "unknown target")
    rejected("GRASP before NAVIGATE", [A("GRASP", ["Machine Learning"])], "NAVIGATE first")
    rejected("PLACE without the book", [A("NAVIGATE", ["S2-A3"]), A("PLACE", ["Physics", "S2-A3"])],
             "GRASP it first")
    full = []
    for b, t in (("Machine Learning", "B"), ("Deep Learning", "B"), ("Robotics", "B")):
        full += [A("NAVIGATE", [f"Table {t}"]), A("GRASP", [b]), A("NAVIGATE", ["S2-A1"]), A("PLACE", [b, "S2-A1"])]
    rejected("a full compartment (3 books into S2-A1, which has room for 2)", full, "is full")
    rejected("a table without a free edge spot", [A("NAVIGATE", ["Table A"]), A("GRASP", ["Calculus"]),
                                                  A("NAVIGATE", ["Table B"]), A("PLACE", ["Calculus", "Table B"])],
             "no free edge spot")
    rejected("a VERIFY that would fail", [A("NAVIGATE", ["Table B"]), A("GRASP", ["Machine Learning"]),
                                          A("PLACE", ["Machine Learning", "Table B"]),
                                          A("VERIFY", ["Machine Learning", "S2-A3"])], "VERIFY would fail")
    v = P.validate([A("VERIFY", ["Machine Learning", "S2-A3"])], world)
    checks.append(("validator allows a question whose answer is no: VERIFY(Machine Learning, S2-A3)", v.ok))
    js = P.from_json('{"plan": [{"action": "navigate", "args": ["Table B"]}]}')
    checks.append(("plan JSON round trip", P.from_json(P.to_json(js)) == js and js[0].name == "NAVIGATE"))
    return checks


# ---------------------------------------------------------------------------- 2. end to end
def trial(command, perception=False, truth=False):
    import numpy as np
    from planner import baseline as B
    from sim.run_command import make_robot, run
    robot = make_robot(truth, perception)
    world0 = __import__("planner.world", fromlist=["snapshot"]).snapshot(robot)
    pr = B.parse(command, world0)
    moved_books = {b for b, s in world0["books"].items()
                   if pr.ok and any(s["title"] in a.args for a in pr.plan)}
    p0 = {b: robot.d.body(f"book_{b}").xpos.copy() for b in world0["books"]}
    res = run(command, robot, say=lambda s: None)
    pushed = set()
    for st in res.steps:
        for name, _ in st.substeps:
            for b, s in world0["books"].items():
                if name.startswith("pushed ") and s["title"] in name:
                    pushed.add(b)
    disturbed = {b: float(np.linalg.norm(robot.d.body(f"book_{b}").xpos - p0[b]))
                 for b in p0 if b not in moved_books and b not in pushed}
    worst = max(disturbed.items(), key=lambda kv: kv[1])
    return {"result": __import__("dataclasses").asdict(res), "worst_other": [worst[0], round(worst[1] * 100, 2)],
            "pushed": sorted(pushed)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner-only", action="store_true")
    ap.add_argument("--perception", action="store_true")
    ap.add_argument("--truth", action="store_true")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("commands", nargs="*", help="only these commands (default: all ten)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    import mujoco
    from planner import world as W
    from sim.robot import Robot
    m = mujoco.MjModel.from_xml_path(str(ROOT / "scene/library.xml"))
    world = W.snapshot(Robot(m, mujoco.MjData(m), pose_source="true"))
    checks = planner_checks(world)
    print("1. planner checks")
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    final = [(f"planner checks ({sum(ok for _, ok in checks)}/{len(checks)})", all(ok for _, ok in checks))]

    if not a.planner_only:
        cmds = a.commands or COMMANDS
        mode = ("detector" if a.perception else "ground-truth") + " book poses, " + ("true" if a.truth else "MCL") + " robot pose"
        print(f"\n2. {len(cmds)} commands end to end ({mode})")
        t0 = time.time()
        workers = a.workers or (3 if a.perception else 4)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            outs = list(ex.map(trial, cmds, [a.perception] * len(cmds), [a.truth] * len(cmds)))
        n_ok = 0
        for cmd, o in zip(cmds, outs):
            r = o["result"]
            stay = o["worst_other"][1] < 1.0
            bad_sub = [f"{s['action']}: {n}" for s in r["steps"] for n, k in s["substeps"] if not k]
            ok = r["success"] and stay and not bad_sub
            n_ok += ok
            print(f"\n  {'PASS' if ok else 'FAIL'}  {cmd}")
            print(f"        plan: {' -> '.join(r['plan']) or '(none)'}")
            for s in r["steps"]:
                print(f"        {'ok' if s['success'] else 'XX'} {s['action']}({', '.join(s['args'])})"
                      f"  {s['sim_time']:5.1f} s  {s['message']}")
            print(f"        {r['message']}; {r['sim_time']:.0f} s simulated, {r['wall_time']:.0f} s wall; "
                  f"other books moved <= {o['worst_other'][1]:.2f} cm ({o['worst_other'][0]})"
                  + (f"; pushed aside: {', '.join(o['pushed'])}" if o["pushed"] else ""))
            if bad_sub:
                print(f"        failed sub-steps: {'; '.join(bad_sub)}")
        (OUT / "summary.json").write_text(json.dumps({"mode": mode, "trials": [
            {"command": c, **o} for c, o in zip(cmds, outs)]}, indent=1))
        print(f"\n  {n_ok}/{len(cmds)} commands ok in {time.time() - t0:.0f} s wall; log: sim/output/phase7/summary.json")
        steps = [s for o in outs for s in o["result"]["steps"]]
        by = {}
        for s in steps:
            by.setdefault(s["action"], []).append(s)
        print("  per skill: " + ", ".join(f"{k} {sum(x['success'] for x in v)}/{len(v)} "
                                         f"(mean {sum(x['sim_time'] for x in v) / len(v):.0f} s)" for k, v in by.items()))
        final.append((f"fixed commands end to end ({n_ok}/{len(cmds)})", n_ok == len(cmds)))

    print()
    for name, ok in final:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("RESULT:", "all passed" if all(ok for _, ok in final) else "FAILED")


if __name__ == "__main__":
    main()
