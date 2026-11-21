"""Phase 9 done-when test: failure injection and replanning.

The five failure tasks of the benchmark (benchmark/tasks.json, F01-F05) run in the simulator
(MCL pose) with each planner. The failure is injected once while the plan runs
(sim/failures.py); after a failing step the planner gets the new world state and, for the LLM,
a description of what happened, and plans again (planner/replan.py, at most 2 times).

    F01 drop_book           the book slips out on the way to the shelf, lands on a table
    F02 compartment_filled  the target compartment is full when the robot arrives
    F03 path_blocked        a cart stands in the target table's work zone
    F04 book_moved          the book was moved to another compartment before GRASP
    F05 grasp_slip          the book slips out while being pulled, stays in its compartment

A task counts as recovered when the failure happened, the goal holds at the end, the hand is
empty and no other book was disturbed. Done when the LLM planner recovers from at least 3
different failures (README: 2-3). The baseline runs the same failures for comparison: it
cannot read what went wrong and just parses the command again against the new world.

Usage (project root, venv active; the LLM needs GEMINI_API_KEY or ANTHROPIC_API_KEY, e.g. in .env):
    python sim/test_phase9.py                     # both planners, all five failures
    python sim/test_phase9.py --planners baseline
    python sim/test_phase9.py --tasks F02 F03
Results: sim/output/phase9/summary.json and benchmark/results/phase9_<planner>.json
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "sim/output/phase9"


def offline_checks():
    """The replanning loop with a scripted planner and executor (no physics)."""
    from types import SimpleNamespace as NS
    from planner import replan as R
    from planner.plan import Action
    checks = []
    a = R.Attempt(["NAVIGATE(Table B)", "PLACE(ML, S2-A3)"], "", False, "PLACE(ML, S2-A3)", "not holding ML",
                  [{"step": "NAVIGATE(Table B)", "ok": True}, {"step": "PLACE(ML, S2-A3)", "ok": False}])
    t = R.describe([a])
    checks.append(("failure description names the steps done, the failed step and the error",
                   "NAVIGATE(Table B)" in t and "Failed step: PLACE(ML, S2-A3)" in t and "not holding ML" in t))

    class Ex:                                   # fails the first run, succeeds the second
        runs = 0

        def __init__(self, robot, say=None):
            pass

        def world(self):
            return {}

        def run(self, plan, command="", on_step=None):
            Ex.runs += 1
            ok = Ex.runs > 1
            st = NS(action="PLACE", args=["ML", "S2-A3"], success=ok, message="" if ok else "not holding ML", sim_time=1.0)
            return NS(success=ok, valid=True, message="", steps=[st])
    contexts = []

    def planner(command, world, context):
        contexts.append(context)
        return [Action("PLACE", ["ML", "S2-A3"])], "ok"
    real = R.Executor
    R.Executor = Ex
    try:
        rec = R.run_with_replanning("cmd", planner, NS(d=NS(time=0.0)), max_replans=2)
    finally:
        R.Executor = real
    checks.append(("replans once after a failure, the second planner call gets the context",
                   rec.success and rec.replans == 1 and contexts[0] is None and "not holding ML" in contexts[1]))
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planners", nargs="*", default=["baseline", "llm"])
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    print("1. offline checks")
    final = []
    for name, ok in offline_checks():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        final.append((name, ok))

    from benchmark.run_eval import run_failure
    spec = json.loads((ROOT / "benchmark/tasks.json").read_text())
    tasks = [t for t in spec["tasks"] if t["category"] == "failure" and (not a.tasks or t["id"] in a.tasks)]
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for planner in a.planners:
        print(f"\n2. {planner}: {len(tasks)} failure tasks in the simulator")
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            res = list(ex.map(run_failure, tasks, [planner] * len(tasks)))
        for t, r in zip(tasks, res):
            print(f"\n  {'PASS' if r['success'] else 'FAIL'}  {t['id']} {t['inject']['type']:18s} {t['command']}")
            print(f"        failure: {r.get('failure') or '(not injected)'}")
            for k, at in enumerate(r.get("attempts", []), 1):
                print(f"        plan {k}: {' -> '.join(at['plan']) or '(none: ' + at['planner_message'][:100] + ')'}")
                if at["failed_step"]:
                    print(f"          failed at {at['failed_step']}: {at['error'][:130]}")
            print(f"        -> {r['reason'][:160]}  ({r.get('sim_time', 0):.0f} s simulated)")
        n = sum(r["success"] for r in res)
        kinds = {t["inject"]["type"] for t, r in zip(tasks, res) if r["success"]}
        print(f"\n  {planner}: recovered {n}/{len(tasks)} ({time.time() - t0:.0f} s wall)")
        summary[planner] = [{"id": t["id"], "failure_type": t["inject"]["type"], **r} for t, r in zip(tasks, res)]
        (ROOT / f"benchmark/results/phase9_{planner}.json").write_text(json.dumps(summary[planner], indent=1, default=str))
        if planner == "llm":
            final.append((f"LLM planner recovers from >= 3 different failures ({len(kinds)}: {', '.join(sorted(kinds))})",
                          len(kinds) >= 3))
        else:
            final.append((f"{planner}: recovered {n}/{len(tasks)} (for comparison)", True))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print()
    for name, ok in final:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("RESULT:", "all passed" if all(ok for _, ok in final) else "FAILED")


if __name__ == "__main__":
    main()
