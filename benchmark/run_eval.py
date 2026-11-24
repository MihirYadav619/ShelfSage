"""Run a planner on the benchmark (benchmark/tasks.json) and score it.

Two levels:
    plan   no physics, seconds: the plan is validated and stepped through symbolically
           (planner/plan.py); the predicted final world is checked against the task's goal.
    exec   the plan is executed on the robot in MuJoCo from the reset scene (parallel processes);
           the real final world is checked. Slow (~1.5-3 min simulated per move).

Scoring (same for every planner):
    move tasks   success = goal holds in the final world AND every book not named in the goal
                 is still where it was (table / compartment); at plan level the plan must also be
                 valid, at exec level every step must succeed
    query        find: the plan looks that book up (FIND_BOOK) and moves nothing;
                 verify: the plan checks that book and target (VERIFY) and moves nothing, and the
                 answer (exec: the VERIFY result; plan: the world) is the expected one
    infeasible   success = the planner refuses (no plan) or the validator rejects the plan;
                 nothing runs
    failure      exec level only: the failure is injected while the plan runs, the planner may
                 plan again (up to 2 times); success = goal holds at the end, nothing else
                 disturbed, hand empty

Usage (project root, venv active):
    python benchmark/run_eval.py                          # baseline, plan level
    python benchmark/run_eval.py --level exec             # baseline, executed in the simulator
    python benchmark/run_eval.py --tasks S01 X01 --level exec -v
    python benchmark/run_eval.py --category spatial
    python benchmark/run_eval.py --planner llm            # Gemini (GEMINI_API_KEY) or Claude, plan level
    python benchmark/run_eval.py --planner llm --level exec --effort high

Results: benchmark/results/<planner>_<level>.json (one record per task) + a table per category.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TASKS = ROOT / "benchmark/tasks.json"
RESULTS = ROOT / "benchmark/results"
MOVES = ("GRASP", "PLACE")


# ---------------------------------------------------------------------------- planners
def baseline_planner(command, world, context=None):
    """-> (plan or None, message, extra info for the record). The baseline cannot use the
    failure context: after a failure it parses the command again against the new world."""
    from planner import baseline
    pr = baseline.parse(command, world)
    return pr.plan, pr.reason, {}


_LLM = {}


CACHE = ROOT / "sim/output/llm_cache"        # git-ignored: one file per answered request


def llm_planner(command, world, context=None):
    """LLM plan, cached: a command already answered for the same world by the same backend /
    model / effort is not asked again (free-tier daily limits). API errors are not cached."""
    import hashlib
    from planner import plan as P
    from planner.llm import LLMPlanner
    if "p" not in _LLM:
        _LLM["p"] = LLMPlanner(effort=os.environ.get("SHELFSAGE_EFFORT", "medium"))
    b = _LLM["p"].backend
    from planner.llm import SYSTEM
    key = json.dumps([b.name, b.model, b.effort, hashlib.sha1(SYSTEM.encode()).hexdigest(),
                      int(os.environ.get("SHELFSAGE_REPEAT", "1")), command,
                      world["books"], world["robot"], context], sort_keys=True)
    f = CACHE / (hashlib.sha1(key.encode()).hexdigest()[:20] + ".json")
    if f.exists():
        c = json.loads(f.read_text())
        plan = P.from_json(c["plan"]) if c["plan"] is not None else None
        return plan, c["message"], {**c["extra"], "cached": True}
    r = _LLM["p"].plan(command, world, context)
    extra = {"llm_status": r.status, "backend": r.backend, "model": r.model, "attempts": r.attempts,
             "latency": r.latency, "usage": r.usage, "transcript": r.transcript}
    if r.status != "error":
        CACHE.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"command": command, "plan": P.to_json(r.plan) if r.plan else None,
                                 "message": r.message, "extra": extra}, indent=1))
    return r.plan, r.message, extra


PLANNERS = {"baseline": baseline_planner, "llm": llm_planner}


# ---------------------------------------------------------------------------- scoring
def goal_holds(goal, world):
    """(ok, reason) for the goal conditions on a world-state dictionary."""
    for g in goal:
        s = world["books"][g["book"]]
        at, title = s["at"], s["title"]
        if "at" in g and at != g["at"]:
            return False, f"{title} is at {at}, wanted {g['at']}"
        if "at_one_of" in g and at not in g["at_one_of"]:
            return False, f"{title} is at {at}, wanted one of {', '.join(g['at_one_of'])}"
        if "shelf" in g and not (s["kind"] == "shelf" and s["where"].startswith(g["shelf"] + "-")):
            return False, f"{title} is at {at}, wanted shelf {g['shelf']}"
        if g.get("in_shelf") and s["kind"] != "shelf":
            return False, f"{title} is at {at}, wanted a shelf"
        if g.get("on_table") and s["kind"] != "table":
            return False, f"{title} is at {at}, wanted a table"
    return True, "goal holds"


def others_stay(task, w0, w1):
    named = {g["book"] for g in task.get("goal", [])}
    moved = [f"{w0['books'][b]['title']} ({w0['books'][b]['at']} -> {w1['books'][b]['at']})"
             for b in w0["books"] if b not in named and w0["books"][b]["at"] != w1["books"][b]["at"]]
    return (not moved), ("other books moved: " + ", ".join(moved)) if moved else ""


def query_plan_ok(task, plan, world):
    """Does the plan ask the right question (and move nothing)? (ok, reason)"""
    from planner import plan as P
    from planner import world as W
    if any(a.name in MOVES for a in plan):
        return False, "the plan moves books for a question"
    e = task["expect"]
    if "find" in e:
        hit = any(a.name == "FIND_BOOK" and a.args and W.book_id(world, a.args[0]) == e["find"] for a in plan)
        return hit, "looks it up" if hit else f"no FIND_BOOK of {world['books'][e['find']]['title']}"
    book, target, answer = e["verify"]
    hit = any(a.name == "VERIFY" and len(a.args) == 2 and W.book_id(world, a.args[0]) == book
              and P.canonical_target(a.args[1]) == target for a in plan)
    return hit, "checks it" if hit else f"no VERIFY({world['books'][book]['title']}, {target})"


def score_plan(task, plan, msg, world):
    """Plan-level score: dict(status, success, reason, plan, valid)."""
    from planner import plan as P
    out = {"plan": [str(a) for a in plan] if plan else [], "planner_message": msg}
    e = task.get("expect", {})
    if plan is None:
        out.update(valid=False, success=bool(e.get("refuse")), reason=f"no plan: {msg}")
        return out
    v = P.validate(plan, world)
    out["valid"], out["validation"] = v.ok, v.errors
    if e.get("refuse"):
        out.update(success=not v.ok, reason="validator rejected it" if not v.ok else "a valid plan for an infeasible task")
        return out
    if "find" in e or "verify" in e:
        ok, why = query_plan_ok(task, plan, world)
        out.update(success=ok and v.ok, reason=why if v.ok else "invalid: " + "; ".join(v.errors))
        return out
    if not v.ok:
        out.update(success=False, reason="invalid: " + "; ".join(v.errors))
        return out
    ok, why = goal_holds(task["goal"], v.world)
    stay, why2 = others_stay(task, world, v.world)
    out.update(success=ok and stay, reason=why if not ok else (why2 or why))
    return out


# ---------------------------------------------------------------------------- execution
def make_world_and_robot(perception=False, truth=False):
    from sim.run_command import make_robot
    from planner import world as W
    robot = make_robot(truth, perception)
    return robot, W.snapshot(robot)


def run_exec(task, planner, perception, truth):
    """Plan and execute one task from the reset scene (runs in a worker process)."""
    from planner import world as W
    from planner.executor import Executor
    robot, w0 = make_world_and_robot(perception, truth)
    t0 = time.time()
    try:
        plan, msg, extra = PLANNERS[planner](task["command"], w0)
    except Exception as e:                       # e.g. the free tier's daily quota: not run
        return {"plan": [], "valid": False, "success": None, "executed": False,
                "reason": f"not run: {type(e).__name__}: {e}"}
    plan_time = time.time() - t0
    out = score_plan(task, plan, msg, w0)          # plan-level result first
    out["plan_time"] = round(plan_time, 4)
    out.update(extra)
    e = task.get("expect", {})
    if plan is None or (not out["valid"] and "verify" not in e):
        out["executed"] = False
        return out
    if "verify" in e and not out["success"]:
        out["executed"] = False
        return out
    res = Executor(robot).run(plan, command=task["command"], planner=planner)
    w1 = W.snapshot(robot, truth=True)
    out.update(executed=True, steps=[{"step": f"{s.action}({', '.join(s.args)})", "ok": s.success, "msg": s.message,
                                     "sim_time": s.sim_time} for s in res.steps],
               sim_time=res.sim_time, wall_time=res.wall_time, exec_message=res.message)
    if "verify" in e:
        v = [s for s in res.steps if s.action == "VERIFY"]
        ok = bool(v) and v[-1].success == e["verify"][2]
        out.update(success=ok, reason=f"answered {'yes' if v and v[-1].success else 'no'}"
                   + ("" if ok else f", expected {'yes' if e['verify'][2] else 'no'}"))
        return out
    if "find" in e:
        out.update(success=res.success, reason=res.message)
        return out
    ok, why = goal_holds(task["goal"], w1)
    stay, why2 = others_stay(task, w0, w1)
    out.update(success=res.success and ok and stay,
               reason=res.message if not res.success else (why if not ok else (why2 or why)))
    return out


def run_failure(task, planner, perception=False, truth=False, max_replans=2):
    """A failure task: plan, execute with the failure injected (sim/failures.py), plan again
    after the failure (planner/replan.py, up to max_replans), score the final world. Success =
    the failure really happened, the goal holds in the end and no other book was disturbed
    (books the failure itself moved excepted)."""
    from planner import world as W
    from planner.replan import run_with_replanning
    from sim.failures import Injector
    robot, w0 = make_world_and_robot(perception, truth)
    inj = Injector(robot, task["inject"])

    def plan_fn(command, world, context):
        plan, msg, _ = PLANNERS[planner](command, world, context)
        return plan, msg
    try:
        rec = run_with_replanning(task["command"], plan_fn, robot, max_replans=max_replans, on_step=inj)
    except Exception as e:                       # e.g. the free tier's daily quota: not run
        return {"plan": [], "success": None, "executed": False, "injected": inj.done,
                "reason": f"not run: {type(e).__name__}: {e}"}
    w1 = W.snapshot(robot, truth=True)
    ok, why = goal_holds(task["goal"], w1)
    w0x = {**w0, "books": {b: s for b, s in w0["books"].items() if b not in inj.moved_by_failure}}
    w1x = {**w1, "books": {b: s for b, s in w1["books"].items() if b not in inj.moved_by_failure}}
    stay, why2 = others_stay(task, w0x, w1x)
    held = w1["robot"]["holding"]
    success = inj.done and ok and stay and held is None
    reason = ("the failure was never injected (no matching step)" if not inj.done else
              why if not ok else why2 if not stay else f"still holding {held}" if held else
              f"recovered: {rec.message}")
    return {"plan": rec.attempts[0]["plan"] if rec.attempts else [], "success": success, "executed": True,
            "injected": inj.done, "failure": inj.description, "replans": rec.replans,
            "run_success": rec.success, "run_message": rec.message, "attempts": rec.attempts,
            "sim_time": rec.sim_time, "reason": reason}


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", default="baseline", choices=sorted(PLANNERS))
    ap.add_argument("--level", default="plan", choices=["plan", "exec"])
    ap.add_argument("--tasks", nargs="*", help="only these task ids")
    ap.add_argument("--category", nargs="*")
    ap.add_argument("--perception", action="store_true", help="exec: book poses from the wrist camera")
    ap.add_argument("--truth", action="store_true", help="exec: ground-truth robot pose instead of MCL")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--effort", default="medium", help="llm: low | medium | high | xhigh | max")
    ap.add_argument("--repeat", type=int, default=1,
                    help="llm: repeat number (1, 2, ...): each repeat asks the model afresh; the plan "
                         "and exec levels of the same repeat share the answers")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    os.environ["SHELFSAGE_EFFORT"] = a.effort
    os.environ["SHELFSAGE_REPEAT"] = str(a.repeat)

    spec = json.loads(TASKS.read_text())
    tasks = [t for t in spec["tasks"] if (not a.tasks or t["id"] in a.tasks)
             and (not a.category or t["category"] in a.category)]
    run = [t for t in tasks if t["category"] != "failure"]
    failures = [t for t in tasks if t["category"] == "failure"]
    skipped = failures if a.level == "plan" else []

    t0 = time.time()
    if a.level == "plan":
        robot, w0 = make_world_and_robot()
        results = []
        from planner.llm import QuotaExhausted
        for t in run:
            s = time.time()
            try:
                plan, msg, extra = PLANNERS[a.planner](t["command"], w0)
            except QuotaExhausted as e:
                results.append({"plan": [], "success": None, "reason": f"not run: {e}"})
                print(f"  {t['id']} not run: {e}", flush=True)
                continue
            r = score_plan(t, plan, msg, w0)
            r["plan_time"] = round(time.time() - s, 4)
            r.update(extra)
            results.append(r)
            if a.planner != "baseline":
                print(f"  {t['id']} {'ok ' if r['success'] else 'XX '} {r['plan_time']:.1f} s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            results = list(ex.map(run_exec, run, [a.planner] * len(run), [a.perception] * len(run),
                                  [a.truth] * len(run)))
            fres = list(ex.map(run_failure, failures, [a.planner] * len(failures),
                               [a.perception] * len(failures), [a.truth] * len(failures)))
        run, results = run + failures, results + fres
    records = [{"id": t["id"], "category": t["category"], "command": t["command"], **r} for t, r in zip(run, results)]
    records += [{"id": t["id"], "category": t["category"], "command": t["command"], "success": None,
                 "reason": "skipped: failure tasks run at exec level only"} for t in skipped]

    print(f"{a.planner} planner, {a.level} level, {len(run)} tasks ({len(skipped)} failure tasks skipped)\n")
    for r in records:
        mark = "skip" if r["success"] is None else ("PASS" if r["success"] else "FAIL")
        print(f"  {mark}  {r['id']}  {r['command'][:70]:70s}  {r['reason'][:90]}")
        if a.verbose and r.get("plan"):
            print(f"              plan: {' -> '.join(r['plan'])}")
            for s in r.get("steps", []):
                print(f"              {'ok' if s['ok'] else 'XX'} {s['step']}  {s['sim_time']:.0f} s  {s['msg']}")
    cats = list(spec["categories"])
    print(f"\n  {'category':12s} {'tasks':>5s} {'plan':>5s} {'success':>8s}")
    tot = [0, 0, 0]
    for c in cats:
        rs = [r for r in records if r["category"] == c and r["success"] is not None]
        if not rs:
            n_skip = sum(1 for r in records if r["category"] == c)
            if n_skip:
                print(f"  {c:12s} {n_skip:5d}  skipped (exec level only)")
            continue
        n, p, s = len(rs), sum(1 for r in rs if r.get("plan")), sum(1 for r in rs if r["success"])
        tot = [tot[0] + n, tot[1] + p, tot[2] + s]
        print(f"  {c:12s} {n:5d} {p:5d} {s:5d}/{n:<2d} ({100 * s / n:3.0f} %)")
    print(f"  {'all':12s} {tot[0]:5d} {tot[1]:5d} {tot[2]:5d}/{tot[0]:<2d} ({100 * tot[2] / max(tot[0], 1):3.0f} %)"
          f"   ['plan' = a plan was produced]")
    RESULTS.mkdir(parents=True, exist_ok=True)
    if a.planner == "llm":
        from planner.llm import default_backend
        tag = f"_{default_backend()}_{a.effort}"
    else:
        tag = ""
    rep = f"_r{a.repeat}" if a.planner == "llm" else ""
    out = RESULTS / f"{a.planner}{tag}_{a.level}{'_perception' if a.perception else ''}{rep}.json"
    if a.tasks or a.category:
        out = out.with_name(out.stem + "_subset.json")
    llm = [r for r in records if r.get("latency") is not None]
    if llm:
        lat = sorted(r["latency"] for r in llm)
        tok_in = sum(r["usage"]["input"] + r["usage"]["cache_read"] + r["usage"]["cache_write"] for r in llm)
        print(f"\n  LLM: latency median {lat[len(lat) // 2]:.1f} s, max {lat[-1]:.1f} s; "
              f"{sum(r['attempts'] > 1 for r in llm)} tasks needed a retry; "
              f"tokens in {tok_in} (cache read {sum(r['usage']['cache_read'] for r in llm)}), "
              f"out {sum(r['usage']['output'] for r in llm)}")
    out.write_text(json.dumps({"planner": a.planner, "level": a.level, "repeat": a.repeat, "effort": a.effort if a.planner == "llm" else None, "perception": a.perception,
                               "robot_pose": "true" if a.truth else "mcl", "date": time.strftime("%Y-%m-%d %H:%M"),
                               "tasks_version": spec["version"], "wall_time": round(time.time() - t0, 1),
                               "results": records}, indent=1))
    print(f"\n  wrote {out.relative_to(ROOT)} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
