"""Plan, execute, and plan again after a failure (Phase 9).

    rec = run_with_replanning(command, planner, robot, max_replans=2, on_step=injector)

1. The planner gets the command and the current world state and returns a plan (or refuses).
2. The executor validates and runs it step by step; the first failing step stops it.
3. On a failure the world is read again (where every book really is now, what the robot holds,
   where it stands) and the planner is asked again with the same command plus a description
   of what happened: the steps that ran, the step that failed and the skill's message. The new
   plan starts from the current state. Up to `max_replans` times (README: 2-3).

`planner(command, world, context)` returns (plan or None, message). The LLM planner reads the
context; the rule-based baseline cannot use it and simply parses the command again against
the new world state (its templates resolve where the book is now), which is what a baseline
without a model of failures can do.
"""
import time
from dataclasses import asdict, dataclass, field

from planner.executor import Executor
from planner.llm import OBSERVED


@dataclass
class Attempt:
    plan: list
    planner_message: str
    success: bool
    failed_step: str = ""
    error: str = ""
    steps: list = field(default_factory=list)
    plan_time: float = 0.0


@dataclass
class ReplanResult:
    command: str
    success: bool            # the last plan ran to the end
    replans: int             # how many times it planned again after a failure
    message: str
    attempts: list
    sim_time: float = 0.0

    def to_dict(self):
        return asdict(self)


def describe(attempts):
    """What happened so far, for the planner: per earlier plan the steps that succeeded, the step
    that failed and why."""
    lines = []
    for k, a in enumerate(attempts, 1):
        done = [s["step"] for s in a.steps if s["ok"]]
        lines.append(f"Plan {k} was executed until step {len(done) + 1} failed.")
        if done:
            lines.append("  Steps that succeeded: " + "; ".join(done))
        lines.append(f"  Failed step: {a.failed_step}")
        lines.append(f"  Error: {a.error}")
    lines.append("The world state above is the current one, after the failure (it may differ from what "
                 "the earlier plan expected). Make a new plan that completes the original command from "
                 "this state: do not repeat steps whose effect is already there, and refuse if the "
                 "command can no longer be carried out.")
    return "\n".join(lines)


def observed(attempts):
    """What the LOOKs so far showed, for the planner (search mode, skills/search.py): the plan
    ended with LOOK(table), and the planner continues the command knowing what lies there."""
    lines = []
    for a in attempts:
        for st in a.steps:
            if st["step"].startswith("LOOK(") and st["ok"]:
                lines.append(f"{st['step']}: {st['msg']}.")
    lines.append("The world state above now includes what the camera saw (and the robot stands at the "
                 "last table it looked at). Plan the rest of the command from here; do not LOOK at a "
                 "table again whose looked_at is true.")
    return OBSERVED + "\n".join(lines)


def run_with_replanning(command, planner, robot, max_replans=2, on_step=None, say=lambda s: None, max_looks=3):
    """Plan, execute, and plan again: after a failure (up to max_replans times, with what went
    wrong) and after a plan that ends with LOOK (up to max_looks times, with what was seen)."""
    ex = Executor(robot, say=say)
    attempts, failures = [], []
    t0 = robot.d.time
    replans = looks = 0
    context = None
    while True:
        world = ex.world()
        s = time.time()
        plan, msg = planner(command, world, context)
        pt = round(time.time() - s, 2)
        if plan is None:
            attempts.append(Attempt([], msg, False, error=f"no plan: {msg}", plan_time=pt))
            return ReplanResult(command, False, replans, f"the planner gave no plan: {msg}",
                                [asdict(a) for a in attempts], round(robot.d.time - t0, 1))
        res = ex.run(plan, command=command, on_step=on_step)
        steps = [{"step": f"{st.action}({', '.join(st.args)})", "ok": st.success, "msg": st.message,
                  "sim_time": st.sim_time} for st in res.steps]
        failed = next((st for st in steps if not st["ok"]), None)
        a = Attempt([str(x) for x in plan], msg, res.success, failed["step"] if failed else "",
                    (failed["msg"] if failed else ("" if res.valid else res.message)), steps, pt)
        attempts.append(a)
        if res.success and plan and plan[-1].name == "LOOK" and looks < max_looks:
            looks += 1
            context = observed(attempts)
            continue
        if res.success:
            return ReplanResult(command, True, replans, "done" if replans == 0 else f"done after {replans} replan(s)",
                                [asdict(a) for a in attempts], round(robot.d.time - t0, 1))
        if not res.valid:
            a.failed_step, a.error = "(validator)", res.message
        if replans >= max_replans:
            return ReplanResult(command, False, replans, f"still failing after {max_replans} replans: {a.error}",
                                [asdict(a) for a in attempts], round(robot.d.time - t0, 1))
        replans += 1
        failures.append(a)
        context = describe(failures)
