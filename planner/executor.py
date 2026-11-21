"""Skill executor: runs a validated plan step by step on the robot and logs every step.

    ex = Executor(robot)                         # binds skills/interface.py to the robot
    result = ex.run(plan, command="put ...")     # ExecutionResult
    result.save(path)                            # JSON log

The plan is validated against the current world state first and not started if invalid. Steps
run in order; the first failing step stops the run, and its message (from the skill's
SkillResult) is what goes back to a planner for replanning (Phase 9). Each step logs the action,
its arguments, success, the skill's message, simulated time and wall-clock time.
"""
import json
import time
from dataclasses import asdict, dataclass, field

from planner import plan as P
from planner import world as W
from skills import interface as sk

SKILL_FUNCS = {"FIND_BOOK": sk.find_book, "NAVIGATE": sk.navigate, "GRASP": sk.grasp,
               "PLACE": sk.place, "VERIFY": sk.verify}
assert set(SKILL_FUNCS) == set(P.SIGNATURES) == set(sk.SKILLS)


@dataclass
class StepLog:
    index: int
    action: str
    args: list
    success: bool
    message: str
    sim_start: float          # s, simulated time at the start of the step
    sim_time: float           # s, simulated duration
    wall_time: float          # s, real duration
    substeps: list = field(default_factory=list)   # (name, ok) of the skill's own checks


@dataclass
class ExecutionResult:
    command: str
    planner: str
    plan: list                # ["NAVIGATE(Table B)", ...]
    valid: bool
    validation: list          # validator errors (empty if valid)
    success: bool
    message: str
    steps: list
    sim_time: float
    wall_time: float
    plan_time: float = 0.0    # s, time the planner took (set by the caller)

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=1))

    def report(self):
        lines = [f"command : {self.command}", f"planner : {self.planner} ({self.plan_time * 1000:.0f} ms)",
                 f"plan    : {' -> '.join(self.plan) or '(none)'}"]
        if not self.valid:
            lines += [f"INVALID : {e}" for e in self.validation]
        for s in self.steps:
            lines.append(f"  {'ok' if s.success else 'XX'} {s.index:2d} {s.action}({', '.join(s.args)})"
                         f"  {s.sim_time:6.1f} s sim  {s.wall_time:6.1f} s wall  {s.message}")
        lines.append(f"result  : {'SUCCESS' if self.success else 'FAILED'}: {self.message}  "
                     f"({self.sim_time:.0f} s simulated, {self.wall_time:.0f} s wall)")
        return "\n".join(lines)


class Executor:
    def __init__(self, robot, say=None):
        self.robot = robot
        self.say = say or (lambda s: None)
        self.sublog = []
        sk.bind(robot, say=self.say, log=self.sublog)

    def world(self):
        return W.snapshot(self.robot)

    def run(self, plan, command="", planner="", on_step=None):
        """Validate, then execute the plan. on_step(index, action) is called before each step
        (e.g. to show it in the viewer)."""
        t0w, t0s = time.time(), self.robot.d.time
        v = P.validate(plan, self.world())
        res = ExecutionResult(command, planner, [str(a) for a in plan], v.ok, v.errors, False, "",
                              [], 0.0, 0.0)
        if not v.ok:
            res.message = "plan rejected by the validator: " + "; ".join(v.errors)
            return res
        ok, msg = True, "all steps succeeded"
        for i, a in enumerate(plan, 1):
            if on_step:
                on_step(i, a)
            self.say(f"[{i}/{len(plan)}] {a}")
            n0, s0, w0 = len(self.sublog), self.robot.d.time, time.time()
            try:
                r = SKILL_FUNCS[a.name](*a.args)
                success, message = bool(r.success), r.message
            except Exception as e:                       # a skill bug must not kill the run
                success, message = False, f"{type(e).__name__}: {e}"
            res.steps.append(StepLog(i, a.name, list(a.args), success, message, round(s0, 2),
                                     round(self.robot.d.time - s0, 2), round(time.time() - w0, 2),
                                     [[n, bool(k)] for n, k in self.sublog[n0:]]))
            if not success:
                ok, msg = False, f"step {i} {a} failed: {message}"
                break
        res.success, res.message = ok, msg
        res.sim_time = round(self.robot.d.time - t0s, 2)
        res.wall_time = round(time.time() - t0w, 2)
        return res
