"""Run one natural-language command end to end with the rule-based baseline planner (Phase 7):
command -> baseline parser -> plan -> validator -> executor (the skills on the robot) -> log.

Usage (project root, venv active):
    python sim/run_command.py "Put Machine Learning in S2-A3"
    python sim/run_command.py "Move Physics to Table A" --viewer           # watch it (+ --speed 2)
    python sim/run_command.py "Return Linear Algebra to the shelf" --perception   # camera book poses
    python sim/run_command.py "Where is Statistics?" --plan-only           # parse + validate only
    python sim/run_command.py "Put away Linear Algebra" --planner llm      # Claude plans (Phase 8)

The robot finds books with its wrist camera (search mode, skills/search.py; --known: book
locations known in advance, as in the benchmark). The robot pose comes from MCL (--truth: ground
truth). The log goes to
sim/output/phase7/logs/<time>_<command>.json.
"""
import argparse
import re
import sys
import time
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from planner import baseline  # noqa: E402
from planner import plan as P  # noqa: E402
from planner.executor import Executor  # noqa: E402
from sim.robot import Robot  # noqa: E402

LOG_DIR = ROOT / "sim/output/phase7/logs"


def make_robot(truth=False, perception=False, search=False):
    """search: the robot does not know where loose books lie and finds them with the camera
    (skills/search.py); otherwise book locations are known (the benchmark setting)."""
    model = mujoco.MjModel.from_xml_path(str(ROOT / "scene/library.xml"))
    data = mujoco.MjData(model)
    robot = Robot(model, data, pose_source="true" if truth else "mcl")
    if perception:
        from skills.perception import Perception
        robot.perception = Perception(say=lambda s: None)
    if search:
        from skills import search as S
        S.enable(robot)
    return robot


class _Parsed:
    def __init__(self, plan, reason):
        self.plan, self.reason = plan, reason

    @property
    def ok(self):
        return self.plan is not None


def run(command, robot, say=print, on_step=None, plan_only=False, planner="baseline"):
    """Plan the command (baseline or llm) and execute it; returns the ExecutionResult (plan only:
    executes nothing; the result says whether the plan is valid)."""
    ex = Executor(robot, say=say)
    world = ex.world()
    t0 = time.time()
    from planner.world import missing_book
    missing = missing_book(command)
    if missing:                           # no such book: answer at once, nothing moves
        from planner.executor import ExecutionResult
        why = f"there is no book called '{missing}' in this library (see Book_Catalog.pdf)"
        return ExecutionResult(command, planner, [], False, [why], False, why, [], 0.0, 0.0, 0.0)
    if planner == "llm":
        from planner.llm import LLMPlanner
        r = LLMPlanner().plan(command, world)
        pr = _Parsed(r.plan, r.message)
    else:
        pr = baseline.parse(command, world)
    plan_time = time.time() - t0
    if not pr.ok:
        from planner.executor import ExecutionResult
        res = ExecutionResult(command, planner, [], False, [pr.reason], False,
                              f"not understood: {pr.reason}", [], 0.0, 0.0, plan_time)
        return res
    if pr.reason:
        say(f"[planner] {pr.reason}")
    if plan_only:
        from planner.executor import ExecutionResult
        v = P.validate(pr.plan, world)
        return ExecutionResult(command, planner, [str(a) for a in pr.plan], v.ok, v.errors, v.ok,
                               "plan only" if v.ok else "invalid plan", [], 0.0, 0.0, plan_time)
    res = ex.run(pr.plan, command=command, planner=planner, on_step=on_step)
    res.plan_time = round(plan_time, 4)
    return res


def log_path(command):
    slug = re.sub(r"[^a-z0-9]+", "_", command.lower()).strip("_")[:50]
    return LOG_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{slug}.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--truth", action="store_true", help="ground-truth robot pose instead of MCL")
    ap.add_argument("--perception", action="store_true", help="book poses from the wrist camera")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--planner", default="baseline", choices=["baseline", "llm"])
    ap.add_argument("--known", action="store_true",
                    help="book locations known in advance (default: the robot searches with its camera)")
    a = ap.parse_args()
    robot = make_robot(a.truth, a.perception, search=not a.known)
    if not a.viewer:
        res = run(a.command, robot, say=lambda s: print("  " + s) if s.startswith("[") else None,
                  plan_only=a.plan_only, planner=a.planner)
    else:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(robot.m, robot.d) as viewer:
            viewer.cam.lookat[:] = [0.0, -0.8, 0.8]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 8.5, 90, -35
            state = {"text": a.command, "step": ""}

            def on_sync():
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                  state["text"], state["step"]))

            robot.viewer, robot.speed, robot.on_sync = viewer, a.speed, on_sync

            def on_step(i, act):
                state["step"] = f"step {i}: {act}"

            try:
                res = run(a.command, robot, say=lambda s: print("  " + s) if s.startswith("[") else None,
                          on_step=on_step, plan_only=a.plan_only, planner=a.planner)
                state["step"] = ("SUCCESS" if res.success else "FAILED") + " - close the window to exit"
                while viewer.is_running():
                    robot.wait(0.1)
            except KeyboardInterrupt:
                if "res" not in locals():
                    print("viewer closed before the command finished")
                    sys.exit(1)
    print(res.report())
    if not a.plan_only:
        p = log_path(a.command)
        res.save(p)
        print(f"log: {p.relative_to(ROOT)}")
    sys.exit(0 if res.success else 1)


if __name__ == "__main__":
    main()
