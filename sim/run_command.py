"""Run one natural-language command end to end with the rule-based baseline planner (Phase 7):
command -> baseline parser -> plan -> validator -> executor (the skills on the robot) -> log.

Usage (project root, venv active):
    python sim/run_command.py "Put Machine Learning in S2-A3"
    python sim/run_command.py "Move Physics to Table A" --viewer           # watch it (+ --speed 2)
    python sim/run_command.py "Return Linear Algebra to the shelf" --perception   # camera book poses
    python sim/run_command.py "Where is Statistics?" --plan-only           # parse + validate only

The robot pose comes from MCL (--truth: ground truth). The log goes to
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


def make_robot(truth=False, perception=False):
    model = mujoco.MjModel.from_xml_path(str(ROOT / "scene/library.xml"))
    data = mujoco.MjData(model)
    robot = Robot(model, data, pose_source="true" if truth else "mcl")
    if perception:
        from skills.perception import Perception
        robot.perception = Perception(say=lambda s: None)
    return robot


def run(command, robot, say=print, on_step=None, plan_only=False):
    """Plan the command with the baseline and execute it; returns the ExecutionResult (plan only:
    executes nothing; the result says whether the plan is valid)."""
    ex = Executor(robot, say=say)
    world = ex.world()
    t0 = time.time()
    pr = baseline.parse(command, world)
    plan_time = time.time() - t0
    if not pr.ok:
        from planner.executor import ExecutionResult
        res = ExecutionResult(command, "baseline", [], False, [pr.reason], False,
                              f"not understood: {pr.reason}", [], 0.0, 0.0, plan_time)
        return res
    if pr.reason:
        say(f"resolved: {pr.reason}")
    if plan_only:
        from planner.executor import ExecutionResult
        v = P.validate(pr.plan, world)
        return ExecutionResult(command, "baseline", [str(a) for a in pr.plan], v.ok, v.errors, v.ok,
                               "plan only" if v.ok else "invalid plan", [], 0.0, 0.0, plan_time)
    res = ex.run(pr.plan, command=command, planner="baseline", on_step=on_step)
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
    a = ap.parse_args()
    robot = make_robot(a.truth, a.perception)
    if not a.viewer:
        res = run(a.command, robot, say=lambda s: print("  " + s) if s.startswith("[") else None,
                  plan_only=a.plan_only)
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
                          on_step=on_step, plan_only=a.plan_only)
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
