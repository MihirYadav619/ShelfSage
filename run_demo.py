"""ShelfSage demo: a natural-language command, planned by the LLM (or the rule-based baseline),
executed by the robot in the MuJoCo viewer, with optional failure injection and replanning.

    python run_demo.py "Take the book on Table D and put it in the empty compartment nearest to the window"
    python run_demo.py "Put away Linear Algebra" --planner baseline        # the baseline refuses
    python run_demo.py --task F03                    # a benchmark task, its failure injected
    python run_demo.py --task X01 --speed 3
    python run_demo.py "Where is Statistics?" --headless
    python run_demo.py                               # interactive: type commands one after another
    python run_demo.py "Pick up the Algorithms book and place it in the shelf"   # searches for it

The robot does not know where loose books lie: FIND_BOOK drives to the tables and looks with the
wrist camera + detector until it sees the book (search mode; --known: book locations known in
advance, the benchmark setting, also the default for --task). Where each book belongs is in the
catalog (Book_Catalog.pdf), so "put it back in the shelf" means its home compartment.
The robot pose comes from MCL (LiDAR + odometry); grasp poses from simulator ground truth
(--perception: from the wrist camera + detector). The viewer shows the command, the plan, the
current step, injected failures and replans (top left). The LLM needs GEMINI_API_KEY (or
ANTHROPIC_API_KEY) in the environment or a git-ignored .env file (see README, Phase 8).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", nargs="?")
    ap.add_argument("--task", help="a benchmark task id (benchmark/tasks.json), e.g. X01 or F03")
    ap.add_argument("--planner", default="llm", choices=["llm", "baseline"])
    ap.add_argument("--speed", type=float, default=2.0)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--perception", action="store_true")
    ap.add_argument("--replans", type=int, default=2)
    ap.add_argument("--known", action="store_true", help="book locations known in advance (no search)")
    ap.add_argument("--search", action="store_true", help="search with the camera also for --task")
    ap.add_argument("--no-camera", action="store_true", help="no window with the wrist camera's view")
    ap.add_argument("--quiet", action="store_true", help="do not print the robot's own messages")
    a = ap.parse_args()

    task = None
    if a.task:
        task = next((t for t in json.loads((ROOT / "benchmark/tasks.json").read_text())["tasks"] if t["id"] == a.task), None)
        if task is None:
            sys.exit(f"no task {a.task} in benchmark/tasks.json")
    command = a.command or (task["command"] if task else None)
    interactive = command is None

    from benchmark.run_eval import PLANNERS, goal_holds
    from planner import world as W
    from planner.replan import run_with_replanning
    from sim.run_command import make_robot
    # search mode (books found with the camera) by default; benchmark tasks assume known book
    # locations ("the book on Table D"), so --task runs without search unless --search is given
    search = a.search or not (a.known or a.task)
    robot = make_robot(truth=False, perception=a.perception, search=search)
    inj = None
    if task and task.get("inject"):
        from sim.failures import Injector
        inj = Injector(robot, task["inject"], say=print)

    lines = {"plan": "", "step": "", "note": ""}

    def planner(cmd, world, context):
        lines["step"] = "planning..." if context is None else "replanning after the failure..."
        missing = W.missing_book(cmd) if context is None else None
        if missing:                       # answer at once: no planner, no search of the tables
            msg = f"there is no book called '{missing}' in this library (see Book_Catalog.pdf)"
            print(f"\n[catalog] {msg}")
            lines["plan"] = f"(no plan) {msg}"
            return None, msg
        plan, msg, extra = PLANNERS[a.planner](cmd, world, context)
        print(f"\n[{a.planner}] {'replan' if context else 'plan'}: {msg}")
        if plan:
            print("   " + " -> ".join(str(x) for x in plan))
            lines["plan"] = " -> ".join(str(x) for x in plan)
        else:
            lines["plan"] = f"(no plan) {msg}"
        return plan, msg

    def on_step(i, action):
        lines["step"] = f"step {i}: {action}"
        print(f"   [{i}] {action}")
        if inj is not None:
            inj(i, action)
            if inj.done and not lines["note"]:
                lines["note"] = "FAILURE: " + inj.description

    def say(text):
        """The skills' own messages (where the robot looks, what the camera saw, ...)."""
        text = text.strip()
        if text and not text.startswith("[") and not a.quiet:
            print("       " + text)

    def run():
        rec = run_with_replanning(command, planner, robot, max_replans=a.replans, on_step=on_step, say=say)
        if interactive:
            lines["step"] = ("SUCCESS" if rec.success else "FAILED") + f" - {rec.message}"
            print(f"\n{lines['step']}")
            return rec
        w = W.snapshot(robot, truth=True)
        verdict = "SUCCESS" if rec.success else "FAILED"
        if task and task.get("goal"):
            ok, why = goal_holds(task["goal"], w)
            verdict += f"; goal {'holds' if ok else 'not met'} ({why})"
        lines["step"] = f"{verdict} - {rec.message} ({rec.sim_time:.0f} s simulated)"
        print(f"\n{lines['step']}")
        return rec

    if not interactive:
        print(f"ShelfSage: \"{command}\"  ({a.planner} planner)")
    if a.headless and interactive:
        sys.exit("--headless needs a command")
    if a.headless:
        run()
        return
    import mujoco
    import mujoco.viewer
    with mujoco.viewer.launch_passive(robot.m, robot.d) as viewer:
        viewer.cam.lookat[:] = [0.0, -0.8, 0.8]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 9.0, 90, -38

        def wrap(t, n=110):
            return "\n".join(t[i:i + n] for i in range(0, len(t), n))

        def on_sync():
            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                              f"{a.planner}: \"{command}\"\nplan: {wrap(lines['plan'])}\n{lines['step']}\n{lines['note']}", ""))

        cam = None
        if not a.no_camera:
            from sim.camera_window import CameraWindow
            cam = CameraWindow(robot)

        def on_sync_all():
            on_sync()
            if cam is not None:
                cam.live()

        robot.viewer, robot.speed, robot.on_sync = viewer, a.speed, on_sync_all
        try:
            if not interactive:
                run()
            else:
                print(f"ShelfSage interactive ({a.planner} planner). Type a command and press Enter; "
                      "empty line or 'quit' to stop. The library keeps its state between commands.")
                # the terminal is read in its own thread, so the camera window stays live while
                # the prompt waits
                import queue
                import threading
                typed, want = queue.Queue(), threading.Event()

                def reader():
                    while True:
                        want.wait()
                        want.clear()
                        try:
                            typed.put(input("\ncommand> ").strip())
                        except EOFError:
                            typed.put(None)
                            return
                threading.Thread(target=reader, daemon=True).start()
                while viewer.is_running():
                    lines.update(plan="", step="waiting for a command (type it in the terminal)", note="")
                    robot.wait(0.05)
                    want.set()
                    command = None
                    while viewer.is_running():
                        try:
                            command = typed.get(timeout=0.03)
                            break
                        except queue.Empty:
                            if cam is not None:
                                cam.pump(1)
                    if not command or command.lower() in ("quit", "exit", "q"):
                        break
                    run()
            while viewer.is_running() and not interactive:
                robot.wait(0.1)
        except KeyboardInterrupt:
            print("viewer closed")


if __name__ == "__main__":
    main()
