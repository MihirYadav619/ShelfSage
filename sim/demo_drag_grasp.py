"""Drag-then-grasp demo (Phase 5): a book lying in the middle of a table is first pulled to the
edge with the fingertips, then grasped around its overhanging spine and lifted.

Default: Calculus, 40 cm inside Table A. See skills/manipulation.py for the steps.

Usage (project root, venv active):
    python sim/demo_drag_grasp.py                    # viewer, real time
    python sim/demo_drag_grasp.py --speed 2          # twice as fast
    python sim/demo_drag_grasp.py --book db          # Databases, middle of Table E
    python sim/demo_drag_grasp.py --book control     # Table C: refuses, Signals and Systems lies
                                                     # in its lane to the edge
    python sim/demo_drag_grasp.py --method arm       # the arm pulls in strokes (default: base pulls)
    python sim/demo_drag_grasp.py --headless         # no window: run it and print a report
    python sim/demo_drag_grasp.py --weld             # weld (attach) grasp instead of friction only
"""
import math
import sys
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sim.robot import Robot  # noqa: E402
from skills.manipulation import (META, carry_target, drag_to_edge, pick_from_table_edge,  # noqa: E402
                                 table_frame, table_parking)

AISLE_X = 1.25      # free lane between tables B and C (straight-line driving until A* in Phase 6)


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def drive_to_table(robot, book, table, say, log):
    """Stowed, to the normal parking line of `table` in line with the book; tables in the north
    row (D-F) are reached through the aisle so the straight lines miss tables A-C."""
    say(f"drive to Table {table}")
    robot.stow()
    lateral = float(robot.d.body(f"book_{book}").xpos @ table_frame(table)[1])
    x, y, yaw = table_parking(robot, table, lateral, META["robot"]["park_clearance"]["table"])
    path = [(x, y, yaw)]
    if y > META["tables"]["A"]["parking"][0][1] + 0.5:
        path = [(AISLE_X, -3.0, yaw), (AISLE_X, y, yaw)] + path
    for wp in path:
        res = robot.go_to(*wp, tol_xy=0.02, tol_yaw=0.02)
        log.append((f"drive to {np.round(wp[:2], 2)}", res.success))


def run(robot, book, table, say, log, method="base"):
    drive_to_table(robot, book, table, say, log)
    res = drag_to_edge(robot, book, table, say=say, log=log, method=method)
    say(f"drag: {res.message}")
    if not res.success:
        return res
    res = pick_from_table_edge(robot, book, table, say=say, log=log)
    say(f"pick: {res.message}")
    if res.success:
        say("carry pose: turn the book upright above the chassis")
        p, f, left = carry_target(robot)
        r = robot.move_to(p, z_axis=f, y_axis=left, duration=2.5, allow=(f"book_{book}",))
        log.append(("carry pose", r is not None and r.success))
    return res


def main():
    headless = "--headless" in sys.argv
    speed = float(arg("--speed", 1.0))
    book = arg("--book", "calculus")
    method = arg("--method", "base")
    table = META["books"][book]["location"].split(":")[1]
    model = mujoco.MjModel.from_xml_path(str(ROOT / "scene/library.xml"))
    data = mujoco.MjData(model)
    robot = Robot(model, data, pose_source="true", grasp_mode="weld" if "--weld" in sys.argv else "friction")
    others0 = {model.body(i).name: data.xpos[i].copy() for i in range(model.nbody)
               if model.body(i).name.startswith("book_") and model.body(i).name != f"book_{book}"}
    p_book0 = data.body(f"book_{book}").xpos.copy()
    log = []
    say = lambda s: print(f"  - {s}", flush=True)  # noqa: E731
    print(f"ShelfSage drag-then-grasp demo: {META['books'][book]['title']} on Table {table}, "
          f"drag method '{method}'")

    if headless:
        run(robot, book, table, say, log, method)
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            tx, ty = META["tables"][table]["pos"]
            viewer.cam.lookat[:] = [tx, ty - 0.6, 0.8]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.6, -135, -30
            robot.viewer, robot.speed = viewer, speed
            try:
                print("(close the window to stop)")
                robot.wait(1.0)
                run(robot, book, table, say, log, method)
                print("  - done, close the window to exit")
                while viewer.is_running():
                    robot.wait(0.1)
            except KeyboardInterrupt:
                print("viewer closed")
                return

    moved = max(np.linalg.norm(data.body(n).xpos - p) for n, p in others0.items())
    log.append(("other books undisturbed (< 1 cm)", moved < 0.01))
    print("\nreport:")
    for name, ok in log:
        print(f"  {'ok ' if ok else 'XX '} {name}")
    print(f"  book started {np.round(p_book0, 3)}, other books moved at most {moved * 100:.2f} cm")
    print("RESULT:", "all ok" if all(ok for _, ok in log) else "FAILED")


if __name__ == "__main__":
    main()
