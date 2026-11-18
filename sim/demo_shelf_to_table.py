"""Demo: three books stand side by side in shelf compartment S2-A1 (Physics, Kinematics,
Dynamics); the robot takes out the middle one and lays it on Table A.

    NAVIGATE(S2-A1) -> GRASP(Kinematics) -> NAVIGATE(Table A) -> PLACE(Kinematics, Table A)
    -> VERIFY(Kinematics, Table A)

The robot pose comes from MCL, the book pose from the wrist camera + detector (shown in the
viewer's top-right corner); the current step is shown top left. On the table the book lies
with its spine towards the robot, 3 cm past the edge, ready to be picked up again (cover down:
see place_on_table in skills/manipulation.py).

Usage (project root, venv active):
    python sim/demo_shelf_to_table.py              # viewer, real time
    python sim/demo_shelf_to_table.py --speed 2    # twice as fast
    python sim/demo_shelf_to_table.py --truth      # book poses from ground truth, no camera
"""
import sys
from pathlib import Path

import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sim.demo_extensions import CameraPanel  # noqa: E402
from sim.robot import Robot  # noqa: E402
from skills import interface as sk  # noqa: E402


def main():
    speed = float(sys.argv[sys.argv.index("--speed") + 1]) if "--speed" in sys.argv else 1.0
    use_camera = "--truth" not in sys.argv
    model = mujoco.MjModel.from_xml_path(str(ROOT / "scene/library.xml"))
    data = mujoco.MjData(model)
    robot = Robot(model, data, pose_source="mcl")
    from perception.detect import BookDetector
    det = BookDetector()
    if use_camera:
        from skills.perception import Perception
        robot.perception = Perception(det, say=lambda s: None)
    log = []
    print("ShelfSage demo: the middle book of S2-A1 (Kinematics) to Table A")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [-0.6, 2.1, 0.9]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.0, 125, -20
        panel = CameraPanel(robot, viewer, det, flip=False)
        state = {"follow": False}

        def on_sync():
            panel()
            if state["follow"]:
                x, y, _ = robot.true_pose()
                viewer.cam.lookat[:2] += 0.05 * (np_xy(x, y + 0.5) - viewer.cam.lookat[:2])

        robot.viewer, robot.speed, robot.on_sync = viewer, speed, on_sync
        sk.bind(robot, say=panel.say, log=log)
        try:
            print("(close the window to stop)")
            robot.wait(1.0)
            steps = (("NAVIGATE", sk.navigate, ("S2-A1",)), ("GRASP", sk.grasp, ("Kinematics",)),
                     ("NAVIGATE", sk.navigate, ("Table A",)), ("PLACE", sk.place, ("Kinematics", "Table A")),
                     ("VERIFY", sk.verify, ("Kinematics", "Table A")))
            for name, fn, args in steps:
                state["follow"] = name == "NAVIGATE"
                if name == "PLACE":
                    viewer.cam.lookat[:] = [-2.5, -2.4, 0.8]
                    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.6, -60, -30
                res = fn(*args)
                print(f"{'ok' if res.success else 'XX'} {name}({', '.join(args)}): {res.message}")
                if not res.success:
                    break
                if name == "NAVIGATE" and args[0] == "S2-A1":
                    viewer.cam.lookat[:] = [-0.75, 2.2, 0.9]
                    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.4, 125, -18
            panel.say("done, close the window to exit")
            while viewer.is_running():
                robot.wait(0.1)
        except KeyboardInterrupt:
            print("viewer closed")
    print("\nsteps:")
    for s, ok in log:
        print(f"  {'ok ' if ok else 'XX '} {s}")


def np_xy(x, y):
    import numpy as np
    return np.array([x, y])


if __name__ == "__main__":
    main()
