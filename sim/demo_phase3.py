"""Watch the robot in the MuJoCo viewer (Phases 3 and 4).

Sequence: stow -> drive to Table B -> unfold -> pre-grasp in front of the Deep Learning book ->
straight-line approach (fingers around the overhanging spine) -> straight-line lift and back
out (the actual grasp is Phase 5) -> stow -> drive through the aisle to shelf S2 -> pre-place
in front of empty compartment S2-B3 -> straight-line in and out -> stow -> back to start.
The arm is always stowed while driving. Arm motions use our own IK (control/ik.py) and
straight-line Cartesian moves. LiDAR rays are drawn in red (every 4th ray). The base uses the
ground-truth pose here; odometry/MCL comes in Phase 6.

Usage (project root, venv active):
    python sim/demo_phase3.py              # real time
    python sim/demo_phase3.py --speed 2    # twice as fast
Close the viewer window to stop.
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
from sim.test_robot import META, table_parking  # noqa: E402

FWD, UP, SIDE = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
BD = META["book_dims"]


def grasp_point(data, book, where):
    """Middle of the book's thickness, grip_in[where] in from the spine."""
    b = data.body(f"book_{book}")
    return b.xpos - b.xmat.reshape(3, 3)[:, 1] * (BD["width"] / 2 - BD["grip_in"][where])


def draw_lidar(robot, viewer, every=4):
    ang, rng = robot.lidar_scan(noise=False)
    p0 = robot.d.site_xpos[robot.lidar_site].copy()
    R = robot.d.site_xmat[robot.lidar_site].reshape(3, 3)
    scn = viewer.user_scn
    scn.ngeom = 0
    for a, r in zip(ang[::every], rng[::every]):
        if not np.isfinite(r) or scn.ngeom >= scn.maxgeom:
            continue
        p1 = p0 + R @ np.array([math.cos(a), math.sin(a), 0.0]) * r
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.zeros(9),
                            np.array([1, 0.15, 0.1, 0.5], dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, 1.5, p0, p1)
        scn.ngeom += 1


def say(msg):
    print(f"  - {msg}", flush=True)


def main():
    speed = float(sys.argv[sys.argv.index("--speed") + 1]) if "--speed" in sys.argv else 1.0
    model = mujoco.MjModel.from_xml_path(str(ROOT / "scene/library.xml"))
    data = mujoco.MjData(model)
    robot = Robot(model, data, pose_source="true")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.6, -1.0, 0.6]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 6.0, -115, -28
        robot.viewer, robot.speed = viewer, speed
        frame = [0]

        def on_sync():
            frame[0] += 1
            if frame[0] % 6 == 0:          # refresh the rays 10x per second
                draw_lidar(robot, viewer)
            # camera follows the robot
            x, y, _ = robot.true_pose()
            viewer.cam.lookat[:2] += 0.05 * (np.array([x, y + 0.6]) - viewer.cam.lookat[:2])

        robot.on_sync = on_sync
        try:
            print("ShelfSage demo, Phases 3-4 (close the window to stop)")
            robot.wait(1.0)

            say("stow the arm for driving")
            robot.stow()
            say("driving to Table B")
            robot.go_to(*table_parking("B"))

            say("unfold the arm (home, elbow up)")
            robot.home(2.0)
            g = grasp_point(data, "dl", "table")
            robot.set_gripper_width(0.06, 0.3)
            say("pre-grasp: 10 cm in front of the Deep Learning book's spine")
            robot.move_to(g - 0.10 * FWD, z_axis=FWD, y_axis=UP, duration=2.5)
            say("straight line in: fingers above and below the overhanging spine")
            robot.move_linear(g, z_axis=FWD, y_axis=UP, speed=0.04)
            robot.wait(0.8)
            say("(Phase 5 closes the gripper here) - straight line back out")
            robot.move_linear(g - 0.10 * FWD, z_axis=FWD, y_axis=UP, speed=0.06)
            say("stow the arm")
            robot.stow()

            say("driving through the aisle to shelf S2")
            sh = META["shelves"]["S2"]                     # centre shelf
            for wp in [(1.25, -2.6, math.pi / 2), (1.25, 1.7, math.pi / 2), tuple(sh["parking"]["3"])]:
                robot.go_to(*wp, tol_xy=0.02, tol_yaw=0.02)

            # where a book in the first position of empty compartment S2-B3 would be held,
            # 3 cm higher (the carried book clears the board while it goes in)
            pos0 = np.array(sh["compartments"]["S2-B3"]["positions"][0])
            spine_y = pos0[1] - BD["width"] / 2
            g = np.array([pos0[0], spine_y + BD["grip_in"]["shelf"], pos0[2] + 0.03])
            robot.set_gripper_width(0.05, 0.3)
            say("pre-place: 15 cm in front of empty compartment S2-B3")
            robot.move_to(g - 0.15 * FWD, z_axis=FWD, y_axis=SIDE, duration=3.0)
            say("straight line in to where a book would be held, then back out")
            robot.move_linear(g, z_axis=FWD, y_axis=SIDE, speed=0.05)
            robot.wait(0.6)
            robot.move_linear(g - 0.15 * FWD, z_axis=FWD, y_axis=SIDE, speed=0.06)
            say("stow the arm")
            robot.stow()

            say("back to the start")
            start = META["robot"]["start"]
            for wp in [(1.25, 1.7, -math.pi / 2), (1.25, -2.6, -math.pi / 2), tuple(start)]:
                robot.go_to(*wp, tol_xy=0.02, tol_yaw=0.02)
            say("done - close the window to exit")
            while viewer.is_running():
                robot.wait(0.1)
        except KeyboardInterrupt:
            pass
    print("demo finished")


if __name__ == "__main__":
    main()
