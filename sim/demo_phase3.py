"""Watch the Phase 3 robot in the MuJoCo viewer.

Sequence: stow arm -> drive to Table B -> arm over the Deep Learning book -> dip, open/close
gripper, lift -> stow -> drive through the aisle to the shelf -> reach in front of empty slot B3
-> stow -> back to start. The arm is always stowed while driving.
LiDAR rays are drawn in red (every 4th ray). The base uses ground-truth pose here so the
motion is clean; odometry/MCL comes later.

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
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sim.robot import Robot, rot2  # noqa: E402
from sim.test_robot import META, table_parking  # noqa: E402


def ik(robot, target, z_axis, y_axis=None, seeds=80):
    """Temporary numerical IK (Phase 4 replaces it): TCP at target, tcp z-axis = z_axis,
    optionally the finger closing axis (tcp y) along +-y_axis. Of all exact solutions found,
    returns the one closest to the current arm configuration."""
    m = robot.m
    d = mujoco.MjData(m)
    d.qpos[:] = robot.d.qpos
    lo, hi = np.array([m.jnt_range[m.jnt_qposadr.tolist().index(a)] for a in robot.arm_qadr]).T
    z_axis = np.asarray(z_axis, float)

    def resid(q):
        d.qpos[robot.arm_qadr] = q
        mujoco.mj_kinematics(m, d)
        R = d.site_xmat[robot.tcp].reshape(3, 3)
        r = [d.site_xpos[robot.tcp] - target, 0.3 * (R[:, 2] - z_axis)]
        if y_axis is not None:
            r.append([0.3 * (1 - abs(R[:, 1] @ y_axis))])
        return np.concatenate(r)

    q_now = robot.arm_q()
    rng = np.random.default_rng(1)
    good, best = [], None
    for k in range(seeds):
        q0 = np.clip(q_now, lo, hi) if k == 0 else rng.uniform(lo, hi)
        sol = least_squares(resid, q0, bounds=(lo, hi), xtol=1e-10, ftol=1e-10)
        if best is None or sol.cost < best.cost:
            best = sol
        if np.linalg.norm(resid(sol.x)) < 2e-3:
            good.append(sol.x)
            if len(good) >= 5:
                break
    if not good:
        print("    (IK: no exact solution, using the closest one)")
        return best.x
    return min(good, key=lambda q: np.linalg.norm(q - q_now))


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
            print("ShelfSage Phase 3 demo (close the window to stop)")
            robot.wait(1.0)

            say("stow the arm for driving")
            robot.stow()
            say("driving to Table B")
            robot.go_to(*table_parking("B"))

            say("unfold the arm (home, elbow up)")
            robot.home(2.0)
            book = data.body("book_dl").xpos.copy()
            bd = META["book_dims"]
            top = META["tables"]["B"]["top_z"] + bd["rib_height"] + bd["thickness"]
            say("arm over the Deep Learning book (10 cm)")
            robot.move_arm(ik(robot, np.array([book[0], book[1], top + 0.10]), [0, 0, -1]), 2.5)
            say("dip to 2 cm, open and close the gripper, lift")
            robot.move_arm(ik(robot, np.array([book[0], book[1], top + 0.02]), [0, 0, -1]), 1.5)
            robot.close_gripper(0.8)
            robot.open_gripper(0.8)
            robot.move_arm(ik(robot, np.array([book[0], book[1], top + 0.15]), [0, 0, -1]), 1.5)
            say("stow the arm")
            robot.stow()

            say("driving through the aisle to the shelf")
            sh = META["shelf"]
            hx = META["robot"]["chassis_half"][0]
            # park ~30 cm from the shelf: the shelf grasp point is then ~0.45 m ahead of the arm
            # mount, in the middle of the reachable range (sim/reach_check.py)
            front = sh["pos"][1] - 0.35 / 2
            shelf_park = tuple(sh["parking"]["3"])          # fixed pose for column 3
            for wp in [(1.25, -2.6, math.pi / 2), (1.25, 1.7, math.pi / 2), shelf_park]:
                robot.go_to(*wp, tol_xy=0.02, tol_yaw=0.02)

            slot = np.array(sh["slots"]["B3"])
            bd = META["book_dims"]
            fwd = np.array([0.0, 1.0, 0.0])   # into the shelf
            # where a book's grasp point would be: 2 cm + 2.5 cm behind the shelf front, mid-height
            grasp = np.array([slot[0], front + 0.045, slot[2] + bd["length"] / 2 + 0.03])
            say("pre-place: 15 cm in front of empty slot B3")
            robot.move_arm(ik(robot, grasp - [0, 0.15, 0], fwd, y_axis=np.array([1.0, 0, 0])), 3.0)
            say("move in to where a book would be held, then back out")
            robot.move_arm(ik(robot, grasp, fwd, y_axis=np.array([1.0, 0, 0])), 2.0)
            robot.wait(0.5)
            robot.move_arm(ik(robot, grasp - [0, 0.15, 0], fwd, y_axis=np.array([1.0, 0, 0])), 2.0)
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
