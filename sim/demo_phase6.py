"""Navigation demo (Phase 6) in the viewer: A* paths, MCL localization and obstacle avoidance.

The robot localizes with MCL only (odometry + LiDAR against the map) and drives with NAVIGATE:
    1. start -> Table E, a cart stands in the gap between tables A and B (not in the map): the
       LiDAR finds it, the robot stops and goes through the other gap
    2. Table E -> shelf S2 (column 2)
    3. S2 -> Table C, a box is dropped onto the path 1.2 m ahead of the driving robot
Drawn: the planned path (green), the MCL particles (blue dots) and estimate (blue arrow), the
LiDAR rays (red), obstacle cells seen by the LiDAR (orange); top left the current step and the
MCL error against the true pose.

Usage (project root, venv active):
    python sim/demo_phase6.py              # viewer, real time
    python sim/demo_phase6.py --speed 2    # twice as fast
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
from skills import navigation as nav  # noqa: E402
from skills.manipulation import META  # noqa: E402

GREEN = np.array([0.1, 0.8, 0.2, 1.0], np.float32)
BLUE = np.array([0.15, 0.35, 1.0, 0.9], np.float32)
RED = np.array([1.0, 0.15, 0.1, 0.35], np.float32)
ORANGE = np.array([1.0, 0.55, 0.0, 0.8], np.float32)


def put(model, data, body, xy):
    adr = model.jnt_qposadr[model.body(body).jntadr[0]]
    data.qpos[adr:adr + 2] = xy
    data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)


def point_ahead(path, here, dist):
    """The point `dist` metres further along the polyline `path` than the point nearest `here`."""
    pts = [np.asarray(p, float) for p in path]
    best, at = np.inf, (0, 0.0)
    for i in range(len(pts) - 1):
        seg = pts[i + 1] - pts[i]
        L = float(np.linalg.norm(seg))
        u = float(np.clip((here - pts[i]) @ seg / max(L * L, 1e-12), 0, 1))
        d = float(np.linalg.norm(here - (pts[i] + u * seg)))
        if d < best:
            best, at = d, (i, u)
    i, u = at
    p = pts[i] + u * (pts[i + 1] - pts[i])
    left = dist
    for q in pts[i + 1:]:
        L = float(np.linalg.norm(q - p))
        if L >= left:
            return p + (q - p) * left / L
        left -= L
        p = q
    return pts[-1]


class Overlay:
    def __init__(self, robot, viewer):
        self.robot, self.viewer = robot, viewer
        self.frame = 0
        self.status = ""

    def say(self, s):
        print(f"  - {s}", flush=True)
        self.status = s

    def _geom(self, kind, size, pos, rgba, mat=None):
        scn = self.viewer.user_scn
        if scn.ngeom >= scn.maxgeom:
            return None
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, kind, np.asarray(size, float), np.asarray(pos, float),
                            np.eye(3).ravel() if mat is None else mat, rgba)
        scn.ngeom += 1
        return g

    def _line(self, a, b, rgba, width=3.0):
        g = self._geom(mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), rgba)
        if g is not None:
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, width, a, b)

    def __call__(self):
        self.frame += 1
        if self.frame % 6:
            return
        r, scn = self.robot, self.viewer.user_scn
        scn.ngeom = 0
        # planned path
        path = getattr(r, "nav_path", None)
        if path is not None:
            for a, b in zip(path[:-1], path[1:]):
                self._line([*a, 0.02], [*b, 0.02], GREEN, 5.0)
        # obstacle cells
        nm = getattr(r, "navmap", None)
        if nm is not None and nm.dynamic.any():
            iy, ix = np.nonzero(nm.dynamic)
            for i, j in zip(ix[:150], iy[:150]):
                x, y = nm.grid.to_world(i, j)
                self._geom(mujoco.mjtGeom.mjGEOM_BOX, [0.025, 0.025, 0.01], [x, y, 0.03], ORANGE)
        # LiDAR rays
        ang, rng = r.lidar_scan(noise=False)
        p0 = r.d.site_xpos[r.lidar_site].copy()
        R = r.d.site_xmat[r.lidar_site].reshape(3, 3)
        for a, d in zip(ang[::8], rng[::8]):
            if np.isfinite(d):
                self._line(p0, p0 + R @ np.array([math.cos(a), math.sin(a), 0.0]) * d, RED, 1.0)
        # MCL particles and estimate
        if r.mcl is not None:
            for p in r.mcl.p[::3]:
                self._geom(mujoco.mjtGeom.mjGEOM_SPHERE, [0.012, 0, 0], [p[0], p[1], 0.04], BLUE)
            x, y, yaw = r.get_pose()
            self._line([x, y, 0.05], [x + 0.4 * math.cos(yaw), y + 0.4 * math.sin(yaw), 0.05], BLUE, 6.0)
            tx, ty, _ = r.true_pose()
            err = math.hypot(tx - x, ty - y)
            sd = r.mcl.spread()
            self.viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                   f"{self.status}\nMCL error\nparticle spread",
                                   f"\n{err * 1000:.1f} mm\n{math.hypot(sd[0], sd[1]) * 1000:.1f} mm"))
        # camera follows the robot
        x, y, _ = r.true_pose()
        self.viewer.cam.lookat[:2] += 0.08 * (np.array([x, y + 0.4]) - self.viewer.cam.lookat[:2])


def main():
    speed = float(sys.argv[sys.argv.index("--speed") + 1]) if "--speed" in sys.argv else 1.0
    model = mujoco.MjModel.from_xml_path(str(ROOT / "scene/library.xml"))
    data = mujoco.MjData(model)
    robot = Robot(model, data, pose_source="mcl")
    nav.navmap(robot)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.0, -3.0, 0.5]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 5.5, -100, -50
        ov = Overlay(robot, viewer)
        robot.viewer, robot.speed, robot.on_sync = viewer, speed, ov
        try:
            print("ShelfSage navigation demo (close the window to stop)")
            robot.wait(1.0)

            put(model, data, "obstacle_cart", (-1.25, -2.0))
            ov.say("1. start -> Table E (a cart stands in the left gap, not in the map)")
            ok, msg = nav.navigate(robot, nav.parking_pose("Table E"), say=ov.say)
            print("   ", msg)

            ov.say("2. Table E -> shelf S2")
            ok, msg = nav.navigate(robot, nav.parking_pose("S2-A2"), say=ov.say)
            print("   ", msg)

            ov.say("3. S2 -> Table C (a box will drop onto the path)")
            state = {"done": False, "start": np.array(robot.true_pose()[:2])}
            orig = robot.step

            def step(n=1):
                orig(n)
                here = np.array(robot.true_pose()[:2])
                if not state["done"] and np.linalg.norm(here - state["start"]) > 0.8:
                    put(model, data, "obstacle_box", point_ahead(robot.nav_path, here, 1.2))
                    state["done"] = True

            robot.step = step
            ok, msg = nav.navigate(robot, nav.parking_pose("Table C"), say=ov.say)
            robot.step = orig
            print("   ", msg)
            ov.say("done, close the window to exit")
            while viewer.is_running():
                robot.wait(0.1)
        except KeyboardInterrupt:
            print("viewer closed")


if __name__ == "__main__":
    main()
