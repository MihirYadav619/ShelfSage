"""Robot interface for ShelfSage (Phase 3).

The controller only ever uses get_pose(), never true_pose():
    get_pose()   pose estimate from the selected source: "odom" (default) or "true";
                 Phase 6 adds "mcl"
    true_pose()  simulator ground truth, for evaluation and debugging only

Sensors:
    odometry     base velocities in the robot frame with a per-axis scale bias and white
                 noise, integrated every physics step, so it drifts like wheel odometry
    lidar_scan() planar LiDAR via mj_multiRay from the 'lidar' site (270 deg, 541 rays, 8 m,
                 Gaussian range noise); the robot's own base is excluded
    get_rgbd()   wrist camera RGB (uint8) + depth (metres), rendered on demand

Example:
    import mujoco
    from sim.robot import Robot
    model = mujoco.MjModel.from_xml_path("scene/library.xml")
    data = mujoco.MjData(model)
    robot = Robot(model, data)          # resets to the 'home' keyframe
    robot.go_to(0.0, -2.28, 1.5708)
"""
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
META_PATH = ROOT / "scene/library_meta.json"

CONTROL_DT = 0.02          # controller period (50 Hz)


def wrap(a):
    """Wrap an angle to [-pi, pi)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def rot2(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


@dataclass
class DriveResult:
    success: bool
    message: str
    final_error_xy: float      # vs ground truth (evaluation only)
    final_error_yaw: float
    overshoot: float           # how far past the goal along the start->goal line (m)
    time: float
    collisions: list = field(default_factory=list)


class Robot:
    def __init__(self, model, data, seed=0, pose_source="odom", reset=True,
                 odom_scale_sigma=0.02, odom_vel_sigma=0.02, odom_yaw_sigma=0.02):
        self.m, self.d = model, data
        self.meta = json.loads(META_PATH.read_text())["robot"]
        self.rng = np.random.default_rng(seed)
        self.pose_source = pose_source

        self.qadr = [model.joint(j).qposadr[0] for j in self.meta["base_joints"]]
        self.vadr = [model.joint(j).dofadr[0] for j in self.meta["base_joints"]]
        self.base_act = [model.actuator(a).id for a in self.meta["base_actuators"]]
        self.arm_qadr = [model.joint(j).qposadr[0] for j in self.meta["arm_joints"]]
        self.arm_act = [model.actuator(a).id for a in self.meta["arm_actuators"]]
        self.grip_act = model.actuator(self.meta["gripper_actuator"]).id
        self.tcp = model.site(self.meta["tcp_site"]).id
        self.lidar_site = model.site(self.meta["lidar_site"]).id
        self.base_body = model.body("base").id
        self.robot_bodies = {b for b in range(model.nbody) if model.body_rootid[b] == self.base_body}
        self.floor_geom = model.geom("floor").id
        self.v_max = [abs(model.actuator_ctrlrange[a][1]) for a in self.base_act]

        # odometry noise: fixed scale error per axis (like wrong wheel radius) + white noise
        self.odom_scale = 1 + self.rng.normal(0, odom_scale_sigma, 3)
        self.odom_sigma = np.array([odom_vel_sigma, odom_vel_sigma, odom_yaw_sigma])

        li = self.meta["lidar"]
        fov = math.radians(li["fov_deg"])
        self.lidar_angles = np.linspace(-fov / 2, fov / 2, li["rays"])
        self.lidar_range, self.lidar_noise = li["range"], li["noise"]
        # see every geom group except 2 (Panda visual meshes; their collision meshes are group 3)
        self.lidar_groups = np.array([1, 1, 0, 1, 1, 1], dtype=np.uint8)

        self._renderer = None
        self.trace = None          # set to [] to record (true_pose, est_pose) every control step
        self.viewer = None         # set to a passive viewer to watch in real time (see demo)
        self.speed = 1.0           # real-time factor when a viewer is attached
        self.on_sync = None        # optional callback before each viewer sync (e.g. draw rays)
        self._sync = None
        if reset:
            self.reset()

    # ------------------------------------------------------------------ state
    def reset(self, key="home"):
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.m.key(key).id)
        mujoco.mj_forward(self.m, self.d)
        self.odom = np.array(self.true_pose())   # odometry starts at the known start pose

    def true_pose(self):
        """Ground truth (x, y, yaw). For evaluation only, the controller must not use it."""
        q = self.d.qpos
        return float(q[self.qadr[0]]), float(q[self.qadr[1]]), float(wrap(q[self.qadr[2]]))

    def get_pose(self):
        if self.pose_source == "true":
            return self.true_pose()
        if self.pose_source == "odom":
            return float(self.odom[0]), float(self.odom[1]), float(wrap(self.odom[2]))
        raise ValueError(f"unknown pose source {self.pose_source}")

    def tcp_pose(self):
        """TCP position (world) and rotation matrix (columns = tcp x, y, z axes)."""
        return self.d.site_xpos[self.tcp].copy(), self.d.site_xmat[self.tcp].reshape(3, 3).copy()

    def arm_q(self):
        return self.d.qpos[self.arm_qadr].copy()

    # ------------------------------------------------------------------ stepping
    def step(self, n=1):
        """Advance physics by n steps and integrate odometry."""
        dt = self.m.opt.timestep
        for _ in range(n):
            mujoco.mj_step(self.m, self.d)
            v_world = self.d.qvel[self.vadr[:2]]
            yaw_true = self.d.qpos[self.qadr[2]]
            v_body = rot2(yaw_true).T @ v_world            # what wheel encoders would see
            meas = np.array([*v_body, self.d.qvel[self.vadr[2]]]) * self.odom_scale
            meas += self.rng.normal(0, 1, 3) * self.odom_sigma
            self.odom[:2] += rot2(self.odom[2]) @ meas[:2] * dt
            self.odom[2] += meas[2] * dt
            if self.viewer is not None:
                self._viewer_sync()

    def _viewer_sync(self, fps=60.0):
        import time
        now = time.perf_counter()
        if self._sync is None:
            self._sync = (now, self.d.time, self.d.time)
        wall0, sim0, next_t = self._sync
        if self.d.time < next_t:
            return
        if not self.viewer.is_running():
            raise KeyboardInterrupt("viewer closed")
        if self.on_sync is not None:
            self.on_sync()
        self.viewer.sync()
        lag = wall0 + (self.d.time - sim0) / self.speed - time.perf_counter()
        if lag > 0:
            time.sleep(lag)
        self._sync = (wall0, sim0, next_t + self.speed / fps)

    def wait(self, seconds):
        self.step(int(round(seconds / self.m.opt.timestep)))

    # ------------------------------------------------------------------ base
    def send_cmd_vel(self, vx, vy, wz):
        """Velocity command in the robot frame (x forward, y left). Uses the *estimated* yaw to
        convert to the world-frame joint velocities, so heading error behaves like a real robot."""
        _, _, yaw = self.get_pose()
        vw = rot2(yaw) @ np.array([vx, vy])
        for a, v, vmax in zip(self.base_act, (*vw, wz), self.v_max):
            self.d.ctrl[a] = float(np.clip(v, -vmax, vmax))

    def stop(self):
        self.send_cmd_vel(0.0, 0.0, 0.0)

    def go_to(self, x, y, yaw, tol_xy=0.01, tol_yaw=0.01, v_max=0.5, w_max=1.0, a_max=0.8,
              kp_xy=1.5, kp_yaw=2.0, timeout=60.0):
        """Drive to (x, y, yaw) with a P-controller on get_pose(), speed and acceleration
        limited. Straight line only; A* waypoints come in Phase 6."""
        start = np.array(self.true_pose()[:2])
        goal = np.array([x, y])
        line = goal - start
        L = float(np.linalg.norm(line)) or 1.0
        n_ctrl = int(round(CONTROL_DT / self.m.opt.timestep))
        v_cmd = np.zeros(2)
        overshoot, t, collisions = 0.0, 0.0, set()
        while t < timeout:
            px, py, pyaw = self.get_pose()
            e_world = goal - np.array([px, py])
            e_yaw = wrap(yaw - pyaw)
            if np.linalg.norm(e_world) < tol_xy and abs(e_yaw) < tol_yaw and \
                    np.linalg.norm(self.d.qvel[self.vadr[:2]]) < 0.01:
                break
            v_des = rot2(pyaw).T @ (kp_xy * e_world)
            sp = np.linalg.norm(v_des)
            if sp > v_max:
                v_des *= v_max / sp
            dv = v_des - v_cmd                                  # acceleration limit
            lim = a_max * CONTROL_DT
            if np.linalg.norm(dv) > lim:
                dv *= lim / np.linalg.norm(dv)
            v_cmd = v_cmd + dv
            self.send_cmd_vel(*v_cmd, float(np.clip(kp_yaw * e_yaw, -w_max, w_max)))
            self.step(n_ctrl)
            t += CONTROL_DT
            p = np.array(self.true_pose()[:2])
            overshoot = max(overshoot, float((p - start) @ line / L - L))
            collisions |= self.robot_contacts()
            if self.trace is not None:
                self.trace.append((self.true_pose(), self.get_pose()))
        self.stop()
        self.wait(0.3)
        tx, ty, tyaw = self.true_pose()
        err_xy = float(np.hypot(tx - x, ty - y))
        err_yaw = float(abs(wrap(tyaw - yaw)))
        ok = t < timeout and not collisions
        msg = "reached" if ok else ("timeout" if t >= timeout else "collision")
        return DriveResult(ok, msg, err_xy, err_yaw, max(overshoot, 0.0), t, sorted(collisions))

    # ------------------------------------------------------------------ arm and gripper
    def move_arm(self, q, duration=2.0, settle=0.5):
        """Joint-space move with a smooth (smoothstep) profile on the position targets."""
        q0 = self.d.ctrl[self.arm_act].copy()
        q = np.asarray(q, dtype=float)
        n_ctrl = int(round(CONTROL_DT / self.m.opt.timestep))
        steps = max(1, int(duration / CONTROL_DT))
        for k in range(1, steps + 1):
            s = k / steps
            s = s * s * (3 - 2 * s)
            self.d.ctrl[self.arm_act] = q0 + s * (q - q0)
            self.step(n_ctrl)
        self.wait(settle)
        return float(np.max(np.abs(self.arm_q() - q)))

    def home(self, duration=2.0):
        return self.move_arm(self.meta["arm_home"], duration)

    def stow(self, duration=2.5):
        """Fold the arm back over the chassis before driving (nothing past the chassis front)."""
        return self.move_arm(self.meta["arm_stow"], duration)

    def open_gripper(self, wait=0.6):
        self.d.ctrl[self.grip_act] = 255.0
        self.wait(wait)

    def close_gripper(self, wait=0.6):
        self.d.ctrl[self.grip_act] = 0.0
        self.wait(wait)

    # ------------------------------------------------------------------ safety
    def robot_contacts(self):
        """Names of things the robot touches (anything except the floor), plus self-contacts."""
        out = set()
        m, d = self.m, self.d
        for c in d.contact[:d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            r1, r2 = b1 in self.robot_bodies, b2 in self.robot_bodies
            if not (r1 or r2) or self.floor_geom in (c.geom1, c.geom2):
                continue
            n1, n2 = m.body(b1).name, m.body(b2).name
            out.add(f"{n1} <-> {n2}" if r1 else f"{n2} <-> {n1}")
        return out

    # ------------------------------------------------------------------ sensors
    def lidar_scan(self, noise=True):
        """Return (angles in the robot frame, ranges in m). No return = range max (inf)."""
        site = self.lidar_site
        pnt = self.d.site_xpos[site].copy()
        R = self.d.site_xmat[site].reshape(3, 3)
        dirs_local = np.stack([np.cos(self.lidar_angles), np.sin(self.lidar_angles),
                               np.zeros_like(self.lidar_angles)], axis=1)
        vec = (dirs_local @ R.T).ravel()
        n = len(self.lidar_angles)
        geomid = np.zeros(n, dtype=np.int32)
        dist = np.zeros(n)
        mujoco.mj_multiRay(self.m, self.d, pnt, vec, self.lidar_groups, True, self.base_body,
                           geomid, dist, None, n, self.lidar_range)
        r = np.where((geomid >= 0) & (dist >= 0), dist, np.inf)
        if noise:
            hit = np.isfinite(r)
            r[hit] += self.rng.normal(0, self.lidar_noise, hit.sum())
        return self.lidar_angles.copy(), r

    def lidar_offset(self):
        """LiDAR position in the robot frame (x forward)."""
        return float(self.meta["lidar"]["x"]), 0.0

    def camera_intrinsics(self):
        cam = self.meta["wrist_cam"]
        w, h = cam["width"], cam["height"]
        fy = (h / 2) / math.tan(math.radians(cam["fovy"]) / 2)
        return np.array([[fy, 0, w / 2], [0, fy, h / 2], [0, 0, 1]])

    def get_rgbd(self):
        """Wrist camera image: RGB uint8 (H, W, 3) and depth in metres (H, W)."""
        cam = self.meta["wrist_cam"]
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.m, cam["height"], cam["width"])
        r = self._renderer
        r.update_scene(self.d, camera=self.meta["wrist_camera"])
        rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(self.d, camera=self.meta["wrist_camera"])
        depth = r.render().copy()
        r.disable_depth_rendering()
        return rgb, depth

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
