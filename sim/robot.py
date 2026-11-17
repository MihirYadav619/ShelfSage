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

import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from control.ik import IKResult, Target, solve, solve_multi  # noqa: E402
from control.kinematics import PandaKinematics  # noqa: E402
META_PATH = ROOT / "scene/library_meta.json"

CONTROL_DT = 0.02          # controller period (50 Hz)
MCL_DT = 0.1               # MCL update period while driving (s)
MCL_MIN_MOVE = 0.003       # ... only after the odometry moved this far (m / rad)
PREFIX_HAND = "panda_hand"


def wrap(a):
    """Wrap an angle to [-pi, pi)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def rot2(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


@dataclass
class LinearMoveResult:
    success: bool
    max_deviation: float       # measured TCP distance from the straight line (m)
    final_error: float         # measured TCP distance from the goal (m)
    ik_failures: int           # waypoints the IK could not track exactly
    aborted: bool = False      # stopped because the line left the reachable workspace


@dataclass
class GuardedMoveResult:
    success: bool              # contact force reached (at least half of it after settling)
    distance: float            # how far the TCP was commanded before the force was reached (m)
    force: float               # normal finger contact force after settling (N)


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
    def __init__(self, model, data, seed=0, pose_source="odom", reset=True, grasp_mode="friction",
                 odom_scale_sigma=0.02, odom_vel_sigma=0.02, odom_yaw_sigma=0.02):
        self.m, self.d = model, data
        self.grasp_mode = grasp_mode     # "friction" (fingers only) or "weld" (attach fallback)
        self.holding = None              # book id held since grasp(), until release()
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
        rear = self.meta.get("lidar_rear_site")
        self.lidar_rear_site = model.site(rear).id if rear else None     # Phase 6: second LiDAR
        self.base_body = model.body("base").id
        self.robot_bodies = {b for b in range(model.nbody) if model.body_rootid[b] == self.base_body}
        hand_prefix = self.meta["arm_joints"][0].replace("joint1", "")
        self.finger_bodies = {model.body(hand_prefix + f).id for f in ("left_finger", "right_finger")}
        self.floor_geom = model.geom("floor").id
        self.v_max = [abs(model.actuator_ctrlrange[a][1]) for a in self.base_act]
        self.kin = PandaKinematics((self.meta["mount_x"], 0.0, self.meta["mount_height"]))
        self._scratch = mujoco.MjData(model)          # for collision checks of IK solutions

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
        self.hold = None           # base pose (x, y, yaw) held while parked, see hold_base()
        self.mcl = None            # pose_source "mcl": the particle filter (control/mcl.py)
        self.mcl_log = []          # (time, true pose, MCL estimate, spread) after every update
        if reset:
            self.reset()

    # ------------------------------------------------------------------ state
    def reset(self, key="home"):
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.m.key(key).id)
        mujoco.mj_forward(self.m, self.d)
        self.odom = np.array(self.true_pose())   # odometry starts at the known start pose
        self.hold = None
        if self.pose_source == "mcl":
            self.start_mcl(self.true_pose())

    def start_mcl(self, pose):
        """Particle filter around a known pose (the start pose; MCL tracks from there)."""
        from control.mcl import MCL
        from control.occupancy import exact_distance_grid
        if getattr(self, "_mcl_grid", None) is None:
            meta = json.loads(META_PATH.read_text())
            hx, hy = (s / 2 for s in meta["room_size"])
            cx, cy = meta["room_center"]
            lz = self.d.site_xpos[self.lidar_site][2]
            self._mcl_grid = exact_distance_grid(
                self.m, self.d, lz - 0.05, lz + 0.05, res=0.01,
                bounds=(cx - hx - 0.3, cx + hx + 0.3, cy - hy - 0.3, cy + hy + 0.3), exclude_root=self.base_body)
        self.mcl = MCL(self._mcl_grid, self.lidar_offset(), pose, seed=int(self.rng.integers(1 << 30)))
        self._mcl_odom = self.odom.copy()       # odometry at the last MCL update
        self._mcl_next = self.d.time + MCL_DT

    def _mcl_step(self):
        """Predict with the odometry increment since the last update, correct with a scan."""
        if self.d.time < self._mcl_next:
            return
        self._mcl_next = self.d.time + MCL_DT
        d = self.odom - self._mcl_odom
        c, s = math.cos(self._mcl_odom[2]), math.sin(self._mcl_odom[2])
        delta = (c * d[0] + s * d[1], -s * d[0] + c * d[1], float(wrap(d[2])))
        if math.hypot(delta[0], delta[1]) < MCL_MIN_MOVE and abs(delta[2]) < MCL_MIN_MOVE:
            return                      # standing: no new information, do not grow overconfident
        self.mcl.predict(delta)
        self.mcl.update(*self.lidar_scan())
        self._mcl_odom = self.odom.copy()
        self.mcl_log.append((self.d.time, self.true_pose(), tuple(self.mcl.est), tuple(self.mcl.spread())))

    def true_pose(self):
        """Ground truth (x, y, yaw). For evaluation only, the controller must not use it."""
        q = self.d.qpos
        return float(q[self.qadr[0]]), float(q[self.qadr[1]]), float(wrap(q[self.qadr[2]]))

    def get_pose(self):
        if self.pose_source == "true":
            return self.true_pose()
        if self.pose_source == "odom":
            return float(self.odom[0]), float(self.odom[1]), float(wrap(self.odom[2]))
        if self.pose_source == "mcl":
            # last MCL estimate + the odometry increment since (as a map -> odom correction)
            d = self.odom - self._mcl_odom
            c, s = math.cos(self._mcl_odom[2]), math.sin(self._mcl_odom[2])
            lx, ly = c * d[0] + s * d[1], -s * d[0] + c * d[1]
            x, y, yaw = self.mcl.est
            ce, se = math.cos(yaw), math.sin(yaw)
            return float(x + ce * lx - se * ly), float(y + se * lx + ce * ly), float(wrap(yaw + d[2]))
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
            if self.hold is not None:
                self._hold_base()
            mujoco.mj_step(self.m, self.d)
            v_world = self.d.qvel[self.vadr[:2]]
            yaw_true = self.d.qpos[self.qadr[2]]
            v_body = rot2(yaw_true).T @ v_world            # what wheel encoders would see
            meas = np.array([*v_body, self.d.qvel[self.vadr[2]]]) * self.odom_scale
            meas += self.rng.normal(0, 1, 3) * self.odom_sigma
            self.odom[:2] += rot2(self.odom[2]) @ meas[:2] * dt
            self.odom[2] += meas[2] * dt
            if self.mcl is not None:
                self._mcl_step()
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

    def hold_base(self, on=True):
        """Station keeping while parked (on: hold the current estimated pose; off: release).
        The velocity actuators alone only damp motion: a steady push (the arm dragging a book,
        ~5 N) made the base creep 1 cm/s, 13 cm in one stroke. Held, it stays within ~5 mm.
        Off by default: with fast arm swings the yaw loop rings (2 deg), so switch it on only
        for slow, forceful arm work. go_to() releases it."""
        self.hold = self.get_pose() if on else None
        if not on:
            self.stop()

    def _hold_base(self, kp_xy=4.0, kp_yaw=4.0, v_max=0.05, w_max=0.1):
        """One step of station keeping: P-control on the estimated pose back to `self.hold`."""
        x, y, yaw = self.get_pose()
        e = rot2(yaw).T @ (np.array(self.hold[:2]) - (x, y))
        sp = float(np.linalg.norm(e)) * kp_xy
        v = kp_xy * e * (min(1.0, v_max / sp) if sp > 0 else 1.0)
        self.send_cmd_vel(*v, float(np.clip(kp_yaw * wrap(self.hold[2] - yaw), -w_max, w_max)))

    def go_to(self, x, y, yaw, tol_xy=0.01, tol_yaw=0.01, v_max=0.5, w_max=1.0, a_max=0.8,
              kp_xy=1.5, kp_yaw=2.0, timeout=60.0):
        """Drive to (x, y, yaw) with a P-controller on get_pose(), speed and acceleration
        limited. Straight line only; A* waypoints come in Phase 6."""
        self.hold = None
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

    # ------------------------------------------------------------------ arm: IK and Cartesian moves
    def base_transform(self):
        """World <- base, from the *estimated* pose (localisation error shows up in the arm)."""
        x, y, yaw = self.get_pose()
        T = np.eye(4)
        T[:2, :2] = rot2(yaw)
        T[:2, 3] = (x, y)
        return T

    def to_base(self, p=None, v=None):
        """World point p / direction v -> base frame."""
        T = self.base_transform()
        if p is not None:
            return T[:3, :3].T @ (np.asarray(p, float) - T[:3, 3])
        return None if v is None else T[:3, :3].T @ np.asarray(v, float)

    def make_target(self, pos, z_axis=None, y_axis=None, R=None):
        """IK target in the base frame from world-frame position / axes."""
        Rb = None if R is None else self.base_transform()[:3, :3].T @ np.asarray(R, float)
        return Target(self.to_base(p=pos),
                      None if z_axis is None else self.to_base(v=z_axis),
                      None if y_axis is None else self.to_base(v=y_axis), Rb)

    def fk_world(self, q=None):
        """TCP pose in the world from our own FK (q defaults to the commanded arm targets)."""
        q = self.d.ctrl[self.arm_act] if q is None else q
        return self.base_transform() @ self.kin.fk(q)

    def arm_collisions(self, q, allow=()):
        """Contacts of the arm at configuration q (base and fingers as they are now). Contacts
        with bodies named in `allow` (e.g. the book being grasped) are ignored."""
        m, d = self.m, self._scratch
        d.qpos[:] = self.d.qpos
        d.qpos[self.arm_qadr] = q
        mujoco.mj_fwdPosition(m, d)
        allow_ids = {m.body(a).id for a in allow}
        out = []
        for c in d.contact[:d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            r1, r2 = b1 in self.robot_bodies, b2 in self.robot_bodies
            if not (r1 or r2) or self.floor_geom in (c.geom1, c.geom2) or {b1, b2} & allow_ids:
                continue
            if {b1, b2} <= self.finger_bodies:      # closed fingers touching each other
                continue
            out.append(f"{m.body(b1).name} <-> {m.body(b2).name}")
        return out

    def ik(self, pos, z_axis=None, y_axis=None, R=None, allow=(), seeds=None, n_random=8,
           reject_margin=None):
        """Collision-free IK for a world-frame TCP target. Returns an IKResult (q in result.q).
        reject_margin: drop solutions with a joint closer than that (rad) to its limit if possible."""
        q_now = self.d.ctrl[self.arm_act].copy()
        seeds = [self.meta["arm_home"]] if seeds is None else seeds
        return solve_multi(self.kin, self.make_target(pos, z_axis, y_axis, R), q_now, seeds=seeds,
                           n_random=n_random, rng=np.random.default_rng(0),
                           collision_fn=lambda q: self.arm_collisions(q, allow),
                           q_pref=self.meta["arm_home"], reject_margin=reject_margin)

    HOLD_JOINT_SPEED = 0.8     # rad/s peak while holding a book with the friction grasp
    HOLD_MARGIN = 0.2          # rad: while holding, avoid postures with a joint this close to a limit
                               # (from one at the limit the next move needed a 5 rad wrist swing)

    def move_to(self, pos, z_axis=None, y_axis=None, R=None, duration=2.0, allow=(), settle=0.4,
                reject_margin=None):
        """IK + joint-space move. Returns the IKResult; the arm does not move if IK fails.
        While holding a book with the friction grasp the move is slowed down to a peak joint
        speed of HOLD_JOINT_SPEED (a fast wrist swing turned the book in the fingers)."""
        if reject_margin is None and self.holding is not None and self.grasp_mode == "friction":
            reject_margin = self.HOLD_MARGIN
        r = self.ik(pos, z_axis, y_axis, R, allow=allow, reject_margin=reject_margin)
        if r is not None and r.success:
            if self.holding is not None and self.grasp_mode == "friction":
                dq = float(np.max(np.abs(r.q - self.d.ctrl[self.arm_act])))
                duration = max(duration, 1.5 * dq / self.HOLD_JOINT_SPEED)   # smoothstep peak 1.5x
            self.move_arm(r.q, duration, settle)
        return r

    def move_linear(self, pos, z_axis=None, y_axis=None, R=None, speed=0.05, settle=0.3):
        """Move the TCP along a straight line from where it is commanded now to `pos`, keeping the
        orientation constraints, with a smooth speed profile. Each control step runs a few
        warm-started DLS iterations (differential IK)."""
        p0 = self.fk_world()[:3, 3]
        p1 = np.asarray(pos, float)
        L = float(np.linalg.norm(p1 - p0))
        steps = max(1, int(np.ceil(L / speed / CONTROL_DT * 1.5)))   # 1.5: smoothstep peak speed
        n_ctrl = int(round(CONTROL_DT / self.m.opt.timestep))
        q = self.d.ctrl[self.arm_act].copy()
        fails, dev, aborted = 0, 0.0, False
        u = (p1 - p0) / L if L > 1e-9 else np.zeros(3)
        for k in range(1, steps + 1):
            s = k / steps
            s = s * s * (3 - 2 * s)
            r = solve(self.kin, self.make_target(p0 + s * (p1 - p0), z_axis, y_axis, R), q,
                      max_iter=8, tol_pos=2e-4, tol_ang=2e-3)
            fails += not r.success
            if not r.success and (r.pos_err > 0.01 or r.ang_err > 0.05):
                aborted = True                       # path leaves the workspace: stop, hold pose
                break
            q = r.q
            self.d.ctrl[self.arm_act] = q
            self.step(n_ctrl)
            # measured TCP (own FK of the measured joints on the estimated base pose) vs the ideal
            # line: the simulator's TCP would add the localization error, which the robot cannot
            # see and which this move does not cause
            p = self.fk_world(self.arm_q())[:3, 3] - p0
            dev = max(dev, float(np.linalg.norm(p - (p @ u) * u)))
        self.wait(settle)
        err = float(np.linalg.norm(self.fk_world(self.arm_q())[:3, 3] - p1))
        return LinearMoveResult(not aborted and err < 0.005, dev, err, fails, aborted)

    def finger_force(self, body=None):
        """Simulated fingertip force sensing (a real Franka estimates it from the joint torques):
        total (normal, tangential) contact force in N on the two fingers, from contacts with the
        body named `body` only, or with anything outside the robot."""
        m, d = self.m, self.d
        only = None if body is None else m.body(body).id
        f6, fn, ft = np.zeros(6), 0.0, 0.0
        for i, c in enumerate(d.contact[:d.ncon]):
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if not ({b1, b2} & self.finger_bodies):
                continue
            other = b2 if b1 in self.finger_bodies else b1
            if other in self.robot_bodies or (only is not None and other != only):
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            fn += f6[0]
            ft += float(np.linalg.norm(f6[1:3]))
        return fn, ft

    def object_contact_force(self, body):
        """Total normal contact force (N) between `body` (name) and everything outside the robot,
        e.g. a held book touching the shelf board (on a real robot: the load change seen by the
        wrist force sensor)."""
        m, d = self.m, self.d
        bid = m.body(body).id
        f6, fn = np.zeros(6), 0.0
        for i, c in enumerate(d.contact[:d.ncon]):
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if bid in (b1, b2) and (b2 if b1 == bid else b1) not in self.robot_bodies:
                mujoco.mj_contactForce(m, d, i, f6)
                fn += f6[0]
        return fn

    def guarded_move(self, direction, force, z_axis=None, y_axis=None, R=None, speed=0.01,
                     max_dist=0.05, body=None, settle=0.3, sense=None, hold=False):
        """Move the TCP along `direction` (world) until the finger contact force reaches `force` N,
        then hold that position (a guarded move, e.g. pressing down on a book). `sense` replaces the
        finger force with another reading (e.g. object_contact_force of a held book, to set it down).
        hold: keep the force while settling, i.e. move on whenever it drops below 80 % (the press
        relaxes to about half once the arm stops). Fails if nothing is touched within `max_dist`."""
        read = sense if sense is not None else (lambda: self.finger_force(body)[0])
        u = np.asarray(direction, float)
        u = u / np.linalg.norm(u)
        n_ctrl = int(round(CONTROL_DT / self.m.opt.timestep))
        q = self.d.ctrl[self.arm_act].copy()
        p0 = self.fk_world()[:3, 3].copy()
        p, fn = p0.copy(), 0.0
        while np.linalg.norm(p - p0) < max_dist:
            p = p + speed * CONTROL_DT * u
            r = solve(self.kin, self.make_target(p, z_axis, y_axis, R), q, max_iter=8,
                      tol_pos=2e-4, tol_ang=2e-3)
            q = r.q
            self.d.ctrl[self.arm_act] = q
            self.step(n_ctrl)
            fn = read()
            if fn >= force:
                break
        if hold and fn >= force:
            for _ in range(int(round(settle / CONTROL_DT))):
                if read() < 0.8 * force and np.linalg.norm(p - p0) < max_dist:
                    p = p + speed * CONTROL_DT * u
                    q = solve(self.kin, self.make_target(p, z_axis, y_axis, R), q, max_iter=8,
                              tol_pos=2e-4, tol_ang=2e-3).q
                    self.d.ctrl[self.arm_act] = q
                self.step(n_ctrl)
        else:
            self.wait(settle)
        fn = read()
        return GuardedMoveResult(fn >= 0.5 * force, float(np.linalg.norm(p - p0)), fn)

    def set_gripper_width(self, width, wait=0.6):
        """Open the fingers to `width` metres between the finger plates (0 .. 0.08)."""
        self.d.ctrl[self.grip_act] = float(np.clip(width / 2 / 0.04 * 255.0, 0, 255))
        self.wait(wait)

    def attach(self, book):
        """Weld ("attach") grasp, the README fallback until the friction grasp of Phase 5:
        switch on the hand<->book weld with the book's current pose relative to the hand."""
        m, d = self.m, self.d
        eq = m.equality(f"grasp_{book}").id
        h, b = m.body(f"{PREFIX_HAND}").id, m.body(f"book_{book}").id
        mujoco.mj_forward(m, d)
        p = d.xmat[h].reshape(3, 3).T @ (d.xpos[b] - d.xpos[h])          # book in the hand frame
        qc, q = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(qc, d.xquat[h])
        mujoco.mju_mulQuat(q, qc, d.xquat[b])
        m.eq_data[eq, :] = 0
        m.eq_data[eq, 3:6], m.eq_data[eq, 6:10], m.eq_data[eq, 10] = p, q, 1.0
        d.eq_active[eq] = 1

    def grasp(self, book, wait=0.8):
        """Close the fingers on a book (they are commanded shut; the stiff gripper servo squeezes a
        4 cm book with ~60 N). grasp_mode "weld" also switches on the attach weld. Returns
        (gripper width, finger normal force on the book): a width near the book thickness and a
        force of tens of N mean the book is held."""
        self.d.ctrl[self.grip_act] = 0.0
        self.wait(wait)
        if self.grasp_mode == "weld":
            self.attach(book)
        self.holding = book
        return self.gripper_width(), self.finger_force(f"book_{book}")[0]

    def release(self, book):
        """Let go of the book: weld off (if on). The caller opens the fingers."""
        self.d.eq_active[self.m.equality(f"grasp_{book}").id] = 0
        self.holding = None

    def gripper_width(self):
        m = self.meta["arm_joints"][0].replace("joint1", "")
        j1 = self.m.joint(m + "finger_joint1").qposadr[0]
        j2 = self.m.joint(m + "finger_joint2").qposadr[0]
        return float(self.d.qpos[j1] + self.d.qpos[j2])

    # ------------------------------------------------------------------ safety
    def robot_contacts(self):
        """Names of things the robot touches (anything except the floor), plus self-contacts
        (except the closed fingers touching each other)."""
        out = set()
        m, d = self.m, self.d
        for c in d.contact[:d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            r1, r2 = b1 in self.robot_bodies, b2 in self.robot_bodies
            if not (r1 or r2) or self.floor_geom in (c.geom1, c.geom2) or {b1, b2} <= self.finger_bodies:
                continue
            n1, n2 = m.body(b1).name, m.body(b2).name
            out.add(f"{n1} <-> {n2}" if r1 else f"{n2} <-> {n1}")
        return out

    # ------------------------------------------------------------------ sensors
    def lidar_scan(self, noise=True, rear=False):
        """Return (angles in the robot frame, ranges in m). No return = range max (inf).
        rear: the second LiDAR at the back, looking backwards (angles still in the robot frame,
        measured from the robot's forward axis, about its own position: see lidar_offset)."""
        site = self.lidar_rear_site if rear else self.lidar_site
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
        ang = self.lidar_angles.copy()
        if rear:
            ang = (ang + 2 * math.pi) % (2 * math.pi) - math.pi        # turned by 180 deg
        return ang, r

    def lidar_offset(self, rear=False):
        """LiDAR position in the robot frame (x forward)."""
        return (-1.0 if rear else 1.0) * float(self.meta["lidar"]["x"]), 0.0

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
