"""Inverse kinematics for the Panda (Phase 4): damped least squares with null-space objectives.

A target fixes the TCP position and some of its orientation:
    z_axis only        approach direction (roll about it is free)
    z_axis + y_axis    approach + finger closing direction (closing axis sign is free)
    R                  full orientation
Secondary objectives in the null space of the task: stay away from joint limits and lean
towards a preferred posture (elbow up). solve_multi() tries several start configurations and
keeps the collision-free solution closest to the current arm configuration.
"""
from dataclasses import dataclass, field

import numpy as np

from control.kinematics import PandaKinematics


@dataclass
class Target:
    pos: np.ndarray
    z_axis: np.ndarray = None
    y_axis: np.ndarray = None
    R: np.ndarray = None


@dataclass
class IKResult:
    success: bool
    q: np.ndarray
    pos_err: float
    ang_err: float          # rad, on the constrained axes
    iters: int
    collision: list = field(default_factory=list)


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _rot_vec(a, b):
    """Rotation vector turning unit vector a onto unit vector b."""
    c = np.clip(a @ b, -1.0, 1.0)
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        if c > 0:
            return np.zeros(3)
        axis = np.cross(a, [1, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1, 0])
        s = np.linalg.norm(axis)
    return axis / s * np.arctan2(s, c)


def _log_so3(R):
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    if np.pi - th < 1e-6:                      # 180 deg: axis from the symmetric part
        w, V = np.linalg.eigh((R + np.eye(3)) / 2)
        return V[:, np.argmax(w)] * th
    return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * th / (2 * np.sin(th))


def task_error(T, J, target):
    """Task error e and the matching Jacobian rows."""
    p, R = T[:3, 3], T[:3, :3]
    e_p = target.pos - p
    if target.R is not None or (target.z_axis is not None and target.y_axis is not None):
        if target.R is not None:
            Rd = target.R
        else:
            z = _unit(target.z_axis)
            y = _unit(target.y_axis)
            y = y - (y @ z) * z
            y = _unit(y) * (1.0 if y @ R[:, 1] >= 0 else -1.0)   # closing axis sign is free
            Rd = np.column_stack([np.cross(y, z), y, z])
        e_w = _log_so3(Rd @ R.T)
        return np.concatenate([e_p, e_w]), J, float(np.linalg.norm(e_w))
    if target.z_axis is not None:
        z = R[:, 2]
        e_w = _rot_vec(z, _unit(target.z_axis))
        P = np.eye(3) - np.outer(z, z)                            # rotation about z is free
        return np.concatenate([e_p, e_w]), np.vstack([J[:3], P @ J[3:]]), float(np.linalg.norm(e_w))
    return e_p, J[:3], 0.0


def solve(kin: PandaKinematics, target: Target, q0, q_pref=None, max_iter=300, tol_pos=1e-4,
          tol_ang=1e-3, damping=0.03, w_ori=0.5, k_null=0.3, step_max=0.25):
    """Damped least squares from q0. Orientation rows are weighted by w_ori (m per rad)."""
    q = np.clip(np.asarray(q0, float), kin.q_min, kin.q_max)
    q_pref = q if q_pref is None else np.asarray(q_pref, float)
    mid, rng = (kin.q_min + kin.q_max) / 2, kin.q_max - kin.q_min
    pe = ae = np.inf
    best_cost, stall = np.inf, 0
    for it in range(1, max_iter + 1):
        J, T = kin.jacobian(q)
        e, Jt, ae = task_error(T, J, target)
        pe = float(np.linalg.norm(e[:3]))
        if pe < tol_pos and ae < tol_ang:
            return IKResult(True, q, pe, ae, it)
        cost = pe + w_ori * ae                      # stop early when stuck in a local minimum
        if cost < 0.98 * best_cost:
            best_cost, stall = cost, 0
        else:
            stall += 1
            if stall > 25:
                break
        W = np.ones(len(e))
        W[3:] = w_ori
        e_w, J_w = e * W, Jt * W[:, None]
        A = J_w @ J_w.T + damping ** 2 * np.eye(len(e))
        dq = J_w.T @ np.linalg.solve(A, e_w)
        # null space: away from joint limits + towards the preferred posture. The projector uses
        # the exact pseudo-inverse; the damped one leaks the posture pull into the task and the
        # solver stalls a few cm short of the target.
        grad = -(q - mid) / rng ** 2 * 4.0 + 0.5 * (q_pref - q)
        dq += (np.eye(7) - np.linalg.pinv(J_w, rcond=1e-4) @ J_w) @ (k_null * grad)
        n = np.linalg.norm(dq)
        if n > step_max:
            dq *= step_max / n
        q = np.clip(q + dq, kin.q_min, kin.q_max)
    # close enough for a real arm (1 mm, 0.3 deg) even if the strict tolerance was not met
    return IKResult(pe < 1e-3 and ae < 5e-3, q, pe, ae, it)


def limit_margin(kin, q):
    """Smallest distance (rad) of any joint to its limit."""
    return float(np.min(np.minimum(q - kin.q_min, kin.q_max - q)))


def solve_multi(kin, target, q_now, seeds=(), n_random=8, rng=None, collision_fn=None, q_pref=None,
                min_margin=0.3, reject_margin=None, **kw):
    """Try q_now, the given seeds and some random starts; among the collision-free solutions pick
    the one with the lowest cost = joint motion from q_now + distance to the preferred posture
    + a penalty for being close to joint limits (a configuration with a joint at its limit can
    reach the target but not move on from it, e.g. along a straight line). With `reject_margin`,
    solutions closer than that to a joint limit are dropped as long as another one exists."""
    rng = rng or np.random.default_rng(0)
    q_now = np.asarray(q_now, float)
    q_pref_v = q_now if q_pref is None else np.asarray(q_pref, float)
    starts = [q_now] + [np.asarray(s, float) for s in seeds]
    starts += [rng.uniform(kin.q_min, kin.q_max) for _ in range(n_random)]

    def cost(r):
        m = limit_margin(kin, r.q)
        return (np.linalg.norm(r.q - q_now) + 0.5 * np.linalg.norm(r.q - q_pref_v)
                + (5.0 * (min_margin - m) if m < min_margin else 0.0))

    good, best = [], None
    for k, q0 in enumerate(starts):
        r = solve(kin, target, q0, q_pref=q_pref_v, **kw)
        if r.success and collision_fn is not None:
            r.collision = collision_fn(r.q)
        if r.success and not r.collision:
            if k == 0 and limit_margin(kin, r.q) >= min_margin:
                return r                            # small move from here and nowhere near a limit
            good.append(r)
            ok_margin = [g for g in good if reject_margin is None or limit_margin(kin, g.q) >= reject_margin]
            if len(ok_margin) >= 4:
                break
        elif best is None or (r.pos_err + 0.1 * r.ang_err) < (best.pos_err + 0.1 * best.ang_err):
            best = r
    if reject_margin is not None and any(limit_margin(kin, g.q) >= reject_margin for g in good):
        good = [g for g in good if limit_margin(kin, g.q) >= reject_margin]
    if good:
        return min(good, key=cost)
    if best is not None and best.success:          # every solution collided
        best.success = False
    return best
