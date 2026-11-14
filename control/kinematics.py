"""Franka Panda kinematics on the ShelfSage robot (Phase 4), written from the DH parameters.

Frames
    base   the mobile base: origin on the floor under the base centre, x forward, z up
    link0  Panda base, at the arm mount (mount_x, 0, mount_h) in the base frame
    tcp    between the fingertips: flange + Menagerie hand (-45 deg about z) + 0.1034 m along z

FK uses Craig's modified DH convention with Franka's published parameters:
    T_i = RotX(alpha_{i-1}) TransX(a_{i-1}) RotZ(theta_i) TransZ(d_i)
Checked against MuJoCo in sim/test_phase4.py (< 0.001 mm, < 0.001 deg).
"""
import numpy as np

# (a_{i-1}, d_i, alpha_{i-1}) for joints 1..7, then the flange (theta = 0)
DH = [(0.0, 0.333, 0.0),
      (0.0, 0.0, -np.pi / 2),
      (0.0, 0.316, np.pi / 2),
      (0.0825, 0.0, np.pi / 2),
      (-0.0825, 0.384, -np.pi / 2),
      (0.0, 0.0, np.pi / 2),
      (0.088, 0.0, np.pi / 2)]
FLANGE = (0.0, 0.107, 0.0)
HAND_YAW = -np.pi / 4          # Menagerie mounts the hand rotated -45 deg about the flange z
TCP_OFFSET = 0.1034            # flange -> fingertip centre

Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


def mdh(a, d, alpha, theta):
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    return np.array([[ct, -st, 0, a],
                     [st * ca, ct * ca, -sa, -d * sa],
                     [st * sa, ct * sa, ca, d * ca],
                     [0, 0, 0, 1]])


def rot_z(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def transform(R=np.eye(3), p=(0, 0, 0)):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


class PandaKinematics:
    """FK and geometric Jacobian of the TCP in the robot base frame."""

    def __init__(self, mount=(0.20, 0.0, 0.65)):
        self.T_base_link0 = transform(p=mount)
        self.T_flange_tcp = transform(rot_z(HAND_YAW), (0, 0, TCP_OFFSET))
        self.q_min, self.q_max = Q_MIN.copy(), Q_MAX.copy()

    def frames(self, q):
        """Joint frames T_base_i (i = 1..7, z = joint axis) and the TCP frame."""
        T = self.T_base_link0.copy()
        out = []
        for (a, d, al), th in zip(DH, q):
            T = T @ mdh(a, d, al, th)
            out.append(T)
        T_tcp = T @ mdh(*FLANGE, 0.0) @ self.T_flange_tcp
        return out, T_tcp

    def fk(self, q):
        """4x4 pose of the TCP in the base frame."""
        return self.frames(q)[1]

    def jacobian(self, q):
        """Geometric Jacobian (6x7, rows = [v; w]) of the TCP in the base frame, and the TCP pose."""
        frames, T_tcp = self.frames(q)
        p = T_tcp[:3, 3]
        J = np.zeros((6, 7))
        for i, Ti in enumerate(frames):
            z, o = Ti[:3, 2], Ti[:3, 3]
            J[:3, i] = np.cross(z, p - o)
            J[3:, i] = z
        return J, T_tcp
