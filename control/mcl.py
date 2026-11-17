"""Monte Carlo Localization (particle filter, as in AMCL) on the known map (Phase 6).

    predict   every particle moves by the odometry increment since the last update (in its own
              frame), plus noise that grows with the distance driven and the angle turned
    update    LiDAR likelihood field: the scan's end points, placed with each particle's pose,
              should lie on the map's obstacles; p(ray) = z_hit * N(dist to nearest obstacle;
              sigma) + z_rand (rays that hit something not in the map: people, the test box)
    resample  low-variance resampling when the effective particle count drops below half
    estimate  weighted mean (circular mean for the yaw)

The robot runs an update every 0.1 s of driving (not while standing: repeated scans of the same
view would make the filter overconfident) and in between adds the odometry increment to the last
estimate, so get_pose() is smooth.
"""
import math

import numpy as np

N_PARTICLES = 300
N_RAYS = 90                    # of the 541 LiDAR rays, evenly spread
SIGMA_HIT = 0.02               # m
Z_HIT, Z_RAND = 0.9, 0.1
MAX_DIST = 0.5                 # likelihood field clipped here (m)
# motion noise: std of the translation from translation / rotation, of the rotation from both
A_TT, A_TR, A_RR, A_RT = 0.05, 0.02, 0.05, 0.05
MIN_T, MIN_R = 0.002, 0.002    # m, rad: noise floor per update


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class MCL:
    def __init__(self, grid, lidar_xy, pose, sigma=(0.03, 0.03, 0.02), n=N_PARTICLES, seed=0):
        """grid: Grid at the LiDAR's height (control/occupancy.py), with an exact distance `field`
        (exact_distance_grid) or else its distance transform is used; lidar_xy: the LiDAR's
        position in the robot frame; pose: the known start pose."""
        self.grid = grid
        field = grid.field if getattr(grid, "field", None) is not None else grid.distance_field()
        self.field = np.minimum(field, MAX_DIST).astype(np.float32)
        self.lidar_xy = np.asarray(lidar_xy, float)
        self.rng = np.random.default_rng(seed)
        self.n = n
        self.p = np.asarray(pose, float) + self.rng.normal(0, 1, (n, 3)) * np.asarray(sigma)
        self.w = np.full(n, 1.0 / n)
        self.est = np.asarray(pose, float).copy()
        self.updates = 0

    # ------------------------------------------------------------------ steps
    def predict(self, delta):
        """delta = (dx, dy, dyaw): odometry increment in the robot frame of the previous pose."""
        dx, dy, dth = delta
        trans = math.hypot(dx, dy)
        st = A_TT * trans + A_TR * abs(dth) + MIN_T
        sr = A_RR * abs(dth) + A_RT * trans + MIN_R
        n = self.n
        ddx = dx + self.rng.normal(0, st, n)
        ddy = dy + self.rng.normal(0, st, n)
        c, s = np.cos(self.p[:, 2]), np.sin(self.p[:, 2])
        self.p[:, 0] += c * ddx - s * ddy
        self.p[:, 1] += s * ddx + c * ddy
        self.p[:, 2] = wrap(self.p[:, 2] + dth + self.rng.normal(0, sr, n))

    def update(self, angles, ranges):
        """Weight the particles by a LiDAR scan (angles in the robot frame, ranges; inf = none)."""
        idx = np.linspace(0, len(angles) - 1, N_RAYS).astype(int)
        a, r = angles[idx], ranges[idx]
        ok = np.isfinite(r)
        if ok.sum() < 10:
            return
        a, r = a[ok], r[ok]
        # end points in the robot frame
        ex = self.lidar_xy[0] + r * np.cos(a)
        ey = self.lidar_xy[1] + r * np.sin(a)
        c, s = np.cos(self.p[:, 2])[:, None], np.sin(self.p[:, 2])[:, None]
        wx = self.p[:, 0:1] + c * ex - s * ey
        wy = self.p[:, 1:2] + s * ex + c * ey
        ix = np.floor((wx - self.grid.origin[0]) / self.grid.res).astype(int)
        iy = np.floor((wy - self.grid.origin[1]) / self.grid.res).astype(int)
        ny, nx = self.field.shape
        inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        d = np.full(ix.shape, MAX_DIST, np.float32)
        d[inside] = self.field[iy[inside], ix[inside]]
        lik = Z_HIT * np.exp(-0.5 * (d / SIGMA_HIT) ** 2) + Z_RAND
        logw = np.log(lik).sum(axis=1)
        logw -= logw.max()
        w = self.w * np.exp(logw)
        if w.sum() <= 0 or not np.isfinite(w.sum()):
            w = np.ones(self.n)
        self.w = w / w.sum()
        self.updates += 1
        if 1.0 / (self.w ** 2).sum() < self.n / 2:
            self.resample()
        self.est = self.mean()

    def resample(self):
        """Low-variance (systematic) resampling."""
        pos = (self.rng.random() + np.arange(self.n)) / self.n
        idx = np.searchsorted(np.cumsum(self.w), pos)
        idx = np.minimum(idx, self.n - 1)
        self.p = self.p[idx].copy()
        self.w = np.full(self.n, 1.0 / self.n)

    def mean(self):
        x = float(self.w @ self.p[:, 0])
        y = float(self.w @ self.p[:, 1])
        yaw = math.atan2(float(self.w @ np.sin(self.p[:, 2])), float(self.w @ np.cos(self.p[:, 2])))
        return np.array([x, y, yaw])

    def spread(self):
        """Weighted standard deviation (x, y in m, yaw in rad), how sure the filter is."""
        m = self.mean()
        dx, dy = self.p[:, 0] - m[0], self.p[:, 1] - m[1]
        dth = wrap(self.p[:, 2] - m[2])
        return np.sqrt([self.w @ dx ** 2, self.w @ dy ** 2, self.w @ dth ** 2])
