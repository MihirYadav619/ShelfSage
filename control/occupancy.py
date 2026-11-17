"""2D occupancy grid of the static library at a given height band.

Built from the compiled MuJoCo model: every box geom that is not part of the robot and not a
free body (books) and that overlaps the height band [z_lo, z_hi] is rasterised. Hidden walls
count (they still collide). This is the "known map": the LiDAR check in Phase 3 uses it, and
Phase 6 uses it for MCL (expected scans) and A* (inflated by the robot radius).
"""
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class Grid:
    occ: np.ndarray        # bool [ny, nx], True = occupied
    origin: tuple          # world (x, y) of cell [0, 0]'s corner
    res: float             # metres per cell

    def to_cell(self, xy):
        xy = np.atleast_2d(xy)
        return np.floor((xy - np.array(self.origin)) / self.res).astype(int)   # (ix, iy)

    def to_world(self, ix, iy):
        return (self.origin[0] + (ix + 0.5) * self.res, self.origin[1] + (iy + 0.5) * self.res)

    def distance_field(self):
        """Distance (m) from every cell to the nearest occupied cell."""
        from scipy.ndimage import distance_transform_edt
        return distance_transform_edt(~self.occ) * self.res


def static_boxes(model, data, z_lo, z_hi, exclude_root=None):
    """World-frame boxes (centre, half-extents, yaw) of static geoms overlapping [z_lo, z_hi]."""
    out = []
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        b = model.geom_bodyid[g]
        root = model.body_rootid[b]
        if exclude_root is not None and root == exclude_root:
            continue
        if model.body_jntnum[root] > 0:          # anything that can move (robot, books)
            continue
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue                             # purely visual (window frame, sign, apron, ...)
        c = data.geom_xpos[g]
        R = data.geom_xmat[g].reshape(3, 3)
        half = model.geom_size[g]
        hz = np.abs(R[2]) @ half                 # vertical half-extent in world
        if c[2] + hz < z_lo or c[2] - hz > z_hi:
            continue
        out.append((c[:2].copy(), half.copy(), float(np.arctan2(R[1, 0], R[0, 0]))))
    return out


def build_grid(model, data, z_lo, z_hi, res=0.02, bounds=None, exclude_root=None):
    """Rasterise static boxes into a grid. bounds = (xmin, xmax, ymin, ymax) in metres."""
    boxes = static_boxes(model, data, z_lo, z_hi, exclude_root)
    if bounds is None:
        pts = np.array([c for c, _, _ in boxes])
        bounds = (pts[:, 0].min() - 0.5, pts[:, 0].max() + 0.5, pts[:, 1].min() - 0.5, pts[:, 1].max() + 0.5)
    xmin, xmax, ymin, ymax = bounds
    nx, ny = int(np.ceil((xmax - xmin) / res)), int(np.ceil((ymax - ymin) / res))
    xs = xmin + (np.arange(nx) + 0.5) * res
    ys = ymin + (np.arange(ny) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)
    occ = np.zeros((ny, nx), dtype=bool)
    for c, half, yaw in boxes:
        cs, sn = np.cos(yaw), np.sin(yaw)
        dx, dy = X - c[0], Y - c[1]
        lx, ly = cs * dx + sn * dy, -sn * dx + cs * dy
        occ |= (np.abs(lx) <= half[0] + res / 2) & (np.abs(ly) <= half[1] + res / 2)
    return Grid(occ, (xmin, ymin), res)


def exact_distance_grid(model, data, z_lo, z_hi, res=0.01, bounds=None, exclude_root=None, max_dist=0.5):
    """Grid whose `field` is the exact distance (m, up to max_dist) from each cell centre to the
    nearest static box at [z_lo, z_hi] (0 inside). For the MCL likelihood field: the rasterised
    grid's distance transform puts every surface up to a cell nearer, a bias of ~1 cm."""
    boxes = static_boxes(model, data, z_lo, z_hi, exclude_root)
    xmin, xmax, ymin, ymax = bounds
    nx, ny = int(np.ceil((xmax - xmin) / res)), int(np.ceil((ymax - ymin) / res))
    field = np.full((ny, nx), max_dist, dtype=np.float32)
    for c, half, yaw in boxes:
        r = float(np.hypot(half[0], half[1])) + max_dist
        i0, i1 = max(0, int((c[0] - r - xmin) / res)), min(nx, int((c[0] + r - xmin) / res) + 1)
        j0, j1 = max(0, int((c[1] - r - ymin) / res)), min(ny, int((c[1] + r - ymin) / res) + 1)
        if i0 >= i1 or j0 >= j1:
            continue
        xs = xmin + (np.arange(i0, i1) + 0.5) * res
        ys = ymin + (np.arange(j0, j1) + 0.5) * res
        X, Y = np.meshgrid(xs, ys)
        cs, sn = np.cos(yaw), np.sin(yaw)
        dx, dy = X - c[0], Y - c[1]
        qx = np.abs(cs * dx + sn * dy) - half[0]
        qy = np.abs(-sn * dx + cs * dy) - half[1]
        d = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
        field[j0:j1, i0:i1] = np.minimum(field[j0:j1, i0:i1], d)
    g = Grid(field <= 0.0, (xmin, ymin), res)
    g.field = field
    return g
