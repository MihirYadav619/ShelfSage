"""Path planning for the base (Phase 6): configuration space from the known map, A*, smoothing.

The base is holonomic and keeps one heading while it drives (every parking pose faces +y), so the
configuration space is 2D: the obstacles grown by the robot's footprint at that heading. Two
height bands, because the footprint depends on the height:
    low  (floor .. 0.95 m)   chassis 0.60 x 0.50 m; the stowed arm stays inside it
    high (0.95 .. 1.6 m)     stowed: the chassis rectangle; holding a book in the carry pose the
                             hand and book reach 0.76 m ahead of the base centre (and 0.35 m to its
                             right, the elbow). Tables (0.75 m) are below this band, so a carried
                             book may pass over a table edge, but not into a shelf.
plus a safety margin. Dynamic obstacles (seen by the LiDAR, not in the map) are added to both
bands.

A* runs 8-connected on the 5 cm grid, with a cost for passing close to obstacles (it prefers the
middle of a passage); the path is then shortened by line-of-sight (a waypoint is dropped if the
straight line past it stays free).
"""
import heapq
import math

import numpy as np

from control.occupancy import Grid, build_grid

RES = 0.05
LOW_BAND, HIGH_BAND = (0.02, 0.95), (0.95, 1.60)
CHASSIS = (-0.30, 0.30, -0.25, 0.25)       # base frame: fwd min / max, left min / max (m)
CARRY = (-0.30, 0.76, -0.35, 0.25)         # high band while holding a book in the carry pose
MARGIN = 0.05
NEAR = 0.25                                # closer than this to an obstacle (C-space) costs extra
NEAR_COST = 3.0
SHORTCUT_CLEAR = 0.10                      # a shortcut keeps this far from obstacles (or as far as A* did)


def footprint_kernel(rect, yaw, res, margin):
    """Boolean kernel (odd size, centre = robot centre) of a base-frame rectangle at heading yaw."""
    f0, f1, l0, l1 = rect[0] - margin, rect[1] + margin, rect[2] - margin, rect[3] + margin
    r = max(abs(f0), abs(f1), abs(l0), abs(l1)) * math.sqrt(2)
    n = int(math.ceil(r / res)) + 1
    ax = (np.arange(-n, n + 1)) * res
    X, Y = np.meshgrid(ax, ax)                 # world offsets from the robot centre
    c, s = math.cos(yaw), math.sin(yaw)
    fwd = c * X + s * Y
    left = -s * X + c * Y
    # a cell counts if any part of it may be inside: grow by half a cell diagonal
    h = res * 0.71
    return (fwd >= f0 - h) & (fwd <= f1 + h) & (left >= l0 - h) & (left <= l1 + h)


def grow(occ, kernel):
    """C-space: cell p is blocked if the footprint placed at p touches an occupied cell, i.e. the
    OR of the obstacle grid shifted by every footprint offset (numpy only: scipy.signal was too
    heavy for six test processes at once)."""
    ny, nx = occ.shape
    kh, kw = kernel.shape
    cy, cx = kh // 2, kw // 2
    pad = np.ones((ny + 2 * cy, nx + 2 * cx), dtype=bool)      # outside the map = blocked
    pad[cy:cy + ny, cx:cx + nx] = occ
    out = np.zeros_like(occ)
    for dy, dx in zip(*np.nonzero(kernel)):
        # footprint offset (dx - cx, dy - cy): out[p] |= occ[p + offset]
        out |= pad[dy:dy + ny, dx:dx + nx]
    return out


class NavMap:
    """Known map of the library in two height bands, and the configuration space for planning."""

    def __init__(self, model, data, bounds, exclude_root=None):
        self.low = build_grid(model, data, *LOW_BAND, res=RES, bounds=bounds, exclude_root=exclude_root)
        self.high = build_grid(model, data, *HIGH_BAND, res=RES, bounds=bounds, exclude_root=exclude_root)
        self.grid = Grid(self.low.occ, self.low.origin, RES)     # cell geometry
        self.dynamic = np.zeros_like(self.low.occ)                # seen by the LiDAR, not in the map
        self._cache = {}

    def clear_dynamic(self):
        self.dynamic[:] = False
        self._cache = {k: v for k, v in self._cache.items() if not k[2]}

    def add_obstacle_points(self, pts):
        """Mark world points (N, 2) as dynamic obstacles. Returns how many cells were new."""
        if len(pts) == 0:
            return 0
        cells = self.grid.to_cell(np.asarray(pts))
        ny, nx = self.dynamic.shape
        ok = (cells[:, 0] >= 0) & (cells[:, 0] < nx) & (cells[:, 1] >= 0) & (cells[:, 1] < ny)
        cells = cells[ok]
        new = int((~self.dynamic[cells[:, 1], cells[:, 0]]).sum())
        if new:
            self.dynamic[cells[:, 1], cells[:, 0]] = True
            self._cache = {k: v for k, v in self._cache.items() if not k[2]}
        return new

    def cspace(self, yaw, holding=False, dynamic=True):
        """(blocked cells, distance to them) for the robot centre at heading yaw (bool [ny, nx]);
        dynamic=False: the known map only."""
        key = (round(yaw, 3), holding, dynamic)
        if key not in self._cache:
            dyn = self.dynamic if dynamic else np.zeros_like(self.dynamic)
            low = grow(self.low.occ | dyn, footprint_kernel(CHASSIS, yaw, RES, MARGIN))
            high = grow(self.high.occ | dyn, footprint_kernel(CARRY if holding else CHASSIS, yaw, RES, MARGIN))
            c = low | high
            c[0, :] = c[-1, :] = c[:, 0] = c[:, -1] = True
            from scipy.ndimage import distance_transform_edt
            self._cache[key] = (c, distance_transform_edt(~c) * RES)
        return self._cache[key]

    def free(self, cs, xy):
        ix, iy = self.grid.to_cell(xy)[0]
        ny, nx = cs.shape
        return 0 <= ix < nx and 0 <= iy < ny and not cs[iy, ix]

    def line_free(self, cs, a, b, step=0.02):
        n = max(2, int(math.ceil(np.linalg.norm(np.subtract(b, a)) / step)) + 1)
        pts = np.linspace(a, b, n)
        cells = self.grid.to_cell(pts)
        ny, nx = cs.shape
        if (cells < 0).any() or (cells[:, 0] >= nx).any() or (cells[:, 1] >= ny).any():
            return False
        return not cs[cells[:, 1], cells[:, 0]].any()

    def nearest_free(self, cs, xy, prefer=None, max_dist=1.0):
        """Nearest free point to xy (searching outward on the grid); with `prefer` (a unit vector)
        first along that direction (e.g. backing straight away from a table)."""
        xy = np.asarray(xy, float)
        if self.free(cs, xy):
            return xy
        if prefer is not None:
            for d in np.arange(RES, max_dist, RES / 2):
                p = xy + d * np.asarray(prefer, float)
                if self.free(cs, p):
                    return p
        ix, iy = self.grid.to_cell(xy)[0]
        ny, nx = cs.shape
        best = None
        r = int(max_dist / RES)
        ys, xs = np.mgrid[max(0, iy - r):min(ny, iy + r + 1), max(0, ix - r):min(nx, ix + r + 1)]
        mask = ~cs[ys, xs]
        if mask.any():
            d2 = (xs[mask] - ix) ** 2 + (ys[mask] - iy) ** 2
            k = int(np.argmin(d2))
            best = np.array(self.grid.to_world(xs[mask][k], ys[mask][k]))
        return best

    def plan(self, start, goal, yaw, holding=False):
        """Waypoints [start, ..., goal] (world xy) for the robot centre at heading yaw, or None.
        start and goal must be free (see nearest_free)."""
        cs, dist = self.cspace(yaw, holding)
        if not (self.free(cs, start) and self.free(cs, goal)):
            return None
        s = tuple(self.grid.to_cell(start)[0])
        g = tuple(self.grid.to_cell(goal)[0])
        ny, nx = cs.shape
        penalty = 1.0 + NEAR_COST * np.clip((NEAR - dist) / NEAR, 0.0, 1.0)
        moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))]
        g_cost = {s: 0.0}
        parent = {s: None}
        heap = [(math.hypot(g[0] - s[0], g[1] - s[1]), 0.0, s)]
        closed = set()
        while heap:
            _, gc, cur = heapq.heappop(heap)
            if cur in closed:
                continue
            if cur == g:
                break
            closed.add(cur)
            for dx, dy, step in moves:
                nb = (cur[0] + dx, cur[1] + dy)
                if not (0 <= nb[0] < nx and 0 <= nb[1] < ny) or cs[nb[1], nb[0]] or nb in closed:
                    continue
                c = gc + step * penalty[nb[1], nb[0]]
                if c < g_cost.get(nb, math.inf):
                    g_cost[nb] = c
                    parent[nb] = cur
                    heapq.heappush(heap, (c + math.hypot(g[0] - nb[0], g[1] - nb[1]), c, nb))
        if g not in parent:
            return None
        cells = []
        cur = g
        while cur is not None:
            cells.append(cur)
            cur = parent[cur]
        cells.reverse()
        pts = [np.asarray(start, float)] + [np.array(self.grid.to_world(*c)) for c in cells[1:-1]] + \
              [np.asarray(goal, float)]
        return self.smooth(cs, dist, pts)

    def smooth(self, cs, dist, pts):
        """Line-of-sight shortcut: from each kept point, jump to the farthest point it sees; a
        shortcut must also not pass closer to obstacles than the A* path did (keeps the margin
        the cost bought)."""
        clear = [self.clearance(dist, p) for p in pts]
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not (self.line_free(cs, pts[i], pts[j]) and
                                     self.line_clearance(dist, pts[i], pts[j]) >= min(SHORTCUT_CLEAR, *clear[i:j + 1]) - 1e-6):
                j -= 1
            out.append(pts[j])
            i = j
        return out

    def clearance(self, dist, xy):
        ix, iy = self.grid.to_cell(xy)[0]
        ny, nx = dist.shape
        return float(dist[iy, ix]) if 0 <= ix < nx and 0 <= iy < ny else 0.0

    def line_clearance(self, dist, a, b, step=0.02):
        n = max(2, int(math.ceil(np.linalg.norm(np.subtract(b, a)) / step)) + 1)
        return min(self.clearance(dist, p) for p in np.linspace(a, b, n))

    def path_blocked(self, cs, pts):
        """True if any segment of the path (world xy points) crosses blocked cells."""
        return any(not self.line_free(cs, a, b) for a, b in zip(pts[:-1], pts[1:]))


def library_bounds(meta, pad=0.3):
    hx, hy = (s / 2 for s in meta["room_size"])
    cx, cy = meta["room_center"]
    return (cx - hx - pad, cx + hx + pad, cy - hy - pad, cy + hy + pad)
