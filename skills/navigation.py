"""NAVIGATE for now: drive to the parking pose of a table or shelf column along fixed lanes.

The robot always faces +y (all parking poses do) and the base is holonomic, so it only translates.
Three parking lines run east-west, free of furniture: the south tables (A-C, y = -3.2), the north
tables (D-F, y = -0.7) and the shelves (y = 1.725). Between lines it goes through the aisle at
x = 1.25 m (between tables B/C and E/F). A* on the occupancy grid replaces this in Phase 6.

Targets: "Table B" / "Table_B" / "B" (the middle parking spot), a compartment "S2-A3" (its
column's parking pose), or a shelf "S2" (column 1).
"""
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
META = json.loads((ROOT / "scene/library_meta.json").read_text())
AISLE_X = 1.25
SAME_LINE = 0.3          # m: two poses within this in y are on the same parking line


def parse_target(target):
    """("table", "B") / ("shelf", "S2", "S2-A3" or None), or None if not understood."""
    t = target.strip()
    m = re.fullmatch(r"(?:shelf[ _]?)?(S\d)-([A-Z])(\d)", t, re.IGNORECASE)
    if m:
        shelf = m.group(1).upper()
        return ("shelf", shelf, f"{shelf}-{m.group(2).upper()}{m.group(3)}")
    m = re.fullmatch(r"(?:shelf[ _]?)?(S\d)", t, re.IGNORECASE)
    if m:
        return ("shelf", m.group(1).upper(), None)
    m = re.fullmatch(r"(?:table[ _]?)?([A-F])", t, re.IGNORECASE)
    if m:
        return ("table", m.group(1).upper())
    return None


def parking_pose(target):
    """Parking pose (x, y, yaw) for a target string, or None."""
    p = parse_target(target)
    if p is None:
        return None
    if p[0] == "table":
        if p[1] not in META["tables"]:
            return None
        return tuple(META["tables"][p[1]]["parking"][1])
    shelf, comp = p[1], p[2]
    if shelf not in META["shelves"]:
        return None
    col = comp.split("-")[1][1:] if comp else "1"
    if comp and comp not in META["shelves"][shelf]["compartments"]:
        return None
    return tuple(META["shelves"][shelf]["parking"][col])


def route(start, goal):
    """Waypoints (x, y, yaw) from start to goal: straight along a parking line, else via the aisle."""
    yaw = goal[2]
    if abs(start[1] - goal[1]) < SAME_LINE:
        return [goal]
    return [(AISLE_X, start[1], yaw), (AISLE_X, goal[1], yaw), goal]


def navigate(robot, goal, say=print, log=None, holding=None):
    """Drive to `goal` (x, y, yaw). The arm is stowed unless a book is held (then it stays in the
    carry pose). Contacts with the held book do not count as collisions. Returns (ok, message)."""
    log = [] if log is None else log
    if holding is None:
        robot.stow()
    for wp in route(robot.get_pose(), goal):
        res = robot.go_to(*wp, tol_xy=0.02, tol_yaw=0.02)
        hits = [c for c in res.collisions if holding is None or f"book_{holding}" not in c]
        ok = not hits and res.message != "timeout"
        log.append((f"drive to ({wp[0]:.2f}, {wp[1]:.2f})", ok))
        if not ok:
            return False, f"drive to ({wp[0]:.2f}, {wp[1]:.2f}) failed: {res.message} {hits}"
    return True, "arrived"


def near(robot, goal, tol=0.05):
    x, y, yaw = robot.get_pose()
    return math.hypot(x - goal[0], y - goal[1]) < tol and abs(
        (yaw - goal[2] + math.pi) % (2 * math.pi) - math.pi) < 0.05
