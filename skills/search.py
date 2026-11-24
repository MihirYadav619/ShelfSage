"""Search mode: the robot does not know where loose books are; it finds them with the wrist camera.

What the robot knows (robot.memory, set by enable(robot)):
    * the map: tables, shelves, compartments and their parking poses (fixed furniture),
    * the catalog: the home compartment of every book (scene/layout.json, Book_Catalog.pdf),
    * the shelf inventory at the start: which books stand in which compartment (the library's
      records). Books lying on the reading tables are NOT known: someone took them off a shelf
      and left them somewhere.
memory maps book id -> {"kind": "table" | "shelf", "where": "B" | "S2-A3", "pos": [x, y, z]}.
A book missing from it is "unknown". The skills keep it up to date: FIND_BOOK adds every book
the camera sees on the way, GRASP / PLACE move the entry, and a GRASP that finds the book gone
forgets it (it was moved by someone else), so a replan searches again.

FIND_BOOK(book) in search mode (find()):
    1. where memory says it is (if anywhere): drive there, look, done if it is seen;
    2. every table, nearest first: park in front of it, look with the wrist camera (up to six
       views, the same as the grasp's first look), stop when the book is seen;
    3. every shelf column, nearest first, rows A and B (a book put back in the wrong place).
The robot stays parked where it found the book, so GRASP can follow directly.
Detections are placed on a table / in a compartment by their estimated centre
(manipulation.locate_point); the YOLOv8 detector is what sees, never simulator ground truth.
"""
import math

import numpy as np

from skills import manipulation as mp
from skills import navigation as nav
from skills.manipulation import BD, META, UP, shelf_axes, table_frame
from skills.result import SkillResult

LOOK_AHEAD, TABLE_LOOK_UP = 0.45, 0.40       # as skills/perception.py: camera 45 cm ahead of the
SIDE = 0.35                                  # arm mount, 40 cm above the table, then 35 cm to
SHELF_LOOK_BACK, SHELF_LOOK_DIST = 0.15, 0.32  # either side; shelf: base 15 cm back, camera 32 cm
SLACK = 0.03                                 # camera centres may be a few cm off


def enable(robot):
    """Switch the robot to search mode: memory = the shelf inventory, tables unknown."""
    robot.memory = {}
    for bid in META["books"]:
        kind, where = mp.locate_book(robot, bid)
        if kind == "shelf":
            remember(robot, bid, kind, where, robot.d.body(f"book_{bid}").xpos)
    return robot


def active(robot):
    return getattr(robot, "memory", None) is not None


def remember(robot, bid, kind, where, pos):
    robot.memory[bid] = {"kind": kind, "where": where, "pos": [round(float(v), 3) for v in pos]}


def forget(robot, bid):
    robot.memory.pop(bid, None)


def target_of(loc):
    return f"Table {loc['where']}" if loc["kind"] == "table" else loc["where"]


def _eyes(robot):
    """The camera + detector used for searching (the grasp's own, if it has one)."""
    if getattr(robot, "perception", None) is not None:
        return robot.perception
    if getattr(robot, "_search_eyes", None) is None:
        from skills.perception import Perception
        robot._search_eyes = Perception(say=lambda s: None)
    return robot._search_eyes


# ---------------------------------------------------------------------------- looking
def _look_all(robot, eyes, cam, view, z_range=None):
    """Aim the wrist camera and return every detection in the frame."""
    from perception.detect import aim_camera
    q = aim_camera(robot, eyes.det, cam, view)
    if q is None:
        return []
    home = np.array(robot.meta["arm_home"])
    if np.max(np.abs(robot.arm_q() - home)) > 1.0:
        robot.home(1.5)                      # via home: a low swing can knock a book off the table
    robot.move_arm(q, 2.0, settle=0.3)
    return eyes.det.detect(robot, z_range=z_range)


def _record(robot, dets, seen):
    """Put every detection on its table / compartment in memory; returns {book: target}."""
    for d in dets:
        if d.book == robot.holding:
            continue
        kind, where = mp.locate_point(d.centre, SLACK)
        if kind in ("table", "shelf"):
            remember(robot, d.book, kind, where, d.centre)
            seen[d.book] = target_of(robot.memory[d.book])
    return seen


def scan_table(robot, table, book=None):
    """Look over a table from where the robot stands (its parking). Stops early once `book` is
    seen. Returns {book id: "Table X"} for every book seen on any table."""
    eyes = _eyes(robot)
    fwd, left, _, _ = table_frame(table)
    top = META["tables"][table]["top_z"]
    x, y, _ = robot.get_pose()
    mount = np.array([x, y, 0.0]) + robot.meta["mount_x"] * fwd
    zr = (top + 0.003, top + 0.08)
    seen = {}
    for view in (0.6 * fwd - UP, fwd - 0.5 * UP):       # near half, then the far half
        for dl in (0.0, SIDE, -SIDE):
            cam = mount + LOOK_AHEAD * fwd + dl * left + np.array([0, 0, top + TABLE_LOOK_UP])
            _record(robot, _look_all(robot, eyes, cam, view, zr), seen)
            if book is not None and seen.get(book) == f"Table {table}":
                robot.stow()
                return seen
    robot.stow()
    return seen


def scan_column(robot, shelf, col, book=None):
    """Look at rows A and B of one shelf column from its parking (backs off for the look and
    returns). Returns {book id: compartment} for every book seen."""
    eyes = _eyes(robot)
    fwd, side = shelf_axes(shelf)
    park = robot.get_pose()
    robot.stow()
    robot.go_to(park[0] - SHELF_LOOK_BACK * fwd[0], park[1] - SHELF_LOOK_BACK * fwd[1], park[2],
                tol_xy=0.01, tol_yaw=0.01)
    seen = {}
    for row in ("A", "B"):
        comp = f"{shelf}-{row}{col}"
        pos = META["shelves"][shelf]["compartments"][comp]["positions"]
        c = np.mean([pos[0], pos[-1]], axis=0)
        cam = (c @ fwd - BD["width"] / 2 - SHELF_LOOK_DIST) * fwd + (c @ side) * side + np.array([0, 0, c[2] + 0.06])
        _record(robot, _look_all(robot, eyes, cam, fwd - 0.18 * UP), seen)
        if book is not None and seen.get(book) == comp:
            break
    robot.stow()
    robot.go_to(*park, tol_xy=0.01, tol_yaw=0.01)
    return seen


# ---------------------------------------------------------------------------- FIND_BOOK
def _places(robot, bid):
    """Places to look, in order: (target name, kind, table or (shelf, col))."""
    x, y, _ = robot.get_pose()

    def dist(target):
        p = nav.parking_pose(target)
        return math.hypot(p[0] - x, p[1] - y)

    out = []
    if bid in robot.memory:
        loc = robot.memory[bid]
        out.append(target_of(loc))
    out += sorted((f"Table {t}" for t in META["tables"]), key=dist)
    out += sorted((f"{s}-A{c}" for s in META["shelves"] for c in META["shelves"][s]["parking"]), key=dist)
    seen, res = set(), []
    for t in out:
        p = nav.parse_target(t)
        key = ("table", p[1]) if p[0] == "table" else ("col", p[1], p[2].split("-")[1][1:])
        if key not in seen:
            seen.add(key)
            res.append((t, key))
    return res


def find(robot, bid, say=print, log=None):
    """Search for the book with the camera (see module docstring); SkillResult."""
    title = META["books"][bid]["title"]
    if robot.holding == bid:
        return SkillResult(True, f"{title} is in the gripper", {"book": bid, "kind": "held", "where": None})
    looked = []
    for target, key in _places(robot, bid):
        goal = nav.parking_pose(target)
        if not nav.near(robot, goal):
            ok, msg = nav.navigate(robot, goal, say=say, log=log, holding=robot.holding)
            if not ok:
                say(f"  cannot reach {target}: {msg}")
                continue
        say(f"  looking at {target if key[0] == 'table' else 'shelf ' + key[1] + ' column ' + key[2]}")
        if key[0] == "table":
            seen = scan_table(robot, key[1], bid)
        else:
            seen = scan_column(robot, key[1], key[2], bid)
        looked.append(target if key[0] == "table" else f"{key[1]} column {key[2]}")
        if seen:
            say("  seen: " + ", ".join(f"{META['books'][b]['title']} ({t})" for b, t in seen.items()))
        if bid in seen:
            loc = robot.memory[bid]
            where = target_of(loc)
            if log is not None:
                log.append((f"found {title} at {where} after looking at {len(looked)} place(s)", True))
            return SkillResult(True, f"found {title} at {where} with the camera (looked at {', '.join(looked)})",
                               {"book": bid, "kind": loc["kind"], "where": loc["where"], "looked": looked})
    if bid in robot.memory:
        forget(robot, bid)
    return SkillResult(False, f"could not find {title}: looked at every table and shelf column", {"looked": looked})
