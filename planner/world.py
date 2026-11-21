"""World model: what the planners know about the library, as plain data (JSON-ready).

    world = snapshot(robot)        # from the simulator (ground truth for the books, the robot's
                                   # own pose estimate for itself)
    world["books"]["ml"]           # {"title": "Machine Learning", "kind": "table", "where": "B", ...}

The planners (rule-based baseline, LLM in Phase 8) and the plan validator only ever see this
dictionary, never MuJoCo. Fixed facts (tables, compartments, capacities, positions) come from
scene/library_meta.json, the current state (where each book is, what the robot holds, where it
is) from the simulator. Spatial facts are computed here, not left to a planner.

Names used everywhere:
    tables        "A".."F"           targets "Table A"
    compartments  "S2-A3"            shelf S1..S3, row A (bottom) / B (top), column 1..4
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
META = json.loads((ROOT / "scene/library_meta.json").read_text())
SLOT_TOL = 0.03            # a slot is taken if a book's centre is within this of it (along the shelf)


def compartments():
    """{comp: (shelf, compartment meta)} in a fixed order: S1-A1 .. S1-B4, S2-A1, ..."""
    out = {}
    for shelf in sorted(META["shelves"]):
        for comp in sorted(META["shelves"][shelf]["compartments"]):
            out[comp] = (shelf, META["shelves"][shelf]["compartments"][comp])
    return out


def target_name(kind, where):
    """'Table B' / 'S2-A3' / 'held' / 'unknown'."""
    return f"Table {where}" if kind == "table" else (where if kind == "shelf" else kind)


def parking_of(target):
    """(x, y, yaw) where the robot parks for a target ('Table B', 'S2-A3'), or None."""
    if target.startswith("Table "):
        t = META["tables"].get(target[6:])
        return tuple(t["parking"][1]) if t else None
    comps = compartments()
    if target not in comps:
        return None
    shelf = comps[target][0]
    return tuple(META["shelves"][shelf]["parking"][target.split("-")[1][1:]])


def snapshot(robot):
    """The current world state from the simulator."""
    from skills import manipulation as mp
    books = {}
    for bid, info in META["books"].items():
        kind, where = mp.locate_book(robot, bid)
        p = robot.d.body(f"book_{bid}").xpos
        books[bid] = {"title": info["title"], "kind": kind, "where": where,
                      "at": target_name(kind, where), "pos": [round(float(v), 3) for v in p]}
    x, y, yaw = robot.get_pose()
    return build(books, (x, y, yaw), robot.holding)


def build(books, pose, holding):
    """World-state dictionary from book locations, the robot pose and the held book (also used by
    the validator to step the state forward without a simulator)."""
    tables = {}
    for t, info in META["tables"].items():
        on = [b for b, s in books.items() if s["kind"] == "table" and s["where"] == t]
        tables[t] = {"pos": info["pos"], "books": on, "edge_spots": len(info["edge_spots"]),
                     "free_edge_spots": free_edge_spots(t, books)}
    comps = {}
    for comp, (shelf, c) in compartments().items():
        inside = [b for b, s in books.items() if s["kind"] == "shelf" and s["where"] == comp]
        taken = sum(1 for q in c["positions"]
                    if any(abs(books[b]["pos"][0] - q[0]) < SLOT_TOL for b in inside))
        taken = max(taken, len(inside))
        comps[comp] = {"shelf": shelf, "row": comp.split("-")[1][0], "column": int(comp.split("-")[1][1:]),
                       "books": inside, "capacity": len(c["positions"]),
                       "free_slots": len(c["positions"]) - taken,
                       "pos": [round(v, 3) for v in c["positions"][len(c["positions"]) // 2]]}
    at = None                    # the target whose parking pose the robot is at (rows A and B of a
    for name in [f"Table {t}" for t in META["tables"]] + list(comps):     # column share one)
        p = parking_of(name)
        if math.hypot(pose[0] - p[0], pose[1] - p[1]) < 0.3:
            at = name
            break
    return {"robot": {"pose": [round(float(v), 3) for v in pose], "at": at, "holding": holding},
            "books": books, "tables": tables, "compartments": comps,
            "window": META["window_pos"][:2]}


def free_edge_spots(table, books):
    """Edge spots of a table with no book lying within 15 cm."""
    spots = META["tables"][table]["edge_spots"]
    on = [s["pos"] for s in books.values() if s["kind"] == "table" and s["where"] == table]
    return sum(1 for sp in spots if all(math.hypot(sp[0] - q[0], sp[1] - q[1]) > 0.15 for q in on))


def book_id(world, name):
    """Book id from a title or an id (case-insensitive, exact), or None."""
    n = name.strip().lower()
    for bid, s in world["books"].items():
        if n in (bid.lower(), s["title"].lower()):
            return bid
    return None


def same_spot(a, b):
    """True if the robot parks at the same pose for targets a and b (e.g. S2-A3 and S2-B3)."""
    if a is None or b is None:
        return False
    pa, pb = parking_of(a), parking_of(b)
    return pa is not None and pb is not None and math.hypot(pa[0] - pb[0], pa[1] - pb[1]) < 0.05


def distance(world, a, b):
    """Floor distance (m) between the parking poses of two targets ('Table B', 'S2-A3')."""
    pa, pb = parking_of(a), parking_of(b)
    return math.hypot(pa[0] - pb[0], pa[1] - pb[1])
