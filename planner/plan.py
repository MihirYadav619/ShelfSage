"""Plans (lists of skill calls) and the plan validator.

A plan is a list of Action(name, args), e.g.
    [Action("NAVIGATE", ["Table B"]), Action("GRASP", ["Machine Learning"]), ...]
It converts to / from JSON ({"plan": [{"action": "NAVIGATE", "args": ["Table B"]}, ...]}) and to
the README's arrow notation (NAVIGATE(Table B) -> GRASP(Machine Learning) -> ...).

validate(plan, world) checks a plan before anything moves (both planners' plans go through it):
  * every action is a known skill with the right number of arguments,
  * every book and target exists,
  * stepping the world state forward symbolically, every precondition holds: GRASP needs an
    empty hand and the robot at the book's table / shelf column, PLACE needs the book in hand,
    the robot there and a free slot / edge spot, and a VERIFY of a book the plan has moved must
    expect it where the plan put it (a VERIFY of a book the plan does not touch is a question,
    "is it there?", and may answer no).
It returns every problem found (as text a planner can act on) and the predicted final world.
"""
import copy
import json
import math
from dataclasses import dataclass, field

from planner import world as W

# name -> argument names; the executor maps the same names to skills/interface.py
SIGNATURES = {
    "FIND_BOOK": ["book"],
    "NAVIGATE": ["target"],
    "GRASP": ["book"],
    "PLACE": ["book", "target"],
    "VERIFY": ["book", "target"],
}


@dataclass
class Action:
    name: str
    args: list = field(default_factory=list)

    def __str__(self):
        return f"{self.name}({', '.join(self.args)})"


def to_text(plan):
    return " -> ".join(str(a) for a in plan)


def to_json(plan):
    return {"plan": [{"action": a.name, "args": list(a.args)} for a in plan]}


def from_json(obj):
    """Plan from {"plan": [...]} or a bare list (a JSON string is parsed first). Raises ValueError
    with a readable reason if the shape is wrong."""
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except json.JSONDecodeError as e:
            raise ValueError(f"not valid JSON: {e}") from None
    steps = obj.get("plan") if isinstance(obj, dict) else obj
    if not isinstance(steps, list):
        raise ValueError('expected {"plan": [...]} or a list of steps')
    plan = []
    for i, s in enumerate(steps):
        if not isinstance(s, dict) or not isinstance(s.get("action"), str):
            raise ValueError(f"step {i + 1}: expected {{\"action\": ..., \"args\": [...]}}")
        args = s.get("args", [])
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            raise ValueError(f"step {i + 1}: args must be a list of strings")
        plan.append(Action(s["action"].strip().upper(), [a.strip() for a in args]))
    return plan


def canonical_target(name):
    """'Table B' or 'S2-A3' for a target string ('table b', 'shelf S2-A3', 's2-a3'), or None."""
    from skills.navigation import parse_target
    p = parse_target(name)
    if p is None:
        return None
    if p[0] == "table":
        return f"Table {p[1]}" if p[1] in W.META["tables"] else None
    if p[2] is None or p[2] not in W.compartments():
        return None
    return p[2]


@dataclass
class Validation:
    ok: bool
    errors: list
    world: dict            # predicted world after the plan (as far as it got)

    def __str__(self):
        return "valid" if self.ok else "; ".join(self.errors)


def validate(plan, world):
    """Check a plan against the world state without running it (see module docstring)."""
    w = copy.deepcopy(world)
    w["_moved"] = set()
    errors = []
    if not plan:
        return Validation(False, ["the plan is empty"], w)
    for i, a in enumerate(plan, 1):
        where = f"step {i} {a}"
        if a.name not in SIGNATURES:
            errors.append(f"{where}: unknown action (allowed: {', '.join(SIGNATURES)})")
            continue
        if len(a.args) != len(SIGNATURES[a.name]):
            errors.append(f"{where}: needs {len(SIGNATURES[a.name])} argument(s) "
                          f"({', '.join(SIGNATURES[a.name])})")
            continue
        err = _step(a, w)
        if err:
            errors.append(f"{where}: {err}")
    w.pop("_moved", None)
    return Validation(not errors, errors, w)


def _step(a, w):
    """Check one action against the symbolic world w and apply its effect; error text or None."""
    robot, books = w["robot"], w["books"]
    bid = W.book_id(w, a.args[0]) if a.name != "NAVIGATE" else None
    if a.name != "NAVIGATE" and bid is None:
        return f"no book called '{a.args[0]}'"
    title = books[bid]["title"] if bid else None
    if a.name == "FIND_BOOK":
        return None if books[bid]["kind"] != "unknown" else f"{title} is not on a table or in a shelf"
    if a.name == "NAVIGATE":
        t = canonical_target(a.args[0])
        if t is None:
            return f"unknown target '{a.args[0]}' (use 'Table A'..'Table F' or a compartment like 'S2-A3')"
        robot["at"], robot["pose"] = t, list(W.parking_of(t))
        return None
    if a.name == "GRASP":
        if robot["holding"] is not None:
            return f"the hand already holds {books[robot['holding']]['title']}"
        s = books[bid]
        if s["kind"] not in ("table", "shelf"):
            return f"{title} is {s['kind']}, cannot be grasped"
        if not W.same_spot(robot["at"], s["at"]):
            return f"{title} is at {s['at']}, the robot is at {robot['at'] or 'no table or shelf'}: NAVIGATE first"
        robot["holding"] = bid
        w["_moved"].add(bid)
        s.update(kind="held", where=None, at="held")
        _rebuild(w)
        return None
    t = canonical_target(a.args[1])
    if t is None:
        return f"unknown target '{a.args[1]}' (use 'Table A'..'Table F' or a compartment like 'S2-A3')"
    if a.name == "PLACE":
        if robot["holding"] != bid:
            return f"the hand does not hold {title}: GRASP it first"
        if not W.same_spot(robot["at"], t):
            return f"the robot is at {robot['at'] or 'no table or shelf'}, not at {t}: NAVIGATE there first"
        if t.startswith("Table "):
            if w["tables"][t[6:]]["free_edge_spots"] == 0:
                return f"{t} has no free edge spot"
            pos = _free_edge_spot(w, t[6:])
            books[bid].update(kind="table", where=t[6:], at=t, pos=pos)
        else:
            c = w["compartments"][t]
            if c["free_slots"] == 0:
                return f"{t} is full ({c['capacity']} books)"
            books[bid].update(kind="shelf", where=t, at=t, pos=_free_slot_pos(w, t))
        robot["holding"] = None
        _rebuild(w)
        return None
    if a.name == "VERIFY":
        if bid in w["_moved"] and books[bid]["at"] != t:
            return f"{title} would be at {books[bid]['at']}, not {t}: VERIFY would fail"
        return None
    return "not handled"


def _rebuild(w):
    nw = W.build(w["books"], w["robot"]["pose"], w["robot"]["holding"])
    nw["robot"]["at"] = w["robot"]["at"]
    moved = w.get("_moved", set())
    w.update(nw)
    w["_moved"] = moved


def _free_slot_pos(w, comp):
    shelf, c = W.compartments()[comp]
    xs = [w["books"][b]["pos"][0] for b in w["compartments"][comp]["books"]]
    for q in c["positions"]:
        if all(abs(q[0] - x) > W.SLOT_TOL for x in xs):
            return list(q)
    return list(c["positions"][-1])


def _free_edge_spot(w, table):
    on = [w["books"][b]["pos"] for b in w["tables"][table]["books"]]
    for sp in W.META["tables"][table]["edge_spots"]:
        if all(math.hypot(sp[0] - q[0], sp[1] - q[1]) > 0.15 for q in on):
            return list(sp)
    return list(W.META["tables"][table]["edge_spots"][0])
