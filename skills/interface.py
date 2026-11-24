"""The skills the planner may call (README section 6). Each returns a SkillResult, so a failure and
its reason can go back to the LLM for replanning.

Bind a robot first:
    from skills import interface as sk
    sk.bind(robot)                       # robot = sim.robot.Robot(...)
    sk.navigate("Table B"); sk.grasp("Machine Learning"); sk.navigate("S2-A3")
    sk.place("Machine Learning", "S2-A3"); sk.verify("Machine Learning", "S2-A3")

Books can be named by title ("Machine Learning", any case) or id ("ml"). Book poses come from
simulator ground truth for now (the detector replaces that); navigation drives fixed lanes until
A* in Phase 6; VERIFY checks ground truth until the camera check of Phase 9.
"""
import numpy as np

from skills import manipulation as mp
from skills import navigation as nav
from skills import search
from skills.result import SkillResult

_ctx = {"robot": None, "say": print, "log": []}


def bind(robot, say=print, log=None):
    """Use this robot for all skills. `log` collects (step, ok) pairs of every sub-step."""
    _ctx.update(robot=robot, say=say, log=[] if log is None else log)


def _robot():
    if _ctx["robot"] is None:
        raise RuntimeError("call skills.interface.bind(robot) first")
    return _ctx["robot"]


def book_id(name):
    """Book id from a title or an id (case-insensitive), or None."""
    n = name.strip().lower()
    for bid, info in mp.META["books"].items():
        if n in (bid.lower(), info["title"].lower()):
            return bid
    return None


def _where(loc):
    return f"Table {loc[1]}" if loc[0] == "table" else (loc[1] if loc[0] == "shelf" else loc[0])


# ---------------------------------------------------------------------------- skills
def find_book(title: str) -> SkillResult:
    """Find the book. Search mode (skills/search.py): drive to the tables (then the shelves) and
    look with the wrist camera until it is seen; the robot stays parked there. Otherwise: look up
    which table / compartment holds it (simulator ground truth, the benchmark setting)."""
    bid = book_id(title)
    if bid is None:
        return SkillResult(False, f"no book called '{title}'")
    if search.active(_robot()):
        _ctx["say"](f"FIND_BOOK {mp.META['books'][bid]['title']} (searching with the camera)")
        return search.find(_robot(), bid, say=_ctx["say"], log=_ctx["log"])
    loc = mp.locate_book(_robot(), bid)
    if loc[0] == "unknown":
        return SkillResult(False, f"{mp.META['books'][bid]['title']} is not on a table or in a shelf")
    return SkillResult(True, f"{mp.META['books'][bid]['title']} is {'held' if loc[0] == 'held' else 'at ' + _where(loc)}",
                       {"book": bid, "kind": loc[0], "where": loc[1]})


def look(target: str) -> SkillResult:
    """Look over a table: drive there and see which books lie on it (wrist camera + detector in
    search mode; the world state otherwise). The planner plans again with what was seen."""
    r = _robot()
    if search.active(r):
        _ctx["say"](f"LOOK {target}")
        return search.look(r, target, say=_ctx["say"], log=_ctx["log"])
    p = nav.parse_target(target)
    if p is None or p[0] != "table":
        return SkillResult(False, f"LOOK takes a table ('Table A'..'Table F'), not '{target}'")
    res = navigate(f"Table {p[1]}")
    if not res.success:
        return res
    on = [mp.META["books"][b]["title"] for b in mp.META["books"] if mp.locate_book(r, b) == ("table", p[1])]
    return SkillResult(True, f"on Table {p[1]} lie " + (", ".join(on) if on else "no books"), {"books": on})


def navigate(target: str) -> SkillResult:
    """Drive the base to the parking pose of a table or shelf, e.g. 'Table B' or 'S2-A3'."""
    r = _robot()
    goal = nav.parking_pose(target)
    if goal is None:
        return SkillResult(False, f"unknown target '{target}' (use 'Table A'..'Table F' or e.g. 'S2-A3')")
    if not nav.near(r, goal):
        _ctx["say"](f"NAVIGATE {target}")
        ok, msg = nav.navigate(r, goal, say=_ctx["say"], log=_ctx["log"], holding=r.holding)
        if not ok:
            return SkillResult(False, msg)
    return SkillResult(True, f"at {target}", {"pose": goal})


def grasp(obj: str) -> SkillResult:
    """Pick up a book with the 2-finger gripper (friction grasp) and bring it to the carry pose.
    On a table: dragged to the edge first if it lies inside, then grasped by its spine. In a shelf:
    pulled out by its spine. The robot must already be at that table / shelf (NAVIGATE)."""
    r, say, log = _robot(), _ctx["say"], _ctx["log"]
    bid = book_id(obj)
    if bid is None:
        return SkillResult(False, f"no book called '{obj}'")
    if r.holding is not None:
        return SkillResult(False, f"already holding {mp.META['books'][r.holding]['title']}")
    loc = mp.locate_book(r, bid)
    if search.active(r):
        title = mp.META['books'][bid]['title']
        known = r.memory.get(bid)
        if known is None:
            return SkillResult(False, f"where {title} is is not known: FIND_BOOK it first")
        if (known["kind"], known["where"]) != loc:
            search.forget(r, bid)
            return SkillResult(False, f"{title} is no longer at {search.target_of(known)} "
                                      "(someone moved it): FIND_BOOK it again")
    say(f"GRASP {mp.META['books'][bid]['title']} ({_where(loc)})")
    if loc[0] == "table":
        here = nav.parking_pose(f"Table {loc[1]}")
        if abs(r.get_pose()[1] - here[1]) > 0.6 or abs(r.get_pose()[0] - here[0]) > 1.0:
            return SkillResult(False, f"the book is at Table {loc[1]}: NAVIGATE there first")
        res = mp.drag_to_edge(r, bid, loc[1], say=say, log=log)
        if not res.success:
            return SkillResult(False, f"could not drag it to the table edge: {res.message}", res.data)
        res = mp.pick_from_table_edge(r, bid, loc[1], say=say, log=log)
        if not res.success:
            return res
        mp.to_carry_pose(r, bid, log)
    elif loc[0] == "shelf":
        goal = nav.parking_pose(loc[1])
        if not nav.near(r, goal, tol=0.6):
            return SkillResult(False, f"the book is in {loc[1]}: NAVIGATE there first")
        if not nav.near(r, goal):
            r.stow()
            r.go_to(*goal, tol_xy=0.01, tol_yaw=0.01)
        res = mp.pick_from_shelf(r, bid, loc[1], say=say, log=log)
        if not res.success:
            return res
    else:
        return SkillResult(False, f"cannot grasp it: the book is {loc[0]}")
    held = r.holding == bid and r.d.body(f"book_{bid}").xpos[2] > 0.95
    if search.active(r):
        search.forget(r, bid)                    # in the hand (or dropped somewhere unknown)
    return SkillResult(held, "holding the book" if held else "lost the book after grasping it")


def place(obj: str, shelf: str) -> SkillResult:
    """Put the held book into a shelf compartment, e.g. 'S2-A3' (shelf S1/S2/S3, row A/B, column
    1-4), in its leftmost free slot, or lay it on a table ('Table A') at a free edge spot. The
    robot must be at that shelf / table (NAVIGATE)."""
    r, say, log = _robot(), _ctx["say"], _ctx["log"]
    bid = book_id(obj)
    if bid is None:
        return SkillResult(False, f"no book called '{obj}'")
    if r.holding != bid:
        return SkillResult(False, f"not holding {mp.META['books'][bid]['title']}: GRASP it first")
    p = nav.parse_target(shelf)
    if p is not None and p[0] == "table":
        here = nav.parking_pose(f"Table {p[1]}")
        if abs(r.get_pose()[1] - here[1]) > 0.6 or abs(r.get_pose()[0] - here[0]) > 1.0:
            return SkillResult(False, f"not at Table {p[1]}: NAVIGATE there first")
        say(f"PLACE {mp.META['books'][bid]['title']} on Table {p[1]}")
        return _placed(r, bid, mp.place_on_table(r, bid, p[1], say=say, log=log))
    if p is None or p[0] != "shelf" or p[2] is None:
        return SkillResult(False, f"'{shelf}' is not a shelf compartment like 'S2-A3' or a table like 'Table A'")
    comp = p[2]
    goal = nav.parking_pose(comp)
    if goal is None:
        return SkillResult(False, f"no compartment {comp}")
    if not nav.near(r, goal, tol=0.6):
        return SkillResult(False, f"not at shelf {p[1]}: NAVIGATE to {comp} first")
    say(f"PLACE {mp.META['books'][bid]['title']} in {comp}")
    if not nav.near(r, goal):
        r.go_to(*goal, tol_xy=0.01, tol_yaw=0.01)
    return _placed(r, bid, mp.place_in_shelf(r, bid, comp, say=say, log=log))


def _placed(r, bid, res):
    """Search mode: remember where a placed book went (the robot put it there)."""
    if search.active(r) and r.holding != bid:
        kind, where = mp.locate_book(r, bid)
        if kind in ("table", "shelf"):
            search.remember(r, bid, kind, where, r.d.body(f"book_{bid}").xpos)
    return res


def verify(obj: str, shelf: str) -> SkillResult:
    """Check with simulator ground truth that obj stands inside the shelf compartment (or lies
    flat on the table, for 'Table A')."""
    r = _robot()
    bid = book_id(obj)
    if bid is None:
        return SkillResult(False, f"no book called '{obj}'")
    p = nav.parse_target(shelf)
    loc = mp.locate_book(r, bid)
    if p is not None and p[0] == "table":
        flat = abs(r.d.body(f"book_{bid}").xmat[8]) > 0.98
        ok = loc == ("table", p[1]) and flat and r.holding != bid
        msg = (f"{mp.META['books'][bid]['title']} lies on Table {p[1]}" if ok else
               f"{mp.META['books'][bid]['title']} is {'held' if loc[0] == 'held' else 'at ' + _where(loc)}"
               f"{'' if flat else ', not lying flat'}")
        return SkillResult(bool(ok), msg, {"where": loc[1], "pos": np.round(r.d.body(f'book_{bid}').xpos, 3).tolist()})
    comp = p[2] if p is not None and p[0] == "shelf" else None
    standing = abs(r.d.body(f"book_{bid}").xmat[8]) < 0.2
    ok = loc == ("shelf", comp) and standing and r.holding != bid
    msg = (f"{mp.META['books'][bid]['title']} stands in {comp}" if ok else
           f"{mp.META['books'][bid]['title']} is {'held' if loc[0] == 'held' else 'at ' + _where(loc)}"
           f"{'' if standing else ', not standing'}")
    return SkillResult(bool(ok), msg, {"where": loc[1], "standing": bool(standing),
                                       "pos": np.round(r.d.body(f'book_{bid}').xpos, 3).tolist()})


# The LLM planner will only be allowed to call actions listed here.
SKILLS = {
    "FIND_BOOK": find_book,
    "LOOK": look,
    "NAVIGATE": navigate,
    "GRASP": grasp,
    "PLACE": place,
    "VERIFY": verify,
}
