"""Failure injection for Phase 9: something goes wrong while a plan runs.

    inj = Injector(robot, {"type": "drop_book", "before": {"action": "NAVIGATE", "args": ["S2-A3"]}})
    executor.run(plan, on_step=inj)          # the executor calls inj(i, action) before each step

The failure is injected once, just before the first step that matches `before` (action name,
and arguments compared as books / targets, so "ml" matches "Machine Learning"). It changes the
simulated world the way the failure would, so the next skills meet it as a real robot would:

    drop_book           the held book slips out and lands on the nearest table with a free edge
                        spot (lying flat); the hand is empty, the arm goes to the stow pose
    grasp_slip          the held book slips out and ends standing back in the compartment it was
                        taken from
    compartment_filled  other books (taken from far compartments) fill every free slot of
                        `target`, as if someone shelved them meanwhile
    path_blocked        the book cart is left in front of `target` (in its robot work zone)
    book_moved          `book` is moved to compartment `to` (someone moved it; the world model
                        is stale)

`moved_by_failure` lists the books the injection moved (other than the robot's own), so a
score does not count them as books the robot disturbed. Teleports set the free joint directly,
like the tests that place obstacles (Phase 6).
"""
import json
import math
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
META = json.loads((ROOT / "scene/library_meta.json").read_text())
QUAT_FLAT = [1.0, 0, 0, 0]
QUAT_STAND = [0.7071068, 0, 0.7071068, 0]          # as scene/generate_scene.py places shelf books
FILLERS = ["fluids", "heat", "materials", "numerics", "dsa", "infotheory", "crypto"]   # S3-B4, S1-B3, S3-B1


class Injector:
    def __init__(self, robot, spec, say=lambda s: None):
        self.robot, self.spec, self.say = robot, spec, say
        self.done = False
        self.description = ""
        self.moved_by_failure = []

    # ------------------------------------------------------------------ matching
    def matches(self, action):
        from planner import plan as P
        from planner import world as W
        want = self.spec["before"]
        if action.name != want["action"] or len(action.args) < len(want.get("args", [])):
            return False
        world = {"books": {b: {"title": i["title"]} for b, i in META["books"].items()}}
        for got, w in zip(action.args, want.get("args", [])):
            gb, wb = W.book_id(world, got), W.book_id(world, w)
            if gb is not None or wb is not None:
                if gb != wb:
                    return False
            elif P.canonical_target(got) != P.canonical_target(w):
                return False
        return True

    def __call__(self, i, action):
        if self.done or not self.matches(action):
            return
        self.done = True
        getattr(self, self.spec["type"])()
        self.say(f"FAILURE INJECTED before step {i} {action}: {self.description}")

    # ------------------------------------------------------------------ helpers
    def _set(self, body, pos, quat):
        m, d = self.robot.m, self.robot.d
        b = m.body(body)
        adr, dof = m.jnt_qposadr[b.jntadr[0]], b.dofadr[0]
        d.qpos[adr:adr + 3] = pos
        d.qpos[adr + 3:adr + 7] = quat
        d.qvel[dof:dof + 6] = 0
        mujoco.mj_forward(m, d)

    def _book_xy(self):
        return {b: self.robot.d.body(f"book_{b}").xpos[:2].copy() for b in META["books"]}

    def _free_slots(self, comp):
        from skills import manipulation as mp
        shelf = comp.split("-")[0]
        slots = META["shelves"][shelf]["compartments"][comp]["positions"]
        xs = [self.robot.d.body(f"book_{b}").xpos[0] for b in mp.books_in(self.robot, comp)]
        return [s for s in slots if all(abs(s[0] - x) > mp.SLOT_TOL for x in xs)]

    def _let_go(self):
        r = self.robot
        book = r.holding
        if book is None:
            raise RuntimeError(f"{self.spec['type']}: the robot holds no book at the injection point")
        r.release(book)
        r.open_gripper(0.3)
        return book

    # ------------------------------------------------------------------ failures
    def drop_book(self):
        r = self.robot
        book = self._let_go()
        x, y = r.true_pose()[:2]
        xy = self._book_xy()
        best = None
        for t, info in META["tables"].items():
            for sp in info["edge_spots"]:
                free = all(math.hypot(sp[0] - q[0], sp[1] - q[1]) > 0.15 for b, q in xy.items() if b != book)
                dist = math.hypot(sp[0] - x, sp[1] - y)
                if free and (best is None or dist < best[0]):
                    best = (dist, t, sp)
        _, t, sp = best
        self._set(f"book_{book}", sp, QUAT_FLAT)
        r.stow()
        self.description = f"{META['books'][book]['title']} slipped out of the gripper and fell onto Table {t}"

    def grasp_slip(self):
        from skills import manipulation as mp
        r = self.robot
        book = self._let_go()
        src = META["books"][book]["location"].split(":")[1]
        slot = self._free_slots(src)[0]
        self._set(f"book_{book}", slot, QUAT_STAND)
        r.stow()
        self.description = f"{META['books'][book]['title']} slipped out while being pulled and stayed in {src}"
        assert mp.locate_book(r, book) == ("shelf", src)

    def compartment_filled(self):
        comp = self.spec["target"]
        held = self.robot.holding
        used = []
        for slot, b in zip(self._free_slots(comp), [f for f in FILLERS if f != held]):
            self._set(f"book_{b}", slot, QUAT_STAND)
            used.append(b)
        self.moved_by_failure += used
        self.robot.wait(0.3)
        self.description = (f"{comp} was filled with {len(used)} other books meanwhile "
                            f"({', '.join(META['books'][b]['title'] for b in used)})")

    def path_blocked(self):
        target = self.spec["target"]
        t = META["tables"][target.split()[-1]]
        x, y, _ = t["parking"][1]
        self._set("obstacle_cart", [x, y, 0.46], QUAT_FLAT)
        self.robot.wait(0.3)
        self.description = f"a book cart was left in front of {target}, in its robot work zone"

    def book_moved(self):
        book, comp = self.spec["book"], self.spec["to"]
        self._set(f"book_{book}", self._free_slots(comp)[0], QUAT_STAND)
        self.robot.wait(0.3)
        self.description = f"someone moved {META['books'][book]['title']} to {comp}"
