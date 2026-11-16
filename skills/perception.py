"""Book poses for the skills from the wrist camera + detector (instead of simulator ground truth).

Attach with `robot.perception = Perception()`; skills/manipulation.py then asks it for the pose of
the book it is about to drag / grasp. Every request moves the arm to look:
    table book   first look: camera 45 cm ahead of the arm mount, 40 cm above the table, looking
                 forward and down (also 35 cm to either side if the book is not in view, then
                 the same three looking farther, for books at the far side of the table);
                 second look: camera 40 cm straight
                 above the detected book (if the arm reaches it, else 20 or 35 cm nearer the
                 robot, looking at the book), which gives the full cover:
                 centre <= 0.2 cm, yaw <= 0.2 deg (perception/check_detect.py)
    shelf book   base 15 cm back, look at the compartment from 32 cm, then camera 32 cm
                 straight in front of the detected book
A lying book's spine is on the side of its length axis that faces the robot, a standing book's
spine faces the room (library convention, as the scene is laid out).
Not from the camera (yet): which table / compartment a book is in (world state, FIND_BOOK), the
free slots of a compartment (the held book hides the camera's view while placing), and the in-hand
correction.
"""
import math

import numpy as np

from perception.detect import BookDetector, aim_camera
from skills.manipulation import BD, META, UP, shelf_axes, table_frame, table_parking

LOOK_AHEAD, TABLE_LOOK_UP = 0.45, 0.40       # first look: camera 45 cm ahead of the arm mount
                                              # (inside the reach), 40 cm above the table
SECOND_LOOK_UP = 0.40
CLOSE_LOOK_CLEARANCE = 0.10                   # chassis this far from the table to look at a far book
SHELF_LOOK_BACK, SHELF_LOOK_DIST = 0.15, 0.32


class Perception:
    def __init__(self, detector=None, say=print):
        self.det = detector or BookDetector()
        self.say = say
        self.last = None                      # last Detection used, for reports

    # ------------------------------------------------------------------ helpers
    def _look(self, robot, cam, view, book, duration=2.0, z_range=None):
        q = aim_camera(robot, self.det, cam, view)
        if q is None:
            return None, False
        home = np.array(robot.meta["arm_home"])
        if np.max(np.abs(robot.arm_q() - home)) > 1.0:
            # via the (high) home pose: straight from the stow pose to a look pose the arm
            # swept low past the table edge and knocked an overhanging book off the table
            robot.home(1.5)
        robot.move_arm(q, duration, settle=0.3)
        hits = [x for x in self.det.detect(robot, z_range=z_range) if x.book == book]
        return (max(hits, key=lambda x: x.pixels) if hits else None), True

    # ------------------------------------------------------------------ table
    def table_book(self, robot, book, table, closer=True):
        """(centre, spine direction, yaw, flat) of a book lying on `table`, or None if not seen.
        closer: if the book is seen only partly and no second look is in reach (a book at the far
        side of the table, seen from the normal parking), drive up to the table in line with it
        and look again (the skills park for their next step anyway)."""
        fwd, left, edge, _ = table_frame(table)
        top = META["tables"][table]["top_z"]
        x, y, _ = robot.get_pose()
        mount = np.array([x, y, 0.0]) + robot.meta["mount_x"] * fwd
        zr = (top + 0.003, top + 0.08)         # cover and ribs only, not the floor past the edge
        found = None
        # first look, then to the sides; then the same looking farther (flatter): from the normal
        # parking the first view (31-88 deg below the horizon) misses books ~60 cm into the table
        for view in (0.6 * fwd - UP, fwd - 0.5 * UP):
            for dl in (0.0, 0.35, -0.35):
                cam = mount + LOOK_AHEAD * fwd + dl * left + np.array([0, 0, top + TABLE_LOOK_UP])
                found, ok = self._look(robot, cam, view, book, z_range=zr)
                if found is not None:
                    break
            if found is not None:
                break
        if found is None:
            return None
        if not found.complete:
            # second look straight above it; above a book at the far side of the table the arm
            # does not reach, then from 20 / 35 cm nearer the robot, looking at the book (seen
            # partly at a grazing angle from the first look, the cover's plane fit was off)
            reached = False
            for back in (0.0, 0.20, 0.35):
                cam = found.centre - back * fwd + np.array([0, 0, SECOND_LOOK_UP])
                again, reached = self._look(robot, cam, found.centre - cam, book, z_range=zr)
                if reached:
                    break
            if again is not None:
                found = again
            elif not reached and closer:
                robot.stow()
                robot.go_to(*table_parking(robot, table, float(found.centre @ left), CLOSE_LOOK_CLEARANCE),
                            tol_xy=0.01, tol_yaw=0.01)
                return self.table_book(robot, book, table, closer=False)
        self.last = found
        if not found.lying:
            return None
        if found.complete:
            a = np.array([math.cos(found.yaw), math.sin(found.yaw), 0.0])
            spine = np.array([-a[1], a[0], 0.0])
            if spine @ fwd > 0:               # the spine is on the robot's side
                spine = -spine
        else:
            # only part of the cover seen (the second look was out of reach): the rectangle's
            # long side may not be the book's, so use the table convention, spine to the robot;
            # the drag looks again from close by before it presses
            spine = -fwd
        yaw = math.atan2(spine[0], -spine[1]) % math.pi
        self.say(f"  seen: {META['books'][book]['title']} at {np.round(found.centre, 3)}, "
                 f"{'whole cover' if found.complete else 'partly'} (conf {found.conf:.2f})")
        return found.centre.copy(), spine, yaw, 1.0

    # ------------------------------------------------------------------ shelf
    def shelf_book(self, robot, book, comp):
        """Centre of a book standing in shelf compartment `comp`, or None. The robot must be at
        the compartment's column parking; it backs off for the look and returns."""
        shelf = comp.split("-")[0]
        fwd, side = shelf_axes(shelf)
        park = robot.get_pose()
        robot.stow()
        robot.go_to(park[0] - SHELF_LOOK_BACK * fwd[0], park[1] - SHELF_LOOK_BACK * fwd[1], park[2],
                    tol_xy=0.005, tol_yaw=0.005)
        pos = META["shelves"][shelf]["compartments"][comp]["positions"]
        c = np.mean([pos[0], pos[-1]], axis=0)
        spine_line = c @ fwd - BD["width"] / 2
        view = fwd - 0.18 * UP

        def cam_at(x_side, z):
            return (spine_line - SHELF_LOOK_DIST) * fwd + x_side * side + np.array([0, 0, z + 0.06])

        found, _ = self._look(robot, cam_at(c @ side, c[2]), view, book)
        if found is not None and not found.complete:
            again, reached = self._look(robot, cam_at(found.centre @ side, c[2]), view, book)
            if not reached:
                # out of reach (e.g. the book at the end of the shelf): slide the base along the
                # shelf until it is in front of the book, the camera pose is then like the first
                robot.stow()
                shift = (found.centre - c) @ side
                robot.go_to(park[0] - SHELF_LOOK_BACK * fwd[0] + shift * side[0],
                            park[1] - SHELF_LOOK_BACK * fwd[1] + shift * side[1], park[2],
                            tol_xy=0.005, tol_yaw=0.005)
                again, reached = self._look(robot, cam_at(found.centre @ side, c[2]), view, book)
            found = again or found
        robot.stow()
        robot.go_to(*park, tol_xy=0.005, tol_yaw=0.005)
        self.last = found
        if found is None or found.lying:
            return None
        self.say(f"  seen: {META['books'][book]['title']} in {comp} at {np.round(found.centre, 3)} "
                 f"(conf {found.conf:.2f})")
        return found.centre.copy()
