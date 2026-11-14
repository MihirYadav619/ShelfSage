"""Pick-and-place demo: Table B -> shelf S2-B3 -> back to Table B (Deep Learning book).

Grasp = fingers close around the book + the "attach" (weld) grasp, the README fallback; the
friction grasp replaces the weld in Phase 5. Everything else is the real Phase 3-4 stack:
parking poses, our own IK, straight-line Cartesian moves, gripper width.

    table:  pre-grasp -> straight in around the overhanging spine -> close + attach -> lift 10 cm
            -> straight back 16 cm (book clears the table) -> rotate the wrist 90 deg (book stands,
            spine towards the robot) -> carry pose
    shelf:  pre-place 15 cm out -> straight in -> down 3 cm onto the board -> release + open ->
            straight out ... then the reverse to take it out again
    table:  book turned flat on the way to behind the table edge -> straight in over the spot ->
            down onto the table (spine overhanging 3 cm) -> release -> back out

Usage (project root, venv active):
    python sim/demo_pick_place.py              # viewer, real time
    python sim/demo_pick_place.py --speed 2    # twice as fast
    python sim/demo_pick_place.py --headless   # no window: run it and print a report
"""
import math
import sys
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sim.robot import Robot  # noqa: E402
from sim.test_robot import META  # noqa: E402

FWD, UP, SIDE = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
BD = META["book_dims"]
BOOK = "dl"
TABLE, SPOT = "B", 1
SHELF, COMP = "S2", "S2-B3"
LIFT, BACK, CARRY_UP = 0.10, 0.16, 0.03
W_OPEN_T, W_OPEN_S, W_HOLD = 0.06, 0.05, 0.036
AISLE_X = 1.25


class Demo:
    def __init__(self, robot, say=print):
        self.r, self.say, self.log = robot, say, []

    def check(self, name, res):
        ok = res is not None and getattr(res, "success", True)
        self.log.append((name, ok))
        if not ok:
            self.say(f"    !! {name} failed: {res}")
        return ok

    def drive(self, waypoints):
        for wp in waypoints:
            res = self.r.go_to(*wp, tol_xy=0.02, tol_yaw=0.02)
            hits = [c for c in res.collisions if "book_" not in c]   # holding the book is fine
            self.log.append((f"drive to {np.round(wp[:2], 2)}", not hits and res.message != "timeout"))

    def carry_pose(self):
        """Book standing, held above the front of the chassis, for driving."""
        x, y, yaw = self.r.get_pose()
        f = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        p = np.array([x, y, 0.0]) + 0.45 * f + np.array([0, 0, 1.15])
        left = np.array([-f[1], f[0], 0.0])
        return self.check("carry pose", self.r.move_to(p, z_axis=f, y_axis=left, duration=2.5, allow=(f"book_{BOOK}",)))

    def run(self):
        r, d = self.r, self.r.d
        book_body = f"book_{BOOK}"
        self.say("stow the arm, drive to Table B")
        r.stow()
        self.drive([tuple(META["tables"][TABLE]["parking"][SPOT])])

        # ------------------------------------------------------------ pick from the table
        b = d.body(book_body)
        g = b.xpos - b.xmat.reshape(3, 3)[:, 1] * (BD["width"] / 2 - BD["grip_in"]["table"])
        self.say("unfold, pre-grasp in front of the spine")
        r.home(2.0)
        r.set_gripper_width(W_OPEN_T, 0.3)
        self.check("table pre-grasp", r.move_to(g - 0.10 * FWD, z_axis=FWD, y_axis=UP, duration=2.5))
        self.say("straight in around the overhanging spine, close, attach")
        self.check("table approach", r.move_linear(g, z_axis=FWD, y_axis=UP, speed=0.04))
        r.set_gripper_width(W_HOLD, 0.6)
        r.attach(BOOK)
        self.say("lift, back away from the table, turn the book upright")
        self.check("lift", r.move_linear(g + LIFT * UP, z_axis=FWD, y_axis=UP, speed=0.05))
        self.check("back off", r.move_linear(g + LIFT * UP - BACK * FWD, z_axis=FWD, y_axis=UP, speed=0.06))
        self.check("rotate upright", r.move_to(g + LIFT * UP - BACK * FWD, z_axis=FWD, y_axis=SIDE,
                                               duration=2.0, allow=(book_body,)))
        self.carry_pose()

        # ------------------------------------------------------------ put it in the shelf
        self.say("drive through the aisle to shelf S2")
        col = COMP.split("-")[1][1:]
        park = tuple(META["shelves"][SHELF]["parking"][col])
        self.drive([(AISLE_X, -2.6, math.pi / 2), (AISLE_X, 1.7, math.pi / 2), park])
        pos0 = np.array(META["shelves"][SHELF]["compartments"][COMP]["positions"][0])
        gs = np.array([pos0[0], pos0[1] - BD["width"] / 2 + BD["grip_in"]["shelf"], pos0[2]])
        self.say(f"pre-place in front of {COMP}, slide in, set it down, let go")
        self.check("shelf pre-place", r.move_to(gs + CARRY_UP * UP - 0.15 * FWD, z_axis=FWD, y_axis=SIDE,
                                                 duration=2.5, allow=(book_body,)))
        self.check("insert", r.move_linear(gs + CARRY_UP * UP, z_axis=FWD, y_axis=SIDE, speed=0.04))
        self.check("set down", r.move_linear(gs + 0.002 * UP, z_axis=FWD, y_axis=SIDE, speed=0.02))
        r.release(BOOK)
        r.set_gripper_width(W_OPEN_S, 0.5)
        self.check("withdraw", r.move_linear(gs + 0.002 * UP - 0.15 * FWD, z_axis=FWD, y_axis=SIDE, speed=0.06))
        r.wait(1.0)
        in_shelf = d.body(book_body).xpos.copy()
        self.log.append((f"book standing in {COMP}", abs(in_shelf[0] - pos0[0]) < 0.02 and
                         abs(in_shelf[2] - pos0[2]) < 0.02 and d.body(book_body).xmat[8] < 0.2))

        # ------------------------------------------------------------ take it out again
        self.say("take it out again")
        self.check("re-approach", r.move_linear(gs + 0.002 * UP, z_axis=FWD, y_axis=SIDE, speed=0.04))
        r.set_gripper_width(W_HOLD, 0.6)
        r.attach(BOOK)
        self.check("lift off the board", r.move_linear(gs + CARRY_UP * UP, z_axis=FWD, y_axis=SIDE, speed=0.02))
        self.check("pull out", r.move_linear(gs + CARRY_UP * UP - 0.15 * FWD, z_axis=FWD, y_axis=SIDE, speed=0.05))
        self.carry_pose()

        # ------------------------------------------------------------ back onto the table
        self.say("back to Table B")
        self.drive([(AISLE_X, 1.7, -math.pi / 2), (AISLE_X, -2.6, -math.pi / 2),
                    (AISLE_X, -3.0, math.pi / 2), tuple(META["tables"][TABLE]["parking"][SPOT])])
        spot = np.array(META["tables"][TABLE]["edge_spots"][SPOT])
        gt = np.array([spot[0], spot[1] - BD["width"] / 2 + BD["grip_in"]["table"], spot[2]])
        self.say("lay the book flat behind the table edge, slide it over the spot, set it down")
        self.check("behind the edge, lying flat", r.move_to(gt + LIFT * UP - BACK * FWD, z_axis=FWD, y_axis=UP,
                                                             duration=3.0, allow=(book_body,)))
        self.check("over the spot", r.move_linear(gt + LIFT * UP, z_axis=FWD, y_axis=UP, speed=0.05))
        self.check("set down", r.move_linear(gt + 0.002 * UP, z_axis=FWD, y_axis=UP, speed=0.03))
        r.release(BOOK)
        r.set_gripper_width(W_OPEN_T, 0.5)
        self.check("withdraw", r.move_linear(gt + 0.002 * UP - 0.10 * FWD, z_axis=FWD, y_axis=UP, speed=0.06))
        r.wait(1.0)
        on_table = d.body(book_body).xpos.copy()
        self.log.append(("book back on the edge spot (< 2 cm), lying flat",
                         np.linalg.norm(on_table[:2] - spot[:2]) < 0.02 and d.body(book_body).xmat[8] > 0.98))
        self.say("stow and go back to the start")
        r.stow()
        self.drive([(0.0, META["robot"]["start"][1], math.pi / 2)])
        return on_table, spot


def main():
    headless = "--headless" in sys.argv
    speed = float(sys.argv[sys.argv.index("--speed") + 1]) if "--speed" in sys.argv else 1.0
    model = mujoco.MjModel.from_xml_path(str(ROOT / "scene/library.xml"))
    data = mujoco.MjData(model)
    robot = Robot(model, data, pose_source="true")
    others0 = {model.body(i).name: data.xpos[i].copy() for i in range(model.nbody)
               if model.body(i).name.startswith("book_") and model.body(i).name != f"book_{BOOK}"}
    demo = Demo(robot, say=lambda s: print(f"  - {s}", flush=True))

    if headless:
        on_table, spot = demo.run()
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = [0.6, -1.5, 0.8]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 4.5, -120, -25
            robot.viewer, robot.speed = viewer, speed

            def follow():
                x, y, _ = robot.true_pose()
                viewer.cam.lookat[:2] += 0.04 * (np.array([x, y + 0.7]) - viewer.cam.lookat[:2])
            robot.on_sync = follow
            try:
                print("ShelfSage pick-and-place demo (close the window to stop)")
                robot.wait(1.0)
                on_table, spot = demo.run()
                print("  - done, close the window to exit")
                while viewer.is_running():
                    robot.wait(0.1)
            except KeyboardInterrupt:
                print("viewer closed")
                return

    moved = max(np.linalg.norm(data.body(n).xpos - p) for n, p in others0.items())
    demo.log.append(("other books undisturbed (< 1 cm)", moved < 0.01))
    print("\nreport:")
    for name, ok in demo.log:
        print(f"  {'ok ' if ok else 'XX '} {name}")
    print(f"  Deep Learning ended {np.linalg.norm(on_table[:2] - spot[:2]) * 100:.1f} cm from its spot; "
          f"other books moved at most {moved * 100:.2f} cm")


if __name__ == "__main__":
    main()
