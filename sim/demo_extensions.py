"""Demo of the Phase 5 extensions in the viewer, with the wrist camera (and what the detector sees
in it) in the corner: a book in the way is pushed aside, a far book is reached, a book lying at
an angle is turned straight while it is dragged, all in one stroke, then grasped.

Usage (project root, venv active):
    python sim/demo_extensions.py                    # all three at once: Calculus 55 cm into
                                                     # Table A, turned 30 deg, Linear Algebra in its way
    python sim/demo_extensions.py --case blocker     # Table C: Signals and Systems in the way of
                                                     # Control Systems
    python sim/demo_extensions.py --case far         # Deep Learning at the far side of Table B
    python sim/demo_extensions.py --case turned      # Thermodynamics turned 40 deg on Table F
    python sim/demo_extensions.py --speed 2          # twice as fast
    python sim/demo_extensions.py --truth            # book poses from ground truth, not the camera
    python sim/demo_extensions.py --flip             # if the camera image shows upside down
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
from skills import interface as sk  # noqa: E402
from skills.manipulation import META, table_frame  # noqa: E402

# case: (target book, table, [(book, x, spine depth from the edge, yaw deg), ...])
CASES = {
    "all": ("calculus", "A", [("calculus", -2.45, 0.55, 30.0), ("linalg", -2.45, 0.05, 0.0)]),
    "blocker": ("control", "C", []),
    "far": ("dl", "B", [("dl", 0.0, 0.58, 0.0)]),
    "turned": ("thermo", "F", [("thermo", 2.45, 0.25, -40.0)]),
}
CAM_W, CAM_H = 400, 300          # size of the camera picture in the viewer (pixels)
CAM_EVERY = 0.2                  # s of simulated time between camera pictures


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def put_book(model, data, book, table, x, depth, yaw_deg):
    """Book lying on `table`, middle of its spine `depth` in from the robot-side edge at world x,
    turned by yaw_deg."""
    yaw = math.radians(yaw_deg)
    _, _, edge, _ = table_frame(table)              # all tables: robot side at -y
    mid = np.array([x, edge + depth])
    c = mid - np.array([math.sin(yaw), -math.cos(yaw)]) * META["book_dims"]["width"] / 2
    adr = model.jnt_qposadr[model.body(f"book_{book}").jntadr[0]]
    data.qpos[adr:adr + 2] = c
    data.qpos[adr + 3:adr + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    mujoco.mj_forward(model, data)


class CameraPanel:
    """Wrist camera picture, with the detector's masks and labels, in the viewer's top-right
    corner; the current step as text in the top-left."""

    def __init__(self, robot, viewer, detector, flip):
        self.robot, self.viewer, self.det, self.flip = robot, viewer, detector, flip
        self.next_t = 0.0
        self.status = ""

    def say(self, s):
        print(f"  - {s}", flush=True)
        self.status = s
        self.viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               s, None))

    def __call__(self):
        if self.robot.d.time < self.next_t:
            return
        self.next_t = self.robot.d.time + CAM_EVERY
        import cv2
        rgb, _ = self.robot.get_rgbd()
        if self.det is not None:
            res = self.det.model.predict(rgb[..., ::-1].copy(), conf=self.det.conf, device=self.det.device,
                                         verbose=False, retina_masks=True)[0]
            rgb = res.plot(line_width=2, font_size=14)[..., ::-1]     # BGR -> RGB
        img = cv2.resize(np.ascontiguousarray(rgb), (CAM_W, CAM_H), interpolation=cv2.INTER_AREA)
        cv2.putText(img, "wrist camera", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if self.flip:
            img = img[::-1]
        vp = self.viewer.viewport
        if vp is None:
            return
        rect = mujoco.MjrRect(vp.width - CAM_W - 10, vp.height - CAM_H - 10, CAM_W, CAM_H)
        self.viewer.set_images((rect, np.ascontiguousarray(img)))


def main():
    case = arg("--case", "all")
    speed = float(arg("--speed", 1.0))
    use_camera = "--truth" not in sys.argv
    book, table, poses = CASES[case]
    model = mujoco.MjModel.from_xml_path(str(ROOT / "scene/library.xml"))
    data = mujoco.MjData(model)
    robot = Robot(model, data, pose_source="true")
    for b, x, depth, yaw in poses:
        put_book(model, data, b, table, x, depth, yaw)
    det = None
    from perception.detect import BookDetector
    det = BookDetector()
    if use_camera:
        from skills.perception import Perception
        robot.perception = Perception(det, say=lambda s: None)
    log = []
    print(f"ShelfSage extensions demo '{case}': {META['books'][book]['title']} on Table {table}, "
          f"book poses from {'the camera' if use_camera else 'ground truth'}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        tx, ty = META["tables"][table]["pos"]
        viewer.cam.lookat[:] = [tx, ty - 0.5, 0.8]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.8, -120, -35
        panel = CameraPanel(robot, viewer, det, "--flip" in sys.argv)
        robot.viewer, robot.speed, robot.on_sync = viewer, speed, panel
        sk.bind(robot, say=panel.say, log=log)
        try:
            print("(close the window to stop)")
            robot.wait(1.0)
            for name, fn, a in (("NAVIGATE", sk.navigate, f"Table {table}"), ("GRASP", sk.grasp, book)):
                res = fn(a)
                print(f"{'ok' if res.success else 'XX'} {name}({a}): {res.message}")
                if not res.success:
                    break
            panel.say("done, close the window to exit")
            while viewer.is_running():
                robot.wait(0.1)
        except KeyboardInterrupt:
            print("viewer closed")
    print("\nsteps:")
    for s, ok in log:
        print(f"  {'ok ' if ok else 'XX '} {s}")


if __name__ == "__main__":
    main()
