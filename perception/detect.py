"""Book detection and pose from the wrist camera: YOLOv8-seg mask + depth -> 3D points -> pose.

    det = BookDetector()                       # loads perception/runs/books/weights/best.pt
    books = det.detect(robot)                  # list of Detection, from one RGB-D frame

For each detected book: the mask pixels with depth become 3D points in the world frame. The
camera pose comes from the robot's own estimate (base pose from get_pose() + arm FK + the fixed
camera mount), not from the simulator, so localisation error shows up here like on a real robot.
From the points (the visible top cover of a lying book / the spine of a standing one):
    lying       whether the surface normal points up (cover) or sideways (spine in a shelf)
    rectangle   smallest rectangle around the points in that surface (cv2.minAreaRect): its
                centre, its long side = the book's length axis (yaw), its size
    complete    the rectangle is about the book's size (22 x 15 cm cover / 22 x 4 cm spine
                with ribs), i.e. the whole face was seen; otherwise the centre is off (a book cut
                by the image border or hidden behind the fingers) and the robot should look again
    centre      rectangle centre moved half the thickness (lying) / half the width (standing)
                into the book along the normal: the camera sees a surface, skills want the body
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
META = json.loads((ROOT / "scene/library_meta.json").read_text())
BD = META["book_dims"]
WEIGHTS = ROOT / "perception/runs/books/weights/best.pt"
BOOKS = list(META["books"])


@dataclass
class Detection:
    book: str                  # book id
    conf: float
    pixels: int                # mask size
    points: np.ndarray         # (N, 3) world points on the visible surface
    centre: np.ndarray         # estimated body centre (world)
    normal: np.ndarray         # visible surface normal (unit, towards the camera)
    yaw: float                 # lying: angle of the length axis in the xy plane (mod pi)
    lying: bool
    size: tuple = (0.0, 0.0)   # fitted rectangle (long, short side), m
    complete: bool = False     # the whole face was seen


class BookDetector:
    def __init__(self, weights=WEIGHTS, conf=0.5, device=0):
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.conf, self.device = conf, device
        self._T_tcp_cam = None

    def camera_pose(self, robot):
        """World <- camera (4x4, MuJoCo camera frame: x right, y up, looking along -z), from the
        robot's estimated base pose and its own arm FK."""
        return robot.fk_world(robot.arm_q()) @ self._camera_offset(robot)

    def _camera_offset(self, robot):
        """TCP <- camera, the fixed camera mount on the hand."""
        if self._T_tcp_cam is None:                       # fixed mount, read once from the model
            m, d = robot.m, robot.d
            mujoco.mj_forward(m, d)
            cam = m.camera(robot.meta["wrist_camera"]).id
            T_w_tcp, T_w_cam = np.eye(4), np.eye(4)
            T_w_tcp[:3, :3], T_w_tcp[:3, 3] = d.site_xmat[robot.tcp].reshape(3, 3), d.site_xpos[robot.tcp]
            T_w_cam[:3, :3], T_w_cam[:3, 3] = d.cam_xmat[cam].reshape(3, 3), d.cam_xpos[cam]
            self._T_tcp_cam = np.linalg.inv(T_w_tcp) @ T_w_cam
        return self._T_tcp_cam

    def detect(self, robot, rgb=None, depth=None):
        if rgb is None:
            rgb, depth = robot.get_rgbd()
        K = robot.camera_intrinsics()
        res = self.model.predict(rgb[..., ::-1].copy(), conf=self.conf, device=self.device, verbose=False,
                                 retina_masks=True)[0]            # ultralytics expects BGR arrays
        if res.masks is None:
            return []
        T = self.camera_pose(robot)
        out = []
        masks = res.masks.data.cpu().numpy() > 0.5
        import cv2
        kernel = np.ones((7, 7), np.uint8)              # erode 3 px: edge pixels often see what
        for mask, cls, conf in zip(masks, res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            mask = cv2.erode(mask.astype(np.uint8), kernel) > 0     # is behind the book (floor)
            vs, us = np.nonzero(mask & (depth > 0.01) & (depth < 3.0))
            if len(us) < 50:
                continue
            z = depth[vs, us]
            pc = np.stack([(us - K[0, 2]) / K[0, 0] * z, -(vs - K[1, 2]) / K[1, 1] * z, -z], axis=1)
            pw = pc @ T[:3, :3].T + T[:3, 3]
            out.append(self._pose(BOOKS[int(cls)], float(conf), len(us), pw, T[:3, 3]))
        return out

    @staticmethod
    def _pose(book, conf, n, pw, cam_pos):
        import cv2
        for _ in range(2):                          # plane fit, then drop points off the plane
            c = np.median(pw, axis=0)
            _, _, vt = np.linalg.svd(pw - c, full_matrices=False)
            normal = vt[2] / np.linalg.norm(vt[2])
            keep = np.abs((pw - c) @ normal) < 0.015
            if keep.sum() > 30:
                pw = pw[keep]
        c = pw.mean(axis=0)
        if normal @ (cam_pos - c) < 0:
            normal = -normal
        lying = abs(normal[2]) > 0.7
        if lying:                                   # cover: rectangle in the horizontal plane
            normal = np.array([0.0, 0.0, 1.0])
            e1, e2 = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
            expect = (BD["length"], BD["width"])
        else:                                       # spine: vertical plane facing the camera
            normal = np.array([normal[0], normal[1], 0.0]) / np.linalg.norm(normal[:2])
            e1, e2 = np.array([-normal[1], normal[0], 0.0]), np.array([0, 0, 1.0])
            expect = (BD["length"], BD["thickness"] + 2 * BD["rib_height"])
        uv = np.stack([(pw - c) @ e1, (pw - c) @ e2], axis=1).astype(np.float32)
        (cu, cv_), (w, h), ang = cv2.minAreaRect(uv)
        long_, short = max(w, h), min(w, h)
        a = math.radians(ang) + (0.0 if w >= h else math.pi / 2)      # direction of the long side
        surface = c + cu * e1 + cv_ * e2
        if lying:
            surface[2] = np.median(pw[:, 2])        # the cover, not the ribs
            yaw = a % math.pi
        else:
            yaw = 0.0
        complete = abs(long_ - expect[0]) < 0.02 and abs(short - expect[1]) < 0.02
        depth_in = BD["thickness"] / 2 if lying else BD["width"] / 2
        return Detection(book, conf, n, pw, surface - normal * depth_in, normal, yaw, lying,
                         (float(long_), float(short)), bool(complete))


def aim_camera(robot, det, cam_pos, view_dir, iters=3):
    """Arm configuration that puts the wrist camera at cam_pos looking along view_dir (e.g. 40 cm
    straight above a book). The camera sits 6 cm beside the TCP, so the TCP target is corrected a
    few times with the FK of the solution. Returns q or None."""
    view_dir = np.asarray(view_dir, float) / np.linalg.norm(view_dir)
    tcp = np.asarray(cam_pos, float).copy()
    q_save = robot.d.ctrl[robot.arm_act].copy()
    q = None
    try:
        for _ in range(iters):
            res = robot.ik(tcp, z_axis=view_dir, n_random=4)
            if res is None or not res.success:
                return None
            q = res.q
            robot.d.ctrl[robot.arm_act] = q                    # next IK starts from here
            cam = (robot.fk_world(q) @ det._camera_offset(robot))[:3, 3]
            tcp = tcp + (np.asarray(cam_pos) - cam)
        return q
    finally:
        robot.d.ctrl[robot.arm_act] = q_save
