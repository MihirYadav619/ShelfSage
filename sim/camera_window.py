"""A second window showing what the robot's wrist camera sees (run_demo.py).

Live view of the wrist camera a few times per second while the robot moves, and, whenever the
detector looks (FIND_BOOK, GRASP with --perception), its frame with every detected book outlined
and named, held for a moment so it can be read. OpenCV window; press nothing, it follows the
robot. Close it with the demo.
"""
import time

import mujoco
import numpy as np

TITLE = "ShelfSage - wrist camera"
LIVE_HZ = 6.0          # live frames per wall second
HOLD = 2.0             # s: a detector frame stays this long before the live view returns
W, H = 640, 480
PALETTE = [(60, 140, 230), (80, 190, 90), (200, 120, 50), (180, 80, 190), (40, 180, 200), (90, 90, 220)]  # BGR


class CameraWindow:
    def __init__(self, robot):
        import cv2
        from perception.detect import BookDetector
        self.cv2, self.robot = cv2, robot
        self.renderer = mujoco.Renderer(robot.m, H, W)
        self.last_live, self.hold_until = 0.0, 0.0
        cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(TITLE, W, H + 34)
        BookDetector.on_frame = self.on_detect
        self.live(force=True)

    def _upside_down(self):
        """True if the wrist camera is rolled so its image's up points down in the world."""
        m, d = self.robot.m, self.robot.d
        cam = m.camera(self.robot.meta["wrist_camera"]).id
        return d.cam_xmat[cam].reshape(3, 3)[2, 1] < 0

    def _show(self, bgr, caption, colour, upright=False):
        cv2 = self.cv2
        if not upright and self._upside_down():                 # for people only: the detector gets the raw image
            bgr = cv2.rotate(bgr, cv2.ROTATE_180)
        bar = np.full((34, bgr.shape[1], 3), (32, 38, 56), np.uint8)
        cv2.putText(bar, caption[:90], (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)
        cv2.imshow(TITLE, np.vstack([bar, bgr]))
        cv2.waitKey(1)

    def on_detect(self, bgr, items):
        """Detector frame: each detected book tinted, boxed and named (drawn here, after the
        image is turned upright, so the labels read the right way)."""
        cv2 = self.cv2
        img = bgr.copy()
        h, w = img.shape[:2]
        flip = self._upside_down()
        if flip:
            img = cv2.rotate(img, cv2.ROTATE_180)
        for k, (title, conf, (x1, y1, x2, y2), mask) in enumerate(items):
            col = PALETTE[k % len(PALETTE)]
            if flip:
                x1, y1, x2, y2 = w - x2, h - y2, w - x1, h - y1
                mask = None if mask is None else mask[::-1, ::-1]
            if mask is not None and mask.shape == img.shape[:2]:
                img[mask] = (0.6 * img[mask] + 0.4 * np.array(col)).astype(np.uint8)
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(img, p1, p2, col, 2)
            label = f"{title} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            ty = max(th + 6, p1[1])
            cv2.rectangle(img, (p1[0], ty - th - 6), (p1[0] + tw + 8, ty), col, -1)
            cv2.putText(img, label, (p1[0] + 4, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        if items:
            text = "DETECTOR: " + ", ".join(f"{t} {c:.2f}" for t, c, _, _ in sorted(items, key=lambda x: -x[1]))
            colour = (120, 220, 140)
        else:
            text, colour = "DETECTOR: no book in view", (120, 160, 240)
        self._show(img, text, colour, upright=True)
        self.hold_until = time.time() + HOLD

    def live(self, force=False):
        """Live wrist-camera frame (rate-limited); call it often (robot.on_sync)."""
        now = time.time()
        if not force and (now < self.hold_until or now - self.last_live < 1.0 / LIVE_HZ):
            self.cv2.waitKey(1)
            return
        self.last_live = now
        self.renderer.update_scene(self.robot.d, camera=self.robot.meta["wrist_camera"])
        rgb = self.renderer.render()
        self._show(rgb[..., ::-1].copy(), "LIVE  wrist camera", (230, 230, 230))

    def pump(self, ms=30):
        """Keep the window responsive while the robot is idle."""
        if time.time() >= self.hold_until:
            self.live()
        self.cv2.waitKey(ms)

    def close(self):
        from perception.detect import BookDetector
        BookDetector.on_frame = None
        self.renderer.close()
        self.cv2.destroyWindow(TITLE)
