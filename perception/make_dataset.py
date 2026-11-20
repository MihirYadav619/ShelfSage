"""Synthetic training data for the book detector (YOLOv8-seg, one class per book, 39 classes).

Every sample is a real wrist-camera image (640 x 480, fovy 58): the robot is put at a random spot
in front of a table or shelf, the arm is moved by our IK so the wrist camera looks at the books,
and the scene is rendered. No physics is stepped; books are placed directly. Labels come for free
from MuJoCo's segmentation render (geom id per pixel -> book), turned into YOLO polygons.

Two kinds of views (as the robot will see them):
    table  2-6 random books (any of the 39) lying anywhere on a random table, any yaw; camera
           30-50 cm above the table looking down, tilted up to 25 deg, any roll
    shelf  1-5 random books standing packed in one or two compartments of a random shelf; camera
           10-30 cm in front of the spines (robot at the column parking), looking in, tilted up
           to 15 deg
Domain randomization: light intensity and direction, robot position, camera pose, which books,
where they lie. Books not in the sample are moved out of sight.

Writes perception/data/{images,labels}/{train,val}/*.png|txt and perception/data/books.yaml.

Usage (project root, venv active):
    python perception/make_dataset.py --n 4000          # ~85 % train, 15 % val
    python perception/make_dataset.py --n 20 --preview  # also save label overlays to check
"""
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sim.robot import Robot  # noqa: E402

META = json.loads((ROOT / "scene/library_meta.json").read_text())
BD = META["book_dims"]
OUT = ROOT / "perception/data"
BOOKS = list(META["books"])                 # class index = position in this list
MIN_PIXELS = 150                            # smaller visible parts are not labelled
QUAT_FLAT = np.array([1.0, 0, 0, 0])
QUAT_STAND = np.array([0.7071068, 0, 0.7071068, 0])
HIDE = np.array([0.0, -20.0, -5.0])         # far below the floor


def yaw_quat(yaw, base=QUAT_FLAT):
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]), base)
    return q


class Generator:
    def __init__(self, seed=0):
        self.m = mujoco.MjModel.from_xml_path(str(ROOT / "scene/library.xml"))
        self.d = mujoco.MjData(self.m)
        self.r = Robot(self.m, self.d, pose_source="true")
        self.rng = np.random.default_rng(seed)
        self.jnt = {b: self.m.joint(f"book_{b}_joint").qposadr[0] for b in BOOKS}
        body_of = {self.m.body(f"book_{b}").id: k for k, b in enumerate(BOOKS)}
        self.geom_class = np.full(self.m.ngeom + 1, -1, dtype=np.int32)   # last = background (-1)
        for g in range(self.m.ngeom):
            self.geom_class[g] = body_of.get(self.m.geom_bodyid[g], -1)
        self.light0 = self.m.light_diffuse.copy()
        self.lightdir0 = self.m.light_dir.copy()
        cam = self.r.meta["wrist_cam"]
        self.ren = mujoco.Renderer(self.m, cam["height"], cam["width"])
        self.cam = self.r.meta["wrist_camera"]

    # ------------------------------------------------------------------ scene setup
    def set_book(self, b, pos, quat):
        a = self.jnt[b]
        self.d.qpos[a:a + 3] = pos
        self.d.qpos[a + 3:a + 7] = quat

    def hide_all(self):
        for k, b in enumerate(BOOKS):
            self.set_book(b, HIDE + np.array([0.3 * k, 0, 0]), QUAT_FLAT)

    def set_base(self, x, y, yaw):
        for a, v in zip(self.r.qadr, (x, y, yaw)):
            self.d.qpos[a] = v
        self.d.qvel[:] = 0

    def randomize_light(self):
        rng = self.rng
        self.m.light_diffuse[:] = self.light0 * rng.uniform(0.45, 1.5)
        d = self.lightdir0 + rng.normal(0, 0.25, self.lightdir0.shape)
        self.m.light_dir[:] = d / np.linalg.norm(d, axis=1, keepdims=True)

    def aim(self, tcp, z_axis):
        """Put the arm so the TCP is at `tcp` with the camera looking along z_axis (any roll)."""
        self.d.ctrl[self.r.arm_act] = self.r.meta["arm_home"]
        res = self.r.ik(tcp, z_axis=z_axis, n_random=6)
        if res is None or not res.success:
            return False
        self.d.qpos[self.r.arm_qadr] = res.q
        self.d.ctrl[self.r.arm_act] = res.q
        q7 = self.r.arm_qadr[6]                  # random roll: joint 7 spins about the view axis
        lo, hi = self.r.kin.q_min[6], self.r.kin.q_max[6]
        self.d.qpos[q7] = float(np.clip(res.q[6] + self.rng.uniform(-1.2, 1.2), lo, hi))
        mujoco.mj_forward(self.m, self.d)
        return True

    def table_scene(self):
        rng = self.rng
        t = META["tables"][rng.choice(list(META["tables"]))]
        (cx, cy), (hx, hy), top = t["pos"], t["half_size"], t["top_z"]
        books = list(rng.choice(BOOKS, size=rng.integers(2, 7), replace=False))
        placed = []
        for b in books:
            for _ in range(30):                  # rejection sampling, no overlaps
                p = np.array([cx + rng.uniform(-hx + 0.12, hx - 0.12), cy + rng.uniform(-hy + 0.05, hy - 0.08)])
                if all(np.linalg.norm(p - q) > 0.24 for q in placed):
                    break
            else:
                continue
            placed.append(p)
            z = top + BD["rib_height"] + BD["thickness"] / 2 + 0.001
            self.set_book(b, [p[0], p[1], z], yaw_quat(rng.uniform(-math.pi, math.pi)))
        if not placed:
            return False
        yaw = t["parking"][0][2]
        fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        edge = cy - hy
        look = placed[rng.integers(len(placed))] + rng.normal(0, 0.06, 2)
        tcp = np.array([look[0], look[1] - rng.uniform(0.0, 0.15), top + rng.uniform(0.30, 0.50)])
        # base placed so the TCP is 0.35-0.65 m ahead of the arm mount (inside the reach), the
        # chassis front at least 5 cm from the table edge
        mount_ahead = self.r.meta["mount_x"]
        base_y = min(tcp[1] - rng.uniform(0.35, 0.65) - mount_ahead, edge - 0.05 - self.r.meta["chassis_half"][0])
        self.set_base(tcp[0] + rng.uniform(-0.15, 0.15), base_y, yaw)
        tilt, az = math.radians(rng.uniform(0, 25)), rng.uniform(-math.pi, math.pi)
        z_axis = np.array([math.sin(tilt) * math.cos(az), math.sin(tilt) * math.sin(az), -math.cos(tilt)])
        z_axis[:2] += 0.25 * fwd[:2] * rng.uniform(0, 1)      # a bit forward, away from the robot
        return self.aim(tcp, z_axis / np.linalg.norm(z_axis))

    def shelf_scene(self):
        rng = self.rng
        shelf = rng.choice(list(META["shelves"]))
        s = META["shelves"][shelf]
        comps = list(rng.choice(list(s["compartments"]), size=rng.integers(1, 3), replace=False))
        pool = list(rng.permutation(BOOKS))
        for comp in comps:
            slots = s["compartments"][comp]["positions"]
            n = rng.integers(1, len(slots) + 1)
            start = rng.integers(0, len(slots) - n + 1)
            for k in range(start, start + n):
                if pool:
                    q = QUAT_STAND.copy()
                    if rng.random() < 0.2:           # head down: turned about the shelf depth axis,
                        mujoco.mju_mulQuat(q, np.array([0.0, 0, 1, 0]), QUAT_STAND)  # spine still out
                    self.set_book(pool.pop(), slots[k], q)
        comp = comps[0]
        col = comp.split("-")[1][1:]
        x, y, yaw = s["parking"][col]
        self.set_base(x + rng.uniform(-0.15, 0.15), y - rng.uniform(0.0, 0.15), yaw)
        c = np.array(s["compartments"][comp]["positions"][2])
        fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        tcp = c - fwd * (BD["width"] / 2 + rng.uniform(0.10, 0.30)) + np.array(
            [rng.uniform(-0.12, 0.12), 0, rng.uniform(-0.05, 0.12)])
        tilt, az = math.radians(rng.uniform(0, 15)), rng.uniform(-math.pi, math.pi)
        side, up = np.array([fwd[1], -fwd[0], 0.0]), np.array([0, 0, 1.0])
        z_axis = fwd * math.cos(tilt) + math.sin(tilt) * (math.cos(az) * side + math.sin(az) * up)
        return self.aim(tcp, z_axis)

    # ------------------------------------------------------------------ render + labels
    def render(self):
        self.ren.update_scene(self.d, camera=self.cam)
        rgb = self.ren.render().copy()
        self.ren.enable_segmentation_rendering()
        try:
            self.ren.update_scene(self.d, camera=self.cam)
            seg = self.ren.render()[..., 0].copy()
        finally:
            self.ren.disable_segmentation_rendering()
        return rgb, self.geom_class[seg]          # -1 background / not a book

    @staticmethod
    def polygons(cls_img):
        """YOLO-seg label lines: class x1 y1 x2 y2 ... (normalised), one per visible book part."""
        import cv2
        h, w = cls_img.shape
        lines = []
        for c in np.unique(cls_img):
            if c < 0:
                continue
            mask = (cls_img == c).astype(np.uint8)
            if mask.sum() < MIN_PIXELS:
                continue
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) < MIN_PIXELS / 2:
                    continue
                cnt = cv2.approxPolyDP(cnt, 1.0, True).reshape(-1, 2)
                if len(cnt) < 3:
                    continue
                xy = (cnt / np.array([w, h])).clip(0, 1).ravel()
                lines.append(f"{c} " + " ".join(f"{v:.5f}" for v in xy))
        return lines

    def sample(self):
        """One (rgb, label lines, kind) or None if the view could not be set up."""
        self.hide_all()
        self.randomize_light()
        kind = "table" if self.rng.random() < 0.6 else "shelf"
        ok = self.table_scene() if kind == "table" else self.shelf_scene()
        if not ok:
            return None
        mujoco.mj_forward(self.m, self.d)
        try:
            rgb, cls_img = self.render()
        except IndexError:        # MuJoCo's segmentation render now and then returns a seg id past
            return None           # the scene's geoms (a blended pixel); skip that view
        lines = self.polygons(cls_img)
        return (rgb, lines, kind, cls_img) if lines else None


def write_yaml():
    names = "\n".join(f"  {k}: {b}" for k, b in enumerate(BOOKS))
    (OUT / "books.yaml").write_text(f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\n"
                                    f"names:\n{names}\n")


def main():
    from PIL import Image
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 4000
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0
    preview = "--preview" in sys.argv
    gen = Generator(seed)
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
    if preview:
        (OUT / "preview").mkdir(parents=True, exist_ok=True)
    write_yaml()
    k, tries, counts = 0, 0, np.zeros(len(BOOKS), int)
    kinds = {"table": 0, "shelf": 0}
    while k < n and tries < 5 * n:
        tries += 1
        s = gen.sample()
        if s is None:
            continue
        rgb, lines, kind, cls_img = s
        split = "val" if k % 7 == 0 else "train"
        name = f"{seed:02d}_{k:05d}"
        Image.fromarray(rgb).save(OUT / "images" / split / f"{name}.png")
        (OUT / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + "\n")
        for line in lines:
            counts[int(line.split()[0])] += 1
        kinds[kind] += 1
        if preview:
            over = rgb.copy()
            colors = (np.random.default_rng(1).random((len(BOOKS), 3)) * 255).astype(np.uint8)
            m = cls_img >= 0
            over[m] = (0.5 * over[m] + 0.5 * colors[cls_img[m]]).astype(np.uint8)
            Image.fromarray(np.hstack([rgb, over])).save(OUT / "preview" / f"{name}.png")
        k += 1
        if k % 200 == 0:
            print(f"  {k}/{n} images ({tries} views tried)", flush=True)
    print(f"wrote {k} images ({kinds['table']} table, {kinds['shelf']} shelf views) to {OUT}")
    print(f"labelled instances per book: min {counts.min()}, max {counts.max()}, total {counts.sum()}")


if __name__ == "__main__":
    main()
