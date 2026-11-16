"""Check the detector on the real scene (not the training data): the robot looks at every table
and at shelf compartments with books from its working poses, and each detection is compared with
ground truth: right book?, centre error, yaw error (lying books, mod 180 deg). Books that are in
view but not detected count as misses.

Usage (project root, venv active):
    python perception/check_detect.py            # report + perception/runs/check_*.png overlays
"""
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from perception.detect import BOOKS, BookDetector, aim_camera  # noqa: E402
from sim.robot import Robot  # noqa: E402
from sim.test_phase4 import place_base  # noqa: E402
from skills.manipulation import BD, META  # noqa: E402

OUT = ROOT / "perception/runs"
UP, FWD = np.array([0, 0, 1.0]), np.array([0, 1.0, 0])


SHELF_LOOK_BACK = 0.15       # base this far behind the shelf parking while looking at a compartment
SHELF_LOOK_DIST = 0.32       # camera this far in front of the spines


def shelf_look(comp):
    """(base pose, camera position, view direction) to look at a shelf compartment."""
    shelf, col = comp.split("-")[0], comp.split("-")[1][1:]
    s = META["shelves"][shelf]
    x, y, yaw = s["parking"][col]
    pos = s["compartments"][comp]["positions"]
    c = np.mean([pos[0], pos[-1]], axis=0)                  # middle of the compartment
    spine_y = c[1] - BD["width"] / 2
    cam = np.array([c[0], spine_y - SHELF_LOOK_DIST, c[2] + 0.06])
    return (x, y - SHELF_LOOK_BACK, yaw), cam, FWD - 0.18 * UP


def views():
    """(name, base pose, TCP target or None, view axis, camera position or None): tables seen
    from 45 cm above from the parking pose, shelf compartments from 32 cm in front."""
    out = []
    for name, t in META["tables"].items():
        x, y, yaw = t["parking"][1]
        out.append((f"Table {name}", (x, y + 0.35, yaw), np.array([t["pos"][0], t["pos"][1] - 0.25, t["top_z"] + 0.45]),
                    np.array([0, 0.25, -1.0]), None))
    for comp in ("S1-A1", "S1-A4", "S1-B1", "S2-A1", "S2-B2", "S3-A1", "S3-B4"):
        base, cam, axis = shelf_look(comp)
        out.append((comp, base, None, axis, cam))
    return out


def in_view(robot, cls_img):
    return {BOOKS[c] for c in np.unique(cls_img) if c >= 0 and (cls_img == c).sum() >= 400}


def main():
    from PIL import Image, ImageDraw
    m = mujoco.MjModel.from_xml_path(str(ROOT / "scene/library.xml"))
    d = mujoco.MjData(m)
    r = Robot(m, d, pose_source="true")
    det = BookDetector()
    body_book = {m.body(f"book_{b}").id: k for k, b in enumerate(BOOKS)}
    geom_cls = np.array([body_book.get(m.geom_bodyid[g], -1) for g in range(m.ngeom)] + [-1])
    ren = mujoco.Renderer(m, 480, 640)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, misses, wrong, n_seen = [], 0, 0, 0
    for name, base, tcp, axis, cam in views():
        place_base(r, base)
        d.ctrl[r.arm_act] = r.meta["arm_home"]
        if cam is None:
            ik = r.ik(tcp, z_axis=axis / np.linalg.norm(axis))
            q = ik.q if ik is not None and ik.success else None
        else:
            q = aim_camera(r, det, cam, axis)
        if q is None:
            print(f"  {name}: no IK for the look pose")
            continue
        d.qpos[r.arm_qadr] = q
        d.ctrl[r.arm_act] = q
        mujoco.mj_forward(m, d)
        rgb, depth = r.get_rgbd()
        ren.enable_segmentation_rendering()
        ren.update_scene(d, camera=r.meta["wrist_camera"])
        seen = in_view(r, geom_cls[ren.render()[..., 0]])
        ren.disable_segmentation_rendering()
        dets = det.detect(r, rgb, depth)
        found = {x.book for x in dets}
        n_seen += len(seen)
        misses += len(seen - found)
        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)
        for x in dets:
            b = d.body(f"book_{x.book}")
            if x.book not in seen and np.linalg.norm(b.xpos - x.centre) > 0.1:
                wrong += 1
                rows.append((name, x.book, x.conf, None, None, "NOT IN VIEW / WRONG BOOK"))
                continue
            err = float(np.linalg.norm(b.xpos - x.centre))
            yaw_err = None
            if x.lying:
                yaw_true = math.atan2(b.xmat[3], b.xmat[0]) % math.pi
                yaw_err = abs((x.yaw - yaw_true + math.pi / 2) % math.pi - math.pi / 2)
            rows.append((name, x.book, x.conf, err, yaw_err, ""))
            K = r.camera_intrinsics()
            T = np.linalg.inv(det.camera_pose(r))
            p = T[:3, :3] @ x.centre + T[:3, 3]
            u, v = K[0, 0] * p[0] / -p[2] + K[0, 2], -K[1, 1] * p[1] / -p[2] + K[1, 2]
            draw.ellipse([u - 5, v - 5, u + 5, v + 5], outline=(255, 0, 0), width=2)
            draw.text((u + 7, v - 6), f"{x.book} {x.conf:.2f}", fill=(255, 255, 0))
        for b in seen - found:
            rows.append((name, b, 0.0, None, None, "MISSED"))
        img.save(OUT / f"check_{name.replace(' ', '_')}.png")
    print(f"{'view':9s} {'book':12s} {'conf':>5s} {'centre err':>11s} {'yaw err':>8s}")
    for name, b, conf, err, yerr, note in rows:
        print(f"{name:9s} {b:12s} {conf:5.2f} {'' if err is None else f'{err * 100:8.1f} cm':>11s} "
              f"{'' if yerr is None else f'{math.degrees(yerr):6.1f} d':>8s}  {note}")
    errs = [e for _, _, _, e, _, _ in rows if e is not None]
    yerrs = [y for _, _, _, _, y, _ in rows if y is not None]
    print(f"\nbooks in view (>= 400 px): {n_seen}, detected: {n_seen - misses}, missed: {misses}, "
          f"false / wrong: {wrong}")
    if errs:
        print(f"centre error: mean {np.mean(errs) * 100:.1f} cm, max {np.max(errs) * 100:.1f} cm")
    if yerrs:
        print(f"yaw error (lying): mean {math.degrees(np.mean(yerrs)):.1f} deg, max {math.degrees(np.max(yerrs)):.1f} deg")


if __name__ == "__main__":
    main()
