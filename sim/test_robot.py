"""Phase 3 done-when test.

    1. Drive from the start pose to the Table B parking pose (controller on ground truth here,
       to test the mechanics), stop without overshoot or collisions, then move the arm so the
       TCP is 10 cm above the middle book (Deep Learning), gripper pointing down.
    2. LiDAR: draw a scan taken at the parking pose over the static map; the points must lie
       on walls / table legs / shelf (at least 95 % within 4 cm of an occupied cell).
    3. Odometry drift: drive a loop in the free area using only odometry and log how far the
       estimate drifts from ground truth (no pass/fail, this is what MCL will fix in Phase 6).
    4. Wrist RGB-D: save the RGB image and a colourised depth image above the table.
    5. Books are still where they were (< 1 cm) after all of this.

Images go to sim/output/phase3/. Usage (project root, venv active):
    python sim/test_robot.py
"""
import json
import math
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from control.occupancy import build_grid  # noqa: E402
from sim.robot import Robot, rot2  # noqa: E402

OUT = ROOT / "sim/output/phase3"
META = json.loads((ROOT / "scene/library_meta.json").read_text())

def table_parking(name, spot=1):
    """Fixed parking pose in front of an edge spot (0, 1, 2; 1 = middle), from library_meta.json."""
    return tuple(META["tables"][name]["parking"][spot])


def book_positions(m, d):
    return {m.body(i).name: d.xpos[i].copy() for i in range(m.nbody) if m.body(i).name.startswith("book_")}


def draw_map(grid, scale=2):
    img = np.where(grid.occ, 40, 245).astype(np.uint8)
    img = np.stack([img] * 3, axis=-1)[::-1]            # row 0 = top (max y)
    return np.kron(img, np.ones((scale, scale, 1), dtype=np.uint8)), scale


def plot_points(img, grid, scale, pts, color, r=1):
    h = img.shape[0]
    for x, y in pts:
        ix = int((x - grid.origin[0]) / grid.res * scale)
        iy = h - 1 - int((y - grid.origin[1]) / grid.res * scale)
        img[max(iy - r, 0):iy + r + 1, max(ix - r, 0):ix + r + 1] = color


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(ROOT / "scene/library.xml"))
    data = mujoco.MjData(model)
    robot = Robot(model, data, seed=0, pose_source="true")
    books0 = book_positions(model, data)
    checks = []

    # ---------------------------------------------------------------- 1. drive + arm
    goal = table_parking("B")
    print(f"[1] start {np.round(robot.true_pose(), 3)} -> Table B parking {np.round(goal, 3)}")
    res = robot.go_to(*goal)
    print(f"    {res.message} in {res.time:.1f} s, error {res.final_error_xy * 100:.2f} cm / "
          f"{math.degrees(res.final_error_yaw):.2f} deg, overshoot {res.overshoot * 100:.2f} cm, "
          f"collisions: {res.collisions or 'none'}")
    checks += [("base reaches the parking pose (< 2 cm, < 1 deg)",
                res.success and res.final_error_xy < 0.02 and res.final_error_yaw < math.radians(1)),
               ("no overshoot (< 1 cm)", res.overshoot < 0.01),
               ("no collisions while driving", not res.collisions)]

    book = data.body("book_dl").xpos.copy()
    bd = META["book_dims"]
    book_top = META["tables"]["B"]["top_z"] + bd["rib_height"] + bd["thickness"]
    target = np.array([book[0], book[1], book_top + 0.10])
    r = robot.move_to(target, z_axis=np.array([0.0, 0.0, -1.0]), duration=3.0)   # Phase 4 IK
    ik_err = r.pos_err
    tcp, R = robot.tcp_pose()
    t = META["tables"]["B"]
    over_table = (abs(tcp[0] - t["pos"][0]) < t["half_size"][0] and abs(tcp[1] - t["pos"][1]) < t["half_size"][1]
                  and tcp[2] > book_top)
    tilt = math.degrees(math.acos(np.clip(-R[2, 2], -1, 1)))
    print(f"    IK residual {ik_err * 1000:.2f} mm")
    print(f"    arm: TCP {np.round(tcp, 3)}, target {np.round(target, 3)}, error "
          f"{np.linalg.norm(tcp - target) * 100:.2f} cm, gripper tilt from vertical {tilt:.1f} deg")
    arm_contacts = robot.robot_contacts()
    checks += [("TCP over Table B, above the books", over_table),
               ("TCP within 1 cm of the target above the book", np.linalg.norm(tcp - target) < 0.01),
               ("arm touches nothing", not arm_contacts)]

    # ---------------------------------------------------------------- 4. wrist RGB-D (arm is over the table now)
    rgb, depth = robot.get_rgbd()
    imageio.imwrite(OUT / "wrist_rgb.png", rgb)
    np.save(OUT / "wrist_depth.npy", depth)
    lo_d, hi_d = np.percentile(depth, [1, 99])
    dvis = np.clip((depth - lo_d) / max(hi_d - lo_d, 1e-6), 0, 1)     # near = red, far = blue
    dvis = (np.stack([1 - dvis, 0.5 * (1 - abs(2 * dvis - 1)), dvis], axis=-1) * 255).astype(np.uint8)
    imageio.imwrite(OUT / "wrist_depth.png", dvis)
    h, w = depth.shape
    centre = float(np.median(depth[h // 2 - 5:h // 2 + 5, w // 2 - 5:w // 2 + 5]))
    print(f"[4] wrist RGB-D saved ({w}x{h}); depth at image centre {centre * 100:.1f} cm, "
          f"range {lo_d * 100:.1f}-{hi_d * 100:.1f} cm")
    cam_z = robot.d.cam_xpos[model.camera(robot.meta["wrist_camera"]).id][2]
    exp_book = cam_z - book_top                      # cover under the camera
    exp_rib = exp_book - bd["rib_height"]            # top ribs: the nearest thing in view
    print(f"    expected: book cover {exp_book * 100:.1f} cm (image centre), ribs {exp_rib * 100:.1f} cm (nearest)")
    checks.append(("wrist depth: book cover at the image centre and ribs nearest, both +-5 mm",
                   abs(centre - exp_book) < 0.005 and abs(lo_d - exp_rib) < 0.005))

    # ---------------------------------------------------------------- 2. LiDAR vs map
    lz = robot.d.site_xpos[robot.lidar_site][2]
    hx, hy = (s / 2 for s in META["room_size"])
    cx, cy = META["room_center"]
    grid = build_grid(model, data, lz - 0.05, lz + 0.05, res=0.02,
                      bounds=(cx - hx - 0.2, cx + hx + 0.2, cy - hy - 0.2, cy + hy + 0.2),
                      exclude_root=robot.base_body)
    ang, rng_ = robot.lidar_scan()
    x, y, yaw = robot.true_pose()
    hit = np.isfinite(rng_)
    lx, ly = robot.lidar_offset()
    local = np.stack([lx + rng_[hit] * np.cos(ang[hit]), ly + rng_[hit] * np.sin(ang[hit])], axis=1)
    pts = local @ rot2(yaw).T + [x, y]
    dist = grid.distance_field()
    cells = grid.to_cell(pts)
    cells[:, 0] = cells[:, 0].clip(0, grid.occ.shape[1] - 1)
    cells[:, 1] = cells[:, 1].clip(0, grid.occ.shape[0] - 1)
    on_map = dist[cells[:, 1], cells[:, 0]] <= 0.04
    frac = float(on_map.mean())
    print(f"[2] LiDAR: {hit.sum()}/{len(rng_)} returns, {frac * 100:.1f} % within 4 cm of the map")
    img, sc = draw_map(grid)
    plot_points(img, grid, sc, pts, (220, 30, 30))
    plot_points(img, grid, sc, [(x, y)], (30, 90, 220), r=4)
    imageio.imwrite(OUT / "lidar_overlay.png", img)
    checks.append(("LiDAR scan lines up with the map (>= 95 % within 4 cm)", frac >= 0.95))

    # ---------------------------------------------------------------- 5. books
    moved = {n: float(np.linalg.norm(data.body(n).xpos - p)) for n, p in books0.items()}
    worst = max(moved, key=moved.get)
    print(f"[5] books: largest move {worst} {moved[worst] * 100:.2f} cm")
    checks.append(("books stay put (< 1 cm)", moved[worst] < 0.01))

    # ---------------------------------------------------------------- 3. odometry drift
    robot.close()
    robot = Robot(model, data, seed=0, pose_source="odom")
    loop = [(2.5, -3.6, math.pi / 2), (2.5, -3.0, math.pi), (-2.5, -3.0, math.pi),
            (-2.5, -3.6, 0.0), (0.0, -3.6, math.pi / 2)]
    robot.trace, length = [], 0.0
    prev = np.array(robot.true_pose()[:2])
    for wp in loop:
        r = robot.go_to(*wp, timeout=40)
        p = np.array(robot.true_pose()[:2])
        length += float(np.linalg.norm(p - prev))
        prev = p
        if r.collisions:
            print("    collision during odometry loop:", r.collisions)
    tx, ty, tyaw = robot.true_pose()
    ox, oy, oyaw = robot.get_pose()
    drift = math.hypot(tx - ox, ty - oy)
    print(f"[3] odometry loop ({length:.1f} m driven): drift {drift * 100:.1f} cm, "
          f"{math.degrees(abs((tyaw - oyaw + math.pi) % (2 * math.pi) - math.pi)):.1f} deg "
          f"({drift / max(length, 1e-6) * 100:.1f} % of distance)")
    img, sc = draw_map(grid)
    plot_points(img, grid, sc, [t[0][:2] for t in robot.trace], (30, 160, 60), r=1)   # true path
    plot_points(img, grid, sc, [t[1][:2] for t in robot.trace], (220, 30, 30), r=1)   # odometry
    plot_points(img, grid, sc, [robot.true_pose()[:2]], (30, 160, 60), r=4)
    plot_points(img, grid, sc, [robot.get_pose()[:2]], (220, 30, 30), r=4)
    imageio.imwrite(OUT / "odometry_loop.png", img)

    print("\nPhase 3 checks:")
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("RESULT:", "all passed" if all(ok for _, ok in checks) else "some checks failed")
    print(f"images: {OUT}")


if __name__ == "__main__":
    main()
