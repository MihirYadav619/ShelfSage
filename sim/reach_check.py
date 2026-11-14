"""Reach analysis: which arm-mount height works, and how far from a table / the shelf to park.

For the mount height in scene/layout.json (and a few others with --compare) we solve a numerical
IK (scipy least_squares, random restarts) for every pose the robot needs in Phase 5, at a range
of forward distances from the arm mount, and print where each task is reachable.

Tasks (book design: 22 x 15 x 4 cm, 1 cm ribs):
    table grasp / lift   gripper horizontal (approach = forward), fingers close vertically;
                         the lower finger goes into the 1 cm gap under the book
    press far book       closed fingertips press on top of a book, gripper tilted 30 deg, to drag
                         it to the edge (needed when the book is not at the edge)
    shelf A / B          gripper horizontal, fingers close sideways around a standing book;
                         "carry" is 3 cm higher (book lifted off the board while inserting)

Finding (Phase 3): with the gripper pointing straight ahead, the Panda cannot reach 0.27-0.38 m
above its mount at any distance (joint 4 / joint 6 limits). The 0.45 m mount put the table
grasp (0.78-0.88 m) inside that band; 0.65 m keeps every task outside it, with shelf row B
moved to 1.00 m.

Robot frame: x forward, y left, z up, origin on the floor under the arm mount.
Usage (project root, venv active):
    python sim/reach_check.py            # configured height + dead-band map
    python sim/reach_check.py --compare  # also 0.35 / 0.45 / 0.55 m
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent.parent
PANDA = ROOT / "assets/mujoco_menagerie/franka_emika_panda/panda.xml"
LAYOUT = json.loads((ROOT / "scene/layout.json").read_text())

TCP_OFFSET = 0.1034     # Franka flange -> fingertip centre
POS_TOL, ANG_TOL = 0.005, np.deg2rad(3)

TABLE_TOP = 0.75
BOOK_L, BOOK_T, RIB_H = 0.22, 0.04, 0.01
SHELF = LAYOUT["shelf_design"]
DISTANCES = np.round(np.arange(0.20, 0.95, 0.05), 2)


def build(mount_h):
    panda = mujoco.MjSpec.from_file(str(PANDA))
    panda.body("hand").add_site(name="tcp", pos=[0, 0, TCP_OFFSET])
    for k in list(panda.keys):
        panda.delete(k)
    parent = mujoco.MjSpec()
    frame = parent.worldbody.add_frame(pos=[0, 0, mount_h])
    parent.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    parent.attach(panda, prefix="p_", frame=frame)
    return parent.compile()


def tasks(rows):
    """name -> (tcp height, wanted tcp z-axis, wanted closing axis or None)."""
    fwd, left, up = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
    tilt30 = np.array([np.sin(np.radians(30)), 0, -np.cos(np.radians(30))])
    z_mid = TABLE_TOP + RIB_H + BOOK_T / 2
    out = {
        f"table grasp z{z_mid:.2f}": (z_mid, fwd, up),
        f"table lift z{z_mid + 0.10:.2f}": (z_mid + 0.10, fwd, up),
        f"press far book z{TABLE_TOP + RIB_H + BOOK_T:.2f}": (TABLE_TOP + RIB_H + BOOK_T, tilt30, None),
    }
    for r, zs in zip("AB", rows):
        out[f"shelf {r} grasp z{zs + BOOK_L / 2:.2f}"] = (zs + BOOK_L / 2, fwd, left)
        out[f"shelf {r} carry z{zs + BOOK_L / 2 + 0.03:.2f}"] = (zs + BOOK_L / 2 + 0.03, fwd, left)
    return out


def solve(model, data, pos, z_axis, y_axis=None, seeds=25, rng=None):
    rng = rng or np.random.default_rng(0)
    lo, hi = model.jnt_range[:7, 0], model.jnt_range[:7, 1]
    sid = model.site("p_tcp").id

    def resid(q):
        data.qpos[:7] = q
        mujoco.mj_kinematics(model, data)
        R = data.site_xmat[sid].reshape(3, 3)
        r = [data.site_xpos[sid] - pos, 0.3 * (R[:, 2] - z_axis)]
        if y_axis is not None:   # fingers close along the hand y axis; either sign
            r.append(0.3 * np.array([1 - abs(R[:, 1] @ y_axis)]))
        return np.concatenate(r)

    best = (np.inf, np.inf)
    for _ in range(seeds):
        sol = least_squares(resid, rng.uniform(lo, hi), bounds=(lo, hi), xtol=1e-10, ftol=1e-10)
        resid(sol.x)
        R = data.site_xmat[sid].reshape(3, 3)
        perr = float(np.linalg.norm(data.site_xpos[sid] - pos))
        aerr = float(np.arccos(np.clip(R[:, 2] @ z_axis, -1, 1)))
        if perr < POS_TOL and aerr < ANG_TOL:
            return True, perr, aerr
        best = min(best, (perr, aerr))
    return False, *best


def table(mount_h, rows):
    model = build(mount_h)
    data = mujoco.MjData(model)
    print(f"\narm mount {mount_h:.2f} m, shelf rows {list(rows)}: reachable (#) vs forward distance from the mount")
    print("  " + "".ljust(26) + "".join(f"{x:>5.2f}" for x in DISTANCES))
    result = {}
    for name, (z, za, ya) in tasks(rows).items():
        row = [solve(model, data, np.array([x, 0, z]), za, ya)[0] for x in DISTANCES]
        result[name] = row
        print("  " + name.ljust(26) + "".join(f"{'#' if ok else '.':>5}" for ok in row))
    return result


def dead_band(mount_h):
    model = build(0.0)
    data = mujoco.MjData(model)
    fwd = np.array([1.0, 0, 0])
    print("\ngripper pointing straight ahead (any roll): height above the mount vs forward distance")
    xs = (0.30, 0.40, 0.50, 0.60, 0.70)
    print("  dz    world z" + "".join(f"{x:>6.2f}" for x in xs))
    for dz in np.round(np.arange(0.0, 0.71, 0.05), 2):
        row = [solve(model, data, np.array([x, 0, dz]), fwd, seeds=40)[0] for x in xs]
        print(f"  {dz:.2f}  {mount_h + dz:5.2f}  " + "".join(f"{'#' if ok else '.':>6}" for ok in row))


def main():
    h = LAYOUT["robot"]["mount_height"]
    rows = SHELF["row_surface_z"]
    table(h, rows)
    if "--compare" in sys.argv:
        for other in (0.35, 0.45, 0.55):
            table(other, rows)
    dead_band(h)


if __name__ == "__main__":
    main()
