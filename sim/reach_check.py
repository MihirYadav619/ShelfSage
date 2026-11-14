"""Pick the chassis height: can the Panda reach the table and shelf targets from a parking pose?

For each candidate height of the arm mount (top of the chassis) we solve a small numerical IK
(scipy least_squares, many random starts) for the poses the robot will need in Phase 5, and
report which ones are reachable. Distances come from scene/layout.json.

Robot frame: x forward, y left, z up, origin on the floor under the arm mount.

Usage (project root, venv active):
    python sim/reach_check.py
"""
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent.parent
PANDA = ROOT / "assets/mujoco_menagerie/franka_emika_panda/panda.xml"
LAYOUT = json.loads((ROOT / "scene/layout.json").read_text())

MOUNT_TO_FRONT = 0.10   # arm mount sits 10 cm behind the chassis front edge
CLEARANCE = 0.08        # parking gap between chassis front and table / shelf
PARK_ERROR = 0.05       # also test every target 5 cm further away
TCP_OFFSET = 0.1034     # Franka flange -> fingertip centre
POS_TOL, ANG_TOL = 0.005, np.deg2rad(3)

TABLE_TOP = 0.75
BOOK_FLAT_HALF = (0.10, 0.07, 0.015)
BOOK_STAND_HALF_Z = 0.12
TABLE_HALF_Y = LAYOUT["table_half_size"][1]
SHELF = LAYOUT["shelf"]


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


def targets():
    """(name, position in robot frame, wanted tcp z-axis, wanted closing axis or None)."""
    down, fwd, left = np.array([0, 0, -1.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    d_table = MOUNT_TO_FRONT + CLEARANCE + TABLE_HALF_Y        # book centre on the table centre line
    z_book_top = TABLE_TOP + 2 * BOOK_FLAT_HALF[2]
    d_slot = MOUNT_TO_FRONT + CLEARANCE + SHELF["depth"] / 2   # book centre in the middle of the slot
    out = []
    for extra, tag in ((0.0, ""), (PARK_ERROR, " +5cm")):
        out += [
            (f"table grasp{tag}", [d_table + extra, 0, z_book_top + 0.005], down, None),
            (f"table pre-grasp (10 cm up){tag}", [d_table + extra, 0, z_book_top + 0.10], down, None),
        ]
        for row, z_s in zip("AB", SHELF["row_surface_z"]):
            zc = z_s + BOOK_STAND_HALF_Z
            out += [
                (f"shelf {row} place{tag}", [d_slot + extra, 0, zc], fwd, left),
                (f"shelf {row} pre-place (15 cm out){tag}", [d_slot - 0.15 + extra, 0, zc], fwd, left),
            ]
    return out


def solve(model, data, pos, z_axis, y_axis, seeds=30, rng=np.random.default_rng(0)):
    lo, hi = model.jnt_range[:7, 0], model.jnt_range[:7, 1]
    sid = model.site("p_tcp").id

    def resid(q):
        data.qpos[:7] = q
        mujoco.mj_kinematics(model, data)
        R = data.site_xmat[sid].reshape(3, 3)
        r = [data.site_xpos[sid] - pos, 0.3 * (R[:, 2] - z_axis)]
        if y_axis is not None:   # fingers close along the hand y axis; allow either sign
            r.append(0.3 * np.array([1 - abs(R[:, 1] @ y_axis)]))
        return np.concatenate(r)

    best = None
    for _ in range(seeds):
        q0 = rng.uniform(lo, hi)
        sol = least_squares(resid, q0, bounds=(lo, hi), xtol=1e-10, ftol=1e-10)
        resid(sol.x)
        R = data.site_xmat[sid].reshape(3, 3)
        perr = np.linalg.norm(data.site_xpos[sid] - pos)
        aerr = np.arccos(np.clip(R[:, 2] @ z_axis, -1, 1))
        if best is None or perr + aerr * 0.1 < best[0] + best[1] * 0.1:
            best = (perr, aerr)
        if perr < POS_TOL and aerr < ANG_TOL:
            return True, perr, aerr
    return False, *best


def main():
    tg = targets()
    heights = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    table = {}
    for h in heights:
        model = build(h)
        data = mujoco.MjData(model)
        table[h] = [solve(model, data, np.array(p), np.array(z), None if y is None else np.array(y))
                    for _, p, z, y in tg]
        print(f"mount height {h:.2f} m: {sum(ok for ok, *_ in table[h])}/{len(tg)} reachable")

    print("\n" + "target".ljust(36) + "".join(f"{h:>7.2f}" for h in heights))
    for i, (name, *_ ) in enumerate(tg):
        row = "".join(f"{'ok' if table[h][i][0] else 'x':>7}" for h in heights)
        print(name.ljust(36) + row)
    good = [h for h in heights if all(ok for ok, *_ in table[h])]
    print("\nheights where everything (incl. +5 cm margin) is reachable:", good or "none")


if __name__ == "__main__":
    main()
