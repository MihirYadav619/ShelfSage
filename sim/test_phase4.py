"""Phase 4 done-when test: FK, Jacobian, IK and straight-line moves on the real scene.

    1. FK (our DH code) vs MuJoCo for 100 random arm configurations
    2. Jacobian vs mj_jacSite for 50 configurations
    3. README done-when: 10 random reachable TCP poses, reached in simulation with < 1 cm error
    4. Phase 5 poses at the fixed parking poses: table edge grasp (pre-grasp / grasp / lift) and
       standing-book grasps in shelves S1, S2, S3 (rows A and B): IK found and collision-free
    5. Straight-line moves: TCP stays within 5 mm of the line
    6. IK speed and joint limits

For sections 4-5 the base is placed directly at the parking pose (navigation is Phase 6).
Usage (project root, venv active):
    python sim/test_phase4.py
"""
import json
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sim.robot import Robot  # noqa: E402

META = json.loads((ROOT / "scene/library_meta.json").read_text())
BD = META["book_dims"]
FWD = np.array([0.0, 1.0, 0.0])        # every parking pose faces +y
UP = np.array([0.0, 0.0, 1.0])
SIDE = np.array([1.0, 0.0, 0.0])
GRIP_IN = BD["grip_in"]                 # grasp point: this far in from the spine (table / shelf)


def place_base(robot, pose):
    """Put the base at a pose (kinematics test only; driving there is Phase 6)."""
    for a, v in zip(robot.qadr, pose):
        robot.d.qpos[a] = v
    robot.d.qvel[:] = 0
    robot.d.ctrl[robot.base_act] = 0
    mujoco.mj_forward(robot.m, robot.d)
    robot.odom[:] = pose


def book_grasp_point(robot, book, where):
    """Grasp point of a book: middle of its thickness, GRIP_IN[where] in from the spine."""
    b = robot.d.body(f"book_{book}")
    R = b.xmat.reshape(3, 3)
    spine_dir = -R[:, 1]                                    # book frame: spine at -y
    return b.xpos + spine_dir * (BD["width"] / 2 - GRIP_IN[where])


def main():
    model = mujoco.MjModel.from_xml_path(str(ROOT / "scene/library.xml"))
    data = mujoco.MjData(model)
    robot = Robot(model, data, pose_source="true")
    kin = robot.kin
    rng = np.random.default_rng(7)
    checks = []
    dadr = [model.joint(j).dofadr[0] for j in robot.meta["arm_joints"]]

    # ---------------------------------------------------------------- 1-2. FK and Jacobian
    pe, ae, je = [], [], []
    for i in range(100):
        q = rng.uniform(kin.q_min, kin.q_max)
        data.qpos[robot.arm_qadr] = q
        mujoco.mj_forward(model, data)
        T = robot.base_transform() @ kin.fk(q)
        pe.append(np.linalg.norm(T[:3, 3] - data.site_xpos[robot.tcp]))
        Rm = data.site_xmat[robot.tcp].reshape(3, 3)
        ae.append(math.degrees(math.acos(np.clip((np.trace(T[:3, :3].T @ Rm) - 1) / 2, -1, 1))))
        if i < 50:
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jp, jr, robot.tcp)
            Rb = robot.base_transform()[:3, :3]
            J, _ = kin.jacobian(q)
            je.append(np.abs(np.vstack([Rb @ J[:3], Rb @ J[3:]]) - np.vstack([jp[:, dadr], jr[:, dadr]])).max())
    print(f"[1] FK vs MuJoCo (100 configs): max {max(pe) * 1000:.5f} mm, {max(ae):.5f} deg")
    print(f"[2] Jacobian vs mj_jacSite (50 configs): max abs diff {max(je):.1e}")
    checks += [("FK matches MuJoCo (< 0.01 mm, < 0.01 deg)", max(pe) < 1e-5 and max(ae) < 0.01),
               ("Jacobian matches mj_jacSite (< 1e-5; the saved XML rounds the base quaternion)", max(je) < 1e-5)]
    robot.reset()

    # ---------------------------------------------------------------- 3. random reachable poses
    margin = 0.1 * (kin.q_max - kin.q_min)
    errs, ik_ms, contacts, n = [], [], set(), 0
    while n < 10:
        q = rng.uniform(kin.q_min + margin, kin.q_max - margin)
        T = robot.fk_world(q)
        if T[2, 3] < 0.3 or robot.arm_collisions(q):        # skip poses in the floor / robot
            continue
        n += 1
        t = time.perf_counter()
        r = robot.ik(T[:3, 3], R=T[:3, :3])
        ik_ms.append((time.perf_counter() - t) * 1000)
        if r is None or not r.success:
            errs.append(np.inf)
            continue
        robot.move_arm(r.q, duration=2.0, settle=0.6)
        contacts |= robot.robot_contacts()
        errs.append(float(np.linalg.norm(data.site_xpos[robot.tcp] - T[:3, 3])))
        robot.home(1.5)
    print(f"[3] 10 random reachable poses: TCP error in sim max {max(errs) * 1000:.2f} mm, "
          f"mean {np.mean(errs) * 1000:.2f} mm; IK {np.mean(ik_ms):.0f} ms avg; contacts: {sorted(contacts) or 'none'}")
    checks += [("10 random poses reached in sim (< 1 cm)", max(errs) < 0.01),
               ("no contacts while moving to them", not contacts)]

    # ---------------------------------------------------------------- 4. Phase 5 poses
    rows = []

    def task(name, pose, pos, z, y, allow, width):
        place_base(robot, pose)
        robot.stow(1.0)
        robot.set_gripper_width(width, 0.3)
        r = robot.ik(pos, z_axis=z, y_axis=y, allow=allow)
        ok = r is not None and r.success and not r.collision
        rows.append((name, ok, r.pos_err * 1000 if r else np.inf, r.collision if r else ["no IK"]))
        return ok, r

    # table B, middle edge spot: Deep Learning
    pose = tuple(META["tables"]["B"]["parking"][1])
    g = book_grasp_point(robot, "dl", "table")
    task("table B pre-grasp (10 cm back)", pose, g - 0.10 * FWD, FWD, UP, (), 0.06)
    task("table B grasp (overhanging spine)", pose, g, FWD, UP, ("book_dl",), 0.044)
    task("table B lift (10 cm up)", pose, g + 0.10 * UP, FWD, UP, ("book_dl",), 0.044)
    # shelves: first book of a row-A and a row-B compartment on each shelf
    for book in ("os", "prob", "physics", "embedded", "cv", "infotheory"):
        loc = META["books"][book]["location"][len("shelf:"):]
        shelf, col = loc.split("-")[0], loc.split("-")[1][1:]
        pose = tuple(META["shelves"][shelf]["parking"][col])
        g = book_grasp_point(robot, book, "shelf")
        task(f"shelf {loc} grasp ({META['books'][book]['title']})", pose, g, FWD, SIDE, (f"book_{book}",), 0.05)
        task(f"shelf {loc} pre-place (15 cm out)", pose, g - 0.15 * FWD, FWD, SIDE, (), 0.05)
    print("[4] Phase 5 poses at the fixed parking poses:")
    for name, ok, err, col in rows:
        print(f"    {'ok ' if ok else 'XX '} {name:45s} IK err {err:.2f} mm  {'' if ok else col}")
    checks.append(("all Phase 5 poses reachable without collisions", all(ok for _, ok, _, _ in rows)))

    # ---------------------------------------------------------------- 5. straight lines
    pose = tuple(META["tables"]["B"]["parking"][1])
    place_base(robot, pose)
    robot.home(1.5)                      # at the table: unfold first (elbow up), as in the demo
    g = book_grasp_point(robot, "dl", "table")
    robot.set_gripper_width(0.06, 0.3)
    robot.move_to(g - 0.10 * FWD, z_axis=FWD, y_axis=UP, duration=2.5)
    lin = [("approach 10 cm into the grasp", robot.move_linear(g, z_axis=FWD, y_axis=UP)),
           ("lift 10 cm", robot.move_linear(g + 0.10 * UP, z_axis=FWD, y_axis=UP)),
           ("retract 12 cm + 10 cm sideways", robot.move_linear(g + 0.10 * UP - 0.12 * FWD + 0.10 * SIDE, z_axis=FWD, y_axis=UP))]
    print("[5] straight-line moves:")
    for name, r in lin:
        print(f"    {name:32s} deviation {r.max_deviation * 1000:.2f} mm, end error {r.final_error * 1000:.2f} mm, "
              f"IK misses {r.ik_failures}")
    tcp_before = data.site_xpos[robot.tcp].copy()
    bad = robot.move_linear(tcp_before - 0.40 * FWD, z_axis=FWD, y_axis=UP)     # into the robot: unreachable
    moved = float(np.linalg.norm(data.site_xpos[robot.tcp] - tcp_before))
    print(f"    unreachable line (40 cm back into the robot): aborted={bad.aborted}, "
          f"stopped after {moved * 100:.1f} cm, deviation {bad.max_deviation * 1000:.2f} mm, contacts {sorted(robot.robot_contacts()) or 'none'}")
    checks += [("straight-line moves stay within 5 mm of the line",
                all(r.success and r.max_deviation < 0.005 for _, r in lin)),
               ("an unreachable line is aborted safely (stays on the line, no contacts)",
                bad.aborted and bad.max_deviation < 0.005 and not robot.robot_contacts())]

    # ---------------------------------------------------------------- 6. limits
    q = robot.arm_q()
    checks.append(("arm within joint limits", bool(np.all(q >= kin.q_min - 1e-3) and np.all(q <= kin.q_max + 1e-3))))
    checks.append(("IK is fast (< 200 ms average incl. collision checks)", np.mean(ik_ms) < 200))

    print("\nPhase 4 checks:")
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("RESULT:", "all passed" if all(ok for _, ok in checks) else "some checks failed")


if __name__ == "__main__":
    main()
