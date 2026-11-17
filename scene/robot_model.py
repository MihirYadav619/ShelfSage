"""ShelfSage robot: holonomic base (x, y, yaw) + Franka Panda from MuJoCo Menagerie.

Used by generate_scene.py. The base body is written as MJCF text; the Panda is attached with the
MjSpec attach API (loaded from Menagerie, not copy-pasted), and we add to its hand:
    tcp        site between the fingertips (Franka flange -> fingertip offset 0.1034 m)
    wrist_cam  RGB-D camera looking along the gripper (eye-in-hand)

Base joints use `ref`, so qpos of base_x / base_y / base_yaw is the world pose of the robot
(x, y, yaw). The base is driven with velocity actuators (cmd_vel style, world frame);
sim/robot.py converts robot-frame commands.

Reach check (sim/reach_check.py): with the gripper pointing straight ahead the Panda cannot reach
0.27-0.38 m above its mount; a 0.65 m mount keeps the table grasp (0.78-0.88 m) and both shelf
rows (0.61, 1.11 m) outside that band.

"""
import math
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parent.parent
PANDA_XML = ROOT / "assets/mujoco_menagerie/franka_emika_panda/panda.xml"
PANDA_ASSETS_REL = "../assets/mujoco_menagerie/franka_emika_panda/assets/"  # relative to scene/
PREFIX = "panda_"

TCP_OFFSET = 0.1034
ARM_HOME = [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853]   # Menagerie "home"
# travel pose: arm folded back over the chassis, nothing sticks out past the chassis front
# (home reaches 0.84 m ahead of the base and hits the shelf when parking there)
ARM_STOW = [0.0, -1.7, 0.0, -2.844, 0.0, 3.6, 0.785]
FINGER_OPEN = 0.04
GRIPPER_OPEN_CTRL = 255.0
FINGER_FRICTION, FINGER_PRIORITY = 1.5, 2   # rubber pads; beats the books' 0.4 (generate_scene.py)
# Friction grasp (Phase 5). The fingers hold a book only by the 1.5 cm next to its spine, 6 cm from
# its centre of mass, so the grasp must resist twisting: condim 4 adds torsional friction (0.02 m,
# about the pad's contact radius). Without it the book turned in the fingers while carried.
FINGER_CONDIM, FINGER_TORSION = 4, 0.02
# Stiffer finger contacts (solref time constant 4 ms = 2 timesteps, default 20 ms) plus the scene's
# noslip iterations (generate_scene.py): a held book crept 2.9 cm in 40 s, now 0.02 cm. Trade-off:
# pressing on a book chatters (the contact keeps breaking), so the base-pull drag needs ~3 strokes
# instead of 1. At 10 ms the drag took 1 stroke, but the ~2 mm creep left in the grasp made the
# shelf insertion fail.
FINGER_SOLREF = (0.004, 1.0)
# Menagerie's gripper is a weak position servo (100 N/m on the finger-gap tendon: ~2 N on a 4 cm
# book, less than its weight). 3000 N/m with the fingers commanded shut squeezes a 4 cm book with
# ~60 N in total, inside the real Franka Hand's 70 N continuous grasp force.
GRIPPER_KP, GRIPPER_FORCE = 3000.0, 140.0

CHASSIS_RGBA = (0.18, 0.21, 0.26, 1)
TRIM_RGBA = (0.85, 0.47, 0.12, 1)     # orange trim stripe
DARK = (0.08, 0.08, 0.09, 1)

# velocity servos on the base (mass ~48 kg incl. arm, yaw inertia ~3 kg m^2 -> tau ~0.1 s)
BASE_KV_XY, BASE_F_XY, BASE_V_MAX = 500.0, 400.0, 1.0
BASE_KV_YAW, BASE_F_YAW, BASE_W_MAX = 40.0, 150.0, 1.5


def _v(*xs):
    return " ".join(f"{x:.4f}" for x in xs)


def base_xml(cfg, start):
    """MJCF text for the base body (goes into <worldbody>) and its actuators."""
    sx, sy, syaw = start
    hx, hy = cfg["chassis_half"]
    clr, top = cfg["ground_clearance"], cfg["mount_height"]
    hz = (top - clr) / 2
    li = cfg["lidar"]
    c, s = math.cos(syaw), math.sin(syaw)
    # slide axes expressed in the (rotated) body frame so they stay world x / world y
    ax_x, ax_y = (c, -s, 0), (s, c, 0)
    wr = clr + 0.03  # wheel radius: wheels touch the floor (visual only)
    nc = 'contype="0" conaffinity="0"'

    wheels = "\n".join(
        f'  <geom name="wheel_{n}" type="cylinder" size="{wr:.3f} 0.025" '
        f'pos="{_v(px * (hx - 0.12), py * (hy + 0.025), wr)}" quat="0.7071068 0.7071068 0 0" rgba="{_v(*DARK)}" {nc}/>'
        for n, px, py in (("fl", 1, 1), ("fr", 1, -1), ("rl", -1, 1), ("rr", -1, -1)))

    # the whole scene uses radians (like the Panda): joint ref of base_yaw is in radians
    body = f"""<body name="base" pos="{_v(sx, sy, 0)}" quat="{_v(math.cos(syaw / 2), 0, 0, math.sin(syaw / 2))}">
  <joint name="base_x" type="slide" axis="{_v(*ax_x)}" ref="{sx:.6f}"/>
  <joint name="base_y" type="slide" axis="{_v(*ax_y)}" ref="{sy:.6f}"/>
  <joint name="base_yaw" type="hinge" axis="0 0 1" ref="{syaw:.6f}"/>
  <geom name="chassis" type="box" pos="{_v(0, 0, clr + hz)}" size="{_v(hx, hy, hz)}" mass="{cfg['mass']}" rgba="{_v(*CHASSIS_RGBA)}"/>
  <geom name="chassis_top" type="box" pos="{_v(0, 0, top - 0.004)}" size="{_v(hx + 0.005, hy + 0.005, 0.006)}" rgba="0.75 0.77 0.80 1" {nc} mass="0"/>
  <geom name="chassis_trim" type="box" pos="{_v(0, 0, clr + 0.6 * hz)}" size="{_v(hx + 0.003, hy + 0.003, 0.012)}" rgba="{_v(*TRIM_RGBA)}" {nc} mass="0"/>
{wheels}
  <geom name="lidar_puck" type="cylinder" pos="{_v(li['x'], 0, li['z'])}" size="0.04 0.025" rgba="{_v(*DARK)}" {nc} mass="0"/>
  <site name="lidar" pos="{_v(li['x'], 0, li['z'])}" size="0.01" rgba="1 0 0 1"/>
  <!-- second LiDAR at the back, looking backwards: the base drives in any direction while
       facing +y, and the front LiDAR's 90 deg blind sector is then ahead when it reverses -->
  <geom name="lidar_rear_puck" type="cylinder" pos="{_v(-li['x'], 0, li['z'])}" size="0.04 0.025" rgba="{_v(*DARK)}" {nc} mass="0"/>
  <site name="lidar_rear" pos="{_v(-li['x'], 0, li['z'])}" euler="0 0 3.14159265" size="0.01" rgba="1 0 0 1"/>
  <frame name="arm_mount" pos="{_v(cfg['mount_x'], 0, top)}"/>
</body>"""

    actuators = f"""<velocity name="base_vx" joint="base_x" kv="{BASE_KV_XY}" ctrlrange="{-BASE_V_MAX} {BASE_V_MAX}" forcerange="{-BASE_F_XY} {BASE_F_XY}"/>
<velocity name="base_vy" joint="base_y" kv="{BASE_KV_XY}" ctrlrange="{-BASE_V_MAX} {BASE_V_MAX}" forcerange="{-BASE_F_XY} {BASE_F_XY}"/>
<velocity name="base_wz" joint="base_yaw" kv="{BASE_KV_YAW}" ctrlrange="{-BASE_W_MAX} {BASE_W_MAX}" forcerange="{-BASE_F_YAW} {BASE_F_YAW}"/>"""
    return body, actuators


def panda_spec(cfg):
    """Menagerie Panda with tcp site and wrist camera added to the hand."""
    panda = mujoco.MjSpec.from_file(str(PANDA_XML))
    for b in panda.bodies:              # gravity compensation, like the real Franka controller
        if b.name != "world":
            b.gravcomp = 1.0
    for k in list(panda.keys):          # we write our own full-model keyframe
        panda.delete(k)
    for f in ("left_finger", "right_finger"):
        for g in panda.body(f).geoms:
            if g.contype or g.conaffinity:      # collision geoms only (mesh hull + pads)
                g.friction = [FINGER_FRICTION, FINGER_TORSION, 0.0001]
                g.priority = FINGER_PRIORITY
                g.condim = FINGER_CONDIM
                g.solref = list(FINGER_SOLREF)
    grip = panda.actuator("actuator8")      # ctrl 0..255 -> finger gap tendon 0..0.04 m
    grip.gainprm[0] = GRIPPER_KP * 0.04 / 255
    grip.biasprm[1], grip.biasprm[2] = -GRIPPER_KP, -2 * (GRIPPER_KP * 0.1) ** 0.5   # ~critical damping
    grip.forcerange = [-GRIPPER_FORCE, GRIPPER_FORCE]
    hand = panda.body("hand")
    hand.add_site(name="tcp", pos=[0, 0, TCP_OFFSET], size=[0.006, 0, 0], rgba=[0, 1, 0, 0.6], group=4)
    cam = cfg["wrist_cam"]
    # camera housing on the side of the hand; camera looks along +z of the hand (towards the fingers)
    hand.add_geom(name="wrist_cam_body", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.012, 0.045, 0.012],
                  pos=[0.062, 0, 0.035], rgba=[0.1, 0.1, 0.11, 1], contype=0, conaffinity=0, group=2,
                  mass=0)
    hand.add_camera(name="wrist_cam", pos=[0.062, 0, 0.048], quat=[0, 1, 0, 0], fovy=cam["fovy"],
                    resolution=[cam["width"], cam["height"]])
    return panda


def add_robot(spec, cfg, start):
    """Attach the Panda to the base's arm_mount frame, add the 'home' keyframe, fix mesh paths.

    `spec` must already contain the base body from base_xml(). Returns the compiled model.
    """
    spec.attach(panda_spec(cfg), prefix=PREFIX, frame=spec.frame("arm_mount"))
    model = spec.compile()

    qpos, ctrl = model.qpos0.copy(), [0.0] * model.nu
    for i, q in enumerate(ARM_HOME, start=1):
        qpos[model.joint(f"{PREFIX}joint{i}").qposadr[0]] = q
        ctrl[model.actuator(f"{PREFIX}actuator{i}").id] = q
    for f in ("finger_joint1", "finger_joint2"):
        qpos[model.joint(PREFIX + f).qposadr[0]] = FINGER_OPEN
    ctrl[model.actuator(f"{PREFIX}actuator8").id] = GRIPPER_OPEN_CTRL
    spec.add_key(name="home", qpos=qpos, ctrl=ctrl)
    return model


def save_xml(spec, path):
    """spec.to_xml() drops the Panda's meshdir; the Panda meshes are the only meshes in the
    scene, so put the Menagerie asset folder (relative to scene/) into <compiler meshdir>."""
    xml = spec.to_xml()
    if 'meshdir=' not in xml:
        xml = xml.replace("<compiler ", f'<compiler meshdir="{PANDA_ASSETS_REL}" ', 1)
    path.write_text(xml)


def robot_meta(cfg, start):
    return {
        "start": start,
        "base_joints": ["base_x", "base_y", "base_yaw"],
        "base_actuators": ["base_vx", "base_vy", "base_wz"],
        "arm_joints": [f"{PREFIX}joint{i}" for i in range(1, 8)],
        "arm_actuators": [f"{PREFIX}actuator{i}" for i in range(1, 8)],
        "gripper_actuator": f"{PREFIX}actuator8",
        "arm_home": ARM_HOME,
        "arm_stow": ARM_STOW,
        "tcp_site": f"{PREFIX}tcp",
        "lidar_site": "lidar",
        "lidar_rear_site": "lidar_rear",
        "wrist_camera": f"{PREFIX}wrist_cam",
        **cfg,
    }
