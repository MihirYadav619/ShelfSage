"""Write the Panda's joints, actuators and tendons into NOTES.md.

Usage (project root, venv active):
    python sim/save_joint_names.py
"""
import mujoco

PATH = "assets/mujoco_menagerie/franka_emika_panda/scene.xml"
model = mujoco.MjModel.from_xml_path(PATH)

JOINT_TYPES = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}


def name(obj, i):
    return mujoco.mj_id2name(model, obj, i) or f"(unnamed {i})"


lines = ["# Panda notes", "", f"Model: `{PATH}`", ""]

lines += ["## Joints", "", "| id | name | type | range (rad or m) | qpos index |", "|---|---|---|---|---|"]
for i in range(model.njnt):
    lo, hi = model.jnt_range[i]
    rng = f"{lo:.3f} to {hi:.3f}" if model.jnt_limited[i] else "unlimited"
    lines.append(f"| {i} | {name(mujoco.mjtObj.mjOBJ_JOINT, i)} | "
                 f"{JOINT_TYPES[int(model.jnt_type[i])]} | {rng} | {model.jnt_qposadr[i]} |")

lines += ["", "## Actuators (what data.ctrl controls)", "",
          "| id | name | ctrl range |", "|---|---|---|"]
for i in range(model.nu):
    lo, hi = model.actuator_ctrlrange[i]
    lines.append(f"| {i} | {name(mujoco.mjtObj.mjOBJ_ACTUATOR, i)} | {lo:.3f} to {hi:.3f} |")

lines += ["", "## Tendons", ""]
if model.ntendon:
    lines += [f"- {name(mujoco.mjtObj.mjOBJ_TENDON, i)}" for i in range(model.ntendon)]
else:
    lines.append("- none")

lines += ["", "## Totals", "",
          f"- joints: {model.njnt}", f"- actuators: {model.nu}", f"- bodies: {model.nbody}", ""]

with open("NOTES.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("Saved NOTES.md\n")
print("\n".join(lines))