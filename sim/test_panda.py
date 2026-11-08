import math
import time
import mujoco
import mujoco.viewer

PATH = "assets/mujoco_menagerie/franka_emika_panda/scene.xml"
model = mujoco.MjModel.from_xml_path(PATH)
data = mujoco.MjData(model)

print("Joints:", model.njnt, "| Actuators:", model.nu)
for i in range(model.njnt):
    print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i))

with mujoco.viewer.launch_passive(model, data) as viewer:
    t0 = time.time()
    while viewer.is_running():
        t = time.time() - t0
        if model.nu > 0:
            data.ctrl[0] = 0.5 * math.sin(t)   # moves the first joint
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)