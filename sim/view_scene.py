"""Open the generated library scene in the MuJoCo viewer.

Usage (from the project root, venv active):
    python sim/view_scene.py
"""
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("scene/library.xml")
data = mujoco.MjData(model)
mujoco.viewer.launch(model, data)
