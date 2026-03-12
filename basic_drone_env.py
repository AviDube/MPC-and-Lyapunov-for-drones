import os
os.environ["MUJOCO_GL"] = "egl"

import mujoco
import numpy as np
import mediapy

XML_PATH = "basic_quadrotor.xml"

model = mujoco.MjModel.from_xml_path(XML_PATH)
data  = mujoco.MjData(model)

DT_SIM       = model.opt.timestep        # 0.002s from your XML
DT_CTRL      = 0.02
STEPS_PER_CTRL = int(DT_CTRL / DT_SIM)  # 10

FPS           = 30
DURATION_S    = 4.0
TOTAL_STEPS   = int(DURATION_S / DT_SIM)           # 2000
STEPS_PER_FRAME = int((1.0 / FPS) / DT_SIM)        # 16

# Hover thrust: mg/4 per rotor
HOVER = (model.body_mass[1] * abs(model.opt.gravity[2])) / 4
print(f"Hover thrust per rotor: {HOVER:.5f} N")

renderer = mujoco.Renderer(model, height=720, width=1280)
mujoco.mj_resetData(model, data)

# Set initial position so drone starts above the floor
data.qpos[2] = 0.1


frames = []

for step in range(TOTAL_STEPS):
    # Apply hover thrust
    # noise = np.random.normal(0, 0.002, 4)  # small random thrust noise
    data.ctrl[:] = [HOVER] * 4
    mujoco.mj_step(model, data)

    # Capture frame at 30fps
    if step % STEPS_PER_FRAME == 0:
        renderer.update_scene(data, camera="track_cam")
        frames.append(renderer.render())

    # Print z height every 200 steps as sanity check
    if step % 200 == 0:
        print(f"Step {step:4d} | z = {data.qpos[2]:.4f}m")

renderer.close()
mediapy.write_video("drone_flight_stable.mp4", frames, fps=FPS)
print(f"Done — {len(frames)} frames written to drone_flight.mp4")


