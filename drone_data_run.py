import os
os.environ["MUJOCO_GL"] = "egl"

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import mediapy

# ── Flags ──────────────────────────────────────────────────────────────────────
RECORD_VIDEO    = True        # set to False to skip video recording
VIDEO_EPISODES  = [0, 1, 2]  # which episode indices to record
VIDEO_DIR       = "videos"
VIDEO_FPS       = 30
VIDEO_HEIGHT    = 720
VIDEO_WIDTH     = 1280

# ── Constants ──────────────────────────────────────────────────────────────────
XML_PATH       = "basic_quadrotor.xml"
DT_SIM         = 0.002
DT_CTRL        = 0.02
STEPS_PER_CTRL = int(DT_CTRL / DT_SIM)
T_MIN          = 0.0
T_MAX          = 0.15

NUM_EPISODES   = 3
EPISODE_LEN    = 200

SAVE_PATH      = "dataset"
os.makedirs(SAVE_PATH, exist_ok=True)
if RECORD_VIDEO:
    os.makedirs(VIDEO_DIR, exist_ok=True)

# ── Load base model ────────────────────────────────────────────────────────────
base_model = mujoco.MjModel.from_xml_path(XML_PATH)
HOVER      = (base_model.body_mass[1] * abs(base_model.opt.gravity[2])) / 4
print(f"Nominal hover thrust per rotor: {HOVER:.5f} N")

# ── Quaternion → Euler ─────────────────────────────────────────────────────────
def quat_to_euler(q):
    w, x, y, z = q
    r = Rotation.from_quat([x, y, z, w])
    return r.as_euler("xyz")

# ── Error state ────────────────────────────────────────────────────────────────
def get_error_state(data, x_ref=None):
    pos   = data.qpos[:3]
    quat  = data.qpos[3:7]
    euler = quat_to_euler(quat)
    vel   = data.qvel[:3]
    omega = data.qvel[3:6]
    state = np.concatenate([pos, euler, vel, omega])
    if x_ref is None:
        x_ref = np.zeros(12)
    return state - x_ref

# ── Domain randomization ───────────────────────────────────────────────────────
def make_randomized_model(xml_path):
    model = mujoco.MjModel.from_xml_path(xml_path)
    m_nom = 0.027
    model.body_mass[1] = np.random.uniform(0.8 * m_nom, 1.2 * m_nom)
    nom_inertia = np.array([1.4e-5, 1.4e-5, 2.17e-5])
    model.body_inertia[1] = nom_inertia * np.random.uniform(0.85, 1.15, 3)
    for i in range(4):
        model.actuator_gear[i, 2] *= np.random.uniform(0.90, 1.10)
    model.body_ipos[1] = np.random.uniform(-0.01, 0.01, 3)
    model.opt.viscosity *= np.random.uniform(0.80, 1.20)
    return model

# ── PID controller ─────────────────────────────────────────────────────────────
class PIDController:
    def __init__(self):
        self.kp_z  = 8.0;  self.ki_z  = 0.5;  self.kd_z  = 4.0
        self.kp_rp = 3.0;  self.ki_rp = 0.1;  self.kd_rp = 1.5
        self.kp_yaw = 1.0; self.kd_yaw = 0.5
        self.reset()

    def reset(self):
        self.int_z    = 0.0
        self.int_rp   = np.zeros(2)
        self.prev_z   = 0.0
        self.prev_rp  = np.zeros(2)
        self.prev_yaw = 0.0

    def __call__(self, state):
        z, roll, pitch, yaw = state[2], state[3], state[4], state[5]

        self.int_z  += z * DT_CTRL
        dz           = (z - self.prev_z) / DT_CTRL
        self.prev_z  = z
        thrust_corr  = -(self.kp_z * z + self.ki_z * self.int_z + self.kd_z * dz)

        rp           = np.array([roll, pitch])
        self.int_rp += rp * DT_CTRL
        drp          = (rp - self.prev_rp) / DT_CTRL
        self.prev_rp = rp
        rp_corr      = -(self.kp_rp * rp + self.ki_rp * self.int_rp + self.kd_rp * drp)

        dyaw          = (yaw - self.prev_yaw) / DT_CTRL
        self.prev_yaw = yaw
        yaw_corr      = -(self.kp_yaw * yaw + self.kd_yaw * dyaw)

        base = HOVER + thrust_corr
        u = np.array([
            base + rp_corr[1] - rp_corr[0] - yaw_corr,
            base - rp_corr[1] + rp_corr[0] - yaw_corr,
            base - rp_corr[1] - rp_corr[0] + yaw_corr,
            base + rp_corr[1] + rp_corr[0] + yaw_corr,
        ])
        return np.clip(u, T_MIN, T_MAX)

# ── Random initial state ───────────────────────────────────────────────────────
def sample_initial_state(data, model):
    mujoco.mj_resetData(model, data)
    data.qpos[0] = np.random.uniform(-0.3,  0.3)
    data.qpos[1] = np.random.uniform(-0.3,  0.3)
    data.qpos[2] = np.random.uniform( 0.05, 0.35)
    roll  = np.random.uniform(-np.deg2rad(15), np.deg2rad(15))
    pitch = np.random.uniform(-np.deg2rad(15), np.deg2rad(15))
    yaw   = np.random.uniform(-np.deg2rad(15), np.deg2rad(15))
    r     = Rotation.from_euler("xyz", [roll, pitch, yaw])
    quat  = r.as_quat()
    data.qpos[3] = quat[3]
    data.qpos[4] = quat[0]
    data.qpos[5] = quat[1]
    data.qpos[6] = quat[2]
    data.qvel[:3]  = np.random.uniform(-0.2, 0.2, 3)
    data.qvel[3:6] = np.random.uniform(-0.1, 0.1, 3)

# ── Safety check ───────────────────────────────────────────────────────────────
def is_safe(data):
    pos   = data.qpos[:3]
    euler = quat_to_euler(data.qpos[3:7])
    pos_ok = np.all(np.abs(pos[:2]) < 1.0) and 0.0 < pos[2] < 2.0
    att_ok = np.all(np.abs(euler[:2]) < np.deg2rad(45))
    return pos_ok and att_ok

# ── Renderer setup ─────────────────────────────────────────────────────────────
def make_renderer(model):
    return mujoco.Renderer(model, height=VIDEO_HEIGHT, width=VIDEO_WIDTH)

# ── Episode runner ─────────────────────────────────────────────────────────────
def run_episode(controller, episode_idx):
    model = make_randomized_model(XML_PATH)
    data  = mujoco.MjData(model)
    controller.reset()
    sample_initial_state(data, model)

    record_this = RECORD_VIDEO and (episode_idx in VIDEO_EPISODES)
    renderer    = make_renderer(model) if record_this else None
    frames      = [] if record_this else None
    steps_per_frame = int((1.0 / VIDEO_FPS) / DT_SIM)

    transitions  = []
    sim_step     = 0

    for step in range(EPISODE_LEN):
        e_k = get_error_state(data)
        u   = controller(e_k)
        data.ctrl[:] = u

        for substep in range(STEPS_PER_CTRL):
            mujoco.mj_step(model, data)

            if record_this and (sim_step % steps_per_frame == 0):
                renderer.update_scene(data, camera="track_cam")
                frames.append(renderer.render())

            sim_step += 1

        data.qvel[:] += np.random.normal(0, 1e-4, data.qvel.shape)

        # if not is_safe(data):
        #     print(f"  Episode {episode_idx}: safety abort at step {step}")
        #     break

        e_k1 = get_error_state(data)
        transitions.append((e_k, u, e_k1))

    if record_this and len(frames) > 0:
        renderer.close()
        path = f"{VIDEO_DIR}/episode_{episode_idx:04d}.mp4"
        mediapy.write_video(path, frames, fps=VIDEO_FPS)
        print(f"  Video saved → {path}  ({len(frames)} frames)")

    return transitions

# ── Collect dataset ────────────────────────────────────────────────────────────
all_states      = []
all_actions     = []
all_next_states = []

controller = PIDController()
aborted    = 0

print(f"Collecting {NUM_EPISODES} episodes "
      f"({'recording episodes ' + str(VIDEO_EPISODES) if RECORD_VIDEO else 'no video'})...\n")

for ep in range(NUM_EPISODES):
    transitions = run_episode(controller, ep)

    if len(transitions) == 0:
        aborted += 1
        continue

    s, a, s1 = zip(*transitions)
    all_states.extend(s)
    all_actions.extend(a)
    all_next_states.extend(s1)

    if (ep + 1) % 50 == 0:
        print(f"Episode {ep+1:4d}/{NUM_EPISODES} | "
              f"transitions: {len(all_states):6d} | "
              f"aborted: {aborted}")

# ── Save dataset ───────────────────────────────────────────────────────────────
states      = np.array(all_states,      dtype=np.float32)
actions     = np.array(all_actions,     dtype=np.float32)
next_states = np.array(all_next_states, dtype=np.float32)

TARGET = 50_000
if len(states) > TARGET:
    idx         = np.random.choice(len(states), TARGET, replace=False)
    states      = states[idx]
    actions     = actions[idx]
    next_states = next_states[idx]

np.save(f"{SAVE_PATH}/states.npy",      states)
np.save(f"{SAVE_PATH}/actions.npy",     actions)
np.save(f"{SAVE_PATH}/next_states.npy", next_states)

print(f"\nDataset saved to '{SAVE_PATH}/'")
print(f"  states:      {states.shape}")
print(f"  actions:     {actions.shape}")
print(f"  next_states: {next_states.shape}")
print(f"  aborted:     {aborted}/{NUM_EPISODES}")

labels = ["x","y","roll","pitch","yaw","vx","vy","vz","p","q","r"]
print("\nState coverage (min / mean / max):")
for i, label in enumerate(labels):
    print(f"  {label:6s}: [{states[:,i].min():+.3f}, "
          f"{states[:,i].mean():+.3f}, "
          f"{states[:,i].max():+.3f}]")