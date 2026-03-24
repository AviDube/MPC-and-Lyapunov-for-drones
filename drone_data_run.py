import os
os.environ["MUJOCO_GL"] = "egl"

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import mediapy

# ── Flags ──────────────────────────────────────────────────────────────────────
RECORD_VIDEO    = True
VIDEO_EPISODES  = [0, 1, 2]
VIDEO_DIR       = "videos"
VIDEO_FPS       = 30
VIDEO_HEIGHT    = 720
VIDEO_WIDTH     = 1280

# ── Constants ──────────────────────────────────────────────────────────────────
XML_PATH       = "basic_quadrotor.xml"
DT_SIM         = 0.002
DT_CTRL        = 0.02
STEPS_PER_CTRL = int(DT_CTRL / DT_SIM)
# FIX: raised T_MIN from 0.0 — prevents integral windup + clipping death spiral
T_MIN          = 0.01
T_MAX          = 0.15

NUM_EPISODES   = 3
# FIX: extended from 200 → 400 so transient behaviour settles before episode ends
EPISODE_LEN    = 400

SAVE_PATH      = "dataset"
os.makedirs(SAVE_PATH, exist_ok=True)
if RECORD_VIDEO:
    os.makedirs(VIDEO_DIR, exist_ok=True)

# ── Load base model ────────────────────────────────────────────────────────────
base_model = mujoco.MjModel.from_xml_path(XML_PATH)
# FIX: kept as a fallback only — per-episode hover is recomputed from randomized mass
HOVER      = (base_model.body_mass[1] * abs(base_model.opt.gravity[2])) / 4
print(f"Nominal hover thrust per rotor: {HOVER:.5f} N")

# ── Hover setpoint ─────────────────────────────────────────────────────────────
# FIX: was np.zeros(12) — drone was trying to reach z=0 (the ground)
# State vector: [x, y, z, roll, pitch, yaw, vx, vy, vz, p, q, r]
HOVER_SETPOINT = np.array([0.0, 0.0, 0.2,
                            0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0], dtype=np.float32)

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
        # FIX: default to hover setpoint, not zeros
        x_ref = HOVER_SETPOINT
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

# ── PID + feedforward controller ──────────────────────────────────────────────
class PIDController:
    def __init__(self):
        # ── Scaled Altitude Gains ──────────────────────────────
        self.kp_z   = 0.5;   self.ki_z  = 0.05;  self.kd_z  = 0.2
        
        # ── Scaled Roll / Pitch Gains ──────────────────────────
        # kd_rp dropped significantly to stop motor "bang-bang" chatter
        # kp_rp dropped slightly to allow smoother tilt corrections
        self.kp_rp  = 0.05;  self.ki_rp = 0.01;  self.kd_rp = 0.005
        
        # ── Scaled Yaw Gains ───────────────────────────────────
        self.kp_yaw = 0.05;  self.kd_yaw = 0.01
        
        # ── Feedforward Gains (Outer Loop) ─────────────────────
        # Reduced so a 3 m/s drift doesn't command a >45 degree tilt
        self.kff_vel_xy  = 0.10
        self.kff_pos_att = 0.05
        
        self.reset()

    def reset(self, hover=None):
        # FIX: accept per-episode hover thrust so gravity comp is exact for the
        #      randomized mass, not anchored to the base model mass
        self.hover  = hover if hover is not None else HOVER
        self.int_z  = 0.0
        self.int_rp = np.zeros(2)
        # FIX: removed prev_z / prev_rp / prev_yaw — D-terms now use velocity
        #      states directly instead of finite-differencing positions

    def __call__(self, err, debug=False):
        # err = state - setpoint
        e_z      = err[2]
        e_rp     = err[3:5].copy()  
        e_yaw    = err[5]
        e_xy     = err[0:2]
        vx, vy   = err[6], err[7]
        vz       = err[8]
        omega    = err[9:11]
        yaw_rate = err[11]

        # ── Optional: Add an integral term for XY position to lock it in place 
        # (Put self.int_xy = np.zeros(2) inside your reset() method if you use this)
        if not hasattr(self, 'int_xy'):
            self.int_xy = np.zeros(2)
        self.int_xy += e_xy * DT_CTRL
        self.int_xy = np.clip(self.int_xy, -1.0, 1.0) # Anti-windup
        ki_pos = 0.01

        # ── 1. Outer Loop: Position to Desired Attitude (Cascade) ──────────
        pos_scale = min(1.0, 0.2 / (np.linalg.norm(e_xy) + 1e-6))
        
        # PITCH (RotX) controls Y-position. 
        # Needs POSITIVE rotation to fix positive Y-error (Pitch Back to move Back)
        desired_x_rot = (e_xy[1] * self.kff_pos_att * pos_scale) + (vy * self.kff_vel_xy) + (self.int_xy[1] * ki_pos)

        # ROLL (RotY) controls X-position. 
        # Needs NEGATIVE rotation to fix positive X-error (Roll Left to move Left)
        desired_y_rot = -(e_xy[0] * self.kff_pos_att * pos_scale) - (vx * self.kff_vel_xy) - (self.int_xy[0] * ki_pos)

        e_rp[0] -= desired_x_rot
        e_rp[1] -= desired_y_rot

        # ── 2. Inner Loop: Feedback (PID) for Attitude ─────────────────────
        self.int_z  += e_z  * DT_CTRL
        thrust_corr  = -(self.kp_z  * e_z  + self.ki_z  * self.int_z  + self.kd_z  * vz)

        self.int_rp += e_rp * DT_CTRL
        rp_corr      = -(self.kp_rp * e_rp + self.ki_rp * self.int_rp + self.kd_rp * omega)

        yaw_corr     = -(self.kp_yaw * e_yaw + self.kd_yaw * yaw_rate)

        # ── 3. Motor Mixing ────────────────────────────────────────────────
        base = self.hover + thrust_corr
        u = np.array([
            base - rp_corr[0] - rp_corr[1] - yaw_corr,
            base - rp_corr[0] + rp_corr[1] + yaw_corr,
            base + rp_corr[0] + rp_corr[1] - yaw_corr,
            base + rp_corr[0] - rp_corr[1] + yaw_corr,
        ])
        
        u_clipped = np.clip(u, T_MIN, T_MAX)

        # ── Debug Logging ──────────────────────────────────────────────────
        if debug:
            print(f"\n--- Controller Debug Log ---")
            print(f"Pos Err (X, Y):    [{e_xy[0]:+.3f}, {e_xy[1]:+.3f}]")
            print(f"Vel (Vx, Vy):      [{vx:+.3f}, {vy:+.3f}]")
            print(f"Desired Attitude:  Roll: {desired_roll:+.3f}, Pitch: {desired_pitch:+.3f}")
            print(f"Att Err (Inner):   Roll: {e_rp[0]:+.3f}, Pitch: {e_rp[1]:+.3f}")
            print(f"PID Torque Out:    Roll: {rp_corr[0]:+.3f}, Pitch: {rp_corr[1]:+.3f}")
            print(f"Motor Cmd (Raw):   {np.round(u, 3)}")
            print(f"Motor Cmd (Clip):  {np.round(u_clipped, 3)}")
            print(f"----------------------------")

        return u_clipped

# ── Random initial state ───────────────────────────────────────────────────────
def sample_initial_state(data, model):
    mujoco.mj_resetData(model, data)
    data.qpos[0] = 0 # np.random.uniform(-0.3,  0.3)
    data.qpos[1] = 0 # np.random.uniform(-0.3,  0.3)
    # FIX: spawn near the hover setpoint altitude (0.2 m), not far from it
    data.qpos[2] = np.random.uniform(0.1, 0.4)
    roll  = 0 # np.random.uniform(-np.deg2rad(15), np.deg2rad(15))
    pitch = 0 # np.random.uniform(-np.deg2rad(15), np.deg2rad(15))
    yaw   = 0 # np.random.uniform(-np.deg2rad(15), np.deg2rad(15))
    r     = Rotation.from_euler("xyz", [roll, pitch, yaw])
    quat  = r.as_quat()
    data.qpos[3] = quat[3]
    data.qpos[4] = quat[0]
    data.qpos[5] = quat[1]
    data.qpos[6] = quat[2]
    data.qvel[:3]  = np.random.uniform(0, 0, 3)
    data.qvel[3:6] = np.random.uniform(0, 0, 3)

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
    hover = (model.body_mass[1] * abs(model.opt.gravity[2])) / 4
    data  = mujoco.MjData(model)
    controller.reset(hover)
    sample_initial_state(data, model)

    record_this     = RECORD_VIDEO and (episode_idx in VIDEO_EPISODES)
    renderer        = make_renderer(model) if record_this else None
    frames          = [] if record_this else None
    steps_per_frame = int((1.0 / VIDEO_FPS) / DT_SIM)

    transitions = []
    sim_step    = 0

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

        if not is_safe(data):
            print(f"  Episode {episode_idx}: safety abort at step {step}")
            break

        e_k1 = get_error_state(data)
        transitions.append((e_k, u, e_k1))

    if record_this and frames and len(frames) > 0:
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

labels = ["x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz", "p", "q", "r"]
print("\nState coverage (min / mean / max):")
for i, label in enumerate(labels):
    print(f"  {label:6s}: [{states[:,i].min():+.3f}, "
          f"{states[:,i].mean():+.3f}, "
          f"{states[:,i].max():+.3f}]")