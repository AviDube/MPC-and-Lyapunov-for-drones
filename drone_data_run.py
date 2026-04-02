import os
os.environ["MUJOCO_GL"] = "glfw"

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# ── Flags ──────────────────────────────────────────────────────────────────────
RECORD_VIDEO = False   # disabled for fast bulk collection

# ── Constants ──────────────────────────────────────────────────────────────────
XML_PATH       = "basic_quadrotor.xml"
DT_SIM         = 0.002
DT_CTRL        = 0.02
STEPS_PER_CTRL = int(DT_CTRL / DT_SIM)
T_MIN          = 0.01
T_MAX          = 0.15

NUM_EPISODES = 15000  
EPISODE_LEN  = 150  

SAVE_PATH = "dataset"
os.makedirs(SAVE_PATH, exist_ok=True)

# ── Load base model ────────────────────────────────────────────────────────────
base_model = mujoco.MjModel.from_xml_path(XML_PATH)
HOVER      = (base_model.body_mass[1] * abs(base_model.opt.gravity[2])) / 4
print(f"Nominal hover thrust per rotor: {HOVER:.5f} N")

# ── Setpoint randomization ─────────────────────────────────────────────────────
SETPOINT_Z_MIN    = 0.15
SETPOINT_Z_MAX    = 0.80
SETPOINT_XY_RANGE = 0.2

def sample_setpoint():
    return np.array([
        np.random.uniform(-SETPOINT_XY_RANGE, SETPOINT_XY_RANGE),
        np.random.uniform(-SETPOINT_XY_RANGE, SETPOINT_XY_RANGE),
        np.random.uniform(SETPOINT_Z_MIN, SETPOINT_Z_MAX),
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ], dtype=np.float32)

# ── Quaternion to Euler ─────────────────────────────────────────────────────────
def quat_to_euler(q):
    w, x, y, z = q
    r = Rotation.from_quat([x, y, z, w])
    return r.as_euler("xyz")

# ── Error state ────────────────────────────────────────────────────────────────
def get_error_state(data, x_ref):
    pos   = data.qpos[:3]
    quat  = data.qpos[3:7]
    euler = quat_to_euler(quat)
    vel   = data.qvel[:3]
    omega = data.qvel[3:6]
    state = np.concatenate([pos, euler, vel, omega])
    return state - x_ref

# ── Domain randomization ───────────────────────────────────────────────────────
def make_randomized_model(xml_path):
    model = mujoco.MjModel.from_xml_path(xml_path)
    m_nom = 0.027
    model.body_mass[1]    = np.random.uniform(0.8 * m_nom, 1.2 * m_nom)
    nom_inertia           = np.array([1.4e-5, 1.4e-5, 2.17e-5])
    model.body_inertia[1] = nom_inertia * np.random.uniform(0.85, 1.15, 3)
    for i in range(4):
        model.actuator_gear[i, 2] *= np.random.uniform(0.90, 1.10)
    model.body_ipos[1]   = np.random.uniform(-0.01, 0.01, 3)
    model.opt.viscosity *= np.random.uniform(0.80, 1.20)
    return model

# ── PID controller ─────────────────────────────────────────────────────────────
class PIDController:
    def __init__(self):
        self.kp_z   = 0.5;  self.ki_z   = 0.05;  self.kd_z   = 0.2
        self.kp_rp  = 0.05; self.ki_rp  = 0.01;  self.kd_rp  = 0.005
        self.kp_yaw = 0.05; self.kd_yaw = 0.01
        self.kff_vel_xy  = 0.15
        self.kff_pos_att = 0.10
        self.reset()

    def reset(self, hover=None):
        self.hover  = hover if hover is not None else HOVER
        self.int_z  = 0.0
        self.int_rp = np.zeros(2)
        self.int_xy = np.zeros(2)

    def __call__(self, err):
        e_z      = err[2]
        e_rp     = err[3:5].copy()
        e_yaw    = err[5]
        e_xy     = err[0:2]
        vx, vy   = err[6], err[7]
        vz       = err[8]
        omega    = err[9:11]
        yaw_rate = err[11]

        ki_pos = 0.03
        self.int_xy += e_xy * DT_CTRL
        self.int_xy  = np.clip(self.int_xy, -1.0, 1.0)

        pos_scale     = min(1.0, 0.2 / (np.linalg.norm(e_xy) + 1e-6))
        desired_pitch = ( e_xy[1] * self.kff_pos_att * pos_scale
                        + vy      * self.kff_vel_xy
                        + self.int_xy[1] * ki_pos)
        desired_roll  = (-e_xy[0] * self.kff_pos_att * pos_scale
                        - vx      * self.kff_vel_xy
                        - self.int_xy[0] * ki_pos)
        e_rp[0] -= desired_pitch
        e_rp[1] -= desired_roll

        self.int_z  += e_z * DT_CTRL
        thrust_corr  = -(self.kp_z  * e_z  + self.ki_z  * self.int_z  + self.kd_z  * vz)

        self.int_rp += e_rp * DT_CTRL
        rp_corr      = -(self.kp_rp * e_rp + self.ki_rp * self.int_rp + self.kd_rp * omega)

        yaw_corr = -(self.kp_yaw * e_yaw + self.kd_yaw * yaw_rate)

        base = self.hover + thrust_corr
        u = np.array([
            base - rp_corr[0] - rp_corr[1] - yaw_corr,
            base - rp_corr[0] + rp_corr[1] + yaw_corr,
            base + rp_corr[0] + rp_corr[1] - yaw_corr,
            base + rp_corr[0] - rp_corr[1] + yaw_corr,
        ])
        return np.clip(u, T_MIN, T_MAX)

# ── Random initial state  ───────────────
def sample_initial_state(data, model, setpoint):
    mujoco.mj_resetData(model, data)

    data.qpos[0] = setpoint[0] + np.random.uniform(-0.3,  0.3)
    data.qpos[1] = setpoint[1] + np.random.uniform(-0.3,  0.3)
    data.qpos[2] = np.clip(
        setpoint[2] + np.random.uniform(-0.4, 0.4),
        0.05, 1.5
    )

    roll  = np.random.uniform(-np.deg2rad(15), np.deg2rad(15))
    pitch = np.random.uniform(-np.deg2rad(15), np.deg2rad(15))
    yaw   = np.random.uniform(-np.deg2rad(15), np.deg2rad(15))
    r     = Rotation.from_euler("xyz", [roll, pitch, yaw])
    quat  = r.as_quat()
    data.qpos[3] = quat[3]
    data.qpos[4] = quat[0]
    data.qpos[5] = quat[1]
    data.qpos[6] = quat[2]

    data.qvel[:3]  = np.random.uniform(-0.5, 0.5, 3)
    data.qvel[3:6] = np.random.uniform(-0.3, 0.3, 3)

# ── Safety check (setpoint-relative) ──────────────────────────────────────────
XY_SAFE_RADIUS = 1.2
Z_SAFE_MARGIN  = 0.6
ATT_SAFE_DEG   = 45

def is_safe(data, setpoint):
    pos   = data.qpos[:3]
    euler = quat_to_euler(data.qpos[3:7])
    xy_ok  = np.all(np.abs(pos[:2] - setpoint[:2]) < XY_SAFE_RADIUS)
    z_ok   = (setpoint[2] - Z_SAFE_MARGIN) < pos[2] < (setpoint[2] + Z_SAFE_MARGIN)
    z_ok   = z_ok and pos[2] > 0.02
    att_ok = np.all(np.abs(euler[:2]) < np.deg2rad(ATT_SAFE_DEG))
    return xy_ok and z_ok and att_ok

# ── Episode runner ─────────────────────────────────────────────────────────────
def run_episode(controller):
    model    = make_randomized_model(XML_PATH)
    hover    = (model.body_mass[1] * abs(model.opt.gravity[2])) / 4
    data     = mujoco.MjData(model)
    setpoint = sample_setpoint()

    controller.reset(hover)
    sample_initial_state(data, model, setpoint)

    transitions = []

    for _ in range(EPISODE_LEN):
        e_k = get_error_state(data, setpoint)
        u   = controller(e_k)
        data.ctrl[:] = u

        for _ in range(STEPS_PER_CTRL):
            mujoco.mj_step(model, data)

        data.qvel[:] += np.random.normal(0, 1e-4, data.qvel.shape)

        if not is_safe(data, setpoint):
            break

        e_k1 = get_error_state(data, setpoint)
        transitions.append((
            e_k.astype(np.float32),
            u.astype(np.float32),
            e_k1.astype(np.float32),
            setpoint.copy(),
        ))

    return transitions

# ── Collect dataset ────────────────────────────────────────────────────────────
all_states, all_actions, all_next_states, all_setpoints = [], [], [], []
controller = PIDController()
aborted    = 0

pbar = tqdm(range(NUM_EPISODES), desc="Collecting", unit="ep", dynamic_ncols=True)

for ep in pbar:
    transitions = run_episode(controller)

    if len(transitions) == 0:
        aborted += 1
    else:
        s, a, s1, sp = zip(*transitions)
        all_states.extend(s)
        all_actions.extend(a)
        all_next_states.extend(s1)
        all_setpoints.extend(sp)

    pbar.set_postfix(
        n=len(all_states),
        aborted=aborted,
        abort_pct=f"{aborted / (ep + 1):.1%}",
    )

# ── Save ───────────────────────────────────────────────────────────────────────
states      = np.array(all_states,      dtype=np.float32)
actions     = np.array(all_actions,     dtype=np.float32)
next_states = np.array(all_next_states, dtype=np.float32)
setpoints   = np.array(all_setpoints,   dtype=np.float32)

TARGET = 150_000
if len(states) > TARGET:
    idx         = np.random.choice(len(states), TARGET, replace=False)
    states      = states[idx]
    actions     = actions[idx]
    next_states = next_states[idx]
    setpoints   = setpoints[idx]

np.save(f"{SAVE_PATH}/states.npy",      states)
np.save(f"{SAVE_PATH}/actions.npy",     actions)
np.save(f"{SAVE_PATH}/next_states.npy", next_states)
np.save(f"{SAVE_PATH}/setpoints.npy",   setpoints)

print(f"\nDataset saved to '{SAVE_PATH}/'")
print(f"  states:      {states.shape}")
print(f"  actions:     {actions.shape}")
print(f"  next_states: {next_states.shape}")
print(f"  setpoints:   {setpoints.shape}")
print(f"  aborted:     {aborted}/{NUM_EPISODES}  ({aborted/NUM_EPISODES:.1%})")

# ── Coverage report ────────────────────────────────────────────────────────────
# States are error-space: states[:,i] = world_state[i] - setpoint[i]
# So states[:,2] = z_error, states[:,8] = vz  (setpoint vz = 0 so no offset)
labels = ["x_err", "y_err", "z_err", "roll", "pitch", "yaw",
          "vx_err", "vy_err", "vz", "p", "q", "r"]
print("\nError-state coverage (min / mean / max):")
for i, label in enumerate(labels):
    print(f"  {label:8s}: [{states[:, i].min():+.3f}, "
          f"{states[:, i].mean():+.3f}, "
          f"{states[:, i].max():+.3f}]")

print("\nSetpoint Z coverage (min / mean / max):")
print(f"  z_sp : [{setpoints[:, 2].min():+.3f}, "
      f"{setpoints[:, 2].mean():+.3f}, "
      f"{setpoints[:, 2].max():+.3f}]")

# ── Bin-occupancy on (z_error, vz) ────────────────────────────────────────────
N_BINS = 10

print("\nPer-dimension bin occupancy:")
print(f"  {'dim':8s}  {'range':>20s}   occupancy")
print(f"  {'-'*8}  {'-'*20}   {'-'*20}")

for i, label in enumerate(labels):
    col  = states[:, i]
    lo   = np.percentile(col, 2)    # 2nd percentile avoids extreme outliers
    hi   = np.percentile(col, 98)   # skewing the bins
    bins = np.linspace(lo, hi, N_BINS + 1)
    counts, _ = np.histogram(col, bins=bins)
    occ  = np.sum(counts > 0) / N_BINS
    bar  = "█" * int(occ * N_BINS) + "░" * (N_BINS - int(occ * N_BINS))
    print(f"  {label:8s}  [{lo:+.3f}, {hi:+.3f}]   {bar}  {occ:.0%}")

# 2D occupancy on the two most important subspaces
print("\n2D bin occupancy (10×10 grid):")
pairs = [
    (2, 8,  "z_err",  "vz"),
    (0, 6,  "x_err",  "vx_err"),
    (3, 9,  "roll",   "p"),
    (4, 10, "pitch",  "q"),
]
for ix, iy, lx, ly in pairs:
    cx, cy = states[:, ix], states[:, iy]
    xbins  = np.linspace(np.percentile(cx, 2), np.percentile(cx, 98), N_BINS + 1)
    ybins  = np.linspace(np.percentile(cy, 2), np.percentile(cy, 98), N_BINS + 1)
    occupied = set()
    for x, y in zip(cx, cy):
        xi = np.searchsorted(xbins, x) - 1
        yi = np.searchsorted(ybins, y) - 1
        if 0 <= xi < N_BINS and 0 <= yi < N_BINS:
            occupied.add((xi, yi))
    occ = len(occupied) / (N_BINS * N_BINS)
    print(f"  ({lx:8s}, {ly:8s}):  {occ:.0%}  ({len(occupied)}/{N_BINS*N_BINS} cells)")