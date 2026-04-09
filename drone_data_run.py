import os
os.environ["MUJOCO_GL"] = "glfw"

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# ==============================
# Config
# ==============================
XML_PATH       = "basic_quadrotor.xml"
DT_SIM         = 0.002
DT_CTRL        = 0.02
STEPS_PER_CTRL = int(DT_CTRL / DT_SIM)

T_MIN = 0.01
T_MAX = 0.15

NUM_EPISODES = 5000   # reduced for faster testing
EPISODE_LEN  = 150

SAVE_PATH = "dataset"
os.makedirs(SAVE_PATH, exist_ok=True)

# ==============================
# Load base model
# ==============================
base_model = mujoco.MjModel.from_xml_path(XML_PATH)
HOVER = (base_model.body_mass[1] * abs(base_model.opt.gravity[2])) / 4
print(f"Nominal hover thrust per rotor: {HOVER:.5f} N")

# ==============================
# State extraction
# ==============================
def get_state(data):
    pos   = data.qpos[:3]
    quat  = data.qpos[3:7]   # (w, x, y, z)
    vel   = data.qvel[:3]
    omega = data.qvel[3:6]
    return np.concatenate([pos, quat, vel, omega])

# ==============================
# Rotor -> Wrench
# ==============================
def rotors_to_wrench(u):
    l = 0.028
    k = 0.02513
    u0, u1, u2, u3 = u
    T  = u0 + u1 + u2 + u3
    tx = l * (-u0 - u1 + u2 + u3)
    ty = l * (-u0 + u1 + u2 - u3)
    tz = k * (-u0 + u1 - u2 + u3)
    return np.array([T, tx, ty, tz])

# ==============================
# Setpoint sampling
# ==============================
SETPOINT_Z_MIN    = 0.05
SETPOINT_Z_MAX    = 5
SETPOINT_XY_RANGE = 5.0

def sample_setpoint():
    return np.array([
        np.random.uniform(-SETPOINT_XY_RANGE, SETPOINT_XY_RANGE),
        np.random.uniform(-SETPOINT_XY_RANGE, SETPOINT_XY_RANGE),
        np.random.uniform(SETPOINT_Z_MIN, SETPOINT_Z_MAX),
    ])

# ==============================
# Domain randomization
# ==============================
def make_randomized_model(xml_path):
    model = mujoco.MjModel.from_xml_path(xml_path)
    m_nom = 0.027
    model.body_mass[1] = np.random.uniform(0.8*m_nom, 1.2*m_nom)
    nom_inertia = np.array([1.4e-5, 1.4e-5, 2.17e-5])
    model.body_inertia[1] = nom_inertia * np.random.uniform(0.85, 1.15, 3)
    for i in range(4):
        model.actuator_gear[i, 2] *= np.random.uniform(0.9, 1.1)
    model.opt.viscosity *= np.random.uniform(0.8, 1.2)
    return model

# ==============================
# Initial state
# ==============================
def sample_initial_state(data, model, setpoint):
    mujoco.mj_resetData(model, data)
    data.qpos[0] = setpoint[0] + np.random.uniform(-0.5, 0.5)
    data.qpos[1] = setpoint[1] + np.random.uniform(-0.5, 0.5)
    data.qpos[2] = np.clip(setpoint[2] + np.random.uniform(-0.8, 0.8), 0.05, 2.0)
    # random orientation
    r = Rotation.from_euler("xyz", np.random.uniform(-0.5,0.5,3))
    q = r.as_quat()
    data.qpos[3:7] = [q[3], q[0], q[1], q[2]]
    data.qvel[:] = np.random.uniform(-1.0, 1.0, 6)

# ==============================
# Smoothed random rotor input
# ==============================
class RandomRotorInput:
    def __init__(self, t_min=T_MIN, t_max=T_MAX, alpha=0.9):
        self.t_min = t_min
        self.t_max = t_max
        self.alpha = alpha
        self.prev_u = np.full(4, (t_min + t_max)/2)

    def __call__(self):
        noise = np.random.uniform(-0.02, 0.02, 4)  # small smooth changes
        u = self.alpha * self.prev_u + (1 - self.alpha) * noise
        u = np.clip(u, self.t_min, self.t_max)
        self.prev_u = u
        return u

# ==============================
# Episode rollout
# ==============================
def run_episode(rotor_input):
    model = make_randomized_model(XML_PATH)
    data = mujoco.MjData(model)
    setpoint = sample_setpoint()
    sample_initial_state(data, model, setpoint)

    transitions = []

    for _ in range(EPISODE_LEN):
        x_k = get_state(data)
        u_rotor = rotor_input()
        u_wrench = rotors_to_wrench(u_rotor)
        data.ctrl[:] = u_rotor

        for _ in range(STEPS_PER_CTRL):
            mujoco.mj_step(model, data)

        # small disturbance
        data.qvel[:] += np.random.normal(0, 1e-3, data.qvel.shape)

        x_k1 = get_state(data)
        xdot = (x_k1 - x_k) / DT_CTRL

        transitions.append((
            x_k.astype(np.float32),
            u_wrench.astype(np.float32),
            x_k1.astype(np.float32),
            xdot.astype(np.float32),
        ))

    return transitions

# ==============================
# Collect dataset
# ==============================
all_states, all_actions, all_next_states, all_xdot = [], [], [], []
rotor_input = RandomRotorInput()

pbar = tqdm(range(NUM_EPISODES), desc="Collecting")
for _ in pbar:
    transitions = run_episode(rotor_input)
    s, a, s1, xd = zip(*transitions)
    all_states.extend(s)
    all_actions.extend(a)
    all_next_states.extend(s1)
    all_xdot.extend(xd)

# ==============================
# Convert to arrays
# ==============================
states      = np.array(all_states, dtype=np.float32)
actions     = np.array(all_actions, dtype=np.float32)
next_states = np.array(all_next_states, dtype=np.float32)
xdot        = np.array(all_xdot, dtype=np.float32)

# ==============================
# Subsample (optional)
# ==============================
TARGET = 300_000
if len(states) > TARGET:
    idx = np.random.choice(len(states), TARGET, replace=False)
    states      = states[idx]
    actions     = actions[idx]
    next_states = next_states[idx]
    xdot        = xdot[idx]

# ==============================
# Save dataset
# ==============================
np.save(f"{SAVE_PATH}/states.npy", states)
np.save(f"{SAVE_PATH}/actions.npy", actions)
np.save(f"{SAVE_PATH}/next_states.npy", next_states)
np.save(f"{SAVE_PATH}/xdot.npy", xdot)

print("\nDataset saved:")
print(" states:", states.shape)
print(" actions:", actions.shape)
print(" next_states:", next_states.shape)
print(" xdot:", xdot.shape)

# ==============================
# Dataset quality metrics
# ==============================
def evaluate_dataset(states, actions, xdot):
    print("\n==== Dataset Quality Metrics ====")

    # 1. Position coverage
    print("\nPosition ranges:")
    print(f"x: {states[:,0].min():.3f} → {states[:,0].max():.3f}")
    print(f"y: {states[:,1].min():.3f} → {states[:,1].max():.3f}")
    print(f"z: {states[:,2].min():.3f} → {states[:,2].max():.3f}")

    # 2. Velocity coverage
    print("\nVelocity ranges:")
    print(f"vx: {states[:,7].min():.3f} → {states[:,7].max():.3f}")
    print(f"vy: {states[:,8].min():.3f} → {states[:,8].max():.3f}")
    print(f"vz: {states[:,9].min():.3f} → {states[:,9].max():.3f}")

    # 3. Rotor input distribution
    print("\nRotor input ranges:")
    for i in range(4):
        print(f"u{i}: {actions[:,i].min():.3f} → {actions[:,i].max():.3f}")

    # 4. Dynamics excitation
    print("\nxdot stats (mean ± std):")
    mean_xdot = np.mean(xdot, axis=0)
    std_xdot  = np.std(xdot, axis=0)
    for i in range(len(mean_xdot)):
        print(f"xdot[{i}]: {mean_xdot[i]:.4f} ± {std_xdot[i]:.4f}")

    # 5. NaNs or extreme values
    print("\nSanity checks:")
    print(f"NaNs in states: {np.isnan(states).sum()}")
    print(f"NaNs in actions: {np.isnan(actions).sum()}")
    print(f"NaNs in xdot: {np.isnan(xdot).sum()}")
    print(f"Extreme z (>5m or <0m): {np.sum((states[:,2]>5) | (states[:,2]<0))}")

# Call the evaluation function
evaluate_dataset(states, actions, xdot)