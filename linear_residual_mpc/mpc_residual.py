"""
controller_with_residual.py
────────────────────────────
Drop-in replacement for your original controller.
The only change to MPC is in the rollout constraint:

    BEFORE:  x[:,k+1] == Ad @ x[:,k] + Bd @ u[:,k] + c
    AFTER:   x[:,k+1] == Ad @ x[:,k] + Bd @ u[:,k] + c + δ_k

where δ_k = NN(x_k, u_k) is computed once per horizon step
(outside CVXPY) and treated as a constant correction vector.

This keeps the MPC problem strictly linear/quadratic (OSQP-friendly)
while still benefiting from the learned nonlinear corrections.
"""

import numpy as np
import cvxpy as cp
import torch
import torch.nn as nn
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════
XML_PATH        = "../basic_quadrotor.xml"
DT_CTRL         = 0.02
SIM_TIME        = 8.0
NN_WEIGHTS_PATH = "../models/residual_nn.pt"
NN_SCALER_PATH  = "../models/residual_scaler.npz"

# ══════════════════════════════════════════════════════════════════════════════
# Load MuJoCo model
# ══════════════════════════════════════════════════════════════════════════════
model = mujoco.MjModel.from_xml_path(XML_PATH)
data  = mujoco.MjData(model)

m  = model.body_mass[1]
g  = abs(model.opt.gravity[2])
Ix, Iy, Iz = model.body_inertia[1]
HOVER = (m * g) / 4

# ══════════════════════════════════════════════════════════════════════════════
# State extraction
# ══════════════════════════════════════════════════════════════════════════════
def quat_to_euler(q):
    w, x, y, z = q
    return Rotation.from_quat([x, y, z, w]).as_euler("xyz")

def get_state(d):
    pos   = d.qpos[:3]
    quat  = d.qpos[3:7]
    euler = quat_to_euler(quat)
    vel   = d.qvel[:3]
    omega = d.qvel[3:6]
    return np.concatenate([pos, euler, vel, omega])

# ══════════════════════════════════════════════════════════════════════════════
# Linearized model
# ══════════════════════════════════════════════════════════════════════════════
nx, nu = 12, 4

A = np.zeros((nx, nx))
B = np.zeros((nx, nu))
A[0,6]=1; A[1,7]=1; A[2,8]=1
A[6,4]=g; A[7,3]=-g
A[3,9]=1; A[4,10]=1; A[5,11]=1
B[8,0]=1/m; B[9,1]=1/Ix; B[10,2]=1/Iy; B[11,3]=1/Iz

Ad = np.eye(nx) + A * DT_CTRL
Bd = B * DT_CTRL
c_gravity = np.zeros(nx); c_gravity[8] = -g * DT_CTRL

# ══════════════════════════════════════════════════════════════════════════════
# Wrench → rotor mapping
# ══════════════════════════════════════════════════════════════════════════════
def wrench_to_rotors(u):
    T, tx, ty, tz = u
    l = 0.028; k = 0.02513
    return np.array([
        T/4 - tx/(4*l) - ty/(4*l) - tz/(4*k),
        T/4 - tx/(4*l) + ty/(4*l) + tz/(4*k),
        T/4 + tx/(4*l) + ty/(4*l) - tz/(4*k),
        T/4 + tx/(4*l) - ty/(4*l) + tz/(4*k),
    ])

# ══════════════════════════════════════════════════════════════════════════════
# Residual NN
# ══════════════════════════════════════════════════════════════════════════════
class ResidualNN(nn.Module):
    def __init__(self, nx=12, nu=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nx + nu, hidden), nn.LayerNorm(hidden), nn.ELU(),
            nn.Linear(hidden, hidden),  nn.LayerNorm(hidden), nn.ELU(),
            nn.Linear(hidden, nx),
        )

    def forward(self, xu):
        return self.net(xu)


def load_residual_nn(weights_path, scaler_path, device="cpu"):
    """
    Returns (nn_model, scaler_mean, scaler_std) ready for inference.
    Falls back to zero residual if files are missing (safe for first run).
    """
    try:
        net = ResidualNN(nx=nx, nu=nu, hidden=64).to(device)
        net.load_state_dict(torch.load(weights_path, map_location=device))
        net.eval()
        sc  = np.load(scaler_path)
        print(f"[ResidualNN] loaded from {weights_path}")
        return net, sc["mean"].astype(np.float32), sc["std"].astype(np.float32)
    except FileNotFoundError:
        print(f"[ResidualNN] weights not found at {weights_path} — "
              "running with zero residual (pure linear model)")
        return None, None, None


_device = torch.device("cpu")   # keep on CPU for low-latency single-sample inference
_nn, _sc_mean, _sc_std = load_residual_nn(NN_WEIGHTS_PATH, NN_SCALER_PATH, _device)


def predict_residual(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Returns δ = NN(x, u) as a numpy (12,) vector.
    If NN is not loaded, returns zeros.
    """
    if _nn is None:
        return np.zeros(nx)

    xu = np.concatenate([x, u]).astype(np.float32)
    xu_n = (xu - _sc_mean) / (_sc_std + 1e-8)       # normalise
    with torch.no_grad():
        delta = _nn(torch.from_numpy(xu_n).unsqueeze(0))
    return delta.squeeze(0).numpy()

# ══════════════════════════════════════════════════════════════════════════════
# MPC with residual corrections
# ══════════════════════════════════════════════════════════════════════════════
class MPC:
    """
    Identical to the original MPC except the rollout constraint becomes:

        x[k+1] = Ad @ x[k] + Bd @ u[k] + c_gravity + delta[k]

    where delta[k] is a CONSTANT (pre-computed outside CVXPY) correction
    obtained from the residual NN evaluated at the current operating point.

    This is sometimes called a "Sequential Linearisation" or
    "RTI-style" (Real-Time Iteration) correction and keeps the QP structure.
    """

    def __init__(self, horizon=50):
        self.N = horizon
        self.Q  = np.diag([40, 40, 100, 5, 5, 50, 2, 2, 5, 1, 1, 10])
        self.Qf = self.Q * 10
        self.R   = 0.01 * np.eye(nu)
        self.Rdu = 0.1  * np.eye(nu)
        self.Ki  = np.array([5.0, 5.0, 20.0, 5.0])

        self.integral_error = np.zeros(4)
        self.integral_clip  = np.array([0.3, 0.3, 1.5, 0.3])
        self.u_prev = np.array([m*g, 0.0, 0.0, 0.0])

        # Cache last solved trajectory for warm-start delta computation
        self._x_traj_prev = None
        self._u_traj_prev = None

    def _compute_delta_sequence(self, x0: np.ndarray) -> np.ndarray:
        """
        Compute NN residual corrections for each step in the horizon.

        Strategy: roll out the CURRENT linearized model from x0 using
        the previous control solution as the warm-start, then query the
        NN at each (x_k, u_k) along that nominal trajectory.

        Returns: (N, 12) array of delta vectors.
        """
        deltas = np.zeros((self.N, nx))

        x_k = x0.copy()
        for k in range(self.N):
            # Use previous solution if available, else hover
            if self._u_traj_prev is not None and k < len(self._u_traj_prev):
                u_k = self._u_traj_prev[k]
            else:
                u_k = np.array([m*g, 0.0, 0.0, 0.0])

            deltas[k] = predict_residual(x_k, u_k)

            # Advance nominal trajectory (linear + delta for next delta eval)
            x_k = Ad @ x_k + Bd @ u_k + c_gravity + deltas[k]

        return deltas

    def solve(self, x0: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
        # ── integral action ────────────────────────────────────────────────────
        err = x_ref[[0,1,2,5]] - x0[[0,1,2,5]]
        self.integral_error += err * DT_CTRL
        self.integral_error  = np.clip(
            self.integral_error, -self.integral_clip, self.integral_clip
        )

        # ── tilt compensation ──────────────────────────────────────────────────
        roll, pitch = x0[3], x0[4]
        cos_tilt    = np.clip(np.cos(roll) * np.cos(pitch), 0.5, 1.0)
        tilt_thrust = m * g / cos_tilt

        u_ff = np.array([
            (tilt_thrust - m*g) + self.Ki[2]*self.integral_error[2],
            -self.Ki[1]*self.integral_error[1],
             self.Ki[0]*self.integral_error[0],
             self.Ki[3]*self.integral_error[3],
        ])
        u_hover = np.array([m*g, 0.0, 0.0, 0.0]) + u_ff

        # ── residual corrections (constant wrt optimisation variables) ─────────
        deltas = self._compute_delta_sequence(x0)   # (N, 12)

        # ── CVXPY problem ──────────────────────────────────────────────────────
        x = cp.Variable((nx, self.N + 1))
        u = cp.Variable((nu, self.N))

        cost        = 0
        constraints = [x[:, 0] == x0]

        for k in range(self.N):
            cost += cp.quad_form(x[:, k] - x_ref, self.Q)
            cost += cp.quad_form(u[:, k] - u_hover, self.R)
            u_prev_k = self.u_prev if k == 0 else u[:, k-1]
            cost += cp.quad_form(u[:, k] - u_prev_k, self.Rdu)

            # Key change: add constant delta_k to the dynamics constraint
            constraints += [
                x[:, k+1] == Ad @ x[:, k] + Bd @ u[:, k]
                             + c_gravity + deltas[k],   # ← residual correction
                u[0, k] >= 0.0,
                u[0, k] <= 2 * m * g,
                cp.abs(u[1, k]) <= 0.005,
                cp.abs(u[2, k]) <= 0.005,
                cp.abs(u[3, k]) <= 0.02,
            ]

        cost += cp.quad_form(x[:, self.N] - x_ref, self.Qf)

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)

        if u.value is None:
            print("MPC infeasible — using previous solution")
            return self.u_prev

        self.u_prev       = u.value[:, 0]
        self._u_traj_prev = u.value.T      # (N, 4) for next warm-start
        self._x_traj_prev = x.value.T      # (N+1, 12)
        return self.u_prev


# ══════════════════════════════════════════════════════════════════════════════
# Main simulation
# ══════════════════════════════════════════════════════════════════════════════
mpc = MPC()

x_ref = np.zeros(nx)
x_ref[0] = 0.5
x_ref[1] = -0.5
x_ref[2] = 1.0
x_ref[5] = 0.0

test_u = wrench_to_rotors(np.array([m*g, 0, 0, 0]))
print(f"m={m:.4f} kg  g={g:.4f} m/s²  m*g={m*g:.4f} N")
print(f"Hover rotor commands : {test_u}")
print(f"Rotor sum            : {sum(test_u):.4f} N  (should = {m*g:.4f} N)")

states, controls, times = [], [], []

with mujoco.viewer.launch_passive(model, data) as viewer:
    t = 0.0
    while viewer.is_running() and t < SIM_TIME:
        x = get_state(data)

        u_wrench = mpc.solve(x, x_ref)

        if len(times) % 25 == 0:
            print(f"t={t:.1f} | z={x[2]:.3f} | z_err={x_ref[2]-x[2]:.3f} | "
                  f"T_cmd={u_wrench[0]:.4f} | int_z={mpc.integral_error[2]:.4f} | "
                  f"rotors={wrench_to_rotors(u_wrench).round(4)}")

        u_rotors = np.clip(wrench_to_rotors(u_wrench), 0.005, 0.25)
        data.ctrl[:] = u_rotors

        states.append(x.copy())
        controls.append(u_rotors.copy())
        times.append(t)

        for _ in range(int(DT_CTRL / model.opt.timestep)):
            mujoco.mj_step(model, data)

        viewer.sync()
        t += DT_CTRL

states   = np.array(states)
controls = np.array(controls)
times    = np.array(times)

# ── plots ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

ax = axes[0]
ax.plot(times, states[:,0], label="x")
ax.plot(times, states[:,1], label="y")
ax.plot(times, states[:,2], label="z")
ax.axhline(x_ref[0], ls="--", color="r", alpha=0.5)
ax.axhline(x_ref[1], ls="--", color="g", alpha=0.5)
ax.axhline(x_ref[2], ls="--", color="b", alpha=0.5)
ax.set_title("Position"); ax.set_xlabel("Time (s)"); ax.legend(); ax.grid()

ax = axes[1]
ax.plot(times, np.degrees(states[:,3]), label="roll")
ax.plot(times, np.degrees(states[:,4]), label="pitch")
ax.plot(times, np.degrees(states[:,5]), label="yaw")
ax.axhline(np.degrees(x_ref[5]), ls="--", color="k", alpha=0.4)
ax.set_title("Orientation (deg)"); ax.set_xlabel("Time (s)"); ax.legend(); ax.grid()

ax = axes[2]
for i in range(4):
    ax.plot(times, controls[:,i], label=f"rotor {i}", alpha=0.7)
ax.axhline(HOVER, ls="--", color="k", alpha=0.4, label="hover/4")
ax.set_title("Rotor thrusts"); ax.set_xlabel("Time (s)"); ax.legend(); ax.grid()

plt.tight_layout()
plt.show()