import os
import numpy as np
import torch
import casadi as ca
import time

# ── Paths ──────────────────────────────────────────────────────────────────────
DYNAMICS_PATH = "dynamics_model"

# ── MPC hyperparameters ────────────────────────────────────────────────────────
N        = 10          # prediction horizon (steps)
DT_CTRL  = 0.02        # control timestep (s) — must match data collection
T_MIN    = 0.01        # motor thrust lower bound (N)
T_MAX    = 0.15        # motor thrust upper bound (N)

# ── Cost weights (Q diagonal, R diagonal) ─────────────────────────────────────
# State error: [x, y, z, roll, pitch, yaw, vx, vy, vz, p, q, r]
# Tune these — z and vz are most important for hover, roll/pitch next.
Q_DIAG = np.array([
    1.0,  1.0,  10.0,   # position error (z weighted highest)
    5.0,  5.0,  1.0,    # attitude error (roll/pitch weighted)
    0.5,  0.5,  2.0,    # velocity error (vz weighted)
    0.1,  0.1,  0.1,    # angular rate error
], dtype=np.float64)

R_DIAG = np.array([0.01, 0.01, 0.01, 0.01], dtype=np.float64)  # action cost

# Terminal cost 
QN_DIAG = 5.0 * Q_DIAG

# ── Load normalisers ───────────────────────────────────────────────────────────
def load_norm(path):
    d = torch.load(path, map_location="cpu")
    return d["mean"].numpy().astype(np.float64), d["std"].numpy().astype(np.float64)

state_mean,  state_std  = load_norm(f"{DYNAMICS_PATH}/state_norm.pt")
action_mean, action_std = load_norm(f"{DYNAMICS_PATH}/action_norm.pt")
delta_mean,  delta_std  = load_norm(f"{DYNAMICS_PATH}/delta_norm.pt")

# ── Load model weights ─────────────────────────────────────────────────────────
ckpt        = torch.load(f"{DYNAMICS_PATH}/best_model.pt", map_location="cpu")
state_dict  = ckpt["model"]
cfg         = ckpt["config"]
HIDDEN_DIMS = cfg["hidden_dims"]
ACTIVATION  = cfg["activation"]   # "silu"
ANGLE_IDXS  = cfg["angle_idxs"]   # [3, 4, 5]

# Extract weights and biases as numpy arrays (float64 for CasADi)
def extract_layers(state_dict, hidden_dims):
    """Returns list of (W, b) tuples for each Linear layer in the MLP."""
    layers = []
    keys   = [k for k in state_dict if k.endswith(".weight")]
    for key in keys:
        b_key = key.replace(".weight", ".bias")
        W = state_dict[key].numpy().astype(np.float64)
        b = state_dict[b_key].numpy().astype(np.float64)
        layers.append((W, b))
    return layers

layers = extract_layers(state_dict, HIDDEN_DIMS)
print(f"Loaded {len(layers)} layers from checkpoint")
for i, (W, b) in enumerate(layers):
    print(f"  Layer {i}: {W.shape[1]} → {W.shape[0]}")

# ── CasADi symbolic forward pass ──────────────────────────────────────────────
# Reimplements the PyTorch DynamicsModel.forward() using CasADi MX operations.
# This allows IPOPT to differentiate through the network analytically.

def silu_ca(x):
    """SiLU (Swish) activation: x * sigmoid(x)."""
    return x / (1.0 + ca.exp(-x))

def relu_ca(x):
    return ca.fmax(0.0, x)

def tanh_ca(x):
    return ca.tanh(x)

ACT_FN = {"silu": silu_ca, "relu": relu_ca, "tanh": tanh_ca}[ACTIVATION]

def encode_state_ca(e):
    """
    Sinusoidal angle encoding in CasADi.
    e: (12,) MX vector
    returns: (15,) MX vector
    [x, y, z, sin_r, sin_p, sin_y, cos_r, cos_p, cos_y, vx, vy, vz, p, q, r]
    """
    non_angle_before = e[:3]             # x, y, z
    non_angle_after  = e[6:]             # vx, vy, vz, p, q, r
    angles           = ca.vertcat(e[3], e[4], e[5])   # roll, pitch, yaw
    sins             = ca.vertcat(ca.sin(e[3]), ca.sin(e[4]), ca.sin(e[5]))
    coss             = ca.vertcat(ca.cos(e[3]), ca.cos(e[4]), ca.cos(e[5]))
    return ca.vertcat(non_angle_before, sins, coss, non_angle_after)

def nn_forward_ca(e_raw, u_raw):
    """
    Full neural dynamics model forward pass in CasADi symbolic math.

    e_raw: (12,) raw error state  (NOT normalised)
    u_raw: (4,)  raw motor thrust (NOT normalised)

    Returns: (12,) predicted Δe in physical units
    """
    # 1. Normalise inputs
    e_norm = (e_raw - state_mean)  / state_std
    u_norm = (u_raw - action_mean) / action_std

    # 2. Encode angles
    enc = encode_state_ca(e_norm)   # (15,)

    # 3. Concatenate
    x = ca.vertcat(enc, u_norm)     # (19,)

    # 4. Forward through hidden layers with activation
    for i, (W, b) in enumerate(layers[:-1]):
        W_ca = ca.DM(W)
        b_ca = ca.DM(b)
        x    = ACT_FN(W_ca @ x + b_ca)

    # 5. Final linear layer (no activation)
    W_ca = ca.DM(layers[-1][0])
    b_ca = ca.DM(layers[-1][1])
    x    = W_ca @ x + b_ca          # (12,) normalised Δe

    # 6. Denormalise output to physical units
    delta_e = x * delta_std + delta_mean

    return delta_e

def dynamics_ca(e, u):
    """
    Single-step transition: e_{k+1} = e_k + fθ(e_k, u_k)
    """
    return e + nn_forward_ca(e, u)

# ── Build NLP once (warm-started at each call) ────────────────────────────────
print("\nBuilding CasADi NLP...")
build_start = time.time()

E_DIM = 12
U_DIM = 4

# Decision variables: [u_0, ..., u_{N-1}, e_1, ..., e_N]
# Using multiple-shooting: both states and actions are decision variables,
# with dynamics enforced as equality constraints.
# This is more numerically stable than single-shooting for long horizons.

opti = ca.Opti()

# Decision variables
U = opti.variable(U_DIM, N)       # actions over horizon
E = opti.variable(E_DIM, N + 1)   # states  over horizon (e_0 fixed by param)

# Parameters (set at each solve)
e0_param = opti.parameter(E_DIM)       # current error state
ref_param = opti.parameter(E_DIM)      # reference error (usually zeros for hover)

# ── Cost ───────────────────────────────────────────────────────────────────────
Q  = ca.diag(Q_DIAG)
R  = ca.diag(R_DIAG)
QN = ca.diag(QN_DIAG)

cost = 0.0
for k in range(N):
    e_err = E[:, k] - ref_param
    cost += ca.mtimes([e_err.T, Q, e_err]) + ca.mtimes([U[:, k].T, R, U[:, k]])

# Terminal cost
e_err_N = E[:, N] - ref_param
cost    += ca.mtimes([e_err_N.T, QN, e_err_N])

opti.minimize(cost)

# ── Constraints ────────────────────────────────────────────────────────────────
# Initial state
opti.subject_to(E[:, 0] == e0_param)

# Dynamics (multiple shooting)
for k in range(N):
    e_next = dynamics_ca(E[:, k], U[:, k])
    opti.subject_to(E[:, k + 1] == e_next)

# Control bounds
opti.subject_to(opti.bounded(T_MIN, U, T_MAX))


ATT_LIMIT = 0.785
for k in range(N + 1):
    opti.subject_to(opti.bounded(-ATT_LIMIT, E[3, k], ATT_LIMIT))  # roll
    opti.subject_to(opti.bounded(-ATT_LIMIT, E[4, k], ATT_LIMIT))  # pitch

# ── Solver options ─────────────────────────────────────────────────────────────
solver_opts = {
    "ipopt.max_iter":          200,
    "ipopt.tol":               1e-4,
    "ipopt.acceptable_tol":    1e-3,    # accept slightly suboptimal solutions
    "ipopt.acceptable_iter":   5,       # stop after 5 acceptable iterations
    "ipopt.warm_start_init_point": "yes",
    "ipopt.print_level":       0,       # suppress IPOPT output
    "print_time":              False,
}
opti.solver("ipopt", solver_opts)

build_time = time.time() - build_start
print(f"NLP built in {build_time:.2f}s")

# ── MPC controller class ───────────────────────────────────────────────────────
class NeuralMPC:
    """
    Receding-horizon MPC controller using the learned neural dynamics model.

    Usage:
        mpc = NeuralMPC()
        u   = mpc.solve(e_current, e_reference)
    """

    def __init__(self):
        # Derive nominal hover thrust from the action normaliser mean.
        # action_mean was fit on PID rollouts where the drone spends most of its
        # time near hover, so action_mean ≈ hover thrust per rotor.
        self._hover = float(action_mean.mean())

        # Warm-start storage
        self._u_prev = np.ones((U_DIM, N)) * self._hover
        self._e_prev = np.zeros((E_DIM, N + 1))
        self._last_solve_time = None
        self._solve_count      = 0
        self._failed_count     = 0

    def solve(self, e_current: np.ndarray,
              e_ref: np.ndarray = None,
              debug: bool = False) -> np.ndarray:
        """
        Solve one MPC step.

        e_current: (12,) current error state
        e_ref:     (12,) reference error state (default: zeros = hover at setpoint)
        returns:   (4,)  optimal motor thrusts for this step
        """
        if e_ref is None:
            e_ref = np.zeros(E_DIM)

        # ── Set parameters ─────────────────────────────────────────────────────
        opti.set_value(e0_param,  e_current.astype(np.float64))
        opti.set_value(ref_param, e_ref.astype(np.float64))

        # ── Warm start: shift previous solution by one step ────────────────────
        u_init      = np.roll(self._u_prev, -1, axis=1)
        u_init[:, -1] = self._hover          # repeat last action
        e_init      = np.roll(self._e_prev, -1, axis=1)
        e_init[:, 0]  = e_current
        e_init[:, -1] = e_init[:, -2]

        opti.set_initial(U, u_init)
        opti.set_initial(E, e_init)

        # ── Solve ──────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            sol = opti.solve()
            u_opt = sol.value(U)         # (4, N)
            e_opt = sol.value(E)         # (12, N+1)

            # Store for next warm start
            self._u_prev = u_opt
            self._e_prev = e_opt
            u_out        = u_opt[:, 0]   # apply first action only

        except RuntimeError as ex:
            # Solver failed — fall back to previous solution shifted by one step
            self._failed_count += 1
            u_out = self._u_prev[:, 0]
            if debug:
                print(f"  [MPC] Solver failed: {ex}")

        self._last_solve_time = time.perf_counter() - t0
        self._solve_count    += 1

        if debug:
            print(f"  [MPC] solve #{self._solve_count:4d}  "
                  f"t={self._last_solve_time*1e3:.1f}ms  "
                  f"e_z={e_current[2]:+.3f}  u={np.round(u_out, 4)}")

        return np.clip(u_out, T_MIN, T_MAX).astype(np.float32)

    @property
    def solve_time_ms(self):
        return self._last_solve_time * 1e3 if self._last_solve_time else None

# ── Standalone test against MuJoCo ────────────────────────────────────────────
if __name__ == "__main__":
    import mujoco
    from scipy.spatial.transform import Rotation

    XML_PATH   = "basic_quadrotor.xml"
    DT_SIM     = 0.002
    STEPS_PER_CTRL = int(DT_CTRL / DT_SIM)

    def quat_to_euler(q):
        w, x, y, z = q
        r = Rotation.from_quat([x, y, z, w])
        return r.as_euler("xyz")

    def get_error_state(data, setpoint):
        pos   = data.qpos[:3]
        euler = quat_to_euler(data.qpos[3:7])
        vel   = data.qvel[:3]
        omega = data.qvel[3:6]
        state = np.concatenate([pos, euler, vel, omega])
        return (state - setpoint).astype(np.float32)

    # ── Test episode ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Running MPC test episode in MuJoCo...")

    model    = mujoco.MjModel.from_xml_path(XML_PATH)
    data     = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Start 0.3m above hover setpoint to test convergence
    SETPOINT  = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    data.qpos[2] = SETPOINT[2] + 0.3    # start 0.3m above setpoint
    mujoco.mj_forward(model, data)

    mpc        = NeuralMPC()
    solve_times = []
    z_errors    = []

    N_STEPS = 200
    print(f"Simulating {N_STEPS} control steps "
          f"({N_STEPS * DT_CTRL:.1f}s)...\n")
    print(f"  {'step':>5s}  {'z':>7s}  {'z_err':>8s}  "
          f"{'vz':>7s}  {'solve_ms':>9s}  {'u_mean':>8s}")
    print(f"  {'-'*5}  {'-'*7}  {'-'*8}  {'-'*7}  {'-'*9}  {'-'*8}")

    for step in range(N_STEPS):
        e_k = get_error_state(data, SETPOINT)
        u   = mpc.solve(e_k, debug=False)
        data.ctrl[:] = u

        for _ in range(STEPS_PER_CTRL):
            mujoco.mj_step(model, data)

        z     = float(data.qpos[2])
        z_err = float(e_k[2])
        vz    = float(data.qvel[2])
        t_ms  = mpc.solve_time_ms

        solve_times.append(t_ms)
        z_errors.append(abs(z_err))

        if step % 20 == 0:
            print(f"  {step:5d}  {z:7.4f}  {z_err:+8.4f}  "
                  f"{vz:+7.4f}  {t_ms:8.1f}ms  {u.mean():8.5f}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"MPC test summary ({N_STEPS} steps):")
    print(f"  Solve time:  mean={np.mean(solve_times):.1f}ms  "
          f"max={np.max(solve_times):.1f}ms  "
          f"min={np.min(solve_times):.1f}ms")
    print(f"  |z_err|:     mean={np.mean(z_errors):.4f}m  "
          f"final={z_errors[-1]:.4f}m")
    print(f"  Solver fails: {mpc._failed_count}/{N_STEPS}")

    budget_ms = DT_CTRL * 1000   # 20ms
    rt_pct    = np.mean(np.array(solve_times) < budget_ms) * 100
    print(f"\n  Real-time budget ({budget_ms:.0f}ms): {rt_pct:.1f}% of solves within budget")
    if rt_pct < 90:
        print("  ⚠  Consider reducing horizon N or simplifying the model.")
    else:
        print("  ✓  Solve times are within real-time budget.")