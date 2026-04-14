import os
os.environ.setdefault("MUJOCO_GL", "glfw")  # needed for WSL/Linux rendering

import numpy as np
import torch
import casadi as ca
import time

# ── Paths ──────────────────────────────────────────────────────────────────────
DYNAMICS_PATH = "dynamics_model"
CODEGEN_DIR   = "mpc_codegen"
os.makedirs(CODEGEN_DIR, exist_ok=True)

# ── MPC hyperparameters ────────────────────────────────────────────────────────
N        = 5           # prediction horizon (steps) — needs ≥5 for xy tracking
                       # because tilt→lateral-acceleration→position takes time
DT_CTRL  = 0.02        # control timestep (s)
T_MIN    = 0.01        # motor thrust lower bound (N)
T_MAX    = 0.15        # motor thrust upper bound (N)

E_DIM = 12
U_DIM = 4

# ── Cost weights ───────────────────────────────────────────────────────────────
# The quadrotor controls xy position indirectly: it must tilt (roll/pitch)
# to accelerate laterally. If roll/pitch penalty is too high relative to
# xy position penalty, the optimizer refuses to tilt and xy error persists.
#
# State: [x, y, z, roll, pitch, yaw, vx, vy, vz, p, q, r]
Q_DIAG = np.array([
    5.0,  5.0,  10.0,   # position error — xy raised from 1.0 to 5.0
    2.0,  2.0,  1.0,    # attitude error — roll/pitch lowered from 5.0 to 2.0
    1.0,  1.0,  2.0,    # velocity error — vx/vy raised for damping
    0.1,  0.1,  0.1,    # angular rate error
], dtype=np.float64)

# Action cost — moderate: prevent unnecessary oscillation but don't
# prevent the optimizer from using thrust when needed.
R_DIAG  = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float64)

QN_DIAG = 5.0 * Q_DIAG

# Hover regularization — light touch now that the NN has good gradients
# across the full thrust range. Just enough to break ties, not enough
# to prevent climbing/descending when the state cost demands it.
R_HOVER = 0.5

ATT_LIMIT = 0.785

# ── Load normalisers ───────────────────────────────────────────────────────────
def load_norm(path):
    d = torch.load(path, map_location="cpu")
    return d["mean"].numpy().astype(np.float64), d["std"].numpy().astype(np.float64)

state_mean,  state_std  = load_norm(f"{DYNAMICS_PATH}/state_norm.pt")
action_mean, action_std = load_norm(f"{DYNAMICS_PATH}/action_norm.pt")
delta_mean,  delta_std  = load_norm(f"{DYNAMICS_PATH}/delta_norm.pt")

# ── Load model weights ─────────────────────────────────────────────────────────
ckpt       = torch.load(f"{DYNAMICS_PATH}/best_model.pt", map_location="cpu")
state_dict = ckpt["model"]
cfg        = ckpt["config"]
HIDDEN_DIMS = cfg["hidden_dims"]
ACTIVATION  = cfg["activation"]

def extract_layers(sd):
    layers = []
    keys = [k for k in sd if k.endswith(".weight")]
    for key in keys:
        W = sd[key].numpy().astype(np.float64)
        b = sd[key.replace(".weight", ".bias")].numpy().astype(np.float64)
        layers.append((W, b))
    return layers

layers = extract_layers(state_dict)
print(f"Loaded {len(layers)} layers from checkpoint")
for i, (W, b) in enumerate(layers):
    print(f"  Layer {i}: {W.shape[1]} → {W.shape[0]}")

# ══════════════════════════════════════════════════════════════════════════════
# NN sanity check — verify the model predicts correct physics
# ══════════════════════════════════════════════════════════════════════════════
def nn_eval_numpy(e_raw, u_raw):
    """Evaluate the NN dynamics in numpy (for diagnostics only)."""
    e_norm = (e_raw - state_mean) / state_std
    u_norm = (u_raw - action_mean) / action_std

    # Encode
    enc = np.concatenate([
        e_norm[:3],
        np.sin(e_norm[3:6]), np.cos(e_norm[3:6]),
        e_norm[6:],
    ])
    x = np.concatenate([enc, u_norm])

    # Forward pass
    act_fn = {"silu": lambda z: z / (1 + np.exp(-z)),
              "relu": lambda z: np.maximum(0, z),
              "tanh": np.tanh}[ACTIVATION]

    for W, b in layers[:-1]:
        x = act_fn(W @ x + b)
    W_last, b_last = layers[-1]
    x = W_last @ x + b_last

    return x * delta_std + delta_mean

print("\n── NN Sanity Check ──────────────────────────────────────")
hover_thrust = float(action_mean.mean())
print(f"  Hover thrust (from action_mean): {hover_thrust:.5f} N/rotor")

# Test 1: at hover state with hover thrust, Δe should be near zero
e_hover = np.zeros(E_DIM)
u_hover = np.full(U_DIM, hover_thrust)
de_hover = nn_eval_numpy(e_hover, u_hover)
print(f"\n  Test 1: e=0, u=hover → Δe_z = {de_hover[2]:+.6f} (expect ≈ 0)")
print(f"          Δe_vz = {de_hover[8]:+.6f} (expect ≈ 0)")

# Test 2: above setpoint (z_err > 0) with hover thrust → should drift up (Δe_z > 0)
#          or at least vz should not get more positive
e_above = np.zeros(E_DIM); e_above[2] = 0.3
de_above = nn_eval_numpy(e_above, u_hover)
print(f"\n  Test 2: e_z=+0.3, u=hover → Δe_z = {de_above[2]:+.6f}")
print(f"          Δe_vz = {de_above[8]:+.6f}")

# Test 3: above setpoint with LOW thrust → should descend (Δe_z < 0 or Δe_vz < 0)
u_low = np.full(U_DIM, T_MIN)
de_low = nn_eval_numpy(e_above, u_low)
print(f"\n  Test 3: e_z=+0.3, u=T_MIN → Δe_z = {de_low[2]:+.6f}")
print(f"          Δe_vz = {de_low[8]:+.6f} (expect negative = descending)")

# Test 4: above setpoint with HIGH thrust → should ascend more
u_high = np.full(U_DIM, T_MAX)
de_high = nn_eval_numpy(e_above, u_high)
print(f"\n  Test 4: e_z=+0.3, u=T_MAX → Δe_z = {de_high[2]:+.6f}")
print(f"          Δe_vz = {de_high[8]:+.6f} (expect positive = ascending)")

# Check: does low thrust give more negative vz than high thrust?
if de_low[8] < de_high[8]:
    print("\n  ✓ Model correctly predicts: lower thrust → more downward acceleration")
else:
    print("\n  ✗ WARNING: Model does NOT correctly predict thrust-to-vz relationship!")
    print("    The optimizer cannot find the right action if this mapping is wrong.")
    print("    This suggests the training data or model needs improvement.")

print("─" * 57)

# ══════════════════════════════════════════════════════════════════════════════
# Build NLP with SX symbolics
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding CasADi NLP with SX symbolics...")
build_start = time.time()

def silu_sx(x):
    return x / (1.0 + ca.exp(-x))

ACT_FN = {"silu": silu_sx, "relu": lambda x: ca.fmax(0.0, x),
           "tanh": lambda x: ca.tanh(x)}[ACTIVATION]

def encode_state_sx(e):
    return ca.vertcat(
        e[:3],
        ca.sin(e[3]), ca.sin(e[4]), ca.sin(e[5]),
        ca.cos(e[3]), ca.cos(e[4]), ca.cos(e[5]),
        e[6:],
    )

# State clamping bounds — derived from training data coverage.
# Keeps NN inputs within the region where it has seen data.
STATE_CLAMP_LO = np.array([
    -0.8, -0.8, -0.8,      # position errors
    -0.5, -0.5, -0.5,      # attitude errors (rad)
    -1.5, -1.5, -1.5,      # velocity errors
    -1.0, -1.0, -1.0,      # angular rate errors
], dtype=np.float64)
STATE_CLAMP_HI = -STATE_CLAMP_LO

def nn_forward_sx(e_raw, u_raw):
    # Clamp state to training distribution before feeding to NN
    e_clamped = ca.fmin(ca.fmax(e_raw, STATE_CLAMP_LO), STATE_CLAMP_HI)
    e_norm = (e_clamped - state_mean) / state_std
    u_norm = (u_raw - action_mean) / action_std
    x = ca.vertcat(encode_state_sx(e_norm), u_norm)
    for W, b in layers[:-1]:
        x = ACT_FN(ca.DM(W) @ x + ca.DM(b))
    W_last, b_last = layers[-1]
    x = ca.DM(W_last) @ x + ca.DM(b_last)
    return x * delta_std + delta_mean

def dynamics_sx(e, u):
    return e + nn_forward_sx(e, u)

# Flat NLP
n_w = N * U_DIM + N * E_DIM
n_p = 2 * E_DIM

w = ca.SX.sym("w", n_w)
p = ca.SX.sym("p", n_p)

def get_u(k):
    return w[k*U_DIM:(k+1)*U_DIM]

def get_e_shoot(k):
    idx = N*U_DIM + (k-1)*E_DIM
    return w[idx:idx+E_DIM]

e0   = p[:E_DIM]
eref = p[E_DIM:]

e_seq = [e0] + [get_e_shoot(k) for k in range(1, N+1)]

Q_dm  = ca.DM(np.diag(Q_DIAG))
R_dm  = ca.DM(np.diag(R_DIAG))
QN_dm = ca.DM(np.diag(QN_DIAG))

# Hover thrust vector for regularization
u_hover_dm = ca.DM(np.full(U_DIM, hover_thrust))

cost = ca.SX(0.0)
g_list = []

for k in range(N):
    e_err = e_seq[k] - eref
    u_k   = get_u(k)

    # State cost + action cost
    cost += ca.mtimes([e_err.T, Q_dm, e_err]) + ca.mtimes([u_k.T, R_dm, u_k])

    # Hover regularization: penalize ||u - u_hover||^2
    # Keeps optimizer near hover where the NN has good gradients
    u_dev = u_k - u_hover_dm
    cost += R_HOVER * ca.dot(u_dev, u_dev)

    # Control rate-of-change penalty: penalize large changes between steps
    # This prevents the bang-bang oscillation (T_MIN → T_MAX → T_MIN)
    if k > 0:
        u_prev = get_u(k - 1)
        du = u_k - u_prev
        cost += R_HOVER * ca.dot(du, du)

    g_list.append(e_seq[k+1] - dynamics_sx(e_seq[k], u_k))

e_err_N = e_seq[N] - eref
cost += ca.mtimes([e_err_N.T, QN_dm, e_err_N])

g   = ca.vertcat(*g_list)
lbg = np.zeros(N * E_DIM)
ubg = np.zeros(N * E_DIM)

lbw = np.full(n_w, -np.inf)
ubw = np.full(n_w,  np.inf)

for k in range(N):
    idx = k * U_DIM
    lbw[idx:idx+U_DIM] = T_MIN
    ubw[idx:idx+U_DIM] = T_MAX

for k in range(1, N+1):
    idx = N*U_DIM + (k-1)*E_DIM
    lbw[idx+3] = -ATT_LIMIT;  ubw[idx+3] = ATT_LIMIT
    lbw[idx+4] = -ATT_LIMIT;  ubw[idx+4] = ATT_LIMIT

nlp = {"x": w, "f": cost, "g": g, "p": p}

solver_opts = {
    "ipopt.max_iter":              50,
    "ipopt.tol":                   1e-3,
    "ipopt.acceptable_tol":        5e-3,
    "ipopt.acceptable_iter":       3,
    "ipopt.warm_start_init_point": "yes",
    "ipopt.mu_init":               1e-3,
    "ipopt.print_level":           0,
    "print_time":                  False,
}

# ── Build solver ──────────────────────────────────────────────────────────────
# CasADi's built-in JIT option compiles NLP callbacks (cost, constraints,
# Jacobians, Hessian) to C and dlopens them automatically.
#
# We use -O0 because the generated C files are huge (thousands of scalar ops
# from unrolling the NN) and -O2 can hang for minutes. The speedup comes
# from eliminating Python/interpreter dispatch, not from gcc optimisation.

solver = None

# Try JIT compilation
try:
    print("Attempting JIT-compiled solver...")
    jit_opts = dict(solver_opts)
    jit_opts["jit"] = True
    jit_opts["compiler"] = "shell"
    jit_opts["jit_options"] = {
        "compiler": "gcc",
        "flags": ["-O0", "-fPIC"],
    }
    jit_opts["jit_temp_suffix"] = False

    solver = ca.nlpsol("mpc_solver", "ipopt", nlp, jit_opts)
    print("  ✓ JIT-compiled solver ready")

except Exception as ex:
    print(f"  ✗ JIT failed: {ex}")
    print("  Using interpreted SX solver (still faster than Opti/MX)")
    solver = ca.nlpsol("mpc_solver", "ipopt", nlp, solver_opts)

build_time = time.time() - build_start
print(f"NLP built in {build_time:.2f}s")

# ── MPC controller class ───────────────────────────────────────────────────────
class NeuralMPC:

    def __init__(self):
        self._hover = float(action_mean.mean())

        self._w_prev = np.zeros(n_w)
        for k in range(N):
            self._w_prev[k*U_DIM:(k+1)*U_DIM] = self._hover

        self._lam_g_prev = np.zeros(N * E_DIM)
        self._lam_x_prev = np.zeros(n_w)

        self._last_solve_time = None
        self._solve_count      = 0
        self._failed_count     = 0

    def solve(self, e_current: np.ndarray,
              e_ref: np.ndarray = None,
              debug: bool = False) -> np.ndarray:

        if e_ref is None:
            e_ref = np.zeros(E_DIM)

        p_val = np.concatenate([e_current.astype(np.float64),
                                e_ref.astype(np.float64)])

        w_init = self._shift_warmstart(e_current)

        t0 = time.perf_counter()
        try:
            sol = solver(
                x0     = w_init,
                lbx    = lbw,
                ubx    = ubw,
                lbg    = lbg,
                ubg    = ubg,
                p      = p_val,
                lam_g0 = self._lam_g_prev,
                lam_x0 = self._lam_x_prev,
            )

            w_opt  = np.array(sol["x"]).flatten()
            stats  = solver.stats()
            success = stats.get("success", False)

            if success:
                self._w_prev     = w_opt
                self._lam_g_prev = np.array(sol["lam_g"]).flatten()
                self._lam_x_prev = np.array(sol["lam_x"]).flatten()
            else:
                self._failed_count += 1
                if debug:
                    print(f"  [MPC] Non-converged: {stats.get('return_status')}")

            u_out = w_opt[:U_DIM]

        except Exception as ex:
            self._failed_count += 1
            u_out = np.full(U_DIM, self._hover)
            if debug:
                print(f"  [MPC] Exception: {ex}")

        self._last_solve_time = time.perf_counter() - t0
        self._solve_count += 1

        if debug:
            print(f"  [MPC] #{self._solve_count:4d}  "
                  f"t={self._last_solve_time*1e3:.1f}ms  "
                  f"e_z={e_current[2]:+.3f}  u={np.round(u_out, 4)}")

        return np.clip(u_out, T_MIN, T_MAX).astype(np.float32)

    def _shift_warmstart(self, e_current):
        w = np.copy(self._w_prev)
        for k in range(N - 1):
            w[k*U_DIM:(k+1)*U_DIM] = self._w_prev[(k+1)*U_DIM:(k+2)*U_DIM]
        w[(N-1)*U_DIM:N*U_DIM] = self._hover
        e_base = N * U_DIM
        for k in range(N - 1):
            src = e_base + (k+1)*E_DIM
            dst = e_base + k*E_DIM
            w[dst:dst+E_DIM] = self._w_prev[src:src+E_DIM]
        return w

    @property
    def solve_time_ms(self):
        return self._last_solve_time * 1e3 if self._last_solve_time else None

# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import mujoco
    import mujoco.viewer
    from scipy.spatial.transform import Rotation
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-viewer", action="store_true",
                        help="Run without MuJoCo viewer (headless)")
    args = parser.parse_args()

    XML_PATH       = "basic_quadrotor.xml"
    DT_SIM         = 0.002
    STEPS_PER_CTRL = int(DT_CTRL / DT_SIM)
    USE_VIEWER     = not args.no_viewer

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

    print("\n" + "="*60)
    print("Running MPC test episode in MuJoCo...")
    print(f"  Hover thrust: {hover_thrust:.5f} N/rotor")
    print(f"  Thrust range: [{T_MIN}, {T_MAX}]")
    print(f"  Horizon N={N}, DT_CTRL={DT_CTRL}s")
    print(f"  Viewer: {'ON' if USE_VIEWER else 'OFF'}")

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    SETPOINT = np.array([0,0,0.5, 0,0,0, 0,0,0, 0,0,0], dtype=np.float32)

    # Start offset in all three axes — within training data range (±0.5m)
    data.qpos[0] = 0.10    # 10cm offset in x
    data.qpos[1] = -0.10   # 10cm offset in y
    data.qpos[2] = 0.35    # 15cm below setpoint z
    mujoco.mj_forward(model, data)

    print(f"  Initial position: x={data.qpos[0]:.2f}, "
          f"y={data.qpos[1]:.2f}, z={data.qpos[2]:.2f}")
    print(f"  Setpoint:         x={SETPOINT[0]:.2f}, "
          f"y={SETPOINT[1]:.2f}, z={SETPOINT[2]:.2f}")

    mpc         = NeuralMPC()
    solve_times = []
    z_errors    = []
    xy_errors   = []

    # ── Trajectory recording for post-sim plots ───────────────────────────
    traj_z      = []
    traj_vz     = []
    traj_u_mean = []
    traj_time   = []
    traj_x      = []
    traj_y      = []

    N_STEPS = 1000   # longer episode to see convergence
    print(f"\nSimulating {N_STEPS} control steps ({N_STEPS*DT_CTRL:.1f}s)...\n")

    # Detailed header — now includes x, y, xy_err
    print(f"  {'step':>4s}  {'x':>6s}  {'y':>6s}  {'z':>6s}  "
          f"{'xy_err':>7s}  {'z_err':>7s}  {'vz':>7s}  "
          f"{'u_mean':>7s}  {'vs_hov':>7s}  "
          f"{'ms':>6s}  {'status':>8s}")
    print(f"  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*6}  "
          f"{'-'*7}  {'-'*7}  {'-'*7}  "
          f"{'-'*7}  {'-'*7}  "
          f"{'-'*6}  {'-'*8}")

    # ── Simulation with optional viewer ───────────────────────────────────
    def run_sim(viewer_handle=None):
        for step in range(N_STEPS):
            e_k = get_error_state(data, SETPOINT)
            u   = mpc.solve(e_k, debug=False)

            data.ctrl[:] = u

            for _ in range(STEPS_PER_CTRL):
                mujoco.mj_step(model, data)

            # Sync viewer if running
            if viewer_handle is not None:
                viewer_handle.sync()

            x     = float(data.qpos[0])
            y     = float(data.qpos[1])
            z     = float(data.qpos[2])
            x_err = float(e_k[0])
            y_err = float(e_k[1])
            z_err = float(e_k[2])
            vz    = float(data.qvel[2])
            xy_err = np.sqrt(x_err**2 + y_err**2)
            t_ms  = mpc.solve_time_ms

            solve_times.append(t_ms)
            z_errors.append(abs(z_err))
            xy_errors.append(xy_err)
            traj_z.append(z)
            traj_vz.append(vz)
            traj_u_mean.append(float(u.mean()))
            traj_time.append(step * DT_CTRL)
            traj_x.append(x)
            traj_y.append(y)

            # Solver status
            try:
                stats = solver.stats()
                status = "OK" if stats.get("success", False) else "FAIL"
                n_iter = stats.get("iter_count", "?")
                status_str = f"{status}({n_iter})"
            except:
                status_str = "??"

            u_vs_hover = u.mean() - hover_thrust

            if step < 50 and step % 5 == 0 or step >= 50 and step % 20 == 0:
                print(f"  {step:4d}  {x:6.3f}  {y:6.3f}  {z:6.3f}  "
                      f"{xy_err:7.4f}  {z_err:+7.4f}  {vz:+7.4f}  "
                      f"{u.mean():7.5f}  {u_vs_hover:+7.4f}  "
                      f"{t_ms:6.1f}  {status_str:>8s}")

            # Early termination if drone crashes or flies away
            if z < 0.01 or z > 5.0:
                print(f"\n  ⚠ Episode terminated at step {step}: z={z:.3f}")
                break

    # Launch with or without viewer
    if USE_VIEWER:
        try:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                # Set camera to see the drone well
                viewer.cam.distance = 2.0
                viewer.cam.elevation = -20
                viewer.cam.azimuth = 135
                viewer.cam.lookat[:] = [0, 0, 0.5]
                run_sim(viewer)
        except Exception as ex:
            print(f"  Viewer failed ({ex}), running headless...")
            run_sim(None)
    else:
        run_sim(None)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"MPC test summary ({len(solve_times)} steps):")
    print(f"  Hover thrust:  {hover_thrust:.5f} N/rotor")
    print(f"  Solve time:    mean={np.mean(solve_times):.1f}ms  "
          f"max={np.max(solve_times):.1f}ms  "
          f"min={np.min(solve_times):.1f}ms")
    print(f"  |z_err|:       mean={np.mean(z_errors):.4f}m  "
          f"final={z_errors[-1]:.4f}m")
    print(f"  |xy_err|:      mean={np.mean(xy_errors):.4f}m  "
          f"final={xy_errors[-1]:.4f}m")
    print(f"  Final pos:     x={traj_x[-1]:.4f}  y={traj_y[-1]:.4f}  z={traj_z[-1]:.4f}")
    print(f"  Solver fails:  {mpc._failed_count}/{len(solve_times)}")

    budget_ms = DT_CTRL * 1000
    rt_pct    = np.mean(np.array(solve_times) < budget_ms) * 100
    print(f"\n  Real-time budget ({budget_ms:.0f}ms): {rt_pct:.1f}% within budget")
    if rt_pct < 90:
        print("  ⚠  Consider reducing horizon N or simplifying the model.")
    else:
        print("  ✓  Solve times are within real-time budget.")

    # ── NN prediction sweep ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("NN prediction sweep at e_z=+0.3, vz=0:")
    print(f"  {'thrust':>8s}  {'Δe_z':>10s}  {'Δe_vz':>10s}  {'next_z_err':>12s}  direction")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*10}")
    e_test = np.zeros(E_DIM); e_test[2] = 0.3
    for t_val in np.linspace(T_MIN, T_MAX, 8):
        u_test = np.full(U_DIM, t_val)
        de = nn_eval_numpy(e_test, u_test)
        next_z = e_test[2] + de[2]
        direction = "↓ DOWN" if de[8] < 0 else "↑ UP"
        print(f"  {t_val:8.5f}  {de[2]:+10.6f}  {de[8]:+10.6f}  {next_z:12.6f}  {direction}")

    # ── Save trajectory to CSV for external plotting ──────────────────────
    traj_path = "mpc_trajectory.csv"
    n = len(traj_z)
    traj_data = np.column_stack([
        traj_time, traj_x, traj_y, traj_z, traj_vz, traj_u_mean,
        z_errors[:n], xy_errors[:n], solve_times[:n]
    ])
    np.savetxt(traj_path, traj_data, delimiter=",",
               header="time,x,y,z,vz,u_mean,z_err_abs,xy_err,solve_ms", comments="")
    print(f"\n  Trajectory saved to '{traj_path}' for plotting.")