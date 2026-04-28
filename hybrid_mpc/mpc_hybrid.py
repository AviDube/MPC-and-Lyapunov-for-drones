"""
MPC using the hybrid physics + NN dynamics model.
"""

import numpy as np
import casadi as ca
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import time
import torch
import torch.nn as nn

from physics import build_physics_fn, rk4_ca

# Config
XML_PATH = "../basic_quadrotor.xml"
DT_CTRL  = 0.02
SIM_TIME = 8.0
HORIZON  = 25   

# Load MuJoCo
mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
mj_data  = mujoco.MjData(mj_model)

MASS     = float(mj_model.body_mass[1])
GRAV     = float(abs(mj_model.opt.gravity[2]))
INERTIA  = tuple(mj_model.body_inertia[1])
HOVER    = MASS * GRAV / 4

nx, nu = 12, 4

def quat_to_euler(q):
    w,x,y,z = q
    return Rotation.from_quat([x,y,z,w]).as_euler("xyz")

def get_state(d):
    return np.concatenate([d.qpos[:3], quat_to_euler(d.qpos[3:7]),
                           d.qvel[:3], d.qvel[3:6]])

def wrench_to_rotors(u):
    T,tx,ty,tz = u
    l=0.028; k=0.02513
    return np.array([T/4-tx/(4*l)-ty/(4*l)-tz/(4*k),
                     T/4-tx/(4*l)+ty/(4*l)+tz/(4*k),
                     T/4+tx/(4*l)+ty/(4*l)-tz/(4*k),
                     T/4+tx/(4*l)-ty/(4*l)+tz/(4*k)])

# casadi functions for physics + NN residuals
class HybridResidualNN(nn.Module):
    def __init__(self, nx=12, nu=4, hidden=64, n_layers=3):
        super().__init__()
        layers = [nn.Linear(nx+nu, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, nx)]
        self.net = nn.Sequential(*layers)

    def forward(self, xu):
        return self.net(xu)


def silu_ca(x):
    return x / (1 + ca.exp(-x))


def linear_ca(x, layer):
    W = ca.DM(layer.weight.detach().numpy().astype(np.float64))
    b = ca.DM(layer.bias.detach().numpy().astype(np.float64).reshape(-1,1))
    return W @ x + b


def build_nn_residual_fn(weights_path="models/hybrid_nn.pt",
                          config_path="models/hybrid_config.npz"):
    cfg      = np.load(config_path)
    hidden   = int(cfg["hidden"][0])
    n_layers = int(cfg["n_layers"][0])
    xu_mean  = cfg["xu_mean"].astype(np.float64)
    xu_std   = cfg["xu_std"].astype(np.float64)

    net = HybridResidualNN(nx=nx, nu=nu, hidden=hidden, n_layers=n_layers)
    net.load_state_dict(torch.load(weights_path, map_location="cpu"))
    net.eval()

    x_sym = ca.MX.sym("x", nx)
    u_sym = ca.MX.sym("u", nu)

    # Normalise input
    xu_raw = ca.vertcat(x_sym, u_sym)
    xu_n   = (xu_raw - ca.DM(xu_mean.reshape(-1,1))) \
             / ca.DM(xu_std.reshape(-1,1))

    # Forward pass through Sequential layers
    h = xu_n
    layers = list(net.net.children())
    for layer in layers:
        if isinstance(layer, nn.Linear):
            h = linear_ca(h, layer)
        elif isinstance(layer, nn.SiLU):
            h = silu_ca(h)
        # ignore any other layer types

    delta_sym = h

    return ca.Function("f_nn_residual", [x_sym, u_sym], [delta_sym],
                       ["x", "u"], ["delta"])


def build_hybrid_fn(f_physics, f_nn):
    x_sym = ca.MX.sym("x", nx)
    u_sym = ca.MX.sym("u", nu)

    x_phys  = f_physics(x_sym, u_sym)
    delta   = f_nn(x_sym, u_sym)
    x_next  = x_phys + delta

    return ca.Function("f_hybrid", [x_sym, u_sym], [x_next],
                       ["x", "u"], ["x_next"])



print("Building hybrid CasADi model...")
f_physics = build_physics_fn(MASS, INERTIA, GRAV, DT_CTRL)

try:
    f_nn     = build_nn_residual_fn()
    f_hybrid = build_hybrid_fn(f_physics, f_nn)
    print("  Loaded hybrid_nn.pt — using physics + NN")
    USE_NN = True
except FileNotFoundError:
    print("  hybrid_nn.pt not found — using physics only (no NN correction)")
    f_hybrid = f_physics
    USE_NN   = False


def build_nlp(horizon=HORIZON):
    N = horizon

    Q   = np.diag([20, 20, 100, 5, 5, 50, 2, 2, 5, 1, 1, 10])
    Qf  = Q * 10
    R   = 0.01 * np.eye(nu)
    Rdu = 0.10 * np.eye(nu)

    Q_ca = ca.DM(Q); Qf_ca = ca.DM(Qf)
    R_ca = ca.DM(R); Rdu_ca = ca.DM(Rdu)

    X_sym = [ca.MX.sym(f"x_{k}", nx) for k in range(N+1)]
    U_sym = [ca.MX.sym(f"u_{k}", nu) for k in range(N)]

    p_sym    = ca.MX.sym("p", nx + nx)
    x0_sym   = p_sym[:nx]
    xref_sym = p_sym[nx:]
    u_hover  = ca.DM([MASS*GRAV, 0.0, 0.0, 0.0])

    obj    = ca.MX(0)
    con    = []
    con_lb = []
    con_ub = []

    con.append(X_sym[0] - x0_sym)
    con_lb.extend([0.0] * nx)
    con_ub.extend([0.0] * nx)

    for k in range(N):
        ex = X_sym[k] - xref_sym
        obj += ca.bilin(Q_ca, ex, ex)

        eu = U_sym[k] - u_hover
        obj += ca.bilin(R_ca, eu, eu)

        u_prev = U_sym[k-1] if k > 0 else u_hover
        obj += ca.bilin(Rdu_ca, U_sym[k] - u_prev, U_sym[k] - u_prev)

        # Hybrid dynamics constraint — physics + NN
        x_next = f_hybrid(X_sym[k], U_sym[k])
        con.append(X_sym[k+1] - x_next)
        con_lb.extend([0.0] * nx)
        con_ub.extend([0.0] * nx)

    ex = X_sym[N] - xref_sym
    obj += ca.bilin(Qf_ca, ex, ex)

    u_lb = ca.DM([0.0,         -0.005, -0.005, -0.02])
    u_ub = ca.DM([2*MASS*GRAV,  0.005,  0.005,  0.02])

    w    = ca.vertcat(*X_sym, *U_sym)
    w_lb = ca.vertcat(*[-ca.inf * ca.DM.ones(nx)] * (N+1), *[u_lb] * N)
    w_ub = ca.vertcat(*[ ca.inf * ca.DM.ones(nx)] * (N+1), *[u_ub] * N)

    con_expr = ca.vertcat(*con)
    con_lb   = ca.DM(con_lb)
    con_ub   = ca.DM(con_ub)

    nlp = {"x": w, "f": obj, "g": con_expr, "p": p_sym}

    opts = {
        "ipopt.max_iter":                   300,
        "ipopt.tol":                        1e-3,
        "ipopt.constr_viol_tol":            1e-3,
        "ipopt.acceptable_tol":             1e-2,
        "ipopt.acceptable_iter":            5,
        "ipopt.warm_start_init_point":      "yes",
        "ipopt.warm_start_bound_push":      1e-6,
        "ipopt.warm_start_mult_bound_push": 1e-6,
        "ipopt.mu_init":                    1e-3,
        "ipopt.print_level":                0,
        "ipopt.sb":                         "yes",
        "print_time":                       False,
    }

    solver = ca.nlpsol("solver", "ipopt", nlp, opts)
    meta   = {
        "N": N, "w_lb": w_lb, "w_ub": w_ub,
        "con_lb": con_lb, "con_ub": con_ub,
        "n_x_vars": (N+1)*nx,
    }
    return solver, meta


print("Building NLP...")
t0 = time.time()
solver, meta = build_nlp()
print(f"NLP built in {time.time()-t0:.1f}s\n")
N = meta["N"]


# MPC
class HybridMPC:
    def __init__(self):
        self.w0     = None
        self.lam_g0 = None
        self.lam_w0 = None
        self.integral_error = np.zeros(4)
        self.integral_clip  = np.array([0.3, 0.3, 1.5, 0.3])

    def _initial_guess(self, x0, x_ref):
        X_init = np.linspace(x0, x_ref, N+1)
        U_init = np.tile([MASS*GRAV, 0, 0, 0], (N, 1))
        return np.concatenate([X_init.flatten(), U_init.flatten()])

    def solve(self, x0, x_ref):
        # Integral action
        self.integral_error += (x_ref[[0,1,2,5]] - x0[[0,1,2,5]]) * DT_CTRL
        self.integral_error  = np.clip(self.integral_error,
                                       -self.integral_clip, self.integral_clip)

        cos_tilt   = np.clip(np.cos(x0[3])*np.cos(x0[4]), 0.5, 1.0)
        u_fallback = np.array([MASS*GRAV/cos_tilt, 0.0, 0.0, 0.0])

        if self.w0 is None:
            w0 = self._initial_guess(x0, x_ref)
        else:
            n_xv   = meta["n_x_vars"]
            X_prev = self.w0[:n_xv].reshape(N+1, nx)
            U_prev = self.w0[n_xv:].reshape(N, nu)
            w0     = np.concatenate([
                np.vstack([X_prev[1:], X_prev[-1:]]).flatten(),
                np.vstack([U_prev[1:], U_prev[-1:]]).flatten(),
            ])

        kwargs = dict(
            x0=w0, p=np.concatenate([x0, x_ref]),
            lbx=meta["w_lb"], ubx=meta["w_ub"],
            lbg=meta["con_lb"], ubg=meta["con_ub"],
        )
        if self.lam_g0 is not None:
            kwargs["lam_g0"] = self.lam_g0
            kwargs["lam_x0"] = self.lam_w0

        sol   = solver(**kwargs)
        stats = solver.stats()
        ok    = stats["success"] or \
                stats["return_status"] == "Solved_To_Acceptable_Level"

        if not ok:
            print(f"  IPOPT: {stats['return_status']} — holding hover")
            return u_fallback

        w_opt       = np.array(sol["x"]).flatten()
        self.w0     = w_opt
        self.lam_g0 = sol["lam_g"]
        self.lam_w0 = sol["lam_x"]
        return w_opt[meta["n_x_vars"]:].reshape(N, nu)[0]


# Simulation
mpc   = HybridMPC()
x_ref = np.zeros(nx)
x_ref[0]=0.5; x_ref[1]=-0.5; x_ref[2]=1.0; x_ref[5]=0.0

print(f"Model: {'Physics + NN' if USE_NN else 'Physics only'}")
print(f"Target: x={x_ref[0]}, y={x_ref[1]}, z={x_ref[2]}\n")

states, controls, times, solve_times = [], [], [], []

with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    t = 0.0
    while viewer.is_running() and t < SIM_TIME:
        x = get_state(mj_data)

        t0       = time.time()
        u_wrench = mpc.solve(x, x_ref)
        dt_solve = time.time() - t0

        if len(times) % 25 == 0:
            print(f"t={t:.1f}s | z={x[2]:.3f} | err_z={x_ref[2]-x[2]:.3f} | "
                  f"T={u_wrench[0]:.3f} | solve={dt_solve*1e3:.0f}ms")

        u_rotors = np.clip(wrench_to_rotors(u_wrench), 0.005, 0.25)
        mj_data.ctrl[:] = u_rotors

        states.append(x.copy())
        controls.append(u_rotors.copy())
        times.append(t)
        solve_times.append(dt_solve)

        for _ in range(int(DT_CTRL / mj_model.opt.timestep)):
            mujoco.mj_step(mj_model, mj_data)

        viewer.sync()
        t += DT_CTRL

states      = np.array(states)
controls    = np.array(controls)
times       = np.array(times)
solve_times = np.array(solve_times)

print(f"\nSolve time: mean={solve_times.mean()*1e3:.0f}ms  "
      f"p95={np.percentile(solve_times,95)*1e3:.0f}ms  "
      f"max={solve_times.max()*1e3:.0f}ms")

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
ax = axes[0]
for i,(lbl,c) in enumerate(zip(["x","y","z"],
                                ["tab:blue","tab:orange","tab:green"])):
    ax.plot(times, states[:,i], label=lbl, color=c)
    ax.axhline(x_ref[i], ls="--", color=c, alpha=0.4)
ax.set_title("Position"); ax.set_xlabel("Time (s)"); ax.legend(); ax.grid()

ax = axes[1]
for i,lbl in enumerate(["roll","pitch","yaw"]):
    ax.plot(times, np.degrees(states[:,3+i]), label=lbl)
ax.set_title("Orientation (deg)"); ax.set_xlabel("Time (s)"); ax.legend(); ax.grid()

ax = axes[2]
for i in range(4):
    ax.plot(times, controls[:,i], label=f"rotor {i}", alpha=0.7)
ax.axhline(HOVER, ls="--", color="k", alpha=0.4, label="hover/4")
ax.set_title("Rotor thrusts"); ax.set_xlabel("Time (s)"); ax.legend(); ax.grid()

ax = axes[3]
ax.plot(times, solve_times*1e3)
ax.axhline(DT_CTRL*1e3, ls="--", color="r", alpha=0.6,
           label=f"budget ({DT_CTRL*1e3:.0f}ms)")
ax.set_title("IPOPT solve time"); ax.set_xlabel("Time (s)")
ax.set_ylabel("ms"); ax.legend(); ax.grid()

plt.tight_layout(); plt.show()