"""
Evaluates the residual policy and Lyapunov function on MuJoCo rollouts.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mujoco
import mujoco.viewer

from mpc_hybrid import HybridMPC, get_state, wrench_to_rotors

XML_PATH = "../basic_quadrotor.xml"
DT_CTRL  = 0.02
nx, nu   = 12, 4

mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
MASS     = float(mj_model.body_mass[1])
GRAV     = float(abs(mj_model.opt.gravity[2]))

X_REF  = np.array([0.5,-0.5,1.0, 0,0,0, 0,0,0, 0,0,0], dtype=np.float32)
U_STAR = np.array([MASS*GRAV, 0.0, 0.0, 0.0], dtype=np.float32)



class ResidualPolicy(nn.Module):
    def __init__(self, nx=12, hidden=128, n_layers=3):
        super().__init__()
        layers = [nn.Linear(nx, hidden), nn.Tanh()]
        for _ in range(n_layers-1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, nu, bias=False)]
        self.delta_net = nn.Sequential(*layers)
        self.register_buffer("u_star",    torch.from_numpy(U_STAR))
        self.register_buffer("delta_max", torch.tensor([MASS*GRAV,0.005,0.005,0.02]))

    def forward(self, e):
        return self.u_star + self.delta_max * torch.tanh(self.delta_net(e))


def load_policy():

    for cfg_path, weights_path in [
        ("models/policy_lyap_config.npz", "models/residual_policy.pt"),
        ("models/policy_config.npz",      "models/neural_policy.pt"),
    ]:
        try:
            cfg     = np.load(cfg_path)
            e_scale = cfg["e_scale"].astype(np.float32)
            policy  = ResidualPolicy()
            policy.load_state_dict(torch.load(weights_path, map_location="cpu"))
            policy.eval()
            print(f"  Loaded policy from {weights_path}")
            print(f"  Config from {cfg_path}")
            return policy, e_scale
        except FileNotFoundError:
            continue
    raise FileNotFoundError(
        "No policy weights found. Run residual_lyap_policy.py first.")


def policy_action(policy, e_scale, x):
    e   = (x - X_REF).astype(np.float32)
    e_n = e / e_scale
    with torch.no_grad():
        u = policy(torch.from_numpy(e_n).unsqueeze(0))
    return u.squeeze(0).numpy()


class LyapunovFunction(nn.Module):
    def __init__(self, nx=12, hidden=128, feat_dim=64):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(nx, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, feat_dim, bias=False))
        self.W = nn.Parameter(torch.eye(feat_dim) * 0.1)

    def forward(self, e):
        feat = self.phi(e); Wf = feat @ self.W.T
        return (Wf**2).sum(dim=1, keepdim=True)


def load_lyapunov():

    # Joint training config
    for weights_path, cfg_path, hidden, feat_dim, label in [
        ("models/lyapunov_function.pt", "models/policy_lyap_config.npz",
         128, 64, "joint training"),
        ("models/lyapunov_net.pt",      "models/lyapunov_config.npz",
         None, None, "separate training"),
    ]:
        try:
            if cfg_path == "models/lyapunov_config.npz":
                cfg      = np.load(cfg_path)
                e_scale  = cfg["e_scale"].astype(np.float32)
                alpha    = float(cfg["alpha"][0])
                hidden   = int(cfg["hidden"][0])
                feat_dim = int(cfg["feat_dim"][0])
                c_roa    = 5.0
            else:
                cfg     = np.load(cfg_path)
                e_scale = cfg["e_scale"].astype(np.float32)
                alpha   = 0.05 
                c_roa   = 5.0

            V_net = LyapunovFunction(nx=nx, hidden=hidden, feat_dim=feat_dim)
            V_net.load_state_dict(torch.load(weights_path, map_location="cpu"))
            V_net.eval()
            print(f"  Loaded Lyapunov V from {weights_path}  [{label}]")
            return V_net, e_scale, alpha, c_roa
        except FileNotFoundError:
            continue

    print("  No Lyapunov weights found — V overlay disabled")
    return None, None, None, None


def eval_V(V_net, e_scale, states):
    e   = (states - X_REF).astype(np.float32)
    e_n = e / e_scale
    with torch.no_grad():
        V = V_net(torch.from_numpy(e_n)).squeeze().numpy()
    return V


# Simulation
def reset(pos_offset=None):
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, mj_data)
    start   = X_REF[:3].copy()
    if pos_offset is not None:
        start = start - np.array(pos_offset)
    mj_data.qpos[:3] = start
    mj_data.qpos[3]  = 1.0; mj_data.qpos[4:7] = 0.0
    mujoco.mj_forward(mj_model, mj_data)
    return mj_data


def run_policy(policy, e_scale, pos_offset, duration=8.0):
    mj_data = reset(pos_offset)
    times, states, controls = [], [], []
    for _ in range(int(duration / DT_CTRL)):
        x = get_state(mj_data)
        u = policy_action(policy, e_scale, x)
        mj_data.ctrl[:] = np.clip(wrench_to_rotors(u), 0.005, 0.25)
        for _ in range(int(DT_CTRL / mj_model.opt.timestep)):
            mujoco.mj_step(mj_model, mj_data)
        times.append(len(times)*DT_CTRL)
        states.append(x.copy()); controls.append(u.copy())
    return np.array(times), np.array(states), np.array(controls)


def run_mpc(pos_offset, duration=8.0):
    mj_data = reset(pos_offset)
    mpc     = HybridMPC()
    times, states, controls, solve_ms = [], [], [], []
    for _ in range(int(duration / DT_CTRL)):
        x  = get_state(mj_data)
        t0 = time.time()
        u  = mpc.solve(x, X_REF)
        solve_ms.append((time.time()-t0)*1e3)
        mj_data.ctrl[:] = np.clip(wrench_to_rotors(u), 0.005, 0.25)
        for _ in range(int(DT_CTRL / mj_model.opt.timestep)):
            mujoco.mj_step(mj_model, mj_data)
        times.append(len(times)*DT_CTRL)
        states.append(x.copy()); controls.append(np.array(u).copy())
    return (np.array(times), np.array(states),
            np.array(controls), np.array(solve_ms))


# Stats
def print_stats(sc):
    def settle(err, times):
        w = int(0.5/DT_CTRL)
        for i in range(len(err)-w):
            if err[i:i+w].max() < 0.05: return times[i]
        return float("inf")

    err_p = np.linalg.norm(sc["s_pol"][:,:3] - X_REF[:3], axis=1)
    err_m = np.linalg.norm(sc["s_mpc"][:,:3] - X_REF[:3], axis=1)
    print(f"\n  {sc['label']}")
    print(f"    Policy — RMSE:{err_p.mean()*100:6.1f}cm  "
          f"peak:{err_p.max()*100:6.1f}cm  "
          f"settle:{settle(err_p, sc['t_pol']):.2f}s")
    print(f"    MPC    — RMSE:{err_m.mean()*100:6.1f}cm  "
          f"peak:{err_m.max()*100:6.1f}cm  "
          f"settle:{settle(err_m, sc['t_mpc']):.2f}s  "
          f"solve:{sc['ms_mpc'].mean():.0f}ms")


# Plot
def plot_comparison(scenarios, V_net, v_scale, alpha, c_roa):
    n_sc   = len(scenarios)
    colors = {"mpc": "#1D9E75", "policy": "#D85A30"}

    fig = plt.figure(figsize=(5*n_sc, 16))
    gs  = gridspec.GridSpec(4, n_sc, figure=fig, hspace=0.45, wspace=0.3)

    for col, sc in enumerate(scenarios):
        t_p=sc["t_pol"]; s_p=sc["s_pol"]; u_p=sc["u_pol"]
        t_m=sc["t_mpc"]; s_m=sc["s_mpc"]; u_m=sc["u_mpc"]

        # Row 0: position tracking
        ax = fig.add_subplot(gs[0, col])
        ax.set_title(sc["label"], fontsize=10, fontweight="500")
        for dim, lbl, c in zip([0,1,2], ["x","y","z"],
                                ["#378ADD","#EF9F27","#7F77DD"]):
            ax.plot(t_m, s_m[:,dim], color=c, lw=1.8,
                    label=f"{lbl} MPC" if col==0 else "")
            ax.plot(t_p, s_p[:,dim], color=c, lw=1.2, ls="--", alpha=0.7,
                    label=f"{lbl} policy" if col==0 else "")
            ax.axhline(X_REF[dim], color=c, lw=0.6, ls=":", alpha=0.4)
        if col==0:
            ax.set_ylabel("Position (m)", fontsize=9)
            ax.legend(fontsize=7, ncol=2, loc="upper right")
        ax.grid(alpha=0.25); ax.set_xlim(0, t_p[-1])

        # position error
        ax = fig.add_subplot(gs[1, col])
        err_m = np.linalg.norm(s_m[:,:3]-X_REF[:3], axis=1)*100
        err_p = np.linalg.norm(s_p[:,:3]-X_REF[:3], axis=1)*100
        ax.plot(t_m, err_m, color=colors["mpc"],    lw=1.8, label="MPC")
        ax.plot(t_p, err_p, color=colors["policy"], lw=1.5, ls="--",
                label="policy")
        ax.axhline(5.0, color="#888780", lw=1.0, ls="--",
                   label="5 cm threshold")
        ax.set_ylim(bottom=0)
        if col==0:
            ax.set_ylabel("Pos error (cm)", fontsize=9)
            ax.legend(fontsize=7)
        ax.grid(alpha=0.25); ax.set_xlim(0, t_p[-1])

        # thrust
        ax = fig.add_subplot(gs[2, col])
        ax.plot(t_m, u_m[:,0], color=colors["mpc"],    lw=1.8, label="MPC")
        ax.plot(t_p, u_p[:,0], color=colors["policy"], lw=1.5, ls="--",
                label="policy")
        ax.axhline(MASS*GRAV, color="#888780", lw=0.8, ls=":",
                   label=f"hover {MASS*GRAV:.3f}N")
        if col==0:
            ax.set_ylabel("Thrust T (N)", fontsize=9)
            ax.legend(fontsize=7)
        ax.grid(alpha=0.25); ax.set_xlim(0, t_p[-1])

        # Lyapunov V
        ax = fig.add_subplot(gs[3, col])
        if V_net is not None:
            V_m = eval_V(V_net, v_scale, s_m)
            V_p = eval_V(V_net, v_scale, s_p)
            ax.plot(t_m, V_m, color=colors["mpc"],    lw=1.8, label="MPC")
            ax.plot(t_p, V_p, color=colors["policy"], lw=1.5, ls="--",
                    label="policy")
            ax.axhline(c_roa, color="#1D9E75", lw=1.2, ls="--",
                       label=f"ROA  V={c_roa}")
            ax.fill_between(t_p, 0, c_roa, alpha=0.06, color="#1D9E75")

            # Mark first step policy exits ROA
            V_p_arr = np.array(V_p) if not isinstance(V_p, np.ndarray) else V_p
            exit_idx = np.where(V_p_arr > c_roa)[0]
            if len(exit_idx) > 0:
                t_exit = t_p[exit_idx[0]]
                ax.axvline(t_exit, color=colors["policy"],
                           lw=1.0, ls=":", alpha=0.7)
                ax.text(t_exit+0.05, ax.get_ylim()[1]*0.5 if ax.get_ylim()[1]>0 else 1,
                        f"exits ROA\nt={t_exit:.1f}s",
                        fontsize=7, color=colors["policy"])
            ax.set_yscale("log")
            if col==0:
                ax.set_ylabel("V(e)  [log scale]", fontsize=9)
                ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "V not available",
                    transform=ax.transAxes, ha="center", fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.grid(alpha=0.25); ax.set_xlim(0, t_p[-1])

    fig.suptitle("Residual policy vs Hybrid MPC — MuJoCo rollout",
                 fontsize=13, y=1.01)
    plt.savefig("models/policy_evaluation.png", dpi=130,
                bbox_inches="tight")
    plt.show()
    print("Saved → models/policy_evaluation.png")


# main
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, default=8.0)
    p.add_argument("--render",   action="store_true")
    args = p.parse_args()

    print("Loading policy...")
    policy, e_scale = load_policy()
    print(f"  e_scale range: [{e_scale.min():.3f}, {e_scale.max():.3f}]")

    # Quick equilibrium check
    with torch.no_grad():
        u0  = policy(torch.zeros(1, nx)).squeeze().numpy()
        err = abs(u0[0] - MASS*GRAV)
    print(f"  pi(0) thrust = {u0[0]:.4f}N  "
          f"(hover = {MASS*GRAV:.4f}N, err = {err:.4e})")

    print("\nLoading Lyapunov function...")
    V_net, v_scale, alpha, c_roa = load_lyapunov()

    scenarios_cfg = [
        ([0.0, 0.0, 0.0], "At hover (near eq.)"),
        ([0.3, 0.0, 0.0], "0.3m x offset"),
        ([0.0, 0.0, 0.4], "0.4m z offset"),
        ([0.4, 0.3, 0.3], "Large 3D offset"),
    ]

    print("\nRunning scenarios...")
    scenarios = []
    for offset, label in scenarios_cfg:
        print(f"  {label}...")
        t_p, s_p, u_p        = run_policy(policy, e_scale, offset, args.duration)
        t_m, s_m, u_m, ms_m = run_mpc(offset, args.duration)
        sc = dict(label=label, offset=offset,
                  t_pol=t_p, s_pol=s_p, u_pol=u_p,
                  t_mpc=t_m, s_mpc=s_m, u_mpc=u_m,
                  ms_mpc=ms_m)
        scenarios.append(sc)
        print_stats(sc)

    plot_comparison(scenarios, V_net, v_scale, alpha, c_roa)

    if args.render:
        print("\nLaunching MuJoCo viewer with residual policy...")
        mj_data = reset([0.3, 0.0, 0.0])
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            t = 0.0
            while viewer.is_running() and t < args.duration:
                x   = get_state(mj_data)
                u   = policy_action(policy, e_scale, x)
                err = np.linalg.norm(x[:3] - X_REF[:3])
                if int(t/DT_CTRL) % 25 == 0:
                    print(f"t={t:.1f}s | err={err*100:.1f}cm | T={u[0]:.3f}N")
                mj_data.ctrl[:] = np.clip(wrench_to_rotors(u), 0.005, 0.25)
                for _ in range(int(DT_CTRL / mj_model.opt.timestep)):
                    mujoco.mj_step(mj_model, mj_data)
                viewer.sync()
                t += DT_CTRL