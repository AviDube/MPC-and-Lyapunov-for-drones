"""
verify_lyapunov.py
──────────────────
Run from hybrid_mpc/:
    cd hybrid_mpc
    python verify_lyapunov.py

Produces four diagnostic figures:

  Figure 1 — V(e) level sets in 2D slices
    Left:   position error plane (ex, ez)
    Centre: velocity error plane (evx, evz)
    Right:  attitude error plane (e_roll, e_pitch)

  Figure 2 — V decrease along sampled trajectories
    Rolls out 12 closed-loop trajectories from random initial errors.
    V should decrease initially; trajectories that leave the certified
    region (V > c_ROA) are marked with a vertical grey line — growth
    beyond that point is expected and does NOT indicate certificate failure.

  Figure 3 — Violation map in position error plane
    Red: decrease condition violated
    Blue: positive definiteness violated
    Green: certified

  Figure 4 — Region of attraction estimate
    Largest level set {e: V(e) <= c} where decrease holds for >= 99%
    of sampled states.

Saved to models/.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mujoco

XML_PATH = "../basic_quadrotor.xml"
DT_CTRL  = 0.02
nx, nu   = 12, 4

mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
MASS     = float(mj_model.body_mass[1])
GRAV     = float(abs(mj_model.opt.gravity[2]))
INERTIA  = tuple(mj_model.body_inertia[1])
Ix,Iy,Iz = INERTIA

os.makedirs("models", exist_ok=True)


# ── model definitions ─────────────────────────────────────────────────────────
class NeuralPolicy(nn.Module):
    def __init__(self, nx=12, hidden=128, n_layers=3):
        super().__init__()
        layers=[nn.Linear(nx,hidden),nn.Tanh()]
        for _ in range(n_layers-1): layers+=[nn.Linear(hidden,hidden),nn.Tanh()]
        layers+=[nn.Linear(hidden,nu,bias=False)]
        self.delta_net=nn.Sequential(*layers)
        self.register_buffer("u_star",
            torch.tensor([MASS*GRAV,0.,0.,0.]))
        self.register_buffer("delta_max",
            torch.tensor([MASS*GRAV,0.005,0.005,0.02]))
    def forward(self,e):
        return self.u_star+self.delta_max*torch.tanh(self.delta_net(e))


class HybridResidualNN(nn.Module):
    def __init__(self,nx=12,nu=4,hidden=64,n_layers=3):
        super().__init__()
        layers=[nn.Linear(nx+nu,hidden),nn.SiLU()]
        for _ in range(n_layers-1): layers+=[nn.Linear(hidden,hidden),nn.SiLU()]
        layers+=[nn.Linear(hidden,nx)]
        self.net=nn.Sequential(*layers)
    def forward(self,xu): return self.net(xu)


class LyapunovNet(nn.Module):
    def __init__(self,nx=12,hidden=128,feat_dim=64):
        super().__init__()
        self.phi=nn.Sequential(
            nn.Linear(nx,hidden),nn.Tanh(),
            nn.Linear(hidden,hidden),nn.Tanh(),
            nn.Linear(hidden,feat_dim,bias=False))
        self.W=nn.Parameter(torch.eye(feat_dim)*0.1)
    def forward(self,e):
        feat=self.phi(e); Wf=feat@self.W.T
        return (Wf**2).sum(dim=1,keepdim=True)


# ── physics ───────────────────────────────────────────────────────────────────
def _R(phi,theta,psi):
    cp=torch.cos(phi);sp=torch.sin(phi);ct=torch.cos(theta)
    cy=torch.cos(psi);sy=torch.sin(psi);st=torch.sin(theta)
    B=phi.shape[0];R=torch.zeros(B,3,3,device=phi.device)
    R[:,0,0]=cy*ct;R[:,0,1]=cy*st*sp-sy*cp;R[:,0,2]=cy*st*cp+sy*sp
    R[:,1,0]=sy*ct;R[:,1,1]=sy*st*sp+cy*cp;R[:,1,2]=sy*st*cp-cy*sp
    R[:,2,0]=-st;R[:,2,1]=ct*sp;R[:,2,2]=ct*cp; return R

def _Wm(phi,theta):
    cp=torch.cos(phi);sp=torch.sin(phi);ct=torch.cos(theta);tt=torch.tan(theta)
    B=phi.shape[0];W=torch.zeros(B,3,3,device=phi.device)
    W[:,0,0]=1;W[:,0,1]=sp*tt;W[:,0,2]=cp*tt
    W[:,1,1]=cp;W[:,1,2]=-sp;W[:,2,1]=sp/ct;W[:,2,2]=cp/ct; return W

def physics_step(x_abs,u,dt=DT_CTRL):
    def ode(x,u):
        phi=x[:,3];theta=x[:,4];psi=x[:,5]
        p=x[:,9];q=x[:,10];r=x[:,11]
        T=u[:,0];tx=u[:,1];ty=u[:,2];tz=u[:,3]
        Rm=_R(phi,theta,psi)
        tb=torch.stack([torch.zeros_like(T),torch.zeros_like(T),T/MASS],1)
        gw=torch.zeros_like(x[:,6:9]);gw[:,2]=-GRAV
        acc=(Rm@tb.unsqueeze(-1)).squeeze(-1)+gw
        omega=x[:,9:12];Wmat=_Wm(phi,theta)
        eta_dot=(Wmat@omega.unsqueeze(-1)).squeeze(-1)
        Io=torch.stack([Ix*p,Iy*q,Iz*r],1)
        gyro=torch.linalg.cross(omega,Io);tau=u[:,1:4]
        od=torch.stack([(tau[:,0]-gyro[:,0])/Ix,
                        (tau[:,1]-gyro[:,1])/Iy,
                        (tau[:,2]-gyro[:,2])/Iz],1)
        return torch.cat([x[:,6:9],eta_dot,acc,od],1)
    k1=ode(x_abs,u);k2=ode(x_abs+dt/2*k1,u)
    k3=ode(x_abs+dt/2*k2,u);k4=ode(x_abs+dt*k3,u)
    return x_abs+(dt/6)*(k1+2*k2+2*k3+k4)

def make_cl_step(policy,nn_res,xu_mean,xu_std,x_ref_t,e_scale_t,delta_eq):
    def cl_step(e):
        B=e.shape[0]; xrb=x_ref_t.expand(B,-1)
        x_abs=e*e_scale_t+xrb; u=policy(e)
        xp=physics_step(x_abs,u)
        xu_n=(torch.cat([x_abs,u],1)-xu_mean)/xu_std
        delta=nn_res(xu_n)
        xn=xp+delta-delta_eq
        return (xn-xrb)/e_scale_t
    return cl_step


# ── load ──────────────────────────────────────────────────────────────────────
def load_all(device):
    cfg     = np.load("models/policy_config.npz")
    e_scale = torch.from_numpy(cfg["e_scale"]).float().to(device)
    X_REF   = cfg["x_ref"].astype(np.float32)
    x_ref_t = torch.from_numpy(X_REF).float().to(device).unsqueeze(0)

    policy  = NeuralPolicy().to(device)
    policy.load_state_dict(torch.load("models/neural_policy.pt",
                                      map_location=device))
    policy.eval()

    cfg2    = np.load("models/hybrid_config.npz")
    nn_res  = HybridResidualNN(hidden=int(cfg2["hidden"][0]),
                                n_layers=int(cfg2["n_layers"][0])).to(device)
    nn_res.load_state_dict(torch.load("models/hybrid_nn.pt",
                                      map_location=device))
    nn_res.eval()
    xu_mean = torch.from_numpy(cfg2["xu_mean"].astype(np.float32)).to(device)
    xu_std  = torch.from_numpy(cfg2["xu_std"].astype(np.float32)).to(device)

    lyap    = np.load("models/lyapunov_config.npz")
    delta_eq= torch.from_numpy(lyap["delta_eq"].astype(np.float32)).to(device)
    alpha   = float(lyap["alpha"][0])
    feat_dim= int(lyap["feat_dim"][0])
    hidden  = int(lyap["hidden"][0])

    V_net   = LyapunovNet(nx=nx,hidden=hidden,feat_dim=feat_dim).to(device)
    V_net.load_state_dict(torch.load("models/lyapunov_net.pt",
                                     map_location=device))
    V_net.eval()

    cl_step = make_cl_step(policy,nn_res,xu_mean,xu_std,
                           x_ref_t,e_scale,delta_eq)
    return V_net, cl_step, alpha, e_scale


# ── Figure 1: level sets ───────────────────────────────────────────────────────
def fig_level_sets(V_net, cl_step, alpha, device):
    N = 100
    slices = [
        ("pos (ex, ez)",         0, 2, 0.6),
        ("vel (evx, evz)",       6, 8, 0.5),
        ("att (eroll, epitch)",  3, 4, 0.3),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (title, i, j, rng) in zip(axes, slices):
        g = np.linspace(-rng, rng, N)
        G1, G2 = np.meshgrid(g, g)
        E = np.zeros((N*N, nx), dtype=np.float32)
        E[:,i] = G1.ravel(); E[:,j] = G2.ravel()
        Et = torch.from_numpy(E).to(device)

        with torch.no_grad():
            V  = V_net(Et).squeeze().cpu().numpy().reshape(N, N)
            En = cl_step(Et)
            Vn = V_net(En).squeeze().cpu().numpy().reshape(N, N)

        dec_ok = (Vn <= (1-alpha)*V).astype(float)
        cf = ax.contourf(G1, G2, V, levels=20, cmap="viridis", alpha=0.85)
        ax.contour(G1, G2, V, levels=[0.05,0.2,0.5,1.0,2.0],
                   colors="white", linewidths=0.7, alpha=0.6)
        ax.contourf(G1, G2, 1-dec_ok, levels=[0.5,1.5],
                    colors=["#E24B4A"], alpha=0.25)
        plt.colorbar(cf, ax=ax, shrink=0.85)
        ax.set_xlabel(f"dim {i}"); ax.set_ylabel(f"dim {j}")
        ax.set_title(f"V(e) — {title}", fontsize=9)
        ax.plot(0, 0, "r*", ms=10, label="equilibrium")
        ax.legend(fontsize=7)

    fig.suptitle(
        "Lyapunov function level sets  (red overlay = decrease violated)",
        fontsize=11)
    plt.tight_layout()
    plt.savefig("models/lyapunov_level_sets.png", dpi=130, bbox_inches="tight")
    plt.show()
    print("Saved → models/lyapunov_level_sets.png")


# ── Figure 2: V decrease along trajectories ────────────────────────────────────
def fig_decrease(V_net, cl_step, alpha, device,
                 n_traj=12, T_steps=80, c_roa=5.0):
    """
    Rolls out n_traj trajectories and plots V(e_k) on a log scale.

    Key addition: a vertical grey line is drawn at the step where each
    trajectory first exits the certified region {V <= c_roa}.  Growth
    after that line is expected — the certificate only applies inside the
    ROA, not outside it.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))
    rng     = np.random.default_rng(42)
    colors  = plt.cm.tab20(np.linspace(0, 1, n_traj))

    exit_steps = []

    for i in range(n_traj):
        e = torch.from_numpy(
            rng.uniform(-0.5, 0.5, (1, nx)).astype(np.float32)).to(device)
        V_traj = []
        exit_k = None

        with torch.no_grad():
            for k in range(T_steps):
                v = V_net(e).item()
                V_traj.append(v)
                if exit_k is None and v > c_roa:
                    exit_k = k
                e = cl_step(e)

        exit_steps.append(exit_k)
        ax.plot(V_traj, color=colors[i], alpha=0.7, lw=1.3)

        # Mark exit from certified region
        if exit_k is not None:
            ax.axvline(exit_k, color=colors[i], lw=0.8,
                       ls=":", alpha=0.5)

    # Theoretical decay envelope anchored at median V(e_0)
    t_arr    = np.arange(T_steps)
    ax.plot(c_roa * (1-alpha)**t_arr, "k--", lw=2.0,
            label=f"$(1-\\alpha)^t \\cdot c_{{ROA}}$  α={alpha}")

    # Shade the certified region
    ax.axhspan(0, c_roa, alpha=0.04, color="#1D9E75")
    ax.axhline(c_roa, color="#1D9E75", lw=1.2, ls="--",
               label=f"ROA boundary  V={c_roa}")

    # Annotation explaining the vertical lines
    ax.text(0.98, 0.97,
            "dotted vertical = trajectory exits ROA\n"
            "growth beyond this is expected",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="#555555",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7,
                      ec="#cccccc"))

    ax.set_yscale("log")
    ax.set_xlabel("Step k", fontsize=11)
    ax.set_ylabel("V(e_k)", fontsize=11)
    ax.set_title("V decrease along closed-loop trajectories\n"
                 "Certificate valid only inside shaded region (V ≤ ROA boundary)",
                 fontsize=10)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, T_steps-1)

    plt.tight_layout()
    plt.savefig("models/lyapunov_decrease.png", dpi=130, bbox_inches="tight")
    plt.show()
    print("Saved → models/lyapunov_decrease.png")

    n_exit = sum(1 for s in exit_steps if s is not None)
    print(f"  {n_exit}/{n_traj} trajectories left the ROA within {T_steps} steps")
    if n_exit > 0:
        mean_exit = np.mean([s for s in exit_steps if s is not None])
        print(f"  Mean exit step: {mean_exit:.1f}  "
              f"({mean_exit*DT_CTRL:.2f}s)")


# ── Figure 3: violation map ────────────────────────────────────────────────────
def fig_violation_map(V_net, cl_step, alpha, device):
    N   = 120
    rng = np.linspace(-0.6, 0.6, N)
    EX, EZ = np.meshgrid(rng, rng)

    E = np.zeros((N*N, nx), dtype=np.float32)
    E[:,0] = EX.ravel(); E[:,2] = EZ.ravel()
    Et = torch.from_numpy(E).to(device)

    with torch.no_grad():
        Ve  = V_net(Et).squeeze().cpu().numpy().reshape(N, N)
        En  = cl_step(Et)
        Ven = V_net(En).squeeze().cpu().numpy().reshape(N, N)

    norm     = np.sqrt(EX**2 + EZ**2)
    viol_dec = Ven > (1-alpha)*Ve
    viol_pd  = (norm > 0.02) & (Ve < 1e-3)
    certified= ~viol_dec & ~viol_pd

    cmap   = mcolors.ListedColormap(["#E24B4A","#E1F5EE","#378ADD"])
    status = np.where(viol_dec, 0, np.where(certified, 1, 2)).astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.pcolormesh(EX, EZ, status, cmap=cmap, vmin=0, vmax=2)
    ax.contour(EX, EZ, Ve, levels=[0.05,0.2,0.5,1.0,2.0,5.0],
               colors="white", linewidths=0.6, alpha=0.7)
    ax.plot(0, 0, "k*", ms=10, label="equilibrium")
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor="#E24B4A", label="decrease violated"),
                  Patch(facecolor="#E1F5EE", label="certified"),
                  Patch(facecolor="#378ADD", label="PD violated")]
    ax.legend(handles=legend_els, fontsize=8, loc="upper right")
    ax.set_xlabel("x error (m)"); ax.set_ylabel("z error (m)")
    ax.set_title("Certification status", fontsize=10)

    ax = axes[1]
    cf = ax.contourf(EX, EZ, Ve, levels=20, cmap="plasma")
    ax.contour(EX, EZ, Ve, levels=[0.05,0.1,0.3,0.5,1.0,2.0,5.0],
               colors="white", linewidths=0.6, alpha=0.6)
    plt.colorbar(cf, ax=ax, shrink=0.85)
    ax.plot(0, 0, "r*", ms=10)
    ax.set_xlabel("x error (m)"); ax.set_ylabel("z error (m)")
    ax.set_title("V(e) — position error plane", fontsize=10)

    pct_viol = viol_dec.mean() * 100
    ax.text(0.02, 0.04,
            f"Decrease violations: {pct_viol:.2f}% of grid",
            transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig("models/lyapunov_violation_map.png", dpi=130,
                bbox_inches="tight")
    plt.show()
    print("Saved → models/lyapunov_violation_map.png")


# ── Stratified sampler ────────────────────────────────────────────────────────
def stratified_sample(shells, nx, device):
    """
    Sample states in concentric shells of increasing radius so that
    every level set — including tiny inner ones — gets adequate coverage.

    shells: list of (r_min, r_max, n_samples)
    Strategy: uniform random direction on the unit sphere, scaled to
    a radius drawn uniformly in [r_min, r_max].
    """
    all_e = []
    for r_min, r_max, n in shells:
        z = torch.randn(n, nx)
        z = z / z.norm(dim=1, keepdim=True).clamp(min=1e-8)
        r = torch.rand(n, 1) * (r_max - r_min) + r_min
        all_e.append((z * r).to(device))
    return torch.cat(all_e, dim=0)


# ── Figure 4: ROA estimate ─────────────────────────────────────────────────────
def fig_roa(V_net, cl_step, alpha, device):
    print("\n── Region of Attraction estimate (stratified sampling) ──")

    # Dense near origin, sparser at large radii — ensures every level set
    # gets thousands of points regardless of how small it is in V-space.
    shells = [
        (0.00, 0.05,  30_000),
        (0.05, 0.15,  40_000),
        (0.15, 0.30,  50_000),
        (0.30, 0.50,  60_000),
        (0.50, 0.70,  70_000),
        (0.70, 0.90,  80_000),
        (0.90, 1.10,  90_000),
        (1.10, 1.40, 100_000),
        (1.40, 2.00, 100_000),
    ]
    e  = stratified_sample(shells, nx, device)
    nm = e.norm(dim=1)
    print(f"  Total samples: {len(e):,}")

    with torch.no_grad():
        Ve  = V_net(e).squeeze()
        En  = cl_step(e)
        Ven = V_net(En).squeeze()

    dec_ok = Ven <= (1-alpha)*Ve
    pd_ok  = (nm <= 0.02) | (Ve > 1e-3)

    levels = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    rows   = []
    for c in levels:
        in_ls = Ve <= c
        n_in  = in_ls.sum().item()
        if n_in == 0:
            continue
        cert = (dec_ok & pd_ok)[in_ls].float().mean().item()
        rows.append((c, n_in, cert*100))
        print(f"  V <= {c:.2f}:  {n_in:7,} states,  "
              f"certified {cert*100:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    cs    = [r[0] for r in rows]
    ns    = [r[1] for r in rows]
    certs = [r[2] for r in rows]

    axes[0].bar([str(c) for c in cs], ns, color="#378ADD", alpha=0.8)
    axes[0].set_xlabel("Level set c  (V ≤ c)")
    axes[0].set_ylabel("Stratified samples in level set")
    axes[0].set_title("Coverage of each level set\n"
                      "(stratified — inner shells oversampled)", fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)
    for bar, n_in in zip(axes[0].patches, ns):
        axes[0].text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()*1.01,
                     f"{n_in:,}", ha="center", va="bottom", fontsize=7)

    colors_bar = ["#1D9E75" if v >= 99 else "#EF9F27" if v >= 90
                  else "#E24B4A" for v in certs]
    axes[1].bar([str(c) for c in cs], certs, color=colors_bar, alpha=0.85)
    axes[1].axhline(99, ls="--", color="#1D9E75", lw=1.2,
                    label="99% threshold")
    axes[1].set_xlabel("Level set c  (V ≤ c)")
    axes[1].set_ylabel("Certified (%)")
    axes[1].set_title("Certificate coverage per level set\n"
                      "(green ≥ 99%, orange ≥ 90%, red < 90%)", fontsize=9)
    axes[1].set_ylim(0, 108)
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)

    best_c = next((c for c, _, cert in rows if cert >= 99.0), None)
    if best_c is not None:
        idx = [str(c) for c in cs].index(str(best_c))
        axes[1].patches[idx].set_edgecolor("#0F6E56")
        axes[1].patches[idx].set_linewidth(2.5)
        axes[1].text(idx, certs[idx] + 1.5,
                     f"ROA ≈ V≤{best_c}",
                     ha="center", fontsize=8,
                     color="#0F6E56", fontweight="bold")

    plt.tight_layout()
    plt.savefig("models/lyapunov_roa.png", dpi=130, bbox_inches="tight")
    plt.show()
    print("Saved → models/lyapunov_roa.png")


# ── Summary ───────────────────────────────────────────────────────────────────
def print_summary(V_net, cl_step, alpha, device):
    print("\n── Certificate summary (stratified samples) ──")

    shells = [
        (0.00, 0.20,  60_000),
        (0.20, 0.60,  80_000),
        (0.60, 1.00, 100_000),
        (1.00, 1.50, 120_000),
        (1.50, 2.00, 140_000),
    ]
    e  = stratified_sample(shells, nx, device)
    nm = e.norm(dim=1)

    with torch.no_grad():
        Ve  = V_net(e).squeeze()
        En  = cl_step(e)
        Ven = V_net(En).squeeze()

    mask   = nm > 0.02
    dec_ok = Ven <= (1-alpha)*Ve

    print(f"  Positive definiteness:  "
          f"{(Ve[mask] > 1e-3).float().mean().item()*100:.1f}% satisfied")
    print(f"  Decrease condition:     "
          f"{dec_ok.float().mean().item()*100:.1f}% satisfied")
    print(f"  Joint certificate:      "
          f"{(dec_ok & ((~mask) | (Ve > 1e-3))).float().mean().item()*100:.1f}%")
    print(f"  V range:  [{Ve.min().item():.3e}, {Ve.max().item():.3e}]")
    print(f"  Decay rate alpha = {alpha}  "
          f"→  half-life ≈ {np.log(0.5)/np.log(1-alpha):.0f} steps  "
          f"({np.log(0.5)/np.log(1-alpha)*DT_CTRL:.2f}s)")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--c_roa", type=float, default=5.0,
                   help="ROA boundary value for decrease plot (default: 5.0)")
    args = p.parse_args()

    device = torch.device("cpu")
    V_net, cl_step, alpha, e_scale = load_all(device)

    print(f"Loaded certificate | alpha={alpha}")
    print(f"V(0) = {V_net(torch.zeros(1,nx)).item():.2e}")

    print_summary(V_net, cl_step, alpha, device)
    fig_level_sets(V_net, cl_step, alpha, device)
    fig_decrease(V_net, cl_step, alpha, device, c_roa=args.c_roa)
    fig_violation_map(V_net, cl_step, alpha, device)
    fig_roa(V_net, cl_step, alpha, device)

    print("\nAll figures saved to models/")