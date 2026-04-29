"""
Distillation of a residual policy + Lyapunov function from the hybrid MPC controller.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import mujoco

from mpc_hybrid import HybridMPC, get_state, wrench_to_rotors

XML_PATH = "../basic_quadrotor.xml"
DT_CTRL  = 0.02
nx, nu   = 12, 4

mj_model  = mujoco.MjModel.from_xml_path(XML_PATH)
mj_data   = mujoco.MjData(mj_model)
MASS      = float(mj_model.body_mass[1])
GRAV      = float(abs(mj_model.opt.gravity[2]))
INERTIA   = tuple(mj_model.body_inertia[1])
Ix,Iy,Iz  = INERTIA

X_REF  = np.array([0.5,-0.5,1.0, 0,0,0, 0,0,0, 0,0,0], dtype=np.float32)
U_STAR = np.array([MASS*GRAV, 0.0, 0.0, 0.0], dtype=np.float32)

os.makedirs("models", exist_ok=True)

_x_ref = torch.from_numpy(X_REF)

class HybridResidualNN(nn.Module):
    def __init__(self, nx=12, nu=4, hidden=64, n_layers=3):
        super().__init__()
        layers = [nn.Linear(nx+nu, hidden), nn.SiLU()]
        for _ in range(n_layers-1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, nx)]
        self.net = nn.Sequential(*layers)
    def forward(self, xu): return self.net(xu)


def _rot(phi, theta, psi):
    cp=torch.cos(phi); sp=torch.sin(phi)
    ct=torch.cos(theta); st=torch.sin(theta)
    cy=torch.cos(psi);  sy=torch.sin(psi)
    B=phi.shape[0]; R=torch.zeros(B,3,3,device=phi.device)
    R[:,0,0]=cy*ct;  R[:,0,1]=cy*st*sp-sy*cp; R[:,0,2]=cy*st*cp+sy*sp
    R[:,1,0]=sy*ct;  R[:,1,1]=sy*st*sp+cy*cp; R[:,1,2]=sy*st*cp-cy*sp
    R[:,2,0]=-st;    R[:,2,1]=ct*sp;           R[:,2,2]=ct*cp
    return R

def _wmat(phi, theta):
    cp=torch.cos(phi); sp=torch.sin(phi)
    ct=torch.cos(theta); tt=torch.tan(theta)
    B=phi.shape[0]; W=torch.zeros(B,3,3,device=phi.device)
    W[:,0,0]=1; W[:,0,1]=sp*tt; W[:,0,2]=cp*tt
    W[:,1,1]=cp; W[:,1,2]=-sp
    W[:,2,1]=sp/ct; W[:,2,2]=cp/ct
    return W

def physics_step(x_abs, u, dt=DT_CTRL):
    def ode(x, u):
        phi=x[:,3]; theta=x[:,4]; psi=x[:,5]
        p=x[:,9];   q=x[:,10];   r=x[:,11]
        T=u[:,0]; tx=u[:,1]; ty=u[:,2]; tz=u[:,3]
        R  = _rot(phi, theta, psi)
        tb = torch.stack([torch.zeros_like(T), torch.zeros_like(T), T/MASS], 1)
        gw = torch.zeros_like(x[:,6:9]); gw[:,2] = -GRAV
        acc = (R @ tb.unsqueeze(-1)).squeeze(-1) + gw
        om  = x[:,9:12]; Wm = _wmat(phi, theta)
        eta_dot = (Wm @ om.unsqueeze(-1)).squeeze(-1)
        Io  = torch.stack([Ix*p, Iy*q, Iz*r], 1)
        gyro= torch.linalg.cross(om, Io); tau = u[:,1:4]
        od  = torch.stack([(tau[:,0]-gyro[:,0])/Ix,
                           (tau[:,1]-gyro[:,1])/Iy,
                           (tau[:,2]-gyro[:,2])/Iz], 1)
        return torch.cat([x[:,6:9], eta_dot, acc, od], 1)
    k1=ode(x_abs,u); k2=ode(x_abs+dt/2*k1,u)
    k3=ode(x_abs+dt/2*k2,u); k4=ode(x_abs+dt*k3,u)
    return x_abs + (dt/6)*(k1+2*k2+2*k3+k4)


def load_hybrid_nn(device):
    cfg     = np.load("models/hybrid_config.npz")
    nn_res  = HybridResidualNN(hidden=int(cfg["hidden"][0]),
                                n_layers=int(cfg["n_layers"][0])).to(device)
    nn_res.load_state_dict(torch.load("models/hybrid_nn.pt", map_location=device))
    nn_res.eval()
    xu_mean = torch.from_numpy(cfg["xu_mean"].astype(np.float32)).to(device)
    xu_std  = torch.from_numpy(cfg["xu_std"].astype(np.float32)).to(device)

    # Equilibrium correction: delta* = f_theta(x*, u*)
    # Subtract this so f_hybrid(x*, u*) = x* exactly
    x_star = torch.from_numpy(X_REF).float().to(device).unsqueeze(0)
    u_star = torch.from_numpy(U_STAR).float().to(device).unsqueeze(0)
    xu_eq  = (torch.cat([x_star, u_star], 1) - xu_mean) / xu_std
    with torch.no_grad():
        delta_eq = nn_res(xu_eq)   # (1, 12)
    print(f"  Hybrid NN loaded  |  "
          f"delta* max: {delta_eq.abs().max().item():.4e}")
    return nn_res, xu_mean, xu_std, delta_eq


def hybrid_next(x_abs, u, nn_res, xu_mean, xu_std, delta_eq):
    x_phys  = physics_step(x_abs, u)
    xu_n    = (torch.cat([x_abs, u], 1) - xu_mean) / xu_std
    delta   = nn_res(xu_n)
    return x_phys + delta - delta_eq


class ResidualPolicy(nn.Module):
    def __init__(self, nx=12, hidden=128, n_layers=3):
        super().__init__()
        layers = [nn.Linear(nx, hidden), nn.Tanh()]
        for _ in range(n_layers-1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, nu, bias=False)]
        self.delta_net = nn.Sequential(*layers)
        nn.init.zeros_(self.delta_net[-1].weight)
        self.register_buffer("u_star",    torch.from_numpy(U_STAR))
        self.register_buffer("delta_max", torch.tensor([MASS*GRAV,0.005,0.005,0.02]))

    def forward(self, e):
        return self.u_star + self.delta_max * torch.tanh(self.delta_net(e))


class LyapunovFunction(nn.Module):

    def __init__(self, nx=12, hidden=128, feat_dim=64):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(nx, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, feat_dim, bias=False),  # phi(0)=0
        )
        self.W = nn.Parameter(torch.eye(feat_dim) * 0.1)

    def forward(self, e):
        feat = self.phi(e)
        Wf   = feat @ self.W.T
        return (Wf**2).sum(dim=1, keepdim=True)   # (B,1), >=0, =0 at e=0



def collect(n_episodes=15):
    offsets = [
        [0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [-0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0], [0.0,-0.5, 0.0], [0.0,  0.0, 0.5],
        [0.0, 0.0,-0.4], [0.4, 0.4, 0.3], [-0.3, 0.3,-0.2],
    ]
    E_buf, U_buf = [], []
    for ep in range(n_episodes):
        mujoco.mj_resetData(mj_model, mj_data)
        offset = offsets[ep % len(offsets)]
        mj_data.qpos[:3]  = X_REF[:3] - np.array(offset)
        mj_data.qpos[3]   = 1.0; mj_data.qpos[4:7] = 0.0
        mujoco.mj_forward(mj_model, mj_data)

        mpc = HybridMPC()
        for _ in range(int(8.0/DT_CTRL)):
            x = get_state(mj_data)
            u = mpc.solve(x, X_REF)
            if isinstance(u, tuple): u = u[0]
            E_buf.append((x - X_REF).astype(np.float32))
            U_buf.append(np.asarray(u, dtype=np.float32))
            u_rot = np.clip(wrench_to_rotors(u), 0.005, 0.25)
            mj_data.ctrl[:] = u_rot
            for _ in range(int(DT_CTRL / mj_model.opt.timestep)):
                mujoco.mj_step(mj_model, mj_data)
        print(f"  Episode {ep+1}/{n_episodes}: {len(E_buf)} transitions")

    # Synthetic equilibrium samples
    rng  = np.random.default_rng(0)
    E_eq = rng.standard_normal((1000, nx)).astype(np.float32) * 0.01
    U_eq = np.tile(U_STAR, (1000, 1))
    E_all = np.concatenate([np.array(E_buf), E_eq])
    U_all = np.concatenate([np.array(U_buf), U_eq])
    print(f"\nTotal: {len(E_all)} (e, u) pairs")
    return E_all, U_all


def train(n_episodes=15, epochs=300, lr=3e-4, hidden=128, n_layers=3,
          alpha=0.05, lambda_lyap=5.0, lambda_pd=1.0, eps_pd=0.01):

    print(f"Collecting {n_episodes} episodes...")
    E, U = collect(n_episodes)

    e_scale = np.clip(np.abs(E).max(0), 1e-3, None).astype(np.float32)
    E_n     = (E / e_scale).astype(np.float32)

    np.savez("models/policy_lyap_config.npz",
             x_ref=X_REF, u_star=U_STAR, e_scale=e_scale)

    dataset  = TensorDataset(torch.from_numpy(E_n), torch.from_numpy(U))
    n_val    = max(1, int(0.1*len(dataset)))
    train_ds, val_ds = random_split(dataset, [len(dataset)-n_val, n_val],
                                    generator=torch.Generator().manual_seed(0))
    train_ld = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_ld   = DataLoader(val_ds,   batch_size=512)

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy     = ResidualPolicy(nx=nx, hidden=hidden, n_layers=n_layers).to(device)
    lyap       = LyapunovFunction(nx=nx, hidden=hidden, feat_dim=64).to(device)

    print("Loading hybrid dynamics model...")
    nn_res, xu_mean, xu_std, delta_eq = load_hybrid_nn(device)

    x_ref_d   = _x_ref.to(device)
    e_scale_d = torch.from_numpy(e_scale).to(device)

    opt   = torch.optim.AdamW(
        list(policy.parameters()) + list(lyap.parameters()),
        lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=1e-5)

    print(f"\nResidualPolicy: {sum(p.numel() for p in policy.parameters())} params")
    print(f"LyapunovFunction: {sum(p.numel() for p in lyap.parameters())} params")
    print(f"alpha={alpha}  lambda_lyap={lambda_lyap}  device={device}")

    with torch.no_grad():
        v0 = lyap(torch.zeros(1,nx,device=device)).item()
    print(f"V(0) = {v0:.2e}  [{'PASS' if v0<1e-8 else 'FAIL'}]\n")

    best_val = float("inf")
    hist = {"imitate":[], "decrease":[], "pd":[], "val":[]}

    for epoch in range(1, epochs+1):
        policy.train(); lyap.train()
        tl_im = tl_dec = tl_pd = 0.0

        for e_n, u_mpc in train_ld:
            e_n, u_mpc = e_n.to(device), u_mpc.to(device)
            u_pred = policy(e_n)

            # Imitation loss
            L_imitate = nn.functional.mse_loss(u_pred, u_mpc)

            x_abs    = e_n * e_scale_d + x_ref_d
            x_next   = hybrid_next(x_abs, u_pred, nn_res, xu_mean, xu_std, delta_eq)
            e_next_n = (x_next - x_ref_d) / e_scale_d

            # Lyapunov decrease loss
            Ve  = lyap(e_n).squeeze()
            Ven = lyap(e_next_n).squeeze()
            L_decrease = torch.relu(Ven - (1-alpha)*Ve).mean()

            # Positive definiteness loss
            norm_e = e_n.norm(dim=1)
            mask   = (norm_e > 0.02).float()
            L_pd   = (mask * torch.relu(eps_pd - Ve)).mean()

            loss = L_imitate + lambda_lyap*L_decrease + lambda_pd*L_pd
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(policy.parameters())+list(lyap.parameters()), 1.0)
            opt.step()

            tl_im  += L_imitate.item()
            tl_dec += L_decrease.item()
            tl_pd  += L_pd.item()

        n_b = len(train_ld)
        hist["imitate"].append(tl_im/n_b)
        hist["decrease"].append(tl_dec/n_b)
        hist["pd"].append(tl_pd/n_b)

        policy.eval(); lyap.eval()
        vl = 0.0
        with torch.no_grad():
            for e_n, u_mpc in val_ld:
                vl += nn.functional.mse_loss(
                    policy(e_n.to(device)), u_mpc.to(device)).item()
        vl /= len(val_ld); sched.step()
        hist["val"].append(vl)

        if vl < best_val:
            best_val = vl
            torch.save(policy.state_dict(), "models/residual_policy.pt")
            torch.save(lyap.state_dict(),   "models/lyapunov_function.pt")

        if epoch % 50 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}/{epochs}  "
                  f"imitate={tl_im/n_b:.3e}  "
                  f"decrease={tl_dec/n_b:.3e}  "
                  f"pd={tl_pd/n_b:.3e}  "
                  f"val={vl:.3e}")

    print(f"\nBest val: {best_val:.4e}")
    print("Saved → models/residual_policy.pt  models/lyapunov_function.pt")

    # Post-training checks
    policy.load_state_dict(torch.load("models/residual_policy.pt",
                                      map_location=device))
    lyap.load_state_dict(torch.load("models/lyapunov_function.pt",
                                    map_location=device))
    policy.to(device).eval(); lyap.to(device).eval()

    with torch.no_grad():
        eq_err = (policy(torch.zeros(1,nx,device=device)).squeeze()
                  - policy.u_star).abs().max().item()
        v0     = lyap(torch.zeros(1,nx,device=device)).item()
    print(f"\nEquilibrium check: max|pi(0)-u*| = {eq_err:.2e}  "
          f"[{'PASS' if eq_err<5e-3 else 'FAIL'}]")
    print(f"V(0) = {v0:.2e}  [{'PASS' if v0<1e-6 else 'FAIL'}]")

    # Training curves
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    for ax, key, title in zip(axes,
        ["imitate","decrease","pd","val"],
        ["Imitation loss","Decrease loss","PD loss","Val imitation"]):
        ax.plot(hist[key]); ax.set_yscale("log")
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.grid(alpha=0.3)
    fig.suptitle("Joint policy + Lyapunov training", fontsize=12)
    plt.tight_layout()
    plt.savefig("models/joint_training.png", dpi=120)
    plt.show()
    print("Saved → models/joint_training.png")

    # Lyapunov scatter
    with torch.no_grad():
        e_all  = torch.from_numpy(E_n).to(device)
        u_all  = policy(e_all)
        x_abs  = e_all * e_scale_d + x_ref_d
        x_next = hybrid_next(x_abs, u_all, nn_res, xu_mean, xu_std, delta_eq)
        e_next = (x_next - x_ref_d) / e_scale_d
        Vk  = lyap(e_all).squeeze().cpu().numpy()
        Vk1 = lyap(e_next).squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.scatter(Vk, Vk1, alpha=0.15, s=3, c="#378ADD")
    lim = max(Vk.max(), Vk1.max()) * 1.05
    ax.plot([0,lim],[0,lim],         "k--", lw=1, label="V_next = V")
    ax.plot([0,lim],[0,lim*(1-alpha)],"g--", lw=1,
            label=f"V_next = (1-α)V  α={alpha}")
    ax.set_xlabel("V(e_k)"); ax.set_ylabel("V(e_k+1)")
    ax.set_title("Lyapunov scatter\n(below green = certified)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ratio = Vk1 / (Vk + 1e-8)
    pct   = (ratio <= 1-alpha).mean() * 100
    ax.hist(ratio, bins=50, color="#1D9E75", alpha=0.8)
    ax.axvline(1-alpha, color="r", lw=1.5, ls="--",
               label=f"threshold {1-alpha:.2f}")
    ax.set_xlabel("V(e_next) / V(e_k)")
    ax.set_title(f"Decrease ratio  —  {pct:.1f}% certified")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("models/lyapunov_scatter.png", dpi=120)
    plt.show()
    print("Saved → models/lyapunov_scatter.png")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--episodes",    type=int,   default=15)
    p.add_argument("--epochs",      type=int,   default=300)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--hidden",      type=int,   default=128)
    p.add_argument("--n_layers",    type=int,   default=3)
    p.add_argument("--alpha",       type=float, default=0.05)
    p.add_argument("--lambda_lyap", type=float, default=5.0)
    p.add_argument("--lambda_pd",   type=float, default=1.0)
    args = p.parse_args()
    train(args.episodes, args.epochs, args.lr, args.hidden, args.n_layers,
          args.alpha, args.lambda_lyap, args.lambda_pd)