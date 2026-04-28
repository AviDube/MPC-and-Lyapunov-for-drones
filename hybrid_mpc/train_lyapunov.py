"""
Trains a Lyapunov network V(e) for the closed-loop system with the learned policy.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import mujoco

from physics import rk4_np

XML_PATH = "../basic_quadrotor.xml"
DT_CTRL  = 0.02
nx, nu   = 12, 4

mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
MASS     = float(mj_model.body_mass[1])
GRAV     = float(abs(mj_model.opt.gravity[2]))
INERTIA  = tuple(mj_model.body_inertia[1])
Ix,Iy,Iz = INERTIA

X_REF  = np.array([0.5,-0.5,1.0, 0,0,0, 0,0,0, 0,0,0], dtype=np.float32)
U_STAR = np.array([MASS*GRAV, 0.0, 0.0, 0.0], dtype=np.float32)

os.makedirs("models", exist_ok=True)


# model definitions
class NeuralPolicy(nn.Module):
    def __init__(self, nx=12, hidden=128, n_layers=3):
        super().__init__()
        layers = [nn.Linear(nx, hidden), nn.Tanh()]
        for _ in range(n_layers-1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, nu, bias=False)]
        self.delta_net = nn.Sequential(*layers)
        self.register_buffer("u_star",
            torch.from_numpy(U_STAR))
        self.register_buffer("delta_max",
            torch.tensor([MASS*GRAV, 0.005, 0.005, 0.02]))
    def forward(self, e):
        return self.u_star + self.delta_max * torch.tanh(self.delta_net(e))


class HybridResidualNN(nn.Module):
    def __init__(self, nx=12, nu=4, hidden=64, n_layers=3):
        super().__init__()
        layers = [nn.Linear(nx+nu, hidden), nn.SiLU()]
        for _ in range(n_layers-1): layers+=[nn.Linear(hidden,hidden),nn.SiLU()]
        layers += [nn.Linear(hidden, nx)]
        self.net = nn.Sequential(*layers)
    def forward(self, xu): return self.net(xu)


class LyapunovNet(nn.Module):
    """
    Simple NN
    """
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


def _R(phi, theta, psi):
    cp=torch.cos(phi); sp=torch.sin(phi)
    ct=torch.cos(theta); st=torch.sin(theta)
    cy=torch.cos(psi);  sy=torch.sin(psi)
    B=phi.shape[0]; R=torch.zeros(B,3,3,device=phi.device)
    R[:,0,0]=cy*ct;  R[:,0,1]=cy*st*sp-sy*cp; R[:,0,2]=cy*st*cp+sy*sp
    R[:,1,0]=sy*ct;  R[:,1,1]=sy*st*sp+cy*cp; R[:,1,2]=sy*st*cp-cy*sp
    R[:,2,0]=-st;    R[:,2,1]=ct*sp;           R[:,2,2]=ct*cp
    return R

def _W(phi, theta):
    cp=torch.cos(phi); sp=torch.sin(phi)
    ct=torch.cos(theta); tt=torch.tan(theta)
    B=phi.shape[0]; W=torch.zeros(B,3,3,device=phi.device)
    W[:,0,0]=1; W[:,0,1]=sp*tt; W[:,0,2]=cp*tt
    W[:,1,0]=0; W[:,1,1]=cp;    W[:,1,2]=-sp
    W[:,2,0]=0; W[:,2,1]=sp/ct; W[:,2,2]=cp/ct
    return W

def physics_step(x_abs, u, dt=DT_CTRL):
    def ode(x, u):
        phi=x[:,3]; theta=x[:,4]; psi=x[:,5]
        p=x[:,9]; q=x[:,10]; r=x[:,11]
        T=u[:,0]; tx=u[:,1]; ty=u[:,2]; tz=u[:,3]
        pos_dot = x[:,6:9]
        Rm = _R(phi,theta,psi)
        tb = torch.stack([torch.zeros_like(T),torch.zeros_like(T),T/MASS],1)
        gw = torch.zeros_like(pos_dot); gw[:,2]=-GRAV
        acc = (Rm @ tb.unsqueeze(-1)).squeeze(-1) + gw
        omega=x[:,9:12]; Wm=_W(phi,theta)
        eta_dot=(Wm@omega.unsqueeze(-1)).squeeze(-1)
        Io=torch.stack([Ix*p,Iy*q,Iz*r],1)
        gyro=torch.linalg.cross(omega,Io)
        tau=u[:,1:4]
        od=torch.stack([(tau[:,0]-gyro[:,0])/Ix,
                        (tau[:,1]-gyro[:,1])/Iy,
                        (tau[:,2]-gyro[:,2])/Iz],1)
        return torch.cat([pos_dot,eta_dot,acc,od],1)
    k1=ode(x_abs,u); k2=ode(x_abs+dt/2*k1,u)
    k3=ode(x_abs+dt/2*k2,u); k4=ode(x_abs+dt*k3,u)
    return x_abs+(dt/6)*(k1+2*k2+2*k3+k4)


def make_cl_step(policy, nn_res, xu_mean, xu_std,
                 x_ref_t, e_scale_t, delta_eq, device):
    """
    cl_step(e) -> e_next   in scaled error coordinates.

    Correction: subtract delta* = f_theta(x*, u*) so f_cl(x*) = x* exactly.
    """
    def cl_step(e):
        B       = e.shape[0]
        x_ref_b = x_ref_t.expand(B, -1)
        x_abs   = e * e_scale_t + x_ref_b          # absolute state
        u       = policy(e)                          # pi_phi(e)

        x_phys  = physics_step(x_abs, u)

        xu      = torch.cat([x_abs, u], dim=1)
        xu_n    = (xu - xu_mean) / xu_std
        delta   = nn_res(xu_n)

        x_next  = x_phys + delta - delta_eq         # equilibrium correction
        return (x_next - x_ref_b) / e_scale_t       # back to scaled error
    return cl_step


def compute_delta_eq(nn_res, xu_mean, xu_std, device):
    x_star = torch.from_numpy(X_REF).float().to(device).unsqueeze(0)
    u_star = torch.from_numpy(U_STAR).float().to(device).unsqueeze(0)
    xu     = torch.cat([x_star, u_star], 1)
    xu_n   = (xu - xu_mean) / xu_std
    with torch.no_grad():
        delta_star = nn_res(xu_n)
    mag = delta_star.abs().max().item()
    print(f"  delta* max magnitude: {mag:.4e}")
    return delta_star


def lyapunov_loss(V, e, e_next, alpha, eps_pd, lam):
    Ve   = V(e).squeeze()
    Ven  = V(e_next).squeeze()
    norm = e.norm(dim=1)

    # Positive definiteness outside tiny ball
    mask = (norm > 0.02).float()
    L_pd = (mask * torch.relu(eps_pd - Ve)).mean()

    # Exponential decrease
    L_dec = torch.relu(Ven - (1-alpha)*Ve).mean()

    # Encourage V to grow toward boundary
    mask_b  = (norm > 0.8).float()
    L_bound = (mask_b * torch.relu(1.0 - Ve)).mean()

    return L_pd + lam*L_dec + 0.1*L_bound, {
        "L_pd":  L_pd.item(),
        "L_dec": L_dec.item(),
        "V_mean": Ve.mean().item(),
        "V_min":  Ve.min().item(),
    }


@torch.no_grad()
def find_violations(V, cl_step, alpha, n=100_000, device="cpu"):
    e    = (torch.rand(n, nx, device=device)*2 - 1)
    en   = cl_step(e)
    Ve   = V(e).squeeze()
    Ven  = V(en).squeeze()
    norm = e.norm(dim=1)

    viol = ((norm > 0.02) & (Ve < 1e-3)) | (Ven > (1-alpha)*Ve)
    n_v  = viol.sum().item()
    return e[viol].detach(), n_v / n, n_v


# Training loop
def train(alpha=0.05, eps_pd=0.01, lam=10.0,
          iters=30, inner_steps=200, batch=1024,
          n_init=50_000, n_verify=100_000, lr=1e-3):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load policy
    cfg     = np.load("models/policy_config.npz")
    e_scale = torch.from_numpy(cfg["e_scale"]).float().to(device)
    policy  = NeuralPolicy().to(device)
    policy.load_state_dict(torch.load("models/neural_policy.pt",
                                      map_location=device))
    policy.eval()

    with torch.no_grad():
        eq_err = (policy(torch.zeros(1,nx,device=device)).squeeze()
                  - torch.from_numpy(U_STAR).to(device)).abs().max().item()
    print(f"Policy equilibrium error: {eq_err:.2e}  "
          f"[{'PASS' if eq_err<1e-5 else 'WARNING'}]")

    # load hybrid NN residual
    cfg2    = np.load("models/hybrid_config.npz")
    nn_res  = HybridResidualNN(hidden=int(cfg2["hidden"][0]),
                                n_layers=int(cfg2["n_layers"][0])).to(device)
    nn_res.load_state_dict(torch.load("models/hybrid_nn.pt",
                                      map_location=device))
    nn_res.eval()
    xu_mean = torch.from_numpy(cfg2["xu_mean"].astype(np.float32)).to(device)
    xu_std  = torch.from_numpy(cfg2["xu_std"].astype(np.float32)).to(device)

    # equilibrium correction
    print("\nComputing equilibrium correction delta*:")
    x_ref_t  = torch.from_numpy(X_REF).float().to(device).unsqueeze(0)
    delta_eq = compute_delta_eq(nn_res, xu_mean, xu_std, device)

    # Verify f_cl(x*) = x*
    cl_step = make_cl_step(policy, nn_res, xu_mean, xu_std,
                           x_ref_t, e_scale, delta_eq, device)
    with torch.no_grad():
        e0_nxt = cl_step(torch.zeros(1, nx, device=device))
    print(f"||f_cl(x*) - x*|| = {e0_nxt.norm().item():.2e}  "
          f"[{'PASS' if e0_nxt.norm().item()<1e-4 else 'FAIL'}]")

    # Lyapunov network
    V_net = LyapunovNet(nx=nx, hidden=128, feat_dim=64).to(device)
    opt   = torch.optim.Adam(V_net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=40, gamma=0.5)

    # Verify V(0) = 0
    with torch.no_grad():
        v0 = V_net(torch.zeros(1,nx,device=device)).item()
    print(f"V(0) = {v0:.2e}  [{'PASS' if v0<1e-8 else 'FAIL'}]\n")
    print(f"LyapunovNet: {sum(p.numel() for p in V_net.parameters())} params")
    print(f"alpha={alpha}  iters={iters}  inner_steps={inner_steps}\n")

    # Training set — unit cube in scaled error space
    train_set = (torch.rand(n_init, nx, device=device)*2 - 1)

    ce_rates, losses, dec_losses = [], [], []

    for outer in range(1, iters+1):

        V_net.train()
        for _ in range(inner_steps):
            idx   = torch.randperm(len(train_set))[:batch]
            e     = train_set[idx]
            e_nxt = cl_step(e)

            e_zero    = torch.zeros(16, nx, device=device)
            e_zero_nxt= cl_step(e_zero)

            loss, info = lyapunov_loss(
                V_net,
                torch.cat([e, e_zero]),
                torch.cat([e_nxt, e_zero_nxt]),
                alpha, eps_pd, lam)

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(V_net.parameters(), 1.0)
            opt.step()

        sched.step()

        V_net.eval()
        ces, ce_rate, n_viol = find_violations(
            V_net, cl_step, alpha, n=n_verify, device=device)

        ce_rates.append(ce_rate)
        losses.append(loss.item())
        dec_losses.append(info["L_dec"])

        print(f"Iter {outer:3d}/{iters}  loss={loss.item():.3e}  "
              f"L_dec={info['L_dec']:.3e}  "
              f"violations={n_viol}/{n_verify} ({100*ce_rate:.1f}%)  "
              f"V_min={info['V_min']:.3e}")

        # Add counterexamples
        if len(ces) > 0:
            repeat    = min(5, max(1, batch // max(len(ces), 1)))
            train_set = torch.cat([train_set, ces.repeat(repeat, 1)])
            if len(train_set) > 200_000:
                idx       = torch.randperm(len(train_set))[:150_000]
                train_set = train_set[idx]

        # Save best
        if outer == 1 or ce_rate < min(ce_rates[:-1]):
            torch.save(V_net.state_dict(), "models/lyapunov_net.pt")
            np.savez("models/lyapunov_config.npz",
                     x_ref=X_REF, u_star=U_STAR,
                     delta_eq=delta_eq.cpu().numpy(),
                     e_scale=e_scale.cpu().numpy(),
                     alpha=np.array([alpha]),
                     feat_dim=np.array([64]),
                     hidden=np.array([128]))

        if ce_rate < 0.001:
            print(f"\nCertificate achieved at iter {outer} (<0.1% violations)")
            break

    # final check
    print("\n── Final certificate check (500k samples) ──")
    V_net.load_state_dict(torch.load("models/lyapunov_net.pt",
                                     map_location=device))
    V_net.eval()
    _, final_rate, final_n = find_violations(
        V_net, cl_step, alpha, n=500_000, device=device)
    print(f"Violation rate: {100*final_rate:.2f}%  ({final_n}/500,000)")
    print("Certificate VALID" if final_rate < 0.01
          else "Certificate PARTIAL — try more iters or smaller alpha")

    fig, axes = plt.subplots(1,3,figsize=(13,3))
    axes[0].plot(losses);     axes[0].set_yscale("log")
    axes[0].set_title("Total loss"); axes[0].grid()
    axes[1].plot(dec_losses); axes[1].set_yscale("log")
    axes[1].set_title("Decrease condition loss"); axes[1].grid()
    axes[2].plot([r*100 for r in ce_rates])
    axes[2].axhline(1, ls="--", color="r", alpha=0.6, label="1% threshold")
    axes[2].set_title("Violation rate (%)"); axes[2].legend(); axes[2].grid()
    plt.tight_layout()
    plt.savefig("models/lyapunov_training.png", dpi=120)
    plt.show()
    print("Saved → models/lyapunov_training.png")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--alpha",       type=float, default=0.05)
    p.add_argument("--eps_pd",      type=float, default=0.01)
    p.add_argument("--lam",         type=float, default=10.0)
    p.add_argument("--iters",       type=int,   default=30)
    p.add_argument("--inner_steps", type=int,   default=200)
    p.add_argument("--batch",       type=int,   default=1024)
    p.add_argument("--n_init",      type=int,   default=50_000)
    p.add_argument("--n_verify",    type=int,   default=100_000)
    p.add_argument("--lr",          type=float, default=1e-3)
    args = p.parse_args()
    train(args.alpha, args.eps_pd, args.lam, args.iters,
          args.inner_steps, args.batch, args.n_init, args.n_verify, args.lr)