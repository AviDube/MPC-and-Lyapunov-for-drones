"""
distil_policy.py
────────────────
Run from hybrid_mpc/:
    cd hybrid_mpc
    python distil_policy.py

Distils the HybridMPC controller into a small neural policy:
    pi_phi(e) -> u,   e = x - x_ref  (error state)

Architecture guarantees pi_phi(0) = u* exactly:
    pi_phi(e) = u* + delta_u(e)
where delta_u has no output bias and zero-init output weights,
so delta_u(0) = 0 by construction.

Outputs:
    models/neural_policy.pt
    models/policy_config.npz   (x_ref, u_star, e_scale)
"""

import os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import mujoco
from scipy.spatial.transform import Rotation

from mpc_hybrid import HybridMPC, get_state, wrench_to_rotors

# ── constants ─────────────────────────────────────────────────────────────────
XML_PATH = "../basic_quadrotor.xml"
DT_CTRL  = 0.02
nx, nu   = 12, 4

mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
mj_data  = mujoco.MjData(mj_model)
MASS     = float(mj_model.body_mass[1])
GRAV     = float(abs(mj_model.opt.gravity[2]))

# Hover equilibrium
X_REF  = np.array([0.5,-0.5,1.0, 0,0,0, 0,0,0, 0,0,0], dtype=np.float32)
U_STAR = np.array([MASS*GRAV, 0.0, 0.0, 0.0], dtype=np.float32)

os.makedirs("models", exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Neural policy  pi_phi(e) = u* + delta_u(e),  delta_u(0) = 0 by construction
# ══════════════════════════════════════════════════════════════════════════════
class NeuralPolicy(nn.Module):
    def __init__(self, nx=12, hidden=128, n_layers=3):
        super().__init__()
        layers = [nn.Linear(nx, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        # No bias on output → output is 0 when hidden activations are 0
        layers += [nn.Linear(hidden, nu, bias=False)]
        self.delta_net = nn.Sequential(*layers)

        # Zero-init: guarantees delta_u(0) = 0 before and after training
        nn.init.zeros_(self.delta_net[-1].weight)

        self.register_buffer("u_star",
            torch.from_numpy(U_STAR))
        self.register_buffer("delta_max",
            torch.tensor([MASS*GRAV, 0.005, 0.005, 0.02]))

    def forward(self, e):
        return self.u_star + self.delta_max * torch.tanh(self.delta_net(e))

    def check_equilibrium(self):
        device = self.u_star.device
        with torch.no_grad():
            u_out = self.forward(torch.zeros(1, nx, device=device))
            err   = (u_out.squeeze() - self.u_star).abs().max().item()
        status = "PASS" if err < 5e-3 else "FAIL"
        print(f"  pi_phi(0) = {u_out.squeeze().cpu().numpy().round(5)}")
        print(f"  u*        = {U_STAR}")
        print(f"  max|pi_phi(0) - u*| = {err:.2e}  [{status}]")


# ══════════════════════════════════════════════════════════════════════════════
# Data collection
# ══════════════════════════════════════════════════════════════════════════════
def collect(n_episodes=15):
    """
    Run HybridMPC from diverse starting offsets, log (e, u) pairs.
    Also adds synthetic near-equilibrium samples so the policy
    learns u*(e≈0) = u* accurately.
    """
    offsets = [
        [0.0,  0.0,  0.0],
        [0.5,  0.0,  0.0],
        [-0.5, 0.0,  0.0],
        [0.0,  0.5,  0.0],
        [0.0, -0.5,  0.0],
        [0.0,  0.0,  0.5],
        [0.0,  0.0, -0.4],
        [0.4,  0.4,  0.3],
        [-0.3, 0.3, -0.2],
    ]

    E_buf, U_buf = [], []

    for ep in range(n_episodes):
        mujoco.mj_resetData(mj_model, mj_data)
        offset = offsets[ep % len(offsets)]
        mj_data.qpos[:3] = X_REF[:3] - np.array(offset)
        mujoco.mj_forward(mj_model, mj_data)

        mpc = HybridMPC()

        for _ in range(int(8.0 / DT_CTRL)):
            x   = get_state(mj_data)
            u   = mpc.solve(x, X_REF)
            if isinstance(u, tuple): u = u[0]

            E_buf.append((x - X_REF).astype(np.float32))
            U_buf.append(np.asarray(u, dtype=np.float32))

            u_rot = np.clip(wrench_to_rotors(u), 0.005, 0.25)
            mj_data.ctrl[:] = u_rot
            for _ in range(int(DT_CTRL / mj_model.opt.timestep)):
                mujoco.mj_step(mj_model, mj_data)

        print(f"  episode {ep+1}/{n_episodes}: {len(E_buf)} transitions")

    # Synthetic near-equilibrium samples
    rng   = np.random.default_rng(0)
    n_eq  = 1000
    E_eq  = rng.standard_normal((n_eq, nx)).astype(np.float32) * 0.01
    U_eq  = np.tile(U_STAR, (n_eq, 1))
    E_buf = np.array(E_buf); U_buf = np.array(U_buf)
    E_all = np.concatenate([E_buf, E_eq])
    U_all = np.concatenate([U_buf, U_eq])

    print(f"\nTotal: {len(E_all)} (e, u) pairs")
    print(f"Error state range: {np.abs(E_all).max(0).round(3)}")
    return E_all, U_all


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════
def train(n_episodes=15, epochs=300, lr=3e-4, hidden=128, n_layers=3):
    print(f"Collecting {n_episodes} episodes...")
    E, U = collect(n_episodes)

    # Scale error state so each dim is in [-1, 1] — keeps 0 at 0
    e_scale = np.clip(np.abs(E).max(0), 1e-3, None).astype(np.float32)
    E_n     = E / e_scale

    np.savez("models/policy_config.npz",
             x_ref=X_REF, u_star=U_STAR, e_scale=e_scale)

    dataset  = TensorDataset(torch.from_numpy(E_n), torch.from_numpy(U))
    n_val    = max(1, int(0.1 * len(dataset)))
    train_ds, val_ds = random_split(dataset, [len(dataset)-n_val, n_val],
                                    generator=torch.Generator().manual_seed(0))
    train_ld = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_ld   = DataLoader(val_ds,   batch_size=512)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = NeuralPolicy(nx=nx, hidden=hidden, n_layers=n_layers).to(device)

    print(f"\nNeuralPolicy: {sum(p.numel() for p in policy.parameters())} params")
    print("Equilibrium check before training:")
    policy.check_equilibrium()

    opt   = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                        eta_min=1e-5)
    best_val = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(1, epochs+1):
        policy.train(); tl = 0
        for eb, ub in train_ld:
            eb, ub = eb.to(device), ub.to(device)
            loss   = nn.functional.mse_loss(policy(eb), ub)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step(); tl += loss.item()
        tl /= len(train_ld)

        policy.eval(); vl = 0
        with torch.no_grad():
            for eb, ub in val_ld:
                vl += nn.functional.mse_loss(
                    policy(eb.to(device)), ub.to(device)).item()
        vl /= len(val_ld)
        sched.step()

        train_losses.append(tl); val_losses.append(vl)
        if vl < best_val:
            best_val = vl
            torch.save(policy.state_dict(), "models/neural_policy.pt")

        if epoch % 50 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}/{epochs}  "
                  f"train={tl:.3e}  val={vl:.3e}")

    print(f"\nBest val: {best_val:.4e}  ->  models/neural_policy.pt")

    # Load best weights back onto the correct device before checking
    policy.load_state_dict(torch.load("models/neural_policy.pt",
                                      map_location=device))
    policy.to(device).eval()
    print("\nEquilibrium check after training:")
    policy.check_equilibrium()

    # Per-channel RMSE — keep everything on device, move to CPU only for numpy
    preds, trues = [], []
    with torch.no_grad():
        for eb, ub in val_ld:
            preds.append(policy(eb.to(device)).cpu().numpy())
            trues.append(ub.numpy())
    pred=np.concatenate(preds); true=np.concatenate(trues)
    rmse=np.sqrt(((pred-true)**2).mean(0))
    print("\nPolicy RMSE per channel:")
    for lbl, r in zip(["T","tau_x","tau_y","tau_z"], rmse):
        print(f"  {lbl:>6}: {r:.4e}")

    plt.figure(figsize=(7,3))
    plt.plot(train_losses, label="train"); plt.plot(val_losses, label="val")
    plt.yscale("log"); plt.xlabel("Epoch"); plt.ylabel("MSE loss")
    plt.title("Policy distillation"); plt.legend(); plt.grid()
    plt.tight_layout()
    plt.savefig("models/policy_training.png", dpi=120)
    plt.show()
    print("Saved → models/policy_training.png")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int,   default=15)
    p.add_argument("--epochs",   type=int,   default=300)
    p.add_argument("--lr",       type=float, default=3e-4)
    p.add_argument("--hidden",   type=int,   default=128)
    p.add_argument("--n_layers", type=int,   default=3)
    args = p.parse_args()
    train(args.episodes, args.epochs, args.lr, args.hidden, args.n_layers)