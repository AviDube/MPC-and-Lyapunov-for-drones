"""
Trains a hybrid dynamics model.

Usage:
    python train_hybrid.py
    python train_hybrid.py --data data/transitions.npz --epochs 300
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import mujoco

from hybrid_mpc.physics import rk4_np

# load params
XML_PATH = "../basic_quadrotor.xml"
mj_model = mujoco.MjModel.from_xml_path(XML_PATH)

MASS    = float(mj_model.body_mass[1])
GRAV    = float(abs(mj_model.opt.gravity[2]))
INERTIA = tuple(mj_model.body_inertia[1])
DT      = 0.02

nx, nu  = 12, 4

print(f"Drone: mass={MASS:.4f}kg  g={GRAV:.4f}  "
      f"I={[f'{v:.6f}' for v in INERTIA]}")


# residual NN
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.l1  = nn.Linear(dim, dim)
        self.l2  = nn.Linear(dim, dim)
        self.act = nn.SiLU()   # SiLU
                               

    def forward(self, x):
        return x + self.l2(self.act(self.l1(x)))


class HybridResidualNN(nn.Module):
    """
    Small MLP that learns the physics residual f_theta(x, u)

    Input  : (x, u) concatenated — 16 dims
    Output : delta = x_next_true - x_next_physics — 12 dims
    """
    def __init__(self, nx=12, nu=4, hidden=64, n_layers=3):
        super().__init__()
        in_dim = nx + nu

        layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, nx)]

        self.net = nn.Sequential(*layers)

        # Zero-init output layer → residual starts at zero
        nn.init.uniform_(self.net[-1].weight, -1e-4, 1e-4)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, xu):
        return self.net(xu)


def compute_physics_residuals(X, U, X_next):
    """
    Compute delta = x_next_true - RK4_physics(x, u) for every transition.
    """
    N = len(X)
    delta = np.zeros_like(X_next)

    for i in range(N):
        x_phys = rk4_np(X[i], U[i], DT, MASS, INERTIA, GRAV)
        delta[i] = X_next[i] - x_phys

    return delta


def train(args):
    os.makedirs("models", exist_ok=True)

    d      = np.load(args.data)
    X      = d["X"].astype(np.float64)
    U      = d["U_wrench"].astype(np.float64)
    X_next = d["X_next"].astype(np.float64)
    N      = len(X)
    print(f"Loaded {N} transitions from {args.data}")

    print("Computing physics residuals (RK4)...")
    delta = compute_physics_residuals(X, U, X_next)

    print(f"\nResidual statistics (what the NN must learn):")
    labels = ["x","y","z","roll","pitch","yaw","vx","vy","vz","p","q","r"]
    for i, lbl in enumerate(labels):
        print(f"  {lbl:>6}: mean={delta[:,i].mean():+.2e}  "
              f"std={delta[:,i].std():.2e}  "
              f"max|d|={np.abs(delta[:,i]).max():.2e}")

    # Compare residual magnitude to full state transition magnitude
    full_mag = np.abs(X_next - X).mean()
    res_mag  = np.abs(delta).mean()
    print(f"\nFull transition magnitude: {full_mag:.4f}")
    print(f"Residual magnitude:        {res_mag:.4f}  "
          f"({100*res_mag/full_mag:.1f}% of full transition)")
    print("(Lower % = physics model is more accurate = easier learning task)")

    # normalise inputs
    XU      = np.concatenate([X, U], axis=1).astype(np.float32)
    delta_f = delta.astype(np.float32)

    xu_mean = XU.mean(axis=0).astype(np.float32)
    xu_std  = np.clip(XU.std(axis=0), 1e-6, None).astype(np.float32)
    XU_n    = (XU - xu_mean) / xu_std

    # Save normalisation stats
    np.savez("models/hybrid_config.npz",
             xu_mean=xu_mean, xu_std=xu_std,
             mass=np.array([MASS]), grav=np.array([GRAV]),
             inertia=np.array(INERTIA), dt=np.array([DT]),
             hidden=np.array([args.hidden]),
             n_layers=np.array([args.n_layers]),
             nx=np.array([nx]), nu=np.array([nu]))

    dataset = TensorDataset(
        torch.from_numpy(XU_n),
        torch.from_numpy(delta_f),
    )
    n_val   = max(1, int(0.1 * N))
    n_train = N - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch)

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net    = HybridResidualNN(nx=nx, nu=nu,
                              hidden=args.hidden,
                              n_layers=args.n_layers).to(device)
    opt    = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
                 opt, T_max=args.epochs, eta_min=1e-5)
    loss_fn = nn.HuberLoss(delta=0.01)

    n_params = sum(p.numel() for p in net.parameters())
    print(f"\nTraining HybridResidualNN on {device}  |  {n_params} params")

    train_losses, val_losses = [], []
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        net.train()
        tl = 0.0
        for xu_n, dlt in train_loader:
            xu_n, dlt = xu_n.to(device), dlt.to(device)
            loss = loss_fn(net(xu_n), dlt)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            tl += loss.item()
        tl /= len(train_loader)

        net.eval()
        vl = 0.0
        with torch.no_grad():
            for xu_n, dlt in val_loader:
                vl += loss_fn(net(xu_n.to(device)),
                              dlt.to(device)).item()
        vl /= len(val_loader)
        sched.step()

        train_losses.append(tl); val_losses.append(vl)
        if vl < best_val:
            best_val = vl
            torch.save(net.state_dict(), "models/hybrid_nn.pt")

        if epoch % 20 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}/{args.epochs}  "
                  f"train={tl:.3e}  val={vl:.3e}  "
                  f"lr={sched.get_last_lr()[0]:.1e}")

    print(f"\nBest val loss: {best_val:.4e}  →  models/hybrid_nn.pt")

    # val
    net.load_state_dict(torch.load("models/hybrid_nn.pt", map_location=device))
    net.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xu_n, dlt in val_loader:
            preds.append(net(xu_n.to(device)).cpu().numpy())
            trues.append(dlt.numpy())
    pred = np.concatenate(preds); true = np.concatenate(trues)
    rmse = np.sqrt(((pred - true)**2).mean(axis=0))

    print("\nVal RMSE of NN residual prediction (physical units):")
    for lbl, r in zip(labels, rmse):
        print(f"  {lbl:>6}: {r:.4e}")

    baseline_rmse = np.sqrt((true**2).mean(axis=0))
    print("\nPhysics-only RMSE (NN not applied):")
    for lbl, b, r in zip(labels, baseline_rmse, rmse):
        pct = 100*r/b if b > 1e-10 else 0
        print(f"  {lbl:>6}: {b:.4e}  →  NN reduces to {r:.4e}  ({pct:.0f}% remaining)")

    # plot
    plt.figure(figsize=(8, 3))
    plt.plot(train_losses, label="train"); plt.plot(val_losses, label="val")
    plt.yscale("log"); plt.xlabel("Epoch"); plt.ylabel("Huber loss")
    plt.title("Hybrid residual NN training"); plt.legend(); plt.grid()
    plt.tight_layout()
    plt.savefig("models/hybrid_training_curve.png", dpi=120)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",     default="../data/transitions.npz")
    parser.add_argument("--epochs",   type=int,   default=200)
    parser.add_argument("--lr",       type=float, default=3e-4)
    parser.add_argument("--batch",    type=int,   default=256)
    parser.add_argument("--hidden",   type=int,   default=64)
    parser.add_argument("--n_layers", type=int,   default=3)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()