"""
train_residual.py
─────────────────
Trains a small MLP to predict the residual between the true next state
and what the linearized model predicts:

    δ_t = x_{t+1} - (Ad @ x_t + Bd @ u_t + c)
    NN(x_t, u_t) → δ_t

Usage:
    python train_residual.py                        # uses data/transitions.npz
    python train_residual.py --data data/run2.npz   # custom data path
    python train_residual.py --epochs 300 --lr 3e-4

Output:
    models/residual_nn.pt      — trained weights
    models/residual_scaler.npz — input normalisation stats (mean / std)
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

# ── linearized model (must match your controller exactly) ─────────────────────
import mujoco

XML_PATH = "../basic_quadrotor.xml"
DT_CTRL  = 0.02

model_mj = mujoco.MjModel.from_xml_path(XML_PATH)
m  = model_mj.body_mass[1]
g  = abs(model_mj.opt.gravity[2])
Ix, Iy, Iz = model_mj.body_inertia[1]

nx, nu = 12, 4
A = np.zeros((nx, nx))
B = np.zeros((nx, nu))
A[0,6]=1; A[1,7]=1; A[2,8]=1
A[6,4]=g; A[7,3]=-g
A[3,9]=1; A[4,10]=1; A[5,11]=1
B[8,0]=1/m; B[9,1]=1/Ix; B[10,2]=1/Iy; B[11,3]=1/Iz

Ad = np.eye(nx) + A * DT_CTRL
Bd = B * DT_CTRL
c  = np.zeros(nx); c[8] = -g * DT_CTRL   # gravity correction term


# ── network definition ────────────────────────────────────────────────────────
class ResidualNN(nn.Module):
    """
    Small MLP: (x, u) → δ
    Input  dim: nx + nu = 16
    Output dim: nx      = 12

    Two hidden layers of 64 units with layer norm + ELU activation.
    Layer norm (vs batch norm) is important here because at MPC solve time
    we do single-sample forward passes — batch norm would behave differently
    during training vs inference.
    """
    def __init__(self, nx=12, nu=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nx + nu, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
            nn.Linear(hidden, nx),
        )
        # initialise last layer near zero so the NN starts as a small correction
        nn.init.uniform_(self.net[-1].weight, -1e-3, 1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, xu):
        return self.net(xu)


# ── normalisation helpers ─────────────────────────────────────────────────────
class Normalizer:
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std

    def transform(self, x):
        return (x - self.mean) / (self.std + 1e-8)

    def save(self, path):
        np.savez(path, mean=self.mean, std=self.std)

    @staticmethod
    def load(path):
        d = np.load(path)
        return Normalizer(d["mean"], d["std"])


# ── training ──────────────────────────────────────────────────────────────────
def train(args):
    os.makedirs("models", exist_ok=True)

    # ── load data ─────────────────────────────────────────────────────────────
    d = np.load(args.data)
    X      = d["X"]          # (N, 12)
    U      = d["U_wrench"]   # (N,  4)  — wrench [T, tx, ty, tz]
    X_next = d["X_next"]     # (N, 12)

    N = len(X)
    print(f"Loaded {N} transitions from {args.data}")

    # ── compute residuals ─────────────────────────────────────────────────────
    # x_lin = Ad @ x + Bd @ u + c  (vectorised over batch)
    X_lin = (Ad @ X.T).T + (Bd @ U.T).T + c    # (N, 12)
    delta  = X_next - X_lin                      # (N, 12)  ← learning target

    print(f"Residual stats (per dim):")
    print(f"  mean  = {delta.mean(axis=0).round(4)}")
    print(f"  std   = {delta.std(axis=0).round(4)}")
    print(f"  max|δ|= {np.abs(delta).max(axis=0).round(4)}")

    # ── build input/output tensors ────────────────────────────────────────────
    XU = np.concatenate([X, U], axis=1).astype(np.float32)  # (N, 16)
    D  = delta.astype(np.float32)                            # (N, 12)

    # Normalise inputs only (output is already small-ish residual)
    scaler = Normalizer(XU.mean(axis=0), XU.std(axis=0))
    scaler.save("models/residual_scaler.npz")
    XU_n = scaler.transform(XU)

    dataset = TensorDataset(torch.from_numpy(XU_n), torch.from_numpy(D))
    n_val   = max(1, int(0.1 * N))
    n_train = N - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch)

    # ── model / optimiser ─────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net    = ResidualNN(nx=nx, nu=nu, hidden=args.hidden).to(device)
    opt    = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    print(f"\nTraining on {device}  |  {sum(p.numel() for p in net.parameters())} params")

    train_losses, val_losses = [], []
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        # train
        net.train()
        tl = 0.0
        for xu, dlt in train_loader:
            xu, dlt = xu.to(device), dlt.to(device)
            pred = net(xu)
            loss = loss_fn(pred, dlt)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tl += loss.item() * len(xu)
        tl /= n_train

        # validate
        net.eval()
        vl = 0.0
        with torch.no_grad():
            for xu, dlt in val_loader:
                xu, dlt = xu.to(device), dlt.to(device)
                vl += loss_fn(net(xu), dlt).item() * len(xu)
        vl /= n_val
        sched.step()

        train_losses.append(tl)
        val_losses.append(vl)

        if vl < best_val:
            best_val = vl
            torch.save(net.state_dict(), "models/residual_nn.pt")

        if epoch % 20 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}/{args.epochs}  "
                  f"train={tl:.2e}  val={vl:.2e}  lr={sched.get_last_lr()[0]:.1e}")

    print(f"\nBest val loss: {best_val:.4e}  →  models/residual_nn.pt")

    # ── plot ──────────────────────────────────────────────────────────────────
    plt.figure(figsize=(8, 3))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses,   label="val")
    plt.yscale("log")
    plt.xlabel("Epoch"); plt.ylabel("MSE loss"); plt.legend(); plt.grid()
    plt.title("Residual NN training")
    plt.tight_layout()
    plt.savefig("models/training_curve.png", dpi=120)
    plt.show()

    # ── quick sanity: per-dimension RMSE on val set ───────────────────────────
    net.load_state_dict(torch.load("models/residual_nn.pt", map_location=device))
    net.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for xu, dlt in val_loader:
            all_pred.append(net(xu.to(device)).cpu().numpy())
            all_true.append(dlt.numpy())
    pred = np.concatenate(all_pred); true = np.concatenate(all_true)
    rmse = np.sqrt(((pred - true)**2).mean(axis=0))
    labels = ["x","y","z","r","p","yw","vx","vy","vz","wx","wy","wz"]
    print("\nVal RMSE per state dim:")
    for lbl, r in zip(labels, rmse):
        print(f"  {lbl:>3}: {r:.4e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="../data/transitions.npz")
    parser.add_argument("--epochs", type=int,   default=200)
    parser.add_argument("--lr",     type=float, default=3e-4)
    parser.add_argument("--batch",  type=int,   default=256)
    parser.add_argument("--hidden", type=int,   default=64)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()