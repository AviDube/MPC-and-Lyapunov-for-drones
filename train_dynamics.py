import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import json

# ── Config ─────────────────────────────────────────────────────────────────────
DATASET_PATH = "dataset"
SAVE_PATH    = "dynamics_model_hybrid"
os.makedirs(SAVE_PATH, exist_ok=True)

# Architecture
HIDDEN_DIMS  = [64, 64, 64]
ACTIVATION   = "silu"

# Training
BATCH_SIZE   = 512
EPOCHS       = 200
LR           = 3e-4
WEIGHT_DECAY = 1e-5
VAL_FRAC     = 0.1
PATIENCE     = 20
LR_PATIENCE  = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── State / action dimensions ──────────────────────────────────────────────────
STATE_DIM    = 13   # x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz
ACTION_DIM   = 4
INPUT_DIM    = STATE_DIM + ACTION_DIM
OUTPUT_DIM   = STATE_DIM  # predict next state delta

# ── Physics prior ─────────────────────────────────────────────────────────────
# Simple physics-based model: Δpos = vel*dt, Δvel = (thrust - gravity)/m * dt
GRAVITY = 9.81
DT_CTRL = 0.02
MASS    = 0.027  # approximate quadrotor mass

def physics_prior(state: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    state: (B, 13) [pos(3), quat(4), vel(3), omega(3)]
    u:     (B, 4)  rotor thrusts
    returns: (B, 13) predicted Δstate from simple physics
    """
    pos = state[:, :3]
    vel = state[:, 7:10]
    thrust_total = u.sum(dim=1, keepdim=True)  # total thrust
    acc = torch.zeros_like(vel)
    acc[:, 2] = thrust_total[:, 0]/MASS - GRAVITY

    delta_pos = vel * DT_CTRL
    delta_vel = acc * DT_CTRL
    delta_state = torch.zeros_like(state)
    delta_state[:, :3]    = delta_pos
    delta_state[:, 7:10]  = delta_vel
    # quaternions and angular rates left as zero (NN will learn residual)
    return delta_state

# ── Hybrid NN model ────────────────────────────────────────────────────────────
class HybridDynamicsModel(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dims=HIDDEN_DIMS,
                 output_dim=OUTPUT_DIM, activation="silu"):
        super().__init__()
        act_fn = {"silu": nn.SiLU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), act_fn()]
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

        nn.init.uniform_(self.net[-1].weight, -1e-3, 1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # Physics prior
        delta_phys = physics_prior(state, u)
        # NN residual
        x = torch.cat([state, u], dim=-1)
        delta_residual = self.net(x)
        return delta_phys + delta_residual

# ── Normalization ─────────────────────────────────────────────────────────────
class Normalizer:
    def __init__(self):
        self.mean = None
        self.std  = None

    def fit(self, x: torch.Tensor):
        self.mean = x.mean(dim=0)
        self.std  = x.std(dim=0).clamp(min=1e-6)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean

    def save(self, path: str):
        torch.save({"mean": self.mean, "std": self.std}, path)

    @classmethod
    def load(cls, path: str):
        n = cls()
        d = torch.load(path, map_location="cpu")
        n.mean = d["mean"]
        n.std  = d["std"]
        return n

# ── Load dataset ───────────────────────────────────────────────────────────────
print("Loading dataset...")
states      = torch.tensor(np.load(f"{DATASET_PATH}/states.npy"),      dtype=torch.float32)
actions     = torch.tensor(np.load(f"{DATASET_PATH}/actions.npy"),     dtype=torch.float32)
next_states = torch.tensor(np.load(f"{DATASET_PATH}/next_states.npy"), dtype=torch.float32)

delta_states = next_states - states
print(f"  Loaded {len(states):,} transitions")
print(f"  states: {states.shape},  Δstates: {delta_states.shape}")

# ── Train / val split ──────────────────────────────────────────────────────────
N     = len(states)
N_val = int(N * VAL_FRAC)
N_tr  = N - N_val

dataset    = TensorDataset(states, actions, delta_states)
tr_set, val_set = random_split(dataset, [N_tr, N_val], generator=torch.Generator().manual_seed(42))

tr_loader  = DataLoader(tr_set,  batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE*4, shuffle=False, num_workers=2, pin_memory=True)

# ── Fit normalizers on training data only ──────────────────────────────────────
tr_states  = states[[i for i in tr_set.indices]]
tr_actions = actions[[i for i in tr_set.indices]]
tr_delta   = delta_states[[i for i in tr_set.indices]]

state_norm  = Normalizer(); state_norm.fit(tr_states)
action_norm = Normalizer(); action_norm.fit(tr_actions)
delta_norm  = Normalizer(); delta_norm.fit(tr_delta)

state_norm.save( f"{SAVE_PATH}/state_norm.pt")
action_norm.save(f"{SAVE_PATH}/action_norm.pt")
delta_norm.save( f"{SAVE_PATH}/delta_norm.pt")
print("Normalisers saved.")

# ── Move normalizer stats to device ───────────────────────────────────────────
for norm in [state_norm, action_norm, delta_norm]:
    norm.mean = norm.mean.to(DEVICE)
    norm.std  = norm.std.to(DEVICE)

# ── Model, optimizer, scheduler ──────────────────────────────────────────────
model = HybridDynamicsModel().to(DEVICE)
optimiser = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="min", factor=0.5, patience=LR_PATIENCE)
loss_fn = nn.MSELoss()

# ── Training loop ─────────────────────────────────────────────────────────────
best_val_loss  = float("inf")
epochs_no_imp  = 0
history        = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    tr_loss_sum = 0.0

    for s_batch, u_batch, ds_batch in tr_loader:
        s_batch, u_batch, ds_batch = s_batch.to(DEVICE), u_batch.to(DEVICE), ds_batch.to(DEVICE)

        s_norm  = state_norm.transform(s_batch)
        u_norm  = action_norm.transform(u_batch)
        ds_norm = delta_norm.transform(ds_batch)

        pred = model(s_norm, u_norm)
        loss = loss_fn(pred, ds_norm)

        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        tr_loss_sum += loss.item() * len(s_batch)

    tr_loss = tr_loss_sum / N_tr

    # Validation
    model.eval()
    val_loss_sum = 0.0
    all_abs_err  = []

    with torch.no_grad():
        for s_batch, u_batch, ds_batch in val_loader:
            s_batch, u_batch, ds_batch = s_batch.to(DEVICE), u_batch.to(DEVICE), ds_batch.to(DEVICE)

            s_norm  = state_norm.transform(s_batch)
            u_norm  = action_norm.transform(u_batch)
            ds_norm = delta_norm.transform(ds_batch)

            pred_norm = model(s_norm, u_norm)
            val_loss_sum += loss_fn(pred_norm, ds_norm).item() * len(s_batch)

            pred_phys = delta_norm.inverse_transform(pred_norm)
            all_abs_err.append((pred_phys - ds_batch).abs().cpu())

    val_loss = val_loss_sum / N_val
    mae_per_dim = torch.cat(all_abs_err, dim=0).mean(dim=0)

    scheduler.step(val_loss)
    current_lr = optimiser.param_groups[0]["lr"]

    history.append({
        "epoch": epoch, "tr_loss": tr_loss, "val_loss": val_loss,
        "lr": current_lr, "mae": mae_per_dim.tolist(),
    })

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:4d}/{EPOCHS} tr={tr_loss:.5f} val={val_loss:.5f} lr={current_lr:.2e}")

    # Checkpoint & early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_imp = 0
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimiser": optimiser.state_dict(),
            "val_loss": val_loss,
        }, f"{SAVE_PATH}/best_model.pt")
    else:
        epochs_no_imp += 1
        if epochs_no_imp >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

# ── Save history ──────────────────────────────────────────────────────────────
with open(f"{SAVE_PATH}/history.json", "w") as f:
    json.dump(history, f, indent=2)

print(f"\nTraining complete. Model and normalizers saved to '{SAVE_PATH}/'")