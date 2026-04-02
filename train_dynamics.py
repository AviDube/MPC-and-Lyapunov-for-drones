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
SAVE_PATH    = "dynamics_model"
os.makedirs(SAVE_PATH, exist_ok=True)

# Architecture
HIDDEN_DIMS  = [256, 256, 256]
ACTIVATION   = "silu"

# Training
BATCH_SIZE   = 512
EPOCHS       = 200
LR           = 3e-4
WEIGHT_DECAY = 1e-5
VAL_FRAC     = 0.1
PATIENCE     = 20          # early stopping — epochs without val loss improvement
LR_PATIENCE  = 10          # halve LR if no improvement for this many epochs

# Angle indices in the 12-dim error state
# state = [x, y, z, roll, pitch, yaw, vx, vy, vz, p, q, r]
ANGLE_IDXS   = [3, 4, 5]  # roll, pitch, yaw

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Sinusoidal angle encoding ──────────────────────────────────────────────────
# Replaces each angle with (sin , cos ) to avoid discontinuities at ±pi.
def encode_state(e: torch.Tensor) -> torch.Tensor:
    """
    e: (..., 12)  →  out: (..., 15)
    Non-angle dims pass through unchanged.
    Each angle dim [3,4,5] is replaced by [sin, cos].
    Output order: [x, y, z, sin_r, cos_r, sin_p, cos_p, sin_y, cos_y,
                   vx, vy, vz, p, q, r]
    """
    non_angle_before = e[..., :3]          # x, y, z
    non_angle_after  = e[..., 6:]          # vx, vy, vz, p, q, r
    angles           = e[..., ANGLE_IDXS]  # roll, pitch, yaw
    sin_cos          = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    # interleave sin/cos pairs: [sin_r, cos_r, sin_p, cos_p, sin_y, cos_y]
    sin_cos_interleaved = torch.stack(
        [sin_cos[..., i % 3] if i % 2 == 0 else sin_cos[..., 3 + i // 2]
         for i in range(6)], dim=-1
    )
    
    enc_angles = torch.cat([
        torch.sin(angles),
        torch.cos(angles),
    ], dim=-1)   # (..., 6)

    return torch.cat([non_angle_before, enc_angles, non_angle_after], dim=-1)  # (..., 15)

ENCODED_STATE_DIM = 15   # 3 + 6 + 6
ACTION_DIM        = 4
INPUT_DIM         = ENCODED_STATE_DIM + ACTION_DIM   # 19
OUTPUT_DIM        = 12   

# ── Model ──────────────────────────────────────────────────────────────────────
class DynamicsModel(nn.Module):
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

        # Output scaling layer — initialise near zero so early predictions
        nn.init.uniform_(self.net[-1].weight, -1e-3, 1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, e: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        e: (B, 12)   raw error state
        u: (B,  4)   motor commands
        returns: (B, 12)  predicted Δe
        """
        enc = encode_state(e)          # (B, 15)
        x   = torch.cat([enc, u], dim=-1)  # (B, 19)
        return self.net(x)

# ── Normalisation ──────────────────────────────────────────────────────────────
# Fit on training split only; applied to model inputs and targets.
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

delta_e = next_states - states   # (N, 12) — what the model predicts

print(f"  Loaded {len(states):,} transitions")
print(f"  states:   {states.shape},  Δe: {delta_e.shape}")

# ── Train / val split ──────────────────────────────────────────────────────────
N     = len(states)
N_val = int(N * VAL_FRAC)
N_tr  = N - N_val

dataset    = TensorDataset(states, actions, delta_e)
tr_set, val_set = random_split(dataset, [N_tr, N_val],
                               generator=torch.Generator().manual_seed(42))

tr_loader  = DataLoader(tr_set,  batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=2, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE * 4, shuffle=False,
                        num_workers=2, pin_memory=True)

print(f"  Train: {N_tr:,}  |  Val: {N_val:,}")

# ── Fit normalisers on training data only ──────────────────────────────────────
tr_states  = states[[i for i in tr_set.indices]]
tr_actions = actions[[i for i in tr_set.indices]]
tr_delta   = delta_e[[i for i in tr_set.indices]]

state_norm  = Normalizer(); state_norm.fit(tr_states)
action_norm = Normalizer(); action_norm.fit(tr_actions)
delta_norm  = Normalizer(); delta_norm.fit(tr_delta)

state_norm.save( f"{SAVE_PATH}/state_norm.pt")
action_norm.save(f"{SAVE_PATH}/action_norm.pt")
delta_norm.save( f"{SAVE_PATH}/delta_norm.pt")
print("Normalisers saved.")

# Move normaliser stats to device for fast batch transforms
def to_device(norm):
    norm.mean = norm.mean.to(DEVICE)
    norm.std  = norm.std.to(DEVICE)

to_device(state_norm)
to_device(action_norm)
to_device(delta_norm)

# ── Model, optimiser, scheduler ────────────────────────────────────────────────
model = DynamicsModel().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {n_params:,}")

optimiser = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimiser, mode="min", factor=0.5, patience=LR_PATIENCE,
)
loss_fn = nn.MSELoss()

# ── Training loop ──────────────────────────────────────────────────────────────
labels = ["x_err", "y_err", "z_err", "roll", "pitch", "yaw",
          "vx_err", "vy_err", "vz", "p", "q", "r"]

best_val_loss  = float("inf")
epochs_no_imp  = 0
history        = []

print(f"\nTraining for up to {EPOCHS} epochs (early stop patience={PATIENCE})...\n")

for epoch in range(1, EPOCHS + 1):

    # ── Train ──────────────────────────────────────────────────────────────────
    model.train()
    tr_loss_sum = 0.0

    for e_batch, u_batch, de_batch in tr_loader:
        e_batch  = e_batch.to(DEVICE)
        u_batch  = u_batch.to(DEVICE)
        de_batch = de_batch.to(DEVICE)

        # Normalise inputs and targets
        e_norm  = state_norm.transform(e_batch)
        u_norm  = action_norm.transform(u_batch)
        de_norm = delta_norm.transform(de_batch)

        pred = model(e_norm, u_norm)
        loss = loss_fn(pred, de_norm)

        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        tr_loss_sum += loss.item() * len(e_batch)

    tr_loss = tr_loss_sum / N_tr

    # ── Validate ───────────────────────────────────────────────────────────────
    model.eval()
    val_loss_sum = 0.0
    all_abs_err  = []   # (N_val, 12) in physical units for MAE breakdown

    with torch.no_grad():
        for e_batch, u_batch, de_batch in val_loader:
            e_batch  = e_batch.to(DEVICE)
            u_batch  = u_batch.to(DEVICE)
            de_batch = de_batch.to(DEVICE)

            e_norm  = state_norm.transform(e_batch)
            u_norm  = action_norm.transform(u_batch)
            de_norm = delta_norm.transform(de_batch)

            pred_norm = model(e_norm, u_norm)
            val_loss_sum += loss_fn(pred_norm, de_norm).item() * len(e_batch)

            # Denormalise to get physical-unit errors
            pred_phys = delta_norm.inverse_transform(pred_norm)
            abs_err   = (pred_phys - de_batch).abs()
            all_abs_err.append(abs_err.cpu())

    val_loss  = val_loss_sum / N_val
    mae_per_dim = torch.cat(all_abs_err, dim=0).mean(dim=0)  # (12,)

    scheduler.step(val_loss)
    current_lr = optimiser.param_groups[0]["lr"]

    history.append({
        "epoch": epoch, "tr_loss": tr_loss, "val_loss": val_loss,
        "lr": current_lr,
        "mae": mae_per_dim.tolist(),
    })

    # ── Logging ────────────────────────────────────────────────────────────────
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:4d}/{EPOCHS}  "
              f"tr={tr_loss:.5f}  val={val_loss:.5f}  lr={current_lr:.2e}")
        print("  MAE per dim:")
        for i, (lbl, mae) in enumerate(zip(labels, mae_per_dim)):
            flag = " ✗" if (
                (i in [0,1,2] and mae > 0.005) or   # position: < 5mm
                (i in [3,4]   and mae > 0.003) or   # roll/pitch: < 0.003 rad
                (i == 5       and mae > 0.005) or   # yaw: < 5 mrad
                (i in [6,7,8] and mae > 0.05)  or   # velocity: < 0.05 m/s
                (i in [9,10,11] and mae > 0.05)     # angular rate: < 0.05 rad/s
            ) else ""
            print(f"    {lbl:8s}: {mae:.5f}{flag}")

    # ── Early stopping & checkpointing ────────────────────────────────────────
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_imp = 0
        torch.save({
            "epoch":      epoch,
            "model":      model.state_dict(),
            "optimiser":  optimiser.state_dict(),
            "val_loss":   val_loss,
            "config": {
                "hidden_dims":  HIDDEN_DIMS,
                "activation":   ACTIVATION,
                "input_dim":    INPUT_DIM,
                "output_dim":   OUTPUT_DIM,
                "angle_idxs":   ANGLE_IDXS,
            },
        }, f"{SAVE_PATH}/best_model.pt")
    else:
        epochs_no_imp += 1
        if epochs_no_imp >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {PATIENCE} epochs)")
            break

# ── Save training history ──────────────────────────────────────────────────────
with open(f"{SAVE_PATH}/history.json", "w") as f:
    json.dump(history, f, indent=2)

# ── Final evaluation on best checkpoint ───────────────────────────────────────
print(f"\n{'='*60}")
print(f"Best val loss: {best_val_loss:.6f}")
print(f"Loading best checkpoint for final evaluation...")

ckpt = torch.load(f"{SAVE_PATH}/best_model.pt", map_location=DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()

all_preds, all_targets = [], []
with torch.no_grad():
    for e_batch, u_batch, de_batch in val_loader:
        e_batch = e_batch.to(DEVICE)
        u_batch = u_batch.to(DEVICE)

        e_norm = state_norm.transform(e_batch)
        u_norm = action_norm.transform(u_batch)

        pred_norm = model(e_norm, u_norm)
        pred_phys = delta_norm.inverse_transform(pred_norm)

        all_preds.append(pred_phys.cpu())
        all_targets.append(de_batch)

preds   = torch.cat(all_preds,   dim=0)
targets = torch.cat(all_targets, dim=0)

mae_final  = (preds - targets).abs().mean(dim=0)
rmse_final = ((preds - targets) ** 2).mean(dim=0).sqrt()

print("\nFinal validation MAE and RMSE per dimension:")
print(f"  {'dim':8s}  {'MAE':>10s}  {'RMSE':>10s}  status")
print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*6}")

thresholds = [0.005, 0.005, 0.005, 0.003, 0.003, 0.005, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
for i, (lbl, mae, rmse, thr) in enumerate(zip(labels, mae_final, rmse_final, thresholds)):
    status = "OK" if mae < thr else "REVIEW"
    print(f"  {lbl:8s}  {mae.item():>10.5f}  {rmse.item():>10.5f}  {status}")

# ── Regional breakdown: near-hover vs transient ────────────────────────────────
val_states = states[[i for i in val_set.indices]]
z_err_val  = val_states[:, 2].abs()

near_hover  = z_err_val < 0.05   # within 5cm of setpoint
transient   = z_err_val > 0.20   # more than 20cm from setpoint

print(f"\nRegional MAE (z dimension only, as proxy for model quality):")
print(f"  Near hover  (|z_err| < 0.05m, n={near_hover.sum():,}): "
      f"{(preds[near_hover, 2] - targets[near_hover, 2]).abs().mean():.5f} m")
print(f"  Transient   (|z_err| > 0.20m, n={transient.sum():,}):  "
      f"{(preds[transient, 2] - targets[transient, 2]).abs().mean():.5f} m")
print(f"\n  Ratio (transient/near-hover): "
      f"{(preds[transient, 2] - targets[transient, 2]).abs().mean() / (preds[near_hover, 2] - targets[near_hover, 2]).abs().mean():.1f}x")
print("  Target: ratio < 3-5x")
print(f"\nModel and normalisers saved to '{SAVE_PATH}/'")