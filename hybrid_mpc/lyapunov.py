"""
certify_nmpc.py
───────────────
Post-hoc stability certification of the Hybrid NMPC controller using
Generalized Lyapunov Functions (Long, Cortés, Atanasov — NeurIPS 2025).

Pipeline
────────
  1. Sample initial error states around hover
  2. Collect M-step closed-loop rollouts via NMPC + hybrid dynamics
     (warm-start preserved within each rollout for speed)
  3. Train  φ(e; θ₁)  — neural residual on top of quadratic base J^π
            σ(e; θ₂)  — step-weight network
  4. Evaluate generalised decrease condition on large test set
  5. Run automated go/no-go checks
  6. Visualise certificate over 2-D state slices

Usage
─────
  python certify_nmpc.py              # full run
  python certify_nmpc.py --fast       # smoke-test (~10 min)
  python certify_nmpc.py --skip-rollouts  # reuse cached .npy, retrain only
  python certify_nmpc.py --expand     # larger domain (after achieving 100%)

Changes vs v1
─────────────
  • M = 30 (full) / 15 (fast)  — was 20/10; covers full settling transient
  • Tighter default DOMAIN_HALF — achieve 100% here, then use --expand
  • N_EPOCHS = 800 (full) / 150 (fast) — loss had not plateaued in v1
  • Single HybridMPC instance per rollout — warm-start preserved → ~3× faster
  • Cache keyed by (M, domain) — auto-invalidates when params change
  • Automated go/no-go checks with actionable failure messages
  • MuJoCo trajectory loader as fast alternative to live NMPC rollouts

Dependencies: numpy, torch, casadi, matplotlib
(same environment as mpc_hybrid.py — no new installs needed)
"""

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── import your existing MPC machinery ────────────────────────────────────────
from mpc_hybrid import (
    f_hybrid, HybridMPC,
    MASS, GRAV, DT_CTRL, nx, nu,
)

# ══════════════════════════════════════════════════════════════════════════════
# 0.  Configuration
# ══════════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser()
parser.add_argument("--fast",           action="store_true",
                    help="Smoke-test with tiny dataset (~10 min)")
parser.add_argument("--skip-rollouts",  action="store_true",
                    help="Load cached rollouts, skip collection")
parser.add_argument("--expand",         action="store_true",
                    help="Use larger domain (run after achieving 100% on default)")
args, _ = parser.parse_known_args()

# ── Hover reference ───────────────────────────────────────────────────────────
X_REF = np.array([0.5, -0.5, 1.0,  0., 0., 0.,  0., 0., 0.,  0., 0., 0.])

# ── Certification domain ──────────────────────────────────────────────────────
# Strategy: start tight → achieve 100% → run --expand to grow the domain.
# State order: [x, y, z,  roll, pitch, yaw,  vx, vy, vz,  wx, wy, wz]

if args.expand:
    # Expanded domain — only use after 100% on default domain
    DOMAIN_HALF = np.array([
        0.50, 0.50, 0.50,       # ±0.5 m
        0.30, 0.30, 0.52,       # ±17°, ±17°, ±30°
        2.00, 2.00, 2.00,       # ±2 m/s
        2.00, 2.00, 2.00,       # ±2 rad/s
    ])
else:
    # Default tight domain — achievable with M=30
    DOMAIN_HALF = np.array([
        0.30, 0.30, 0.30,       # ±0.3 m
        0.20, 0.20, 0.35,       # ±11°, ±11°, ±20°
        1.00, 1.00, 1.00,       # ±1 m/s
        1.00, 1.00, 1.00,       # ±1 rad/s
    ])

# State normalisation — divide error by these before feeding NNs
STATE_SCALE = DOMAIN_HALF.copy()

# ── Rollout horizon ───────────────────────────────────────────────────────────
# M=30 → 0.6s lookahead.  Attitude settles ~0.3s, position ~0.5–1s.
# If weight concentration is still front-loaded after training, increase to 40.
M = 30 if not args.fast else 15

# Small exclusion ball — MPC settles to neighbourhood, not exactly 0
DELTA = 0.02

# ── Training ──────────────────────────────────────────────────────────────────
N_TRAIN   = 5_000  if not args.fast else  300
N_TEST    = 30_000 if not args.fast else 1_500
BATCH     = 256
N_EPOCHS  = 800    if not args.fast else  150
LR        = 3e-4
ALPHA_BAR = 0.02   # mild per-step decay — intentionally loose
BETA      = 0.01   # positivity regulariser weight on ‖e‖²

OUT_DIR = Path("certificate_outputs")
OUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cpu")

# ── Qf: terminal cost matrix from your NLP  (Qf = 10 × Q) ───────────────────
QF_DIAG = np.array([200., 200., 1000., 50., 50., 500.,
                      20.,  20.,   50., 10., 10.,  100.])
QF = torch.tensor(np.diag(QF_DIAG), dtype=torch.float32)

# ── Cache key — invalidates automatically when M or domain changes ────────────
_cache_key = hashlib.md5(
    np.concatenate([[M], DOMAIN_HALF]).tobytes()
).hexdigest()[:8]
TRAIN_CACHE = OUT_DIR / f"trajs_train_{_cache_key}.npy"
TEST_CACHE  = OUT_DIR / f"trajs_test_{_cache_key}.npy"


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Networks
# ══════════════════════════════════════════════════════════════════════════════
class ResidualNet(nn.Module):
    """
    φ(e; θ₁) : R^12 → R

    Neural correction on top of the quadratic base J^π = e^T Qf e.
    Tanh activations ensure global smoothness (Lipschitz everywhere),
    required for Theorem 4.2 of the paper.
    Initialised small so the quadratic base dominates early in training.
    """
    def __init__(self, ne: int = 12, hidden: int = 128, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(ne, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, e_norm: torch.Tensor) -> torch.Tensor:
        return self.net(e_norm).squeeze(-1)   # (B,)


class WeightNet(nn.Module):
    """
    σ(e; θ₂) : R^12 → R^M_≥0   with  Σᵢ σᵢ = M  (softmax × M)

    Learns where in the horizon to concentrate the decrease requirement.
    Healthy certificates back-load weight (>30% in the last quintile).
    """
    def __init__(self, ne: int = 12, M: int = 30, hidden: int = 64):
        super().__init__()
        self.M = M
        self.net = nn.Sequential(
            nn.Linear(ne, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, M),
        )

    def forward(self, e_norm: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(e_norm), dim=-1) * self.M   # (B, M)


# Origin tensor — used to shift φ so V(0) = 0
_ORIGIN_NORM = torch.zeros(1, nx, device=DEVICE)


def compute_V(e_batch: torch.Tensor,
              phi_net:  ResidualNet,
              beta:     float = BETA) -> torch.Tensor:
    """
    V(e) = e^T Qf e                 ← quadratic base (your MPC terminal cost)
           + |φ(ê) − φ(0)|          ← neural residual, shifted so V(0) = 0
           + β ‖e‖²                 ← strict positivity near origin

    All terms ≥ 0.  V(e) = 0  iff  e = 0.

    Args:
        e_batch : (B, 12)  raw error states
    Returns:
        V       : (B,)
    """
    e_norm = e_batch / torch.tensor(STATE_SCALE, dtype=torch.float32)

    # Quadratic base — always 0 at origin by construction
    V_quad = torch.einsum('bi,ij,bj->b', e_batch, QF, e_batch)

    # Neural residual — absolute value guarantees non-negativity
    phi_e = phi_net(e_norm)
    phi_0 = phi_net(_ORIGIN_NORM.expand(e_batch.size(0), -1))
    V_res = torch.abs(phi_e - phi_0)

    # Positivity — prevents V being flat near origin
    V_pos = beta * torch.sum(e_batch ** 2, dim=-1)

    return V_quad + V_res + V_pos


def generalised_decrease(
        traj_batch: torch.Tensor,
        phi_net:    ResidualNet,
        sigma_net:  WeightNet,
        alpha_bar:  float = ALPHA_BAR,
) -> torch.Tensor:
    """
    F(e_k) = (1/M) Σᵢ σᵢ(e_k)·V(eᵢ)  −  (1−ᾱ)·V(e_k)

    Certificate valid where F(e_k) ≤ 0  (paper eq. 14).

    Args:
        traj_batch : (B, M+1, 12)
    Returns:
        F          : (B,)
    """
    M_steps = traj_batch.shape[1] - 1

    e0      = traj_batch[:, 0, :]
    e_norm0 = e0 / torch.tensor(STATE_SCALE, dtype=torch.float32)

    V0      = compute_V(e0, phi_net)
    weights = sigma_net(e_norm0)                              # (B, M)

    V_future = torch.stack(
        [compute_V(traj_batch[:, i + 1, :], phi_net) for i in range(M_steps)],
        dim=1,
    )                                                          # (B, M)

    weighted_avg = (weights * V_future).sum(dim=1) / M_steps
    return weighted_avg - (1.0 - alpha_bar) * V0


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Rollout collection
# ══════════════════════════════════════════════════════════════════════════════
def rollout_one(e0: np.ndarray, x_ref: np.ndarray, M: int) -> np.ndarray:
    """
    Simulate M closed-loop steps from error state e0.

    FIX vs v1: a single HybridMPC instance is used for the whole rollout so
    IPOPT warm-starts carry over between steps.  Only the first solve from a
    cold start is slow (~1–1.5s); subsequent steps warm-start (~50ms).
    Total per rollout: ~1.5 + (M-1)×0.05s  ≈  3s for M=30.

    Dynamics propagated via f_hybrid (CasADi) — same model the MPC uses.

    Returns:
        traj : (M+1, 12)  error states  float32
    """
    mpc  = HybridMPC()   # one instance → warm-start preserved across steps
    traj = np.empty((M + 1, nx), dtype=np.float32)
    x    = x_ref + e0
    traj[0] = e0.astype(np.float32)

    for i in range(M):
        u_wrench = mpc.solve(x, x_ref)
        x_next   = np.array(f_hybrid(x, u_wrench)).flatten()
        traj[i + 1] = (x_next - x_ref).astype(np.float32)
        x = x_next

    return traj


def sample_error_states(N: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform samples in domain, excluding δ-ball around origin."""
    out: list[np.ndarray] = []
    while sum(len(a) for a in out) < N:
        e    = rng.uniform(-DOMAIN_HALF, DOMAIN_HALF, size=(N * 2, nx))
        mask = np.linalg.norm(e, axis=1) > DELTA
        out.append(e[mask])
    return np.concatenate(out)[:N]


def collect_rollouts(N: int, M: int, x_ref: np.ndarray,
                     seed: int = 0) -> np.ndarray:
    """Collect N rollouts of M steps each via live NMPC."""
    rng   = np.random.default_rng(seed)
    e0s   = sample_error_states(N, rng)
    trajs = np.empty((N, M + 1, nx), dtype=np.float32)

    print(f"\n{'─'*60}")
    print(f"Collecting {N} rollouts × {M} steps  "
          f"({M * DT_CTRL:.2f}s each) …")
    print(f"Domain: ±{DOMAIN_HALF[:3]} m  ±{np.degrees(DOMAIN_HALF[3:6]).round(0)}°")
    print(f"Tip: for faster collection pipe in MuJoCo trajectories via "
          f"load_mujoco_trajs().\n")

    t0 = time.time()
    for i, e0 in enumerate(e0s):
        trajs[i] = rollout_one(e0, x_ref, M)
        if (i + 1) % max(1, N // 10) == 0:
            elapsed = time.time() - t0
            eta     = elapsed / (i + 1) * (N - i - 1)
            print(f"  [{i+1:5d}/{N}]  "
                  f"{elapsed/60:.1f}min elapsed  "
                  f"{eta/60:.1f}min ETA  "
                  f"‖e₀‖={np.linalg.norm(e0):.3f}")

    print(f"\nDone — {(time.time()-t0)/60:.1f} min\n{'─'*60}\n")
    return trajs


def load_mujoco_trajs(npy_path: str, M: int) -> np.ndarray:
    """
    Fast alternative: load pre-recorded MuJoCo state trajectories.

    Run mpc_hybrid.py with many random starting positions, save `states`
    (shape: N_traj × T_steps × 12, absolute states), then point here.
    Converts absolute → error states and truncates/pads to M+1 steps.

    Usage:
        trajs_train = load_mujoco_trajs("mujoco_trajs.npy", M)
    """
    raw = np.load(npy_path).astype(np.float32)    # (N, T, 12)
    N, T, _ = raw.shape
    T_use = min(T, M + 1)
    trajs = np.zeros((N, M + 1, nx), dtype=np.float32)
    trajs[:, :T_use, :] = raw[:, :T_use, :] - X_REF
    print(f"Loaded {N} MuJoCo trajectories ({T_use} steps each) from {npy_path}")
    return trajs


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Training
# ══════════════════════════════════════════════════════════════════════════════
def train(trajs:      np.ndarray,
          phi_net:    ResidualNet,
          sigma_net:  WeightNet,
          n_epochs:   int   = N_EPOCHS,
          batch_size: int   = BATCH,
          lr:         float = LR) -> list[float]:

    trajs_t = torch.tensor(trajs, dtype=torch.float32, device=DEVICE)
    N       = len(trajs_t)

    opt   = torch.optim.Adam(
        list(phi_net.parameters()) + list(sigma_net.parameters()), lr=lr
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    history:   list[float] = []
    log_every = max(1, n_epochs // 20)

    print(f"Training φ + σ  for {n_epochs} epochs "
          f"(batch={batch_size}, lr={lr}) …\n")

    for epoch in range(1, n_epochs + 1):
        phi_net.train(); sigma_net.train()

        idx     = torch.randperm(N, device=DEVICE)
        ep_loss = 0.0
        n_viol  = 0

        for start in range(0, N, batch_size):
            batch = trajs_t[idx[start:start + batch_size]]
            F     = generalised_decrease(batch, phi_net, sigma_net)
            loss  = torch.relu(F).mean()

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(phi_net.parameters()) + list(sigma_net.parameters()), 1.0
            )
            opt.step()

            ep_loss += loss.item() * len(batch)
            n_viol  += int((F > 0).sum().item())

        sched.step()
        avg_loss = ep_loss / N
        pct_ok   = 100.0 * (1.0 - n_viol / N)
        history.append(avg_loss)

        if epoch % log_every == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{n_epochs}  "
                  f"loss={avg_loss:.6f}  "
                  f"satisfying={pct_ok:.1f}%  "
                  f"lr={sched.get_last_lr()[0]:.2e}")

    print()
    return history


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Evaluation
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(trajs_test: np.ndarray,
             phi_net:    ResidualNet,
             sigma_net:  WeightNet) -> dict:
    phi_net.eval(); sigma_net.eval()

    trajs_t   = torch.tensor(trajs_test, dtype=torch.float32, device=DEVICE)
    F_chunks: list[torch.Tensor] = []
    for start in range(0, len(trajs_t), 512):
        F_chunks.append(
            generalised_decrease(trajs_t[start:start + 512], phi_net, sigma_net)
        )
    F_all = torch.cat(F_chunks).numpy()

    pct_ok   = 100.0 * float(np.mean(F_all <= 0))
    max_viol = float(np.max(F_all))
    mean_F   = float(np.mean(F_all))

    print(f"{'─'*60}")
    print(f"Evaluation on {len(F_all):,} test rollouts")
    print(f"  Satisfying F(e) ≤ 0 : {pct_ok:.2f}%")
    print(f"  Mean F              : {mean_F:.4f}")
    print(f"  Max violation       : {max_viol:.4f}")
    print(f"{'─'*60}\n")

    return {"pct_ok": pct_ok, "max_viol": max_viol,
            "mean_F": mean_F, "F_all": F_all}


@torch.no_grad()
def weight_concentration(trajs_test: np.ndarray,
                         sigma_net:  WeightNet,
                         n_samples:  int = 5_000) -> np.ndarray:
    """Mean normalised weight per timestep — paper Table 2 analogue."""
    sigma_net.eval()
    n  = min(n_samples, len(trajs_test))
    e0 = torch.tensor(trajs_test[:n, 0, :], dtype=torch.float32, device=DEVICE)
    en = e0 / torch.tensor(STATE_SCALE, dtype=torch.float32)
    w  = sigma_net(en)
    return (w / w.sum(dim=1, keepdim=True)).mean(dim=0).numpy()


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Go / no-go checks
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def run_checks(phi_net:   ResidualNet,
               sigma_net: WeightNet,
               eval_res:  dict,
               w_conc:    np.ndarray) -> bool:
    """
    Six checks.  All must pass for a valid certificate.
    Prints PASS/FAIL for each with a concrete fix if it fails.
    """
    results: list[bool] = []

    def check(name: str, passed: bool, fix: str) -> None:
        tag = "✓  PASS" if passed else "✗  FAIL"
        print(f"  {tag}  {name}")
        if not passed:
            print(f"           → {fix}")
        results.append(passed)

    print(f"\n{'═'*60}")
    print("  Go / No-Go Certificate Checks")
    print(f"{'═'*60}")

    # 1. 100% satisfaction on test set
    pct = eval_res["pct_ok"]
    check(
        f"F(e) ≤ 0 on 100% of test states  (got {pct:.2f}%)",
        pct == 100.0,
        f"Increase M (currently {M}), increase N_TRAIN, or reduce DOMAIN_HALF",
    )

    # 2. Mean F clearly negative
    mF = eval_res["mean_F"]
    check(
        f"Mean F clearly negative  (got {mF:.4f}, want < −0.5)",
        mF < -0.5,
        "Increase N_EPOCHS or reduce ALPHA_BAR — certificate is too marginal",
    )

    # 3. Max violation ≤ 0
    mv = eval_res["max_viol"]
    check(
        f"Max violation ≤ 0  (got {mv:.4f})",
        mv <= 0.0,
        "Increase M or reduce DOMAIN_HALF to shrink the problem",
    )

    # 4. V(0) ≈ 0
    V0_val = compute_V(torch.zeros(1, nx), phi_net).item()
    check(
        f"V(0) ≈ 0  (got {V0_val:.6f}, want < 1e-3)",
        V0_val < 1e-3,
        "φ(0) subtraction broken in compute_V — check _ORIGIN_NORM logic",
    )

    # 5. V grows away from origin
    e_mid = torch.tensor(DOMAIN_HALF[None] * 0.5, dtype=torch.float32)
    e_far = torch.tensor(DOMAIN_HALF[None] * 0.9, dtype=torch.float32)
    V_mid = compute_V(e_mid, phi_net).item()
    V_far = compute_V(e_far, phi_net).item()
    check(
        f"V grows with ‖e‖  (V_mid={V_mid:.1f}, V_far={V_far:.1f}, V(0)={V0_val:.3f})",
        V_far > V_mid > V0_val,
        "V is not bowl-shaped — neural residual may dominate; reduce LR or BETA",
    )

    # 6. Weight concentration back-loaded
    last_q = float(np.array_split(w_conc, 5)[-1].sum())
    check(
        f"Weights back-loaded  (last quintile = {last_q:.3f}, want > 0.30)",
        last_q > 0.30,
        f"Increase M — {M} steps ({M*DT_CTRL:.2f}s) may not cover full transient",
    )

    all_pass = all(results)
    n_fail   = sum(1 for r in results if not r)
    print(f"{'═'*60}")
    if all_pass:
        print("  ✓  ALL CHECKS PASSED — certificate is valid.")
        print(f"     Domain: ±{DOMAIN_HALF[:3]} m  "
              f"±{np.degrees(DOMAIN_HALF[3:6]).round(0)}°  "
              f"±{DOMAIN_HALF[6:9]} m/s")
        print(f"     Next: run with --expand to certify a larger region.")
    else:
        print(f"  ✗  {n_fail}/{len(results)} checks failed — see actions above.")
    print(f"{'═'*60}\n")
    return all_pass


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Visualisation
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def plot_certificate(phi_net:    ResidualNet,
                     sigma_net:  WeightNet,
                     trajs_test: np.ndarray,
                     history:    list[float],
                     eval_res:   dict,
                     w_conc:     np.ndarray,
                     save_path:  Path = OUT_DIR / "certificate.png") -> None:
    phi_net.eval(); sigma_net.eval()

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(
        f"Generalised Lyapunov Certificate — Quadrotor NMPC  "
        f"[M={M} steps,  {eval_res['pct_ok']:.2f}% valid]",
        fontsize=14, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    # (A) Training loss
    ax = fig.add_subplot(gs[0, 0])
    ax.semilogy(history, color="#2563EB", lw=1.5)
    ax.set_title("Training loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Mean ReLU(F)")
    ax.grid(True, alpha=0.3)

    # (B) Step-weight concentration
    ax = fig.add_subplot(gs[0, 1])
    ax.bar(np.arange(1, len(w_conc) + 1), w_conc,
           color="#7C3AED", alpha=0.8, width=0.85)
    ax.set_title("Step-weight σᵢ  (healthy = back-loaded)")
    ax.set_xlabel("Horizon step i"); ax.set_ylabel("Mean normalised weight")
    ax.grid(True, alpha=0.3, axis="y")

    # (C) F distribution
    ax = fig.add_subplot(gs[0, 2])
    F_all = eval_res["F_all"]
    ax.hist(F_all, bins=80, color="#059669", alpha=0.8, density=True)
    ax.axvline(0, color="red", lw=1.5, ls="--", label="F=0")
    ax.set_title(f"F(e) distribution\n{eval_res['pct_ok']:.2f}% satisfy F≤0")
    ax.set_xlabel("F(e)"); ax.legend(); ax.grid(True, alpha=0.3)

    # (D) V along sample trajectories
    ax = fig.add_subplot(gs[0, 3])
    rng = np.random.default_rng(42)
    for i in rng.choice(len(trajs_test), size=10, replace=False):
        traj   = torch.tensor(trajs_test[i], dtype=torch.float32)
        V_traj = compute_V(traj, phi_net).numpy()
        ax.plot(V_traj, alpha=0.65, lw=1.1)
    ax.set_title("V(e) along trajectories\n(non-monotonic OK, trend ↓)")
    ax.set_xlabel("Step"); ax.set_ylabel("V(e)")
    ax.grid(True, alpha=0.3)

    # 2-D slices
    SLICES = [
        ("x – z position",    0, 2, DOMAIN_HALF[0], DOMAIN_HALF[2],
         "x error (m)",       "z error (m)"),
        ("roll – pitch",      3, 4, DOMAIN_HALF[3], DOMAIN_HALF[4],
         "roll error (rad)",  "pitch error (rad)"),
        ("vx – vz",           6, 8, DOMAIN_HALF[6], DOMAIN_HALF[8],
         "vx error (m/s)",    "vz error (m/s)"),
        ("x – roll coupling", 0, 3, DOMAIN_HALF[0], DOMAIN_HALF[3],
         "x error (m)",       "roll error (rad)"),
    ]
    N_GRID = 60

    # (E–H) V slices
    for col, (title, i1, i2, lim1, lim2, xl, yl) in enumerate(SLICES):
        ax = fig.add_subplot(gs[1, col])
        g1 = np.linspace(-lim1, lim1, N_GRID)
        g2 = np.linspace(-lim2, lim2, N_GRID)
        G1, G2  = np.meshgrid(g1, g2)
        V_grid  = np.zeros((N_GRID, N_GRID), dtype=np.float32)
        for r in range(N_GRID):
            E = np.zeros((N_GRID, nx), dtype=np.float32)
            E[:, i1] = G1[r]; E[:, i2] = G2[r]
            V_grid[r] = compute_V(torch.tensor(E), phi_net).numpy()
        cf = ax.contourf(G1, G2, V_grid, levels=20, cmap="plasma")
        ax.contour(G1, G2, V_grid, levels=10,
                   colors="white", linewidths=0.4, alpha=0.5)
        fig.colorbar(cf, ax=ax, shrink=0.85)
        ax.set_title(f"V(e)  [{title}]", fontsize=9)
        ax.set_xlabel(xl, fontsize=8); ax.set_ylabel(yl, fontsize=8)

    # (I–L) F scatter on test set (green = OK, red = violation)
    F_vals = eval_res["F_all"]
    e0s    = trajs_test[:, 0, :]
    vmin   = float(np.percentile(F_vals, 2))
    vmax   = float(np.percentile(F_vals, 98))
    for col, (title, i1, i2, lim1, lim2, xl, yl) in enumerate(SLICES):
        ax = fig.add_subplot(gs[2, col])
        sc = ax.scatter(e0s[:, i1], e0s[:, i2],
                        c=F_vals, cmap="RdYlGn_r",
                        vmin=vmin, vmax=vmax,
                        s=2, alpha=0.4, rasterized=True)
        fig.colorbar(sc, ax=ax, shrink=0.85)
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.axvline(0, color="k", lw=0.5, ls="--")
        ax.set_title(f"F(e)  [{title}]\ngreen=OK  red=violation", fontsize=9)
        ax.set_xlabel(xl, fontsize=8); ax.set_ylabel(yl, fontsize=8)
        ax.set_xlim(-lim1, lim1); ax.set_ylim(-lim2, lim2)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure → {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Main
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    print("=" * 60)
    print("  Generalised Lyapunov Certificate — Quadrotor NMPC  v2")
    print("=" * 60)
    print(f"  Horizon M        : {M} steps  ({M*DT_CTRL:.2f}s)")
    print(f"  Training rollouts: {N_TRAIN}")
    print(f"  Test rollouts    : {N_TEST}")
    print(f"  Epochs           : {N_EPOCHS}")
    print(f"  Domain (pos)     : ±{DOMAIN_HALF[:3]} m")
    print(f"  Domain (att)     : ±{np.degrees(DOMAIN_HALF[3:6]).round(1)}°")
    print(f"  Domain (vel)     : ±{DOMAIN_HALF[6:9]} m/s")
    print(f"  Cache key        : {_cache_key}")
    if args.expand:
        print("  *** EXPAND MODE — using larger domain ***")
    print()

    # 1. Collect / load rollouts ───────────────────────────────────────────────
    use_cache = (TRAIN_CACHE.exists() and TEST_CACHE.exists()
                 and (args.skip_rollouts or not args.fast))

    if use_cache:
        reason = "--skip-rollouts" if args.skip_rollouts else "cache hit"
        print(f"Loading cached rollouts ({reason}) …")
        trajs_train = np.load(TRAIN_CACHE)
        trajs_test  = np.load(TEST_CACHE)
        print(f"  Train: {trajs_train.shape}  Test: {trajs_test.shape}\n")
    else:
        trajs_train = collect_rollouts(N_TRAIN, M, X_REF, seed=0)
        trajs_test  = collect_rollouts(N_TEST,  M, X_REF, seed=1)
        if not args.fast:
            np.save(TRAIN_CACHE, trajs_train)
            np.save(TEST_CACHE,  trajs_test)
            print(f"Cached → {TRAIN_CACHE.name}  /  {TEST_CACHE.name}\n")

    # 2. Build networks ────────────────────────────────────────────────────────
    phi_net   = ResidualNet(ne=nx, hidden=128, depth=3).to(DEVICE)
    sigma_net = WeightNet(ne=nx, M=M, hidden=64).to(DEVICE)
    n_phi     = sum(p.numel() for p in phi_net.parameters())
    n_sig     = sum(p.numel() for p in sigma_net.parameters())
    print(f"Parameters: φ={n_phi:,}  σ={n_sig:,}  total={n_phi+n_sig:,}\n")

    # 3. Train ─────────────────────────────────────────────────────────────────
    history = train(trajs_train, phi_net, sigma_net,
                    n_epochs=N_EPOCHS, batch_size=BATCH, lr=LR)

    # 4. Evaluate ──────────────────────────────────────────────────────────────
    eval_res = evaluate(trajs_test, phi_net, sigma_net)
    w_conc   = weight_concentration(trajs_test, sigma_net)

    # Weight concentration table
    print("Step-weight concentration (fraction per quintile):")
    for lbl, b in zip(
        ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"],
        np.array_split(w_conc, 5),
    ):
        bar = "█" * int(b.sum() * 40)
        print(f"  {lbl} : {b.sum():.3f}  {bar}")
    print()

    # 5. Go / no-go ────────────────────────────────────────────────────────────
    all_pass = run_checks(phi_net, sigma_net, eval_res, w_conc)

    # 6. Save checkpoint ───────────────────────────────────────────────────────
    ckpt = OUT_DIR / "certificate.pt"
    torch.save({
        "phi_state":     phi_net.state_dict(),
        "sigma_state":   sigma_net.state_dict(),
        "M":             M,
        "alpha_bar":     ALPHA_BAR,
        "beta":          BETA,
        "QF_diag":       QF_DIAG,
        "domain_half":   DOMAIN_HALF,
        "state_scale":   STATE_SCALE,
        "x_ref":         X_REF,
        "eval_pct_ok":   eval_res["pct_ok"],
        "eval_max_viol": eval_res["max_viol"],
        "all_pass":      all_pass,
        "cache_key":     _cache_key,
    }, ckpt)
    print(f"Checkpoint saved → {ckpt}\n")

    # 7. Visualise ─────────────────────────────────────────────────────────────
    plot_certificate(phi_net, sigma_net, trajs_test,
                     history, eval_res, w_conc)


if __name__ == "__main__":
    main()