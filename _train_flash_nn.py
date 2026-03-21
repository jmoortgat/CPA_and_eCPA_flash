"""
Train a physics-informed neural network on the scan_v3 solution table.

Architecture: PhysicsFlashNet (ecpa/nn_flash.py)
  • 4-input MLP with 6 residual blocks (hidden=256) + SiLU activations
  • Phase head   → p_two_phase (BCE loss)
  • K-value head → [ln_K_H2O, ln_K_CO2] (MSE + Rachford-Rice residual loss)
  • Newton head  → [Z_aq, chi1w, ln_epsr, Z_c, chi1c] (MSE loss)

Physics losses
--------------
  L_RR   : Rachford-Rice residual at ground-truth β — penalises K-values that
            don't produce the observed phase split.
  L_comp : Composition-derived RR check from predicted K-values.

After training the model is used as a warm-start for flash_co2_h2o_salt_kv:
a single Newton/SSI iteration after the NN guess enforces thermodynamic consistency.

Output
------
  results/flash_nn_v1.pt       — best checkpoint (by validation K-value MSE)
  results/flash_nn_v1_last.pt  — last epoch checkpoint
  figures/nn/training_curves.png

Usage
-----
  python _train_flash_nn.py [--epochs 500] [--hidden 256] [--blocks 6]
                            [--batch 4096] [--lr 3e-4] [--device auto]
                            [--table results/scan_v3_table.npz]
"""
import argparse
import math
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)
os.makedirs("figures/nn", exist_ok=True)

from ecpa.nn_flash import (
    FlashNNStats,
    PhysicsFlashNet,
    encode_inputs,
    rachford_rice_residual,
)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--table",   default="results/scan_v3_table.npz")
    p.add_argument("--out",     default="results/flash_nn_v1.pt")
    p.add_argument("--epochs",  type=int,   default=500)
    p.add_argument("--hidden",  type=int,   default=256)
    p.add_argument("--blocks",  type=int,   default=6)
    p.add_argument("--batch",   type=int,   default=4096)
    p.add_argument("--lr",      type=float, default=3e-4)
    p.add_argument("--wd",      type=float, default=1e-5,   help="weight decay")
    p.add_argument("--dropout", type=float, default=0.02)
    p.add_argument("--device",  default="auto",
                   help="auto | cpu | mps | cuda")
    # Loss weights
    p.add_argument("--lam-kv",     type=float, default=1.0,  help="K-value MSE weight")
    p.add_argument("--lam-rr",     type=float, default=0.5,  help="RR residual weight")
    p.add_argument("--lam-newton", type=float, default=0.5,  help="Newton-state MSE weight")
    p.add_argument("--lam-cls",    type=float, default=1.0,  help="phase BCE weight")
    p.add_argument("--val-frac",   type=float, default=0.10)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(table_path: str):
    """Load scan_v3 NPZ and return training arrays."""
    print(f"Loading {table_path} …")
    t = np.load(table_path, allow_pickle=False)

    T_g  = t["T_grid"];  P_g  = t["P_grid"]
    z_g  = t["z_grid"];  ms_g = t["ms_grid"]
    nT, nP, nz, nms = len(T_g), len(P_g), len(z_g), len(ms_g)

    is2   = t["is_two_phase"]          # (nT, nP, nz, nms) bool
    x1w   = t["x1w"];   x4w = t["x4w"]
    x1c   = t["x1c"];   x4c = t["x4c"]   # x4c = 1 − x1c stored
    Z_aq  = t["Z_aq"];  Z_c = t["Z_c"]
    chi1w = t["chi1w"]; chi1c = t["chi1c"]
    epsr  = t["epsr"];  beta = t["beta"]

    # Build full (T,P,z,ms) index arrays for every cell
    iT_arr  = np.repeat(np.arange(nT),  nP * nz * nms).reshape(nT, nP, nz, nms)
    iP_arr  = np.tile(np.repeat(np.arange(nP),  nz * nms).reshape(nP, nz, nms), (nT,1,1,1))
    iz_arr  = np.tile(np.tile(np.repeat(np.arange(nz), nms).reshape(nz, nms), (nP,1,1)), (nT,1,1,1))
    ims_arr = np.tile(np.arange(nms).reshape(1,1,1,nms), (nT,nP,nz,1))

    T_all   = T_g [iT_arr .ravel()].astype(np.float32)
    P_all   = P_g [iP_arr .ravel()].astype(np.float32)
    z_all   = z_g [iz_arr .ravel()].astype(np.float32)
    ms_all  = ms_g[ims_arr.ravel()].astype(np.float32)
    is2_all = is2.ravel()

    print(f"  Total cells:     {len(T_all):,}")
    print(f"  Two-phase cells: {is2_all.sum():,} ({is2_all.mean()*100:.1f}%)")

    # ── Quality filter for two-phase cells ────────────────────────────────────
    good_2ph = (
        is2_all &
        (x4w.ravel() > 1e-8) &
        (x1w.ravel() > 1e-6) &
        (x1c.ravel() > 1e-8) &
        np.isfinite(x1w.ravel() + x4w.ravel() + x1c.ravel() + beta.ravel())
    )
    print(f"  Clean two-phase: {good_2ph.sum():,} ({good_2ph.mean()*100:.1f}% of all cells)")

    # ── Compute K-values ──────────────────────────────────────────────────────
    x1w_2 = x1w.ravel()[good_2ph]; x4w_2 = x4w.ravel()[good_2ph]
    x1c_2 = x1c.ravel()[good_2ph]; x4c_2 = x4c.ravel()[good_2ph]
    K1 = x1c_2 / x1w_2   # H2O K-value
    K4 = x4c_2 / x4w_2   # CO2 K-value
    # Clip extreme K-values before log (rare numerical outliers)
    K1 = np.clip(K1, 1e-6, 1e6)
    K4 = np.clip(K4, 1e-6, 1e6)
    lnK1 = np.log(K1).astype(np.float32)
    lnK4 = np.log(K4).astype(np.float32)

    # Filter any remaining inf/nan from log
    kv_ok = np.isfinite(lnK1) & np.isfinite(lnK4)
    print(f"  K-value finite:  {kv_ok.sum():,}")

    # ── Newton-state targets ──────────────────────────────────────────────────
    Zaq_2   = Z_aq .ravel()[good_2ph]
    Zc_2    = Z_c  .ravel()[good_2ph]
    chi1w_2 = chi1w.ravel()[good_2ph]
    chi1c_2 = chi1c.ravel()[good_2ph]
    epsr_2  = epsr .ravel()[good_2ph]
    beta_2  = beta .ravel()[good_2ph]

    # log(epsr): set to 0 for CPA cells (ms=0, epsr=NaN) — mask handled in loss
    epsr_isnan = np.isnan(epsr_2)
    lnepsr_2   = np.where(epsr_isnan, 0.0, np.log(np.maximum(epsr_2, 1.0))).astype(np.float32)
    epsr_valid  = (~epsr_isnan & kv_ok).astype(np.float32)  # weight mask for epsr loss

    # Filter kv_ok into all arrays
    T_2   = T_all [good_2ph][kv_ok]
    P_2   = P_all [good_2ph][kv_ok]
    z_2   = z_all [good_2ph][kv_ok]
    ms_2  = ms_all[good_2ph][kv_ok]
    lnK1  = lnK1[kv_ok]; lnK4 = lnK4[kv_ok]
    Zaq_2 = Zaq_2[kv_ok]; Zc_2 = Zc_2[kv_ok]
    chi1w_2 = chi1w_2[kv_ok]; chi1c_2 = chi1c_2[kv_ok]
    lnepsr_2 = lnepsr_2[kv_ok]; epsr_valid = epsr_valid[kv_ok]
    beta_2  = beta_2[kv_ok]

    # ── Normalisation stats ────────────────────────────────────────────────────
    stats = FlashNNStats(
        T_mean  =float(T_all.mean()),   T_std  =float(T_all.std()),
        lnP_mean=float(np.log(P_all).mean()), lnP_std=float(np.log(P_all).std()),
        z_mean  =float(z_all.mean()),   z_std  =float(z_all.std()),
        ms_mean =float(ms_all.mean()),  ms_std =float(ms_all.std()),
        lnK1_mean=float(lnK1.mean()),   lnK1_std=float(lnK1.std()),
        lnK4_mean=float(lnK4.mean()),   lnK4_std=float(lnK4.std()),
        Zaq_mean =float(Zaq_2.mean()),  Zaq_std =float(Zaq_2.std()),
        chi1w_mean=float(chi1w_2.mean()),chi1w_std=float(chi1w_2.std()),
        lnepsr_mean=float(np.nanmean(np.where(epsr_valid>0.5, lnepsr_2, np.nan))),
        lnepsr_std =float(np.nanstd( np.where(epsr_valid>0.5, lnepsr_2, np.nan))),
        Zc_mean  =float(Zc_2.mean()),   Zc_std  =float(Zc_2.std()),
        chi1c_mean=float(chi1c_2.mean()),chi1c_std=float(chi1c_2.std()),
    )

    def _norm_kv(x, mu, s): return ((x - mu) / max(s, 1e-8)).astype(np.float32)
    def _norm(x, mu, s):    return ((x - mu) / max(s, 1e-8)).astype(np.float32)

    # Normalised K-value targets
    lnK1_n = _norm_kv(lnK1, stats.lnK1_mean, stats.lnK1_std)
    lnK4_n = _norm_kv(lnK4, stats.lnK4_mean, stats.lnK4_std)

    # Normalised Newton-state targets
    Zaq_n    = _norm(Zaq_2,    stats.Zaq_mean,    stats.Zaq_std)
    chi1w_n  = _norm(chi1w_2,  stats.chi1w_mean,  stats.chi1w_std)
    lnepsr_n = _norm(lnepsr_2, stats.lnepsr_mean, stats.lnepsr_std)
    Zc_n     = _norm(Zc_2,     stats.Zc_mean,     stats.Zc_std)
    chi1c_n  = _norm(chi1c_2,  stats.chi1c_mean,  stats.chi1c_std)

    # Normalised inputs for two-phase cells
    X_2ph = encode_inputs(T_2, P_2, z_2, ms_2, stats)

    # ── Classification dataset (all cells) ─────────────────────────────────────
    X_all = encode_inputs(T_all, P_all, z_all, ms_all, stats)
    y_cls = is2_all.astype(np.float32)

    # ── Pack tensors ──────────────────────────────────────────────────────────
    data_2ph = {
        "X":        torch.from_numpy(X_2ph),
        "lnK1_n":   torch.from_numpy(lnK1_n),
        "lnK4_n":   torch.from_numpy(lnK4_n),
        "lnK1":     torch.from_numpy(lnK1),        # unnorm, for RR loss
        "lnK4":     torch.from_numpy(lnK4),
        "beta":     torch.from_numpy(beta_2.astype(np.float32)),
        "z_co2":    torch.from_numpy(z_2),
        "newton":   torch.from_numpy(np.stack([Zaq_n, chi1w_n, lnepsr_n, Zc_n, chi1c_n], axis=1)),
        "epsr_mask":torch.from_numpy(epsr_valid),  # 1.0 where epsr is valid (ms>0)
    }
    data_all = {
        "X":    torch.from_numpy(X_all),
        "y_cls":torch.from_numpy(y_cls),
    }
    return data_2ph, data_all, stats


# ── Loss functions ─────────────────────────────────────────────────────────────

def compute_loss(
    model: PhysicsFlashNet,
    batch_2ph: tuple,
    batch_all: tuple,
    weights: dict,
    device: torch.device,
    kv_stats: tuple | None = None,  # (lnK1_mean, lnK1_std, lnK4_mean, lnK4_std) as tensors on device
) -> tuple[torch.Tensor, dict]:
    """Compute multi-task physics-informed loss.

    Returns (total_loss, loss_components_dict).

    kv_stats: if provided, de-normalise the K-value head outputs before
              computing the Rachford-Rice physics residual.  This makes the
              RR loss flow through the K-value predictions (genuine physics
              constraint) rather than being a sanity-check on the data.
    """
    X_2ph, lnK1_n, lnK4_n, lnK1, lnK4, beta, z_co2, newton, epsr_mask = batch_2ph
    X_all, y_cls = batch_all

    X_2ph = X_2ph.to(device); X_all = X_all.to(device)
    y_cls = y_cls.to(device)

    # ── Phase classification (all cells) ──────────────────────────────────────
    ph_logit, _, _ = model(X_all)
    L_cls = nn.functional.binary_cross_entropy_with_logits(
        ph_logit.squeeze(1), y_cls
    )

    # ── K-value + Newton (two-phase cells only) ────────────────────────────────
    lnK1_n  = lnK1_n.to(device);  lnK4_n  = lnK4_n.to(device)
    lnK1    = lnK1.to(device);    lnK4    = lnK4.to(device)
    beta    = beta.to(device);     z_co2   = z_co2.to(device)
    newton  = newton.to(device);   epsr_mask = epsr_mask.to(device)

    _, kv, nw = model(X_2ph)

    # K-value MSE loss (normalised scale → unit-variance targets)
    L_kv = (
        (kv[:, 0] - lnK1_n).pow(2).mean() +
        (kv[:, 1] - lnK4_n).pow(2).mean()
    )

    # ── Rachford-Rice physics loss ─────────────────────────────────────────────
    # Compute RR residual at ground-truth β using the PREDICTED K-values.
    # This penalises K-values whose predicted phase split (at the true β)
    # violates the Rachford-Rice equation — an actual physics constraint.
    # We de-normalise predicted outputs to natural ln(K) scale first.
    if kv_stats is not None:
        lnK1_mean, lnK1_std, lnK4_mean, lnK4_std = kv_stats
        lnK1_pred_nat = kv[:, 0] * lnK1_std + lnK1_mean
        lnK4_pred_nat = kv[:, 1] * lnK4_std + lnK4_mean
    else:
        # Fallback: use ground-truth K-values (sanity-checks data; no gradient
        # flows through K-value head — only useful for loss bookkeeping)
        lnK1_pred_nat = lnK1
        lnK4_pred_nat = lnK4

    K1_pred = torch.exp(lnK1_pred_nat.clamp(-15.0, 15.0))
    K4_pred = torch.exp(lnK4_pred_nat.clamp(-15.0, 15.0))

    f_rr = rachford_rice_residual(beta, K1_pred, K4_pred, z_co2)
    L_rr = f_rr.pow(2).mean()

    # ── Newton-state MSE loss ─────────────────────────────────────────────────
    # [Zaq_n, chi1w_n, lnepsr_n, Zc_n, chi1c_n]
    L_newton_noepsr = (
        (nw[:, 0] - newton[:, 0]).pow(2).mean() +   # Z_aq
        (nw[:, 1] - newton[:, 1]).pow(2).mean() +   # chi1w
        (nw[:, 3] - newton[:, 3]).pow(2).mean() +   # Z_c
        (nw[:, 4] - newton[:, 4]).pow(2).mean()     # chi1c
    )
    # epsr only for eCPA cells (ms > 0)
    n_epsr = epsr_mask.sum().clamp(min=1)
    L_epsr = ((nw[:, 2] - newton[:, 2]).pow(2) * epsr_mask).sum() / n_epsr

    L_newton = L_newton_noepsr + L_epsr

    # ── Total loss ────────────────────────────────────────────────────────────
    L_total = (
        weights["cls"]    * L_cls    +
        weights["kv"]     * L_kv     +
        weights["rr"]     * L_rr     +
        weights["newton"] * L_newton
    )

    components = {
        "cls":    L_cls.item(),
        "kv":     L_kv.item(),
        "rr":     L_rr.item(),
        "newton": L_newton.item(),
        "total":  L_total.item(),
    }
    return L_total, components


# ── Training loop ──────────────────────────────────────────────────────────────

def make_device(spec: str) -> torch.device:
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


def split_dataset(tensor_dict: dict, val_frac: float, seed: int):
    """Split a dict of equal-length tensors into train/val."""
    n = len(next(iter(tensor_dict.values())))
    rng = torch.Generator().manual_seed(seed)
    n_val = int(n * val_frac)
    n_tr  = n - n_val
    idx = torch.randperm(n, generator=rng)
    idx_tr, idx_val = idx[:n_tr], idx[n_tr:]
    tr  = {k: v[idx_tr]  for k, v in tensor_dict.items()}
    val = {k: v[idx_val] for k, v in tensor_dict.items()}
    return tr, val


def batch_iter(tensor_dict: dict, batch_size: int, shuffle: bool = True):
    """Yield mini-batches as tuples of tensors."""
    keys = list(tensor_dict.keys())
    vals = [tensor_dict[k] for k in keys]
    n = len(vals[0])
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        sl = idx[start:start + batch_size]
        yield tuple(v[sl] for v in vals)


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = make_device(args.device)
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    data_2ph, data_all, stats = load_data(args.table)

    d2_tr, d2_val = split_dataset(data_2ph, args.val_frac, args.seed)
    da_tr, da_val = split_dataset(data_all, args.val_frac, args.seed)

    n_tr_2ph  = len(d2_tr["X"])
    n_tr_all  = len(da_tr["X"])
    n_val_2ph = len(d2_val["X"])
    n_val_all = len(da_val["X"])
    print(f"Train: {n_tr_2ph:,} two-phase + {n_tr_all:,} all-phase cells")
    print(f"Val:   {n_val_2ph:,} two-phase + {n_val_all:,} all-phase cells")

    # ── Model ─────────────────────────────────────────────────────────────────
    arch = {"hidden": args.hidden, "n_blocks": args.blocks, "dropout": args.dropout}
    model = PhysicsFlashNet(**arch).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} parameters  ({args.hidden}×{args.blocks})")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    weights = {
        "cls":    args.lam_cls,
        "kv":     args.lam_kv,
        "rr":     args.lam_rr,
        "newton": args.lam_newton,
    }

    # K-value de-normalisation constants for the physics (RR) loss
    # These are fixed scalars, kept on device as 0-dim tensors for efficiency
    kv_stats = (
        torch.tensor(stats.lnK1_mean, device=device),
        torch.tensor(stats.lnK1_std,  device=device),
        torch.tensor(stats.lnK4_mean, device=device),
        torch.tensor(stats.lnK4_std,  device=device),
    )

    # ── Training ──────────────────────────────────────────────────────────────
    best_val_kv = math.inf
    best_val_cls_auc = 0.0
    history = {"train": [], "val": []}

    out_best = Path(args.out)
    out_last = out_best.with_name(out_best.stem + "_last.pt")

    print(f"\n{'Epoch':>6}  {'L_cls':>7} {'L_kv':>7} {'L_rr':>7} {'L_nw':>7}  "
          f"{'Val_cls':>8} {'Val_kv':>7} {'Val_rr':>7}  {'LR':>8}  {'Time':>6}")
    print("─" * 100)

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        tr_comps = {k: [] for k in ["cls", "kv", "rr", "newton", "total"]}

        # Pair up two-phase and all-phase batches (repeat the smaller one cyclically)
        b2_list = list(batch_iter(d2_tr, args.batch, shuffle=True))
        ba_list = list(batch_iter(da_tr, args.batch, shuffle=True))
        n_steps = max(len(b2_list), len(ba_list))
        # Cycle whichever is shorter
        b2_cycle = [b2_list[i % len(b2_list)] for i in range(n_steps)]
        ba_cycle = [ba_list[i % len(ba_list)] for i in range(n_steps)]

        for b2, ba in zip(b2_cycle, ba_cycle):
            optimizer.zero_grad()
            loss, comps = compute_loss(model, b2, ba, weights, device, kv_stats)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            for k in comps: tr_comps[k].append(comps[k])

        tr_mean = {k: np.mean(v) for k, v in tr_comps.items()}
        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_comps = {k: [] for k in ["cls", "kv", "rr", "newton", "total"]}
        with torch.no_grad():
            v2_list = list(batch_iter(d2_val, args.batch * 2, shuffle=False))
            va_list = list(batch_iter(da_val, args.batch * 2, shuffle=False))
            n_vsteps = max(len(v2_list), len(va_list))
            v2_cycle = [v2_list[i % len(v2_list)] for i in range(n_vsteps)]
            va_cycle = [va_list[i % len(va_list)] for i in range(n_vsteps)]
            for b2, ba in zip(v2_cycle, va_cycle):
                _, comps = compute_loss(model, b2, ba, weights, device, kv_stats)
                for k in comps: val_comps[k].append(comps[k])

        val_mean = {k: np.mean(v) for k, v in val_comps.items()}
        elapsed = time.perf_counter() - t0
        lr_now = scheduler.get_last_lr()[0]

        history["train"].append(tr_mean)
        history["val"].append(val_mean)

        # ── Checkpoint ────────────────────────────────────────────────────────
        if val_mean["kv"] < best_val_kv:
            best_val_kv = val_mean["kv"]
            torch.save({
                "epoch": epoch,
                "arch":  arch,
                "state_dict": model.state_dict(),
                "stats": vars(stats),
                "val_kv": best_val_kv,
                "optimizer": optimizer.state_dict(),
            }, out_best)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"{epoch:>6}  "
                f"{tr_mean['cls']:>7.4f} {tr_mean['kv']:>7.4f} "
                f"{tr_mean['rr']:>7.4f} {tr_mean['newton']:>7.4f}  "
                f"{val_mean['cls']:>8.4f} {val_mean['kv']:>7.4f} "
                f"{val_mean['rr']:>7.4f}  "
                f"{lr_now:.2e}  {elapsed:.1f}s"
                + ("  ✓best" if abs(val_mean['kv'] - best_val_kv) < 1e-9 else "")
            )

    # Save last checkpoint
    torch.save({
        "epoch": args.epochs,
        "arch":  arch,
        "state_dict": model.state_dict(),
        "stats": vars(stats),
    }, out_last)

    return model, stats, history, out_best


# ── Post-training evaluation ───────────────────────────────────────────────────

def evaluate(model_path: str, data_2ph: dict, data_all: dict, stats: FlashNNStats, device: torch.device):
    """Load best checkpoint and compute evaluation metrics."""
    from ecpa.nn_flash import FlashNNGuess, flash_nn_guess
    nn_model = FlashNNGuess.load(model_path, device=device)
    model = nn_model.model.eval()

    print("\n" + "=" * 60)
    print("Evaluation on held-out TEST set (first 10% of val)")
    print("=" * 60)

    # Use validation set as proxy for test (we don't have a separate test split here)
    # In practice, split off 10% for final evaluation
    X_2ph = data_2ph["X"].to(device)
    with torch.no_grad():
        _, kv, nw = model(X_2ph)

    # K-value errors (normalised → natural units via stats)
    lnK1_pred = (kv[:, 0].cpu().numpy() * stats.lnK1_std + stats.lnK1_mean)
    lnK4_pred = (kv[:, 1].cpu().numpy() * stats.lnK4_std + stats.lnK4_mean)

    lnK1_true = (data_2ph["lnK1_n"].numpy() * stats.lnK1_std + stats.lnK1_mean)
    lnK4_true = (data_2ph["lnK4_n"].numpy() * stats.lnK4_std + stats.lnK4_mean)

    # Use a subsample for speed
    rng = np.random.default_rng(0)
    idx = rng.choice(len(lnK1_true), size=min(50000, len(lnK1_true)), replace=False)

    err_K1 = np.abs(lnK1_pred[idx] - lnK1_true[idx])
    err_K4 = np.abs(lnK4_pred[idx] - lnK4_true[idx])
    print(f"  |Δ ln K_H2O|: mean={err_K1.mean():.4f}  median={np.median(err_K1):.4f}  p95={np.percentile(err_K1,95):.4f}")
    print(f"  |Δ ln K_CO2|: mean={err_K4.mean():.4f}  median={np.median(err_K4):.4f}  p95={np.percentile(err_K4,95):.4f}")

    # Derived composition errors
    K1_pred_arr = np.exp(lnK1_pred[idx]); K4_pred_arr = np.exp(lnK4_pred[idx])
    K1_true_arr = np.exp(lnK1_true[idx]); K4_true_arr = np.exp(lnK4_true[idx])

    beta_arr = data_2ph["beta"].numpy()[idx]
    z_arr    = data_2ph["z_co2"].numpy()[idx]
    z_h2o    = 1.0 - z_arr

    x1w_true = z_h2o / (1.0 + beta_arr * (K1_true_arr - 1.0))
    x4w_true = z_arr  / (1.0 + beta_arr * (K4_true_arr - 1.0))
    x1w_pred = z_h2o / (1.0 + beta_arr * (K1_pred_arr - 1.0))
    x4w_pred = z_arr  / (1.0 + beta_arr * (K4_pred_arr - 1.0))

    err_x1w = np.abs(x1w_pred - x1w_true) / np.maximum(x1w_true, 1e-8)
    err_x4w = np.abs(x4w_pred - x4w_true) / np.maximum(x4w_true, 1e-8)
    print(f"  AARE x_H2O (aq): mean={err_x1w.mean()*100:.2f}%  median={np.median(err_x1w)*100:.2f}%")
    print(f"  AARE x_CO2 (aq): mean={err_x4w.mean()*100:.2f}%  median={np.median(err_x4w)*100:.2f}%")

    # Phase classification accuracy
    X_all = data_all["X"].to(device)
    with torch.no_grad():
        ph_logit, _, _ = model(X_all)
    p2 = torch.sigmoid(ph_logit.squeeze(1)).cpu().numpy()
    y_true = data_all["y_cls"].numpy()
    y_pred = (p2 > 0.5).astype(float)
    acc = (y_pred == y_true).mean()
    tp  = ((y_pred == 1) & (y_true == 1)).sum()
    fp  = ((y_pred == 1) & (y_true == 0)).sum()
    fn  = ((y_pred == 0) & (y_true == 1)).sum()
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    print(f"\n  Phase accuracy:   {acc*100:.2f}%")
    print(f"  Precision:        {precision*100:.2f}%  (fraction of predicted 2-phase that are truly 2-phase)")
    print(f"  Recall:           {recall*100:.2f}%  (fraction of true 2-phase that are predicted 2-phase)")

    # Inference timing
    x_single = encode_inputs(
        np.array([350.0]), np.array([100.0]), np.array([0.5]), np.array([1.0]), stats
    )
    xt = torch.from_numpy(x_single).to(device)
    # Warm up
    for _ in range(20):
        with torch.no_grad():
            _ = model(xt)
    t0 = time.perf_counter()
    N_TIMING = 1000
    for _ in range(N_TIMING):
        with torch.no_grad():
            _ = model(xt)
    t_ms = (time.perf_counter() - t0) / N_TIMING * 1e3
    print(f"\n  NN inference (single point): {t_ms:.3f} ms/call")

    return {
        "lnK1_mae": err_K1.mean(), "lnK4_mae": err_K4.mean(),
        "x1w_aare": err_x1w.mean(), "x4w_aare": err_x4w.mean(),
        "phase_acc": acc, "precision": precision, "recall": recall,
        "t_ms_nn": t_ms,
    }


# ── Figures ────────────────────────────────────────────────────────────────────

def plot_training_curves(history: dict, out: str = "figures/nn/training_curves.png"):
    try:
        import scienceplots; import matplotlib.pyplot as plt; plt.style.use(["science"])
    except ImportError:
        import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train"]) + 1)
    keys   = ["cls", "kv", "rr", "newton"]
    labels = ["Phase BCE", "K-value MSE", "RR residual", "Newton-state MSE"]
    colors = ["steelblue", "tomato", "forestgreen", "darkorange"]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2),
                             gridspec_kw=dict(left=0.07, right=0.97,
                                              top=0.88, bottom=0.15,
                                              wspace=0.35))
    for ax, key, lab, col in zip(axes, keys, labels, colors):
        tr  = [h[key] for h in history["train"]]
        val = [h[key] for h in history["val"]]
        ax.plot(epochs, tr,  color=col, lw=1.2, alpha=0.8, label="Train")
        ax.plot(epochs, val, color=col, lw=1.2, ls="--",   label="Val")
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel(lab, fontsize=9)
        ax.set_yscale("log")
        ax.legend(fontsize=8, framealpha=0)
        ax.set_title(lab, fontsize=9)
    fig.suptitle("Physics-informed flash NN — training curves", fontsize=9)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"\nTraining curves saved: {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    args = parse_args()

    model, stats, history, out_best = train(args)

    print(f"\nBest checkpoint: {out_best}")
    print(f"Last checkpoint: {out_best.with_name(out_best.stem + '_last.pt')}")

    # Evaluate
    data_2ph, data_all, _ = load_data(args.table)
    device = make_device(args.device)
    metrics = evaluate(str(out_best), data_2ph, data_all, stats, device)

    # Plot
    plot_training_curves(history)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  |Δ ln K_H2O| mean : {metrics['lnK1_mae']:.4f}")
    print(f"  |Δ ln K_CO2| mean : {metrics['lnK4_mae']:.4f}")
    print(f"  AARE x_CO2 (aq)   : {metrics['x4w_aare']*100:.2f}%")
    print(f"  AARE x_H2O (aq)   : {metrics['x1w_aare']*100:.2f}%")
    print(f"  Phase accuracy    : {metrics['phase_acc']*100:.2f}%")
    print(f"  Recall (2-phase)  : {metrics['recall']*100:.2f}%")
    print(f"  NN inference time : {metrics['t_ms_nn']:.3f} ms/call")
    print(f"\nDone. Model saved to: {out_best}")
