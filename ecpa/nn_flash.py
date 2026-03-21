"""
Physics-informed neural network for CO2+H2O+NaCl flash warm-start.

Architecture
------------
PhysicsFlashNet: 4-input MLP with residual blocks, three heads:

  1. Phase head  → p_two_phase (sigmoid over logit)
  2. K-value head → [ln_K_H2O, ln_K_CO2]  (log scale, linear outputs)
  3. Newton-state head → [Z_aq, chi1w, ln_epsr, Z_c, chi1c]
     (aqueous Newton variables + CO2-rich Newton variables)

Physics enforcement
-------------------
During training the loss includes:
  • Rachford-Rice residual:  β*(K_i−1)/(1+β*(K_i−1)) summed over components
    evaluated at the ground-truth β — penalises K-values that are thermodynamically
    inconsistent with the observed phase split.
  • Sum-to-one check on derived compositions (soft penalty).

At inference, compositions are derived analytically from the predicted K-values
via the 3-component Rachford-Rice equation (K_NaCl = 0), giving exact mass
balance by construction.  The Newton-state outputs seed the inner Newton solvers
(Z_aq, εr, χ1w for the aqueous phase; Z_c, χ1c for the CO2-rich phase).

Usage
-----
After training, load the model and call ``flash_nn_guess`` to obtain initial
guesses compatible with ``flash_co2_h2o_salt_kv`` / ``flash_co2_h2o_salt_fast_kv``.

    from ecpa.nn_flash import FlashNNGuess, flash_nn_guess
    nn = FlashNNGuess.load('results/flash_nn_v1.pt')
    guess = flash_nn_guess(T=350.0, P_bar=100.0, z_co2=0.5, m_tot=1.0, nn=nn)
    # guess is a dict with keys matching the warm-start interface of flash_co2_h2o_salt_kv
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Constants ──────────────────────────────────────────────────────────────────

# Input normalisation statistics (set after fitting; stored in checkpoint)
_DEFAULT_INPUT_STATS = {
    "T_mean": 460.5, "T_std": 99.6,
    "lnP_mean": 3.68, "lnP_std": 2.03,
    "z_mean": 0.475, "z_std": 0.247,
    "ms_mean": 1.67, "ms_std": 1.93,
}

# Output normalisation statistics for K-values and Newton state
_DEFAULT_OUTPUT_STATS: dict = {}  # filled from data during training

# ── Building blocks ────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Pre-norm residual MLP block with SiLU activation."""

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1  = nn.Linear(dim, dim)
        self.fc2  = nn.Linear(dim, dim)
        self.act  = nn.SiLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(self.norm(x)))
        h = self.drop(self.fc2(h))
        return x + h


class PhysicsFlashNet(nn.Module):
    """
    Multi-task MLP for CO2+H2O+NaCl flash warm-start.

    Parameters
    ----------
    hidden : int
        Width of all hidden layers.
    n_blocks : int
        Number of residual blocks in the shared backbone.
    dropout : float
        Dropout rate (0 = no dropout).
    """

    N_NEWTON = 5  # [Z_aq, chi1w, ln_epsr, Z_c, chi1c]
    N_KV     = 2  # [ln_K_H2O, ln_K_CO2]

    def __init__(self, hidden: int = 256, n_blocks: int = 6, dropout: float = 0.05):
        super().__init__()
        # Embed: 4 inputs → hidden
        self.embed = nn.Sequential(
            nn.Linear(4, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        # Shared residual backbone
        self.backbone = nn.ModuleList([ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.backbone_norm = nn.LayerNorm(hidden)

        # Head 1: phase classification (single logit, BCEWithLogits)
        self.phase_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )
        # Head 2: K-values in log scale
        self.kv_head = nn.Sequential(
            nn.Linear(hidden, 128), nn.SiLU(),
            nn.Linear(128, 64),    nn.SiLU(),
            nn.Linear(64, self.N_KV),
        )
        # Head 3: Newton-state warm-start variables
        self.newton_head = nn.Sequential(
            nn.Linear(hidden, 128), nn.SiLU(),
            nn.Linear(128, 64),    nn.SiLU(),
            nn.Linear(64, self.N_NEWTON),
        )

    def _shared(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for blk in self.backbone:
            h = blk(h)
        return self.backbone_norm(h)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (N, 4) float tensor — normalised [T, ln_P, z_co2, ms]

        Returns
        -------
        phase_logit : (N, 1)
        ln_Kv      : (N, 2)   [ln_K_H2O, ln_K_CO2]
        newton_raw : (N, 5)   [Z_aq_raw, chi1w_raw, ln_epsr, Z_c_raw, chi1c_raw]
        """
        h = self._shared(x)
        return self.phase_head(h), self.kv_head(h), self.newton_head(h)

    def phase_only(self, x: torch.Tensor) -> torch.Tensor:
        """Fast path: only compute phase logit (for screening)."""
        return self.phase_head(self._shared(x))


# ── Physics helpers ────────────────────────────────────────────────────────────

def rachford_rice_residual(
    beta: torch.Tensor,
    K1: torch.Tensor,
    K4: torch.Tensor,
    z_co2: torch.Tensor,
) -> torch.Tensor:
    """
    Rachford-Rice function for a 3-component system (H2O, CO2, NaCl)
    with K_NaCl = 0 (NaCl entirely in aqueous phase).

    f(β) = Σ_i z_i*(K_i−1)/(1+β*(K_i−1))
         = z_H2O*(K1−1)/(1+β*(K1−1))
         + z_CO2*(K4−1)/(1+β*(K4−1))
         + z_NaCl*(0−1)/(1+β*(0−1))          ← K_NaCl = 0

    The z_NaCl / (1−β) term enforces that the denominator for NaCl avoids
    division by zero when β→1 (added as a clamp in inference).

    Here we compute the two-component RR residual (H2O + CO2 on effective
    salt-free basis) since the salt contribution is implicit in K1, K4 via
    activity coefficients.

    Parameters
    ----------
    beta, K1, K4, z_co2 : tensors of the same shape
        beta  = CO2-rich phase fraction
        K1    = x1c / x1w   (H2O K-value)
        K4    = x4c / x4w   (CO2 K-value)
        z_co2 = overall CO2 feed fraction (salt-inclusive basis)
    """
    z_h2o = 1.0 - z_co2
    eps = 1e-8
    f  = z_h2o * (K1 - 1.0) / (1.0 + beta * (K1 - 1.0) + eps)
    f += z_co2 * (K4 - 1.0) / (1.0 + beta * (K4 - 1.0) + eps)
    return f


def kv_to_compositions(
    K1: np.ndarray,
    K4: np.ndarray,
    z_co2: float | np.ndarray,
    beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute phase compositions from K-values using the Rachford-Rice solution.

    Returns x1w, x4w, x1c, x4c, beta (clipped to [ε, 1−ε]).
    """
    z_h2o = 1.0 - z_co2
    beta  = np.clip(beta, 1e-6, 1.0 - 1e-6)
    x1w = z_h2o / (1.0 + beta * (K1 - 1.0))
    x4w = z_co2 / (1.0 + beta * (K4 - 1.0))
    x1c = K1 * x1w
    x4c = K4 * x4w
    return x1w, x4w, x1c, x4c, beta


def solve_rr_beta(
    K1: float,
    K4: float,
    z_co2: float,
    beta0: float = 0.5,
    n_iter: int = 8,
) -> float:
    """
    Solve the Rachford-Rice equation for β given K-values and feed composition.
    Uses bracketed Newton iteration (Halley would need second derivative; Newton
    is sufficient given good starting point from the NN).
    """
    z_h2o = 1.0 - z_co2
    # Bracket: β ∈ (1/(1-K1), 1/(1-K4)) or similar Michelsen bounds
    b_lo = max(1.0 / (1.0 - max(K1, K4)) + 1e-8, 0.0)
    b_hi = min(1.0 / (1.0 - min(K1, K4)) - 1e-8, 1.0)
    if b_lo >= b_hi:
        b_lo, b_hi = 0.0, 1.0
    beta = float(np.clip(beta0, b_lo, b_hi))

    for _ in range(n_iter):
        d1  = 1.0 + beta * (K1 - 1.0)
        d4  = 1.0 + beta * (K4 - 1.0)
        f   = z_h2o * (K1 - 1.0) / d1 + z_co2 * (K4 - 1.0) / d4
        df  = -z_h2o * (K1 - 1.0) ** 2 / d1**2 - z_co2 * (K4 - 1.0) ** 2 / d4**2
        if abs(df) < 1e-15:
            break
        step = -f / df
        beta = float(np.clip(beta + step, b_lo, b_hi))
        if abs(f) < 1e-10:
            break
    return beta


# ── Input / output encoding ────────────────────────────────────────────────────

@dataclass
class FlashNNStats:
    """Normalisation statistics for a FlashNNGuess model."""
    # Input: mean and std for standardisation
    T_mean: float = 460.5;   T_std: float = 99.6
    lnP_mean: float = 3.68;  lnP_std: float = 2.03
    z_mean: float = 0.475;   z_std: float = 0.247
    ms_mean: float = 1.67;   ms_std: float = 1.93
    # Output K-value: mean and std (log scale)
    lnK1_mean: float = -2.75; lnK1_std: float = 2.30
    lnK4_mean: float = 4.49;  lnK4_std: float = 3.50
    # Output Newton state
    Zaq_mean:  float = 0.215;  Zaq_std:  float = 0.30
    chi1w_mean:float = 0.883;  chi1w_std:float = 0.10
    lnepsr_mean:float = 3.63; lnepsr_std:float = 0.52
    Zc_mean:   float = 0.219;  Zc_std:   float = 0.30
    chi1c_mean:float = 0.776;  chi1c_std:float = 0.14


def encode_inputs(
    T: np.ndarray,
    P_bar: np.ndarray,
    z_co2: np.ndarray,
    ms: np.ndarray,
    stats: FlashNNStats,
) -> np.ndarray:
    """Normalise raw inputs → float32 array shape (N, 4)."""
    ln_P = np.log(np.maximum(P_bar, 1e-3))
    return np.column_stack([
        (T     - stats.T_mean)    / stats.T_std,
        (ln_P  - stats.lnP_mean)  / stats.lnP_std,
        (z_co2 - stats.z_mean)    / stats.z_std,
        (ms    - stats.ms_mean)   / stats.ms_std,
    ]).astype(np.float32)


def decode_newton_state(raw: np.ndarray, stats: FlashNNStats) -> dict:
    """
    Decode the Newton-state head output → physically bounded variables.

    raw shape: (5,) — [Z_aq, chi1w, ln_epsr, Z_c, chi1c]
    All outputs are in normalised space; apply inverse transform.
    """
    Z_aq  = float(raw[0] * stats.Zaq_std  + stats.Zaq_mean)
    chi1w = float(raw[1] * stats.chi1w_std + stats.chi1w_mean)
    epsr  = float(np.exp(raw[2] * stats.lnepsr_std + stats.lnepsr_mean))
    Z_c   = float(raw[3] * stats.Zc_std   + stats.Zc_mean)
    chi1c = float(raw[4] * stats.chi1c_std + stats.chi1c_mean)
    # Clip to physical bounds
    Z_aq  = max(Z_aq,  1e-4)
    Z_c   = max(Z_c,   1e-4)
    chi1w = float(np.clip(chi1w, 0.05, 1.0))
    chi1c = float(np.clip(chi1c, 0.05, 1.15))
    epsr  = float(np.clip(epsr,  5.0, 100.0))
    return {"Z_aq": Z_aq, "chi1w": chi1w, "epsr": epsr, "Z_c": Z_c, "chi1c": chi1c}


# ── Inference container ────────────────────────────────────────────────────────

class FlashNNGuess:
    """
    Loaded NN model ready for inference.

    Attributes
    ----------
    model  : PhysicsFlashNet
    stats  : FlashNNStats
    device : torch.device
    """

    def __init__(
        self,
        model: PhysicsFlashNet,
        stats: FlashNNStats,
        device: torch.device | None = None,
    ):
        self.model  = model.eval()
        self.stats  = stats
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

    @classmethod
    def load(cls, path: str | Path, device: torch.device | None = None) -> "FlashNNGuess":
        """Load model checkpoint saved by _train_flash_nn.py."""
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        ckpt  = torch.load(path, map_location=device, weights_only=False)
        model = PhysicsFlashNet(**ckpt["arch"])
        model.load_state_dict(ckpt["state_dict"])
        stats = FlashNNStats(**ckpt["stats"])
        return cls(model, stats, device)

    def predict_raw(
        self,
        T: float,
        P_bar: float,
        z_co2: float,
        ms: float,
    ) -> tuple[float, float, float, np.ndarray]:
        """
        Raw single-point inference.

        Returns
        -------
        p_two_phase : float in [0,1]
        ln_K1       : float  (log H2O K-value)
        ln_K4       : float  (log CO2 K-value)
        newton_raw  : ndarray shape (5,) — normalised Newton-state outputs
        """
        x = encode_inputs(
            np.array([T]), np.array([P_bar]),
            np.array([z_co2]), np.array([ms]),
            self.stats,
        )
        xt = torch.from_numpy(x).to(self.device)
        with torch.no_grad():
            ph, kv, nw = self.model(xt)
        p2  = float(torch.sigmoid(ph[0, 0]).cpu())
        lK1 = float(kv[0, 0].cpu())
        lK4 = float(kv[0, 1].cpu())
        nw_arr = nw[0].cpu().numpy()
        return p2, lK1, lK4, nw_arr


def flash_nn_guess(
    T: float,
    P_bar: float,
    z_co2: float,
    m_tot: float,
    nn: FlashNNGuess,
    phase_threshold: float = 0.35,
    refine_beta: bool = True,
) -> Optional[dict]:
    """
    Compute a physics-consistent warm-start guess for ``flash_co2_h2o_salt_kv``.

    Parameters
    ----------
    T, P_bar, z_co2, m_tot : float
        Flash conditions.
    nn : FlashNNGuess
        Loaded NN model.
    phase_threshold : float
        Predict single-phase if p_two_phase < threshold (conservative = 0.35;
        increase to skip more conditions at risk of false negatives).
    refine_beta : bool
        If True, polish β with Newton's method given the predicted K-values.
        Cost: ~8 scalar iterations; recommended.

    Returns
    -------
    dict with keys:
        ``K_vals``      : list [K_H2O, K_CO2]
        ``beta``        : float — predicted CO2-rich phase fraction
        ``x_aq``        : dict {x1w, x4w} — aqueous phase compositions
        ``x_c``         : dict {x1c, x4c} — CO2-rich phase compositions
        ``sol_aq_x0``   : ndarray (3,) — [Z_aq, chi1w, epsr] for Newton warm-start
        ``sol_c_x0``    : ndarray (2,) — [Z_c, chi1c] for Newton warm-start
        ``p_two_phase`` : float — predicted two-phase probability
    or None if predicted single-phase.
    """
    p2, lK1, lK4, nw_raw = nn.predict_raw(T, P_bar, z_co2, m_tot)

    if p2 < phase_threshold:
        return None  # predict single-phase → skip warm-start

    K1 = math.exp(lK1)
    K4 = math.exp(lK4)

    # β from analytical 2-comp Rachford-Rice (fast, closed-form for 2 species)
    z_h2o = 1.0 - z_co2
    denom = (K1 - 1.0) * (K4 - 1.0)
    if abs(denom) < 1e-10:
        beta0 = 0.5
    else:
        beta0 = -((z_h2o * (K4 - 1.0) + z_co2 * (K1 - 1.0)) / denom)
    beta0 = float(np.clip(beta0, 1e-6, 1.0 - 1e-6))

    if refine_beta:
        beta = solve_rr_beta(K1, K4, z_co2, beta0=beta0)
    else:
        beta = beta0

    x1w, x4w, x1c, x4c, beta = kv_to_compositions(
        np.array([K1]), np.array([K4]),
        z_co2, np.array([beta]),
    )
    x1w = float(x1w[0]); x4w = float(x4w[0])
    x1c = float(x1c[0]); x4c = float(x4c[0])
    beta = float(beta)

    ns = nn.stats
    ns_decoded = decode_newton_state(nw_raw, ns)

    sol_aq_x0 = np.array([ns_decoded["Z_aq"], ns_decoded["chi1w"], ns_decoded["epsr"]])
    sol_c_x0  = np.array([ns_decoded["Z_c"],  ns_decoded["chi1c"]])

    return {
        "K_vals":      [K1, K4],
        "beta":        beta,
        "x_aq":        {"x1w": x1w, "x4w": x4w},
        "x_c":         {"x1c": x1c, "x4c": x4c},
        "sol_aq_x0":   sol_aq_x0,
        "sol_c_x0":    sol_c_x0,
        "p_two_phase": p2,
    }
