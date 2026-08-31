"""
Warm-start providers for ``flash_co2_h2o_salt_kv``.

Two strategies are provided, both implementing the same callable protocol:

    guess = provider(T, P_bar, z_co2, m_tot)
    # Returns WarmStartGuess or None

    if guess is not None:
        result = flash_co2_h2o_salt_kv(
            T, P_bar, z_co2, m_tot,
            K_init     = guess.K_init,
            sol_aq_x0  = guess.sol_aq_x0,
            sol_c_x0   = guess.sol_c_x0,
        )

Strategies
----------
ScanTableWarmStart
    Interpolates K-values and Newton initial states from the pre-computed
    scan table (scan_v4_table.npz, shipped with the repository; the legacy
    4-D scan_v3 format is also supported).
    Uses 4-D linear interpolation in (T, log₁₀P, z, ms) space with
    nearest-neighbour NaN-fill for single-phase cells.
    *Fastest* warm-start — no model inference, just array lookups.

Usage
-----
    from ecpa.warmstart import ScanTableWarmStart

    ws = ScanTableWarmStart.load("results/scan_v4_table.npz")
    result = flash_co2_h2o_salt_kv(T, P, z, ms, warm_start=ws)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt


# ── Return type ────────────────────────────────────────────────────────────────

@dataclass
class WarmStartGuess:
    """
    Initial-guess bundle for ``flash_co2_h2o_salt_kv``.

    Attributes
    ----------
    K_init : (K1, K4)
        K-value pair: K₁ = x1c/x1w (H₂O), K₄ = x4c/x4w (CO₂).
    sol_aq_x0 : ndarray(3,) — [Z_aq, ε_r, χ₁w]
        Warm-start vector for the aqueous-phase Newton inner solve.
    sol_c_x0 : ndarray(2,) — [Z_c, χ₁c]
        Warm-start vector for the CO₂-rich-phase Newton inner solve.
    is_two_phase : bool
        Hint from the provider — not authoritative.  The flash solver always
        runs stability analysis; this is only used to skip obviously
        single-phase conditions in bulk scan loops.
    source : str
        Human-readable label: 'table' or 'nn'.
    """
    K_init:       Tuple[float, float]
    sol_aq_x0:    Optional[np.ndarray]
    sol_c_x0:     Optional[np.ndarray]
    is_two_phase: bool
    source:       str = "unknown"


# ── Table-based warm start ─────────────────────────────────────────────────────

class ScanTableWarmStart:
    """
    Warm-start provider backed by pre-computed flash tables.

    Primary table (4-D, required)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ``scan_v3_table.npz`` — 70T × 50P × 25z × 14ms regular grid covering
    the full ms range.  Interpolation axes: T [K], log₁₀P [bar], z_CO₂,
    ms [mol/kg].

    CPA table (3-D, optional but recommended)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ``cpa_table.npz`` — 139T × 50P × 49z at ms≈0, built by combining the
    scan_v3 ms=0 slice with the CPA enrichment data (2.5 K T-resolution,
    double z-resolution).  Used automatically when ``m_tot < ms_cpa_thresh``
    (default 0.05 mol/kg).  Auto-detected from ``cpa_table.npz`` in the same
    directory as the primary table if ``cpa_table_path`` is not given.

    Interpolated quantities (NaN-cells nearest-neighbour filled before
    building interpolators):
        ln K₁, ln K₄    — K-values in log-space (more linear)
        Z_aq, ε_r, χ₁w  — aqueous Newton state
        Z_c,  χ₁c       — CO₂-rich Newton state
        is_two_phase     — nearest-neighbour stability flag

    Parameters
    ----------
    table_path : str or Path
        Path to scan_v3_table.npz (4-D primary table).
    cpa_table_path : str, Path, or None
        Path to cpa_table.npz (3-D CPA table).  If None, auto-detected from
        the same directory as ``table_path``.  Pass ``False`` to disable.
    ms_cpa_thresh : float
        Queries with m_tot < ms_cpa_thresh use the CPA 3-D table.
        Default 0.05 mol/kg.
    clip_K : bool
        Clip K-values to physical bounds before returning.
    """

    _MS_CPA_DEFAULT = 0.05   # mol/kg — switch to CPA table below this

    def __init__(
        self,
        table_path: str | Path,
        *,
        cpa_table_path=None,   # None = auto-detect; False = disable
        ms_cpa_thresh: float = _MS_CPA_DEFAULT,
        clip_K: bool = True,
    ):
        self._clip_K = clip_K
        self._ms_cpa_thresh = ms_cpa_thresh
        self._table_3d = False   # set properly by _build_4d
        path = Path(table_path)
        self._build_4d(path)

        # Resolve CPA table path
        if cpa_table_path is False:
            self._cpa_interps = None
        else:
            if cpa_table_path is None:
                auto = path.parent / "cpa_table.npz"
                cpa_table_path = auto if auto.exists() else False
            if cpa_table_path is not False:
                self._build_cpa(Path(cpa_table_path))
            else:
                self._cpa_interps = None

    @classmethod
    def load(cls, table_path: str | Path, **kw) -> "ScanTableWarmStart":
        """Convenience constructor. Same as ``ScanTableWarmStart(path)``."""
        return cls(table_path, **kw)

    # ── Internal build helpers ─────────────────────────────────────────────────

    @staticmethod
    def _build_interps(arrays_dict: dict, axes: tuple) -> dict:
        """Build one RegularGridInterpolator per named array."""
        kw = dict(method="linear", bounds_error=False, fill_value=None)
        interps = {k: RegularGridInterpolator(axes, v, **kw)
                   for k, v in arrays_dict.items() if k != "stable"}
        interps["stable"] = RegularGridInterpolator(
            axes, arrays_dict["stable"].astype(float),
            method="nearest", bounds_error=False, fill_value=0.0,
        )
        return interps

    @staticmethod
    def _nn_fill_3d(arrays: list, valid: np.ndarray) -> list:
        invalid = ~valid
        if not invalid.any():
            return arrays
        _, nn_idx = distance_transform_edt(invalid, return_indices=True)
        i0, i1, i2 = nn_idx
        return [np.where(valid, a, a[i0, i1, i2]) for a in arrays]

    @staticmethod
    def _nn_fill_4d(arrays: list, valid: np.ndarray) -> list:
        invalid = ~valid
        if not invalid.any():
            return arrays
        _, nn_idx = distance_transform_edt(invalid, return_indices=True)
        i0, i1, i2, i3 = nn_idx
        return [np.where(valid, a, a[i0, i1, i2, i3]) for a in arrays]

    def _build_4d(self, path: Path) -> None:
        """Build interpolators from a scan table.

        Supports both formats:
          * 4-D (legacy scan_v3): arrays shaped (nT, nP, nz, nms), with z_grid key.
          * 3-D (scan_v4, z-free): arrays shaped (nT, nP, nms), no z_grid key.
            In this case the z argument is ignored at query time.
        """
        data  = np.load(path, allow_pickle=False)
        valid = data["is_two_phase"].astype(bool)

        x1w = data["x1w"].astype(float)
        x1c = data["x1c"].astype(float)
        x4w = data["x4w"].astype(float)
        x4c = data["x4c"].astype(float)

        with np.errstate(divide="ignore", invalid="ignore"):
            lnK1 = np.where(valid & (x1w > 1e-10) & (x1c > 1e-10),
                            np.log(x1c / x1w), np.nan)
            lnK4 = np.where(valid & (x4w > 1e-10) & (x4c > 1e-10),
                            np.log(x4c / x4w), np.nan)

        logP = np.log10(data["P_grid"].astype(float))

        if "z_grid" in data:
            # ── 4-D legacy format ─────────────────────────────────────────────
            filled = self._nn_fill_4d(
                [lnK1, lnK4,
                 data["Z_aq"].astype(float), data["epsr"].astype(float),
                 data["chi1w"].astype(float), data["Z_c"].astype(float),
                 data["chi1c"].astype(float)],
                valid,
            )
            lnK1, lnK4, Z_aq, epsr, chi1w, Z_c, chi1c = filled
            axes = (data["T_grid"].astype(float), logP,
                    data["z_grid"].astype(float),
                    data["ms_grid"].astype(float))
            self._4d_interps = self._build_interps(
                dict(lnK1=lnK1, lnK4=lnK4, Z_aq=Z_aq, epsr=epsr,
                     chi1w=chi1w, Z_c=Z_c, chi1c=chi1c, stable=valid),
                axes,
            )
            self._table_3d = False
        else:
            # ── 3-D format (scan_v4, z-free) ─────────────────────────────────
            filled = self._nn_fill_3d(
                [lnK1, lnK4,
                 data["Z_aq"].astype(float), data["epsr"].astype(float),
                 data["chi1w"].astype(float), data["Z_c"].astype(float),
                 data["chi1c"].astype(float)],
                valid,
            )
            lnK1, lnK4, Z_aq, epsr, chi1w, Z_c, chi1c = filled
            axes = (data["T_grid"].astype(float), logP,
                    data["ms_grid"].astype(float))
            self._4d_interps = self._build_interps(
                dict(lnK1=lnK1, lnK4=lnK4, Z_aq=Z_aq, epsr=epsr,
                     chi1w=chi1w, Z_c=Z_c, chi1c=chi1c, stable=valid),
                axes,
            )
            self._table_3d = True

    def _build_cpa(self, path: Path) -> None:
        """Build interpolators from the 3-D CPA (ms≈0) table."""
        data  = np.load(path, allow_pickle=False)
        valid = data["is_two_phase"].astype(bool)   # (nT,nP,nz)

        filled = self._nn_fill_3d(
            [data["lnK1"].astype(float), data["lnK4"].astype(float),
             data["Z_aq"].astype(float), data["epsr"].astype(float),
             data["chi1w"].astype(float), data["Z_c"].astype(float),
             data["chi1c"].astype(float)],
            valid,
        )
        lnK1, lnK4, Z_aq, epsr, chi1w, Z_c, chi1c = filled

        logP = np.log10(data["P_grid"].astype(float))
        axes = (data["T_grid"].astype(float), logP,
                data["z_grid"].astype(float))

        self._cpa_interps = self._build_interps(
            dict(lnK1=lnK1, lnK4=lnK4, Z_aq=Z_aq, epsr=epsr,
                 chi1w=chi1w, Z_c=Z_c, chi1c=chi1c, stable=valid),
            axes,
        )

    # ── Internal scalar lookup ─────────────────────────────────────────────────

    def _eval(self, interps: dict, pt: np.ndarray) -> WarmStartGuess:
        K1    = float(math.exp(interps["lnK1"](pt)[0]))
        K4    = float(math.exp(interps["lnK4"](pt)[0]))
        Z_aq  = float(interps["Z_aq"](pt)[0])
        epsr  = float(interps["epsr"](pt)[0])
        chi1w = float(interps["chi1w"](pt)[0])
        Z_c   = float(interps["Z_c"](pt)[0])
        chi1c = float(interps["chi1c"](pt)[0])
        is2ph = bool(interps["stable"](pt)[0] > 0.5)
        if self._clip_K:
            K1 = float(np.clip(K1, 1e-6, 1.0 - 1e-9))
            K4 = float(np.clip(K4, 1.0 + 1e-9, 1e6))
        src = "table_cpa" if interps is self._cpa_interps else "table"
        return WarmStartGuess(
            K_init       = (K1, K4),
            sol_aq_x0    = np.array([Z_aq, epsr, chi1w], dtype=float),
            sol_c_x0     = np.array([Z_c,  chi1c],       dtype=float),
            is_two_phase = is2ph,
            source       = src,
        )

    # ── Callable interface ─────────────────────────────────────────────────────

    def __call__(
        self,
        T: float,
        P_bar: float,
        z_co2: float,
        m_tot: float,
    ) -> WarmStartGuess:
        """
        Return an interpolated warm-start guess.

        Routes to the denser 3-D CPA table when m_tot < ms_cpa_thresh and
        the CPA table is loaded; otherwise uses the 4-D eCPA table.
        Always returns a WarmStartGuess (never None).
        """
        logP = math.log10(max(P_bar, 1e-9))
        if m_tot < self._ms_cpa_thresh and self._cpa_interps is not None:
            pt      = np.array([[T, logP, z_co2]])
            interps = self._cpa_interps
        elif self._table_3d:
            # 3-D table (scan_v4): z_co2 not used as axis
            pt      = np.array([[T, logP, m_tot]])
            interps = self._4d_interps
        else:
            pt      = np.array([[T, logP, z_co2, m_tot]])
            interps = self._4d_interps
        return self._eval(interps, pt)

    def batch(
        self,
        T:     np.ndarray,
        P_bar: np.ndarray,
        z_co2: np.ndarray,
        m_tot: np.ndarray,
    ) -> dict:
        """
        Vectorised lookup for N points simultaneously.

        Returns dict: K1, K4 (N,); sol_aq_x0 (N,3); sol_c_x0 (N,2);
        is_two_phase (N,bool); source (N, object).
        """
        T     = np.asarray(T,     dtype=float)
        P_bar = np.asarray(P_bar, dtype=float)
        z_co2 = np.asarray(z_co2, dtype=float)
        m_tot = np.asarray(m_tot, dtype=float)
        N     = len(T)
        logP  = np.log10(np.maximum(P_bar, 1e-9))

        K1    = np.empty(N);  K4    = np.empty(N)
        Z_aq  = np.empty(N);  epsr  = np.empty(N); chi1w = np.empty(N)
        Z_c   = np.empty(N);  chi1c = np.empty(N)
        is2ph = np.zeros(N, dtype=bool)
        source = np.empty(N, dtype=object)

        cpa_mask = (m_tot < self._ms_cpa_thresh) & (self._cpa_interps is not None)

        for mask, interps, label in [
            (cpa_mask,  self._cpa_interps, "table_cpa"),
            (~cpa_mask, self._4d_interps,  "table"),
        ]:
            if interps is None:
                # CPA table not loaded — fall back to 4D for those points too
                interps = self._4d_interps
                label   = "table"
                mask    = np.ones(N, dtype=bool) if not (~cpa_mask).any() else mask
            if not mask.any():
                continue
            if label == "table_cpa":
                pts = np.column_stack([T[mask], logP[mask], z_co2[mask]])
            elif self._table_3d:
                pts = np.column_stack([T[mask], logP[mask], m_tot[mask]])
            else:
                pts = np.column_stack([T[mask], logP[mask], z_co2[mask], m_tot[mask]])
            K1[mask]    = np.exp(interps["lnK1"](pts))
            K4[mask]    = np.exp(interps["lnK4"](pts))
            Z_aq[mask]  = interps["Z_aq"](pts)
            epsr[mask]  = interps["epsr"](pts)
            chi1w[mask] = interps["chi1w"](pts)
            Z_c[mask]   = interps["Z_c"](pts)
            chi1c[mask] = interps["chi1c"](pts)
            is2ph[mask] = interps["stable"](pts) > 0.5
            source[mask]= label

        if self._clip_K:
            K1 = np.clip(K1, 1e-6, 1.0 - 1e-9)
            K4 = np.clip(K4, 1.0 + 1e-9, 1e6)

        return {
            "K1":           K1,
            "K4":           K4,
            "sol_aq_x0":    np.column_stack([Z_aq, epsr, chi1w]),
            "sol_c_x0":     np.column_stack([Z_c, chi1c]),
            "is_two_phase": is2ph,
            "source":       source,
        }
