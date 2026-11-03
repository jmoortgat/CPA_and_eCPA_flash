"""Sweep grids for the multi-salt eCPA stability+flash benchmark.

Each grid is a generator of ``SweepPoint`` records covering a region of
the (T, P, z, mi) input space. The sweep harness iterates these
generators, distributes them across workers, and writes the converged
phase compositions, inner-state warm-start vectors, and performance
metrics to parquet for later analysis.

Grids
-----
* :data:`GRIDS["smoke"]`     ~ 500 points (validation / CI)
* :data:`GRIDS["standard"]`  ~ 60k--100k points (first real run)
* :data:`GRIDS["full"]`      ~ 1M+ points (full coverage of paper window)

Adjust the resolution arrays at the top of the file to scale up/down
without rewriting the structure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Single-point record consumed by the worker
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SweepPoint:
    """One (T, P, z, mi, convention, tags) input to evaluate."""
    point_id: int
    config_name: str
    feed_family: str           # 'binary_CO2', 'binary_CH4', 'ternary', 'salt-free-vle'
    salt_name: str             # e.g. 'no_salt', 'NaCl_1m', 'mixed_brine_LIU_2011_B1'
    T_K: float
    P_bar: float
    z_H2O: float
    z_CH4: float
    z_CO2: float
    m_Na: float
    m_Cl: float
    m_K: float
    m_Ca: float
    m_Mg: float
    m_SO4: float
    salinity_convention: str


# ---------------------------------------------------------------------------
# Grid resolution arrays
# ---------------------------------------------------------------------------

# Temperature isotherms (K). Span the full multi-salt paper window plus a
# few extra at the low/high ends so we can characterise edge behaviour.
T_FULL = tuple(int(t) for t in np.unique(np.concatenate([
    np.linspace(283.0, 623.0, 19),   # 20 K spacing, ~20 points
])))
T_STANDARD = (283, 308, 323, 348, 373, 398, 423, 448, 473, 498, 523, 573, 623)
T_SMOKE = (323.0, 423.0, 523.0)
# T_DENSE: 3x finer than T_STANDARD across the same 283-623 K window
# (13 -> 39 uniformly-spaced isotherms).
T_DENSE = tuple(float(t) for t in np.linspace(283.0, 623.0, 39))

# Pressures (bar). Log-spaced; covers atmospheric to deep-reservoir.
P_FULL = tuple(float(p) for p in np.geomspace(1.0, 2000.0, 30))
P_STANDARD = tuple(float(p) for p in np.geomspace(1.0, 2000.0, 20))
P_SMOKE = (1.0, 10.0, 100.0, 500.0, 1500.0)
# P_DENSE: 3x finer than P_STANDARD on the same log-spaced 1-2000 bar range
# (20 -> 60 pressures).
P_DENSE = tuple(float(p) for p in np.geomspace(1.0, 2000.0, 60))

# Binary z grids: z_gas ∈ (0, 1).
ZBIN_FULL = tuple(float(z) for z in np.concatenate([
    np.linspace(0.02, 0.20, 7),   # gas-poor end
    np.linspace(0.30, 0.70, 5),   # mid
    np.linspace(0.80, 0.98, 7),   # gas-rich end
]))
ZBIN_STANDARD = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
ZBIN_SMOKE = (0.05, 0.50, 0.90)

# Ternary z grids: (z_CH4, z_CO2) triangle (sum < 1; z_w = 1 - z_CH4 - z_CO2).
def _triangle(n: int) -> tuple[tuple[float, float], ...]:
    """Triangle of n×n - upper-half points with z_CH4 + z_CO2 < 0.98."""
    out = []
    grid = np.linspace(0.02, 0.96, n)
    for zc in grid:
        for zh in grid:
            if zc + zh < 0.98:
                out.append((float(zh), float(zc)))
    return tuple(out)

ZTRI_FULL = _triangle(8)        # ~25 points
ZTRI_STANDARD = _triangle(5)    # ~10 points
ZTRI_SMOKE = _triangle(3)       # ~3 points


# ---------------------------------------------------------------------------
# Salt configurations
# ---------------------------------------------------------------------------
# Each entry: (salt_name, mi-vector). mi = (m_Na, m_Cl, m_K, m_Ca, m_Mg, m_SO4).

# Single-salt at a few representative molalities each.
SINGLE_SALTS = (
    ("no_salt",       np.array([0,   0,   0,   0,   0,   0])),
    ("NaCl_0p5m",     np.array([0.5, 0.5, 0,   0,   0,   0])),
    ("NaCl_1m",       np.array([1.0, 1.0, 0,   0,   0,   0])),
    ("NaCl_3m",       np.array([3.0, 3.0, 0,   0,   0,   0])),
    ("KCl_1m",        np.array([0,   1.0, 1.0, 0,   0,   0])),
    ("KCl_3m",        np.array([0,   3.0, 3.0, 0,   0,   0])),
    ("CaCl2_1m",      np.array([0,   2.0, 0,   1.0, 0,   0])),
    ("CaCl2_3m",      np.array([0,   6.0, 0,   3.0, 0,   0])),
    ("MgCl2_1m",      np.array([0,   2.0, 0,   0,   1.0, 0])),
    ("Na2SO4_1m",     np.array([2.0, 0,   0,   0,   0,   1.0])),
)

# Mixed brines drawn from the paper validation set (Liu 2011 B1, Wang 2014 B1,
# Poulain 2019 B1, ZhaoC 2015 B1). Recipes copied from the experimental cache.
MIXED_BRINES = (
    ("mix_LIU2011_B1",  np.array([0.30, 0.85, 0.23, 0.16, 0, 0])),
    ("mix_WANG2014_B1", np.array([0.43, 1.04, 0,    0.29, 0.02, 0.01])),
    ("mix_synthetic_1", np.array([1.0, 1.5,  0.2, 0.3, 0.1, 0.1])),  # rich brine
    ("mix_synthetic_2", np.array([0.1, 0.3,  0.05, 0.1, 0.05, 0.05])),  # dilute
)

SALT_CONFIGS_FULL = SINGLE_SALTS + MIXED_BRINES
SALT_CONFIGS_STANDARD = SINGLE_SALTS[:6] + MIXED_BRINES[:2]
SALT_CONFIGS_SMOKE = (SINGLE_SALTS[0], SINGLE_SALTS[2])  # no_salt + NaCl_1m


# ---------------------------------------------------------------------------
# Validity domain
# ---------------------------------------------------------------------------
# Empirical convergence-safe envelope established from sweep iterations 1-8:
# the multi-salt eCPA stability+flash framework converges in 100% of points
# tested when both of the following hold:
#
#     P_bar       <= 1000.0
#     min(z_i)    >= 0.10   over partitioning species i with z_i > 0
#
# Outside this envelope the model is at its formulation edge: trace-species
# feeds stress the inner Newton's basin coverage, and very-high-P (>1000 bar)
# states push the aqueous compressibility / permittivity solutions to extremes
# the seed ladder cannot always reach.  Inside the envelope, the v8 patches
# (Iter 1-8 in sweep/notes/iteration_log.md) clear every failure mode observed
# in the standard sweep.

VALIDITY_P_MAX_BAR = 1000.0
VALIDITY_Z_MIN = 0.10   # strict lower bound (z_i > VALIDITY_Z_MIN)


def is_in_validity_domain(P_bar: float, z) -> bool:
    """Return True iff ``(P_bar, z)`` is inside the convergence-safe envelope.

    ``z`` is the 3-tuple of partition-component mole fractions
    ``(z_H2O, z_CH4, z_CO2)``.  Absent species (``z_i == 0``) are
    ignored when computing the minimum present mole fraction.  The
    bound on ``z`` is STRICT (``z_i > 0.10``) so that exactly-0.10
    boundary points -- e.g. ``z = (0.90, 0.10, 0)`` -- which sit on the
    failure surface stay outside the domain.
    """
    if float(P_bar) > VALIDITY_P_MAX_BAR:
        return False
    present_zs = [float(zi) for zi in z if float(zi) > 0.0]
    if not present_zs:
        return False
    return min(present_zs) > VALIDITY_Z_MIN


# ---------------------------------------------------------------------------
# Feed families
# ---------------------------------------------------------------------------

def _binary_co2(z_CO2: float) -> tuple[float, float, float]:
    return (1.0 - z_CO2, 0.0, z_CO2)

def _binary_ch4(z_CH4: float) -> tuple[float, float, float]:
    return (1.0 - z_CH4, z_CH4, 0.0)

def _ternary(zh: float, zc: float) -> tuple[float, float, float]:
    return (1.0 - zh - zc, zh, zc)


# ---------------------------------------------------------------------------
# Grid generator
# ---------------------------------------------------------------------------

def iter_grid(config_name: str) -> Generator[SweepPoint, None, None]:
    """Yield every ``SweepPoint`` for the named config.

    Iteration order: salt outer → feed-family → isotherm → pressure → z.
    This keeps each isotherm's worth of points contiguous, which helps the
    SLURM array chunking remain efficient (similar wall-clocks per chunk).
    """
    if config_name == "smoke":
        Ts, Ps, zbins, ztri, salts = (
            T_SMOKE, P_SMOKE, ZBIN_SMOKE, ZTRI_SMOKE, SALT_CONFIGS_SMOKE)
    elif config_name == "standard":
        Ts, Ps, zbins, ztri, salts = (
            T_STANDARD, P_STANDARD, ZBIN_STANDARD, ZTRI_STANDARD,
            SALT_CONFIGS_STANDARD)
    elif config_name == "full":
        Ts, Ps, zbins, ztri, salts = (
            T_FULL, P_FULL, ZBIN_FULL, ZTRI_FULL, SALT_CONFIGS_FULL)
    elif config_name == "valid":
        # Validity-domain restriction of "standard": every point also
        # satisfies the convergence-safe predicate documented in
        # :func:`is_in_validity_domain`.  The grid arrays are the same;
        # the filter is applied at yield time below.
        Ts, Ps, zbins, ztri, salts = (
            T_STANDARD, P_STANDARD, ZBIN_STANDARD, ZTRI_STANDARD,
            SALT_CONFIGS_STANDARD)
    elif config_name == "valid_dense":
        # Same salts and z-grids as "valid", but 3x finer in each of T
        # and P (39 x 60 T-P cells vs 13 x 20). Used to resolve the
        # phase-boundary detail that the coarser grid smears.
        Ts, Ps, zbins, ztri, salts = (
            T_DENSE, P_DENSE, ZBIN_STANDARD, ZTRI_STANDARD,
            SALT_CONFIGS_STANDARD)
    else:
        raise ValueError(f"unknown config_name: {config_name!r}")

    convention = "as-aqueous"   # paper baseline; sweep_runner can override
    apply_validity_filter = config_name in ("valid", "valid_dense")
    point_id = 0
    for salt_name, mi in salts:
        for T in Ts:
            for P in Ps:
                # Binary CO2-H2O
                for z_CO2 in zbins:
                    z = _binary_co2(z_CO2)
                    if apply_validity_filter and not is_in_validity_domain(P, z):
                        continue
                    yield SweepPoint(
                        point_id=point_id, config_name=config_name,
                        feed_family="binary_CO2", salt_name=salt_name,
                        T_K=float(T), P_bar=float(P),
                        z_H2O=z[0], z_CH4=z[1], z_CO2=z[2],
                        m_Na=float(mi[0]), m_Cl=float(mi[1]),
                        m_K=float(mi[2]), m_Ca=float(mi[3]),
                        m_Mg=float(mi[4]), m_SO4=float(mi[5]),
                        salinity_convention=convention,
                    )
                    point_id += 1
                # Binary CH4-H2O
                for z_CH4 in zbins:
                    z = _binary_ch4(z_CH4)
                    if apply_validity_filter and not is_in_validity_domain(P, z):
                        continue
                    yield SweepPoint(
                        point_id=point_id, config_name=config_name,
                        feed_family="binary_CH4", salt_name=salt_name,
                        T_K=float(T), P_bar=float(P),
                        z_H2O=z[0], z_CH4=z[1], z_CO2=z[2],
                        m_Na=float(mi[0]), m_Cl=float(mi[1]),
                        m_K=float(mi[2]), m_Ca=float(mi[3]),
                        m_Mg=float(mi[4]), m_SO4=float(mi[5]),
                        salinity_convention=convention,
                    )
                    point_id += 1
                # Ternary CH4-CO2-H2O
                for zh, zc in ztri:
                    z = _ternary(zh, zc)
                    if apply_validity_filter and not is_in_validity_domain(P, z):
                        continue
                    yield SweepPoint(
                        point_id=point_id, config_name=config_name,
                        feed_family="ternary", salt_name=salt_name,
                        T_K=float(T), P_bar=float(P),
                        z_H2O=z[0], z_CH4=z[1], z_CO2=z[2],
                        m_Na=float(mi[0]), m_Cl=float(mi[1]),
                        m_K=float(mi[2]), m_Ca=float(mi[3]),
                        m_Mg=float(mi[4]), m_SO4=float(mi[5]),
                        salinity_convention=convention,
                    )
                    point_id += 1


def estimate_size(config_name: str) -> dict:
    """Return a dict summarising point counts per family for ``config_name``.

    For ``"valid"`` we walk the generator to get the exact filtered count;
    the validity-domain predicate doesn't admit a closed-form formula.
    """
    if config_name == "smoke":
        salts, Ts, Ps = SALT_CONFIGS_SMOKE, T_SMOKE, P_SMOKE
        n_bin, n_tri = len(ZBIN_SMOKE), len(ZTRI_SMOKE)
    elif config_name == "standard":
        salts, Ts, Ps = SALT_CONFIGS_STANDARD, T_STANDARD, P_STANDARD
        n_bin, n_tri = len(ZBIN_STANDARD), len(ZTRI_STANDARD)
    elif config_name == "full":
        salts, Ts, Ps = SALT_CONFIGS_FULL, T_FULL, P_FULL
        n_bin, n_tri = len(ZBIN_FULL), len(ZTRI_FULL)
    elif config_name in ("valid", "valid_dense"):
        # Count by iterating; predicate-filtered.  Compares against the
        # "standard" total so we can report a "validity-domain coverage"
        # percentage straight from this dict.
        n_valid = sum(1 for _ in iter_grid(config_name))
        n_std = sum(1 for _ in iter_grid("standard"))
        return dict(config_name=config_name,
                    n_salts=len(SALT_CONFIGS_STANDARD),
                    total=n_valid, standard_total=n_std,
                    pct_of_standard=100.0 * n_valid / max(n_std, 1),
                    rule=f"P_bar <= {VALIDITY_P_MAX_BAR} and min(z_i present) > {VALIDITY_Z_MIN}")
    else:
        raise ValueError(config_name)
    n_TP = len(Ts) * len(Ps)
    per_salt = n_TP * (2 * n_bin + n_tri)
    total = len(salts) * per_salt
    return dict(config_name=config_name,
                n_salts=len(salts), n_T=len(Ts), n_P=len(Ps),
                n_zbin=n_bin, n_ztri=n_tri,
                per_salt=per_salt, total=total)


if __name__ == "__main__":
    # Quick CLI dump of grid sizes.
    for name in ("smoke", "standard", "full"):
        print(estimate_size(name))
