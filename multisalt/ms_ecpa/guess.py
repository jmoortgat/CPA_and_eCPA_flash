"""Initial-guess construction for the eCPA equilibrium solver.

Guesses are bootstrapped from precomputed binary CPA equilibrium tables
(pure CO2-H2O and CH4-H2O). These tables are NOT experimental data; they
are reference solutions used only to seed the solver.

Storage formats (the loader prefers parquet; falls back to per-T .dat files):
    <guess_dir>/CO2.parquet            (consolidated; shipped in data/guess_tables)
    <guess_dir>/CH4.parquet
    <guess_dir>/CO2/CPA_ELV_T{t}K.dat  (251 files per gas; legacy text format)
    <guess_dir>/CH4/CPA_ELV_T{t}K.dat
"""
from functools import lru_cache
from pathlib import Path

import numpy as np

# Validity window of the binary guess tables (integer K files in this range).
T0 = 283
Tf = 533

# The warm-start guess tables ship in data/guess_tables/. MS_ECPA_GUESS_DIR
# points the loader at another copy.
def _default_guess_dir() -> Path:
    import os
    env = os.environ.get("MS_ECPA_GUESS_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1] / "data" / "guess_tables"


DEFAULT_GUESS_DIR = _default_guess_dir()



class SinglePhaseError(ValueError):
    """Raised when P is below the water vapor pressure (single-phase system)."""


def _read_dat(path):
    """Read a CPA_ELV_T{t}K.dat file -> (pressures[bar], rows[N, 10 cols])."""
    pressures = []
    rows = []
    with open(path, "r") as fh:
        next(fh)  # header line
        for line in fh:
            info = line.split()
            if not info:
                continue
            pressures.append(float(info[0]))
            rows.append([float(info[i + 1]) for i in range(10)])
    return np.asarray(pressures, dtype=float), np.asarray(rows, dtype=float)


@lru_cache(maxsize=8)
def _load_table(gas, guess_dir):
    """Load one gas's full guess table, cached by (gas, guess_dir).

    Prefers the consolidated parquet file; falls back to scanning the per-T
    ``.dat`` files when no parquet is present -- or when the interpreter has
    pandas but no parquet engine (pyarrow / fastparquet), in which case
    ``pd.read_parquet`` raises ImportError.  The two stores are bit-identical
    (the parquet was generated from these very ``.dat`` files by
    ``scripts/convert_guess_tables_to_parquet.py``; verified equal for all
    251 temperature slices of both gases), so the fallback changes no result
    -- it only keeps the solver usable in an environment without pyarrow
    instead of surfacing the missing engine as a spurious non-convergence
    several frames up.  Returns a dict
    ``{T (int K): (pressures (N,), rows (N, 10))}``.
    """
    guess_dir = Path(guess_dir)
    parquet = guess_dir / f"{gas}.parquet"
    gas_dir = guess_dir / gas
    if parquet.exists():
        import pandas as pd  # local import keeps pandas optional for .dat-only setups
        try:
            df = pd.read_parquet(parquet)
        except ImportError:
            # No parquet engine in this interpreter: use the .dat store if it
            # is there, otherwise let the caller see the real cause.
            if not gas_dir.is_dir():
                raise
            df = None
        if df is not None:
            table = {}
            # Columns after dropping T and P are the 10 guess components in the
            # canonical order used by residual_elv (Z_W, xw_W, eps_r, Z_g, xw_g,
            # chiw_W, chiw_g, Ndchi1w_dNw, Ndchi1w_dNg, Vdchi1w_dV).
            for t, sub in df.groupby("T", sort=True):
                sub = sub.sort_values("P")
                pressures = sub["P"].to_numpy()
                rows = sub.drop(columns=["T", "P"]).to_numpy()
                table[int(t)] = (pressures, rows)
            return table

    if not gas_dir.is_dir():
        raise FileNotFoundError(
            f"No {gas}.parquet at {parquet} and no fallback directory at {gas_dir}"
        )
    table = {}
    for f in sorted(gas_dir.glob("CPA_ELV_T*K.dat")):
        t = int(f.stem.replace("CPA_ELV_T", "").rstrip("K"))
        table[t] = _read_dat(f)
    return table


def build_initial_guess(T, P, YC, guess_dir=None):
    """Build the initial-guess vector for ``residual_elv`` at (T, P, YC).

    Parameters
    ----------
    T : float        absolute temperature [K]
    P : float        pressure [bar]
    YC : float       water-free CO2 mole fraction in the gas phase, in [0, 1]
    guess_dir : str or pathlib.Path, optional
        Directory containing ``CO2.parquet``/``CH4.parquet`` (preferred) or
        the legacy ``CO2/``/``CH4/`` table subfolders.
        Defaults to ``<repo>/../data/guess_tables``.

    Returns
    -------
    numpy.ndarray  initial guess (length depends on YC: 10 for pure gas, 12 for a mixture)
    """
    guess_dir = Path(guess_dir) if guess_dir is not None else DEFAULT_GUESS_DIR

    # Nearest integer temperature for which a binary table exists.
    temps = np.linspace(T0, Tf, num=Tf - T0 + 1)
    t = int(temps[np.argmin(np.abs(temps - T))])

    # --- pure CO2-H2O table -> GUESS_c ---
    co2_table = _load_table("CO2", guess_dir)
    pc, rows_c = co2_table[t]
    iPc = int(np.argmin(np.abs(pc - P)))
    GUESS_c = list(rows_c[iPc])
    GUESS_c[1] = 1 - GUESS_c[1]  # xw -> xc

    # --- pure CH4-H2O table -> GUESS_h ---
    ch4_table = _load_table("CH4", guess_dir)
    ph, rows_h = ch4_table[t]
    iPh = int(np.argmin(np.abs(ph - P)))
    GUESS_h = list(rows_h[iPh])
    GUESS_h[1] = 1 - GUESS_h[1]  # xw -> xh

    # Single-phase gate. The first row of each table sits at that binary's own
    # water vapor pressure, and the two differ at 17 of the 251 tabulated
    # temperatures (1.12 vs 1.05 bar at 373 K, 3.00 vs 2.81 at 403 K, ...), so
    # the threshold has to come from the table(s) that actually seed this
    # guess.  Testing every YC against the CO2 column rejected pure CH4-H2O
    # points the CH4 table covers, and at the three temperatures where the CO2
    # column is the lower of the two it let a sub-vapor-pressure point through
    # to be seeded from a row above the requested pressure.  A mixture is
    # assembled from one row of each table, so it needs both -> the larger.
    if YC == 0:
        P_min = ph[0]
    elif YC == 1:
        P_min = pc[0]
    else:
        P_min = max(pc[0], ph[0])
    if P < P_min:
        raise SinglePhaseError(
            "Target pressure lower than water vapor pressure at the target "
            "temperature -> single phase system"
        )

    if YC == 0:
        GUESS_ms0 = np.array(GUESS_h)
    elif YC == 1:
        GUESS_ms0 = np.array(GUESS_c)
    else:
        # Gas mixture: assemble the 12-variable guess by interleaving the two
        # binary guesses and weighting by the gas ratio.
        GUESS_c.insert(1, 0)             # insert x2w (CH4 in water) slot
        GUESS_c.insert(9, GUESS_h[8])    # insert Ndchi1WdNh slot
        GUESS_h.insert(2, 0)             # insert x3w (CO2 in water) slot
        GUESS_h.insert(10, GUESS_c[-2])  # insert Ndchi1WdNc slot
        GUESS_ms0 = np.array(GUESS_c) * YC + np.array(GUESS_h) * (1 - YC)

    return GUESS_ms0
