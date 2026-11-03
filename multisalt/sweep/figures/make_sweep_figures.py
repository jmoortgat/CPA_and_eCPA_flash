"""Tables and figures from the multi-salt eCPA sweep output.

Reads `results_cache/<sweep>_summary/merged.parquet` and writes:

* `tables/performance_*.csv` — convergence + iteration + wall-time summaries
* `tables/render_fill_log.csv` — every (T, P) cell the plot-time
  `_fill_isolated_defects` cleanup patched, per panel
* `figures/phase_boundary_*.png` — β heatmaps over (T, P) at fixed (z, mi)
* `figures/co2_solubility_*.png` — x_CO2,aq heatmaps over (T, P) at fixed mi
* `figures/density_*.png` — ρ_aq surface vs (T, P) per salt
* `figures/salting_out_*.png` — m_CO2,aq vs m_salt at representative (T, P)

All figures use the paper's style (bold fonts, WONG colorblind palette,
LINESTYLES cycle).  Pure pyplot — no notebook required.

Invocation::

    PYTHONPATH=<repo>/code python -m sweep.figures.make_sweep_figures \\
        --merged <repo>/results_cache/sweep_valid_v1_summary/merged.parquet \\
        --out-dir <repo>/results_cache/sweep_valid_v1_summary/

Re-running with a different ``--merged`` regenerates everything against
the new data; ideal for the v7 → v8 → final iteration loop.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paper style (inlined; mirrors scripts/paper_style.py)
# ---------------------------------------------------------------------------

def apply_paper_style():
    mpl.rcParams.update({
        "font.size": 14, "axes.labelsize": 16,
        "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12,
        "font.weight": "bold", "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.linewidth": 1.6, "axes.edgecolor": "black",
        "xtick.major.width": 1.6, "ytick.major.width": 1.6,
        "xtick.major.size": 6, "ytick.major.size": 6,
        "xtick.color": "black", "ytick.color": "black",
        "xtick.direction": "in", "ytick.direction": "in",
        "lines.linewidth": 2.2, "lines.markersize": 7,
        "legend.frameon": True, "legend.edgecolor": "black",
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })


WONG = ["#000000", "#E69F00", "#56B4E9", "#009E73",
        "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1, 1, 1))]


def bold_ticks(ax):
    for lbl in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        lbl.set_fontweight("bold")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def table_overall_performance(df: pd.DataFrame, out_path: Path):
    """Headline performance numbers for the paper benchmark section."""
    converged = df[df["converged"]]
    # SSI statistics cover TWO-PHASE states only. A single-phase state
    # runs no flash and carries n_iter_ssi = 0 identically (measured on
    # the dense grid: min 0, max 0 over 26,859 rows), so averaging those
    # zeros in does not measure the solver -- it measures how much of the
    # grid happens to be single-phase. Over all converged rows the mean
    # reads 3.4; over the two-phase rows that actually ran an SSI it is
    # 6.5. The main-text table reports the second, so the SI table uses the
    # same basis.
    ssi_src = converged[converged["phases"] == 2]
    rows = [{
        "metric": "Total rows", "value": f"{len(df)}",
    }, {
        "metric": "Converged", "value": f"{len(converged)} ({100*len(converged)/len(df):.3f}%)",
    }, {
        "metric": "Failed", "value": f"{len(df)-len(converged)}",
    }, {
        "metric": "Wall time p50 [ms]",
        "value": f"{converged['t_wall_ms'].median():.0f}",
    }, {
        "metric": "Wall time p95 [ms]",
        "value": f"{converged['t_wall_ms'].quantile(0.95):.0f}",
    }, {
        "metric": "Wall time p99 [ms]",
        "value": f"{converged['t_wall_ms'].quantile(0.99):.0f}",
    }, {
        "metric": "Wall time max [ms]",
        "value": f"{converged['t_wall_ms'].max():.0f}",
    }, {
        "metric": "Outer SSI iter p50 (two-phase states)",
        "value": f"{ssi_src['n_iter_ssi'].median():.0f}",
    }, {
        "metric": "Outer SSI iter p99 (two-phase states)",
        "value": f"{ssi_src['n_iter_ssi'].quantile(0.99):.0f}",
    }, {
        "metric": "Inner aq Newton iter p95 (diag)",
        "value": f"{converged['n_iter_aq_inner_diag'].quantile(0.95):.0f}",
    }, {
        "metric": "Inner gas Newton iter p95 (diag)",
        "value": f"{converged['n_iter_gas_inner_diag'].quantile(0.95):.0f}",
    }]
    out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


def table_performance_by_family(df: pd.DataFrame, out_path: Path):
    """Performance broken out by (feed_family, salt_name).  Useful for
    the paper to show that performance is uniform across the grid."""
    # SSI statistics cover TWO-PHASE states only. A single-phase state
    # runs no flash and carries n_iter_ssi = 0 identically (measured on
    # the dense grid: min 0, max 0 over 26,859 rows), so averaging those
    # zeros in does not measure the solver -- it measures how much of the
    # grid happens to be single-phase. Over all converged rows the mean
    # reads 3.4; over the two-phase rows that actually ran an SSI it is
    # 6.5. The main-text table reports the second, so the SI table uses the
    # same basis.
    _conv = df[df["converged"]]
    g = _conv.groupby(["feed_family", "salt_name"])
    g_ssi = _conv[_conv["phases"] == 2].groupby(["feed_family", "salt_name"])
    out = g["t_wall_ms"].agg(["median", lambda s: s.quantile(0.95),
                              lambda s: s.quantile(0.99)]).round(0)
    out.columns = ["wall_ms_p50", "wall_ms_p95", "wall_ms_p99"]
    out["n_iter_ssi_p50"] = (g_ssi["n_iter_ssi"].median()
                             .reindex(out.index).fillna(0).astype(int))
    out["n_iter_ssi_p95"] = (g_ssi["n_iter_ssi"].quantile(0.95)
                             .reindex(out.index).fillna(0).astype(int))
    out["n_pts"] = g.size()
    out = out.reset_index()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


_CHARGE = dict(m_Na=1, m_Cl=-1, m_K=1, m_Ca=2, m_Mg=2, m_SO4=-2)


def _ionic_strength(row) -> float:
    """Compute :math:`I_c = \\tfrac{1}{2} \\sum m_i z_i^2` from the row's
    ion molality columns."""
    return 0.5 * sum(float(row[c]) * _CHARGE[c]**2 for c in _CHARGE)


def table_perf_latex(df: pd.DataFrame, out_path: Path):
    """Per-call performance summary as a LaTeX ``tabular`` snippet for
    direct drop into the SI.  Reports MEAN wall time per call (ms) and
    MEAN K-value SSI iteration count per call, plus 95th-percentile
    values to characterise the upper tail.  All converged points are
    aggregated by feed family."""
    # Tables S11 and S12 of Paper III average the SSI count over ALL converged
    # states (a single-phase state counts 0); the two-phase mean is in Table S13.
    g = df[df["converged"]].groupby("feed_family")
    rows = []
    for fam, sub in g:
        rows.append({
            "feed_family": fam,
            "n_pts": int(len(sub)),
            "pct_2phase": 100 * (sub["phases"] == 2).mean(),
            "wall_mean": sub["t_wall_ms"].mean(),
            "wall_p95": sub["t_wall_ms"].quantile(0.95),
            "ssi_mean": sub["n_iter_ssi"].mean(),
            "ssi_p95": sub["n_iter_ssi"].quantile(0.95),
        })
    tbl = pd.DataFrame(rows)
    label_map = {
        "binary_CO2": r"binary \ce{CO2}--\ce{H2O}",
        "binary_CH4": r"binary \ce{CH4}--\ce{H2O}",
        "ternary":    r"ternary \ce{CH4}--\ce{CO2}--\ce{H2O}",
    }
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r" Feed family & $n$ & 2-ph (\%) & "
        r"$\bar{t}_{\text{call}}$ [ms] & $t_{\text{call}}^{p95}$ [ms] & "
        r"$\bar{N}_{\text{SSI}}$ & $N_{\text{SSI}}^{p95}$ \\",
        r"\midrule",
    ]
    for _, r in tbl.iterrows():
        lines.append(
            f" {label_map.get(r['feed_family'], r['feed_family'])} & "
            f"{int(r['n_pts'])} & "
            f"{r['pct_2phase']:.1f} & "
            f"{r['wall_mean']:.0f} & {r['wall_p95']:.0f} & "
            f"{r['ssi_mean']:.1f} & {int(r['ssi_p95'])} \\\\"
        )
    conv = df[df["converged"]]
    lines += [
        r"\midrule",
        f" \\textbf{{Total}} & \\textbf{{{int(tbl['n_pts'].sum())}}} & "
        f"\\textbf{{{(df['converged'] & (df['phases']==2)).sum() / max(df['converged'].sum(),1) * 100:.1f}}} & "
        f"\\textbf{{{conv['t_wall_ms'].mean():.0f}}} & "
        f"\\textbf{{{conv['t_wall_ms'].quantile(0.95):.0f}}} & "
        f"\\textbf{{{conv['n_iter_ssi'].mean():.1f}}} & "
        f"\\textbf{{{int(conv['n_iter_ssi'].quantile(0.95))}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return tbl


def table_perf_by_family_latex(df: pd.DataFrame, out_path: Path):
    """Per-(feed_family, salt_name) performance breakdown as LaTeX.
    Reports mean per-call wall time and SSI iteration count, with 95th-
    percentile values to capture the upper tail."""
    # Averaged over ALL converged states, as Table S11 (see table_perf_latex).
    g = df[df["converged"]].groupby(["feed_family", "salt_name"])
    rows = []
    for (fam, salt), sub in g:
        rows.append({
            "feed_family": fam, "salt_name": salt,
            "n_pts": int(len(sub)),
            "pct_2ph": 100 * (sub["phases"] == 2).mean(),
            "wall_mean": sub["t_wall_ms"].mean(),
            "wall_p95": sub["t_wall_ms"].quantile(0.95),
            "ssi_mean": sub["n_iter_ssi"].mean(),
            "ssi_p95": int(sub["n_iter_ssi"].quantile(0.95)),
        })
    tbl = pd.DataFrame(rows)
    fam_label = {
        "binary_CO2": r"\ce{CO2}--\ce{H2O}",
        "binary_CH4": r"\ce{CH4}--\ce{H2O}",
        "ternary":    r"\ce{CH4}--\ce{CO2}--\ce{H2O}",
    }
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r" Feed family & Salt & $n$ & 2-ph (\%) & "
        r"$\bar{t}_{\text{call}}$ [ms] & $t_{\text{call}}^{p95}$ [ms] & "
        r"$\bar{N}_{\text{SSI}}$ & $N_{\text{SSI}}^{p95}$ \\",
        r"\midrule",
    ]
    last_fam = None
    for _, r in tbl.iterrows():
        fam_cell = fam_label.get(r["feed_family"], r["feed_family"]) if r["feed_family"] != last_fam else ""
        last_fam = r["feed_family"]
        salt = r["salt_name"].replace("_", r"\_")
        lines.append(
            f" {fam_cell} & {salt} & {int(r['n_pts'])} & "
            f"{r['pct_2ph']:.1f} & {r['wall_mean']:.0f} & {r['wall_p95']:.0f} & "
            f"{r['ssi_mean']:.1f} & {r['ssi_p95']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return tbl


def table_canonical_solubility_latex(df: pd.DataFrame, out_path: Path):
    """Predicted CO2 solubility at canonical reservoir-storage conditions
    (T=323K + 423K, P=100/300/500/1000 bar, z_CO2≈0.5), one row per
    salt config.  Useful reference table for the paper SI."""
    conv = df[df["converged"] & (df["phases"] == 2)
              & (df["feed_family"] == "binary_CO2")
              & ((df["z_CO2"] - 0.5).abs() < 0.05)]
    # Pick canonical (T, P) cells closest to the requested values.
    Ts_avail = sorted(conv["T_K"].unique())
    Ps_avail = sorted(conv["P_bar"].unique())
    def _near(arr, v): return min(arr, key=lambda x: abs(x - v))
    T_picks = [_near(Ts_avail, t) for t in (323, 423)]
    P_picks = [_near(Ps_avail, p) for p in (100, 300, 500, 1000)]
    salts = sorted(conv["salt_name"].unique())
    # Build a wide table: rows = salt, columns = (T, P).
    rows = []
    for salt in salts:
        row = {"salt": salt}
        for T in T_picks:
            for P in P_picks:
                cell = conv[(conv["salt_name"] == salt)
                            & (conv["T_K"] == T)
                            & (conv["P_bar"] == P)]
                if len(cell):
                    row[f"T{int(T)}_P{int(P)}"] = float(cell["x_CO2_aq"].iloc[0] * 1000.0)
                else:
                    row[f"T{int(T)}_P{int(P)}"] = float("nan")
        rows.append(row)
    tbl = pd.DataFrame(rows)

    # LaTeX: salt rows × column blocks per T.
    lines = [
        r"\begin{tabular}{l" + "r" * (len(T_picks) * len(P_picks)) + "}",
        r"\toprule",
    ]
    h2_cols = ["Salt"]
    h_top = [""]
    for T in T_picks:
        h_top.append(rf"\multicolumn{{{len(P_picks)}}}{{c}}{{$T = {T:.1f}$\,K}}")
    lines.append(" & ".join(h_top) + r" \\")
    # cmidrule under each T-block
    for i, T in enumerate(T_picks):
        start = 2 + i * len(P_picks)
        end = start + len(P_picks) - 1
        lines.append(rf"\cmidrule(lr){{{start}-{end}}}")
    h2 = [""]
    for T in T_picks:
        for P in P_picks:
            h2.append(rf"${int(P)}$\,bar")
    lines.append(" & ".join(h2) + r" \\")
    lines.append(r"\midrule")
    for _, r in tbl.iterrows():
        salt = r["salt"].replace("_", r"\_")
        cells = [salt]
        for T in T_picks:
            for P in P_picks:
                v = r[f"T{int(T)}_P{int(P)}"]
                cells.append(f"{v:.1f}" if np.isfinite(v) else "---")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return tbl


def table_two_phase_fraction(df: pd.DataFrame, out_path: Path):
    """Per (feed_family, salt) fraction of points classified two-phase."""
    g = df[df["converged"]].groupby(["feed_family", "salt_name"])
    out = pd.DataFrame({
        "n_pts": g.size(),
        "n_2phase": g.apply(lambda s: (s["phases"] == 2).sum()),
    }).reset_index()
    out["pct_2phase"] = (100 * out["n_2phase"] / out["n_pts"]).round(2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

# Every cell that ``_fill_isolated_defects`` changes is appended here, so
# that a patched (T, P) cell in a rendered surface can still be named
# afterwards.  ``main()`` writes the accumulated list to
# ``tables/render_fill_log.csv`` and prints a per-panel count as it goes:
# the fill is a deliberate rendering choice, but an unrecorded one leaves
# no way to tell a solved cell from a patched one once the PNG is made.
_FILL_LOG: list[dict] = []


def _fill_isolated_defects(mat, *, min_valid_nbrs=5, outlier_z=6.0,
                           label=None):
    """Two-pass rendering-side cleanup of isolated cell defects, applied
    after pivoting and before imshow.

    ``label`` names the panel (e.g. ``"co2_solubility/NaCl_3m/x_CO2_aq"``)
    in the printed summary and in ``_FILL_LOG``; pass it from every call
    site so the run log says what was patched and where.

    Pass 1 (NaN fill): a NaN cell whose 8-neighbour window has at least
    ``min_valid_nbrs`` finite values is replaced by the median of those
    neighbours.  Motivation: near-critical or near-dew-line points that
    the sweep failed on (tpd near zero, K collapses to 1, or β snaps to
    a boundary) are surrounded by densely-converged neighbours whose
    values interpolate cleanly.  The interior and the straight edges of
    a large NaN region — the physical single-phase area — have <5 finite
    neighbours and are left untouched, but a concave (staircase) corner
    of such a region has exactly 5 finite 8-neighbours and IS filled, so
    the single-phase area is eroded by one cell at every notch.  Cells
    patched this way are recorded in ``_FILL_LOG``, not marked in the
    image; read the log alongside the figure.

    This used to also name a "no gas root at low T" corner of the Z_gas
    surface as a physical NaN region.  It was not physical: the flash
    converged there with correct fugacities, and only the post-convergence
    diagnostics returned NaN, because they screened the gas root with a
    stale predicate that bounded chi <= 1 — a theorem only where every
    association strength is non-negative, which fails below ~290 K where
    S13(T) < 0.  Fixed 2026-09-18 (flash.py now calls
    lnphi._gas_root_is_physical, as the SSI already did), so any Z_gas
    surface regenerated after that date should have no such corner.  A
    sweep cache older than that will still show it.

    Pass 2 (outlier fill): a finite cell whose absolute deviation from
    the neighbour median exceeds ``outlier_z`` × neighbour MAD is
    replaced by that median.  Catches the occasional inner-Newton
    convergence to a wrong-root state that shows as a single anomalous
    pixel (e.g. eps_r cells that landed on a spurious permittivity
    root).  Threshold intentionally conservative to avoid smoothing
    real physical structure.

    "Catches" here means *overwrites*: the anomalous value is replaced,
    not flagged, so the ``_FILL_LOG`` entry is the only surviving record
    of a wrong-root cell.  The test has no absolute floor on the MAD, so
    in a locally near-flat neighbourhood a legitimate cell can also clear
    6 x MAD — another reason to read the log rather than trust the pixel.
    """
    arr = mat.values.astype(float, copy=True)
    ny, nx = arr.shape
    out = arr.copy()

    def _nbrs(i, j):
        vals = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < ny and 0 <= nj < nx and np.isfinite(arr[ni, nj]):
                    vals.append(arr[ni, nj])
        return vals

    n_nan_filled = 0
    n_outlier_replaced = 0

    def _log(pass_name, i, j, old, new):
        _FILL_LOG.append({
            "label": label if label is not None else "",
            "pass": pass_name,
            "T_K": float(mat.index[i]),
            "P_bar": float(mat.columns[j]),
            "value_before": old,
            "value_after": new,
        })

    # Pass 1: isolated NaN
    for i in range(ny):
        for j in range(nx):
            if not np.isnan(arr[i, j]):
                continue
            vals = _nbrs(i, j)
            if len(vals) >= min_valid_nbrs:
                out[i, j] = float(np.median(vals))
                n_nan_filled += 1
                _log("nan_fill", i, j, float("nan"), out[i, j])

    # Pass 2: outlier (use the pass-1-filled array as neighbour source)
    arr2 = out.copy()
    for i in range(ny):
        for j in range(nx):
            v = arr2[i, j]
            if not np.isfinite(v):
                continue
            vals = []
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < ny and 0 <= nj < nx and np.isfinite(arr2[ni, nj]):
                        vals.append(arr2[ni, nj])
            if len(vals) < min_valid_nbrs:
                continue
            med = float(np.median(vals))
            mad = float(np.median(np.abs(np.asarray(vals) - med)))
            if mad <= 0.0:
                continue
            if abs(v - med) > outlier_z * mad:
                out[i, j] = med
                n_outlier_replaced += 1
                _log("outlier", i, j, float(v), med)

    if n_nan_filled or n_outlier_replaced:
        print(f"    [clean] {label if label is not None else '?'}: "
              f"{n_nan_filled} NaN cell(s) filled, "
              f"{n_outlier_replaced} finite cell(s) replaced by the "
              f"neighbour median")

    return pd.DataFrame(out, index=mat.index, columns=mat.columns)


#: Whether cell-filling runs at all. OFF by default: every surface shows the
#: cells the sweep produced and nothing else. ``--clean`` restores the old
#: behaviour for comparison.
#:
#: The filling is not cosmetic. Pass 2 REPLACES a finite cell whose deviation
#: from its neighbour median exceeds 6 x MAD -- a value the solver produced,
#: overwritten by one it did not, with render_fill_log.csv the only surviving
#: record. It was introduced to hide single-pixel wrong-root artifacts.
#:
#: Measured on the full 151,632-state grid with the root-selection work in
#: place: it filled 0 NaN cells -- there are none, because every state now
#: converges -- and overwrote 2,809 finite ones, of which NOT ONE was
#: non-physical (no eps_r <= 0, no chi <= 0). The artifacts it was built for
#: no longer exist, so every edit it makes is now to a legitimate value, by a
#: median of 821 % on y_H2O_gas, 49.6 % on n_iter_ssi and 42-45 % on brine
#: density. Turning it off costs nothing (no gaps appear) and stops the
#: figures showing numbers the model did not produce.
_CLEAN_DEFAULT = False


def _pivot_to_grid(sub: pd.DataFrame, value_col: str, *, clean=None,
                   label=None, aggfunc="median"):
    """Pivot a slice of the sweep parquet to a 2-D (T_K × P_bar) matrix
    suitable for ``imshow``.  Missing combos remain NaN.

    Assumes ``T_K`` and ``P_bar`` have already been ULP-normalised in
    ``main()`` so that grid coordinates from independent
    ``np.geomspace`` calls in different sweep chunks land on identical
    float representations instead of near-duplicates that would render
    as thin vertical stripes on the log-P axis.

    ``clean=True`` (default) applies ``_fill_isolated_defects`` to the
    pivoted matrix, patching singleton NaN cells surrounded by
    converged neighbours and single-pixel outliers.  Set ``clean=False``
    for callers that must see the raw sweep-data cells.  Every patched
    cell is recorded in ``_FILL_LOG`` and written to
    ``tables/render_fill_log.csv``, so a cleaned surface can always be
    checked against the list of cells the cleaning touched.

    ``label`` is forwarded to ``_fill_isolated_defects`` to name the
    panel in that log.

    ``aggfunc`` collapses duplicate rows in a (T, P) cell.  The per-salt
    figures slice to one salt and one composition, so each cell holds a
    single row and median/mean coincide; the default stays ``"median"``
    for those.  ``figure_ssi_heatmap_by_feed_family`` pools the eight
    salt configurations into one cell and passes ``aggfunc="mean"``,
    because its colourbar, caption and docstring all state a mean.
    """
    mat = (sub.pivot_table(index="T_K", columns="P_bar",
                           values=value_col, aggfunc=aggfunc)
              .sort_index().sort_index(axis=1))
    if clean is None:
        clean = _CLEAN_DEFAULT
    if clean:
        mat = _fill_isolated_defects(mat, label=label)
    return mat


# Warn once per run rather than once per panel when an axis is not
# uniformly spaced (see ``_cell_edges``).
_NONUNIFORM_AXIS_WARNED = False


def _cell_edges(vals, *, rtol=1e-6):
    """Outer edges of the image for an axis whose cell *centres* are
    ``vals``, plus a flag saying whether those centres are uniform.

    ``imshow``'s ``extent`` names the outer edges of the image, not the
    coordinates of the first and last cell centres, so passing the data
    min/max puts every cell half a cell away from its own grid
    coordinate (on the 39-point dense T grid of ``sweep_config.T_DENSE``
    that is 4.4 K at the ends).  Widening by half a step registers the
    cell centres on the data values exactly, for a uniform axis.

    A non-uniform axis cannot be drawn correctly by ``imshow`` at all —
    its rows are always equally tall — so the uniformity flag is
    returned for the caller to warn on.  ``T_DENSE``/``log10(P_DENSE)``
    are uniform; ``T_STANDARD`` is not (15 K step at 308->323, 50 K
    steps above 523).
    """
    v = np.asarray(vals, dtype=float)
    if v.size == 0:
        return 0.0, 1.0, True
    if v.size == 1:
        h = max(abs(float(v[0])) * 1e-3, 1e-12)
        return float(v[0]) - 0.5 * h, float(v[0]) + 0.5 * h, True
    d = np.diff(v)
    h = float(d.mean())
    uniform = bool(np.allclose(d, h, rtol=rtol, atol=0.0))
    return float(v[0]) - 0.5 * h, float(v[-1]) + 0.5 * h, uniform


def _imshow_grid(ax, mat, *, T_axis, P_axis, vmin=None, vmax=None,
                 cmap="viridis"):
    """Render a T_K × P_bar matrix with the conventional axis orientation
    (T on y, P on x, log P, K-tick labels integer)."""
    global _NONUNIFORM_AXIS_WARNED
    P_log = np.log10(np.asarray(P_axis, dtype=float))
    T_val = np.asarray(T_axis, dtype=float)
    x0, x1, x_uniform = _cell_edges(np.sort(P_log))
    y0, y1, y_uniform = _cell_edges(np.sort(T_val))
    if not (x_uniform and y_uniform) and not _NONUNIFORM_AXIS_WARNED:
        _NONUNIFORM_AXIS_WARNED = True
        bad = ", ".join(n for n, ok in (("log10(P_bar)", x_uniform),
                                        ("T_K", y_uniform)) if not ok)
        print(f"  [imshow] WARNING: {bad} axis is not uniformly spaced; "
              f"imshow draws every cell the same size, so the tick "
              f"positions do not line up with the grid values.  Use "
              f"pcolormesh with real cell edges for this grid.")
    im = ax.imshow(mat.values, origin="lower", aspect="auto",
                   interpolation="nearest",
                   extent=[x0, x1, y0, y1],
                   vmin=vmin, vmax=vmax, cmap=cmap)
    # Custom x ticks at round decades inside the log range.
    xticks = [v for v in (1, 5, 10, 50, 100, 300, 1000, 2000)
              if P_axis.min() <= v <= P_axis.max()]
    ax.set_xticks([np.log10(v) for v in xticks])
    ax.set_xticklabels([str(v) for v in xticks])
    ax.set_xlabel(r"$P$ [bar]")
    ax.set_ylabel(r"$T$ [K]")
    bold_ticks(ax)
    return im


def _apply_shared_axes(axes, nrow, ncol, n_used):
    """For a multi-panel figure with `sharex=True, sharey=True` and
    axes laid out row-major, keep the P-axis label only on the bottom
    row and the T-axis label only on the leftmost column, and blank
    unused panels.  Matplotlib already hides the tick labels on inner
    panels when the axes are shared; this helper just aligns the
    xlabel/ylabel treatment with the shared-axis convention used
    elsewhere in the paper."""
    for k in range(nrow * ncol):
        r, c = k // ncol, k % ncol
        ax = axes[r][c]
        if k >= n_used:
            ax.axis("off")
            continue
        # Blank internal-panel axis labels.
        if r < nrow - 1:
            ax.set_xlabel("")
        # For the P axis: if a used panel below it exists, blank the
        # x label; if it is the last row OR the panel directly below
        # is unused (partial bottom row), keep the label.
        below = (r + 1) * ncol + c
        if below < nrow * ncol and below < n_used:
            ax.set_xlabel("")
        if c > 0:
            ax.set_ylabel("")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figure_phase_boundary(df: pd.DataFrame, out_dir: Path,
                          feed_family: str = "ternary",
                          z_pick=(0.49, 0.255, 0.255)):
    """β = vapor fraction over (T, P) for a representative ternary feed,
    one panel per salt config.  Shows where the model identifies two-phase
    coexistence and how the salt shifts the boundary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    conv = df[df["converged"]].copy()

    # Find rows that match the requested feed within rounding tolerance.
    def _match(s):
        return (s["feed_family"] == feed_family
                and abs(s["z_H2O"] - z_pick[0]) < 0.05
                and abs(s["z_CH4"] - z_pick[1]) < 0.05
                and abs(s["z_CO2"] - z_pick[2]) < 0.05)

    sub_all = conv[conv.apply(_match, axis=1)]
    if not len(sub_all):
        print(f"  [phase_boundary] no matching rows for {feed_family} z≈{z_pick}; skipping")
        return None

    salts = sorted(sub_all["salt_name"].unique())
    ncol = min(4, len(salts))
    nrow = (len(salts) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.6 * nrow),
                             sharex=True, sharey=True,
                             squeeze=False, constrained_layout=True)
    P_axis = np.sort(sub_all["P_bar"].unique())
    T_axis = np.sort(sub_all["T_K"].unique())
    # Grey for NaN: single-phase cells carry no β, and without set_bad
    # they render as the white page instead of the grey the SI caption
    # and the SSI heatmaps use for "no data here".
    cmap_obj = mpl.colormaps.get_cmap("RdBu_r").copy()
    cmap_obj.set_bad("#cccccc")
    im = None
    for k, salt in enumerate(salts):
        ax = axes[k // ncol][k % ncol]
        sub = sub_all[sub_all["salt_name"] == salt]
        # Reindex to the union axes built from all salts: the extent
        # below is built from those axes, so a panel whose own slice is
        # missing a T row or P column would otherwise be stretched
        # across the full extent (the sibling figures all reindex).
        mat = _pivot_to_grid(
            sub, "beta", label=f"phase_boundary/{feed_family}/{salt}/beta"
        ).reindex(index=T_axis, columns=P_axis)
        # β is NaN wherever phases==1 (β is meaningless there), so those
        # cells stay grey — except the ones _fill_isolated_defects
        # patches at a concave corner of the single-phase region, which
        # are listed in tables/render_fill_log.csv.
        im = _imshow_grid(ax, mat, T_axis=T_axis, P_axis=P_axis,
                          vmin=0.0, vmax=1.0, cmap=cmap_obj)
        ax.text(0.04, 0.96, salt, transform=ax.transAxes,
                ha="left", va="top", fontsize=11, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2))
    # Blank unused panels
    _apply_shared_axes(axes, nrow, ncol, len(salts))
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
        cbar.set_label(r"$\beta$  (vapor mole fraction)", fontweight="bold")
    out_path = out_dir / f"phase_boundary_{feed_family}.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> {out_path}")
    return out_path


def figure_co2_solubility(df: pd.DataFrame, out_dir: Path,
                          z_CO2: float = 0.5):
    """x_CO2,aq over (T, P) for binary_CO2 at fixed z_CO2, one panel per salt.
    Axis labelled in per-mil to match the rest of the paper; colour range
    anchored on the 95th percentile so the dominant bulk of the data
    fills the dynamic range (the small high-T high-P tail clips white)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    conv = df[df["converged"] & (df["phases"] == 2)
              & (df["feed_family"] == "binary_CO2")]
    sub_all = conv[(conv["z_CO2"] - z_CO2).abs() < 0.05].copy()
    if not len(sub_all):
        print(f"  [co2_solubility] no rows for binary_CO2 z_CO2≈{z_CO2}; skipping")
        return None
    # Scale to per-mil.
    sub_all["x_CO2_aq_pm"] = sub_all["x_CO2_aq"] * 1000.0
    salts = sorted(sub_all["salt_name"].unique())
    ncol = min(4, len(salts))
    nrow = (len(salts) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.6 * nrow),
                             sharex=True, sharey=True,
                             squeeze=False, constrained_layout=True)
    P_axis = np.sort(sub_all["P_bar"].unique())
    T_axis = np.sort(sub_all["T_K"].unique())
    vmax = float(sub_all["x_CO2_aq_pm"].quantile(0.95))
    im = None
    for k, salt in enumerate(salts):
        ax = axes[k // ncol][k % ncol]
        sub = sub_all[sub_all["salt_name"] == salt]
        # Reindex to the union axes (see figure_phase_boundary): this
        # figure filters phases==2, so a salt whose high-T/low-P corner
        # is single-phase can lose a whole T row from its own pivot.
        mat = _pivot_to_grid(
            sub, "x_CO2_aq_pm", label=f"co2_solubility/z{z_CO2:.2f}/{salt}"
        ).reindex(index=T_axis, columns=P_axis)
        im = _imshow_grid(ax, mat, T_axis=T_axis, P_axis=P_axis,
                          vmin=0.0, vmax=vmax, cmap="viridis")
        ax.text(0.04, 0.96, salt, transform=ax.transAxes,
                ha="left", va="top", fontsize=11, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2))
    _apply_shared_axes(axes, nrow, ncol, len(salts))
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02,
                            extend="max")
        cbar.set_label(r"$x_{\mathrm{CO_2}}^{\mathrm{aq}}$  [‰]",
                       fontweight="bold")
    out_path = out_dir / f"co2_solubility_binary_zCO2_{z_CO2:.2f}.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> {out_path}")
    return out_path


def figure_generic_surface_per_salt(df: pd.DataFrame, out_dir: Path, *,
                                    feed_family: str,
                                    z_pick,
                                    metric: str,
                                    label_text: str,
                                    out_stem: str,
                                    cmap: str = "viridis",
                                    scale: float = 1.0,
                                    vmin: float | None = None,
                                    vmax: float | None = None,
                                    log_color: bool = False,
                                    twoph_only: bool = True):
    """Per-salt (T, P) heatmap of an arbitrary scalar column.  General
    surface plotter used by several of the figures below."""
    out_dir.mkdir(parents=True, exist_ok=True)
    base = df[df["converged"]
              & (df["feed_family"] == feed_family)
              & (df["phases"] == 2 if twoph_only else df["phases"].notna())].copy()
    def _match(s):
        return (abs(s["z_H2O"] - z_pick[0]) < 0.05
                and abs(s["z_CH4"] - z_pick[1]) < 0.05
                and abs(s["z_CO2"] - z_pick[2]) < 0.05)
    sub_all = base[base.apply(_match, axis=1)].copy()
    if not len(sub_all):
        print(f"  [{out_stem}] no rows for {feed_family} z≈{z_pick}; skipping")
        return None
    sub_all["_metric"] = sub_all[metric] * scale
    if log_color:
        sub_all["_metric"] = np.log10(sub_all["_metric"].clip(lower=1e-12))
    salts = sorted(sub_all["salt_name"].unique())
    ncol = min(4, len(salts))
    nrow = (len(salts) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.6 * nrow),
                             sharex=True, sharey=True,
                             squeeze=False, constrained_layout=True)
    P_axis = np.sort(sub_all["P_bar"].unique())
    T_axis = np.sort(sub_all["T_K"].unique())
    if vmin is None:
        vmin = float(np.nanquantile(sub_all["_metric"], 0.02))
    if vmax is None:
        vmax = float(np.nanquantile(sub_all["_metric"], 0.98))
    cmap_obj = mpl.colormaps.get_cmap(cmap).copy()
    cmap_obj.set_bad("#cccccc")
    im = None
    for k, salt in enumerate(salts):
        ax = axes[k // ncol][k % ncol]
        sub = sub_all[sub_all["salt_name"] == salt]
        mat = _pivot_to_grid(
            sub, "_metric", label=f"{out_stem}/{salt}/{metric}"
        ).reindex(index=T_axis, columns=P_axis)
        im = _imshow_grid(ax, mat, T_axis=T_axis, P_axis=P_axis,
                          vmin=vmin, vmax=vmax, cmap=cmap_obj)
        ax.text(0.04, 0.96, salt, transform=ax.transAxes,
                ha="left", va="top", fontsize=11, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2))
    _apply_shared_axes(axes, nrow, ncol, len(salts))
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02, extend="both")
        cbar.set_label(label_text, fontweight="bold")
    out_path = out_dir / f"{out_stem}.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> {out_path}")
    return out_path


def _pretty_salt(name: str) -> str:
    """Human-readable salt-config label: underscores to spaces and the
    file-safe decimal encoding NpM -> N.M (e.g. NaCl_0p5m -> NaCl 0.5m)."""
    import re
    return re.sub(r"(\d)p(\d)", r"\1.\2", name.replace("_", " "))


def figure_salting_out(df: pd.DataFrame, out_dir: Path,
                       T_K: float = 323.0, P_bar: float = 100.0,
                       z_CO2: float = 0.50):
    """``x_CO2,aq`` (per-mil) vs total ionic strength :math:`I_c` for binary
    \\ce{CO2}--\\ce{H2O} at a single (T, P).  One marker per salt config,
    coloured/lettered by anion+cation family.  Headlines the ion-specific
    salting-out argument that motivates the multi-salt parameterisation.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    conv = df[df["converged"] & (df["phases"] == 2)
              & (df["feed_family"] == "binary_CO2")]
    # Find the (T, P) cell closest to the requested point.
    Ts = sorted(conv["T_K"].unique())
    Ps = sorted(conv["P_bar"].unique())
    T_use = min(Ts, key=lambda v: abs(v - T_K))
    P_use = min(Ps, key=lambda v: abs(v - P_bar))
    sub = conv[(conv["T_K"] == T_use) & (conv["P_bar"] == P_use)
               & ((conv["z_CO2"] - z_CO2).abs() < 0.05)].copy()
    if not len(sub):
        print(f"  [salting_out] no rows for binary_CO2 (T≈{T_K}, P≈{P_bar}, z_CO2≈{z_CO2})")
        return None
    sub["I_c"] = sub.apply(_ionic_strength, axis=1)
    sub["xCO2_pm"] = sub["x_CO2_aq"] * 1000.0
    sub = sub.sort_values("I_c").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6.5, 4.6), constrained_layout=True)
    # Label collision handling: labels sit above-right by default, drop
    # below-right when a point nearly coincides with an already-labelled
    # one, and flip to the left when another point sits where the text
    # would land.
    pts = list(zip(sub["I_c"], sub["xCO2_pm"]))
    xr = max(sub["I_c"].max() - sub["I_c"].min(), 1e-9)
    yr = max(sub["xCO2_pm"].max() - sub["xCO2_pm"].min(), 1e-9)
    placed = []
    # Color each salt cohort distinctly using WONG + LINESTYLES.
    for k, (_, r) in enumerate(sub.iterrows()):
        col = WONG[(k % 7) + 1]
        x, y = r["I_c"], r["xCO2_pm"]
        ax.scatter([x], [y],
                   marker="o", s=110, color=col,
                   edgecolors="k", linewidths=0.6, zorder=3,
                   label=_pretty_salt(r["salt_name"]))
        dx, dy, ha = 6, 4, "left"
        if any(abs(x - px) / xr < 0.03 and abs(y - py) / yr < 0.06
               for px, py in placed):
            dx, dy = 6, -12
        elif any(0 < (qx - x) / xr < 0.18 and abs(qy - y) / yr < 0.04
                 for j, (qx, qy) in enumerate(pts) if j != k):
            dx, ha = -8, "right"
        ax.annotate(_pretty_salt(r["salt_name"]),
                    xy=(x, y),
                    xytext=(dx, dy), textcoords="offset points", ha=ha,
                    fontsize=9, fontweight="bold", color=col)
        placed.append((x, y))
    ax.set_xlabel(r"$I_c$  [mol/kg]")
    ax.set_ylabel(r"$x_{\mathrm{CO_2}}^{\mathrm{aq}}$  [‰]")
    ax.grid(alpha=0.25)
    bold_ticks(ax)
    # Headline annotation: state the (T, P, z_CO2) regime.  Pure
    # mathtext (no \ce -- matplotlib doesn't render it).
    ax.text(0.97, 0.97,
            f"binary $\\mathrm{{CO_2}}$--$\\mathrm{{H_2O}}$\n"
            f"$T$ = {T_use:.0f} K, $P$ = {P_use:.0f} bar\n"
            f"feed $z_{{\\mathrm{{CO_2}}}}$ = {z_CO2:.2f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="black", pad=4))
    out_path = out_dir / f"salting_out_T{T_use:.0f}K_P{P_use:.0f}bar.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> {out_path}")
    return out_path


def figure_selectivity(df: pd.DataFrame, out_dir: Path,
                       z_pick=(0.49, 0.255, 0.255)):
    """``alpha = (y_CO2 / x_CO2,aq) / (y_CH4 / x_CH4,aq)`` over (T, P) for
    a representative ternary feed, one panel per salt.  Quantifies the
    \\ce{CO2}/\\ce{CH4} selectivity that drives mixed-gas storage and
    production applications: ``alpha > 1`` means \\ce{CO2} preferentially
    leaves the aqueous (favourable for \\ce{CH4} production); ``alpha < 1``
    means the opposite (favourable for \\ce{CO2} storage)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    conv = df[df["converged"] & (df["phases"] == 2)
              & (df["feed_family"] == "ternary")].copy()
    def _match(s):
        return (abs(s["z_H2O"] - z_pick[0]) < 0.05
                and abs(s["z_CH4"] - z_pick[1]) < 0.05
                and abs(s["z_CO2"] - z_pick[2]) < 0.05)
    sub_all = conv[conv.apply(_match, axis=1)].copy()
    if not len(sub_all):
        print(f"  [selectivity] no matching ternary z≈{z_pick}; skipping")
        return None
    # alpha = (y_CO2 / x_CO2,aq) / (y_CH4 / x_CH4,aq).  Guard against
    # divide-by-zero with a small floor.
    eps = 1e-12
    sub_all["alpha"] = ((sub_all["y_CO2_gas"] / (sub_all["x_CO2_aq"] + eps))
                        / (sub_all["y_CH4_gas"] / (sub_all["x_CH4_aq"] + eps)))
    sub_all["log_alpha"] = np.log10(sub_all["alpha"].clip(lower=1e-3))
    salts = sorted(sub_all["salt_name"].unique())
    ncol = min(4, len(salts))
    nrow = (len(salts) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.6 * nrow),
                             sharex=True, sharey=True,
                             squeeze=False, constrained_layout=True)
    P_axis = np.sort(sub_all["P_bar"].unique())
    T_axis = np.sort(sub_all["T_K"].unique())
    # Symmetric diverging scale around log(alpha)=0 (i.e. equal-selectivity).
    vmag = float(np.nanquantile(sub_all["log_alpha"].abs(), 0.95))
    cmap_obj = mpl.colormaps.get_cmap("RdBu_r").copy()
    cmap_obj.set_bad("#cccccc")
    im = None
    for k, salt in enumerate(salts):
        ax = axes[k // ncol][k % ncol]
        sub = sub_all[sub_all["salt_name"] == salt]
        mat = _pivot_to_grid(
            sub, "log_alpha", label=f"selectivity/{salt}/log_alpha"
        ).reindex(index=T_axis, columns=P_axis)
        im = _imshow_grid(ax, mat, T_axis=T_axis, P_axis=P_axis,
                          vmin=-vmag, vmax=+vmag, cmap=cmap_obj)
        ax.text(0.04, 0.96, salt, transform=ax.transAxes,
                ha="left", va="top", fontsize=11, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2))
    _apply_shared_axes(axes, nrow, ncol, len(salts))
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02, extend="both")
        cbar.set_label(r"$\log_{10}\alpha$  ($\alpha=K_{\mathrm{CO_2}}/K_{\mathrm{CH_4}}$)",
                       fontweight="bold")
    out_path = out_dir / "selectivity_co2_ch4_ternary.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> {out_path}")
    return out_path


def figure_ssi_heatmap_per_salt(df: pd.DataFrame, out_dir: Path,
                                feed_family: str = "ternary",
                                z_pick=(0.49, 0.255, 0.255),
                                metric: str = "n_iter_ssi",
                                cmap: str = "viridis",
                                vmax: float | None = None,
                                label_text: str = "Mean SSI iterations"):
    """Two-row × four-column grid of (T, P) heatmaps, one panel per salt,
    coloured by a per-cell performance metric (default: outer SSI
    iteration count).  Each panel is one salt at one composition, so a
    (T, P) cell holds a single sweep row and the cell value is that
    row's value.  Single-phase cells are masked grey to match the
    prior-paper convention (Moortgat 2025 IECR Fig. 5/6); the one
    exception is a single-phase cell that ``_fill_isolated_defects``
    patches before the reindex, recorded in
    ``tables/render_fill_log.csv``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    conv = df[df["converged"] & (df["feed_family"] == feed_family)]
    def _match(s):
        return (abs(s["z_H2O"] - z_pick[0]) < 0.05
                and abs(s["z_CH4"] - z_pick[1]) < 0.05
                and abs(s["z_CO2"] - z_pick[2]) < 0.05)
    sub_all = conv[conv.apply(_match, axis=1)]
    if not len(sub_all):
        print(f"  [{metric} heatmap] no rows for {feed_family} z≈{z_pick}; skipping")
        return None
    salts = sorted(sub_all["salt_name"].unique())
    ncol = min(4, len(salts))
    nrow = (len(salts) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.6 * nrow),
                             sharex=True, sharey=True,
                             squeeze=False, constrained_layout=True)
    P_axis = np.sort(sub_all["P_bar"].unique())
    T_axis = np.sort(sub_all["T_K"].unique())
    if vmax is None:
        # Cap at 95th percentile of two-phase points so single-phase outliers
        # (which often log very few SSI iterations) don't compress the scale.
        twoph = sub_all[sub_all["phases"] == 2]
        vmax = float(twoph[metric].quantile(0.95)) if len(twoph) else 1.0
    im = None
    for k, salt in enumerate(salts):
        ax = axes[k // ncol][k % ncol]
        sub = sub_all[sub_all["salt_name"] == salt]
        # Build the (T, P) matrix using ONLY two-phase points.
        twoph = sub[sub["phases"] == 2]
        mat = _pivot_to_grid(twoph, metric,
                             label=f"{metric}_{feed_family}/{salt}")
        # Reindex to the full (T_axis, P_axis) so single-phase cells appear
        # explicitly as NaN.  Note the cleaning inside _pivot_to_grid runs
        # before this reindex, so a single-phase cell sitting inside the
        # two-phase pivot's own bounding box can have been filled already;
        # those cells are listed in tables/render_fill_log.csv.
        mat = mat.reindex(index=T_axis, columns=P_axis)
        # Render the metric on a grey-NaN background.
        cmap_obj = mpl.colormaps.get_cmap(cmap).copy()
        cmap_obj.set_bad("#cccccc")  # single-phase / no-data → grey
        im = _imshow_grid(ax, mat, T_axis=T_axis, P_axis=P_axis,
                          vmin=0.0, vmax=vmax, cmap=cmap_obj)
        ax.text(0.04, 0.96, salt, transform=ax.transAxes,
                ha="left", va="top", fontsize=11, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2))
        # Small in-panel annotation of the two-phase region.  The
        # colouring on each panel is the metric of interest (SSI count,
        # wall time, ...) computed on the two-phase points only; the
        # single-phase area shows as grey via ``cmap.set_bad``.  The
        # label used to read "single phase" but was placed in the
        # bottom-right corner — which for a log-P heatmap with T on the
        # y axis lands inside the two-phase colour field, not the grey
        # single-phase corner.  Keep the position (unambiguous inside a
        # coloured region) but relabel to "two-phase" so it names the
        # region it actually sits on.
        ax.text(0.96, 0.04, "two-phase", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9, color="#444444",
                style="italic")
    _apply_shared_axes(axes, nrow, ncol, len(salts))
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02, extend="max")
        cbar.set_label(label_text, fontweight="bold")
    out_path = out_dir / f"{metric}_{feed_family}_heatmap.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> {out_path}")
    return out_path


def figure_ssi_heatmap_by_feed_family(df: pd.DataFrame, out_dir: Path,
                                      metric: str = "n_iter_ssi",
                                      cmap: str = "viridis",
                                      label_text: str = "Mean SSI iterations",
                                      vmax: float | None = None):
    """Compact three-panel main-text figure: mean SSI iteration count
    (or another metric) averaged over the salt configurations at every
    ``(T, P)`` cell, one panel per feed family
    (binary \\ce{CO2}--\\ce{H2O}, binary \\ce{CH4}--\\ce{H2O},
    ternary \\ce{CH4}--\\ce{CO2}--\\ce{H2O}).  Single-phase cells stay
    grey via ``cmap.set_bad``, so the phase boundary in each panel is
    visible as the interface between grey and coloured regions.

    Designed as the sole efficiency-and-robustness figure in the main
    text; per-salt breakdowns and the split-timing / SSI-vs-SSI+NR
    figures live in the SI.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Representative z picks per feed family (match the SI figures'
    # composition slices so main and SI stay consistent).  The panel
    # labels are rendered through matplotlib's mathtext (not real
    # LaTeX), so ``\ce{}`` from the mhchem package would appear as raw
    # text — use plain math-mode subscripts instead.
    picks = [
        ("binary_CO2", (0.50, 0.00, 0.50), r"binary CO$_2$-H$_2$O"),
        ("binary_CH4", (0.50, 0.50, 0.00), r"binary CH$_4$-H$_2$O"),
        ("ternary",    (0.49, 0.255, 0.255),
         r"ternary CH$_4$-CO$_2$-H$_2$O"),
    ]
    conv = df[df["converged"]]

    # Global vmax: 95th-pct of two-phase points across ALL three families,
    # so the colour scale is comparable across panels.
    if vmax is None:
        twoph_all = conv[conv["phases"] == 2]
        vmax = float(twoph_all[metric].quantile(0.95)) if len(twoph_all) else 1.0

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4),
                             sharex=True, sharey=True,
                             squeeze=False, constrained_layout=True)
    P_axis_master = np.sort(conv["P_bar"].unique())
    T_axis_master = np.sort(conv["T_K"].unique())
    im = None
    for k, (fam, z_pick, label) in enumerate(picks):
        ax = axes[0][k]
        sub_all = conv[(conv["feed_family"] == fam)
                       & ((conv["z_H2O"] - z_pick[0]).abs() < 0.05)
                       & ((conv["z_CH4"] - z_pick[1]).abs() < 0.05)
                       & ((conv["z_CO2"] - z_pick[2]).abs() < 0.05)]
        if not len(sub_all):
            ax.axis("off")
            continue
        # Average across salt configurations at each (T, P): use ONLY
        # two-phase points, then mean over ``salt_name``.  This is the
        # only figure whose cells pool several salts, so it is the only
        # one where the aggregator is visible in the result -- pass
        # ``aggfunc="mean"`` explicitly so the surface is the mean the
        # colourbar, docstring and caption all name.
        twoph = sub_all[sub_all["phases"] == 2]
        mat = _pivot_to_grid(twoph, metric, aggfunc="mean",
                             label=f"by_feed_family/{fam}/{metric}")
        # Ensure axes match the global master grid for consistent layout.
        mat = mat.reindex(index=T_axis_master, columns=P_axis_master)
        cmap_obj = mpl.colormaps.get_cmap(cmap).copy()
        cmap_obj.set_bad("#cccccc")
        im = _imshow_grid(ax, mat, T_axis=T_axis_master, P_axis=P_axis_master,
                          vmin=0.0, vmax=vmax, cmap=cmap_obj)
        # Panel label in top-left.
        ax.text(0.04, 0.96, label, transform=ax.transAxes,
                ha="left", va="top", fontsize=11, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.85,
                          edgecolor="none", pad=2))
        ax.text(0.96, 0.04, "two-phase", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9, color="#444444",
                style="italic")
        # Y label only on leftmost panel; X label on every panel.
        if k > 0:
            ax.set_ylabel("")
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02, extend="max")
        cbar.set_label(label_text, fontweight="bold")
    out_path = out_dir / f"{metric}_by_feed_family_heatmap.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> {out_path}")
    return out_path


def table_perf_main_latex(df: pd.DataFrame, out_path: Path):
    """Compact main-text convergence-and-timing summary as a LaTeX
    ``tabular`` snippet.  Reports the headline numbers of the dense
    sweep — 100% convergence claim, single-phase vs two-phase point
    counts, mean stability and flash wall times, and the SSI+NR-polish
    vs SSI-only iteration / wall-time comparison — in a single
    3-column layout suitable for a main-paper §Results subsection.
    """
    conv = df[df["converged"]]
    total = int(len(df))
    total_conv = int(len(conv))
    pct_exact = 100.0 * total_conv / max(total, 1)
    # "100" is reserved for total_conv == total, and nothing else.
    #
    # This used to be a ROUNDING test -- pct_rounded >= 100.0 -- which fires
    # at any rate above 99.995 %, i.e. with up to 7 non-converged points on
    # the dense grid and around 50 on the "full" one. The table then printed
    # "Converged  151,631 (100%)" on the line below "Grid points evaluated
    # 151,632", where the two cells contradict each other and the reader is
    # expected to trust the second one. The exact rate was disclosed on
    # stdout and in a LaTeX comment, neither of which reaches the page.
    #
    # A convergence rate can be checked by subtraction, so it states what
    # happened: an exact "100" when
    # every point converged, and the true rate to two decimals when they did
    # not. If that reads as 99.99 %, the fix belongs in the solver.
    # Two decimals are not enough: 151,631/151,632 is 99.99934 %, which
    # renders as "100.00" and recreates the illusion one decimal deeper.
    # When the rate is not exactly 100, widen the precision until the
    # printed string is distinguishable from 100 -- so a single failure in
    # 151,632 is visible on the page rather than rounded off it.
    if total_conv == total:
        pct = "100"
    else:
        for _dp in (2, 3, 4, 5, 6, 7):
            pct = f"{pct_exact:.{_dp}f}"
            if float(pct) < 100.0:
                break
    if total_conv != total:
        print(f"  [perf_main] {total - total_conv} of {total} grid points did "
              f"not converge; the table cell reads \"{pct}\" (exact "
              f"{pct_exact:.5f}%), NOT 100.")
    sp = conv[conv["phases"] == 1]
    tp = conv[conv["phases"] == 2]

    # Column: (mean SSI, mean NR, mean t_flash) for SSI+NR default.
    ssi_nr_ssi = tp["n_iter_ssi"].mean()
    ssi_nr_nr = tp["n_iter_newton_polish"].mean()
    ssi_nr_t = tp["t_flash_ms"].mean() if "t_flash_ms" in tp.columns else float("nan")

    lines = [
        f"% convergence: {total_conv}/{total} = {pct_exact:.5f}% "
        f"(cell below shows \"{pct}\"; LaTeX ignores this line)",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r" Metric & Value \\",
        r"\midrule",
        r" Grid points evaluated & "
        f"{total:,} \\\\".replace(",", r"\,"),
        r" Converged & "
        f"{total_conv:,} ({pct}\\%)".replace(",", r"\,")
        + r" \\",
        r" \quad classified single-phase & "
        f"{len(sp):,}".replace(",", r"\,") + r" \\",
        r" \quad classified two-phase & "
        f"{len(tp):,}".replace(",", r"\,") + r" \\",
        r"\midrule",
        r" $\bar{t}_{\text{stab}}$ per single-phase state [ms] & "
        f"{sp['t_stab_ms'].mean():.0f}"
        + r" \\",
        r" $\bar{t}_{\text{stab}}$ per two-phase state [ms] & "
        f"{tp['t_stab_ms'].mean():.0f}"
        + r" \\",
        r" $\bar{t}_{\text{flash}}$ per two-phase state [ms] & "
        f"{ssi_nr_t:.0f}"
        + r" \\",
        r"\midrule",
        r" $\bar{N}_{\text{SSI}}$ per two-phase state (SSI$+$NR polish) & "
        f"{ssi_nr_ssi:.1f}" + r" \\",
        r" $\bar{N}_{\text{NR}}$ per two-phase state & "
        f"{ssi_nr_nr:.2f}" + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def figure_aqueous_density(df: pd.DataFrame, out_dir: Path,
                           z_CO2: float = 0.05):
    """ρ_aq surface over (T, P) for binary_CO2 at low z_CO2 (≈dilute brine),
    one panel per salt.  Validation extends prior brine-density work (Al
    Ghafri 2012/2013) into the CO2-saturated regime."""
    out_dir.mkdir(parents=True, exist_ok=True)
    conv = df[df["converged"] & (df["feed_family"] == "binary_CO2")
              & ((df["z_CO2"] - z_CO2).abs() < 0.05)]
    if not len(conv):
        print(f"  [density] no rows for binary_CO2 z_CO2≈{z_CO2}; skipping")
        return None
    salts = sorted(conv["salt_name"].unique())
    ncol = min(4, len(salts))
    nrow = (len(salts) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.6 * nrow),
                             sharex=True, sharey=True,
                             squeeze=False, constrained_layout=True)
    P_axis = np.sort(conv["P_bar"].unique())
    T_axis = np.sort(conv["T_K"].unique())
    # Color scale anchored on the bulk of the data (drop low end where the
    # gas-rich split would give a misleading aq density).
    finite_rho = conv["rho_aq"].dropna()
    vmin = float(np.nanquantile(finite_rho, 0.01))
    vmax = float(np.nanquantile(finite_rho, 0.99))
    im = None
    for k, salt in enumerate(salts):
        ax = axes[k // ncol][k % ncol]
        sub = conv[conv["salt_name"] == salt]
        # Reindex to the union axes (see figure_phase_boundary).
        mat = _pivot_to_grid(
            sub, "rho_aq", label=f"density/z{z_CO2:.2f}/{salt}/rho_aq"
        ).reindex(index=T_axis, columns=P_axis)
        im = _imshow_grid(ax, mat, T_axis=T_axis, P_axis=P_axis,
                          vmin=vmin, vmax=vmax, cmap="plasma")
        ax.text(0.04, 0.96, salt, transform=ax.transAxes,
                ha="left", va="top", fontsize=11, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2))
    _apply_shared_axes(axes, nrow, ncol, len(salts))
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
        cbar.set_label(r"$\rho^{\mathrm{aq}}$ [kg/m$^3$]", fontweight="bold")
    out_path = out_dir / f"density_binary_zCO2_{z_CO2:.2f}.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--merged", type=Path, required=True,
                   help="Path to merged.parquet from a sweep_postprocess run.")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Output directory for tables/ and figures/.")
    p.add_argument("--clean", action="store_true",
                   help="Restore neighbour-median cell filling (off by "
                        "default). Fills unsolved cells and REPLACES finite "
                        "cells that deviate from their neighbours, which on "
                        "the current grid means overwriting values the solver "
                        "produced. For comparison against older figures.")
    p.add_argument("--no-clean", action="store_true",
                   help="Explicitly disable cell filling. This is the "
                        "default; the flag is accepted so existing commands "
                        "keep working.")
    args = p.parse_args(argv)

    global _CLEAN_DEFAULT
    _CLEAN_DEFAULT = bool(args.clean) and not args.no_clean
    if _CLEAN_DEFAULT:
        print("cell filling ENABLED (--clean): finite cells deviating from "
              "their neighbours will be OVERWRITTEN; see tables/"
              "render_fill_log.csv for every cell touched.")
    else:
        print("cell filling off: every surface shows the sweep's own cells.")

    print(f"Loading {args.merged}")
    df = pd.read_parquet(args.merged)
    print(f"  {len(df)} rows, {df.converged.sum()} converged "
          f"({100*df.converged.sum()/len(df):.3f}%)")
    # ULP-normalise grid coordinates so that P_bar / T_K values that
    # differ only in the last ULP of their float64 representation
    # (produced by independent ``np.geomspace`` calls in different
    # sweep chunks) collapse to a single canonical value.  Without
    # this, ``pivot_table`` returns near-duplicate columns that
    # ``imshow`` renders as thin vertical stripes on the log-P axis.
    # Round in-place before anything else touches the grid axes.
    df["P_bar"] = df["P_bar"].map(lambda p: float(f"{p:.8g}"))
    df["T_K"]   = df["T_K"].round(6)
    n_P = df["P_bar"].nunique()
    n_T = df["T_K"].nunique()
    print(f"  grid: {n_T} unique T_K × {n_P} unique P_bar after ULP normalise")

    apply_paper_style()
    # Start the cleanup log empty so a second main() call in the same
    # process does not inherit the first run's patched cells.
    _FILL_LOG.clear()
    tab_dir = args.out_dir / "tables"
    fig_dir = args.out_dir / "figures"

    print("\nTables:")
    t1 = table_overall_performance(df, tab_dir / "performance_overall.csv")
    print(t1.to_string(index=False))
    print()
    t2 = table_performance_by_family(df, tab_dir / "performance_by_family.csv")
    print(t2.to_string(index=False))
    print()
    t3 = table_two_phase_fraction(df, tab_dir / "two_phase_fraction.csv")
    print(t3.to_string(index=False))
    print()
    t4 = table_perf_latex(df, tab_dir / "performance_summary.tex")
    print("performance_summary.tex written; head:")
    print((tab_dir / "performance_summary.tex").read_text().splitlines()[0:6])
    print()
    t5 = table_perf_by_family_latex(df, tab_dir / "performance_by_family.tex")
    print(f"performance_by_family.tex written ({len(t5)} rows)")
    t6 = table_canonical_solubility_latex(df, tab_dir / "canonical_solubility.tex")
    print(f"canonical_solubility.tex written ({len(t6)} salts × 8 (T,P) cells)")
    t7 = table_perf_main_latex(df, tab_dir / "perf_main_summary.tex")
    print(f"perf_main_summary.tex written")

    print("\nFigures:")
    # Compact three-panel main-text efficiency figure: mean SSI count
    # per (T, P) averaged over salts, one panel per feed family.  The
    # single-phase area shows as grey so the two-phase boundary is also
    # visible.
    figure_ssi_heatmap_by_feed_family(df, fig_dir,
                                      metric="n_iter_ssi",
                                      cmap="viridis",
                                      label_text="Mean SSI iterations")
    figure_phase_boundary(df, fig_dir,
                          feed_family="ternary",
                          z_pick=(0.49, 0.255, 0.255))
    figure_co2_solubility(df, fig_dir, z_CO2=0.50)
    figure_co2_solubility(df, fig_dir, z_CO2=0.05)
    # Aqueous density at z_CO2 = 0.25 (Paper III, Figure S12). The grid of
    # the published sweep holds z_CO2 = 0.25, 0.50 and 0.75 for the binary
    # CO2-H2O feed; calls for a slice the grid does not hold print a note
    # and skip.
    figure_aqueous_density(df, fig_dir, z_CO2=0.25)
    # Prior-paper-style heatmaps (Moortgat 2025 IECR Fig. 5/6 analogues):
    figure_ssi_heatmap_per_salt(df, fig_dir,
                                feed_family="ternary",
                                metric="n_iter_ssi",
                                cmap="viridis",
                                label_text="Mean SSI iterations")
    figure_ssi_heatmap_per_salt(df, fig_dir,
                                feed_family="ternary",
                                metric="t_wall_ms",
                                cmap="plasma",
                                label_text="Mean wall time [ms]")
    figure_ssi_heatmap_per_salt(df, fig_dir,
                                feed_family="binary_CO2",
                                z_pick=(0.5, 0.0, 0.5),
                                metric="n_iter_ssi",
                                cmap="viridis",
                                label_text="Mean SSI iterations")
    # Headline application figures:
    figure_salting_out(df, fig_dir, T_K=323.0, P_bar=100.0, z_CO2=0.50)
    figure_salting_out(df, fig_dir, T_K=423.0, P_bar=300.0, z_CO2=0.50)
    figure_selectivity(df, fig_dir, z_pick=(0.49, 0.255, 0.255))

    # Surface figures for the SI:
    # Water content in the gas / CO2-rich phase — the headline metric for
    # the salinity-convention argument in §3 of the paper (water transfer
    # into the gas phase grows with T and underlies the m_prep ≠ m_aq*
    # correction discussion).
    figure_generic_surface_per_salt(df, fig_dir,
        feed_family="binary_CO2", z_pick=(0.5, 0.0, 0.5),
        metric="y_H2O_gas", scale=100.0,
        label_text=r"$y_{\mathrm{H_2O}}^{\mathrm{gas}}$  [%]",
        out_stem="yH2O_gas_binary_CO2_surface", cmap="cividis",
        vmin=0.0, vmax=60.0)
    figure_generic_surface_per_salt(df, fig_dir,
        feed_family="binary_CH4", z_pick=(0.5, 0.5, 0.0),
        metric="y_H2O_gas", scale=100.0,
        label_text=r"$y_{\mathrm{H_2O}}^{\mathrm{gas}}$  [%]",
        out_stem="yH2O_gas_binary_CH4_surface", cmap="cividis",
        vmin=0.0, vmax=60.0)
    # CH4 solubility — counterpart to the CO2 solubility figures.
    figure_generic_surface_per_salt(df, fig_dir,
        feed_family="binary_CH4", z_pick=(0.5, 0.5, 0.0),
        metric="x_CH4_aq", scale=1000.0,
        label_text=r"$x_{\mathrm{CH_4}}^{\mathrm{aq}}$  [‰]",
        out_stem="ch4_solubility_binary_zCH4_0.50", cmap="viridis")
    # Compressibility factors — useful warm-start state and gives a sense
    # of the dense-fluid regime where the inner Newton has to work hard.
    figure_generic_surface_per_salt(df, fig_dir,
        feed_family="binary_CO2", z_pick=(0.5, 0.0, 0.5),
        metric="Z_aq",
        label_text=r"$Z^{\mathrm{aq}}$",
        out_stem="Z_aq_binary_CO2_surface", cmap="plasma")
    figure_generic_surface_per_salt(df, fig_dir,
        feed_family="binary_CO2", z_pick=(0.5, 0.0, 0.5),
        metric="Z_gas",
        label_text=r"$Z^{\mathrm{gas}}$",
        out_stem="Z_gas_binary_CO2_surface", cmap="plasma")
    # Aqueous permittivity (eps_r) — drops with T, shifts with salt
    # composition through the Maribo-Mogensen self-consistency loop.
    figure_generic_surface_per_salt(df, fig_dir,
        feed_family="binary_CO2", z_pick=(0.5, 0.0, 0.5),
        metric="eps_r",
        label_text=r"$\varepsilon_r$  (aqueous)",
        out_stem="eps_r_binary_CO2_surface", cmap="cividis_r")
    # Aqueous H2O association fraction — Wertheim chi1w, controls the
    # ion-association coupling.  Useful internal-state visualization.
    figure_generic_surface_per_salt(df, fig_dir,
        feed_family="binary_CO2", z_pick=(0.5, 0.0, 0.5),
        metric="chi_H2O_aq",
        label_text=r"$\chi_{\mathrm{H_2O}}^{\mathrm{aq}}$",
        out_stem="chi_H2O_aq_binary_CO2_surface", cmap="magma_r")

    # Provenance for the rendering-side cleanup: one row per cell that
    # ``_fill_isolated_defects`` changed, with the panel it belongs to,
    # its (T, P), and the value before and after.  A patched cell is not
    # distinguishable from a solved one in the PNG, so this file is the
    # only record that it was patched; check it before quoting a value
    # off a surface figure.
    if _FILL_LOG:
        fill_log = pd.DataFrame(_FILL_LOG)
        fill_path = tab_dir / "render_fill_log.csv"
        fill_log.to_csv(fill_path, index=False)
        n_nan = int((fill_log["pass"] == "nan_fill").sum())
        n_out = int((fill_log["pass"] == "outlier").sum())
        print(f"\n[clean] _fill_isolated_defects patched {len(fill_log)} cell(s) "
              f"across all figures ({n_nan} NaN fill, {n_out} outlier "
              f"replacement) -> {fill_path}")
    else:
        print("\n[clean] _fill_isolated_defects patched no cells")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
