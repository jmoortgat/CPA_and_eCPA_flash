"""
Validation of the eCPA CO₂ + H₂O + NaCl flash against experimental data.

Public API
----------
load_co2nacl_exp(data_dir, parquet_cache)
    Parse EXP/CO2-NaCl directory → long-format DataFrame.

run_validation(exp_df, guess_table_fn, params, ...)
    Run flash at every experimental condition; return results DataFrame.

compute_metrics(results_df)
    Compute AARE, RMSE, bias overall and grouped by T, ms, qty type.

plot_validation_parity(results_df, ...)
    Parity plots (predicted vs experimental) per quantity type.

plot_validation_T(results_df, ...)
    Solubility curves vs P, one panel per temperature.

plot_error_heatmap(results_df, ...)
    AARE heatmap over (T, ms) grid.
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable

from .constants import Mw


# ── Data loading ───────────────────────────────────────────────────────────────

_DEFAULT_NACL_DIR     = Path("EXP/CO2-NaCl")
_DEFAULT_NACL_PARQUET = Path("EXP/CO2-NaCl/co2_nacl_exp.parquet")


def _parse_co2nacl_dir(data_dir: Path) -> pd.DataFrame:
    """Parse all EXP*.txt files in the CO2-NaCl directory tree.

    Returns a long-format DataFrame with one row per (datapoint, quantity).
    """
    records = []
    for txt_path in sorted(data_dir.rglob("EXP*.txt")):
        # Skip superseded copies marked with _X suffix
        if txt_path.stem.endswith("_X") or "(copy)" in txt_path.name.lower():
            continue
        m_T = re.search(r"T(\d+)K", str(txt_path))
        if not m_T:
            continue
        T_K = int(m_T.group(1))
        m_exp = re.search(r"EXP(\d+)_", txt_path.name)
        exp_id = int(m_exp.group(1)) if m_exp else 0

        lines = txt_path.read_text(encoding="utf-8", errors="replace").splitlines()

        reference = ""
        col_header_idx = None
        data_start = 0

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("#"):
                if not reference:
                    reference = stripped.lstrip("#").strip()
            elif stripped.lower().startswith("p ["):
                col_header_idx = i
                data_start = i + 1
                break

        if col_header_idx is None:
            continue

        # Parse column header: identify quantity types and their column indices
        header_raw = lines[col_header_idx]
        # Separate bracket annotations from token names
        header_clean = re.sub(r"\[.*?\]", "", header_raw)
        tokens = header_clean.lower().split()

        # Find bracket annotations per token position to distinguish SALTfree/SALTincl
        bracket_map = {}
        for m in re.finditer(r"(\S+)\s*\[([^\]]*)\]", header_raw, re.IGNORECASE):
            name_clean = re.sub(r"\s+", "", m.group(1)).lower()
            bracket_map[name_clean] = m.group(2).strip()

        # Build column specs: list of (col_idx, qty_name)
        qty_cols = []  # (col_idx, qty_name)
        for j, tok in enumerate(tokens):
            if tok == "p":
                pass  # col 0 always
            elif tok == "ms":
                pass  # col 1 always
            elif tok == "mc":
                qty_cols.append((j, "mc"))
            elif tok == "xc_w":
                bracket = bracket_map.get("xc_w", "")
                if "saltfree" in bracket.lower():
                    qty_cols.append((j, "xc_W_SALTfree"))
                else:
                    qty_cols.append((j, "xc_W_SALTincl"))
            elif tok == "xc_c":
                bracket = bracket_map.get("xc_c", "")
                qty_cols.append((j, "xc_C"))

        if not qty_cols:
            continue

        def _v(s):
            try:
                return np.nan if s.lower() == "x" else float(s)
            except ValueError:
                return np.nan

        for line in lines[data_start:]:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 3:
                continue
            try:
                P_bar = float(parts[0])
                ms    = float(parts[1])
            except ValueError:
                continue

            for col_idx, qty_name in qty_cols:
                if col_idx < len(parts):
                    val = _v(parts[col_idx])
                    if not np.isnan(val):
                        records.append({
                            "T_K":        T_K,
                            "P_bar":      P_bar,
                            "ms":         ms,
                            "value":      val,
                            "qty":        qty_name,
                            "exp_id":     exp_id,
                            "reference":  reference,
                            "source_file": txt_path.name,
                        })

    df = pd.DataFrame(records)
    df = df.sort_values(["T_K", "ms", "P_bar"]).reset_index(drop=True)
    return df


def load_co2nacl_exp(
    data_dir: Path = _DEFAULT_NACL_DIR,
    parquet_cache: Path = _DEFAULT_NACL_PARQUET,
    force_reparse: bool = False,
) -> pd.DataFrame:
    """Load CO2-NaCl experimental data (parses from text files or loads cache)."""
    if not force_reparse and parquet_cache.exists():
        return pd.read_parquet(parquet_cache)
    df = _parse_co2nacl_dir(data_dir)
    parquet_cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_cache, index=False)
    return df


# ── Flash runner ───────────────────────────────────────────────────────────────

def _z_candidates(qty: str, value: float) -> list:
    """Return z_co2 candidates to try, in priority order."""
    if qty == "mc":
        x1w_est = value * Mw          # rough mole fraction
    elif qty == "xc_W_SALTincl":
        x1w_est = value
    elif qty in ("xc_W_SALTfree", "xc_C"):
        x1w_est = value
    else:
        x1w_est = 0.3

    candidates = [
        x1w_est * 2.0,
        x1w_est * 3.0,
        0.3,
        0.5,
        0.1,
        0.7,
    ]
    # Clip to physically valid range, deduplicate
    seen = set()
    out = []
    for z in candidates:
        z = float(np.clip(z, 0.02, 0.95))
        key = round(z, 4)
        if key not in seen:
            seen.add(key)
            out.append(z)
    return out


def _pred_to_unit(qty: str, x_aq: dict, x_c: dict) -> float:
    """Convert flash output to the experimental unit.

    eCPA ELV component numbering convention:
      x1w = H2O mole fraction in aqueous phase  (component 1 = H2O)
      x2w = Na+ mole fraction in aqueous phase
      x3w = Cl- mole fraction in aqueous phase
      x4w = CO2 mole fraction in aqueous phase  (component 4 = CO2 in aq.)
      x1c = H2O mole fraction in CO2-rich phase
      x4c = CO2 mole fraction in CO2-rich phase = 1 - x1c
    """
    x1w = x_aq["x1w"]   # H2O in aqueous phase
    x4w = x_aq["x4w"]   # CO2 in aqueous phase
    x1c = x_c["x1c"]    # H2O in CO2-rich phase
    # x4c = CO2 in CO2-rich phase = 1 - x1c
    if qty == "mc":
        # CO2 molality = n_CO2 / (n_H2O * Mw) = x4w / (x1w * Mw)
        if x1w <= 0:
            return np.nan
        return x4w / (x1w * Mw)
    elif qty == "xc_W_SALTincl":
        # CO2 mole fraction in aqueous phase, salt-inclusive basis
        return x4w
    elif qty == "xc_W_SALTfree":
        # CO2 mole fraction on CO2 + H2O basis (excluding salt ions)
        denom = x4w + x1w
        return x4w / denom if denom > 0 else np.nan
    elif qty == "xc_C":
        # CO2 mole fraction in CO2-rich phase = 1 - x1c
        return 1.0 - x1c
    return np.nan


def _predict_one(row, guess_table_fn, params):
    """Run flash for one experimental row; return result dict."""
    from .stability import ecpa_stability_flash

    T    = float(row["T_K"])
    P    = float(row["P_bar"])
    ms   = float(row["ms"])
    qty  = row["qty"]
    val_exp = float(row["value"])

    base = {
        "T_K": T, "P_bar": P, "ms": ms, "qty": qty,
        "value_exp": val_exp,
        "value_pred": np.nan,
        "rel_err":    np.nan,
        "abs_rel_err": np.nan,
        "status":     "flash_failed",
        "z_co2_used": np.nan,
        "ms_aq_pred": np.nan,
        "x1w_pred":   np.nan,
        "x1c_pred":   np.nan,
        "n_iter_ms":  -1,
        "reference":  row["reference"],
        "source_file": row["source_file"],
        "exp_id":     row["exp_id"],
    }

    candidates = _z_candidates(qty, val_exp)
    last_err = ""

    for z in candidates:
        try:
            out = ecpa_stability_flash(
                z_co2=z, ms=ms, T=T, P=P,
                params=params,
            )
            if out.get("phase") in ("single_phase",):
                raise ValueError(f"single_phase at z={z:.4f}")
        except Exception as e:
            last_err = str(e)
            continue

        # Sanity: ms_aq should be close to the input ms (salt non-volatile)
        ms_aq = float(out["ms_aq"])
        if ms_aq <= 0 or abs(ms_aq - ms) / ms > 0.5:
            last_err = f"ms_aq={ms_aq:.3f} far from input ms={ms:.3f}"
            continue

        x_aq = out["x_aq"]
        x_c  = out["x_c"]
        # x1w = H2O in aqueous phase; x4w = CO2 in aqueous phase
        if x_aq["x1w"] <= 0:
            last_err = "x1w <= 0 (nonphysical H2O)"
            base["status"] = "nonphysical"
            continue

        pred = _pred_to_unit(qty, x_aq, x_c)
        if pred is None or not np.isfinite(pred) or pred <= 0:
            last_err = f"pred={pred} nonphysical"
            base["status"] = "nonphysical"
            continue

        rel_err = (pred - val_exp) / val_exp

        base.update({
            "value_pred":  pred,
            "rel_err":     rel_err,
            "abs_rel_err": abs(rel_err),
            "status":      "ok",
            "z_co2_used":  z,
            "ms_aq_pred":  ms_aq,
            "x1w_pred":    x_aq["x1w"],
            "x1c_pred":    x_c["x1c"],
            "n_iter_ms":   out.get("n_iter_ms", -1),
        })
        return base

    base["status"] = "flash_failed"
    base["_last_err"] = last_err
    return base


def run_validation(
    exp_df,
    guess_table_fn,
    params=None,
    T_max: float = 523.0,
    ms_max: float = 3.5,
    n_workers: int = 1,
    verbose: bool = False,
    failure_log: str = "results/validation_failures_co2nacl.txt",
) -> pd.DataFrame:
    """
    Run flash at every experimental (T, P, ms) condition and compare to data.

    Parameters
    ----------
    exp_df : DataFrame from load_co2nacl_exp()
    guess_table_fn : callable (T, P_bar) → np.ndarray, from make_guess_fn()
    params : eCPA parameters dict (from make_params())
    T_max : filter; skip T > T_max K
    ms_max : filter; skip ms > ms_max mol/kg
    n_workers : parallel workers (1 = serial)
    verbose : print progress
    failure_log : path to write failed conditions

    Returns
    -------
    DataFrame with one row per experimental point + model predictions.
    """
    df = exp_df[(exp_df["T_K"] <= T_max) & (exp_df["ms"] <= ms_max)].copy()
    rows = df.to_dict("records")
    n = len(rows)

    if verbose:
        print(f"Running validation on {n} points "
              f"(T ≤ {T_max}K, ms ≤ {ms_max} mol/kg) ...")

    results = []
    if n_workers <= 1:
        for k, row in enumerate(rows):
            r = _predict_one(row, guess_table_fn, params)
            results.append(r)
            if verbose and (k + 1) % 50 == 0:
                print(f"  {k+1}/{n}", flush=True)
    else:
        import concurrent.futures
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=__import__("multiprocessing").get_context("spawn"),
            initializer=_val_worker_init,
            initargs=(guess_table_fn, params),
        ) as exe:
            futs = {exe.submit(_val_worker, row): k for k, row in enumerate(rows)}
            done = 0
            for fut in concurrent.futures.as_completed(futs):
                results.append(fut.result())
                done += 1
                if verbose and done % 50 == 0:
                    print(f"  {done}/{n}", flush=True)

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values(["T_K", "ms", "P_bar"]).reset_index(drop=True)

    # Summary
    ok = (res_df["status"] == "ok").sum()
    failed = n - ok
    print(f"\nValidation summary")
    print(f"  Total points (filtered): {n}")
    print(f"  Converged (ok):          {ok}")
    print(f"  Failed / skipped:        {failed}")
    for s in ["flash_failed", "nonphysical", "single_phase"]:
        cnt = (res_df["status"] == s).sum()
        if cnt:
            print(f"    {s}: {cnt}")

    # Write failure log
    if failure_log and failed > 0:
        Path(failure_log).parent.mkdir(parents=True, exist_ok=True)
        with open(failure_log, "w") as fh:
            fh.write("# CO2-NaCl validation failures\n")
            for _, r in res_df[res_df["status"] != "ok"].iterrows():
                fh.write(
                    f"T={r['T_K']}K P={r['P_bar']:.1f}bar ms={r['ms']:.2f} "
                    f"qty={r['qty']} status={r['status']}\n"
                )

    return res_df


# Parallel worker helpers (must be top-level for spawn)
_VAL_GUESS_FN = None
_VAL_PARAMS   = None

def _val_worker_init(guess_table_fn, params):
    global _VAL_GUESS_FN, _VAL_PARAMS
    _VAL_GUESS_FN = guess_table_fn
    _VAL_PARAMS   = params

def _val_worker(row):
    return _predict_one(row, _VAL_GUESS_FN, _VAL_PARAMS)


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(results_df: pd.DataFrame) -> dict:
    """Compute AARE, RMSE, bias overall and grouped by T, ms range, qty type."""
    ok = results_df[results_df["status"] == "ok"].copy()
    n_total = len(results_df)
    n_ok    = len(ok)

    def _metrics(sub):
        if len(sub) == 0:
            return {"N": 0, "AARE_%": np.nan, "bias_%": np.nan,
                    "RMSE": np.nan, "max_ARE_%": np.nan}
        return {
            "N":        len(sub),
            "AARE_%":   sub["abs_rel_err"].mean() * 100,
            "bias_%":   sub["rel_err"].mean() * 100,
            "RMSE":     np.sqrt(((sub["value_pred"] - sub["value_exp"])**2).mean()),
            "max_ARE_%": sub["abs_rel_err"].max() * 100,
        }

    overall = _metrics(ok)
    overall["N_total"] = n_total
    overall["N_failed"] = n_total - n_ok

    def _groupby_metrics(df, col):
        rows = []
        for key, sub in df.groupby(col):
            m = _metrics(sub)
            m[col] = key
            rows.append(m)
        return pd.DataFrame(rows)

    by_T = _groupby_metrics(ok, "T_K").sort_values("T_K").reset_index(drop=True)

    ms_bins   = [0, 1, 2, 3.5, 5.0, 7.0]
    ms_labels = ["0–1", "1–2", "2–3.5", "3.5–5", "5–7"]
    ok2 = ok.copy()
    ok2["ms_bin"] = pd.cut(ok2["ms"], bins=ms_bins, labels=ms_labels, right=False)
    by_ms  = _groupby_metrics(ok2.dropna(subset=["ms_bin"]), "ms_bin").reset_index(drop=True)
    by_qty = _groupby_metrics(ok, "qty").reset_index(drop=True)

    return {
        "overall": overall,
        "by_T":    by_T,
        "by_ms":   by_ms,
        "by_qty":  by_qty,
    }


def print_metrics(metrics: dict):
    """Pretty-print the metrics dict."""
    o = metrics["overall"]
    print(f"\n{'='*60}")
    print(f"Overall  N={o['N']}/{o['N_total']}  "
          f"AARE={o['AARE_%']:.2f}%  bias={o['bias_%']:+.2f}%  "
          f"RMSE={o['RMSE']:.4g}  max_ARE={o['max_ARE_%']:.1f}%")
    print(f"  Failed/skipped: {o['N_failed']}")

    print(f"\n{'─'*60}")
    print("By temperature:")
    print(metrics["by_T"].to_string(index=False, float_format="{:.2f}".format))

    print(f"\n{'─'*60}")
    print("By ms range:")
    print(metrics["by_ms"].to_string(index=False, float_format="{:.2f}".format))

    print(f"\n{'─'*60}")
    print("By quantity type:")
    print(metrics["by_qty"].to_string(index=False, float_format="{:.2f}".format))


# ── Plots ──────────────────────────────────────────────────────────────────────

def _save(fig, path):
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")


def plot_validation_T(results_df, save_path=None, T_max=523, ms_max=3.5):
    """
    Solubility vs P, one panel per temperature.
    Experimental data as open circles; model as filled dots.
    Color encodes ms value.
    """
    ok = results_df[
        (results_df["status"] == "ok") &
        (results_df["T_K"] <= T_max) &
        (results_df["ms"] <= ms_max)
    ].copy()

    T_vals = sorted(ok["T_K"].unique())
    ms_vals = sorted(ok["ms"].unique())

    cmap = plt.cm.viridis
    ms_norm = mcolors.Normalize(vmin=min(ms_vals), vmax=max(ms_vals))

    n_T  = len(T_vals)
    ncols = min(4, n_T)
    nrows = int(np.ceil(n_T / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 3.5 * nrows),
                             squeeze=False)

    qty_labels = {
        "mc":           "CO₂ molality [mol/kg]",
        "xc_W_SALTincl": "x_CO₂ (salt-incl.)",
        "xc_W_SALTfree": "x_CO₂ (salt-free)",
        "xc_C":         "x_CO₂ in CO₂-rich phase",
    }

    for idx, T in enumerate(T_vals):
        ax = axes[idx // ncols][idx % ncols]
        sub = ok[ok["T_K"] == T]
        # One series per (ms, qty) combo
        for ms_i in sorted(sub["ms"].unique()):
            for qty in sorted(sub["qty"].unique()):
                s = sub[(sub["ms"] == ms_i) & (sub["qty"] == qty)].sort_values("P_bar")
                if s.empty:
                    continue
                c = cmap(ms_norm(ms_i))
                ax.scatter(s["P_bar"], s["value_exp"], marker="o",
                           facecolors="none", edgecolors=c, s=40, zorder=3)
                ax.scatter(s["P_bar"], s["value_pred"], marker=".", color=c,
                           s=40, zorder=3)
                ax.plot(s["P_bar"], s["value_pred"], color=c, lw=0.8, alpha=0.6)

        ax.set_xlabel("P [bar]", fontsize=9)
        ax.set_ylabel(qty_labels.get(sub["qty"].iloc[0], "value"), fontsize=9)
        ax.set_title(f"T = {T} K", fontsize=10)
        ax.set_xscale("log")

    # Hide unused axes
    for idx in range(len(T_vals), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    # Colorbar for ms
    sm = ScalarMappable(cmap=cmap, norm=ms_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("ms [mol NaCl/kg H₂O]", fontsize=9)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="k", lw=0, markerfacecolor="none",
               markersize=7, label="Experiment"),
        Line2D([0], [0], marker=".", color="k", lw=0.8,
               markersize=7, label="eCPA model"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=2, fontsize=9, frameon=True,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("eCPA vs Experimental CO₂ Solubility in NaCl Brine", fontsize=12)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_validation_parity(results_df, save_path=None):
    """
    Parity plots (predicted vs experimental) per quantity type.
    ±10% and ±20% error bands; color by temperature.
    """
    ok  = results_df[results_df["status"] == "ok"]
    bad = results_df[results_df["status"] != "ok"]

    qtys = sorted(ok["qty"].unique())
    fig, axes = plt.subplots(1, len(qtys),
                             figsize=(5 * len(qtys), 5),
                             squeeze=False)

    qty_labels = {
        "mc":            "CO₂ molality [mol/kg]",
        "xc_W_SALTincl": "x_CO₂ (salt-incl.)",
        "xc_W_SALTfree": "x_CO₂ (salt-free)",
        "xc_C":          "x_CO₂ in CO₂-rich phase",
    }

    T_all = sorted(ok["T_K"].unique())
    T_norm = mcolors.Normalize(vmin=min(T_all), vmax=max(T_all))
    cmap = plt.cm.plasma

    for j, qty in enumerate(qtys):
        ax = axes[0][j]
        sub = ok[ok["qty"] == qty]
        aare = sub["abs_rel_err"].mean() * 100
        N    = len(sub)

        vmin = sub[["value_exp", "value_pred"]].min().min()
        vmax = sub[["value_exp", "value_pred"]].max().max()
        pad  = (vmax - vmin) * 0.05
        lims = (max(0, vmin - pad), vmax + pad)

        lv = np.linspace(lims[0], lims[1], 200)
        ax.plot(lv, lv, "k-", lw=1.2, label="1:1")
        ax.fill_between(lv, lv * 0.8, lv * 1.2,
                        color="orange", alpha=0.15, label="±20%")
        ax.fill_between(lv, lv * 0.9, lv * 1.1,
                        color="green", alpha=0.20, label="±10%")

        sc = ax.scatter(sub["value_exp"], sub["value_pred"],
                        c=sub["T_K"], cmap=cmap, norm=T_norm,
                        s=18, zorder=4, linewidths=0.2, edgecolors="k")

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f"Experimental {qty_labels.get(qty, qty)}", fontsize=9)
        ax.set_ylabel(f"Predicted", fontsize=9)
        ax.set_title(f"{qty}  (N={N}, AARE={aare:.1f}%)", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")

        cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
        cbar.set_label("T [K]", fontsize=8)

    fig.suptitle("eCPA Parity: Predicted vs Experimental CO₂–NaCl", fontsize=11)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ── Per-ms visual style (consistent across all T figures) ──────────────────────
# Up to 10 distinct ms levels in data; styles assigned in sorted order globally.
_MS_COLORS     = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
_MS_MARKERS    = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'X', 'P']
_MS_LINESTYLES = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']


def _build_ms_styles(ms_vals_sorted):
    """Return dict  ms_val → (color, marker, linestyle)  in sorted order."""
    styles = {}
    for i, ms_i in enumerate(ms_vals_sorted):
        styles[ms_i] = (
            _MS_COLORS[i % len(_MS_COLORS)],
            _MS_MARKERS[i % len(_MS_MARKERS)],
            _MS_LINESTYLES[i % len(_MS_LINESTYLES)],
        )
    return styles


def plot_nacl_T_figures(results_df, fig_dir='figures/co2nacl',
                        T_max=523.0, ms_max=7.0, smooth_data=None):
    """
    Generate one figure per temperature for the CO2-NaCl validation.

    Each figure has up to two panels:
      Left : CO₂ aqueous molality  mc [mol/kg H₂O] vs P  (log-log)
      Right: CO₂ mole fraction in CO₂-rich phase  xc_C vs P  (log-linear, if data exist)

    xc_W_SALTfree data are converted to mc via  mc = v / ((1-v) * Mw).
    xc_W_SALTincl rows are excluded (conversion requires knowing x_NaCl).
    NaCl molality encoded by color + marker + linestyle; two-column legend
    shows experiment symbol and eCPA line side by side for each ms level.
    """
    import os
    from pathlib import Path
    from matplotlib.lines import Line2D

    os.makedirs(fig_dir, exist_ok=True)

    ok = results_df[
        (results_df['status'] == 'ok') &
        (results_df['T_K'] <= T_max) &
        (results_df['ms'] <= ms_max)
    ].copy()

    # Convert xc_W_SALTfree → mc  (mc = v / ((1-v)*Mw))
    mask_sf = ok['qty'] == 'xc_W_SALTfree'
    for col in ('value_exp', 'value_pred'):
        v = ok.loc[mask_sf, col].clip(upper=1 - 1e-9)
        ok.loc[mask_sf, col] = v / ((1.0 - v) * Mw)
    ok.loc[mask_sf, 'qty'] = 'mc'

    # Drop xc_W_SALTincl — conversion to mc is ambiguous without x_NaCl
    ok = ok[ok['qty'] != 'xc_W_SALTincl'].copy()

    # Build global style map so same ms always gets same color/marker/linestyle
    ms_styles = _build_ms_styles(sorted(ok['ms'].unique()))

    T_vals = sorted(ok['T_K'].unique())
    saved = []

    for T in T_vals:
        sub_T = ok[ok['T_K'] == T]
        has_mc  = (sub_T['qty'] == 'mc').any()
        has_xcc = (sub_T['qty'] == 'xc_C').any()
        n_panels = int(has_mc) + int(has_xcc)
        if n_panels == 0:
            continue

        fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.5),
                                 squeeze=False)
        panel = 0
        legend_ms_vals = []  # ms values present in this figure (for legend)

        # ── Pre-compute smooth ribbon curves (if smooth_data provided) ─────────
        _ribbon_curves = {}   # (qty, ms_val) -> y array on P_FINE
        if smooth_data is not None:
            import matplotlib.colors as mcolors
            _P_FINE  = np.logspace(0.0, np.log10(1500.0), 3000)
            _logPf   = np.log10(_P_FINE)
            _CMAP    = plt.cm.rainbow
            _CNORM   = mcolors.Normalize(vmin=0.0, vmax=6.0)
            _SLOPE_MAX = 4.0

            def _get_ribbon(ms_v, qty_r):
                df_sm = smooth_data[ms_v]
                sub_sm = (df_sm[np.abs(df_sm['T_K'] - T) < 0.5
                               ].query('ecpa_converged')
                           .sort_values('P_bar'))
                if len(sub_sm) < 2:
                    return np.full(len(_P_FINE), np.nan)
                P_sm = sub_sm['P_bar'].values
                if qty_r == 'mc':
                    xc = sub_sm['ecpa_xc_W'].values
                    ok = np.isfinite(xc) & (xc > 0) & (xc < 1)
                    if ok.sum() < 2:
                        return np.full(len(_P_FINE), np.nan)
                    y_sm  = xc[ok] / ((1.0 - xc[ok]) * Mw)
                    lP    = np.log10(P_sm[ok])
                    lY    = np.log10(y_sm)
                    # trim steep near-boundary onset
                    start = 0
                    if len(lP) > 2:
                        sl = np.abs(np.diff(lY) /
                                    np.clip(np.diff(lP), 1e-9, None))
                        while start < len(sl) and sl[start] > _SLOPE_MAX:
                            start += 1
                    lP = lP[start:]; lY = lY[start:]
                    if len(lP) < 2:
                        return np.full(len(_P_FINE), np.nan)
                    w = (_logPf >= lP.min()) & (_logPf <= lP.max())
                    out = np.full(len(_P_FINE), np.nan)
                    if w.sum() >= 2:
                        out[w] = 10.0 ** np.interp(_logPf[w], lP, lY)
                    return out
                elif qty_r == 'xc_C':
                    yw  = sub_sm['ecpa_yw_C'].values
                    y_sm = 1.0 - yw                          # y_CO2 = 1 - y_H2O
                    ok = np.isfinite(y_sm) & (y_sm > 0)
                    if ok.sum() < 2:
                        return np.full(len(_P_FINE), np.nan)
                    lP = np.log10(P_sm[ok]); Y = y_sm[ok]
                    w  = (_logPf >= lP.min()) & (_logPf <= lP.max())
                    out = np.full(len(_P_FINE), np.nan)
                    if w.sum() >= 2:
                        out[w] = np.interp(_logPf[w], lP, Y)  # linear y
                    return out
                return np.full(len(_P_FINE), np.nan)

            _sorted_sm = sorted(smooth_data.keys())
            for _qty_r in ('mc', 'xc_C'):
                for ms_v in _sorted_sm:
                    _ribbon_curves[(_qty_r, ms_v)] = _get_ribbon(ms_v, _qty_r)

        for qty, ylabel, yscale in [
            ('mc',   r'$m_{\mathrm{CO_2}}$ [mol kg$^{-1}$]',               'log'),
            ('xc_C', r'$y_{\mathrm{CO_2}}$ (CO$_2$-rich phase)',            'linear'),
        ]:
            if not (sub_T['qty'] == qty).any():
                continue
            ax = axes[0][panel]

            # ── Rainbow ribbon background ─────────────────────────────────────
            if smooth_data is not None and _ribbon_curves:
                _sorted_sm = sorted(smooth_data.keys())
                for _i in range(len(_sorted_sm) - 1):
                    ms_lo, ms_hi = _sorted_sm[_i], _sorted_sm[_i + 1]
                    y_lo = _ribbon_curves[(qty, ms_lo)]
                    y_hi = _ribbon_curves[(qty, ms_hi)]
                    _valid = np.isfinite(y_lo) & np.isfinite(y_hi)
                    if _valid.sum() < 2:
                        continue
                    _y1 = np.where(_valid, np.maximum(y_lo, y_hi), np.nan)
                    _y2 = np.where(_valid, np.minimum(y_lo, y_hi), np.nan)
                    ax.fill_between(_P_FINE, _y1, _y2, where=_valid,
                                    color=_CMAP(_CNORM(0.5*(ms_lo + ms_hi))),
                                    alpha=0.35, linewidth=0, zorder=1)
                for ms_v in _sorted_sm:
                    y = _ribbon_curves[(qty, ms_v)]
                    _v = np.isfinite(y)
                    if _v.sum() > 1:
                        ax.plot(_P_FINE[_v], y[_v], '-',
                                color=_CMAP(_CNORM(ms_v if ms_v > 0.01 else 0.0)),
                                lw=0.6, alpha=0.55, zorder=1)

            sub_q = sub_T[sub_T['qty'] == qty].sort_values(['ms', 'P_bar'])

            for ms_i in sorted(sub_q['ms'].unique()):
                c, mk, ls = ms_styles[ms_i]
                s = sub_q[sub_q['ms'] == ms_i]
                ax.scatter(s['P_bar'], s['value_exp'], marker=mk,
                           facecolors='none', edgecolors=c, s=40,
                           linewidths=1.0, zorder=4)
                ax.plot(s['P_bar'].values, s['value_pred'].values,
                        ls, color=c, lw=1.5, zorder=3)
                if ms_i not in legend_ms_vals:
                    legend_ms_vals.append(ms_i)

            ax.set_xscale('log')
            ax.set_yscale(yscale)
            ax.set_xlabel('P [bar]', fontsize=12, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
            ax.set_title(f'T = {int(T)} K', fontsize=12, fontweight='bold')
            ax.tick_params(labelsize=10)
            panel += 1

        # Two-column legend: left col = experiment markers, right col = eCPA lines.
        # matplotlib fills columns in column-major order, so we pass all exp
        # handles first, then all mod handles.  Row i then shows (exp_i, mod_i).
        exp_handles = []
        mod_handles = []
        for ms_i in sorted(legend_ms_vals):
            c, mk, ls = ms_styles[ms_i]
            exp_handles.append(
                Line2D([0], [0], marker=mk, color=c, lw=0,
                       mfc='none', mec=c, ms=6, label=f'ms = {ms_i:.2g}')
            )
            mod_handles.append(
                Line2D([0], [0], color=c, lw=1.5, ls=ls, label='')
            )
        # column-major: [exp_0..exp_n, mod_0..mod_n] → row i = (exp_i | mod_i)
        all_handles = exp_handles + mod_handles

        # Legend inside the rightmost panel
        leg = axes[0][-1].legend(
            handles=all_handles,
            ncol=2, fontsize=9,
            loc='best',
            framealpha=0.9,
            handlelength=2.2,
            title='Exp.   eCPA',
            title_fontsize=9,
        )

        fig.tight_layout()
        fpath = Path(fig_dir) / f'T{int(T)}K.png'
        fig.savefig(fpath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved.append(str(fpath))

    print(f"Saved {len(saved)} per-T figures to {fig_dir}/")
    return saved


def plot_error_heatmap(results_df, save_path=None):
    """AARE heatmap over individual (T, ms) cells."""
    ok = results_df[results_df["status"] == "ok"].copy()

    # Use individual ms values (round to 2 dp to merge near-identical labels)
    ok["ms_r"] = ok["ms"].round(2)
    ms_vals = sorted(ok["ms_r"].unique())
    T_vals  = sorted(ok["T_K"].unique())

    aare_grid = np.full((len(ms_vals), len(T_vals)), np.nan)
    n_grid    = np.zeros_like(aare_grid, dtype=int)

    for i, ms_v in enumerate(ms_vals):
        for j, T in enumerate(T_vals):
            sub = ok[(ok["ms_r"] == ms_v) & (ok["T_K"] == T)]
            if len(sub) > 0:
                aare_grid[i, j] = sub["abs_rel_err"].mean() * 100
                n_grid[i, j]    = len(sub)

    fig_w = max(8, len(T_vals) * 0.65)
    fig_h = max(4, len(ms_vals) * 0.50)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(aare_grid, aspect="auto", origin="lower",
                   cmap="RdYlGn_r",
                   norm=mcolors.TwoSlopeNorm(vcenter=5, vmin=0, vmax=30))

    ax.set_xticks(range(len(T_vals)))
    ax.set_xticklabels([str(int(T)) for T in T_vals], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(ms_vals)))
    ax.set_yticklabels([f"{v:.2g}" for v in ms_vals], fontsize=8)
    ax.set_xlabel("T [K]", fontsize=11, fontweight='bold')
    ax.set_ylabel(r"$m_s$ [mol kg$^{-1}$]", fontsize=11, fontweight='bold')

    # Annotate non-empty cells with AARE and N
    for i in range(len(ms_vals)):
        for j in range(len(T_vals)):
            if not np.isnan(aare_grid[i, j]):
                txt = f"{aare_grid[i,j]:.1f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=6.5)

    fig.colorbar(im, ax=ax, label="AARE [%]", shrink=0.8)
    fig.tight_layout()
    _save(fig, save_path)
    return fig
