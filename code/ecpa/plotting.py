"""
Plotting functions for eCPA results.

All functions accept a `save_path` keyword argument (default None).
Pass a file path to save the figure; omit it (or pass None) to skip saving.
"""
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.interpolate import NearestNDInterpolator

from .exp_data import exp_at_T


# ── Helpers ────────────────────────────────────────────────────────────────────

def _save(fig, save_path):
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")


def _ref_colors(exp_df):
    all_refs = sorted(exp_df["reference"].unique())
    return {r: plt.cm.tab20(i / max(len(all_refs) - 1, 1))
            for i, r in enumerate(all_refs)}


# ── Experimental comparison plots ──────────────────────────────────────────────

def _cpa2_vle_worker(args):
    """Top-level worker for plot_cpa2_vs_exp (parallel CPA flashes)."""
    import CPA, numpy as np, warnings
    T_K, P_bar = args[0], args[1]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = CPA.flash_co2_h2o_tpz(T=float(T_K), P_bar=float(P_bar), z_co2=0.5)
        if out["phase"] != "two_phase":
            return T_K, P_bar, np.nan, np.nan
        return T_K, P_bar, float(out["x"][0]), float(out["y"][1])
    except Exception:
        return T_K, P_bar, np.nan, np.nan


def plot_cpa2_vs_exp(T_list, exp_df, N_P=40, n_workers=48,
                     save_path=None):
    """CPA flash vs experimental VLE: grid of T rows × 2 columns."""
    tasks   = []
    P_grids = {}
    for T_K in T_list:
        df_T    = exp_at_T(T_K, exp_df)
        P_lo    = df_T["P_bar"].min()
        P_hi    = df_T["P_bar"].max()
        P_grids[T_K] = np.linspace(P_lo, P_hi, N_P)
        for P in P_grids[T_K]:
            tasks.append((T_K, P))

    raw = {}
    with ProcessPoolExecutor(max_workers=min(n_workers, len(tasks))) as ex:
        futs = {ex.submit(_cpa2_vle_worker, t): t for t in tasks}
        for fut in as_completed(futs):
            T_K, P, xc_W, yw_C = fut.result()
            raw[(T_K, round(float(P), 6))] = (xc_W, yw_C)

    cpa2 = {}
    for T_K in T_list:
        P_grid = P_grids[T_K]
        vals   = [raw.get((T_K, round(float(P), 6)), (np.nan, np.nan)) for P in P_grid]
        cpa2[T_K] = dict(P=P_grid,
                         xc_W=np.array([v[0] for v in vals]),
                         yw_C=np.array([v[1] for v in vals]))

    ref_color = _ref_colors(exp_df)
    fig, axes = plt.subplots(len(T_list), 2, figsize=(11, 2.8*len(T_list)),
                              squeeze=False)
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.04,
                        top=0.96, hspace=0.55, wspace=0.30)

    for row, T_K in enumerate(T_list):
        ax_x, ax_y = axes[row, 0], axes[row, 1]
        d = cpa2[T_K]
        mask = np.isfinite(d["xc_W"])
        if mask.any():
            ax_x.plot(d["P"][mask], d["xc_W"][mask], "k-", lw=1.4, zorder=3)
        mask = np.isfinite(d["yw_C"])
        if mask.any():
            ax_y.plot(d["P"][mask], d["yw_C"][mask], "k-", lw=1.4, zorder=3)

        df_T = exp_at_T(T_K, exp_df)
        seen_refs = set()
        for _, grp in df_T.groupby("reference"):
            ref   = grp["reference"].iloc[0]
            color = ref_color[ref]
            label = ref if ref not in seen_refs else "_nolegend_"
            seen_refs.add(ref)
            sub_x = grp[grp["xc_W"].notna()]
            if not sub_x.empty:
                ax_x.scatter(sub_x["P_bar"], sub_x["xc_W"], s=28, color=color,
                             zorder=4, label=label, linewidths=0)
            sub_y = grp[grp["yw_C"].notna()]
            if not sub_y.empty:
                ax_y.scatter(sub_y["P_bar"], sub_y["yw_C"], s=28, color=color,
                             zorder=4, label=label, linewidths=0)

        for ax in (ax_x, ax_y):
            ax.set_xlabel("$P$ [bar]", fontsize=9)
            ax.set_xlim(left=0)
            ax.grid(True, ls=":", lw=0.4, alpha=0.5)
            ax.tick_params(labelsize=8)
        ax_x.set_ylabel("$x_{c,W}$  (CO$_2$ in aq.)", fontsize=9)
        ax_y.set_ylabel("$y_{w,C}$  (H$_2$O in CO$_2$)", fontsize=9)
        ax_y.set_yscale("log")
        fig.text(0.5, axes[row, 0].get_position().y1 + 0.005,
                 f"$T = {T_K}$ K", ha="center", va="bottom",
                 fontsize=10, fontweight="bold", transform=fig.transFigure)

    _save(fig, save_path)
    plt.show()
    return fig


def plot_combined_vs_exp(T_list, exp_df, CPA_GROUPS, CPA_TEMPS,
                          N_P=60, n_workers=48,
                          P_MAX=1000.0,
                          save_path=None):
    """eCPA table + CPA flash vs experiment: grid of T rows × 2 columns."""
    tasks, P_grids, exp_data_by_T = [], {}, {}
    for T_K in T_list:
        df_T = exp_at_T(T_K, exp_df)
        df_T = df_T[df_T["P_bar"] <= P_MAX].copy()
        exp_data_by_T[T_K] = df_T
        if df_T.empty:
            P_grids[T_K] = np.array([], dtype=float)
            continue
        P_lo = float(df_T["P_bar"].min())
        P_hi = float(df_T["P_bar"].max())
        if P_lo <= P_hi:
            P_grids[T_K] = np.linspace(P_lo, P_hi, N_P)
            for P in P_grids[T_K]:
                tasks.append((T_K, P))
        else:
            P_grids[T_K] = np.array([], dtype=float)

    raw = {}
    if tasks:
        with ProcessPoolExecutor(max_workers=min(n_workers, len(tasks))) as ex:
            futs = {ex.submit(_cpa2_vle_worker, t): t for t in tasks}
            for fut in as_completed(futs):
                T_K, P, xc_W, yw_C = fut.result()
                raw[(T_K, round(float(P), 6))] = (xc_W, yw_C)

    cpa2 = {}
    for T_K in T_list:
        P_grid = P_grids[T_K]
        vals   = [raw.get((T_K, round(float(P), 6)), (np.nan, np.nan)) for P in P_grid]
        cpa2[T_K] = dict(P=P_grid,
                         xc_W=np.array([v[0] for v in vals]),
                         yw_C=np.array([v[1] for v in vals]))

    ref_color = _ref_colors(exp_df)
    fig, axes = plt.subplots(len(T_list), 2, figsize=(11, 2.8*len(T_list)),
                              squeeze=False)
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.06,
                        top=0.96, hspace=0.55, wspace=0.30)

    temps_arr = np.array(sorted(CPA_GROUPS.keys()), dtype=float)

    for row, T_K in enumerate(T_list):
        ax_x, ax_y = axes[row, 0], axes[row, 1]
        df_T = exp_data_by_T[T_K]

        # eCPA table line
        T_near = int(temps_arr[np.argmin(np.abs(temps_arr - T_K))])
        if T_near in CPA_GROUPS and not df_T.empty:
            tab    = CPA_GROUPS[T_near].sort_values("P_bar")
            P_tab  = tab["P_bar"].to_numpy(dtype=float)
            xc_tab = 1.0 - tab["xw_W"].to_numpy(dtype=float)
            yw_tab = tab["xw_C"].to_numpy(dtype=float)
            P_lo   = float(df_T["P_bar"].min())
            P_hi   = float(df_T["P_bar"].max())
            P_start = max(P_lo, float(P_tab.min()))
            P_end   = min(P_hi, float(P_tab.max()), P_MAX)
            if P_start <= P_end:
                P_i = np.linspace(P_start, P_end, N_P)
                ax_x.plot(P_i, np.interp(P_i, P_tab, xc_tab),
                          "k--", lw=1.6, zorder=3, label="eCPA table")
                ax_y.plot(P_i, np.interp(P_i, P_tab, yw_tab),
                          "k--", lw=1.6, zorder=3, label="eCPA table")

        # CPA line
        d = cpa2[T_K]
        mask_x = np.isfinite(d["xc_W"])
        if mask_x.any():
            ax_x.plot(d["P"][mask_x], d["xc_W"][mask_x],
                      "k-.", lw=1.4, zorder=4, label="CPA flash")
        mask_y = np.isfinite(d["yw_C"])
        if mask_y.any():
            ax_y.plot(d["P"][mask_y], d["yw_C"][mask_y],
                      "k-.", lw=1.4, zorder=4, label="CPA flash")

        # Experimental
        seen_refs = set()
        for _, grp in df_T.groupby("reference"):
            ref   = grp["reference"].iloc[0]
            color = ref_color[ref]
            label = ref if ref not in seen_refs else "_nolegend_"
            seen_refs.add(ref)
            sub_x = grp[grp["xc_W"].notna()]
            if not sub_x.empty:
                ax_x.scatter(sub_x["P_bar"], sub_x["xc_W"], s=28, color=color,
                             zorder=5, label=label, linewidths=0)
            sub_y = grp[grp["yw_C"].notna()]
            if not sub_y.empty:
                ax_y.scatter(sub_y["P_bar"], sub_y["yw_C"], s=28, color=color,
                             zorder=5, label=label, linewidths=0)

        for ax in (ax_x, ax_y):
            ax.set_xlabel("$P$ [bar]", fontsize=9)
            ax.set_xlim(left=0)
            ax.grid(True, ls=":", lw=0.4, alpha=0.5)
            ax.tick_params(labelsize=8)
            _, xmax = ax.get_xlim()
            ax.set_xlim(0, min(xmax, P_MAX))
        ax_x.set_ylabel("$x_{c,W}$  (CO$_2$ in aq.)", fontsize=9)
        ax_y.set_ylabel("$y_{w,C}$  (H$_2$O in CO$_2$)", fontsize=9)
        ax_y.set_yscale("log")
        fig.text(0.5, axes[row, 0].get_position().y1 + 0.005,
                 f"$T = {T_K}$ K", ha="center", va="bottom",
                 fontsize=10, fontweight="bold", transform=fig.transFigure)

    legend_handles = [
        Line2D([0], [0], color="k", lw=1.6, ls="--", label="eCPA table (salt-free)"),
        Line2D([0], [0], color="k", lw=1.4, ls="-.", label="CPA flash (SSI)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, 0.0), framealpha=0.9)
    _save(fig, save_path)
    plt.show()
    return fig


# ── Convergence map ────────────────────────────────────────────────────────────

def plot_convergence_map(df_results, z_co2_values, ms_values,
                          save_path=None):
    """
    Heatmap of flash convergence/failure types across the (T, P, z, ms) grid.
    One panel per (z_co2, ms) pair.
    """
    # Colorblind-safe palette (distinguishable for deuteranopia/protanopia):
    #   blue, gray, orange, purple, teal, near-black
    etype_code = {
        "none":               0,
        "out_of_range":       1,
        "salting_out":        2,
        "no_sign_change":     3,   # flash failed, CPA uncertain
        "single_phase_stable":3,   # flash failed, ecpa_stability confirmed
        "single_phase_gas":   3,   # flash failed, CPA confirmed gas
        "single_phase_liquid":3,   # flash failed, CPA confirmed liquid
        "ssi_no_converge":    3,
        "elv_solver":         4,
        "cache_empty":        4,
        "runtime_other":      4,
        "exception":          4,
    }
    shade_colors = ["#4878cf",   # blue        – converged
                    "#d3d3d3",   # light gray  – out of range
                    "#ff8c00",   # dark orange – salting-out
                    "#9467bd",   # purple      – single-phase (confirmed or likely)
                    "#2d2d2d"]   # near-black  – solver failed
    shade_labels = ["converged",
                    "single-phase (out of range)",
                    "salting-out (x₄w<0)",
                    "single-phase (confirmed or likely)",
                    "solver failed"]
    marker_map   = {
        "none":               ("#4878cf", "o",  15),
        "out_of_range":       ("#aaaaaa", "s",  12),
        "salting_out":        ("#ff8c00", "^",  15),
        "no_sign_change":     ("#9467bd", "D",  18),  # purple diamond  – uncertain
        "single_phase_stable":("#17becf", "v",  18),  # teal ▼ – stability confirmed
        "single_phase_gas":   ("#17becf", "<",  18),  # teal ◄ – CPA confirmed gas
        "single_phase_liquid":("#17becf", ">",  18),  # teal ► – CPA confirmed liquid
        "ssi_no_converge":    ("#9467bd", "D",  18),
        "elv_solver":         ("#2d2d2d", "x",  30),
        "cache_empty":        ("#2d2d2d", "x",  30),
        "runtime_other":      ("#2d2d2d", "P",  30),
        "exception":          ("#2d2d2d", "*",  30),
    }

    cmap   = mpl.colors.ListedColormap(shade_colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm   = mpl.colors.BoundaryNorm(bounds, cmap.N)

    P_grid = np.logspace(0, np.log10(1500), 300)
    T_grid = np.linspace(283, 533, 300)
    PP, TT = np.meshgrid(P_grid, T_grid)

    n_rows = len(z_co2_values)
    n_cols = len(ms_values)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 3.5*n_rows),
                              sharey=True, sharex=True)

    for i, z_co2_i in enumerate(z_co2_values):
        for j, ms_i in enumerate(ms_values):
            ax  = axes[i, j]
            sub = df_results[
                np.isclose(df_results["z_co2"], round(z_co2_i, 2)) &
                (df_results["ms"] == ms_i)
            ].copy()
            sub["code"] = sub["error_type"].map(etype_code).fillna(4).astype(int)

            interp = NearestNDInterpolator(
                np.c_[np.log10(sub["P"].values), sub["T"].values],
                sub["code"].values,
            )
            ZZ = interp(np.c_[np.log10(PP.ravel()), TT.ravel()]).reshape(PP.shape)
            ax.contourf(PP, TT, ZZ, levels=bounds, cmap=cmap, norm=norm, alpha=0.35)

            for etype, (color, marker, size) in marker_map.items():
                grp = sub[sub["error_type"] == etype]
                if len(grp):
                    ax.scatter(grp["P"], grp["T"], c=color, marker=marker,
                               s=size, zorder=3, alpha=0.9)

            ax.set_xscale("log")
            ax.grid(True, which="both", alpha=0.2)
            if j == 0:
                ax.set_ylabel(f"z_CO₂ = {z_co2_i:.1f}\nT (K)", fontsize=10)
            if i == 0:
                ax.set_title(f"ms = {ms_i} mol/kg", fontsize=11)
            if i == n_rows - 1:
                ax.set_xlabel("P (bar)", fontsize=10)

    # Background-shade patches + distinctive markers for boundary types
    legend_handles = [
        mpatches.Patch(color=shade_colors[k], alpha=0.5, label=shade_labels[k])
        for k in range(len(shade_colors))
    ]
    # Add explicit marker entries for types that share a shade code
    legend_handles += [
        Line2D([0],[0], marker="D", color="w", markerfacecolor="#9467bd",
               markersize=8, label="no_sign_change — flash uncertain (◆)"),
        Line2D([0],[0], marker="v", color="w", markerfacecolor="#17becf",
               markersize=8, label="single_phase_stable — ecpa_stability (▼)"),
        Line2D([0],[0], marker="<", color="w", markerfacecolor="#17becf",
               markersize=8, label="single_phase_gas — CPA confirmed (◄)"),
        Line2D([0],[0], marker=">", color="w", markerfacecolor="#17becf",
               markersize=8, label="single_phase_liquid — CPA confirmed (►)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.08), frameon=True)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()
    return fig


# ── Phase envelope ─────────────────────────────────────────────────────────────

def plot_phase_envelope(envelopes_ecpa, z_co2_values, ms_values,
                         envelopes_cpa2=None,
                         ms_colors=("steelblue", "darkorange", "firebrick"),
                         save_path=None):
    """
    Multi-panel phase envelope: T vs P, rows = z_co2, cols = ms.
    eCPA solid/dashed; CPA salt-free dotted (if provided).
    Single-phase region shaded blue, two-phase region shaded orange.
    """
    # Fill limits well beyond the plotted range — clipped by axes limits
    P_FILL_MIN = 0.1
    P_FILL_MAX = 5000.0

    n_rows = len(z_co2_values)
    n_cols = len(ms_values)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 3.5*n_rows),
                              sharey=True, sharex=True)

    for i, z_co2_i in enumerate(z_co2_values):
        z_key   = round(z_co2_i, 2)
        df_cpa2 = (envelopes_cpa2.get(z_key, None)
                   if envelopes_cpa2 else None)

        for j, ms_i in enumerate(ms_values):
            ax    = axes[i, j]
            color = ms_colors[j % len(ms_colors)]

            df_env = envelopes_ecpa[z_key][ms_i].dropna(subset=["P_lo"])
            if len(df_env) > 0:
                T_plot = df_env["T"].values
                P_lo   = df_env["P_lo"].values

                # Region shading (drawn first, behind curves)
                ax.fill_between(T_plot, P_FILL_MIN, P_lo,
                                color="#cce5f5", alpha=0.55, zorder=0,
                                label="single-phase")
                ax.fill_between(T_plot, P_lo, P_FILL_MAX,
                                color="#fde5c8", alpha=0.55, zorder=0,
                                label="two-phase")

                ax.plot(T_plot, P_lo, color=color, lw=2, label="bubble pt (eCPA)")

                # Dew-point branch: plot if data exist, but suppress legend entry
                hi_mask = np.isfinite(df_env["P_hi"].values)
                if hi_mask.any():
                    ax.plot(T_plot[hi_mask], df_env["P_hi"].values[hi_mask],
                            color=color, lw=2, ls="--", label="_nolegend_")
                    T_fill = np.concatenate([T_plot[hi_mask], T_plot[hi_mask][::-1]])
                    P_fill = np.concatenate([df_env["P_hi"].values[hi_mask],
                                             P_lo[hi_mask][::-1]])
                    ax.fill(T_fill, P_fill, color=color, alpha=0.12)

            if df_cpa2 is not None and len(df_cpa2) > 0:
                df_c2_lo = df_cpa2.dropna(subset=["P_lo"])
                if len(df_c2_lo) > 0:
                    ax.plot(df_c2_lo["T"], df_c2_lo["P_lo"], color="black",
                            lw=1.5, ls=":", label="bubble pt (CPA)")
                df_c2_hi = df_cpa2.dropna(subset=["P_hi"])
                if len(df_c2_hi) > 0:
                    ax.plot(df_c2_hi["T"], df_c2_hi["P_hi"], color="black",
                            lw=1.5, ls="-.", label="_nolegend_")

            ax.set_yscale("log")
            ax.set_xlim(283, 533)
            ax.grid(True, which="both", alpha=0.2)
            if j == 0:
                ax.set_ylabel(f"z_CO₂={z_co2_i:.1f}\nP (bar)", fontsize=10)
            if i == 0:
                ax.set_title(f"ms = {ms_i} mol/kg", fontsize=11)
            if i == n_rows - 1:
                ax.set_xlabel("T (K)", fontsize=10)

    # Deduplicate legend entries; skip internal matplotlib labels (prefixed "_")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    seen = {}
    for lbl, hdl in zip(labels, handles):
        if not lbl.startswith("_") and lbl not in seen:
            seen[lbl] = hdl
    fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=4,
               fontsize=10, bbox_to_anchor=(0.5, -0.04), frameon=True)

    plt.tight_layout()
    _save(fig, save_path)
    plt.show()
    return fig


# ── Compositions vs P ──────────────────────────────────────────────────────────

def plot_guess_table(CPA_GROUPS, CPA_TEMPS, save_path=None):
    """x_w,W and x_w,C vs P coloured by T — quick sanity check of the table."""
    PANELS = [
        ("xw_W", r"$x_{w,W}$  (H$_2$O mol-frac, aqueous)",     False),
        ("xw_C", r"$x_{w,C}$  (H$_2$O mol-frac, CO$_2$-rich)", True),
    ]
    T_all  = np.array(sorted(CPA_GROUPS.keys()), dtype=float)[::25]
    T_min, T_max = T_all.min(), T_all.max()
    cmap   = cm.plasma
    norm   = mcolors.Normalize(vmin=T_min, vmax=T_max)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.subplots_adjust(left=0.07, right=0.90, bottom=0.12, top=0.88, wspace=0.28)

    for ax, (col, ylabel, use_log) in zip(axes, PANELS):
        for T_K in T_all:
            dfT   = CPA_GROUPS[int(T_K)]
            P_arr = dfT["P_bar"].to_numpy(dtype=float)
            y_arr = dfT[col].to_numpy(dtype=float)
            color = cmap(norm(T_K))
            ax.plot(P_arr, y_arr, lw=0.6, color=color, alpha=0.55, zorder=1)
            ax.scatter(P_arr, y_arr, s=4, color=color, alpha=0.75,
                       linewidths=0, zorder=2)
        ax.set_xlabel(r"$P$  [bar]", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.tick_params(labelsize=9)
        ax.grid(True, ls=":", lw=0.4, alpha=0.5)
        if use_log:
            ax.set_yscale("log")

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.12, 0.018, 0.76])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(r"$T$  [K]", fontsize=11, labelpad=8)
    cbar.ax.tick_params(labelsize=9)
    _save(fig, save_path)
    plt.show()
    return fig
