"""
_plot_scan_results.py — Generate publication figures from scan_results.npz.

Figures:
1. SSI iteration count heatmaps (standard Wilson vs accelerated Wilson vs accelerated+stabK)
2. Stability test initial guess requirement map
3. Phase diagram (single-phase vs two-phase) in T-P space
4. Speedup ratio (standard/accelerated) heatmap
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, BoundaryNorm
from matplotlib.ticker import MaxNLocator
import os

plt.rcParams.update({
    'font.weight':       'bold',
    'axes.labelweight':  'bold',
    'axes.titleweight':  'bold',
    'axes.labelsize':    11,
    'axes.titlesize':    11,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
})

def bold_ticks(ax):
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight('bold')

# ── Load data ──────────────────────────────────────────────────────────────────
d = np.load("results/scan_results_extended.npz", allow_pickle=True)

T_grid = d["T_grid"]
P_grid = d["P_grid"]
z_grid = d["z_grid"]
phase_map = list(d["phase_map"])
phase_id = d["phase_id"]              # [iT, iP, iz]
flash_conv = d["flash_conv"]          # [iT, iP, iz, 4]
flash_iter = d["flash_iter"]          # [iT, iP, iz, 4]
stab_stable = d["stab_stable"]        # [iT, iP, iz]
stab_best_trial = d["stab_best_trial"]  # [iT, iP, iz]
stab_n_unstable = d["stab_n_unstable"]  # [iT, iP, iz]
stab_trial_labels = list(d["stab_trial_labels"])
flash_strategy_names = list(d["flash_strategy_names"])
robust_attempt = d["robust_attempt"]
wall_time = d["wall_time"]

tp_idx = phase_map.index("two_phase")
sp_idx = phase_map.index("single_phase")

figdir = "figures/scan"
os.makedirs(figdir, exist_ok=True)

T_C  = T_grid - 273.15  # Celsius axis
logP = np.log10(P_grid)

_Ptick_bar = [1, 3, 10, 30, 100, 300, 1000, 1500]
_Ptick_pos = [np.log10(p) for p in _Ptick_bar]
_Ptick_lbl = [str(p) for p in _Ptick_bar]

# ── Fine phase boundary from scan_v4 table (ms=0, 361T×100P grid) ────────────
_v4      = np.load("results/scan_v4_table.npz")
_ims0    = 0   # ms_grid[0] = 0.0 (salt-free)
_v4_T_C  = _v4["T_grid"] - 273.15                        # (361,) °C
_v4_logP = np.log10(_v4["P_grid"])                       # (100,)
_v4_is2  = _v4["is_two_phase"][:, :, _ims0]              # (361, 100) bool

# Split into dew (lower-P) and bubble (upper-P) boundary curves
_dew_T, _dew_logP, _bub_T, _bub_logP = [], [], [], []
for _iT in range(len(_v4_T_C)):
    _idx = np.where(_v4_is2[_iT, :])[0]
    if len(_idx):
        _dew_T.append(_v4_T_C[_iT]); _dew_logP.append(_v4_logP[_idx[0]])
        _bub_T.append(_v4_T_C[_iT]); _bub_logP.append(_v4_logP[_idx[-1]])
_dew_T   = np.array(_dew_T);   _dew_logP = np.array(_dew_logP)
_bub_T   = np.array(_bub_T);   _bub_logP = np.array(_bub_logP)

# ── Helper: average over z for T-P heatmaps ──────────────────────────────────
def tp_average(arr_3d, mask_3d=None):
    """Average arr_3d[iT, iP, iz] over iz, optionally only where mask_3d is True."""
    if mask_3d is None:
        return np.nanmean(arr_3d.astype(float), axis=2)
    out = np.full((len(T_grid), len(P_grid)), np.nan)
    for iT in range(len(T_grid)):
        for iP in range(len(P_grid)):
            m = mask_3d[iT, iP, :]
            if np.any(m):
                out[iT, iP] = np.nanmean(arr_3d[iT, iP, m].astype(float))
    return out


# ============================================================================
# Figure 1: SSI iteration count heatmaps (T °C on x, log P on y, averaged over z)
# ============================================================================
two_ph = phase_id == tp_idx
any_two_ph = two_ph.any(axis=-1)   # (nT, nP) — two-phase for any z

fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True,
                         gridspec_kw={"left": 0.07, "right": 0.88,
                                      "bottom": 0.12, "top": 0.90,
                                      "wspace": 0.08})
strategy_labels = ["Standard SSI\n(Wilson K)", "Accelerated SSI\n(Wilson K)",
                   "Accelerated SSI\n(Stability K)"]
strategy_idx = [0, 1, 2]  # std_wilson, acc_wilson, acc_stabK

for ax, si, label in zip(axes, strategy_idx, strategy_labels):
    conv_mask = flash_conv[:, :, :, si] & two_ph
    iters = flash_iter[:, :, :, si].copy().astype(float)
    iters[~conv_mask] = np.nan
    avg = tp_average(iters, two_ph)   # (nT, nP)

    # Single-phase background (gray) — from old scan's any-z envelope
    ax.contourf(T_C, logP, any_two_ph.astype(float).T,
                levels=[-0.5, 0.5], colors=["0.85"], zorder=0)

    # Iteration heatmap
    im = ax.pcolormesh(T_C, logP, avg.T, cmap="viridis",
                       vmin=1, vmax=50, shading="nearest", zorder=1)


    ax.set_xlim(T_C[0], T_C[-1])
    ax.set_ylim(logP[0], logP[-1])
    ax.set_xlabel("Temperature (°C)", fontsize=11, fontweight='bold')
    ax.set_title(label, fontsize=11, fontweight='bold')
    ax.text(340, 0.25, "single-phase", fontsize=9, fontweight='bold', color="0.4",
            ha="center", va="center",
            bbox=dict(fc="0.85", ec="none", pad=2))
    bold_ticks(ax)

axes[0].set_ylabel("Pressure (bar)", fontsize=11, fontweight='bold')
axes[0].set_yticks(_Ptick_pos)
axes[0].set_yticklabels(_Ptick_lbl, fontsize=9)
for tick in axes[0].get_yticklabels():
    tick.set_fontweight('bold')
cax = fig.add_axes([0.90, 0.12, 0.015, 0.78])
cb = fig.colorbar(im, cax=cax, label="Mean SSI iterations")
cb.set_label("Mean SSI iterations", fontsize=11, fontweight='bold')
for tick in cb.ax.get_yticklabels():
    tick.set_fontweight('bold')
fig.savefig(f"{figdir}/ssi_iterations_heatmap.pdf", bbox_inches="tight", dpi=150)
fig.savefig(f"{figdir}/ssi_iterations_heatmap.png", bbox_inches="tight", dpi=150)
print(f"Saved {figdir}/ssi_iterations_heatmap.pdf")
plt.close(fig)


# ============================================================================
# Figure 2: Speedup ratio (standard / accelerated Wilson) heatmap
# ============================================================================
both_conv = flash_conv[:, :, :, 0] & flash_conv[:, :, :, 1] & two_ph
std_it = flash_iter[:, :, :, 0].astype(float)
acc_it = flash_iter[:, :, :, 1].astype(float)
ratio = np.full_like(std_it, np.nan)
ratio[both_conv] = std_it[both_conv] / np.maximum(acc_it[both_conv], 1.0)

avg_ratio = tp_average(ratio, two_ph)   # (nT, nP)

fig, ax = plt.subplots(figsize=(5.5, 4.0),
                       gridspec_kw={"left": 0.12, "right": 0.80,
                                    "bottom": 0.12, "top": 0.92})

# Single-phase background (gray) — from old scan's any-z envelope
ax.contourf(T_C, logP, any_two_ph.astype(float).T,
            levels=[-0.5, 0.5], colors=["0.85"], zorder=0)

# Speedup heatmap
im = ax.pcolormesh(T_C, logP, avg_ratio.T, cmap="RdYlGn",
                   vmin=1.0, vmax=4.0, shading="nearest", zorder=1)


ax.set_xlim(T_C[0], T_C[-1])
ax.set_ylim(logP[0], logP[-1])
ax.set_xlabel("Temperature (°C)", fontsize=11, fontweight='bold')
ax.set_ylabel("Pressure (bar)", fontsize=11, fontweight='bold')
ax.text(340, 0.25, "single-phase", fontsize=9, fontweight='bold', color="0.4",
        ha="center", va="center",
        bbox=dict(fc="0.85", ec="none", pad=2))
ax.set_yticks(_Ptick_pos)
ax.set_yticklabels(_Ptick_lbl, fontsize=9)
bold_ticks(ax)
cax = fig.add_axes([0.83, 0.12, 0.04, 0.80])
cb2 = fig.colorbar(im, cax=cax, label="Speedup ratio (Std/Acc SSI)")
cb2.set_label("Speedup ratio (Std/Acc SSI)", fontsize=11, fontweight='bold')
for tick in cb2.ax.get_yticklabels():
    tick.set_fontweight('bold')
fig.savefig(f"{figdir}/speedup_ratio_heatmap.pdf", bbox_inches="tight", dpi=150)
fig.savefig(f"{figdir}/speedup_ratio_heatmap.png", bbox_inches="tight", dpi=150)
print(f"Saved {figdir}/speedup_ratio_heatmap.pdf")
plt.close(fig)


# ============================================================================
# Figure 3: Phase diagram (T-P) at selected z values
# ============================================================================
z_select = [0.01, 0.05, 0.1, 0.3, 0.5, 0.9]
z_indices = [np.argmin(np.abs(z_grid - zv)) for zv in z_select]

fig, ax = plt.subplots(figsize=(7, 5))
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(z_select)))

for iz, zv, c in zip(z_indices, z_select, colors):
    # Find two-phase boundary in T-P space
    is_2ph = phase_id[:, :, iz] == tp_idx  # [iT, iP]
    # For each T, find min and max P that are two-phase
    T_boundary = []
    P_lo = []
    P_hi = []
    for iT in range(len(T_grid)):
        idx_2ph = np.where(is_2ph[iT, :])[0]
        if len(idx_2ph) > 0:
            T_boundary.append(T_C[iT])
            P_lo.append(P_grid[idx_2ph[0]])
            P_hi.append(P_grid[idx_2ph[-1]])

    if T_boundary:
        T_boundary = np.array(T_boundary)
        P_lo = np.array(P_lo)
        P_hi = np.array(P_hi)
        ax.fill_betweenx(T_boundary, P_lo, P_hi, alpha=0.15, color=c)
        ax.plot(P_lo, T_boundary, '-', color=c, linewidth=1.2, label=f"$z_{{CO_2}}$ = {zv}")
        ax.plot(P_hi, T_boundary, '-', color=c, linewidth=1.2)

ax.set_xlabel("Pressure (bar)")
ax.set_ylabel(r"$T$ (°C)")
ax.set_title("Two-Phase Envelope (CPA, CO$_2$ + H$_2$O)")
ax.legend(fontsize=8, loc="upper right")
ax.set_xlim(0, P_grid[-1])
fig.tight_layout()
fig.savefig(f"{figdir}/phase_envelope_Tz.pdf", bbox_inches="tight", dpi=150)
fig.savefig(f"{figdir}/phase_envelope_Tz.png", bbox_inches="tight", dpi=150)
print(f"Saved {figdir}/phase_envelope_Tz.pdf")
plt.close(fig)


# ============================================================================
# Figure 4: Stability test — which initial guess found the lowest TPD
# ============================================================================
# For unstable points, stab_best_trial gives the trial index (0-5)
unstable = ~stab_stable

# Count per-trial across all unstable points
n_unstable = np.sum(unstable)
trial_counts = np.array([np.sum(stab_best_trial[unstable] == i) for i in range(6)])
trial_fracs = 100 * trial_counts / n_unstable

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(range(6), trial_fracs, color=plt.cm.Set2(np.arange(6)))
for bar, frac, count in zip(bars, trial_fracs, trial_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{frac:.1f}%\n({count})", ha="center", va="bottom",
            fontsize=9, fontweight='bold')

ax.set_xticks(range(6))
ax.set_xticklabels(stab_trial_labels, fontsize=9, rotation=15, ha="right",
                   fontweight='bold')
ax.set_ylabel("Fraction of unstable points (%)", fontsize=11, fontweight='bold')
ax.set_title("Best Stability Initial Guess (lowest TPD)", fontsize=11, fontweight='bold')
ax.set_ylim(0, max(trial_fracs) + 8)
bold_ticks(ax)
fig.tight_layout()
fig.savefig(f"{figdir}/stability_best_trial.pdf", bbox_inches="tight", dpi=150)
fig.savefig(f"{figdir}/stability_best_trial.png", bbox_inches="tight", dpi=150)
print(f"Saved {figdir}/stability_best_trial.pdf")
plt.close(fig)


# ============================================================================
# Figure 5: Which initial guess is "best" as a function of T and z (at mid P)
# ============================================================================
# Pick a representative pressure (e.g. 200 bar)
iP_mid = np.argmin(np.abs(P_grid - 200))

fig, ax = plt.subplots(figsize=(7, 5))
best_Tz = stab_best_trial[:, iP_mid, :].astype(float)  # [iT, iz]
# Mask stable points
stable_Tz = stab_stable[:, iP_mid, :]
best_Tz[stable_Tz] = np.nan

cmap = plt.cm.Set2
bounds = np.arange(-0.5, 6.5, 1)
norm = BoundaryNorm(bounds, cmap.N)
im = ax.pcolormesh(T_C, z_grid, best_Tz.T, cmap=cmap, norm=norm, shading="nearest")
cbar = fig.colorbar(im, ax=ax, ticks=range(6))
cbar.set_ticklabels(stab_trial_labels)
cbar.set_label("Best initial guess")
ax.set_xlabel(r"$T$ (°C)")
ax.set_ylabel("$z_{CO_2}$")
ax.set_title(f"Best Stability Initial Guess at P = {P_grid[iP_mid]:.0f} bar")
fig.tight_layout()
fig.savefig(f"{figdir}/stability_best_trial_Tz.pdf", bbox_inches="tight", dpi=150)
fig.savefig(f"{figdir}/stability_best_trial_Tz.png", bbox_inches="tight", dpi=150)
print(f"Saved {figdir}/stability_best_trial_Tz.pdf")
plt.close(fig)


# ============================================================================
# Figure 6: Convergence statistics summary bar chart
# ============================================================================
n2 = np.sum(two_ph)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Panel (a): Convergence rate by strategy
conv_rates = []
mean_iters = []
for i, name in enumerate(flash_strategy_names):
    conv_2ph = flash_conv[:, :, :, i][two_ph]
    it_2ph = flash_iter[:, :, :, i][two_ph]
    cr = 100 * np.sum(conv_2ph) / n2
    mi = np.mean(it_2ph[conv_2ph]) if np.any(conv_2ph) else 0
    conv_rates.append(cr)
    mean_iters.append(mi)

ax = axes[0]
x = np.arange(4)
labels_short = ["Std+Wilson", "Acc+Wilson", "Acc+StabK", "Robust"]
bars = ax.bar(x, conv_rates, color=["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"])
for bar, cr in zip(bars, conv_rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f"{cr:.1f}%", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(labels_short, fontsize=9)
ax.set_ylabel("Convergence rate (%)")
ax.set_title("(a) Convergence rate (two-phase points)")
ax.set_ylim(98, 101)

# Panel (b): Mean iterations
ax = axes[1]
bars = ax.bar(x, mean_iters, color=["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"])
for bar, mi in zip(bars, mean_iters):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{mi:.1f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(labels_short, fontsize=9)
ax.set_ylabel("Mean SSI iterations")
ax.set_title("(b) Mean iterations (converged two-phase)")

fig.suptitle(f"Flash Strategy Comparison ({n2} two-phase points)", fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(f"{figdir}/convergence_summary.pdf", bbox_inches="tight", dpi=150)
fig.savefig(f"{figdir}/convergence_summary.png", bbox_inches="tight", dpi=150)
print(f"Saved {figdir}/convergence_summary.pdf")
plt.close(fig)


# ============================================================================
# Print summary statistics for the paper
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY STATISTICS FOR PAPER")
print("=" * 70)
N = phase_id.size
print(f"Total grid points: {N}")
print(f"  T: {T_grid[0]:.0f}–{T_grid[-1]:.0f} K ({len(T_grid)} points)")
print(f"  P: {P_grid[0]:.0f}–{P_grid[-1]:.0f} bar ({len(P_grid)} points)")
print(f"  z: {z_grid[0]:.3f}–{z_grid[-1]:.3f} ({len(z_grid)} points)")
print(f"Single-phase: {np.sum(phase_id == sp_idx)} ({100*np.sum(phase_id == sp_idx)/N:.1f}%)")
print(f"Two-phase: {n2} ({100*n2/N:.1f}%)")
print()
for i, name in enumerate(flash_strategy_names):
    conv_2ph = flash_conv[:, :, :, i][two_ph]
    it_2ph = flash_iter[:, :, :, i][two_ph]
    nc = np.sum(conv_2ph)
    if nc > 0:
        print(f"{name}: {nc}/{n2} conv ({100*nc/n2:.1f}%), "
              f"mean iter={np.mean(it_2ph[conv_2ph]):.1f}, "
              f"median={np.median(it_2ph[conv_2ph]):.0f}")
print()

# Speedup
both = flash_conv[:, :, :, 0] & flash_conv[:, :, :, 1] & two_ph
nb = np.sum(both)
r = std_it[both] / np.maximum(acc_it[both], 1)
print(f"Speedup (std vs acc Wilson, n={nb}):")
print(f"  Mean: {np.mean(r):.2f}x, Median: {np.median(r):.2f}x")
print(f"  Iteration reduction: {100*(1-np.mean(acc_it[both])/np.mean(std_it[both])):.0f}%")
print()

print(f"Unstable points: {n_unstable}")
for i, label in enumerate(stab_trial_labels):
    print(f"  Trial {i} ({label}): {trial_counts[i]} ({trial_fracs[i]:.1f}%)")
print()

# Cumulative: how many guesses needed
print("Cumulative guess requirement:")
# For each unstable point, find the first trial (in order) that found TPD < -1e-7
stab_trial_tpd = d["stab_trial_tpd"]  # [iT, iP, iz, 6]
cum = 0
for k in range(6):
    # Points where trial k is the first to detect instability
    first_unstable_at_k = np.zeros_like(unstable)
    for iT in range(len(T_grid)):
        for iP in range(len(P_grid)):
            for iz in range(len(z_grid)):
                if not unstable[iT, iP, iz]:
                    continue
                for t in range(6):
                    if stab_trial_tpd[iT, iP, iz, t] < -1e-7:
                        if t == k:
                            first_unstable_at_k[iT, iP, iz] = True
                        break
    cum += np.sum(first_unstable_at_k)
    print(f"  With {k+1} guess(es): {cum}/{n_unstable} ({100*cum/n_unstable:.1f}%)")

print("\nDone.")
