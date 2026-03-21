"""
Newton inner solver iteration histogram — eCPA ternary scan.

Single panel: overlapping histograms of mean iterations-per-call for the
aqueous-phase (Zw, εr, χ1w) and CO2-rich-phase (Zc) Newton solvers,
aggregated over all two-phase (T, P, z, ms) conditions.

CO2-rich phase plotted in background; aqueous phase in foreground.
Median lines shown for each phase.

Output: figures/scan_v3/ecpa_newton_stats.png/.pdf
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc
import scienceplots  # noqa: F401
import os

OUT_DIR = "figures/scan_v3"
os.makedirs(OUT_DIR, exist_ok=True)

plt.style.use(["science"])
plt.rcParams.update({
    "figure.dpi":   150,
    "savefig.dpi":  300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.5,
})

# ── data ──────────────────────────────────────────────────────────────────────
print("Loading data ...")
df     = pd.read_parquet("results/scan_v3_metrics.parquet")
df_2ph = df[(df["eos_type"] == "eCPA") & df["is_two_phase"]].copy()

# mean iterations per Newton call (one call per SSI step per phase)
aq_iters = (df_2ph["n_newton_aq_iters"] /
            df_2ph["n_newton_aq"].clip(lower=1)
            ).where(df_2ph["n_newton_aq"] > 0).dropna()
c_iters  = (df_2ph["n_newton_c_iters"] /
            df_2ph["n_newton_c"].clip(lower=1)
            ).where(df_2ph["n_newton_c"] > 0).dropna()

med_aq = aq_iters.median()
med_c  = c_iters.median()

print(f"  Aqueous  : median={med_aq:.2f}, max={aq_iters.max():.2f}, "
      f"frac>8={( aq_iters>8).mean()*100:.1f}%")
print(f"  CO2-rich : median={med_c:.2f},  max={c_iters.max():.2f}")

# ── figure ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(3.6, 2.5),
                       gridspec_kw=dict(left=0.15, right=0.97,
                                        top=0.95, bottom=0.17))

BINS   = np.arange(1.0, 8.5, 0.25)   # 0.25-wide bins, x-range 1–8
XLIM   = (1.0, 8.0)

COL_C  = cmc.lajolla(0.60)   # warm orange for CO2-rich
COL_AQ = cmc.lapaz(0.40)     # cool blue for aqueous

# CO2-rich phase in background
ax.hist(np.clip(c_iters, *XLIM),  bins=BINS, density=True,
        color=COL_C,  alpha=0.65, label=r"CO$_2$-rich phase", zorder=2)

# aqueous phase in foreground
ax.hist(np.clip(aq_iters, *XLIM), bins=BINS, density=True,
        color=COL_AQ, alpha=0.65, label="Aqueous phase", zorder=3)

# ── median lines ───────────────────────────────────────────────────────────────
ymax = ax.get_ylim()[1]
ax.axvline(med_c,  color=COL_C,  lw=1.0, ls="--", zorder=4)
ax.axvline(med_aq, color=COL_AQ, lw=1.0, ls="--", zorder=5)

# annotate medians (offset slightly so labels don't collide)
offset_c  = +0.12
offset_aq = +0.12
# if medians are close, stagger vertically
vert_c  = 0.93
vert_aq = 0.75 if abs(med_aq - med_c) < 1.0 else 0.93
ax.text(med_c  + offset_c,  ymax * vert_c,
        rf"$\tilde{{n}}={med_c:.1f}$",
        color=COL_C,  fontsize=6.5, va="top", zorder=6)
ax.text(med_aq + offset_aq, ymax * vert_aq,
        rf"$\tilde{{n}}={med_aq:.1f}$",
        color=COL_AQ, fontsize=6.5, va="top", zorder=7)

# ── axes formatting ────────────────────────────────────────────────────────────
ax.set_xlim(*XLIM)
ax.set_xticks(range(1, 9))
ax.set_xticklabels([str(i) for i in range(1, 9)], fontsize=6.5)
ax.set_xlabel("Newton iterations per call", fontsize=8)
ax.set_ylabel("Probability density", fontsize=8)
ax.tick_params(axis="both", length=2, pad=1.5, labelsize=6.5)

ax.legend(fontsize=6.5, framealpha=0.0, loc="upper right",
          handlelength=1.2, handletextpad=0.4)

# ── save ───────────────────────────────────────────────────────────────────────
outname = "ecpa_newton_stats"
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT_DIR}/{outname}.{ext}")
plt.close(fig)
print(f"  -> {OUT_DIR}/{outname}.png/.pdf")
print("Done.")
