"""Regenerate figures/flash_vs_z.png with clean labels."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ecpa.parameters import make_params
from ecpa.solution_table import make_solution_guess_fn
from ecpa.flash import flash_co2_h2o_salt_fast

# ── load solution table ──────────────────────────────────────────────────────
params = make_params()
npz      = np.load("results/solution_table.npz")
guess_fn = make_solution_guess_fn(dict(npz))

# ── conditions ───────────────────────────────────────────────────────────────
P_bar  = 200.0
m_feed = 1.0          # mol/kg feed NaCl
Mw     = 0.018015     # kg/mol
temps  = [350.0, 400.0, 450.0, 500.0]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

z_arr  = np.linspace(0.01, 0.99, 200)

# ── run flash ────────────────────────────────────────────────────────────────
results = {}   # T → dict of arrays

for T in temps:
    beta_arr   = []
    msrat_arr  = []   # ms_aq / m_feed
    mc_arr     = []   # CO2 molality in aq phase
    x1c_arr    = []   # H2O mole fraction in CO2-rich phase

    for z in z_arr:
        try:
            r = flash_co2_h2o_salt_fast(
                T=T, P_bar=P_bar, z_co2=z, m_tot=m_feed,
                solution_guess_fn=guess_fn, params=params,
            )
        except Exception:
            beta_arr.append(np.nan)
            msrat_arr.append(np.nan)
            mc_arr.append(np.nan)
            x1c_arr.append(np.nan)
            continue

        if r.get("phase") == "single_phase":
            beta_arr.append(np.nan)
            msrat_arr.append(np.nan)
            mc_arr.append(np.nan)
            x1c_arr.append(np.nan)
        else:
            beta  = float(r["beta"])
            ms_aq = float(r["ms_aq"])
            x1w   = float(r["x_aq"]["x1w"])
            x4w   = float(r["x_aq"]["x4w"])
            x1c   = float(r["x_c"]["x1c"])

            beta_arr.append(beta)
            msrat_arr.append(ms_aq / m_feed)
            mc = x4w / (x1w * Mw) if x1w > 1e-10 else np.nan
            mc_arr.append(mc)
            x1c_arr.append(x1c)

    results[T] = {
        "beta":  np.array(beta_arr),
        "msrat": np.array(msrat_arr),
        "mc":    np.array(mc_arr),
        "x1c":  np.array(x1c_arr),
    }

# ── plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle(
    r"$P = 200\,\mathrm{bar}$, $m_\mathrm{feed} = 1.0\,\mathrm{mol\,kg^{-1}}$ NaCl",
    fontsize=12,
)

ax_beta, ax_ms, ax_mc, ax_x1c = axes.flat

for T, col in zip(temps, colors):
    d  = results[T]
    lbl = f"$T = {int(T)}\,\mathrm{{K}}$"

    # mask to two-phase only (beta in (0,1))
    mask = np.isfinite(d["beta"]) & (d["beta"] > 1e-4) & (d["beta"] < 1 - 1e-4)

    ax_beta.plot(z_arr[mask], d["beta"][mask],  color=col, label=lbl)
    ax_ms.plot(  z_arr[mask], d["msrat"][mask], color=col, label=lbl)
    ax_mc.plot(  z_arr[mask], d["mc"][mask],    color=col, label=lbl)
    ax_x1c.plot( z_arr[mask], d["x1c"][mask],  color=col, label=lbl)

# reference line ms_aq/m_feed = 1 (no salinity amplification)
ax_ms.axhline(1.0, color="k", lw=0.8, ls="--", label="$m_s^\\mathrm{aq} = m_\\mathrm{feed}$")

# ── labels ───────────────────────────────────────────────────────────────────
z_label = "Overall CO$_2$ mole fraction $z$"

ax_beta.set_xlabel(z_label)
ax_beta.set_ylabel(r"CO$_2$-rich phase fraction $\beta$")
ax_beta.set_title("(a) Phase split")
ax_beta.set_xlim(0, 1)
ax_beta.set_ylim(0, 1)
ax_beta.legend(fontsize=8)

ax_ms.set_xlabel(z_label)
ax_ms.set_ylabel(r"$m_s^\mathrm{aq}\,/\,m_\mathrm{feed}$")
ax_ms.set_title("(b) NaCl salinity amplification")
ax_ms.set_xlim(0, 1)
ax_ms.set_ylim(bottom=0.9)
ax_ms.legend(fontsize=8)

ax_mc.set_xlabel(z_label)
ax_mc.set_ylabel(r"CO$_2$ molality $m_c$ [mol kg$^{-1}$]")
ax_mc.set_title("(c) CO$_2$ solubility in aqueous phase")
ax_mc.set_xlim(0, 1)
ax_mc.legend(fontsize=8)

ax_x1c.set_xlabel(z_label)
ax_x1c.set_ylabel(r"H$_2$O mole fraction $x_\mathrm{H_2O}^C$")
ax_x1c.set_title("(d) Water content of CO$_2$-rich phase")
ax_x1c.set_xlim(0, 1)
ax_x1c.set_ylim(bottom=0)
ax_x1c.legend(fontsize=8)

fig.tight_layout()
out = "figures/flash_vs_z.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
