"""Assemble the ms-eCPA fitting set from GasBrineBench, and cut the folds.

What goes in
------------
CO2 and CH4 only, because those are the two gases that have ion parameters
of their own. Four families:

    co2_binary / ch4_binary   salt-free gas-water solubility  -> kij(T)
    co2_solubility / ch4_...  brine solubility                -> ion energies
    co2_yh2o / ch4_yh2o       water content of the gas        -> ion energies
                                                                 + association
    rho                       brine density (gas-free)        -> Peneloux

What stays out, and why
-----------------------
**Henry constants.** Derived from Henry's law, whose limited validity is the
reason for building an equation of state at all. Fitting to them would be
circular. Figure input only, never a fit or validation target.

**N2, H2, C2H6, C3H8, O2 brine data.** Those gases have no ion parameters:
they enter by slot swap with ion energies DERIVED by corresponding states
(A_N2,i = A_CH4,i * Tc(N2)/Tc(CH4)). Fitting to their brine data would turn
the transferability result into a tautology. They are the held-out test.
Their gas-WATER binaries are a separate matter -- each gas has its own three
kij coefficients, fitted independently, which do not couple to CO2/CH4.

**Complement-derived y_H2O.** Where a source printed the gas fraction and
y_H2O was obtained as 1 - y_gas, the subtraction destroys relative precision
at high gas purity; GasBrineBench codes those `U`. Excluded by the quality
filter below rather than by name.

Exclusions are by DOCUMENTED RULE, never by source identity
---------------------------------------------------------
Quality `U`, the `lle-regime` tag, and the stated T/P envelope. Never
"we dropped Nighswander". Reproducing the previous fit's undocumented
source exclusions is precisely what this refit exists to escape.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from ms_ecpa import gbb as _gbb

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parents[1]

IONS = ["m_Na", "m_Cl", "m_K", "m_Ca", "m_Mg", "m_SO4"]

#: Per-family 1-sigma, as a RELATIVE fraction unless noted absolute.
#: These are measured inter-source reproducibility floors -- the median
#: disagreement between different laboratories at the same state, computed
#: from GasBrineBench itself -- not judgement calls. Weighting by 1/sigma^2
#: then says "do not fit tighter than the data can distinguish", which is an
#: estimate rather than a preference and is correspondingly hard to argue
#: with in review.
#:
#:   co2 brine solubility   2.80 %   (124 cross-source pairs)
#:   co2 water binary       3.65 %   (104)
#:   co2 y_H2O             15.94 %   ( 70)   <- 5x looser than solubility
#:   ch4 water binary       5.77 %   (150)
#:   ch4 y_H2O              4.14 %   ( 28, thin -- treated as 8 %)
#:   brine density          no cross-source pairs at all (single source)
SIGMA = {
    "co2_solubility": ("rel", 0.028),
    "ch4_solubility": ("rel", 0.028),   # no pairs; assume the CO2 floor
    "co2_binary":     ("rel", 0.037),
    "ch4_binary":     ("rel", 0.058),
    "co2_yh2o":       ("rel", 0.159),
    "ch4_yh2o":       ("rel", 0.080),   # 28 pairs gave 4.1 %; widened
    "rho":            ("abs", 0.5),     # kg/m3; no independent check exists
    # The two halves of each y_H2O family inherit its floor: the split is by
    # which PARAMETERS can reach a row, not by how well it was measured.
    "co2_yh2o_sf":    ("rel", 0.159),
    "co2_yh2o_brine": ("rel", 0.159),
    "ch4_yh2o_sf":    ("rel", 0.080),
    "ch4_yh2o_brine": ("rel", 0.080),
}


@dataclass
class Family:
    """One fitted family: its rows, its sigma, and its balance weight."""
    name: str
    df: pd.DataFrame
    sigma_kind: str
    sigma: float
    budget: float = 1.0          # B_f; the Pareto knob

    @property
    def n(self) -> int:
        return len(self.df)

    def weights(self) -> np.ndarray:
        """w_i = sqrt(B_f / N_f): decouples a family's pull from its size."""
        if self.n == 0:
            return np.zeros(0)
        return np.full(self.n, np.sqrt(self.budget / self.n))


def _gbb_root() -> Path:
    """Where GasBrineBench lives -- delegated to :mod:`ms_ecpa.gbb`.

    The upward search that used to live here was the correct half of a pair:
    the figure scripts carried a second, level-counting copy that resolved to
    nothing and silently fell back to the legacy parquets. Both now call one
    resolver, so they cannot disagree about which measurements are in play.
    """
    return _gbb.ensure()


def _load(family: str) -> pd.DataFrame:
    # exclude_tags=None on purpose: this assembly keeps every row and applies
    # the documented exclusion rules at the top of this module, rather than
    # inheriting the package default.
    return _gbb.load(family, exclude_tags=None)


def _salt_free(d: pd.DataFrame) -> pd.Series:
    return d[IONS].fillna(0.0).abs().sum(axis=1) == 0


def build(*, drop_quality=("U",), t_range=(273.0, 633.0),
          p_max_bar=3500.0, budgets=None, split_yh2o=False,
          keep_unsplit=True, relative_sigma=False) -> dict[str, Family]:
    """Assemble every fitted family. ``budgets`` overrides B_f per family.

    ``split_yh2o`` cuts each y_H2O family into its SALT-FREE and BRINE halves.
    The split is by which parameters can reach a row, and it is not a fine
    distinction: 84.8 % of the CO2 y_H2O rows and 100 % of the CH4 rows carry
    no salt at all, so the ion energies -- which act only through ion molality
    -- cannot move them. Fitting y_H2O inside the ion-energy stage therefore
    reweights rows that the parameters being fitted do not reach. The
    salt-free half belongs with the binary/solvation block that does reach it.

    ``keep_unsplit`` additionally keeps the undivided family under its
    original name, for SCORING only, so the reported AAD stays directly
    comparable with a run that did not split.
    """
    budgets = dict(budgets or {})
    sol = _load("solubility")
    yh = _load("y_h2o")
    rho = _load("rho")

    def clean(d):
        d = d[~d["quality"].isin(drop_quality)]
        d = d[d["tag"] != "lle-regime"]
        d = d[(d["T_K"] >= t_range[0]) & (d["T_K"] <= t_range[1])]
        if "P_bar" in d:
            d = d[d["P_bar"].isna() | (d["P_bar"] <= p_max_bar)]
        return d.reset_index(drop=True)

    fams: dict[str, Family] = {}

    for gas in ("co2", "ch4"):
        # Brine solubility. Molality is the model's native aqueous output for
        # CH4 and the mole fraction for CO2, so take each gas on the basis its
        # own data is densest in and convert once, rather than mixing bases.
        prop = "xc_saltfree" if gas == "co2" else "solubility_molality"
        s = clean(sol[(sol["gas"] == gas) & (sol["property"] == prop)])
        fams[f"{gas}_solubility"] = _mk(f"{gas}_solubility",
                                        s[~_salt_free(s)], budgets)
        fams[f"{gas}_binary"] = _mk(f"{gas}_binary",
                                    s[_salt_free(s)], budgets)
        y = clean(yh[yh["gas"] == gas])
        if split_yh2o:
            sf = _salt_free(y)
            fams[f"{gas}_yh2o_sf"] = _mk(f"{gas}_yh2o_sf", y[sf], budgets)
            fams[f"{gas}_yh2o_brine"] = _mk(f"{gas}_yh2o_brine", y[~sf],
                                            budgets)
        if (not split_yh2o) or keep_unsplit:
            fams[f"{gas}_yh2o"] = _mk(f"{gas}_yh2o", y, budgets)

    fams["rho"] = _mk("rho", clean(rho), budgets)

    if relative_sigma:
        # AAD is a mean of RELATIVE deviations, so an objective meant to track
        # it must make every residual relative. Six families already are;
        # brine density carries Al Ghafri's absolute 0.5 kg/m3, which makes its
        # residual proportional to ABSOLUTE error instead. Re-express that same
        # floor at the data's own median density, which preserves its magnitude
        # while making the residual relative. Nothing else changes.
        f = fams["rho"]
        med = float(np.median(np.abs(f.df["value"].to_numpy(dtype=float))))
        fams["rho"] = Family(f.name, f.df, "rel", f.sigma / med, f.budget)
    return fams


def _mk(name, df, budgets) -> Family:
    kind, sig = SIGMA[name]
    df = df.reset_index(drop=True).copy()
    df["source_group"] = _source_group(df["source"])
    return Family(name, df, kind, sig,
                  budget=float(budgets.get(name, 1.0)))


def _source_group(src: pd.Series) -> pd.Series:
    """Collapse a `source` cell to the PUBLICATION that produced it.

    Some families carry the transcription filename in the source cell --
    "ALGHAFRI(2012) / EXP_CaCl_T283K.txt". Grouping folds on the raw string
    would make brine density look like 42 independent sources when it is two
    papers from one laboratory, and leave-one-source-out would then hold out
    a single isotherm of a single salt from the same lab: interpolation
    dressed up as prediction, reporting a flatteringly small error.

    Collapsing on the publication is what makes the fold boundary mean
    "a laboratory we did not train on".
    """
    return (src.astype(str)
               .str.replace(r"\s*/.*$", "", regex=True)   # drop " / file.txt"
               .str.strip())


# ---------------------------------------------------------------------------
# folds
# ---------------------------------------------------------------------------

def loso_folds(fam: Family, min_sources: int = 5):
    """Leave-one-source-out, CYCLING: every source is held out exactly once.

    Not a fixed validation set. A fixed split makes the result hostage to
    which laboratories happen to land in it; cycling turns source choice into
    a measured spread instead of an unexamined assumption.

    Folds are grouped BY SOURCE, never by row. Adjacent pressure points on
    one isotherm from one laboratory are near-duplicates, so a shuffled
    row-wise split puts near-copies on both sides and reports a flattering
    number that measures interpolation, not prediction.

    Returns [] when the pool is too thin to hold one out and still fit --
    N2 brine solubility has a single source, and brine density has one
    laboratory (Al Ghafri), so LOSO is impossible there by construction and
    those families need an extrapolation-shell split instead.
    """
    srcs = sorted(fam.df["source_group"].dropna().unique())
    if len(srcs) < min_sources:
        return []
    return [(fam.df["source_group"] != s, fam.df["source_group"] == s)
            for s in srcs]


def shell_split(fam: Family, quantile: float = 0.85):
    """Hold out the outer T/P shell: train inside, predict outside.

    For families too thin for leave-one-source-out. This is the harder and
    more honest test anyway -- users need extrapolation, not interpolation.
    """
    d = fam.df
    if len(d) < 20:
        return None
    t_cut = d["T_K"].quantile(quantile)
    p_cut = d["P_bar"].quantile(quantile) if d["P_bar"].notna().any() else np.inf
    test = (d["T_K"] > t_cut) | (d["P_bar"] > p_cut)
    return ~test, test
