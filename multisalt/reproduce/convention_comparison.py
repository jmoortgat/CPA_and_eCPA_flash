"""What the salinity convention costs, measured on the scored data.

Section 3 of Paper II equates the equilibrium salt molality with the
prepared-brine molality, while the flash of Paper III lets salt concentrate
as water leaves the aqueous phase. This script quantifies the effect of the
two conventions on the reported solubilities and phase fractions.

The two conventions:

``as-aqueous`` (the default, and what the papers report)
    ``mi`` IS the equilibrium aqueous molality. This is how the experimental
    papers back-calculate solubility -- they use the prepared-brine water
    fraction throughout -- so scoring against them this way is like-for-like.

``chain``
    ``mi`` is the prepared-brine FEED molality, and the equilibrium aqueous
    molality is amplified by salt conservation as water leaves:
    ``m_aq* = m_feed * z_w / [(1 - beta) * x_w_aq]``.

The difference grows with how much water leaves the aqueous phase, so it is
largest where beta is large: high temperature, low pressure, high salinity.
This script measures it rather than arguing about it -- per row, per source,
and as the change in the headline AAD.

Run:  python3 reproduce/convention_comparison.py [--backend fortran]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np

_HERE = Path(__file__).resolve().parent
_CODE = _HERE.parent
for p in (str(_CODE), str(_CODE / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from fitting import dataset as ds          # noqa: E402
from fitting.predict import Z_GAS_RETRIES, _slot   # noqa: E402
from ms_ecpa import stability_flash        # noqa: E402
from ms_ecpa.parameters import Mw          # noqa: E402

IONS = ds.IONS


def solubility_under(T, P, mi, gas, convention):
    """Salt-free aqueous gas mole fraction, under one salinity convention.

    Mirrors fitting.predict.solubility -- same feed ladder, same
    degenerate-split guard -- but passes the convention through, which
    predict() does not expose.
    """
    k = _slot(gas)
    for zg in Z_GAS_RETRIES:
        z = np.zeros(3)
        z[0], z[k] = 1.0 - zg, zg
        try:
            r = stability_flash(float(T), float(P), z, mi,
                                salinity_convention=convention)
        except Exception:
            continue
        if r.phases != 2 or r.flash_result is None:
            continue
        x = np.asarray(r.flash_result.x_aq, dtype=float)
        xH, xg = float(x[0]), float(x[k])
        if xH + xg <= 0:
            continue
        xsf = xg / (xH + xg)
        if abs(xsf - zg) < 1e-3 * max(zg, 1e-6):
            continue
        return xsf, float(r.flash_result.beta)
    return float("nan"), float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="co2_solubility")
    ap.add_argument("--gas", default="co2")
    ap.add_argument("--max-rows", type=int, default=0,
                    help="0 = every row in the family")
    ap.add_argument("--out", default=str(_CODE / "results" /
                                         "convention_comparison.json"))
    args = ap.parse_args()

    fams = ds.build(split_yh2o=True, keep_unsplit=True)
    d = fams[args.family].df
    if args.max_rows:
        d = d.iloc[:: max(1, len(d) // args.max_rows)]
    y = np.asarray(d["value"], dtype=float)

    rows = []
    for i, (_, row) in enumerate(d.iterrows()):
        mi = np.nan_to_num(np.array([row[c] for c in IONS], dtype=float))
        T, P = float(row["T_K"]), float(row["P_bar"])
        xa, ba = solubility_under(T, P, mi, args.gas, "as-aqueous")
        xc, bc = solubility_under(T, P, mi, args.gas, "chain")
        rows.append(dict(T_K=T, P_bar=P, m_tot=float(np.sum(mi)),
                         source=str(row.get("source", "")),
                         exp=float(y[i]), as_aq=xa, chain=xc,
                         beta_as_aq=ba, beta_chain=bc))
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(d)}", flush=True)

    a = np.array([r["as_aq"] for r in rows], float)
    c = np.array([r["chain"] for r in rows], float)
    m_tot = np.array([r["m_tot"] for r in rows], float)
    T = np.array([r["T_K"] for r in rows], float)
    both = np.isfinite(a) & np.isfinite(c)
    rel = np.full(len(rows), np.nan)
    rel[both] = 100.0 * (c[both] - a[both]) / a[both]

    def aad(pred):
        ok = np.isfinite(pred) & (y != 0)
        return 100.0 * float(np.mean(np.abs(pred[ok] - y[ok]) / y[ok])), int(ok.sum())

    aad_a, n_a = aad(a)
    aad_c, n_c = aad(c)

    print(f"\nfamily {args.family}: {len(rows)} rows, both conventions solved "
          f"on {int(both.sum())}")
    print(f"  AAD as-aqueous : {aad_a:7.4f} %   (N = {n_a})")
    print(f"  AAD chain      : {aad_c:7.4f} %   (N = {n_c})")
    print(f"  chain - as-aqueous on the AAD: {aad_c - aad_a:+.4f} pp")
    print("\nrelative change in predicted solubility, chain vs as-aqueous:")
    if both.any():
        print(f"  median {np.nanmedian(rel):+.3f} %   p95 {np.nanpercentile(rel[both], 95):+.3f} %"
              f"   max {np.nanmax(rel[both]):+.3f} %")
        print("\n  by salinity:")
        for lo, hi in ((0, 0.5), (0.5, 2), (2, 4), (4, 99)):
            s = both & (m_tot >= lo) & (m_tot < hi)
            if s.any():
                print(f"    m_tot {lo:>4}-{hi:<4} n={int(s.sum()):4d}  "
                      f"median {np.nanmedian(rel[s]):+7.3f} %   max {np.nanmax(rel[s]):+8.3f} %")
        print("\n  by temperature:")
        for lo, hi in ((250, 350), (350, 450), (450, 550), (550, 700)):
            s = both & (T >= lo) & (T < hi)
            if s.any():
                print(f"    T {lo}-{hi} K  n={int(s.sum()):4d}  "
                      f"median {np.nanmedian(rel[s]):+7.3f} %   max {np.nanmax(rel[s]):+8.3f} %")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        dict(family=args.family, n_rows=len(rows), n_both=int(both.sum()),
             aad_as_aqueous=round(aad_a, 4), aad_chain=round(aad_c, 4),
             rows=rows), indent=1) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
