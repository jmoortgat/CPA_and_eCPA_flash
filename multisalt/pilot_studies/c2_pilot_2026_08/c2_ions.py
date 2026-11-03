"""Two-parameter ethane-ion Setschenow calibration (final C2 ion story).

With the cleaned quadratic kij(T) (refit_state.json fitQ), calibrate:
  s_NaCl  scaling the transferred (Na+, Cl-) energy pair, fitted to the
          IUPAC smoothed NaCl k_sca at KS_FIT_T;
  s_K     scaling the transferred K+ energies (Cl- from s_NaCl), fitted
          to the Ben-Naim & Yaacobi KCl k_sca (283-303 K).
Remaining ions (Ca, Mg, SO4) keep the pure transfer, declared untested.
The temperature SHAPE of both k_S(T) curves is predicted, not fitted.

    python c2_ions.py         # runs everything, ~15-25 min serial
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c2_pilot as cp                     # noqa: E402
import ms_ecpa.parameters as prm          # noqa: E402

OUT = HERE / "c2_ions_results.json"


def rebind_ions(kij, s_nacl, s_k, pol2=cp.POL_C2):
    """rebind_c2 variant with per-ion scaling: (Na, Cl) by s_nacl, K by
    s_k, remaining ions pure transfer. Ion order: Na, Cl, K, Ca, Mg, SO4."""
    cp.rebind_c2(*kij, s_all=1.0, pol2=pol2)
    scale = np.array([s_nacl, s_nacl, s_k, 1.0, 1.0, 1.0])
    prm.A_U2i[:] = cp._ORIG_A_U2i * cp.TC_RATIO * scale
    prm.B_U2i[:] = cp._ORIG_B_U2i * scale


def solub(kij, s_nacl, s_k, T, mi6):
    try:
        rebind_ions(kij, s_nacl, s_k)
        from ms_ecpa import stability_flash
        for zg in cp.Z_GAS_RETRIES:
            z = np.array([1.0 - zg, zg, 0.0])
            try:
                res = stability_flash(float(T), cp.KS_P_TOTAL_BAR, z,
                                      np.asarray(mi6, dtype=float))
            except Exception:
                continue
            if res.phases == 2 and res.flash_result is not None:
                fr = res.flash_result
                x1w, x2w, x3w = fr.x_aq
                m_gas = x2w / (prm.Mw * x1w)
                P_Pa = cp.KS_P_TOTAL_BAR * 1e5
                V = float(fr.Z_aq) * cp.R_J * float(T) / P_Pa
                M_mix = x1w * prm.Mw + x2w * cp.M_C2 + x3w * 0.04401
                for k in range(6):
                    M_mix += prm.Mw * x1w * float(mi6[k]) * cp.M_ION[k]
                return (float(m_gas), float(fr.fugacities_gas[1]),
                        float(M_mix / V))
        return (float("nan"),) * 3
    finally:
        cp.restore_ch4()


def ks(kij, s_nacl, s_k, T, salt, m=1.0, wcache={}):
    if T not in wcache:
        wcache[T] = solub(kij, 1.0, 1.0, T, cp.MI_WATER)
    sw = wcache[T]
    sb = solub(kij, s_nacl, s_k, T, cp.mi_for(salt, m))
    if not (np.isfinite(sw[0]) and np.isfinite(sb[0])):
        return float("nan")
    Shw = (sw[0] * (sw[2] / 1000.0)) / sw[1]
    Shb = (sb[0] * (sb[2] / 1000.0) / (1.0 + m * cp.M_SALT_KG[salt])) / sb[1]
    c_salt = m * (sb[2] / 1000.0) / (1.0 + m * cp.M_SALT_KG[salt])
    return float(np.log10(Shw / Shb) / c_salt)


def scalar_fit(fun, targets, x0, label):
    from scipy.optimize import least_squares

    def resid(x):
        return np.array([fun(float(x[0]), T) / t - 1.0 for T, t in targets])
    t0 = time.perf_counter()
    res = least_squares(resid, [x0], diff_step=1e-3, method="trf",
                        bounds=([0.1], [4.0]))
    print(f"  [{label}] s = {res.x[0]:.4f}, rms {np.sqrt(np.mean(res.fun**2))*100:.1f}%, "
          f"nfev {res.nfev} ({time.perf_counter()-t0:.0f}s)", flush=True)
    return float(res.x[0])


def main():
    kij = json.load(open(HERE / "refit_state.json"))["fitQ"]["best"]["x"]
    ksdf = cp.load_ks()
    sm = ksdf[(ksdf.salt == "NaCl") & (ksdf.lab == "smoothed")]
    nacl_t = {float(r.T_K): float(r.ks_dm3_mol) for r in sm.itertuples()}
    kcl = ksdf[(ksdf.salt == "KCl")]
    kcl_t = {float(r.T_K): float(r.ks_dm3_mol) for r in kcl.itertuples()}

    wc = {}
    print("=== fit s_NaCl (5 T) ===", flush=True)
    s_nacl = scalar_fit(lambda s, T: ks(kij, s, 1.0, T, "NaCl", wcache=wc),
                        [(T, nacl_t[T]) for T in cp.KS_FIT_T], 0.75, "s_NaCl")
    print("=== fit s_K (5 T, Cl from s_NaCl) ===", flush=True)
    s_k = scalar_fit(lambda s, T: ks(kij, s_nacl, s, T, "KCl", wcache=wc),
                     sorted(kcl_t.items()), 1.3, "s_K")

    print("=== validation: full-range k_S(T) with fitted scales ===",
          flush=True)
    out = {"kij_quadratic": kij, "s_nacl": s_nacl, "s_k": s_k,
           "NaCl": [], "KCl": []}
    for T in np.arange(283.15, 348.16, 5.0):
        v = ks(kij, s_nacl, s_k, float(T), "NaCl", wcache=wc)
        out["NaCl"].append({"T_K": float(T), "ks": v,
                            "exp": nacl_t.get(float(T))})
        print(f"  NaCl T={T:.2f}: model {v:.4f} exp "
              f"{nacl_t.get(float(T), float('nan'))}", flush=True)
    for T in sorted(kcl_t):
        v = ks(kij, s_nacl, s_k, float(T), "KCl", wcache=wc)
        out["KCl"].append({"T_K": T, "ks": v, "exp": kcl_t[T]})
        print(f"  KCl  T={T:.2f}: model {v:.4f} exp {kcl_t[T]}", flush=True)
    for salt in ("NaCl", "KCl"):
        d = [r["ks"] / r["exp"] - 1 for r in out[salt]
             if r["exp"] and np.isfinite(r["ks"])]
        out[f"{salt}_aad_pct"] = float(np.mean(np.abs(d)) * 100)
        print(f"{salt}: AAD {out[f'{salt}_aad_pct']:.1f}%")
    OUT.write_text(json.dumps(out, indent=2))
    print("->", OUT)


if __name__ == "__main__":
    main()
