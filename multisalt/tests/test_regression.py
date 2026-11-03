"""Regression test: refactored ms_ecpa.solve_elv must reproduce the original
eCPA_ELV.py outputs.

Baseline values were captured by running the original interactive script
(reference_original/eCPA_ELV_ORIGINAL.py) with piped stdin. Case A is the
worked example published in docs/Readme.pdf. Values are at the precision the
original printed; tolerances are set to the corresponding rounding half-width.

RE-ANCHORING (2026-07-08, commit 4f7fbf7): cases A (Ca2+/Mg2+-bearing mixed
brine) and E (CaCl2 1 m) were recaptured from the current library after the
ADOPTED repaired Ca2+ row + Mg2+ polarizability -- those parameters now
deliberately differ from the original script, so original-script equivalence
for Ca2+/Mg2+ cases holds only at tag v1.2-pre-ca-adoption. Both solver
paths (fsolve/newton) agree at print precision on the recaptured values.
Cases B/C/D (salt-free and NaCl-only) still cross-check the ORIGINAL script.

RE-ANCHORING (2026-09-18, `ws` only): the original script -- and this library
until today -- computed the aqueous mass fractions on a denominator that
omitted the dissolved-CO2 mass term while keeping the CH4 one, so `wh`, `wc`
and `ws` were mass ratios rather than the mass fractions ELVResult documents.
For case A the four aqueous mass fractions summed to 103.18 %. `eos.py` now
divides by the full solution mass (MMw), they sum to 100.00 %, and the three
salt-bearing `ws` anchors move: A 13.00 -> 12.60, D 10.47 -> 10.28,
E 9.99 -> 9.50. No other field in any case changes -- Zw, rho_w, eps_r, mh,
mc, Zh, rho_h and all three fugacities are bit-identical -- so this is a
reporting correction, not a physics change, and original-script equivalence
for `ws` is deliberately given up.

RE-ANCHORING (2026-09-18, `rho_w` and one `mc`): the library's water molar
mass moved from the rounded 0.018 kg/mol to 0.01801528 (n_w 55.5556 ->
55.5084), so that the model constant agrees with the molality conversions
applied to the data it is fitted against (scripts/total_mole_basis.py and
GasBrineBench both use 0.01801528). The shift is +0.085 %, and it reaches
exactly two reported quantities: `rho_w` in all five cases (+0.07-0.08 %)
and `mc` in case E (1.2946 -> 1.2933). Zw, eps_r, mh, Zh, rho_h, xs and all
three fugacities are unchanged within tolerance. Paper I reproduction is
NOT affected: `paper1_config` pins Mw back to 0.018 for the duration of
apply_paper1(), and tests/test_paper1_config.py is bit-identical across
this change.

Run directly:   python tests/test_regression.py
Or with pytest: pytest tests/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ms_ecpa import solve_elv  # noqa: E402

# Each case: inputs + expected printed outputs from the ORIGINAL script.
CASES = [
    dict(name="A: 323K 500bar YC=0.8 mixed brine",  # re-anchored 4f7fbf7
         T=323, P=500, YC=0.8, mi=[1, 2, 0.6, 0.2, 0.2, 0.2],
         salinity=4.20, ws=12.60, xs=6.93,
         Zw=0.3805, rho_w=1104.10, eps_r=55.42, mh=0.0393, mc=0.8314,
         Zh=0.9028, rho_h=790.29, xw_g=0.3896, xh_g=19.9221, xc_g=79.6883,
         f_h2o=0.1532, f_ch4=124.0502, f_co2=120.3425),
    dict(name="B: 350K 100bar YC=1 (pure CO2-H2O)",
         T=350, P=100, YC=1, mi=[0, 0, 0, 0, 0, 0],
         salinity=0.00, ws=0.00, xs=0.00,
         Zw=0.0649, rho_w=976.69, eps_r=58.80, mh=0.0000, mc=0.9063,
         Zh=0.6778, rho_h=222.05, xw_g=0.7754, xh_g=0.0000, xc_g=99.2246,
         f_h2o=0.4243, f_ch4=0.0000, f_co2=73.8011),
    dict(name="C: 350K 100bar YC=0 (pure CH4-H2O)",
         T=350, P=100, YC=0, mi=[0, 0, 0, 0, 0, 0],
         salinity=0.00, ws=0.00, xs=0.00,
         Zw=0.0638, rho_w=970.04, eps_r=61.06, mh=0.0710, mc=0.0000,
         Zh=0.9431, rho_h=58.34, xw_g=0.5091, xh_g=99.4909, xc_g=0.0000,
         f_h2o=0.4301, f_ch4=92.6764, f_co2=0.0000),
    dict(name="D: 400K 300bar YC=0.5 NaCl 2m",
         T=400, P=300, YC=0.5, mi=[2, 2, 0, 0, 0, 0],
         salinity=4.00, ws=10.28, xs=6.66,
         Zw=0.1936, rho_w=1009.43, eps_r=39.08, mh=0.0583, mc=0.4620,
         Zh=0.9236, rho_h=290.75, xw_g=1.9454, xh_g=49.0273, xc_g=49.0273,
         f_h2o=2.5516, f_ch4=151.5458, f_co2=100.6419),
    dict(name="E: 300K 600bar YC=1 CaCl2 1m",  # re-anchored 4f7fbf7
         T=300, P=600, YC=1, mi=[0, 2, 0, 1, 0, 0],
         salinity=3.00, ws=9.50, xs=5.01,
         Zw=0.4737, rho_w=1104.65, eps_r=65.31, mh=0.0000, mc=1.2933,
         Zh=0.9688, rho_h=1091.80, xw_g=0.1153, xh_g=0.0000, xc_g=99.8847,
         f_h2o=0.0497, f_ch4=0.0000, f_co2=124.4205),
]

# field -> (result attribute or callable, absolute tolerance)
FIELDS = {
    "salinity": (lambda r: r.salinity, 0.01),
    "ws": (lambda r: r.ws, 0.01),
    "xs": (lambda r: r.xs, 0.01),
    "Zw": (lambda r: r.Zw, 1e-3),
    "rho_w": (lambda r: r.rho_w, 0.1),
    "eps_r": (lambda r: r.eps_r, 0.02),
    "mh": (lambda r: r.mh, 1e-3),
    "mc": (lambda r: r.mc, 1e-3),
    "Zh": (lambda r: r.Zh, 1e-3),
    "rho_h": (lambda r: r.rho_h, 0.1),
    "xw_g": (lambda r: 100 * r.x1h, 0.02),
    "xh_g": (lambda r: 100 * r.x2h, 0.02),
    "xc_g": (lambda r: 100 * r.x3h, 0.02),
    "f_h2o": (lambda r: r.f_h2o, 0.02),
    "f_ch4": (lambda r: r.f_ch4, 0.02),
    "f_co2": (lambda r: r.f_co2, 0.02),
}


SOLVERS = ("fsolve", "newton")


def _check_case(case, solver):
    r = solve_elv(case["T"], case["P"], case["YC"], case["mi"], solver=solver)
    assert r.converged, f"{case['name']} [{solver}]: did not converge"
    failures = []
    for field, (getter, tol) in FIELDS.items():
        got = getter(r)
        exp = case[field]
        if abs(got - exp) > tol:
            failures.append(f"  {field}: got {got:.4f}, expected {exp:.4f} (tol {tol})")
    return failures


def test_regression():
    """pytest entry point: both solvers must reproduce the original."""
    for solver in SOLVERS:
        for case in CASES:
            failures = _check_case(case, solver)
            assert not failures, (
                f"{case['name']} [{solver}] mismatches:\n" + "\n".join(failures)
            )


def main():
    n_fail = 0
    for solver in SOLVERS:
        print(f"--- solver: {solver} ---")
        for case in CASES:
            failures = _check_case(case, solver)
            if failures:
                n_fail += 1
                print(f"FAIL  {case['name']}")
                print("\n".join(failures))
            else:
                print(f"PASS  {case['name']}")
        print()
    if n_fail:
        print(f"{n_fail} case/solver combinations FAILED")
        return 1
    print(f"All {len(CASES)} cases reproduce the original for both solvers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
