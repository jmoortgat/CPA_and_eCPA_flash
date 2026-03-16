"""
Utilities: flash result reporting, percent difference, tie-line validation.
"""
import numpy as np


def pct_diff(a: float, b: float) -> float:
    """Symmetric percent difference: 100·(a−b) / (0.5·(|a|+|b|))."""
    denom = 0.5 * (abs(a) + abs(b))
    return 100.0 * (a - b) / denom if denom > 1e-30 else 0.0


def print_flash_report(out: dict) -> None:
    """
    Print a formatted summary of a flash result dict, including:
      - phase split (beta, N_aq, N_c)
      - aqueous-phase composition, density, saturation
      - CO₂-rich-phase composition, density, saturation
      - mass balance check
    """
    T     = float(out["T"])
    P_bar = float(out["P_bar"])
    z_co2 = float(out["z_co2"])
    m_tot = float(out["m_tot"])
    ms_aq = float(out["ms_aq"])
    N_aq  = float(out["N_aq"])
    N_c   = float(out["N_c"])
    beta  = float(out["beta"])
    x_aq  = out["x_aq"]
    x_c   = out["x_c"]
    sol   = out["sol"]
    Z_aq  = float(sol[0])
    Z_c   = float(sol[3])

    n_co2_tot  = z_co2
    n_h2o_tot  = 1.0 - z_co2
    n_salt_tot = m_tot * n_h2o_tot * 0.018  # Mw hard-coded for standalone use

    # Molecular weights [kg/mol]
    M_H2O = 18.01528e-3;  M_NA = 22.98977e-3
    M_CL  = 35.453e-3;    M_CO2 = 44.0095e-3

    Mbar_aq = (x_aq["x1w"]*M_H2O + x_aq["x2w"]*M_NA +
               x_aq["x3w"]*M_CL  + x_aq["x4w"]*M_CO2)
    Mbar_c  =  x_c["x1c"]*M_H2O  + x_c["x4c"]*M_CO2

    R_val  = 8.314462618
    P_Pa   = P_bar * 1e5
    rho_mol_aq = P_Pa / (Z_aq * R_val * T)
    rho_mol_c  = P_Pa / (Z_c  * R_val * T)
    rho_mass_aq = rho_mol_aq * Mbar_aq
    rho_mass_c  = rho_mol_c  * Mbar_c

    V_aq  = N_aq / rho_mol_aq
    V_c   = N_c  / rho_mol_c
    V_tot = V_aq + V_c
    S_aq  = V_aq / V_tot
    S_c   = V_c  / V_tot
    m_aq  = N_aq * Mbar_aq
    m_c   = N_c  * Mbar_c

    n_h2o_a = N_aq * x_aq["x1w"];  n_co2_a = N_aq * x_aq["x4w"]
    n_na_a  = N_aq * x_aq["x2w"];  n_cl_a  = N_aq * x_aq["x3w"]
    n_h2o_c = N_c  * x_c["x1c"];   n_co2_c = N_c  * x_c["x4c"]

    def pct(x): return f"{100*x:6.2f}%"
    def mol(x): return f"{x:.4f} mol"
    def sci(x): return f"{x:.4e}"

    print("\n================ FLASH RESULT ================\n")
    print(f"  T = {T:.2f} K          P = {P_bar:.2f} bar")
    print(f"  z_CO2 (feed)  = {pct(z_co2)}    m_tot = {m_tot:.4f} mol/kg")
    print(f"  n_salt (basis)= {sci(n_salt_tot)} mol")

    print("\n-------- Phase Split --------")
    print(f"  beta  (CO2-rich fraction) = {pct(beta)}")
    print(f"  N_aq  = {sci(N_aq)} mol     N_c = {sci(N_c)} mol")

    print("\n-------- Aqueous Phase --------")
    print(f"  ms_aq = {ms_aq:.4f} mol/kg    Z_aq = {Z_aq:.6f}")
    print(f"  rho   = {rho_mass_aq:.1f} kg/m³    ({sci(rho_mol_aq)} mol/m³)")
    print(f"  V_aq  = {sci(V_aq)} m³    S_aq = {pct(S_aq)}")
    print(f"  x_H2O = {pct(x_aq['x1w'])}   x_CO2 = {pct(x_aq['x4w'])}")
    print(f"  x_Na+ = {pct(x_aq['x2w'])}   x_Cl- = {pct(x_aq['x3w'])}")
    print(f"  n_H2O = {mol(n_h2o_a)}   n_CO2 = {mol(n_co2_a)}")
    print(f"  n_Na+ = {mol(n_na_a)}   n_Cl- = {mol(n_cl_a)}")

    print("\n-------- CO2-Rich Phase --------")
    print(f"  Z_c   = {Z_c:.6f}")
    print(f"  rho   = {rho_mass_c:.1f} kg/m³    ({sci(rho_mol_c)} mol/m³)")
    print(f"  V_c   = {sci(V_c)} m³    S_c  = {pct(S_c)}")
    print(f"  x_H2O = {pct(x_c['x1c'])}   x_CO2 = {pct(x_c['x4c'])}")
    print(f"  n_H2O = {mol(n_h2o_c)}   n_CO2 = {mol(n_co2_c)}")

    print("\n-------- Mass Balance --------")
    print(f"  H2O : {sci(n_h2o_a+n_h2o_c)} mol  (target {sci(n_h2o_tot)})")
    print(f"  CO2 : {sci(n_co2_a+n_co2_c)} mol  (target {sci(n_co2_tot)})")
    print(f"  salt: {sci(n_na_a)}  mol  (target {sci(n_salt_tot)})")

    print("\n-------- Totals --------")
    print(f"  mass: {sci(m_aq)} kg (aq) + {sci(m_c)} kg (c) = {sci(m_aq+m_c)} kg")
    print(f"  vol : {sci(V_aq)} m³ (aq) + {sci(V_c)} m³ (c) = {sci(V_tot)} m³")
    print("\n=============================================\n")


def test_tieline_invariance(T, P_bar, z_co2_values, m_tot,
                             guess_table_fn, params,
                             flash_fn=None,
                             verbose=False):
    """
    Tie-line invariance test: flash a range of feed compositions at fixed T, P.

    At fixed T and P the equilibrium phase compositions must be independent
    of z_co2.  Only beta (phase split) should vary.

    Parameters
    ----------
    flash_fn : callable or None
        Flash function to use.  Defaults to flash_co2_h2o_salt_ssi.

    Returns
    -------
    list of dicts with keys: z_co2, z_h2o, beta, x1w, x4w, ms_aq,
                              x1c, x4c, converged, status
    """
    if flash_fn is None:
        from .flash import flash_co2_h2o_salt_ssi
        flash_fn = flash_co2_h2o_salt_ssi

    print(f"Tie-line invariance test: T={T} K, P={P_bar} bar, m_tot={m_tot} mol/kg")
    print(f"Varying z_co2 over {len(z_co2_values)} feed compositions\n")
    print(f"{'z_co2':>7s} {'z_H2O':>7s} {'beta':>7s} "
          f"{'x_H2O_aq':>10s} {'x_CO2_aq':>10s} {'ms_aq':>8s} "
          f"{'x_H2O_c':>9s} {'x_CO2_c':>9s} {'status':>22s}")
    print("-" * 103)

    results = []
    for z_co2 in z_co2_values:
        try:
            out = flash_fn(T=T, P_bar=P_bar, z_co2=float(z_co2), m_tot=m_tot,
                           guess_table_fn=guess_table_fn, params=params)
            converged = True
            x1w  = out["x_aq"]["x1w"];  x4w  = out["x_aq"]["x4w"]
            x1c  = out["x_c"]["x1c"];   x4c  = out["x_c"]["x4c"]
            ms_i = out["ms_aq"];         beta = out["beta"]
            hint = "converged"
        except Exception as exc:
            converged = False
            x1w = x4w = x1c = x4c = ms_i = beta = np.nan
            hint = ("single_phase_liquid" if "liquid" in str(exc) else
                    "single_phase_gas"    if "gas"    in str(exc) else "failed")
            if verbose:
                print(f"  z_co2={z_co2:.3f} failed: {exc}")

        results.append(dict(z_co2=z_co2, z_h2o=1-z_co2, beta=beta,
                            x1w=x1w, x4w=x4w, ms_aq=ms_i,
                            x1c=x1c, x4c=x4c, converged=converged, status=hint))

        if np.isfinite(beta):
            print(f"{z_co2:7.4f} {1-z_co2:7.4f} {beta:7.4f} ", end="")
        else:
            print(f"{z_co2:7.4f} {1-z_co2:7.4f} {'---':>7s} ", end="")

        if converged:
            print(f"{x1w:10.6f} {x4w:10.6f} {ms_i:8.4f} {x1c:9.6f} {x4c:9.6f} "
                  f"{hint:>22s}")
        else:
            print(f"{'---':>10s} {'---':>10s} {'---':>8s} {'---':>9s} {'---':>9s} "
                  f"{hint:>22s}")

    return results
