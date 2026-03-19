"""
_test_accelerated_ssi.py — Verification of accelerated SSI and stability test.

Compares standard vs accelerated SSI iteration counts, verifies stability test
phase identification, and checks that converged solutions match.
"""
import numpy as np
import CPA2

# ── Grid of (T, P) conditions ──────────────────────────────────────────────
T_grid = np.array([300, 325, 350, 375, 400, 425, 450, 475, 500])  # K
P_grid = np.array([10, 50, 100, 200, 300, 400, 500])               # bar
z_co2 = 0.01  # typical overall CO₂ mole fraction

comps = CPA2.make_components_co2_h2o()
Omega, Tc, Pc, Mw = comps["Omega"], comps["Tc"], comps["Pc"], comps["Mw"]

# ── Test 1: Standard vs Accelerated SSI iteration counts ───────────────────
print("=" * 90)
print("TEST 1: Standard SSI vs Accelerated SSI — iteration count comparison")
print("=" * 90)
header = f"{'T(K)':>6} {'P(bar)':>7} {'std_it':>7} {'acc_it':>7} {'ratio':>7} {'std_conv':>9} {'acc_conv':>9} {'x_CO2 match':>12}"
print(header)
print("-" * 90)

n_total = 0
n_faster = 0
n_both_conv = 0
total_std = 0
total_acc = 0

for T in T_grid:
    kij = CPA2.kij_ecpa(T)
    swc = CPA2.s14_ecpa(T)
    for P in P_grid:
        z = np.array([z_co2, 1.0 - z_co2])
        kw = dict(Omega=Omega, Tc=Tc, Pc=Pc, Mw=Mw, kij12=kij, swc=swc,
                  tol=1e-10, maxiter=1000)

        # Standard SSI
        tie_std = CPA2.tie_line_two_comp(T=T, P_bar=P, accelerated=False, **kw)
        # Accelerated SSI
        tie_acc = CPA2.tie_line_two_comp(T=T, P_bar=P, accelerated=True, **kw)

        it_std = tie_std["iterations"]
        it_acc = tie_acc["iterations"]
        c_std  = tie_std["converged"]
        c_acc  = tie_acc["converged"]

        ratio = f"{it_std/it_acc:.2f}" if it_acc > 0 else "N/A"

        # Check composition match
        match = "—"
        if c_std and c_acc:
            diff = abs(tie_std["x"][0] - tie_acc["x"][0])
            match = f"{diff:.1e}"
            n_both_conv += 1
            total_std += it_std
            total_acc += it_acc
            if it_acc < it_std:
                n_faster += 1

        n_total += 1
        print(f"{T:6.0f} {P:7.0f} {it_std:7d} {it_acc:7d} {ratio:>7} "
              f"{'Y' if c_std else 'N':>9} {'Y' if c_acc else 'N':>9} {match:>12}")

print("-" * 90)
if n_both_conv > 0:
    print(f"Both converged: {n_both_conv}/{n_total}")
    print(f"Accelerated faster: {n_faster}/{n_both_conv} "
          f"({100*n_faster/n_both_conv:.0f}%)")
    print(f"Average iterations — std: {total_std/n_both_conv:.1f}, "
          f"acc: {total_acc/n_both_conv:.1f} "
          f"(reduction: {100*(1 - total_acc/total_std):.0f}%)")

# ── Test 2: Stability test ─────────────────────────────────────────────────
print("\n" + "=" * 90)
print("TEST 2: Stability test — phase identification")
print("=" * 90)
header2 = f"{'T(K)':>6} {'P(bar)':>7} {'stable':>7} {'tpd_min':>12} {'flash_phase':>14} {'agree':>6}"
print(header2)
print("-" * 90)

n_agree = 0
n_stab_total = 0

for T in T_grid:
    kij = CPA2.kij_ecpa(T)
    swc = CPA2.s14_ecpa(T)
    for P in P_grid:
        z = np.array([z_co2, 1.0 - z_co2])

        # Stability test
        stab = CPA2.stability_test(
            T, P, z, Omega=Omega, Tc=Tc, Pc=Pc, Mw=Mw,
            kij12=kij, swc=swc, accelerated=True,
        )

        # Flash for ground truth
        flash = CPA2.flash_co2_h2o_tpz(T=T, P_bar=P, z_co2=z_co2,
                                         kij12=kij, swc=swc)
        flash_phase = flash.get("phase", "failed")
        flash_2ph = flash_phase == "two_phase"

        stab_2ph = not stab["stable"]
        agree = stab_2ph == flash_2ph

        n_stab_total += 1
        if agree:
            n_agree += 1

        print(f"{T:6.0f} {P:7.0f} {'N' if stab_2ph else 'Y':>7} "
              f"{stab['tpd_min']:12.4e} {flash_phase:>14} "
              f"{'Y' if agree else '*** N ***':>6}")

print("-" * 90)
print(f"Agreement: {n_agree}/{n_stab_total} ({100*n_agree/n_stab_total:.0f}%)")

# ── Test 3: Hierarchical flash — K from stability vs Wilson K ──────────────
print("\n" + "=" * 90)
print("TEST 3: Hierarchical flash — stability K vs Wilson K convergence")
print("=" * 90)
header3 = f"{'T(K)':>6} {'P(bar)':>7} {'rob_it':>8} {'wilson_it':>10} {'robust_conv':>12} {'phase':>14} {'x_CO2':>10}"
print(header3)
print("-" * 90)

for T in T_grid:
    kij = CPA2.kij_ecpa(T)
    swc = CPA2.s14_ecpa(T)
    for P in P_grid:
        robust = CPA2.flash_co2_h2o_tpz_robust(
            T=T, P_bar=P, z_co2=z_co2,
            kij12=kij, swc=swc, accelerated=True,
        )

        # Also do standard flash for comparison
        std = CPA2.flash_co2_h2o_tpz(T=T, P_bar=P, z_co2=z_co2,
                                       kij12=kij, swc=swc)

        rob_tie = robust.get("tie")
        rob_phase = robust.get("phase", "failed")

        if rob_phase == "single_phase":
            rob_it = 0; rob_conv = True
            x_co2 = "(single)"
        elif rob_tie is not None:
            rob_it = rob_tie["iterations"]
            rob_conv = rob_tie["converged"]
            x_co2 = f"{rob_tie['x'][0]:.6f}" if rob_conv else "—"
        else:
            rob_it = 0; rob_conv = False; x_co2 = "—"

        std_it  = std["tie"]["iterations"] if std.get("tie") else 0

        print(f"{T:6.0f} {P:7.0f} {rob_it:8d} {std_it:10d} "
              f"{'Y' if rob_conv else 'N':>12} {rob_phase:>14} {x_co2:>10}")

print("-" * 90)
print("\nDone.")
