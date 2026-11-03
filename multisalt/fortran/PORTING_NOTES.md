# PORTING_NOTES -- Python-side subtleties mirrored (or deliberately not)

Notes from transcribing `ms_ecpa` (Python/NumPy) to F2008.  Everything
below was load-bearing for reaching 1e-12 agreement; the last section
lists the three intentional deviations.

## 1. One complex kernel serves real evaluation + complex step

The Python inner Newton evaluates the residual twice per iteration
family: once with a real vector (line-search norm) and n times with a
complex-perturbed vector (Jacobian).  NumPy's real path yields NaN on
log/sqrt of negative arguments (under `errstate(ignore)`), which the
Python `residual_norm` maps to an infinite norm -- that is how the
line search avoids invalid regions.  The complex path, however, walks
straight through branch cuts and returns finite complex numbers.

The Fortran port uses ONE complex kernel for both.  Key facts that make
this exact:

* complex arithmetic with zero imaginary parts produces real parts
  bit-identical to real arithmetic (+, -, *, /, sqrt, exp, log, **),
  so the real evaluation is unchanged;
* with all-real inputs, a NONZERO imaginary part can appear only via a
  branch cut (log/sqrt/x**0.5 of a negative number) -- exactly the NaN
  set of the NumPy real path.  `residual_norm` therefore flags
  `aimag(f) /= 0` (or any non-finite component) as invalid, and the
  damped Newton makes the same accept/reject decisions as Python.

Never compile with -ffast-math / value-unsafe flags: NaN/Inf
propagation is part of the algorithm.

## 2. Sum ordering

NumPy `np.sum` uses pairwise summation only for >= 8 (blocked at 128)
elements; every reduction in these kernels is over 3 or 6 elements,
i.e. plain left-to-right -- matched by sequential Fortran loops /
array `sum()`.  Expression order was transcribed operator-for-operator
from lnphi.py (left-associative chains like `a*P/R**2/T**2` parse the
same in both languages), keeping agreement at the last-ulp level.

## 3. Seed heuristics use literal constants, not parameters

The aqueous cold seed is `Zw_liq = P*18.015e-6/(8.314*T)` with LITERAL
18.015e-6 and 8.314 in lnphi.py (not `parameters.R`); same in
`dew_point_flash`'s diagnostic seed.  The Fortran port reproduces the
literals (`zw_liq_seed`).  If a refit ever changed `R`, the seeds
must NOT change with it.

## 4. Two different aqueous fallback ladders

`lnphi_aq` has a 30-rung ladder (plus any rungs an extension adds
through `aq_seeds_hook`) whose 5th rung is `[0.10, 30, 0.05]`;
`flash_kv._ssi_step` carries its own 29-rung variant that inserts
`[0.05, 50, 0.10]` as rung 5 and omits the last two (25/0.12, 1.4x) rungs.  They
are ported as two separate tables (`aq_seed_ladder_lnphi`,
`aq_seed_ladder_flash`).  Also mirrored: on a warm-Newton failure
inside `_ssi_step`, Python first calls `lnphi_aq` (full ladder) for the
lnphi VALUES and then re-derives the warm-start vector with the flash
ladder -- the same double pass is kept, including the case where the
lnphi values come from a different root than the warm vector.

## 5. Convergence bookkeeping quirks kept verbatim

* `tpd_ssi` checks `max_g < tol` AFTER updating W, so the reported
  W/w/tpd are one SSI step past the converged residual.
* `flash_kv` counts degenerate-aqueous K-reset iterations against
  `maxiter` (the `continue` consumes an iteration) but does not
  advance `n_iter_ssi`.
* On the SSI-failure return path `n_iter_ssi = it` (0-based), not
  `it+1`.
* Jex acceleration turns off for exactly one step after a K-reset
  because `g_prev` is cleared -- no special counter.
* `wilson_k` operates in bar; the lnphi kernels in Pa.

## 6. Paper-I quadratic gas-ion DU shim

`paper1_config` swaps `A_U3i` for a `_QuadraticGasIonDU` object whose
`__mul__` intercepts the kernel's `A_U3i * (T/Tc3)` product.  The
Fortran side has no operator overloading tricks: the .par format
carries per-ion-slot form flags (`U3i_form`) plus quadratic
coefficients, and `u_gas_ion()` dispatches.  `gen_params.py` unpacks
the shim's internals when it detects the applied configuration.

## 7. Extension points

`ecpa_extensions.f90` holds four procedure pointers (aqueous terms,
extra seeds, colligative dielectric factors, colligative Helmholtz
term). All are null in this library, and every kernel then uses the
neutral constants (density factor 1, dipole weight 1, zero chains and
additions) in exactly the expressions it would evaluate without them;
verified bit-identical against the pre-extension build.

## 8. brentq is scipy's C algorithm, verbatim

`rachford_rice` (xtol=rtol=1e-12, maxiter=200) and the dew-point
rescue (xtol=1e-8, scipy default rtol=4*eps) go through a line-by-line
port of scipy/optimize/Zeros/brentq.c, including the
secant/inverse-quadratic selection rule and the delta-step fallback.

## 9. NaN-mask conventions for absent species

Absent species (z_i = 0) carry K = NaN through the flash exactly as in
NumPy (`np.where(present, ...)` patterns became explicit masked loops);
comparisons with NaN are false in both languages, which several guards
rely on (e.g. `ieee_is_finite(K(1)) .and. K(1) > 0`).

## 10. Intentional deviations (all measure-zero or diagnostic-only)

1. **Singular Newton Jacobian**: Python falls back to `lstsq` on an
   exactly singular J (LinAlgError); the Fortran Newton treats it as
   "no further progress".  Exact singularity is measure-zero and was
   never observed in the golden corpus.
2. **Newton-polish base evaluation failure**: Python lets the
   RuntimeError escape `flash_kv` entirely (callers see an exception);
   Fortran returns a result with `errcode = 5`.  In `stability_flash`
   both behave as a failed attempt; the only observable difference is
   which diagnostic `last_fr` lands in the hard-failure result.
3. **stability_flash `last_fr` on attempt exceptions**: Python sets
   `last_fr = None` when attempt 2 raises; Fortran keeps the
   error-coded result object.  Affects only the diagnostic flash
   attached to the (unconverged) hard-failure classification, never
   the phase decision or any converged quantity.

## 11. The chain convention loses the degenerate-aqueous K-reset

Under `salinity_convention="chain"`, `_ssi_step` calls
`compute_aqueous_x1w_chain` and raises `RuntimeError("chain closure
invalid ...")` whenever the helper returns a non-finite `x1w_true`.  The
degenerate-aqueous guard (`x1w_true < 0.02`) sits BELOW that raise, so a
transient SSI state with `x_part[0] <= 0` is reported as an invalid
chain closure -- which the SSI loop treats as fatal -- where the
`as-aqueous` branch would have reported `degenerate-aqueous:` and
recovered via the Wilson K-reset (up to five times).

The port mirrors this exactly (`ssi_step` returns `stat_ = 2`, which is
NOT in the recoverable branch).  Observed consequence: the chain route
gives up on a handful of water-poor / high-beta states that the default
convention rescues.  Two of the messages are also misleading, reporting
`m_tot_feed=0.0000` on salt-free feeds where `sigma_x` is identically
zero and the closure cannot in fact be invalid.

FLAGGED FOR THE OWNER, NOT FIXED: moving the degenerate-aqueous guard
above the chain branch (or letting the chain raise reuse the
`degenerate-aqueous:` prefix) would restore the K-reset under `chain`.
That is a behaviour change to the reference, so it belongs in
`ms_ecpa`, not here.

## 12. One knife-edge state where the two implementations disagree

`paper1`, T = 473.15 K, P = 400 bar, z = (0.95, 0, 0.05), salt-free:
the Python SSI drives `K_H2O -> NaN` and dies on the degenerate-aqueous
guard at iteration 11; the port's trajectory stays finite and converges
at SSI = 7 + 1 Newton polish.  The same bifurcation occurs under the
`paperII_III` defaults, and it is present under BOTH salinity
conventions, so it is a property of the shared aqueous kernel rather
than of any phase-2 work.

It is the only such state in 270 chain flash states x 3 parameter sets,
and it is not in the phase-1 golden grid (which is why phase 1 never
saw it).  `tools/test_fortran_chain.py` carries it on an explicit
`KNOWN_PREEXISTING_DIVERGENCES` allowlist, guarded by
`test_known_divergences_are_pre_existing`, which FAILS if a listed
state turns out to agree under `as-aqueous` (i.e. the allowlist can
only ever excuse a pre-existing bifurcation, never a phase-2 defect).

Not investigated further here: the aqueous lnphi kernel was under
concurrent repair (a chi-squared derivative-row swap and a spurious
`(delta - 1)` grouping) while this work was done, and either fix will
move last-ulp values near the NaN branch cut where this state sits.
Re-check the allowlist after that lands.

## 13. Colligative routes: three different Python sources

The three phase-2 colligative routes do NOT share one assembly, because
the reference does not:

* `phi_osm_total` <- `scripts/total_mole_basis.py::phi_osm_total`, the
  phase-4 "oracle": an INDEPENDENT total-mole-basis Helmholtz assembly
  that never calls the production lnphi kernels;
* `rho_brine` <- `scripts/fig_brine_density.py::predict_rho_brine`,
  which DOES use the production aqueous kernel plus the Peneloux ion
  shifts (this is the route behind the paper's density figure);
* `psat_brine` <- the vapor-pressure scripts' `psat_model`, water isofugacity between the production aqueous and
  gas kernels with Brent on `ln P`.

`psat_model` gets its Brent bracket from CoolProp's IAPWS saturation
line (`PropsSI("P", "T", T, "Q", 0, "Water")`).  The port does NOT
reimplement IAPWS: `p0_Pa` is an explicit argument, so a caller passing
the same anchor gets a bit-identical bracket and therefore the same
Brent trajectory.  The two variants differ only in the lower
bracket factor (0.50, and 0.55 in
`pilot_studies/vpsat_salt_2026_08/vpsat.py`); 0.50 is the default and
0.55 is reachable via `lo_fac`.

Also mirrored: `total_mole_basis` snapshots ion parameters into its own
`IP` namespace at import, so a parameter change requires an explicit
`reload_parameters()`.  The Fortran routine reads `par_t` directly, so
there is no snapshot to stale -- but the GENERATOR
(`tools/gen_golden_colligative.py`) must call `reload_parameters()`
after every configuration switch, and does.

The reference caches `ln f_w(pure water)` per `(T, P)`; the
port recomputes it.  That is a pure speed device with no effect on the
value.

## 14. rho_brine uses a DIFFERENT seed ladder again

`fig_brine_density._solve_Zw` has its own 1 + 6 rung ladder
(`[0.05, 50, 0.10]` then `[0.07,60,0.13]`, `[0.10,30,0.05]`,
`[0.30,70,0.20]`, `[0.20,40,0.10]`, `[0.50,50,0.10]`,
`[0.03,78,0.07]`).  That is a third table, distinct from both
`lnphi_aq`'s 30-rung ladder and `flash_kv._ssi_step`'s 29-rung variant
(note 4).  All three are ported separately; do not consolidate them.

## 15. phi_osm cannot reach 1e-12, by construction

Every `phi_osm_total` evaluation runs two Brent solves (permittivity,
volume) and TWO NESTED central differences: `dA/dV` inside `pressure`
(h = V*1e-7) and `dA/dn_w` inside `ln_f_w` (h = n_w*1e-7).  A central
difference with h ~ 1e-7 amplifies a last-ulp difference in `A` by
~1e7, and `phi` is then formed as a DIFFERENCE of two `ln f_w` values
(so the cancellation grows without bound as m -> 0).

Consequences, both deliberate:

* the gate uses 1e-9 relative for `phi_osm` (converged-quantity class)
  and 1e-12 for `rho_brine` (kernel class -- one inner Newton, then
  closed-form algebra);
* the golden molality grid starts at 0.1, not at the 1e-3
  Debye-Hueckel probe used in `total_mole_basis`'s own smoke test: at
  1e-3 the `ln f_w` difference is ~3e-5 in absolute terms, so ANY
  implementation difference at the 1e-14 level shows up at ~1e-9
  relative in phi and the comparison stops measuring the port.

`np.linspace` is reproduced exactly in `solve_v_liquid`'s bracket march
(including NumPy's "last element is forced to stop" rule), because the
marched bracket determines which root Brent lands on.

The 1e-7 figure is NOT a porting slip: `A_r` evaluated at a volume
handed to both sides is bit-identical on almost every state (worst
1.1e-16 relative), which is exactly why the gate carries a separate
`a_res` block at 1e-13.  The volume root itself agrees only to ~1e-9
relative for the same reason -- liquid water is stiff
(dP/dV ~ -1e14 Pa/m3) and the reference's `pressure` is a central
difference with a noise floor of order 1 Pa.

## 16. (removed)

## 17. FLAG FOR THE OWNER: psat_brine is ill-posed below ~0.1 bar

Porting `psat_model` surfaced a problem in the REFERENCE, not in the
port, and it is the most substantive thing this phase turned up.

`pilot_studies/vpsat_salt_2026_08/vpsat.py` sweeps
`T_GRID = np.arange(283.15, 574.0, 10.0)`.  At the bottom of that grid
the pure-water saturation pressure is 0.0123 bar, and down there
`lnphi_aq` is MULTIVALUED: the aqueous 3-var inner Newton lands on
different roots at neighbouring pressures, so the isofugacity residual
is not a function of P in any useful sense.

Measured with `ms_ecpa` ALONE (no Fortran involved), counting sign
changes of the residual over the Brent bracket at 41 pressures:

| T [K] | p0 [bar] | m=0 | m=1 | m=3 | m=6 |
|---|---|---|---|---|---|
| 283.15 | 0.01228 | 7 | 20 | 3 | 7 |
| 323.15 | 0.12352 | 1 | 1 | 1 | 1 |
| 373.15 | 1.01418 | 1 | 1 | 1 | 1 |
| 423.15 | 4.76165 | 1 | 1 | 1 | 1 |
| 473.15 | 15.54928 | 1 | 1 | 1 | 1 |
| 523.15 | 39.76175 | 1 | 1 | 1 | 1 |
| 573.15 | 85.87905 | 1 | 1 | 1 | 1 |

A well-posed bracket has exactly one sign change.  Every T >= 323.15 K
does.  283.15 K does not, and `scipy.optimize.brentq` there simply
returns whichever branch its sample points happened to straddle -- a
value with no reproducible meaning.  The port lands on a different
branch sequence (its lnphi_aq root selection differs in the last ulp)
and, at m = 3, finds no bracket at all and returns NaN.

Two consequences:

1. The psat golden grid starts at 323.15 K.  Those 24 states per
   parameter set agree to ~5e-15 relative.  The 283.15 K row is
   excluded because it is not a reproducible target for ANY
   reimplementation, not because the port cannot hit it.
2. **Any published 283.15 K point from these vpsat sweeps should be
   re-examined.** The relevant regime is roughly P_sat < 0.1 bar; the
   underlying cause is aqueous-kernel root selection at very low
   pressure, which is also 160x below the lowest pressure in the
   phase-1 golden grid (1 bar), so nothing else had probed it.

Not fixed here: widening `lnphi_aq`'s low-pressure root selection is a
change to `ms_ecpa`, and this port does not get to have opinions about
the reference.

### 17a. UPDATE 2026-09-14 — mechanism identified, Python side repaired

The "multivalued `lnphi_aq`" above has a specific mechanism. At low T and
low P the aqueous 3-var inner Newton stalls just above its `1e-10`
tolerance; `lnphi_aq` then falls through its 30-rung seed ladder and
returns the **vapour** root of the aqueous mixture. At such a pressure
both sides of the isofugacity balance are the same phase, so the
residual collapses to zero to machine precision (measured `|f|` from
9e-19 to 2.6e-12) with an arbitrary sign. Those are the extra "sign
changes" in the table above — they are not roots. On the physical liquid
branch the residual is smooth and monotone with `|df/d ln P| ~ 1` and
has exactly ONE root, at 283.15 K as everywhere else.

The Python side is now routed through `ms_ecpa/psat.py`
(`solve_psat_isofugacity`): it verifies the `[0.50 p0, 1.10 p0]` bracket
before trusting it, discards contaminated samples, falls back to an
adaptive scan, confirms the root is a real sign crossing, and raises
`PsatBracketError` rather than returning a bracket edge. Where the old
bracket was well posed — which is every state in the psat golden grid
(323.15–573.15 K) — it runs the identical `brentq` call and returns a
bit-identical root, so nothing here moves.

**The Fortran `psat_brine` is deliberately NOT updated.** It remains a
byte-faithful mirror of the OLD reference: fixed `lo_fac`/`hi_fac`, bare
Brent, NaN on no-sign-change. That is correct for the golden grid and
keeps the port's contract, but below ~300 K it still carries the
original defect and can return a bracket edge. Mirroring
`ms_ecpa.psat.solve_psat_isofugacity` is the open item if the port is
ever asked to serve that regime.

## 17b. OPEN ITEM: `golden_colligative_*.json` not yet regenerated

The 2026-09-14 gas-side chi guard regenerated `golden_*.json` and
`golden_chain_*.json` for all three parameter sets. The three
`golden_colligative_*.json` files were **not** regenerated in that pass
and still carry their 2026-09-13 content. The colligative gate passes
against them (16/16) with the rebuilt library, and there is a physical
reason to expect them to be unaffected: the colligative routes are
gas-free except `psat`, which calls `lnphi_gas` only at pure steam
`y = (1, 0, 0)` over 323.15–573.15 K, where the guard cannot fire. That
is an expectation, not a measurement. To close it:

```bash
cd multisalt
PYTHONPATH=. python3 fortran/tools/gen_golden_colligative.py   # ~30 min
python3 -m pytest fortran/tools/test_fortran_colligative.py -q
```

## 18. Golden vectors are tied to a reference REVISION

The 2026-09-13 repair of the chi-chain derivatives in
`ms_ecpa/lnphi.py` and `ms_ecpa/eos.py` (the chi-squared row swap and
the spurious `(delta - 1)` grouping) changed `lnphi_aq` VALUES, not
just Jacobian entries, so it moved every golden vector that goes
through the aqueous kernel.

Which phase-2 blocks are affected:

* `golden_chain_*.json` -- yes, entirely (flash + stability).
* `golden_colligative_*.json`, `rho_brine` and `psat` blocks -- yes.
* `golden_colligative_*.json`, `a_res` / `ln_f_w` / `phi_osm` blocks --
  NO.  The total-mole oracle in `scripts/total_mole_basis.py` is an
  independent assembly that never calls the lnphi kernels, so it is
  insensitive to the repair.  (Verified: those blocks reproduce
  unchanged across the repair.)

Practical rule: regenerate the phase-1 AND phase-2 golden sets together
after any change to `ms_ecpa`, and never mix a golden file from one
reference revision with a library built from another.  The chain gate
will fail loudly if you do -- which is how the staleness was caught.

## 19. Warm-start extension (beyond the Python reference)

The caller-owned `ecpa_warm_t` plumbing (seeding SSI K + both inner
Newtons, and the stability route-10 direct warm flash) has NO Python
counterpart (Python can only seed the gas side via `gas_Z_init`).  It
is strictly opt-in: NULL/valid=0 reproduces Python behaviour, which is
what the golden gate certifies.
