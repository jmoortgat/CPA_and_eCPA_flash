# fortran/ -- Fortran port of the multi-salt eCPA library (phase 1)

Modern-Fortran (F2008) port of `ms_ecpa`, the ion-specific electrolyte
CPA library for H2O-CH4-CO2-(Na, Cl, K, Ca, Mg, SO4) phase equilibria.
The Python library in `code/ms_ecpa/` is the reference implementation;
this port reproduces it to floating-point noise (see Verification) and
is the evaluation backbone for (a) reservoir-simulator integration and
(b) future refit campaigns driving the engine through the C API.

## Layout

```
fortran/
  src/               F2008 modules (no vendor extensions)
    ecpa_kinds.f90      precision kinds
    ecpa_params.f90     parameter store + name table + .par reader
    ecpa_water.f90      pure-water auxiliary correlations
    ecpa_extensions.f90 extension points (procedure pointers; null = off)
    ecpa_newton.f90     damped Newton, complex-step Jacobian (h=1e-30)
    ecpa_lnphi.f90      lnphi kernels: aqueous 3-var (Z, eps_r, chi_H2O)
                        inner Newton; gas 2-var (Z, chi_H2O)
    ecpa_brent.f90      Brent root finder (scipy brentq port)
    ecpa_flash.f90      Rachford-Rice + K-value flash (accelerated SSI
                        + Newton polish) + dew-point rescue
    ecpa_stability.f90  Michelsen TPD + combined stability_flash driver
    ecpa_colligative.f90 phase-2 colligative routes: total-mole-basis
                        osmotic coefficient (the phase-4 'oracle'),
                        brine density, brine vapour pressure
    ecpa_capi.f90       iso_c_binding API (opaque context, warm struct)
  params/            runtime parameter files (generated, do not hand-edit)
    paperII_III.par     pristine library defaults (Papers II/III)
    paper1.par          Paper-I CO2-H2O-NaCl configuration
  golden/            Python-generated golden verification vectors
  tools/
    gen_params.py       regenerates params/*.par from ms_ecpa
    gen_golden.py       regenerates golden/*.json from ms_ecpa (phase 1)
    gen_golden_chain.py       golden vectors for salinity "chain"
    gen_golden_colligative.py golden vectors for the colligative routes
    ecpa_ctypes.py      minimal ctypes wrapper (verification + example)
    test_fortran_port.py        pytest gate, phase 1 ("as-aqueous")
    test_fortran_chain.py       pytest gate, salinity "chain"
    test_fortran_colligative.py pytest gate, colligative routes
    bench.py            Fortran-vs-Python timing comparison
    bench_fortran.py    single-variant timing (pick with ECPA_LIB)
  Makefile           gfortran default; make FC=ifx for Intel oneAPI
  PORTING_NOTES.md   Python-side subtleties that were hard to mirror
```

## Build

```bash
make -C fortran                          # gfortran 14 (default)
make -C fortran FC=ifx                   # Intel oneAPI 2024.2, portable
make -C fortran FC=ifx ARCH=sapphire     # + -xSAPPHIRERAPIDS (Sapphire Rapids nodes only)
```

Produces `build/libecpa.dylib` (macOS) or `build/libecpa.so` (Linux).
Never add fast-math flags: the solvers rely on IEEE NaN/Inf propagation
mirroring NumPy semantics.

### Intel build -- what is shipped and why

Shipped: **ifx** (oneAPI 2024.2.0) at **`-O3 -stand f08 -fpp -fPIC
-fp-model strict`**, portable and `ARCH=sapphire` variants.

`ifort` 2021.13 also works, but it announces its own
deprecation (remark #10448) and is the worse choice here: with
`-fp-model strict` it runs 11-13 % slower than gfortran, while ifx pays
nothing for `strict` (ifx's `precise` already disables FMA
contraction, so `strict` only adds exception semantics).

Floating point: all 15 compiler/flag variants were built and put
through the full acceptance gate on a Sapphire Rapids compute node of an
HPC cluster.
Intel's default `-fp-model fast` PASSED, but it does change the
arithmetic (every reported deviation moves) and it assumes finite math,
which this port cannot afford -- so the strictest setting that passes
is the one shipped. On ifx, `strict` is bit-identical to `precise` AND
to the `-xSAPPHIRERAPIDS` build, so the portable and CPU-targeted
libraries return the same numbers; on ifort that is not true.

**MKL: not linked, and it should not be.** The library calls no BLAS
and no LAPACK. The 3x3 aqueous and 2x2 gas inner solves use a
hand-rolled LU with partial pivoting (`ecpa_newton.f90`, a dgesv
equivalent for a tiny fixed size), and the only matrix expression in
the codebase is one 6x6 `MATMUL` in the ion-ion Huron-Vidal term, which
every compiler inlines. Evidence: the complete `nm -D --undefined-only`
list for the gnu, ifort and ifx builds is libm + libc + compiler
runtime -- no `dgemm_`, `dgesv_`, `dgetrf_` or `mkl_*` symbol of any
kind. Linking `-qmkl` would add a dependency that buys nothing. Should
that ever change, use `-qmkl=sequential`: the library is driven from
multiprocessing workers, where a threaded BLAS fights the pool.

To check a build, run the acceptance tests against it:

```bash
ECPA_LIB=fortran/build/libecpa.so python3 -m pytest fortran/tools
```

## Parameter files -- design

Plain-text `name = value [value ...]` (# comments), one parameter per
line; ion arrays are 6 values in the fixed order Na Cl K Ca Mg SO4;
`Uref_ii` is 36 values row-major; switches are 0/1 entries.
Chosen over Fortran namelist because (i) the same name list is shared
with the programmatic C-API setter -- one authoritative namespace, no
namelist-formatting quirks across compilers; (ii) files are generated
and diffed on the Python side, so a dumb line format with full-repr
floats (shortest round-trip decimal = bit-exact float64) is the most
transparent contract.  The reader is STRICT: unknown names or wrong
value counts abort with the offending line number.

Everything the kernels evaluate comes from the file: rounded universal
constants (R = 8.314, Na = 6.02e23, ... -- they are part of the model),
final pure-component CPA values, ion parameters, gas-ion Huron-Vidal
energies with a per-slot functional-form flag (`U3i_form` = 1 selects
the Paper-I quadratic DU form on that ion slot), Peneloux ion shifts,
Wilson
omegas.  Refits therefore never touch the Fortran core: regenerate a
.par file (or push values through `ecpa_set_param`) and run.

Parameters owned by an extension layer (src/ecpa_extensions.f90) are
accepted by the same reader and setter once that layer has installed its
name table (ecpa_params::ext_param_lookup); they live in `par%ext`.

## C API (iso_c_binding)

Opaque context; all state a caller can vary lives either in the context
(parameters) or in caller-owned per-call structs (warm start).  One
context per worker/thread; contexts are independent.

```c
void*  ecpa_create(void);
void   ecpa_destroy(void* ctx);
int    ecpa_load_params(void* ctx, const char* path);
int    ecpa_set_param(void* ctx, const char* name, const double* v, int n);
int    ecpa_get_param(void* ctx, const char* name, double* v, int nmax);
int    ecpa_lnphi_aq (void* ctx, const double x_aq[3], const double mi[6],
                      double T, double P_Pa, const double v_init[3],
                      double lnphi[3], double v_out[3], int* niter);
int    ecpa_lnphi_gas(void* ctx, const double y[3], double T, double P_Pa,
                      double Z_init, double chi_init,
                      double lnphi[3], double v_out[2]);
int    ecpa_flash_kv (void* ctx, double T, double P_bar, const double z[3],
                      const double mi[6], int salinity_convention,
                      const ecpa_warm_t* warm_in, ecpa_flash_out_t* out,
                      ecpa_warm_t* warm_out);
int    ecpa_stability_flash(void* ctx, double T, double P_bar,
                      const double z[3], const double mi[6],
                      int salinity_convention, const ecpa_warm_t* warm_in,
                      int* phases, double* tpd_min, int* route,
                      int* has_flash, ecpa_flash_out_t* out,
                      ecpa_warm_t* warm_out);
```

`ecpa_warm_t` = {K[3], Z_aq, Z_gas, eps_r, chi_H2O_aq, chi_H2O_gas,
beta, valid} -- exactly the warm-start tuple `FlashKVResult` exposes.
Pass NULL / valid=0 for a cold start (bit-faithful Python behaviour;
what the golden gate exercises).  With a valid warm struct, flash_kv
seeds the SSI K vector and both inner Newtons from it, and
stability_flash first tries a direct warm flash before the cold ladder.
Converged state is always written back to `warm_out` when non-NULL, so
a reservoir simulator can store it per cell for the next time step.

Phase-2 additions (colligative routes -- see `ecpa_colligative.f90` for
which Python function each one reproduces):

```c
int    ecpa_a_res    (void* ctx, const double n[7], double V, double T,
                      double* A);
int    ecpa_ln_f_w   (void* ctx, const double n[7], double T, double P_Pa,
                      double* lnf, double* V);
int    ecpa_phi_osm  (void* ctx, int salt, double m, double T, double P_Pa,
                      double* phi);
int    ecpa_rho_brine(void* ctx, double T, double P_bar, const double mi[6],
                      const double v_init[3], double* rho, double v_out[3]);
int    ecpa_psat_brine(void* ctx, double T, double m, double p0_Pa,
                      double lo_fac, double hi_fac, double* P_bar);
```

`salt` = 1 NaCl, 2 KCl, 3 CaCl2, 4 MgCl2, 5 Na2SO4. `n` is the
total-mole 7-vector `[n_w, n_Na, n_Cl, n_K, n_Ca, n_Mg, n_SO4]` for 1 kg
of water (`n_w = 55.5084`). `p0_Pa` is the pure-water saturation pressure the
Python reference reads from CoolProp and uses ONLY to anchor the Brent
bracket -- the port takes it as an argument rather than reimplementing
IAPWS, so passing the same value gives a bit-identical bracket.

`salinity_convention`: 0 = "as-aqueous", 1 = "chain". **0 is the
default and stays the default**, exactly as in `ms_ecpa.flash.flash_kv`
-- `"as-aqueous"` is the convention the published validation uses, and
the port has no opinion of its own about that. `"chain"` is opt-in
(`m_aq* = m_prep * z_w / [(1-beta) * x_w_aq]`, applied per ion at every
SSI step); any other value returns error 101.

Units: `lnphi_*` take P in Pa (SI, like the Python kernels);
`flash_kv` / `stability_flash` take P in bar (like the Python drivers).

Python usage: see `tools/ecpa_ctypes.py` (class `Engine`).

## Numerics -- derivative choice

Same as the Python reference: complex-step Jacobians with h = 1e-30.
The residual kernels are written once in `complex(real64)` and serve
both the real evaluation (zero imaginary parts propagate exactly; the
real parts are bit-identical to a real-arithmetic implementation) and
the perturbed evaluations.  Invalid-region detection: with all-real
inputs, a nonzero imaginary part can only come from a log/sqrt branch
cut -- exactly where NumPy's real path yields NaN -- and is mapped to
an infinite residual norm, so the damped line search makes the same
accept/reject decisions as the Python solver.  All solver tolerances
match the Python values exactly (inner Newton 1e-10/80 iters, flash SSI
1e-8/80, Newton-polish handoff 1e-4/5, TPD 1e-8/80, Rachford-Rice
brentq 1e-12/1e-12).

## Verification

```bash
make -C fortran
python3 fortran/tools/gen_params.py
python3 fortran/tools/gen_golden.py             # ~10 min (phase 1)
python3 fortran/tools/gen_golden_chain.py       # ~4 min  (phase 2)
python3 fortran/tools/gen_golden_colligative.py # ~30 min (phase 2)
# (of which ~25 min is the 600-state rho_brine grid; the total-mole
#  oracle blocks take about 3 s)
python3 -m pytest fortran/tools/test_fortran_port.py \
                 fortran/tools/test_fortran_chain.py \
                 fortran/tools/test_fortran_colligative.py -v -s
```

The gate compares, for each of the three parameter files: lnphi_aq /
lnphi_gas components + (Z, eps_r, chi) at ~200 states (1e-12 relative),
full flash_kv results at ~100 states (1e-9 on converged quantities,
iteration counts +/- 2), and stability_flash phase decisions at ~50
states (phases exact, tpd_min 1e-6).  Convergence FAILURES must match
too (Python raise <-> Fortran error code).  The phase-2 gates add the
same discipline for `salinity_convention="chain"` and the colligative
routes; `test_fortran_colligative.py` documents why its tolerances
differ per block.

An auxiliary tool:

```bash
python3 fortran/tools/regen_psat_block.py    # psat block only (fast)
```

`tools/bench.py` prints the measured single-thread speedups.

### Verified state (2026-09-12, gfortran 14.2, Apple M-series)

All 13 gate tests pass for every parameter file.  Max deviations
vs the Python golden vectors:

| block | paperII_III | paper1 |
|---|---|---|
| lnphi_aq (126 states each) | 2.7e-15 | 5.4e-15 |
| lnphi_gas (84 states each) | 1.8e-15 | 2.9e-15 |
| flash_kv (90/60/90 states, rel) | 1.7e-13 | 1.2e-13 |
| stability tpd_min (50/20/50, abs) | 2.6e-13 | 4.4e-15 |
| phase decisions / failure cases | all matched | all matched |

SSI/Newton iteration counts matched EXACTLY on every converged flash
(the +/- 2 test slack was never used).

### Verified state -- phase 2 (2026-09-13, gfortran 14.2, macOS)

Against the REPAIRED `ms_ecpa` (the 2026-09-13 chi-chain derivative
fix; see PORTING_NOTES.md 18 on why golden vectors are tied to a
reference revision).

`salinity_convention="chain"` -- 11 gate tests pass:

| block | paperII_III | paper1 |
|---|---|---|
| flash_kv incl. mi_aq (90 states each, rel) | 9.2e-14 | 1.4e-13 |
| stability flash (32/12/32 states, rel) | 8.1e-14 | 5.4e-14 |
| stability tpd_min (abs) | 1.1e-13 | 4.4e-15 |
| converged / invalid-closure states | 78 / 12 | 79 / 10 |

Every invalid-chain-closure state is reported unconverged by both
implementations, and SSI/Newton counts match exactly.  One knife-edge
paper1 state is allowlisted (PORTING_NOTES.md 12).

Colligative routes -- 16 gate tests pass:

| block | paperII_III | paper1 |
|---|---|---|
| `a_res` (420/84/420 values, rel) | 2.1e-16 | 1.9e-16 |
| `solve_V` (rel) | 4.5e-9 | 3.3e-9 |
| `ln_f_w` (abs) | 1.6e-7 | 2.4e-8 |
| `phi_osm` (90/18/90 states, ABS) | 9.5e-6 | 8.6e-6 |
| `rho_brine` (484/97/470 states, rel) | 2.9e-14 | 2.7e-15 |
| `psat_brine` (24 states each, rel) | 3.7e-15 | 3.7e-15 |

Read that table downwards: the total-mole assembly is reproduced at the
last ulp, and everything above `a_res` is the reference's OWN numerical
differentiation -- two nested central differences and a Brent solve on
a numerically-differentiated pressure -- degrading it.  `phi_osm` is
gated on the absolute deviation because its relative error is
unbounded wherever phi itself is small.

### Cross-compiler verification (Intel Xeon Platinum 8468, Sapphire Rapids)

All three gates (13 phase-1 + 13 chain + 16 colligative = 40 passed,
2 skipped) pass on a Sapphire Rapids compute node for
`build-gnu-O3`, `build-gnu-O3-sap`,
`build-ifx-strict`, `build-ifx-strict-sap`, `build-ifx-prec-sap` and
`build-ifort-strict-sap`.

### Measured performance (Sapphire Rapids node, single thread,
best-of-7 round-robin over interleaved variants, via ctypes)

| variant | lnphi_aq | lnphi_gas | flash_kv | stability_flash |
|---|---:|---:|---:|---:|
| gfortran -O3 | 27,493 | 175,283 | 519 | 400 |
| gfortran -O3 -march=sapphirerapids | 29,175 | 181,098 | 566 | 424 |
| ifort -fp-model precise | 28,345 | 178,320 | 545 | 408 |
| ifort -fp-model precise -x | 29,419 | 177,796 | 565 | 420 |
| ifort -fp-model strict -x | 24,433 | 171,869 | 450 | 350 |
| ifx -fp-model precise | 27,235 | 188,282 | 522 | 414 |
| ifx -fp-model precise -x | 28,801 | 187,558 | 555 | 435 |
| **ifx -fp-model strict -x (shipped)** | **27,304** | **187,294** | **519** | **408** |
| ifx (default fast) -x | 30,700 | 207,075 | 546 | 449 |

evals/s; `-x` = `-xSAPPHIRERAPIDS`.  Python reference on the same node:
201, 3,145, 3.7, 3.0 evals/s -- i.e. 130-140x.

The honest reading: apart from `ifort -fp-model strict` (11-13 % slow)
and Intel's unsafe default `fast` (5-18 % quick), every combination
lands within ~9 % of gfortran.  This code is a dependent chain of
scalar work and transcendental calls, so there is little for a
vectorising backend to win.  `ifx -fp-model strict` was chosen for
reproducibility, not speed -- it costs nothing versus `precise` and
returns identical numbers from the portable and CPU-targeted builds.

### Measured performance (same machine, single thread, via ctypes --
i.e. including Python call overhead on the Fortran side)

| kernel | Python | Fortran | speedup |
|---|---|---|---|
| lnphi_aq | 3.02 ms (331 /s) | 0.015 ms (65,300 /s) | 197x |
| lnphi_gas | 0.167 ms (6,000 /s) | 0.003 ms (335,000 /s) | 56x* |
| flash_kv | 163 ms (6.1 /s) | 0.76 ms (1,320 /s) | 215x |
| stability_flash | 203 ms (4.9 /s) | 1.10 ms (909 /s) | 185x |

*lnphi_gas is dominated by the ~2-3 us ctypes crossing; the in-Fortran
kernel cost is well below that, so native callers see more.

## Roadmap

Phase 2 delivered `salinity_convention="chain"` and the three
colligative routes (see above). Still open, in intended order:

1. `solve_elv` route (`eos.py::solve_elv` / `residual_elv` +
   `guess.py::build_initial_guess`) -- needed for the CH4/CO2 refit
   predictor path (the refit gas blocks call `solve_elv`, not
   `flash_kv`). NOT started, and it is much heavier than its position
   in this list suggests; two hard dependencies have to be paid first:
   * **the default solver is `scipy.optimize.fsolve`**, i.e. MINPACK
     `hybrd`. A faithful port has to reproduce `hybrd` (dogleg trust
     region, `qrfac`/`qform`/`r1updt`/`r1mpyq` Broyden update,
     `fdjac1`, `enorm`) including scipy's defaults. `solver="newton"`
     exists and would reuse the already-ported damped Newton, but it is
     NOT the default and so is not what a faithful port may ship.
   * **the initial guess comes from 43 MB of binary CO2-H2O / CH4-H2O
     tables** (`data/guess_tables/{CO2,CH4}.parquet`, 251 integer
     temperatures x N pressures x 10 columns). `build_initial_guess`
     nearest-neighbour looks up both, then interleaves and
     YC-weights them into a 10- or 12-variable vector. The Fortran
     side needs those tables in a readable format (a raw float64 dump
     plus an index is the obvious choice) and the same
     nearest-neighbour tie-breaking as `np.argmin`.
2. Three-phase (V + Laq + L_CO2) stability/flash (flash3.py).
3. Simulator integration of the warm-start plumbing: use the previous
   time-step state to SKIP/shorten the TPD stage (previous two-phase
   cell -> direct flash from previous K, stability test only on
   suspicion of a phase-state change).  The per-call warm struct and
   the route-10 warm attempt in stability_flash are the hooks.
4. Density reporting for the FLASH phases (Peneloux-translated
   rho_aq/rho_gas of a two-phase result, incl. the Paper-I H2O
   polynomial). Distinct from the phase-2 `rho_brine` route, which is
   the CO2-free brine density behind the paper figure.

Refit harnesses can already run a HYBRID objective today: the Fortran
engine for the expensive flash/solubility rows through the
`ecpa_set_param` + evaluate C API, and -- now -- the Fortran
colligative routes for the ~150 colligative rows as well. Only the
`solve_elv` gas blocks still have to go through Python.
