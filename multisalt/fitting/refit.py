"""The ms-eCPA refit: objective, least-squares harness, scoring, Pareto scan.

Objective
---------
For row i of family f the residual is

    r_i = w_f * (pred_i - obs_i) / sigma_i

* ``sigma_i`` is the family's measured INTER-SOURCE REPRODUCIBILITY FLOOR --
  the median disagreement between different laboratories at the same state,
  computed from GasBrineBench itself (see fitting/dataset.SIGMA). Weighting
  by 1/sigma^2 then says "do not fit tighter than the data can distinguish",
  which is an estimate rather than a preference, and is correspondingly hard
  to argue with in review.
* ``w_f = sqrt(B_f / N_f)`` decouples a family's pull on the fit from how
  many rows it happens to have. ``B_f`` is the family budget -- and the
  y_H2O budget is the Pareto knob (see :func:`pareto_scan`).
* A row the model cannot converge on contributes ``PENALTY`` sigma rather
  than being dropped, so a parameter set that stops converging is punished
  instead of being quietly scored on fewer points.

Why a budget at all, rather than pure 1/sigma^2
-----------------------------------------------
Pure 1/sigma^2 is maximum likelihood *for a correctly specified model*.
This model is misspecified -- it cannot fit every property to the noise
floor at once -- so the fit is a Pareto trade-off, and 1/sigma^2 silently
resolves it in favour of whichever family has the most rows. With 4,520 CO2
solubility rows against 657 y_H2O rows at a 5x looser floor, y_H2O would
land at under 0.5 % of the objective and never move. The budget makes that
trade-off explicit and scannable instead of accidental.

Staging
-------
Brine density is gas-free, so the Peneloux ion shifts are determined without
any gas-ion parameter entering -- but they DO affect solubility. The
coupling is one-way, so the default stages the fit: Peneloux first from
density alone, then held fixed while the gas-ion energies are fitted. That
is correct rather than a compromise, and it insulates the brine-density
result from the solubility/y_H2O trade-off.

Usage
-----
    python3 -m fitting.refit --stage peneloux --out 01_peneloux.json
    python3 -m fitting.refit --stage ions --peneloux 01_peneloux.json
    python3 -m fitting.refit --pareto --backend fortran

The second command needs ``--peneloux``: nothing is carried between two
processes, so without it the ion energies would be fitted on the PUBLISHED
shifts rather than on the block the first command just produced.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from . import blocks as B
from . import dataset as ds
from . import predict as pr

_CODE = Path(__file__).resolve().parents[1]

#: A row the model cannot converge on is charged as if it had been predicted
#: with this RELATIVE error -- 100 %, i.e. a prediction of zero (or double).
#:
#: This was a FIXED 10 sigma, which was wrong in a way that mattered. Sigma is
#: absolute for brine density (0.5 kg/m3, Al Ghafri's stated uncertainty) and
#: the model reaches about 18 kg/m3 there, so the median CONVERGED density row
#: already costs 23 sigma and the 95th percentile costs 78. A flat penalty of
#: 10 therefore made non-convergence CHEAPER than a typical successful
#: prediction: the optimiser was rewarded for breaking the flash. The same
#: inversion reached into the upper tail of four other families.
#:
#: Expressing it as a relative error puts every family on the same footing
#: automatically -- for a relative-sigma family the observation cancels and it
#: is simply 1/sigma_rel, while for density it is rho/sigma_abs. Measured
#: against the baseline residual distributions, 100 % lands almost exactly on
#: the WORST converged residual each family actually produces (CO2 solubility
#: 35.7 vs a max of 35.6 sigma; CO2 binary 27.0 vs 27.0; CH4 binary 17.2 vs
#: 17.2), which is the rule worth stating: a row the model cannot solve costs
#: what its worst solved row costs. Converged residuals are clipped to the
#: same magnitude, which makes "non-convergence is never cheaper than
#: convergence" exact rather than approximate.
#:
#: KNOWN CONSEQUENCE, visible in the output rather than hidden: 11.7 % of
#: brine-density rows do not converge at the published parameters, and at this
#: penalty they dominate that stage's cost. The density stage is therefore
#: convergence-led, which is defensible when its accuracy (1.74 % AAD) is
#: already good and its failures are not -- but it IS a choice. ``converged_fraction`` is reported next to every AAD so
#: that it can be read off the result: if convergence stays flat through a
#: stage, the penalty was an inert constant and accuracy drove the fit anyway.
PENALTY_REL = 1.0


#: Temperatures over which physical constraints on parameter COMBINATIONS are
#: enforced. The data spans 273-633 K; a constraint is meaningless outside the
#: range the parameters are asked to describe, and enforcing it far outside
#: would distort the fit to defend a region nobody uses.
CONSTRAINT_T_GRID = np.linspace(273.0, 633.0, 73)

#: Weight on a physical-constraint violation.
#:
#: Sized so the constraint is SATISFIED rather than traded. At 100 the fit
#: bought a partial reduction in the violation with accuracy elsewhere and
#: still left S13 negative at 2 of 73 grid temperatures -- which is the
#: failure mode of every soft penalty that is merely "large". With families
#: contributing B_f * AAD_f ~ 0.1-0.3 under the AAD transform, a violation of
#: 1e-3 in S13 now costs 100 and one of 1e-2 costs 1e4: there is no trade to
#: make.
CONSTRAINT_WEIGHT = 1.0e4


def s13_values(vals, T_grid=None):
    """The CO2-H2O solvation strength S13(T) implied by a parameter set.

    S13(T) = AS13*(T/Tc3)^2 + BS13*(T/Tc3) + CS13, exactly as ms_ecpa.eos and
    ms_ecpa.lnphi evaluate it.
    """
    import ms_ecpa.parameters as _P
    vals = B.expand(vals)          # accepts either coordinate system
    T = CONSTRAINT_T_GRID if T_grid is None else np.asarray(T_grid, float)
    tau = T / float(_P.Tc3)
    return (float(np.ravel(vals["AS13"])[0]) * tau ** 2
            + float(np.ravel(vals["BS13"])[0]) * tau
            + float(np.ravel(vals["CS13"])[0]))


def constraint_residuals(vals) -> np.ndarray:
    """Residuals enforcing physical conditions that box bounds cannot express.

    S13(T) >= 0. The solvation strength multiplies the CO2-water
    cross-association directly (chi3w = Zw/(Zw + 2 x1w chi1w S13 DELTA)), so a
    negative value is an ANTI-bond: it pushes water out of the CO2-rich phase
    instead of holding it. A bond strength cannot be negative.

    This is not hypothetical. The published coefficients give S13 < 0 below
    289.8 K, and that is exactly where the measured enhancement factor
    E = y_H2O*P/Psat_w(T) collapses -- at 273-304 K and 100-300 bar the model
    carries 15 % of the water it should. The coefficients were fitted to CO2
    SOLUBILITY, the liquid side, with y_H2O in no objective, so nothing
    penalised the extrapolation. lnphi._gas_root_is_physical then relaxed the
    Wertheim chi <= 1 bound precisely where S13 < 0 -- correct about the
    theorem, whose precondition is "every association strength is
    non-negative", but it accepted that the precondition should fail.

    Positivity is a JOINT condition on AS13/BS13/CS13, so a per-coefficient
    box cannot state it; it has to enter the objective as a residual.
    """
    ex = B.expand(vals)
    if not all(k in ex for k in ("AS13", "BS13", "CS13")):
        return np.zeros(0)
    # Under the (SA13, SB13, SC13) parameterisation this is satisfied by
    # construction and returns zeros -- it is kept as a cheap assertion, and
    # it still binds if the raw coefficients are ever fitted directly.
    return CONSTRAINT_WEIGHT * np.maximum(0.0, -s13_values(vals))


def observed(fam: ds.Family) -> np.ndarray:
    return fam.df["value"].to_numpy(dtype=float)


def predicted(fam: ds.Family, backend: str, pool=None,
              vals=None) -> np.ndarray:
    """Model prediction for every row of ``fam``.

    With a ``pool``, the rows are evaluated across the node's cores and
    ``vals`` (the trial parameters) is rebound inside each worker. The
    parallel path returns the same array as the serial one -- density is cut
    along its own warm-start group boundaries, never across them. See
    fitting/parallel.

    ``vals`` is bound on BOTH paths. The serial branch used to ignore it and
    predict with whatever happened to be bound in the process, so a reporting
    call such as ``score(fams, backend, vals=PUB)`` returned the AAD of some
    other parameter set under PUB's label -- and the two paths disagreed
    whenever they were not accidentally in step, with nothing reporting it.
    ``parallel.start`` returns None for a single core, so any one-worker run
    took that branch.
    """
    if pool is not None:
        if vals is None:
            raise ValueError("parallel evaluation needs the trial parameters")
        from . import parallel as par
        return par.predict_family_parallel(fam, vals, backend, pool)
    if vals is not None:
        # Exactly what parallel._predict_chunk does inside a worker.
        B.bind_python(vals)
        if backend == "fortran":
            B.bind_fortran(pr._engine(), vals)
    # Density must go through the family-level path: its inner Newton has
    # several roots and which one it finds depends on the traversal, so a
    # row-by-row cold call reproduces neither the published numbers nor the
    # convergence count. See fitting/predict.predict_rho_family.
    if fam.name == "rho":
        return pr.predict_rho_family(fam.df, backend)
    return np.array([pr.predict_family(fam.name, r, backend)
                     for _, r in fam.df.iterrows()], dtype=float)


def _sigma(fam: ds.Family, obs: np.ndarray) -> np.ndarray:
    if fam.sigma_kind == "rel":
        return np.maximum(np.abs(obs) * fam.sigma, 1e-12)
    return np.full_like(obs, fam.sigma)


def _penalty(fam: ds.Family, obs: np.ndarray) -> np.ndarray:
    """Per-row residual charged for non-convergence, in sigma units."""
    return PENALTY_REL * np.abs(obs) / _sigma(fam, obs)


def family_residuals(fam: ds.Family, backend: str, mask=None, pool=None,
                     vals=None, exact_aad=False) -> np.ndarray:
    obs = observed(fam)
    pred = predicted(fam, backend, pool=pool, vals=vals)
    if exact_aad:
        # r^2 = w * |relative deviation|, so sum(r^2) IS the weighted AAD.
        rel = np.where(np.isfinite(pred),
                       np.abs(pred - obs) / np.maximum(np.abs(obs), 1e-300),
                       PENALTY_REL)
        rel = np.minimum(rel, PENALTY_REL)      # same cap as the penalty
        # w^2, not w. fam.weights() is sqrt(B_f/N_f), sized for the SQUARED
        # objective where a family contributes B_f * mean(r^2) -- independent
        # of row count, which is the whole point of the balance weight. Under
        # this transform sum(r^2) = sum_i w_i * rel_i, so w_i must be B_f/N_f
        # for the family to contribute B_f * AAD_f. Using w itself makes the
        # contribution sqrt(B_f*N_f) * AAD_f, and the sqrt(N) dependence the
        # weight was invented to remove comes straight back: in the CO2
        # binary/solvation stage it gave y_H2O 2.7x the pull it should have,
        # and the stage duly traded 8.3 points of binary accuracy for 1.9 of
        # y_H2O.
        r = np.sqrt(fam.weights() ** 2 * np.maximum(rel, AAD_FLOOR))
        if mask is not None:
            r = r[np.asarray(mask, dtype=bool)]
        return r
    sig = _sigma(fam, obs)
    w = fam.weights()
    pen = _penalty(fam, obs)
    r = np.where(np.isfinite(pred), (pred - obs) / sig, pen)
    # Clipping to the same magnitude is what makes non-convergence never
    # cheaper than convergence. In practice it binds only on a handful of
    # extreme outliers, so it is a mild robustification rather than a
    # reshaping of the objective.
    r = np.clip(r, -pen, pen) * w
    if mask is not None:
        r = r[np.asarray(mask, dtype=bool)]
    return r


def aad_percent(fam: ds.Family, backend: str, mask=None, pool=None,
                vals=None) -> float:
    """Average absolute RELATIVE deviation [%], the paper's reported metric.

    Non-converged rows are EXCLUDED here, unlike in the objective. The AAD is
    the papers' reported accuracy metric and must keep meaning what it meant;
    the convergence count is reported alongside it by :func:`score`.
    """
    obs = observed(fam)
    pred = predicted(fam, backend, pool=pool, vals=vals)
    ok = np.isfinite(pred) & (np.abs(obs) > 0)
    if mask is not None:
        ok &= np.asarray(mask, dtype=bool)
    if not ok.any():
        return float("nan")
    return float(100.0 * np.mean(np.abs(pred[ok] - obs[ok]) / np.abs(obs[ok])))


def score(fams: dict, backend: str, pool=None, vals=None) -> dict:
    return {k: aad_percent(f, backend, pool=pool, vals=vals)
            for k, f in fams.items()}


def converged_fraction(fams: dict, backend: str, pool=None, vals=None) -> dict:
    """Fraction of rows the model returns a finite prediction for.

    Reported next to every AAD. An AAD that improves while this falls is a
    parameter set that got better at the rows it still solves and gave up on
    the rest -- which the objective now punishes, but which must still be
    visible in the result rather than inferred from it.
    """
    out = {}
    for k, f in fams.items():
        p = predicted(f, backend, pool=pool, vals=vals)
        out[k] = float(np.mean(np.isfinite(p))) if f.n else float("nan")
    return out


# ---------------------------------------------------------------------------
# the harness
# ---------------------------------------------------------------------------

#: Loss shaping, so the fit can optimise what the papers actually report.
#:
#: The papers report AAD -- the MEAN ABSOLUTE relative deviation. The default
#: objective is a sum of SQUARED sigma-weighted residuals, and the two do not
#: have the same minimum: squaring chases the worst rows, which is why job
#: 11126525 drove CH4 solubility's cost down while its AAD went 9.341 ->
#: 9.531 %. Fitting one number and reporting another is hard to defend, and
#: worse, it can make the headline number go backwards.
#:
#: ``OBJECTIVE_AAD`` selects scipy's ``soft_l1``, whose cost grows like r^2
#: for residuals inside ``f_scale`` and like |r| beyond it. With residuals
#: expressed in sigma units, f_scale = 1 means "quadratic within the
#: measurement floor, absolute outside it" -- so beyond the noise the fit is
#: minimising a sum of absolute relative deviations, which is AAD up to the
#: per-family constant already carried by the budget weight.
#:
#: It is offered rather than imposed: jobs 11126525 and 11126541 both ran on
#: the squared objective, and switching the default underneath them would
#: make the three runs incomparable.
OBJECTIVE_AAD = dict(loss="soft_l1", f_scale=1.0)
OBJECTIVE_L2 = dict(loss="linear", f_scale=1.0)

#: AAD exactly, not approximately.
#:
#: soft_l1 at f_scale = 1 sigma turned out to be too mild to matter: on the
#: CH4 stage it moved AAD from +0.190 to +0.164 against baseline -- still
#: worse than not fitting at all. Typical residuals there are ~2.4 sigma,
#: which is close enough to f_scale that the loss is still essentially
#: quadratic.
#:
#: This mode is exact instead of shaped. least_squares minimises sum(r_i^2),
#: so defining
#:
#:     r_i = sqrt( w_f * |pred_i - obs_i| / |obs_i| )
#:
#: makes sum(r_i^2) = sum_f w_f * sum_i |relative deviation| -- which IS the
#: reported AAD, up to the per-family constant the budget already carries.
#: The square root is not a trick: it is the change of variable that turns a
#: least-squares solver into an absolute-deviation solver.
#:
#: The cost is that d(sqrt(u))/du diverges as a row's error approaches zero,
#: so a row the model nails contributes an ill-conditioned Jacobian column.
#: ``AAD_FLOOR`` keeps the argument away from zero; it is far below any
#: measurement floor, so it changes the objective nowhere it matters.
OBJECTIVE_AAD_EXACT = dict(loss="linear", f_scale=1.0, exact_aad=True)

#: Same, but additionally requiring S13(T) >= 0 (see constraint_residuals).
#: OPT-IN, and it must stay opt-in: the constraint carries weight 1e4 against
#: a data term that cannot exceed ~1, so wherever it is on it does not inform
#: the fit, it IS the fit. Made unconditional, it turns a refit of the
#: published model form into a constrained one: the fitted S13 is then pinned
#: at zero.
OBJECTIVE_AAD_CONSTRAINED = dict(loss="linear", f_scale=1.0, exact_aad=True,
                                 constrain_s13=True)

#: Relative deviation below which the sqrt transform is flattened.
AAD_FLOOR = 1e-8


def run_stage(vectors, families, *, backend="fortran", max_nfev=None,
              verbose=True, label="stage", pool=None, context=None,
              objective=None):
    """Fit ``vectors`` against ``families`` and return the fitted values.

    ``context`` carries the blocks EARLIER stages already fitted. It matters
    only on the parallel path, and it matters absolutely: a worker rebinds the
    vectors it is handed and nothing else, so without the context it would
    keep evaluating stage 2 against the PUBLISHED Peneloux shifts while the
    parent believes the fitted ones are in force. The two would differ by a
    few per cent and nothing would report an error.
    """
    template = B.published_values(vectors)
    lo_d, hi_d = B.brackets(vectors)
    x0 = B.flatten(template)
    lo = B.flatten({k: lo_d[k] for k in template})
    hi = B.flatten({k: hi_d[k] for k in template})
    z0 = B.to_z(x0, lo, hi)

    obj = dict(objective or OBJECTIVE_L2)
    engine = pr._engine() if backend == "fortran" else None
    n_eval = [0]
    t0 = time.perf_counter()

    def fun(z):
        vals = B.unflatten(B.from_z(z, lo, hi), template)
        B.bind_python(vals)
        if engine is not None:
            B.bind_fortran(engine, vals)
        send = dict(context or {})
        send.update(vals)
        out = [family_residuals(f, backend, pool=pool, vals=send,
                                exact_aad=obj.get("exact_aad", False))
               for f in families]
        if obj.get("constrain_s13", False):
            con = constraint_residuals(send)
            if con.size:
                out.append(con)
        n_eval[0] += 1
        r = np.concatenate(out) if out else np.zeros(0)
        if verbose and n_eval[0] % 5 == 1:
            print(f"    [{label}] eval {n_eval[0]:4d}  "
                  f"cost {0.5*float(np.dot(r, r)):.6g}  "
                  f"{time.perf_counter()-t0:6.1f}s", flush=True)
        return r

    res = least_squares(fun, z0, bounds=(1.0, 2.0), method="trf",
                        diff_step=1e-3, x_scale=1.0, max_nfev=max_nfev,
                        loss=obj["loss"], f_scale=obj["f_scale"])
    fitted = B.unflatten(B.from_z(res.x, lo, hi), template)
    B.bind_python(fitted)
    if engine is not None:
        B.bind_fortran(engine, fitted)

    # least_squares' own verdict. status <= 0 means it did NOT converge
    # (0 = ran out of evaluations, -1 = improper input), and status 4 means it
    # stopped on the gradient alone. Discarding this made a truncated fit
    # indistinguishable from a converged one -- and several stages in this
    # project have terminated at status 3 after very few evaluations.
    if res.status <= 0:
        print(f"    [{label}] WARNING: optimiser did NOT converge "
              f"(status={res.status}, nfev={res.nfev}) -- the returned "
              f"parameters are where it stopped, not a fitted optimum",
              flush=True)

    # A parameter that converged ONTO its bracket is a finding, not a
    # success: the optimiser wanted to go further and was stopped.
    #
    # Reported like the status warning above: unconditionally, and on the
    # result object as well as on stdout. Gating it on ``verbose`` meant the
    # batch drivers, which run verbose=False, computed the list and threw it
    # away -- a bracket-pinned block was then indistinguishable from a
    # converged one in every artefact they wrote. ``res.on_edge`` gives a
    # caller something to record; run_stage's (fitted, res) shape is
    # unchanged, so no call site has to move.
    on_edge = [k for k, v in fitted.items()
               if np.any(np.isclose(B.to_z(v, lo_d[k], hi_d[k]), 1.0, atol=1e-6)
                         | np.isclose(B.to_z(v, lo_d[k], hi_d[k]), 2.0, atol=1e-6))]
    res.on_edge = on_edge
    if on_edge:
        print(f"    [{label}] WARNING: at bracket edge: {on_edge} -- the "
              f"bracket, not the data, set these", flush=True)
    return fitted, res


def refit_all(*, backend="fortran", budgets=None, staged=True, verbose=True,
              pool=None, fams=None, peneloux=None, stage="all"):
    """Full ms-eCPA refit. Returns (fitted parameters, before/after scores).

    ``peneloux`` supplies an already-fitted Peneloux block, which is then bound
    and its stage skipped. The density family is gas-free, so nothing in the
    Pareto scan (which moves only the y_H2O budget) can change that stage's
    answer -- refitting it once per scan point would burn ~45 min of identical
    arithmetic each time.

    ``stage`` selects which of the staged blocks are actually fitted, and is
    what makes the documented two-command workflow work: "peneloux" stops
    after the density stage, "ions" skips it and fits only the gas-ion
    energies on top of the Peneloux block currently bound (or the one handed
    in via ``peneloux=``), "all" runs both. It is ignored when
    ``staged=False``, where there is one joint stage and nothing to select.

    ``fams`` reuses an assembled dataset. Only the per-family BUDGET changes
    between scan points, and the budget lives in the weight the parent applies,
    not in the rows, so the same dataframes -- and therefore the same forked
    pool -- are valid throughout.
    """
    # Checked before the dataset is assembled: ds.build is not cheap, and a
    # misspelt stage should not cost it.
    if stage not in ("all", "peneloux", "ions"):
        raise ValueError(f"unknown stage {stage!r}; "
                         "expected 'all', 'peneloux' or 'ions'")
    if fams is None:
        fams = ds.build(budgets=budgets)
    elif budgets:
        for k, b in budgets.items():
            if k in fams:
                fams[k].budget = float(b)
    #: Every vector the refit may touch. The "before" score must be taken
    #: with these bound, and the workers rebind whatever they are handed, so
    #: the current values have to travel with the request rather than being
    #: assumed to already be in place inside each worker.
    ALL = ("A_Peni", "B_Peni", "A_U3i", "B_U3i", "A_U2i", "B_U2i")
    current = B.published_values(ALL)
    before = score(fams, backend, pool=pool, vals=current)
    fitted = {}

    if staged:
        # 1. Peneloux from gas-free density alone (one-way coupling).
        if peneloux is not None:
            fitted.update(peneloux)
            B.bind_python(peneloux)
            if backend == "fortran":
                B.bind_fortran(pr._engine(), peneloux)
            if verbose:
                print("    [peneloux] reusing the fitted block "
                      "(gas-free; the y_H2O budget cannot reach it)",
                      flush=True)
        elif stage != "ions":
            f, _ = run_stage(("A_Peni", "B_Peni"), [fams["rho"]],
                             backend=backend, verbose=verbose,
                             label="peneloux", pool=pool, context=fitted)
            fitted.update(f)
        elif verbose:
            # No block supplied and the stage held: the ion energies are
            # fitted on top of whatever Peneloux shifts are bound, which for
            # a fresh process are the published ones. Say so, because the
            # output will not otherwise contain A_Peni/B_Peni at all.
            print("    [peneloux] stage held -- fitting the ion energies on "
                  "the currently bound shifts", flush=True)
        # 2. Gas-ion energies from solubility + y_H2O, per gas. The two gases
        #    have SEPARATE parameter blocks and separate data, so they do not
        #    compete and are fitted independently.
        if stage != "peneloux":
            for gas, vecs in (("co2", ("A_U3i", "B_U3i")),
                              ("ch4", ("A_U2i", "B_U2i"))):
                targets = [fams[f"{gas}_solubility"], fams[f"{gas}_binary"],
                           fams[f"{gas}_yh2o"]]
                f, _ = run_stage(vecs, targets, backend=backend,
                                 verbose=verbose, label=f"{gas}_ion",
                                 pool=pool, context=fitted)
                fitted.update(f)
    else:
        vecs = ("A_Peni", "B_Peni", "A_U3i", "B_U3i", "A_U2i", "B_U2i")
        f, _ = run_stage(vecs, list(fams.values()), backend=backend,
                         verbose=verbose, label="joint", pool=pool,
                         context=fitted)
        fitted.update(f)

    after = score(fams, backend, pool=pool, vals=fitted)
    return fitted, {"before": before, "after": after}


#: The restaged fit, in dependency order. Each entry is
#: (label, parameter vectors, family names it is fitted against).
#:
#: The ordering is forced by what each block can physically reach, and every
#: claim below was MEASURED by perturbation rather than reasoned about:
#:
#:   * the salt-free binaries do not respond to the ion energies (exactly 0)
#:     nor to the Peneloux ion shifts (exactly 0) -- they carry no ions;
#:   * brine density does not respond to any of the nine binary coefficients
#:     (exactly 0) -- it is gas-free;
#:   * so stages A and B are independent of each other and of C, and only C
#:     depends on both.
#:
#: The consequence that matters: y_H2O is 84.8 % (CO2) and 100 % (CH4)
#: salt-free, so almost all of it belongs in stage A with the solvation block
#: that reaches it, NOT in stage C with the ion energies that do not.
STAGES_BINARY_FIRST = (
    # Solvation fitted as (SA13, SB13, SC13): S13 = (SA13*tau+SB13)^2 + SC13,
    # SC13 >= 0, so the strength cannot go negative. See blocks.SOLVATION_VIRTUAL.
    ("co2_binary", ("Ak13", "Bk13", "Ck13", "SA13", "SB13", "SC13"),
     ("co2_binary", "co2_yh2o_sf")),
    ("ch4_binary", ("Ak12", "Bk12", "Ck12"),
     ("ch4_binary", "ch4_yh2o_sf")),
    ("peneloux", ("A_Peni", "B_Peni"), ("rho",)),
    ("co2_ion", ("A_U3i", "B_U3i"), ("co2_solubility", "co2_yh2o_brine")),
    ("ch4_ion", ("A_U2i", "B_U2i"), ("ch4_solubility", "ch4_yh2o_brine")),
)

#: The Papers II/III staging: y_H2O is not in any objective.
#:
#: That is the objective of the published parametrization. Around it, this
#: refit uses the larger GasBrineBench validation set, the dew-point rescue
#: that solves the fugacity balances, the liquid root guard on brine density,
#: and an objective that minimises the AAD the papers report.
#:
#: y_H2O stays out deliberately. Constraining the solvation strength to be
#: physical reaches its measured reproducibility floor but costs 3-4x on
#: solubility (reproduce/enhancement_factor.py and the constrained refits).
#: The blocks here are therefore the PUBLISHED ones -- AS13/BS13/CS13
#: directly, not the non-negative reparameterisation, and MCR1_CROSS_ENERGY
#: off -- so the model form is exactly the published one.
STAGES_NO_YH2O = (
    ("co2_binary", ("Ak13", "Bk13", "Ck13", "AS13", "BS13", "CS13"),
     ("co2_binary",)),
    ("ch4_binary", ("Ak12", "Bk12", "Ck12"), ("ch4_binary",)),
    ("peneloux", ("A_Peni", "B_Peni"), ("rho",)),
    ("co2_ion", ("A_U3i", "B_U3i"), ("co2_solubility",)),
    ("ch4_ion", ("A_U2i", "B_U2i"), ("ch4_solubility",)),
)


#: Families kept only so the reported AAD matches an unsplit run; never fitted.
_REPORT_ONLY = ("co2_yh2o", "ch4_yh2o")


def refit_binary_first(*, backend="fortran", budgets=None, verbose=True,
                       pool=None, fams=None, max_nfev=None, preset=None,
                       objective=None, stages=None):
    """Refit with the binary/solvation block fitted FIRST.

    The alternative staging (``refit_all``) never fits the binary coefficients
    at all, so the ion energies are asked to explain salting-out on top of a
    gas-water baseline that is itself 15 % off for CO2. Here the baseline is
    fitted first, from the salt-free data that determines it, and the ion
    energies then explain only what salt actually does.

    Returns (fitted parameters, {"before": ..., "after": ...}).
    """
    if fams is None:
        fams = ds.build(budgets=budgets, split_yh2o=True)
    elif budgets:
        for k, b in budgets.items():
            if k in fams:
                fams[k].budget = float(b)

    ALL = tuple(v for _, vecs, _ in (stages or STAGES_BINARY_FIRST)
                for v in vecs)
    current = B.published_values(ALL)
    before = score(fams, backend, pool=pool, vals=current)

    fitted = {}
    for label, vecs, fam_names in (stages or STAGES_BINARY_FIRST):
        # A stage whose whole block is supplied is bound and skipped. Used by
        # the Pareto scan for the Peneloux block, which is gas-free and
        # therefore cannot be reached by the y_H2O budget being scanned.
        if preset and all(v in preset for v in vecs):
            vals = {v: preset[v] for v in vecs}
            fitted.update(vals)
            B.bind_python(vals)
            if backend == "fortran":
                B.bind_fortran(pr._engine(), vals)
            if verbose:
                print(f"    [{label}] reusing the fitted block", flush=True)
            continue
        targets = [fams[n] for n in fam_names if n in fams and fams[n].n]
        if not targets:
            if verbose:
                print(f"    [{label}] no rows -- stage skipped", flush=True)
            continue
        if verbose:
            print(f"\n--- stage {label}: {len(vecs)} vectors against "
                  f"{', '.join(f'{t.name}({t.n})' for t in targets)} ---",
                  flush=True)
        f, _ = run_stage(vecs, targets, backend=backend, verbose=verbose,
                         label=label, pool=pool, context=fitted,
                         max_nfev=max_nfev, objective=objective)
        fitted.update(f)

    after = score(fams, backend, pool=pool, vals=fitted)
    return fitted, {"before": before, "after": after}


def pareto_scan(budgets_yh2o, *, backend="fortran", out=None, verbose=True,
                pool=None, fams=None, peneloux=None):
    """Scan the y_H2O family budget and record the trade-off.

    The point is not to justify one weight but to show what moving it costs:
    the Pareto front gives the price of each choice of weight.

    Brine density is scored on every point too, because the Peneloux shifts
    are shared and a y_H2O gain must not quietly cost brine-density accuracy.

    Every point starts from the published parameters. ``B.published_values``
    reads the LIVE library and ``B.bind_python`` writes into it in place, so
    without the snapshot-and-restore below point N+1 would warm-start from
    point N's solution and re-centre its own +/-80 kJ/mol brackets there: the
    front would depend on the order ``budgets_yh2o`` happens to be listed in,
    the bracket-edge warning would be measured against a moved bracket, and
    ``refit_all``'s "before" columns would report the previous point's fit
    while being labelled the baseline.
    """
    #: Every vector refit_all may touch, so the restore covers all of them.
    ALL = ("A_Peni", "B_Peni", "A_U3i", "B_U3i", "A_U2i", "B_U2i")
    published = B.published_values(ALL)
    rows = []
    for Bf in budgets_yh2o:
        B.bind_python(published)
        if backend == "fortran":
            B.bind_fortran(pr._engine(), published)
        if verbose:
            print(f"\n=== y_H2O budget B_f = {Bf} ===", flush=True)
        budgets = {"co2_yh2o": Bf, "ch4_yh2o": Bf}
        _, sc = refit_all(backend=backend, budgets=budgets, verbose=verbose,
                          pool=pool, fams=fams, peneloux=peneloux)
        row = {"B_yh2o": Bf}
        row.update({f"after_{k}": v for k, v in sc["after"].items()})
        row.update({f"before_{k}": v for k, v in sc["before"].items()})
        rows.append(row)
        if verbose:
            print(f"  -> co2_yh2o {sc['after']['co2_yh2o']:.2f}%  "
                  f"co2_solubility {sc['after']['co2_solubility']:.2f}%  "
                  f"rho {sc['after']['rho']:.2f}%", flush=True)
    if out:
        Path(out).write_text(json.dumps(rows, indent=2))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ms-eCPA refit")
    ap.add_argument("--backend", default="fortran",
                    choices=("python", "fortran"))
    ap.add_argument("--stage", choices=("peneloux", "ions", "all", "joint"),
                    default="all",
                    help="which staged block to fit: peneloux (density only), "
                         "ions (gas-ion energies on a held Peneloux block), "
                         "all (both), joint (one unstaged stage)")
    ap.add_argument("--peneloux", default=None,
                    help="JSON holding a fitted Peneloux block (A_Peni/"
                         "B_Peni), as written by --stage peneloux --out; it "
                         "is bound and its stage skipped")
    ap.add_argument("--pareto", action="store_true",
                    help="scan the y_H2O budget and write the trade-off")
    ap.add_argument("--budgets", default="0.25,1,4,16,64,256")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    if a.pareto:
        bs = [float(x) for x in a.budgets.split(",")]
        rows = pareto_scan(bs, backend=a.backend, out=a.out)
        print("\nB_yh2o   co2_yh2o%   co2_sol%    rho%")
        for r in rows:
            print(f"{r['B_yh2o']:7g} {r['after_co2_yh2o']:10.2f} "
                  f"{r['after_co2_solubility']:10.2f} {r['after_rho']:8.2f}")
        return 0

    peneloux = None
    if a.peneloux:
        blk = json.loads(Path(a.peneloux).read_text())
        peneloux = {k: np.asarray(v, dtype=float)
                    for k, v in blk.items() if k in ("A_Peni", "B_Peni")}
        if not peneloux:
            raise SystemExit(f"{a.peneloux}: no A_Peni/B_Peni block")
    fitted, sc = refit_all(backend=a.backend, staged=(a.stage != "joint"),
                           stage="all" if a.stage == "joint" else a.stage,
                           peneloux=peneloux)
    print("\nfamily            before%    after%")
    for k in sc["before"]:
        print(f"  {k:16s} {sc['before'][k]:8.2f} {sc['after'][k]:9.2f}")
    if a.out:
        Path(a.out).write_text(json.dumps(
            {k: np.asarray(v).tolist() for k, v in fitted.items()}, indent=2))
        print(f"\nparameters -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
