"""Damped Newton solver with a complex-step Jacobian.

The eCPA residual system (:func:`ms_ecpa.eos.residual_elv`) is smooth and
free of branches on the unknowns, so it can be differentiated by the
**complex-step** method: perturbing one unknown by ``i*h`` and reading the
imaginary part of the residual gives ``dF/dx`` to machine precision, with no
subtractive-cancellation error (unlike the finite differences ``fsolve`` uses
internally). The result is an analytical-precision Jacobian obtained directly
from the verbatim physics, fed to a damped Newton iteration with a
backtracking line search.
"""
import numpy as np

# Complex-step size. Small enough that the O(h^2) truncation error is
# negligible; complex-step has no cancellation floor, so it need not balance
# roundoff the way a real finite difference must.
_CS_H = 1e-30


def complex_step_jacobian(func, x, args=(), h=_CS_H):
    """Jacobian of ``func`` at ``x`` via complex-step differentiation.

    Returns ``(J, f0)`` where ``J[i, j] = dF_i/dx_j`` and ``f0 = func(x)``
    (real). ``func`` must accept a complex input vector and propagate complex
    arithmetic (no ``abs``/``min``/``max``/comparisons on the unknowns).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    with np.errstate(all="ignore"):  # invalid-region probes are expected
        f0 = np.asarray(func(x, *args), dtype=float)
        m = f0.size
        J = np.empty((m, n))
        xc = x.astype(complex)
        for j in range(n):
            xc[j] = x[j] + 1j * h
            fj = np.asarray(func(xc, *args), dtype=complex)
            J[:, j] = fj.imag / h
            xc[j] = x[j]
    return J, f0


def newton_solve(func, x0, args=(), tol=1e-10, maxiter=100,
                 min_lambda=1e-6, max_backtracks=40):
    """Solve ``func(x, *args) = 0`` by damped Newton with a CS Jacobian.

    Parameters
    ----------
    func : callable        residual function ``func(x, *args) -> array``
    x0 : array_like        initial guess
    args : tuple           extra arguments passed to ``func``
    tol : float            convergence on ``max|F|``
    maxiter : int          maximum Newton iterations
    min_lambda : float     smallest line-search step fraction before accepting
    max_backtracks : int   max halvings per line search

    Returns
    -------
    x : ndarray            solution estimate
    converged : bool       whether ``max|F| < tol``
    niter : int            Newton iterations performed
    """
    def residual_norm(xv):
        """Infinity norm of the residual; +inf if any component is non-finite
        (so the line search rejects steps that wander into invalid regions,
        e.g. negative arguments to log)."""
        with np.errstate(all="ignore"):
            Fx = np.asarray(func(xv, *args), dtype=float)
        if not np.all(np.isfinite(Fx)):
            return np.inf, Fx
        return float(np.max(np.abs(Fx))), Fx

    x = np.array(x0, dtype=float)
    fnorm, F = residual_norm(x)

    for it in range(maxiter):
        if fnorm < tol:
            return x, True, it
        if not np.isfinite(fnorm):
            return x, False, it  # stuck in an invalid region

        J, _ = complex_step_jacobian(func, x, args)
        if not np.all(np.isfinite(J)):
            return x, False, it

        try:
            dx = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            try:
                dx, *_ = np.linalg.lstsq(J, -F, rcond=None)
            except np.linalg.LinAlgError:
                return x, False, it

        # Backtracking line search on the residual infinity norm.
        lam = 1.0
        x_new = x + dx
        f_new, F_new = residual_norm(x_new)
        n_bt = 0
        while f_new >= fnorm and lam > min_lambda and n_bt < max_backtracks:
            lam *= 0.5
            x_new = x + lam * dx
            f_new, F_new = residual_norm(x_new)
            n_bt += 1

        if f_new >= fnorm:
            return x, (fnorm < tol), it  # no further progress possible

        x, F, fnorm = x_new, F_new, f_new

    return x, (fnorm < tol), maxiter
