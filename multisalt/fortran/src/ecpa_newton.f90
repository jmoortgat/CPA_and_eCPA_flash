! ecpa_newton.f90 -- damped Newton with a complex-step Jacobian.
! Faithful transcription of ms_ecpa/newton.py (h = 1e-30, backtracking
! line search on the residual infinity norm, min_lambda = 1e-6,
! max_backtracks = 40).
!
! Derivative strategy (documented choice): identical to the Python
! reference -- COMPLEX-STEP differentiation.  The residual kernels are
! written once in complex(real64) arithmetic and reused for both the
! real evaluation (zero imaginary parts propagate exactly, so the real
! parts match a real-arithmetic implementation bit-for-bit) and the
! perturbed evaluations J(:,j) = Im f(x + i*h*e_j)/h.  Invalid-region
! semantics mirror NumPy's NaN behaviour: with all-real inputs a
! nonzero imaginary part can only arise from a negative branch-cut
! argument (log/sqrt of a negative number), exactly the places where
! the Python real path produces NaN -- both are mapped to an infinite
! residual norm so the line search rejects the step.
module ecpa_newton
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite, ieee_value, &
       ieee_quiet_nan, ieee_positive_inf
  use ecpa_kinds, only: wp
  use ecpa_params, only: par_t
  implicit none
  private
  public :: rextra_t, resid_if, newton_solve, lin_solve, nan_wp, inf_wp, &
            clip_np, maxabs_nan, nansum_np

  real(wp), parameter :: cs_h = 1.0e-30_wp

  ! Context payload for the residual callbacks: composition (x_aq or
  ! y_gas), ion molalities, T [K], P [Pa].
  type :: rextra_t
     real(wp) :: comp(3) = 0.0_wp
     real(wp) :: mi(6) = 0.0_wp
     real(wp) :: T = 0.0_wp
     real(wp) :: P = 0.0_wp
  end type rextra_t

  abstract interface
     subroutine resid_if(par, ex, n, v, f)
       import :: wp, par_t, rextra_t
       type(par_t), intent(in) :: par
       type(rextra_t), intent(in) :: ex
       integer, intent(in) :: n
       complex(wp), intent(in) :: v(n)
       complex(wp), intent(out) :: f(n)
     end subroutine resid_if
  end interface

contains

  ! NumPy-semantics helpers: np.clip / np.max / np.nansum PROPAGATE or
  ! IGNORE NaN in specific ways that Fortran's min/max/maxval/sum need
  ! not honor (IEEE maxNum drops NaN); several solver trajectories
  ! depend on the NumPy behaviour (see PORTING_NOTES.md).
  elemental function clip_np(x, lo, hi) result(y)
    real(wp), intent(in) :: x, lo, hi
    real(wp) :: y
    if (x /= x) then          ! NaN propagates (np.clip semantics)
       y = x
    else
       y = min(max(x, lo), hi)
    end if
  end function clip_np

  pure function maxabs_nan(v) result(m)
    ! max(abs(v)) with NaN propagation (np.max(np.abs(v)) semantics).
    real(wp), intent(in) :: v(:)
    real(wp) :: m
    integer :: i
    do i = 1, size(v)
       if (v(i) /= v(i)) then
          m = v(i)
          return
       end if
    end do
    m = 0.0_wp
    do i = 1, size(v)
       m = max(m, abs(v(i)))
    end do
  end function maxabs_nan

  pure function nansum_np(v) result(s)
    ! sum treating NaN as 0 (np.nansum semantics).
    real(wp), intent(in) :: v(:)
    real(wp) :: s
    integer :: i
    s = 0.0_wp
    do i = 1, size(v)
       if (v(i) == v(i)) s = s + v(i)
    end do
  end function nansum_np

  pure function nan_wp() result(v)
    real(wp) :: v
    v = ieee_value(0.0_wp, ieee_quiet_nan)
  end function nan_wp

  pure function inf_wp() result(v)
    real(wp) :: v
    v = ieee_value(0.0_wp, ieee_positive_inf)
  end function inf_wp

  ! Real-point residual evaluation via the complex kernel.
  ! fnorm = +inf when any component is non-finite OR carries a nonzero
  ! imaginary part (NumPy-NaN-equivalent invalid-region flag).
  subroutine residual_norm(par, ex, fn, n, x, fnorm, Fr)
    type(par_t), intent(in) :: par
    type(rextra_t), intent(in) :: ex
    procedure(resid_if) :: fn
    integer, intent(in) :: n
    real(wp), intent(in) :: x(n)
    real(wp), intent(out) :: fnorm, Fr(n)
    complex(wp) :: v(n), f(n)
    integer :: i
    logical :: bad
    v = cmplx(x, 0.0_wp, kind=wp)
    call fn(par, ex, n, v, f)
    bad = .false.
    do i = 1, n
       if (.not. (ieee_is_finite(real(f(i))) .and. &
                  ieee_is_finite(aimag(f(i))))) bad = .true.
       if (aimag(f(i)) /= 0.0_wp) bad = .true.
    end do
    Fr = real(f)
    if (bad) then
       fnorm = inf_wp()
    else
       fnorm = maxval(abs(Fr))
    end if
  end subroutine residual_norm

  ! Complex-step Jacobian at x (x must be a valid real point).
  subroutine cs_jacobian(par, ex, fn, n, x, Jm, ok)
    type(par_t), intent(in) :: par
    type(rextra_t), intent(in) :: ex
    procedure(resid_if) :: fn
    integer, intent(in) :: n
    real(wp), intent(in) :: x(n)
    real(wp), intent(out) :: Jm(n, n)
    logical, intent(out) :: ok
    complex(wp) :: v(n), f(n)
    integer :: i1, j1
    ok = .true.
    do j1 = 1, n
       v = cmplx(x, 0.0_wp, kind=wp)
       v(j1) = cmplx(x(j1), cs_h, kind=wp)
       call fn(par, ex, n, v, f)
       do i1 = 1, n
          Jm(i1, j1) = aimag(f(i1)) / cs_h
          if (.not. ieee_is_finite(Jm(i1, j1))) ok = .false.
       end do
    end do
  end subroutine cs_jacobian

  ! Dense LU solve with partial pivoting (dgesv-equivalent for the
  ! tiny 2x2/3x3 systems here).  singular=.true. on an exact zero
  ! pivot; the caller treats that as a failed Newton step (the Python
  ! reference falls back to lstsq in this measure-zero event -- see
  ! PORTING_NOTES.md).
  subroutine lin_solve(n, A_in, b_in, x, singular)
    integer, intent(in) :: n
    real(wp), intent(in) :: A_in(n, n), b_in(n)
    real(wp), intent(out) :: x(n)
    logical, intent(out) :: singular
    real(wp) :: A(n, n), b(n), tmp, amax
    integer :: i, j, k, ip
    A = A_in
    b = b_in
    singular = .false.
    do k = 1, n
       ip = k
       amax = abs(A(k, k))
       do i = k + 1, n
          if (abs(A(i, k)) > amax) then
             amax = abs(A(i, k)); ip = i
          end if
       end do
       if (amax == 0.0_wp) then
          singular = .true.
          x = 0.0_wp
          return
       end if
       if (ip /= k) then
          do j = 1, n
             tmp = A(k, j); A(k, j) = A(ip, j); A(ip, j) = tmp
          end do
          tmp = b(k); b(k) = b(ip); b(ip) = tmp
       end if
       do i = k + 1, n
          A(i, k) = A(i, k) / A(k, k)
          do j = k + 1, n
             A(i, j) = A(i, j) - A(i, k) * A(k, j)
          end do
          b(i) = b(i) - A(i, k) * b(k)
       end do
    end do
    do i = n, 1, -1
       tmp = b(i)
       do j = i + 1, n
          tmp = tmp - A(i, j) * x(j)
       end do
       x(i) = tmp / A(i, i)
    end do
  end subroutine lin_solve

  ! Solve fn(x) = 0 by damped Newton with a complex-step Jacobian.
  ! Mirrors ms_ecpa.newton.newton_solve exactly (same exits, same
  ! backtracking schedule).
  subroutine newton_solve(par, ex, fn, n, x0, tol, maxiter, x, converged, niter)
    type(par_t), intent(in) :: par
    type(rextra_t), intent(in) :: ex
    procedure(resid_if) :: fn
    integer, intent(in) :: n, maxiter
    real(wp), intent(in) :: x0(n), tol
    real(wp), intent(out) :: x(n)
    logical, intent(out) :: converged
    integer, intent(out) :: niter
    real(wp), parameter :: min_lambda = 1.0e-6_wp
    integer, parameter :: max_backtracks = 40
    real(wp) :: Fv(n), Fv_new(n), Jm(n, n), dx(n), xnew(n)
    real(wp) :: fnorm, fnew, lam
    integer :: it, n_bt
    logical :: jac_ok, singular

    x = x0
    call residual_norm(par, ex, fn, n, x, fnorm, Fv)

    do it = 0, maxiter - 1
       if (fnorm < tol) then
          converged = .true.; niter = it
          return
       end if
       if (.not. ieee_is_finite(fnorm)) then
          converged = .false.; niter = it
          return
       end if

       call cs_jacobian(par, ex, fn, n, x, Jm, jac_ok)
       if (.not. jac_ok) then
          converged = .false.; niter = it
          return
       end if

       call lin_solve(n, Jm, -Fv, dx, singular)
       if (singular) then
          ! Python: lstsq fallback; measure-zero event -- treat as no
          ! further progress (documented deviation).
          converged = (fnorm < tol); niter = it
          return
       end if

       lam = 1.0_wp
       xnew = x + dx
       call residual_norm(par, ex, fn, n, xnew, fnew, Fv_new)
       n_bt = 0
       do while (fnew >= fnorm .and. lam > min_lambda .and. n_bt < max_backtracks)
          lam = lam * 0.5_wp
          xnew = x + lam * dx
          call residual_norm(par, ex, fn, n, xnew, fnew, Fv_new)
          n_bt = n_bt + 1
       end do

       if (fnew >= fnorm) then
          converged = (fnorm < tol); niter = it
          return
       end if

       x = xnew
       Fv = Fv_new
       fnorm = fnew
    end do

    converged = (fnorm < tol)
    niter = maxiter
  end subroutine newton_solve

end module ecpa_newton
