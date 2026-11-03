! ecpa_brent.f90 -- Brent's method root finder, a faithful port of
! scipy.optimize.brentq (scipy/optimize/Zeros/brentq.c), so the
! Rachford-Rice and dew-point-rescue solves land on the same roots to
! the same tolerances as the Python reference.
module ecpa_brent
  use ecpa_kinds, only: wp
  implicit none
  private
  public :: brent_fn, brentq

  ! Callers pass internal procedures (host association carries the
  ! closure state) -- an F2008 feature supported by gfortran and ifort.
  abstract interface
     function brent_fn(x) result(y)
       import :: wp
       real(wp), intent(in) :: x
       real(wp) :: y
     end function brent_fn
  end interface

contains

  ! Roots of f on [xa, xb] with f(xa)*f(xb) <= 0.
  ! ierr: 0 converged; 1 no sign change; 2 maxiter reached (returns best).
  subroutine brentq(f, xa, xb, xtol, rtol, maxiter, root, ierr)
    procedure(brent_fn) :: f
    real(wp), intent(in) :: xa, xb, xtol, rtol
    integer, intent(in) :: maxiter
    real(wp), intent(out) :: root
    integer, intent(out) :: ierr

    real(wp) :: xpre, xcur, fpre, fcur, xblk, fblk
    real(wp) :: spre, scur, sbis, delta, stry, dpre, dblk
    integer :: i

    xpre = xa; xcur = xb
    xblk = 0.0_wp; fblk = 0.0_wp
    spre = 0.0_wp; scur = 0.0_wp
    ierr = 0

    fpre = f(xpre)
    fcur = f(xcur)
    if (fpre * fcur > 0.0_wp) then
       ierr = 1
       root = 0.0_wp
       return
    end if
    if (fpre == 0.0_wp) then
       root = xpre
       return
    end if
    if (fcur == 0.0_wp) then
       root = xcur
       return
    end if

    do i = 1, maxiter
       if (fpre * fcur < 0.0_wp) then
          xblk = xpre
          fblk = fpre
          spre = xcur - xpre
          scur = xcur - xpre
       end if
       if (abs(fblk) < abs(fcur)) then
          xpre = xcur; xcur = xblk; xblk = xpre
          fpre = fcur; fcur = fblk; fblk = fpre
       end if

       delta = (xtol + rtol * abs(xcur)) / 2.0_wp
       sbis = (xblk - xcur) / 2.0_wp
       if (fcur == 0.0_wp .or. abs(sbis) < delta) then
          root = xcur
          return
       end if

       if (abs(spre) > delta .and. abs(fcur) < abs(fpre)) then
          if (xpre == xblk) then
             ! interpolate (secant)
             stry = -fcur * (xcur - xpre) / (fcur - fpre)
          else
             ! extrapolate (inverse quadratic)
             dpre = (fpre - fcur) / (xpre - xcur)
             dblk = (fblk - fcur) / (xblk - xcur)
             stry = -fcur * (fblk * dblk - fpre * dpre) &
                    / (dblk * dpre * (fblk - fpre))
          end if
          if (2.0_wp * abs(stry) < min(abs(spre), 3.0_wp * abs(sbis) - delta)) then
             spre = scur
             scur = stry
          else
             spre = sbis
             scur = sbis
          end if
       else
          spre = sbis
          scur = sbis
       end if

       xpre = xcur
       fpre = fcur
       if (abs(scur) > delta) then
          xcur = xcur + scur
       else
          if (sbis > 0.0_wp) then
             xcur = xcur + delta
          else
             xcur = xcur - delta
          end if
       end if
       fcur = f(xcur)
    end do
    ierr = 2
    root = xcur
  end subroutine brentq

end module ecpa_brent
