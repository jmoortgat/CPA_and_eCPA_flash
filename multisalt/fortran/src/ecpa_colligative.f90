! ecpa_colligative.f90 -- phase-2 colligative routes.
!
! WHICH PYTHON FUNCTION EACH ROUTINE REPRODUCES (read this first; the
! three routes deliberately do NOT share one assembly, because the
! Python reference does not either):
!
!   phi_osm_total(salt, m, T, Pp)
!       <- code/scripts/total_mole_basis.py :: phi_osm_total
!          the phase-4 "oracle": an INDEPENDENT total-mole-basis
!          Helmholtz assembly (dU_wi, a_w_pure, hv_a_mix, solve_chi,
!          solve_epsr, A_r, pressure, solve_V, ln_f_w) verified
!          term by term against the production kernels.  It does NOT
!          call the production lnphi kernels.
!
!   rho_brine(T, Pbar, mi, v_init, ...)
!       <- code/scripts/fig_brine_density.py :: predict_rho_brine
!          (plus its _solve_Zw seed ladder).  This one DOES use the
!          production aqueous kernel (aq_residual_3var) and the
!          Peneloux ion shifts -- it is the route behind the paper's
!          brine-density figure, not the oracle.
!
!   psat_brine(T, m, p0_Pa, lo_fac, hi_fac)
!       <- pilot_studies/vpsat_salt_2026_08/vpsat.py :: psat_model
!          (lower bracket factor 0.55; the default 0.50 is the
!          other variant -- pass lo_fac=0.55 for vpsat.py).  Water isofugacity
!          between the production aqueous and gas kernels, Brent on
!          ln P.  The reference takes the bracket anchor p0 from
!          CoolProp's IAPWS saturation line; this port takes p0 as an
!          ARGUMENT rather than reimplementing IAPWS, so a caller that
!          supplies the same p0 gets the identical bracket and hence
!          the identical Brent trajectory.  See PORTING_NOTES.md 13.
!
! Basis note: the oracle's component vector is
!   n = [n_w, n_Na, n_Cl, n_K, n_Ca, n_Mg, n_SO4]  [mol], built for
! 1 kg of water (n_w = 55.5084).  Ions are ordinary mole-counted
! components (they enter -n_T ln(1 - B/V) and the P(V) closure).
module ecpa_colligative
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use ecpa_kinds, only: wp
  use ecpa_params, only: par_t
  use ecpa_water, only: rho_w_sat
  use ecpa_extensions, only: coll_dielectric_hook, coll_helmholtz_hook
  use ecpa_newton, only: rextra_t, newton_solve, nan_wp
  use ecpa_lnphi, only: aq_residual_3var, lnphi_aq, lnphi_gas
  use ecpa_brent, only: brentq
  implicit none
  private
  public :: N_W, T_REF, SALT_NACL, SALT_KCL, SALT_CACL2, SALT_MGCL2, &
            SALT_NA2SO4, salt_nu, n_vec, A_r, ln_f_w, solve_v_liquid, &
            phi_osm_total, rho_brine, psat_brine

  ! mol water per kg water (total_mole_basis.N_W)
  real(wp), parameter :: N_W = 55.5084_wp
  real(wp), parameter :: T_REF = 298.15_wp

  ! salt codes (total_mole_basis._SALT_STOICH)
  integer, parameter :: SALT_NACL = 1, SALT_KCL = 2, SALT_CACL2 = 3
  integer, parameter :: SALT_MGCL2 = 4, SALT_NA2SO4 = 5

contains

  ! nu (total ions per formula unit), total_mole_basis.NU
  pure function salt_nu(salt) result(nu)
    integer, intent(in) :: salt
    real(wp) :: nu
    select case (salt)
    case (SALT_NACL);   nu = 2.0_wp
    case (SALT_KCL);    nu = 2.0_wp
    case (SALT_CACL2);  nu = 3.0_wp
    case (SALT_MGCL2);  nu = 3.0_wp
    case (SALT_NA2SO4); nu = 3.0_wp
    case default;       nu = nan_wp()
    end select
  end function salt_nu

  ! 1 kg water + m molal salt -> 7-component total-mole vector.
  ! Ion slots 2..7 are (Na, Cl, K, Ca, Mg, SO4), i.e. n(1+i) is ion i.
  pure function n_vec(salt, m) result(n)
    integer, intent(in) :: salt
    real(wp), intent(in) :: m
    real(wp) :: n(7)
    n = 0.0_wp
    n(1) = N_W
    if (m > 0.0_wp) then
       select case (salt)
       case (SALT_NACL)                       ! Na 1, Cl 1
          n(2) = 1.0_wp * m; n(3) = 1.0_wp * m
       case (SALT_KCL)                        ! K 1, Cl 1
          n(4) = 1.0_wp * m; n(3) = 1.0_wp * m
       case (SALT_CACL2)                      ! Ca 1, Cl 2
          n(5) = 1.0_wp * m; n(3) = 2.0_wp * m
       case (SALT_MGCL2)                      ! Mg 1, Cl 2
          n(6) = 1.0_wp * m; n(3) = 2.0_wp * m
       case (SALT_NA2SO4)                     ! Na 2, SO4 1
          n(2) = 2.0_wp * m; n(7) = 1.0_wp * m
       end select
    end if
  end function n_vec

  ! --------------------------------------------------------------
  ! published-equation assembly (total_mole_basis.py, basis="total")
  ! --------------------------------------------------------------

  ! Water-ion HV interaction energy, J/mol (Schlaikjer eq 6).
  pure function dU_wi(par, T) result(dU)
    type(par_t), intent(in) :: par
    real(wp), intent(in) :: T
    real(wp) :: dU(6)
    dU = par%Uref1i + par%alfa1i * par%R * &
         ((1.0_wp - T / par%Talfa1i) ** 2 - &
          (1.0_wp - T_REF / par%Talfa1i) ** 2)
  end function dU_wi

  pure function a_w_pure(par, T) result(a)
    type(par_t), intent(in) :: par
    real(wp), intent(in) :: T
    real(wp) :: a
    a = par%a01 * (1.0_wp + par%c11 * (1.0_wp - sqrt(T / par%Tc1))) ** 2
  end function a_w_pure

  ! b vector on the 7-component basis: [b_w, b_ions]
  pure function b_vec(par) result(b)
    type(par_t), intent(in) :: par
    real(wp) :: b(7)
    b(1) = par%b1
    b(2:7) = par%bi
  end function b_vec

  ! a(T,n) intensive on the total-mole basis [J m3 mol-2] (eqs 4-5/15).
  pure subroutine hv_a_mix(par, n, T, a, b, gE)
    type(par_t), intent(in) :: par
    real(wp), intent(in) :: n(7), T
    real(wp), intent(out) :: a, b, gE
    real(wp) :: nT, x(7), BV(7), ai(7), dU(6), xw, xi(6), acc, tmp(6)
    integer :: i, j
    BV = b_vec(par)
    nT = sum(n)
    x = n / nT
    b = dot_product(x, BV)
    ai(1) = a_w_pure(par, T)
    ai(2:7) = par%a0i                        ! c1i = 0 for ions
    dU = dU_wi(par, T)
    xw = x(1)
    xi = x(2:7)
    gE = (1.0_wp / b) * sum(xw * xi * (BV(1) + BV(2:7)) * dU)
    ! Ion-ion HV energies (paper-2 slot; production = all zeros, so the
    ! branch is skipped and the baseline stays bit-exact).
    if (any(par%Uref_ii /= 0.0_wp)) then
       do i = 1, 6
          tmp(i) = 0.0_wp
          do j = 1, 6
             tmp(i) = tmp(i) + par%Uref_ii(i, j) * (BV(1 + j) * xi(j))
          end do
       end do
       gE = gE + dot_product(xi, tmp) / b
    end if
    acc = 0.0_wp
    do i = 1, 7
       acc = acc + x(i) * (ai(i) / BV(i))
    end do
    a = b * (acc - gE / log(2.0_wp))
  end subroutine hv_a_mix

  ! 4C water monomer fraction; cw = n_w/V [mol/m3].
  pure subroutine solve_chi(par, cw, Bext, V, T, X, delta, eta, g)
    type(par_t), intent(in) :: par
    real(wp), intent(in) :: cw, Bext, V, T
    real(wp), intent(out) :: X, delta, eta, g
    real(wp) :: q
    eta = Bext / (4.0_wp * V)
    g = 1.0_wp / (1.0_wp - 1.9_wp * eta)
    delta = g * par%kappaW * (exp(par%epsW / T) - 1.0_wp)   ! m3/mol
    q = 2.0_wp * cw * delta
    if (q > 0.0_wp) then
       X = (-1.0_wp + sqrt(1.0_wp + 4.0_wp * q)) / (2.0_wp * q)
    else
       X = 1.0_wp
    end if
  end subroutine solve_chi

  ! Static permittivity, Maribo eqs 13-14. An extension layer may supply
  ! a density factor r_d (Clausius-Mossotti and dipole terms) and a weight
  ! w_d on the water dipole term (ecpa_extensions); both are 1 otherwise,
  ! the published production form.
  subroutine solve_epsr(par, n, V, T, er, eps_inf, X, gw, ierr)
    type(par_t), intent(in) :: par
    real(wp), intent(in) :: n(7), V, T
    real(wp), intent(out) :: er, eps_inf, X, gw
    integer, intent(out) :: ierr
    real(wp) :: c(7), r_d, w_d, M, POL(7), Bext, delta, eta, g
    real(wp) :: Pww, rhs, BV(7)

    c = n / V
    r_d = 1.0_wp
    w_d = 1.0_wp
    if (associated(coll_dielectric_hook)) &
         call coll_dielectric_hook(par, n, V, T, r_d, w_d)
    POL(1) = par%pol1
    POL(2:7) = par%poli
    M = (par%Navo / (3.0_wp * par%eps0)) * dot_product(c, POL) * r_d
    eps_inf = (2.0_wp * M + 1.0_wp) / (1.0_wp - M)
    BV = b_vec(par)
    Bext = dot_product(n, BV)
    call solve_chi(par, c(1), Bext, V, T, X, delta, eta, g)
    Pww = 2.0_wp * c(1) * delta * X ** 2
    gw = 1.0_wp + par%zww * Pww * par%cosGAMMA1 / &
         (Pww * par%cosTHETA1 + 1.0_wp)
    rhs = (par%Navo / (9.0_wp * par%eps0 * par%kb * T)) * c(1) * r_d * &
          w_d * gw * par%dip01 ** 2
    ! scipy brentq(f, eps_inf + 1e-9, 500.0, xtol=1e-12)  (default
    ! rtol = 4*eps, default maxiter = 100)
    call brentq(eps_residual, eps_inf + 1.0e-9_wp, 500.0_wp, 1.0e-12_wp, &
                4.0_wp * epsilon(1.0_wp), 100, er, ierr)

  contains

    function eps_residual(erx) result(y)
      real(wp), intent(in) :: erx
      real(wp) :: y
      y = (2.0_wp * erx + eps_inf) * (erx - eps_inf) / &
          (erx * (eps_inf + 2.0_wp) ** 2) - rhs
    end function eps_residual
  end subroutine solve_epsr

  ! Residual Helmholtz energy / RT (extensive), basis="total".
  subroutine A_r(par, n, V, T, A, ierr)
    type(par_t), intent(in) :: par
    real(wp), intent(in) :: n(7), V, T
    real(wp), intent(out) :: A
    integer, intent(out) :: ierr
    ! NB: the local mixture `a` is `a_hv` here -- plain `a` would clash
    ! with the intent(out) result `A` (Fortran is case-insensitive).
    real(wp) :: nT_all, Bext, a_hv, b_hv, gE, D, A_srk, cw, X, delta, eta, g
    real(wp) :: A_assoc, ni(6), zi(6), er, eps_inf, X2, gw, SnZ2, kap
    real(wp) :: chi_i(6), A_dh_J, A_dh, A_born_J, A_born, BV(7), SG(6), A_x
    logical :: ext_on

    ierr = 0
    BV = b_vec(par)
    nT_all = sum(n)
    Bext = dot_product(n, BV)
    call hv_a_mix(par, n, T, a_hv, b_hv, gE)
    D = a_hv * nT_all ** 2                 ! a n_T^2, basis-independent
    A_srk = -nT_all * log(1.0_wp - Bext / V) - &
            D / (Bext * par%R * T) * log(1.0_wp + Bext / V)

    cw = n(1) / V
    call solve_chi(par, cw, Bext, V, T, X, delta, eta, g)
    A_assoc = n(1) * 4.0_wp * (log(X) - X / 2.0_wp + 0.5_wp)

    ni = n(2:7)
    zi = par%Zi
    SG = par%Sgi
    if (sum(ni) > 0.0_wp) then
       call solve_epsr(par, n, V, T, er, eps_inf, X2, gw, ierr)
       if (ierr /= 0) then
          A = nan_wp()
          return
       end if
       ! Debye-Hueckel (eqs 10-11, limiting-law normalisation)
       SnZ2 = dot_product(ni, zi ** 2)
       kap = sqrt(par%e ** 2 * par%Navo ** 2 * SnZ2 / &
                  (V * par%eps0 * er * par%kb * T * par%Navo))
       chi_i = (1.0_wp / SG ** 3) * (log(1.0_wp + kap * SG) - kap * SG + &
                                     0.5_wp * (kap * SG) ** 2)
       A_dh_J = -(V * par%kb * T / (4.0_wp * acos(-1.0_wp))) * &
                dot_product(ni * zi ** 2, chi_i) / SnZ2
       A_dh = A_dh_J / (par%R * T)
       ! Born (eq 12)
       A_born_J = par%Navo * par%e ** 2 / &
                  (8.0_wp * acos(-1.0_wp) * par%eps0) * &
                  (1.0_wp / er - 1.0_wp) * &
                  dot_product(ni * zi ** 2, 1.0_wp / par%Rbi)
       A_born = A_born_J / (par%R * T)
    else
       A_dh = 0.0_wp
       A_born = 0.0_wp
    end if
    ! extension Helmholtz term (none by default)
    ext_on = .false.
    if (associated(coll_helmholtz_hook)) &
         call coll_helmholtz_hook(par, n, V, T, ext_on, A_x)
    if (ext_on) then
       A = A_srk + A_assoc + A_dh + A_born + A_x
    else
       A = A_srk + A_assoc + A_dh + A_born
    end if
  end subroutine A_r

  ! P = -RT dA^r/dV + n_T RT/V  (central difference, h = V*1e-7).
  subroutine pressure(par, n, V, T, Pout, ierr)
    type(par_t), intent(in) :: par
    real(wp), intent(in) :: n(7), V, T
    real(wp), intent(out) :: Pout
    integer, intent(out) :: ierr
    real(wp) :: h, Ap, Am, dA
    integer :: e1, e2
    h = V * 1.0e-7_wp
    call A_r(par, n, V + h, T, Ap, e1)
    call A_r(par, n, V - h, T, Am, e2)
    ierr = max(e1, e2)
    if (ierr /= 0) then
       Pout = nan_wp()
       return
    end if
    dA = (Ap - Am) / (2.0_wp * h)
    Pout = -par%R * T * dA + sum(n) * par%R * T / V
  end subroutine pressure

  ! np.linspace(start, stop, num) -- reproduced exactly, including
  ! NumPy's "last element is set to stop" rule.
  pure subroutine linspace_np(start, stop_, num, out)
    real(wp), intent(in) :: start, stop_
    integer, intent(in) :: num
    real(wp), intent(out) :: out(num)
    real(wp) :: step
    integer :: i
    if (num <= 0) return
    if (num == 1) then
       out(1) = start
       return
    end if
    step = (stop_ - start) / real(num - 1, wp)
    do i = 1, num
       out(i) = start + step * real(i - 1, wp)
    end do
    out(num) = stop_
  end subroutine linspace_np

  ! Liquid-branch volume root: march out from the co-volume.
  ! ierr: 0 ok; 1 no bracket / brentq failure; 2 A_r failure.
  subroutine solve_v_liquid(par, n, T, Pp, V, ierr)
    type(par_t), intent(in) :: par
    real(wp), intent(in) :: n(7), T, Pp
    real(wp), intent(out) :: V
    integer, intent(out) :: ierr
    real(wp) :: Bext, Vlo, Vhi, f_lo, f_hi, facs(90), BV(7)
    integer :: i, e

    BV = b_vec(par)
    Bext = dot_product(n, BV)
    Vlo = Bext * 1.02_wp
    call pressure(par, n, Vlo, T, f_lo, e)
    if (e /= 0) then
       V = nan_wp(); ierr = 2; return
    end if
    f_lo = f_lo - Pp
    ! np.concatenate([linspace(1.05, 2.5, 60), linspace(2.55, 4.0, 30)])
    call linspace_np(1.05_wp, 2.5_wp, 60, facs(1:60))
    call linspace_np(2.55_wp, 4.0_wp, 30, facs(61:90))
    Vhi = Vlo
    do i = 1, 90
       Vhi = Bext * facs(i)
       call pressure(par, n, Vhi, T, f_hi, e)
       ! The reference has no error branch here: a solve_epsr bracket
       ! failure raises straight out of solve_V (and phi_osm_total
       ! reports nan).  Propagate rather than skipping the rung.
       if (e /= 0) then
          V = nan_wp(); ierr = 2; return
       end if
       if ((f_hi - Pp) * f_lo < 0.0_wp) exit
    end do
    ! scipy brentq(..., xtol=1e-16, rtol=8.9e-16) -- default maxiter 100.
    call brentq(pres_residual, Vlo, Vhi, 1.0e-16_wp, 8.9e-16_wp, 100, V, ierr)
    if (ierr /= 0) V = nan_wp()

  contains

    function pres_residual(Vx) result(y)
      real(wp), intent(in) :: Vx
      real(wp) :: y
      integer :: ee
      call pressure(par, n, Vx, T, y, ee)
      y = y - Pp
    end function pres_residual
  end subroutine solve_v_liquid

  ! ln f_w [ln Pa] = ln(n_w R T / V) + d(A^r/RT)/dn_w|_{T,V}
  subroutine ln_f_w(par, n, T, Pp, lnf, Vout, ierr)
    type(par_t), intent(in) :: par
    real(wp), intent(in) :: n(7), T, Pp
    real(wp), intent(out) :: lnf, Vout
    integer, intent(out) :: ierr
    real(wp) :: V, h, np1(7), nm1(7), Ap, Am, dA
    integer :: e1, e2

    call solve_v_liquid(par, n, T, Pp, V, ierr)
    Vout = V
    if (ierr /= 0) then
       lnf = nan_wp()
       return
    end if
    h = max(n(1) * 1.0e-7_wp, 1.0e-9_wp)
    np1 = n; np1(1) = np1(1) + h
    nm1 = n; nm1(1) = nm1(1) - h
    call A_r(par, np1, V, T, Ap, e1)
    call A_r(par, nm1, V, T, Am, e2)
    ierr = max(e1, e2)
    if (ierr /= 0) then
       lnf = nan_wp()
       return
    end if
    dA = (Ap - Am) / (2.0_wp * h)
    lnf = log(n(1) * par%R * T / V) + dA
  end subroutine ln_f_w

  ! --------------------------------------------------------------
  ! Osmotic coefficient on the total-mole basis.
  !   phi = -55.5084 * [ln f_w(brine) - ln f_w(pure)] / (nu * m)
  ! Returns NaN where the reference returns nan (m <= 0, or a brentq
  ! bracket failure inside solve_V -- the reference catches ValueError
  ! and reports nan rather than patching it).
  !
  ! NOTE: the reference caches ln f_w(pure) per (T, P).
  ! This port recomputes it; the cache is a pure speed device with no
  ! effect on the value (same inputs -> same deterministic result).
  ! --------------------------------------------------------------
  function phi_osm_total(par, salt, m, T, Pp) result(phi)
    type(par_t), intent(in) :: par
    integer, intent(in) :: salt
    real(wp), intent(in) :: m, T, Pp
    real(wp) :: phi
    real(wp) :: lnf_br, lnf_pu, Vb, Vp
    integer :: e1, e2

    if (m <= 0.0_wp) then
       phi = nan_wp()
       return
    end if
    call ln_f_w(par, n_vec(salt, m), T, Pp, lnf_br, Vb, e1)
    call ln_f_w(par, n_vec(SALT_NACL, 0.0_wp), T, Pp, &
                lnf_pu, Vp, e2)
    if (e1 /= 0 .or. e2 /= 0) then
       phi = nan_wp()
       return
    end if
    phi = -N_W * (lnf_br - lnf_pu) / (salt_nu(salt) * m)
  end function phi_osm_total

  ! --------------------------------------------------------------
  ! Brine (CO2-free) mass density -- fig_brine_density.predict_rho_brine
  !   x1w  = 1 / (1 + Mw * sum(mi));  x_aq = (x1w, 0, 0)
  !   v    = Zw R T / P + sum(xiw * Pen_i)          [m3/mol]
  !   rho  = MMw / v                                [kg/m3]
  ! v_init/have_v_init mirror the optional warm start; the cold seed and
  ! the 6-rung fallback ladder are _solve_Zw's, NOT lnphi_aq's 30-rung
  ! ladder (they differ -- see PORTING_NOTES.md 14).
  ! --------------------------------------------------------------
  subroutine rho_brine(par, T, Pbar, mi, have_v_init, v_init, rho, v_out, ok)
    type(par_t), intent(in) :: par
    real(wp), intent(in) :: T, Pbar, mi(6), v_init(3)
    logical, intent(in) :: have_v_init
    real(wp), intent(out) :: rho, v_out(3)
    logical, intent(out) :: ok
    real(wp) :: x1w, x_aq(3), P_Pa, Pen_i(6), xiw(6), MMw, v, v0(3)
    real(wp) :: seeds(3, 13), v_try(3), rho_try, rho_ref, Zw_liq
    type(rextra_t) :: ex
    integer :: nit, k
    logical :: okn
    ! Liquid-branch band, mirroring ms_ecpa/roots.py: a gas-free brine is a
    ! compressed liquid, so it cannot be far from pure water's saturated
    ! liquid density at the same T. Loose on purpose -- it rejects roots wrong
    ! by orders of magnitude, not by per cent.
    real(wp), parameter :: LIQUID_LO = 0.5_wp, LIQUID_HI = 3.0_wp

    rho = nan_wp()
    v_out = nan_wp()
    ok = .false.
    x1w = 1.0_wp / (1.0_wp + par%Mw * sum(mi))
    x_aq = [x1w, 0.0_wp, 0.0_wp]
    P_Pa = Pbar * 1.0e5_wp

    ex%comp = x_aq
    ex%mi = mi
    ex%T = T
    ex%P = P_Pa
    if (have_v_init) then
       v0 = v_init
    else
       v0 = [P_Pa * 18.015e-6_wp / (8.314_wp * T), 50.0_wp, 0.10_wp]
    end if
    ! The seed ladder used to run ONLY when the Newton failed, and the first
    ! converged root was accepted with no test of which branch it was on. The
    ! three-variable closure has several roots: in the Python twin of this
    ! routine that accepted 0.8-5.5 kg/m3 as a brine density at 200-500 bar,
    ! where the measurement is 1100-1215. Convergence is not correctness. Every
    ! seed is now tried in turn and the first root that is actually a LIQUID is
    ! taken.
    seeds(:, 1) = v0
    seeds(:, 2) = [0.07_wp, 60.0_wp, 0.13_wp]
    seeds(:, 3) = [0.10_wp, 30.0_wp, 0.05_wp]
    seeds(:, 4) = [0.30_wp, 70.0_wp, 0.20_wp]
    seeds(:, 5) = [0.20_wp, 40.0_wp, 0.10_wp]
    seeds(:, 6) = [0.50_wp, 50.0_wp, 0.10_wp]
    seeds(:, 7) = [0.03_wp, 78.0_wp, 0.07_wp]
    ! The COLD default, as an 8th seed. When a warm start is supplied it takes
    ! slot 1 and this basin is otherwise never tried -- and exactly one brine
    ! density state in the validation set converges from here and from nowhere
    ! else. It used to be reached only because the Python fallback restarted
    ! cold; with that fallback removed for speed, the seed has to be here.

    Pen_i = (par%A_Peni * T / par%Tc1 + par%B_Peni) * 1.0e-6_wp
    xiw = x1w * par%Mw * mi
    MMw = par%Mw * x1w + par%Mc * 0.0_wp + sum(par%MMi * xiw)
    rho_ref = rho_w_sat(T) * par%Mw        ! mol/m3 -> kg/m3

    seeds(:, 8) = [0.05_wp, 50.0_wp, 0.10_wp]
    ! Pressure-scaled liquid seeds. Every rung above is a FIXED Zw, but a
    ! liquid's Zw = P v /(R T) MOVES with pressure: ~0.0065 at 9 bar against
    ! ~0.05 at 70 bar. Below ~20 bar the entire fixed ladder sits far above
    ! the liquid basin and never finds it -- 100 of the 105 unsolved
    ! brine-density states are under 20 bar, at every temperature. lnphi_aq
    ! already seeds its own inner Newton with exactly this quantity.
    Zw_liq = P_Pa * 18.015e-6_wp / (8.314_wp * T)
    seeds(:, 9)  = [Zw_liq,           50.0_wp, 0.10_wp]
    seeds(:, 10) = [0.85_wp * Zw_liq, 60.0_wp, 0.13_wp]
    seeds(:, 11) = [1.25_wp * Zw_liq, 40.0_wp, 0.07_wp]
    ! Low-permittivity rungs for concentrated brine: ions suppress eps_r to
    ! about 30 at 5 molal MgCl2, so a scaled-Zw rung paired with eps_r near
    ! pure water's 78 is in the wrong basin in two coordinates at once.
    seeds(:, 12) = [1.5_wp * Zw_liq, 30.0_wp, 0.10_wp]
    seeds(:, 13) = [0.4_wp * Zw_liq, 30.0_wp, 0.05_wp]

    do k = 1, 13
       call newton_solve(par, ex, aq_residual_3var, 3, seeds(:, k), &
                         1.0e-10_wp, 80, v_try, okn, nit)
       if (.not. okn) cycle
       ! chi1w is a monomer fraction: never negative. The 3x3 Newton can
       ! settle on the negative branch of the association closure and the
       ! implied density can still fall inside the liquid bounds tested
       ! below, so that test does not catch it. Mirrors the chi1w > 0
       ! guard in _solve_Zw / lnphi_aq.
       if (.not. (v_try(3) > 0.0_wp)) cycle
       v = v_try(1) * par%R * T / P_Pa + sum(xiw * Pen_i)
       if (v <= 0.0_wp) cycle
       rho_try = MMw / v
       if (.not. ieee_is_finite(rho_try)) cycle
       if (rho_try < LIQUID_LO * rho_ref) cycle
       if (rho_try > LIQUID_HI * rho_ref) cycle
       v_out = v_try
       rho = rho_try
       ok = .true.
       return
    end do

    ! No seed landed on the liquid branch. NaN is the honest answer.
    v_out = nan_wp()
    rho = nan_wp()
  end subroutine rho_brine

  ! --------------------------------------------------------------
  ! Brine vapor (bubble) pressure from water isofugacity:
  !   ln x_w + lnphi_w^aq(T,P) = lnphi_w^gas(T,P),  Brent on ln P.
  ! NaCl only (mi = (m, m, 0, 0, 0, 0)), exactly as the reference.
  ! p0_Pa is the pure-water saturation pressure the reference obtains
  ! from CoolProp; supply the same value for an identical bracket.
  ! Returns P in BAR (the reference divides by 1e5), NaN on failure.
  ! --------------------------------------------------------------
  function psat_brine(par, T, m, p0_Pa, lo_fac, hi_fac) result(Pbar)
    type(par_t), intent(in) :: par
    real(wp), intent(in) :: T, m, p0_Pa
    real(wp), intent(in), optional :: lo_fac, hi_fac
    real(wp) :: Pbar
    real(wp) :: mi(6), x_w, x_aq(3), y(3), lo, hi, lnP, flo, fhi
    real(wp) :: lof, hif
    ! Fraction of the bracket WIDTH within which a returned root counts as
    ! sitting on an edge rather than inside.  Brent stops at rtol = 1e-10
    ! on ln P, so an edge return lands ~1e-9 of the width from it; the
    ! whole psat golden grid (323.15-573.15 K) sits at least 0.10 of the
    ! width from either edge.  1e-6 is three orders above the first and
    ! five below the second.
    real(wp), parameter :: EDGE_MARGIN = 1.0e-6_wp
    integer :: ierr

    lof = 0.50_wp                ! vpsat_newphys.py psat_model
    hif = 1.10_wp
    if (present(lo_fac)) lof = lo_fac    ! 0.55 -> vpsat_salt_2026_08
    if (present(hi_fac)) hif = hi_fac

    mi = [m, m, 0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp]
    x_w = 1.0_wp / (1.0_wp + par%Mw * 2.0_wp * m)
    x_aq = [x_w, 0.0_wp, 0.0_wp]
    y = [1.0_wp, 0.0_wp, 0.0_wp]

    lo = log(lof * p0_Pa)
    hi = log(hif * p0_Pa)
    flo = iso_residual(lo)
    fhi = iso_residual(hi)
    if (.not. (ieee_is_finite(flo) .and. ieee_is_finite(fhi))) then
       Pbar = nan_wp()
       return
    end if
    if (flo * fhi > 0.0_wp) then
       Pbar = nan_wp()          ! the reference's brentq raises -> caught
       return
    end if
    ! scipy brentq(f, lo, hi, xtol=1e-12, rtol=1e-10) -- maxiter 100.
    call brentq(iso_residual, lo, hi, 1.0e-12_wp, 1.0e-10_wp, 100, lnP, ierr)
    if (ierr /= 0) then
       Pbar = nan_wp()
       return
    end if
    ! A root ON a bracket edge is an artefact, not a saturation pressure.
    ! Below ~300 K the aqueous inner Newton stalls just above its 1e-10
    ! tolerance and lnphi_aq falls through its ladder onto the VAPOUR root
    ! of the aqueous mixture; both sides of the isofugacity balance are
    ! then the same phase, so the residual collapses to |f| ~ 1e-19..1e-12
    ! with an arbitrary sign.  When such a point sits at lo or hi, Brent
    ! converges onto the edge and 0.50*p0 is returned as a pressure.  NaN
    ! is this routine's failure value (the C API maps it to rc = 1), so an
    ! ill-posed bracket reports as ill-posed instead of as a number.  The
    ! bracket VERIFICATION of ms_ecpa.psat.solve_psat_isofugacity stays
    ! unported on purpose -- PORTING_NOTES 17a.
    if (min(lnP - lo, hi - lnP) <= EDGE_MARGIN * (hi - lo)) then
       Pbar = nan_wp()
       return
    end if
    Pbar = exp(lnP) / 1.0e5_wp

  contains

    function iso_residual(lnPx) result(r)
      real(wp), intent(in) :: lnPx
      real(wp) :: r
      real(wp) :: P, la(3), lg(3), dum3(3), dum2(2)
      logical :: oka, okg
      integer :: nit
      P = exp(lnPx)
      call lnphi_aq(par, x_aq, mi, T, P, la, dum3, oka, nit)
      if (.not. oka) then
         r = nan_wp()
         return
      end if
      call lnphi_gas(par, y, T, P, -1.0_wp, 0.99_wp, lg, dum2, okg)
      if (.not. okg) then
         r = nan_wp()
         return
      end if
      r = (log(x_w) + la(1)) - lg(1)
    end function iso_residual
  end function psat_brine

end module ecpa_colligative
