! ecpa_lnphi.f90 -- per-phase log-fugacity-coefficient kernels.
!
! Faithful transcription of ms_ecpa/lnphi.py:
!   * aqueous 3-variable inner Newton on (Zw, eps_r, chi1w) with
!     analytically solved chi-derivative fixed points,
!   * gas 2-variable inner Newton on (Zh, chi1h),
!   * the exact cold-start seed ladders of the Python drivers.
!
! All residual/state algebra is written in complex(real64) so ONE code
! path serves both the real evaluation (zero imaginary parts) and the
! complex-step Jacobian (see ecpa_newton.f90 header for the fidelity
! argument).  Expression order mirrors the Python source line by line.
!
! Units: T [K], P [Pa] (SI -- callers multiply bar by 1e5).
module ecpa_lnphi
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use ecpa_kinds, only: wp
  use ecpa_params, only: par_t
  use ecpa_newton, only: rextra_t, newton_solve, nan_wp
  use ecpa_extensions, only: aq_terms_t, aq_terms_hook, aq_seeds_hook, &
                             MAX_EXTRA_SEEDS
  implicit none
  private
  public :: aq_residual_3var, gas_residual, aq_lnphi_evaluate_3var, &
            gas_lnphi_evaluate, lnphi_aq, lnphi_gas, aq_default_seed, &
            aq_seed_ladder_lnphi, aq_seed_ladder_flash, zw_liq_seed, &
            gas_root_is_physical, gas_chi_positive_branch, &
            gas_accept_or_repair, LIQUID_EPSR_MIN

  real(wp), parameter :: pi = 3.14159265358979323846_wp

  ! Cached aqueous intermediates shared by residual / chi-derivs / lnphi.
  type :: aqstate_t
     real(wp) :: x1w = 0, x2w = 0, x3w = 0
     real(wp) :: xiw(6) = 0
     real(wp) :: cB1 = 0, cB2 = 0, cB3 = 0, cBi(6) = 0, Bcap = 0, Acap = 0
     real(wp) :: b_mix = 0, gE = 0, S13 = 0, xiZi = 0
     logical  :: has_ions = .false.
     complex(wp) :: rho = 0, rho1 = 0
     complex(wp) :: Zphys = 0, Zassoc = 0, ZDH = 0
     complex(wp) :: lnPHI1phys = 0, lnPHI2phys = 0, lnPHI3phys = 0
     complex(wp) :: lnPHI1assoc = 0, lnPHI2assoc = 0, lnPHI3assoc = 0
     complex(wp) :: eta = 0, g_eta = 0, dg_deta = 0
     complex(wp) :: delta = 0, DELTAc = 0
     complex(wp) :: chi3w = 0, chi2w = 0, chi1w_new = 0
     real(wp) :: S12A = 0
     complex(wp) :: M = 0, eps_inf = 0, r_d = 1.0_wp
     complex(wp) :: Pww = 0, Pwc = 0, Pwh = 0, Pw = 0, gw = 0
     ! extension terms (ecpa_extensions); neutral values by default
     complex(wp) :: w_d = 1.0_wp, VdwdV = 0, NdwdN1 = 0
     complex(wp) :: Z_ext = 0, lnphi_ext(3) = 0
     complex(wp) :: T1 = 0, T2 = 0
     complex(wp) :: debye = 0
  end type aqstate_t

  ! Least relative permittivity that can belong to an AQUEOUS phase.
  ! eps_r > 0 rejects a negative permittivity but not a VAPOUR: the seed
  ! ladder can converge on Zw ~ 0.5, eps_r ~ 1.0003, chi ~ 0.998 and that
  ! passes "positive". Water and brine sit between about 25 and 80 over the
  ! range this model covers, so 5.0 is far below any real aqueous state and
  ! far above any vapour. Mirrors LIQUID_EPSR_MIN in ms_ecpa/lnphi.py.
  real(wp), parameter :: LIQUID_EPSR_MIN = 5.0_wp

contains

  ! ------------------------------------------------------------------
  ! T-only mixture helpers (never complex-step perturbed).
  ! ------------------------------------------------------------------
  subroutine pure_a_coeffs(par, T, a1, a2, a3)
    type(par_t), intent(in) :: par
    real(wp), intent(in) :: T
    real(wp), intent(out) :: a1, a2, a3
    a1 = par%a01 * (1.0_wp + par%c11 * (1.0_wp - sqrt(T / par%Tc1)))**2
    a2 = par%a02 * (1.0_wp + par%c12 * (1.0_wp - sqrt(T / par%Tc2)))**2
    a3 = par%a03 * (1.0_wp + par%c13 * (1.0_wp - sqrt(T / par%Tc3)))**2
  end subroutine pure_a_coeffs

  subroutine kij_s13(par, T, k12, k13, S13)
    type(par_t), intent(in) :: par
    real(wp), intent(in) :: T
    real(wp), intent(out) :: k12, k13, S13
    k13 = par%Ak13 * (T / par%Tc3)**2 + par%Bk13 * (T / par%Tc3) + par%Ck13
    S13 = par%AS13 * (T / par%Tc3)**2 + par%BS13 * (T / par%Tc3) + par%CS13
    ! m-CR1: the cross-association energy is eps_water/2, not eps_water. The
    ! cross strength is formed downstream as S13 * DELTA_water, and DELTA
    ! carries exp(epsW/T)-1, so rescaling S13 by the ratio of the two
    ! exponentials makes the product carry exp(epsW/2T)-1 instead. Identical
    ! algebra to ms_ecpa.parameters.s13_of_T; see that docstring for why.
    if (par%MCR1_CROSS_ENERGY /= 0) then
       ! S13 * DELTA must equal Q * (exp(epsW/2T)-1) * (the rest of DELTA),
       ! and DELTA carries (exp(epsW/T)-1), so the factor is half-over-full.
       ! Written the other way up first, which made the cross strength ~40x
       ! too large at 273 K and put this kernel 31-99 % away from Python.
       S13 = S13 * (exp(0.5_wp * par%epsW / T) - 1.0_wp) &
                 / (exp(par%epsW / T) - 1.0_wp)
    end if
    k12 = par%Ak12 * (T / par%Tc1)**2 + par%Bk12 * (T / par%Tc1) + par%Ck12
  end subroutine kij_s13

  ! Gas-ion Huron-Vidal energies with per-slot functional form.
  ! slot = 2 (CH4) or 3 (CO2).
  subroutine u_gas_ion(par, slot, T, U)
    type(par_t), intent(in) :: par
    integer, intent(in) :: slot
    real(wp), intent(in) :: T
    real(wp), intent(out) :: U(6)
    integer :: i
    if (slot == 2) then
       do i = 1, 6
          if (par%U2i_form(i) == 0) then
             U(i) = par%A_U2i(i) * (T / par%Tc2) + par%B_U2i(i)
          else
             U(i) = par%U2i_quad_Uref(i) + par%U2i_quad_alfa(i) * par%R * &
                    ((1.0_wp - T / par%U2i_quad_Talfa(i))**2 &
                     - (1.0_wp - par%Tref / par%U2i_quad_Talfa(i))**2)
          end if
       end do
    else
       do i = 1, 6
          if (par%U3i_form(i) == 0) then
             U(i) = par%A_U3i(i) * (T / par%Tc3) + par%B_U3i(i)
          else
             U(i) = par%U3i_quad_Uref(i) + par%U3i_quad_alfa(i) * par%R * &
                    ((1.0_wp - T / par%U3i_quad_Talfa(i))**2 &
                     - (1.0_wp - par%Tref / par%U3i_quad_Talfa(i))**2)
          end if
       end do
    end if
  end subroutine u_gas_ion

  ! ------------------------------------------------------------------
  ! Aqueous core state (mirrors _aq_core_state).
  ! ------------------------------------------------------------------
  subroutine aq_core_state(par, Zw, epsr, chi1w, x_aq, mi, T, P, st)
    type(par_t), intent(in) :: par
    complex(wp), intent(in) :: Zw, epsr, chi1w
    real(wp), intent(in) :: x_aq(3), mi(6), T, P
    type(aqstate_t), intent(out) :: st

    real(wp) :: x1w, x2w, x3w, xiw(6), ai_loc(6)
    real(wp) :: a1, a2, a3, k12, k13, S13
    real(wp) :: U12, U21, U13, U31, U23, U32, U1i(6), U2i(6), U3i(6)
    real(wp) :: cA1, cB1, cA2, cB2, cA3, cB3, cBi(6)
    real(wp) :: b, gE, a, Acap, Bcap, ln2, xiZi
    complex(wp) :: rho, Zphys, eta, g_eta, dg_deta, delta, DELTAc
    complex(wp) :: chi3w, chi2w, chi1w_new, Zassoc, assoc_sum
    complex(wp) :: debye, ZDH, Xis
    complex(wp) :: rho1, M, eps_inf
    complex(wp) :: Pww, Pwc, Pwh, Pw, gw
    type(aq_terms_t) :: ext
    logical :: ext_on
    complex(wp) :: T1, T2
    real(wp) :: S12A_
    logical :: has_ions
    integer :: i

    ln2 = log(2.0_wp)
    x1w = x_aq(1); x2w = x_aq(2); x3w = x_aq(3)
    xiw = x1w * par%Mw * mi
    ai_loc = par%a0i * (1.0_wp + par%c1i * (1.0_wp - sqrt(T / par%Tref)))**2

    call pure_a_coeffs(par, T, a1, a2, a3)
    call kij_s13(par, T, k12, k13, S13)
    U12 = ln2 * (a2 / par%b2 - 2.0_wp * sqrt(a1 * a2) * (1.0_wp - k12) / (par%b1 + par%b2))
    U21 = ln2 * (a1 / par%b1 - 2.0_wp * sqrt(a2 * a1) * (1.0_wp - k12) / (par%b2 + par%b1))
    U13 = ln2 * (a3 / par%b3 - 2.0_wp * sqrt(a1 * a3) * (1.0_wp - k13) / (par%b1 + par%b3))
    U31 = ln2 * (a1 / par%b1 - 2.0_wp * sqrt(a3 * a1) * (1.0_wp - k13) / (par%b3 + par%b1))
    U23 = ln2 * (a3 / par%b3 - 2.0_wp * sqrt(a2 * a3) * (1.0_wp - par%k23) / (par%b2 + par%b3))
    U32 = ln2 * (a2 / par%b2 - 2.0_wp * sqrt(a3 * a2) * (1.0_wp - par%k23) / (par%b3 + par%b2))
    U1i = par%Uref1i + par%alfa1i * par%R * ((1.0_wp - T / par%Talfa1i)**2 &
          - (1.0_wp - par%Tref / par%Talfa1i)**2)
    call u_gas_ion(par, 2, T, U2i)
    call u_gas_ion(par, 3, T, U3i)

    cA1 = a1 * P / par%R**2 / T**2
    cB1 = par%b1 * P / par%R / T
    cA2 = a2 * P / par%R**2 / T**2
    cB2 = par%b2 * P / par%R / T
    cA3 = a3 * P / par%R**2 / T**2
    cB3 = par%b3 * P / par%R / T
    cBi = par%bi * P / par%R / T

    rho = P / Zw / par%R / T
    b = par%b1 * x1w + par%b2 * x2w + par%b3 * x3w + sum(par%bi * xiw)
    gE = 1.0_wp / b * (x1w * x2w * (par%b1 * U12 + par%b2 * U21) &
         + x1w * x3w * (par%b1 * U13 + par%b3 * U31) &
         + x2w * x3w * (par%b2 * U23 + par%b3 * U32) &
         + x1w * sum(xiw * U1i * (par%b1 + par%bi)) &
         + x2w * sum(xiw * U2i * (par%b2 + par%bi)) &
         + x3w * sum(xiw * U3i * (par%b3 + par%bi)))
    if (any(par%Uref_ii /= 0.0_wp)) then
       gE = gE + dot_product(xiw, matmul(par%Uref_ii, par%bi * xiw)) / b
    end if
    a = b * (x1w * a1 / par%b1 + x2w * a2 / par%b2 + x3w * a3 / par%b3 &
        + sum(xiw * ai_loc / par%bi) - gE / ln2)
    Acap = a * P / par%R**2 / T**2
    Bcap = b * P / par%R / T

    Zphys = Zw / (Zw - Bcap) - Acap / (Zw + Bcap)
    st%lnPHI1phys = (-log(Zw - Bcap) + cB1 / Bcap * (Bcap / (Zw - Bcap) - Acap / (Zw + Bcap)) &
         - log((Zw + Bcap) / Zw) &
         * (cA1 / cB1 - 1.0_wp / (Bcap * ln2) &
            * (x2w / par%R / T * (cB1 * U12 + cB2 * U21) &
               + x3w / par%R / T * (cB1 * U13 + cB3 * U31) &
               + sum(xiw * U1i / par%R / T * (cB1 + cBi)) &
               - cB1 * gE / par%R / T)))
    st%lnPHI2phys = (-log(Zw - Bcap) + cB2 / Bcap * (Bcap / (Zw - Bcap) - Acap / (Zw + Bcap)) &
         - log((Zw + Bcap) / Zw) &
         * (cA2 / cB2 - 1.0_wp / (Bcap * ln2) &
            * (x1w / par%R / T * (cB1 * U12 + cB2 * U21) &
               + x3w / par%R / T * (cB2 * U23 + cB3 * U32) &
               + sum(xiw * U2i / par%R / T * (cB2 + cBi)) &
               - cB2 * gE / par%R / T)))
    st%lnPHI3phys = (-log(Zw - Bcap) + cB3 / Bcap * (Bcap / (Zw - Bcap) - Acap / (Zw + Bcap)) &
         - log((Zw + Bcap) / Zw) &
         * (cA3 / cB3 - 1.0_wp / (Bcap * ln2) &
            * (x1w / par%R / T * (cB1 * U13 + cB3 * U31) &
               + x2w / par%R / T * (cB3 * U32 + cB2 * U23) &
               + sum(xiw * U3i / par%R / T * (cB3 + cBi)) &
               - cB3 * gE / par%R / T)))

    ! Association
    S12A_ = par%S12A
    eta = Bcap / 4.0_wp / Zw
    g_eta = 1.0_wp / (1.0_wp - 1.9_wp * eta)
    dg_deta = 1.9_wp / (1.0_wp - 1.9_wp * eta)**2
    delta = (g_eta * par%kappaW) * (exp(par%epsW / T) - 1.0_wp)
    DELTAc = delta * P / par%R / T
    chi3w = Zw / (Zw + 2.0_wp * x1w * chi1w * S13 * DELTAc)
    chi2w = Zw / (Zw + 2.0_wp * x1w * chi1w * S12A_ * DELTAc)
    chi1w_new = Zw / (Zw + 2.0_wp * x1w * chi1w * DELTAc &
                + 2.0_wp * x3w * chi3w * S13 * DELTAc &
                + 2.0_wp * x2w * chi2w * S12A_ * DELTAc)
    Zassoc = -2.0_wp * (1.0_wp + eta / g_eta * dg_deta) &
             * (x1w * (1.0_wp - chi1w) + x3w * (1.0_wp - chi3w) &
                + x2w * (1.0_wp - chi2w))
    assoc_sum = (x1w * (chi1w - 1.0_wp) + x3w * (chi3w - 1.0_wp) &
                 + x2w * (chi2w - 1.0_wp))
    st%lnPHI1assoc = 4.0_wp * log(chi1w) + cB1 / (2.0_wp * g_eta * Zw) * dg_deta * assoc_sum
    st%lnPHI2assoc = 4.0_wp * log(chi2w) + cB2 / (2.0_wp * g_eta * Zw) * dg_deta * assoc_sum
    st%lnPHI3assoc = 4.0_wp * log(chi3w) + cB3 / (2.0_wp * g_eta * Zw) * dg_deta * assoc_sum

    ! Debye-Hueckel
    xiZi = sum(xiw * par%Zi**2)
    has_ions = (sum(mi) > 0.0_wp)
    if (has_ions) then
       debye = (par%e**2 * par%Navo * rho * xiZi / (par%kb * T * epsr * par%eps0))**0.5_wp
       ZDH = (0.0_wp, 0.0_wp)
       do i = 1, 6
          Xis = 1.0_wp / par%Sgi(i)**3 * (log(1.0_wp + debye * par%Sgi(i)) &
                - debye * par%Sgi(i) + 0.5_wp * (debye * par%Sgi(i))**2)
          ZDH = ZDH + xiw(i) * par%Zi(i)**2 &
                * (Xis - 0.5_wp * debye**3 / (1.0_wp + debye * par%Sgi(i)))
       end do
       ZDH = 1.0_wp / (4.0_wp * pi * par%Navo * rho * xiZi) * ZDH
    else
       debye = (0.0_wp, 0.0_wp)
       ZDH = (0.0_wp, 0.0_wp)
    end if

    ! Permittivity (Maribo-Mogensen)
    rho1 = rho * x1w
    ! Extension terms (ecpa_extensions); neutral values (r_d = w_d = 1,
    ! derivatives, Z and lnphi additions 0) unless an extension is active.
    ext = aq_terms_t()
    if (has_ions .and. associated(aq_terms_hook)) then
       call aq_terms_hook(par, T, rho, x1w, x2w, x3w, xiw, ext_on, ext)
       if (.not. ext_on) ext = aq_terms_t()
    end if
    M = (par%Navo * rho * ext%r_d / (3.0_wp * par%eps0) &
         * (x1w * par%pol1 + x2w * par%pol2 + x3w * par%pol3 + sum(xiw * par%poli)))
    eps_inf = (2.0_wp * M + 1.0_wp) / (1.0_wp - M)
    Pww = 2.0_wp * rho * x1w * delta * chi1w**2
    Pwc = 2.0_wp * rho * x3w * S13 * delta * chi1w * chi3w
    Pwh = 2.0_wp * rho * x2w * S12A_ * delta * chi1w * chi2w
    Pw = Pww + Pwc + Pwh
    gw = 1.0_wp + par%zww * Pww * par%cosGAMMA1 / (Pw * par%cosTHETA1 + 1.0_wp)

    T1 = (2.0_wp * epsr + eps_inf) * (epsr - eps_inf) / (epsr * (eps_inf + 2.0_wp)**2)
    T2 = par%Navo * rho * ext%r_d / (9.0_wp * par%eps0 * par%kb * T) &
         * (x1w * ext%w * gw * par%dip01**2)

    st%x1w = x1w; st%x2w = x2w; st%x3w = x3w; st%xiw = xiw
    st%cB1 = cB1; st%cB2 = cB2; st%cB3 = cB3; st%cBi = cBi
    st%Bcap = Bcap; st%Acap = Acap
    st%b_mix = b; st%gE = gE; st%S13 = S13; st%xiZi = xiZi
    st%has_ions = has_ions
    st%rho = rho; st%rho1 = rho1
    st%Zphys = Zphys; st%Zassoc = Zassoc; st%ZDH = ZDH
    st%eta = eta; st%g_eta = g_eta; st%dg_deta = dg_deta
    st%delta = delta; st%DELTAc = DELTAc
    st%chi3w = chi3w; st%chi2w = chi2w; st%chi1w_new = chi1w_new
    st%S12A = S12A_
    st%M = M; st%eps_inf = eps_inf; st%r_d = ext%r_d
    st%Pww = Pww; st%Pwc = Pwc; st%Pwh = Pwh; st%Pw = Pw; st%gw = gw
    st%w_d = ext%w; st%VdwdV = ext%VdwdV; st%NdwdN1 = ext%NdwdN1
    st%Z_ext = ext%Z; st%lnphi_ext = ext%lnphi
    st%T1 = T1; st%T2 = T2
    st%debye = debye
  end subroutine aq_core_state

  ! ------------------------------------------------------------------
  ! Analytic chi-derivative fixed points (mirrors _aq_chi_derivatives).
  ! ------------------------------------------------------------------
  subroutine aq_chi_derivatives(par, st, chi1w, x_aq, &
                                Vdchi1WdV, Ndchi1WdNw, Ndchi1WdNc, Ndchi1WdNh)
    type(par_t), intent(in) :: par
    type(aqstate_t), intent(in) :: st
    complex(wp), intent(in) :: chi1w
    real(wp), intent(in) :: x_aq(3)
    complex(wp), intent(out) :: Vdchi1WdV, Ndchi1WdNw, Ndchi1WdNc, Ndchi1WdNh

    real(wp) :: x1w, x2w, x3w, S13, S12A_, b
    complex(wp) :: delta, chi3w, g_eta, dg_deta, eta, rho, chi2w
    complex(wp) :: dFdchiw, dFdchic, dFdchih, dGdchiw, dHdchiw
    complex(wp) :: VddeltadV, dFddelta, dGddelta, dHddelta
    complex(wp) :: alpha, beta_V, beta_Nw, beta_Nc, beta_Nh
    complex(wp) :: dFdV, dGdV, dHdV, dFdNw, dGdNw, dHdNw
    complex(wp) :: NddeltadN1, NddeltadN3, NddeltadN2
    complex(wp) :: dFdNc, dFdNh

    x1w = x_aq(1); x2w = x_aq(2); x3w = x_aq(3)
    delta = st%delta; chi3w = st%chi3w; g_eta = st%g_eta
    dg_deta = st%dg_deta; eta = st%eta; b = st%b_mix
    rho = st%rho; S13 = st%S13
    chi2w = st%chi2w; S12A_ = st%S12A

    ! Each row carries chi_new**2 of its OWN residual (F -> chi1w, G ->
    ! chi3w, H -> chi2w), and d/ddelta is +chi**2 * dQ/ddelta.  Corrected
    ! 2026-09 in lockstep with ms_ecpa/lnphi.py; ground truth in
    ! tests/test_chi_chain_derivatives.py.
    dFdchiw = 1.0_wp + 2.0_wp * x1w * rho * delta * chi1w**2
    dFdchic = 2.0_wp * x3w * rho * S13 * delta * chi1w**2
    dFdchih = 2.0_wp * x2w * rho * S12A_ * delta * chi1w**2
    dGdchiw = 2.0_wp * x1w * rho * S13 * delta * chi3w**2
    dHdchiw = 2.0_wp * x1w * rho * S12A_ * delta * chi2w**2
    VddeltadV = -delta * eta * dg_deta / g_eta
    dFddelta = chi1w**2 * (2.0_wp * rho * x1w * chi1w &
                           + 2.0_wp * rho * x3w * chi3w * S13 &
                           + 2.0_wp * rho * x2w * chi2w * S12A_)
    dGddelta = chi3w**2 * (2.0_wp * rho * x1w * chi1w * S13)
    dHddelta = chi2w**2 * (2.0_wp * rho * x1w * chi1w * S12A_)

    alpha = (dFdchic * dGdchiw + dFdchih * dHdchiw) / dFdchiw

    dFdV = -(2.0_wp * rho * x1w * delta * chi1w + 2.0_wp * rho * x3w * S13 * delta * chi3w &
           + 2.0_wp * rho * x2w * S12A_ * delta * chi2w) * chi1w**2
    dGdV = -(2.0_wp * rho * x1w * S13 * delta * chi1w) * chi3w**2
    dHdV = -(2.0_wp * rho * x1w * S12A_ * delta * chi1w) * chi2w**2
    beta_V = (dFdchic * (dGdV + dGddelta * VddeltadV) &
              + dFdchih * (dHdV + dHddelta * VddeltadV) &
              - dFdV - dFddelta * VddeltadV) / dFdchiw
    Vdchi1WdV = beta_V / (1.0_wp - alpha)

    NddeltadN1 = delta * eta * par%b1 * dg_deta / (g_eta * b)
    dFdNw = 2.0_wp * rho * delta * chi1w**3
    dGdNw = 2.0_wp * rho * S13 * delta * chi1w * chi3w**2
    dHdNw = 2.0_wp * rho * S12A_ * delta * chi1w * chi2w**2
    beta_Nw = (dFdchic * (dGdNw + dGddelta * NddeltadN1) &
               + dFdchih * (dHdNw + dHddelta * NddeltadN1) &
               - dFdNw - dFddelta * NddeltadN1) / dFdchiw
    Ndchi1WdNw = beta_Nw / (1.0_wp - alpha)

    if (x3w > 0.0_wp) then
       NddeltadN3 = delta * eta * par%b3 * dg_deta / (g_eta * b)
       dFdNc = 2.0_wp * rho * S13 * delta * chi1w**2 * chi3w
       beta_Nc = (dFdchic * (0.0_wp + dGddelta * NddeltadN3) &
                  + dFdchih * (0.0_wp + dHddelta * NddeltadN3) &
                  - dFdNc - dFddelta * NddeltadN3) / dFdchiw
       Ndchi1WdNc = beta_Nc / (1.0_wp - alpha)
    else
       Ndchi1WdNc = (1.0_wp, 0.0_wp)
    end if

    if (x2w > 0.0_wp) then
       NddeltadN2 = delta * eta * par%b2 * dg_deta / (g_eta * b)
       dFdNh = 2.0_wp * rho * S12A_ * delta * chi1w**2 * chi2w
       beta_Nh = (dFdchic * (0.0_wp + dGddelta * NddeltadN2) &
                  + dFdchih * (0.0_wp + dHddelta * NddeltadN2) &
                  - dFdNh - dFddelta * NddeltadN2) / dFdchiw
       Ndchi1WdNh = beta_Nh / (1.0_wp - alpha)
    else
       Ndchi1WdNh = (1.0_wp, 0.0_wp)
    end if
  end subroutine aq_chi_derivatives

  ! ------------------------------------------------------------------
  ! Three-equation aqueous residual in (Zw, epsr, chi1w)
  ! (mirrors _aq_residual_3var; complex-generic).
  ! ------------------------------------------------------------------
  subroutine aq_residual_3var(par, ex, n, v, f)
    type(par_t), intent(in) :: par
    type(rextra_t), intent(in) :: ex
    integer, intent(in) :: n
    complex(wp), intent(in) :: v(n)
    complex(wp), intent(out) :: f(n)

    type(aqstate_t) :: st
    complex(wp) :: Zw, epsr, chi1w
    complex(wp) :: Vdchi1WdV, Ndchi1WdNw, Ndchi1WdNc, Ndchi1WdNh
    real(wp) :: x1w, x2w, x3w, S13, S12A_, b
    complex(wp) :: M, eps_inf, rho, rho1, Pw, Pww, Pwc, Pwh, gw, delta
    complex(wp) :: chi3w, g_eta, dg_deta, eta, chi2w, debye
    real(wp) :: xiZi
    complex(wp) :: VddeltadV, dGdchiw, dGddelta, dGdV, Vdchi3WdV
    complex(wp) :: dHdchiw, dHddelta, dHdV, Vdchi2WdV
    complex(wp) :: dgwdPw, dgwdPww, VdPwwdV, VdPwcdV, VdPwhdV, VdPwdV, VdgwdV
    complex(wp) :: dFder, dFdeinf, r_d, VdeinfdV
    complex(wp) :: w_d, VdwdV
    complex(wp) :: VdFdV, VderdV, daDHBder, ZPerm, Zw_new

    Zw = v(1); epsr = v(2); chi1w = v(3)
    call aq_core_state(par, Zw, epsr, chi1w, ex%comp, ex%mi, ex%T, ex%P, st)
    call aq_chi_derivatives(par, st, chi1w, ex%comp, &
                            Vdchi1WdV, Ndchi1WdNw, Ndchi1WdNc, Ndchi1WdNh)

    x1w = ex%comp(1); x2w = ex%comp(2); x3w = ex%comp(3)
    M = st%M; eps_inf = st%eps_inf
    rho = st%rho; rho1 = st%rho1
    Pw = st%Pw; Pww = st%Pww; Pwc = st%Pwc; Pwh = st%Pwh
    gw = st%gw; delta = st%delta
    chi3w = st%chi3w; g_eta = st%g_eta; dg_deta = st%dg_deta
    eta = st%eta; b = st%b_mix; S13 = st%S13
    chi2w = st%chi2w; S12A_ = st%S12A
    debye = st%debye; xiZi = st%xiZi

    VddeltadV = -delta * eta * dg_deta / g_eta
    dGdchiw = 2.0_wp * x1w * rho * S13 * delta * chi3w**2
    dGddelta = chi3w**2 * (2.0_wp * rho * x1w * chi1w * S13)
    dGdV = -(2.0_wp * rho * x1w * S13 * delta * chi1w) * chi3w**2
    Vdchi3WdV = -(dGdchiw * Vdchi1WdV + dGdV + dGddelta * VddeltadV)
    dHdchiw = 2.0_wp * x1w * rho * S12A_ * delta * chi2w**2
    dHddelta = chi2w**2 * (2.0_wp * rho * x1w * chi1w * S12A_)
    dHdV = -(2.0_wp * rho * x1w * S12A_ * delta * chi1w) * chi2w**2
    Vdchi2WdV = -(dHdchiw * Vdchi1WdV + dHdV + dHddelta * VddeltadV)

    dgwdPw = -(gw - 1.0_wp) * par%cosTHETA1 / (Pw * par%cosTHETA1 + 1.0_wp)
    if (abs(Pww) < 1.0e-30_wp) then
       dgwdPww = (0.0_wp, 0.0_wp)
    else
       dgwdPww = (gw - 1.0_wp) / Pww
    end if
    VdPwwdV = Pww * (-1.0_wp + 1.0_wp / delta * VddeltadV + 2.0_wp / chi1w * Vdchi1WdV)
    if (x3w > 0.0_wp) then
       VdPwcdV = Pwc * (-1.0_wp + 1.0_wp / delta * VddeltadV + 1.0_wp / chi1w * Vdchi1WdV &
                 + 1.0_wp / chi3w * Vdchi3WdV)
    else
       VdPwcdV = (0.0_wp, 0.0_wp)
    end if
    if (x2w > 0.0_wp .and. S12A_ /= 0.0_wp) then
       VdPwhdV = Pwh * (-1.0_wp + 1.0_wp / delta * VddeltadV + 1.0_wp / chi1w * Vdchi1WdV &
                 + 1.0_wp / chi2w * Vdchi2WdV)
    else
       VdPwhdV = (0.0_wp, 0.0_wp)
    end if
    VdPwdV = VdPwwdV + VdPwcdV + VdPwhdV
    VdgwdV = dgwdPw * VdPwdV + dgwdPww * VdPwwdV

    dFder = (2.0_wp * epsr**2 + eps_inf**2) / (epsr**2 * (eps_inf + 2.0_wp)**2)
    dFdeinf = (epsr * eps_inf - 4.0_wp * eps_inf - 4.0_wp * epsr**2 - 2.0_wp * epsr) &
              / (epsr * (eps_inf + 2.0_wp)**3)
    r_d = st%r_d
    VdeinfdV = -3.0_wp * M * r_d / (1.0_wp - M)**2
    w_d = st%w_d
    VdwdV = st%VdwdV
    VdFdV = (par%Navo * rho1 * r_d * par%dip01**2 / (9.0_wp * par%eps0 * par%kb * ex%T) &
             * (w_d * (gw * r_d - VdgwdV) - gw * VdwdV))
    VderdV = -(dFder)**(-1) * (dFdeinf * VdeinfdV + VdFdV)

    if (st%has_ions) then
       daDHBder = (debye**2 / (8.0_wp * pi * par%Navo * rho * epsr * xiZi) &
                   * sum(st%xiw * par%Zi**2 &
                         * (debye / (1.0_wp + debye * par%Sgi) - 1.0_wp / par%Rbi)))
    else
       daDHBder = (0.0_wp, 0.0_wp)
    end if
    ZPerm = -daDHBder * VderdV

    Zw_new = st%Zphys + st%Zassoc + st%ZDH + ZPerm + st%Z_ext

    f(1) = (Zw - Zw_new) / Zw
    f(2) = (chi1w - st%chi1w_new) / chi1w
    f(3) = (st%T1 - st%T2) / st%T2
  end subroutine aq_residual_3var

  ! ------------------------------------------------------------------
  ! Aqueous lnphi at a converged (Zw, epsr, chi1w)
  ! (mirrors _aq_lnphi_evaluate_3var).
  ! ------------------------------------------------------------------
  subroutine aq_lnphi_evaluate_3var(par, v, x_aq, mi, T, P, lnphi)
    type(par_t), intent(in) :: par
    real(wp), intent(in) :: v(3), x_aq(3), mi(6), T, P
    real(wp), intent(out) :: lnphi(3)

    type(aqstate_t) :: st
    complex(wp) :: Zw, epsr, chi1w
    complex(wp) :: Vdchi1WdV, Ndchi1WdNw, Ndchi1WdNc, Ndchi1WdNh
    real(wp) :: x1w, x2w, x3w, S13, S12A_, b
    complex(wp) :: M, eps_inf, rho, Pw, Pww, Pwc, Pwh, gw, delta, chi3w
    complex(wp) :: g_eta, dg_deta, eta, debye, chi2w
    real(wp) :: xiZi
    logical :: s12_on
    complex(wp) :: NddeltadN1, NddeltadN2, NddeltadN3
    complex(wp) :: dGdchiw, dGdchic, dGddelta, dGdNw
    complex(wp) :: dHdchiw, dHddelta, dHdNw
    complex(wp) :: Ndchi3WdNw, Ndchi2WdNw, Ndchi3WdNc, Ndchi2WdNc
    complex(wp) :: Ndchi3WdNh, Ndchi2WdNh
    complex(wp) :: dgwdPw, dgwdPww
    complex(wp) :: NdPwwdN1, NdPwcdN1, NdPwhdN1, NdgwdN1
    complex(wp) :: NdPwwdN3, NdPwcdN3, NdPwhdN3, NdgwdN3
    complex(wp) :: NdPwwdN2, NdPwcdN2, NdPwhdN2, NdgwdN2
    complex(wp) :: dFder, dFdeinf, r_d
    complex(wp) :: NdeinfdN1, NdeinfdN2, NdeinfdN3
    complex(wp) :: w_d, NdwdN1
    complex(wp) :: NdFdN1, NdFdN2, NdFdN3, NderdN1, NderdN2, NderdN3
    complex(wp) :: daDHBder, lnPHI1perm, lnPHI2perm, lnPHI3perm
    complex(wp) :: l1, l2, l3

    Zw = cmplx(v(1), 0.0_wp, kind=wp)
    epsr = cmplx(v(2), 0.0_wp, kind=wp)
    chi1w = cmplx(v(3), 0.0_wp, kind=wp)
    x1w = x_aq(1); x2w = x_aq(2); x3w = x_aq(3)

    call aq_core_state(par, Zw, epsr, chi1w, x_aq, mi, T, P, st)
    call aq_chi_derivatives(par, st, chi1w, x_aq, &
                            Vdchi1WdV, Ndchi1WdNw, Ndchi1WdNc, Ndchi1WdNh)

    M = st%M; eps_inf = st%eps_inf
    rho = st%rho; Pw = st%Pw; Pww = st%Pww; Pwc = st%Pwc; Pwh = st%Pwh
    gw = st%gw; delta = st%delta; chi3w = st%chi3w
    g_eta = st%g_eta; dg_deta = st%dg_deta; eta = st%eta
    b = st%b_mix; S13 = st%S13; debye = st%debye
    chi2w = st%chi2w; S12A_ = st%S12A
    xiZi = st%xiZi
    s12_on = (S12A_ /= 0.0_wp)

    NddeltadN1 = delta * eta * par%b1 * dg_deta / (g_eta * b)
    NddeltadN2 = delta * eta * par%b2 * dg_deta / (g_eta * b)
    NddeltadN3 = delta * eta * par%b3 * dg_deta / (g_eta * b)
    dGdchiw = 2.0_wp * x1w * rho * S13 * delta * chi3w**2
    dGdchic = (1.0_wp, 0.0_wp)
    dGddelta = chi3w**2 * (2.0_wp * rho * x1w * chi1w * S13)
    dGdNw = 2.0_wp * rho * S13 * delta * chi1w * chi3w**2
    dHdchiw = 2.0_wp * x1w * rho * S12A_ * delta * chi2w**2
    dHddelta = chi2w**2 * (2.0_wp * rho * x1w * chi1w * S12A_)
    dHdNw = 2.0_wp * rho * S12A_ * delta * chi1w * chi2w**2

    Ndchi3WdNc = (0.0_wp, 0.0_wp); Ndchi2WdNc = (0.0_wp, 0.0_wp)
    Ndchi3WdNh = (0.0_wp, 0.0_wp); Ndchi2WdNh = (0.0_wp, 0.0_wp)
    Ndchi3WdNw = -dGdchic**(-1) * (dGdchiw * Ndchi1WdNw + dGdNw + dGddelta * NddeltadN1)
    Ndchi2WdNw = -(dHdchiw * Ndchi1WdNw + dHdNw + dHddelta * NddeltadN1)
    if (x3w > 0.0_wp) then
       Ndchi3WdNc = -dGdchic**(-1) * (dGdchiw * Ndchi1WdNc + 0.0_wp + dGddelta * NddeltadN3)
       Ndchi2WdNc = -(dHdchiw * Ndchi1WdNc + 0.0_wp + dHddelta * NddeltadN3)
    end if
    if (x2w > 0.0_wp) then
       Ndchi3WdNh = -dGdchic**(-1) * (dGdchiw * Ndchi1WdNh + 0.0_wp + dGddelta * NddeltadN2)
       Ndchi2WdNh = -(dHdchiw * Ndchi1WdNh + 0.0_wp + dHddelta * NddeltadN2)
    end if

    dgwdPw = -(gw - 1.0_wp) * par%cosTHETA1 / (Pw * par%cosTHETA1 + 1.0_wp)
    if (abs(Pww) < 1.0e-30_wp) then
       dgwdPww = (0.0_wp, 0.0_wp)
    else
       dgwdPww = (gw - 1.0_wp) / Pww
    end if

    NdPwwdN1 = Pww * (1.0_wp / x1w + 1.0_wp / delta * NddeltadN1 &
               + 2.0_wp / chi1w * Ndchi1WdNw)
    if (x3w > 0.0_wp) then
       NdPwcdN1 = Pwc * (1.0_wp / chi1w * Ndchi1WdNw + 1.0_wp / delta * NddeltadN1 &
                  + 1.0_wp / chi3w * Ndchi3WdNw)
    else
       NdPwcdN1 = (0.0_wp, 0.0_wp)
    end if
    if (x2w > 0.0_wp .and. s12_on) then
       NdPwhdN1 = Pwh * (1.0_wp / chi1w * Ndchi1WdNw + 1.0_wp / delta * NddeltadN1 &
                  + 1.0_wp / chi2w * Ndchi2WdNw)
    else
       NdPwhdN1 = (0.0_wp, 0.0_wp)
    end if
    NdgwdN1 = dgwdPw * (NdPwwdN1 + NdPwcdN1 + NdPwhdN1) + dgwdPww * NdPwwdN1

    if (x3w > 0.0_wp) then
       NdPwwdN3 = Pww * (2.0_wp / chi1w * Ndchi1WdNc + 1.0_wp / delta * NddeltadN3)
       NdPwcdN3 = Pwc * (1.0_wp / x3w + 1.0_wp / chi1w * Ndchi1WdNc &
                  + 1.0_wp / delta * NddeltadN3 + 1.0_wp / chi3w * Ndchi3WdNc)
       if (x2w > 0.0_wp .and. s12_on) then
          NdPwhdN3 = Pwh * (1.0_wp / chi1w * Ndchi1WdNc + 1.0_wp / delta * NddeltadN3 &
                     + 1.0_wp / chi2w * Ndchi2WdNc)
       else
          NdPwhdN3 = (0.0_wp, 0.0_wp)
       end if
       NdgwdN3 = dgwdPw * (NdPwwdN3 + NdPwcdN3 + NdPwhdN3) + dgwdPww * NdPwwdN3
    else
       NdgwdN3 = (0.0_wp, 0.0_wp)
    end if

    if (x2w > 0.0_wp) then
       NdPwwdN2 = Pww * (2.0_wp / chi1w * Ndchi1WdNh + 1.0_wp / delta * NddeltadN2)
       if (x3w > 0.0_wp) then
          NdPwcdN2 = Pwc * (1.0_wp / chi1w * Ndchi1WdNh + 1.0_wp / delta * NddeltadN2 &
                     + 1.0_wp / chi3w * Ndchi3WdNh)
       else
          NdPwcdN2 = (0.0_wp, 0.0_wp)
       end if
       if (s12_on) then
          NdPwhdN2 = Pwh * (1.0_wp / x2w + 1.0_wp / chi1w * Ndchi1WdNh &
                     + 1.0_wp / delta * NddeltadN2 + 1.0_wp / chi2w * Ndchi2WdNh)
       else
          NdPwhdN2 = (0.0_wp, 0.0_wp)
       end if
       NdgwdN2 = dgwdPw * (NdPwwdN2 + NdPwcdN2 + NdPwhdN2) + dgwdPww * NdPwwdN2
    else
       NdgwdN2 = (0.0_wp, 0.0_wp)
    end if

    dFder = (2.0_wp * epsr**2 + eps_inf**2) / (epsr**2 * (eps_inf + 2.0_wp)**2)
    dFdeinf = (epsr * eps_inf - 4.0_wp * eps_inf - 4.0_wp * epsr**2 - 2.0_wp * epsr) &
              / (epsr * (eps_inf + 2.0_wp)**3)
    r_d = st%r_d
    NdeinfdN1 = par%Navo * par%pol1 * rho * r_d / (par%eps0 * (1.0_wp - M)**2)
    NdeinfdN2 = par%Navo * par%pol2 * rho * r_d / (par%eps0 * (1.0_wp - M)**2)
    NdeinfdN3 = par%Navo * par%pol3 * rho * r_d / (par%eps0 * (1.0_wp - M)**2)
    w_d = st%w_d
    NdwdN1 = st%NdwdN1
    NdFdN1 = (-par%Navo * rho * r_d * par%dip01**2 / (9.0_wp * par%eps0 * par%kb * T) &
              * (gw * w_d + x1w * (w_d * NdgwdN1 + gw * NdwdN1)))
    NdFdN2 = -par%Navo * rho * r_d * par%dip01**2 / (9.0_wp * par%eps0 * par%kb * T) &
             * (x1w * w_d * NdgwdN2)
    NdFdN3 = -par%Navo * rho * r_d * par%dip01**2 / (9.0_wp * par%eps0 * par%kb * T) &
             * (x1w * w_d * NdgwdN3)
    NderdN1 = -(dFder)**(-1) * (dFdeinf * NdeinfdN1 + NdFdN1)
    NderdN2 = -(dFder)**(-1) * (dFdeinf * NdeinfdN2 + NdFdN2)
    NderdN3 = -(dFder)**(-1) * (dFdeinf * NdeinfdN3 + NdFdN3)

    if (st%has_ions) then
       daDHBder = (debye**2 / (8.0_wp * pi * par%Navo * rho * epsr * xiZi) &
                   * sum(st%xiw * par%Zi**2 &
                         * (debye / (1.0_wp + debye * par%Sgi) - 1.0_wp / par%Rbi)))
    else
       daDHBder = (0.0_wp, 0.0_wp)
    end if
    lnPHI1perm = daDHBder * NderdN1
    lnPHI2perm = daDHBder * NderdN2
    lnPHI3perm = daDHBder * NderdN3

    l1 = st%lnPHI1phys + st%lnPHI1assoc + lnPHI1perm + st%lnphi_ext(1)
    l2 = st%lnPHI2phys + st%lnPHI2assoc + lnPHI2perm + st%lnphi_ext(2)
    l3 = st%lnPHI3phys + st%lnPHI3assoc + lnPHI3perm + st%lnphi_ext(3)

    ! Branch-cut guard.  Every argument of this routine is real, so the
    ! assembled lnphi can only acquire an imaginary part through a log (or
    ! the Debye sqrt) evaluated on a negative argument: a root inside the
    ! mixture co-volume (Zw <= Bcap in -log(Zw - Bcap)), a non-positive Zw
    ! in log((Zw + Bcap)/Zw), or a non-positive association fraction in
    ! 4*log(chi).  The real-arithmetic reference gets NaN at exactly those
    ! points; keeping real() alone would return log|.| instead -- a finite,
    ! ordinary-looking lnphi for a root that is not a state of the fluid,
    ! which an SSI can then converge on.  Return the all-NaN vector, the
    ! same failure signal gas_lnphi_evaluate raises on a non-positive chi.
    ! The threshold is slack because a legitimate imaginary part here is
    ! exactly zero, while a branch cut contributes O(pi).
    if (abs(aimag(l1)) > 1.0e-10_wp .or. abs(aimag(l2)) > 1.0e-10_wp .or. &
        abs(aimag(l3)) > 1.0e-10_wp) then
       lnphi = nan_wp()
       return
    end if

    lnphi(1) = real(l1)
    lnphi(2) = real(l2)
    lnphi(3) = real(l3)
  end subroutine aq_lnphi_evaluate_3var

  ! ------------------------------------------------------------------
  ! Gas-phase residual (mirrors _gas_residual; complex-generic).
  ! ex%comp = y_gas.
  ! ------------------------------------------------------------------
  subroutine gas_residual(par, ex, n, v, f)
    type(par_t), intent(in) :: par
    type(rextra_t), intent(in) :: ex
    integer, intent(in) :: n
    complex(wp), intent(in) :: v(n)
    complex(wp), intent(out) :: f(n)

    complex(wp) :: Zh, chi1h
    real(wp) :: y1, y2, y3, T, P
    real(wp) :: a1, a2, a3, k12, k13, S13, a12, a13, a23
    real(wp) :: b, a, Acap, Bcap, S12A_
    complex(wp) :: Zphys, eta, g_eta, dg_deta, delta, DELTAc
    complex(wp) :: chi3h, chi2h, chi1h_new, Zassoc, Zh_new

    Zh = v(1); chi1h = v(2)
    y1 = ex%comp(1); y2 = ex%comp(2); y3 = ex%comp(3)
    T = ex%T; P = ex%P

    call pure_a_coeffs(par, T, a1, a2, a3)
    call kij_s13(par, T, k12, k13, S13)
    a12 = sqrt(a1 * a2) * (1.0_wp - k12)
    a13 = sqrt(a1 * a3) * (1.0_wp - k13)
    a23 = sqrt(a2 * a3) * (1.0_wp - par%k23)

    b = par%b1 * y1 + par%b2 * y2 + par%b3 * y3
    a = (a1 * y1**2 + 2.0_wp * a12 * y1 * y2 + a2 * y2**2 &
         + 2.0_wp * a13 * y1 * y3 + a3 * y3**2 + 2.0_wp * a23 * y2 * y3)
    Acap = a * P / par%R**2 / T**2
    Bcap = b * P / par%R / T
    Zphys = Zh / (Zh - Bcap) - Acap / (Zh + Bcap)

    eta = Bcap / 4.0_wp / Zh
    g_eta = 1.0_wp / (1.0_wp - 1.9_wp * eta)
    dg_deta = 1.9_wp / (1.0_wp - 1.9_wp * eta)**2
    delta = (g_eta * par%kappaW) * (exp(par%epsW / T) - 1.0_wp)
    DELTAc = delta * P / par%R / T
    S12A_ = par%S12A
    chi3h = Zh / (Zh + 2.0_wp * y1 * chi1h * S13 * DELTAc)
    chi2h = Zh / (Zh + 2.0_wp * y1 * chi1h * S12A_ * DELTAc)
    chi1h_new = Zh / (Zh + 2.0_wp * y1 * chi1h * DELTAc &
                + 2.0_wp * y3 * chi3h * S13 * DELTAc &
                + 2.0_wp * y2 * chi2h * S12A_ * DELTAc)
    Zassoc = -2.0_wp * (1.0_wp + eta / g_eta * dg_deta) &
             * (y1 * (1.0_wp - chi1h) + y3 * (1.0_wp - chi3h) &
                + y2 * (1.0_wp - chi2h))
    Zh_new = Zphys + Zassoc

    f(1) = (Zh - Zh_new) / Zh
    f(2) = (chi1h - chi1h_new) / chi1h
  end subroutine gas_residual

  ! ------------------------------------------------------------------
  ! Gas association state at a GIVEN Zh (mirrors _gas_assoc_state).
  ! ok = .false. when Zh is outside the closure domain: inside the
  ! mixture co-volume (Zh <= B, where log(Zh - B) is undefined) or at/past
  ! the pole of g(eta) = 1/(1 - 1.9 eta).
  ! ------------------------------------------------------------------
  subroutine gas_assoc_state(par, Zh, y_gas, T, P, Bcap, DELTAc, S13, S12A_, ok)
    type(par_t), intent(in) :: par
    real(wp), intent(in) :: Zh, y_gas(3), T, P
    real(wp), intent(out) :: Bcap, DELTAc, S13, S12A_
    logical, intent(out) :: ok

    real(wp) :: b, eta, g_eta, delta, k12, k13

    Bcap = 0.0_wp; DELTAc = 0.0_wp; S13 = 0.0_wp; S12A_ = par%S12A
    ok = .false.
    b = par%b1 * y_gas(1) + par%b2 * y_gas(2) + par%b3 * y_gas(3)
    Bcap = b * P / par%R / T
    if (.not. ieee_is_finite(Zh)) return
    if (Zh <= Bcap) return
    eta = Bcap / 4.0_wp / Zh
    if (1.0_wp - 1.9_wp * eta <= 0.0_wp) return
    g_eta = 1.0_wp / (1.0_wp - 1.9_wp * eta)
    delta = (g_eta * par%kappaW) * (exp(par%epsW / T) - 1.0_wp)
    DELTAc = delta * P / par%R / T
    call kij_s13(par, T, k12, k13, S13)
    ok = .true.
  end subroutine gas_assoc_state

  ! ------------------------------------------------------------------
  ! Is a converged (Zh, chi1h) pair on the physical chi branch?
  ! Mirrors _gas_root_is_physical.  Every one of the three monomer
  ! fractions enters lnphi through 4*log(chi), so all three must be
  ! strictly positive; the closure chi*(Z + 2 y1 chi Delta) = Z also has
  ! a NEGATIVE root that satisfies the residual exactly and yields NaN.
  ! The Wertheim upper bound chi <= 1 holds only when every association
  ! strength is non-negative (S13(T) goes negative below ~290 K), so it
  ! is asserted only there.
  ! ------------------------------------------------------------------
  function gas_root_is_physical(par, v, y_gas, T, P) result(good)
    type(par_t), intent(in) :: par
    real(wp), intent(in) :: v(2), y_gas(3), T, P
    logical :: good

    real(wp) :: Zh, chi1h, Bcap, DELTAc, S13, S12A_, den3, den2
    real(wp) :: chi2h, chi3h
    real(wp), parameter :: slack = 1.0e-12_wp
    logical :: okst

    good = .false.
    Zh = v(1); chi1h = v(2)
    if (.not. (ieee_is_finite(Zh) .and. ieee_is_finite(chi1h))) return
    call gas_assoc_state(par, Zh, y_gas, T, P, Bcap, DELTAc, S13, S12A_, okst)
    if (.not. okst) return
    den3 = Zh + 2.0_wp * y_gas(1) * chi1h * S13 * DELTAc
    den2 = Zh + 2.0_wp * y_gas(1) * chi1h * S12A_ * DELTAc
    if (den3 == 0.0_wp .or. den2 == 0.0_wp) return
    chi3h = Zh / den3
    chi2h = Zh / den2
    if (.not. (ieee_is_finite(chi2h) .and. ieee_is_finite(chi3h))) return
    if (.not. (chi1h > 0.0_wp .and. chi2h > 0.0_wp .and. chi3h > 0.0_wp)) return
    if (DELTAc >= 0.0_wp .and. S13 >= 0.0_wp .and. S12A_ >= 0.0_wp) then
       if (chi1h > 1.0_wp + slack) return
       if (chi2h > 1.0_wp + slack) return
       if (chi3h > 1.0_wp + slack) return
    end if
    good = .true.
  end function gas_root_is_physical

  ! ------------------------------------------------------------------
  ! chi1h on the POSITIVE branch at a given Zh (mirrors
  ! _gas_chi_positive_branch): the closed-form positive root of the
  ! water-water quadratic 2 y1 Delta c^2 + Z c - Z = 0, written
  ! cancellation-free as 2Z / (Z + sqrt(Z^2 + 8 y1 Delta Z)), refined by
  ! damped substitution on the full three-species closure.
  ! ------------------------------------------------------------------
  subroutine gas_chi_positive_branch(par, Zh, y_gas, T, P, c, ok)
    type(par_t), intent(in) :: par
    real(wp), intent(in) :: Zh, y_gas(3), T, P
    real(wp), intent(out) :: c
    logical, intent(out) :: ok

    real(wp) :: Bcap, DELTAc, S13, S12A_, u, disc, den, c_new, c_next
    real(wp) :: chi2h, chi3h, y1, y2, y3
    real(wp), parameter :: ctol = 1.0e-14_wp
    integer, parameter :: cmax = 200
    integer :: it
    logical :: okst

    c = nan_wp()
    ok = .false.
    y1 = y_gas(1); y2 = y_gas(2); y3 = y_gas(3)
    call gas_assoc_state(par, Zh, y_gas, T, P, Bcap, DELTAc, S13, S12A_, okst)
    if (.not. okst) return
    u = 2.0_wp * y1 * DELTAc
    if (u <= 0.0_wp) then
       c = 1.0_wp                      ! no water-water association
       ok = .true.
       return
    end if
    disc = Zh * Zh + 4.0_wp * u * Zh
    if (disc < 0.0_wp) return
    c = 2.0_wp * Zh / (Zh + sqrt(disc))
    if (.not. (ieee_is_finite(c) .and. c > 0.0_wp)) then
       c = nan_wp()
       return
    end if
    do it = 1, cmax
       chi3h = Zh + 2.0_wp * y1 * c * S13 * DELTAc
       chi2h = Zh + 2.0_wp * y1 * c * S12A_ * DELTAc
       if (chi3h == 0.0_wp .or. chi2h == 0.0_wp) exit
       chi3h = Zh / chi3h
       chi2h = Zh / chi2h
       den = Zh + u * c + 2.0_wp * y3 * chi3h * S13 * DELTAc &
             + 2.0_wp * y2 * chi2h * S12A_ * DELTAc
       if (.not. (den > 0.0_wp)) exit
       c_new = Zh / den
       if (.not. (ieee_is_finite(c_new) .and. c_new > 0.0_wp)) exit
       c_next = 0.5_wp * (c + c_new)
       if (abs(c_next - c) <= ctol * max(1.0_wp, abs(c))) then
          c = c_next
          exit
       end if
       c = c_next
    end do
    ok = ieee_is_finite(c) .and. c > 0.0_wp
    if (.not. ok) c = nan_wp()
  end subroutine gas_chi_positive_branch

  ! ------------------------------------------------------------------
  ! Accept a converged gas root only if it is physical; if it is not,
  ! re-solve once from the SAME Zh with chi1h re-seeded on the positive
  ! branch (mirrors _gas_accept_or_repair).  A non-converged input is
  ! passed through untouched, so a state whose Newton merely failed takes
  ! exactly the seed ladder it took before this guard existed.
  ! ------------------------------------------------------------------
  subroutine gas_accept_or_repair(par, ex, v, conv, y_gas, T, P, tol, maxiter)
    type(par_t), intent(in) :: par
    type(rextra_t), intent(in) :: ex
    real(wp), intent(inout) :: v(2)
    logical, intent(inout) :: conv
    real(wp), intent(in) :: y_gas(3), T, P, tol
    integer, intent(in) :: maxiter

    real(wp) :: c0, v2(2)
    logical :: okc, conv2
    integer :: nit

    if (.not. conv) return
    if (gas_root_is_physical(par, v, y_gas, T, P)) return
    call gas_chi_positive_branch(par, v(1), y_gas, T, P, c0, okc)
    if (okc) then
       call newton_solve(par, ex, gas_residual, 2, [v(1), c0], tol, maxiter, &
                         v2, conv2, nit)
       if (conv2) then
          if (gas_root_is_physical(par, v2, y_gas, T, P)) then
             v = v2
             conv = .true.
             return
          end if
       end if
    end if
    conv = .false.
  end subroutine gas_accept_or_repair

  ! ------------------------------------------------------------------
  ! Gas lnphi at a converged (Zh, chi1h) (mirrors _gas_lnphi_evaluate).
  ! ------------------------------------------------------------------
  subroutine gas_lnphi_evaluate(par, v, y_gas, T, P, lnphi)
    type(par_t), intent(in) :: par
    real(wp), intent(in) :: v(2), y_gas(3), T, P
    real(wp), intent(out) :: lnphi(3)

    real(wp) :: Zh, chi1h, y1, y2, y3
    real(wp) :: a1, a2, a3, k12, k13, S13, a12, a13, a23
    real(wp) :: cA1, cB1, cA2, cB2, cA3, cB3, gA12, gA13, gA23
    real(wp) :: b, a, Acap, Bcap
    real(wp) :: lnPHI1phys, lnPHI2phys, lnPHI3phys
    real(wp) :: eta, g_eta, dg_deta, delta, DELTAc, S12A_
    real(wp) :: chi3h, chi2h, assoc_sum
    real(wp) :: lnPHI1assoc, lnPHI2assoc, lnPHI3assoc

    Zh = v(1); chi1h = v(2)
    y1 = y_gas(1); y2 = y_gas(2); y3 = y_gas(3)

    call pure_a_coeffs(par, T, a1, a2, a3)
    call kij_s13(par, T, k12, k13, S13)
    a12 = sqrt(a1 * a2) * (1.0_wp - k12)
    a13 = sqrt(a1 * a3) * (1.0_wp - k13)
    a23 = sqrt(a2 * a3) * (1.0_wp - par%k23)
    cA1 = a1 * P / par%R**2 / T**2
    cB1 = par%b1 * P / par%R / T
    cA2 = a2 * P / par%R**2 / T**2
    cB2 = par%b2 * P / par%R / T
    cA3 = a3 * P / par%R**2 / T**2
    cB3 = par%b3 * P / par%R / T
    gA12 = a12 * P / par%R**2 / T**2
    gA13 = a13 * P / par%R**2 / T**2
    gA23 = a23 * P / par%R**2 / T**2

    b = par%b1 * y1 + par%b2 * y2 + par%b3 * y3
    a = (a1 * y1**2 + 2.0_wp * a12 * y1 * y2 + a2 * y2**2 &
         + 2.0_wp * a13 * y1 * y3 + a3 * y3**2 + 2.0_wp * a23 * y2 * y3)
    Acap = a * P / par%R**2 / T**2
    Bcap = b * P / par%R / T

    lnPHI1phys = (-log(Zh - Bcap) + cB1 / Bcap * (Bcap / (Zh - Bcap) - Acap / (Zh + Bcap)) &
         + Acap / Bcap * (cB1 / Bcap - 2.0_wp * (y1 * cA1 + y2 * gA12 + y3 * gA13) / Acap) &
         * log(1.0_wp + Bcap / Zh))
    lnPHI2phys = (-log(Zh - Bcap) + cB2 / Bcap * (Bcap / (Zh - Bcap) - Acap / (Zh + Bcap)) &
         + Acap / Bcap * (cB2 / Bcap - 2.0_wp * (y1 * gA12 + y2 * cA2 + y3 * gA23) / Acap) &
         * log(1.0_wp + Bcap / Zh))
    lnPHI3phys = (-log(Zh - Bcap) + cB3 / Bcap * (Bcap / (Zh - Bcap) - Acap / (Zh + Bcap)) &
         + Acap / Bcap * (cB3 / Bcap - 2.0_wp * (y1 * gA13 + y2 * gA23 + y3 * cA3) / Acap) &
         * log(1.0_wp + Bcap / Zh))

    eta = Bcap / 4.0_wp / Zh
    g_eta = 1.0_wp / (1.0_wp - 1.9_wp * eta)
    dg_deta = 1.9_wp / (1.0_wp - 1.9_wp * eta)**2
    delta = (g_eta * par%kappaW) * (exp(par%epsW / T) - 1.0_wp)
    DELTAc = delta * P / par%R / T
    S12A_ = par%S12A
    chi3h = Zh / (Zh + 2.0_wp * y1 * chi1h * S13 * DELTAc)
    chi2h = Zh / (Zh + 2.0_wp * y1 * chi1h * S12A_ * DELTAc)
    ! Last line of defence: every chi below feeds a logarithm.  The
    ! Python reference RAISES here; Fortran has no exceptions, so this
    ! returns the all-NaN vector, which every caller already treats as a
    ! failure.  Callers screen their root with gas_root_is_physical, so
    ! this branch should be unreachable in normal operation.
    if (.not. (chi1h > 0.0_wp .and. chi2h > 0.0_wp .and. chi3h > 0.0_wp)) then
       lnphi = nan_wp()
       return
    end if
    assoc_sum = (y1 * (chi1h - 1.0_wp) + y3 * (chi3h - 1.0_wp) &
                 + y2 * (chi2h - 1.0_wp))
    lnPHI1assoc = 4.0_wp * log(chi1h) + cB1 / (2.0_wp * g_eta * Zh) * dg_deta * assoc_sum
    lnPHI2assoc = 4.0_wp * log(chi2h) + cB2 / (2.0_wp * g_eta * Zh) * dg_deta * assoc_sum
    lnPHI3assoc = 4.0_wp * log(chi3h) + cB3 / (2.0_wp * g_eta * Zh) * dg_deta * assoc_sum

    lnphi(1) = lnPHI1phys + lnPHI1assoc
    lnphi(2) = lnPHI2phys + lnPHI2assoc
    lnphi(3) = lnPHI3phys + lnPHI3assoc
  end subroutine gas_lnphi_evaluate

  ! ------------------------------------------------------------------
  ! Seed helpers (mirror the Python cold-start ladders exactly).
  ! ------------------------------------------------------------------
  pure function zw_liq_seed(T, P) result(z)
    real(wp), intent(in) :: T, P
    real(wp) :: z
    ! Literal constants mirror the Python seed heuristic (NOT par%R).
    z = P * 18.015e-6_wp / (8.314_wp * T)
  end function zw_liq_seed

  pure function aq_default_seed(T, P) result(v0)
    real(wp), intent(in) :: T, P
    real(wp) :: v0(3)
    v0 = [zw_liq_seed(T, P), 50.0_wp, 0.10_wp]
  end function aq_default_seed

  ! lnphi_aq's own fallback ladder (30 rungs + 2 flag-gated).
  subroutine aq_seed_ladder_lnphi(par, T, P, seeds, nseeds)
    type(par_t), intent(in) :: par
    real(wp), intent(in) :: T, P
    real(wp), intent(out) :: seeds(3, 30 + MAX_EXTRA_SEEDS)
    integer, intent(out) :: nseeds
    real(wp) :: zl, xseeds(3, MAX_EXTRA_SEEDS)
    integer :: nx
    zl = zw_liq_seed(T, P)
    seeds(:, 1)  = [zl, 60.0_wp, 0.07_wp]
    seeds(:, 2)  = [zl, 50.0_wp, 0.80_wp]
    seeds(:, 3)  = [zl, 50.0_wp, 0.90_wp]
    seeds(:, 4)  = [0.07_wp, 60.0_wp, 0.13_wp]
    seeds(:, 5)  = [0.10_wp, 30.0_wp, 0.05_wp]
    seeds(:, 6)  = [0.30_wp, 70.0_wp, 0.20_wp]
    seeds(:, 7)  = [0.20_wp, 40.0_wp, 0.10_wp]
    seeds(:, 8)  = [0.50_wp, 50.0_wp, 0.10_wp]
    seeds(:, 9)  = [0.50_wp, 80.0_wp, 0.50_wp]
    seeds(:, 10) = [zl, 78.0_wp, 0.85_wp]
    seeds(:, 11) = [zl, 78.0_wp, 0.95_wp]
    seeds(:, 12) = [zl, 90.0_wp, 0.07_wp]
    seeds(:, 13) = [zl, 100.0_wp, 0.07_wp]
    seeds(:, 14) = [zl, 110.0_wp, 0.07_wp]
    seeds(:, 15) = [zl, 90.0_wp, 0.85_wp]
    seeds(:, 16) = [zl, 100.0_wp, 0.90_wp]
    seeds(:, 17) = [zl, 80.0_wp, 0.97_wp]
    seeds(:, 18) = [zl, 85.0_wp, 0.95_wp]
    seeds(:, 19) = [zl, 30.0_wp, 0.10_wp]
    seeds(:, 20) = [zl, 40.0_wp, 0.20_wp]
    seeds(:, 21) = [zl, 50.0_wp, 0.30_wp]
    seeds(:, 22) = [zl, 75.0_wp, 0.98_wp]
    seeds(:, 23) = [zl, 70.0_wp, 0.99_wp]
    seeds(:, 24) = [zl, 70.0_wp, 0.93_wp]
    seeds(:, 25) = [zl, 65.0_wp, 0.95_wp]
    seeds(:, 26) = [zl, 5.0_wp, 0.05_wp]
    seeds(:, 27) = [zl, 10.0_wp, 0.05_wp]
    seeds(:, 28) = [zl, 15.0_wp, 0.10_wp]
    seeds(:, 29) = [zl, 25.0_wp, 0.12_wp]
    seeds(:, 30) = [1.4_wp * zl, 42.0_wp, 0.09_wp]
    nseeds = 30
    ! seeds an extension adds for its own terms (none by default)
    if (associated(aq_seeds_hook)) then
       call aq_seeds_hook(par, zl, nx, xseeds)
       nx = max(0, min(nx, MAX_EXTRA_SEEDS))
       if (nx > 0) seeds(:, 31:30 + nx) = xseeds(:, 1:nx)
       nseeds = 30 + nx
    end if
  end subroutine aq_seed_ladder_lnphi

  ! The flash _ssi_step warm-restart ladder (29 rungs, fixed).
  subroutine aq_seed_ladder_flash(T, P, seeds, nseeds)
    real(wp), intent(in) :: T, P
    real(wp), intent(out) :: seeds(3, 29)
    integer, intent(out) :: nseeds
    real(wp) :: zl
    zl = zw_liq_seed(T, P)
    seeds(:, 1)  = [zl, 60.0_wp, 0.07_wp]
    seeds(:, 2)  = [zl, 50.0_wp, 0.80_wp]
    seeds(:, 3)  = [zl, 50.0_wp, 0.90_wp]
    seeds(:, 4)  = [0.07_wp, 60.0_wp, 0.13_wp]
    seeds(:, 5)  = [0.05_wp, 50.0_wp, 0.10_wp]
    seeds(:, 6)  = [0.10_wp, 30.0_wp, 0.05_wp]
    seeds(:, 7)  = [0.30_wp, 70.0_wp, 0.20_wp]
    seeds(:, 8)  = [0.20_wp, 40.0_wp, 0.10_wp]
    seeds(:, 9)  = [0.50_wp, 50.0_wp, 0.10_wp]
    seeds(:, 10) = [0.50_wp, 80.0_wp, 0.50_wp]
    seeds(:, 11) = [zl, 78.0_wp, 0.85_wp]
    seeds(:, 12) = [zl, 78.0_wp, 0.95_wp]
    seeds(:, 13) = [zl, 90.0_wp, 0.07_wp]
    seeds(:, 14) = [zl, 100.0_wp, 0.07_wp]
    seeds(:, 15) = [zl, 110.0_wp, 0.07_wp]
    seeds(:, 16) = [zl, 90.0_wp, 0.85_wp]
    seeds(:, 17) = [zl, 100.0_wp, 0.90_wp]
    seeds(:, 18) = [zl, 80.0_wp, 0.97_wp]
    seeds(:, 19) = [zl, 85.0_wp, 0.95_wp]
    seeds(:, 20) = [zl, 30.0_wp, 0.10_wp]
    seeds(:, 21) = [zl, 40.0_wp, 0.20_wp]
    seeds(:, 22) = [zl, 50.0_wp, 0.30_wp]
    seeds(:, 23) = [zl, 75.0_wp, 0.98_wp]
    seeds(:, 24) = [zl, 70.0_wp, 0.99_wp]
    seeds(:, 25) = [zl, 70.0_wp, 0.93_wp]
    seeds(:, 26) = [zl, 65.0_wp, 0.95_wp]
    seeds(:, 27) = [zl, 5.0_wp, 0.05_wp]
    seeds(:, 28) = [zl, 10.0_wp, 0.05_wp]
    seeds(:, 29) = [zl, 15.0_wp, 0.10_wp]
    nseeds = 29
  end subroutine aq_seed_ladder_flash

  ! ------------------------------------------------------------------
  ! Self-consistent aqueous lnphi driver (mirrors lnphi_aq).
  ! ok=.false. when the whole seed ladder fails (Python raises).
  ! niter: iterations of the successful Newton (diagnostic).
  ! ------------------------------------------------------------------
  subroutine lnphi_aq(par, x_aq, mi, T, P, lnphi, v_out, ok, niter, v_init)
    type(par_t), intent(in) :: par
    real(wp), intent(in) :: x_aq(3), mi(6), T, P
    real(wp), intent(out) :: lnphi(3), v_out(3)
    logical, intent(out) :: ok
    integer, intent(out) :: niter
    real(wp), intent(in), optional :: v_init(3)

    type(rextra_t) :: ex
    real(wp) :: v0(3), v(3), seeds(3, 30 + MAX_EXTRA_SEEDS)
    integer :: nseeds, k, nit
    logical :: conv, asym
    integer :: i, j
    real(wp), parameter :: tol = 1.0e-10_wp
    integer, parameter :: maxiter = 80

    lnphi = nan_wp()
    v_out = nan_wp()
    ok = .false.
    niter = 0

    ! Symmetry guard on an active ion-ion matrix (mirrors lnphi_aq).
    asym = .false.
    if (any(par%Uref_ii /= 0.0_wp)) then
       do i = 1, 6
          do j = 1, 6
             if (par%Uref_ii(i, j) /= par%Uref_ii(j, i)) asym = .true.
          end do
       end do
    end if
    if (asym) return

    ex%comp = x_aq
    ex%mi = mi
    ex%T = T
    ex%P = P

    if (present(v_init)) then
       v0 = v_init
    else
       v0 = aq_default_seed(T, P)
    end if

    call newton_solve(par, ex, aq_residual_3var, 3, v0, tol, maxiter, v, conv, nit)
    if (conv .and. (v(3) <= 0.0_wp .or. v(2) <= LIQUID_EPSR_MIN)) conv = .false.
    if (.not. conv) then
       call aq_seed_ladder_lnphi(par, T, P, seeds, nseeds)
       do k = 1, nseeds
          call newton_solve(par, ex, aq_residual_3var, 3, seeds(:, k), tol, maxiter, &
                            v, conv, nit)
          if (conv .and. (v(3) <= 0.0_wp .or. v(2) <= LIQUID_EPSR_MIN)) conv = .false.
          if (conv) exit
       end do
    end if
    if (.not. conv) return

    call aq_lnphi_evaluate_3var(par, v, x_aq, mi, T, P, lnphi)
    v_out = v
    ok = .true.
    niter = nit
  end subroutine lnphi_aq

  ! ------------------------------------------------------------------
  ! Self-consistent gas lnphi driver (mirrors lnphi_gas).
  ! Z_init <= 0 selects the default 1.0.
  ! ------------------------------------------------------------------
  ! ------------------------------------------------------------------
  ! Last-resort gas root: eliminate chi1h analytically, then bracket and
  ! bisect on Zh alone.  Mirrors _gas_nested_root in ms_ecpa/lnphi.py.
  !
  ! As y_H2O -> 0 the association closure degenerates, because chi1h
  ! enters it only multiplied by y_H2O: the Jacobian column for chi1h
  ! vanishes and the 2x2 Newton either diverges or lands on the negative
  ! branch.  The Zh direction stays well conditioned, so eliminating
  ! chi1h removes the failure mode outright, and at y_H2O = 0 it returns
  ! the exact answer chi1h = 1 (no water, hence no association).
  !
  ! Matters below ~290 K, where S13(T) < 0 puts the closure on its
  ! extended (anti-solvation) domain and folds the root locus, so no
  ! fixed seed ladder in (Zh, chi1h) covers every composition.  Inside
  ! that fold the nested residual has three roots; the one of lowest
  ! molar Gibbs energy sum_i y_i ln phi_i is returned.
  ! ------------------------------------------------------------------
  subroutine gas_nested_residual(par, ex, Zh, y_gas, T, P, r, ok)
    type(par_t), intent(in) :: par
    type(rextra_t), intent(in) :: ex
    real(wp), intent(in) :: Zh, y_gas(3), T, P
    real(wp), intent(out) :: r
    logical, intent(out) :: ok

    real(wp) :: c
    complex(wp) :: vc(2), fc(2)
    logical :: okc

    r = nan_wp()
    ok = .false.
    call gas_chi_positive_branch(par, Zh, y_gas, T, P, c, okc)
    if (.not. okc) return
    vc(1) = cmplx(Zh, 0.0_wp, wp)
    vc(2) = cmplx(c, 0.0_wp, wp)
    call gas_residual(par, ex, 2, vc, fc)
    r = real(fc(1), wp)
    ok = ieee_is_finite(r)
    if (.not. ok) r = nan_wp()
  end subroutine gas_nested_residual

  subroutine gas_nested_root(par, ex, y_gas, T, P, v, ok)
    type(par_t), intent(in) :: par
    type(rextra_t), intent(in) :: ex
    real(wp), intent(in) :: y_gas(3), T, P
    real(wp), intent(out) :: v(2)
    logical, intent(out) :: ok

    integer, parameter :: ngrid = 240
    integer, parameter :: nbis = 80
    real(wp), parameter :: zmax = 3.0_wp
    real(wp) :: b_mix, Bcap, lo, hi, step, xa, xb, fa, fb
    real(wp) :: lo_i, hi_i, f_lo, mid, f_mid, c, g, best_g, vbest(2), vt(2)
    real(wp) :: lnphi(3)
    logical :: oka, okb, okm, okc, have_best
    integer :: i, k

    v = nan_wp()
    ok = .false.
    have_best = .false.
    best_g = nan_wp()
    vbest = nan_wp()

    b_mix = par%b1 * y_gas(1) + par%b2 * y_gas(2) + par%b3 * y_gas(3)
    Bcap = b_mix * P / (par%R * T)
    lo = Bcap * 1.0001_wp + 1.0e-9_wp
    hi = zmax
    if (.not. ieee_is_finite(lo)) return
    if (lo >= hi) return
    step = (hi - lo) / real(ngrid - 1, wp)

    xb = lo
    call gas_nested_residual(par, ex, xb, y_gas, T, P, fb, okb)
    do i = 1, ngrid - 1
       xa = xb; fa = fb; oka = okb
       xb = lo + real(i, wp) * step
       call gas_nested_residual(par, ex, xb, y_gas, T, P, fb, okb)
       if (.not. (oka .and. okb)) cycle
       if (fa /= 0.0_wp .and. fa * fb >= 0.0_wp) cycle
       lo_i = xa; hi_i = xb; f_lo = fa
       if (fa /= 0.0_wp) then
          do k = 1, nbis
             mid = 0.5_wp * (lo_i + hi_i)
             call gas_nested_residual(par, ex, mid, y_gas, T, P, f_mid, okm)
             if (.not. okm) exit
             if (f_lo * f_mid <= 0.0_wp) then
                hi_i = mid
             else
                lo_i = mid; f_lo = f_mid
             end if
          end do
       end if
       mid = 0.5_wp * (lo_i + hi_i)
       call gas_chi_positive_branch(par, mid, y_gas, T, P, c, okc)
       if (.not. okc) cycle
       vt = [mid, c]
       if (.not. gas_root_is_physical(par, vt, y_gas, T, P)) cycle
       call gas_lnphi_evaluate(par, vt, y_gas, T, P, lnphi)
       g = y_gas(1) * lnphi(1) + y_gas(2) * lnphi(2) + y_gas(3) * lnphi(3)
       if (.not. ieee_is_finite(g)) cycle
       if ((.not. have_best) .or. g < best_g) then
          best_g = g; vbest = vt; have_best = .true.
       end if
    end do

    if (.not. have_best) return
    v = vbest
    ok = .true.
  end subroutine gas_nested_root

  subroutine lnphi_gas(par, y_gas, T, P, Z_init, chi_init, lnphi, v_out, ok)
    type(par_t), intent(in) :: par
    real(wp), intent(in) :: y_gas(3), T, P, Z_init, chi_init
    real(wp), intent(out) :: lnphi(3), v_out(2)
    logical, intent(out) :: ok

    type(rextra_t) :: ex
    real(wp) :: v0(2), v(2), alts(8), b_mix, Zh_B, Zh_liq_co2, zi0
    integer :: nit, k
    logical :: conv
    real(wp), parameter :: tol = 1.0e-10_wp
    integer, parameter :: maxiter = 80

    lnphi = nan_wp()
    v_out = nan_wp()
    ok = .false.

    ex%comp = y_gas
    ex%mi = 0.0_wp
    ex%T = T
    ex%P = P

    zi0 = Z_init
    if (zi0 <= 0.0_wp) zi0 = 1.0_wp
    v0 = [zi0, chi_init]
    call newton_solve(par, ex, gas_residual, 2, v0, tol, maxiter, v, conv, nit)
    ! Physical guard, mirroring the chi1w>0 guard on the aqueous side: a
    ! root the Newton reports as converged is accepted only on the
    ! physical chi branch; a mis-branched root is repaired in place.
    call gas_accept_or_repair(par, ex, v, conv, y_gas, T, P, tol, maxiter)
    if (.not. conv) then
       b_mix = par%b1 * y_gas(1) + par%b2 * y_gas(2) + par%b3 * y_gas(3)
       Zh_B = b_mix * P / (par%R * T) * 1.2_wp + 0.3_wp
       Zh_liq_co2 = b_mix * P / (par%R * T) * 2.0_wp
       alts = [max(Zh_B, 1.5_wp), max(Zh_B * 1.3_wp, 2.0_wp), 3.0_wp, 0.7_wp, &
               0.5_wp, 0.3_wp, Zh_liq_co2, b_mix * P / (par%R * T) * 1.5_wp]
       do k = 1, 8
          call newton_solve(par, ex, gas_residual, 2, [alts(k), chi_init], tol, &
                            maxiter, v, conv, nit)
          call gas_accept_or_repair(par, ex, v, conv, y_gas, T, P, tol, maxiter)
          if (conv) exit
       end do
    end if
    if (.not. conv) then
       ! Purely additive: reached only where the code above would have
       ! returned ok=.false., so no state that already solved can change.
       call gas_nested_root(par, ex, y_gas, T, P, v, conv)
    end if
    if (.not. conv) return

    call gas_lnphi_evaluate(par, v, y_gas, T, P, lnphi)
    v_out = v
    ok = .true.
  end subroutine lnphi_gas

end module ecpa_lnphi
