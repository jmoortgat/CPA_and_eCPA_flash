! ecpa_flash.f90 -- multi-salt K-value flash (faithful port of
! ms_ecpa/flash.py: wilson_k, rachford_rice, flash_kv, dew_point_flash).
!
! salinity_convention: 0 = "as-aqueous" (the Python DEFAULT, and the
! convention the published validation uses), 1 = "chain" (opt-in
! salt-conservation closure, flash.compute_aqueous_x1w_chain).
!
! Units: T [K], P [bar] at this level (kernels get P*1e5 Pa).
module ecpa_flash
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use ecpa_kinds, only: wp
  use ecpa_params, only: par_t
  use ecpa_newton, only: rextra_t, newton_solve, lin_solve, nan_wp, inf_wp, &
       clip_np, maxabs_nan, nansum_np
  use ecpa_lnphi, only: aq_residual_3var, gas_residual, &
       aq_lnphi_evaluate_3var, gas_lnphi_evaluate, lnphi_aq, lnphi_gas, &
       aq_seed_ladder_flash, gas_root_is_physical, zw_liq_seed, &
       LIQUID_EPSR_MIN
  use ecpa_brent, only: brentq
  implicit none
  private
  public :: flashopt_t, flashres_t, wilson_k, rachford_rice, flash_kv, &
            dew_point_flash, aqueous_x1w_chain, SALINITY_AS_AQUEOUS, &
            SALINITY_CHAIN

  ! salinity_convention codes (mirror the Python string flag).
  integer, parameter :: SALINITY_AS_AQUEOUS = 0
  integer, parameter :: SALINITY_CHAIN = 1

  type :: flashopt_t
     real(wp) :: tol = 1.0e-8_wp
     integer :: maxiter = 80
     logical :: accelerated = .true.
     logical :: use_newton = .true.
     real(wp) :: newton_tol = 1.0e-4_wp
     integer :: max_newton = 5
     logical :: have_kinit = .false.
     real(wp) :: K_init(3) = 0.0_wp
     logical :: have_gas_zinit = .false.
     real(wp) :: gas_Z_init = 0.0_wp
     ! Caller-owned warm-start seeds (reservoir-simulator integration):
     ! full inner-Newton state from a previous converged flash.
     logical :: have_warm_aq = .false.
     real(wp) :: warm_aq(3) = 0.0_wp       ! (Z_aq, eps_r, chi_H2O_aq)
     logical :: have_warm_gas = .false.
     real(wp) :: warm_gas(2) = 0.0_wp      ! (Z_gas, chi_H2O_gas)
     ! Salinity convention: SALINITY_AS_AQUEOUS (default, matches the
     ! Python default) or SALINITY_CHAIN (opt-in).
     integer :: salinity_convention = SALINITY_AS_AQUEOUS
  end type flashopt_t

  type :: flashres_t
     logical :: converged = .false.
     ! errcode: 0 clean; 1 SSI step failed (non-degenerate);
     ! 2 did-not-converge in maxiter; 5 Newton-polish base eval failed
     ! (the Python reference RAISES here -- see PORTING_NOTES.md);
     ! 100+ input validation.
     integer :: errcode = 0
     real(wp) :: T = 0.0_wp, P = 0.0_wp
     real(wp) :: K(3) = 0.0_wp
     real(wp) :: x_aq(3) = 0.0_wp
     real(wp) :: y_gas(3) = 0.0_wp
     real(wp) :: beta = 0.0_wp, x1w = 0.0_wp
     real(wp) :: Z_aq = 0.0_wp, Z_gas = 0.0_wp, eps_r = 0.0_wp
     real(wp) :: chi_aq = 0.0_wp, chi_gas = 0.0_wp
     integer :: n_iter_ssi = 0, n_iter_newton = 0
     integer :: n_iter_aq_inner = 0, n_iter_gas_inner = 0
     real(wp) :: max_residual = 0.0_wp
     real(wp) :: fug_aq(3) = 0.0_wp, fug_gas(3) = 0.0_wp
     real(wp) :: mi_aq(6) = 0.0_wp
  end type flashres_t

contains

  ! Wilson K-values for (H2O, CH4, CO2) at (T [K], P [bar]).
  function wilson_k(par, T, Pbar) result(K)
    type(par_t), intent(in) :: par
    real(wp), intent(in) :: T, Pbar
    real(wp) :: K(3)
    real(wp) :: Tc(3), Pc_bar(3)
    Tc = [par%Tc1, par%Tc2, par%Tc3]
    Pc_bar = [par%Pc1, par%Pc2, par%Pc3] / 1.0e5_wp
    K = Pc_bar / Pbar * exp(5.373_wp * (1.0_wp + par%omega_wilson) * (1.0_wp - Tc / T))
  end function wilson_k

  ! ------------------------------------------------------------------
  ! Salt-conservation chain closure -- line-for-line port of
  ! flash.compute_aqueous_x1w_chain.
  !
  !   S        = Mw * z_w * m_tot_feed / (1 - beta)
  !   scale    = 1 / (1 + S)
  !   sigma_x  = S * scale                     (= S/(1+S))
  !   x1w_true = x_part_w * scale
  !   mi_aq    = mi_feed * z_w / ((1 - beta) * x_part_w)
  !
  ! S is the ion moles per mole of aqueous PARTITIONING species; the ion
  ! mole-fraction sum sigma_x = Mw * x_w_aq * m_tot_aq then gives
  ! sigma_x = S/(1+S) and x_w_aq = x_part_w/(1+S).  Both lie in (0, 1]
  ! for every physical input, so the closure cannot report a spuriously
  ! "unphysical" state; the earlier scale = 1 - S form agrees only to
  ! first order in S and went non-positive at high m_tot / high beta.
  !
  ! The molality denominator is the PARTITION-basis (salt-free) water
  ! mole fraction, not the ion-inclusive x1w_true: molality is per kg of
  ! water, and the aqueous water moles are (1-beta)*x_part_w per mole of
  ! partitioning feed.  Dividing by x1w_true instead inflates every
  ! molality by 1/(1 - sigma_x) and breaks salt conservation.
  !
  ! Invalid states return NaN exactly where the Python helper does; the
  ! caller (ssi_step) tests isfinite(x1w_true)/x1w_true>0/scale>0 and
  ! raises, mirroring the Python RuntimeError.
  ! ------------------------------------------------------------------
  subroutine aqueous_x1w_chain(Mw, x_part_w, beta, z_w, mi_feed, &
                               x1w_true, scale, mi_aq, sigma_x)
    real(wp), intent(in) :: Mw, x_part_w, beta, z_w, mi_feed(6)
    real(wp), intent(out) :: x1w_true, scale, mi_aq(6), sigma_x
    real(wp) :: m_tot_feed, one_minus_beta, S

    m_tot_feed = sum(mi_feed)
    one_minus_beta = 1.0_wp - beta
    if (one_minus_beta <= 0.0_wp) then
       ! beta -> 1: no aqueous phase; sigma_x diverges.
       x1w_true = nan_wp(); scale = nan_wp()
       mi_aq = nan_wp(); sigma_x = inf_wp()
       return
    end if
    S = Mw * z_w * m_tot_feed / one_minus_beta
    if ((.not. ieee_is_finite(S)) .or. S < 0.0_wp) then
       ! Guards a non-finite or negative S (non-finite feed molality, or a
       ! beta that overshot the bracket), which would otherwise make
       ! 1/(1+S) silently finite and wrong at S = -1 - eps.
       x1w_true = nan_wp(); scale = nan_wp()
       mi_aq = nan_wp(); sigma_x = nan_wp()
       return
    end if
    scale = 1.0_wp / (1.0_wp + S)
    sigma_x = S * scale
    x1w_true = x_part_w * scale
    if (x1w_true <= 0.0_wp) then
       x1w_true = nan_wp(); mi_aq = nan_wp()
       return
    end if
    mi_aq = mi_feed * (z_w / (one_minus_beta * x_part_w))
  end subroutine aqueous_x1w_chain

  ! Rachford-Rice vapor fraction (mirrors flash.rachford_rice).
  function rachford_rice(K, z) result(beta)
    real(wp), intent(in) :: K(3), z(3)
    real(wp) :: beta
    logical :: pres(3)
    real(wp) :: K_p(3), z_p(3)
    integer :: np_, i, ierr
    real(wp) :: beta_lo, beta_hi, f_lo, f_hi
    real(wp), parameter :: eps = 1.0e-12_wp

    np_ = 0
    do i = 1, 3
       pres(i) = (z(i) > 0.0_wp) .and. ieee_is_finite(K(i)) .and. ieee_is_finite(z(i))
       if (pres(i)) then
          np_ = np_ + 1
          K_p(np_) = K(i)
          z_p(np_) = z(i)
       end if
    end do
    if (np_ == 0) then
       beta = 0.0_wp
       return
    end if
    if (all(K_p(1:np_) <= 1.0_wp)) then
       beta = 0.0_wp
       return
    end if
    if (all(K_p(1:np_) >= 1.0_wp)) then
       beta = 1.0_wp
       return
    end if

    beta_lo = max(1.0_wp / (1.0_wp - maxval(K_p(1:np_))) + eps, 0.0_wp)
    beta_hi = min(1.0_wp / (1.0_wp - minval(K_p(1:np_))) - eps, 1.0_wp)
    beta_lo = max(beta_lo, 0.0_wp)
    beta_hi = min(beta_hi, 1.0_wp)

    f_lo = rr_f(beta_lo)
    f_hi = rr_f(beta_hi)
    if (f_lo * f_hi > 0.0_wp) then
       if (abs(f_lo) < abs(f_hi)) then
          beta = 0.0_wp
       else
          beta = 1.0_wp
       end if
       return
    end if
    call brentq(rr_f, beta_lo, beta_hi, 1.0e-12_wp, 1.0e-12_wp, 200, beta, ierr)

  contains

    function rr_f(b) result(y)
      real(wp), intent(in) :: b
      real(wp) :: y
      integer :: k_
      y = 0.0_wp
      do k_ = 1, np_
         y = y + z_p(k_) * (K_p(k_) - 1.0_wp) / (1.0_wp + b * (K_p(k_) - 1.0_wp))
      end do
    end function rr_f
  end function rachford_rice

  ! ------------------------------------------------------------------
  ! flash_kv (mirrors flash.flash_kv, "as-aqueous" convention).
  ! ------------------------------------------------------------------
  subroutine flash_kv(par, T, Pbar, z, mi, opt, res)
    type(par_t), intent(in) :: par
    real(wp), intent(in) :: T, Pbar, z(3), mi(6)
    type(flashopt_t), intent(in) :: opt
    type(flashres_t), intent(out) :: res

    logical :: present(3)
    real(wp) :: P_Pa, K(3), K_wil_now(3)
    real(wp) :: g_prev(3)
    integer :: n_g_prev
    logical :: have_g_prev
    real(wp) :: m_jex, max_resid
    integer :: n_iter_ssi, n_iter_newton, n_degenerate_resets
    logical :: converged
    real(wp) :: x_aq(3), y_gas(3), beta, x1w
    real(wp) :: v_aq_warm(3), v_gas_warm(2)
    logical :: have_aq_warm, have_gas_warm
    ! ssi_step outputs
    real(wp) :: s_Knew(3), s_xaq(3), s_ygas(3), s_beta, s_x1w, s_g(3)
    real(wp) :: s_la(3), s_lg(3)
    integer :: sstat
    real(wp) :: g_present(3)
    integer :: idx_present(3), n_p
    real(wp) :: lnK_now(3), lnK_new_vec(3)
    real(wp) :: num_a, denom_a, K_H2O_max, z_H2O
    real(wp) :: mi_aq_final(6), S_chain
    integer :: it, i
    real(wp), parameter :: X1W_FLOOR = 0.10_wp

    res = flashres_t()
    res%T = T
    res%P = Pbar
    res%mi_aq = mi
    res%K = nan_wp()
    res%x_aq = nan_wp(); res%y_gas = nan_wp()
    res%beta = nan_wp(); res%x1w = nan_wp()
    res%Z_aq = nan_wp(); res%Z_gas = nan_wp(); res%eps_r = nan_wp()
    res%chi_aq = nan_wp(); res%chi_gas = nan_wp()
    res%fug_aq = nan_wp(); res%fug_gas = nan_wp()
    res%max_residual = inf_wp()

    if (abs(sum(z) - 1.0_wp) > 1.0e-8_wp) then
       res%errcode = 100
       return
    end if

    present = (z > 0.0_wp)
    P_Pa = Pbar * 1.0e5_wp

    if (opt%have_kinit) then
       K = opt%K_init
    else
       K = wilson_k(par, T, Pbar)
    end if
    do i = 1, 3
       if (.not. present(i)) K(i) = nan_wp()
    end do

    n_p = 0
    do i = 1, 3
       if (present(i)) then
          n_p = n_p + 1
          idx_present(n_p) = i
       end if
    end do

    have_g_prev = .false.
    n_g_prev = 0
    g_prev = 0.0_wp
    m_jex = 1.0_wp
    n_iter_ssi = 0
    n_iter_newton = 0
    converged = .false.
    max_resid = inf_wp()
    x_aq = nan_wp(); y_gas = nan_wp(); beta = nan_wp(); x1w = nan_wp()

    have_aq_warm = opt%have_warm_aq
    v_aq_warm = 0.0_wp
    if (have_aq_warm) v_aq_warm = opt%warm_aq
    have_gas_warm = .false.
    v_gas_warm = 0.0_wp
    if (opt%have_warm_gas) then
       have_gas_warm = .true.
       v_gas_warm = opt%warm_gas
    else if (opt%have_gas_zinit) then
       have_gas_warm = .true.
       v_gas_warm = [opt%gas_Z_init, 0.95_wp]
    end if

    ! -- SSI loop ----------------------------------------------------
    n_degenerate_resets = 0
    ssi_loop: do it = 0, opt%maxiter - 1
       call ssi_step(K, s_Knew, s_xaq, s_ygas, s_beta, s_x1w, s_g, s_la, s_lg, sstat)
       if (sstat /= 0) then
          if (sstat == 1 .and. n_degenerate_resets < 5) then
             n_degenerate_resets = n_degenerate_resets + 1
             K_wil_now = wilson_k(par, T, Pbar)
             if (present(1)) then
                if (ieee_is_finite(K(1)) .and. K(1) > 0.0_wp) then
                   K(1) = min(K(1), K_wil_now(1))
                else
                   K(1) = K_wil_now(1)
                end if
             end if
             do i = 2, 3
                if (present(i)) K(i) = max(K_wil_now(i), 5.0_wp)
             end do
             have_aq_warm = .false.
             have_g_prev = .false.
             m_jex = 1.0_wp
             cycle ssi_loop
          end if
          ! SSI failed
          res%converged = .false.
          res%errcode = 1
          res%K = K
          res%x_aq = x_aq; res%y_gas = y_gas
          res%beta = beta; res%x1w = x1w
          res%n_iter_ssi = it
          res%n_iter_newton = 0
          res%max_residual = inf_wp()
          res%mi_aq = mi
          return
       end if

       x_aq = s_xaq; y_gas = s_ygas; beta = s_beta; x1w = s_x1w

       g_present = 0.0_wp
       do i = 1, n_p
          g_present(i) = s_g(idx_present(i))
       end do
       max_resid = maxabs_nan(g_present(1:n_p))

       if (max_resid < opt%tol) then
          K = s_Knew
          n_iter_ssi = it + 1
          converged = .true.
          exit ssi_loop
       end if
       if (opt%use_newton .and. max_resid < opt%newton_tol) then
          n_iter_ssi = it + 1
          exit ssi_loop
       end if

       do i = 1, 3
          if (present(i)) then
             lnK_now(i) = log(K(i))
          else
             lnK_now(i) = 0.0_wp
          end if
       end do
       if (opt%accelerated .and. have_g_prev) then
          num_a = 0.0_wp
          denom_a = 0.0_wp
          do i = 1, n_p
             num_a = num_a + g_prev(i) * g_prev(i)
             denom_a = denom_a + g_prev(i) * (g_prev(i) - g_present(i))
          end do
          if (abs(denom_a) > 1.0e-30_wp) then
             m_jex = abs(num_a / denom_a * m_jex)
             m_jex = min(max(m_jex, 1.0_wp), 10.0_wp)
          else
             m_jex = 1.0_wp
          end if
       end if
       g_prev(1:n_p) = g_present(1:n_p)
       n_g_prev = n_p
       have_g_prev = .true.

       lnK_new_vec = lnK_now
       do i = 1, n_p
          lnK_new_vec(idx_present(i)) = lnK_now(idx_present(i)) + m_jex * g_present(i)
       end do
       lnK_new_vec = clip_np(lnK_new_vec, -30.0_wp, 30.0_wp)

       ! K_H2O upper clamp (Iter 7 Fix A / Iter 8 Fix C)
       if (present(1) .and. beta > 1.0e-4_wp .and. beta < 1.0_wp - 1.0e-4_wp) then
          z_H2O = z(1)
          if (z_H2O > X1W_FLOOR) then
             K_H2O_max = (z_H2O / X1W_FLOOR - 1.0_wp) / beta + 1.0_wp
             if (K_H2O_max > 1.0_wp) then
                if (lnK_new_vec(1) == lnK_new_vec(1)) then
                   lnK_new_vec(1) = min(lnK_new_vec(1), log(K_H2O_max))
                end if
             end if
          end if
       end if

       do i = 1, 3
          if (present(i)) then
             K(i) = exp(lnK_new_vec(i))
          else
             K(i) = nan_wp()
          end if
       end do
       n_iter_ssi = it + 1
    end do ssi_loop

    ! -- Newton polish ----------------------------------------------
    if (opt%use_newton .and. .not. converged .and. ieee_is_finite(max_resid) &
        .and. max_resid < opt%newton_tol) then
       call newton_polish()
       if (res%errcode == 5) return
    end if

    ! -- Equilibrium aqueous-phase molality (chain output) ------------
    ! Mirrors the flash_kv tail: under "chain" recompute m_aq* from the
    ! converged (beta, x1w); under "as-aqueous" mi_aq == input mi.
    if (opt%salinity_convention == SALINITY_CHAIN .and. &
        ieee_is_finite(beta) .and. (1.0_wp - beta) > 0.0_wp .and. x1w > 0.0_wp) then
       ! x1w is the ion-INCLUSIVE aqueous water fraction; molality needs
       ! the partition-basis one, x_part_w = x1w * (1 + S).  See
       ! aqueous_x1w_chain.
       S_chain = par%Mw * z(1) * sum(mi) / (1.0_wp - beta)
       mi_aq_final = mi * (z(1) / ((1.0_wp - beta) * x1w * (1.0_wp + S_chain)))
    else
       mi_aq_final = mi
    end if

    ! -- Diagnostics from the converged state -------------------------
    call diagnostics()

    res%converged = converged
    if (.not. converged .and. res%errcode == 0) res%errcode = 2
    res%K = K
    res%x_aq = x_aq
    res%y_gas = y_gas
    res%beta = beta
    res%x1w = x1w
    res%n_iter_ssi = n_iter_ssi
    res%n_iter_newton = n_iter_newton
    res%max_residual = max_resid
    res%mi_aq = mi_aq_final

  contains

    ! ................................................................
    ! Single SSI evaluation (mirrors flash_kv._ssi_step).
    ! sstat: 0 ok; 1 degenerate-aqueous (triggers the K-reset);
    !        2 chain-closure invalid; 3 aq inner Newton failed;
    !        4 gas lnphi failed.  Only stat_ == 1 is recoverable, exactly
    !        as in Python (only the "degenerate-aqueous:" message is).
    ! ................................................................
    subroutine ssi_step(K_in, K_new, x_aq_loc, y_gas_loc, beta_loc, x1w_true, &
                        g_loc, la, lg, stat_)
      real(wp), intent(in) :: K_in(3)
      real(wp), intent(out) :: K_new(3), x_aq_loc(3), y_gas_loc(3)
      real(wp), intent(out) :: beta_loc, x1w_true, g_loc(3), la(3), lg(3)
      integer, intent(out) :: stat_

      real(wp) :: x_part(3), y_part(3), sx, sy, denom
      real(wp) :: m_tot, scale, mi_aq_step(6), sigma_chain
      real(wp) :: v0_aq(3), v_aq(3), v0_g(2), v_g(2)
      real(wp) :: seeds(3, 29), lnK_new(3), dum3(3)
      integer :: nseeds, kseed, nit, ii
      logical :: ok_aq, ok_g, ok_full, warm_ok
      type(rextra_t) :: ex_a, ex_g

      stat_ = 0
      K_new = nan_wp(); la = nan_wp(); lg = nan_wp()
      g_loc = 0.0_wp

      beta_loc = rachford_rice(K_in, z)
      do ii = 1, 3
         if (present(ii)) then
            denom = 1.0_wp + beta_loc * (K_in(ii) - 1.0_wp)
            x_part(ii) = z(ii) / denom
            y_part(ii) = K_in(ii) * x_part(ii)
         else
            x_part(ii) = 0.0_wp
            y_part(ii) = 0.0_wp
         end if
      end do
      sx = nansum_np(x_part)
      sy = nansum_np(y_part)
      if (sx > 0.0_wp) x_part = x_part / sx
      if (sy > 0.0_wp) y_part = y_part / sy

      if (opt%salinity_convention == SALINITY_CHAIN) then
         ! Rigorous salt-conservation chain (opt-in; Python
         ! salinity_convention="chain").  mi is the prepared-brine FEED
         ! molality.
         call aqueous_x1w_chain(par%Mw, x_part(1), beta_loc, z(1), mi, &
                                x1w_true, scale, mi_aq_step, sigma_chain)
         if ((.not. ieee_is_finite(x1w_true)) .or. x1w_true <= 0.0_wp &
             .or. scale <= 0.0_wp) then
            ! Python raises RuntimeError("chain closure invalid ...");
            ! not a "degenerate-aqueous:" message, so the SSI loop
            ! returns an unconverged result (errcode 1).
            stat_ = 2
            return
         end if
      else
         ! "as-aqueous" closure (default)
         m_tot = sum(mi)
         if (x_part(1) > 0.0_wp) then
            x1w_true = x_part(1) / (1.0_wp + par%Mw * m_tot * x_part(1))
         else
            x1w_true = 0.0_wp
         end if
         scale = 1.0_wp - par%Mw * m_tot * x1w_true
         mi_aq_step = mi
      end if

      x_aq_loc(1) = x1w_true
      if (present(2)) then
         x_aq_loc(2) = x_part(2) * scale
      else
         x_aq_loc(2) = 0.0_wp
      end if
      if (present(3)) then
         x_aq_loc(3) = x_part(3) * scale
      else
         x_aq_loc(3) = 0.0_wp
      end if
      y_gas_loc(1) = y_part(1)
      if (present(2)) then
         y_gas_loc(2) = y_part(2)
      else
         y_gas_loc(2) = 0.0_wp
      end if
      if (present(3)) then
         y_gas_loc(3) = y_part(3)
      else
         y_gas_loc(3) = 0.0_wp
      end if

      if (present(1) .and. x1w_true < 0.02_wp) then
         stat_ = 1
         return
      end if

      ! Aqueous side: warm Newton, then flash fallback ladder.
      ex_a%comp = x_aq_loc
      ex_a%mi = mi_aq_step
      ex_a%T = T
      ex_a%P = P_Pa
      if (have_aq_warm) then
         v0_aq = v_aq_warm
      else
         ! Cold start on the SAME physics-scaled seed lnphi_aq uses by
         ! default. The fixed Zw = 0.05 that used to sit here is the seed
         ! lnphi_aq's own notes record as landing outside the liquid basin
         ! at low pressure, and the root found here is taken as
         ! authoritative below, so both entry points must look in the same
         ! basin. Mirrors ms_ecpa/flash.py::_ssi_step.
         v0_aq = [zw_liq_seed(T, P_Pa), 50.0_wp, 0.10_wp]
      end if
      call newton_solve(par, ex_a, aq_residual_3var, 3, v0_aq, 1.0e-10_wp, 80, &
                        v_aq, ok_aq, nit)
      if (ok_aq .and. (v_aq(3) <= 0.0_wp .or. v_aq(2) <= LIQUID_EPSR_MIN)) ok_aq = .false.
      warm_ok = .true.
      if (.not. ok_aq) then
         ! lnphi_aq has its own, longer seed ladder. This `la` is the
         ! authoritative aqueous lnphi and, if we get here, it succeeded.
         call lnphi_aq(par, x_aq_loc, mi_aq_step, T, P_Pa, la, dum3, ok_full, nit)
         if (.not. ok_full) then
            stat_ = 3
            return
         end if
         ! Everything below only re-derives v_aq for the WARM START of the
         ! next SSI step -- an optimisation, not a correctness requirement.
         call aq_seed_ladder_flash(T, P_Pa, seeds, nseeds)
         do kseed = 1, nseeds
            call newton_solve(par, ex_a, aq_residual_3var, 3, seeds(:, kseed), &
                              1.0e-10_wp, 80, v_aq, ok_aq, nit)
            if (ok_aq .and. (v_aq(3) <= 0.0_wp .or. v_aq(2) <= LIQUID_EPSR_MIN)) ok_aq = .false.
            if (ok_aq) exit
         end do
         ! Do NOT abort here. `la` above is valid, and returning stat_ = 3
         ! failed a flash that had actually solved, purely because this
         ! shorter re-derive ladder missed the warm-start root. Mirrors the
         ! same fix in flash.py::_ssi_step (2026-09-18).
         if (.not. ok_aq) warm_ok = .false.
      else
         call aq_lnphi_evaluate_3var(par, v_aq, x_aq_loc, mi_aq_step, T, P_Pa, la)
      end if
      if (warm_ok) then
         v_aq_warm = v_aq
         have_aq_warm = .true.
      else
         ! v_aq holds the last failed attempt; do not store it.
         have_aq_warm = .false.
      end if

      ! Gas side
      ex_g%comp = y_gas_loc
      ex_g%mi = 0.0_wp
      ex_g%T = T
      ex_g%P = P_Pa
      if (have_gas_warm) then
         v0_g = v_gas_warm
      else
         v0_g = [1.0_wp, 0.99_wp]
      end if
      call newton_solve(par, ex_g, gas_residual, 2, v0_g, 1.0e-10_wp, 80, &
                        v_g, ok_g, nit)
      ! reject a spurious chi<=0 gas fixed point (mirrors flash.py)
      if (ok_g) then
         if (.not. gas_root_is_physical(par, v_g, y_gas_loc, T, P_Pa)) &
              ok_g = .false.
      end if
      if (.not. ok_g) then
         block
           real(wp) :: lg_full(3), vg_full(2)
           logical :: okg_full
           call lnphi_gas(par, y_gas_loc, T, P_Pa, -1.0_wp, 0.99_wp, &
                          lg_full, vg_full, okg_full)
           if (.not. okg_full) then
              stat_ = 4
              return
           end if
           lg = lg_full
         end block
         block
           real(wp) :: alts(2, 3)
           integer :: ka
           alts(:, 1) = [1.0_wp, 0.99_wp]
           alts(:, 2) = [0.7_wp, 0.95_wp]
           alts(:, 3) = [0.3_wp, 0.90_wp]
           do ka = 1, 3
              call newton_solve(par, ex_g, gas_residual, 2, alts(:, ka), &
                                1.0e-10_wp, 80, v_g, ok_g, nit)
              if (ok_g) then
                 if (.not. gas_root_is_physical(par, v_g, y_gas_loc, T, &
                                                P_Pa)) ok_g = .false.
              end if
              if (ok_g) exit
           end do
         end block
      else
         call gas_lnphi_evaluate(par, v_g, y_gas_loc, T, P_Pa, lg)
      end if
      if (ok_g) then
         v_gas_warm = v_g
         have_gas_warm = .true.
      end if

      ! K_new on the partition basis
      do ii = 1, 3
         if (.not. present(ii)) then
            lnK_new(ii) = nan_wp()
            cycle
         end if
         lnK_new(ii) = log(scale) + (la(ii) - lg(ii))
      end do
      do ii = 1, 3
         if (present(ii)) then
            g_loc(ii) = lnK_new(ii) - log(K_in(ii))
            K_new(ii) = exp(lnK_new(ii))
         else
            g_loc(ii) = 0.0_wp
            K_new(ii) = nan_wp()
         end if
      end do
    end subroutine ssi_step

    ! ................................................................
    subroutine newton_polish()
      real(wp) :: J(3, 3), base_g(3), dlnK(3), K_pert(3), K_new_full(3)
      real(wp) :: gp(3), new_resid
      real(wp), parameter :: h = 1.0e-5_wp
      integer :: ipol, jcol, ii, sstat2
      logical :: singular, jac_fail

      do ipol = 1, opt%max_newton
         call ssi_step(K, s_Knew, s_xaq, s_ygas, s_beta, s_x1w, s_g, s_la, s_lg, sstat2)
         if (sstat2 /= 0) then
            ! Python raises out of flash_kv here (uncaught).
            res%converged = .false.
            res%errcode = 5
            res%K = K
            res%x_aq = x_aq; res%y_gas = y_gas
            res%beta = beta; res%x1w = x1w
            res%n_iter_ssi = n_iter_ssi
            res%n_iter_newton = n_iter_newton
            res%max_residual = max_resid
            res%mi_aq = mi
            return
         end if
         do ii = 1, n_p
            base_g(ii) = s_g(idx_present(ii))
         end do
         jac_fail = .false.
         do jcol = 1, n_p
            K_pert = K
            K_pert(idx_present(jcol)) = K(idx_present(jcol)) * exp(h)
            call ssi_step(K_pert, s_Knew, s_xaq, s_ygas, s_beta, s_x1w, s_g, &
                          s_la, s_lg, sstat2)
            if (sstat2 /= 0) then
               jac_fail = .true.
               exit
            end if
            do ii = 1, n_p
               J(ii, jcol) = (s_g(idx_present(ii)) - base_g(ii)) / h
            end do
         end do
         if (jac_fail) return
         call lin_solve(n_p, J(1:n_p, 1:n_p), -base_g(1:n_p), dlnK(1:n_p), singular)
         if (singular) return
         dlnK(1:n_p) = clip_np(dlnK(1:n_p), -5.0_wp, 5.0_wp)
         K_new_full = K
         do ii = 1, n_p
            K_new_full(idx_present(ii)) = K(idx_present(ii)) * exp(dlnK(ii))
         end do
         call ssi_step(K_new_full, s_Knew, s_xaq, s_ygas, s_beta, s_x1w, s_g, &
                       s_la, s_lg, sstat2)
         if (sstat2 /= 0) return
         do ii = 1, n_p
            gp(ii) = s_g(idx_present(ii))
         end do
         new_resid = maxabs_nan(gp(1:n_p))
         if (new_resid > max_resid) return
         K = K_new_full
         x_aq = s_xaq; y_gas = s_ygas; beta = s_beta; x1w = s_x1w
         max_resid = new_resid
         n_iter_newton = n_iter_newton + 1
         if (max_resid < opt%tol) then
            converged = .true.
            return
         end if
      end do
    end subroutine newton_polish

    ! ................................................................
    ! Post-convergence diagnostics (mirrors the flash_kv tail).
    ! ................................................................
    subroutine diagnostics()
      type(rextra_t) :: ex_a, ex_g
      real(wp) :: v_aq_init(3), v_aq(3), v_g_init(2), v_g(2), v_try(2)
      real(wp) :: la(3), lg(3), dum3(3), dum2(2)
      real(wp) :: b_mix_g, B_mix_cap, Zh_B, Zh_liq
      real(wp) :: alts(2, 11)
      integer :: nit, ka, ii
      logical :: ok_aq, ok_g, ok_try, ok_la, ok_lg

      ex_a%comp = x_aq
      ex_a%mi = mi_aq_final
      ex_a%T = T
      ex_a%P = P_Pa
      if (have_aq_warm) then
         v_aq_init = v_aq_warm
      else
         ! Cold start on the SAME physics-scaled seed lnphi_aq uses by
         ! default. The fixed Zw = 0.05 that used to sit here is the seed
         ! lnphi_aq's own notes record as landing outside the liquid basin
         ! at low pressure, and the root found here is taken as
         ! authoritative below, so both entry points must look in the same
         ! basin. Mirrors ms_ecpa/flash.py::_ssi_step.
         v_aq_init = [zw_liq_seed(T, P_Pa), 50.0_wp, 0.10_wp]
      end if
      call newton_solve(par, ex_a, aq_residual_3var, 3, v_aq_init, 1.0e-10_wp, 80, &
                        v_aq, ok_aq, nit)
      res%n_iter_aq_inner = nit
      ! Same acceptance the equilibrium path uses (ssi_step, lnphi_aq).
      ! These are diagnostics, but they are reported alongside the
      ! fugacities as though they described the same state, so they must
      ! come from a root the rest of the package would accept -- not
      ! merely from one the Newton stopped at. aq_residual_3var has a
      ! genuine fixed point at chi1w < 0 and at eps_r <= 0 (chi1w enters
      ! only as products and as 1/chi1w, so no logarithm flags it here),
      ! and that root is exported as res%eps_r/chi_aq and reused as the
      ! next cell's warm seed. Left as NaN otherwise, which is readable
      ! as "not available" in a way a wrong number is not.
      ! Mirrors ms_ecpa/flash.py::flash_kv.
      if (ok_aq .and. v_aq(3) > 0.0_wp .and. v_aq(2) > 0.0_wp) then
         res%Z_aq = v_aq(1)
         res%eps_r = v_aq(2)
         res%chi_aq = v_aq(3)
      end if

      ex_g%comp = y_gas
      ex_g%mi = 0.0_wp
      ex_g%T = T
      ex_g%P = P_Pa
      if (have_gas_warm) then
         v_g_init = v_gas_warm
      else
         v_g_init = [1.0_wp, 0.99_wp]
      end if
      call newton_solve(par, ex_g, gas_residual, 2, v_g_init, 1.0e-10_wp, 80, &
                        v_g, ok_g, nit)
      res%n_iter_gas_inner = nit
      if (.not. (ok_g .and. gas_root_is_physical(par, v_g, y_gas, T, P_Pa))) then
         b_mix_g = par%b1 * y_gas(1) + par%b2 * y_gas(2) + par%b3 * y_gas(3)
         B_mix_cap = b_mix_g * P_Pa / par%R / T
         Zh_B = max(B_mix_cap * 1.2_wp + 0.3_wp, 1.0_wp)
         Zh_liq = max(B_mix_cap * 2.0_wp, 0.4_wp)
         alts(:, 1) = [Zh_B, 0.99_wp]
         alts(:, 2) = [Zh_B * 1.3_wp, 0.99_wp]
         alts(:, 3) = [3.0_wp, 0.99_wp]
         alts(:, 4) = [Zh_liq, 0.5_wp]
         alts(:, 5) = [Zh_liq * 1.2_wp, 0.7_wp]
         alts(:, 6) = [0.5_wp, 0.99_wp]
         alts(:, 7) = [0.3_wp, 0.99_wp]
         alts(:, 8) = [B_mix_cap * 1.5_wp, 0.9_wp]
         alts(:, 9) = [B_mix_cap * 3.0_wp, 0.7_wp]
         ! Unfloored dense rungs, mirroring flash.py. Every seed above is
         ! either >= 0.4 (Zh_liq carries a max(..., 0.4) floor) or scaled
         ! >= 1.5x B_mix, which sits above the dense-gas root at low T
         ! where S13 < 0 and chi may exceed 1.
         alts(:, 10) = [B_mix_cap * 1.2_wp, 0.99_wp]
         alts(:, 11) = [B_mix_cap * 1.2_wp, 1.3_wp]
         ok_g = .false.
         do ka = 1, 11
            call newton_solve(par, ex_g, gas_residual, 2, alts(:, ka), &
                              1.0e-10_wp, 80, v_try, ok_try, nit)
            if (ok_try .and. gas_root_is_physical(par, v_try, y_gas, T, P_Pa)) then
               v_g = v_try
               ok_g = .true.
               res%n_iter_gas_inner = nit
               exit
            end if
         end do
      end if
      if (ok_g) then
         if (gas_root_is_physical(par, v_g, y_gas, T, P_Pa)) then
            res%Z_gas = v_g(1)
            res%chi_gas = v_g(2)
         end if
      end if

      call lnphi_aq(par, x_aq, mi_aq_final, T, P_Pa, la, dum3, ok_la, nit)
      if (.not. ok_la) return
      call lnphi_gas(par, y_gas, T, P_Pa, -1.0_wp, 0.99_wp, lg, dum2, ok_lg)
      if (.not. ok_lg) return
      do ii = 1, 3
         res%fug_aq(ii) = exp(la(ii)) * x_aq(ii) * Pbar
         res%fug_gas(ii) = exp(lg(ii)) * y_gas(ii) * Pbar
      end do
    end subroutine diagnostics

  end subroutine flash_kv

  ! accept_v_gas was removed on 2026-09-18. It carried its own rule --
  ! Z > 0 and chi in [0, 1+eps] -- which had drifted from the SSI step and
  ! from lnphi_gas, both of which screen roots with gas_root_is_physical.
  ! The upper bound chi <= 1 is a theorem only when every association
  ! strength is non-negative, and this model lets S13(T) go negative below
  ! about 290 K, so the diagnostics rejected the very root the SSI had just
  ! used and returned NaN for Z_gas and chi_gas. The diagnostics now call
  ! gas_root_is_physical, as ecpa_flash.f90:571/596 already did.

  ! ------------------------------------------------------------------
  ! dew_point_flash (mirrors flash.dew_point_flash).
  !
  ! An ITERATION SCHEME, not different physics. The equations solved here are
  ! the same as everywhere else -- same EoS, same fugacity equality for all
  ! three components, same tolerance -- and only the path to the solution
  ! differs, because near the water saturation line the K-space successive
  ! substitution approaches its trivial fixed point (K = 1). Do not describe
  ! it as a limiting case or an approximation.
  ! ok=.false. corresponds to the Python "return None".
  ! salinity_convention is OPTIONAL and defaults to "as-aqueous", as in
  ! Python; it affects ONLY the reported mi_aq (the internal lnphi_aq
  ! calls use the raw input mi in the reference too -- see
  ! PORTING_NOTES.md).
  ! ------------------------------------------------------------------
  subroutine dew_point_flash(par, T, Pbar, z, mi, x_aq_fixed, res, ok, &
                             salinity_convention)
    type(par_t), intent(in) :: par
    real(wp), intent(in) :: T, Pbar, z(3), mi(6), x_aq_fixed(3)
    type(flashres_t), intent(out) :: res
    logical, intent(out) :: ok
    integer, intent(in), optional :: salinity_convention
    integer :: sconv

    real(wp) :: x_aq(3), P_Pa, ln_aq(3), dum3(3), dum2(2)
    real(wp) :: phi_w_aq, f_w_aq, beta
    real(wp) :: y_gas(3), ln_gas(3), K(3)
    real(wp) :: v_aq0(3), v_aq(3), v_g0(2), v_g(2)
    ! NB: the component-presence mask is named `pres` here (not
    ! `present`) so the PRESENT() intrinsic stays visible for the
    ! optional salinity_convention argument.
    logical :: ok_aq, pres(3), okg, okb, okg2
    integer :: nit, i, it_ref
    real(wp) :: x_new(3), gas_sum, step, resid, fa, fg, fscale
    real(wp) :: m_tot_dew, x_part_dew(3), x1w_true_dew, scale_dew, S_dew
    ! Dissolved-gas refinement (mirrors ms_ecpa/flash.py: DEW_REFINE_*).
    integer, parameter :: DEW_REFINE_ITERS = 40
    real(wp), parameter :: DEW_REFINE_DAMP = 0.7_wp
    real(wp), parameter :: DEW_REFINE_TOL = 1.0e-12_wp
    real(wp), parameter :: DEW_FUGACITY_TOL = 1.0e-6_wp
    real(wp), parameter :: beta_tol = 1.0e-8_wp
    ! scipy brentq default rtol = 4 * DBL_EPSILON
    real(wp), parameter :: rtol_default = 4.0_wp * epsilon(1.0_wp)
    type(rextra_t) :: ex_a, ex_g

    ok = .false.
    res = flashres_t()
    res%T = T
    res%P = Pbar
    sconv = SALINITY_AS_AQUEOUS
    if (present(salinity_convention)) sconv = salinity_convention

    x_aq = x_aq_fixed
    P_Pa = Pbar * 1.0e5_wp
    ! Put the trial composition on the SAME basis the rest of the flash uses.
    ! x_aq_fixed arrives on the partition basis (the three partitioning
    ! species, summing to 1), but the aqueous phase also holds the ions, so
    ! the closure the aqueous kernel assumes is
    !     x1w + x2w + x3w + sum(x_ion) = 1,  x_ion = x1w * Mw * mi,
    ! i.e. the three species sum to 1 - sum(x_ion). ssi_step (flash_kv)
    ! builds x_aq on exactly this basis; both feed the same lnphi_aq and it
    ! consumes x_aq(1) as given. Mirrors ms_ecpa/flash.py::dew_point_flash.
    m_tot_dew = sum(mi)
    if (sum(x_aq) > 0.0_wp) then
       x_part_dew = x_aq / sum(x_aq)
       if (x_part_dew(1) > 0.0_wp) then
          x1w_true_dew = x_part_dew(1) &
                         / (1.0_wp + par%Mw * m_tot_dew * x_part_dew(1))
       else
          x1w_true_dew = 0.0_wp
       end if
       scale_dew = 1.0_wp - par%Mw * m_tot_dew * x1w_true_dew
       x_aq(1) = x1w_true_dew
       x_aq(2) = x_part_dew(2) * scale_dew
       x_aq(3) = x_part_dew(3) * scale_dew
    end if

    call lnphi_aq(par, x_aq, mi, T, P_Pa, ln_aq, dum3, ok_aq, nit)
    if (.not. ok_aq) return
    phi_w_aq = exp(ln_aq(1))
    f_w_aq = x_aq(1) * phi_w_aq * Pbar

    call solve_beta(beta, okb)
    if (.not. okb) return

    ! ---- refine the dissolved-gas fractions --------------------------
    ! Water fugacity balance fixes beta; the CH4/CO2 aqueous fractions used
    ! to be left at whatever x_aq_fixed carried in, which is the quantity a
    ! solubility caller reads. Successive substitution on their own fugacity
    ! equality, water as the balance, beta re-solved each sweep because the
    ! mass balance moves with x_aq. Damped: the map overshoots on a trace
    ! composition. Mirrors ms_ecpa/flash.py::dew_point_flash exactly.
    do it_ref = 1, DEW_REFINE_ITERS
       y_gas = y_gas_of_beta(beta)
       call lnphi_gas(par, y_gas, T, P_Pa, -1.0_wp, 0.99_wp, ln_gas, dum2, okg2)
       if (.not. okg2) return
       call lnphi_aq(par, x_aq, mi, T, P_Pa, ln_aq, dum3, ok_aq, nit)
       if (.not. ok_aq) return
       x_new = x_aq
       do i = 2, 3
          if (z(i) <= 0.0_wp) then
             x_new(i) = 0.0_wp
          else
             x_new(i) = y_gas(i) * exp(ln_gas(i) - ln_aq(i))
          end if
       end do
       gas_sum = x_new(2) + x_new(3)
       if (.not. ieee_is_finite(gas_sum)) return
       if (gas_sum >= 1.0_wp) return
       ! Water is the balance against the ion closure above:
       ! x1w (1 + Mw m_tot) + gas_sum = 1. At m_tot = 0 this is the plain
       ! 1 - gas_sum, which is why salt-free states are unaffected.
       x_new(1) = (1.0_wp - gas_sum) / (1.0_wp + par%Mw * m_tot_dew)
       step = maxval(abs(x_new - x_aq))
       x_aq = (1.0_wp - DEW_REFINE_DAMP) * x_aq + DEW_REFINE_DAMP * x_new
       call lnphi_aq(par, x_aq, mi, T, P_Pa, ln_aq, dum3, ok_aq, nit)
       if (.not. ok_aq) return
       f_w_aq = x_aq(1) * exp(ln_aq(1)) * Pbar
       call solve_beta(beta, okb)
       if (.not. okb) return
       if (step < DEW_REFINE_TOL) exit
    end do

    call lnphi_aq(par, x_aq, mi, T, P_Pa, ln_aq, dum3, ok_aq, nit)
    if (.not. ok_aq) return
    y_gas = y_gas_of_beta(beta)
    block
      logical :: oklg
      call lnphi_gas(par, y_gas, T, P_Pa, -1.0_wp, 0.99_wp, ln_gas, dum2, oklg)
      if (.not. oklg) return
    end block

    ! ---- the honest residual -----------------------------------------
    ! Across every component present, RELATIVE to each one's own fugacity
    ! scale: dissolved CO2 sits orders of magnitude below water, and an
    ! absolute tolerance waves it through -- which is how these states came
    ! to be reported with max_residual = 0.
    resid = 0.0_wp
    do i = 1, 3
       if (z(i) <= 0.0_wp) cycle
       fa = exp(ln_aq(i)) * x_aq(i) * Pbar
       fg = exp(ln_gas(i)) * y_gas(i) * Pbar
       fscale = max(abs(fa), abs(fg), 1.0e-300_wp)
       resid = max(resid, abs(fa - fg) / fscale)
    end do
    if (.not. ieee_is_finite(resid)) return
    if (resid > DEW_FUGACITY_TOL) return

    pres = (z > 0.0_wp)
    do i = 1, 3
       if (pres(i) .and. x_aq(i) > 1.0e-12_wp) then
          K(i) = y_gas(i) / x_aq(i)
       else
          K(i) = nan_wp()
       end if
    end do

    res%Z_aq = nan_wp(); res%eps_r = nan_wp(); res%chi_aq = nan_wp()
    res%Z_gas = nan_wp(); res%chi_gas = nan_wp()
    v_aq0 = [P_Pa * 18.015e-6_wp / (8.314_wp * T), 50.0_wp, 0.10_wp]
    ex_a%comp = x_aq
    ex_a%mi = mi
    ex_a%T = T
    ex_a%P = P_Pa
    call newton_solve(par, ex_a, aq_residual_3var, 3, v_aq0, 1.0e-10_wp, 80, &
                      v_aq, ok_aq, nit)
    if (ok_aq) then
       res%Z_aq = v_aq(1)
       res%eps_r = v_aq(2)
       res%chi_aq = v_aq(3)
    end if
    res%n_iter_aq_inner = nit
    v_g0 = [1.0_wp, 0.99_wp]
    ex_g%comp = y_gas
    ex_g%mi = 0.0_wp
    ex_g%T = T
    ex_g%P = P_Pa
    call newton_solve(par, ex_g, gas_residual, 2, v_g0, 1.0e-10_wp, 80, &
                      v_g, okg, nit)
    if (okg) then
       res%Z_gas = v_g(1)
       res%chi_gas = v_g(2)
    end if
    res%n_iter_gas_inner = nit

    do i = 1, 3
       res%fug_aq(i) = exp(ln_aq(i)) * x_aq(i) * Pbar
       res%fug_gas(i) = exp(ln_gas(i)) * y_gas(i) * Pbar
    end do

    res%converged = .true.
    res%K = K
    res%x_aq = x_aq
    res%y_gas = y_gas
    res%beta = beta
    res%x1w = x_aq(1)
    ! Iteration counts under this route's own convention, copied from
    ! ms_ecpa/flash.py::dew_point_flash so the two kernels report the
    ! same integers. They are NOT comparable with the K-space flash:
    !   n_iter_ssi     = 0  -- no K-space SSI runs here; this is a
    !                         specialised rescue, not an SSI flash. The
    !                         work is the DEW_REFINE_ITERS successive-
    !                         substitution sweeps above, each re-solving
    !                         solve_beta, and that count is not recorded.
    !   n_iter_newton  = 1  -- one Brent solve on beta, not one Newton
    !                         step. The two diagnostic Newtons below are
    !                         counted honestly in n_iter_aq_inner /
    !                         n_iter_gas_inner.
    ! A caller averaging solver cost over mixed routes must split on
    ! solve_path first; dew-rescued rows bias both means down.
    res%n_iter_ssi = 0
    res%n_iter_newton = 1
    res%max_residual = resid
    ! Under "chain", recompute mi_aq at the converged beta / x_aq(1).
    if (sconv == SALINITY_CHAIN .and. (1.0_wp - beta) > 0.0_wp &
        .and. x_aq(1) > 0.0_wp) then
       ! partition-basis water fraction = x_aq(1) * (1 + S); see
       ! aqueous_x1w_chain.
       S_dew = par%Mw * z(1) * sum(mi) / (1.0_wp - beta)
       res%mi_aq = mi * (z(1) / ((1.0_wp - beta) * x_aq(1) * (1.0_wp + S_dew)))
    else
       res%mi_aq = mi
    end if
    ok = .true.

  contains

    subroutine solve_beta(bout, okb_out)
      !! Brent on the water fugacity balance at the CURRENT x_aq / f_w_aq,
      !! both reached by host association so each refinement sweep re-solves
      !! against the composition it just produced.
      real(wp), intent(out) :: bout
      logical, intent(out) :: okb_out
      real(wp) :: blo, bhi, rlo, rhi
      integer :: j, ierr_l
      okb_out = .false.
      bout = 0.0_wp
      blo = 1.0e-8_wp
      do j = 1, 3
         if (x_aq(j) > 1.0e-12_wp) then
            blo = max(blo, 1.0_wp - z(j) / x_aq(j))
         end if
      end do
      blo = blo + 1.0e-8_wp
      bhi = 1.0_wp - 1.0e-8_wp
      if (blo >= bhi) return
      rlo = water_residual(blo)
      rhi = water_residual(bhi)
      if (.not. (ieee_is_finite(rlo) .and. ieee_is_finite(rhi))) return
      if (rlo * rhi > 0.0_wp) return
      call brentq(water_residual, blo, bhi, beta_tol, rtol_default, 200, &
                  bout, ierr_l)
      okb_out = (ierr_l == 0)
    end subroutine solve_beta

    function y_gas_of_beta(b) result(y)
      real(wp), intent(in) :: b
      real(wp) :: y(3), s
      y = (z - (1.0_wp - b) * x_aq) / b
      y = max(y, 0.0_wp)
      s = sum(y)
      if (s > 0.0_wp) y = y / s
    end function y_gas_of_beta

    function water_residual(b) result(r)
      real(wp), intent(in) :: b
      real(wp) :: r
      real(wp) :: y(3), lg_loc(3), vg_loc(2), f_w_gas
      logical :: okg_loc
      y = y_gas_of_beta(b)
      call lnphi_gas(par, y, T, P_Pa, -1.0_wp, 0.99_wp, lg_loc, vg_loc, okg_loc)
      if (.not. okg_loc) then
         r = nan_wp()
         return
      end if
      f_w_gas = y(1) * exp(lg_loc(1)) * Pbar
      r = f_w_aq - f_w_gas
    end function water_residual

  end subroutine dew_point_flash

end module ecpa_flash
