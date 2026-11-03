! ecpa_stability.f90 -- Michelsen TPD stability + combined driver
! (faithful port of ms_ecpa/stability.py: tpd_ssi, michelsen_tpd,
! make_k_init_from_trial, stability_flash, incl. the dew-point rescue).
module ecpa_stability
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use ecpa_kinds, only: wp
  use ecpa_params, only: par_t
  use ecpa_newton, only: nan_wp, inf_wp, clip_np, maxabs_nan
  use ecpa_lnphi, only: lnphi_aq, lnphi_gas
  use ecpa_flash, only: flashopt_t, flashres_t, flash_kv, wilson_k, &
       dew_point_flash, SALINITY_AS_AQUEOUS, SALINITY_CHAIN
  implicit none
  private
  public :: tpdres_t, stabres_t, stabflash_t, tpd_ssi, michelsen_tpd, &
            make_k_init_from_trial, stability_flash

  integer, parameter :: SIDE_AQ = 1, SIDE_GAS = 2
  integer, parameter :: MAXTRIALS = 8

  ! Evaluate aqueous stability trials on the flash's composition basis.
  ! See the note at the call site. Mirrors ms_ecpa/stability.py.
  logical, parameter :: AQ_TRIAL_ON_FLASH_BASIS = .true.

  type :: tpdres_t
     real(wp) :: tpd = 0.0_wp
     real(wp) :: W(3) = 0.0_wp
     real(wp) :: w_frac(3) = 0.0_wp
     real(wp) :: lnphi_trial(3) = 0.0_wp
     integer :: n_iter = 0
     logical :: converged = .false.
     integer :: side = 0
  end type tpdres_t

  type :: stabres_t
     logical :: stable = .true.
     real(wp) :: tpd_min = 0.0_wp
     integer :: ntrials = 0
     type(tpdres_t) :: trials(MAXTRIALS)
     integer :: best_idx = 0            ! 0 = none (Python -1)
     real(wp) :: d_ref(3) = 0.0_wp
     integer :: ierr = 0                ! 0 ok; 1 reference lnphi failed everywhere
  end type stabres_t

  type :: stabflash_t
     integer :: phases = 1
     real(wp) :: tpd_min = 0.0_wp
     logical :: stable = .true.
     logical :: has_flash = .false.
     type(flashres_t) :: flash
     ! route: 0 hard single-phase guard; 1 trial-K; 2 Wilson; 3 water-
     ! biased; 4 pure-SSI; 5 dual-trial; 6 dew rescue; 7 beta-degenerate
     ! single-phase; 8 soft fallback; 9 hard failure; -1 stability error
     integer :: route = 0
  end type stabflash_t

contains

  pure function safe_log(x) result(y)
    real(wp), intent(in) :: x
    real(wp) :: y
    y = log(max(x, 1.0e-300_wp))
  end function safe_log

  ! ------------------------------------------------------------------
  ! Single Michelsen TPD SSI trial in log space (mirrors tpd_ssi).
  ! ------------------------------------------------------------------
  subroutine tpd_ssi(par, d_i, trial_init, T, Pbar, mi, side, present, &
                     accelerated, tol, maxiter, res)
    type(par_t), intent(in) :: par
    real(wp), intent(in) :: d_i(3), trial_init(3), T, Pbar, mi(6)
    integer, intent(in) :: side
    logical, intent(in) :: present(3)
    logical, intent(in) :: accelerated
    real(wp), intent(in) :: tol
    integer, intent(in) :: maxiter
    type(tpdres_t), intent(out) :: res

    real(wp) :: W(3), P_Pa, sum_W, tpd, wn(3), sw
    real(wp) :: ln_s, s_tr, x1w_tr, m_tot_tr
    real(wp) :: lnphi_t(3), lnW(3), g_full(3), g(3), g_p(3), g_prev(3)
    real(wp) :: max_g, lnW_new(3), lnW_step(3), lnW_next(3), m_acc
    real(wp) :: v3(3), v2(2)
    integer :: it, i, n_p, nit
    logical :: converged, have_g_prev, ok
    real(wp) :: w_final(3), tpd_final

    W = trial_init
    do i = 1, 3
       if (present(i)) then
          W(i) = max(W(i), 1.0e-300_wp)
       else
          W(i) = 0.0_wp
       end if
    end do
    P_Pa = Pbar * 1.0e5_wp

    have_g_prev = .false.
    g_prev = 0.0_wp
    m_acc = 1.0_wp
    res%n_iter = 0
    converged = .false.
    lnphi_t = nan_wp()
    sum_W = sum(W)
    tpd = 1.0_wp - sum_W

    iter_loop: do it = 0, maxiter - 1
       sum_W = sum(W)
       if (.not. ieee_is_finite(sum_W) .or. sum_W <= 0.0_wp) exit iter_loop
       do i = 1, 3
          if (present(i)) then
             wn(i) = W(i) / sum_W
             wn(i) = max(wn(i), 1.0e-12_wp)
          else
             wn(i) = 0.0_wp
          end if
       end do
       sw = sum(wn)
       if (sw <= 0.0_wp) exit iter_loop
       wn = wn / sw

       ! Aqueous trials are evaluated on the composition basis the FLASH
       ! uses: x_aq = s*x with s the ion closure, so f_i = s x_i phi_i(s x) P
       ! and ln s enters g_i.  Passing the partition-basis wn with no ln s
       ! (the original convention) leaves the closure factor uncancelled,
       ! because the reference is built on the ion-free gas side.  Mirrors
       ! AQ_TRIAL_ON_FLASH_BASIS in ms_ecpa/stability.py.  Salt-free: s = 1.
       ln_s = 0.0_wp
       if (side == SIDE_AQ) then
          if (AQ_TRIAL_ON_FLASH_BASIS) then
             m_tot_tr = sum(mi)
             if (m_tot_tr > 0.0_wp .and. wn(1) > 0.0_wp) then
                x1w_tr = wn(1) / (1.0_wp + par%Mw * m_tot_tr * wn(1))
                s_tr = 1.0_wp - par%Mw * m_tot_tr * x1w_tr
                ln_s = log(s_tr)
                call lnphi_aq(par, s_tr * wn, mi, T, P_Pa, lnphi_t, v3, ok, nit)
             else
                call lnphi_aq(par, wn, mi, T, P_Pa, lnphi_t, v3, ok, nit)
             end if
          else
             call lnphi_aq(par, wn, mi, T, P_Pa, lnphi_t, v3, ok, nit)
          end if
       else
          call lnphi_gas(par, wn, T, P_Pa, -1.0_wp, 0.99_wp, lnphi_t, v2, ok)
       end if
       if (.not. ok) exit iter_loop
       lnphi_t = lnphi_t + ln_s

       do i = 1, 3
          if (present(i)) then
             lnW(i) = log(max(W(i), 1.0e-300_wp))
          else
             lnW(i) = 0.0_wp
          end if
       end do
       g_full = lnW + lnphi_t - d_i
       do i = 1, 3
          if (present(i)) then
             g(i) = g_full(i)
          else
             g(i) = 0.0_wp
          end if
       end do
       n_p = 0
       do i = 1, 3
          if (present(i)) then
             n_p = n_p + 1
             g_p(n_p) = g(i)
          end if
       end do
       max_g = maxabs_nan(g_p(1:n_p))

       lnW_new = d_i - lnphi_t
       do i = 1, 3
          if (present(i)) then
             lnW_step(i) = lnW_new(i) - lnW(i)
          else
             lnW_step(i) = 0.0_wp
          end if
       end do
       lnW_step = clip_np(lnW_step, -5.0_wp, 5.0_wp)

       if (accelerated .and. have_g_prev) then
          block
            real(wp) :: num_a, denom_a
            integer :: k_
            num_a = 0.0_wp
            denom_a = 0.0_wp
            do k_ = 1, n_p
               num_a = num_a + g_prev(k_) * g_prev(k_)
               denom_a = denom_a + g_prev(k_) * (g_prev(k_) - g_p(k_))
            end do
            if (abs(denom_a) > 1.0e-30_wp) then
               m_acc = abs(num_a / denom_a * m_acc)
               m_acc = min(max(m_acc, 1.0_wp), 10.0_wp)
            else
               m_acc = 1.0_wp
            end if
          end block
       end if
       g_prev(1:n_p) = g_p(1:n_p)
       have_g_prev = .true.

       lnW_next = lnW + m_acc * lnW_step
       lnW_next = clip_np(lnW_next, -500.0_wp, 500.0_wp)
       do i = 1, 3
          if (present(i)) then
             W(i) = exp(lnW_next(i))
          else
             W(i) = 0.0_wp
          end if
       end do

       sum_W = sum(W)
       if (ieee_is_finite(sum_W)) then
          tpd = 1.0_wp - sum_W
       else
          tpd = nan_wp()
       end if
       res%n_iter = it + 1

       if (max_g < tol) then
          converged = .true.
          exit iter_loop
       end if
    end do iter_loop

    if (all(ieee_is_finite(W))) then
       sum_W = sum(W)
    else
       sum_W = nan_wp()
    end if
    if (ieee_is_finite(sum_W)) then
       tpd_final = 1.0_wp - sum_W
    else
       tpd_final = nan_wp()
    end if
    if (ieee_is_finite(sum_W) .and. sum_W > 0.0_wp) then
       w_final = W / sum_W
    else
       w_final = 0.0_wp
    end if

    res%tpd = tpd_final
    res%W = W
    res%w_frac = w_final
    res%lnphi_trial = lnphi_t
    res%converged = converged
    res%side = side
  end subroutine tpd_ssi

  ! ------------------------------------------------------------------
  ! 8-trial schedule (mirrors _build_trial_list).
  ! ------------------------------------------------------------------
  subroutine build_trial_list(par, z, present, T, Pbar, w0s, sides, ntr)
    type(par_t), intent(in) :: par
    real(wp), intent(in) :: z(3), T, Pbar
    logical, intent(in) :: present(3)
    real(wp), intent(out) :: w0s(3, MAXTRIALS)
    integer, intent(out) :: sides(MAXTRIALS), ntr
    real(wp) :: K_wil(3), w(3), gas_share, z_gases(2), split(2), gas_total
    real(wp) :: w_aq_wil(3), w_gas_wil(3), s_aq, s_gas
    integer :: n_present_gas, i

    K_wil = wilson_k(par, T, Pbar)
    ntr = 0

    if (present(3)) then
       w = [0.01_wp, 0.0_wp, 0.99_wp]
       ntr = ntr + 1; w0s(:, ntr) = w; sides(ntr) = SIDE_GAS
       w = [0.10_wp, 0.0_wp, 0.90_wp]
       ntr = ntr + 1; w0s(:, ntr) = w; sides(ntr) = SIDE_GAS
    end if
    if (present(2)) then
       w = [0.01_wp, 0.99_wp, 0.0_wp]
       ntr = ntr + 1; w0s(:, ntr) = w; sides(ntr) = SIDE_GAS
       w = [0.10_wp, 0.90_wp, 0.0_wp]
       ntr = ntr + 1; w0s(:, ntr) = w; sides(ntr) = SIDE_GAS
    end if

    ! aq water-rich
    w = 0.0_wp
    w(1) = 0.99_wp
    n_present_gas = 0
    if (present(2)) n_present_gas = n_present_gas + 1
    if (present(3)) n_present_gas = n_present_gas + 1
    if (n_present_gas > 0) then
       gas_share = (1.0_wp - w(1)) / real(n_present_gas, wp)
       if (present(2)) w(2) = gas_share
       if (present(3)) w(3) = gas_share
    else
       w(1) = 1.0_wp
    end if
    ntr = ntr + 1; w0s(:, ntr) = w; sides(ntr) = SIDE_AQ

    ! aq moderate
    w = 0.0_wp
    if (n_present_gas == 2) then
       z_gases = [z(2), z(3)]
       if (sum(z_gases) > 0.0_wp) then
          split = z_gases / sum(z_gases)
       else
          split = [0.5_wp, 0.5_wp]
       end if
       gas_total = 0.10_wp
       w(2) = gas_total * split(1)
       w(3) = gas_total * split(2)
       w(1) = 1.0_wp - gas_total
    else if (n_present_gas == 1) then
       gas_total = 0.05_wp
       if (present(2)) w(2) = gas_total
       if (present(3)) w(3) = gas_total
       w(1) = 1.0_wp - gas_total
    else
       w(1) = 1.0_wp
    end if
    ntr = ntr + 1; w0s(:, ntr) = w; sides(ntr) = SIDE_AQ

    ! Wilson trials
    do i = 1, 3
       if (present(i)) then
          w_aq_wil(i) = z(i) / K_wil(i)
          w_gas_wil(i) = K_wil(i) * z(i)
       else
          w_aq_wil(i) = 0.0_wp
          w_gas_wil(i) = 0.0_wp
       end if
    end do
    s_aq = sum(w_aq_wil)
    s_gas = sum(w_gas_wil)
    if (s_aq > 0.0_wp) then
       ntr = ntr + 1; w0s(:, ntr) = w_aq_wil / s_aq; sides(ntr) = SIDE_AQ
    end if
    if (s_gas > 0.0_wp) then
       ntr = ntr + 1; w0s(:, ntr) = w_gas_wil / s_gas; sides(ntr) = SIDE_GAS
    end if
  end subroutine build_trial_list

  ! ------------------------------------------------------------------
  ! michelsen_tpd (mirrors the Python function; tpd_threshold is the
  ! POSITIVE classification threshold as in the Python signature).
  ! ------------------------------------------------------------------
  subroutine michelsen_tpd(par, z, mi, T, Pbar, tpd_threshold, res)
    type(par_t), intent(in) :: par
    real(wp), intent(in) :: z(3), mi(6), T, Pbar, tpd_threshold
    type(stabres_t), intent(out) :: res

    logical, parameter :: accelerated = .true.
    real(wp), parameter :: tol = 1.0e-8_wp
    integer, parameter :: maxiter = 80
    real(wp), parameter :: early_tpd = -0.5_wp
    real(wp), parameter :: trivial_w_atol = 5.0e-3_wp

    logical :: present(3), got_ref, ok
    real(wp) :: P_Pa, z_ref(3), s, b_mix, B_mix_cap, Zh_B
    real(wp) :: Z_hints(7), lnphi_ref(3), v2(2), v3(3)
    real(wp) :: m_tot_ref, x1w_ref, sc_ref, x_aq_ref(3), x1w_brine
    real(wp) :: d_ref(3), w0s(3, MAXTRIALS), W0(3)
    integer :: sides(MAXTRIALS), ntr, i, i_t, nit
    real(wp) :: tpd_min, z_present(3), w_arr(3), maxdiff
    integer :: best_idx
    logical :: trivial
    type(tpdres_t) :: tr

    res = stabres_t()
    present = (z > 0.0_wp)
    P_Pa = Pbar * 1.0e5_wp

    do i = 1, 3
       if (present(i)) then
          z_ref(i) = z(i)
       else
          z_ref(i) = 0.0_wp
       end if
    end do
    s = sum(z_ref)
    if (s <= 0.0_wp) then
       res%ierr = 1
       return
    end if
    z_ref = z_ref / s

    b_mix = par%b1 * z_ref(1) + par%b2 * z_ref(2) + par%b3 * z_ref(3)
    B_mix_cap = b_mix * P_Pa / par%R / T
    Zh_B = max(B_mix_cap * 1.2_wp + 0.3_wp, 1.0_wp)
    Z_hints = [1.0_wp, Zh_B, max(Zh_B * 1.3_wp, 2.0_wp), 0.7_wp, 0.5_wp, &
               max(B_mix_cap * 1.5_wp, 0.4_wp), 0.3_wp]
    got_ref = .false.
    do i = 1, 7
       call lnphi_gas(par, z_ref, T, P_Pa, Z_hints(i), 0.99_wp, lnphi_ref, v2, ok)
       if (ok) then
          if (all(ieee_is_finite(lnphi_ref))) then
             got_ref = .true.
             exit
          end if
       end if
    end do
    if (.not. got_ref) then
       ! Aqueous-side reference fallback
       m_tot_ref = sum(mi)
       if (m_tot_ref > 0.0_wp) then
          x1w_ref = z_ref(1) / (1.0_wp + par%Mw * m_tot_ref * z_ref(1))
          sc_ref = 1.0_wp - par%Mw * m_tot_ref * x1w_ref
       else
          x1w_ref = z_ref(1)
          sc_ref = 1.0_wp
       end if
       x_aq_ref = [x1w_ref, z_ref(2) * sc_ref, z_ref(3) * sc_ref]
       call lnphi_aq(par, x_aq_ref, mi, T, P_Pa, lnphi_ref, v3, ok, nit)
       if (.not. ok) then
          if (m_tot_ref > 0.0_wp) then
             x1w_brine = 1.0_wp / (1.0_wp + par%Mw * m_tot_ref)
          else
             x1w_brine = 1.0_wp
          end if
          call lnphi_aq(par, [x1w_brine, 0.0_wp, 0.0_wp], mi, T, P_Pa, &
                        lnphi_ref, v3, ok, nit)
          if (.not. ok) then
             res%ierr = 1
             return
          end if
       end if
    end if

    do i = 1, 3
       d_ref(i) = safe_log(z_ref(i)) + lnphi_ref(i)
    end do
    do i = 1, 3
       if (.not. present(i)) d_ref(i) = 0.0_wp
    end do

    call build_trial_list(par, z, present, T, Pbar, w0s, sides, ntr)

    tpd_min = 0.0_wp
    best_idx = 0
    do i = 1, 3
       if (present(i)) then
          z_present(i) = z(i)
       else
          z_present(i) = 0.0_wp
       end if
    end do
    res%ntrials = 0
    do i_t = 1, ntr
       do i = 1, 3
          if (present(i)) then
             W0(i) = max(w0s(i, i_t), 1.0e-12_wp)
          else
             W0(i) = 0.0_wp
          end if
       end do
       call tpd_ssi(par, d_ref, W0, T, Pbar, mi, sides(i_t), present, &
                    accelerated, tol, maxiter, tr)
       res%ntrials = res%ntrials + 1
       res%trials(res%ntrials) = tr
       w_arr = tr%w_frac
       maxdiff = maxabs_nan(w_arr - z_present)
       trivial = (maxdiff < trivial_w_atol)
       ! A trial that did not converge has not found a stationary point:
       ! its tpd is wherever the SSI loop happened to stop -- maxiter
       ! exhaustion, a non-finite or non-positive sum_W, or a failed
       ! lnphi_aq/lnphi_gas -- evaluated on a partially-updated W.  Such a
       ! trial must not set tpd_min, nor become best_idx and so supply the
       ! w_frac that seeds make_k_init_from_trial and the FIXED aqueous
       ! composition of the dew-point rescue.  Mirrors the same guard in
       ! ms_ecpa/stability.py::michelsen_tpd; the dual-trial block in
       ! stability_flash below already requires tr%converged.
       if (tr%converged) then
          if ((.not. trivial) .and. ieee_is_finite(tr%tpd) .and. tr%tpd < tpd_min) then
             tpd_min = tr%tpd
             best_idx = res%ntrials
          end if
          if ((.not. trivial) .and. ieee_is_finite(tr%tpd) .and. &
              tr%tpd < early_tpd) exit
       end if
    end do

    if (best_idx <= 0) then
       res%stable = .true.
    else
       res%stable = (tpd_min > -tpd_threshold)
    end if
    res%tpd_min = tpd_min
    res%best_idx = best_idx
    res%d_ref = d_ref
  end subroutine michelsen_tpd

  ! ------------------------------------------------------------------
  ! K init from best trial (mirrors make_k_init_from_trial).
  ! ------------------------------------------------------------------
  function make_k_init_from_trial(par, tr, z, T, Pbar) result(K)
    type(par_t), intent(in) :: par
    type(tpdres_t), intent(in) :: tr
    real(wp), intent(in) :: z(3), T, Pbar
    real(wp) :: K(3)
    logical :: present(3)
    real(wp) :: K_wil(3), num, denom
    real(wp), parameter :: eps = 1.0e-12_wp
    integer :: i

    present = (z > 0.0_wp)
    K_wil = wilson_k(par, T, Pbar)
    do i = 1, 3
       if (present(i)) then
          K(i) = K_wil(i)
       else
          K(i) = nan_wp()
       end if
    end do
    do i = 1, 3
       if (.not. present(i)) cycle
       if (tr%side == SIDE_GAS) then
          denom = max(z(i), eps)
          num = max(tr%w_frac(i), eps)
          K(i) = num / denom
       else
          num = max(z(i), eps)
          denom = max(tr%w_frac(i), eps)
          K(i) = num / denom
       end if
    end do
    do i = 1, 3
       if (present(i)) then
          K(i) = clip_np(K(i), 1.0e-6_wp, 1.0e6_wp)
       else
          K(i) = nan_wp()
       end if
    end do
    if (present(1) .and. ieee_is_finite(K(1))) then
       if (K(1) > 2.0_wp) K(1) = K_wil(1)
    end if
  end function make_k_init_from_trial

  ! ------------------------------------------------------------------
  ! Split / phase-swap classifiers (mirror _is_real_split /
  ! _is_phase_swapped).
  ! ------------------------------------------------------------------
  logical function is_real_split(fr)
    type(flashres_t), intent(in) :: fr
    real(wp), parameter :: eps_beta = 1.0e-4_wp
    is_real_split = .false.
    if (.not. fr%converged) return
    if (.not. ieee_is_finite(fr%beta)) return
    is_real_split = (fr%beta > eps_beta .and. fr%beta < 1.0_wp - eps_beta)
  end function is_real_split

  logical function is_phase_swapped(fr)
    type(flashres_t), intent(in) :: fr
    real(wp) :: x_w_aq, y_w_gas
    is_phase_swapped = .false.
    if (.not. fr%converged) return
    x_w_aq = fr%x_aq(1)
    y_w_gas = fr%y_gas(1)
    if (.not. (ieee_is_finite(x_w_aq) .and. ieee_is_finite(y_w_gas))) return
    if (y_w_gas > 0.5_wp .and. y_w_gas > 2.0_wp * x_w_aq) then
       is_phase_swapped = .true.
       return
    end if
    if (ieee_is_finite(fr%Z_aq) .and. ieee_is_finite(fr%Z_gas)) then
       if (fr%Z_aq > fr%Z_gas + 0.1_wp) then
          is_phase_swapped = .true.
          return
       end if
    end if
    if (ieee_is_finite(fr%chi_aq)) then
       if (x_w_aq > 0.1_wp .and. fr%chi_aq > 0.8_wp) then
          is_phase_swapped = .true.
          return
       end if
    end if
  end function is_phase_swapped

  ! ------------------------------------------------------------------
  ! Combined stability + flash driver (mirrors stability_flash).
  ! warm (optional, caller-owned): previous-step state; when valid it
  ! adds a leading warm flash attempt and seeds the inner solves --
  ! cold calls are bit-identical to the Python reference.
  ! ------------------------------------------------------------------
  ! salinity_convention is OPTIONAL and defaults to "as-aqueous"
  ! (SALINITY_AS_AQUEOUS), matching the Python default.  It is threaded
  ! into every flash_kv attempt and into the dew-point rescue, exactly
  ! as Python forwards it through **flash_kwargs.  The Michelsen TPD
  ! stage does NOT see it -- neither does the reference implementation.
  subroutine stability_flash(par, T, Pbar, z, mi, sf, stab_out, warm_opt, &
                             salinity_convention)
    type(par_t), intent(in) :: par
    real(wp), intent(in) :: T, Pbar, z(3), mi(6)
    type(stabflash_t), intent(out) :: sf
    type(stabres_t), intent(out) :: stab_out
    type(flashopt_t), intent(in), optional :: warm_opt
    integer, intent(in), optional :: salinity_convention
    integer :: sconv

    real(wp), parameter :: tpd_threshold = -0.005_wp
    real(wp), parameter :: tpd_soft_threshold = -0.1_wp

    type(stabres_t) :: stab
    type(flashres_t) :: fr, last_fr, last_degenerate_fr
    logical :: have_last_fr, have_last_degen
    type(flashopt_t) :: opt
    real(wp) :: K_trial(3), K_wb(3), K_dual(3), x_aq_TPD(3)
    integer :: i, ig, ia
    logical :: found_dual, dew_ok
    real(wp) :: tg, ta
    real(wp), parameter :: eps = 1.0e-12_wp

    sf = stabflash_t()
    sconv = SALINITY_AS_AQUEOUS
    if (present(salinity_convention)) sconv = salinity_convention

    ! Optional simulator warm-start attempt (extension; OFF for golden
    ! verification -- see module header).
    if (present(warm_opt)) then
       if (warm_opt%have_kinit) then
          call flash_kv(par, T, Pbar, z, mi, warm_opt, fr)
          if (is_real_split(fr)) then
             if (.not. is_phase_swapped(fr)) then
                call michelsen_tpd(par, z, mi, T, Pbar, -tpd_threshold, stab)
                ! The warm K-vector is a guess carried over from another
                ! state, so its split is only accepted when the TPD test
                ! agrees the feed is unstable -- otherwise a held-over
                ! K-vector can give Rachford-Rice a split inside the
                ! weakly-negative band that tpd_threshold exists to absorb,
                ! and the same state would come back single-phase on a cold
                ! call.  ierr /= 0 means no reference lnphi was obtained, so
                ! stab%tpd_min is the type default (0.0) and not a measured
                ! tangent-plane distance; it must not be published as one.
                ! Both cases fall through to the cold path below, which
                ! recomputes the test and reports route 0 or route -1.
                if (stab%ierr == 0 .and. stab%tpd_min <= tpd_threshold) then
                   stab_out = stab
                   sf%phases = 2
                   sf%tpd_min = stab%tpd_min
                   sf%stable = .false.
                   sf%has_flash = .true.
                   sf%flash = fr
                   sf%route = 10
                   return
                end if
             end if
          end if
       end if
    end if

    call michelsen_tpd(par, z, mi, T, Pbar, -tpd_threshold, stab)
    stab_out = stab
    if (stab%ierr /= 0) then
       sf%route = -1
       sf%phases = 1
       sf%stable = .true.
       sf%tpd_min = nan_wp()
       return
    end if

    sf%tpd_min = stab%tpd_min

    ! Hard guard
    if (stab%tpd_min > tpd_threshold) then
       sf%phases = 1
       sf%stable = .true.
       sf%route = 0
       return
    end if

    have_last_fr = .false.
    have_last_degen = .false.

    ! -- Attempt 1: K_init from best trial ---------------------------
    if (stab%best_idx > 0) then
       K_trial = make_k_init_from_trial(par, stab%trials(stab%best_idx), z, T, Pbar)
       opt = flashopt_t()
       opt%salinity_convention = sconv
       opt%have_kinit = .true.
       opt%K_init = K_trial
       call flash_kv(par, T, Pbar, z, mi, opt, fr)
       if (is_real_split(fr)) then
          if (.not. is_phase_swapped(fr)) then
             call finish_two_phase(fr, 1)
             return
          end if
       end if
       if (fr%converged) then
          last_degenerate_fr = fr
          have_last_degen = .true.
       end if
    end if

    ! -- Attempt 2: Wilson K_init ------------------------------------
    opt = flashopt_t()
    opt%salinity_convention = sconv
    opt%have_kinit = .true.
    opt%K_init = wilson_k(par, T, Pbar)
    call flash_kv(par, T, Pbar, z, mi, opt, fr)
    if (is_real_split(fr)) then
       if (.not. is_phase_swapped(fr)) then
          call finish_two_phase(fr, 2)
          return
       end if
    end if
    if (fr%converged) then
       last_degenerate_fr = fr
       have_last_degen = .true.
    end if
    last_fr = fr
    have_last_fr = .true.

    ! -- Attempt 3: water-biased K -----------------------------------
    K_wb = wilson_k(par, T, Pbar)
    do i = 1, 3
       if (.not. (z(i) > 0.0_wp)) K_wb(i) = nan_wp()
    end do
    if (z(1) > 0.0_wp) K_wb(1) = min(K_wb(1), 1.0e-3_wp)
    do i = 2, 3
       if (z(i) > 0.0_wp) K_wb(i) = max(K_wb(i), 5.0_wp)
    end do
    opt = flashopt_t()
    opt%salinity_convention = sconv
    opt%have_kinit = .true.
    opt%K_init = K_wb
    call flash_kv(par, T, Pbar, z, mi, opt, fr)
    if (is_real_split(fr)) then
       if (.not. is_phase_swapped(fr)) then
          call finish_two_phase(fr, 3)
          return
       end if
    end if
    if (fr%converged) then
       last_degenerate_fr = fr
       have_last_degen = .true.
    end if
    if (.not. have_last_fr) then
       last_fr = fr
       have_last_fr = .true.
    else if (fr%n_iter_ssi > last_fr%n_iter_ssi) then
       last_fr = fr
    end if

    ! -- Attempt 4: pure SSI -----------------------------------------
    opt = flashopt_t()
    opt%salinity_convention = sconv
    opt%have_kinit = .true.
    opt%K_init = wilson_k(par, T, Pbar)
    opt%accelerated = .false.
    opt%use_newton = .false.
    opt%maxiter = 200
    call flash_kv(par, T, Pbar, z, mi, opt, fr)
    if (is_real_split(fr)) then
       if (.not. is_phase_swapped(fr)) then
          call finish_two_phase(fr, 4)
          return
       end if
    end if
    if (fr%converged) then
       last_degenerate_fr = fr
       have_last_degen = .true.
    end if
    if (.not. have_last_fr) then
       last_fr = fr
       have_last_fr = .true.
    else if (fr%n_iter_ssi > last_fr%n_iter_ssi) then
       last_fr = fr
    end if

    ! -- Attempt 4b: dual-trial K-init -------------------------------
    ig = 0; ia = 0
    tg = 0.0_wp; ta = 0.0_wp
    do i = 1, stab%ntrials
       associate (tr => stab%trials(i))
         if (.not. (tr%converged .and. ieee_is_finite(tr%tpd) .and. &
                    tr%tpd < 0.0_wp)) cycle
         if (tr%side == SIDE_GAS) then
            if (ig == 0 .or. tr%tpd < tg) then
               ig = i; tg = tr%tpd
            end if
         end if
         if (tr%side == SIDE_AQ) then
            if (ia == 0 .or. tr%tpd < ta) then
               ia = i; ta = tr%tpd
            end if
         end if
       end associate
    end do
    found_dual = (ig > 0 .and. ia > 0)
    if (found_dual) then
       do i = 1, 3
          if (z(i) > 0.0_wp) then
             K_dual(i) = max(stab%trials(ig)%w_frac(i), eps) &
                         / max(stab%trials(ia)%w_frac(i), eps)
             K_dual(i) = min(max(K_dual(i), 1.0e-6_wp), 1.0e6_wp)
          else
             K_dual(i) = nan_wp()
          end if
       end do
       opt = flashopt_t()
       opt%salinity_convention = sconv
       opt%have_kinit = .true.
       opt%K_init = K_dual
       call flash_kv(par, T, Pbar, z, mi, opt, fr)
       if (is_real_split(fr)) then
          if (.not. is_phase_swapped(fr)) then
             call finish_two_phase(fr, 5)
             return
          end if
       end if
       if (fr%converged) then
          last_degenerate_fr = fr
          have_last_degen = .true.
       end if
    end if

    ! -- Attempt 5: dew-point rescue ---------------------------------
    if (stab%tpd_min < -1.0e-3_wp .and. stab%best_idx > 0) then
       associate (bt => stab%trials(stab%best_idx))
         if (bt%side == SIDE_AQ .and. bt%w_frac(1) > 0.5_wp) then
            x_aq_TPD = bt%w_frac / sum(bt%w_frac)
            call dew_point_flash(par, T, Pbar, z, mi, x_aq_TPD, fr, dew_ok, &
                                 salinity_convention=sconv)
            if (dew_ok) then
               if (is_real_split(fr)) then
                  call finish_two_phase(fr, 6)
                  return
               end if
            end if
         end if
       end associate
    end if

    ! -- beta-degenerate fallback ------------------------------------
    if (have_last_degen) then
       sf%phases = 1
       sf%stable = .true.
       sf%has_flash = .true.
       sf%flash = last_degenerate_fr
       sf%route = 7
       return
    end if

    ! -- soft fallback -----------------------------------------------
    if (stab%tpd_min > tpd_soft_threshold) then
       sf%phases = 1
       sf%stable = .true.
       sf%route = 8
       return
    end if

    ! -- hard failure ------------------------------------------------
    sf%phases = 2
    sf%stable = .false.
    if (have_last_fr) then
       sf%has_flash = .true.
       sf%flash = last_fr
    end if
    sf%route = 9

  contains

    subroutine finish_two_phase(fres, route)
      type(flashres_t), intent(in) :: fres
      integer, intent(in) :: route
      sf%phases = 2
      sf%tpd_min = stab%tpd_min
      sf%stable = .false.
      sf%has_flash = .true.
      sf%flash = fres
      sf%route = route
    end subroutine finish_two_phase

  end subroutine stability_flash

end module ecpa_stability
