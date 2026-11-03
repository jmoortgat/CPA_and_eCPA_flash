! ecpa_capi.f90 -- C-interoperable API (iso_c_binding) for the eCPA
! multi-salt Fortran engine.
!
! Design (agreed API contract):
!   * Opaque engine context: ecpa_create() / ecpa_destroy().  The
!     context holds ONLY parameters -- no per-call solution state -- so
!     one context per thread/worker serves millions of cells, and
!     multiprocessing pools create one context per worker trivially.
!   * Parameters: ecpa_load_params(ctx, path) reads a .par file;
!     ecpa_set_param(ctx, name, vals, n) / ecpa_get_param update or
!     read a named parameter IN MEMORY (no file round trip) -- for the
!     refit harness objective loop.  File keys and setter names are the
!     SAME namespace (see ecpa_params.f90).
!   * Warm-start state is CALLER-OWNED and per-call: ecpa_warm_t
!     carries (K, Z_aq, Z_gas, eps_r, chi_H2O_aq, chi_H2O_gas, beta)
!     -- exactly the FlashKVResult warm-start tuple.  Pass NULL (or
!     valid=0) for a cold start (bit-faithful Python behaviour, used by
!     the golden-vector gate); pass a previous-step state to seed the
!     SSI K and both inner Newtons.  Converged state is always written
!     back to warm_out when non-NULL; its valid flag is set only for a
!     genuine two-phase split, so a trivial-fixed-point result (beta at
!     0 or 1) is never re-seeded into the next call.
!
! Return convention: 0 = success; nonzero = error/exception analog.
module ecpa_capi
  use, intrinsic :: iso_c_binding
  use ecpa_kinds, only: wp
  use ecpa_params, only: par_t, set_param, get_param, read_par_file
  use ecpa_lnphi, only: lnphi_aq, lnphi_gas
  use ecpa_flash, only: flashopt_t, flashres_t, flash_kv, wilson_k, &
       SALINITY_AS_AQUEOUS, SALINITY_CHAIN
  use ecpa_stability, only: stabres_t, stabflash_t, stability_flash, &
       michelsen_tpd
  use ecpa_newton, only: nan_wp
  use ecpa_colligative, only: phi_osm_total, rho_brine, psat_brine, &
       A_r, ln_f_w, solve_v_liquid
  implicit none
  private

  type :: ctx_t
     type(par_t) :: par
  end type ctx_t

  type, bind(c) :: ecpa_warm_t
     real(c_double) :: K(3)
     real(c_double) :: Z_aq, Z_gas, eps_r, chi_aq, chi_gas, beta
     integer(c_int) :: valid
  end type ecpa_warm_t

  type, bind(c) :: ecpa_flash_out_t
     integer(c_int) :: converged, errcode
     integer(c_int) :: n_iter_ssi, n_iter_newton
     integer(c_int) :: n_iter_aq_inner, n_iter_gas_inner
     real(c_double) :: K(3), x_aq(3), y_gas(3)
     real(c_double) :: beta, x1w
     real(c_double) :: Z_aq, Z_gas, eps_r, chi_aq, chi_gas
     real(c_double) :: max_residual
     real(c_double) :: fug_aq(3), fug_gas(3), mi_aq(6)
  end type ecpa_flash_out_t

  public :: ecpa_warm_t, ecpa_flash_out_t

contains

  subroutine c_to_f_string(cstr, fstr)
    character(kind=c_char), intent(in) :: cstr(*)
    character(len=:), allocatable, intent(out) :: fstr
    integer :: n, i
    n = 0
    do
       if (cstr(n + 1) == c_null_char) exit
       n = n + 1
    end do
    allocate(character(len=n) :: fstr)
    do i = 1, n
       fstr(i:i) = cstr(i)
    end do
  end subroutine c_to_f_string

  ! ------------------------------------------------------------------
  function ecpa_create() bind(c, name='ecpa_create') result(h)
    type(c_ptr) :: h
    type(ctx_t), pointer :: ctx
    allocate(ctx)
    h = c_loc(ctx)
  end function ecpa_create

  subroutine ecpa_destroy(h) bind(c, name='ecpa_destroy')
    type(c_ptr), value :: h
    type(ctx_t), pointer :: ctx
    if (.not. c_associated(h)) return
    call c_f_pointer(h, ctx)
    deallocate(ctx)
  end subroutine ecpa_destroy

  ! ------------------------------------------------------------------
  function ecpa_load_params(h, path) bind(c, name='ecpa_load_params') result(rc)
    type(c_ptr), value :: h
    character(kind=c_char), intent(in) :: path(*)
    integer(c_int) :: rc
    type(ctx_t), pointer :: ctx
    character(len=:), allocatable :: fpath
    integer :: ierr, badline
    if (.not. c_associated(h)) then
       rc = -1
       return
    end if
    call c_f_pointer(h, ctx)
    call c_to_f_string(path, fpath)
    call read_par_file(ctx%par, fpath, ierr, badline)
    if (ierr /= 0) then
       rc = int(ierr * 100000 + badline, c_int)
    else
       rc = 0
    end if
  end function ecpa_load_params

  function ecpa_set_param(h, name, vals, n) bind(c, name='ecpa_set_param') result(rc)
    type(c_ptr), value :: h
    character(kind=c_char), intent(in) :: name(*)
    real(c_double), intent(in) :: vals(*)
    integer(c_int), value :: n
    integer(c_int) :: rc
    type(ctx_t), pointer :: ctx
    character(len=:), allocatable :: fname
    integer :: ierr
    if (.not. c_associated(h)) then
       rc = -1
       return
    end if
    call c_f_pointer(h, ctx)
    call c_to_f_string(name, fname)
    call set_param(ctx%par, fname, vals(1:n), int(n), ierr)
    rc = int(ierr, c_int)
  end function ecpa_set_param

  ! Returns number of values written (>0), or negative error code.
  function ecpa_get_param(h, name, vals, nmax) bind(c, name='ecpa_get_param') result(rc)
    type(c_ptr), value :: h
    character(kind=c_char), intent(in) :: name(*)
    real(c_double), intent(out) :: vals(*)
    integer(c_int), value :: nmax
    integer(c_int) :: rc
    type(ctx_t), pointer :: ctx
    character(len=:), allocatable :: fname
    integer :: ierr, nv
    real(wp) :: buf(36)
    if (.not. c_associated(h)) then
       rc = -1
       return
    end if
    call c_f_pointer(h, ctx)
    call c_to_f_string(name, fname)
    nv = 36
    call get_param(ctx%par, fname, buf, nv, ierr)
    if (ierr /= 0) then
       rc = int(-ierr, c_int)
       return
    end if
    if (nv > nmax) then
       rc = -2
       return
    end if
    vals(1:nv) = buf(1:nv)
    rc = int(nv, c_int)
  end function ecpa_get_param

  ! ------------------------------------------------------------------
  ! lnphi_aq: v_init(1) <= 0 selects the default cold seed + ladder.
  ! ------------------------------------------------------------------
  function ecpa_lnphi_aq(h, x_aq, mi, T, P_Pa, v_init, lnphi, v_out, niter) &
       bind(c, name='ecpa_lnphi_aq') result(rc)
    type(c_ptr), value :: h
    real(c_double), intent(in) :: x_aq(3), mi(6), v_init(3)
    real(c_double), value :: T, P_Pa
    real(c_double), intent(out) :: lnphi(3), v_out(3)
    integer(c_int), intent(out) :: niter
    integer(c_int) :: rc
    type(ctx_t), pointer :: ctx
    real(wp) :: lp(3), vo(3)
    logical :: ok
    integer :: nit
    if (.not. c_associated(h)) then
       rc = -1
       return
    end if
    call c_f_pointer(h, ctx)
    if (v_init(1) > 0.0_c_double) then
       call lnphi_aq(ctx%par, real(x_aq, wp), real(mi, wp), real(T, wp), &
                     real(P_Pa, wp), lp, vo, ok, nit, v_init=real(v_init, wp))
    else
       call lnphi_aq(ctx%par, real(x_aq, wp), real(mi, wp), real(T, wp), &
                     real(P_Pa, wp), lp, vo, ok, nit)
    end if
    lnphi = lp
    v_out = vo
    niter = int(nit, c_int)
    if (ok) then
       rc = 0
    else
       rc = 1
    end if
  end function ecpa_lnphi_aq

  ! lnphi_gas: Z_init <= 0 selects the default (1.0).
  function ecpa_lnphi_gas(h, y_gas, T, P_Pa, Z_init, chi_init, lnphi, v_out) &
       bind(c, name='ecpa_lnphi_gas') result(rc)
    type(c_ptr), value :: h
    real(c_double), intent(in) :: y_gas(3)
    real(c_double), value :: T, P_Pa, Z_init, chi_init
    real(c_double), intent(out) :: lnphi(3), v_out(2)
    integer(c_int) :: rc
    type(ctx_t), pointer :: ctx
    real(wp) :: lp(3), vo(2)
    logical :: ok
    if (.not. c_associated(h)) then
       rc = -1
       return
    end if
    call c_f_pointer(h, ctx)
    call lnphi_gas(ctx%par, real(y_gas, wp), real(T, wp), real(P_Pa, wp), &
                   real(Z_init, wp), real(chi_init, wp), lp, vo, ok)
    lnphi = lp
    v_out = vo
    if (ok) then
       rc = 0
    else
       rc = 1
    end if
  end function ecpa_lnphi_gas

  ! ------------------------------------------------------------------
  subroutine warm_to_opt(warm, opt)
    type(ecpa_warm_t), intent(in) :: warm
    type(flashopt_t), intent(inout) :: opt
    opt%have_kinit = .true.
    opt%K_init = real(warm%K, wp)
    opt%have_warm_aq = .true.
    opt%warm_aq = [real(warm%Z_aq, wp), real(warm%eps_r, wp), real(warm%chi_aq, wp)]
    opt%have_warm_gas = .true.
    opt%warm_gas = [real(warm%Z_gas, wp), real(warm%chi_gas, wp)]
  end subroutine warm_to_opt

  subroutine res_to_out(res, out)
    type(flashres_t), intent(in) :: res
    type(ecpa_flash_out_t), intent(out) :: out
    out%converged = merge(1_c_int, 0_c_int, res%converged)
    out%errcode = int(res%errcode, c_int)
    out%n_iter_ssi = int(res%n_iter_ssi, c_int)
    out%n_iter_newton = int(res%n_iter_newton, c_int)
    out%n_iter_aq_inner = int(res%n_iter_aq_inner, c_int)
    out%n_iter_gas_inner = int(res%n_iter_gas_inner, c_int)
    out%K = res%K
    out%x_aq = res%x_aq
    out%y_gas = res%y_gas
    out%beta = res%beta
    out%x1w = res%x1w
    out%Z_aq = res%Z_aq
    out%Z_gas = res%Z_gas
    out%eps_r = res%eps_r
    out%chi_aq = res%chi_aq
    out%chi_gas = res%chi_gas
    out%max_residual = res%max_residual
    out%fug_aq = res%fug_aq
    out%fug_gas = res%fug_gas
    out%mi_aq = res%mi_aq
  end subroutine res_to_out

  ! Interior-beta test for warm-start validity.  A converged flash can
  ! sit on the trivial fixed point (K -> 1, beta pinned at 0 or 1);
  ! re-seeding the next call from such a state keeps it there, so it is
  ! not a usable two-phase warm start.  Same eps_beta as is_real_split
  ! in ecpa_stability.f90, which is module-private there.
  logical function interior_beta(beta)
    real(wp), intent(in) :: beta
    real(wp), parameter :: eps_beta = 1.0e-4_wp
    interior_beta = .false.
    if (beta /= beta) return    ! NaN test without ieee_arithmetic here
    interior_beta = (beta > eps_beta .and. beta < 1.0_wp - eps_beta)
  end function interior_beta

  subroutine res_to_warm(res, two_phase, warm)
    type(flashres_t), intent(in) :: res
    logical, intent(in) :: two_phase
    type(ecpa_warm_t), intent(out) :: warm
    warm%K = res%K
    warm%Z_aq = res%Z_aq
    warm%Z_gas = res%Z_gas
    warm%eps_r = res%eps_r
    warm%chi_aq = res%chi_aq
    warm%chi_gas = res%chi_gas
    warm%beta = res%beta
    warm%valid = merge(1_c_int, 0_c_int, two_phase .and. res%converged)
  end subroutine res_to_warm

  ! ------------------------------------------------------------------
  ! flash_kv.  salinity_convention: 0 = "as-aqueous" (DEFAULT, as in
  ! Python); 1 = "chain" (opt-in salt-conservation closure).  Any other
  ! value returns error 101.
  ! warm_in / warm_out may be NULL.
  ! ------------------------------------------------------------------
  function ecpa_flash_kv(h, T, Pbar, z, mi, salinity_convention, &
                         warm_in, out, warm_out) &
       bind(c, name='ecpa_flash_kv') result(rc)
    type(c_ptr), value :: h
    real(c_double), value :: T, Pbar
    real(c_double), intent(in) :: z(3), mi(6)
    integer(c_int), value :: salinity_convention
    type(c_ptr), value :: warm_in
    type(ecpa_flash_out_t), intent(out) :: out
    type(c_ptr), value :: warm_out
    integer(c_int) :: rc
    type(ctx_t), pointer :: ctx
    type(ecpa_warm_t), pointer :: wi, wo
    type(flashopt_t) :: opt
    type(flashres_t) :: res
    if (.not. c_associated(h)) then
       rc = -1
       return
    end if
    if (salinity_convention /= SALINITY_AS_AQUEOUS .and. &
        salinity_convention /= SALINITY_CHAIN) then
       rc = 101
       return
    end if
    call c_f_pointer(h, ctx)
    opt = flashopt_t()
    opt%salinity_convention = int(salinity_convention)
    if (c_associated(warm_in)) then
       call c_f_pointer(warm_in, wi)
       if (wi%valid /= 0) call warm_to_opt(wi, opt)
    end if
    call flash_kv(ctx%par, real(T, wp), real(Pbar, wp), real(z, wp), &
                  real(mi, wp), opt, res)
    call res_to_out(res, out)
    if (c_associated(warm_out)) then
       call c_f_pointer(warm_out, wo)
       ! Export as a valid warm start only for a genuine split: the
       ! sibling ecpa_stability_flash has sf%phases for this, flash_kv
       ! has only beta.
       call res_to_warm(res, interior_beta(res%beta), wo)
    end if
    rc = int(res%errcode, c_int)
  end function ecpa_flash_kv

  ! ------------------------------------------------------------------
  ! stability_flash.  phases/tpd_min/route always set; flash outputs
  ! populated when has_flash = 1.
  ! ------------------------------------------------------------------
  function ecpa_stability_flash(h, T, Pbar, z, mi, salinity_convention, &
                                warm_in, phases, tpd_min, route, has_flash, &
                                out, warm_out) &
       bind(c, name='ecpa_stability_flash') result(rc)
    type(c_ptr), value :: h
    real(c_double), value :: T, Pbar
    real(c_double), intent(in) :: z(3), mi(6)
    integer(c_int), value :: salinity_convention
    type(c_ptr), value :: warm_in
    integer(c_int), intent(out) :: phases, route, has_flash
    real(c_double), intent(out) :: tpd_min
    type(ecpa_flash_out_t), intent(out) :: out
    type(c_ptr), value :: warm_out
    integer(c_int) :: rc
    type(ctx_t), pointer :: ctx
    type(ecpa_warm_t), pointer :: wi, wo
    type(flashopt_t) :: wopt
    type(stabflash_t) :: sf
    type(stabres_t) :: stab
    logical :: have_warm
    if (.not. c_associated(h)) then
       rc = -1
       return
    end if
    if (salinity_convention /= SALINITY_AS_AQUEOUS .and. &
        salinity_convention /= SALINITY_CHAIN) then
       rc = 101
       return
    end if
    call c_f_pointer(h, ctx)
    have_warm = .false.
    wopt = flashopt_t()
    wopt%salinity_convention = int(salinity_convention)
    if (c_associated(warm_in)) then
       call c_f_pointer(warm_in, wi)
       if (wi%valid /= 0) then
          call warm_to_opt(wi, wopt)
          have_warm = .true.
       end if
    end if
    if (have_warm) then
       call stability_flash(ctx%par, real(T, wp), real(Pbar, wp), real(z, wp), &
                            real(mi, wp), sf, stab, warm_opt=wopt, &
                            salinity_convention=int(salinity_convention))
    else
       call stability_flash(ctx%par, real(T, wp), real(Pbar, wp), real(z, wp), &
                            real(mi, wp), sf, stab, &
                            salinity_convention=int(salinity_convention))
    end if
    phases = int(sf%phases, c_int)
    tpd_min = sf%tpd_min
    route = int(sf%route, c_int)
    has_flash = merge(1_c_int, 0_c_int, sf%has_flash)
    if (sf%has_flash) then
       call res_to_out(sf%flash, out)
    else
       block
         type(flashres_t) :: empty
         empty%K = nan_wp(); empty%x_aq = nan_wp(); empty%y_gas = nan_wp()
         empty%beta = nan_wp(); empty%x1w = nan_wp()
         empty%Z_aq = nan_wp(); empty%Z_gas = nan_wp(); empty%eps_r = nan_wp()
         empty%chi_aq = nan_wp(); empty%chi_gas = nan_wp()
         empty%fug_aq = nan_wp(); empty%fug_gas = nan_wp()
         empty%mi_aq = nan_wp(); empty%max_residual = nan_wp()
         call res_to_out(empty, out)
       end block
    end if
    if (c_associated(warm_out)) then
       call c_f_pointer(warm_out, wo)
       if (sf%has_flash) then
          call res_to_warm(sf%flash, sf%phases == 2, wo)
       else
          wo%valid = 0_c_int
          wo%K = nan_wp(); wo%Z_aq = nan_wp(); wo%Z_gas = nan_wp()
          wo%eps_r = nan_wp(); wo%chi_aq = nan_wp(); wo%chi_gas = nan_wp()
          wo%beta = nan_wp()
       end if
    end if
    if (sf%route == -1) then
       rc = 2   ! stability reference failure (Python raises RuntimeError)
    else
       rc = 0
    end if
  end function ecpa_stability_flash

  ! ------------------------------------------------------------------
  ! Phase-2 colligative routes.  See ecpa_colligative.f90 for which
  ! Python function each one reproduces.
  ! ------------------------------------------------------------------

  ! Total-mole-basis residual Helmholtz energy A^r/RT (extensive) at a
  ! GIVEN volume: total_mole_basis.py::A_r(basis="total").  This is the
  ! un-amplified primitive underneath phi_osm -- gating it directly is
  ! what separates "the assembly is right" from "the two nested central
  ! differences in phi_osm amplify last-ulp noise" (PORTING_NOTES 15).
  ! n is the 7-vector [n_w, n_Na, n_Cl, n_K, n_Ca, n_Mg, n_SO4] [mol].
  function ecpa_a_res(h, n, V, T, A) &
       bind(c, name='ecpa_a_res') result(rc)
    type(c_ptr), value :: h
    real(c_double), intent(in) :: n(7)
    real(c_double), value :: V, T
    real(c_double), intent(out) :: A
    integer(c_int) :: rc
    type(ctx_t), pointer :: ctx
    real(wp) :: av
    integer :: e
    if (.not. c_associated(h)) then
       rc = -1
       return
    end if
    call c_f_pointer(h, ctx)
    call A_r(ctx%par, real(n, wp), real(V, wp), real(T, wp), &
             av, e)
    A = av
    rc = int(e, c_int)
  end function ecpa_a_res

  ! Liquid-branch volume root and ln f_w [ln Pa] at (T, P):
  ! total_mole_basis.py::solve_V and ::ln_f_w (basis="total").
  function ecpa_ln_f_w(h, n, T, P_Pa, lnf, V) &
       bind(c, name='ecpa_ln_f_w') result(rc)
    type(c_ptr), value :: h
    real(c_double), intent(in) :: n(7)
    real(c_double), value :: T, P_Pa
    real(c_double), intent(out) :: lnf, V
    integer(c_int) :: rc
    type(ctx_t), pointer :: ctx
    real(wp) :: lf, vv
    integer :: e
    if (.not. c_associated(h)) then
       rc = -1
       return
    end if
    call c_f_pointer(h, ctx)
    call ln_f_w(ctx%par, real(n, wp), real(T, wp), real(P_Pa, wp), &
                lf, vv, e)
    lnf = lf
    V = vv
    rc = int(e, c_int)
  end function ecpa_ln_f_w

  ! Osmotic coefficient on the total-mole basis (the phase-4 oracle).
  ! salt: 1 NaCl, 2 KCl, 3 CaCl2, 4 MgCl2, 5 Na2SO4.
  ! (the reference's defaults).
  ! rc 0 = value written (which may still be NaN exactly where the
  ! reference returns nan); -1 = bad context; 102 = unknown salt code.
  function ecpa_phi_osm(h, salt, m, T, P_Pa, phi) &
       bind(c, name='ecpa_phi_osm') result(rc)
    type(c_ptr), value :: h
    integer(c_int), value :: salt
    real(c_double), value :: m, T, P_Pa
    real(c_double), intent(out) :: phi
    integer(c_int) :: rc
    type(ctx_t), pointer :: ctx
    if (.not. c_associated(h)) then
       rc = -1
       return
    end if
    if (salt < 1 .or. salt > 5) then
       rc = 102
       return
    end if
    call c_f_pointer(h, ctx)
    phi = phi_osm_total(ctx%par, int(salt), real(m, wp), real(T, wp), &
                        real(P_Pa, wp))
    rc = 0
  end function ecpa_phi_osm

  ! CO2-free brine mass density [kg/m3] at (T [K], P [bar]).
  ! v_init(1) <= 0 selects the cold seed + fallback ladder.
  ! v_out receives the converged (Z_aq, eps_r, chi_H2O_aq) so a caller
  ! marching an isotherm can feed it back as v_init.
  function ecpa_rho_brine(h, T, Pbar, mi, v_init, rho, v_out) &
       bind(c, name='ecpa_rho_brine') result(rc)
    type(c_ptr), value :: h
    real(c_double), value :: T, Pbar
    real(c_double), intent(in) :: mi(6), v_init(3)
    real(c_double), intent(out) :: rho, v_out(3)
    integer(c_int) :: rc
    type(ctx_t), pointer :: ctx
    real(wp) :: r, vo(3)
    logical :: ok
    if (.not. c_associated(h)) then
       rc = -1
       return
    end if
    call c_f_pointer(h, ctx)
    call rho_brine(ctx%par, real(T, wp), real(Pbar, wp), real(mi, wp), &
                   v_init(1) > 0.0_c_double, real(v_init, wp), r, vo, ok)
    rho = r
    v_out = vo
    if (ok) then
       rc = 0
    else
       rc = 1
    end if
  end function ecpa_rho_brine

  ! NaCl-brine vapour (bubble) pressure [bar] at (T [K], m [mol/kg]).
  ! p0_Pa is the pure-water saturation pressure used to anchor the
  ! Brent bracket -- the reference reads it from CoolProp (IAPWS), so
  ! pass the same value for an identical bracket.  lo_fac/hi_fac <= 0
  ! select the reference defaults (0.50, 1.10).
  function ecpa_psat_brine(h, T, m, p0_Pa, lo_fac, hi_fac, Pbar) &
       bind(c, name='ecpa_psat_brine') result(rc)
    type(c_ptr), value :: h
    real(c_double), value :: T, m, p0_Pa, lo_fac, hi_fac
    real(c_double), intent(out) :: Pbar
    integer(c_int) :: rc
    type(ctx_t), pointer :: ctx
    real(wp) :: p
    if (.not. c_associated(h)) then
       rc = -1
       return
    end if
    call c_f_pointer(h, ctx)
    if (lo_fac > 0.0_c_double .and. hi_fac > 0.0_c_double) then
       p = psat_brine(ctx%par, real(T, wp), real(m, wp), real(p0_Pa, wp), &
                      lo_fac=real(lo_fac, wp), hi_fac=real(hi_fac, wp))
    else
       p = psat_brine(ctx%par, real(T, wp), real(m, wp), real(p0_Pa, wp))
    end if
    Pbar = p
    if (p == p) then            ! NaN test without ieee_arithmetic here
       rc = 0
    else
       rc = 1
    end if
  end function ecpa_psat_brine

end module ecpa_capi
