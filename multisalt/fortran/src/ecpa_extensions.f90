! ecpa_extensions.f90 -- extension points of the aqueous model.
! Mirrors ms_ecpa/extensions.py.
!
! A separately compiled layer may add an aqueous Helmholtz contribution
! and modify the permittivity closure by associating the procedure
! pointers below (and ecpa_params::ext_param_lookup for its own named
! parameters, stored per context in par%ext). With every pointer
! null -- the state of this library on load -- the kernels use the
! neutral values and evaluate exactly the unextended model.
!
! aq_terms_hook(par, T, rho, x1w, x2w, x3w, xiw, active, terms)
!     called once per aqueous state evaluation, only with ions present.
!     active = .false. means "neutral". terms fields (complex, for the
!     complex-step Jacobian):
!       r_d     density factor of the permittivity closure (M and the
!               dipole term at rho*r_d; V-chains carry r_d)
!       w       weight on the water dipole term
!       VdwdV   V dw/dV at fixed N;   NdwdN1  N dw/dN_water at fixed V
!       Z       additive to Z;        lnphi(3) additive to ln phi
! aq_seeds_hook(par, Zw_liq, nextra, seeds)
!     up to 8 extra (Zw, eps_r, chi1w) seeds after the built-in ladder.
! coll_dielectric_hook(par, n, V, T, r_d, w)
! coll_helmholtz_hook(par, n, V, T, active, A)
!     the same modifications for the total-mole colligative assembly.
module ecpa_extensions
  use ecpa_kinds, only: wp
  use ecpa_params, only: par_t
  implicit none
  private
  public :: aq_terms_t, aq_terms_hook, aq_seeds_hook, &
            coll_dielectric_hook, coll_helmholtz_hook, MAX_EXTRA_SEEDS, &
            aq_terms_iface, aq_seeds_iface, coll_dielectric_iface, &
            coll_helmholtz_iface

  integer, parameter :: MAX_EXTRA_SEEDS = 8

  type :: aq_terms_t
     complex(wp) :: r_d = (1.0_wp, 0.0_wp), w = (1.0_wp, 0.0_wp)
     complex(wp) :: VdwdV = (0.0_wp, 0.0_wp), NdwdN1 = (0.0_wp, 0.0_wp)
     complex(wp) :: Z = (0.0_wp, 0.0_wp)
     complex(wp) :: lnphi(3) = (0.0_wp, 0.0_wp)
  end type aq_terms_t

  abstract interface
     subroutine aq_terms_iface(par, T, rho, x1w, x2w, x3w, xiw, active, terms)
       import :: wp, par_t, aq_terms_t
       type(par_t), intent(in) :: par
       real(wp), intent(in) :: T, x1w, x2w, x3w, xiw(6)
       complex(wp), intent(in) :: rho
       logical, intent(out) :: active
       type(aq_terms_t), intent(out) :: terms
     end subroutine aq_terms_iface
     subroutine aq_seeds_iface(par, Zw_liq, nextra, seeds)
       import :: wp, par_t
       type(par_t), intent(in) :: par
       real(wp), intent(in) :: Zw_liq
       integer, intent(out) :: nextra
       real(wp), intent(out) :: seeds(3, 8)
     end subroutine aq_seeds_iface
     subroutine coll_dielectric_iface(par, n, V, T, r_d, w)
       import :: wp, par_t
       type(par_t), intent(in) :: par
       real(wp), intent(in) :: n(7), V, T
       real(wp), intent(out) :: r_d, w
     end subroutine coll_dielectric_iface
     subroutine coll_helmholtz_iface(par, n, V, T, active, A)
       import :: wp, par_t
       type(par_t), intent(in) :: par
       real(wp), intent(in) :: n(7), V, T
       logical, intent(out) :: active
       real(wp), intent(out) :: A
     end subroutine coll_helmholtz_iface
  end interface

  procedure(aq_terms_iface), pointer :: aq_terms_hook => null()
  procedure(aq_seeds_iface), pointer :: aq_seeds_hook => null()
  procedure(coll_dielectric_iface), pointer :: coll_dielectric_hook => null()
  procedure(coll_helmholtz_iface), pointer :: coll_helmholtz_hook => null()

end module ecpa_extensions
