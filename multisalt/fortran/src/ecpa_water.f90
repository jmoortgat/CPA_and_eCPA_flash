! ecpa_water.f90 -- pure-water auxiliary correlations (not the EoS).
! Mirrors ms_ecpa/water.py.
module ecpa_water
  use ecpa_kinds, only: wp
  implicit none
  private
  public :: rho_w_sat

contains

  ! Saturated-liquid molar density of pure water [mol/m3]
  ! (Wagner & Pruss auxiliary saturation equation, IAPWS-95 eq 2.6).
  pure function rho_w_sat(T) result(r_out)
    real(wp), intent(in) :: T
    real(wp) :: r_out
    real(wp), parameter :: Tc = 647.096_wp, rhoc = 322.0_wp
    real(wp) :: tau, r
    tau = 1.0_wp - min(T, Tc - 1.0e-6_wp) / Tc
    r = 1.0_wp + 1.99274064_wp * tau**(1.0_wp/3.0_wp) &
        + 1.09965342_wp * tau**(2.0_wp/3.0_wp) &
        - 0.510839303_wp * tau**(5.0_wp/3.0_wp) &
        - 1.75493479_wp * tau**(16.0_wp/3.0_wp) &
        - 45.5170352_wp * tau**(43.0_wp/3.0_wp) &
        - 6.74694450e5_wp * tau**(110.0_wp/3.0_wp)
    r_out = r * rhoc / 0.018015268_wp
  end function rho_w_sat

end module ecpa_water
