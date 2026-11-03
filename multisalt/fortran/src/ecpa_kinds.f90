! ecpa_kinds.f90 -- precision kinds for the eCPA multi-salt Fortran port.
! F2008, no vendor extensions.
module ecpa_kinds
  use, intrinsic :: iso_fortran_env, only: real64
  implicit none
  public
  integer, parameter :: wp = real64
end module ecpa_kinds
