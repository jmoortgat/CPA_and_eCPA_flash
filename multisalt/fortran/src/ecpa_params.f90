! ecpa_params.f90 -- runtime parameter store for the eCPA multi-salt port.
!
! Mirrors ms_ecpa/parameters.py, but with EVERY model quantity read at
! runtime (file or C-API setter) -- nothing a refit could change is
! hard-wired in the physics kernels.  One authoritative parameter
! namespace serves both the .par file reader and the programmatic
! set/get API (see set_param / get_param): the file format is literally
! "name = value(s)" over the same names the setter accepts.
!
! Component indexing convention (identical to the Python library):
!   partitioning species 1..3 = H2O, CH4, CO2
!   ions 1..6               = Na+, Cl-, K+, Ca2+, Mg2+, SO4(2-)
!
! Gas-ion Huron-Vidal energy functional forms (per ion slot):
!   U2i_form/U3i_form = 0 : linear   U = A*(T/Tc_gas) + B
!   U2i_form/U3i_form = 1 : quadratic (Paper-I DU_cs form)
!       U = quad_Uref + quad_alfa*R*((1-T/quad_Talfa)**2
!                                    - (1-Tref/quad_Talfa)**2)
module ecpa_params
  use ecpa_kinds, only: wp
  implicit none
  private
  public :: par_t, set_param, get_param, read_par_file, param_count, &
            ext_param_lookup, ext_lookup_iface, EXT_SLOTS

  integer, parameter :: EXT_SLOTS = 64

  type :: par_t
     ! -- universal constants (rounded values are part of the model) --
     real(wp) :: R = 0, Navo = 0, kb = 0, e = 0, eps0 = 0
     ! -- molar masses --
     real(wp) :: Mw = 0, Mh = 0, Mc = 0
     ! -- critical properties --
     real(wp) :: Tc1 = 0, Pc1 = 0, Tc2 = 0, Pc2 = 0, Tc3 = 0, Pc3 = 0
     ! -- pure-component CPA (final numeric values; no re-derivation) --
     real(wp) :: b1 = 0, c11 = 0, a01 = 0
     real(wp) :: b2 = 0, c12 = 0, a02 = 0, w2 = 0
     real(wp) :: b3 = 0, c13 = 0, a03 = 0
     ! -- association --
     real(wp) :: epsW = 0, kappaW = 0
     ! -- permittivity closure --
     real(wp) :: dip01 = 0, pol1 = 0, pol2 = 0, pol3 = 0
     real(wp) :: cosGAMMA1 = 0, cosTHETA1 = 0, zww = 0
     ! -- reference temperature --
     real(wp) :: Tref = 0
     ! -- binary interaction / cross-association coefficients --
     real(wp) :: Ak12 = 0, Bk12 = 0, Ck12 = 0
     real(wp) :: Ak13 = 0, Bk13 = 0, Ck13 = 0
     real(wp) :: AS13 = 0, BS13 = 0, CS13 = 0
     real(wp) :: k23 = 0, S12A = 0
     ! m-CR1 combining rule for the CO2-H2O cross-association ENERGY:
     ! eps_cross = eps_water/2 instead of eps_water. Mirrors
     ! ms_ecpa.parameters.MCR1_CROSS_ENERGY. OFF reproduces Papers I-III.
     integer  :: MCR1_CROSS_ENERGY = 0
     ! -- ion parameters (Na, Cl, K, Ca, Mg, SO4) --
     real(wp) :: Sgi(6) = 0, bi(6) = 0, Rbi(6) = 0, a0i(6) = 0, c1i(6) = 0
     real(wp) :: Uref1i(6) = 0, Talfa1i(6) = 0, alfa1i(6) = 0
     real(wp) :: Zi(6) = 0, poli(6) = 0, MMi(6) = 0
     real(wp) :: A_U2i(6) = 0, B_U2i(6) = 0, A_U3i(6) = 0, B_U3i(6) = 0
     real(wp) :: A_Peni(6) = 0, B_Peni(6) = 0
     integer  :: U2i_form(6) = 0, U3i_form(6) = 0
     real(wp) :: U2i_quad_Uref(6) = 0, U2i_quad_Talfa(6) = 1.0_wp
     real(wp) :: U2i_quad_alfa(6) = 0
     real(wp) :: U3i_quad_Uref(6) = 0, U3i_quad_Talfa(6) = 1.0_wp
     real(wp) :: U3i_quad_alfa(6) = 0
     ! -- ion-ion Huron-Vidal reference energies (row-major symmetric) --
     real(wp) :: Uref_ii(6,6) = 0
     ! -- Wilson-K acentric factors (H2O, CH4, CO2) --
     real(wp) :: omega_wilson(3) = 0
     ! -- storage for parameters owned by an extension layer (see
     !    ecpa_extensions.f90); names are mapped to slots by
     !    ext_param_lookup. Untouched (all 0) without an extension. --
     real(wp) :: ext(EXT_SLOTS) = 0
  end type par_t

  ! Name -> slot map for extension-owned parameters. set_param/get_param
  ! and read_par_file consult it for any name not in PARAM_NAMES.
  ! slot = first index into par%ext, nlen = number of values; slot <= 0
  ! means "not an extension parameter either".
  abstract interface
     subroutine ext_lookup_iface(name, slot, nlen)
       character(len=*), intent(in) :: name
       integer, intent(out) :: slot, nlen
     end subroutine ext_lookup_iface
  end interface
  procedure(ext_lookup_iface), pointer :: ext_param_lookup => null()

  ! ------------------------------------------------------------------
  ! The parameter namespace itself.  This table is the single
  ! authority: expected_len reads the lengths off it, and
  ! read_par_file checks a loaded file against it.  A new quantity
  ! added to par_t is added here and to set_param/get_param.
  ! ------------------------------------------------------------------
  integer, parameter :: NPAR = 73
  character(len=24), parameter :: PARAM_NAMES(NPAR) = &
       [character(len=24) :: &
        'R', 'Na', 'kb', &
        'e', 'eps0', 'Mw', &
        'Mh', 'Mc', 'Tc1', &
        'Pc1', 'Tc2', 'Pc2', &
        'Tc3', 'Pc3', 'b1', &
        'c11', 'a01', 'b2', &
        'c12', 'a02', 'w2', &
        'b3', 'c13', 'a03', &
        'epsW', 'kappaW', 'dip01', &
        'pol1', 'pol2', 'pol3', &
        'cosGAMMA1', 'cosTHETA1', 'zww', &
        'Tref', 'Ak12', 'Bk12', &
        'Ck12', 'Ak13', 'Bk13', &
        'Ck13', 'AS13', 'BS13', &
        'CS13', 'k23', 'S12A', &
        'MCR1_CROSS_ENERGY', 'omega_wilson', &
        'Sgi', 'bi', &
        'Rbi', 'a0i', 'c1i', &
        'Uref1i', 'Talfa1i', 'alfa1i', &
        'Zi', 'poli', 'MMi', &
        'A_U2i', 'B_U2i', 'A_U3i', &
        'B_U3i', 'A_Peni', 'B_Peni', &
        'U2i_form', 'U3i_form', 'U2i_quad_Uref', &
        'U2i_quad_Talfa', 'U2i_quad_alfa', 'U3i_quad_Uref', &
        'U3i_quad_Talfa', 'U3i_quad_alfa', 'Uref_ii']
  ! values per name, in the same order as PARAM_NAMES
  integer, parameter :: PARAM_LENS(NPAR) = &
       [ &
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, &
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, &
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, &
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 6, &
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, &
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, &
        36]

  ! Names a .par file may legitimately leave out.  MCR1_CROSS_ENERGY
  ! selects the m-CR1 cross-association ENERGY rule; its 0 default is
  ! the Papers I-III form, and fortran/tools/gen_params.py does not
  ! write it.  Every other name must appear in the file.
  character(len=24), parameter :: OPTIONAL_NAMES(1) = &
       [character(len=24) :: 'MCR1_CROSS_ENERGY']

contains

  ! Number of names the file reader and the set/get API accept.
  pure function param_count() result(n)
    integer :: n
    n = NPAR
  end function param_count

  ! ------------------------------------------------------------------
  ! set_param: the single authoritative name -> storage map.
  ! nv must match the parameter's length (1, 3, 6 or 36).
  ! ierr: 0 ok; 1 unknown name; 2 wrong length.
  ! ------------------------------------------------------------------
  subroutine set_param(par, name, vals, nv, ierr)
    type(par_t), intent(inout) :: par
    character(len=*), intent(in) :: name
    real(wp), intent(in) :: vals(:)
    integer, intent(in) :: nv
    integer, intent(out) :: ierr
    integer :: need, slot, nlen
    ierr = 0
    need = expected_len(name)
    if (need < 0) then
       ! not a core name: an extension layer's, if one is installed
       ierr = 1
       if (.not. associated(ext_param_lookup)) return
       call ext_param_lookup(name, slot, nlen)
       if (slot <= 0) return
       if (nv /= nlen .or. size(vals) < nv .or. slot + nlen - 1 > EXT_SLOTS) then
          ierr = 2
          return
       end if
       par%ext(slot:slot + nlen - 1) = vals(1:nlen)
       ierr = 0
       return
    end if
    if (nv /= need .or. size(vals) < nv) then
       ierr = 2
       return
    end if
    select case (name)
    case ('R');       par%R = vals(1)
    case ('Na');      par%Navo = vals(1)
    case ('kb');      par%kb = vals(1)
    case ('e');       par%e = vals(1)
    case ('eps0');    par%eps0 = vals(1)
    case ('Mw');      par%Mw = vals(1)
    case ('Mh');      par%Mh = vals(1)
    case ('Mc');      par%Mc = vals(1)
    case ('Tc1');     par%Tc1 = vals(1)
    case ('Pc1');     par%Pc1 = vals(1)
    case ('Tc2');     par%Tc2 = vals(1)
    case ('Pc2');     par%Pc2 = vals(1)
    case ('Tc3');     par%Tc3 = vals(1)
    case ('Pc3');     par%Pc3 = vals(1)
    case ('b1');      par%b1 = vals(1)
    case ('c11');     par%c11 = vals(1)
    case ('a01');     par%a01 = vals(1)
    case ('b2');      par%b2 = vals(1)
    case ('c12');     par%c12 = vals(1)
    case ('a02');     par%a02 = vals(1)
    case ('w2');      par%w2 = vals(1)
    case ('b3');      par%b3 = vals(1)
    case ('c13');     par%c13 = vals(1)
    case ('a03');     par%a03 = vals(1)
    case ('epsW');    par%epsW = vals(1)
    case ('kappaW');  par%kappaW = vals(1)
    case ('dip01');   par%dip01 = vals(1)
    case ('pol1');    par%pol1 = vals(1)
    case ('pol2');    par%pol2 = vals(1)
    case ('pol3');    par%pol3 = vals(1)
    case ('cosGAMMA1'); par%cosGAMMA1 = vals(1)
    case ('cosTHETA1'); par%cosTHETA1 = vals(1)
    case ('zww');     par%zww = vals(1)
    case ('Tref');    par%Tref = vals(1)
    case ('Ak12');    par%Ak12 = vals(1)
    case ('Bk12');    par%Bk12 = vals(1)
    case ('Ck12');    par%Ck12 = vals(1)
    case ('Ak13');    par%Ak13 = vals(1)
    case ('Bk13');    par%Bk13 = vals(1)
    case ('Ck13');    par%Ck13 = vals(1)
    case ('AS13');    par%AS13 = vals(1)
    case ('BS13');    par%BS13 = vals(1)
    case ('CS13');    par%CS13 = vals(1)
    case ('k23');     par%k23 = vals(1)
    case ('S12A');    par%S12A = vals(1)
    case ('MCR1_CROSS_ENERGY');    par%MCR1_CROSS_ENERGY = nint(vals(1))
    case ('Sgi');     par%Sgi = vals(1:6)
    case ('bi');      par%bi = vals(1:6)
    case ('Rbi');     par%Rbi = vals(1:6)
    case ('a0i');     par%a0i = vals(1:6)
    case ('c1i');     par%c1i = vals(1:6)
    case ('Uref1i');  par%Uref1i = vals(1:6)
    case ('Talfa1i'); par%Talfa1i = vals(1:6)
    case ('alfa1i');  par%alfa1i = vals(1:6)
    case ('Zi');      par%Zi = vals(1:6)
    case ('poli');    par%poli = vals(1:6)
    case ('MMi');     par%MMi = vals(1:6)
    case ('A_U2i');   par%A_U2i = vals(1:6)
    case ('B_U2i');   par%B_U2i = vals(1:6)
    case ('A_U3i');   par%A_U3i = vals(1:6)
    case ('B_U3i');   par%B_U3i = vals(1:6)
    case ('A_Peni');  par%A_Peni = vals(1:6)
    case ('B_Peni');  par%B_Peni = vals(1:6)
    case ('U2i_form');      par%U2i_form = nint(vals(1:6))
    case ('U3i_form');      par%U3i_form = nint(vals(1:6))
    case ('U2i_quad_Uref');  par%U2i_quad_Uref = vals(1:6)
    case ('U2i_quad_Talfa'); par%U2i_quad_Talfa = vals(1:6)
    case ('U2i_quad_alfa');  par%U2i_quad_alfa = vals(1:6)
    case ('U3i_quad_Uref');  par%U3i_quad_Uref = vals(1:6)
    case ('U3i_quad_Talfa'); par%U3i_quad_Talfa = vals(1:6)
    case ('U3i_quad_alfa');  par%U3i_quad_alfa = vals(1:6)
    case ('Uref_ii'); par%Uref_ii = reshape(vals(1:36), [6, 6], order=[2, 1])
    case ('omega_wilson'); par%omega_wilson = vals(1:3)
    case default
       ierr = 1
    end select
  end subroutine set_param

  subroutine get_param(par, name, vals, nv, ierr)
    type(par_t), intent(in) :: par
    character(len=*), intent(in) :: name
    real(wp), intent(out) :: vals(:)
    integer, intent(inout) :: nv     ! in: buffer size; out: values written
    integer, intent(out) :: ierr
    integer :: need, slot, nlen
    real(wp) :: tmp(36)
    ierr = 0
    need = expected_len(name)
    if (need < 0) then
       ierr = 1
       if (associated(ext_param_lookup)) then
          call ext_param_lookup(name, slot, nlen)
          if (slot > 0 .and. slot + nlen - 1 <= EXT_SLOTS) then
             if (nv < nlen) then
                ierr = 2; nv = nlen
                return
             end if
             vals(1:nlen) = par%ext(slot:slot + nlen - 1)
             nv = nlen
             ierr = 0
             return
          end if
       end if
       nv = 0
       return
    end if
    if (nv < need) then
       ierr = 2; nv = need
       return
    end if
    tmp = 0
    select case (name)
    case ('R');       tmp(1) = par%R
    case ('Na');      tmp(1) = par%Navo
    case ('kb');      tmp(1) = par%kb
    case ('e');       tmp(1) = par%e
    case ('eps0');    tmp(1) = par%eps0
    case ('Mw');      tmp(1) = par%Mw
    case ('Mh');      tmp(1) = par%Mh
    case ('Mc');      tmp(1) = par%Mc
    case ('Tc1');     tmp(1) = par%Tc1
    case ('Pc1');     tmp(1) = par%Pc1
    case ('Tc2');     tmp(1) = par%Tc2
    case ('Pc2');     tmp(1) = par%Pc2
    case ('Tc3');     tmp(1) = par%Tc3
    case ('Pc3');     tmp(1) = par%Pc3
    case ('b1');      tmp(1) = par%b1
    case ('c11');     tmp(1) = par%c11
    case ('a01');     tmp(1) = par%a01
    case ('b2');      tmp(1) = par%b2
    case ('c12');     tmp(1) = par%c12
    case ('a02');     tmp(1) = par%a02
    case ('w2');      tmp(1) = par%w2
    case ('b3');      tmp(1) = par%b3
    case ('c13');     tmp(1) = par%c13
    case ('a03');     tmp(1) = par%a03
    case ('epsW');    tmp(1) = par%epsW
    case ('kappaW');  tmp(1) = par%kappaW
    case ('dip01');   tmp(1) = par%dip01
    case ('pol1');    tmp(1) = par%pol1
    case ('pol2');    tmp(1) = par%pol2
    case ('pol3');    tmp(1) = par%pol3
    case ('cosGAMMA1'); tmp(1) = par%cosGAMMA1
    case ('cosTHETA1'); tmp(1) = par%cosTHETA1
    case ('zww');     tmp(1) = par%zww
    case ('Tref');    tmp(1) = par%Tref
    case ('Ak12');    tmp(1) = par%Ak12
    case ('Bk12');    tmp(1) = par%Bk12
    case ('Ck12');    tmp(1) = par%Ck12
    case ('Ak13');    tmp(1) = par%Ak13
    case ('Bk13');    tmp(1) = par%Bk13
    case ('Ck13');    tmp(1) = par%Ck13
    case ('AS13');    tmp(1) = par%AS13
    case ('BS13');    tmp(1) = par%BS13
    case ('CS13');    tmp(1) = par%CS13
    case ('k23');     tmp(1) = par%k23
    case ('S12A');    tmp(1) = par%S12A
    case ('MCR1_CROSS_ENERGY');    tmp(1) = real(par%MCR1_CROSS_ENERGY, wp)
    case ('Sgi');     tmp(1:6) = par%Sgi
    case ('bi');      tmp(1:6) = par%bi
    case ('Rbi');     tmp(1:6) = par%Rbi
    case ('a0i');     tmp(1:6) = par%a0i
    case ('c1i');     tmp(1:6) = par%c1i
    case ('Uref1i');  tmp(1:6) = par%Uref1i
    case ('Talfa1i'); tmp(1:6) = par%Talfa1i
    case ('alfa1i');  tmp(1:6) = par%alfa1i
    case ('Zi');      tmp(1:6) = par%Zi
    case ('poli');    tmp(1:6) = par%poli
    case ('MMi');     tmp(1:6) = par%MMi
    case ('A_U2i');   tmp(1:6) = par%A_U2i
    case ('B_U2i');   tmp(1:6) = par%B_U2i
    case ('A_U3i');   tmp(1:6) = par%A_U3i
    case ('B_U3i');   tmp(1:6) = par%B_U3i
    case ('A_Peni');  tmp(1:6) = par%A_Peni
    case ('B_Peni');  tmp(1:6) = par%B_Peni
    case ('U2i_form');      tmp(1:6) = real(par%U2i_form, wp)
    case ('U3i_form');      tmp(1:6) = real(par%U3i_form, wp)
    case ('U2i_quad_Uref');  tmp(1:6) = par%U2i_quad_Uref
    case ('U2i_quad_Talfa'); tmp(1:6) = par%U2i_quad_Talfa
    case ('U2i_quad_alfa');  tmp(1:6) = par%U2i_quad_alfa
    case ('U3i_quad_Uref');  tmp(1:6) = par%U3i_quad_Uref
    case ('U3i_quad_Talfa'); tmp(1:6) = par%U3i_quad_Talfa
    case ('U3i_quad_alfa');  tmp(1:6) = par%U3i_quad_alfa
    case ('Uref_ii'); tmp(1:36) = reshape(transpose(par%Uref_ii), [36])
    case ('omega_wilson'); tmp(1:3) = par%omega_wilson
    case default
       ierr = 1; nv = 0
       return
    end select
    vals(1:need) = tmp(1:need)
    nv = need
  end subroutine get_param

  ! ------------------------------------------------------------------
  ! param_index: position of a name in PARAM_NAMES, or -1 if unknown.
  ! expected_len: how many values that name takes, or -1 if unknown.
  ! ------------------------------------------------------------------
  pure function param_index(name) result(i)
    character(len=*), intent(in) :: name
    integer :: i
    do i = 1, NPAR
       if (PARAM_NAMES(i) == name) return
    end do
    i = -1
  end function param_index

  pure function expected_len(name) result(n)
    character(len=*), intent(in) :: name
    integer :: n
    integer :: i
    i = param_index(name)
    if (i < 0) then
       n = -1
    else
       n = PARAM_LENS(i)
    end if
  end function expected_len

  ! ------------------------------------------------------------------
  ! read_par_file: strict "name = value(s)" reader.  '#' starts a
  ! comment; blank lines skipped; unknown names or wrong value counts
  ! are hard errors (catches typos in generated files).  The file must
  ! parameterise the engine on its own: every name in PARAM_NAMES
  ! except OPTIONAL_NAMES has to appear in it (see the completeness
  ! check at the end).
  ! ierr: 0 ok; 10 cannot open; 11 parse error; 12 required parameter
  ! absent from the file, with badline carrying its PARAM_NAMES index
  ! instead of a line number; 1/2 from set_param.
  ! ------------------------------------------------------------------
  subroutine read_par_file(par, path, ierr, badline)
    type(par_t), intent(inout) :: par
    character(len=*), intent(in) :: path
    integer, intent(out) :: ierr
    integer, intent(out) :: badline
    character(len=4096) :: line, rhs
    character(len=64) :: key
    real(wp) :: vals(36)
    integer :: unit, ios, ieq, ihash, nv, lineno, idx, i
    logical :: seen(NPAR)
    ierr = 0
    badline = 0
    seen = .false.
    open(newunit=unit, file=path, status='old', action='read', iostat=ios)
    if (ios /= 0) then
       ierr = 10
       return
    end if
    lineno = 0
    do
       read(unit, '(A)', iostat=ios) line
       if (ios /= 0) exit
       lineno = lineno + 1
       ihash = index(line, '#')
       if (ihash > 0) line = line(1:ihash-1)
       if (len_trim(line) == 0) cycle
       ieq = index(line, '=')
       if (ieq <= 1) then
          ierr = 11; badline = lineno
          exit
       end if
       key = adjustl(line(1:ieq-1))
       rhs = adjustl(line(ieq+1:))
       call count_tokens(rhs, nv)
       if (nv < 1 .or. nv > 36) then
          ierr = 11; badline = lineno
          exit
       end if
       read(rhs, *, iostat=ios) vals(1:nv)
       if (ios /= 0) then
          ierr = 11; badline = lineno
          exit
       end if
       call set_param(par, trim(key), vals(1:nv), nv, ierr)
       if (ierr /= 0) then
          badline = lineno
          exit
       end if
       idx = param_index(trim(key))
       if (idx > 0) seen(idx) = .true.
    end do
    close(unit)
    if (ierr /= 0) return
    ! A clean parse only establishes that the lines PRESENT were valid.
    ! par_t defaults every physical quantity to 0, and 0 is a legitimate
    ! fitted value for several of them (k23, S12A, c1i, Uref_ii), so a
    ! name the file never mentions is indistinguishable downstream from
    ! an explicit zero -- while an absent R or b1 turns every kernel
    ! expression that divides by it into Inf/NaN under a success return.
    ! Refuse the file instead.
    do i = 1, NPAR
       if (seen(i)) cycle
       if (any(OPTIONAL_NAMES == PARAM_NAMES(i))) cycle
       ierr = 12
       badline = i
       return
    end do
  end subroutine read_par_file

  subroutine count_tokens(str, n)
    character(len=*), intent(in) :: str
    integer, intent(out) :: n
    logical :: intok
    integer :: i
    character :: c
    n = 0
    intok = .false.
    do i = 1, len_trim(str)
       c = str(i:i)
       if (c == ' ' .or. c == char(9)) then
          intok = .false.
       else
          if (.not. intok) n = n + 1
          intok = .true.
       end if
    end do
  end subroutine count_tokens

end module ecpa_params
