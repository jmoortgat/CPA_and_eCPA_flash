# Makefile for the eCPA multi-salt Fortran port.
#
# Default compiler: gfortran (Homebrew gfortran 14 on macOS).
# Intel (oneAPI 2024.2, e.g. on a Linux HPC cluster):
#   make FC=ifx                       portable Intel build   -> build/
#   make FC=ifx ARCH=sapphire         Sapphire Rapids build  -> build-sapphire/
#
# WHICH INTEL COMPILER: ship ifx, not ifort.  ifort 2021.13 still works
# but emits its own deprecation remark (#10448, "discontinued late
# 2024"), and it is measurably WORSE for us: with -fp-model strict it
# runs 11-13 % slower than gfortran on Sapphire Rapids, whereas ifx
# with -fp-model strict costs nothing (ifx's `precise` already turns
# FMA contraction off, so `strict` only adds exception semantics).
#
# WHY -fp-model strict (measured on an Intel Xeon Platinum 8468 node, all 13
# acceptance tests x 3 parameter files, 15 compiler/flag variants):
#   * -fp-model fast (the Intel DEFAULT) does change the arithmetic --
#     it perturbs every reported deviation -- though it still passed
#     the gate.  Not shipped: it assumes finite math, and this port
#     depends on NaN/Inf propagation mirroring NumPy.
#   * -fp-model precise passes and matches gfortran.
#   * -fp-model strict passes, and on ifx is BIT-IDENTICAL to precise
#     AND to the -xSAPPHIRERAPIDS build.  That last property is the
#     reason to prefer it: the portable and CPU-targeted libraries then
#     return the same numbers, so which node built the .so cannot
#     change a result.  On ifort, precise does NOT have that property
#     (FMA contraction survives and the -x build diverges).
# Never add -ffast-math / -fp-model fast.
#
# ARCH=sapphire adds -xSAPPHIRERAPIDS (Intel) / -march=sapphirerapids
# (gfortran): the resulting library runs ONLY on Sapphire Rapids nodes
# (on a SLURM cluster, request them with a node constraint), and gives a
# consistent 5-9 % speed-up.  The
# plain build stays portable.  Variants land in separate build dirs;
# point the ctypes wrapper at a variant with the ECPA_LIB env var
# (tools/ecpa_ctypes.py).
#
# NO BLAS/LAPACK: the library calls none -- the 3x3 / 2x2 inner solves
# are a hand-rolled LU with partial pivoting (ecpa_newton.f90) and the
# only matrix expression is a 6x6 MATMUL the compiler inlines.  Verified
# with `nm -D --undefined-only`: the complete undefined-symbol list is
# libm + libc + compiler runtime, with no dgemm_/dgesv_/mkl_* of any
# kind.  Do NOT link -qmkl; it would add a dependency that buys
# nothing.  (If that ever changes, use -qmkl=sequential: this library
# is called from multiprocessing workers, where a threaded BLAS would
# fight the pool.)
#
# Targets:
#   make            -> build the shared library libecpa.(dylib|so)
#   make bench      -> run the Fortran-vs-Python timing comparison
#   make clean

# make's built-in default FC is f77, so it must be replaced -- but not with a
# hard-coded path, which would exist on one kind of machine only.
#
# Search order, first hit wins:
#   1. FC= on the command line, or the FC environment variable
#   2. ifort   -- the Intel compiler
#   3. ifx     -- its LLVM successor
#   4. gfortran on PATH (any platform, including Homebrew's)
#   5. the Homebrew path, as a last resort for a macOS shell whose PATH
#      has not been set up
ifeq ($(origin FC),default)
  FC := $(firstword $(shell command -v ifort 2>/dev/null) \
                    $(shell command -v ifx 2>/dev/null) \
                    $(shell command -v gfortran 2>/dev/null) \
                    /opt/homebrew/bin/gfortran)
endif
ifeq ($(wildcard $(FC)),)
  ifeq ($(shell command -v $(FC) 2>/dev/null),)
    $(error No Fortran compiler found. Set FC=, e.g. \
      FC=ifx (Intel oneAPI) or FC=$$(brew --prefix)/bin/gfortran (macOS). \
      On an HPC cluster, loading a compiler module first is usually enough \
      for the search to find one.)
  endif
endif
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  SOEXT := dylib
  SOFLAGS := -dynamiclib
else
  SOEXT := so
  SOFLAGS := -shared
endif

# Compiler family: ifort and ifx share Intel conventions (-module, -stand).
FCNAME := $(notdir $(FC))
ifneq (,$(filter ifort ifx,$(FCNAME)))
  INTEL := 1
endif

# Optional CPU-targeted variant (see header).
ARCHFLAG :=
ifeq ($(ARCH),sapphire)
  ifdef INTEL
    ARCHFLAG := -xSAPPHIRERAPIDS
  else
    ARCHFLAG := -march=sapphirerapids
  endif
endif

# F2008 standard, no vendor extensions.  IEEE semantics: keep value
# safety (no -ffast-math ever -- the port relies on NaN/Inf propagation
# mirroring NumPy).  See the header for the measured justification of
# -fp-model strict.
ifdef INTEL
  FFLAGS ?= -O3 -stand f08 -fpp -fPIC -fp-model strict
  MODFLAG := -module
else
  FFLAGS ?= -O2 -std=f2008 -fPIC -Wall -Wno-unused-dummy-argument
  MODFLAG := -J
endif

# gfortran note (NOT applied by default, so the verified macOS
# baseline is not perturbed): gfortran defaults to -ffp-contract=fast,
# so its portable and -march=sapphirerapids builds differ in the last
# ulp (both pass the gate).  Add -ffp-contract=off to FFLAGS for the
# same build-target independence the Intel -fp-model strict build has.

SRCDIR := src
OBJDIR ?= build$(if $(ARCH),-$(ARCH))
BINDIR := bin

SRCS := ecpa_kinds.f90 ecpa_params.f90 ecpa_water.f90 ecpa_extensions.f90 \
        ecpa_newton.f90 ecpa_lnphi.f90 ecpa_brent.f90 ecpa_flash.f90 \
        ecpa_stability.f90 ecpa_colligative.f90 ecpa_capi.f90
OBJS := $(SRCS:%.f90=$(OBJDIR)/%.o)

# Out-of-tree extension sources (see src/ecpa_extensions.f90), compiled
# after the core into the same library under another name, e.g.
#   make EXT_SRCS="/path/a.f90 /path/b.f90" LIBNAME=libecpa_ext OBJDIR=build-ext
# Listed in dependency order. Empty by default.
EXT_SRCS ?=
LIBNAME ?= libecpa
EXT_OBJS := $(foreach f,$(EXT_SRCS),$(OBJDIR)/ext_$(basename $(notdir $(f))).o)

LIB := $(OBJDIR)/$(LIBNAME).$(SOEXT)

.PHONY: all clean bench

all: $(LIB)

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(BINDIR):
	mkdir -p $(BINDIR)

$(OBJDIR)/%.o: $(SRCDIR)/%.f90 | $(OBJDIR)
	$(FC) $(FFLAGS) $(ARCHFLAG) $(MODFLAG) $(OBJDIR) -c $< -o $@

define EXT_RULE
$(OBJDIR)/ext_$(basename $(notdir $(1))).o: $(1) $(OBJS) | $(OBJDIR)
	$(FC) $(FFLAGS) $(ARCHFLAG) $(MODFLAG) $(OBJDIR) -c $(1) -o $$@
endef
$(foreach f,$(EXT_SRCS),$(eval $(call EXT_RULE,$(f))))

$(LIB): $(OBJS) $(EXT_OBJS)
	$(FC) $(SOFLAGS) -o $@ $(OBJS) $(EXT_OBJS)

# module dependencies
$(OBJDIR)/ecpa_params.o: $(OBJDIR)/ecpa_kinds.o
$(OBJDIR)/ecpa_water.o: $(OBJDIR)/ecpa_kinds.o
$(OBJDIR)/ecpa_extensions.o: $(OBJDIR)/ecpa_params.o
$(OBJDIR)/ecpa_newton.o: $(OBJDIR)/ecpa_params.o
$(OBJDIR)/ecpa_lnphi.o: $(OBJDIR)/ecpa_newton.o $(OBJDIR)/ecpa_extensions.o
$(OBJDIR)/ecpa_brent.o: $(OBJDIR)/ecpa_kinds.o
$(OBJDIR)/ecpa_flash.o: $(OBJDIR)/ecpa_lnphi.o $(OBJDIR)/ecpa_brent.o
$(OBJDIR)/ecpa_stability.o: $(OBJDIR)/ecpa_flash.o
$(OBJDIR)/ecpa_colligative.o: $(OBJDIR)/ecpa_lnphi.o $(OBJDIR)/ecpa_brent.o \
                              $(OBJDIR)/ecpa_water.o $(OBJDIR)/ecpa_extensions.o
$(OBJDIR)/ecpa_capi.o: $(OBJDIR)/ecpa_stability.o $(OBJDIR)/ecpa_colligative.o

bench: $(LIB)
	python3 tools/bench.py

clean:
	rm -rf build build-sapphire build-ext $(BINDIR)
