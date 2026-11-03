"""ms_ecpa: ion-specific electrolyte CPA equation of state for
H2O-CH4-CO2-ions phase equilibria.

Refactored from the original ``eCPA_ELV.py`` (Mourao Coelho & Firoozabadi,
2025) into an importable, non-interactive package. Physics is preserved
verbatim; see :mod:`ms_ecpa.eos` and :mod:`ms_ecpa.parameters`.

Example
-------
>>> from ms_ecpa import solve_elv
>>> res = solve_elv(T=323, P=500, YC=0.8, mi=[1, 2, 0.6, 0.2, 0.2, 0.2])
>>> round(res.rho_w, 2)
1104.31
"""
from .eos import ELVResult, gas_fugacities, residual_elv, solve_elv
from .guess import SinglePhaseError, build_initial_guess
from .lnphi import lnphi_aq, lnphi_gas
from .flash import (
    FlashKVResult,
    aqueous_closure,
    compute_aqueous_x1w_from_KP,
    flash_kv,
    rachford_rice,
    wilson_k,
)
from .stability import (
    StabilityFlashResult,
    StabilityResult,
    TPDResult,
    make_k_init_from_trial,
    michelsen_tpd,
    stability_flash,
    tpd_ssi,
)
from .flash3 import (
    Flash3Result,
    StabilityFlash3Result,
    flash3_kv,
    rachford_rice_3p,
    stability_flash3,
)
from .psat import PsatBracketError, solve_psat_isofugacity

# ---------------------------------------------------------------------------
# Optional compiled backend.
#
# MS_ECPA_BACKEND=fortran routes stability_flash to the Fortran port (~190x,
# agreeing to 1e-13) for EVERY caller -- figure scripts, sweep, fitting
# driver -- without any of them changing. as-aqueous only; a chain call
# falls back to this Python kernel rather than returning numbers that differ
# silently from it. See ms_ecpa/backend.py.
from .backend import dispatch as _dispatch          # noqa: E402
stability_flash = _dispatch(stability_flash)        # noqa: F811

__all__ = [
    "solve_elv",
    "ELVResult",
    "residual_elv",
    "gas_fugacities",
    "build_initial_guess",
    "SinglePhaseError",
    # Per-phase lnφ kernels (Phase A)
    "lnphi_aq",
    "lnphi_gas",
    # K-value flash (Phase C)
    "FlashKVResult",
    "aqueous_closure",
    "compute_aqueous_x1w_from_KP",
    "flash_kv",
    "rachford_rice",
    "wilson_k",
    # Michelsen TPD stability + combined driver (Step 2)
    "TPDResult",
    "StabilityResult",
    "StabilityFlashResult",
    "tpd_ssi",
    "michelsen_tpd",
    "make_k_init_from_trial",
    "stability_flash",
    # Three-phase (V + Laq + L_CO2) stability + flash (additive extension)
    "Flash3Result",
    "StabilityFlash3Result",
    "flash3_kv",
    "rachford_rice_3p",
    "stability_flash3",
    # Saturation pressure by water isofugacity, on a verified bracket
    "PsatBracketError",
    "solve_psat_isofugacity",
]
