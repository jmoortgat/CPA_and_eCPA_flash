"""Transition alias: ``ecpa_ms`` IS ``ms_ecpa``, not a copy of it.

The package was renamed on 2026-09-18. Code that still imports the old
name -- older research scripts in particular -- must reach the SAME module
objects the kernel uses, not a second import of the same source.

Why this matters more than tidiness
-----------------------------------
The pilots configure a gas by REBINDING module attributes: c2_pilot does

    import ecpa_ms.parameters as prm
    ...
    setattr(prm, "Tc2", <C2H6 value>)      # and Pc2, w2, b2, a02, ...

to swap ethane into the CH4 slot. If ``ecpa_ms.parameters`` and
``ms_ecpa.parameters`` were two separate module objects, that rebind would
land in one while the flash read the other: the slot swap would silently not
happen, ethane would be computed with methane's parameters, and the result
would be a plausible wrong number rather than an error. That is exactly what
happened to the C2H6 and C3H8 anchors before this shim existed.

So this module does not re-implement or re-export anything. It installs the
real modules under the old names in ``sys.modules``, so the two names are
one object and a rebind through either is visible to both.
"""
from __future__ import annotations

import importlib
import sys

import ms_ecpa as _pkg

#: Submodules the old name must reach. A missing one is skipped rather than
#: raising, so this keeps working as the kernel grows or loses modules.
_SUBMODULES = (
    "parameters", "lnphi", "eos", "flash", "flash3", "stability",
    "guess", "newton", "psat", "rachford_rice", "extensions", "water",
    "paper1_config", "backend",
)

for _name in _SUBMODULES:
    try:
        sys.modules[f"{__name__}.{_name}"] = importlib.import_module(
            f"ms_ecpa.{_name}")
    except ModuleNotFoundError:
        pass

# The package itself resolves to the real one, so `import ecpa_ms` and
# `import ms_ecpa` are the same object.
sys.modules[__name__] = _pkg
