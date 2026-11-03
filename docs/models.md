# The two models

The repository holds two equations of state. Both are electrolyte
Cubic-Plus-Association (eCPA) models: a Soave–Redlich–Kwong cubic term, the
Wertheim association term for water, and Debye–Hückel and Born terms for the
ions. They differ in how the salt is described, in the gases they cover, and
in the computations that come with them. [`papers.md`](papers.md) gives the
articles that define them (Paper I, Paper II, Paper III).

| | eCPA of Paper I | ms-eCPA of Papers II and III |
|---|---|---|
| Folder, Python package | [`code/`](../code/README.md), `ecpa` | [`multisalt/`](../multisalt/README.md), `ms_ecpa` |
| Installation | `pip install ecpa-flash`, or from the source tree | from the source tree (`PYTHONPATH=multisalt`) |
| Salts | NaCl | brines of Na⁺, K⁺, Ca²⁺, Mg²⁺, Cl⁻ and SO₄²⁻, single salts and mixtures |
| Salt description | NaCl parameters of Coelho et al. (2025) | ion-specific parameters, fitted in Paper II |
| Gases | CO₂ | CO₂, CH₄ (Paper II); N₂, H₂, C₂H₆, C₃H₈ and CH₄–CO₂ mixtures (Paper III) |
| Phases | two (aqueous, CO₂-rich) | two, and three below the critical temperature of a condensable gas |
| Stability and flash | tangent-plane-distance test; K-value flash with warm start from a precomputed solution table | tangent-plane-distance test; accelerated K-value flash; three-phase flash |
| Ranges in the papers | 273–698 K, 1–1500 bar, 0–6 mol/kg NaCl | calibration: CO₂–brine 283–473 K, 1.3–1200 bar, 0.09–6 mol/kg; CH₄–brine 298–598 K, 4–1379 bar, 0.25–6 mol/kg; flash: 283–623 K, 1–1000 bar |
| Experimental data | `EXP/` and `CO2/` of this repository | GasBrineBench v1.1.1, downloaded on first use |
| Compiled code | none | optional Fortran library with a C interface; same results as the Python code under the default (as-aqueous) salinity convention (`multisalt/fortran/`) |
| Simulator | prototype reservoir simulator (`code/co2brine_simulator.py`) | none |

## Which one to use

- **Brines other than NaCl, gases other than CO₂, gas mixtures, or three
  phases:** use ms-eCPA. The eCPA of Paper I has no parameters for these.
- **CO₂ in water or NaCl brine:** both models apply. The eCPA of Paper I can
  be installed from PyPI, covers higher temperatures and pressures (to 698 K
  and 1500 bar), and comes with the precomputed solution table for fast warm
  starts. ms-eCPA, with its ion-specific parameters, gives the same model for
  NaCl as for the other salts, which matters when results for several brines
  are compared.
- **Reproducing a published figure or table:** use the code of that paper and
  its executable notebook; see [`papers.md`](papers.md).

The two packages share no code. The NaCl parameters of the two models are
different, so they do not give identical results for CO₂–NaCl brine.
`multisalt/ms_ecpa/paper1_config.py` sets ms-eCPA to the parameters of
Paper I; with that setting the two packages agree to within 3e-5 relative at
the test states of `multisalt/tests/test_paper1_config.py` (2e-4 at two
states).
