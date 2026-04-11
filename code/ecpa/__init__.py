"""
ecpa — electrolyte CPA equation of state for CO₂–H₂O–NaCl

Package layout
--------------
constants.py    Physical and EoS constants (scalars only)
parameters.py   make_params() — builds the params dict passed to ELV
guess_table.py  CPA initial-guess table I/O and interpolation
exp_data.py     Experimental VLE data loading and lookup
elv.py          ELV() residual system + complex-step Jacobian
flash.py        Continuation cache, ELV solver, Brent + SSI flash
stability.py         Phase-stability analysis: ecpa_lnphi_aq/c, ecpa_stability, ecpa_stability_flash
flash_simplified.py  Simplified flash assuming y_H2O=0 (pure-CO2 phase): flash_co2_h2o_simplified
warmstart.py         Warm-start providers: ScanTableWarmStart, NNWarmStart, WarmStartGuess
scan.py         run_flash_scan() — grid scan over (T, P, z, ms)
envelope.py     find_envelope_from_scan(), build_cpa2_envelope()
plotting.py     All figure-generating functions
utils.py        print_flash_report, pct_diff, test_tieline_invariance
"""
