# Experimental validation database

Raw literature data used to validate the CPA and eCPA flash calculations.
These files are the **source of record**; the code reads the derived parquet
caches and only re-parses this directory if a cache is missing.

## Layout

```
EXP/
├── CO2-WATER/T<temp>K/EXP<n>_T<temp>K.txt   # CO2 + H2O VLE, one file per literature source
└── CO2-NaCl/T<temp>K/...                    # CO2 + NaCl brine, same convention
    └── co2_nacl_exp.parquet                 # parsed cache (used by the code)
```

Each text file holds the digitized measurements from a single publication at
a single temperature. The first line is a comment naming the source (e.g.
`#TODHEIDE (1963)`), followed by a header and whitespace-separated columns
(pressure in bar; compositions as mole fractions `xc_W`, `yw_C`, molality
`mc`, or density `rho_W`, depending on what the source reports).

## Parsed caches used by the code

- `CO2/CO2_WATER_exp.parquet` — built by `ecpa.exp_data.load_exp_data()`
  (631 points, 273–623 K).
- `EXP/CO2-NaCl/co2_nacl_exp.parquet` — built by
  `ecpa.validate_nacl.load_co2nacl_exp()` (708 points, 278–723 K).

Deleting a parquet cache forces the corresponding loader to rebuild it from
these raw files (`load_co2nacl_exp` also accepts `force_reparse=True`).
