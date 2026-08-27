# Contributing

Thank you for your interest in improving **eCPA Flash**. Contributions of all
kinds are welcome: bug reports, questions, documentation fixes, new
validation data, and code.

## Reporting problems and asking questions

Please open a [GitHub issue](https://github.com/jmoortgat/CPA_and_eCPA_flash/issues)
with:

- what you ran (the exact command or a minimal script),
- what you expected, and what happened instead,
- your platform and Python version (`python --version`), and
- the full traceback if there was an error.

Questions about usage or about the thermodynamic model are welcome in the
issue tracker as well.

## Setting up a development environment

```bash
git clone https://github.com/jmoortgat/CPA_and_eCPA_flash.git
cd CPA_and_eCPA_flash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
pytest
```

Alternatively use the conda environment in `code/environment.yml`.

## Making changes

1. Fork the repository and create a feature branch.
2. Make your changes. Please keep the numerical core free of new required
   dependencies; heavyweight extras (e.g., the optional neural-network
   warm-start, which needs `torch`) belong behind optional imports.
3. Run the test suite (`pytest`) and confirm it passes.
4. If your change affects computed results, say so explicitly in the pull
   request and include before/after numbers for at least one of the
   validation conditions in `tests/`.
5. Open a pull request with a clear description of the problem and the fix.

## Scientific scope

The package implements the CPA and eCPA equations of state for the
CO2 + H2O + NaCl system, with the parameterization of Coelho, Franco &
Firoozabadi (2025) and the stability/flash algorithms described in the
companion journal article (see `README.md`). Extensions (additional salts,
gases, or algorithms) are welcome as long as the published validation
results remain reproducible.

## Code of conduct

All participation in this project is governed by the
[Code of Conduct](CODE_OF_CONDUCT.md).
