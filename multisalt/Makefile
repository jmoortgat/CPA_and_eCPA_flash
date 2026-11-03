# ms-eCPA (Papers II and III) -- regeneration Makefile. Run from multisalt/.
#
# `make figures`        Regenerate every figure and stats sidecar from the
#                       cached model results in results_cache/.
# `make figures-force`  Recompute every cached model result first, then
#                       regenerate the figures. Cold: several minutes to hours,
#                       depending on the backend (see README.md).
# `make tables`         Regenerate the LaTeX tables and number macros in
#                       paper_tables/ (or $ECPA_PAPER_DIR).
# `make clean`          Remove the generated tables and number macros. Does
#                       NOT remove the cached model results.

CODE_DIR := $(shell pwd)
PAPER_DIR := $(or $(ECPA_PAPER_DIR),$(CODE_DIR)/paper_tables)
PY := PYTHONPATH=$(CODE_DIR):$(CODE_DIR)/scripts python3

.PHONY: figures figures-force tables clean help

help:
	@echo "ms-eCPA regeneration targets"
	@echo ""
	@echo "  make figures        Regenerate figures + stats from cached model results"
	@echo "  make figures-force  Recompute the cached model results, then regenerate"
	@echo "  make tables         Regenerate LaTeX tables + number macros"
	@echo "  make clean          Remove the generated tables + number macros"

figures:
	$(PY) scripts/make_all_paper_figures.py

figures-force:
	$(PY) scripts/make_all_paper_figures.py --force

tables:
	$(PY) scripts/build_paper_tables.py
	$(PY) scripts/build_paper_stats_tex.py

clean:
	rm -rf $(PAPER_DIR)
	@echo "Removed $(PAPER_DIR). Cached model results are untouched."
