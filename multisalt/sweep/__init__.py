"""Multi-salt eCPA stability+flash sweep harness.

Entry points:

* :func:`sweep_config.iter_grid` — generate (T, P, z, mi) sweep points.
* :func:`sweep_core.run_one` — evaluate one point, return a flat dict.
* :mod:`sweep_runner` — parallelise a chunk across CPUs, write parquet.

See ``README.md`` in this directory for usage and SLURM job submission.
"""
from . import sweep_config, sweep_core   # noqa: F401
