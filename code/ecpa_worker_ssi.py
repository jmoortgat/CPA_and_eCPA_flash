"""
ecpa_worker_ssi.py — parallel eCPA flash worker using the SSI algorithm.

Must live at the top level so ProcessPoolExecutor can pickle tasks.

Task tuple:
    (T, P_bar, z_co2, ms, parquet_path, params)

Usage
-----
from concurrent.futures import ProcessPoolExecutor, as_completed
import ecpa_worker_ssi

tasks = [
    (T, P, z, ms, "CPA_ELV_all.parquet", params)
    for T, P, z, ms in grid
]
with ProcessPoolExecutor(max_workers=N) as executor:
    futures = {executor.submit(ecpa_worker_ssi.compute_one_point, task): task
               for task in tasks}
    for future in as_completed(futures):
        result = future.result()
        # result keys: T, P_bar, z_co2, ms, flash_algo,
        #              converged, beta, ms_aq, Z_aq, Z_c, error, error_type
"""

from ecpa_worker import _compute


def compute_one_point(task: tuple) -> dict:
    """Run one eCPA flash point with the SSI algorithm."""
    return _compute(task, flash_algo="ssi")
