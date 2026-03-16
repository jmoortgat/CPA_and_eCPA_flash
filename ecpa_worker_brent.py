"""
ecpa_worker_brent.py — parallel eCPA flash worker using the Brent algorithm.

Must live at the top level so ProcessPoolExecutor can pickle tasks.

Task tuple:
    (T, P_bar, z_co2, ms, parquet_path, params)

Usage
-----
from concurrent.futures import ProcessPoolExecutor, as_completed
import ecpa_worker_brent

tasks = [
    (T, P, z, ms, "CPA_ELV_all.parquet", params)
    for T, P, z, ms in grid
]
with ProcessPoolExecutor(max_workers=N) as executor:
    futures = {executor.submit(ecpa_worker_brent.compute_one_point, task): task
               for task in tasks}
    for future in as_completed(futures):
        result = future.result()
        # result keys: T, P_bar, z_co2, ms, flash_algo,
        #              converged, beta, ms_aq, Z_aq, Z_c, error, error_type

NOTE: To remove the Brent algorithm entirely:
    1. Delete this file (ecpa_worker_brent.py)
    2. Remove flash_co2_h2o_salt_1d from ecpa/flash.py
    3. Remove the "brent" entry from FLASH_ALGORITHMS in ecpa/flash.py
"""

from ecpa_worker import _compute


def compute_one_point(task: tuple) -> dict:
    """Run one eCPA flash point with the Brent algorithm."""
    return _compute(task, flash_algo="brent")
