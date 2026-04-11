import warnings; warnings.filterwarnings('ignore')

if __name__ == '__main__':
    from ecpa.parameters import make_params
    from ecpa.guess_table import load_cpa_guess_table
    from ecpa.solution_table import build_solution_table, print_table_summary

    params     = make_params()
    CPA_GROUPS, CPA_TEMPS = load_cpa_guess_table()

    grid_data = build_solution_table(
        params     = params,
        CPA_GROUPS = CPA_GROUPS,
        CPA_TEMPS  = CPA_TEMPS,
        save_path  = "results/solution_table.npz",
        n_workers  = 7,
        force_recompute = True,
    )
    print_table_summary(grid_data)
