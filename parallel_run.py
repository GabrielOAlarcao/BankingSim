# run_parallel_joblib.py

import os
import random
import numpy as np
from joblib import Parallel, delayed


# ============================================================
# GLOBAL RESET UTILITIES
# ============================================================


# ============================================================
# SINGLE SIMULATION WORKER (ONE PROCESS)
# ============================================================

def run_single_simulation(
    run_id,
    firm_seed,
    model_seed,
    capital_requirement,
    max_steps,
    output_dir
):
    """
    Runs exactly ONE model instance in ONE process.
    Safe for joblib multiprocessing on Windows.
    """

    # --- imports INSIDE function (Windows spawn-safe) ---
    from model import MyModel, ExogenousFactors
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)

    # --- hard reset of all global state ---
    #reset_global_state(model_seed)

    # --- set exogenous parameters ---
    #ExogenousFactors.isCapitalRequirementActive = False
    #ExogenousFactors.DefaultEWADampingFactor = 0.9
    #ExogenousFactors.minimumCapitalAdequacyRatio = 0.08
    #ExogenousFactors.beta0FirmTypeModel = -1.3
    #ExogenousFactors.beta2FirmTypeModel = -1.1

    print(f"[PID {os.getpid()}] Run {run_id} START", flush=True)

    # --- run model ---
    model = MyModel(firm_seed = firm_seed, model_seed = model_seed,
                    capital_requirement = capital_requirement)
    
    #np.random.seed()
    
    model.run_model(max_steps)

    # --- save outputs (same format as before) ---
    model.datacollector.get_model_vars_dataframe().to_csv(
        f"{output_dir}/output_model_{run_id}.csv",
        sep=";",
        decimal=".",
        index=False
    )

    model.datacollector.get_agent_vars_dataframe().to_csv(
        f"{output_dir}/output_agents_{run_id}.csv",
        sep=";",
        decimal=".",
        index=False
    )

    print(f"[PID {os.getpid()}] Run {run_id} END", flush=True)

    return run_id


# ============================================================
# PARALLEL LAUNCHER
# ============================================================

if __name__ == "__main__":

    # --- import ONLY here ---
    from model import ExogenousFactors

    OUTPUT_DIR = "prudential_10"
    MAX_STEPS = 10000
    N_SIMULATIONS = 20
    N_JOBS = 5        # change to 1 for debugging

    base_firm_seed = ExogenousFactors.firmSizeSeed
    base_model_seed = ExogenousFactors.modelSeed
    capital_requirement = True

    print(
        f"Starting {N_SIMULATIONS} simulations "
        f"for {MAX_STEPS} steps\n"
        f"using {N_JOBS} parallel workers\n"
    )

    results = Parallel(
        n_jobs=N_JOBS,
        backend="loky",      # multiprocessing (NOT threading)
        verbose=10
    )(
        delayed(run_single_simulation)(
            run_id=i,
            firm_seed=base_firm_seed + i * 4,
            model_seed=base_model_seed + i,
            capital_requirement = capital_requirement,
            max_steps=MAX_STEPS,
            output_dir=OUTPUT_DIR
        )
        for i in range(N_SIMULATIONS)
    )

    print("\nAll simulations completed:", results)
