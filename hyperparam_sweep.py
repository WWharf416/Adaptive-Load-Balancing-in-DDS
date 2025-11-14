"""
This script runs a hyperparameter sweep for the PRO-LBR simulation
by testing different combinations of Q-learning parameters
(ALPHA, GAMMA, EPSILON) defined in `config.py`.

It imports the `run_simulation` function from `main.py` and the
`metrics` collector from `metrics.py` to run the simulations
and capture the results.

At the end, it prints a formatted table of all runs, sorted by
the best percentage improvement.
"""

import itertools
import sys
import io
import time

# Import the modules we need to interact with
import config
from main import run_simulation
from metrics import metrics

# --- Define the Parameter Grid to Test ---
# Add or remove values here to broaden or narrow the search.
# Warning: Each combination adds two full simulation runs.
alphas_to_test = [0.1, 0.3, 0.5]
gammas_to_test = [0.7, 0.9]
epsilons_to_test = [0.1, 0.25]

# Create all possible combinations
param_grid = list(itertools.product(
    alphas_to_test,
    gammas_to_test,
    epsilons_to_test
))

total_runs = len(param_grid)
results_list = []

print("=" * 80)
print("STARTING HYPERPARAMETER SWEEP")
print(f"Total combinations to test: {total_runs}")
print("=" * 80)

# Store the original stdout to suppress simulation output
original_stdout = sys.stdout
start_time = time.time()

for i, (alpha, gamma, epsilon) in enumerate(param_grid):
    run_num = i + 1
    print(f"\n--- Running {run_num}/{total_runs} (Alpha={alpha}, Gamma={gamma}, Epsilon={epsilon}) ---")

    # 1. Set the hyperparameters in the config module
    #    This modifies the imported module's values in memory for this run
    config.ALPHA = alpha
    config.GAMMA = gamma
    config.EPSILON = epsilon
    
    # 2. Suppress the verbose stdout from run_simulation
    #    to keep the sweep output clean.
    sys.stdout = io.StringIO()

    try:
        # 3. Run both simulations. The metrics object is a singleton
        #    and will be updated by each run.
        run_simulation('reactive')
        run_simulation('proactive')

    except Exception as e:
        # Restore stdout to print any errors
        sys.stdout = original_stdout
        print(f"ERROR during run {run_num}: {e}")
        # Skip this combination
        continue
    
    finally:
        # 4. Always restore stdout
        sys.stdout = original_stdout

    # 5. Get results from the metrics collector's final report
    r_results = metrics.final_report.get('reactive', {})
    p_results = metrics.final_report.get('proactive', {})

    r_p99 = r_results.get('steady_p99', 0.0)
    p_p99 = p_results.get('steady_p99', 0.0)

    # 6. Calculate improvement
    pct_improvement = 0.0
    if r_p99 > 0 and p_p99 > 0:
        # (1 - new/old) * 100
        # A positive value is an improvement (p_p99 is lower)
        # A negative value is a degradation (p_p99 is higher)
        pct_improvement = (1 - p_p99 / r_p99) * 100
    
    # 7. Store results
    results_list.append({
        'alpha': alpha,
        'gamma': gamma,
        'epsilon': epsilon,
        'r_p99': r_p99,
        'p_p99': p_p99,
        'improvement_pct': pct_improvement
    })
    
    print(f"Result: Reactive p99={r_p99:.2f}ms, Proactive p99={p_p99:.2f}ms. Improvement: {pct_improvement:+.2f}%")

end_time = time.time()
print("\n" + "=" * 80)
print(f"SWEEP COMPLETED in {end_time - start_time:.2f} seconds")
print("=" * 80)

# --- Print Final Results Table ---

if results_list:
    # Sort by the best improvement (highest percentage)
    results_list.sort(key=lambda x: x['improvement_pct'], reverse=True)

    print("\nHyperparameter Sweep Results (Sorted by Best Improvement)")
    print("-" * 80)
    print(f"{'Alpha':<7} | {'Gamma':<7} | {'Epsilon':<9} | {'Reactive p99':<14} | {'Proactive p99':<15} | {'Improvement':<12}")
    print("-" * 80)

    for res in results_list:
        print(f"{res['alpha']:<7.2f} | {res['gamma']:<7.2f} | {res['epsilon']:<9.2f} | "
              f"{res['r_p99']:<14.2f} | {res['p_p99']:<15.2f} | "
              f"{res['improvement_pct']:<+12.2f}%")
    
    print("-" * 80)
    best = results_list[0]
    print(f"\nBest Run: A={best['alpha']}, G={best['gamma']}, E={best['epsilon']} "
          f"with {best['improvement_pct']:+.2f}% improvement.")
else:
    print("No results to display. The sweep may have encountered errors.")