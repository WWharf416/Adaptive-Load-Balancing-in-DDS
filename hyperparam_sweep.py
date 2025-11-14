"""
This script runs a hyperparameter sweep for the PRO-LBR simulation
by testing different combinations of Q-learning parameters
(ALPHA, GAMMA, EPSILON) defined in `config.py`.

It imports the `run_simulation` function from `main.py` and the
`metrics` collector from `metrics.py` to run the simulations
and capture the results.

At the end, it prints a formatted table of all runs, sorted by
the best percentage improvement, AND saves the results to
a CSV file named 'sweep_results.csv'.
"""

import itertools
import sys
import io
import time
import csv  # Import the CSV module

# Import the modules we need to interact with
import config
from main import run_simulation
from metrics import metrics

# --- Define the Parameter Grid to Test ---
# We've expanded the epsilon list to create more combinations
alphas_to_test = [0.1, 0.3, 0.5]
gammas_to_test = [0.7, 0.9]
epsilons_to_test = [0.1, 0.25, 0.5]  # Was [0.1, 0.25]

# Create all possible combinations
param_grid = list(itertools.product(
    alphas_to_test,
    gammas_to_test,
    epsilons_to_test
))

total_runs = len(param_grid)
results_list = []
output_filename = "sweep_results.csv"

print("=" * 80)
print("STARTING HYPERPARAMETER SWEEP")
print(f"Total combinations to test: {total_runs}")
print(f"Results will be saved to: {output_filename}")
print("=" * 80)

# Store the original stdout to suppress simulation output
original_stdout = sys.stdout
start_time = time.time()

for i, (alpha, gamma, epsilon) in enumerate(param_grid):
    
    run_num = i + 1
    print(f"\n--- Running {run_num}/{total_runs} (Alpha={alpha}, Gamma={gamma}, Epsilon={epsilon}) ---")

    # 1. Set the config parameters for this run
    config.ALPHA = alpha
    config.GAMMA = gamma
    config.EPSILON = epsilon
    
    # 2. Suppress print output from the simulation runs
    sys.stdout = io.StringIO()
    
    # 3. Run Reactive Simulation
    try:
        run_simulation('reactive')
        r_p99 = metrics.final_report.get('reactive', {}).get('steady_p99', 0)
    except Exception as e:
        print(f"Error in reactive run: {e}")
        r_p99 = 0
    
    # 4. Run Proactive Simulation
    try:
        run_simulation('proactive')
        p_p99 = metrics.final_report.get('proactive', {}).get('steady_p99', 0)
    except Exception as e:
        print(f"Error in proactive run: {e}")
        p_p99 = 0

    # 5. Restore stdout
    sys.stdout = original_stdout
    
    # 6. Calculate improvement
    pct_improvement = 0.0
    if r_p99 > 0 and p_p99 > 0:
        # (1 - new/old) * 100
        # Positive is improvement (p_p99 is lower)
        # Negative is degradation (p_p99 is higher)
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

# --- Print Final Results Table (to console) ---

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
              f"{res['improvement_pct']:<+11.2f} %")
    print("-" * 80)
    
    best = results_list[0]
    print(f"\nBest Run: A={best['alpha']}, G={best['gamma']}, E={best['epsilon']} "
          f"with {best['improvement_pct']:+.2f}% improvement.")
          
    # --- Save Results to CSV ---
    
    print(f"\nSaving results to {output_filename}...")
    
    try:
        # We use the sorted list, so the CSV will also be sorted
        fieldnames = ['alpha', 'gamma', 'epsilon', 'r_p99', 'p_p99', 'improvement_pct']
        
        with open(output_filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for res in results_list:
                # Format numbers for consistent CSV output
                formatted_res = {
                    'alpha': f"{res['alpha']:.2f}",
                    'gamma': f"{res['gamma']:.2f}",
                    'epsilon': f"{res['epsilon']:.2f}",
                    'r_p99': f"{res['r_p99']:.2f}",
                    'p_p99': f"{res['p_p99']:.2f}",
                    'improvement_pct': f"{res['improvement_pct']:.2f}"
                }
                writer.writerow(formatted_res)
        
        print(f"Successfully saved {len(results_list)} results to {output_filename}.")
    
    except IOError as e:
        print(f"Error: Could not write to file {output_filename}. {e}")
    except Exception as e:
        print(f"An unexpected error occurred while writing CSV: {e}")
else:
    print("No results to save.")