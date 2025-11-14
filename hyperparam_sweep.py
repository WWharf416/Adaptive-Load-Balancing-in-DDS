"""
This script runs an expanded hyperparameter sweep for the PRO-LBR simulation
by testing different combinations of Q-learning parameters
(ALPHA, GAMMA, EPSILON) and reward function parameters
(IMBALANCE_PENALTY_FACTOR) defined in `config.py`.
"""

import itertools
import sys
import io
import time
import csv

import config
from main import run_simulation
from metrics import metrics

# --- Define the Parameter Grid to Test ---
alphas_to_test = [0.1, 0.3, 0.5]
gammas_to_test = [0.7, 0.9]
epsilons_to_test = [0.1, 0.25, 0.5]
# --- NEW: Add reward penalty factor to the sweep ---
penalties_to_test = [1.0, 1.5, 2.0]

# Create all possible combinations
param_grid = list(itertools.product(
    alphas_to_test,
    gammas_to_test,
    epsilons_to_test,
    penalties_to_test  # <-- Added new parameter
))

total_runs = len(param_grid)
results_list = []
output_filename = "sweep_results_enhanced.csv" # New output file

print("=" * 80)
print("STARTING ENHANCED HYPERPARAMETER SWEEP (State w/ Velocity)")
print(f"Total combinations to test: {total_runs}")
print(f"Results will be saved to: {output_filename}")
print("=" * 80)

original_stdout = sys.stdout
start_time = time.time()

# --- MODIFIED: Loop includes new parameter ---
for i, (alpha, gamma, epsilon, penalty) in enumerate(param_grid):
    
    run_num = i + 1
    print(f"\n--- Running {run_num}/{total_runs} (A={alpha}, G={gamma}, E={epsilon}, P={penalty}) ---")

    # 1. Set the config parameters for this run
    config.ALPHA = alpha
    config.GAMMA = gamma
    config.EPSILON = epsilon
    config.IMBALANCE_PENALTY_FACTOR = penalty # <-- Set new parameter
    
    # 2. Suppress print output
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
        pct_improvement = (1 - p_p99 / r_p99) * 100
    
    # 7. Store results
    results_list.append({
        'alpha': alpha,
        'gamma': gamma,
        'epsilon': epsilon,
        'penalty': penalty, # <-- Store new parameter
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
    results_list.sort(key=lambda x: x['improvement_pct'], reverse=True)

    print("\nHyperparameter Sweep Results (Sorted by Best Improvement)")
    print("-" * 90)
    # --- MODIFIED: Updated header ---
    print(f"{'Alpha':<7} | {'Gamma':<7} | {'Epsilon':<9} | {'Penalty':<9} | {'Reactive p99':<14} | {'Proactive p99':<15} | {'Improvement':<12}")
    print("-" * 90)

    for res in results_list:
        # --- MODIFIED: Updated row ---
        print(f"{res['alpha']:<7.2f} | {res['gamma']:<7.2f} | {res['epsilon']:<9.2f} | "
              f"{res['penalty']:<9.2f} | {res['r_p99']:<14.2f} | {res['p_p99']:<15.2f} | "
              f"{res['improvement_pct']:<+11.2f} %")
    print("-" * 90)
    
    best = results_list[0]
    print(f"\nBest Run: A={best['alpha']}, G={best['gamma']}, E={best['epsilon']}, P={best['penalty']} "
          f"with {best['improvement_pct']:+.2f}% improvement.")
          
    # --- Save Results to CSV ---
    print(f"\nSaving results to {output_filename}...")
    
    try:
        # --- MODIFIED: Updated fieldnames ---
        fieldnames = ['alpha', 'gamma', 'epsilon', 'penalty', 'r_p99', 'p_p99', 'improvement_pct']
        
        with open(output_filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for res in results_list:
                formatted_res = {
                    'alpha': f"{res['alpha']:.2f}",
                    'gamma': f"{res['gamma']:.2f}",
                    'epsilon': f"{res['epsilon']:.2f}",
                    'penalty': f"{res['penalty']:.2f}", # <-- Format new field
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