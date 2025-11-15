"""
This script runs two separate hyperparameter sweeps for the Q-Table and DQN
agents, benchmarking both against the 'reactive' baseline.

***
V2 UPDATE: Now includes DQN_EPSILON_DECAY in the DQN sweep,
as this is critical for short simulations.
***
"""

import itertools
import sys
import time
import math # Need for epsilon decay calculation

# Import from your uploaded files
import config
import simulation  # Contains the run_simulation function
from metrics import metrics # The global metrics collector

print("=" * 80)
print("STARTING DUAL HYPERPARAMETER SWEEP (Q-TABLE & DQN vs. REACTIVE)")
print("=" * 80)

# --- 1. Run Baseline (Reactive) ---
print("\nRunning baseline 'reactive' simulation...")
try:
    simulation.run_simulation('reactive')
    reactive_metrics = metrics.final_report.get('reactive', {})
    reactive_p99 = reactive_metrics.get('steady_p99', 0)
    
    if reactive_p99 == 0:
        print("ERROR: Reactive simulation failed or returned p99 of 0. Aborting.")
        sys.exit(1)
        
    print(f"Baseline Reactive p99: {reactive_p99:.2f} ms")

except Exception as e:
    print(f"ERROR during reactive simulation: {e}")
    sys.exit(1)


# --- 2. Q-Table Hyperparameter Sweep ---
print("\n" + "=" * 80)
print("STARTING: Q-Table Sweep")
print("=" * 80)

# --- Customize your Q-Table parameters here ---
q_alphas_to_test = [0.1, 0.3, 0.5]
q_gammas_to_test = [0.7, 0.9]
q_epsilons_to_test = [0.1, 0.25, 0.4]
# ---

q_param_grid = list(itertools.product(
    q_alphas_to_test,
    q_gammas_to_test,
    q_epsilons_to_test
))

q_results = []
total_q_runs = len(q_param_grid)

print(f"Total Q-Table combinations to test: {total_q_runs} (3 alphas * 2 gammas * 3 epsilons)")

for i, (alpha, gamma, epsilon) in enumerate(q_param_grid):
    print(f"\n--- Q-Table Run {i+1}/{total_q_runs} ---")
    print(f"Params: Alpha={alpha}, Gamma={gamma}, Epsilon={epsilon}")
    
    # Modify config parameters on the fly
    config.Q_TABLE_ALPHA = alpha
    config.Q_TABLE_GAMMA = gamma
    config.Q_TABLE_EPSILON = epsilon
    
    try:
        start_time = time.time()
        # Run the simulation using the function from simulation.py
        simulation.run_simulation('q_table')
        run_duration = time.time() - start_time
        
        # Get results from the global metrics object
        q_metrics = metrics.final_report.get('q_table', {})
        q_p99 = q_metrics.get('steady_p99', 0)
        
        if q_p99 > 0:
            improvement_pct = (1 - (q_p99 / reactive_p99)) * 100
            print(f"Result: Q-Table p99: {q_p99:.2f} ms (Took {run_duration:.1f}s)")
            print(f"IMPROVEMENT vs Reactive: {improvement_pct:+.2f}%")
            q_results.append({'params': (alpha, gamma, epsilon), 'p99': q_p99, 'improvement': improvement_pct})
        else:
            print(f"Result: Q-Table simulation failed or returned p99 of 0. (Took {run_duration:.1f}s)")

    except Exception as e:
        print(f"ERROR during Q-Table simulation run {i+1}: {e}")


# --- 3. DQN Hyperparameter Sweep ---
print("\n" + "=" * 80)
print("STARTING: DQN Sweep")
print("=" * 80)

# --- Customize your DQN parameters here ---
dqn_gammas_to_test = [0.9, 0.99]
dqn_batch_sizes_to_test = [64, 128]
dqn_target_updates_to_test = [5, 10]
# --- NEW: Added Epsilon Decay to the sweep ---
# These values are much smaller to force exploitation
# within the 400s simulation time.
dqn_decays_to_test = [30, 60, 100] # <-- NEW
# ---

dqn_param_grid = list(itertools.product(
    dqn_gammas_to_test,
    dqn_batch_sizes_to_test,
    dqn_target_updates_to_test,
    dqn_decays_to_test # <-- NEW
))

dqn_results = []
total_dqn_runs = len(dqn_param_grid)

print(f"Total DQN combinations to test: {total_dqn_runs} (2 gammas * 2 batches * 2 targets * 3 decays)")

# Calculate total steps for info printout
total_steps = (config.SIM_TIME / config.PROACTIVE_CHECK_INTERVAL) - 1 # rough steps

for i, (gamma, batch_size, target_update, decay) in enumerate(dqn_param_grid):
    print(f"\n--- DQN Run {i+1}/{total_dqn_runs} ---")
    print(f"Params: Gamma={gamma}, BatchSize={batch_size}, TargetUpdate={target_update}, EpsilonDecay={decay}")
    
    # Calculate what the final epsilon will be, just for info
    final_eps = config.DQN_EPSILON_END + (config.DQN_EPSILON_START - config.DQN_EPSILON_END) * \
                math.exp(-1. * total_steps / decay)
    print(f"  (Note: Final Epsilon at 400s will be ~{final_eps:.2f})")
    
    # Modify config parameters on the fly
    config.DQN_GAMMA = gamma
    config.DQN_BATCH_SIZE = batch_size
    config.DQN_TARGET_UPDATE = target_update
    config.DQN_EPSILON_DECAY = decay # <-- NEW
    
    try:
        start_time = time.time()
        # Run the simulation
        simulation.run_simulation('dqn')
        run_duration = time.time() - start_time
        
        # Get results
        d_metrics = metrics.final_report.get('dqn', {})
        d_p99 = d_metrics.get('steady_p99', 0)
        
        if d_p99 > 0:
            improvement_pct = (1 - (d_p99 / reactive_p99)) * 100
            print(f"Result: DQN p99: {d_p99:.2f} ms (Took {run_duration:.1f}s)")
            print(f"IMPROVEMENT vs Reactive: {improvement_pct:+.2f}%")
            dqn_results.append({'params': (gamma, batch_size, target_update, decay), 'p99': d_p99, 'improvement': improvement_pct})
        else:
            print(f"Result: DQN simulation failed or returned p99 of 0. (Took {run_duration:.1f}s)")

    except Exception as e:
        print(f"ERROR during DQN simulation run {i+1}: {e}")


# --- 4. Final Summary ---
print("\n" + "=" * 80)
print("SWEEP COMPLETE - FINAL SUMMARY")
print("=" * 80)
print(f"Baseline Reactive p99: {reactive_p99:.2f} ms")

if q_results:
    best_q = max(q_results, key=lambda x: x['improvement'])
    print(f"\nBest Q-Table Run:")
    print(f"  Params (A, G, E): {best_q['params']}")
    print(f"  Improvement: {best_q['improvement']:+.2f}% (p99: {best_q['p99']:.2f} ms)")
else:
    print("\nNo successful Q-Table runs.")
    
if dqn_results:
    best_dqn = max(dqn_results, key=lambda x: x['improvement'])
    print(f"\nBest DQN Run:")
    print(f"  Params (G, Batch, Target, Decay): {best_dqn['params']}") # <-- Updated params
    print(f"  Improvement: {best_dqn['improvement']:+.2f}% (p99: {best_dqn['p99']:.2f} ms)")
else:
    print("\nNo successful DQN runs.")

print("\n" + "=" * 80)