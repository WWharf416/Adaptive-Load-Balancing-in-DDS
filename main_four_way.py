"""
Main entry point for the 4-way comparison:
1. Reactive (Baseline)
2. Q-Table (Tuned, 108-state)
3. Q-Table-Large (Tuned, 1400-state) <-- NEW
4. DQN (Advanced Approach)
"""

import config
from simulation import run_simulation
from metrics import metrics

def print_simulation_header():
    """Prints the shared simulation parameters"""
    print("\n" + "="*65)
    print("PRO-LBR SIMULATION: 4-Way Balancer Comparison")
    print("="*65)
    print("\nShared Scenario:")
    
    node_capacity = 1000 / config.PROCESSING_TIME_MS
    total_capacity = config.NUM_NODES * node_capacity
    utilization = (config.REQUEST_RATE / total_capacity) * 100
    
    print(f"  • {config.NUM_NODES} nodes @ {node_capacity:.0f} req/s each "
          f"(total capacity: {total_capacity:.0f} req/s)")
    print(f"  • Incoming load: {config.REQUEST_RATE} req/s "
          f"({utilization:.0f}% utilization)")
    print("  • 35% of traffic goes to shifting hotspot (shifts every 150s)")
    print("  • All balancers migrate the *hottest* available chunk.")
    print("\nBalancers:")
    print(f"  1. REACTIVE: Checks every {config.REACTIVE_CHECK_INTERVAL}s, "
          f"migrates if imbalance > {config.REACTIVE_THRESHOLD}")
    print(f"  2. Q-TABLE: Checks every {config.PROACTIVE_CHECK_INTERVAL}s, "
          f"(108 states, A={config.Q_TABLE_ALPHA}, G={config.Q_TABLE_GAMMA}, E={config.Q_TABLE_EPSILON})")
    print(f"  3. Q-TABLE-LRG: Checks every {config.PROACTIVE_CHECK_INTERVAL}s, "
          f"(1400 states, same params)")
    print(f"  4. DQN: Checks every {config.PROACTIVE_CHECK_INTERVAL}s, "
          f"(LR={config.DQN_LEARNING_RATE}, G={config.DQN_GAMMA}, Epsilon-decay)")
    print("\n" + "="*65)

def print_4_way_comparison():
    """Prints a final 4-way comparison table"""
    print("\n" + "="*65)
    print("FINAL 4-WAY COMPARISON")
    print("="*65)

    r_metrics = metrics.final_report.get('reactive', {})
    q_metrics = metrics.final_report.get('q_table', {})
    ql_metrics = metrics.final_report.get('q_table_large', {}) # <-- NEW
    d_metrics = metrics.final_report.get('dqn', {})

    r_p99 = r_metrics.get('steady_p99', 0)
    q_p99 = q_metrics.get('steady_p99', 0)
    ql_p99 = ql_metrics.get('steady_p99', 0) # <-- NEW
    d_p99 = d_metrics.get('steady_p99', 0)
    
    r_migs = r_metrics.get('migrations', 0)
    q_migs = q_metrics.get('migrations', 0)
    ql_migs = ql_metrics.get('migrations', 0) # <-- NEW
    d_migs = d_metrics.get('migrations', 0)

    print(f"         | Reactive   | Q-Table    | Q-Table-LRG | DQN (Advanced)")
    print(f"---------|------------|------------|-------------|----------------")
    print(f"Migs     | {r_migs:<10} | {q_migs:<10} | {ql_migs:<11} | {d_migs:<10}")
    print(f"Steady p99| {r_p99:<10.2f} | {q_p99:<10.2f} | {ql_p99:<11.2f} | {d_p99:<10.2f}")

    if r_p99 > 0:
        if q_p99 > 0:
            q_pct = (1 - q_p99/r_p99) * 100
            print(f"\n• Q-Table vs Reactive:   {q_pct:+.1f}% improvement")
        if ql_p99 > 0: # <-- NEW
            ql_pct = (1 - ql_p99/r_p99) * 100
            print(f"• Q-Large vs Reactive:   {ql_pct:+.1f}% improvement")
        if d_p99 > 0:
            d_pct = (1 - d_p99/r_p99) * 100
            print(f"• DQN vs Reactive:       {d_pct:+.1f}% improvement")
        if q_p99 > 0 and ql_p99 > 0: # <-- NEW
            ql_vs_q_pct = (1 - ql_p99/q_p99) * 100
            print(f"• Q-Large vs Q-Table:    {ql_vs_q_pct:+.1f}% (Expected to be negative)")

    print("="*65)

if __name__ == '__main__':
    
    print_simulation_header()

    # --- Run all four simulations ---
    run_simulation('reactive')
    run_simulation('q_table')
    run_simulation('q_table_large') # <-- NEW
    run_simulation('dqn')

    # --- Print final comparison ---
    print_4_way_comparison()