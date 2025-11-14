import random
import numpy as np
import simpy

import config
from cluster import Cluster
from workload import workload_generator
from balancers import reactive_balancer, ProLBR_Agent
from metrics import metrics


def run_simulation(balancer_type):
    """Run a single simulation with the specified balancer"""
    print(f"\n{'='*65}")
    print(f"RUNNING: {balancer_type.upper()} BALANCER")
    print(f"{'='*65}")

    # Set random seeds for reproducibility
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    # Reset metrics for this run
    metrics.reset()

    # Initialize simulation environment
    env = simpy.Environment()
    cluster = Cluster(env)
    
    # Start workload generator
    env.process(workload_generator(env, cluster))

    # Start appropriate balancer
    if balancer_type == 'reactive':
        env.process(reactive_balancer(env, cluster))
    elif balancer_type == 'proactive':
        agent = ProLBR_Agent(env, cluster)

    # Start metrics logger
    def logger(env):
        while True:
            yield env.timeout(10)
            metrics.log_metrics(env, balancer_type, cluster)

    env.process(logger(env))

    # Run simulation
    env.run(until=config.SIM_TIME)
    
    # Generate final report
    metrics.report(balancer_type)


def print_comparison():
    """Print comparison between reactive and proactive balancers"""
    print("\n" + "="*65)
    print("COMPARISON")
    print("="*65)

    r_metrics = metrics.final_report.get('reactive', {})
    p_metrics = metrics.final_report.get('proactive', {})

    r_migs = r_metrics.get('migrations', 0)
    p_migs = p_metrics.get('migrations', 0)

    r_p99 = r_metrics.get('steady_p99', 0)
    p_p99 = p_metrics.get('steady_p99', 0)

    print(f"         | Reactive   | Proactive")
    print(f"---------|------------|------------")
    print(f"Migs     | {r_migs:<10} | {p_migs:<10} ")
    print(f"Steady p99| {r_p99:<10.2f} | {p_p99:<10.2f} ")

    if r_p99 and p_p99:
        if p_p99 < r_p99:
            improvement = r_p99 - p_p99
            pct_improvement = (1 - p_p99/r_p99) * 100
            print(f"\n✅ Proactive was {improvement:.2f}ms "
                  f"({pct_improvement:.0f}%) better!")
        else:
            degradation = p_p99 - r_p99
            print(f"\n❌ Proactive was {degradation:.2f}ms worse.")

    print("="*65)


if __name__ == '__main__':
    print("\n" + "="*65)
    print("PRO-LBR SIMULATION: Proactive vs Reactive Load Balancing")
    print("="*65)
    print("\nScenario:")
    
    node_capacity = 1000 / config.PROCESSING_TIME_MS
    total_capacity = config.NUM_NODES * node_capacity
    utilization = (config.REQUEST_RATE / total_capacity) * 100
    
    print(f"  • {config.NUM_NODES} nodes @ {node_capacity:.0f} req/s each "
          f"(total capacity: {total_capacity:.0f} req/s)")
    print(f"  • Incoming load: {config.REQUEST_RATE} req/s "
          f"({utilization:.0f}% utilization)")
    print("  • 35% of traffic goes to shifting hotspot")
    print("  • Hotspot shifts every 150 seconds")
    print("\nBalancers:")
    print(f"  • REACTIVE: Checks every {config.REACTIVE_CHECK_INTERVAL}s, "
          f"migrates if imbalance > {config.REACTIVE_THRESHOLD}")
    print(f"  • PROACTIVE: Checks every {config.PROACTIVE_CHECK_INTERVAL}s, "
          "learns via Q-learning")
    print("\n" + "="*65)

    # Run both simulations
    run_simulation('reactive')
    run_simulation('proactive')

    # Print comparison
    print_comparison()