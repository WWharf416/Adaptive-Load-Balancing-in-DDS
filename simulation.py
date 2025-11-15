"""
Contains the main simulation running logic.
"""
import random
import numpy as np
import simpy
import config
import torch

from cluster import Cluster
from workload import workload_generator
from metrics import metrics
from balancers import reactive_balancer
from q_table_agent import QTableAgent
from q_table_large_agent import QTableLargeAgent  # <-- 1. NEW IMPORT
from dqn_agent import DQNAgent

def run_simulation(balancer_type):
    """Run a single simulation with the specified balancer"""
    print(f"\n{'='*65}")
    print(f"RUNNING: {balancer_type.upper()} BALANCER")
    print(f"{'='*65}")

    # Set random seeds for reproducibility
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)

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
    elif balancer_type == 'q_table':
        agent = QTableAgent(env, cluster)
    elif balancer_type == 'q_table_large':  # <-- 2. NEW ELIF BLOCK
        agent = QTableLargeAgent(env, cluster)
    elif balancer_type == 'dqn':
        agent = DQNAgent(env, cluster)
    else:
        raise ValueError(f"Unknown balancer type: {balancer_type}")

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