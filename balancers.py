"""
Contains the Reactive Balancer and the Base Class for RL Balancers.
"""
import random
import numpy as np
import config
from metrics import metrics


def reactive_balancer(env, cluster):
    """
    Reactive balancer: Periodically checks for imbalance and migrates
    the *hottest* chunk when threshold is exceeded.
    """
    while True:
        yield env.timeout(config.REACTIVE_CHECK_INTERVAL) 

        loads = cluster.get_node_loads()
        if not loads:
            continue

        max_id = max(loads, key=loads.get)
        min_id = min(loads, key=loads.get)
        diff = loads[max_id] - loads[min_id]

        if diff > config.REACTIVE_THRESHOLD:
            from_node = cluster.nodes[max_id]
            to_node = cluster.nodes[min_id]

            # --- MODIFIED: Migrate hottest chunk, not random ---
            chunk_to_move = cluster.get_hottest_chunk(from_node.node_id)
            
            if chunk_to_move is not None:
                env.process(cluster.migrate_chunk(
                    chunk_to_move, from_node, to_node, 'reactive'))


class BaseRLAgent:
    """
    Base class for Proactive Q-Learning and DQN Agents.
    Handles the simulation loop, state calculation, and reward logic.
    """
    
    def __init__(self, env, cluster, balancer_type):
        self.env = env
        self.cluster = cluster
        self.balancer_type = balancer_type
        
        # State tracking for velocity
        self.last_max_load = 0
        self.last_imbalance = 0
        
        # Main simulation process
        self.action = env.process(self.run())

    def get_raw_state(self):
        """
        Calculates the 4-dimensional continuous state of the system.
        Returns: (max_load, imbalance, load_velocity, imbalance_velocity)
        """
        loads = self.cluster.get_node_loads()
        if not loads:
            max_load, imbalance = 0, 0
        else:
            max_load = max(loads.values())
            imbalance = max_load - min(loads.values())

        # Calculate velocity (diff from last step)
        load_v = max_load - self.last_max_load
        imb_v = imbalance - self.last_imbalance
            
        self.last_max_load = max_load
        self.last_imbalance = imbalance

        return np.array([max_load, imbalance, load_v, imb_v], dtype=np.float32)

    def get_reward(self):
        """
        Calculate reward based on load imbalance and latency.
        """
        loads = self.cluster.get_node_loads()
        imbalance = max(loads.values()) - min(loads.values()) if loads else 0

        imbalance_penalty = -imbalance * config.IMBALANCE_PENALTY_FACTOR

        recent = metrics.response_times[-1000:]
        if len(recent) < 200:
            return imbalance_penalty

        p99 = np.percentile(recent, 99)

        if p99 < config.LATENCY_REWARD_THRESH[0]:
            latency_reward = config.LATENCY_REWARD_VALUES[0]
        elif p99 < config.LATENCY_REWARD_THRESH[1]:
            latency_reward = config.LATENCY_REWARD_VALUES[1]
        elif p99 < config.LATENCY_REWARD_THRESH[2]:
            latency_reward = config.LATENCY_REWARD_VALUES[2]
        elif p99 < config.LATENCY_REWARD_THRESH[3]:
            latency_reward = config.LATENCY_REWARD_VALUES[3]
        else:
            latency_reward = config.LATENCY_REWARD_VALUES[4]

        return imbalance_penalty + latency_reward

    def execute_action(self, action):
        """
        Executes the chosen action (0=Do Nothing, 1=Migrate).
        Returns the cost (penalty) of this action.
        """
        if action == 0:
            return 0  # No cost

        # Action == 1: Migrate
        loads = self.cluster.get_node_loads()
        if not loads:
            return 0

        from_id = max(loads, key=loads.get)
        from_node = self.cluster.nodes[from_id]
        to_node = self.cluster.get_least_loaded_node()

        diff = loads[from_id] - loads[to_node.node_id]

        # Only migrate if imbalance is meaningful
        if diff > 2:
            cost = 0
            for _ in range(config.MAX_MIGRATIONS_PER_CYCLE):
                # --- MODIFIED: Migrate hottest chunk ---
                chunk = self.cluster.get_hottest_chunk(from_node.node_id)
                if chunk is not None:
                    self.env.process(self.cluster.migrate_chunk(
                        chunk, from_node, to_node, self.balancer_type))
                    cost += config.MIGRATION_COST_PENALTY
                else:
                    break # No more migratable chunks
            return cost
        
        return 0 # No migration occurred

    def run(self):
        """Main Q-learning loop, implemented by child classes"""
        raise NotImplementedError

    def get_state(self, raw_state):
        """Convert raw state to agent-specific state (tuple or tensor)"""
        raise NotImplementedError

    def choose_action(self, state):
        """Choose an action based on the current state"""
        raise NotImplementedError

    def update_model(self, last_state, last_action, reward, new_state, done):
        """Update the agent's model (Q-table or DQN)"""
        raise NotImplementedError