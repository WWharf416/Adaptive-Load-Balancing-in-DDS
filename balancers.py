"""Load balancing strategies: Reactive and Proactive (Q-learning)"""
import random
import collections
import numpy as np
import config
from metrics import metrics


def reactive_balancer(env, cluster):
    """
    Reactive balancer: Periodically checks for imbalance and migrates
    chunks when threshold is exceeded.
    """
    while True:
        yield env.timeout(config.REACTIVE_CHECK_INTERVAL) 

        loads = cluster.get_node_loads()
        if not loads:
            continue

        # Find most and least loaded nodes
        max_id = max(loads, key=loads.get)
        min_id = min(loads, key=loads.get)
        diff = loads[max_id] - loads[min_id]

        # Migrate if imbalance exceeds threshold
        if diff > config.REACTIVE_THRESHOLD:
            from_node = cluster.nodes[max_id]
            to_node = cluster.nodes[min_id]

            # Find chunks that can be migrated (not on cooldown)
            available = [c for c in from_node.chunks 
                        if cluster.can_migrate_chunk(c)]
            
            if available:
                chunk = random.choice(available)
                env.process(cluster.migrate_chunk(
                    chunk, from_node, to_node, 'reactive'))


class ProLBR_Agent:
    """
    Proactive Q-learning agent that learns optimal migration policies
    """
    
    def __init__(self, env, cluster):
        self.env = env
        self.cluster = cluster
        # Q-table: state -> [Q(s,do_nothing), Q(s,migrate)]
        self.q_table = collections.defaultdict(lambda: np.zeros(2))
        self.action = env.process(self.run())

    def get_state(self):
        """Discretize system state into (load_level, imbalance_level)"""
        loads = self.cluster.get_node_loads()
        if not loads:
            return (1, 1)

        max_load = max(loads.values())
        imbalance = max(loads.values()) - min(loads.values())

        # Discretize load level
        if max_load <= 3:
            load_lvl = 1
        elif max_load <= 7:
            load_lvl = 2
        elif max_load <= 12:
            load_lvl = 3
        else:
            load_lvl = 4

        # Discretize imbalance level
        if imbalance <= 2:
            imb_lvl = 1
        elif imbalance <= 6:
            imb_lvl = 2
        else:
            imb_lvl = 3

        return (load_lvl, imb_lvl)

    def get_reward(self):
        """
        Calculate reward based on load imbalance and latency.
        Penalizes imbalance and high latency, rewards low latency.
        """
        loads = self.cluster.get_node_loads()
        imbalance = max(loads.values()) - min(loads.values()) if loads else 0

        # Primary: Penalize imbalance directly (-1.5 per request of imbalance)
        imbalance_penalty = -imbalance * 1.5

        # Secondary: Check p99 latency
        recent = metrics.response_times[-1000:]
        if len(recent) < 200:
            # Not enough data, just return imbalance penalty
            return imbalance_penalty

        p99 = np.percentile(recent, 99)

        # New, more granular latency-based reward
        # The goal is to keep p99 under 50ms
        if p99 < 30:
            latency_reward = 15     # Excellent
        elif p99 < 50:
            latency_reward = 5      # Good (Target)
        elif p99 < 100:
            latency_reward = -5     # High
        elif p99 < 200:
            latency_reward = -10    # Very high
        else:
            latency_reward = -20    # Critical

        return imbalance_penalty + latency_reward

    def run(self):
        """Main Q-learning loop"""
        # Warmup period
        yield self.env.timeout(15)

        last_state = self.get_state()
        last_action = 0
        last_cost = 0

        while True:
            state = self.get_state()

            # Calculate reward (subtract cost of last action)
            reward = self.get_reward() - last_cost

            # Q-learning update: Q(s,a) += α[r + γ·max Q(s',a') - Q(s,a)]
            best_q = np.max(self.q_table[state])
            self.q_table[last_state][last_action] += config.ALPHA * (
                reward + config.GAMMA * best_q - self.q_table[last_state][last_action]
            )

            # Epsilon-greedy action selection
            if random.random() < config.EPSILON:
                action = random.randint(0, 1)  # Explore
            else:
                action = int(np.argmax(self.q_table[state]))  # Exploit

            last_cost = 0

            # Execute action: 0 = do nothing, 1 = migrate
            if action == 1:
                loads = self.cluster.get_node_loads()
                if loads:
                    from_id = max(loads, key=loads.get)
                    from_node = self.cluster.nodes[from_id]
                    to_node = self.cluster.get_least_loaded_node()

                    diff = loads[from_id] - loads[to_node.node_id]

                    # Only migrate if imbalance is meaningful
                    if diff > 2 and from_node.chunks:
                        available = [c for c in from_node.chunks 
                                   if self.cluster.can_migrate_chunk(c)]

                        if available:
                            # Migrate up to MAX_MIGRATIONS_PER_CYCLE chunks
                            chunks_to_move = available[:config.MAX_MIGRATIONS_PER_CYCLE]

                            for chunk in chunks_to_move:
                                self.env.process(self.cluster.migrate_chunk(
                                    chunk, from_node, to_node, 'proactive'))
                                last_cost += config.MIGRATION_COST_PENALTY

            last_state = state
            last_action = action

            yield self.env.timeout(config.PROACTIVE_CHECK_INTERVAL)