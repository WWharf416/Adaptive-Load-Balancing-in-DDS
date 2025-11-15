"""
Contains the QTableLargeAgent, designed to show the curse of dimensionality.
It uses the same logic as the base QTableAgent but with a
much finer-grained (larger) state space.
"""
import random
import collections
import numpy as np
import config
from balancers import BaseRLAgent


class QTableLargeAgent(BaseRLAgent):
    """
    Implements the Proactive Balancer using a simple Q-table
    with a VERY LARGE state space (1400 states) to demonstrate
    the curse of dimensionality.
    """
    
    def __init__(self, env, cluster):
        # We use 'q_table_large' as the type for metrics
        super().__init__(env, cluster, 'q_table_large')
        
        # Q-table: state -> [Q(s,do_nothing), Q(s,migrate)]
        self.q_table = collections.defaultdict(lambda: np.zeros(2))
        
        # We use the *same* tuned parameters as the base Q-table
        # for a fair comparison.
        self.alpha = config.Q_TABLE_ALPHA
        self.gamma = config.Q_TABLE_GAMMA
        self.epsilon = config.Q_TABLE_EPSILON

    def get_state(self, raw_state):
        """
        Discretizes the 4D raw state into a hashable tuple
        for the Q-table, using a MUCH LARGER state space.
        """
        max_load, imbalance, load_v, imb_v = raw_state

        # Discretize load (8 levels instead of 4)
        if max_load <= 2: load_lvl = 1
        elif max_load <= 4: load_lvl = 2
        elif max_load <= 6: load_lvl = 3
        elif max_load <= 8: load_lvl = 4
        elif max_load <= 10: load_lvl = 5
        elif max_load <= 13: load_lvl = 6
        elif max_load <= 16: load_lvl = 7
        else: load_lvl = 8 # 8 levels total

        # Discretize imbalance (7 levels instead of 3)
        if imbalance <= 1: imb_lvl = 1
        elif imbalance <= 2: imb_lvl = 2
        elif imbalance <= 3: imb_lvl = 3
        elif imbalance <= 4: imb_lvl = 4
        elif imbalance <= 5: imb_lvl = 5
        elif imbalance <= 6: imb_lvl = 6
        else: imb_lvl = 7 # 7 levels total

        # Discretize velocities (5 levels instead of 3)
        if load_v > 2: load_v_lvl = 2
        elif load_v > 0.5: load_v_lvl = 1
        elif load_v < -2: load_v_lvl = -2
        elif load_v < -0.5: load_v_lvl = -1
        else: load_v_lvl = 0 # 5 levels total

        if imb_v > 2: imb_v_lvl = 2
        elif imb_v > 0.5: imb_v_lvl = 1
        elif imb_v < -2: imb_v_lvl = -2
        elif imb_v < -0.5: imb_v_lvl = -1
        else: imb_v_lvl = 0 # 5 levels total
        
        # Total states = 8 * 7 * 5 * 5 = 1400 states
        return (load_lvl, imb_lvl, load_v_lvl, imb_v_lvl)

    def choose_action(self, state):
        """Epsilon-greedy action selection from Q-table"""
        if random.random() < self.epsilon:
            return random.randint(0, 1)  # Explore
        else:
            return int(np.argmax(self.q_table[state]))  # Exploit

    def update_model(self, last_state, last_action, reward, new_state):
        """Update the Q-table using the Bellman equation"""
        best_q = np.max(self.q_table[new_state])
        
        self.q_table[last_state][last_action] += self.alpha * (
            reward + self.gamma * best_q - self.q_table[last_state][last_action]
        )

    def run(self):
        """Main Q-learning loop"""
        yield self.env.timeout(15)  # Warmup

        last_state = self.get_state(self.get_raw_state())
        last_action = 0
        last_cost = 0

        while True:
            # 1. Get reward for previous action
            reward = self.get_reward() - last_cost

            # 2. Get new state
            raw_state = self.get_raw_state()
            new_state = self.get_state(raw_state)

            # 3. Update model
            self.update_model(last_state, last_action, reward, new_state)

            # 4. Choose new action
            action = self.choose_action(new_state)

            # 5. Execute action and get cost
            last_cost = self.execute_action(action)
            
            # 6. Cycle state
            last_state = new_state
            last_action = action

            yield self.env.timeout(config.PROACTIVE_CHECK_INTERVAL)