"""
Contains the QTableAgent (your original tuned submission).
"""
import random
import collections
import numpy as np
import config
from balancers import BaseRLAgent


class QTableAgent(BaseRLAgent):
    """
    Implements the Proactive Balancer using a simple Q-table.
    This represents your tuned submission.
    """
    
    def __init__(self, env, cluster):
        super().__init__(env, cluster, 'q_table')
        
        # Q-table: state -> [Q(s,do_nothing), Q(s,migrate)]
        self.q_table = collections.defaultdict(lambda: np.zeros(2))
        
        # Load tuned parameters from config
        self.alpha = config.Q_TABLE_ALPHA
        self.gamma = config.Q_TABLE_GAMMA
        self.epsilon = config.Q_TABLE_EPSILON

    def get_state(self, raw_state):
        """
        Discretizes the 4D raw state into a hashable tuple
        for the Q-table.
        """
        max_load, imbalance, load_v, imb_v = raw_state

        # Discretize load
        if max_load <= config.LOAD_LVL_THRESH[0]: load_lvl = 1
        elif max_load <= config.LOAD_LVL_THRESH[1]: load_lvl = 2
        elif max_load <= config.LOAD_LVL_THRESH[2]: load_lvl = 3
        else: load_lvl = 4

        # Discretize imbalance
        if imbalance <= config.IMB_LVL_THRESH[0]: imb_lvl = 1
        elif imbalance <= config.IMB_LVL_THRESH[1]: imb_lvl = 2
        else: imb_lvl = 3

        # Discretize velocities (1: Inc, 0: Stable, -1: Dec)
        load_v_lvl = 1 if load_v > 1 else (-1 if load_v < -1 else 0)
        imb_v_lvl = 1 if imb_v > 1 else (-1 if imb_v < -1 else 0)

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