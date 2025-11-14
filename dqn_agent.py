"""
Contains the DQNAgent (Advanced Submission).
"""
import random
import math
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

import config
from balancers import BaseRLAgent
from dqn_model import DQN, ReplayBuffer, Transition, device


class DQNAgent(BaseRLAgent):
    """
    Implements the Proactive Balancer using a Deep Q-Network.
    """
    def __init__(self, env, cluster):
        super().__init__(env, cluster, 'dqn')
        
        self.n_observations = 4  # [load, imbalance, load_v, imb_v]
        self.n_actions = 2       # [no_op, migrate]

        # Load DQN-specific parameters
        self.gamma = config.DQN_GAMMA
        self.eps_start = config.DQN_EPSILON_START
        self.eps_end = config.DQN_EPSILON_END
        self.eps_decay = config.DQN_EPSILON_DECAY
        self.batch_size = config.DQN_BATCH_SIZE
        self.target_update = config.DQN_TARGET_UPDATE
        
        # Initialize Policy and Target Networks
        self.policy_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only for evaluation

        self.optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=config.DQN_LEARNING_RATE
        )
        self.memory = ReplayBuffer(config.DQN_REPLAY_BUFFER_SIZE)
        self.steps_done = 0

    def get_state(self, raw_state):
        """
        Normalizes the raw state and converts it to a PyTorch Tensor.
        """
        # Simple normalization (dividing by a reasonable expected max)
        # This is crucial for neural network stability
        raw_state[0] /= 50  # Max load
        raw_state[1] /= 50  # Max imbalance
        raw_state[2] /= 10  # Max load velocity
        raw_state[3] /= 10  # Max imbalance velocity
        
        # Add a batch dimension and send to device
        return torch.tensor(raw_state, dtype=torch.float32, device=device).unsqueeze(0)

    def choose_action(self, state):
        """Epsilon-greedy action selection using the DQN"""
        # Calculate current epsilon
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        
        if random.random() > eps_threshold:
            # Exploit: Choose action with highest Q-value
            with torch.no_grad():
                # state is [1, 4] -> policy_net(state) is [1, 2]
                # .max(1) finds max value in dim 1
                # .view(1, 1) converts it to a [1, 1] tensor
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            # Explore: Choose a random action
            return torch.tensor(
                [[random.randrange(self.n_actions)]], 
                device=device, dtype=torch.long
            )

    def update_model(self, last_state, last_action, reward, new_state):
        """Saves transition to buffer and runs one optimization step"""
        
        # Convert reward to a tensor
        reward = torch.tensor([reward], device=device)
        
        # Store the transition in memory
        self.memory.push(last_state, last_action, new_state, reward)

        # Don't optimize until we have enough samples
        if len(self.memory) < self.batch_size:
            return

        # --- Perform one step of optimization ---
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch
        batch = Transition(*zip(*transitions))

        # Concatenate all states, actions, rewards
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        # Q(s, a) - Q-values for the actions we *actually took*
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # max(Q(s', a')) - Max Q-value for the *next* state
        # We use target_net for stability
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()

        # Compute the expected Q values (Bellman equation)
        # Expected = R + gamma * max(Q(s', a'))
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(
            state_action_values, 
            expected_state_action_values.unsqueeze(1)
        )

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1) # Gradient clipping
        self.optimizer.step()

    def run(self):
        """Main DQN learning loop"""
        yield self.env.timeout(15)  # Warmup
        
        # Get initial state
        last_state = self.get_state(self.get_raw_state())
        last_action = self.choose_action(last_state)
        last_cost = self.execute_action(last_action.item())
        
        while True:
            # 1. Get reward for previous action
            reward = self.get_reward() - last_cost

            # 2. Get new state
            raw_state = self.get_raw_state()
            new_state = self.get_state(raw_state)

            # 3. Update model (push to buffer and optimize)
            self.update_model(last_state, last_action, reward, new_state)

            # 4. Choose new action
            action = self.choose_action(new_state)

            # 5. Execute action and get cost
            last_cost = self.execute_action(action.item())
            
            # 6. Cycle state
            last_state = new_state
            last_action = action
            
            # 7. Update target network periodically
            if self.steps_done % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            yield self.env.timeout(config.PROACTIVE_CHECK_INTERVAL)