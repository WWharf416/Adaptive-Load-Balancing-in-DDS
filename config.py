# ====================================================================
# --- SIMULATION SETTINGS ---
# ====================================================================
RANDOM_SEED = 42
NUM_NODES = 4
NUM_CHUNKS = 64
SIM_TIME = 400  # Total simulation time in seconds

# ====================================================================
# --- SYSTEM CAPACITY & WORKLOAD ---
# ====================================================================
REQUEST_RATE = 220      # req/s - incoming load
PROCESSING_TIME_MS = 8  # base processing time in ms

# ====================================================================
# --- BALANCER SETTINGS ---
# ====================================================================
# --- 1. Reactive Balancer ---
REACTIVE_CHECK_INTERVAL = 45    # seconds between checks
REACTIVE_THRESHOLD = 5          # minimum imbalance to trigger migration
MIGRATION_TIME = 3              # seconds to complete migration

# --- 2. Shared RL Agent Settings ---
PROACTIVE_CHECK_INTERVAL = 10   # seconds between checks for ALL RL agents
MIGRATION_COST_PENALTY = 2      # cost penalty for migrations
MAX_MIGRATIONS_PER_CYCLE = 2    # max migrations per decision

# State discretization thresholds
# Load Level: 1 (<=3), 2 (4-7), 3 (8-12), 4 (>12)
LOAD_LVL_THRESH = [3, 7, 12]
# Imbalance Level: 1 (<=2), 2 (3-6), 3 (>6)
IMB_LVL_THRESH = [2, 6]

# Reward function weights
# Penalty = -imbalance * IMBALANCE_PENALTY_FACTOR
IMBALANCE_PENALTY_FACTOR = 1.5
# p99 thresholds for reward
LATENCY_REWARD_THRESH = [30, 50, 100, 200]
# Reward values: [Excellent, Good, High, Very High, Critical]
LATENCY_REWARD_VALUES = [15, 5, -5, -10, -20]

# --- 3. Q-Table Agent (Your Tuned Submission) ---
# Best params from your sweep_results.csv
Q_TABLE_ALPHA = 0.1       # learning rate
Q_TABLE_GAMMA = 0.7       # discount factor
Q_TABLE_EPSILON = 0.1     # exploration rate

# --- 4. DQN Agent (Advanced) ---
DQN_GAMMA = 0.9           # discount factor (higher for DQN)
DQN_EPSILON_START = 1.0   # exploration start
DQN_EPSILON_END = 0.05    # exploration end
DQN_EPSILON_DECAY = 500   # steps to decay
DQN_LEARNING_RATE = 1e-4  # learning rate for Adam optimizer
DQN_BATCH_SIZE = 64       # samples from replay buffer
DQN_REPLAY_BUFFER_SIZE = 10000
DQN_TARGET_UPDATE = 10    # update target network every 10 steps

# ====================================================================
# --- METRICS & MONITORING ---
# ====================================================================
RECENT_SAMPLE_SIZE = 1500       # recent requests for p99 calculation
P99_WARMUP_MIN_SAMPLES = 100    # minimum samples before calculating p99