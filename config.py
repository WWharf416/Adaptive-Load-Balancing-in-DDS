# Simulation settings
RANDOM_SEED = 42
NUM_NODES = 4
NUM_CHUNKS = 64
KEYS_PER_CHUNK = 1000
SIM_TIME = 400

# System capacity parameters
REQUEST_RATE = 220          # req/s - incoming load (WAS 180)
PROCESSING_TIME_MS = 8      # base processing time in ms (125 req/s per node)

# Reactive balancer parameters
REACTIVE_CHECK_INTERVAL = 45    # seconds between checks
REACTIVE_THRESHOLD = 5          # minimum imbalance to trigger migration (WAS 6)
MIGRATION_TIME = 3              # seconds to complete migration

# Proactive balancer parameters (Q-learning)
PROACTIVE_CHECK_INTERVAL = 10   # seconds between checks
ALPHA = 0.3                     # learning rate (This will be overridden by sweep script)
GAMMA = 0.7                     # discount factor (This will be overridden by sweep script)
EPSILON = 0.25                  # exploration rate (This will be overridden by sweep script)
MIGRATION_COST_PENALTY = 2      # cost penalty for migrations
MAX_MIGRATIONS_PER_CYCLE = 2    # max migrations per decision

# Metrics and monitoring
RECENT_SAMPLE_SIZE = 1500       # recent requests for p99 calculation
P99_WARMUP_MIN_SAMPLES = 100    # minimum samples before calculating p99