import simpy
import random
import numpy as np
import collections

# --- Simulation Parameters ---
RANDOM_SEED = 42
NUM_NODES = 4
NUM_CHUNKS = 64
KEYS_PER_CHUNK = 1000
SIM_TIME = 400

# --- WORKING PARAMETERS (System can handle this!) ---
REQUEST_RATE = 180      # 180 req/s
# --- CHANGED ---
# Increased node capacity (from 10ms to 8ms)
# New capacity: 125 req/s per node.
# Hot node load of 92.25 req/s is now ~74% utilization (healthy!)
PROCESSING_TIME_MS = 8 # 8ms base = 125 req/s capacity per node

# --- Balancer Parameters ---
REACTIVE_CHECK_INTERVAL = 45  # Checks every 45s (slow)
REACTIVE_THRESHOLD = 6        # Only migrates if imbalance > 6
MIGRATION_TIME = 3

PROACTIVE_CHECK_INTERVAL = 10 # Checks every 10s (fast)
ALPHA = 0.3
GAMMA = 0.7
EPSILON = 0.25
# --- CHANGED ---
# Migration cost MUST be small, otherwise the agent learns
# that "doing nothing" is better than "trying and paying a cost".
MIGRATION_COST_PENALTY = 2 # Was 8
MAX_MIGRATIONS_PER_CYCLE = 2

# --- Metrics ---
class MetricsCollector:
    def __init__(self):
        self.response_times = []
        self.migrations_reactive = 0
        self.migrations_proactive = 0
        self.p99_samples = []
        self.load_samples = []
        self.last_log_time = 0
        
        # --- CHANGED ---
        # Store final results for comparison
        self.final_report = {}

    def record_response(self, time):
        self.response_times.append(time)

    def record_migration(self, balancer_type):
        if balancer_type == 'reactive':
            self.migrations_reactive += 1
        elif balancer_type == 'proactive':
            self.migrations_proactive += 1

    def log_metrics(self, env, balancer_type, cluster):
        # Only sample from recent requests to get a more responsive p99
        recent = self.response_times[-1500:] # Was -800
        if len(recent) < 100:
            return
        
        p99 = np.percentile(recent, 99)
        self.p99_samples.append((env.now, p99))
        
        loads = cluster.get_node_loads()
        self.load_samples.append(loads.copy())
        
        if env.now - self.last_log_time > 60:
            max_q = max(loads.values()) if loads else 0
            min_q = min(loads.values()) if loads else 0
            avg_q = np.mean(list(loads.values())) if loads else 0
            migs = self.migrations_reactive if balancer_type=='reactive' else self.migrations_proactive
            print(f"  t={env.now:.0f}s | p99={p99:.1f}ms | Q: max={max_q} min={min_q} avg={avg_q:.1f} | Migs={migs}")
            self.last_log_time = env.now

    def reset(self):
        self.response_times = []
        self.migrations_reactive = 0
        self.migrations_proactive = 0
        self.p99_samples = []
        self.load_samples = []
        self.last_log_time = 0
        # Do NOT reset final_report

    def report(self, balancer_type):
        print(f"\n{'='*65}")
        print(f"FINAL RESULTS: {balancer_type.upper()}")
        print(f"{'='*65}")
        
        # --- CHANGED ---
        # Store results in the dictionary
        self.final_report[balancer_type] = {}
        report_data = self.final_report[balancer_type]
        
        if len(self.response_times) < 1000:
            print("❌ System overloaded - insufficient data")
            report_data['requests'] = len(self.response_times)
            report_data['migrations'] = 0
            report_data['p99'] = 0
            report_data['steady_p99'] = 0
            return

        p50 = np.percentile(self.response_times, 50)
        p95 = np.percentile(self.response_times, 95)
        p99 = np.percentile(self.response_times, 99)
        p999 = np.percentile(self.response_times, 99.9)
        avg = np.mean(self.response_times)
        max_lat = np.max(self.response_times)
        migrations = self.migrations_reactive if balancer_type == 'reactive' else self.migrations_proactive
        
        print(f"Completed Requests:  {len(self.response_times):,}")
        print(f"Total Migrations:    {migrations}")
        print(f"Average Latency:     {avg:.2f} ms")
        print(f"p50 Latency:         {p50:.2f} ms")
        print(f"p95 Latency:         {p95:.2f} ms")
        print(f"p99 Latency:         {p99:.2f} ms")
        print(f"p99.9 Latency:       {p999:.2f} ms")
        print(f"Max Latency:         {max_lat:.2f} ms")
        
        steady_p99_val = 0.0
        steady_p99 = [p for t, p in self.p99_samples if t > 80]
        if steady_p99:
            steady_p99_val = np.mean(steady_p99)
            print(f"Steady-State p99:    {steady_p99_val:.2f} ms")
        
        if self.load_samples:
            imbalances = [max(l.values()) - min(l.values()) for l in self.load_samples if l]
            if imbalances:
                print(f"Avg Load Imbalance:  {np.mean(imbalances):.1f} requests")
                print(f"Max Load Imbalance:  {np.max(imbalances):.1f} requests")
        
        # --- CHANGED ---
        # Store key metrics for final comparison
        report_data['requests'] = len(self.response_times)
        report_data['migrations'] = migrations
        report_data['p99'] = p99
        report_data['steady_p99'] = steady_p99_val


metrics = MetricsCollector()

# --- Cluster ---
class Node:
    def __init__(self, env, node_id):
        self.env = env
        self.node_id = node_id
        self.proc_queue = simpy.Resource(env, capacity=1)
        self.chunks = set()

    def process_request(self, request):
        start_time = self.env.now
        
        processing_time_ms = PROCESSING_TIME_MS + random.uniform(-0.5, 0.5)
        queue_depth = len(self.proc_queue.queue)
        processing_time_ms += queue_depth * 2
        
        with self.proc_queue.request() as req:
            yield req
            # Add a small delay for context switching, etc.
            # This helps prevent p99s of exactly 0.0
            yield self.env.timeout((processing_time_ms / 1000.0) + 0.0001) 
            
        end_time = self.env.now
        response_time_ms = (end_time - start_time) * 1000.0
        metrics.record_response(response_time_ms)

    def get_load(self):
        return len(self.proc_queue.queue)


class Cluster:
    def __init__(self, env):
        self.env = env
        self.nodes = [Node(env, i) for i in range(NUM_NODES)]
        self.chunk_map = {}
        
        for i in range(NUM_CHUNKS):
            node_id = i % NUM_NODES
            self.nodes[node_id].chunks.add(i)
            self.chunk_map[i] = node_id
            
        self.chunks_in_migration = set()
        self.recently_migrated = {}
        self.migration_cooldown = 25

    def get_node_for_chunk(self, chunk_id):
        return self.nodes[self.chunk_map[chunk_id]]

    def get_node_loads(self):
        return {n.node_id: n.get_load() for n in self.nodes}

    def get_least_loaded_node(self):
        loads = self.get_node_loads()
        # Find node with minimum load
        min_load = min(loads.values())
        min_nodes = [node_id for node_id, load in loads.items() if load == min_load]
        # Break ties randomly
        return self.nodes[random.choice(min_nodes)] 

    def can_migrate_chunk(self, chunk_id):
        if chunk_id in self.chunks_in_migration:
            return False
        if chunk_id in self.recently_migrated:
            if self.env.now - self.recently_migrated[chunk_id] < self.migration_cooldown:
                return False
        return True

    def migrate_chunk(self, chunk_id, from_node, to_node, balancer_type):
        if from_node.node_id == to_node.node_id or chunk_id not in from_node.chunks:
            return

        metrics.record_migration(balancer_type)
        self.chunks_in_migration.add(chunk_id)
        
        yield self.env.timeout(MIGRATION_TIME)
        
        if chunk_id in from_node.chunks:
            from_node.chunks.remove(chunk_id)
            to_node.chunks.add(chunk_id)
            self.chunk_map[chunk_id] = to_node.node_id
            self.recently_migrated[chunk_id] = self.env.now

        if chunk_id in self.chunks_in_migration:
            self.chunks_in_migration.remove(chunk_id)


# --- WORKLOAD: Shifting hotspots across different chunks ---
def workload_generator(env, cluster):
    """
    Realistic workload with CONCENTRATED, shifting hotspots:
    - Phase 1 (0-150s):   35% to 5 chunks on Node 0
    - Phase 2 (150-300s): 35% to 5 chunks on Node 1
    - Phase 3 (300-400s): 35% to 5 chunks on Node 2
    """
    print("  Workload: 35% hotspot concentration (CONCENTRATED), shifts every 150s")
    
    # These chunks are all guaranteed to be on specific nodes
    hot_chunks_p1 = [0, 4, 8, 12, 16] # All on Node 0
    hot_chunks_p2 = [1, 5, 9, 13, 17] # All on Node 1
    hot_chunks_p3 = [2, 6, 10, 14, 18] # All on Node 2

    while True:
        t = env.now
        
        # Determine hot chunk list
        if t < 150:
            hot_list = hot_chunks_p1
        elif t < 300:
            hot_list = hot_chunks_p2
        else:
            hot_list = hot_chunks_p3
        
        # 35% to hot chunks, 65% uniform
        if random.random() < 0.35:
            # Pick a random chunk from the concentrated hot list
            chunk_id = random.choice(hot_list)
        else:
            # Uniformly distribute across all chunks
            chunk_id = random.randint(0, NUM_CHUNKS - 1)
        
        node = cluster.get_node_for_chunk(chunk_id)
        env.process(node.process_request({}))
        
        yield env.timeout(1.0 / REQUEST_RATE)


# --- Reactive Balancer ---
def reactive_balancer(env, cluster):
    """Slow, threshold-based balancing."""
    while True:
        # --- CHANGED ---
        # Removed 'self.' - this is a generator function, not a class method
        yield env.timeout(REACTIVE_CHECK_INTERVAL) 
        
        loads = cluster.get_node_loads()
        if not loads:
            continue
        
        max_id = max(loads, key=loads.get)
        min_id = min(loads, key=loads.get)
        diff = loads[max_id] - loads[min_id]
        
        if diff > REACTIVE_THRESHOLD:
            from_node = cluster.nodes[max_id]
            to_node = cluster.nodes[min_id]
            
            # Be smarter: find a chunk that ISN'T on cooldown
            available = [c for c in from_node.chunks if cluster.can_migrate_chunk(c)]
            if available:
                chunk = random.choice(available) # Still random, but at least valid
                env.process(cluster.migrate_chunk(chunk, from_node, to_node, 'reactive'))

# --- Pro-LBR Agent ---
class ProLBR_Agent:
    """Proactive Q-learning agent."""
    def __init__(self, env, cluster):
        self.env = env
        self.cluster = cluster
        self.q_table = collections.defaultdict(lambda: np.zeros(2))
        self.action = env.process(self.run())

    def get_state(self):
        loads = self.cluster.get_node_loads()
        if not loads:
            return (1, 1)
            
        max_load = max(loads.values())
        imbalance = max(loads.values()) - min(loads.values())
        
        # State discretization
        if max_load <= 3:      # Adjusted for 8ms proc time
            load_lvl = 1
        elif max_load <= 7:
            load_lvl = 2
        elif max_load <= 12:
            load_lvl = 3
        else:
            load_lvl = 4
        
        if imbalance <= 2:
            imb_lvl = 1
        elif imbalance <= 6:
            imb_lvl = 2
        else:
            imb_lvl = 3
            
        return (load_lvl, imb_lvl)

    # --- CHANGED ---
    # New reward function.
    # The old one was "saturated" (p99 was always > 40, so reward was always -5).
    # This new one directly rewards the agent for what we want it
    # to fix (imbalance) and what we want it to preserve (latency).
    def get_reward(self):
        loads = self.cluster.get_node_loads()
        imbalance = max(loads.values()) - min(loads.values()) if loads else 0
        
        # Primary reward: Penalize imbalance directly.
        # -1 reward for every 1 request of imbalance.
        imbalance_penalty = -imbalance
        
        # Secondary reward: Check p99 latency
        recent = metrics.response_times[-1000:]
        if len(recent) < 200:
            return imbalance_penalty # Not enough data, just use imbalance
            
        p99 = np.percentile(recent, 99)
        
        latency_reward = 0
        if p99 < 15:
            latency_reward = 10  # Strong reward for excellent latency
        elif p99 < 25:
            latency_reward = 5   # Small reward for good latency
        elif p99 < 50:
            latency_reward = -5  # Penalize high latency
        else:
            latency_reward = -10 # Strongly penalize very high latency

        # Return combined reward
        return imbalance_penalty + latency_reward

    def run(self):
        # Start learning after a brief warmup
        yield self.env.timeout(15) 
        
        last_state = self.get_state()
        last_action = 0
        last_cost = 0

        while True:
            state = self.get_state()
            
            # Reward is based on the new state we just entered,
            # and we penalize it with the cost of the *last* action.
            reward = self.get_reward() - last_cost
            
            # Q-learning update
            best_q = np.max(self.q_table[state])
            self.q_table[last_state][last_action] += ALPHA * (
                reward + GAMMA * best_q - self.q_table[last_state][last_action]
            )
            
            # Epsilon-greedy action selection
            if random.random() < EPSILON:
                action = random.randint(0, 1) # Explore
            else:
                action = np.argmax(self.q_table[state]) # Exploit
                
            last_cost = 0
            
            if action == 1:  # MIGRATE
                loads = self.cluster.get_node_loads()
                if loads:
                    from_id = max(loads, key=loads.get)
                    from_node = self.cluster.nodes[from_id]
                    to_node = self.cluster.get_least_loaded_node()
                    
                    diff = loads[from_id] - loads[to_node.node_id]
                    
                    # Only migrate if imbalance is meaningful
                    if diff > 2 and from_node.chunks:
                        # Find chunks that can actually be migrated
                        available = [c for c in from_node.chunks if self.cluster.can_migrate_chunk(c)]
                        
                        if available:
                            # --- CHANGED ---
                            # Don't just pick randomly.
                            # Pick the *first* available chunk. This is more deterministic
                            # and stable for the agent to learn about.
                            # (A smarter agent would estimate chunk "hotness"
                            # but this is a good first step)
                            chunks_to_move = available[:MAX_MIGRATIONS_PER_CYCLE]
                            
                            for chunk in chunks_to_move:
                                self.env.process(self.cluster.migrate_chunk(chunk, from_node, to_node, 'proactive'))
                                last_cost += MIGRATION_COST_PENALTY
            
            last_state = state
            last_action = action
            
            yield self.env.timeout(PROACTIVE_CHECK_INTERVAL)


# --- Simulation ---
def run_simulation(balancer_type):
    print(f"\n{'='*65}")
    print(f"RUNNING: {balancer_type.upper()} BALANCER")
    print(f"{'='*65}")
    
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # --- CHANGED ---
    # Only reset the metrics *for the specific run*
    metrics.reset() 
    
    env = simpy.Environment()
    cluster = Cluster(env)
    env.process(workload_generator(env, cluster))
    
    if balancer_type == 'reactive':
        env.process(reactive_balancer(env, cluster))
    elif balancer_type == 'proactive':
        agent = ProLBR_Agent(env, cluster)

    def logger(env):
        while True:
            yield env.timeout(10)
            metrics.log_metrics(env, balancer_type, cluster)
    
    env.process(logger(env))
    
    env.run(until=SIM_TIME)
    metrics.report(balancer_type)


# --- Main ---
if __name__ == "__main__":
    print("\n" + "="*65)
    print("PRO-LBR SIMULATION: Proactive vs Reactive Load Balancing")
    print("="*65)
    print("\nScenario:")
    print(f"  • {NUM_NODES} nodes @ {1000/PROCESSING_TIME_MS:.0f} req/s each (total capacity: {NUM_NODES*1000/PROCESSING_TIME_MS:.0f} req/s)")
    print(f"  • Incoming load: {REQUEST_RATE} req/s ({REQUEST_RATE/(NUM_NODES*1000/PROCESSING_TIME_MS)*100:.0f}% utilization)")
    print("  • 35% of traffic goes to shifting hotspot")
    print("  • Hotspot shifts every 150 seconds")
    print("\nBalancers:")
    print("  • REACTIVE: Checks every 45s, migrates if imbalance > 6")
    print("  • PROACTIVE: Checks every 10s, learns via Q-learning")
    print("\n" + "="*65)
    
    run_simulation('reactive')
    run_simulation('proactive')
    
    print("\n" + "="*65)
    print("COMPARISON")
    print("="*65)
    
    # --- CHANGED ---
    # Correctly read from the stored metrics
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

    if p_p99 < r_p99:
        print(f"\n✅ Proactive was {r_p99 - p_p99:.2f}ms ({(1 - p_p99/r_p99)*100:.0f}%) better!")
    else:
        print(f"\n❌ Proactive was {p_p99 - r_p99:.2f}ms worse.")
    
    print("="*65)