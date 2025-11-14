import random
import simpy
from metrics import metrics
import config


class Node:
    """Represents a single node in the cluster"""
    
    def __init__(self, env, node_id):
        self.env = env
        self.node_id = node_id
        self.proc_queue = simpy.Resource(env, capacity=1)
        self.chunks = set()

    def process_request(self, request):
        """Process a request with queuing and timing"""
        start_time = self.env.now

        # Base processing time with small variance
        processing_time_ms = config.PROCESSING_TIME_MS + random.uniform(-0.5, 0.5)
        
        # --- THIS LINE WAS THE BUG CAUSING HIGH LATENCY ---
        # The simpy.Resource already models queuing delay. Adding this
        # created a positive feedback loop that made the system unstable.
        # queue_depth = len(self.proc_queue.queue)
        # processing_time_ms += queue_depth * 2
        # --- END BUGFIX ---

        # Simulate processing
        with self.proc_queue.request() as req:
            yield req
            # Convert ms to seconds and add small delay for context switching
            yield self.env.timeout((processing_time_ms / 1000.0) + 0.0001)

        end_time = self.env.now
        response_time_ms = (end_time - start_time) * 1000.0
        metrics.record_response(response_time_ms)

    def get_load(self):
        """Return current queue depth"""
        return len(self.proc_queue.queue)