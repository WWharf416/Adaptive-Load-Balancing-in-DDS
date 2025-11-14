import random
import simpy
import collections
from metrics import metrics
import config


class Node:
    """Represents a single node in the cluster"""
    
    def __init__(self, env, node_id):
        self.env = env
        self.node_id = node_id
        self.proc_queue = simpy.Resource(env, capacity=1)
        self.chunks = set()
        
        # --- NEW: Track requests per chunk for intelligent migration ---
        self.chunk_request_counts = collections.defaultdict(int)

    def process_request(self, chunk_id):
        """Process a request with queuing and timing"""
        start_time = self.env.now
        
        # --- NEW: Record request for this chunk ---
        self.chunk_request_counts[chunk_id] += 1

        # Base processing time with small variance
        processing_time_ms = config.PROCESSING_TIME_MS + random.uniform(-0.5, 0.5)

        # Simulate processing
        with self.proc_queue.request() as req:
            yield req
            # Convert ms to seconds
            yield self.env.timeout(processing_time_ms / 1000.0)

        end_time = self.env.now
        response_time_ms = (end_time - start_time) * 1000.0
        metrics.record_response(response_time_ms)

    def get_load(self):
        """Return current queue depth"""
        return len(self.proc_queue.queue)