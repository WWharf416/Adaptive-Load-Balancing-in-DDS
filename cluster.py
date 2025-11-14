import random
from node import Node
import config
from metrics import metrics


class Cluster:
    """Manages nodes and chunk placement"""
    
    def __init__(self, env):
        self.env = env
        self.nodes = [Node(env, i) for i in range(config.NUM_NODES)]
        self.chunk_map = {}
        
        # Initial chunk distribution (round-robin)
        for i in range(config.NUM_CHUNKS):
            node_id = i % config.NUM_NODES
            self.nodes[node_id].chunks.add(i)
            self.chunk_map[i] = node_id
        
        self.chunks_in_migration = set()
        self.recently_migrated = {}
        self.migration_cooldown = 25  # seconds

    def get_node_for_chunk(self, chunk_id):
        """Get the node currently responsible for a chunk"""
        return self.nodes[self.chunk_map[chunk_id]]

    def get_node_loads(self):
        """Get current load (queue depth) for all nodes"""
        return {n.node_id: n.get_load() for n in self.nodes}

    def get_least_loaded_node(self):
        """Find the node with minimum load, breaking ties randomly"""
        loads = self.get_node_loads()
        min_load = min(loads.values())
        min_nodes = [node_id for node_id, load in loads.items() 
                    if load == min_load]
        return self.nodes[random.choice(min_nodes)]

    def can_migrate_chunk(self, chunk_id):
        """Check if a chunk is eligible for migration"""
        if chunk_id in self.chunks_in_migration:
            return False
        
        if chunk_id in self.recently_migrated:
            if self.env.now - self.recently_migrated[chunk_id] < self.migration_cooldown:
                return False
        
        return True

    # --- NEW: Find hottest migratable chunk on a node ---
    def get_hottest_chunk(self, node_id):
        """
        Finds the chunk on the given node that has the highest
        request count and is available for migration.
        """
        node = self.nodes[node_id]
        available_chunks = [c for c in node.chunks if self.can_migrate_chunk(c)]
        
        if not available_chunks:
            return None

        # Sort available chunks by request count (descending)
        hottest_chunk = max(
            available_chunks, 
            key=lambda chunk_id: node.chunk_request_counts[chunk_id]
        )
        return hottest_chunk

    def migrate_chunk(self, chunk_id, from_node, to_node, balancer_type):
        """Migrate a chunk from one node to another"""
        if from_node.node_id == to_node.node_id:
            return
        if chunk_id not in from_node.chunks:
            return

        metrics.record_migration(balancer_type)
        self.chunks_in_migration.add(chunk_id)

        yield self.env.timeout(config.MIGRATION_TIME)

        if chunk_id in from_node.chunks:
            from_node.chunks.remove(chunk_id)
            to_node.chunks.add(chunk_id)
            self.chunk_map[chunk_id] = to_node.node_id
            self.recently_migrated[chunk_id] = self.env.now
            
            # --- NEW: Reset request count on old node ---
            # This helps new "hottest" chunks to be identified
            from_node.chunk_request_counts[chunk_id] = 0

        if chunk_id in self.chunks_in_migration:
            self.chunks_in_migration.remove(chunk_id)