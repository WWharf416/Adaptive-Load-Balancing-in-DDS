import random
from config import REQUEST_RATE, NUM_CHUNKS


def workload_generator(env, cluster):
    """
    Generate realistic workload with concentrated, shifting hotspots:
    - Phase 1 (0-150s):   35% to 5 chunks on Node 0
    - Phase 2 (150-300s): 35% to 5 chunks on Node 1
    - Phase 3 (300-400s): 35% to 5 chunks on Node 2
    - Remaining 65% distributed uniformly across all chunks
    """
    print("  Workload: 35% hotspot concentration (CONCENTRATED), "
          "shifts every 150s")

    # Hot chunks for each phase (guaranteed to be on specific nodes)
    hot_chunks_p1 = [0, 4, 8, 12, 16]   # All on Node 0
    hot_chunks_p2 = [1, 5, 9, 13, 17]   # All on Node 1
    hot_chunks_p3 = [2, 6, 10, 14, 18]  # All on Node 2

    while True:
        t = env.now

        # Determine current hot chunk list based on time
        if t < 150:
            hot_list = hot_chunks_p1
        elif t < 300:
            hot_list = hot_chunks_p2
        else:
            hot_list = hot_chunks_p3

        # 35% to hot chunks, 65% uniform distribution
        if random.random() < 0.35:
            chunk_id = random.choice(hot_list)
        else:
            chunk_id = random.randint(0, NUM_CHUNKS - 1)

        # Route request to appropriate node
        node = cluster.get_node_for_chunk(chunk_id)
        env.process(node.process_request({}))

        # Inter-arrival time based on Poisson process
        yield env.timeout(1.0 / REQUEST_RATE)