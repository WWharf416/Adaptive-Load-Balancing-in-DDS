import random
import config


def workload_generator(env, cluster):
    """
    Generate realistic workload with concentrated, shifting hotspots.
    """
    print("  Workload: 35% hotspot concentration (CONCENTRATED), "
          "shifts every 150s")

    hot_chunks_p1 = [0, 4, 8, 12, 16]   # All on Node 0
    hot_chunks_p2 = [1, 5, 9, 13, 17]   # All on Node 1
    hot_chunks_p3 = [2, 6, 10, 14, 18]  # All on Node 2

    while True:
        t = env.now

        if t < 150:
            hot_list = hot_chunks_p1
        elif t < 300:
            hot_list = hot_chunks_p2
        else:
            hot_list = hot_chunks_p3

        if random.random() < 0.35:
            chunk_id = random.choice(hot_list)
        else:
            chunk_id = random.randint(0, config.NUM_CHUNKS - 1)

        node = cluster.get_node_for_chunk(chunk_id)
        
        # --- MODIFIED: Pass chunk_id to node for tracking ---
        env.process(node.process_request(chunk_id))

        yield env.timeout(1.0 / config.REQUEST_RATE)