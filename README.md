# Adaptive Load Balancing in Distributed Database Systems (PRO-LBR)

A discrete-event simulation comparing **Reactive** and **Proactive (Q-Learning)** load balancing strategies for distributed database systems with dynamic workload patterns.

## 🎯 Overview

This project implements and compares two load balancing approaches for distributed databases:

1. **Reactive Balancer**: Traditional threshold-based approach that reacts to imbalances after they occur
2. **PRO-LBR (Proactive Load Balancer)**: Q-learning agent that learns to anticipate and prevent imbalances

The simulation models a realistic distributed database scenario with:
- Multiple nodes processing requests
- Chunk-based data partitioning
- Shifting hotspot workload patterns
- Request queuing and processing delays
- Chunk migration overhead

## ✨ Features

- **Discrete-Event Simulation**: Built using SimPy for accurate modeling
- **Q-Learning Agent**: Reinforcement learning-based proactive balancing
- **Realistic Workload**: Shifting hotspots simulating real-world traffic patterns
- **Comprehensive Metrics**: p50, p95, p99, p99.9 latency tracking
- **Load Imbalance Analysis**: Queue depth monitoring across nodes
- **Migration Tracking**: Records and analyzes chunk migrations
- **Modular Design**: Clean separation of concerns for easy extension

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Simulation Environment                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐        ┌──────────────┐                  │
│  │   Workload   │───────▶│   Cluster    │                  │
│  │  Generator   │        │  (4 Nodes)   │                  │
│  └──────────────┘        └──────┬───────┘                  │
│        │                         │                          │
│        │                         │                          │
│        ▼                         ▼                          │
│  ┌──────────────────────────────────────┐                  │
│  │         Load Balancer                │                  │
│  │  ┌─────────────┐  ┌───────────────┐ │                  │
│  │  │  Reactive   │  │  Proactive    │ │                  │
│  │  │  (Threshold)│  │  (Q-Learning) │ │                  │
│  │  └─────────────┘  └───────────────┘ │                  │
│  └──────────────────────────────────────┘                  │
│                         │                                   │
│                         ▼                                   │
│                  ┌─────────────┐                           │
│                  │   Metrics   │                           │
│                  │  Collector  │                           │
│                  └─────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Adaptive-Load-Balancing-in-DDS.git
   cd Adaptive-Load-Balancing-in-DDS
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install simpy numpy
   ```

## 💻 Usage

### Basic Usage

Run the simulation with default parameters:

```bash
python main.py
```

This will:
1. Run the reactive balancer simulation
2. Run the proactive (Q-learning) balancer simulation
3. Display comparative results


## 📊 Results

### Key Metrics

The simulation tracks and reports:

- **Latency Percentiles**: p50, p95, p99, p99.9
- **Steady-State Performance**: p99 after initial warmup (t > 80s)
- **Load Imbalance**: Maximum queue depth difference across nodes
- **Migration Count**: Total chunk migrations performed
- **Throughput**: Completed requests per second

### Why PRO-LBR Performs Better

1. **Anticipatory Actions**: Learns to migrate before severe imbalance occurs
2. **Faster Response**: Checks every 10s vs 45s for reactive
3. **Adaptive Policy**: Learns optimal migration timing through experience
4. **Multiple Migrations**: Can migrate up to 2 chunks per cycle when needed

## 📁 Project Structure

```
Adaptive-Load-Balancing-in-DDS/
│
├── config.py              # Configuration parameters
├── metrics.py             # Metrics collection and reporting
├── node.py               # Node implementation
├── cluster.py            # Cluster management and migrations
├── workload.py           # Workload generation with hotspots
├── balancers.py          # Reactive and Proactive balancers
├── main.py               # Main simulation runner
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

### Module Descriptions

- **config.py**: Centralized configuration for all simulation parameters
- **metrics.py**: Collects response times, migrations, and generates reports
- **node.py**: Models individual database nodes with queuing
- **cluster.py**: Manages nodes, chunk placement, and migration logic
- **workload.py**: Generates realistic traffic with shifting hotspots
- **balancers.py**: Implements both balancing strategies
- **main.py**: Orchestrates simulations and comparison

## 🔍 How It Works

### Workload Pattern

The simulation generates a realistic workload with **shifting hotspots**:

```
Time      Hotspot Location    Traffic Distribution
─────────────────────────────────────────────────────
0-150s    Node 0 (5 chunks)   35% → hot chunks
                              65% → uniform
150-300s  Node 1 (5 chunks)   35% → hot chunks
                              65% → uniform
300-400s  Node 2 (5 chunks)   35% → hot chunks
                              65% → uniform
```

This simulates real-world scenarios like:
- Geographic traffic patterns (day/night cycles)
- Trending content (viral posts, popular products)
- Batch processing jobs

### Reactive Balancer Algorithm

```python
Every 45 seconds:
  1. Measure load on all nodes
  2. Calculate max_load - min_load
  3. If difference > 6:
     - Select random chunk from max-loaded node
     - Migrate to min-loaded node
```

**Limitations**:
- Reacts only after imbalance occurs
- Slow response (45s interval)
- May miss short-lived imbalances

### Proactive Balancer (Q-Learning) Algorithm

```python
State: (load_level, imbalance_level)
Actions: [do_nothing, migrate]

Every 10 seconds:
  1. Observe current state
  2. Calculate reward:
     - Penalize imbalance (-1 per request)
     - Reward low latency (+10 if p99 < 15ms)
     - Penalize high latency (-10 if p99 > 50ms)
  3. Update Q-table: Q(s,a) += α[r + γ·max Q(s',a') - Q(s,a)]
  4. Choose action (ε-greedy):
     - Explore: random action (25%)
     - Exploit: argmax Q(s,a) (75%)
  5. If migrate:
     - Move up to 2 chunks from max to min node
     - Incur migration cost penalty
```

**Advantages**:
- Learns from experience
- Anticipates imbalances
- Faster decision cycle
- Considers both latency and balance

### State Space

The Q-learning agent discretizes the continuous state space:

**Load Level** (max queue depth):
- Level 1: ≤ 3 requests (low)
- Level 2: 4-7 requests (medium)
- Level 3: 8-12 requests (high)
- Level 4: > 12 requests (very high)

**Imbalance Level** (max - min queue depth):
- Level 1: ≤ 2 requests (balanced)
- Level 2: 3-6 requests (moderate)
- Level 3: > 6 requests (severe)

**Total States**: 4 × 3 = 12 states

## 📚 References

1. **Reinforcement Learning**: Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*
2. **Load Balancing**: Cardellini, V., et al. (2002). "Dynamic Load Balancing on Web-server Systems"
3. **Distributed Databases**: Özsu, M. T., & Valduriez, P. (2011). *Principles of Distributed Database Systems*
4. **SimPy Documentation**: https://simpy.readthedocs.io/

**⭐ Star this repository if you find it helpful!**