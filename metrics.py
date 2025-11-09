import numpy as np


class MetricsCollector:
    """Collects and reports simulation metrics"""
    
    def __init__(self):
        self.response_times = []
        self.migrations_reactive = 0
        self.migrations_proactive = 0
        self.p99_samples = []
        self.load_samples = []
        self.last_log_time = 0
        self.final_report = {}

    def record_response(self, time):
        """Record a request response time"""
        self.response_times.append(time)

    def record_migration(self, balancer_type):
        """Record a chunk migration"""
        if balancer_type == 'reactive':
            self.migrations_reactive += 1
        elif balancer_type == 'proactive':
            self.migrations_proactive += 1

    def log_metrics(self, env, balancer_type, cluster):
        """Periodically log system metrics"""
        recent = self.response_times[-1500:]
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
            migs = (self.migrations_reactive if balancer_type == 'reactive' 
                   else self.migrations_proactive)
            print(f"  t={env.now:.0f}s | p99={p99:.1f}ms | "
                  f"Q: max={max_q} min={min_q} avg={avg_q:.1f} | Migs={migs}")
            self.last_log_time = env.now

    def reset(self):
        """Reset metrics for a new simulation run"""
        self.response_times = []
        self.migrations_reactive = 0
        self.migrations_proactive = 0
        self.p99_samples = []
        self.load_samples = []
        self.last_log_time = 0

    def report(self, balancer_type):
        """Generate final report for a simulation run"""
        print(f"\n{'='*65}")
        print(f"FINAL RESULTS: {balancer_type.upper()}")
        print(f"{'='*65}")

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
        migrations = (self.migrations_reactive if balancer_type == 'reactive' 
                     else self.migrations_proactive)

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
            imbalances = [max(l.values()) - min(l.values()) 
                         for l in self.load_samples if l]
            if imbalances:
                print(f"Avg Load Imbalance:  {np.mean(imbalances):.1f} requests")
                print(f"Max Load Imbalance:  {np.max(imbalances):.1f} requests")

        report_data['requests'] = len(self.response_times)
        report_data['migrations'] = migrations
        report_data['p99'] = p99
        report_data['steady_p99'] = steady_p99_val


# Shared singleton instance
metrics = MetricsCollector()