"""
Monte Carlo Simulation: Discrete-Event Queueing Engine (N=10,000 Runs)
Saveetha College Canteen Queue Assistant
"""
from dataclasses import dataclass
import heapq
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from canteen_queue.module1_probabilistic.queue_theory import QueueTheoryMMC


@dataclass
class MonteCarloSimulationResult:
    num_customers: int
    arrival_rate: float
    service_rate: float
    servers: int
    # Empirical Metrics
    mean_wait_in_queue_min: float
    std_wait_in_queue_min: float
    mean_total_system_time_min: float
    p50_wait_min: float
    p90_wait_min: float
    p95_wait_min: float
    prob_wait_gt_5min: float
    prob_wait_gt_10min: float
    # Analytic M/M/c Benchmark
    analytic_wq_min: float
    analytic_w_min: float
    analytic_lq: float
    absolute_error_wq_min: float
    percentage_error_wq: float
    simulated_wait_times: np.ndarray


class MonteCarloQueueSimulator:
    """
    Discrete-Event Simulation (DES) of M/M/c (or M/G/c) queueing system.
    Tracks event timeline across parallel servers with priority queue.
    """

    def __init__(
        self,
        arrival_rate: float,
        service_rate: float,
        servers: int = 1,
        service_distribution: str = "exponential",
        gamma_shape: float = 2.5,
        seed: int = 42,
    ):
        self.lam = float(arrival_rate)
        self.mu = float(service_rate)
        self.c = int(servers)
        self.service_dist = service_distribution
        self.gamma_shape = gamma_shape
        self.seed = seed

    def simulate(self, num_customers: int = 10000) -> MonteCarloSimulationResult:
        """
        Runs discrete-event simulation for N customers through c servers.
        Uses min-heap for server availability times.
        """
        np.random.seed(self.seed)

        # 1. Generate customer inter-arrival times ~ Exp(lambda)
        inter_arrivals = np.random.exponential(scale=1.0 / self.lam, size=num_customers)
        arrival_times = np.cumsum(inter_arrivals)

        # 2. Generate customer service durations
        if self.service_dist == "gamma":
            scale = (1.0 / self.mu) / self.gamma_shape
            service_durations = np.random.gamma(shape=self.gamma_shape, scale=scale, size=num_customers)
        else:
            service_durations = np.random.exponential(scale=1.0 / self.mu, size=num_customers)

        # 3. Simulate c parallel servers using a min-heap of free times
        # Initialize heap with c servers available at t = 0
        server_free_times = [0.0] * self.c
        heapq.heapify(server_free_times)

        wait_times_in_queue = np.zeros(num_customers, dtype=float)
        total_system_times = np.zeros(num_customers, dtype=float)

        for i in range(num_customers):
            t_arr = arrival_times[i]
            s_dur = service_durations[i]

            # Earliest available server
            earliest_free = heapq.heappop(server_free_times)

            # Service starts at max(arrival_time, server_free_time)
            service_start = max(t_arr, earliest_free)
            wait_q = service_start - t_arr
            departure = service_start + s_dur

            wait_times_in_queue[i] = wait_q
            total_system_times[i] = wait_q + s_dur

            # Put server back into heap with new departure time
            heapq.heappush(server_free_times, departure)

        # Compute empirical statistics (discard first 10% warm-up period)
        warmup = int(0.10 * num_customers)
        valid_waits = wait_times_in_queue[warmup:]
        valid_totals = total_system_times[warmup:]

        mean_wq = float(np.mean(valid_waits))
        std_wq = float(np.std(valid_waits, ddof=1))
        mean_w = float(np.mean(valid_totals))

        p50 = float(np.percentile(valid_waits, 50))
        p90 = float(np.percentile(valid_waits, 90))
        p95 = float(np.percentile(valid_waits, 95))

        p_gt_5 = float(np.mean(valid_waits > 5.0))
        p_gt_10 = float(np.mean(valid_waits > 10.0))

        # Analytic benchmark
        analytic = QueueTheoryMMC(arrival_rate=self.lam, service_rate=self.mu, servers=self.c).compute_metrics()
        analytic_wq = analytic.wq_minutes if analytic.is_stable else 999.0
        analytic_w = analytic.w_minutes if analytic.is_stable else 999.0
        analytic_lq = analytic.lq if analytic.is_stable else 999.0

        abs_err = abs(mean_wq - analytic_wq) if analytic.is_stable else 0.0
        pct_err = (abs_err / max(1e-4, analytic_wq)) * 100.0 if analytic.is_stable else 0.0

        return MonteCarloSimulationResult(
            num_customers=num_customers,
            arrival_rate=self.lam,
            service_rate=self.mu,
            servers=self.c,
            mean_wait_in_queue_min=round(mean_wq, 3),
            std_wait_in_queue_min=round(std_wq, 3),
            mean_total_system_time_min=round(mean_w, 3),
            p50_wait_min=round(p50, 3),
            p90_wait_min=round(p90, 3),
            p95_wait_min=round(p95, 3),
            prob_wait_gt_5min=round(p_gt_5, 4),
            prob_wait_gt_10min=round(p_gt_10, 4),
            analytic_wq_min=round(analytic_wq, 3),
            analytic_w_min=round(analytic_w, 3),
            analytic_lq=round(analytic_lq, 3),
            absolute_error_wq_min=round(abs_err, 3),
            percentage_error_wq=round(pct_err, 2),
            simulated_wait_times=valid_waits,
        )


def run_canteen_monte_carlo(
    arrival_rate: float,
    service_rate: float,
    servers: int = 1,
    num_runs: int = 10000,
    seed: int = 42,
) -> MonteCarloSimulationResult:
    """
    Convenience function to execute Monte Carlo simulation.
    """
    sim = MonteCarloQueueSimulator(
        arrival_rate=arrival_rate,
        service_rate=service_rate,
        servers=servers,
        seed=seed,
    )
    return sim.simulate(num_customers=num_runs)


def plot_simulation_histogram(
    sim_result: MonteCarloSimulationResult,
    title: str = "Monte Carlo Simulated Waiting Times (N=10,000)",
    save_path: Optional[str] = None,
):
    """
    Plots the empirical waiting time distribution with P50, P90, and P95 markers.
    """
    waits = sim_result.simulated_wait_times
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        waits,
        bins=50,
        density=True,
        alpha=0.6,
        color="#2B2D42",
        edgecolor="white",
        label=f"Simulated Wait Distribution (Mean={sim_result.mean_wait_in_queue_min:.2f}m)",
    )

    # Percentiles
    ax.axvline(
        sim_result.p50_wait_min,
        color="#06D6A0",
        linestyle="--",
        linewidth=2.2,
        label=f"P50 (Median): {sim_result.p50_wait_min:.2f} min",
    )
    ax.axvline(
        sim_result.p90_wait_min,
        color="#FFD166",
        linestyle="--",
        linewidth=2.2,
        label=f"P90: {sim_result.p90_wait_min:.2f} min",
    )
    ax.axvline(
        sim_result.p95_wait_min,
        color="#EF476F",
        linestyle="--",
        linewidth=2.2,
        label=f"P95: {sim_result.p95_wait_min:.2f} min",
    )

    if sim_result.analytic_wq_min < 100:
        ax.axvline(
            sim_result.analytic_wq_min,
            color="#118AB2",
            linestyle="-",
            linewidth=2.5,
            label=f"Analytic Erlang-C Wq: {sim_result.analytic_wq_min:.2f} min",
        )

    ax.set_title(
        f"Saveetha Canteen: {title}\n"
        f"Sim Error vs Erlang-C: {sim_result.absolute_error_wq_min:.2f}m ({sim_result.percentage_error_wq:.1f}%) | "
        f"P(Wait > 5m) = {sim_result.prob_wait_gt_5min*100:.1f}%",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Waiting Time in Queue (minutes)", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    return fig
