"""
Unit tests for Monte Carlo Discrete-Event Queue Simulation
"""
import pytest
from canteen_queue.module1_probabilistic.monte_carlo import (
    MonteCarloQueueSimulator,
    run_canteen_monte_carlo,
)
from canteen_queue.module1_probabilistic.queue_theory import QueueTheoryMMC


def test_monte_carlo_convergence_to_analytic_erlang_c():
    """
    Simulates N=10,000 customers under stationary M/M/2 conditions:
    Verifies empirical waiting time in queue matches theoretical Erlang-C
    within statistical sampling error (< 15% relative or < 0.20 min absolute).
    """
    lam = 1.2  # cust/min
    mu = 1.0   # cust/min/server
    c = 2      # servers
    # rho = 1.2 / (2 * 1.0) = 0.60

    analytic = QueueTheoryMMC(arrival_rate=lam, service_rate=mu, servers=c).compute_metrics()
    expected_wq = analytic.wq_minutes

    res = run_canteen_monte_carlo(
        arrival_rate=lam,
        service_rate=mu,
        servers=c,
        num_runs=10000,
        seed=42,
    )

    assert res.num_customers == 10000
    assert len(res.simulated_wait_times) == 9000  # 10% warm-up removed
    assert res.absolute_error_wq_min < 0.20  # absolute error within 0.20 minutes
    assert res.percentage_error_wq < 15.0   # statistical relative tolerance


def test_monte_carlo_percentiles_order():
    """
    Verifies monotonicity of simulated percentiles: P50 <= P90 <= P95.
    """
    res = run_canteen_monte_carlo(
        arrival_rate=1.2,
        service_rate=1.5,
        servers=1,
        num_runs=5000,
        seed=123,
    )
    assert res.p50_wait_min <= res.p90_wait_min <= res.p95_wait_min
    assert 0.0 <= res.prob_wait_gt_5min <= 1.0
    assert 0.0 <= res.prob_wait_gt_10min <= res.prob_wait_gt_5min
