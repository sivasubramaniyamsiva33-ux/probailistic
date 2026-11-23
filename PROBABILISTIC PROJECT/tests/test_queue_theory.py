"""
Unit tests for Queue Theory (M/M/1, M/M/c, Erlang-C, and Little's Law)
"""
import pytest
from canteen_queue.module1_probabilistic.queue_theory import (
    QueueTheoryMMC,
    check_littles_law,
    predict_wait_time,
)


def test_erlang_c_against_known_textbook_values():
    """
    Standard textbook queueing verification:
    c = 2 servers, lambda = 4.0 cust/min, mu = 3.0 cust/min/server
    r = 4/3, rho = 4 / (2 * 3) = 2/3.
    Exact Textbook Analytical Values:
        P0 = 0.20 (1/5)
        P_C (Erlang-C wait probability) = 8/15 ~= 0.53333
        L_q (mean queue length) = 16/15 ~= 1.06667
        W_q (mean waiting time in queue) = 4/15 ~= 0.26667 min
        W (mean total system time) = 3/5 = 0.60 min
        L (mean customers in system) = 2.40
    """
    system = QueueTheoryMMC(arrival_rate=4.0, service_rate=3.0, servers=2)
    metrics = system.compute_metrics()

    assert metrics.is_stable is True
    assert metrics.utilization_rho == pytest.approx(2.0 / 3.0, rel=1e-4)
    assert metrics.p0 == pytest.approx(0.20, abs=1e-4)
    assert metrics.erlang_c_prob == pytest.approx(8.0 / 15.0, abs=1e-4)
    assert metrics.lq == pytest.approx(16.0 / 15.0, abs=1e-4)
    assert metrics.wq_minutes == pytest.approx(4.0 / 15.0, abs=1e-4)
    assert metrics.w_minutes == pytest.approx(0.60, abs=1e-4)
    assert metrics.l_total == pytest.approx(2.40, abs=1e-4)


def test_littles_law_exact_consistency():
    """
    Validates Little's Law conservation identity:
        L = lambda * W
        L_q = lambda * W_q
    across various server counts and load conditions to tolerance < 1e-6.
    """
    test_cases = [
        (1.5, 2.0, 1),
        (3.0, 2.5, 2),
        (5.0, 2.0, 3),
        (8.0, 3.0, 4),
    ]

    for lam, mu, c in test_cases:
        system = QueueTheoryMMC(arrival_rate=lam, service_rate=mu, servers=c)
        metrics = system.compute_metrics()

        assert metrics.is_stable is True
        assert check_littles_law(metrics, tolerance=1e-6) is True
        assert metrics.littles_law_error < 1e-6
        assert metrics.littles_law_q_error < 1e-6


def test_mm1_exact_closed_form():
    """
    For c = 1 server, M/M/1 closed form:
        P0 = 1 - rho
        Lq = rho^2 / (1 - rho)
        Wq = rho / (mu - lambda)
        W = 1 / (mu - lambda)
    """
    lam = 1.2
    mu = 2.0
    rho = lam / mu  # 0.6

    system = QueueTheoryMMC(arrival_rate=lam, service_rate=mu, servers=1)
    metrics = system.compute_metrics()

    expected_p0 = 1.0 - rho
    expected_lq = (rho ** 2) / (1.0 - rho)
    expected_wq = rho / (mu - lam)
    expected_w = 1.0 / (mu - lam)

    assert metrics.p0 == pytest.approx(expected_p0, abs=1e-4)
    assert metrics.lq == pytest.approx(expected_lq, abs=1e-4)
    assert metrics.wq_minutes == pytest.approx(expected_wq, abs=1e-4)
    assert metrics.w_minutes == pytest.approx(expected_w, abs=1e-4)


def test_unstable_queue_handling():
    """
    When arrival rate exceeds system capacity (rho >= 1), the system must report
    is_stable = False and infinite wait time.
    """
    system = QueueTheoryMMC(arrival_rate=5.0, service_rate=2.0, servers=2)
    metrics = system.compute_metrics()
    assert metrics.is_stable is False
    assert metrics.utilization_rho >= 1.0
    assert metrics.wq_minutes == float("inf")


def test_predict_wait_time_function():
    """
    Tests wait-time prediction function with observed queue length:
    Returns expected wait, 90% credible bounds, and correct traffic light.
    """
    pred = predict_wait_time(
        counter="Meals",
        time_slot="12:30",
        current_queue_length=6,
        servers=2,
        default_mu=1.0,
        default_lam=1.5,
    )
    assert "expected_total_wait_min" in pred
    assert "ci_90_lower_min" in pred
    assert "ci_90_upper_min" in pred
    assert pred["ci_90_lower_min"] <= pred["expected_total_wait_min"] <= pred["ci_90_upper_min"]
    assert pred["traffic_light"] in ["Green", "Amber", "Red"]
