"""
Queueing Theory Engine: Pure From-Scratch M/M/1 & M/M/c Implementation
Saveetha College Canteen Queue Assistant
"""
from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class QueueMetrics:
    arrival_rate_per_min: float  # lambda
    service_rate_per_min: float  # mu
    servers: int                 # c
    traffic_intensity: float     # r = lambda / mu
    utilization_rho: float       # rho = lambda / (c * mu)
    is_stable: bool              # rho < 1
    p0: float                    # P0 probability of empty system
    erlang_c_prob: float         # P_C = P(Wait > 0)
    lq: float                    # Expected queue length L_q
    wq_minutes: float            # Expected waiting time in queue W_q
    w_minutes: float             # Expected total system time W = W_q + 1/mu
    l_total: float               # Expected number of customers in system L
    littles_law_error: float     # |L - lambda * W|
    littles_law_q_error: float   # |L_q - lambda * W_q|


def erlang_c_formula(c: int, r: float, rho: float) -> Tuple[float, float]:
    """
    Computes P0 and the Erlang-C probability of delay from first principles.

    Formulas:
        P0 = [ sum_{n=0}^{c-1} (r^n / n!) + (r^c / (c! * (1 - rho))) ]^(-1)
        P_C = P0 * [ r^c / (c! * (1 - rho)) ]
    """
    if rho >= 1.0:
        return 0.0, 1.0

    # Finite sum for n = 0 to c - 1
    sum_terms = 0.0
    for n in range(c):
        sum_terms += (r ** n) / math.factorial(n)

    # Queueing delay term at capacity
    delay_term = (r ** c) / (math.factorial(c) * (1.0 - rho))

    p0 = 1.0 / (sum_terms + delay_term)
    erlang_c_prob = p0 * delay_term
    return p0, erlang_c_prob


class QueueTheoryMMC:
    """
    Analytic M/M/c queueing system implemented strictly from first principles.
    Does not use external queueing packages.
    """

    def __init__(self, arrival_rate: float, service_rate: float, servers: int = 1):
        """
        Args:
            arrival_rate: lambda (customers per minute)
            service_rate: mu (customers per server per minute)
            servers: c (number of parallel servers)
        """
        if arrival_rate <= 0 or service_rate <= 0 or servers < 1:
            raise ValueError("Parameters lambda > 0, mu > 0, and c >= 1 are required.")

        self.lam = float(arrival_rate)
        self.mu = float(service_rate)
        self.c = int(servers)
        self.r = self.lam / self.mu
        self.rho = self.lam / (self.c * self.mu)

    def compute_metrics(self) -> QueueMetrics:
        """
        Evaluates exact steady-state queue performance metrics and verifies Little's Law.
        """
        if self.rho >= 1.0:
            # Unstable queue (demand exceeds capacity)
            return QueueMetrics(
                arrival_rate_per_min=self.lam,
                service_rate_per_min=self.mu,
                servers=self.c,
                traffic_intensity=round(self.r, 4),
                utilization_rho=round(self.rho, 4),
                is_stable=False,
                p0=0.0,
                erlang_c_prob=1.0,
                lq=float("inf"),
                wq_minutes=float("inf"),
                w_minutes=float("inf"),
                l_total=float("inf"),
                littles_law_error=0.0,
                littles_law_q_error=0.0,
            )

        if self.c == 1:
            # Exact M/M/1 Closed Form
            p0 = 1.0 - self.rho
            erlang_c = self.rho
            lq = (self.rho ** 2) / (1.0 - self.rho)
            wq = lq / self.lam
            w = wq + (1.0 / self.mu)
            l = self.lam * w
        else:
            # Exact M/M/c Formula
            p0, erlang_c = erlang_c_formula(self.c, self.r, self.rho)
            lq = erlang_c * (self.rho / (1.0 - self.rho))
            wq = lq / self.lam
            w = wq + (1.0 / self.mu)
            l = lq + self.r

        # Little's Law consistency checks: L = lambda * W and L_q = lambda * W_q
        ll_error = abs(l - (self.lam * w))
        ll_q_error = abs(lq - (self.lam * wq))

        return QueueMetrics(
            arrival_rate_per_min=round(self.lam, 4),
            service_rate_per_min=round(self.mu, 4),
            servers=self.c,
            traffic_intensity=round(self.r, 4),
            utilization_rho=round(self.rho, 4),
            is_stable=True,
            p0=round(p0, 5),
            erlang_c_prob=round(erlang_c, 5),
            lq=round(lq, 4),
            wq_minutes=round(wq, 4),
            w_minutes=round(w, 4),
            l_total=round(l, 4),
            littles_law_error=float(ll_error),
            littles_law_q_error=float(ll_q_error),
        )


def check_littles_law(metrics: QueueMetrics, tolerance: float = 1e-6) -> bool:
    """
    Validates Little's Law conservation identity:
        L = lambda * W
        L_q = lambda * W_q
    """
    if not metrics.is_stable:
        return True
    return bool(metrics.littles_law_error < tolerance and metrics.littles_law_q_error < tolerance)


def predict_wait_time(
    counter: str,
    time_slot: str,
    current_queue_length: int,
    servers: Optional[int] = None,
    df: Optional[pd.DataFrame] = None,
    default_mu: Optional[float] = None,
    default_lam: Optional[float] = None,
) -> Dict[str, float]:
    """
    Predicts expected waiting time and 90% confidence/credible interval for a customer
    arriving at a counter with an observed queue length.

    Mathematical Formulation:
    If a customer arrives and observes Q customers in line ahead with c servers:
    1. If Q < c, an idle server is available immediately; wait in line is 0,
       total time = customer's own service time ~ Exp(mu).
    2. If Q >= c, the newcomer waits for (Q - c + 1) service completions from
       the c active servers. Each completion has rate (c * mu).
       Wait in queue W_q ~ Erlang(k = Q - c + 1, rate = c * mu).
       Total wait W = W_q + S, where S ~ Exp(mu) is own service.
       E[W | Q] = (Q - c + 1) / (c * mu) + (1 / mu)
       Var[W | Q] = (Q - c + 1) / (c * mu)^2 + (1 / mu)^2
    """
    from canteen_queue.config import DEFAULT_SERVERS, DEFAULT_SERVICE_RATES

    c = servers if servers is not None else DEFAULT_SERVERS.get(counter, 1)
    mu = default_mu if default_mu is not None else DEFAULT_SERVICE_RATES.get(counter, 1.5)

    # Estimate local arrival rate lambda from historical data if provided
    if df is not None and default_lam is None:
        sub = df[(df["counter"] == counter) & (df["time_slot"] == time_slot)]
        if len(sub) > 0:
            mean_arrivals_5min = float(sub["arrival_count"].mean())
            lam = mean_arrivals_5min / 5.0  # arrivals per minute
        else:
            lam = 0.8
    else:
        lam = default_lam if default_lam is not None else 0.8

    Q = max(0, int(current_queue_length))

    if Q < c:
        # No queue wait; only own service
        expected_wq = 0.0
        expected_total = 1.0 / mu
        # 90% interval for Exp(mu): [F^-1(0.05), F^-1(0.95)]
        lower_bound = float(stats.expon.ppf(0.05, scale=1.0 / mu))
        upper_bound = float(stats.expon.ppf(0.95, scale=1.0 / mu))
    else:
        k_steps = Q - c + 1
        rate_pool = c * mu
        expected_wq = k_steps / rate_pool
        expected_total = expected_wq + (1.0 / mu)
        var_total = (k_steps / (rate_pool ** 2)) + (1.0 / (mu ** 2))

        # Match moments to Gamma distribution Gamma(alpha_eff, scale_eff)
        alpha_eff = (expected_total ** 2) / max(1e-6, var_total)
        scale_eff = var_total / max(1e-6, expected_total)
        lower_bound = float(stats.gamma.ppf(0.05, a=alpha_eff, scale=scale_eff))
        upper_bound = float(stats.gamma.ppf(0.95, a=alpha_eff, scale=scale_eff))

    # Steady state M/M/c metrics for system reference
    steady_metrics = QueueTheoryMMC(arrival_rate=lam, service_rate=mu, servers=c).compute_metrics()

    return {
        "counter": counter,
        "time_slot": time_slot,
        "current_queue_length": Q,
        "servers": c,
        "arrival_rate_per_min": round(lam, 3),
        "service_rate_per_min": round(mu, 3),
        "expected_queue_wait_min": round(expected_wq, 2),
        "expected_total_wait_min": round(expected_total, 2),
        "ci_90_lower_min": round(max(0.1, lower_bound), 2),
        "ci_90_upper_min": round(upper_bound, 2),
        "steady_state_wq_min": steady_metrics.wq_minutes if steady_metrics.is_stable else 999.0,
        "utilization_rho": steady_metrics.utilization_rho,
        "traffic_light": "Green" if expected_total < 5.0 else ("Amber" if expected_total <= 12.0 else "Red"),
    }
