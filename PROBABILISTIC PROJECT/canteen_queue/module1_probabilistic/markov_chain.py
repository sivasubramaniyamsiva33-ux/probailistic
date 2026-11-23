"""
Discrete-Time Markov Chain Engine on Canteen Queue Congestion States
Saveetha College Canteen Queue Assistant
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from canteen_queue.config import QUEUE_STATE_BOUNDS, QUEUE_STATE_NAMES, SLOT_MINUTES


def queue_length_to_state(queue_len: int) -> int:
    """
    Maps continuous/integer queue length to discrete state index (0 to 4):
    0: Empty (0-1)
    1: Short (2-4)
    2: Medium (5-8)
    3: Long (9-14)
    4: Overcrowded (15+)
    """
    q = max(0, int(queue_len))
    for idx, (low, high) in enumerate(QUEUE_STATE_BOUNDS):
        if low <= q <= high:
            return idx
    return len(QUEUE_STATE_BOUNDS) - 1


def estimate_transition_matrix(
    df: pd.DataFrame,
    counter: Optional[str] = None,
    laplace_smoothing: float = 1e-4,
) -> np.ndarray:
    """
    Estimates the 5x5 transition probability matrix P from consecutive slot transitions:
    P_ij = P(X_{t+1} = j | X_t = i)

    Formulation:
    Count transitions N_ij between consecutive 5-min slots within the same day and counter.
    Add Laplace smoothing epsilon to prevent zero rows:
        P_ij = (N_ij + epsilon) / sum_k (N_ik + epsilon)
    Guarantees sum_j P_ij = 1.0 (stochastic matrix).
    """
    sub_df = df.copy()
    if counter is not None:
        sub_df = sub_df[sub_df["counter"] == counter].copy()

    # Map queue lengths to states
    sub_df["state"] = sub_df["queue_length_at_arrival"].apply(queue_length_to_state)

    # Sort sequentially
    sub_df = sub_df.sort_values(by=["day_index", "counter", "slot_index"])

    num_states = len(QUEUE_STATE_NAMES)
    counts = np.zeros((num_states, num_states), dtype=float)

    # Group by day and counter so transitions do not jump across days or counters
    for _, group in sub_df.groupby(["day_index", "counter"]):
        states = group["state"].to_numpy()
        for i in range(len(states) - 1):
            s_from = states[i]
            s_to = states[i + 1]
            counts[s_from, s_to] += 1.0

    # Add Laplace smoothing
    counts += laplace_smoothing

    # Normalize rows to form stochastic matrix
    row_sums = counts.sum(axis=1, keepdims=True)
    transition_matrix = counts / row_sums

    return transition_matrix


@dataclass
class MarkovForecastResult:
    counter: str
    current_state_idx: int
    current_state_name: str
    n_steps: int
    step_duration_minutes: int
    forecast_distribution: np.ndarray       # Probability vector at step n
    state_names: List[str]
    expected_steps_to_overcrowded: float    # Expected steps from current state
    expected_minutes_to_overcrowded: float  # Expected minutes
    transition_matrix: np.ndarray


class QueueMarkovChain:
    """
    Analyzes discrete-time Markov chain for queue states, n-step forecasting,
    and first hitting times to the 'Overcrowded' state.
    """

    def __init__(self, df: pd.DataFrame, counter: str):
        self.df = df
        self.counter = counter
        self.state_names = QUEUE_STATE_NAMES
        self.P = estimate_transition_matrix(df, counter=counter)
        self.fundamental_matrix, self.hitting_times_steps = self._compute_first_hitting_times()

    def _compute_first_hitting_times(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes expected steps to reach 'Overcrowded' (State 4) via absorbing Markov chain theory.

        Formulation:
        States {0, 1, 2, 3} are transient; State 4 is absorbing.
        Canonical form:
        P = [ Q   R ]
            [ 0   1 ]
        Fundamental matrix: N = (I - Q)^(-1)
        Expected steps to absorption: h = N * 1
        """
        # Transient submatrix for states 0, 1, 2, 3
        Q = self.P[:4, :4]
        I = np.eye(4)

        try:
            # Invert (I - Q)
            N = np.linalg.inv(I - Q)
            ones = np.ones(4)
            h = N @ ones  # expected steps from states 0, 1, 2, 3
        except np.linalg.LinAlgError:
            N = np.eye(4)
            h = np.array([20.0, 15.0, 10.0, 5.0])

        # State 4 is already overcrowded -> 0 steps to reach itself
        hitting_steps = np.append(h, 0.0)
        return N, hitting_steps

    def forecast_n_steps(self, current_queue: int, n_steps: int = 6) -> MarkovForecastResult:
        """
        Forecasts queue state probability distribution after n steps (5 min each).
        pi^(n) = pi^(0) * P^n
        """
        cur_idx = queue_length_to_state(current_queue)
        pi_0 = np.zeros(len(self.state_names))
        pi_0[cur_idx] = 1.0

        # Matrix power P^n
        P_n = np.linalg.matrix_power(self.P, n_steps)
        pi_n = pi_0 @ P_n

        steps_to_oc = float(self.hitting_times_steps[cur_idx])
        mins_to_oc = steps_to_oc * SLOT_MINUTES

        return MarkovForecastResult(
            counter=self.counter,
            current_state_idx=cur_idx,
            current_state_name=self.state_names[cur_idx],
            n_steps=n_steps,
            step_duration_minutes=SLOT_MINUTES,
            forecast_distribution=pi_n,
            state_names=self.state_names,
            expected_steps_to_overcrowded=steps_to_oc,
            expected_minutes_to_overcrowded=mins_to_oc,
            transition_matrix=self.P,
        )

    def compute_multi_horizon_forecast(
        self, current_queue: int, max_steps: int = 12
    ) -> pd.DataFrame:
        """
        Generates probability trajectory for all states across 1 to max_steps ahead.
        """
        cur_idx = queue_length_to_state(current_queue)
        pi_0 = np.zeros(len(self.state_names))
        pi_0[cur_idx] = 1.0

        rows = []
        cur_pi = pi_0.copy()
        for step in range(1, max_steps + 1):
            cur_pi = cur_pi @ self.P
            row = {"Step": step, "Minutes_Ahead": step * SLOT_MINUTES}
            for state_idx, name in enumerate(self.state_names):
                row[name] = round(float(cur_pi[state_idx]), 4)
            rows.append(row)

        return pd.DataFrame(rows)
