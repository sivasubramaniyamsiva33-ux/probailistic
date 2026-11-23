"""
Unit tests for Discrete-Time Markov Chain Engine
"""
import numpy as np
import pytest
from canteen_queue.data.generate_data import load_or_generate_data
from canteen_queue.module1_probabilistic.markov_chain import (
    QueueMarkovChain,
    estimate_transition_matrix,
    queue_length_to_state,
)


@pytest.fixture(scope="module")
def dataset():
    return load_or_generate_data()


def test_queue_state_mapping():
    """
    Validates state mapping bounds:
    0: Empty (0-1), 1: Short (2-4), 2: Medium (5-8), 3: Long (9-14), 4: Overcrowded (15+)
    """
    assert queue_length_to_state(0) == 0
    assert queue_length_to_state(1) == 0
    assert queue_length_to_state(2) == 1
    assert queue_length_to_state(4) == 1
    assert queue_length_to_state(5) == 2
    assert queue_length_to_state(8) == 2
    assert queue_length_to_state(9) == 3
    assert queue_length_to_state(14) == 3
    assert queue_length_to_state(15) == 4
    assert queue_length_to_state(50) == 4


def test_markov_transition_matrix_stochastic_property(dataset):
    """
    Verifies that the transition matrix P is a valid row-stochastic matrix:
    1. Every entry P_ij in [0, 1]
    2. Every row sums to 1.0 to machine precision
    """
    for counter in ["Meals", "Snacks", "Beverages"]:
        P = estimate_transition_matrix(dataset, counter=counter)
        assert P.shape == (5, 5)
        assert np.all(P >= 0.0)
        assert np.all(P <= 1.0)
        row_sums = P.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)


def test_markov_n_step_forecast_and_hitting_times(dataset):
    """
    Verifies n-step probability distribution and first hitting times to Overcrowded state.
    """
    mc = QueueMarkovChain(dataset, counter="Meals")
    res = mc.forecast_n_steps(current_queue=3, n_steps=4)

    assert len(res.forecast_distribution) == 5
    assert np.all(res.forecast_distribution >= 0.0)
    assert np.isclose(np.sum(res.forecast_distribution), 1.0, atol=1e-5)
    assert res.expected_steps_to_overcrowded > 0.0
    assert res.expected_minutes_to_overcrowded == pytest.approx(
        res.expected_steps_to_overcrowded * 5.0, abs=1e-4
    )
