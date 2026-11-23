"""
Unit tests for Markov Matrix Spectral Analysis & Manual Power Iteration
"""
import numpy as np
import pytest
from canteen_queue.data.generate_data import load_or_generate_data
from canteen_queue.module1_probabilistic.markov_chain import estimate_transition_matrix
from canteen_queue.module2_linear_algebra.markov_matrix_analysis import (
    MarkovSpectralAnalysis,
    manual_power_iteration,
    perform_spectral_analysis,
)


@pytest.fixture(scope="module")
def dataset():
    return load_or_generate_data()


def test_power_iteration_convergence_and_invariance(dataset):
    """
    Validates that the stationary vector pi obtained via manual power iteration:
    1. Sums to 1.0 (valid probability distribution)
    2. Has all non-negative entries
    3. Is invariant under P: pi * P = pi
    """
    P = estimate_transition_matrix(dataset, counter="Meals")
    pi, num_iter, errors = manual_power_iteration(P, tol=1e-11)

    assert np.isclose(np.sum(pi), 1.0, atol=1e-8)
    assert np.all(pi >= 0.0)
    # Stationarity invariance: pi * P = pi
    pi_next = pi @ P
    assert np.allclose(pi_next, pi, atol=1e-8)


def test_power_iteration_vs_numpy_eig_agreement(dataset):
    """
    Key Prompt Requirement:
    Verify manual power iteration output against exact numpy.linalg.eig left eigenvector.
    """
    P = estimate_transition_matrix(dataset, counter="Snacks")
    res = perform_spectral_analysis(P, "Snacks")

    # Spectral radius must be 1.0 by Perron-Frobenius theorem
    assert res.spectral_radius == pytest.approx(1.0, abs=1e-5)
    # SLEM < 1.0
    assert res.second_largest_modulus < 1.0
    assert res.spectral_gap > 0.0

    # High precision agreement between power iteration and eig
    assert res.agreement_error < 1e-4
