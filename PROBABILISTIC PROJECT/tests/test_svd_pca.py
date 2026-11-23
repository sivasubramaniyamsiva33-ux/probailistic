"""
Unit tests for SVD & PCA Demand Patterns & Eckart-Young Theorem
"""
import numpy as np
import pytest
from canteen_queue.data.generate_data import load_or_generate_data
from canteen_queue.module2_linear_algebra.demand_matrix import DemandMatrixBuilder
from canteen_queue.module2_linear_algebra.svd_pca_patterns import (
    SVDDemandPatterns,
    perform_svd_analysis,
)


@pytest.fixture(scope="module")
def dataset():
    return load_or_generate_data()


def test_svd_orthonormality_and_energy(dataset):
    """
    Tests SVD right singular vectors orthonormality: V^T V = I,
    and cumulative energy monotonically reaches 1.0.
    """
    builder = DemandMatrixBuilder(dataset)
    D = builder.build_day_slot_matrix(counter="Meals")
    res = perform_svd_analysis(D, "Meals")

    V_top3 = res.top_v_patterns
    VtV = V_top3.T @ V_top3
    assert np.allclose(VtV, np.eye(3), atol=1e-10)

    # Energy monotonically increasing
    assert np.all(np.diff(res.cumulative_energy) >= -1e-10)
    assert 0.0 < res.cumulative_energy[0] <= 1.0
    assert res.components_for_90pct > 0
    assert res.components_for_95pct >= res.components_for_90pct


def test_eckart_young_monotonic_error_decrease(dataset):
    """
    Key Prompt Requirement:
    Verify that low-rank SVD reconstruction Frobenius error strictly decreases
    as rank k increases, proving the Eckart-Young-Mirsky theorem.
    """
    builder = DemandMatrixBuilder(dataset)
    D = builder.build_day_slot_matrix(counter="Snacks")
    res = perform_svd_analysis(D, "Snacks")

    errors = res.reconstruction_errors
    # Monotonically strictly decreasing
    assert np.all(np.diff(errors) < 0.0)

    # Verify exact Eckart-Young formula: ||D - D_k||_F = sqrt(sum_{j=k+1}^r sigma_j^2)
    s = res.singular_values
    for k in [1, 2, 5, 10]:
        expected_err = np.sqrt(np.sum(s[k:] ** 2))
        actual_err = errors[k - 1]
        assert actual_err == pytest.approx(expected_err, rel=1e-5)
