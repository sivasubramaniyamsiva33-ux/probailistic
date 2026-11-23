"""
Unit tests for Demand Matrix Construction, Imputation, and Matrix Properties
"""
import numpy as np
import pytest
from canteen_queue.data.generate_data import load_or_generate_data
from canteen_queue.module2_linear_algebra.demand_matrix import DemandMatrixBuilder


@pytest.fixture(scope="module")
def dataset():
    return load_or_generate_data()


def test_demand_matrix_dimensions_and_imputation(dataset):
    """
    Verifies D in R^(60 x 120) with no missing values.
    """
    builder = DemandMatrixBuilder(dataset)
    for c in ["Meals", "Snacks", "Beverages"]:
        D = builder.build_day_slot_matrix(counter=c)
        assert D.shape == (60, 120)
        assert not np.isnan(D).any()
        assert (D >= 0).all()


def test_matrix_normalization_methods():
    """
    Tests mean-centering and z-score standardizations.
    """
    D = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
    D_center, p_c = DemandMatrixBuilder.normalize_matrix(D, method="center")
    assert np.allclose(np.mean(D_center, axis=0), 0.0, atol=1e-6)

    D_z, p_z = DemandMatrixBuilder.normalize_matrix(D, method="zscore")
    assert np.allclose(np.mean(D_z, axis=0), 0.0, atol=1e-6)
    assert np.allclose(np.std(D_z, axis=0), 1.0, atol=1e-6)


def test_matrix_structural_properties(dataset):
    """
    Validates rank, condition number, and Frobenius norm calculations.
    """
    builder = DemandMatrixBuilder(dataset)
    D = builder.build_day_slot_matrix(counter="Meals")
    analysis = DemandMatrixBuilder.analyze_matrix_properties(D, "Meals")

    assert analysis.matrix_rank > 0
    assert analysis.frobenius_norm > 0.0
    assert analysis.spectral_norm > 0.0
    assert analysis.condition_number >= 1.0
    # Singular values must be strictly sorted in descending order
    s = analysis.singular_values
    assert np.all(np.diff(s) <= 1e-8)
