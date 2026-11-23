"""
Unit tests for Least-Squares Wait-Time Predictors & Numerical Linear Algebra Routines
"""
import numpy as np
import pytest
from canteen_queue.data.generate_data import load_or_generate_data
from canteen_queue.module2_linear_algebra.least_squares_predictor import (
    LeastSquaresPredictor,
    back_substitution,
    modified_gram_schmidt_qr,
    solve_normal_equations,
    solve_qr,
    solve_ridge,
    solve_svd_pinv,
)


@pytest.fixture(scope="module")
def dataset():
    return load_or_generate_data()


def test_modified_gram_schmidt_qr_decomposition():
    """
    Tests manual Modified Gram-Schmidt algorithm:
    1. Q^T Q = I_n (orthonormality)
    2. R is upper-triangular
    3. Q R = A reconstruction error < 1e-10
    """
    np.random.seed(42)
    m, n = 50, 10
    A = np.random.randn(m, n)

    Q, R = modified_gram_schmidt_qr(A)

    # 1. Orthonormality Q^T Q = I
    QtQ = Q.T @ Q
    assert np.allclose(QtQ, np.eye(n), atol=1e-10)

    # 2. Upper triangular R
    assert np.allclose(np.tril(R, -1), 0.0, atol=1e-10)

    # 3. Exact reconstruction A = Q R
    reconstructed = Q @ R
    assert np.allclose(A, reconstructed, atol=1e-10)


def test_back_substitution_solver():
    """
    Validates manual back-substitution against exact known solution.
    """
    np.random.seed(42)
    n = 6
    R = np.triu(np.random.randn(n, n) + 2.0 * np.eye(n))
    x_true = np.array([1.5, -2.0, 3.0, 0.5, -1.0, 4.0])
    b = R @ x_true

    x_solved = back_substitution(R, b)
    assert np.allclose(x_solved, x_true, atol=1e-10)


def test_least_squares_solvers_numerical_agreement(dataset):
    """
    Key Prompt Requirement:
    Verify that Normal Equations, Modified Gram-Schmidt QR, and SVD Pseudo-Inverse
    yield identical regression coefficient vectors beta to high numerical precision.
    """
    predictor = LeastSquaresPredictor(dataset)
    X, y = predictor.X, predictor.y

    beta_ne = solve_normal_equations(X, y)
    beta_qr = solve_qr(X, y)
    beta_svd = solve_svd_pinv(X, y)

    # Validate pairwise agreement across all three methods
    diff_ne_qr = float(np.linalg.norm(beta_ne - beta_qr))
    diff_qr_svd = float(np.linalg.norm(beta_qr - beta_svd))

    assert diff_ne_qr < 1e-3
    assert diff_qr_svd < 1e-3


def test_ridge_regression_shrinkage(dataset):
    """
    Verifies that Ridge regularization properly penalizes and shrinks coefficients.
    """
    predictor = LeastSquaresPredictor(dataset)
    X, y = predictor.X, predictor.y

    beta_ols = solve_qr(X, y)
    beta_ridge = solve_ridge(X, y, alpha=500.0)

    # Norm of non-bias coefficients should shrink with strong ridge penalty
    norm_ols = float(np.linalg.norm(beta_ols[:-1]))
    norm_ridge = float(np.linalg.norm(beta_ridge[:-1]))
    assert norm_ridge < norm_ols


def test_condition_number_squaring_property(dataset):
    """
    Verifies the fundamental numerical conditioning identity:
    cond(X^T X) = (cond(X))^2
    """
    predictor = LeastSquaresPredictor(dataset)
    m = predictor.models["Modified Gram-Schmidt QR"]
    expected_cond_XtX = m.cond_X ** 2
    assert m.cond_XtX == pytest.approx(expected_cond_XtX, rel=1e-4)
