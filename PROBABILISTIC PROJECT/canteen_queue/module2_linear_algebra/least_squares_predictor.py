"""
Least-Squares Wait-Time Predictor: Normal Equations, Modified Gram-Schmidt QR,
SVD Pseudo-Inverse, Ridge Regularization, and Conditioning Analysis.
Saveetha College Canteen Queue Assistant
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def modified_gram_schmidt_qr(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes thin QR decomposition A = Q R using the numerically stable
    Modified Gram-Schmidt (MGS) orthogonalization algorithm.

    Mathematical Formulation:
    For matrix A in R^(m x n) with m >= n:
    Columns of Q are mutually orthonormal: Q^T Q = I_n.
    R is upper triangular with positive diagonal entries.

    MGS orthogonalizes each vector against the updated components,
    avoiding catastrophic cancellation that occurs in Classical Gram-Schmidt (CGS).
    """
    m, n = A.shape
    Q = np.zeros((m, n), dtype=float)
    R = np.zeros((n, n), dtype=float)
    V = A.astype(float).copy()

    for i in range(n):
        R[i, i] = np.linalg.norm(V[:, i])
        if R[i, i] > 1e-12:
            Q[:, i] = V[:, i] / R[i, i]
        else:
            Q[:, i] = 0.0

        for j in range(i + 1, n):
            R[i, j] = np.dot(Q[:, i], V[:, j])
            V[:, j] = V[:, j] - R[i, j] * Q[:, i]

    return Q, R


def back_substitution(R: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solves upper-triangular linear system R x = b via back-substitution.
    x_i = (b_i - sum_{j=i+1}^{n-1} R_ij * x_j) / R_ii
    """
    n = len(b)
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        if abs(R[i, i]) < 1e-12:
            x[i] = 0.0
        else:
            x[i] = (b[i] - np.dot(R[i, i + 1:], x[i + 1:])) / R[i, i]
    return x


def solve_normal_equations(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Solves normal equations: (X^T X) beta = X^T y
    beta = (X^T X)^(-1) X^T y
    """
    XtX = X.T @ X
    Xty = X.T @ y
    beta = np.linalg.solve(XtX, Xty)
    return beta


def solve_qr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Solves least squares using Modified Gram-Schmidt QR decomposition:
    X = Q R  =>  R beta = Q^T y
    """
    Q, R = modified_gram_schmidt_qr(X)
    Qty = Q.T @ y
    beta = back_substitution(R, Qty)
    return beta


def solve_svd_pinv(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Solves least squares using Singular Value Decomposition pseudo-inverse:
    X = U Sigma V^T  =>  X^+ = V Sigma^+ U^T
    beta = X^+ y
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    # Reciprocal of singular values with thresholding
    s_inv = np.zeros_like(s)
    threshold = 1e-12 * s[0]
    nonzero = s > threshold
    s_inv[nonzero] = 1.0 / s[nonzero]

    # X^+ y = V * diag(s_inv) * (U^T y)
    beta = Vt.T @ (s_inv * (U.T @ y))
    return beta


def solve_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Solves L2 Ridge Regression in closed matrix form:
    beta_ridge = (X^T X + alpha * I)^(-1) X^T y
    Does not regularize the intercept term (last column).
    """
    XtX = X.T @ X
    n_features = X.shape[1]
    penalty = alpha * np.eye(n_features)
    # Exclude bias/intercept (last column) from shrinkage
    penalty[-1, -1] = 0.0

    beta_ridge = np.linalg.solve(XtX + penalty, X.T @ y)
    return beta_ridge


@dataclass
class LeastSquaresMetrics:
    method: str
    beta: np.ndarray
    residual_norm: float
    r_squared: float
    rmse: float
    mae: float
    cond_X: float
    cond_XtX: float


class LeastSquaresPredictor:
    """
    Predicts waiting time using manual linear algebra implementations
    and provides condition number stability analysis.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.feature_names: List[str] = []
        self.X, self.y = self._build_design_matrix(self.df)
        self.models: Dict[str, LeastSquaresMetrics] = {}
        self.fit_all_methods()

    def _build_design_matrix(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Builds feature matrix X and target vector y:
        Features:
        1. Time-of-day harmonic cyclical features: sin(2*pi*t/24), cos(2*pi*t/24)
        2. Daytime peak harmonics: sin(4*pi*t/24), cos(4*pi*t/24)
        3. Day-of-week dummy encoding (Tuesday-Saturday; Monday is reference base)
        4. Counter dummy encoding (Snacks, Beverages; Meals is reference base)
        5. Observed queue length at arrival
        6. Intercept / Bias term (all 1s)

        Dropping the reference category avoids the Dummy Variable Trap (multicollinearity),
        ensuring full column rank rank(X) = d and non-singular X^T X.
        """
        slot_mins = data["slot_index"].to_numpy() * 5.0
        time_decimal = 8.0 + (slot_mins / 60.0)

        # 1. Harmonic cyclical encodings
        sin_t = np.sin(2.0 * np.pi * time_decimal / 24.0)
        cos_t = np.cos(2.0 * np.pi * time_decimal / 24.0)
        sin_2t = np.sin(4.0 * np.pi * time_decimal / 24.0)
        cos_2t = np.cos(4.0 * np.pi * time_decimal / 24.0)

        feature_cols = [sin_t, cos_t, sin_2t, cos_2t]
        names = ["sin_diurnal", "cos_diurnal", "sin_semidiurnal", "cos_semidiurnal"]

        # 2. Queue length
        q_len = data["queue_length_at_arrival"].to_numpy(dtype=float)
        feature_cols.append(q_len)
        names.append("queue_length")

        # 3. Day of week dummy encoding (drop Monday as reference)
        days_non_ref = ["Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        for d in days_non_ref:
            feature_cols.append((data["day_of_week"] == d).to_numpy(dtype=float))
            names.append(f"day_{d}")

        # 4. Counter dummy encoding (drop Meals as reference)
        counters_non_ref = ["Snacks", "Beverages"]
        for c in counters_non_ref:
            feature_cols.append((data["counter"] == c).to_numpy(dtype=float))
            names.append(f"counter_{c}")

        # 5. Bias column (1s)
        bias = np.ones(len(data), dtype=float)
        feature_cols.append(bias)
        names.append("bias_intercept")

        X = np.column_stack(feature_cols)
        y = data["observed_wait_time"].to_numpy(dtype=float)

        self.feature_names = names
        return X, y

    def fit_all_methods(self, ridge_alpha: float = 10.0):
        """
        Fits models across all 4 linear algebra solvers and computes metrics.
        """
        # Singular values and condition numbers
        s = np.linalg.svd(self.X, compute_uv=False)
        cond_X = float(s[0] / s[-1])
        cond_XtX = cond_X ** 2

        methods = {
            "Normal Equations": solve_normal_equations(self.X, self.y),
            "Modified Gram-Schmidt QR": solve_qr(self.X, self.y),
            "SVD Pseudo-Inverse": solve_svd_pinv(self.X, self.y),
            "Ridge Regression": solve_ridge(self.X, self.y, alpha=ridge_alpha),
        }

        y_mean = np.mean(self.y)
        ss_tot = np.sum((self.y - y_mean) ** 2)

        for name, beta in methods.items():
            y_pred = self.X @ beta
            residuals = self.y - y_pred
            res_norm = float(np.linalg.norm(residuals))
            ss_res = float(np.sum(residuals ** 2))
            r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
            rmse = float(np.sqrt(np.mean(residuals ** 2)))
            mae = float(np.mean(np.abs(residuals)))

            self.models[name] = LeastSquaresMetrics(
                method=name,
                beta=beta,
                residual_norm=round(res_norm, 2),
                r_squared=round(r2, 4),
                rmse=round(rmse, 4),
                mae=round(mae, 4),
                cond_X=round(cond_X, 2),
                cond_XtX=round(cond_XtX, 2),
            )

    def get_comparison_summary(self) -> pd.DataFrame:
        """
        Summarizes numerical comparisons across solvers.
        """
        rows = []
        for name, m in self.models.items():
            rows.append(
                {
                    "Method": name,
                    "R² Score": m.r_squared,
                    "RMSE (min)": m.rmse,
                    "MAE (min)": m.mae,
                    "Residual Norm": m.residual_norm,
                    "Cond(X)": m.cond_X,
                    "Cond(X^T X)": f"{m.cond_XtX:.2e}",
                }
            )
        df_res = pd.DataFrame(rows)
        return df_res

    def predict_sample(
        self,
        time_slot: str,
        day_of_week: str,
        counter: str,
        queue_length: int,
        method: str = "Modified Gram-Schmidt QR",
    ) -> float:
        """
        Predicts wait time for an individual input query vector x_query.
        """
        hour, minute = map(int, time_slot.split(":"))
        time_decimal = hour + (minute / 60.0)

        sin_t = np.sin(2.0 * np.pi * time_decimal / 24.0)
        cos_t = np.cos(2.0 * np.pi * time_decimal / 24.0)
        sin_2t = np.sin(4.0 * np.pi * time_decimal / 24.0)
        cos_2t = np.cos(4.0 * np.pi * time_decimal / 24.0)

        feat = [sin_t, cos_t, sin_2t, cos_2t, float(queue_length)]

        # Tuesday - Saturday (Monday is base)
        days_non_ref = ["Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        for d in days_non_ref:
            feat.append(1.0 if day_of_week == d else 0.0)

        # Snacks, Beverages (Meals is base)
        counters_non_ref = ["Snacks", "Beverages"]
        for c in counters_non_ref:
            feat.append(1.0 if counter == c else 0.0)

        feat.append(1.0)  # bias

        x_query = np.array(feat)
        beta = self.models[method].beta
        pred = float(np.dot(x_query, beta))
        return float(max(0.2, round(pred, 2)))
