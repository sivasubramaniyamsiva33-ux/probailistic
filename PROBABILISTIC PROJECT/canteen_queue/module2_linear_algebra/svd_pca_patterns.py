"""
SVD & PCA Spectral Decomposition of Canteen Demand:
Singular Value Spectrum, Canonical Diurnal Modes, and Eckart-Young Low-Rank Approximations.
Saveetha College Canteen Queue Assistant
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from canteen_queue.module2_linear_algebra.demand_matrix import DemandMatrixBuilder


@dataclass
class SVDAnalysisResult:
    counter: str
    shape: Tuple[int, int]
    singular_values: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_energy: np.ndarray
    components_for_90pct: int
    components_for_95pct: int
    reconstruction_errors: np.ndarray    # Frobenius error ||D - D_k||_F for k=1..r
    top_v_patterns: np.ndarray           # Top 3 right singular vectors (temporal patterns)
    top_u_loadings: np.ndarray           # Top 3 left singular vectors (day loadings)


def perform_svd_analysis(
    D: np.ndarray, counter_name: str = "Meals"
) -> SVDAnalysisResult:
    """
    Computes SVD decomposition D = U Sigma V^T and derives spectral energy capture.

    Mathematical Formulations:
    1. SVD Decomposition:
       D = sum_{i=1}^r sigma_i * u_i * v_i^T
       where u_i in R^m, v_i in R^n, and sigma_1 >= sigma_2 >= ... >= sigma_r >= 0.

    2. Energy / Variance Captured:
       Total Energy E_total = ||D||_F^2 = sum_{i=1}^r sigma_i^2
       Cumulative Energy E_k = (sum_{i=1}^k sigma_i^2) / E_total

    3. Eckart-Young-Mirsky Theorem:
       For rank-k approximation D_k = sum_{i=1}^k sigma_i * u_i * v_i^T:
       min_{rank(B) <= k} ||D - B||_F = ||D - D_k||_F = sqrt(sum_{j=k+1}^r sigma_j^2)
    """
    U, s, Vt = np.linalg.svd(D, full_matrices=False)
    V = Vt.T

    # Energy calculations
    squared_singular = s ** 2
    total_energy = np.sum(squared_singular)
    explained_ratio = squared_singular / total_energy
    cum_energy = np.cumsum(explained_ratio)

    # Number of components needed for 90% and 95%
    k_90 = int(np.argmax(cum_energy >= 0.90) + 1)
    k_95 = int(np.argmax(cum_energy >= 0.95) + 1)

    # Low-rank reconstruction errors for k = 1 to min(20, len(s))
    max_k_eval = min(30, len(s))
    rec_errors = np.zeros(max_k_eval)
    for k in range(1, max_k_eval + 1):
        D_k = (U[:, :k] * s[:k]) @ Vt[:k, :]
        rec_errors[k - 1] = np.linalg.norm(D - D_k, ord="fro")

    return SVDAnalysisResult(
        counter=counter_name,
        shape=D.shape,
        singular_values=s,
        explained_variance_ratio=explained_ratio,
        cumulative_energy=cum_energy,
        components_for_90pct=k_90,
        components_for_95pct=k_95,
        reconstruction_errors=rec_errors,
        top_v_patterns=V[:, :3],   # shape (120, 3)
        top_u_loadings=U[:, :3],   # shape (60, 3)
    )


class SVDDemandPatterns:
    """
    High-level facade for SVD diurnal analysis and visualization.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.builder = DemandMatrixBuilder(df)
        self.results: Dict[str, SVDAnalysisResult] = {}
        for counter in ["Meals", "Snacks", "Beverages"]:
            D = self.builder.build_day_slot_matrix(counter=counter)
            self.results[counter] = perform_svd_analysis(D, counter_name=counter)

    def plot_scree_and_energy(
        self,
        counter: str = "Meals",
        save_path: Optional[str] = None,
    ):
        """
        Plots Scree plot (singular value spectrum) and Cumulative Energy % curve.
        """
        res = self.results[counter]
        s = res.singular_values[:20]
        k_axis = np.arange(1, len(s) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 1. Singular Values (Scree)
        ax1.plot(k_axis, s, "o-", color="#1D3557", linewidth=2.2, markersize=6)
        ax1.set_title(f"Scree Plot: Singular Values $\\sigma_k$ ({counter})", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Principal Component Index $k$", fontsize=10)
        ax1.set_ylabel("Singular Value $\\sigma_k$", fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.5)

        # 2. Cumulative Energy Captured
        cum_pct = res.cumulative_energy[:20] * 100.0
        ax2.plot(k_axis, cum_pct, "s-", color="#E63946", linewidth=2.2, markersize=6)
        ax2.axhline(90.0, color="#2A9D8F", linestyle="--", label="90% Energy Threshold")
        ax2.axvline(
            res.components_for_90pct,
            color="#2A9D8F",
            linestyle=":",
            label=f"Rank $k={res.components_for_90pct}$ reaches 90%",
        )
        ax2.set_title(
            f"Cumulative Energy ($E_k = \\sum_{{i=1}}^k \\sigma_i^2 / \\|D\\|_F^2$)\n"
            f"Top {res.components_for_90pct} components capture ≥90% variance",
            fontsize=11,
            fontweight="bold",
        )
        ax2.set_xlabel("Number of Principal Components ($k$)", fontsize=10)
        ax2.set_ylabel("Cumulative Energy (%)", fontsize=10)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(frameon=True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        return fig

    def plot_canonical_daily_patterns(
        self,
        counter: str = "Meals",
        save_path: Optional[str] = None,
    ):
        """
        Plots the top 3 right singular vectors v1, v2, v3 (canonical diurnal basis modes).
        """
        res = self.results[counter]
        V_top3 = res.top_v_patterns  # shape (120, 3)
        time_slots = np.arange(120)

        fig, ax = plt.subplots(figsize=(10, 5))
        pattern_labels = [
            f"Mode 1 (Baseline Profile: {res.explained_variance_ratio[0]*100:.1f}% var)",
            f"Mode 2 (Lunch Rush Mode: {res.explained_variance_ratio[1]*100:.1f}% var)",
            f"Mode 3 (Class-Break Spikes: {res.explained_variance_ratio[2]*100:.1f}% var)",
        ]
        colors = ["#1D3557", "#E63946", "#2A9D8F"]

        for idx in range(3):
            ax.plot(
                time_slots,
                V_top3[:, idx],
                label=pattern_labels[idx],
                color=colors[idx],
                linewidth=2.2,
            )

        x_ticks = [0, 24, 48, 72, 96, 119]
        x_labels = ["08:00", "10:00", "12:00", "14:00", "16:00", "17:55"]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)

        ax.set_title(
            f"Saveetha Canteen: Canonical Diurnal Basis Modes $v_1, v_2, v_3$ ({counter})\n"
            f"Orthonormal Right Singular Vectors: $V^T V = I_3$",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Time of Day", fontsize=11)
        ax.set_ylabel("Singular Vector Amplitude $v_{k, j}$", fontsize=11)
        ax.axhline(0, color="black", linestyle="--", alpha=0.3)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        return fig

    def plot_reconstruction_error(
        self,
        counter: str = "Meals",
        save_path: Optional[str] = None,
    ):
        """
        Plots Frobenius reconstruction error ||D - D_k||_F as a function of rank k,
        demonstrating the Eckart-Young-Mirsky theorem.
        """
        res = self.results[counter]
        errors = res.reconstruction_errors
        ranks = np.arange(1, len(errors) + 1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ranks, errors, "d-", color="#8338EC", linewidth=2.2, markersize=6)
        ax.axvline(
            res.components_for_90pct,
            color="#2A9D8F",
            linestyle="--",
            label=f"Rank k={res.components_for_90pct} (90% Energy)",
        )
        ax.set_title(
            f"Eckart-Young Theorem: Frobenius Error $\\|D - D_k\\|_F$ vs. Rank $k$ ({counter})\n"
            f"Monotonically Decreasing Error $\\|D - D_k\\|_F = \\sqrt{{\\sum_{{j=k+1}}^r \\sigma_j^2}}$",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("Approximation Rank $k$", fontsize=10)
        ax.set_ylabel("Frobenius Reconstruction Error", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        return fig
