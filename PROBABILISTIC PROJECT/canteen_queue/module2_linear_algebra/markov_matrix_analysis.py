"""
Spectral Eigen-Analysis of Markov Transition Matrix:
Perron-Frobenius Theorem, Manual Power Iteration, Stationary Distribution, and Mixing Rates.
Saveetha College Canteen Queue Assistant
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from canteen_queue.config import QUEUE_STATE_NAMES
from canteen_queue.module1_probabilistic.markov_chain import estimate_transition_matrix


def manual_power_iteration(
    P: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> Tuple[np.ndarray, int, List[float]]:
    """
    Computes the stationary distribution pi of a row-stochastic matrix P
    using the manual Power Iteration algorithm on left eigenvectors:
        pi^(k+1) = pi^(k) * P

    Mathematical Formulation:
    Since sum_j P_ij = 1, P * 1 = 1.
    The stationary distribution is the left eigenvector satisfying pi * P = 1 * pi.
    Equivalently, it is the right eigenvector of P^T: P^T * pi^T = pi^T.

    Convergence rate is governed by the second largest eigenvalue modulus:
        ||pi^(k) - pi|| = O(|lambda_2|^k)
    """
    n = P.shape[0]
    # Initialize with uniform probability distribution
    pi = np.full(n, 1.0 / n, dtype=float)
    history_errors = []

    for iteration in range(1, max_iter + 1):
        pi_next = pi @ P
        # Re-normalize to guarantee numerical sum == 1
        pi_next = pi_next / np.sum(pi_next)

        error = float(np.linalg.norm(pi_next - pi, ord=1))
        history_errors.append(error)

        if error < tol:
            return pi_next, iteration, history_errors

        pi = pi_next

    return pi, max_iter, history_errors


@dataclass
class SpectralAnalysisResult:
    counter: str
    transition_matrix: np.ndarray
    eigenvalues: np.ndarray             # Sorted descending by modulus
    eigenvalues_moduli: np.ndarray      # |lambda_i|
    spectral_radius: float              # Should be 1.0 (Perron-Frobenius)
    second_largest_modulus: float       # |lambda_2| (SLEM)
    spectral_gap: float                 # gamma = 1 - |lambda_2|
    mixing_relaxation_time: float       # tau_rel = 1 / gamma (steps)
    mixing_relaxation_minutes: float    # tau_rel * 5 minutes
    stationary_dist_power_iter: np.ndarray
    stationary_dist_eig: np.ndarray
    power_iter_steps: int
    agreement_error: float              # ||pi_power - pi_eig||_2


def perform_spectral_analysis(
    P: np.ndarray, counter_name: str = "Meals"
) -> SpectralAnalysisResult:
    """
    Performs complete eigen-decomposition on row-stochastic transition matrix P.
    """
    # 1. Manual Power Iteration
    pi_power, num_steps, err_history = manual_power_iteration(P)

    # 2. Exact Eigen-decomposition via numpy.linalg.eig on P^T
    # Left eigenvectors of P are right eigenvectors of P^T
    eigenvals, eigenvecs = np.linalg.eig(P.T)

    # Sort eigenvalues by modulus in descending order
    sort_idx = np.argsort(-np.abs(eigenvals))
    eigenvals_sorted = eigenvals[sort_idx]
    eigenvecs_sorted = eigenvecs[:, sort_idx]

    # Stationary distribution corresponding to lambda_1 = 1.0
    v1 = np.real(eigenvecs_sorted[:, 0])
    pi_eig = v1 / np.sum(v1)
    # Ensure non-negative
    if np.any(pi_eig < 0):
        pi_eig = -pi_eig

    agreement = float(np.linalg.norm(pi_power - pi_eig))

    # Second largest eigenvalue modulus (SLEM)
    moduli = np.abs(eigenvals_sorted)
    spectral_radius = float(moduli[0])
    slem = float(moduli[1]) if len(moduli) > 1 else 0.0
    spectral_gap = max(1e-6, 1.0 - slem)
    tau_rel_steps = 1.0 / spectral_gap
    tau_rel_mins = tau_rel_steps * 5.0

    return SpectralAnalysisResult(
        counter=counter_name,
        transition_matrix=P,
        eigenvalues=eigenvals_sorted,
        eigenvalues_moduli=moduli,
        spectral_radius=round(spectral_radius, 6),
        second_largest_modulus=round(slem, 5),
        spectral_gap=round(spectral_gap, 5),
        mixing_relaxation_time=round(tau_rel_steps, 2),
        mixing_relaxation_minutes=round(tau_rel_mins, 1),
        stationary_dist_power_iter=pi_power,
        stationary_dist_eig=pi_eig,
        power_iter_steps=num_steps,
        agreement_error=round(agreement, 8),
    )


class MarkovSpectralAnalysis:
    """
    Spectral engine and visualization for Markov queue dynamics.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results: Dict[str, SpectralAnalysisResult] = {}
        for counter in ["Meals", "Snacks", "Beverages"]:
            P = estimate_transition_matrix(df, counter=counter)
            self.results[counter] = perform_spectral_analysis(P, counter_name=counter)

    def plot_stationary_distribution(
        self,
        counter: str = "Meals",
        save_path: Optional[str] = None,
    ):
        """
        Plots stationary probability distribution across the 5 queue congestion states.
        """
        res = self.results[counter]
        pi = res.stationary_dist_power_iter
        names = QUEUE_STATE_NAMES

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(names, pi * 100.0, color="#1D3557", edgecolor="black", alpha=0.85)

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.8,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax.set_title(
            f"Saveetha Canteen: Stationary Congestion Distribution $\\pi$ ({counter})\n"
            f"Computed via Manual Power Iteration ({res.power_iter_steps} iterations, tol < 1e-10)\n"
            f"Spectral Gap $\\gamma = {res.spectral_gap:.3f}$ | Relaxation Time $\\tau_{{mix}} = {res.mixing_relaxation_minutes:.1f}$ min",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("Queue Congestion State", fontsize=11)
        ax.set_ylabel("Long-Term Steady-State Probability (%)", fontsize=11)
        ax.set_ylim(0, max(pi * 100.0) + 10)
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        return fig
