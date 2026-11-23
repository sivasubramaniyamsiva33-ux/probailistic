"""
Demand Matrix Construction, Imputation, Normalization, and Matrix Properties
Saveetha College Canteen Queue Assistant
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class DemandMatrixAnalysis:
    counter: str
    shape: Tuple[int, int]
    matrix_rank: int
    frobenius_norm: float
    spectral_norm: float         # largest singular value sigma_max
    condition_number: float      # kappa = sigma_max / sigma_min
    singular_values: np.ndarray
    mean_demand: float
    std_demand: float


class DemandMatrixBuilder:
    """
    Constructs, cleans, normalizes, and computes structural linear algebra properties
    of the Canteen Demand Matrix D.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def build_day_slot_matrix(
        self,
        counter: str = "Meals",
        metric: str = "arrival_count",
        impute_missing: bool = True,
    ) -> np.ndarray:
        """
        Builds matrix D in R^(m x n):
        m = 60 days (rows)
        n = 120 time slots (columns, 08:00 to 17:55)
        D_ij represents arrivals on Day i at Time Slot j.
        """
        sub = self.df[self.df["counter"] == counter].copy()

        # Pivot: index=day_index, columns=slot_index, values=metric
        pivot = sub.pivot(index="day_index", columns="slot_index", values=metric)

        # Handle missing values
        if impute_missing and pivot.isna().sum().sum() > 0:
            # Impute column median (time-slot median across days)
            pivot = pivot.apply(lambda col: col.fillna(col.median()), axis=0)
            # Impute any remaining NaNs with global median
            pivot = pivot.fillna(pivot.median().median())

        D = pivot.to_numpy(dtype=float)
        return D

    def build_slot_counter_matrix(
        self,
        metric: str = "arrival_count",
        aggregation: str = "mean",
    ) -> np.ndarray:
        """
        Builds matrix D in R^(120 x 3):
        Rows = 120 Time Slots
        Columns = 3 Counters (Meals, Snacks, Beverages)
        Entries are average arrivals per slot.
        """
        pivot = self.df.pivot_table(
            index="slot_index",
            columns="counter",
            values=metric,
            aggfunc=aggregation,
        )
        return pivot.to_numpy(dtype=float)

    @staticmethod
    def normalize_matrix(
        D: np.ndarray, method: str = "zscore"
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Applies linear normalization to demand matrix:
        - 'center': mean-centered (D - mu)
        - 'zscore': standardized ((D - mu) / sigma)
        - 'minmax': scaled to [0, 1]
        """
        if method == "center":
            mean_col = np.mean(D, axis=0, keepdims=True)
            D_norm = D - mean_col
            params = {"mean": mean_col}
        elif method == "zscore":
            mean_col = np.mean(D, axis=0, keepdims=True)
            std_col = np.std(D, axis=0, keepdims=True)
            std_col[std_col < 1e-6] = 1.0
            D_norm = (D - mean_col) / std_col
            params = {"mean": mean_col, "std": std_col}
        elif method == "minmax":
            d_min = np.min(D, axis=0, keepdims=True)
            d_max = np.max(D, axis=0, keepdims=True)
            denom = d_max - d_min
            denom[denom < 1e-6] = 1.0
            D_norm = (D - d_min) / denom
            params = {"min": d_min, "max": d_max}
        else:
            D_norm = D.copy()
            params = {}

        return D_norm, params

    @staticmethod
    def analyze_matrix_properties(
        D: np.ndarray, counter_name: str = "Meals"
    ) -> DemandMatrixAnalysis:
        """
        Computes fundamental linear algebra properties:
        - Rank: rank(D)
        - Frobenius norm: ||D||_F = sqrt(sum_i sigma_i^2)
        - Spectral norm: sigma_1 = sigma_max
        - Condition number: kappa(D) = sigma_max / sigma_min
        """
        # SVD: D = U S V^T
        U, s, Vt = np.linalg.svd(D, full_matrices=False)

        # Numerical rank with machine tolerance
        eps = np.finfo(float).eps
        tol = max(D.shape) * eps * s[0]
        rank = int(np.sum(s > tol))

        frobenius = float(np.linalg.norm(D, ord="fro"))
        spectral = float(s[0])
        # Condition number
        min_singular = s[s > 1e-12][-1] if np.any(s > 1e-12) else 1e-12
        cond_num = float(s[0] / min_singular)

        return DemandMatrixAnalysis(
            counter=counter_name,
            shape=D.shape,
            matrix_rank=rank,
            frobenius_norm=round(frobenius, 2),
            spectral_norm=round(spectral, 2),
            condition_number=round(cond_num, 2),
            singular_values=s,
            mean_demand=round(float(np.mean(D)), 2),
            std_demand=round(float(np.std(D)), 2),
        )

    def plot_demand_matrix_heatmap(
        self,
        counter: str = "Meals",
        save_path: Optional[str] = None,
    ):
        """
        Plots 2D heatmap of the Day x Slot demand matrix.
        """
        D = self.build_day_slot_matrix(counter=counter)
        analysis = self.analyze_matrix_properties(D, counter)

        fig, ax = plt.subplots(figsize=(10, 6))
        cax = ax.imshow(D, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label("Arrival Count per 5-min Slot", fontsize=11)

        # Slot tick labels (every 2 hours = 24 slots)
        x_ticks = [0, 24, 48, 72, 96, 119]
        x_labels = ["08:00", "10:00", "12:00", "14:00", "16:00", "17:55"]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)

        ax.set_title(
            f"Saveetha Canteen: Demand Matrix Heatmap D ({counter})\n"
            f"Shape: {D.shape[0]} Days × {D.shape[1]} Slots | Rank = {analysis.matrix_rank} | "
            f"Cond(D) = {analysis.condition_number:.1f} | ||D||_F = {analysis.frobenius_norm:.1f}",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Time of Day", fontsize=11)
        ax.set_ylabel("Day of Operation (0 to 59)", fontsize=11)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        return fig
