"""
Subspace Projection & Cosine-Similarity Canteen Visit Recommender
Saveetha College Canteen Queue Assistant
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from canteen_queue.config import COUNTERS, START_HOUR
from canteen_queue.module2_linear_algebra.demand_matrix import DemandMatrixBuilder


@dataclass
class RecommendationResult:
    current_time_slot: str
    target_counter: str
    best_counter_alternative: str
    current_slot_expected_wait: float
    best_counter_expected_wait: float
    recommended_slots: List[Dict[str, str]]
    most_similar_day_index: int
    cosine_similarity_score: float
    subspace_coordinates: np.ndarray


class CanteenVisitRecommender:
    """
    Principal Subspace Projection and Cosine Similarity Recommender Engine.

    Formulation:
    1. Historical day demand matrix D in R^(m x n), where each row is a historical day vector d_i in R^n.
    2. Principal Subspace:
       SVD: D = U Sigma V^T. The top k right singular vectors V_k = [v_1, ..., v_k] in R^(n x k)
       span the k-dimensional optimal principal subspace.
    3. Orthogonal Projection:
       The projection of an observed partial or complete day vector d_obs onto the subspace is:
       p = (d_obs V_k) V_k^T
       Coordinate representation in principal coordinates: z = d_obs V_k in R^k.
    4. Cosine Similarity Matching:
       similarity(d_i, d_obs) = <d_i, d_obs> / (||d_i||_2 * ||d_obs||_2)
       Matches today's partial trajectory against historical days to forecast
       future congestion and recommend the optimal low-queue windows.
    """

    def __init__(self, df: pd.DataFrame, k_subspace: int = 3):
        self.df = df
        self.k_subspace = k_subspace
        self.builder = DemandMatrixBuilder(df)
        self.D_matrices: Dict[str, np.ndarray] = {}
        self.V_subspaces: Dict[str, np.ndarray] = {}

        for c in COUNTERS:
            D = self.builder.build_day_slot_matrix(counter=c)
            self.D_matrices[c] = D
            _, _, Vt = np.linalg.svd(D, full_matrices=False)
            self.V_subspaces[c] = Vt[:k_subspace, :].T  # shape (120, k)

    def recommend(
        self,
        current_time_slot: str = "12:30",
        target_counter: str = "Meals",
        current_queue_len: int = 8,
        num_recommendations: int = 3,
    ) -> RecommendationResult:
        """
        Recommends best upcoming slots and least-crowded alternative counter.
        """
        hour, minute = map(int, current_time_slot.split(":"))
        slot_idx = int(((hour - START_HOUR) * 60 + minute) // 5)
        slot_idx = min(max(0, slot_idx), 119)

        D_target = self.D_matrices[target_counter]
        V_k = self.V_subspaces[target_counter]

        # 1. Construct partial day query vector up to current slot
        # Average historical profile up to current slot, augmented with current observation
        query_vector = np.mean(D_target, axis=0).copy()
        query_vector[slot_idx] = float(current_queue_len)

        # 2. Project query vector onto principal subspace
        z_coords = query_vector @ V_k  # shape (k,)
        projected_vector = z_coords @ V_k.T  # shape (120,)

        # 3. Cosine similarity against historical days
        query_norm = np.linalg.norm(query_vector)
        if query_norm < 1e-6:
            query_norm = 1.0

        similarities = np.zeros(D_target.shape[0])
        for i in range(D_target.shape[0]):
            d_i = D_target[i]
            d_i_norm = np.linalg.norm(d_i)
            if d_i_norm > 1e-6:
                similarities[i] = np.dot(d_i, query_vector) / (d_i_norm * query_norm)
            else:
                similarities[i] = 0.0

        best_day_idx = int(np.argmax(similarities))
        best_sim_score = float(similarities[best_day_idx])

        # 4. Search upcoming slots for lowest congestion
        # Upcoming window: slots from slot_idx to min(119, slot_idx + 24) (next 2 hours)
        end_slot = min(120, slot_idx + 24)
        if end_slot - slot_idx < 3:
            # If near end of day, search afternoon window
            upcoming_indices = np.arange(max(0, slot_idx - 12), slot_idx)
        else:
            upcoming_indices = np.arange(slot_idx, end_slot)

        # Predict future demand using weighted blend of projection and best-matched day
        matched_day_demand = D_target[best_day_idx]
        predicted_upcoming = 0.6 * matched_day_demand[upcoming_indices] + 0.4 * projected_vector[upcoming_indices]

        # Rank slots by lowest predicted arrivals/congestion
        sorted_rel_order = np.argsort(predicted_upcoming)
        recommended_slots = []

        for rank_idx in sorted_rel_order[:num_recommendations]:
            actual_slot = int(upcoming_indices[rank_idx])
            s_hour = START_HOUR + (actual_slot * 5) // 60
            s_min = (actual_slot * 5) % 60
            time_str = f"{s_hour:02d}:{s_min:02d}"
            est_wait = max(1.0, round(float(predicted_upcoming[rank_idx]) * 0.45, 1))
            recommended_slots.append(
                {
                    "time_slot": time_str,
                    "predicted_demand": f"{predicted_upcoming[rank_idx]:.1f} arrivals",
                    "estimated_wait": f"{est_wait:.1f} min",
                }
            )

        # 5. Evaluate alternative counters at the requested time
        slot_demand_by_counter = {}
        for c in COUNTERS:
            mean_c = float(np.mean(self.D_matrices[c][:, slot_idx]))
            slot_demand_by_counter[c] = mean_c

        # Best alternative counter is one with lowest demand/wait at this slot
        best_alt_counter = min(slot_demand_by_counter, key=slot_demand_by_counter.get)
        current_wait = round(float(slot_demand_by_counter[target_counter]) * 0.5, 1)
        alt_wait = round(float(slot_demand_by_counter[best_alt_counter]) * 0.35, 1)

        return RecommendationResult(
            current_time_slot=current_time_slot,
            target_counter=target_counter,
            best_counter_alternative=best_alt_counter,
            current_slot_expected_wait=current_wait,
            best_counter_expected_wait=alt_wait,
            recommended_slots=recommended_slots,
            most_similar_day_index=best_day_idx,
            cosine_similarity_score=round(best_sim_score, 4),
            subspace_coordinates=np.round(z_coords, 3),
        )


def recommend_best_time_to_visit(
    df: pd.DataFrame,
    current_time_slot: str,
    target_counter: str,
    current_queue_len: int,
) -> RecommendationResult:
    """
    Convenience wrapper for the recommender engine.
    """
    engine = CanteenVisitRecommender(df)
    return engine.recommend(
        current_time_slot=current_time_slot,
        target_counter=target_counter,
        current_queue_len=current_queue_len,
    )
