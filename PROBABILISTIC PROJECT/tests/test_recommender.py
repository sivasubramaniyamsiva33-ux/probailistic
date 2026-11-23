"""
Unit tests for Subspace Projection Recommender
"""
import pytest
from canteen_queue.data.generate_data import load_or_generate_data
from canteen_queue.module2_linear_algebra.recommender import (
    CanteenVisitRecommender,
    recommend_best_time_to_visit,
)


@pytest.fixture(scope="module")
def dataset():
    return load_or_generate_data()


def test_recommender_subspace_projection_and_cosine_similarity(dataset):
    """
    Verifies that recommender computes valid subspace coordinates, cosine similarity,
    and returns ranked upcoming low-wait recommendations.
    """
    rec = CanteenVisitRecommender(dataset, k_subspace=3)
    res = rec.recommend(
        current_time_slot="12:30",
        target_counter="Meals",
        current_queue_len=10,
        num_recommendations=3,
    )

    assert len(res.recommended_slots) == 3
    assert res.target_counter == "Meals"
    assert res.best_counter_alternative in ["Meals", "Snacks", "Beverages"]
    assert 0 <= res.most_similar_day_index < 60
    assert 0.0 <= res.cosine_similarity_score <= 1.0
    assert len(res.subspace_coordinates) == 3

    # Check that recommendation entries contain valid time strings and wait estimates
    for slot_info in res.recommended_slots:
        assert ":" in slot_info["time_slot"]
        assert "min" in slot_info["estimated_wait"]
