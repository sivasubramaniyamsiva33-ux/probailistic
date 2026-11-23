"""
Unit tests for Bayesian Updating Engine (Conjugate Gamma-Poisson Model)
"""
import pytest
from canteen_queue.data.generate_data import load_or_generate_data
from canteen_queue.module1_probabilistic.bayesian_update import (
    BayesianArrivalTracker,
    GammaPoissonModel,
)


@pytest.fixture(scope="module")
def dataset():
    return load_or_generate_data()


def test_gamma_poisson_conjugate_update_formulas():
    """
    Verifies that live Bayesian updating strictly obeys conjugate formulas:
    alpha_post = alpha_0 + sum(K)
    beta_post = beta_0 + T
    """
    model = GammaPoissonModel(prior_alpha=12.0, prior_beta=10.0, slot_duration_minutes=5.0)
    assert model.alpha_0 == 12.0
    assert model.beta_0 == 10.0
    initial_mean = 12.0 / 10.0  # 1.2 cust/min

    # Observe 3 slots with arrivals [8, 10, 12] -> total K = 30 over 15 minutes
    state = model.update_live([8, 10, 12])

    expected_alpha = 12.0 + 30.0  # 42.0
    expected_beta = 10.0 + 15.0   # 25.0
    expected_mean = 42.0 / 25.0   # 1.68

    assert state.posterior_alpha == pytest.approx(expected_alpha, abs=1e-4)
    assert state.posterior_beta == pytest.approx(expected_beta, abs=1e-4)
    assert state.posterior_mean == pytest.approx(expected_mean, abs=1e-4)


def test_posterior_variance_shrinkage():
    """
    Bayesian property: As more empirical data is gathered, posterior variance
    strictly decreases relative to prior variance (epistemic uncertainty reduction).
    """
    model = GammaPoissonModel(prior_alpha=5.0, prior_beta=4.0, slot_duration_minutes=5.0)
    prior_var = 5.0 / (4.0 ** 2)

    # 10 observed slots of 10 arrivals each
    state = model.update_live([10] * 10)
    post_var = state.posterior_variance

    assert post_var < prior_var
    assert state.ci_90_lower < state.posterior_mean < state.ci_90_upper


def test_empirical_bayes_prior_and_predictive(dataset):
    """
    Tests automated Empirical Bayes prior fitting and Negative Binomial posterior predictive.
    """
    tracker = BayesianArrivalTracker(dataset)
    state = tracker.live_update_counter("Meals", "12:30", [15, 18])

    assert state.prior_alpha > 0.0
    assert state.prior_beta > 0.0
    assert state.predictive_mean_next_slot > 0.0
    assert state.predictive_p10 <= state.predictive_mean_next_slot <= state.predictive_p90
