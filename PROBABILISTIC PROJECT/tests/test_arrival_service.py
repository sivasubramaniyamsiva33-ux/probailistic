"""
Unit tests for Arrival & Service Distribution Modeling
"""
import numpy as np
import pandas as pd
import pytest
from canteen_queue.data.generate_data import load_or_generate_data
from canteen_queue.module1_probabilistic.arrival_model import (
    ArrivalModel,
    fit_exponential_interarrivals,
    fit_poisson_arrivals,
)
from canteen_queue.module1_probabilistic.service_model import (
    ServiceModel,
    fit_service_distributions,
)


@pytest.fixture(scope="module")
def dataset():
    return load_or_generate_data()


def test_poisson_arrivals_mle_and_chisquare(dataset):
    """
    Validates Poisson MLE equals empirical mean and chi2 test produces valid p-value.
    """
    fit = fit_poisson_arrivals(dataset, counter="Meals", time_band="Lunch")
    sub_df = dataset[(dataset["counter"] == "Meals") & (dataset["slot_index"] >= 36) & (dataset["slot_index"] < 72)]
    expected_mle = float(sub_df["arrival_count"].mean())

    assert fit.lambda_mle == pytest.approx(expected_mle, abs=1e-3)
    assert fit.chi2_stat >= 0.0
    assert 0.0 <= fit.p_value <= 1.0
    assert fit.degrees_of_freedom >= 1


def test_exponential_interarrivals_kstest(dataset):
    """
    Validates Exponential fit on inter-arrivals produces a rate parameter and valid KS statistic.
    """
    fit = fit_exponential_interarrivals(dataset, counter="Beverages")
    assert fit.mean_interarrival_sec > 0.0
    assert fit.rate_parameter == pytest.approx(1.0 / fit.mean_interarrival_sec, rel=1e-3)
    assert 0.0 <= fit.ks_stat <= 1.0
    assert 0.0 <= fit.ks_p_value <= 1.0


def test_service_distributions_mle_and_aic(dataset):
    """
    Tests Exponential and Gamma MLE estimation and AIC/BIC model comparison.
    """
    fit = fit_service_distributions(dataset, counter="Snacks")
    assert fit.sample_size > 0
    assert fit.mean_seconds > 0.0
    assert fit.mu_exp_per_min == pytest.approx(60.0 / fit.mean_seconds, rel=1e-3)
    assert fit.aic_exp > 0.0
    assert fit.aic_gamma > 0.0
    assert fit.alpha_gamma > 0.0
    assert fit.scale_gamma > 0.0
    assert fit.best_distribution in ["Gamma", "Exponential"]


def test_service_model_summary_table(dataset):
    """
    Verifies ServiceModel facade generates a complete summary table across all counters.
    """
    sm = ServiceModel(dataset)
    df_sum = sm.get_summary_table()
    assert len(df_sum) == 3
    assert "Counter" in df_sum.columns
    assert "Delta AIC (Exp - Gamma)" in df_sum.columns
