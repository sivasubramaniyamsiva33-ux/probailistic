"""
Bayesian Updating Engine: Conjugate Gamma-Poisson Model for Dynamic Arrival Rates
Saveetha College Canteen Queue Assistant
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class BayesianPosteriorState:
    counter: str
    time_slot: str
    prior_alpha: float
    prior_beta: float            # rate parameter (scale = 1/beta)
    prior_mean: float
    prior_variance: float
    observed_intervals: int      # number of observed 5-min intervals
    observed_total_arrivals: int # sum of observed arrivals K
    total_observation_time: float# T (in minutes)
    posterior_alpha: float
    posterior_beta: float
    posterior_mean: float        # arrivals per minute
    posterior_mode: float
    posterior_variance: float
    ci_90_lower: float
    ci_90_upper: float
    predictive_mean_next_slot: float # expected arrivals in next 5-min slot
    predictive_p10: int
    predictive_p90: int


class GammaPoissonModel:
    """
    Gamma-Poisson Bayesian Conjugate Engine.

    Formulation:
    1. Prior:
       lambda ~ Gamma(alpha_0, beta_0) where:
       f(lambda) = (beta_0^alpha_0 / Gamma(alpha_0)) * lambda^(alpha_0 - 1) * e^(-beta_0 * lambda)
       E[lambda] = alpha_0 / beta_0
       Var[lambda] = alpha_0 / beta_0^2

    2. Likelihood:
       Arrivals K in duration T follow Poisson(lambda * T):
       P(K | lambda) = ((lambda * T)^K * e^(-lambda * T)) / K!

    3. Posterior:
       lambda | K ~ Gamma(alpha_post, beta_post)
       alpha_post = alpha_0 + K
       beta_post  = beta_0 + T
       E[lambda | K] = (alpha_0 + K) / (beta_0 + T)

    4. Posterior Predictive Distribution for arrivals Y in next interval tau:
       Negative Binomial(r = alpha_post, p = beta_post / (beta_post + tau))
       P(Y = y) = Gamma(y + r)/(y! * Gamma(r)) * p^r * (1 - p)^y
    """

    def __init__(
        self,
        prior_alpha: float = 10.0,
        prior_beta: float = 10.0,
        slot_duration_minutes: float = 5.0,
    ):
        self.alpha_0 = float(prior_alpha)
        self.beta_0 = float(prior_beta)
        self.slot_duration = float(slot_duration_minutes)

        self.alpha_post = self.alpha_0
        self.beta_post = self.beta_0
        self.observed_arrivals_history: List[int] = []

    @classmethod
    def fit_empirical_bayes_prior(
        cls,
        df: pd.DataFrame,
        counter: str,
        time_slot: Optional[str] = None,
        slot_duration_minutes: float = 5.0,
    ) -> "GammaPoissonModel":
        """
        Derives informative empirical Bayes prior from historical variance and mean.
        Uses method of moments:
        mean_rate = sample_mean / slot_duration
        var_rate  = sample_var / (slot_duration^2)
        beta_0    = mean_rate / var_rate
        alpha_0   = mean_rate * beta_0
        """
        sub = df[df["counter"] == counter]
        if time_slot is not None:
            slot_sub = sub[sub["time_slot"] == time_slot]
            if len(slot_sub) >= 5:
                sub = slot_sub

        arrivals = sub["arrival_count"].dropna().to_numpy()
        sample_mean = float(np.mean(arrivals))
        sample_var = float(np.var(arrivals, ddof=1)) if len(arrivals) > 1 else max(1.0, sample_mean)

        # Arrival rate per minute
        mean_rate = max(0.05, sample_mean / slot_duration_minutes)
        var_rate = max(0.01, sample_var / (slot_duration_minutes ** 2))

        # Moment matching for Gamma(alpha, beta)
        beta_0 = max(0.5, mean_rate / var_rate)
        alpha_0 = max(1.0, mean_rate * beta_0)

        return cls(prior_alpha=alpha_0, prior_beta=beta_0, slot_duration_minutes=slot_duration_minutes)

    def update_live(self, new_arrivals: List[int]) -> BayesianPosteriorState:
        """
        Performs live conjugate Bayesian update with newly observed slot arrivals.
        """
        self.observed_arrivals_history.extend(new_arrivals)
        total_k = sum(self.observed_arrivals_history)
        total_time = len(self.observed_arrivals_history) * self.slot_duration

        self.alpha_post = self.alpha_0 + total_k
        self.beta_post = self.beta_0 + total_time

        post_mean = self.alpha_post / self.beta_post
        post_mode = (self.alpha_post - 1.0) / self.beta_post if self.alpha_post > 1.0 else 0.0
        post_var = self.alpha_post / (self.beta_post ** 2)

        # 90% Credible Interval (5th to 95th percentiles of Gamma)
        ci_lower = float(stats.gamma.ppf(0.05, a=self.alpha_post, scale=1.0 / self.beta_post))
        ci_upper = float(stats.gamma.ppf(0.95, a=self.alpha_post, scale=1.0 / self.beta_post))

        # Posterior Predictive for next 5-min slot
        tau = self.slot_duration
        r_nb = self.alpha_post
        p_nb = self.beta_post / (self.beta_post + tau)
        # scipy.stats.nbinom uses n=r, p=p
        pred_mean = float(r_nb * (1.0 - p_nb) / p_nb)
        pred_p10 = int(stats.nbinom.ppf(0.10, n=r_nb, p=p_nb))
        pred_p90 = int(stats.nbinom.ppf(0.90, n=r_nb, p=p_nb))

        return BayesianPosteriorState(
            counter="",
            time_slot="",
            prior_alpha=round(self.alpha_0, 3),
            prior_beta=round(self.beta_0, 3),
            prior_mean=round(self.alpha_0 / self.beta_0, 4),
            prior_variance=round(self.alpha_0 / (self.beta_0 ** 2), 4),
            observed_intervals=len(self.observed_arrivals_history),
            observed_total_arrivals=total_k,
            total_observation_time=round(total_time, 1),
            posterior_alpha=round(self.alpha_post, 3),
            posterior_beta=round(self.beta_post, 3),
            posterior_mean=round(post_mean, 4),
            posterior_mode=round(post_mode, 4),
            posterior_variance=round(post_var, 5),
            ci_90_lower=round(ci_lower, 4),
            ci_90_upper=round(ci_upper, 4),
            predictive_mean_next_slot=round(pred_mean, 2),
            predictive_p10=pred_p10,
            predictive_p90=pred_p90,
        )


class BayesianArrivalTracker:
    """
    Manages live Bayesian updating across all 3 counters for the interactive Streamlit dashboard.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.models: Dict[str, GammaPoissonModel] = {}
        for counter in df["counter"].unique():
            self.models[counter] = GammaPoissonModel.fit_empirical_bayes_prior(df, counter=counter)

    def live_update_counter(
        self,
        counter: str,
        time_slot: str,
        observed_arrivals: List[int],
    ) -> BayesianPosteriorState:
        model = self.models[counter]
        state = model.update_live(observed_arrivals)
        state.counter = counter
        state.time_slot = time_slot
        return state

    def plot_prior_vs_posterior(
        self,
        counter: str,
        save_path: Optional[str] = None,
    ):
        """
        Plots Prior PDF vs Posterior PDF along with the 90% credible interval.
        """
        model = self.models[counter]
        x_max = max(model.alpha_post / model.beta_post * 2.5, 3.0)
        x_grid = np.linspace(0.01, x_max, 500)

        prior_pdf = stats.gamma.pdf(x_grid, a=model.alpha_0, scale=1.0 / model.beta_0)
        post_pdf = stats.gamma.pdf(x_grid, a=model.alpha_post, scale=1.0 / model.beta_post)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            x_grid,
            prior_pdf,
            "--",
            color="#6C757D",
            linewidth=2.0,
            label=f"Prior Gamma({model.alpha_0:.1f}, {model.beta_0:.1f})",
        )
        ax.plot(
            x_grid,
            post_pdf,
            "-",
            color="#2A9D8F",
            linewidth=2.5,
            label=f"Posterior Gamma({model.alpha_post:.1f}, {model.beta_post:.1f})",
        )

        ci_lower = stats.gamma.ppf(0.05, a=model.alpha_post, scale=1.0 / model.beta_post)
        ci_upper = stats.gamma.ppf(0.95, a=model.alpha_post, scale=1.0 / model.beta_post)
        ax.axvspan(ci_lower, ci_upper, color="#2A9D8F", alpha=0.15, label="90% Posterior Credible Interval")

        ax.set_title(
            f"Saveetha Canteen: Bayesian Parameter Update for Arrival Rate λ ({counter})\n"
            f"Posterior Mean: {model.alpha_post/model.beta_post:.3f} arrivals/min",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Arrival Rate λ (arrivals per minute)", fontsize=11)
        ax.set_ylabel("Probability Density", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        return fig
