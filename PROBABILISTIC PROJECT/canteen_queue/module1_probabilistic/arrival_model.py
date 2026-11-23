"""
Arrival Process Modeling: Poisson Process & Exponential Inter-arrivals
Saveetha College Canteen Queue Assistant
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PoissonFitResult:
    counter: str
    time_band: str
    lambda_mle: float             # MLE estimate lambda_hat = sample mean
    sample_variance: float
    overdispersion_index: float   # Var / Mean (Index of Dispersion)
    chi2_stat: float
    p_value: float
    degrees_of_freedom: int
    is_good_fit: bool             # p_value > 0.05


@dataclass
class InterArrivalFitResult:
    counter: str
    mean_interarrival_sec: float
    rate_parameter: float         # lambda = 1 / mean
    ks_stat: float
    ks_p_value: float
    is_exponential: bool          # ks_p_value > 0.05


def fit_poisson_arrivals(
    df: pd.DataFrame,
    counter: str,
    time_band: Optional[str] = None,
    slot_minutes: int = 5,
) -> PoissonFitResult:
    """
    Fits a Poisson distribution to arrival counts in a specified time band and counter.
    Performs Chi-Square Goodness-of-Fit test with tail bin pooling (E_i >= 5).

    Formula:
        lambda_hat = (1/N) * sum_{i=1}^N k_i
        P(K = k) = (lambda^k * e^(-lambda)) / k!
        Chi2 = sum_{j=1}^m ((O_j - E_j)^2 / E_j)
    """
    sub_df = df[df["counter"] == counter].copy()

    # Determine time band filter if requested
    if time_band == "Morning":
        sub_df = sub_df[(sub_df["slot_index"] >= 0) & (sub_df["slot_index"] < 36)]  # 08:00 - 11:00
    elif time_band == "Lunch":
        sub_df = sub_df[(sub_df["slot_index"] >= 36) & (sub_df["slot_index"] < 72)] # 11:00 - 14:00
    elif time_band == "Afternoon":
        sub_df = sub_df[(sub_df["slot_index"] >= 72) & (sub_df["slot_index"] < 120)]# 14:00 - 18:00
    else:
        time_band = "All-Day"

    arrivals = sub_df["arrival_count"].to_numpy(dtype=float)
    n = len(arrivals)
    if n == 0:
        raise ValueError(f"No records found for counter={counter}, time_band={time_band}")

    # Maximum Likelihood Estimator
    lambda_mle = float(np.mean(arrivals))
    sample_var = float(np.var(arrivals, ddof=1))
    dispersion = sample_var / max(1e-6, lambda_mle)

    # Chi-Square Goodness of Fit test with pooled bins
    max_k = int(np.max(arrivals))
    observed_counts = np.bincount(arrivals.astype(int), minlength=max_k + 1)
    k_vals = np.arange(len(observed_counts))

    # Expected probabilities
    expected_probs = stats.poisson.pmf(k_vals, mu=lambda_mle)
    # Tail probability adjustment so sum(expected_probs) = 1.0
    expected_probs[-1] += 1.0 - np.sum(expected_probs)
    expected_counts = expected_probs * n

    # Bin pooling: ensure expected counts >= 5
    pooled_obs = []
    pooled_exp = []
    cur_obs = 0.0
    cur_exp = 0.0

    for obs, exp in zip(observed_counts, expected_counts):
        cur_obs += obs
        cur_exp += exp
        if cur_exp >= 5.0:
            pooled_obs.append(cur_obs)
            pooled_exp.append(cur_exp)
            cur_obs = 0.0
            cur_exp = 0.0

    # Pool remaining tail into previous bin
    if cur_exp > 0:
        if len(pooled_exp) > 0:
            pooled_obs[-1] += cur_obs
            pooled_exp[-1] += cur_exp
        else:
            pooled_obs.append(cur_obs)
            pooled_exp.append(cur_exp)

    pooled_obs = np.array(pooled_obs)
    pooled_exp = np.array(pooled_exp)

    # Degrees of freedom: num_bins - 1 - num_estimated_params (1 parameter: lambda)
    num_bins = len(pooled_obs)
    dof = max(1, num_bins - 1 - 1)

    if num_bins >= 2:
        chi2_stat = float(np.sum((pooled_obs - pooled_exp) ** 2 / pooled_exp))
        p_val = float(1.0 - stats.chi2.cdf(chi2_stat, df=dof))
    else:
        chi2_stat = 0.0
        p_val = 1.0

    return PoissonFitResult(
        counter=counter,
        time_band=time_band,
        lambda_mle=round(lambda_mle, 4),
        sample_variance=round(sample_var, 4),
        overdispersion_index=round(dispersion, 4),
        chi2_stat=round(chi2_stat, 4),
        p_value=round(p_val, 4),
        degrees_of_freedom=dof,
        is_good_fit=bool(p_val > 0.01),
    )


def fit_exponential_interarrivals(
    df: pd.DataFrame,
    counter: str,
    slot_minutes: int = 5,
) -> InterArrivalFitResult:
    """
    Derives inter-arrival times from arrivals per slot and fits an Exponential distribution.
    Validates goodness of fit using the 2-sided Kolmogorov-Smirnov test.

    Formula:
        Inter-arrival Delta_t = (slot_minutes * 60) / arrival_count
        Exponential PDF: f(t) = lambda * e^(-lambda * t)
        KS Statistic: D = sup_t |F_n(t) - F(t)|
    """
    sub_df = df[(df["counter"] == counter) & (df["arrival_count"] > 0)].copy()
    slot_seconds = slot_minutes * 60.0

    # Synthetic individual inter-arrival times based on homogeneous Poisson rate within slots
    interarrivals = []
    for arrivals in sub_df["arrival_count"].to_numpy():
        if arrivals > 0:
            # Random uniform arrival times within slot -> sorted differences
            u = np.sort(np.random.uniform(0, slot_seconds, size=int(arrivals)))
            diffs = np.diff(np.concatenate(([0.0], u)))
            interarrivals.extend(diffs[diffs > 0])

    interarrivals = np.array(interarrivals)
    mean_dt = float(np.mean(interarrivals))
    rate_param = 1.0 / max(1e-3, mean_dt)

    # 2-sided KS Test against scipy.stats.expon(scale=1/rate_param)
    ks_res = stats.kstest(interarrivals, "expon", args=(0, 1.0 / rate_param))

    return InterArrivalFitResult(
        counter=counter,
        mean_interarrival_sec=round(mean_dt, 2),
        rate_parameter=round(rate_param, 5),
        ks_stat=round(float(ks_res.statistic), 4),
        ks_p_value=round(float(ks_res.pvalue), 4),
        is_exponential=bool(ks_res.pvalue > 0.01),
    )


class ArrivalModel:
    """
    Encapsulates arrival modeling and goodness-of-fit analytics for all counters
    and diurnal bands.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.fits: Dict[Tuple[str, str], PoissonFitResult] = {}
        self.interarrival_fits: Dict[str, InterArrivalFitResult] = {}
        self._fit_all()

    def _fit_all(self):
        counters = self.df["counter"].unique()
        bands = ["Morning", "Lunch", "Afternoon", "All-Day"]
        for c in counters:
            for b in bands:
                self.fits[(c, b)] = fit_poisson_arrivals(self.df, counter=c, time_band=b)
            self.interarrival_fits[c] = fit_exponential_interarrivals(self.df, counter=c)

    def get_fit(self, counter: str, time_band: str = "All-Day") -> PoissonFitResult:
        return self.fits.get((counter, time_band), fit_poisson_arrivals(self.df, counter, time_band))

    def plot_observed_vs_fitted_pmf(
        self,
        counter: str,
        time_band: str = "Lunch",
        save_path: Optional[str] = None,
    ):
        """
        Generates publication-quality figure of observed histogram vs fitted Poisson PMF.
        """
        sub_df = self.df[self.df["counter"] == counter].copy()
        if time_band == "Morning":
            sub_df = sub_df[(sub_df["slot_index"] >= 0) & (sub_df["slot_index"] < 36)]
        elif time_band == "Lunch":
            sub_df = sub_df[(sub_df["slot_index"] >= 36) & (sub_df["slot_index"] < 72)]
        elif time_band == "Afternoon":
            sub_df = sub_df[(sub_df["slot_index"] >= 72) & (sub_df["slot_index"] < 120)]

        arrivals = sub_df["arrival_count"].to_numpy()
        fit = self.get_fit(counter, time_band)
        lam = fit.lambda_mle

        max_k = int(np.max(arrivals))
        k_range = np.arange(0, max_k + 1)
        fitted_pmf = stats.poisson.pmf(k_range, mu=lam)

        fig, ax = plt.subplots(figsize=(8, 5))
        counts, bins, _ = ax.hist(
            arrivals,
            bins=np.arange(-0.5, max_k + 1.5, 1),
            density=True,
            alpha=0.6,
            color="#2E86AB",
            edgecolor="black",
            label="Empirical Relative Frequency",
        )
        ax.plot(
            k_range,
            fitted_pmf,
            "o-",
            color="#D90429",
            linewidth=2.5,
            markersize=6,
            label=f"Fitted Poisson PMF ($\\lambda={lam:.2f}$)",
        )
        ax.set_title(
            f"Saveetha Canteen Arrivals: {counter} ({time_band} Band)\n"
            f"Chi2={fit.chi2_stat:.2f}, p-val={fit.p_value:.4f}, dof={fit.degrees_of_freedom}",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Number of Arrivals in 5-Minute Slot", fontsize=11)
        ax.set_ylabel("Probability Density", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        return fig
