"""
Service Process Modeling: Exponential vs. Gamma/Erlang MLE & AIC/BIC Model Selection
Saveetha College Canteen Queue Assistant
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln


@dataclass
class ServiceFitResult:
    counter: str
    sample_size: int
    mean_seconds: float
    std_seconds: float
    # Exponential parameters
    mu_exp_per_sec: float
    mu_exp_per_min: float
    log_lik_exp: float
    aic_exp: float
    bic_exp: float
    # Gamma parameters (Erlang generalization)
    alpha_gamma: float          # shape k
    beta_gamma: float           # rate (1/scale)
    scale_gamma: float          # scale theta = 1 / beta
    log_lik_gamma: float
    aic_gamma: float
    bic_gamma: float
    # Best distribution by AIC
    best_distribution: str      # "Gamma" or "Exponential"
    delta_aic: float            # AIC_exp - AIC_gamma (positive means Gamma is better)


def fit_service_distributions(df: pd.DataFrame, counter: str) -> ServiceFitResult:
    """
    Fits Exponential and Gamma distributions to empirical service times using MLE.
    Compares models using Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC).

    Mathematical Formulations:
    1. Exponential Distribution:
       f(x; mu) = mu * e^(-mu * x), x >= 0
       MLE: mu_hat = n / sum(x_i) = 1 / mean(x)
       ln L_exp = n * ln(mu) - mu * sum(x_i)
       AIC_exp = 2(1) - 2 * ln L_exp

    2. Gamma Distribution:
       f(x; alpha, beta) = (beta^alpha / Gamma(alpha)) * x^(alpha - 1) * e^(-beta * x)
       ln L_gamma = n * (alpha * ln(beta) - ln(Gamma(alpha))) + (alpha - 1) * sum(ln(x_i)) - beta * sum(x_i)
       AIC_gamma = 2(2) - 2 * ln L_gamma
    """
    sub_df = df[df["counter"] == counter].copy()
    raw_services = sub_df["service_time_seconds"].dropna().to_numpy(dtype=float)
    # Filter positive valid service values
    services = raw_services[raw_services > 0]
    n = len(services)
    if n == 0:
        raise ValueError(f"No valid service time data found for counter: {counter}")

    mean_sec = float(np.mean(services))
    std_sec = float(np.std(services, ddof=1))

    # --- 1. Exponential Fit (k=1 param) ---
    mu_exp = 1.0 / max(1e-4, mean_sec)  # rate per second
    mu_exp_min = mu_exp * 60.0          # rate per minute
    # Log-likelihood: n * ln(mu) - mu * sum(x)
    log_lik_exp = float(n * np.log(mu_exp) - mu_exp * np.sum(services))
    k_exp = 1
    aic_exp = float(2 * k_exp - 2 * log_lik_exp)
    bic_exp = float(k_exp * np.log(n) - 2 * log_lik_exp)

    # --- 2. Gamma Fit (k=2 params: shape alpha, rate beta = 1/scale) ---
    # SciPy fit with fixed loc=0
    shape_alpha, loc, scale_theta = stats.gamma.fit(services, floc=0)
    beta_gamma = 1.0 / max(1e-6, scale_theta)

    # Analytic exact Log-Likelihood for Gamma
    sum_ln_x = float(np.sum(np.log(services)))
    sum_x = float(np.sum(services))
    log_lik_gamma = float(
        n * (shape_alpha * np.log(beta_gamma) - gammaln(shape_alpha))
        + (shape_alpha - 1.0) * sum_ln_x
        - beta_gamma * sum_x
    )
    k_gamma = 2
    aic_gamma = float(2 * k_gamma - 2 * log_lik_gamma)
    bic_gamma = float(k_gamma * np.log(n) - 2 * log_lik_gamma)

    # Comparison
    delta_aic = aic_exp - aic_gamma
    best_dist = "Gamma" if delta_aic > 0 else "Exponential"

    return ServiceFitResult(
        counter=counter,
        sample_size=n,
        mean_seconds=round(mean_sec, 2),
        std_seconds=round(std_sec, 2),
        mu_exp_per_sec=round(mu_exp, 5),
        mu_exp_per_min=round(mu_exp_min, 4),
        log_lik_exp=round(log_lik_exp, 2),
        aic_exp=round(aic_exp, 2),
        bic_exp=round(bic_exp, 2),
        alpha_gamma=round(shape_alpha, 4),
        beta_gamma=round(beta_gamma, 5),
        scale_gamma=round(scale_theta, 4),
        log_lik_gamma=round(log_lik_gamma, 2),
        aic_gamma=round(aic_gamma, 2),
        bic_gamma=round(bic_gamma, 2),
        best_distribution=best_dist,
        delta_aic=round(delta_aic, 2),
    )


class ServiceModel:
    """
    Encapsulates service time distribution evaluation and density visualization
    across all canteen counters.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.fits: Dict[str, ServiceFitResult] = {}
        for c in df["counter"].unique():
            self.fits[c] = fit_service_distributions(df, counter=c)

    def get_summary_table(self) -> pd.DataFrame:
        """
        Returns a clean formatted comparison table of Exponential vs Gamma fits.
        """
        rows = []
        for c, res in self.fits.items():
            rows.append(
                {
                    "Counter": c,
                    "Mean (sec)": res.mean_seconds,
                    "Std (sec)": res.std_seconds,
                    "Exp Rate (cust/min)": res.mu_exp_per_min,
                    "Exp AIC": res.aic_exp,
                    "Gamma Shape (α)": res.alpha_gamma,
                    "Gamma Scale (θ)": res.scale_gamma,
                    "Gamma AIC": res.aic_gamma,
                    "Delta AIC (Exp - Gamma)": res.delta_aic,
                    "Best Fit": res.best_distribution,
                }
            )
        return pd.DataFrame(rows)

    def plot_service_density_comparison(
        self,
        counter: str,
        save_path: Optional[str] = None,
    ):
        """
        Plots the empirical service time distribution along with the fitted
        Exponential and Gamma probability density functions.
        """
        sub_df = self.df[self.df["counter"] == counter]
        services = sub_df["service_time_seconds"].dropna().to_numpy()
        fit = self.fits[counter]

        x_grid = np.linspace(0.1, np.percentile(services, 99.5), 500)
        exp_pdf = stats.expon.pdf(x_grid, scale=1.0 / fit.mu_exp_per_sec)
        gamma_pdf = stats.gamma.pdf(x_grid, a=fit.alpha_gamma, scale=fit.scale_gamma)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(
            services,
            bins=40,
            density=True,
            alpha=0.55,
            color="#457B9D",
            edgecolor="white",
            label="Empirical Service Times",
        )
        ax.plot(
            x_grid,
            exp_pdf,
            "--",
            color="#E63946",
            linewidth=2.2,
            label=f"Exponential PDF (AIC={fit.aic_exp:.1f})",
        )
        ax.plot(
            x_grid,
            gamma_pdf,
            "-",
            color="#1D3557",
            linewidth=2.6,
            label=f"Gamma PDF (α={fit.alpha_gamma:.2f}, AIC={fit.aic_gamma:.1f})",
        )

        ax.set_title(
            f"Saveetha Canteen: Service Time Model Fit ({counter})\n"
            f"Best Model: {fit.best_distribution} (ΔAIC = {fit.delta_aic:.1f})",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Service Time (seconds)", fontsize=11)
        ax.set_ylabel("Probability Density", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        return fig
