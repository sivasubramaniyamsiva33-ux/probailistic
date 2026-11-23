"""
Generates static figures, evaluation tables, and holdout benchmarks for academic reports.
Saveetha College Canteen Queue Assistant
"""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Path setup
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from canteen_queue.config import FIGURES_DIR, REPORTS_DIR
from canteen_queue.data.generate_data import load_or_generate_data
from canteen_queue.module1_probabilistic.arrival_model import ArrivalModel
from canteen_queue.module1_probabilistic.bayesian_update import BayesianArrivalTracker
from canteen_queue.module1_probabilistic.markov_chain import QueueMarkovChain
from canteen_queue.module1_probabilistic.monte_carlo import plot_simulation_histogram, run_canteen_monte_carlo
from canteen_queue.module1_probabilistic.queue_theory import predict_wait_time
from canteen_queue.module1_probabilistic.service_model import ServiceModel
from canteen_queue.module2_linear_algebra.demand_matrix import DemandMatrixBuilder
from canteen_queue.module2_linear_algebra.least_squares_predictor import LeastSquaresPredictor
from canteen_queue.module2_linear_algebra.markov_matrix_analysis import MarkovSpectralAnalysis
from canteen_queue.module2_linear_algebra.svd_pca_patterns import SVDDemandPatterns


def evaluate_on_held_out_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluates wait-time predictors on a held-out test set consisting of the last 10 days
    (Days 50 to 59), trained strictly on Days 0 to 49.
    """
    train_df = df[df["day_index"] < 50].copy()
    test_df = df[df["day_index"] >= 50].copy()

    # Train Least Squares on Train Set
    ls_train = LeastSquaresPredictor(train_df)

    y_test = test_df["observed_wait_time"].to_numpy()

    # Predictions
    preds = {}

    # 1. Least Squares Methods
    for method in [
        "Normal Equations",
        "Modified Gram-Schmidt QR",
        "SVD Pseudo-Inverse",
        "Ridge Regression",
    ]:
        beta = ls_train.models[method].beta
        # Build design matrix for test_df
        X_test, _ = ls_train._build_design_matrix(test_df)
        preds[method] = np.maximum(0.2, X_test @ beta)

    # 2. Analytic M/M/c
    mmc_preds = []
    for _, row in test_df.iterrows():
        p = predict_wait_time(
            counter=row["counter"],
            time_slot=row["time_slot"],
            current_queue_length=int(row["queue_length_at_arrival"]),
            df=train_df,
        )
        mmc_preds.append(p["expected_total_wait_min"])
    preds["Analytic M/M/c"] = np.array(mmc_preds)

    # 3. Bayesian M/M/c
    tracker = BayesianArrivalTracker(train_df)
    bayes_preds = []
    for _, row in test_df.iterrows():
        state = tracker.live_update_counter(
            counter=row["counter"],
            time_slot=row["time_slot"],
            observed_arrivals=[int(row["arrival_count"])],
        )
        p = predict_wait_time(
            counter=row["counter"],
            time_slot=row["time_slot"],
            current_queue_length=int(row["queue_length_at_arrival"]),
            default_lam=state.posterior_mean,
        )
        bayes_preds.append(p["expected_total_wait_min"])
    preds["Bayesian-Updated M/M/c"] = np.array(bayes_preds)

    # 4. Ensemble
    preds["Weighted Ensemble"] = (
        0.40 * preds["Analytic M/M/c"]
        + 0.35 * preds["Bayesian-Updated M/M/c"]
        + 0.25 * preds["Modified Gram-Schmidt QR"]
    )

    # Evaluate metrics
    results = []
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)

    for name, y_hat in preds.items():
        residuals = y_test - y_hat
        mae = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        ss_res = np.sum(residuals ** 2)
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

        results.append(
            {
                "Model / Predictor": name,
                "MAE (min)": round(mae, 3),
                "RMSE (min)": round(rmse, 3),
                "R² Score": round(r2, 4),
            }
        )

    eval_df = pd.DataFrame(results).sort_values(by="RMSE (min)")
    return eval_df


def generate_all_figures_and_tables():
    """
    Generates and saves all required plots and evaluation tables.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_or_generate_data()

    print("[1/9] Plotting Observed vs Fitted Poisson PMF...")
    am = ArrivalModel(df)
    am.plot_observed_vs_fitted_pmf("Meals", "Lunch", save_path=str(FIGURES_DIR / "pmf_observed_vs_fitted.png"))

    print("[2/9] Plotting Service Distribution Comparison (Exponential vs Gamma)...")
    sm = ServiceModel(df)
    sm.plot_service_density_comparison("Meals", save_path=str(FIGURES_DIR / "service_density_comparison.png"))

    print("[3/9] Plotting Bayesian Prior vs Posterior Evolution...")
    tracker = BayesianArrivalTracker(df)
    tracker.live_update_counter("Meals", "12:30", [12, 16, 14])
    tracker.plot_prior_vs_posterior("Meals", save_path=str(FIGURES_DIR / "bayesian_prior_vs_posterior.png"))

    print("[4/9] Plotting Monte Carlo Waiting Time Histogram (N=10,000)...")
    mc_res = run_canteen_monte_carlo(arrival_rate=1.2, service_rate=1.0, servers=2, num_runs=10000, seed=42)
    plot_simulation_histogram(mc_res, save_path=str(FIGURES_DIR / "monte_carlo_histogram.png"))

    print("[5/9] Plotting Markov Stationary Distribution...")
    msa = MarkovSpectralAnalysis(df)
    msa.plot_stationary_distribution("Meals", save_path=str(FIGURES_DIR / "markov_stationary_distribution.png"))

    print("[6/9] Plotting Demand Matrix Heatmap D...")
    dmb = DemandMatrixBuilder(df)
    dmb.plot_demand_matrix_heatmap("Meals", save_path=str(FIGURES_DIR / "demand_matrix_heatmap.png"))

    print("[7/9] Plotting SVD Scree & Energy Capture...")
    svd_patterns = SVDDemandPatterns(df)
    svd_patterns.plot_scree_and_energy("Meals", save_path=str(FIGURES_DIR / "svd_scree_energy.png"))

    print("[8/9] Plotting Canonical Diurnal Modes (Top 3 Singular Vectors)...")
    svd_patterns.plot_canonical_daily_patterns("Meals", save_path=str(FIGURES_DIR / "canonical_daily_modes.png"))
    svd_patterns.plot_reconstruction_error("Meals", save_path=str(FIGURES_DIR / "eckart_young_reconstruction_error.png"))

    print("[9/9] Running Hold-Out Test Set Evaluation (Days 50-59)...")
    eval_df = evaluate_on_held_out_days(df)
    eval_path = REPORTS_DIR / "model_evaluation_table.csv"
    eval_df.to_csv(eval_path, index=False)
    print("\n=== Hold-Out Test Set Performance (Days 50-59) ===")
    print(eval_df.to_string(index=False))

    # Plot Holdout Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(eval_df))
    width = 0.35
    ax.bar(x - width/2, eval_df["MAE (min)"], width, label="MAE (min)", color="#2A9D8F", edgecolor="black")
    ax.bar(x + width/2, eval_df["RMSE (min)"], width, label="RMSE (min)", color="#E76F51", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(eval_df["Model / Predictor"], rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Error in Minutes", fontsize=11)
    ax.set_title("Saveetha Canteen: Model Performance Comparison on Held-Out Test Set (Days 50-59)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "holdout_model_evaluation.png", dpi=300)
    plt.close()

    print(f"\nAll report assets successfully written to {FIGURES_DIR} and {REPORTS_DIR}")


if __name__ == "__main__":
    generate_all_figures_and_tables()
