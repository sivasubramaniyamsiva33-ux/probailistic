"""
Saveetha College Canteen Queue Assistant: Interactive Streamlit Dashboard
Integration Layer combining Module 1 (Probabilistic Methods) and Module 2 (Linear Algebra)
"""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import streamlit as st

# Setup python path so imports resolve from both package and local directory
root_dir = Path(__file__).resolve().parent.parent.parent
queue_dir = Path(__file__).resolve().parent.parent
for d in [str(root_dir), str(queue_dir)]:
    if d not in sys.path:
        sys.path.insert(0, d)

from canteen_queue.config import (
    COUNTERS,
    DEFAULT_SERVERS,
    DEFAULT_SERVICE_RATES,
    QUEUE_STATE_NAMES,
    SLOT_MINUTES,
    START_HOUR,
    TRAFFIC_LIGHT_AMBER_MAX,
    TRAFFIC_LIGHT_GREEN_MAX,
)
from canteen_queue.data.generate_data import load_or_generate_data
from canteen_queue.module1_probabilistic.arrival_model import ArrivalModel
from canteen_queue.module1_probabilistic.bayesian_update import GammaPoissonModel
from canteen_queue.module1_probabilistic.markov_chain import QueueMarkovChain, queue_length_to_state
from canteen_queue.module1_probabilistic.monte_carlo import run_canteen_monte_carlo
from canteen_queue.module1_probabilistic.queue_theory import QueueTheoryMMC, predict_wait_time
from canteen_queue.module1_probabilistic.service_model import ServiceModel
from canteen_queue.module2_linear_algebra.demand_matrix import DemandMatrixBuilder
from canteen_queue.module2_linear_algebra.least_squares_predictor import LeastSquaresPredictor
from canteen_queue.module2_linear_algebra.markov_matrix_analysis import MarkovSpectralAnalysis
from canteen_queue.module2_linear_algebra.recommender import CanteenVisitRecommender
from canteen_queue.module2_linear_algebra.svd_pca_patterns import SVDDemandPatterns

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Saveetha Canteen Queue Assistant",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def get_cached_data() -> pd.DataFrame:
    return load_or_generate_data()


@st.cache_resource
def get_cached_models(df: pd.DataFrame):
    arrival_mod = ArrivalModel(df)
    service_mod = ServiceModel(df)
    ls_predictor = LeastSquaresPredictor(df)
    svd_patterns = SVDDemandPatterns(df)
    markov_spectral = MarkovSpectralAnalysis(df)
    recommender = CanteenVisitRecommender(df)
    markov_chains = {c: QueueMarkovChain(df, counter=c) for c in COUNTERS}
    return {
        "arrival": arrival_mod,
        "service": service_mod,
        "ls_predictor": ls_predictor,
        "svd": svd_patterns,
        "markov_spectral": markov_spectral,
        "recommender": recommender,
        "markov_chains": markov_chains,
    }


def main():
    # Load dataset & pretrained models
    df = get_cached_data()
    models = get_cached_models(df)

    # Header / Branding Banner
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #1D3557, #457B9D); padding: 22px; border-radius: 12px; margin-bottom: 24px; color: white;">
            <h1 style="margin: 0; color: #F1FAEE; font-size: 2.2rem; font-weight: 700;">
                🍽️ Saveetha College Canteen Queue Assistant
            </h1>
            <p style="margin: 6px 0 0 0; color: #A8DADC; font-size: 1.05rem;">
                Smart Queue Prediction, Congestion Forecasting & Waiting-Time Decision Support
                <br><em>Dual Mathematical Architecture: Module 1 (Probabilistic Modeling) & Module 2 (Linear Algebra)</em>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------ SIDEBAR CONTROLS ------------------
    st.sidebar.markdown("### 🎛️ Live Simulation Controls")
    st.sidebar.markdown("Configure current real-time observations:")

    selected_counter = st.sidebar.selectbox("Select Counter", COUNTERS, index=0)

    # Time selector
    time_options = []
    for h in range(8, 18):
        for m in range(0, 60, 5):
            time_options.append(f"{h:02d}:{m:02d}")
    default_time_idx = time_options.index("12:35") if "12:35" in time_options else 55
    selected_time = st.sidebar.select_slider(
        "Current Time of Day", options=time_options, value=time_options[default_time_idx]
    )

    day_options = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    selected_day = st.sidebar.selectbox("Day of Week", day_options, index=1)

    selected_queue = st.sidebar.slider("Observed Queue Length (Customers Ahead)", 0, 30, 8)

    default_c = DEFAULT_SERVERS.get(selected_counter, 1)
    selected_servers = st.sidebar.slider("Active Servers (c)", 1, 4, default_c)

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### ⚡ Dynamic Live Parameters")
    base_mu = DEFAULT_SERVICE_RATES.get(selected_counter, 1.5)
    mu_slider = st.sidebar.slider(
        "Service Rate μ (cust/min/server)",
        min_value=0.4,
        max_value=3.5,
        value=float(base_mu),
        step=0.1,
    )

    # Real-time traffic light calculation
    mmc_pred = predict_wait_time(
        counter=selected_counter,
        time_slot=selected_time,
        current_queue_length=selected_queue,
        servers=selected_servers,
        df=df,
        default_mu=mu_slider,
    )
    expected_wait = mmc_pred["expected_total_wait_min"]

    # Traffic light banner
    if expected_wait < TRAFFIC_LIGHT_GREEN_MAX:
        light_color = "#2A9D8F"
        light_label = "GREEN: Low Wait Time"
        light_icon = "🟢"
        light_desc = "Quick service expected! Great time to grab your meal or beverage."
    elif expected_wait <= TRAFFIC_LIGHT_AMBER_MAX:
        light_color = "#E76F51"
        light_label = "AMBER: Moderate Wait Time"
        light_icon = "🟡"
        light_desc = "Moderate queue detected. Expected wait is reasonable but building."
    else:
        light_color = "#E63946"
        light_label = "RED: High Congestion Alert"
        light_icon = "🔴"
        light_desc = "Heavy queue congestion! Check Tab 3 for best time recommendations."

    st.markdown(
        f"""
        <div style="background-color: {light_color}18; border-left: 6px solid {light_color}; padding: 14px 20px; border-radius: 8px; margin-bottom: 20px;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <h3 style="margin: 0; color: {light_color}; font-size: 1.3rem;">{light_icon} {light_label}</h3>
                    <p style="margin: 4px 0 0 0; color: #333333; font-size: 0.95rem;">{light_desc}</p>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 1.8rem; font-weight: 700; color: {light_color};">{expected_wait:.1f} min</span>
                    <br><span style="font-size: 0.8rem; color: #666666;">90% CI: [{mmc_pred['ci_90_lower_min']:.1f} - {mmc_pred['ci_90_upper_min']:.1f} min]</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------ MAIN NAVIGATION TABS ------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "⏱️ Live Wait-Time Prediction",
            "📊 Crowd Forecast (Markov)",
            "🧭 Best Time to Visit",
            "🔬 Model Insights & Linear Algebra",
            "📐 Theory & Mathematical Foundations",
        ]
    )

    # ==============================================================
    # TAB 1: LIVE WAIT-TIME PREDICTION
    # ==============================================================
    with tab1:
        st.subheader("Multi-Engine Wait Time Prediction Comparison")
        st.markdown(
            "Side-by-side comparison of queue predictions derived from analytical stochastic queue theory, "
            "Bayesian parameter updating, empirical least-squares regression, and Monte Carlo discrete-event simulation."
        )

        # 1. Compute predictions across engines
        # Least Squares prediction
        ls_pred_val = models["ls_predictor"].predict_sample(
            time_slot=selected_time,
            day_of_week=selected_day,
            counter=selected_counter,
            queue_length=selected_queue,
        )

        # Bayesian Live Update
        st.markdown("#### 🧬 Live Bayesian Observation Feed")
        col_b1, col_b2 = st.columns([2, 3])
        with col_b1:
            live_arrivals_input = st.slider(
                "Observed arrivals in the last 5-min slot:",
                min_value=0,
                max_value=25,
                value=int(mmc_pred["arrival_rate_per_min"] * 5),
            )
        bayes_model = GammaPoissonModel.fit_empirical_bayes_prior(
            df, counter=selected_counter, time_slot=selected_time
        )
        bayes_state = bayes_model.update_live([live_arrivals_input])
        updated_lambda = bayes_state.posterior_mean

        bayes_wait_pred = predict_wait_time(
            counter=selected_counter,
            time_slot=selected_time,
            current_queue_length=selected_queue,
            servers=selected_servers,
            df=df,
            default_mu=mu_slider,
            default_lam=updated_lambda,
        )

        with col_b2:
            st.info(
                f"**Prior Arrival Rate:** {bayes_state.prior_mean:.2f} cust/min  |  "
                f"**Updated Posterior Rate:** {bayes_state.posterior_mean:.2f} cust/min\n\n"
                f"**90% Credible Interval:** [{bayes_state.ci_90_lower:.2f}, {bayes_state.ci_90_upper:.2f}] cust/min  |  "
                f"**Predicted Next-Slot Arrivals:** {bayes_state.predictive_mean_next_slot:.1f}"
            )

        # Weighted Ensemble Model
        ensemble_pred = round(
            0.45 * mmc_pred["expected_total_wait_min"]
            + 0.35 * bayes_wait_pred["expected_total_wait_min"]
            + 0.20 * ls_pred_val,
            2,
        )

        # Prediction Metric Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                label="Analytic M/M/c",
                value=f"{mmc_pred['expected_total_wait_min']:.1f} min",
                delta=f"Queue: {mmc_pred['expected_queue_wait_min']:.1f}m",
            )
        with col2:
            st.metric(
                label="Bayesian-Updated M/M/c",
                value=f"{bayes_wait_pred['expected_total_wait_min']:.1f} min",
                delta=f"λ = {updated_lambda:.2f}/min",
            )
        with col3:
            st.metric(
                label="Least-Squares (QR/SVD)",
                value=f"{ls_pred_val:.1f} min",
                delta="Empirical fit",
            )
        with col4:
            st.metric(
                label="⭐ Final Ensemble Estimate",
                value=f"{ensemble_pred:.1f} min",
                delta=f"CI: [{mmc_pred['ci_90_lower_min']:.1f}-{mmc_pred['ci_90_upper_min']:.1f}]m",
            )

        st.markdown("---")

        # Monte Carlo Simulation (N=10,000 runs)
        st.subheader("🎲 Monte Carlo Discrete-Event Queue Simulation (N=10,000 Runs)")
        with st.spinner("Executing 10,000 customer stochastic simulation..."):
            mc_res = run_canteen_monte_carlo(
                arrival_rate=updated_lambda,
                service_rate=mu_slider,
                servers=selected_servers,
                num_runs=10000,
            )

        mc_col1, mc_col2, mc_col3, mc_col4 = st.columns(4)
        mc_col1.metric("P50 Wait (Median)", f"{mc_res.p50_wait_min:.1f} min")
        mc_col2.metric("P90 Wait", f"{mc_res.p90_wait_min:.1f} min")
        mc_col3.metric("P95 Wait", f"{mc_res.p95_wait_min:.1f} min")
        mc_col4.metric("P(Wait > 5 min)", f"{mc_res.prob_wait_gt_5min*100:.1f}%")

        # Interactive Plotly Histogram of Monte Carlo Waits
        fig_mc = go.Figure()
        fig_mc.add_trace(
            go.Histogram(
                x=mc_res.simulated_wait_times,
                histnorm="probability density",
                name="Simulated Wait Time",
                nbinsx=60,
                marker_color="#457B9D",
                opacity=0.7,
            )
        )
        fig_mc.add_vline(
            x=mc_res.p50_wait_min,
            line_dash="dash",
            line_color="#2A9D8F",
            annotation_text=f"P50: {mc_res.p50_wait_min:.1f}m",
            annotation_position="top left",
        )
        fig_mc.add_vline(
            x=mc_res.p90_wait_min,
            line_dash="dash",
            line_color="#E76F51",
            annotation_text=f"P90: {mc_res.p90_wait_min:.1f}m",
            annotation_position="top right",
        )
        fig_mc.add_vline(
            x=mc_res.p95_wait_min,
            line_dash="dash",
            line_color="#E63946",
            annotation_text=f"P95: {mc_res.p95_wait_min:.1f}m",
            annotation_position="top right",
        )
        fig_mc.update_layout(
            title=f"Empirical Waiting Time Distribution for {selected_counter} (N=10,000 Simulated Customers)",
            xaxis_title="Wait Time in Queue (minutes)",
            yaxis_title="Probability Density",
            template="plotly_white",
            height=380,
        )
        st.plotly_chart(fig_mc, use_container_width=True)

    # ==============================================================
    # TAB 2: CROWD FORECAST (MARKOV CHAIN)
    # ==============================================================
    with tab2:
        st.subheader("Discrete-Time Markov Chain Congestion Forecasting")
        st.markdown(
            "State space $\\mathcal{S} = \\{ \\text{Empty, Short, Medium, Long, Overcrowded} \\}$. "
            "Transitions are updated every 5-minute interval based on Maximum Likelihood estimation."
        )

        mc_engine = models["markov_chains"][selected_counter]
        cur_state_idx = queue_length_to_state(selected_queue)
        cur_state_name = QUEUE_STATE_NAMES[cur_state_idx]

        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("Current Queue State", f"{cur_state_name} ({selected_queue} cust)")
        with m_col2:
            forecast_result = mc_engine.forecast_n_steps(selected_queue, n_steps=6)
            st.metric(
                "Expected Steps to Overcrowded",
                f"{forecast_result.expected_steps_to_overcrowded:.1f} steps",
            )
        with m_col3:
            st.metric(
                "Time until Overcrowded",
                f"{forecast_result.expected_minutes_to_overcrowded:.0f} minutes",
            )

        st.markdown("#### Multi-Step Ahead State Probabilities (1 to 12 slots = up to 1 Hour)")
        df_trajectory = mc_engine.compute_multi_horizon_forecast(selected_queue, max_steps=12)

        fig_traj = go.Figure()
        colors = ["#2A9D8F", "#457B9D", "#E76F51", "#F4A261", "#E63946"]
        for s_idx, s_name in enumerate(QUEUE_STATE_NAMES):
            fig_traj.add_trace(
                go.Scatter(
                    x=df_trajectory["Minutes_Ahead"],
                    y=df_trajectory[s_name],
                    mode="lines+markers",
                    name=s_name,
                    line=dict(color=colors[s_idx], width=2.5),
                )
            )
        fig_traj.update_layout(
            title=f"Evolution of State Probability Vector π^(n) over Time for {selected_counter}",
            xaxis_title="Minutes Ahead into Future",
            yaxis_title="State Probability",
            yaxis_range=[0, 1.05],
            template="plotly_white",
            height=380,
        )
        st.plotly_chart(fig_traj, use_container_width=True)

        # Transition matrix heatmap & stationary distribution
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("##### 5×5 Estimated Transition Probability Matrix P")
            P_mat = mc_engine.P
            fig_p = px.imshow(
                P_mat,
                x=QUEUE_STATE_NAMES,
                y=QUEUE_STATE_NAMES,
                labels=dict(x="Destination State", y="Source State", color="P_ij"),
                color_continuous_scale="Viridis",
                text_auto=".2f",
            )
            fig_p.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_p, use_container_width=True)

        with c_right:
            st.markdown("##### Long-Term Stationary Distribution π (Power Iteration)")
            spec_res = models["markov_spectral"].results[selected_counter]
            fig_stat = px.bar(
                x=QUEUE_STATE_NAMES,
                y=spec_res.stationary_dist_power_iter * 100.0,
                labels={"x": "Queue State", "y": "Stationary Probability (%)"},
                color=QUEUE_STATE_NAMES,
                color_discrete_sequence=colors,
            )
            fig_stat.update_layout(
                height=350,
                showlegend=False,
                margin=dict(l=20, r=20, t=30, b=20),
                title=f"Spectral Gap γ = {spec_res.spectral_gap:.3f} | τ_mix = {spec_res.mixing_relaxation_minutes:.1f}m",
            )
            st.plotly_chart(fig_stat, use_container_width=True)

    # ==============================================================
    # TAB 3: BEST TIME TO VISIT (RECOMMENDER)
    # ==============================================================
    with tab3:
        st.subheader("🧭 Best Time to Visit & Counter Recommender")
        st.markdown(
            "Projects today's demand vector onto the principal subspace $V_k$ and identifies "
            "most similar historical operating days to pinpoint lowest-wait upcoming slots."
        )

        rec_res = models["recommender"].recommend(
            current_time_slot=selected_time,
            target_counter=selected_counter,
            current_queue_len=selected_queue,
            num_recommendations=3,
        )

        r_col1, r_col2 = st.columns([1, 1])
        with r_col1:
            st.markdown(
                f"""
                <div style="background-color: #EBF2FA; padding: 18px; border-radius: 10px; border-left: 5px solid #1D3557;">
                    <h4 style="margin: 0 0 10px 0; color: #1D3557;">🎯 Immediate Recommendations</h4>
                    <p><strong>Target Counter:</strong> {selected_counter} (Expected Wait: {rec_res.current_slot_expected_wait:.1f} min)</p>
                    <p><strong>Best Fast Alternative:</strong> <span style="color: #2A9D8F; font-weight: bold;">{rec_res.best_counter_alternative}</span> (Expected Wait: {rec_res.best_counter_expected_wait:.1f} min)</p>
                    <p><strong>Most Similar Historical Day:</strong> Day #{rec_res.most_similar_day_index} (Cosine Similarity: <strong>{rec_res.cosine_similarity_score*100:.1f}%</strong>)</p>
                    <p><strong>Subspace Coordinates (z):</strong> <code>{list(rec_res.subspace_coordinates)}</code></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with r_col2:
            st.markdown("#### 🌟 Top 3 Recommended Upcoming Low-Congestion Slots")
            rec_df = pd.DataFrame(rec_res.recommended_slots)
            st.table(rec_df)

        st.markdown("---")
        st.markdown("#### Diurnal Demand Heatmap across All Counters")
        builder = DemandMatrixBuilder(df)
        D_all = builder.build_slot_counter_matrix()

        slot_labels = [f"{START_HOUR + (i * 5)//60:02d}:{(i * 5)%60:02d}" for i in range(120)]
        fig_heat = px.imshow(
            D_all.T,
            x=slot_labels,
            y=COUNTERS,
            labels=dict(x="Time of Day", y="Counter", color="Avg Arrivals"),
            color_continuous_scale="Thermal",
            aspect="auto",
        )
        fig_heat.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_heat, use_container_width=True)

    # ==============================================================
    # TAB 4: MODEL INSIGHTS & LINEAR ALGEBRA
    # ==============================================================
    with tab4:
        st.subheader("Linear Algebra & Dimensionality Insights")
        st.markdown(
            "Spectral decomposition of the Day × Time-Slot demand matrix $D \\in \\mathbb{R}^{60 \\times 120}$ "
            "and numerical stability analysis across solvers."
        )

        svd_res = models["svd"].results[selected_counter]

        ins_col1, ins_col2, ins_col3, ins_col4 = st.columns(4)
        ins_col1.metric("Matrix Rank", f"{svd_res.shape[0]} × {svd_res.shape[1]}")
        ins_col2.metric("Components for 90% Energy", f"k = {svd_res.components_for_90pct}")
        ins_col3.metric("Components for 95% Energy", f"k = {svd_res.components_for_95pct}")
        ins_col4.metric(
            "Top-1 Mode Energy",
            f"{svd_res.explained_variance_ratio[0]*100:.1f}%",
        )

        # Scree and Canonical Modes Plots
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            fig_scree = go.Figure()
            k_axis = np.arange(1, 16)
            fig_scree.add_trace(
                go.Scatter(
                    x=k_axis,
                    y=svd_res.cumulative_energy[:15] * 100.0,
                    mode="lines+markers",
                    name="Cumulative Energy %",
                    line=dict(color="#E63946", width=2.5),
                )
            )
            fig_scree.add_hline(
                y=90.0, line_dash="dash", line_color="#2A9D8F", annotation_text="90% Threshold"
            )
            fig_scree.update_layout(
                title=f"SVD Energy Capture (%): {selected_counter}",
                xaxis_title="Principal Components k",
                yaxis_title="Energy %",
                template="plotly_white",
                height=350,
            )
            st.plotly_chart(fig_scree, use_container_width=True)

        with p_col2:
            fig_modes = go.Figure()
            slot_indices = np.arange(120)
            slot_times = [f"{START_HOUR + (i * 5)//60:02d}:{(i * 5)%60:02d}" for i in range(120)]
            mode_names = ["Mode 1 (Baseline)", "Mode 2 (Lunch Rush)", "Mode 3 (Class Breaks)"]
            mode_colors = ["#1D3557", "#E63946", "#2A9D8F"]
            for idx in range(3):
                fig_modes.add_trace(
                    go.Scatter(
                        x=slot_times[::4],
                        y=svd_res.top_v_patterns[::4, idx],
                        mode="lines+markers",
                        name=mode_names[idx],
                        line=dict(color=mode_colors[idx], width=2.2),
                    )
                )
            fig_modes.update_layout(
                title="Top 3 Right Singular Vectors v1, v2, v3 (Diurnal Modes)",
                xaxis_title="Time of Day",
                yaxis_title="Mode Amplitude",
                template="plotly_white",
                height=350,
            )
            st.plotly_chart(fig_modes, use_container_width=True)

        st.markdown("#### Solver Comparison Table: Conditioning & Numerical Accuracy")
        comp_table = models["ls_predictor"].get_comparison_summary()
        st.table(comp_table)

    # ==============================================================
    # TAB 5: THEORY & MATHEMATICAL FOUNDATIONS
    # ==============================================================
    with tab5:
        st.subheader("Comprehensive Mathematical Formulations & Derivations")
        st.markdown(
            "Detailed academic derivations of all probabilistic models and linear algebra methods."
        )

        with st.expander("1. Poisson Arrival Process & Goodness-of-Fit Testing", expanded=True):
            st.markdown(
                r"""
                Arrivals $K$ within a slot duration $\tau = 5$ min are modeled as a homogeneous Poisson process:
                $$P(K = k) = \frac{(\lambda \tau)^k e^{-\lambda \tau}}{k!}, \quad k \in \{0, 1, 2, \dots\}$$
                - **Maximum Likelihood Estimator**: $\hat{\lambda} = \frac{1}{N \tau} \sum_{i=1}^N k_i$.
                - **Goodness-of-fit Chi-Square Statistic**:
                  $$\chi^2 = \sum_{j=1}^m \frac{(O_j - E_j)^2}{E_j} \sim \chi^2_{m - 1 - 1}$$
                  Tail bins are pooled such that expected frequency $E_j \ge 5$.
                - **Exponential Inter-Arrival Times**: $\Delta t \sim \text{Exp}(\lambda)$ with CDF $F(t) = 1 - e^{-\lambda t}$.
                  Validated with the two-sided Kolmogorov-Smirnov test: $D = \sup_t |F_n(t) - F(t)|$.
                """
            )

        with st.expander("2. Service Time Modeling: Exponential vs. Gamma & AIC Selection"):
            st.markdown(
                r"""
                Service times $X$ are compared between Exponential and Gamma distributions via Maximum Likelihood Estimation:
                - **Exponential Distribution** ($k=1$):
                  $$f(x; \mu) = \mu e^{-\mu x}, \quad \ln L = n \ln \mu - \mu \sum_{i=1}^n x_i$$
                - **Gamma Distribution** ($k=2$):
                  $$f(x; \alpha, \beta) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha - 1} e^{-\beta x}$$
                  $$\ln L = n(\alpha \ln \beta - \ln \Gamma(\alpha)) + (\alpha - 1)\sum \ln x_i - \beta \sum x_i$$
                - **Model Selection**: Akaike Information Criterion $\text{AIC} = 2k - 2\ln L$. The model with lower AIC is selected.
                """
            )

        with st.expander("3. Queueing Theory: M/M/c Closed Forms & Little's Law"):
            st.markdown(
                r"""
                Traffic intensity $r = \frac{\lambda}{\mu}$ and server utilization $\rho = \frac{\lambda}{c\mu}$.
                Stability requirement: $\rho < 1$.
                - **Normalizing Probability $P_0$**:
                  $$P_0 = \left[ \sum_{n=0}^{c-1} \frac{(c\rho)^n}{n!} + \frac{(c\rho)^c}{c!(1-\rho)} \right]^{-1}$$
                - **Erlang-C Formula** (Probability of waiting in queue):
                  $$P_C = \mathbb{P}(W_q > 0) = P_0 \frac{(c\rho)^c}{c!(1-\rho)}$$
                - **Queue Length and Wait Times**:
                  $$L_q = P_C \frac{\rho}{1-\rho}, \quad W_q = \frac{L_q}{\lambda} = \frac{P_C}{c\mu - \lambda}, \quad W = W_q + \frac{1}{\mu}$$
                - **Little's Law Check**: Verifies conservation laws $L = \lambda W$ and $L_q = \lambda W_q$ with $|L - \lambda W| < 10^{-6}$.
                """
            )

        with st.expander("4. Bayesian Conjugate Updating: Gamma-Poisson Model"):
            st.markdown(
                r"""
                - **Conjugate Prior**: $\lambda \sim \text{Gamma}(\alpha_0, \beta_0)$ where $\mathbb{E}[\lambda] = \frac{\alpha_0}{\beta_0}$ and $\text{Var}[\lambda] = \frac{\alpha_0}{\beta_0^2}$.
                - **Likelihood**: Total arrivals $K = \sum k_i$ over total duration $T$ follow $\text{Poisson}(\lambda T)$.
                - **Posterior Distribution**:
                  $$\lambda | K \sim \text{Gamma}(\alpha_{\text{post}}, \beta_{\text{post}}), \quad \alpha_{\text{post}} = \alpha_0 + K, \quad \beta_{\text{post}} = \beta_0 + T$$
                  $$\mathbb{E}[\lambda | K] = \frac{\alpha_0 + K}{\beta_0 + T}$$
                - **Posterior Predictive Distribution** (for next slot of duration $\tau$):
                  $$Y \sim \text{Negative-Binomial}\left(r = \alpha_{\text{post}}, p = \frac{\beta_{\text{post}}}{\beta_{\text{post}} + \tau}\right)$$
                """
            )

        with st.expander("5. Linear Algebra: QR, SVD Pseudo-Inverse & Conditioning"):
            st.markdown(
                r"""
                Given overdetermined design matrix $X \in \mathbb{R}^{m \times d}$ ($m \gg d$):
                - **Normal Equations**: $\beta = (X^T X)^{-1} X^T y$.
                  $\text{cond}(X^T X) = (\text{cond}(X))^2$. When $X$ has condition number $10^4$, $X^T X$ has condition number $10^8$, squaring rounding errors.
                - **Modified Gram-Schmidt QR**: $X = Q R$, where $Q^T Q = I_d$ and $R$ is upper triangular.
                  Solves $R \beta = Q^T y$ via back-substitution. Operates with $\text{cond}(R) = \text{cond}(X)$, preserving precision.
                - **SVD Pseudo-Inverse**: $X = U \Sigma V^T \implies \beta = V \Sigma^+ U^T y$.
                - **Ridge Regularization**: $\beta_{\text{ridge}} = (X^T X + \lambda I)^{-1} X^T y$.
                """
            )

        with st.expander("6. Markov Spectral Analysis & Perron-Frobenius Theorem"):
            st.markdown(
                r"""
                For row-stochastic matrix $P$ ($P \mathbf{1} = \mathbf{1}$, $\sum_j P_{ij} = 1$):
                - **Perron-Frobenius Theorem**: Spectral radius $\rho(P) = 1$. Dominant eigenvalue $\lambda_1 = 1$.
                - **Stationary Distribution**: Left eigenvector $\pi P = \pi$ with $\sum \pi_i = 1$.
                - **Power Iteration**: $\pi^{(k+1)} = \pi^{(k)} P$. Convergence error $\| \pi^{(k)} - \pi \| = \mathcal{O}(|\lambda_2|^k)$.
                - **Spectral Gap & Mixing Rate**: $\gamma = 1 - |\lambda_2|$ where $|\lambda_2|$ is the second-largest eigenvalue modulus.
                  Relaxation time: $\tau_{\text{rel}} \approx \frac{1}{1 - |\lambda_2|}$.
                """
            )


if __name__ == "__main__":
    main()
