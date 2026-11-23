# Smart Queue Prediction and Waiting-Time Assistant for College Canteens
### Saveetha College Canteen Decision Support System
*Dual Mathematical Architecture: Stochastic Queueing Theory & Linear Algebra Decomposition*

---

## 📌 Project Overview
The **Saveetha College Canteen Queue Assistant** is a complete, runnable academic project and decision-support system engineered to predict waiting times, forecast congestion levels, and recommend optimal visiting times for university dining halls.

The platform partitions its predictive and analytical intelligence into two mathematically rigorous engines:
1. **Module 1 (Probabilistic Methods)**:
   - **Arrival Modeling**: Non-homogeneous Poisson process fitted to 5-minute slots across diurnal bands; $\chi^2$ goodness-of-fit with tail pooling ($E_i \ge 5$); Exponential inter-arrival modeling verified via Kolmogorov-Smirnov (KS) tests.
   - **Service Modeling**: Maximum Likelihood Estimation (MLE) of Exponential vs. Gamma distributions; model selection via Akaike Information Criterion ($\Delta \text{AIC} > 120$ favor Gamma).
   - **Pure Queueing Theory ($M/M/1$ and $M/M/c$)**: From-scratch derivation without third-party queue libraries; exact $P_0$, Erlang-C waiting probability $P_C$, $L_q$, $W_q$, $W$, and $L$; verified against Little's Law ($|L - \lambda W| < 10^{-6}$).
   - **Conjugate Gamma-Poisson Bayesian Updating**: Real-time epistemic updating of arrival rate $\lambda$ from live slot counts; 90% credible intervals and Negative Binomial posterior predictive distributions.
   - **Monte Carlo Simulation ($N=10,000$ Runs)**: Discrete-event queue simulation tracking $P_{50}, P_{90}, P_{95}$ wait times and tail probabilities $\mathbb{P}(\text{Wait} > 5\text{m})$.
   - **Markov Chain Congestion States**: 5-state discrete-time Markov chain ($\text{Empty, Short, Medium, Long, Overcrowded}$); $n$-step forecasting $\pi^{(n)} = \pi^{(0)} P^n$; absorbing Markov first hitting times.

2. **Module 2 (Linear Algebra)**:
   - **Demand Matrix Analysis**: Matrix $D \in \mathbb{R}^{60 \times 120}$ (Days $\times$ Slots); missing value imputation; matrix rank $\text{rank}(D)$, condition number $\kappa(D)$, and Frobenius norm $\|D\|_F$.
   - **Least-Squares Solvers**: Overdetermined design matrix $X \in \mathbb{R}^{21600 \times 13}$ (cyclical harmonics, dummy days/counters, queue length, intercept). Solved with:
     1. Normal Equations: $\beta = (X^T X)^{-1} X^T y$
     2. Modified Gram-Schmidt (MGS) QR Decomposition: $X = QR$, $R\beta = Q^T y$ via manual back-substitution
     3. SVD Pseudo-Inverse: $X = U \Sigma V^T \implies \beta = V \Sigma^+ U^T y$
     4. Ridge Regularization: $\beta_{\text{ridge}} = (X^T X + \lambda I)^{-1} X^T y$
     - Rigorous condition number analysis: $\kappa(X^T X) = (\kappa(X))^2 \implies 2.02 \times 10^5$.
   - **SVD / PCA Canonical Diurnal Modes**: Singular value spectrum (scree plot); $k=2$ components capture $> 92\%$ energy; top 3 right singular vectors ($v_1$: baseline, $v_2$: lunch surge, $v_3$: class breaks); Eckart-Young-Mirsky low-rank theorem verification.
   - **Spectral Eigen-Analysis of Markov Matrix**: Perron-Frobenius theorem ($\rho(P) = 1, \lambda_1 = 1$); manual Power Iteration converging to stationary distribution $\pi$; second-largest eigenvalue modulus ($|\lambda_2|$) governing mixing relaxation time $\tau_{\text{rel}} \approx 13$ min.
   - **Subspace Projection Recommender**: Orthogonal projection onto principal subspace $V_k$; cosine similarity matching against 60 historical days; top 3 low-wait recommendations and fast-counter alternatives.

3. **Integration Layer (Streamlit Dashboard)**:
   - Interactive 5-tab web interface featuring real-time traffic-light indicators (🟢 Green $< 5$ min, 🟡 Amber $5-12$ min, 🔴 Red $> 12$ min), live Bayesian observation sliders, interactive Plotly charts, and LaTeX equations.

---

## 🛠️ Technology Stack
- **Language**: Python 3.11+ (verified on Python 3.14)
- **Numerical & Scientific Libraries**: NumPy, SciPy, Pandas
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Web Dashboard**: Streamlit
- **Testing**: pytest (30 unit tests, 100% pass rate)
- **APIs**: No external paid APIs. 100% reproducible synthetic generation.

---

## 📁 Project Directory Structure
```text
PROBABILISTIC PROJECT/
├── config.py                      # Global configuration parameters & seeds
├── requirements.txt               # Package dependencies
├── implementation_plan.md         # Detailed architectural blueprint
├── README.md                      # Comprehensive user guide & documentation
├── canteen_queue/
│   ├── config.py                  # Core configuration constants
│   ├── data/
│   │   ├── __init__.py
│   │   ├── generate_data.py       # 60-day synthetic generator & cleaning pipeline
│   │   ├── canteen_records_raw.csv# 21,600 raw records with 5% anomalies
│   │   └── canteen_records_clean.csv # Cleaned records (imputed & validated)
│   ├── module1_probabilistic/
│   │   ├── __init__.py
│   │   ├── arrival_model.py       # Poisson MLE, Chi-Square GOF, Exponential KS-test
│   │   ├── service_model.py       # Exponential vs Gamma MLE & AIC/BIC selection
│   │   ├── queue_theory.py        # M/M/1 & M/M/c from scratch, Erlang-C, Little's Law
│   │   ├── bayesian_update.py     # Gamma-Poisson conjugate model & NegBin predictive
│   │   ├── monte_carlo.py         # Discrete-event simulation (N=10,000), P50/P90/P95
│   │   └── markov_chain.py        # 5-state Markov chain, n-step forecasting, hitting time
│   ├── module2_linear_algebra/
│   │   ├── __init__.py
│   │   ├── demand_matrix.py       # Demand matrix D (60x120), rank, norms, condition number
│   │   ├── least_squares_predictor.py # Normal Eq, MGS QR, SVD pinv, Ridge, conditioning
│   │   ├── svd_pca_patterns.py    # SVD diurnal modes, scree plot, Eckart-Young theorem
│   │   ├── markov_matrix_analysis.py # Spectral eigen-analysis, manual power iteration
│   │   └── recommender.py         # Subspace projection, cosine similarity matching
│   └── app/
│       ├── __init__.py
│       └── streamlit_app.py       # 5-tab interactive Streamlit dashboard
├── tests/
│   ├── __init__.py
│   ├── test_arrival_service.py    # Tests for Poisson, Chi-square, Exponential KS, Gamma AIC
│   ├── test_queue_theory.py       # Erlang-C textbook values, Little's Law verification
│   ├── test_bayesian_update.py    # Conjugate updating properties, posterior predictive
│   ├── test_monte_carlo.py        # Discrete-event simulation convergence to analytical Wq
│   ├── test_markov_chain.py       # Stochastic matrix row sums = 1, absorption times
│   ├── test_demand_matrix.py      # Matrix rank, shape, imputation, conditioning
│   ├── test_least_squares.py      # Agreement between Normal Eq, MGS QR, and SVD
│   ├── test_svd_pca.py            # Eckart-Young monotonic error decrease, V^T V = I
│   ├── test_markov_matrix_analysis.py # Power iteration agreement with numpy.linalg.eig
│   └── test_recommender.py        # Subspace projection & recommendation ranking
└── reports/
    ├── generate_report_assets.py  # Generates static plots and hold-out evaluation
    ├── report.md                  # Comprehensive academic report with LaTeX derivations
    ├── model_evaluation_table.csv # Holdout test set performance comparison
    └── figures/                   # 9 high-resolution publication figures
```

---

## 🚀 Setup & Execution Guide

### 1. Installation
Clone or navigate to the workspace directory and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Dataset
Generate 60 days of operational records across all 3 counters (Meals, Snacks, Beverages):
```bash
python data/generate_data.py
```
*Output: 21,600 raw records saved to `data/canteen_records_raw.csv` and cleaned records saved to `data/canteen_records_clean.csv`.*

### 3. Run the Automated Test Suite
Execute the full pytest suite:
```bash
pytest -v
```
*Output: 30 passed in ~5 seconds with 100% coverage.*

### 4. Launch the Interactive Streamlit Dashboard
Launch the web interface:
```bash
streamlit run app/streamlit_app.py
```
Open your browser to `http://localhost:8501`.

### 5. Generate Academic Report Figures & Holdout Evaluation
Run the asset generation script:
```bash
python reports/generate_report_assets.py
```
*Generates 9 publication-grade figures in `reports/figures/` and evaluates models on the held-out test set.*

---

## 📊 Held-Out Test Set Performance Benchmark (Days 50–59)

The models were trained strictly on Days 0 to 49 (18,000 records) and evaluated on Days 50 to 59 (3,600 records):

| Model / Predictor | Mathematical Family | MAE (min) | RMSE (min) | $R^2$ Score |
| :--- | :--- | :---: | :---: | :---: |
| **Weighted Ensemble Predictor** | **Hybrid Stochastic + Linear Algebra** | **0.547** | **1.312** | **0.9464** |
| Analytic $M/M/c$ Predictor | Stochastic Queueing Theory (Erlang-C) | 0.517 | 1.323 | 0.9455 |
| Bayesian-Updated $M/M/c$ | Conjugate Gamma-Poisson Updating | 0.517 | 1.323 | 0.9455 |
| Modified Gram-Schmidt QR | Numerical Linear Algebra (Decomposition) | 0.801 | 1.545 | 0.9257 |
| SVD Pseudo-Inverse | Singular Value Decomposition ($X^+$) | 0.801 | 1.545 | 0.9257 |
| Normal Equations | Closed-Form Matrix Inverse ($(X^T X)^{-1}$) | 0.801 | 1.545 | 0.9257 |
| Ridge Regression ($\alpha=10$) | Regularized Linear Algebra | 0.801 | 1.545 | 0.9257 |

---

## 🖥️ Streamlit Dashboard Walkthrough

### 🚦 Live Traffic-Light Indicator
Located prominently at the top of the application:
- 🟢 **Green ($< 5$ min)**: Quick service, optimal visiting window.
- 🟡 **Amber ($5 - 12$ min)**: Moderate congestion, acceptable wait.
- 🔴 **Red ($> 12$ min)**: High congestion alert, visiting alternative counter recommended.

### 📑 Tab 1: Live Wait-Time Prediction
- Compares predictions from Analytic $M/M/c$, Bayesian-updated model, Least-Squares regression, and Weighted Ensemble.
- Interactive **Live Bayesian Slider**: Enter newly observed customer arrivals to observe dynamic posterior distribution shifts.
- Interactive Plotly histogram of **10,000 Monte Carlo simulated customers** with $P_{50}, P_{90}, P_{95}$ percentile lines.

### 📑 Tab 2: Crowd Forecast (Markov Chain)
- Displays current discrete queue state (`Empty, Short, Medium, Long, Overcrowded`).
- Trajectory curves of state probabilities across 1 to 12 steps ahead (up to 1 hour).
- 5×5 transition probability matrix heatmap and stationary distribution $\pi$.
- Expected steps and minutes until queue becomes `Overcrowded`.

### 📑 Tab 3: Best Time to Visit (Recommender)
- Subspace projection of today's partial demand vector onto $V_k$.
- Identifies the most similar historical operating day via cosine similarity ($> 90\%$).
- Ranks the top 3 upcoming low-wait time slots over the next 2 hours.
- Recommends the fastest alternative counter at the current time.

### 📑 Tab 4: Model Insights & Linear Algebra
- SVD singular value spectrum and cumulative energy scree plot ($k=2$ captures $> 92\%$ energy).
- Canonical daily diurnal modes ($v_1, v_2, v_3$).
- Condition number analysis ($\kappa(X) = 449.27$ vs. $\kappa(X^T X) = 2.02 \times 10^5$).
- Numerical solver comparison table.

### 📑 Tab 5: Theory & Mathematical Foundations
- Beautifully formatted LaTeX derivations explaining all formulas: Poisson MLE, $\chi^2$ test, Gamma vs. Exponential AIC, Erlang-C, Little's Law, Conjugate Gamma-Poisson updating, Gram-Schmidt QR, SVD low-rank approximations, and Markov spectral mixing rates.

---

## 🏆 Academic & Technical Highlights
- **100% First-Principles Implementation**: No third-party black-box queueing packages (`simpy`, `queueing_tool`). Everything from Erlang-C to Modified Gram-Schmidt QR and Power Iteration is written manually.
- **Strict Little's Law Verification**: Validated in pytest to $< 10^{-6}$ precision.
- **Multicollinearity Elimination**: Standardized dummy encodings to prevent the Dummy Variable Trap, guaranteeing invertibility of $(X^T X)$.
- **Reproducible Science**: Single global random seed (`GLOBAL_RANDOM_SEED = 42`) across data generation, Monte Carlo simulations, and evaluations.
