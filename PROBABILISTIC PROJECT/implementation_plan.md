# Implementation Plan: Smart Queue Prediction and Waiting-Time Assistant for College Canteens
**Institution/Branding Context**: Saveetha College Canteen Queue Assistant  
**Role**: Senior Full-Stack + Data-Science Engineer  

---

## 1. Project Overview & Objectives
The goal of this project is to develop an academic, mathematically rigorous, and completely runnable decision-support system titled **"Smart Queue Prediction and Waiting-Time Assistant for College Canteens"**. The architecture partitions predictive analytics into two mathematically rich engines:
1. **Module 1 (Probabilistic Methods)**: Stochastic queue modeling, Poisson arrival processes, Erlang/Gamma service models, Erlang-C M/M/c queueing theory, conjugate Gamma-Poisson Bayesian updating, 10,000-run Monte Carlo discrete-event simulation, and discrete-time Markov chains on congestion states with absorption analysis.
2. **Module 2 (Linear Algebra)**: High-dimensional canteen demand matrix structuring, manual implementation of Least-Squares via Normal Equations, Modified Gram-Schmidt QR decomposition, and SVD pseudo-inverse, Ridge regularization, SVD/PCA spectral decomposition for canonical diurnal modes, Markov transition matrix spectral eigen-analysis (Perron-Frobenius, stationary distribution, mixing rate, manual power iteration), and a subspace projection cosine-similarity recommender.
3. **Integration Layer (Streamlit Dashboard)**: An interactive 5-tab web dashboard equipped with traffic-light wait-time indicators, real-time Bayesian parameter adjustments, live crowd forecasting, visual heatmaps, model diagnostics, and LaTeX theoretical formulas.

---

## 2. Directory Architecture & File Layout
```text
canteen_queue/
├── config.py                      # Global seeds, counters, opening hours (8:00-18:00), parameters
├── data/
│   ├── __init__.py
│   └── generate_data.py          # 60-day synthetic dataset with lunch peaks, class breaks, outliers
├── module1_probabilistic/
│   ├── __init__.py
│   ├── arrival_model.py          # Poisson MLE, Chi-square goodness-of-fit, Exponential KS test
│   ├── service_model.py          # Exponential vs Gamma/Erlang MLE, AIC/BIC selection
│   ├── queue_theory.py           # M/M/1 & M/M/c from scratch, Erlang-C, Little's Law, predict_wait
│   ├── bayesian_update.py        # Gamma-Poisson conjugate model, credible intervals, NegBin predictive
│   ├── monte_carlo.py            # Discrete-event simulation (N=10,000), empirical percentiles, error
│   └── markov_chain.py           # 5-state Markov chain, n-step transitions, hitting time to Overcrowded
├── module2_linear_algebra/
│   ├── __init__.py
│   ├── demand_matrix.py          # Demand matrix D construction, cleaning, rank, condition number
│   ├── least_squares_predictor.py# Normal Eq, Modified Gram-Schmidt QR, SVD pseudo-inverse, Ridge
│   ├── svd_pca_patterns.py       # SVD diurnal decomposition, scree plot, top 3 singular vectors
│   ├── markov_matrix_analysis.py # Spectral eigen-analysis, power iteration, stationary state, mixing
│   └── recommender.py            # Subspace projection, cosine similarity, optimal visiting time
├── app/
│   ├── __init__.py
│   └── streamlit_app.py          # 5-tab interactive Streamlit dashboard with traffic-light status
├── tests/
│   ├── __init__.py
│   ├── test_arrival_service.py   # Tests for Poisson, Chi-square, Exponential KS, Gamma AIC
│   ├── test_queue_theory.py      # Erlang-C textbook values, Little's law verification
│   ├── test_bayesian_update.py   # Conjugate update properties, posterior predictive
│   ├── test_monte_carlo.py       # Discrete-event simulation convergence to analytical expectation
│   ├── test_markov_chain.py      # Stochastic matrix row sums = 1, absorption times
│   ├── test_demand_matrix.py     # Matrix rank, shape, imputation, conditioning
│   ├── test_least_squares.py     # Agreement between Normal Eq, MGS QR, and SVD; Ridge shrinkage
│   ├── test_svd_pca.py           # Eckart-Young monotonic error decrease, orthogonality
│   ├── test_markov_matrix_analysis.py # Power iteration agreement with numpy.linalg.eig
│   └── test_recommender.py       # Subspace projection & recommendation ranking
├── reports/
│   ├── generate_report_assets.py # Generates static figures and evaluation metrics
│   ├── report.md                 # Complete academic report with LaTeX mathematics and results
│   └── figures/                  # Generated plots (heatmaps, scree, PMF fits, MC histograms)
├── requirements.txt
└── README.md
```

---

## 3. Mathematical Foundations & Implementation Specifications

### 3.1 Synthetic Data Generation (`data/generate_data.py`)
- **Parameters**: 60 days of operations (Days 1 to 60), 3 counters (`Meals`, `Snacks`, `Beverages`), 8:00 AM to 6:00 PM (10 hours = 120 5-minute time slots per day; 7,200 slots per counter; 21,600 total observations).
- **Temporal Patterns**:
  - Base arrival rate: $\lambda_0(t)$ modulated by time of day.
  - Lunch rush (12:30 PM - 2:00 PM): Major peak $\lambda_{\text{lunch}} = 3.5 \times \text{base}$.
  - Lecture break spikes (10:30 AM, 3:30 PM): Sharp bell curves $\lambda_{\text{break}} = 2.2 \times \text{base}$.
  - Day of week modulation: Monday-Thursday standard; Friday lower demand (80%); Exam week (Days 45-50) surge in `Beverages` (+75%) and `Snacks` (+40%).
- **Counter Characteristics**:
  - `Meals`: Higher service time ($\mu^{-1} \approx 60-90$s), steady queue.
  - `Snacks`: Fast to moderate service ($\mu^{-1} \approx 30-45$s), bursty breaks.
  - `Beverages`: Very fast service ($\mu^{-1} \approx 20-30$s), high volume.
- **Anomalies & Noise**:
  - Realistic Gaussian/Poisson noise on queue counts and arrivals.
  - 5% synthetic anomalies (sensor dropouts with NaNs, negative values, extreme spikes) to demonstrate data cleaning pipelines.

### 3.2 Module 1: Probabilistic Methods
1. **Arrival Model (`arrival_model.py`)**:
   - Fits Poisson distribution to arrivals $k \sim \text{Poisson}(\lambda)$.
   - $\chi^2$ goodness-of-fit test: calculates test statistic $\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$ with pooled low-frequency tails ($E_i \ge 5$).
   - Inter-arrival times modeled as Exponential $\Delta t \sim \text{Exp}(\lambda_{\text{rate}})$; evaluated with two-sided Kolmogorov-Smirnov test against empirical CDF.
2. **Service Model (`service_model.py`)**:
   - Exponential distribution $f(t) = \mu e^{-\mu t}$ vs Gamma distribution $f(t) = \frac{\beta^\alpha}{\Gamma(\alpha)} t^{\alpha-1} e^{-\beta t}$.
   - Numerical Maximum Likelihood Estimation (MLE) of parameters.
   - Model selection using Akaike Information Criterion ($\text{AIC} = 2k - 2\ln L$) and Bayesian Information Criterion ($\text{BIC} = k\ln n - 2\ln L$).
3. **Queueing Theory (`queue_theory.py`)**:
   - Pure from-scratch implementation without third-party queue packages.
   - Traffic intensity $r = \lambda / \mu$, utilization $\rho = \lambda / (c\mu)$ ($0 < \rho < 1$).
   - Normalizing state probability:
     $$P_0 = \left[ \sum_{n=0}^{c-1} \frac{(c\rho)^n}{n!} + \frac{(c\rho)^c}{c!(1-\rho)} \right]^{-1}$$
   - Erlang-C waiting probability:
     $$P_C = \mathbb{P}(W_q > 0) = \frac{\frac{(c\rho)^c}{c!(1-\rho)}}{\sum_{n=0}^{c-1}\frac{(c\rho)^n}{n!} + \frac{(c\rho)^c}{c!(1-\rho)}} = P_0 \frac{(c\rho)^c}{c!(1-\rho)}$$
   - Queue metrics: $L_q = P_C \frac{\rho}{1-\rho}$, $W_q = \frac{L_q}{\lambda}$, $W = W_q + \frac{1}{\mu}$, $L = \lambda W$.
   - Rigorous check for Little's Law ($|L - \lambda W| < 10^{-6}$).
   - `predict_wait(counter, time_slot, current_queue_length, c)`: combines observed customer count in queue $Q$ with steady-state processing rate $c\mu$. Conditional wait follows an $\text{Erlang}(Q - c + 1, c\mu)$ when $Q \ge c$, plus own service time. 90% credible intervals computed analytically.
4. **Bayesian Updating (`bayesian_update.py`)**:
   - Conjugate prior: $\lambda \sim \text{Gamma}(\alpha_0, \beta_0)$ where $\alpha_0, \beta_0$ are estimated from historical daytime band statistics.
   - Likelihood: Poisson with $K = \sum_{i=1}^m k_i$ arrivals over duration $T = m \times \Delta t$.
   - Posterior: $\lambda | K \sim \text{Gamma}(\alpha_0 + K, \beta_0 + T)$.
   - Posterior mean $\mathbb{E}[\lambda | K] = \frac{\alpha_0 + K}{\beta_0 + T}$, 90% Credible Interval $[\text{Gamma}_{0.05}, \text{Gamma}_{0.95}]$.
   - Posterior predictive distribution for next slot: Negative Binomial $\text{NB}\left(r = \alpha_{post}, p = \frac{\beta_{post}}{\beta_{post} + \tau}\right)$.
5. **Monte Carlo Simulation (`monte_carlo.py`)**:
   - Discrete-event simulation (DES) engine tracking individual arrivals, queue join times, service start times, and departure events across $c$ servers.
   - $N = 10,000$ simulation runs.
   - Empirical percentiles: $P_{50}$, $P_{90}$, $P_{95}$, $\mathbb{P}(W_q > 5\text{ min})$, $\mathbb{P}(W_q > 10\text{ min})$.
   - Error verification against analytical Erlang-C results.
6. **Markov Chain Queue States (`markov_chain.py`)**:
   - States: `0: Empty (0-1)`, `1: Short (2-4)`, `2: Medium (5-8)`, `3: Long (9-14)`, `4: Overcrowded (15+)`.
   - Transition probability matrix $P \in \mathbb{R}^{5 \times 5}$ estimated via Maximum Likelihood with Laplace smoothing.
   - $n$-step forecasting $\pi^{(n)} = \pi^{(0)} P^n$.
   - First hitting time to absorption at "Overcrowded": $h = (I - Q)^{-1} \mathbf{1}$.

### 3.3 Module 2: Linear Algebra
1. **Demand Matrix (`demand_matrix.py`)**:
   - Demand matrix $D \in \mathbb{R}^{60 \times 120}$ (Days $\times$ Slots).
   - Missing value imputation via robust localized median/mean interpolation.
   - Standardization and normalization (centering and scaling).
   - Rank computation $\text{rank}(D)$, Frobenius norm $\|D\|_F$, singular values $\sigma_i$, condition number $\kappa(D) = \sigma_{\max} / \sigma_{\min}$.
2. **Least Squares Predictor (`least_squares_predictor.py`)**:
   - Design matrix $X \in \mathbb{R}^{N \times d}$: harmonic cyclic time features $(\sin, \cos)$, day-of-week one-hot, counter one-hot, observed queue length, bias intercept.
   - Manual solver implementations:
     1. Normal Equations: $\beta = (X^T X)^{-1} X^T y$
     2. Modified Gram-Schmidt (MGS) QR Decomposition: $X = QR$, solve $R\beta = Q^T y$ via back-substitution.
     3. SVD Pseudo-inverse: $X = U \Sigma V^T$, $\beta = V \Sigma^+ U^T y$.
     4. Ridge Regularization: $\beta_{\text{ridge}} = (X^T X + \lambda I)^{-1} X^T y$.
   - Numerical comparison, residual norms $\|y - X\beta\|_2$, $R^2$, RMSE, MAE.
   - In-depth discussion of condition number squaring: $\kappa(X^T X) = (\kappa(X))^2$.
3. **SVD / PCA Patterns (`svd_pca_patterns.py`)**:
   - SVD: $D = U \Sigma V^T$.
   - Energy capture: $E_k = \frac{\sum_{i=1}^k \sigma_i^2}{\sum \sigma_i^2}$; determine components for $\ge 90\%$ energy.
   - Plot top 3 right singular vectors $v_1, v_2, v_3$ (canonical daily patterns: Baseline, Lunch Surge, Break Spikes).
   - Low-rank reconstruction $D_k = \sum_{i=1}^k \sigma_i u_i v_i^T$; verify Eckart-Young-Mirsky theorem.
4. **Markov Matrix Spectral Analysis (`markov_matrix_analysis.py`)**:
   - Eigen-decomposition of stochastic matrix $P$: $P v_i = \lambda_i v_i$, left eigenvector $\pi P = \pi$.
   - Perron-Frobenius theorem verification: $\lambda_1 = 1$, all $|\lambda_i| \le 1$.
   - Manual Power Iteration implementation: $\pi^{(k+1)} = \pi^{(k)} P$.
   - Mixing rate and relaxation time governed by Second Largest Eigenvalue Modulus (SLEM) $|\lambda_2|$:
     $$\tau_{\text{mix}} \approx \frac{1}{1 - |\lambda_2|}$$
5. **Recommender System (`recommender.py`)**:
   - Orthogonal projection of live day demand vector $d_{\text{today}}$ onto principal subspace $V_k$.
   - Cosine similarity matching against historical days.
   - Identification of optimal low-congestion visiting slots and optimal counter choice.

### 3.4 Integration & Streamlit Dashboard (`app/streamlit_app.py`)
- Clean, aesthetic UI with Saveetha College Canteen branding.
- Dynamic sidebar controls: Counter selection, current time slider, observed queue length, active servers.
- Prominent Traffic Light Indicator (Green < 5 min, Amber 5-12 min, Red > 12 min).
- 5 comprehensive tabs:
  1. **Live Wait-Time Prediction**: Analytic (M/M/c), Bayesian-updated, Least-Squares regression, Monte Carlo 10,000-run distribution, and an ensemble estimate.
  2. **Crowd Forecast**: Markov state transitions, $n$-step horizon curves, stationary distribution, expected time to Overcrowded.
  3. **Best Time to Visit**: Interactive demand heatmap, subspace projection recommendations, least-crowded slot recommendations.
  4. **Model Insights**: SVD spectrum, canonical daily modes, residual diagnostics, test-set performance comparison table.
  5. **Theory & Mathematics**: Formatted LaTeX derivations and mathematical explanations.

### 3.5 Testing & Validation Suite (`tests/`)
- Pure pytest test cases covering all requirements:
  - `test_queue_theory.py`: Validates Erlang-C against textbook values ($c=2, \lambda=4, \mu=3 \implies \rho=0.667$, $P_0=0.2$, $P_C=0.5333$), checks Little's Law to precision $< 10^{-6}$.
  - `test_least_squares.py`: Compares Normal Equations, Modified Gram-Schmidt QR, and SVD solutions ($\|\beta_{NE} - \beta_{QR}\| < 10^{-5}$).
  - `test_svd_pca.py`: Verifies monotonic decrease in reconstruction error as rank $k$ increases, checks $V^T V = I$.
  - `test_markov_matrix_analysis.py`: Verifies power iteration matches `np.linalg.eig`, $\sum \pi_i = 1$, $\lambda_1 = 1$.
  - `test_bayesian_update.py`: Verifies posterior variance shrinkage and credible interval bounds.
  - `test_monte_carlo.py`: Verifies that Monte Carlo sample mean approaches Erlang-C theoretical mean within 95% confidence bounds.

---

## 4. Execution Plan & Milestones
- **Step 1**: Finalize implementation plan & verify dependencies.
- **Step 2**: Implement `config.py` and `data/generate_data.py`.
- **Step 3**: Implement Module 1 (`arrival_model.py`, `service_model.py`, `queue_theory.py`, `bayesian_update.py`, `monte_carlo.py`, `markov_chain.py`).
- **Step 4**: Implement Module 2 (`demand_matrix.py`, `least_squares_predictor.py`, `svd_pca_patterns.py`, `markov_matrix_analysis.py`, `recommender.py`).
- **Step 5**: Implement Streamlit App (`app/streamlit_app.py`).
- **Step 6**: Implement Pytest suite in `tests/` and execute `pytest -v`.
- **Step 7**: Generate report figures and write comprehensive `reports/report.md` and `README.md`.
- **Step 8**: Verify full end-to-end functionality, launch testing, and present final deliverables.
