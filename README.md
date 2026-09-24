# CVaR Portfolio Optimization: A Risk Management Analysis

An empirical and theoretical examination of linear programming models for Conditional Value-at-Risk (CVaR) minimization across varying market regimes. 

This project evaluates five portfolio optimization strategies—standard CVaR, confidence level ($\beta$) sensitivity, minimax tail risk protection, dynamic rolling-window rebalancing, and stability-constrained turnover management—using 2019 (in-sample) and 2020 (out-of-sample COVID-19 shock) market data.

**Author**: Prem Charan Badri  
**Course**: Optimization 1 (RM 294)  
**Instructor**: Dr. Daniel Mitchell  
**Date**: January 17, 2026  

---

## Executive Summary

Using stock price data from the NASDAQ-100 (NDX) index constituents during normal market conditions (2019) and severe crisis conditions (2020), this study analyzes how CVaR optimization balances tail risk mitigation, diversification, and real-world execution constraints.

### Policy & Strategy Comparison Summary

| Optimization Strategy | Key Objective / Focus | In-Sample Risk (2019) | Out-of-Sample Risk (2020) | Out-of-Sample Risk Deterioration | Portfolio Concentration (Top 5 Weight) | Primary Trade-Off / Finding |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Static CVaR ($\beta=0.95$)** | Minimize expected worst 5% loss | CVaR = 0.0111 | CVaR = 0.0458 | 313.3% | 77.80% (13 stocks) | Baseline strategy; vulnerable to regime shifts without rebalancing. |
| **Moderate Tail ($\beta=0.90$)** | Focus on worst 10% loss | CVaR = 0.0089 | CVaR = 0.0321 | 260.5% | 70.73% (16 stocks) | Higher diversification; superior out-of-sample stability. |
| **Extreme Tail ($\beta=0.99$)** | Focus on worst 1% loss | CVaR = 0.0125 | CVaR = 0.0909 | 630.0% | 90.24% (9 stocks) | Extreme concentration; highly vulnerable to regime change. |
| **Minimax CVaR** | Minimize maximum monthly CVaR | Max CVaR = 0.0124 | Crisis Avg = 0.0271 | 157.3% | 90.19% (9 stocks) | Pays 11.7% in-sample efficiency cost for 40.8% better crisis performance. |
| **Dynamic Rebalancing** | Monthly rolling 12M window | N/A | Avg CVaR = 0.0322 | N/A | Dynamic | Outperforms static in 11/12 months (91.7%), but incurs high turnover. |
| **Stability-Constrained** | Dynamic + $\pm 5\%$ weight cap | N/A | Avg CVaR = 0.0306 | N/A | Controlled | Eliminates 8/9 unstable transitions; boosts CVaR by 5.0% by preventing over-fitting. |

### Key Takeaways
1. **Regime Sensitivity & Concentration**: Higher confidence levels ($\beta = 0.99$) cause severe portfolio concentration (top 5 holdings reach 90.24%). This over-specifies the portfolio to past tail events, leading to a 630% risk deterioration during unexpected crisis regimes (2020 COVID-19 shock).
2. **Insurance Premium of Minimax Optimization**: Accepting an 11.7% "efficiency penalty" in calm markets (2019) via Minimax CVaR yields a 40.8% improvement in average performance and a 156 percentage point improvement in robustness during market crises.
3. **Dynamic Rebalancing Effectiveness**: Monthly rolling re-optimization reduces average out-of-sample CVaR by 29.6% compared to a static allocation, outperforming the static baseline in 11 out of 12 months in 2020.
4. **The Stability Paradox**: Imposing an explicit $\pm 5\%$ monthly rebalancing constraint reduces transition violations from 81.8% to 9.1% while *improving* average CVaR performance by 5.0%, proving that friction constraints act as an effective regularizer against sample noise.

---

## Mathematical Formulations

The portfolio optimization models are formulated as Linear Programs (LPs) solved via Gurobi. Let $x \in \mathbb{R}^n$ be the portfolio weight vector, $y_k \in \mathbb{R}^n$ be asset returns for scenario $k \in \{1, \dots, q\}$, $R_{\text{target}} = 0.0002$ (0.02% daily), and $\beta$ be the confidence level.

### 1. Standard CVaR Linear Program (Rockafellar & Uryasev)
$$\min_{x, \alpha, u} \quad \alpha + \frac{1}{(1 - \beta)q} \sum_{k=1}^q u_k$$

$$\begin{aligned} \text{subject to} \quad & \sum_{j=1}^n x_j = 1 & & \text{(Budget Constraint)} \\ & x_j \ge 0, \quad \forall j=1,\dots,n & & \text{(No Short-Selling)} \\ & u_k \ge -x^T y_k - \alpha, \quad \forall k=1,\dots,q & & \text{(Tail Loss Capture)} \\ & u_k \ge 0, \quad \forall k=1,\dots,q & & \text{(Auxiliary Non-Negativity)} \\ & x^T \bar{y} \ge R_{\text{target}} & & \text{(Minimum Expected Return)} \end{aligned}$$

Where $\alpha \in \mathbb{R}$ represents Value-at-Risk ($\text{VaR}_\beta$) and $u_k \in \mathbb{R}_+$ linearizes the loss excess function $[ -x^T y_k - \alpha ]^+$.

### 2. Minimax CVaR Formulation
Let $m \in \{1, \dots, M\}$ denote monthly subsets of scenarios. The minimax formulation minimizes the worst monthly CVaR across all calendar months:

$$\min_{x, z, \alpha, u} \quad z$$

$$\begin{aligned} \text{subject to} \quad & z \ge \alpha_m + \frac{1}{(1 - \beta)q_m} \sum_{k \in \text{month } m} u_k^m, \quad \forall m \in \{1, \dots, M\} \\ & u_k^m \ge -x^T y_k - \alpha_m, \quad \forall k \in \text{month } m, \, \forall m \\ & u_k^m \ge 0, \quad \forall k \in \text{month } m, \, \forall m \\ & \sum_{j=1}^n x_j = 1, \quad x_j \ge 0, \quad x^T \bar{y} \ge R_{\text{target}} \end{aligned}$$

### 3. Stability-Constrained Dynamic Model
For consecutive months $t-1$ and $t$, the change in individual asset allocation is bounded by a maximum rebalancing band $\Delta_{\text{max}} = 0.05$:

$$w_{i, t-1} - 0.05 \le w_{i, t} \le w_{i, t-1} + 0.05, \quad \forall i = 1, \dots, n$$

---

## Detailed Section Analysis

### Section 1 & 2 — Baseline CVaR Optimization & Out-of-Sample Performance
* **Objective**: Evaluate in-sample performance (2019) vs. out-of-sample performance (2020 COVID-19 market crisis) at $\beta = 0.95$.
* **In-Sample Performance (2019)**: $\text{CVaR}_{0.95} = 0.0111$ (1.11% daily tail loss).
* **Out-of-Sample Performance (2020)**: $\text{CVaR}_{0.95} = 0.0458$ (4.58% daily tail loss).
* **Deterioration**: Portfolio CVaR degraded by **313.3%**. By comparison, the NASDAQ-100 benchmark (NDX) CVaR deteriorated by **128.9%** (from 0.0244 to 0.0559).
* **Insight**: Although the static CVaR portfolio suffered significant performance degradation due to non-stationary market conditions in 2020, it still outperformed the NDX index benchmark in absolute out-of-sample risk (0.0458 vs. 0.0559).

### Section 3 — Impact of Confidence Level ($\beta$)
Optimizing across varying confidence thresholds demonstrates how tail focus alters concentration and robustness:

| Metric | $\beta = 0.90$ (10% Tail) | $\beta = 0.95$ (5% Tail) | $\beta = 0.99$ (1% Tail) |
| :--- | :---: | :---: | :---: |
| **In-Sample CVaR (2019)** | 0.0089 | 0.0111 | 0.0125 |
| **Out-of-Sample CVaR (2020)** | 0.0321 | 0.0458 | 0.0909 |
| **CVaR Deterioration (%)** | **260.5%** | **313.3%** | **630.0%** |
| **Active Assets Used** | 16 | 13 | 9 |
| **Maximum Asset Weight** | 24.06% | 30.39% | 44.70% |
| **Top 5 Concentration** | 70.73% | 77.80% | 90.24% |
| **Expected Return** | 0.0012 | 0.0012 | 0.0013 |

* **Finding**: As $\beta \to 0.99$, the optimization focuses on rare tail events, selecting a smaller subset of assets that historically protected against extreme losses. This creates a highly concentrated portfolio that fails when out-of-sample stress exhibits different correlation structures.

### Section 4 — Minimax CVaR Approach
Shifting the objective from minimizing average tail risk to minimizing maximum monthly tail risk across 2019:

| Metric | Standard Average CVaR (Sec 2) | Minimax CVaR (Sec 4) | Performance Impact |
| :--- | :---: | :---: | :--- |
| **Number of Holdings** | 13 | 9 | 31% reduction in asset breadth |
| **Top 5 Concentration** | 77.80% | 90.19% | Increased asset concentration |
| **Primary Objective (2019)** | 0.0111 (Avg CVaR) | 0.0124 (Max Monthly CVaR) | +11.7% efficiency penalty |
| **2020 Crisis Performance** | 0.0458 (Avg CVaR) | 0.0271 (Avg CVaR) | **40.8% lower crisis risk** |
| **Risk Deterioration** | 313.3% | 157.3% | **+156.0 %pt robustness gain** |

* **Finding**: The Minimax strategy acts as an insurance policy. It sacrifices 11.7% efficiency during normal market periods to build resilience against worst-case monthly scenarios, resulting in vastly superior performance during market disruptions.

### Section 5 — Dynamic Monthly Rebalancing
Using a 12-month rolling training window to rebalance portfolios monthly throughout 2020:

| Performance Metric | Static Allocation (Sec 2) | Dynamic Rolling Allocation (Sec 5) |
| :--- | :---: | :---: |
| **Average Daily CVaR (2020)** | 0.04580 | **0.03223** (-29.6% Risk) |
| **Outperforming Months** | Base Baseline | **11 / 12 Months (91.7%)** |
| **Worst Month Performance** | March 2020 (Static) | March 2020 (CVaR = 0.10123) |
| **Best Month Performance** | N/A | January 2020 (CVaR = 0.00805) |
| **CVaR Volatility ($\sigma$)** | Fixed | 70.6% |

* **Methodological Limitation**: A temporal aggregation bias exists when comparing annual CVaR (worst 5% of 252 days $\approx$ 12-13 worst days globally) to averaged monthly CVaR (worst 1 day per month $\times$ 12 months). Peak crisis days in March 2020 are concentrated in annual CVaR, whereas monthly averaging dilutes single-period spikes.

### Section 6 — Portfolio Stability & Rebalancing Constraints
Evaluating month-to-month asset weight turnover under a strict 5 percentage point ($\le 0.05$) change constraint per asset:

| Stability Metric | Unconstrained Dynamic Model | Stability-Constrained Model |
| :--- | :---: | :---: |
| **Stable Monthly Transitions ($\le 5\%$)** | 2 / 11 (18.2%) | **10 / 11 (90.9%)** |
| **Average Maximum Weight Change** | 15.6% | **4.1%** |
| **Peak Observed Weight Change** | 45.2% | **5.0%** |
| **Average Violations per Month** | 3.9 assets | **0.1 assets** |
| **Average Out-of-Sample CVaR** | 0.03223 | **0.03062** (5.0% Improvement) |

* **Key Finding**: Imposing rebalancing bounds eliminates impractical portfolio churn (such as a 45.2% single-month swing in March-April 2020). Restricting weight changes prevents the model from overfitting to short-term monthly noise, enhancing out-of-sample risk control.

---

## Code Implementations

### Gurobi Formulation for Standard CVaR LP
```python
import gurobipy as gp

def solve_cvar_portfolio(returns_data, beta, R_target):
    n_stocks = returns_data.shape[1]
    q = len(returns_data)
    Y = returns_data.values
    mean_returns = returns_data.mean().values
    optMod = gp.Model("CVaR_Optimization")

    optMod.Params.OutputFlag = 0

    #---------- Decision variables ----------------------------------------
    optx = optMod.addMVar(shape=n_stocks, lb=0.0)       # Portfolio weights
    alpha = optMod.addVar(lb=-gp.GRB.INFINITY)           # VaR threshold
    u = optMod.addMVar(shape=q, lb=0.0)                  # Tail loss variables

    #---------- Constraints ------------------
    optMod.addConstr(optx.sum() == 1.0)                  # Budget
    optMod.addConstr(u >= -Y @ optx - alpha)             # Tail loss capture
    optMod.addConstr(mean_returns @ optx >= R_target)    # Return target

    #--------- Objective: Minimize CVaR ------------------
    cvar_obj = alpha + (1.0 / ((1.0 - beta) * q)) * u.sum()
    optMod.setObjective(cvar_obj, gp.GRB.MINIMIZE)
    optMod.optimize()
    return optx.X, alpha.X, optMod
