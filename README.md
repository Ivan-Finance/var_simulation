# Historical VaR Simulation

A Python-based portfolio risk analysis tool that computes **Value at Risk (VaR)** and **Conditional Value at Risk (CVaR)** using three different methods: Historical Simulation, Parametric (Gaussian), and Monte Carlo. Includes backtesting and historical stress testing.

---

## What is VaR?

Value at Risk answers a simple but powerful question:

> *"What is the maximum loss I can expect in a single day, with 95% probability?"*

For example, a 1-day VaR of $100 at 95% confidence means that on 95% of trading days, the portfolio will not lose more than $100. On the remaining 5% of days, losses may exceed that threshold.

For a deeper explanation of the theory behind this project, see [THEORY.md](THEORY.md).

---

## Usage

```bash
python var_simulation.py
```

The script will automatically download market data, compute risk metrics, run backtesting and stress tests, and display an interactive dashboard.

---

## Configuration

All parameters are at the top of `var_simulation.py`:

```python
PORTFOLIO = {
    "SPY":  0.50,   # S&P 500 ETF          - 50%
    "TLT":  0.30,   # 20yr Treasury Bond   - 30%
    "GLD":  0.20,   # Gold ETF             - 20%
}

PORTFOLIO_VALUE     = 10_000      # Portfolio value in $ or €
START_DATE          = "2018-01-01"
END_DATE            = "2024-12-31"
HISTORICAL_WINDOW   = 500         # Days used for simulation
CONFIDENCE_LEVEL    = 0.95        # 95% confidence level
N_MC_SIMULATIONS    = 10_000      # Monte Carlo scenarios
```

You can replace tickers with any valid Yahoo Finance symbol (stocks, ETFs, crypto).

---

## Output

The script prints a summary table to the console:

```
============================================================
  VaR SUMMARY — Portfolio: $10,000 | Confidence: 95%
============================================================
  Method                         VaR %     VaR $    CVaR $
  ────────────────────────────────────────────────────────
  Historical Simulation          0.92%       92$      129$
  Parametric (Gaussian)          1.06%      106$      120$
  Monte Carlo                    0.95%       95$      121$
============================================================
```

And generates a 4-panel visual dashboard:

| Chart | Description |
|-------|-------------|
| **Return Distribution** | Histogram of historical returns with VaR thresholds from all 3 methods |
| **Returns Over Time** | Daily returns with rolling VaR line and violation markers |
| **Monte Carlo Distribution** | Simulated scenario distribution with VaR and CVaR lines |
| **VaR & CVaR Comparison** | Bar chart comparing all methods side by side |

---

## Stress Test

The script also applies known historical crisis scenarios to the portfolio.
