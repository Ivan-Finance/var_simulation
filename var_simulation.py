"""
HISTORICAL SIMULATION: VaR & CVaR
Calculating VaR and CVaR using historical simulation, parametric method and Monte
Carlo on a customizable portfolio (including backtesting and stress testing)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

#  PORTFOLIO CONFIGURATION

PORTFOLIO = {
    "SPY":  0.50,   # S&P 500 ETF          - 50%
    "TLT":  0.30,   # 20yr Treasury Bond   - 30%
    "GLD":  0.20,   # Gold ETF             - 20%
}

PORTFOLIO_VALUE     = 10_000      # Portfolio value in $ or €
START_DATE          = "2018-01-01"
END_DATE            = "2024-12-31"
HISTORICAL_WINDOW   = 500         # Days used for the simulation
CONFIDENCE_LEVEL    = 0.95        # 95% (VaR at the 5th percentile)
N_MC_SIMULATIONS    = 10_000      # Number of Monte Carlo simulations


#  1. DATA DOWNLOAD

def download_data(tickers: list, start: str, end: str) -> pd.DataFrame:
    """Download adjusted closing prices for the given tickers"""
    print(f"\n Downloading data for: {', '.join(tickers)}")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    # Handle both single and multiple tickers
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers

    prices.dropna(inplace=True)
    print(f" Downloaded {len(prices)} days from {prices.index[0].date()} to {prices.index[-1].date()}")
    return prices

#  2. PORTFOLIO RETURNS

def compute_portfolio_returns(prices: pd.DataFrame, weights: dict) -> pd.Series:
    """
    Compute daily portfolio returns using a weighted average.
    - portfolio_return = sum(weight_i * return_i)
    """
    weights_series = pd.Series(weights)
    individual_returns = prices.pct_change().dropna()

    # Aligning weights to available tickers
    aligned_weights = weights_series.reindex(individual_returns.columns).fillna(0)
    aligned_weights /= aligned_weights.sum()  # renormalize in case of missing tickers

    portfolio_returns = individual_returns.dot(aligned_weights)
    return portfolio_returns


#  3. METHOD 1 — HISTORICAL SIMULATION VaR

def historical_var(returns: pd.Series, value: float, confidence: float, window: int):
    """
    Historical Simulation VaR.
    Taking the last window days of actual returns and compute the percentile
    corresponding to the maximum loss at the given confidence level.
    """
    recent_returns = returns.iloc[-window:]
    var  = np.percentile(recent_returns, (1 - confidence) * 100)
    cvar = recent_returns[recent_returns <= var].mean()

    return {
        "method": "Historical Simulation",
        "VaR_%": abs(var),
        "CVaR_%": abs(cvar),
        "VaR_$": abs(var)  * value,
        "CVaR_$": abs(cvar) * value,
        "distribution": recent_returns,
    }


#  4. METHOD 2 — PARAMETRIC VaR (Gaussian)

def parametric_var(returns: pd.Series, value: float, confidence: float, window: int):
    """
    Parametric VaR (variance-covariance method).
    Assumption: returns follow a normal distribution.
    VaR = mu - z * sigma

    Faster but less accurate: real returns have fatter tails
    than a normal distribution (excess kurtosis).
    """
    recent_returns = returns.iloc[-window:]
    mu    = recent_returns.mean()
    sigma = recent_returns.std()
    z     = stats.norm.ppf(1 - confidence)

    var  = mu - z * sigma
    # Parametric CVaR = mu - sigma * phi(z) / (1 - c)
    cvar = mu - sigma * (stats.norm.pdf(z) / (1 - confidence))

    return {
        "method": "Parametric (Gaussian)",
        "VaR_%": abs(var),
        "CVaR_%": abs(cvar),
        "VaR_$": abs(var)  * value,
        "CVaR_$": abs(cvar) * value,
        "mu": mu,
        "sigma": sigma,
    }


#  5. METHOD 3 — MONTE CARLO VaR

def monte_carlo_var(returns: pd.Series, value: float, confidence: float,
                    window: int, n_sim: int):
    """
    Monte Carlo VaR.
    Simulating n_sim future scenarios by randomly sampling from a
    normal distribution calibrated on recent historical data.
    """
    recent_returns = returns.iloc[-window:]
    mu    = recent_returns.mean()
    sigma = recent_returns.std()

    # Sampling n_sim random returns
    np.random.seed(42)
    scenarios = np.random.normal(mu, sigma, n_sim)

    var  = np.percentile(scenarios, (1 - confidence) * 100)
    cvar = scenarios[scenarios <= var].mean()

    return {
        "method": "Monte Carlo",
        "VaR_%": abs(var),
        "CVaR_%": abs(cvar),
        "VaR_$": abs(var)  * value,
        "CVaR_$": abs(cvar) * value,
        "scenarios": scenarios,
    }


#  6. BACKTESTING

def backtesting(returns: pd.Series, confidence: float, window: int):
    """
    Checking how many times the historical VaR was breached.

    Method: rolling window
    - For each day t, compute VaR using the previous window days
    - Check if the actual return on day t exceeded the VaR

    Expected violation rate = 1 - confidence (5% for a 95% VaR)
    Kupiec test: if the actual rate deviates significantly the model is inaccurate.
    """
    violations  = []
    rolling_var = []

    for i in range(window, len(returns)):
        past_window  = returns.iloc[i - window:i]
        var_today    = np.percentile(past_window, (1 - confidence) * 100)
        return_today = returns.iloc[i]

        rolling_var.append(var_today)
        violations.append(return_today < var_today)

    violations  = pd.Series(violations, index=returns.index[window:])
    rolling_var = pd.Series(rolling_var, index=returns.index[window:])
    rolling_ret = returns.iloc[window:]

    n_violations  = violations.sum()
    actual_rate   = n_violations / len(violations)
    expected_rate = 1 - confidence

    print(f"\n BACKTESTING RESULTS")
    print(f"   Days tested:         {len(violations)}")
    print(f"   Violations:          {n_violations}")
    print(f"   Actual rate:         {actual_rate:.2%}")
    print(f"   Expected rate:       {expected_rate:.2%}")
    print(f"   Assessment:          {'OK' if abs(actual_rate - expected_rate) < 0.02 else 'Model inaccurate'}")

    return violations, rolling_var, rolling_ret

#  7. STRESS TEST

def stress_test(value: float):
    """
    Simulating known historical crisis scenarios on the current portfolio
    to understand potential losses under extreme market conditions.
    """
    scenarios = {
        "COVID Crash (Feb-Mar 2020)":       -0.34,
        "Global Financial Crisis 2008":     -0.50,
        "Dot-com Crash (2000-2002)":        -0.49,
        "Black Monday 1987":                -0.22,
        "Flash Crash 2010":                 -0.09,
        "2022 Rate Hike Correction":        -0.25,
    }

    print(f"\n STRESS TEST — Portfolio Value: ${value:,.0f}")
    print(f"   {'Scenario':<38} {'Return':>10} {'Loss $':>10}")
    print("   " + "─" * 60)
    for scenario, ret in scenarios.items():
        loss = abs(ret) * value
        print(f"   {scenario:<38} {ret:>9.1%} {loss:>9,.0f}$")

    return scenarios

#  8. VISUALIZATION

def plot_results(results: dict, returns: pd.Series,
                 violations, rolling_var, rolling_ret,
                 mc_scenarios, value: float):
    """
    Dashboard with 4 main charts:
    1. Return distribution + VaR lines from all methods
    2. Returns over time + rolling VaR + violations
    3. Monte Carlo scenario distribution
    4. VaR & CVaR comparison across methods (bar chart)
    """
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("#0d1117")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    colors = {
        "Historical Simulation": "#00d4aa",
        "Parametric (Gaussian)": "#ff6b6b",
        "Monte Carlo":           "#ffd93d",
    }
    bg   = "#161b22"
    text = "white"

    # Chart 1: Historical return distribution + VaR lines 
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(bg)
    ax1.hist(returns.iloc[-500:], bins=60, color="#58a6ff", alpha=0.6, edgecolor="none", density=True)
    for name, res in results.items():
        ax1.axvline(-res["VaR_%"], color=colors[name], linewidth=1.8,
                    linestyle="--", label=f"{name}: {res['VaR_%']:.2%}")
    ax1.set_title("Historical Return Distribution", color=text, fontsize=11, pad=10)
    ax1.set_xlabel("Daily return", color=text)
    ax1.set_ylabel("Density", color=text)
    ax1.tick_params(colors=text)
    ax1.legend(fontsize=7.5, facecolor=bg, labelcolor=text)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#30363d")

    # Chart 2: Returns over time + rolling VaR + violations 
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(bg)
    ax2.plot(rolling_ret.index, rolling_ret.values,
             color="#58a6ff", linewidth=0.6, alpha=0.7, label="Daily return")
    ax2.plot(rolling_var.index, rolling_var.values,
             color="#00d4aa", linewidth=1.2, label="Rolling VaR 95%")
    ax2.scatter(violations[violations].index,
                rolling_ret[violations],
                color="#ff6b6b", s=15, zorder=5, label="VaR violation", alpha=0.8)
    ax2.set_title("Returns Over Time & VaR Violations", color=text, fontsize=11, pad=10)
    ax2.set_xlabel("Date", color=text)
    ax2.set_ylabel("Return", color=text)
    ax2.tick_params(colors=text)
    ax2.legend(fontsize=8, facecolor=bg, labelcolor=text)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#30363d")

    # Chart 3: Monte Carlo scenario distribution
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(bg)
    mc_res = results["Monte Carlo"]
    ax3.hist(mc_res["scenarios"], bins=80, color="#ffd93d", alpha=0.5, edgecolor="none", density=True)
    ax3.axvline(-mc_res["VaR_%"],  color="#ff6b6b", linewidth=2,
                label=f"VaR 95%: {mc_res['VaR_%']:.2%}")
    ax3.axvline(-mc_res["CVaR_%"], color="#ff9900", linewidth=2, linestyle=":",
                label=f"CVaR 95%: {mc_res['CVaR_%']:.2%}")
    ax3.set_title(f"Monte Carlo — {N_MC_SIMULATIONS:,} Scenarios", color=text, fontsize=11, pad=10)
    ax3.set_xlabel("Simulated return", color=text)
    ax3.set_ylabel("Density", color=text)
    ax3.tick_params(colors=text)
    ax3.legend(fontsize=8.5, facecolor=bg, labelcolor=text)
    for spine in ax3.spines.values():
        spine.set_edgecolor("#30363d")

    # Chart 4: VaR & CVaR comparison across methods
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(bg)
    methods   = list(results.keys())
    var_vals  = [results[m]["VaR_$"]  for m in methods]
    cvar_vals = [results[m]["CVaR_$"] for m in methods]
    x = np.arange(len(methods))
    w = 0.35
    bars1 = ax4.bar(x - w/2, var_vals,  w, label="VaR $",
                    color=["#00d4aa", "#ff6b6b", "#ffd93d"], alpha=0.85)
    bars2 = ax4.bar(x + w/2, cvar_vals, w, label="CVaR $",
                    color=["#00d4aa", "#ff6b6b", "#ffd93d"], alpha=0.45)
    ax4.set_title(f"VaR & CVaR Comparison (portfolio ${value:,})", color=text, fontsize=11, pad=10)
    ax4.set_xticks(x)
    ax4.set_xticklabels(["Historical", "Parametric", "Monte Carlo"], color=text, fontsize=8.5)
    ax4.set_ylabel("Potential loss ($)", color=text)
    ax4.tick_params(colors=text)
    ax4.legend(fontsize=8.5, facecolor=bg, labelcolor=text)
    for spine in ax4.spines.values():
        spine.set_edgecolor("#30363d")
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, h + 5,
                 f"${h:.0f}", ha="center", va="bottom", color=text, fontsize=7.5)

    fig.suptitle(" Portfolio Risk Dashboard — Historical VaR Simulation",
                 color=text, fontsize=14, fontweight="bold", y=1.01)
    
    plt.show()
    
    #  9. SUMMARY PRINT

def print_summary(results: dict, value: float, confidence: float):
    print(f"\n{'='*60}")
    print(f"  VaR SUMMARY — Portfolio: ${value:,} | Confidence: {confidence:.0%}")
    print(f"{'='*60}")
    print(f"  {'Method':<28} {'VaR %':>7} {'VaR $':>9} {'CVaR $':>9}")
    print(f"  {'─'*56}")
    for name, res in results.items():
        print(f"  {name:<28} {res['VaR_%']:>6.2%} {res['VaR_$']:>8,.0f}$ {res['CVaR_$']:>8,.0f}$")
    print(f"{'='*60}")
    print(f"\n  Interpretation at {confidence:.0%} confidence:")
    hist_var = results["Historical Simulation"]["VaR_$"]
    print(f"  With {confidence:.0%} probability, the maximum 1-day loss")
    print(f"  will not exceed ${hist_var:,.0f} (historical method).")
    print(f"\n  CVaR (Expected Shortfall) is the average loss")
    print(f"  in the worst {(1-confidence):.0%} of scenarios.")
    
    #  MAIN

def main():
    print("=" * 60)
    print("  HISTORICAL VaR SIMULATION")
    print("  Portfolio:", " | ".join([f"{t} {w:.0%}" for t, w in PORTFOLIO.items()]))
    print("=" * 60)

    # 1. Downloading data
    tickers = list(PORTFOLIO.keys())
    prices  = download_data(tickers, START_DATE, END_DATE)

    # 2. Compute portfolio returns
    returns = compute_portfolio_returns(prices, PORTFOLIO)
    print(f"\n Average daily return:   {returns.mean():.4%}")
    print(f"   Daily volatility:       {returns.std():.4%}")
    print(f"   Skewness:               {returns.skew():.3f}")
    print(f"   Excess kurtosis:        {returns.kurtosis():.3f}  <- >0 means fat tails")

    # 3. Compute VaR with all 3 methods
    print(f"\n Computing VaR with a {HISTORICAL_WINDOW}-day window...")
    res_hist       = historical_var(returns, PORTFOLIO_VALUE, CONFIDENCE_LEVEL, HISTORICAL_WINDOW)
    res_parametric = parametric_var(returns, PORTFOLIO_VALUE, CONFIDENCE_LEVEL, HISTORICAL_WINDOW)
    res_mc         = monte_carlo_var(returns, PORTFOLIO_VALUE, CONFIDENCE_LEVEL,
                                     HISTORICAL_WINDOW, N_MC_SIMULATIONS)

    results = {
        "Historical Simulation": res_hist,
        "Parametric (Gaussian)": res_parametric,
        "Monte Carlo":           res_mc,
    }

    # 4. Printing summary
    print_summary(results, PORTFOLIO_VALUE, CONFIDENCE_LEVEL)

    # 5. Backtesting
    violations, rolling_var, rolling_ret = backtesting(returns, CONFIDENCE_LEVEL, HISTORICAL_WINDOW)

    # 6. Stress test
    stress_test(PORTFOLIO_VALUE)

    # 7. Visual dashboard
    plot_results(results, returns, violations, rolling_var, rolling_ret,
                 res_mc["scenarios"], PORTFOLIO_VALUE)


if __name__ == "__main__":
    main()