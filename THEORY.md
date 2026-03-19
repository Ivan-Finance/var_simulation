# Theoretical Background

This document explains the theory behind each component of the VaR simulation project.

---

## 1. From Prices to Returns

The starting point of any risk model is not the price itself, but the **daily return**:

$$r_t = \frac{P_t - P_{t-1}}{P_{t-1}}$$

Where $P_t$ is the adjusted closing price on day $t$. Using **adjusted prices** is essential: they account for dividends and stock splits, ensuring that a price drop caused by a dividend payment is not mistakenly interpreted as a loss.

For a portfolio of multiple assets, the daily return is a **weighted average** of individual returns:

$$r_{portfolio,t} = \sum_{i=1}^{n} w_i \cdot r_{i,t}$$

Where $w_i$ is the portfolio weight of asset $i$, with $\sum w_i = 1$.

---

## 2. Value at Risk (VaR)

Value at Risk is defined as the **maximum loss not exceeded with a given probability** over a specific time horizon.

Formally, for a confidence level $c$ (e.g. 95%):

$$P(Loss > VaR_c) = 1 - c$$

In other words, VaR at 95% is the threshold such that losses exceed it only 5% of the time. This project computes **1-day VaR**, the standard horizon for trading book risk.

### Limitations of VaR

VaR is widely used but has a well-known weakness: it tells *how often* losses exceed the threshold, but not *by how much*. Two portfolios can have the same VaR but very different tail behavior. This is why CVaR was developed as a complement.

---

## 3. Conditional VaR (CVaR) — Expected Shortfall

CVaR, also called **Expected Shortfall (ES)**, answers the question VaR ignores:

> *"On the days when losses exceed the VaR, what is the average loss?"*

$$CVaR_c = E[Loss \mid Loss > VaR_c]$$

CVaR is considered a superior risk measure because it is **coherent** and fully captures the behavior of the loss tail. Since 2016, the Basel III framework requires banks to use Expected Shortfall instead of VaR for internal capital calculations.

---

## 4. The Three Methods

### 4.1 Historical Simulation

The simplest and most intuitive method. Given a window of $T$ historical returns, sort them from worst to best and take the $(1-c) \cdot T$-th observation as the VaR estimate.

**Advantages:**
- No distributional assumption: fat tails and skewness are captured naturally
- Simple to implement and explain

**Disadvantages:**
- Completely dependent on the historical window chosen
- Cannot predict events that never occurred in the sample period
- Slow to react to sudden changes in volatility

### 4.2 Parametric Method (Variance-Covariance)

This method assumes portfolio returns follow a **normal distribution** with mean $\mu$ and standard deviation $\sigma$, estimated from historical data. VaR is computed analytically:

$$VaR_c = -(\mu - z_c \cdot \sigma)$$

Where $z_c$ is the z-score corresponding to confidence level $c$ (e.g. $z_{0.95} = 1.645$).

The parametric CVaR is:

$$CVaR_c = -\left(\mu - \sigma \cdot \frac{\phi(z_c)}{1-c}\right)$$

Where $\phi$ is the standard normal probability density function.

**Advantages:**
- Fast and analytically clean
- Easy to extend to multi-asset portfolios using covariance matrices

**Disadvantages:**
- The normality assumption is unrealistic: financial returns exhibit **excess kurtosis** (fat tails) and **negative skewness**, meaning extreme losses occur far more often than a normal distribution would predict. This causes the parametric method to systematically **underestimate risk** during market stress.

### 4.3 Monte Carlo Simulation

Instead of using historical observations directly, Monte Carlo generates a large number of **synthetic scenarios** by sampling randomly from a calibrated distribution. In this project, returns are sampled from a normal distribution with $\mu$ and $\sigma$ estimated from the historical window:

$$r_{sim} \sim \mathcal{N}(\mu, \sigma^2)$$

VaR and CVaR are then computed from the empirical distribution of the $N$ simulated returns, in the same way as the historical method.

**Advantages:**
- Highly flexible: the normal distribution can be replaced with a t-Student (for fatter tails), a GARCH-based distribution (for time-varying volatility), or any other model
- With enough simulations, estimates are very stable

**Disadvantages:**
- Results depend heavily on the distributional assumption
- More computationally intensive than the parametric method

---

## 5. Fat Tails and Excess Kurtosis

A recurring theme in financial risk management is that **real return distributions are not normal**. They exhibit:

- **Negative skewness**: large negative returns (crashes) are more common than large positive ones
- **Excess kurtosis > 0**: extreme events occur far more frequently than a normal distribution predicts

This is why the historical method and Monte Carlo (with appropriate distributions) generally provide more realistic risk estimates than the parametric method, especially during periods of market stress.

---

## 6. Backtesting

A risk model is only useful if it is accurate. Backtesting verifies this by applying the model retroactively and checking whether the VaR was violated at the expected rate.

The procedure uses a **rolling window**: for each day $t$ in the test period, VaR is estimated using only the $T$ days preceding $t$, simulating what the model would have predicted in real time. A **violation** occurs when the actual return on day $t$ falls below the predicted VaR.

The expected violation rate is $1 - c$ (e.g. 5% for a 95% VaR). If the actual rate deviates significantly, the model is miscalibrated.

This is formalized by the **Kupiec test** (1995), a likelihood ratio test with hypotheses:

- $H_0$: actual violation rate $= 1 - c$ (model is correctly calibrated)
- $H_1$: actual violation rate $\neq 1 - c$ (model is over- or under-estimating risk)

---

## 7. Stress Testing

VaR measures risk under **normal market conditions**. It is not designed to capture the severity of extreme, low-probability events. Stress testing complements VaR by directly applying the returns observed during **known historical crises** to the current portfolio.
