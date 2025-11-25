# Merton Jump-Diffusion: vectorized simulation + Monte Carlo pricing for a European call
# Requirements: numpy, matplotlib, pandas
# Run with: python merton_mc.py  (or paste into a Jupyter cell)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt, exp

# -----------------------------
# Model / market parameters
# -----------------------------
S0 = 100.0       # initial underlying price
r = 0.01         # risk-free rate (annual)
q = 0.0          # continuous dividend yield
sigma = 0.2      # diffusive volatility (annual)
T = 1.0          # time to maturity (years)
K = 100.0        # strike price

# Jump parameters (Merton)
lam = 0.75       # Poisson intensity (expected jumps per year)
mu_j = -0.1      # mean of jump size in log (E[exp(Y)] relates to this)
sigma_j = 0.25   # std dev of jump size (log-space)

# Monte-Carlo parameters
M = 200_000      # number of simulated paths
N = 252          # number of time steps (e.g., daily)
seed = 123       # RNG seed for reproducibility

# -----------------------------
# Vectorized path simulator
# -----------------------------
def simulate_merton_paths(S0, r, q, sigma, T, lam, mu_j, sigma_j, M, N, seed=None):
    """
    Returns:
      S_paths: array shape (M, N+1) with price at times [0, dt, 2dt, ..., T]
      times:  array of times shape (N+1,)
    Notes:
      - Uses vectorized Poisson for jumps per dt and aggregates jump sizes per step.
      - If there are k jumps in a dt, total jump factor = exp(sum(Y_i)) where Y_i ~ N(mu_j, sigma_j^2).
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / N
    sqrt_dt = sqrt(dt)

    # Brownian increments
    dW = np.random.randn(M, N) * sqrt_dt

    # Poisson counts per (path, step) with mean lam*dt
    jump_counts = np.random.poisson(lam * dt, size=(M, N))

    # For each (path,step) when there are k jumps, sum of k independent N(mu_j, sigma_j^2)
    # has mean = k*mu_j and variance = k*sigma_j^2. When k=0, mean=0 var=0.
    total_jump_mean = jump_counts * mu_j
    total_jump_std = np.sqrt(jump_counts) * sigma_j
    # Draw the total jump log increment (Y_total) for each (path,step)
    # When jump_counts==0, total_jump_std==0 and rand*0 -> 0, giving Y_total==0.
    Y_total = np.random.randn(M, N) * total_jump_std + total_jump_mean
    jump_multiplier = np.exp(Y_total)  # >= 0, equals 1 when k=0

    # continuous part log-return per step:
    drift = (r - q - 0.5 * sigma**2) * dt
    log_returns = drift + sigma * dW + np.log(jump_multiplier)

    # cumulative log price
    logS = np.log(S0) + np.cumsum(log_returns, axis=1)
    S_paths = np.exp(logS)

    # include initial S0 column
    S_paths = np.hstack((np.full((M, 1), S0), S_paths))
    times = np.linspace(0, T, N+1)
    return S_paths, times


# -----------------------------
# Monte Carlo pricing
# -----------------------------
def price_european_call_mc(S0, K, r, T, S_T_paths):
    payoffs = np.maximum(S_T_paths - K, 0.0)
    discounted = np.exp(-r * T) * payoffs
    price = np.mean(discounted)
    stderr = np.std(discounted, ddof=1) / np.sqrt(len(discounted))
    return price, stderr

# Run simulation
S_paths, times = simulate_merton_paths(S0, r, q, sigma, T, lam, mu_j, sigma_j, M, N, seed=seed)
S_T = S_paths[:, -1]

mc_price, mc_se = price_european_call_mc(S0, K, r, T, S_T)

# Summarize
summary = {
    "MC Price": mc_price,
    "Std Error": mc_se,
    "Paths": M,
    "Steps": N,
    "Strike": K,
    "T (yrs)": T,
    "lambda": lam,
    "mu_j": mu_j,
    "sigma_j": sigma_j
}
print("---- Monte Carlo Result (Merton) ----")
for k,v in summary.items():
    print(f"{k:10s} : {v}")

# -----------------------------
# Plots: sample paths and terminal histogram
# -----------------------------
# Plot first 20 sample paths
plt.figure(figsize=(10,5))
for i in range(20):
    plt.plot(times, S_paths[i])
plt.title("Sample Merton Jump-Diffusion paths (20 samples)")
plt.xlabel("Time (years)")
plt.ylabel("Price")
plt.grid(True)
plt.show()

# Histogram of terminal prices
plt.figure(figsize=(8,4))
plt.hist(S_T, bins=100)
plt.title("Histogram of Terminal Prices S_T")
plt.xlabel("S_T")
plt.ylabel("Frequency")
plt.show()

# Optional: save results to DataFrame
df_res = pd.DataFrame({
    "mc_price": [mc_price],
    "mc_stderr": [mc_se],
    "S0": [S0],
    "K": [K],
    "T": [T],
    "sigma": [sigma],
    "lambda": [lam],
    "mu_j": [mu_j],
    "sigma_j": [sigma_j],
    "M": [M],
    "N": [N]
})
print("\nExample row of parameters/results:")
print(df_res.to_string(index=False))
