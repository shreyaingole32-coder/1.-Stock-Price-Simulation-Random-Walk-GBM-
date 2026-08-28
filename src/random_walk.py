"""
Random Walk simulation module for stock prices.
"""

from typing import Optional
import numpy as np


def simulate_random_walk(
    initial_price: float,
    num_days: int,
    daily_volatility: float,
    daily_drift: float = 0.0,
    num_simulations: int = 100,
    mode: str = "geometric",
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    Simulate stock price paths using a Random Walk model.

    In financial economics, stock prices are modeled as a Geometric Random Walk
    to ensure prices remain strictly non-negative (limited liability):

        ln(S_t / S_{t-1}) = mu_daily + sigma_daily * Z_t
        S_t = S_{t-1} * exp(mu_daily + sigma_daily * Z_t)

    where Z_t ~ N(0, 1) are i.i.d. standard normal random shocks.

    Parameters
    ----------
    initial_price : float
        Starting stock price S_0 (must be > 0).
    num_days : int
        Number of forward trading days to simulate (must be >= 1).
    daily_volatility : float
        Daily standard deviation of returns sigma_daily (must be >= 0).
    daily_drift : float, default 0.0
        Expected daily drift mu_daily (0.0 for pure martingale / un-drifted random walk).
    num_simulations : int, default 100
        Number of simulated paths (must be >= 1).
    mode : str, default 'geometric'
        Simulation mode: 'geometric' (multiplicative log-returns, S_t > 0)
        or 'arithmetic' (additive price increments with non-negativity floor).
    seed : int or None, default 42
        Random seed for exact reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (num_simulations, num_days + 1) containing simulated price paths,
        where column 0 is initial_price.
    """
    if initial_price <= 0:
        raise ValueError("Initial price must be strictly positive.")
    if num_days < 1:
        raise ValueError("Number of days must be at least 1.")
    if daily_volatility < 0:
        raise ValueError("Daily volatility cannot be negative.")
    if num_simulations < 1:
        raise ValueError("Number of simulations must be at least 1.")

    rng = np.random.default_rng(seed)

    if mode == "geometric":
        # Standard financial random walk on log-prices
        # Delta ln(S_t) ~ N(mu_daily, sigma_daily^2)
        shocks = rng.normal(
            loc=daily_drift,
            scale=daily_volatility,
            size=(num_simulations, num_days)
        )
        # Vectorized path construction using cumulative sum of log increments
        log_paths = np.zeros((num_simulations, num_days + 1))
        log_paths[:, 1:] = np.cumsum(shocks, axis=1)
        prices = initial_price * np.exp(log_paths)

    elif mode == "arithmetic":
        # Additive arithmetic increments: S_t = S_{t-1} + S_{t-1}*(mu + sigma*Z)
        returns = rng.normal(
            loc=daily_drift,
            scale=daily_volatility,
            size=(num_simulations, num_days)
        )
        growth_factors = np.maximum(0.0, 1.0 + returns)
        prices = np.zeros((num_simulations, num_days + 1))
        prices[:, 0] = initial_price
        prices[:, 1:] = initial_price * np.cumprod(growth_factors, axis=1)

    else:
        raise ValueError(f"Unknown simulation mode '{mode}'. Use 'geometric' or 'arithmetic'.")

    return prices

