"""
Geometric Brownian Motion (GBM) simulation module for stock prices.
"""

from typing import Optional
import numpy as np


def simulate_gbm(
    initial_price: float,
    drift: float,
    volatility: float,
    num_days: int,
    num_simulations: int = 100,
    dt: float = 1 / 252,
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    Simulate stock price paths using Geometric Brownian Motion (GBM).

    Continuous Stochastic Differential Equation (SDE):
        dS_t = mu * S_t * dt + sigma * S_t * dW_t

    By Ito's Lemma, the exact discrete analytical solution is:
        S(t + dt) = S(t) * exp((mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z)

    where:
        mu        = Annualized expected rate of return (continuous drift)
        sigma     = Annualized volatility
        dt        = Time step in years (default: 1/252 for daily steps)
        Z ~ N(0, 1) = Standard normal random variable
        W_t       = Standard Brownian motion (Wiener process)

    Parameters
    ----------
    initial_price : float
        Starting stock price S_0 (must be > 0).
    drift : float
        Annualized continuous drift parameter mu.
    volatility : float
        Annualized volatility parameter sigma (must be >= 0).
    num_days : int
        Number of trading days to simulate (must be >= 1).
    num_simulations : int, default 100
        Number of simulated Monte Carlo paths (must be >= 1).
    dt : float, default 1/252
        Time step per trading day in years.
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
    if volatility < 0:
        raise ValueError("Volatility cannot be negative.")
    if num_simulations < 1:
        raise ValueError("Number of simulations must be at least 1.")
    if dt <= 0:
        raise ValueError("Time step dt must be strictly positive.")

    rng = np.random.default_rng(seed)

    # Generate standard normal random shocks Z ~ N(0, 1)
    shocks = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(num_simulations, num_days)
    )

    # Compute daily log increments: (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z
    deterministic_drift = (drift - 0.5 * (volatility ** 2)) * dt
    stochastic_diffusion = volatility * np.sqrt(dt) * shocks
    log_increments = deterministic_drift + stochastic_diffusion

    # Vectorized path accumulation using cumulative sum of log increments
    log_paths = np.zeros((num_simulations, num_days + 1))
    log_paths[:, 1:] = np.cumsum(log_increments, axis=1)

    prices = initial_price * np.exp(log_paths)
    return prices

