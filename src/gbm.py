import numpy as np


def simulate_gbm(
    initial_price,
    drift,
    volatility,
    num_days,
    num_simulations=5,
    dt=1 / 252,
    seed=42
):
    """
    Simulate stock prices using Geometric Brownian Motion.

    dS = μSdt + σSdW

    Discrete solution:

    S(t+dt) = S(t) *
              exp((μ - 0.5σ²)dt + σ√dt Z)
    """

    np.random.seed(seed)

    prices = np.zeros((num_simulations, num_days + 1))
    prices[:, 0] = initial_price

    random_shocks = np.random.normal(
        0,
        1,
        size=(num_simulations, num_days)
    )

    for t in range(1, num_days + 1):

        prices[:, t] = prices[:, t - 1] * np.exp(
            (drift - 0.5 * volatility ** 2) * dt
            + volatility * np.sqrt(dt) * random_shocks[:, t - 1]
        )

    return prices
