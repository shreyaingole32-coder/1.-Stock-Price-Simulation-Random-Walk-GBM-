import numpy as np


def simulate_random_walk(
    initial_price,
    num_days,
    daily_volatility=0.02,
    num_simulations=5,
    seed=42
):
    """
    Simulate stock prices using a simple random walk.

    P(t+1) = P(t) + sigma * epsilon

    where epsilon ~ N(0,1)
    """

    np.random.seed(seed)

    random_shocks = np.random.normal(
        loc=0,
        scale=daily_volatility,
        size=(num_simulations, num_days)
    )

    prices = np.zeros((num_simulations, num_days + 1))
    prices[:, 0] = initial_price

    for t in range(1, num_days + 1):
        prices[:, t] = (
            prices[:, t - 1]
            + initial_price * random_shocks[:, t - 1]
        )

    return prices
