import matplotlib.pyplot as plt
import numpy as np


def plot_simulations(
    prices,
    title,
    actual_prices=None,
    max_paths=20
):

    plt.figure(figsize=(12, 6))

    num_paths = min(prices.shape[0], max_paths)

    for i in range(num_paths):
        plt.plot(
            prices[i],
            linewidth=1,
            alpha=0.6
        )

    if actual_prices is not None:
        plt.plot(
            np.arange(len(actual_prices)),
            actual_prices,
            linewidth=2,
            label="Actual Stock Price"
        )

    plt.title(title)
    plt.xlabel("Trading Days")
    plt.ylabel("Stock Price")
    plt.grid(alpha=0.3)

    if actual_prices is not None:
        plt.legend()

    plt.tight_layout()
    plt.show()


def plot_comparison(
    actual_prices,
    random_walk_prices,
    gbm_prices
):

    plt.figure(figsize=(13, 7))

    plt.plot(
        actual_prices,
        linewidth=2,
        label="Actual Price"
    )

    plt.plot(
        random_walk_prices,
        linestyle="--",
        linewidth=1.5,
        label="Random Walk"
    )

    plt.plot(
        gbm_prices,
        linestyle="--",
        linewidth=1.5,
        label="GBM"
    )

    plt.title("Actual vs Simulated Stock Price")
    plt.xlabel("Trading Days")
    plt.ylabel("Stock Price")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()
