"""
Visualization module for stock price simulations and model comparisons.
"""

from typing import Optional, Union
import matplotlib.pyplot as plt
import numpy as np


def plot_simulations(
    prices: np.ndarray,
    title: str,
    actual_prices: Optional[np.ndarray] = None,
    max_paths: int = 30,
    confidence_level: float = 0.95,
    save_path: Optional[str] = None
) -> None:
    """
    Plot Monte Carlo simulation paths with confidence interval bands and actual price.

    Parameters
    ----------
    prices : np.ndarray
        2D array of simulated price paths (num_simulations, num_days + 1).
    title : str
        Figure title.
    actual_prices : np.ndarray, optional
        1D array of historical ground-truth stock prices.
    max_paths : int, default 30
        Maximum number of individual trajectory paths to render.
    confidence_level : float, default 0.95
        Confidence level for shaded percentile interval (e.g. 0.95 for 95% band).
    save_path : str, optional
        File path to save the generated figure.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    num_simulations, num_steps = prices.shape
    num_paths_to_plot = min(num_simulations, max_paths)
    days = np.arange(num_steps)

    # Plot sample paths
    for i in range(num_paths_to_plot):
        ax.plot(
            days,
            prices[i],
            linewidth=0.8,
            alpha=0.35,
            color="#4A90E2"
        )

    # Calculate and plot statistics across all simulated paths
    mean_path = np.mean(prices, axis=0)
    alpha = (1.0 - confidence_level) / 2.0
    lower_band = np.percentile(prices, alpha * 100.0, axis=0)
    upper_band = np.percentile(prices, (1.0 - alpha) * 100.0, axis=0)

    ax.plot(
        days,
        mean_path,
        color="#1E3A8A",
        linewidth=2.2,
        linestyle="--",
        label="Simulated Mean Path"
    )

    ax.fill_between(
        days,
        lower_band,
        upper_band,
        color="#3B82F6",
        alpha=0.18,
        label=f"{int(confidence_level * 100)}% Confidence Interval"
    )

    if actual_prices is not None:
        actual_len = len(actual_prices)
        ax.plot(
            np.arange(actual_len),
            actual_prices,
            color="#DC2626",
            linewidth=2.2,
            label="Actual Historical Price"
        )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Trading Days (t)", fontsize=11)
    ax.set_ylabel("Stock Price ($ USD)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if plt.get_backend().lower() not in ["agg", "template", "cairo"]:
        plt.show()
    else:
        plt.close(fig)


def plot_comparison(
    actual_prices: np.ndarray,
    random_walk_paths: np.ndarray,
    gbm_paths: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Plot comprehensive comparison between Random Walk, GBM, and Historical Prices,
    including trajectory comparisons and terminal price distributions.

    Parameters
    ----------
    actual_prices : np.ndarray
        1D array of historical ground-truth stock prices.
    random_walk_paths : np.ndarray
        2D array of simulated Random Walk paths.
    gbm_paths : np.ndarray
        2D array of simulated GBM paths.
    save_path : str, optional
        File path to save the generated figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [2.2, 1.0]})

    days = np.arange(len(actual_prices))

    # Path statistics
    rw_mean = np.mean(random_walk_paths, axis=0)
    gbm_mean = np.mean(gbm_paths, axis=0)

    # 95% Confidence Intervals
    rw_lower = np.percentile(random_walk_paths, 2.5, axis=0)
    rw_upper = np.percentile(random_walk_paths, 97.5, axis=0)
    gbm_lower = np.percentile(gbm_paths, 2.5, axis=0)
    gbm_upper = np.percentile(gbm_paths, 97.5, axis=0)

    # Panel 1: Trajectory Comparison
    ax1.plot(days, actual_prices, color="#111827", linewidth=2.2, label="Actual Price")
    ax1.plot(days, rw_mean, color="#D97706", linewidth=2.0, linestyle="--", label="Random Walk (Mean)")
    ax1.plot(days, gbm_mean, color="#2563EB", linewidth=2.0, linestyle="-.", label="GBM (Mean)")

    ax1.fill_between(days, rw_lower, rw_upper, color="#F59E0B", alpha=0.12, label="Random Walk 95% Band")
    ax1.fill_between(days, gbm_lower, gbm_upper, color="#3B82F6", alpha=0.12, label="GBM 95% Band")

    ax1.set_title("Model Trajectories vs Historical Price", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Trading Days (t)", fontsize=11)
    ax1.set_ylabel("Stock Price ($ USD)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.9)

    # Panel 2: Terminal Price Distribution S_T
    rw_terminal = random_walk_paths[:, -1]
    gbm_terminal = gbm_paths[:, -1]
    actual_final = actual_prices[-1]

    ax2.hist(
        rw_terminal,
        bins=25,
        density=True,
        alpha=0.45,
        color="#F59E0B",
        label="Random Walk S_T"
    )
    ax2.hist(
        gbm_terminal,
        bins=25,
        density=True,
        alpha=0.45,
        color="#3B82F6",
        label="GBM S_T"
    )

    ax2.axvline(
        actual_final,
        color="#DC2626",
        linewidth=2.2,
        linestyle="-",
        label=f"Actual S_T (${actual_final:.1f})"
    )

    ax2.set_title("Terminal Price Distribution ($S_T$)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Terminal Price ($ USD)", fontsize=11)
    ax2.set_ylabel("Probability Density", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

