"""
Stock Price Simulation using Random Walk and Geometric Brownian Motion (GBM).
"""

from src.data_loader import (
    download_stock_data,
    extract_close_prices,
    calculate_returns
)
from src.random_walk import simulate_random_walk
from src.gbm import simulate_gbm
from src.metrics import (
    calculate_statistics,
    calculate_rmse,
    calculate_mae,
    calculate_mape,
    calculate_coverage
)
from src.visualization import (
    plot_simulations,
    plot_comparison
)

__all__ = [
    "download_stock_data",
    "extract_close_prices",
    "calculate_returns",
    "simulate_random_walk",
    "simulate_gbm",
    "calculate_statistics",
    "calculate_rmse",
    "calculate_mae",
    "calculate_mape",
    "calculate_coverage",
    "plot_simulations",
    "plot_comparison"
]

