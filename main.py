"""
Main execution script for Stock Price Simulation (Random Walk vs GBM).
"""

import sys
import numpy as np

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

# ==========================================
# Configuration Parameters
# ==========================================
TICKER = "AAPL"
START_DATE = "2020-01-01"
END_DATE = "2025-01-01"
NUM_SIMULATIONS = 100
SEED = 42
TRADING_DAYS_PER_YEAR = 252


def main() -> None:
    print("=" * 60)
    print("  STOCK PRICE SIMULATION: RANDOM WALK vs GBM")
    print("=" * 60)

    # 1. Fetch Historical Data
    print(f"\n[1/4] Fetching historical market data for '{TICKER}'...")
    try:
        data = download_stock_data(TICKER, START_DATE, END_DATE)
        close_prices = extract_close_prices(data)
    except Exception as e:
        print(f"Error fetching data: {e}", file=sys.stderr)
        return

    actual_prices = close_prices.values
    returns = calculate_returns(close_prices, method="log")

    initial_price = float(actual_prices[0])
    final_price = float(actual_prices[-1])
    num_days = len(actual_prices) - 1

    # 2. Compute Market Statistics
    print("\n[2/4] Calculating return and volatility statistics...")
    stats = calculate_statistics(returns, trading_days=TRADING_DAYS_PER_YEAR)

    daily_volatility = stats["daily_volatility"]
    annualized_volatility = stats["annualized_volatility"]
    annualized_mean_return = stats["annualized_mean_return"]
    annualized_drift = stats["annualized_drift"]

    print("\n" + "-" * 40)
    print(" Market Statistics & Parameters")
    print("-" * 40)
    print(f" Ticker Symbol:            {TICKER}")
    print(f" Historical Range:         {START_DATE} to {END_DATE}")
    print(f" Observation Days:         {len(actual_prices)} trading days")
    print(f" Initial Price (S_0):      ${initial_price:.2f}")
    print(f" Final Actual Price (S_T): ${final_price:.2f}")
    print(f" Daily Volatility (sigma): {daily_volatility * 100:.3f}%")
    print(f" Annualized Log Return:    {annualized_mean_return * 100:.2f}%")
    print(f" Annualized Volatility:    {annualized_volatility * 100:.2f}%")
    print(f" Annualized Drift (mu):    {annualized_drift * 100:.2f}% (Ito-adjusted)")
    print("-" * 40)

    # 3. Simulate Monte Carlo Paths
    print(f"\n[3/4] Generating {NUM_SIMULATIONS} stochastic simulations (N={num_days} days)...")

    # Random Walk: Geometric random walk with zero drift (pure stochastic walk)
    rw_paths = simulate_random_walk(
        initial_price=initial_price,
        num_days=num_days,
        daily_volatility=daily_volatility,
        daily_drift=0.0,
        num_simulations=NUM_SIMULATIONS,
        mode="geometric",
        seed=SEED
    )

    # Geometric Brownian Motion: SDE with drift mu and volatility sigma
    gbm_paths = simulate_gbm(
        initial_price=initial_price,
        drift=annualized_drift,
        volatility=annualized_volatility,
        num_days=num_days,
        num_simulations=NUM_SIMULATIONS,
        dt=1.0 / TRADING_DAYS_PER_YEAR,
        seed=SEED
    )

    # 4. Model Evaluation & Performance
    print("\n[4/4] Evaluating simulated paths against historical trajectory...")

    rw_mean = np.mean(rw_paths, axis=0)
    gbm_mean = np.mean(gbm_paths, axis=0)

    rw_rmse = calculate_rmse(actual_prices, rw_mean)
    gbm_rmse = calculate_rmse(actual_prices, gbm_mean)

    rw_mae = calculate_mae(actual_prices, rw_mean)
    gbm_mae = calculate_mae(actual_prices, gbm_mean)

    rw_mape = calculate_mape(actual_prices, rw_mean)
    gbm_mape = calculate_mape(actual_prices, gbm_mean)

    rw_coverage = calculate_coverage(actual_prices, rw_paths, confidence_level=0.95)
    gbm_coverage = calculate_coverage(actual_prices, gbm_paths, confidence_level=0.95)

    rw_rmse_str = f"${rw_rmse:.2f}"
    gbm_rmse_str = f"${gbm_rmse:.2f}"
    rw_mae_str = f"${rw_mae:.2f}"
    gbm_mae_str = f"${gbm_mae:.2f}"
    rw_mape_str = f"{rw_mape:.2f}%"
    gbm_mape_str = f"{gbm_mape:.2f}%"
    rw_cov_str = f"{rw_coverage['coverage_pct']:.1f}%"
    gbm_cov_str = f"{gbm_coverage['coverage_pct']:.1f}%"

    print("\n" + "=" * 50)
    print(" Model Evaluation Metrics")
    print("=" * 50)
    print(f"{'Metric':<26} {'Random Walk':<12} {'GBM':<12}")
    print("-" * 50)
    print(f"{'RMSE ($):':<26} {rw_rmse_str:<12} {gbm_rmse_str:<12}")
    print(f"{'MAE ($):':<26} {rw_mae_str:<12} {gbm_mae_str:<12}")
    print(f"{'MAPE (%):':<26} {rw_mape_str:<12} {gbm_mape_str:<12}")
    print(f"{'95% Band Coverage (%):':<26} {rw_cov_str:<12} {gbm_cov_str:<12}")
    print("=" * 50)

    print("\n[Evaluation Note]")
    print(
        "Note: RMSE/MAE against the sample mean path evaluate how closely the\n"
        "deterministic expected trajectory aligns with historical prices. GBM is a\n"
        "stochastic scenario simulation tool (used in risk management and derivative\n"
        "pricing), NOT a point-forecasting model."
    )

    # 5. Visualizations
    print("\nGenerating visualization figures...")
    plot_simulations(
        rw_paths,
        title=f"Random Walk Simulation Paths vs Actual Price ({TICKER})",
        actual_prices=actual_prices
    )

    plot_simulations(
        gbm_paths,
        title=f"Geometric Brownian Motion (GBM) Simulation Paths vs Actual Price ({TICKER})",
        actual_prices=actual_prices
    )

    plot_comparison(
        actual_prices=actual_prices,
        random_walk_paths=rw_paths,
        gbm_paths=gbm_paths
    )

    print("\nSimulation and analysis completed successfully.")


if __name__ == "__main__":
    main()

