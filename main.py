import numpy as np

from src.data_loader import (
    download_stock_data,
    calculate_returns
)

from src.random_walk import simulate_random_walk

from src.gbm import simulate_gbm

from src.metrics import (
    calculate_statistics,
    calculate_rmse,
    calculate_mae
)

from src.visualization import (
    plot_simulations,
    plot_comparison
)



TICKER = "AAPL"

START_DATE = "2020-01-01"
END_DATE = "2025-01-01"

NUM_SIMULATIONS = 100
SEED = 42


# ============================================================
# 1. DOWNLOAD REAL STOCK DATA
# ============================================================

print(f"Downloading data for {TICKER}...")

data = download_stock_data(
    TICKER,
    START_DATE,
    END_DATE
)

close_prices = data["Close"]

if hasattr(close_prices, "columns"):
    close_prices = close_prices.iloc[:, 0]

close_prices = close_prices.dropna()

returns = calculate_returns(data)


# ============================================================
# 2. CALCULATE MARKET PARAMETERS
# ============================================================

statistics = calculate_statistics(
    returns.values
)

daily_volatility = statistics["daily_volatility"]
annualized_volatility = statistics["annualized_volatility"]

annualized_return = statistics["annualized_return"]

initial_price = float(close_prices.iloc[0])

num_days = len(close_prices) - 1


print("\nMarket Statistics")
print("-------------------------")
print(f"Initial Price: ${initial_price:.2f}")
print(
    f"Annualized Return: "
    f"{annualized_return * 100:.2f}%"
)
print(
    f"Annualized Volatility: "
    f"{annualized_volatility * 100:.2f}%"
)


# ============================================================
# 3. RANDOM WALK SIMULATION
# ============================================================

random_walk_paths = simulate_random_walk(
    initial_price=initial_price,
    num_days=num_days,
    daily_volatility=daily_volatility,
    num_simulations=NUM_SIMULATIONS,
    seed=SEED
)


# ============================================================
# 4. GBM SIMULATION
# ============================================================

gbm_paths = simulate_gbm(
    initial_price=initial_price,
    drift=annualized_return,
    volatility=annualized_volatility,
    num_days=num_days,
    num_simulations=NUM_SIMULATIONS,
    seed=SEED
)


# ============================================================
# 5. CALCULATE EXPECTED SIMULATED PATH
# ============================================================

random_walk_mean = np.mean(
    random_walk_paths,
    axis=0
)

gbm_mean = np.mean(
    gbm_paths,
    axis=0
)

actual_prices = close_prices.values


# ============================================================
# 6. ERROR METRICS
# ============================================================

random_walk_rmse = calculate_rmse(
    actual_prices,
    random_walk_mean
)

gbm_rmse = calculate_rmse(
    actual_prices,
    gbm_mean
)

random_walk_mae = calculate_mae(
    actual_prices,
    random_walk_mean
)

gbm_mae = calculate_mae(
    actual_prices,
    gbm_mean
)


print("\nModel Performance")
print("-------------------------")

print(
    f"Random Walk RMSE: "
    f"${random_walk_rmse:.2f}"
)

print(
    f"GBM RMSE: "
    f"${gbm_rmse:.2f}"
)

print(
    f"Random Walk MAE: "
    f"${random_walk_mae:.2f}"
)

print(
    f"GBM MAE: "
    f"${gbm_mae:.2f}"
)


# ============================================================
# 7. VISUALIZE RANDOM WALK
# ============================================================

plot_simulations(
    random_walk_paths,
    "Random Walk Stock Price Simulations",
    actual_prices=actual_prices
)


# ============================================================
# 8. VISUALIZE GBM
# ============================================================

plot_simulations(
    gbm_paths,
    "Geometric Brownian Motion Stock Price Simulations",
    actual_prices=actual_prices
)


# ============================================================
# 9. FINAL COMPARISON
# ============================================================

plot_comparison(
    actual_prices=actual_prices,
    random_walk_prices=random_walk_mean,
    gbm_prices=gbm_mean
)
