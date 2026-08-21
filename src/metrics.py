import numpy as np


def calculate_statistics(returns):

    mean_return = np.mean(returns)
    volatility = np.std(returns)

    annualized_return = mean_return * 252
    annualized_volatility = volatility * np.sqrt(252)

    return {
        "daily_mean_return": mean_return,
        "daily_volatility": volatility,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility
    }


def calculate_rmse(actual, predicted):

    return np.sqrt(
        np.mean((actual - predicted) ** 2)
    )


def calculate_mae(actual, predicted):

    return np.mean(
        np.abs(actual - predicted)
    )
