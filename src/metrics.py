"""
Statistical calculation and model evaluation metrics module.
"""

from typing import Dict, Tuple, Union
import numpy as np
import pandas as pd


def calculate_statistics(
    returns: Union[np.ndarray, pd.Series],
    trading_days: int = 252
) -> Dict[str, float]:
    """
    Calculate return and volatility summary statistics from daily log returns.

    Statistical Formulas
    --------------------
    - Daily Mean Return: bar{r} = (1/N) * sum(r_i)
    - Daily Volatility (sample std): s = sqrt( (1 / (N - 1)) * sum( (r_i - bar{r})^2 ) )
    - Annualized Mean Return: bar{r}_{ann} = bar{r} * trading_days
    - Annualized Volatility: sigma_{ann} = s * sqrt(trading_days)
    - Annualized Continuous Drift: mu = bar{r}_{ann} + 0.5 * sigma_{ann}^2
      (Derived from Ito's Lemma: E[ln(S_t/S_{t-1})] = (mu - 0.5*sigma^2)*dt)

    Parameters
    ----------
    returns : np.ndarray or pd.Series
        1D array or Series of daily returns.
    trading_days : int, default 252
        Number of trading days in a standard calendar year.

    Returns
    -------
    dict of str -> float
        Summary statistics dictionary.
    """
    if isinstance(returns, pd.Series):
        arr = returns.dropna().values
    else:
        arr = np.asarray(returns)
        arr = arr[~np.isnan(arr)]

    if len(arr) < 2:
        raise ValueError("At least 2 valid return observations are required for statistics.")

    daily_mean = float(np.mean(arr))
    daily_vol = float(np.std(arr, ddof=1))  # Sample standard deviation

    annualized_mean = daily_mean * trading_days
    annualized_vol = daily_vol * np.sqrt(trading_days)
    # Continuous drift mu for GBM accounting for Ito correction
    annualized_drift = annualized_mean + 0.5 * (annualized_vol ** 2)

    return {
        "daily_mean_return": daily_mean,
        "daily_volatility": daily_vol,
        "annualized_mean_return": annualized_mean,
        "annualized_volatility": annualized_vol,
        "annualized_drift": annualized_drift
    }


def calculate_rmse(
    actual: np.ndarray,
    predicted: np.ndarray
) -> float:
    """
    Calculate Root Mean Squared Error (RMSE) between actual and predicted price paths.

    RMSE = sqrt( (1/N) * sum( (actual_i - predicted_i)^2 ) )

    Parameters
    ----------
    actual : np.ndarray
        Ground truth actual price series (1D).
    predicted : np.ndarray
        Model predicted or mean simulated price series (1D).

    Returns
    -------
    float
        RMSE value in price units ($).
    """
    y_true = np.asarray(actual).flatten()
    y_pred = np.asarray(predicted).flatten()

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Shape mismatch: actual length ({len(y_true)}) != predicted length ({len(y_pred)})."
        )
    if len(y_true) == 0:
        raise ValueError("Input arrays must not be empty.")

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def calculate_mae(
    actual: np.ndarray,
    predicted: np.ndarray
) -> float:
    """
    Calculate Mean Absolute Error (MAE) between actual and predicted price paths.

    MAE = (1/N) * sum( |actual_i - predicted_i| )

    Parameters
    ----------
    actual : np.ndarray
        Ground truth actual price series (1D).
    predicted : np.ndarray
        Model predicted or mean simulated price series (1D).

    Returns
    -------
    float
        MAE value in price units ($).
    """
    y_true = np.asarray(actual).flatten()
    y_pred = np.asarray(predicted).flatten()

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Shape mismatch: actual length ({len(y_true)}) != predicted length ({len(y_pred)})."
        )
    if len(y_true) == 0:
        raise ValueError("Input arrays must not be empty.")

    return float(np.mean(np.abs(y_true - y_pred)))


def calculate_mape(
    actual: np.ndarray,
    predicted: np.ndarray
) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    MAPE = (100 / N) * sum( |(actual_i - predicted_i) / actual_i| )

    Parameters
    ----------
    actual : np.ndarray
        Ground truth actual price series (1D).
    predicted : np.ndarray
        Model predicted or mean simulated price series (1D).

    Returns
    -------
    float
        MAPE expressed as a percentage (%).
    """
    y_true = np.asarray(actual).flatten()
    y_pred = np.asarray(predicted).flatten()

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Shape mismatch: actual length ({len(y_true)}) != predicted length ({len(y_pred)})."
        )
    if np.any(y_true == 0):
        raise ZeroDivisionError("Actual prices contain zero; MAPE is undefined.")

    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)


def calculate_coverage(
    actual: np.ndarray,
    simulated_paths: np.ndarray,
    confidence_level: float = 0.95
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Calculate the empirical coverage of simulated Monte Carlo confidence bands.

    Computes what percentage of the actual historical price path fell within
    the simulated prediction interval (e.g. 2.5th to 97.5th percentile for 95% band).

    Parameters
    ----------
    actual : np.ndarray
        1D array of actual stock prices.
    simulated_paths : np.ndarray
        2D array of shape (num_simulations, num_days + 1).
    confidence_level : float, default 0.95
        Coverage confidence level (e.g., 0.95 for 95% interval).

    Returns
    -------
    dict
        Dictionary containing 'coverage_pct', 'lower_band', 'upper_band', 'median_path'.
    """
    y_true = np.asarray(actual).flatten()
    if simulated_paths.shape[1] != len(y_true):
        raise ValueError(
            f"Dimension mismatch: simulation length ({simulated_paths.shape[1]}) "
            f"!= actual price length ({len(y_true)})."
        )

    alpha = (1.0 - confidence_level) / 2.0
    lower_pct = alpha * 100.0
    upper_pct = (1.0 - alpha) * 100.0

    lower_band = np.percentile(simulated_paths, lower_pct, axis=0)
    upper_band = np.percentile(simulated_paths, upper_pct, axis=0)
    median_path = np.median(simulated_paths, axis=0)

    inside = (y_true >= lower_band) & (y_true <= upper_band)
    coverage_pct = float(np.mean(inside) * 100.0)

    return {
        "coverage_pct": coverage_pct,
        "lower_band": lower_band,
        "upper_band": upper_band,
        "median_path": median_path
    }

