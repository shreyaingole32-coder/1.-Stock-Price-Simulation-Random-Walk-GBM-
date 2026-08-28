"""
Data loading and preprocessing module for stock price simulation.
"""

from typing import Union
import numpy as np
import pandas as pd
import yfinance as yf


def download_stock_data(
    ticker: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Download historical stock price data from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL', 'MSFT').
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.

    Returns
    -------
    pd.DataFrame
        Historical price dataset including adjusted OHLCV data.

    Raises
    ------
    ValueError
        If no data is returned for the given ticker and date range.
    """
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("Ticker symbol must be a non-empty string.")

    cleaned_ticker = ticker.strip().upper()
    data = yf.download(
        cleaned_ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        raise ValueError(
            f"No price data found for ticker '{cleaned_ticker}' between {start_date} and {end_date}. "
            f"Please verify the ticker symbol and date format."
        )

    return data


def extract_close_prices(data: pd.DataFrame) -> pd.Series:
    """
    Extract a clean 1D Series of Close prices from downloaded stock data.

    Handles both standard Index and MultiIndex column structures returned
    by modern versions of yfinance.

    Parameters
    ----------
    data : pd.DataFrame
        Stock data returned by download_stock_data.

    Returns
    -------
    pd.Series
        Clean 1D Series of adjusted close prices with datetime index.
    """
    if "Close" not in data.columns and not any("Close" in str(col) for col in data.columns):
        raise KeyError("Data does not contain a 'Close' price column.")

    close = data["Close"]

    # Flatten MultiIndex if yfinance returns a DataFrame for single ticker
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = close.dropna()
    close.name = "Close"

    if len(close) < 2:
        raise ValueError("Insufficient price observations (at least 2 required).")

    return close


def calculate_returns(
    data_or_prices: Union[pd.DataFrame, pd.Series],
    method: str = "log"
) -> pd.Series:
    """
    Calculate daily returns from price data.

    Parameters
    ----------
    data_or_prices : pd.DataFrame or pd.Series
        Stock DataFrame containing 'Close' or a Series of close prices.
    method : str, default 'log'
        Return calculation method: 'log' for continuous log returns ln(S_t / S_{t-1}),
        or 'simple' for arithmetic percentage returns (S_t - S_{t-1}) / S_{t-1}.

    Returns
    -------
    pd.Series
        Series of daily returns with the first NaN dropped.
    """
    if isinstance(data_or_prices, pd.DataFrame):
        prices = extract_close_prices(data_or_prices)
    elif isinstance(data_or_prices, pd.Series):
        prices = data_or_prices.dropna()
    else:
        raise TypeError("Input must be a pandas DataFrame or Series.")

    if method == "log":
        # Vectorized logarithmic returns: r_t = ln(S_t / S_{t-1})
        returns = np.log(prices / prices.shift(1))
    elif method == "simple":
        # Vectorized arithmetic returns: R_t = (S_t - S_{t-1}) / S_{t-1}
        returns = prices.pct_change()
    else:
        raise ValueError(f"Unknown return method '{method}'. Choose 'log' or 'simple'.")

    returns = returns.dropna()
    returns.name = f"{method}_return"
    return returns

