import yfinance as yf
import pandas as pd


def download_stock_data(ticker, start_date, end_date):
    """
    Download historical stock price data.

    Parameters:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date, e.g. '2020-01-01'.
        end_date (str): End date, e.g. '2025-01-01'.

    Returns:
        pd.DataFrame: Historical stock data.
    """

    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    return data


def calculate_returns(data):
    """
    Calculate daily logarithmic returns.
    """

    close = data["Close"]

    # Handle MultiIndex columns returned by yfinance
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    log_returns = (close / close.shift(1)).apply(
        lambda x: __import__("numpy").log(x)
    )

    return log_returns.dropna()
