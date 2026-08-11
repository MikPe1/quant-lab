from typing import Optional

import yfinance as yf
import pandas as pd


def fetch_historical_data(
    ticker: str,
    start_date: str,
    end_date: Optional[str] = None
) -> pd.Series:
    """
    Fetches historical stock data for a given ticker and date range using yfinance.

    Args:
        ticker (str): The stock ticker symbol (e.g., "AAPL").
        start_date (str): The start date for fetching data (e.g., "2018-01-01").

    Returns:
        pd.Series: Historical closing prices, or an empty Series if data cannot be fetched.
    """
    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
        )
        if data.empty:
            print(f"No data found for ticker: {ticker} from {start_date}")
            return pd.Series(dtype=float)
        data.index = pd.to_datetime(data.index)
        data = data.sort_index()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(0)
        return data.iloc[:, 0]
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.Series(dtype=float)
