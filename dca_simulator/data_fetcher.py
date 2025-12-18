import yfinance as yf
import pandas as pd
###
def fetch_historical_data(ticker: str, start_date: str) -> pd.DataFrame:
    """
    Fetches historical stock data for a given ticker and date range using yfinance.

    Args:
        ticker (str): The stock ticker symbol (e.g., "AAPL").
        start_date (str): The start date for fetching data (e.g., "2018-01-01").

    Returns:
        pd.DataFrame: A DataFrame containing historical stock data with a 'Close' column,
                      or an empty DataFrame if data cannot be fetched.
    """
    try:
        # yfinance fetches data up to the current date if end date is not specified
        data = yf.download(ticker, start=start_date, auto_adjust=False)  # Explicitly set to avoid future warning
        if data.empty:
            print(f"No data found for ticker: {ticker} from {start_date}")
            return pd.DataFrame()
        # Ensure the index is a DatetimeIndex and sort it
        data.index = pd.to_datetime(data.index)
        data = data.sort_index()
        # Handle MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(0)
        return data.iloc[:, 0]  # Return the first column, which is 'Close' prices
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()


if __name__ == '__main__':
    # Example usage
    ticker_symbol = "MSFT"
    start = "2020-01-01"
    historical_df = fetch_historical_data(ticker_symbol, start)

    if not historical_df.empty:
        print(f"Successfully fetched data for {ticker_symbol}:")
        print(historical_df.head())
        print(historical_df.tail())
    else:
        print(f"Failed to fetch data for {ticker_symbol}.")

    ticker_symbol_invalid = "INVALIDTICKER"
    historical_df_invalid = fetch_historical_data(ticker_symbol_invalid, start)
    if historical_df_invalid.empty:
        print(f"Correctly handled invalid ticker: {ticker_symbol_invalid}")
