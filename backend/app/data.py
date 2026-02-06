import yfinance as yf
import pandas as pd
from typing import Optional
from .config import DATA_PERIOD


def fetch_stock_data(symbol: str, period: Optional[str] = None) -> pd.DataFrame:
    """Fetch stock data with error handling and explicit typing.

    Raises ValueError when no data is available or download fails.
    """
    if period is None:
        period = DATA_PERIOD

    try:
        df: Optional[pd.DataFrame] = yf.download(symbol, period=period, progress=False)
        if df is None:
            raise ValueError(f"Failed to download data for symbol: {symbol}")
        if df.empty:
            raise ValueError(f"No data found for symbol: {symbol}")
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        raise ValueError(f"Failed to fetch data for {symbol}: {str(e)}")
