import pandas as pd
import numpy as np

def add_features(df: pd.DataFrame):
    """Add technical indicators to dataframe"""
    df = df.copy()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    
    # Handle division by zero
    rs = gain / loss
    rs = rs.replace([np.inf, -np.inf], 1)
    df["RSI"] = 100 - (100 / (1 + rs))

    df.dropna(inplace=True)
    return df
