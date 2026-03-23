import pandas as pd
import numpy as np

def calculate_levels(df, window=20):
    """
    Calculates dynamic Support and Resistance levels based on rolling highs and lows.
    """
    # Create a copy to avoid SettingWithCopyWarning
    data = df.copy()
    
    # Calculate rolling 20-day high (Resistance) and low (Support)
    data['Resistance'] = data['Close'].rolling(window=window).max()
    data['Support'] = data['Close'].rolling(window=window).min()
    
    # Drop the initial NaN values caused by the rolling window
    data.dropna(inplace=True)
    
    return data

if __name__ == "__main__":
    print("Volume Profile module loaded. Ready to calculate Support & Resistance.")
    # In the final dashboard, this will receive the data from scanner.py