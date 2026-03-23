import numpy as np
import pandas as pd

def calculate_expiration_range(df, days_to_expiry=20):
    """
    Calculates the expected price range until expiration using Historical Volatility.
    Assumes ~20 trading days in a standard Indian F&O monthly expiration cycle.
    """
    # Create a copy to prevent modifying the original dataframe
    data = df.copy()
    
    # 1. Calculate daily logarithmic returns
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    
    # 2. Calculate the standard deviation of returns (Recent Volatility)
    # We use a 20-day rolling window to capture current market conditions
    daily_volatility = data['Log_Returns'].rolling(window=20).std()
    
    # 3. Project volatility forward for the remaining days to expiry
    # Volatility scales with the square root of time
    projected_volatility = daily_volatility.iloc[-1] * np.sqrt(days_to_expiry)
    
    current_price = data['Close'].iloc[-1]
    
    # 4. Calculate the 1 Standard Deviation range (68% probability zone)
    upper_bound = current_price * (1 + projected_volatility)
    lower_bound = current_price * (1 - projected_volatility)
    
    return round(lower_bound, 2), round(upper_bound, 2), round(current_price, 2)

if __name__ == "__main__":
    print("Option Range module loaded. Ready to calculate probability boxes.")
    # This will be called by our master dashboard