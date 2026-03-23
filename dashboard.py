import pandas as pd
import numpy as np

# Import our custom modules
from scanner import fetch_market_data
from vol_profile import calculate_levels
from option_range import calculate_expiration_range

# 1. Define the F&O Universe (Testing with 10 major names; you can expand this to 207)
fno_tickers = [
    "RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "SBIN.NS", 
    "TCS.NS", "ITC.NS", "LT.NS", "TATAMOTORS.NS", "SUNPHARMA.NS"
]

print("Initializing F&O Market Dashboard...")

# 2. Fetch the Data
market_data = fetch_market_data(fno_tickers, period="1y")

# 3. Analyze and Rank the Stocks
results = []
for ticker, df in market_data.items():
    if len(df) < 50: 
        continue # Skip if the data didn't load properly
    
    # Calculate a Daily VWAP proxy
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (df['Typical_Price'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    current_price = df['Close'].iloc[-1]
    vwap_price = df['VWAP'].iloc[-1]
    
    # Calculate how far it is stretching away from VWAP (Trend Strength)
    deviation = (current_price - vwap_price) / vwap_price
    
    # Get Support & Resistance from Step 2
    df_levels = calculate_levels(df)
    support = df_levels['Support'].iloc[-1]
    resistance = df_levels['Resistance'].iloc[-1]
    
    # Get Monthly Expiration Boundaries from Step 3
    lower_bound, upper_bound, _ = calculate_expiration_range(df, days_to_expiry=20)
    
    # Package the data for this stock
    results.append({
        "Ticker": ticker,
        "Price": round(current_price, 2),
        "Trend_Strength": abs(deviation), # Using absolute value to find biggest movers, up or down
        "Direction": "UP 🟢" if deviation > 0 else "DOWN 🔴",
        "Support": round(support, 2),
        "Resistance": round(resistance, 2),
        "Exp_Lower": lower_bound,
        "Exp_Upper": upper_bound
    })

# 4. Filter for the Top 6 and Display
df_results = pd.DataFrame(results)

# Sort by the strongest trends (highest deviation) and take the top 6
top_6 = df_results.sort_values(by="Trend_Strength", ascending=False).head(6)

print("\n" + "="*85)
print("🚀 TOP 6 ACTIONABLE INDIAN F&O STOCKS 🚀")
print("="*85)
# Print the final table cleanly
print(top_6[['Ticker', 'Direction', 'Price', 'Support', 'Resistance', 'Exp_Lower', 'Exp_Upper']].to_string(index=False))
print("="*85)
print("Scan Complete. Ready for Live Trading verification.")