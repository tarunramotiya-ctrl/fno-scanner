import yfinance as yf
import pandas as pd

def fetch_market_data(tickers, period="1y", interval="1d"):
    """
    Downloads daily historical data for a list of NSE tickers.
    """
    print(f"Fetching data for {len(tickers)} stocks with interval {interval}...")
    
    # Download data for all tickers at once (groups by ticker)
    data = yf.download(tickers, period=period, interval=interval, group_by='ticker')
    
    clean_data = {}
    
    for ticker in tickers:
        # Extract individual stock data and drop any empty rows
        df = data[ticker].dropna()
        clean_data[ticker] = df
        print(f"Successfully loaded {len(df)} days of data for {ticker}")
        
    return clean_data

# Test the Data Engine with 5 major Nifty F&O stocks
if __name__ == "__main__":
    nifty_test_list = ["RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "SBIN.NS"]
    
    # Run the fetcher
    market_data = fetch_market_data(nifty_test_list, period="1y")
    
    # Preview the data for Reliance
    print("\n--- RELIANCE.NS Data Preview ---")
    print(market_data["RELIANCE.NS"].tail())