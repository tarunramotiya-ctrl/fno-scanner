import yfinance as yf
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def fetch_market_data(tickers, period="1y", interval="1d"):
    """
    Downloads daily historical data for a list of NSE tickers.
    """
    print(f"Fetching data for {len(tickers)} stocks with interval {interval}...")
    
    # Download data for all tickers at once (groups by ticker)
    data = yf.download(tickers, period=period, interval=interval, group_by='ticker', threads=False, progress=False)
    
    clean_data = {}
    
    for ticker in tickers:
        # Extract individual stock data and drop any empty rows
        df = data[ticker].dropna()
        clean_data[ticker] = df
        print(f"Successfully loaded {len(df)} days of data for {ticker}")
        
    return clean_data

def fetch_nse_live_options(symbol):
    """
    Scrapes live PCR from NSE India. Uses a Session to bypass basic firewalls.
    """
    clean_symbol = symbol.replace('.NS', '').replace('&', '%26').upper()
    url = f"https://www.nseindia.com/api/option-chain-equities?symbol={clean_symbol}"
    
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'accept': '*/*',
        'accept-encoding': 'gzip, deflate, br',
        'accept-language': 'en-US,en;q=0.9'
    }
    
    try:
        session = requests.Session()
        req_retry = Retry(total=3, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=req_retry))
        
        # Ping homepage to generate requisite cookies
        session.get("https://www.nseindia.com", headers=headers, timeout=5)
        # Fetch actual API Option Chain
        response = session.get(url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if not data: return "N/A"
            
            records = data.get('records', {}).get('data', [])
            if not records: return "N/A"
            
            total_ce_oi = sum(r.get('CE', {}).get('openInterest', 0) for r in records if 'CE' in r)
            total_pe_oi = sum(r.get('PE', {}).get('openInterest', 0) for r in records if 'PE' in r)
            if total_ce_oi == 0: return "N/A"
            
            pcr = total_pe_oi / total_ce_oi
            return round(pcr, 2)
        return "N/A"
    except Exception as e:
        print(f"NSE Scrape Error {symbol}: {e}")
        return "N/A"

# Test the Data Engine with 5 major Nifty F&O stocks
if __name__ == "__main__":
    nifty_test_list = ["RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "SBIN.NS"]
    
    # Run the fetcher
    market_data = fetch_market_data(nifty_test_list, period="1y")
    
    # Preview the data for Reliance
    print("\n--- RELIANCE.NS Data Preview ---")
    print(market_data["RELIANCE.NS"].tail())
