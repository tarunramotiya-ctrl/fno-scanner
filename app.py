import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import requests
from datetime import datetime

# --- SETTINGS ---
st.set_page_config(page_title="Indian F&O Scanner Pro", layout="wide")

# Import custom modules
from scanner import fetch_market_data
from vol_profile import calculate_levels
from option_range import calculate_expiration_range

st.title("📈 Institutional F&O Trading Engine")
st.markdown("Advanced MTFA Scanner, Options Analytics, Backtesting, & Telegram Integration.")

# --- INITIALIZATION & CSV LOAD ---
csv_path = os.path.join(os.path.dirname(__file__), "stock update f&o.csv")
if os.path.exists(csv_path):
    df_csv = pd.read_csv(csv_path)
    df_csv.columns = df_csv.columns.str.strip()
    fno_tickers = (df_csv['SYMBOL'].str.strip() + '.NS').tolist() if 'SYMBOL' in df_csv.columns else ["RELIANCE.NS"]
else:
    st.error("Missing F&O CSV File.")
    fno_tickers = ["RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "SBIN.NS"]

# --- SIDEBAR UI ---
st.sidebar.title("⚙️ Pro Control Panel")

st.sidebar.header("1. Filter Watchlist:")
selected_tickers = st.sidebar.multiselect("Limit scan to specific stocks:", options=fno_tickers, default=[])
tickers_to_scan = selected_tickers if selected_tickers else fno_tickers

st.sidebar.header("2. Options Data (CE/PE)")
st.sidebar.caption("Provide your Options CSV. Make sure it contains 'SYMBOL', 'CE', and 'PE' columns.")
options_file = st.sidebar.file_uploader("Upload Market PE/CE Data", type=['csv'])

st.sidebar.header("3. API Telegram Setup")
tg_token = st.sidebar.text_input("Bot API Token", type="password")
tg_chat_id = st.sidebar.text_input("Chat ID")

# --- DATA PROCESS ENGINE ---
@st.cache_data(ttl=1800) # Cache for 30 minutes
def load_and_process_data(tickers):
    # Fetch Daily
    market_data = fetch_market_data(tickers, period="1y", interval="1d")
    
    # Fetch Hourly for MTFA (Limited to 1mo for accuracy)
    market_data_1h = fetch_market_data(tickers, period="1mo", interval="1h") if tickers else {}
    
    results = []
    
    for ticker, df in market_data.items():
        if len(df) < 50: continue
            
        df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['VWAP'] = (df['Typical_Price'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        
        current_price = df['Close'].iloc[-1]
        vwap_price = df['VWAP'].iloc[-1]
        deviation = (current_price - vwap_price) / vwap_price
        
        # MACD
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = (macd - signal).iloc[-1] if not pd.isna((macd - signal).iloc[-1]) else 0
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        
        # Volume
        avg_vol_20 = df['Volume'].rolling(window=20).mean().iloc[-1]
        vol_spike = df['Volume'].iloc[-1] / avg_vol_20 if avg_vol_20 > 0 else 1
        
        # MTFA (Hourly Trend Validation)
        hourly_trend = "FLAT"
        if ticker in market_data_1h and not market_data_1h[ticker].empty:
            df_1h = market_data_1h[ticker]
            if len(df_1h) >= 20:
                ema9_1h = df_1h['Close'].ewm(span=9).mean().iloc[-1]
                ema20_1h = df_1h['Close'].ewm(span=20).mean().iloc[-1]
                hourly_trend = "BULL 🟢" if ema9_1h > ema20_1h else "BEAR 🔴"
                
        # Pro Score Calculation
        score = abs(deviation) * vol_spike * 100
        
        df_levels = calculate_levels(df)
        support = df_levels['Support'].iloc[-1] if not df_levels.empty else current_price
        resistance = df_levels['Resistance'].iloc[-1] if not df_levels.empty else current_price
        
        lower_bound, upper_bound, _ = calculate_expiration_range(df, days_to_expiry=20)
        
        results.append({
            "Ticker": ticker,
            "1H_Trend": hourly_trend,
            "Daily_MACD": "BULL 📈" if macd_hist > 0 else "BEAR 📉",
            "Score": round(score, 2),
            "Price": round(current_price, 2),
            "RSI_14": round(current_rsi, 1),
            "Vol_Spike": f"{round(vol_spike, 1)}x",
            "Support": round(support, 2),
            "Resistance": round(resistance, 2),
            "Exp_Lower": lower_bound,
            "Exp_Upper": upper_bound
        })
        
    df_results = pd.DataFrame(results)
    
    # Options Data Integration
    if options_file is not None:
        try:
            opt_df = pd.read_csv(options_file)
            opt_df.columns = opt_df.columns.str.strip().str.upper()
            if 'SYMBOL' in opt_df.columns:
                opt_df['Ticker'] = opt_df['SYMBOL'].str.strip() + '.NS'
                
                # Automatically calculate Put-Call Ratio if CE and PE columns exist
                if 'PE' in opt_df.columns and 'CE' in opt_df.columns:
                    # Avoid division by zero
                    opt_df['PCR'] = np.where(opt_df['CE'] > 0, opt_df['PE'] / opt_df['CE'], 1)
                    opt_df['PCR'] = opt_df['PCR'].round(2)
                    df_results = df_results.merge(opt_df[['Ticker', 'PCR']], on='Ticker', how='left')
                    
        except Exception as e:
            st.sidebar.error(f"Error reading options CSV: {e}")
            
    if not df_results.empty:
        # Default sort by Score
        top_stocks = df_results.sort_values(by="Score", ascending=False).head(12)
        return top_stocks, market_data
        
    return pd.DataFrame(), market_data

# --- APP LAYOUT (TABS) ---
tab1, tab2 = st.tabs(["📊 Live Market Scanner", "🔄 Strategy Backtester"])

# --- TAB 1: LIVE SCANNER ---
with tab1:
    with st.spinner("Executing Multi-Timeframe Institutional Scan..."):
        top_12_df, full_market_data = load_and_process_data(tickers_to_scan)

    if not top_12_df.empty:
        st.subheader(f"🚀 Top Actionable Stocks ({len(top_12_df)} symbols)")
        
        # Render the ultimate DataFrame
        st.dataframe(
            top_12_df,
            hide_index=True,
            use_container_width=True
        )
        
        # Telegram API Dispatcher
        if tg_token and tg_chat_id:
            msg = "⚡ *PRO SCANNER EXCLUSIVE* ⚡\n\n"
            for r in top_12_df.itertuples():
                msg += f"*{r.Ticker}* | Score: {r.Score} | RSI: {r.RSI_14} | 1H: {r._2}\n"
                
            url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
            if st.button("Dispatch Alert to Telegram Channel ✉️"):
                res = requests.post(url, json={"chat_id": tg_chat_id, "text": msg, "parse_mode": "Markdown"})
                if res.status_code == 200:
                    st.success("Successfully delivered to Telegram!")
                else:
                    st.error(f"Failed to send: {res.text}")

        st.divider()
        st.subheader("Actionable Candlestick Scans")
        cols = st.columns(3)
        
        for idx, row in enumerate(top_12_df.itertuples()):
            ticker = row.Ticker
            df_ticker = full_market_data[ticker].tail(60).copy()
            df_ticker['EMA_9'] = df_ticker['Close'].ewm(span=9, adjust=False).mean()
            df_ticker['EMA_20'] = df_ticker['Close'].ewm(span=20, adjust=False).mean()
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
            
            # Candlestick
            fig.add_trace(go.Candlestick(x=df_ticker.index, open=df_ticker['Open'], high=df_ticker['High'], low=df_ticker['Low'], close=df_ticker['Close'], name="Price"), row=1, col=1)
            
            # EMAs
            fig.add_trace(go.Scatter(x=df_ticker.index, y=df_ticker['EMA_9'], line=dict(color='blue', width=1.5), name='9 EMA'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_ticker.index, y=df_ticker['EMA_20'], line=dict(color='orange', width=1.5), name='20 EMA'), row=1, col=1)

            # Volume Colors
            colors = ['rgba(0, 255, 0, 0.5)' if c >= o else 'rgba(255, 0, 0, 0.5)' for c, o in zip(df_ticker['Close'], df_ticker['Open'])]
            fig.add_trace(go.Bar(x=df_ticker.index, y=df_ticker['Volume'], marker_color=colors, name="Volume"), row=2, col=1)
            
            # Key Levels
            fig.add_hline(y=row.Support, line_color="green", row=1, col=1)
            fig.add_hline(y=row.Resistance, line_color="red", row=1, col=1)
            fig.add_hline(y=row.Exp_Lower, line_dash="dash", line_color="yellow", row=1, col=1)
            fig.add_hline(y=row.Exp_Upper, line_dash="dash", line_color="yellow", row=1, col=1)
            
            fig.update_layout(title=f"{ticker}", xaxis_rangeslider_visible=False, height=500, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
            with cols[idx % 3]: 
                st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: BACKTESTER ---
with tab2:
    st.header("Automated Strategy Backtester")
    st.markdown("Run a simulated 2-year backtest using the EMA 9/20 Crossover Momentum Strategy to determine win rates.")
    
    colA, colB = st.columns([1, 3])
    with colA:
        test_stock = st.selectbox("Select Asset to Backtest", fno_tickers)
        run_sim = st.button("Run Simulation >")
        
    if run_sim:
        with st.spinner("Processing historical order book..."):
            hist_data = fetch_market_data([test_stock], period="2y", interval="1d")[test_stock]
            hist_data['EMA_9'] = hist_data['Close'].ewm(span=9).mean()
            hist_data['EMA_20'] = hist_data['Close'].ewm(span=20).mean()
            
            # Rules: Buy when EMA9 > EMA20, Sell/Short when EMA9 < EMA20
            hist_data['Signal'] = np.where(hist_data['EMA_9'] > hist_data['EMA_20'], 1, -1)
            hist_data['Position'] = hist_data['Signal'].shift() # We enter on next candle
            hist_data['Strategy_Return'] = hist_data['Position'] * (hist_data['Close'].pct_change())
            
            # Cumulative Math
            cum_strat = (1 + hist_data['Strategy_Return']).cumprod()
            cum_bh = (1 + hist_data['Close'].pct_change()).cumprod()
            
            # Plot
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=hist_data.index, y=cum_strat, line=dict(color='green', width=2), name="Strategy Equity Curve"))
            fig2.add_trace(go.Scatter(x=hist_data.index, y=cum_bh, line=dict(color='gray', width=1, dash='dash'), name="Hold Equity Curve"))
            fig2.update_layout(title=f"{test_stock} Trading Output", height=500, template="plotly_dark")
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Metrics
            end_strat = round((cum_strat.iloc[-1] - 1) * 100, 2) if not pd.isna(cum_strat.iloc[-1]) else 0
            end_bh = round((cum_bh.iloc[-1] - 1) * 100, 2) if not pd.isna(cum_bh.iloc[-1]) else 0
            
            m1, m2 = st.columns(2)
            m1.metric("Bot Total Return", f"{end_strat}%")
            m2.metric("Buy & Hold Return", f"{end_bh}%")
