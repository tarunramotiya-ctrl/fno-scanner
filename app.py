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
from scanner import fetch_market_data, fetch_nse_live_options
from vol_profile import calculate_levels
from option_range import calculate_expiration_range

st.title("📈 Institutional F&O Trading Engine")
st.markdown("Advanced MTFA Scanner, Options Analytics, Backtesting, & Telegram Integration.")
st.caption(f"⏱️ Scan Executed At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Free Yahoo Finance data is typically delayed by 15-20 mins)")

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

st.sidebar.header("2. Live NSE Options Data")
st.sidebar.caption("The engine will automatically scrape the exact Put-Call ratio direct from NSE servers for your Top 12 stocks.")
fetch_options = st.sidebar.checkbox("Enable Live NSE Options Scraping (Beta)", value=False)

st.sidebar.header("3. API Telegram Setup")
tg_token = st.sidebar.text_input("Bot API Token", type="password")
tg_chat_id = st.sidebar.text_input("Chat ID")

# --- DATA PROCESS ENGINE ---
@st.cache_data(ttl=60) # Cache reduced to 1 minute for live market testing
def load_and_process_data(tickers, scrape_options):
    # Determine base fetch list (includes Nifty for RS baseline)
    fetch_list = list(set(tickers + ["^NSEI"]))
    market_data = fetch_market_data(fetch_list, period="1y", interval="1d")
    
    # Process Nifty50 RS Baseline
    nifty_data = market_data.get("^NSEI", pd.DataFrame())
    nifty_20d_ret = (nifty_data['Close'].iloc[-1] - nifty_data['Close'].iloc[-20]) / nifty_data['Close'].iloc[-20] if not nifty_data.empty and len(nifty_data) > 20 else 0

    # Fetch Hourly for MTFA (Limited to 1mo for accuracy)
    market_data_1h = fetch_market_data(tickers, period="1mo", interval="1h") if tickers else {}
    
    results = []
    
    for ticker in tickers:
        if ticker not in market_data: continue
        df = market_data[ticker]
        if len(df) < 50: continue
            
        # PREDICTIVE BREAKOUT LOGIC (Day 1 Engines)
        current_price = df['Close'].iloc[-1]
        daily_return = df['Close'].pct_change().iloc[-1] if not pd.isna(df['Close'].pct_change().iloc[-1]) else 0
        deviation = daily_return # Rename concept for UI column
        
        # Bollinger Band Squeeze Calculation (20-day)
        sma_20 = df['Close'].rolling(window=20).mean()
        std_20 = df['Close'].rolling(window=20).std()
        upper_bb = (sma_20 + (2 * std_20)).iloc[-1]
        lower_bb = (sma_20 - (2 * std_20)).iloc[-1]
        bb_width = (upper_bb - lower_bb) / sma_20.iloc[-1] if sma_20.iloc[-1] > 0 else 0.1
        
        # Distance from 20-Day Resistance
        recent_20d_high = df['High'].rolling(20).max().shift(1).iloc[-1]
        dist_from_high = (current_price - recent_20d_high) / recent_20d_high if recent_20d_high > 0 else 0
        
        # Relative Strength (RS) vs Nifty 50
        rs_rating = "NEUTRAL ➖"
        if not nifty_data.empty and len(df) >= 20:
            stock_20d_ret = (df['Close'].iloc[-1] - df['Close'].iloc[-20]) / df['Close'].iloc[-20]
            rs_diff = stock_20d_ret - nifty_20d_ret
            if rs_diff > 0.05:
                rs_rating = "OUTPERFORM 🔥"
            elif rs_diff < -0.05:
                rs_rating = "WEAK 🧨"

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
                
        # Upgraded Predictive Score Calculation
        base_momentum = abs(daily_return) * 100
        squeeze_multiplier = max(1.0, 0.10 / (bb_width + 0.001))
        capped_vol_spike = min(vol_spike, 5.0) 
        
        alignment_multiplier = 1.0
        if daily_return > 0 and -0.05 <= dist_from_high <= 0.05: alignment_multiplier += 0.5 
        if (deviation > 0 and macd_hist > 0) or (deviation < 0 and macd_hist < 0): alignment_multiplier += 0.3 
        if deviation > 0 and 55 <= current_rsi <= 75: alignment_multiplier += 0.2 
        if (deviation > 0 and hourly_trend == "BULL 🟢") or (deviation < 0 and hourly_trend == "BEAR 🔴"): alignment_multiplier += 0.5 
            
        score = base_momentum * squeeze_multiplier * capped_vol_spike * alignment_multiplier
        
        df_levels = calculate_levels(df)
        support = df_levels['Support'].iloc[-1] if not df_levels.empty else current_price
        resistance = df_levels['Resistance'].iloc[-1] if not df_levels.empty else current_price
        
        lower_bound, upper_bound, _ = calculate_expiration_range(df, days_to_expiry=20)
        
        results.append({
            "Ticker": ticker,
            "1H_Trend": hourly_trend,
            "RS_Rating": rs_rating,
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
    
    if not df_results.empty:
        # Default sort by Score
        top_stocks = df_results.sort_values(by="Score", ascending=False).head(12).copy()
        
        # LIVE NSE SCRAPER
        if scrape_options:
            pcr_list = []
            for t in top_stocks['Ticker']:
                pcr_list.append(fetch_nse_live_options(t))
            top_stocks['Live_PCR'] = pcr_list
            
        return top_stocks, market_data
        
    return pd.DataFrame(), market_data

# --- APP LAYOUT (TABS) ---
tab1, tab2, tab3 = st.tabs(["📊 Live Market Scanner", "🔄 Strategy Backtester", "🔍 Stock Deep-Dive"])

# --- TAB 1: LIVE SCANNER ---
with tab1:
    with st.spinner("Executing Multi-Timeframe Institutional Scan..."):
        top_12_df, full_market_data = load_and_process_data(tickers_to_scan, fetch_options)

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
    st.markdown("Run a simulated 2-year backtest using various Institutional Strategies to determine win rates.")
    
    colA, colB = st.columns([1, 3])
    with colA:
        test_stock = st.selectbox("Select Asset to Backtest", fno_tickers)
        strategy = st.radio("Select Strategy Engine", ["EMA 9/20 Crossover", "RSI Oversold Bounce", "Bollinger Breakout"])
        run_sim = st.button("Run Simulation >")
        
    if run_sim:
        with st.spinner(f"Simulating {strategy} Strategy..."):
            hist_data = fetch_market_data([test_stock], period="2y", interval="1d").get(test_stock, pd.DataFrame()).copy()
            
            if not hist_data.empty:
                if strategy == "EMA 9/20 Crossover":
                    hist_data['EMA_9'] = hist_data['Close'].ewm(span=9).mean()
                    hist_data['EMA_20'] = hist_data['Close'].ewm(span=20).mean()
                    hist_data['Signal'] = np.where(hist_data['EMA_9'] > hist_data['EMA_20'], 1, -1)
                    
                elif strategy == "RSI Oversold Bounce":
                    delta = hist_data['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    hist_data['RSI'] = 100 - (100 / (1 + rs))
                    # Buy when RSI < 35, Sell when RSI > 65
                    hist_data['Signal'] = np.where(hist_data['RSI'] < 35, 1, np.where(hist_data['RSI'] > 65, -1, 0))
                    hist_data['Signal'] = hist_data['Signal'].replace(to_replace=0, method='ffill')
                    
                elif strategy == "Bollinger Breakout":
                    sma_20 = hist_data['Close'].rolling(20).mean()
                    std_20 = hist_data['Close'].rolling(20).std()
                    hist_data['Upper_BB'] = sma_20 + (2 * std_20)
                    hist_data['Lower_BB'] = sma_20 - (2 * std_20)
                    hist_data['Signal'] = np.where(hist_data['Close'] > hist_data['Upper_BB'], 1, np.where(hist_data['Close'] < hist_data['Lower_BB'], -1, 0))
                    hist_data['Signal'] = hist_data['Signal'].replace(to_replace=0, method='ffill')
                
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
                m1.metric("Strategy Total Return", f"{end_strat}%")
                m2.metric("Buy & Hold Return", f"{end_bh}%")
            else:
                st.error("Failed to load historical data for simulation.")

# --- TAB 3: DEEP-DIVE PROFILE ---
with tab3:
    st.header("Single Stock Deep-Dive Analytics")
    st.markdown("Upload your specific Historical Data (CSV) and Option Chain (CSV) to generate exact Monthly Returns, PCR, IV, and Options Support/Resistance.")
    
    col_opt, col_hist = st.columns(2)
    with col_opt:
        st.subheader("1. Options Data Input (NSE Format)")
        option_file = st.file_uploader("Upload Option Chain CSV", type=['csv'], key="dd_opt_csv")
    with col_hist:
        st.subheader("2. Historical Data Input (1-2 Years)")
        hist_file = st.file_uploader("Upload Historical Equity CSV", type=['csv'], key="dd_hist_csv")
        
    if st.button("Generate Deep-Dive Profile >"):
        if option_file is not None or hist_file is not None:
            # 1. Option Analytics Extraction
            pcr_val, iv_val, opt_support, opt_resist = "N/A", "N/A", "N/A", "N/A"
            if option_file is not None:
                try:
                    # NSE Format usually has 2 headers. Read skipping first
                    opt_df = pd.read_csv(option_file, header=1)
                    # Convert to numeric safely
                    opt_df = opt_df.apply(pd.to_numeric, errors='coerce')
                    
                    # In standard NSE Option CSV (Skipping Row 0):
                    # Index 1 = Call OI, Index 4 = Call IV, Index 11 = Strike
                    # Index 17 = Put IV, Index 21 = Put OI
                    call_oi_col = opt_df.iloc[:, 1]
                    call_iv_col = opt_df.iloc[:, 4]
                    strike_col = opt_df.iloc[:, 11]
                    put_iv_col = opt_df.iloc[:, 17]
                    put_oi_col = opt_df.iloc[:, 21]
                    
                    ce_oi_total = call_oi_col.sum()
                    pe_oi_total = put_oi_col.sum()
                    
                    if ce_oi_total > 0: pcr_val = round(pe_oi_total / ce_oi_total, 2)
                    
                    iv_mean = pd.concat([call_iv_col, put_iv_col]).dropna().mean()
                    iv_val = round(iv_mean, 2) if not pd.isna(iv_mean) else "N/A"
                    
                    # Support is Strike with Max Put OI, Resistance is Strike with Max Call OI
                    max_put_idx = put_oi_col.idxmax()
                    max_call_idx = call_oi_col.idxmax()
                    opt_support = f"₹{strike_col.loc[max_put_idx]}" if not pd.isna(max_put_idx) else "N/A"
                    opt_resist = f"₹{strike_col.loc[max_call_idx]}" if not pd.isna(max_call_idx) else "N/A"
                except Exception as e:
                    st.warning(f"Could not parse Options CSV as strict NSE format. {e}")
                    
            # UI Row 1
            st.subheader("🎯 Options Profile (Derived from CSV)")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Options PCR", pcr_val)
            k2.metric("Implied Vol (IV)", iv_val)
            k3.metric("Max Put Support", opt_support)
            k4.metric("Max Call Resistance", opt_resist)
            st.divider()
            
            # 2. Historical Monthly Return
            if hist_file is not None:
                try:
                    st.subheader("📅 Monthly Percentage Returns (Month-over-Month)")
                    hist_df = pd.read_csv(hist_file)
                    
                    # Locate Date and Close Columns
                    date_col = next((c for c in hist_df.columns if 'Date' in c), None)
                    close_col = next((c for c in hist_df.columns if 'Close Price' in c or 'Close' in c), None)
                    
                    if date_col and close_col:
                        hist_df[date_col] = pd.to_datetime(hist_df[date_col])
                        hist_df = hist_df.sort_values(date_col)
                        hist_df.set_index(date_col, inplace=True)
                        
                        # Resample to Month End
                        montly_close = hist_df[close_col].resample('ME').last().dropna()
                        monthly_ret = montly_close.pct_change() * 100
                        
                        m_df = monthly_ret.to_frame(name='Return').dropna()
                        m_df['Month'] = m_df.index.strftime('%b %Y')
                        
                        # Create output strings: "Apr 2026: -10%"
                        out_strings = []
                        for idx, row in m_df.iterrows():
                            val = row['Return']
                            emoji = "🟢 Upward" if val > 0 else "🔴 Downward"
                            out_strings.append(f"**{row['Month']}**: {round(val, 2)}% {emoji}")
                        
                        # Display in columns
                        cols = st.columns(3)
                        for i, txt in enumerate(reversed(out_strings)): # Show newest first
                            cols[i % 3].markdown(txt)
                            
                        # Plot Bar
                        colors = ['rgba(0, 255, 0, 0.6)' if val > 0 else 'rgba(255, 0, 0, 0.6)' for val in m_df['Return']]
                        fig3 = go.Figure(data=[go.Bar(
                            x=m_df['Month'], 
                            y=m_df['Return'], 
                            marker_color=colors,
                            text=m_df['Return'].apply(lambda x: f"{round(x, 2)}%"),
                            textposition='auto'
                        )])
                        fig3.update_layout(title="Historical Month-over-Month Returns vs Previous Month Close", template="plotly_dark", height=400)
                        st.plotly_chart(fig3, use_container_width=True)
                    else:
                        st.warning("Could not find 'Date' or 'Close Price' in the historical CSV.")
                except Exception as e:
                    st.error(f"Error parsing Historical Data: {e}")
        else:
            st.info("Please upload at least one CSV to generate the profile.")
