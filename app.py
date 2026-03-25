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
from index_simulator import run_monte_carlo, run_gap_fade_strategy

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
st.sidebar.caption("By default, we scan the top 10 stocks. Type below to add more, or use the Deep-Dive tab to manually search 1 stock instantly.")
safe_default = fno_tickers[:10] if len(fno_tickers) >= 10 else fno_tickers
selected_tickers = st.sidebar.multiselect("Target Specific Stocks:", options=fno_tickers, default=safe_default)
tickers_to_scan = selected_tickers if selected_tickers else safe_default

st.sidebar.header("2. Live NSE Options Data")
st.sidebar.caption("The engine will automatically scrape the exact Put-Call ratio direct from NSE servers for your Top 12 stocks.")
fetch_options = st.sidebar.checkbox("Enable Live NSE Options Scraping (Beta)", value=False)

st.sidebar.header("3. API Telegram Setup")
tg_token = st.sidebar.text_input("Bot API Token", type="password")
tg_chat_id = st.sidebar.text_input("Chat ID")

# --- DATA PROCESS ENGINE ---
@st.cache_data(ttl=300) # Cache increased to 5 minutes to prevent Cloud Container Thread exhaustion
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
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Live Market Scanner", "🔄 Strategy Backtester", "🔍 Stock Deep-Dive", "🏢 Portfolio Matrix", "📈 Index Prediction & Lab"])

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
    st.markdown("Run a simulated mathematical backtest using various Institutional Strategies to determine true statistical edge.")
    
    colA, colB = st.columns([1, 3])
    with colA:
        test_stock = st.selectbox("Select Asset to Backtest", fno_tickers)
        
        duration_map = {"1 Year": "1y", "2 Years": "2y", "3 Years": "3y", "5 Years": "5y", "10 Years": "10y"}
        dur_choice = st.selectbox("Historical Duration", list(duration_map.keys()), index=3) # Default 5y
        
        strategy_options = ["EMA 9/20 Crossover", "RSI Oversold Bounce", "Bollinger Breakout", "200 DMA Reversal", "VWAP Bounce/Reversal", "Combined Master (RSI + BB)", "🌟 Auto-Optimize (Test All)"]
        strategy = st.radio("Select Strategy Engine", strategy_options)
        run_sim = st.button("Run Simulation >")
        
    if run_sim:
        with st.spinner(f"Simulating {strategy} Strategy over {dur_choice}..."):
            hist_data = fetch_market_data([test_stock], period=duration_map[dur_choice], interval="1d").get(test_stock, pd.DataFrame()).copy()
            
            if not hist_data.empty:
                strats_to_run = ["EMA 9/20 Crossover", "RSI Oversold Bounce", "Bollinger Breakout", "200 DMA Reversal", "VWAP Bounce/Reversal", "Combined Master (RSI + BB)"] if "Auto-Optimize" in strategy else [strategy]
                
                best_strat_name = ""
                best_strat_ret = -9999.0
                best_cum_strat = None
                
                for s in strats_to_run:
                    temp_data = hist_data.copy()
                    if s == "EMA 9/20 Crossover":
                        temp_data['EMA_9'] = temp_data['Close'].ewm(span=9).mean()
                        temp_data['EMA_20'] = temp_data['Close'].ewm(span=20).mean()
                        temp_data['Signal'] = np.where(temp_data['EMA_9'] > temp_data['EMA_20'], 1, -1)
                        
                    elif s == "RSI Oversold Bounce":
                        delta = temp_data['Close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                        rs = gain / loss
                        temp_data['RSI'] = 100 - (100 / (1 + rs))
                        temp_data['Signal'] = np.where(temp_data['RSI'] < 35, 1, np.where(temp_data['RSI'] > 65, -1, 0))
                        temp_data['Signal'] = temp_data['Signal'].replace(to_replace=0, method='ffill')
                        
                    elif s == "Bollinger Breakout":
                        sma_20 = temp_data['Close'].rolling(20).mean()
                        std_20 = temp_data['Close'].rolling(20).std()
                        temp_data['Upper_BB'] = sma_20 + (2 * std_20)
                        temp_data['Lower_BB'] = sma_20 - (2 * std_20)
                        temp_data['Signal'] = np.where(temp_data['Close'] > temp_data['Upper_BB'], 1, np.where(temp_data['Close'] < temp_data['Lower_BB'], -1, 0))
                        temp_data['Signal'] = temp_data['Signal'].replace(to_replace=0, method='ffill')
                        
                    elif s == "200 DMA Reversal":
                        temp_data['SMA_200'] = temp_data['Close'].rolling(200).mean()
                        # Buy when Crossing UP over 200 DMA, Short when Crossing DOWN below 200 DMA
                        temp_data['Signal'] = np.where((temp_data['Close'] > temp_data['SMA_200']) & (temp_data['Close'].shift(1) <= temp_data['SMA_200'].shift(1)), 1,
                                              np.where((temp_data['Close'] < temp_data['SMA_200']) & (temp_data['Close'].shift(1) >= temp_data['SMA_200'].shift(1)), -1, 0))
                        temp_data['Signal'] = temp_data['Signal'].replace(to_replace=0, method='ffill')
                        
                    elif s == "VWAP Bounce/Reversal":
                        typical_price = (temp_data['High'] + temp_data['Low'] + temp_data['Close']) / 3
                        # We use a 20-Day Anchored/Rolling VWAP approximation
                        temp_data['VWAP_20'] = (typical_price * temp_data['Volume']).rolling(20).sum() / temp_data['Volume'].rolling(20).sum()
                        temp_data['Signal'] = np.where((temp_data['Close'] > temp_data['VWAP_20']) & (temp_data['Close'].shift(1) <= temp_data['VWAP_20'].shift(1)), 1,
                                              np.where((temp_data['Close'] < temp_data['VWAP_20']) & (temp_data['Close'].shift(1) >= temp_data['VWAP_20'].shift(1)), -1, 0))
                        temp_data['Signal'] = temp_data['Signal'].replace(to_replace=0, method='ffill')
                        
                    elif s == "Combined Master (RSI + BB)":
                        sma_20 = temp_data['Close'].rolling(20).mean()
                        std_20 = temp_data['Close'].rolling(20).std()
                        upper_bb = sma_20 + (2 * std_20)
                        lower_bb = sma_20 - (2 * std_20)
                        
                        delta = temp_data['Close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                        rs = gain / loss
                        rsi = 100 - (100 / (1 + rs))
                        
                        # Master logic: Buy when breaking out OR deeply oversold. Sell when crashing OR deeply overbought.
                        temp_data['Signal'] = np.where((rsi < 35) | (temp_data['Close'] > upper_bb), 1, 
                                              np.where((rsi > 65) | (temp_data['Close'] < lower_bb), -1, 0))
                        temp_data['Signal'] = temp_data['Signal'].replace(to_replace=0, method='ffill')
                
                    temp_data['Position'] = temp_data['Signal'].shift()
                    temp_data['Strategy_Return'] = temp_data['Position'] * (temp_data['Close'].pct_change())
                    
                    cum_strat = (1 + temp_data['Strategy_Return']).cumprod()
                    end_ret = round((cum_strat.iloc[-1] - 1) * 100, 2) if not pd.isna(cum_strat.iloc[-1]) else 0
                    
                    if end_ret > best_strat_ret:
                        best_strat_ret = end_ret
                        best_strat_name = s
                        best_cum_strat = cum_strat
                
                # Execute Best Plot
                cum_bh = (1 + hist_data['Close'].pct_change()).cumprod()
                
                if "Auto-Optimize" in strategy:
                    st.success(f"🏆 Auto-Optimization Complete! The absolute best strategy for {test_stock} is **{best_strat_name}** with a {best_strat_ret}% Return.")
                
                # Plot
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=hist_data.index, y=best_cum_strat, line=dict(color='green', width=2), name=f"{best_strat_name} Equity"))
                fig2.add_trace(go.Scatter(x=hist_data.index, y=cum_bh, line=dict(color='gray', width=1, dash='dash'), name="Hold Equity Curve"))
                fig2.update_layout(title=f"{test_stock} Trading Output ({best_strat_name})", height=500, template="plotly_dark")
                
                st.plotly_chart(fig2, use_container_width=True)
                
                # Metrics
                end_bh = round((cum_bh.iloc[-1] - 1) * 100, 2) if not pd.isna(cum_bh.iloc[-1]) else 0
                
                m1, m2 = st.columns(2)
                m1.metric(f"Winner: {best_strat_name}", f"{best_strat_ret}%")
                m2.metric("Buy & Hold Return", f"{end_bh}%")
            else:
                st.error("Failed to load historical data for simulation.")

# --- TAB 3: DEEP-DIVE PROFILE ---
with tab3:
    st.header("Single Stock Deep-Dive Analytics")
    st.markdown("Upload your Option Chain (CSV) to generate exact Monthly Returns, PCR, IV, and Options Support/Resistance.")
    
    col_sel, col_up = st.columns(2)
    with col_sel:
        focus_stock = st.selectbox("Select Target Asset", fno_tickers, key="dd_stock")
    
    col_opt, col_hist = st.columns(2)
    with col_opt:
        st.subheader("1. Options Data Input (NSE Format)")
        option_file = st.file_uploader("Upload Option Chain CSV", type=['csv'], key="dd_opt_csv")
    with col_hist:
        st.subheader("2. Historical Data (Automated)")
        st.success("Historical Volatility & Net Change calculations are now fully automated! No CSV upload required.")
        
    if st.button("Generate Deep-Dive Profile >"):
        with st.spinner(f"Analyzing {focus_stock} 3-Year History..."):
            hd = fetch_market_data([focus_stock], period="3y", interval="1d").get(focus_stock, pd.DataFrame()).copy()
            
        if hd.empty:
            st.error(f"Failed to fetch data for {focus_stock} from Yahoo Finance.")
        else:
            current_price = hd['Close'].iloc[-1]
            
            # 1. Option Analytics Extraction
            pcr_val, iv_val, opt_support, opt_resist = "N/A", "N/A", "N/A", "N/A"
            ext_support, ext_resist = "N/A", "N/A" # Extended (S2/R2)
            
            if option_file is not None:
                try:
                    import io
                    raw_bytes = option_file.getvalue()
                    try:
                        opt_text = raw_bytes.decode('utf-8')
                    except UnicodeDecodeError:
                        opt_text = raw_bytes.decode('latin-1')
                        
                    lines = opt_text.split('\n')
                    # Find header row containing STRIKE
                    header_idx = 0
                    for i, line in enumerate(lines):
                        if "STRIKE" in line.upper():
                            header_idx = i
                            break
                            
                    opt_df = pd.read_csv(io.StringIO(opt_text), header=header_idx)
                    
                    colnames = [str(x).upper().strip() for x in opt_df.columns]
                    strike_idx = next((i for i, x in enumerate(colnames) if "STRIKE" in x), None)
                    
                    # Usually: Index 1 is CE OI, Index 21 is PE OI. 
                    oi_indices = [i for i, x in enumerate(colnames) if x in ["OI", "OI.1"] or ("OPEN INT" in x and "CH" not in x)]
                    iv_indices = [i for i, x in enumerate(colnames) if "IV" in x and "CH" not in x]
                    
                    if strike_idx is not None and len(oi_indices) >= 2:
                        call_oi_col = pd.to_numeric(opt_df.iloc[:, oi_indices[0]].astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce')
                        put_oi_col = pd.to_numeric(opt_df.iloc[:, oi_indices[-1]].astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce')
                        strike_col = pd.to_numeric(opt_df.iloc[:, strike_idx].astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce')
                        
                        ce_oi_total = call_oi_col.sum()
                        pe_oi_total = put_oi_col.sum()
                        
                        if ce_oi_total > 0: pcr_val = round(pe_oi_total / ce_oi_total, 2)
                        
                        if len(iv_indices) >= 2:
                            call_iv = pd.to_numeric(opt_df.iloc[:, iv_indices[0]].astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce')
                            put_iv = pd.to_numeric(opt_df.iloc[:, iv_indices[-1]].astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce')
                            iv_mean = pd.concat([call_iv, put_iv]).dropna().mean()
                            if not pd.isna(iv_mean): iv_val = round(iv_mean, 2)
                        
                        # Max Support logic (Top put OI strikes)
                        valid_put_oi = put_oi_col.dropna()
                        valid_call_oi = call_oi_col.dropna()
                        
                        if not valid_put_oi.empty:
                            put_sorted = valid_put_oi.sort_values(ascending=False)
                            opt_support = f"₹{strike_col.loc[put_sorted.index[0]]}"
                            if len(put_sorted) > 1: ext_support = f"₹{strike_col.loc[put_sorted.index[1]]}"
                            
                        if not valid_call_oi.empty:
                            call_sorted = valid_call_oi.sort_values(ascending=False)
                            opt_resist = f"₹{strike_col.loc[call_sorted.index[0]]}"
                            if len(call_sorted) > 1: ext_resist = f"₹{strike_col.loc[call_sorted.index[1]]}"
                except Exception as e:
                    st.warning(f"Could not parse Options CSV. Found error: {e}")

            # UI Row 1
            st.subheader(f"🎯 Options Profile for {focus_stock} (Derived from CSV)")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Options PCR", pcr_val)
            k2.metric("Implied Vol (IV)", iv_val)
            k3.metric("Max Put Supports (S1, S2)", f"{opt_support} | {ext_support}")
            k4.metric("Max Call Resists (R1, R2)", f"{opt_resist} | {ext_resist}")
            st.divider()
            
            # 2. Automated Historical Monthly Return (User's Custom Algorithm)
            st.subheader("📅 Monthly Percentage Returns & Volatility (Custom Alg)")
            try:
                monthly_data = hd.resample('ME').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'})
                monthly_data['Volatility % (Swing)'] = ((monthly_data['High'] - monthly_data['Low']) / monthly_data['Low']) * 100
                monthly_data['Net Change % (Direction)'] = ((monthly_data['Close'] - monthly_data['Open']) / monthly_data['Open']) * 100
                monthly_data['Volatility % (Swing)'] = monthly_data['Volatility % (Swing)'].round(2)
                monthly_data['Net Change % (Direction)'] = monthly_data['Net Change % (Direction)'].round(2)
                
                monthly_data = monthly_data.sort_index(ascending=False)
                monthly_data['Month_Str'] = monthly_data.index.strftime('%B %Y')
                
                out_strings = []
                for idx, row in monthly_data.head(36).iterrows():
                    val = row['Net Change % (Direction)']
                    vol = row['Volatility % (Swing)']
                    emoji = "🟢 Upward" if val > 0 else "🔴 Downward"
                    out_strings.append(f"**{row['Month_Str']}**: {val}% {emoji}  \n*(Swing: {vol}%)*")
                
                if out_strings:
                    cols = st.columns(3)
                    for i, txt in enumerate(out_strings):
                        cols[i % 3].markdown(txt)
                        
                    m_df = monthly_data.head(36).copy().sort_index(ascending=True)
                    colors = ['rgba(0, 255, 0, 0.6)' if val > 0 else 'rgba(255, 0, 0, 0.6)' for val in m_df['Net Change % (Direction)']]
                    
                    fig3 = go.Figure()
                    fig3.add_trace(go.Bar(x=m_df['Month_Str'], y=m_df['Net Change % (Direction)'], marker_color=colors, name="Net Change %"))
                    fig3.add_trace(go.Scatter(x=m_df['Month_Str'], y=m_df['Volatility % (Swing)'], mode='lines+markers', line=dict(color='yellow', width=2), name="Volatility % (Swing)"))
                    fig3.update_layout(title="Historical Monthly Net Change vs Volatility Swing", template="plotly_dark", height=400, barmode='group')
                    st.plotly_chart(fig3, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error calculating Automated Historical Data: {e}")

# --- TAB 4: PORTFOLIO MATRIX SCREENER ---
with tab4:
    st.header("🏢 Mass Automated Strategy Matrix")
    st.markdown("Batch scan all F&O stocks across 6 Strategies and 5 Macro-Timeframes to mathematically enforce your Institutional Rules. Generates the final CSV DataFrame.")
    
    if st.button("🚀 Run Master Matrix Calculation"):
        with st.spinner("Executing 6,000 simulations... Please leave this tab open until complete."):
            
            # Fetch globally to prevent threaded timeout (download 10 years for all)
            df_full = fetch_market_data(fno_tickers, period="10y", interval="1d")
            
            rule_map = {
                '1y': (252, 17.0),
                '2y': (504, 35.0),
                '3y': (756, 50.0),
                '5y': (1260, 100.0),
                '10y': (2520, 200.0)
            }
            
            strategies = [
                "EMA 9/20 Crossover", "RSI Oversold Bounce", "Bollinger Breakout", 
                "200 DMA Reversal", "VWAP Bounce/Reversal", "Combined Master (RSI + BB)"
            ]
            
            results_dict = {strat: [] for strat in strategies}
            
            prog_bar = st.progress(0)
            status_text = st.empty()
            tot_stocks = len(fno_tickers)
            
            for i, ticker in enumerate(fno_tickers):
                prog_bar.progress((i + 1) / tot_stocks)
                status_text.text(f"Crunching Mathematics for {ticker} ({i+1}/{tot_stocks})...")
                
                stock_data = df_full.get(ticker, pd.DataFrame())
                # Need absolute minimum 200 days to even test 1 year
                if stock_data.empty or len(stock_data) < 200: continue
                
                for s in strategies:
                    passed_timeframes = []
                    
                    for tf_label, (days_cut, min_thresh) in rule_map.items():
                        sliced_data = stock_data.tail(days_cut).copy()
                        
                        # Only proceed if there is enough historical data for this specific timeframe!
                        if len(sliced_data) < (days_cut * 0.8): continue
                        
                        # Apply specific mathematical logic to sliced_data
                        if s == "EMA 9/20 Crossover":
                            sliced_data['EMA_9'] = sliced_data['Close'].ewm(span=9).mean()
                            sliced_data['EMA_20'] = sliced_data['Close'].ewm(span=20).mean()
                            sliced_data['Signal'] = np.where(sliced_data['EMA_9'] > sliced_data['EMA_20'], 1, -1)
                        elif s == "RSI Oversold Bounce":
                            delta = sliced_data['Close'].diff()
                            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                            rs = gain / loss
                            sliced_data['RSI'] = 100 - (100 / (1 + rs))
                            sliced_data['Signal'] = np.where(sliced_data['RSI'] < 35, 1, np.where(sliced_data['RSI'] > 65, -1, 0))
                            sliced_data['Signal'] = sliced_data['Signal'].replace(to_replace=0, method='ffill')
                        elif s == "Bollinger Breakout":
                            sma_20 = sliced_data['Close'].rolling(20).mean()
                            std_20 = sliced_data['Close'].rolling(20).std()
                            upper_bb = sma_20 + (2 * std_20)
                            lower_bb = sma_20 - (2 * std_20)
                            sliced_data['Signal'] = np.where(sliced_data['Close'] > upper_bb, 1, np.where(sliced_data['Close'] < lower_bb, -1, 0))
                            sliced_data['Signal'] = sliced_data['Signal'].replace(to_replace=0, method='ffill')
                        elif s == "200 DMA Reversal":
                            sma_200 = sliced_data['Close'].rolling(200).mean()
                            sliced_data['Signal'] = np.where((sliced_data['Close'] > sma_200) & (sliced_data['Close'].shift(1) <= sma_200.shift(1)), 1,
                                                      np.where((sliced_data['Close'] < sma_200) & (sliced_data['Close'].shift(1) >= sma_200.shift(1)), -1, 0))
                            sliced_data['Signal'] = sliced_data['Signal'].replace(to_replace=0, method='ffill')
                        elif s == "VWAP Bounce/Reversal":
                            typical_price = (sliced_data['High'] + sliced_data['Low'] + sliced_data['Close']) / 3
                            vwap_20 = (typical_price * sliced_data['Volume']).rolling(20).sum() / sliced_data['Volume'].rolling(20).sum()
                            sliced_data['Signal'] = np.where((sliced_data['Close'] > vwap_20) & (sliced_data['Close'].shift(1) <= vwap_20.shift(1)), 1,
                                                      np.where((sliced_data['Close'] < vwap_20) & (sliced_data['Close'].shift(1) >= vwap_20.shift(1)), -1, 0))
                            sliced_data['Signal'] = sliced_data['Signal'].replace(to_replace=0, method='ffill')
                        elif s == "Combined Master (RSI + BB)":
                            sma_20 = sliced_data['Close'].rolling(20).mean()
                            std_20 = sliced_data['Close'].rolling(20).std()
                            upper_bb = sma_20 + (2 * std_20)
                            lower_bb = sma_20 - (2 * std_20)
                            delta = sliced_data['Close'].diff()
                            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                            rs = gain / loss
                            rsi = 100 - (100 / (1 + rs))
                            sliced_data['Signal'] = np.where((rsi < 35) | (sliced_data['Close'] > upper_bb), 1, 
                                                      np.where((rsi > 65) | (sliced_data['Close'] < lower_bb), -1, 0))
                            sliced_data['Signal'] = sliced_data['Signal'].replace(to_replace=0, method='ffill')
                            
                        sliced_data['Position'] = sliced_data['Signal'].shift()
                        sliced_data['Strategy_Return'] = sliced_data['Position'] * (sliced_data['Close'].pct_change())
                        cum_strat = (1 + sliced_data['Strategy_Return']).cumprod()
                        
                        if len(cum_strat) > 0 and not pd.isna(cum_strat.iloc[-1]):
                            end_ret = round((cum_strat.iloc[-1] - 1) * 100, 2)
                            if end_ret > min_thresh:
                                passed_timeframes.append(tf_label.upper())
                                
                    if passed_timeframes:
                        clean_symbol = ticker.replace('.NS', '')
                        results_dict[s].append(f"{clean_symbol}({','.join(passed_timeframes)})")
                        
            # Pad arrays for DataFrame conversion
            max_len = max([len(v) for v in results_dict.values()]) if results_dict else 0
            if max_len > 0:
                for k in results_dict.keys():
                    while len(results_dict[k]) < max_len:
                        results_dict[k].append("")
                        
                final_df = pd.DataFrame(results_dict)
                status_text.empty()
                st.success("🎉 Matrix Computation Complete! View the results below.")
                st.dataframe(final_df, use_container_width=True)
                
                csv_bytes = final_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Excel Matrix",
                    data=csv_bytes,
                    file_name="Master_Strategy_Matrix.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No stocks passed the intense mathematical thresholds.")

# --- TAB 5: INDEX PREDICTION & LAB ---
with tab5:
    st.header("📈 Index Prediction & Strategy Lab")
    st.markdown("Powerful prediction setup and backtesting environment exclusively for Nifty 50 and BankNifty. Test signals and experiments on the indices.")
    
    # Top section: Visual Analytics
    col_idx1, col_idx2 = st.columns(2)
    with col_idx1:
        idx_period = st.selectbox("Select History Period for Charts", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3, key="idx_period")
    
    if st.button("Generate Technical Charts 🚀", key="btn_idx"):
        with st.spinner("Fetching Index Data..."):
            idx_data = fetch_market_data(["^NSEI", "^NSEBANK"], period=idx_period, interval="1d")
            nifty_df = idx_data.get("^NSEI", pd.DataFrame())
            banknifty_df = idx_data.get("^NSEBANK", pd.DataFrame())
            
            if not nifty_df.empty and not banknifty_df.empty:
                chart_col1, chart_col2 = st.columns(2)
                
                for title, df, col, color_th in [("NIFTY 50 (^NSEI)", nifty_df, chart_col1, '#00d4ff'), ("BANKNIFTY (^NSEBANK)", banknifty_df, chart_col2, '#ff9900')]:
                    df_plot = df.copy()
                    df_plot['EMA_9'] = df_plot['Close'].ewm(span=9, adjust=False).mean()
                    df_plot['EMA_20'] = df_plot['Close'].ewm(span=20, adjust=False).mean()
                    
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
                    fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name="Price"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['EMA_9'], line=dict(color='cyan', width=1.5), name='9 EMA'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['EMA_20'], line=dict(color='magenta', width=1.5), name='20 EMA'), row=1, col=1)
                    
                    colors = ['rgba(0, 255, 0, 0.5)' if c >= o else 'rgba(255, 0, 0, 0.5)' for c, o in zip(df_plot['Close'], df_plot['Open'])]
                    fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['Volume'], marker_color=colors, name="Volume"), row=2, col=1)
                    
                    fig.update_layout(title=title, xaxis_rangeslider_visible=False, height=500, showlegend=False, template="plotly_dark")
                    with col:
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Could not fetch data for both indices. Please try again.")

    st.divider()
    
    # Bottom Section: Advanced Simulation & Backtesting
    st.subheader("🧮 Prediction Engine & Signal Backtester")
    ext_col1, ext_col2 = st.columns([1, 2])
    with ext_col1:
        st.write("Configure Experimental Settings")
        sim_asset = st.selectbox("Target Index", ["^NSEI", "^NSEBANK"], format_func=lambda x: "NIFTY 50" if x == "^NSEI" else "BANKNIFTY")
        sim_history = st.selectbox("Historical Training Data", ["1y", "2y", "3y", "5y"], index=0, key="sim_training")
        
        st.markdown("##### Monte Carlo Setup")
        mc_days = st.slider("Days Into Future", min_value=10, max_value=90, value=30, step=5)
        mc_runs = st.slider("Simulation Paths", min_value=100, max_value=1000, value=250, step=50)
        run_mc = st.button("▶ Run Monte Carlo Prediction", use_container_width=True)
        
        st.markdown("##### Backtest Experimental Signals")
        index_strat = st.selectbox("Custom Metric/Signal", ["Intraday Gap Fade", "Mean Reversion (Future)"])
        run_idx_strat = st.button("▶ Run Index Signal", use_container_width=True)
        
    with ext_col2:
        if run_mc:
            with st.spinner(f"Running {mc_runs} Monte Carlo Predictions for {sim_asset}..."):
                sim_data = fetch_market_data([sim_asset], period=sim_history, interval="1d").get(sim_asset, pd.DataFrame())
                if not sim_data.empty:
                    fig_mc, mc_stats = run_monte_carlo(sim_data, days_to_simulate=mc_days, num_simulations=mc_runs)
                    if fig_mc:
                        st.plotly_chart(fig_mc, use_container_width=True)
                        st.subheader("Simulation Probabilities (At End Date)")
                        s1, s2, s3, s4 = st.columns(4)
                        s1.metric("Current", mc_stats["Current Price"])
                        s2.metric("Expected", mc_stats["Expected Price (Mean)"])
                        s3.metric("95% Bull High", mc_stats["95% Prob. High"])
                        s4.metric("95% Bear Low", mc_stats["95% Prob. Low"])
                        st.info(f"**Mathematical Daily Volatility constraint used:** {mc_stats['Daily Volatility']}.")
                    else:
                        st.error(mc_stats)
                else:
                    st.error("Failed to load historical data for simulation.")
                    
        elif run_idx_strat:
            with st.spinner(f"Backtesting {index_strat} signal on {sim_asset}..."):
                sim_data = fetch_market_data([sim_asset], period=sim_history, interval="1d").get(sim_asset, pd.DataFrame())
                if not sim_data.empty:
                    if index_strat == "Intraday Gap Fade":
                        fig_strat, strat_ret = run_gap_fade_strategy(sim_data)
                        if fig_strat:
                            st.plotly_chart(fig_strat, use_container_width=True)
                            st.success(f"Signal Strategy Final Return: **{strat_ret}%**")
                        else:
                            st.error("Not enough data to run strategy.")
                    else:
                        st.warning("Algorithm under development by engine.")
                else:
                    st.error("Failed to load data for signal testing.")
        else:
            st.info("👈 System Idle: Configure settings on left pane and hit Run to initiate Index predictive calculations.")

