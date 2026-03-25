import pandas as pd
import numpy as np
import plotly.graph_objects as go

def run_monte_carlo(df, days_to_simulate=30, num_simulations=100):
    """
    Run a Monte Carlo price path simulation based on historical daily returns.
    """
    if df.empty or len(df) < 50:
        return None, "Not enough data for simulation."

    # Using log returns for more accurate multi-period modeling
    returns = np.log(1 + df['Close'].pct_change().dropna())
    mu = returns.mean()
    vol = returns.std()
    
    last_price = df['Close'].iloc[-1]
    
    simulation_df = pd.DataFrame()
    
    for x in range(num_simulations):
        # Generate random variables
        daily_returns = np.random.normal(mu, vol, days_to_simulate)
        price_series = [last_price]
        
        for r in daily_returns:
            price_series.append(price_series[-1] * np.exp(r))
            
        simulation_df[f'Sim_{x}'] = price_series

    # Plotting
    fig = go.Figure()
    
    # Plot all simulations with low opacity
    for col in simulation_df.columns:
        fig.add_trace(go.Scatter(
            x=list(range(days_to_simulate + 1)), 
            y=simulation_df[col], 
            mode='lines', 
            line=dict(width=1, color='rgba(0, 150, 255, 0.08)'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
    # Find mean, upper, lower bands (95% CI)
    mean_path = simulation_df.mean(axis=1)
    upper_band = simulation_df.quantile(0.95, axis=1)
    lower_band = simulation_df.quantile(0.05, axis=1)
    
    fig.add_trace(go.Scatter(x=list(range(days_to_simulate + 1)), y=mean_path, name="Expected Path", line=dict(color='yellow', width=3)))
    fig.add_trace(go.Scatter(x=list(range(days_to_simulate + 1)), y=upper_band, name="95% Confidence High", line=dict(color='red', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=list(range(days_to_simulate + 1)), y=lower_band, name="95% Confidence Low", line=dict(color='green', width=2, dash='dash')))
    
    fig.update_layout(
        title=f"Advanced Monte Carlo Price Simulation ({days_to_simulate} Days, {num_simulations} Runs)",
        xaxis_title="Days Into Future",
        yaxis_title="Simulated Asset Price",
        template="plotly_dark",
        height=550,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    stats = {
        "Current Price": round(last_price, 2),
        "Expected Price (Mean)": round(mean_path.iloc[-1], 2),
        "95% Prob. High": round(upper_band.iloc[-1], 2),
        "95% Prob. Low": round(lower_band.iloc[-1], 2),
        "Daily Volatility": f"{round(vol * 100, 2)}%"
    }
    
    return fig, stats

def run_gap_fade_strategy(df):
    """
    Index specific strategy: Fade extreme morning gaps.
    Simulates buying dips on Gap Downs and shorting rips on Gap Ups.
    """
    if df.empty or len(df) < 20:
        return None, 0
        
    temp = df.copy()
    temp['Prev_Close'] = temp['Close'].shift(1)
    temp['Gap_%'] = ((temp['Open'] - temp['Prev_Close']) / temp['Prev_Close']) * 100
    
    # Gap Logic: If it gaps up > 0.8%, fade it (short opening to close). If it gaps down > 0.8%, buy opening to close.
    temp['Signal'] = np.where(temp['Gap_%'] < -0.8, 1, np.where(temp['Gap_%'] > 0.8, -1, 0))
    temp['Signal'] = temp['Signal'].replace(to_replace=0, method='ffill')
    
    temp['Position'] = temp['Signal'].shift()
    # Return from Open to Close that day (intra-day fade)
    temp['Strategy_Return'] = temp['Position'] * (temp['Close'] - temp['Open']) / temp['Open'] 
    
    cum_strat = (1 + temp['Strategy_Return'].fillna(0)).cumprod()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=temp.index, y=cum_strat, name="Intraday Gap Fade Equity", line=dict(color='#00ffcc', width=2.5)))
    
    cum_bh = (1 + temp['Close'].pct_change().fillna(0)).cumprod()
    fig.add_trace(go.Scatter(x=temp.index, y=cum_bh, name="Buy & Hold Baseline", line=dict(color='gray', dash='dash')))
    
    fig.update_layout(
        title="Institutional Gap Fade Strategy (Intraday Execution)", 
        template="plotly_dark", 
        height=400,
        yaxis_title="Cumulative Return"
    )
    
    end_ret = round((cum_strat.iloc[-1] - 1) * 100, 2)
    return fig, end_ret
