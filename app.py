import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from datetime import datetime, timedelta
import ta
import time

st.set_page_config(page_title="Stock ML Dashboard", page_icon="📈", layout="wide")

st.title("📈 Stock Market ML Prediction Dashboard")
st.markdown("Real-time stock analysis with machine learning predictions")

# Sidebar
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Stock Ticker", "AAPL").upper()
    period = st.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)
    
    st.divider()
    st.markdown("### Popular Stocks")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("AAPL"): ticker = "AAPL"
        if st.button("GOOGL"): ticker = "GOOGL"
        if st.button("NVDA"): ticker = "NVDA"
    with col2:
        if st.button("MSFT"): ticker = "MSFT"
        if st.button("TSLA"): ticker = "TSLA"
        if st.button("AMZN"): ticker = "AMZN"

# Better caching - cache for 1 hour
@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_data(ticker, period):
    try:
        time.sleep(0.5)  # Rate limit protection
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if df.empty:
            return None, None
        
        # Add technical indicators
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        df['MACD'] = ta.trend.MACD(df['Close']).macd()
        
        # Get basic info only
        info = {
            'longName': ticker,
            'previousClose': df['Close'].iloc[-2] if len(df) > 1 else df['Close'].iloc[-1],
            'marketCap': 0,
            'fiftyTwoWeekHigh': df['High'].max(),
            'sector': 'N/A',
            'industry': 'N/A',
            'website': 'N/A',
            'exchange': 'N/A'
        }
        
        return df, info
    except Exception as e:
        st.error(f"⚠️ Rate limit hit or API error. Please wait 60 seconds and try again.")
        return None, None

# Load data with spinner
with st.spinner(f"Loading {ticker} data..."):
    result = get_stock_data(ticker, period)
    
if result[0] is None:
    st.error(f"❌ Could not load data for {ticker}")
    st.info("**Troubleshooting:**\n- Wait 60 seconds and refresh\n- Check if ticker symbol is correct\n- Try a different stock")
    st.stop()

df, info = result

# Display key metrics
col1, col2, col3, col4 = st.columns(4)

current_price = df['Close'].iloc[-1]
prev_close = info.get('previousClose', df['Close'].iloc[-2])
change = current_price - prev_close
change_pct = (change / prev_close) * 100

col1.metric("Current Price", f"${current_price:.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
col2.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")
col3.metric("52W High", f"${info.get('fiftyTwoWeekHigh', 0):.2f}")
col4.metric("Days Loaded", len(df))

# Price chart
st.subheader("📊 Price Chart with Technical Indicators")

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name='Price'
))
fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='orange', width=1)))
fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='blue', width=1)))

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Price ($)",
    height=500,
    hovermode='x unified'
)
st.plotly_chart(fig, use_container_width=True)

# ML Prediction
st.subheader("🤖 ML Price Prediction (Next 5 Days)")

try:
    # Prepare features
    df_ml = df.dropna()
    features = ['Open', 'High', 'Low', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'MACD']
    X = df_ml[features].values
    y = df_ml['Close'].values
    
    # Train model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled[:-5], y[:-5])
    
    # Predict next 5 days
    last_data = X_scaled[-1].reshape(1, -1)
    predictions = []
    dates = []
    
    for i in range(5):
        pred = model.predict(last_data)[0]
        predictions.append(pred)
        dates.append(df.index[-1] + timedelta(days=i+1))
    
    # Display predictions
    col1, col2 = st.columns(2)
    
    with col1:
        pred_df = pd.DataFrame({
            'Date': dates,
            'Predicted Price': [f"${p:.2f}" for p in predictions]
        })
        st.dataframe(pred_df, use_container_width=True, hide_index=True)
    
    with col2:
        # Prediction chart
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=df.index[-30:],
            y=df['Close'].iloc[-30:],
            name='Historical',
            line=dict(color='blue')
        ))
        fig_pred.add_trace(go.Scatter(
            x=dates,
            y=predictions,
            name='Predicted',
            line=dict(color='red', dash='dash'),
            mode='lines+markers'
        ))
        fig_pred.update_layout(
            title="5-Day Price Forecast",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=300
        )
        st.plotly_chart(fig_pred, use_container_width=True)
    
    # Technical Analysis
    st.subheader("📉 Technical Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rsi_val = df['RSI'].iloc[-1]
        rsi_signal = "🔴 Overbought" if rsi_val > 70 else "🟢 Oversold" if rsi_val < 30 else "⚪ Neutral"
        st.metric("RSI (14)", f"{rsi_val:.2f}", rsi_signal)
    
    with col2:
        macd_signal = "🟢 Bullish" if df['MACD'].iloc[-1] > 0 else "🔴 Bearish"
        st.metric("MACD Signal", macd_signal)
    
    with col3:
        trend = "📈 Uptrend" if df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1] else "📉 Downtrend"
        st.metric("Trend", trend)
    
    # Volume analysis
    st.subheader("📊 Volume Analysis")
    fig_vol = go.Figure()
    colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] else 'green' for i in range(len(df))]
    fig_vol.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors))
    fig_vol.update_layout(height=250, showlegend=False)
    st.plotly_chart(fig_vol, use_container_width=True)

except Exception as e:
    st.error(f"Error in ML prediction: {str(e)}")

st.divider()
st.markdown("**Data Source:** Yahoo Finance (Free) | **ML Model:** Random Forest | **Cache:** 1 hour")
st.caption("⚠️ If you see rate limit errors, wait 60 seconds before trying again")
