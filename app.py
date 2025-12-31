import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from datetime import datetime, timedelta
import ta

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
    if st.button("AAPL - Apple"): ticker = "AAPL"
    if st.button("MSFT - Microsoft"): ticker = "MSFT"
    if st.button("GOOGL - Google"): ticker = "GOOGL"
    if st.button("TSLA - Tesla"): ticker = "TSLA"
    if st.button("NVDA - Nvidia"): ticker = "NVDA"

# Fetch data
@st.cache_data(ttl=3600)
def get_stock_data(ticker, period):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    
    # Add technical indicators
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    
    return df, stock.info

try:
    df, info = get_stock_data(ticker, period)
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = df['Close'].iloc[-1]
    prev_close = info.get('previousClose', df['Close'].iloc[-2])
    change = current_price - prev_close
    change_pct = (change / prev_close) * 100
    
    col1.metric("Current Price", f"${current_price:.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
    col2.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")
    col3.metric("Market Cap", f"${info.get('marketCap', 0)/1e9:.2f}B")
    col4.metric("52W High", f"${info.get('fiftyTwoWeekHigh', 0):.2f}")
    
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
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='blue')))
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=500,
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # ML Prediction
    st.subheader("🤖 ML Price Prediction (Next 5 Days)")
    
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
            'Predicted Price': predictions
        })
        st.dataframe(pred_df, use_container_width=True)
    
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
            title="Price Forecast",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=300
        )
        st.plotly_chart(fig_pred, use_container_width=True)
    
    # Technical Analysis
    st.subheader("📉 Technical Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("RSI (14)", f"{df['RSI'].iloc[-1]:.2f}", 
                  "Overbought" if df['RSI'].iloc[-1] > 70 else "Oversold" if df['RSI'].iloc[-1] < 30 else "Neutral")
    
    with col2:
        macd_signal = "Bullish" if df['MACD'].iloc[-1] > 0 else "Bearish"
        st.metric("MACD Signal", macd_signal)
    
    with col3:
        trend = "Uptrend" if df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1] else "Downtrend"
        st.metric("Trend", trend)
    
    # Volume analysis
    st.subheader("📊 Volume Analysis")
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'))
    fig_vol.update_layout(height=250)
    st.plotly_chart(fig_vol, use_container_width=True)
    
    # Company info
    with st.expander("ℹ️ Company Information"):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Company:** {info.get('longName', ticker)}")
            st.write(f"**Sector:** {info.get('sector', 'N/A')}")
            st.write(f"**Industry:** {info.get('industry', 'N/A')}")
        with col2:
            st.write(f"**Employees:** {info.get('fullTimeEmployees', 'N/A'):,}")
            st.write(f"**Website:** {info.get('website', 'N/A')}")
            st.write(f"**Exchange:** {info.get('exchange', 'N/A')}")

except Exception as e:
    st.error(f"Error loading data for {ticker}. Please check the ticker symbol.")
    st.write(f"Details: {str(e)}")

st.divider()
st.markdown("**Data Source:** Yahoo Finance | **ML Model:** Random Forest | **Update Frequency:** Real-time")
