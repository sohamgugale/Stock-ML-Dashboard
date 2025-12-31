import streamlit as st
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
    use_demo = st.checkbox("Use Demo Data (Yahoo Finance down)", value=True)
    ticker = st.text_input("Stock Ticker", "AAPL").upper()
    period = st.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y"], index=3)
    
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

def generate_demo_data(ticker, days=252):
    """Generate realistic demo stock data"""
    np.random.seed(hash(ticker) % 10000)
    
    # Base prices for different stocks
    base_prices = {
        'AAPL': 180, 'MSFT': 380, 'GOOGL': 140, 
        'TSLA': 250, 'NVDA': 500, 'AMZN': 170
    }
    base_price = base_prices.get(ticker, 100)
    
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Generate realistic price movement
    returns = np.random.normal(0.001, 0.02, days)
    price = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC
    df = pd.DataFrame({
        'Open': price * (1 + np.random.uniform(-0.01, 0.01, days)),
        'High': price * (1 + np.random.uniform(0, 0.02, days)),
        'Low': price * (1 - np.random.uniform(0, 0.02, days)),
        'Close': price,
        'Volume': np.random.randint(50e6, 150e6, days)
    }, index=dates)
    
    # Add technical indicators
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    
    return df

def get_live_data(ticker, period):
    """Try to get live data from Yahoo Finance"""
    try:
        import yfinance as yf
        import time
        
        time.sleep(1)  # Rate limit
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if df.empty:
            return None
        
        # Add technical indicators
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        df['MACD'] = ta.trend.MACD(df['Close']).macd()
        
        return df
    except:
        return None

# Load data
if use_demo:
    st.info(f"📊 Showing demo data for {ticker} (Yahoo Finance currently unavailable)")
    df = generate_demo_data(ticker)
else:
    with st.spinner(f"Loading {ticker} data from Yahoo Finance..."):
        df = get_live_data(ticker, period)
        if df is None:
            st.warning(f"⚠️ Could not load live data. Switching to demo mode...")
            df = generate_demo_data(ticker)

# Display key metrics
col1, col2, col3, col4 = st.columns(4)

current_price = df['Close'].iloc[-1]
prev_close = df['Close'].iloc[-2]
change = current_price - prev_close
change_pct = (change / prev_close) * 100

col1.metric("Current Price", f"${current_price:.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
col2.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")
col3.metric("52W High", f"${df['High'].max():.2f}")
col4.metric("Days of Data", len(df))

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
    
    # Model accuracy on test set
    test_predictions = model.predict(X_scaled[-5:])
    test_actual = y[-5:]
    accuracy = 100 * (1 - np.mean(np.abs(test_predictions - test_actual) / test_actual))
    
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
            'Date': [d.strftime('%Y-%m-%d') for d in dates],
            'Predicted Price': [f"${p:.2f}" for p in predictions],
            'Change': [f"{((p/current_price - 1)*100):+.2f}%" for p in predictions]
        })
        st.dataframe(pred_df, use_container_width=True, hide_index=True)
        st.caption(f"Model Accuracy: {accuracy:.1f}%")
    
    with col2:
        # Prediction chart
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=df.index[-30:],
            y=df['Close'].iloc[-30:],
            name='Historical',
            line=dict(color='blue', width=2)
        ))
        fig_pred.add_trace(go.Scatter(
            x=dates,
            y=predictions,
            name='Predicted',
            line=dict(color='red', dash='dash', width=2),
            mode='lines+markers',
            marker=dict(size=8)
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
    
    # Key Statistics
    with st.expander("📈 Key Statistics"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Volume", f"{df['Volume'].mean():,.0f}")
            st.metric("Volatility (30d)", f"{df['Close'].pct_change().tail(30).std()*100:.2f}%")
        with col2:
            st.metric("52W Low", f"${df['Low'].min():.2f}")
            st.metric("30d High", f"${df['High'].tail(30).max():.2f}")
        with col3:
            st.metric("30d Low", f"${df['Low'].tail(30).min():.2f}")
            st.metric("Avg Daily Return", f"{df['Close'].pct_change().mean()*100:.2f}%")

except Exception as e:
    st.error(f"Error in ML prediction: {str(e)}")

st.divider()
data_source = "Demo Data" if use_demo else "Yahoo Finance (Live)"
st.markdown(f"**Data Source:** {data_source} | **ML Model:** Random Forest Regressor | **Features:** 8 technical indicators")
