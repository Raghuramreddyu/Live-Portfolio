import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.optimize import minimize
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from textblob import TextBlob
import random
from config import INSTRUMENT_MAP

# --- AUTHENTICATION ---
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    
    if not st.session_state["password_correct"]:
        pwd = st.text_input("Password (use: admin123)", type="password")
        if pwd == "admin123":
            st.session_state["password_correct"] = True
            st.rerun()
        return False
    return True

# --- DATA PROVIDER ---
class DataProvider:
    def __init__(self, api_config):
        self.access_token = api_config["access_token"]
        self.headers = {'Accept': 'application/json', 'Authorization': f'Bearer {self.access_token}'}

    def fetch_ohlc(self, symbol, days=30, interval="1minute"):
        instrument_key = INSTRUMENT_MAP.get(symbol)
        if not instrument_key: return pd.DataFrame()

        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        url = f"https://api.upstox.com/v2/historical-candle/{instrument_key}/{interval}/{to_date}/{from_date}"

        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                data = response.json()
                if "data" in data and "candles" in data["data"] and data["data"]["candles"]:
                    candles = data["data"]["candles"]
                    df = pd.DataFrame(candles, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI'])
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                    df.set_index('Timestamp', inplace=True)
                    df.sort_index(inplace=True)
                    return df
        except Exception:
            pass
        return self._generate_fallback_data(interval)

    def _generate_fallback_data(self, interval):
        freq = 'D' if interval == 'day' else '1min'
        periods = 100
        dates = pd.date_range(end=datetime.now(), periods=periods, freq=freq)
        df = pd.DataFrame(index=dates)
        base = 2000
        df['Close'] = base + np.cumsum(np.random.randn(periods) * 10)
        df['Open'] = df['Close'] + np.random.uniform(-5, 5, periods)
        df['High'] = df['Close'] + 10
        df['Low'] = df['Close'] - 10
        df['Volume'] = np.random.randint(10000, 50000, periods)
        return df

# --- ANALYTICS FUNCTIONS ---
def predict_future_prices(df, days_to_predict=90):
    if df.empty: return None, None
    df_train = df.reset_index().copy()
    df_train['Time_Num'] = df_train['Timestamp'].map(datetime.timestamp)
    X = df_train[['Time_Num']].values
    y = df_train['Close'].values
    model = LinearRegression()
    model.fit(X, y)
    last_time = df_train['Timestamp'].iloc[-1]
    future_dates = [last_time + timedelta(days=i) for i in range(1, days_to_predict + 1)]
    future_timestamps = np.array([t.timestamp() for t in future_dates]).reshape(-1, 1)
    predicted_prices = model.predict(future_timestamps)
    return future_dates, predicted_prices

def optimize_portfolio(prices_df):
    if prices_df.empty: return []
    returns = prices_df.pct_change().dropna()
    if returns.empty: return [1/len(prices_df.columns)] * len(prices_df.columns)
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(mean_returns)
    def negative_sharpe(weights):
        p_ret = np.sum(returns.mean() * weights) * 252
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        return -p_ret / p_vol if p_vol > 0 else 0
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    init_guess = num_assets * [1. / num_assets,]
    try:
        result = minimize(negative_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x
    except:
        return init_guess

def analyze_stock(df, symbol):
    if df.empty: return {"symbol": symbol, "price": 0, "signal": "N/A", "signal_color": "grey"}
    current_price = df['Close'].iloc[-1]
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    signal, color = "HOLD", "grey"
    if len(df) > 50:
        if df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1] and df['SMA_20'].iloc[-2] <= df['SMA_50'].iloc[-2]:
            signal, color = "BUY", "green"
        elif df['SMA_20'].iloc[-1] < df['SMA_50'].iloc[-1] and df['SMA_20'].iloc[-2] >= df['SMA_50'].iloc[-2]:
            signal, color = "SELL", "red"
    return {"symbol": symbol, "price": current_price, "signal": signal, "signal_color": color}

def backtest_strategy(df, initial_capital=100000):
    if df.empty or len(df) < 55: return None, 0
    df = df.copy()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Signal'] = 0
    df.loc[df['SMA_20'] > df['SMA_50'], 'Signal'] = 1
    df['Position'] = df['Signal'].diff()
    capital = initial_capital
    holdings = 0
    trades = []
    for i, row in df.iterrows():
        price = row['Close']
        if row['Position'] == 1:
            if capital > 0:
                holdings = capital / price
                capital = 0
                trades.append({"Time": i, "Action": "BUY", "Price": price, "Value": holdings*price})
        elif row['Position'] == -1:
            if holdings > 0:
                capital = holdings * price
                holdings = 0
                trades.append({"Time": i, "Action": "SELL", "Price": price, "Value": capital})
    final_value = capital if holdings == 0 else holdings * df['Close'].iloc[-1]
    return trades, final_value

def analyze_news_sentiment(symbol):
    base_headlines = [
        f"{symbol} announces strong quarterly results.",
        f"Market analysts upgrade rating for {symbol}.",
        f"Global headwinds might affect {symbol}'s export revenue.",
        f"{symbol} launches new AI-driven product line.",
        f"Regulatory concerns loom over {symbol}'s sector."
    ]
    news_data = []
    for headline in random.sample(base_headlines, 3):
        analysis = TextBlob(headline)
        sentiment_score = analysis.sentiment.polarity
        sentiment_label = "Neutral 😐"
        if sentiment_score > 0.1: sentiment_label = "Positive 🟢"
        if sentiment_score < -0.1: sentiment_label = "Negative 🔴"
        news_data.append({"Headline": headline, "Sentiment": sentiment_label, "Score": round(sentiment_score, 2)})
    return pd.DataFrame(news_data)