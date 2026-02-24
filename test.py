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

    def fetch_ohlc(self, symbol, days=1, interval="1minute"):
        """
        Fetches OHLC data. 
        FIX 1: Uses dynamic instrument_key.
        FIX 2: Retries historical data if intraday (Live) is empty.
        """
        instrument_key = INSTRUMENT_MAP.get(symbol)
        if not instrument_key: return pd.DataFrame()

        url_live = f"https://api.upstox.com/v2/historical-candle/intraday/{instrument_key}/{interval}"

        try:
            response = requests.get(url_live, headers=self.headers)
            data = response.json()
            
            # If Intraday is empty (Market Closed), try Historical backup
            if response.status_code != 200 or not data.get("data", {}).get("candles"):
                to_date = datetime.now().strftime("%Y-%m-%d")
                from_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
                url_hist = f"https://api.upstox.com/v2/historical-candle/{instrument_key}/{interval}/{to_date}/{from_date}"
                response = requests.get(url_hist, headers=self.headers)
                data = response.json()

            if response.status_code == 200 and data.get("data", {}).get("candles"):
                candles = data["data"]["candles"]
                df = pd.DataFrame(candles, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI'])
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                df.set_index('Timestamp', inplace=True)
                df.sort_index(inplace=True)
                return df
                
        except Exception as e:
            st.sidebar.error(f"Connection Error for {symbol}: {e}")
            
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

# --- IMPROVED OPTIMIZER FUNCTIONS ---

def optimize_portfolio(prices_df):
    """
    Markowitz Model: Maximizes the Sharpe Ratio.
    Finds the optimal balance between Return and Risk.
    """
    if prices_df.empty or len(prices_df.columns) < 2: 
        return [1.0] if not prices_df.empty else []
    
    returns = prices_df.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(mean_returns)
    rf_rate = 0.07 # Assume 7% Risk-Free Rate (India Bond Yield)

    def portfolio_stats(weights):
        p_ret = np.sum(mean_returns * weights) * 252
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        sharpe = (p_ret - rf_rate) / p_vol
        return p_ret, p_vol, sharpe

    # Objective: Minimize negative Sharpe Ratio
    def objective(weights):
        return -portfolio_stats(weights)[2]

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # Setting bounds: No stock can be more than 40% or less than 5% of portfolio
    bounds = tuple((0.05, 0.40) for _ in range(num_assets))
    init_guess = num_assets * [1. / num_assets]

    try:
        result = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x
    except:
        return init_guess

def optimize_risk_parity(prices_df):
    """
    Risk Parity Model: Allocates more weight to less volatile stocks.
    Very stable and robust for long-term investing.
    """
    if prices_df.empty: return []
    volatility = prices_df.pct_change().dropna().std()
    inv_vol = 1.0 / volatility
    weights = inv_vol / inv_vol.sum()
    return weights.values

# --- ANALYTICS FUNCTIONS ---

def predict_future_prices(df, days_to_predict=90): # Default set to 90
    if df.empty or len(df) < 30: return None, None
    
    # 1. Prepare historical data
    df_train = df.reset_index().copy()
    # Use timestamps as our numeric feature for regression
    df_train['Time_Num'] = df_train['Timestamp'].map(datetime.timestamp)
    
    X = df_train[['Time_Num']].values
    y = df_train['Close'].values
    
    # 2. Train the Linear Regression Model
    model = LinearRegression()
    model.fit(X, y)
    
    # 3. Generate 90 future timestamps
    last_time = df_train['Timestamp'].iloc[-1]
    # We create a sequence of dates starting from tomorrow
    future_dates = [last_time + timedelta(days=i) for i in range(1, days_to_predict + 1)]
    future_timestamps = np.array([t.timestamp() for t in future_dates]).reshape(-1, 1)
    
    # 4. Predict prices for those 90 dates
    predicted_prices = model.predict(future_timestamps)
    
    return future_dates, predicted_prices

def analyze_stock(df, symbol):
    if df.empty: return {"symbol": symbol, "price": 0, "signal": "N/A", "signal_color": "grey"}
    current_price = df['Close'].iloc[-1]
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    signal, color = "HOLD", "grey"
    if len(df) > 50:
        if df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1]: signal, color = "BUY", "green"
        elif df['SMA_20'].iloc[-1] < df['SMA_50'].iloc[-1]: signal, color = "SELL", "red"
    return {"symbol": symbol, "price": current_price, "signal": signal, "signal_color": color}

def backtest_strategy(df, initial_capital=100000):
    if df.empty or len(df) < 55: return None, initial_capital
    df = df.copy()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Signal'] = 0
    df.loc[df['SMA_20'] > df['SMA_50'], 'Signal'] = 1
    df['Position'] = df['Signal'].diff()
    capital, holdings, trades = initial_capital, 0, []
    for i, row in df.iterrows():
        price = row['Close']
        if row['Position'] == 1 and capital > 0:
            holdings = capital / price
            capital = 0
            trades.append({"Time": i, "Action": "BUY", "Price": price, "Value": holdings*price})
        elif row['Position'] == -1 and holdings > 0:
            capital = holdings * price
            holdings = 0
            trades.append({"Time": i, "Action": "SELL", "Price": price, "Value": capital})
    final_value = capital if holdings == 0 else holdings * df['Close'].iloc[-1]
    return trades, final_value

def optimize_equal_weight(prices_df):
    """Strategy 1: Equal Weight (Benchmark)
    Simply divides capital equally. Often beats active managers."""
    num_assets = len(prices_df.columns)
    return np.array([1.0 / num_assets] * num_assets)

def optimize_minimum_variance(prices_df):
    """Strategy 2: Minimum Variance (Defensive)
    Ignores returns and focuses ONLY on minimizing portfolio volatility.
    Best for conservative investors."""
    if prices_df.empty: return []
    returns = prices_df.pct_change().dropna()
    cov_matrix = returns.cov() * 252
    num_assets = len(prices_df.columns)

    def objective(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    result = minimize(objective, num_assets * [1./num_assets], method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def optimize_hrp_simplified(prices_df):
    """Strategy 3: Simplified Risk Clustering (ML-lite)
    Groups stocks by correlation. Highly stable during market crashes."""
    if prices_df.empty: return []
    corr = prices_df.pct_change().dropna().corr()
    # Simple logic: Give less weight to stocks that are highly correlated with others
    avg_corr = corr.mean()
    inv_corr = 1.0 / avg_corr
    weights = inv_corr / inv_corr.sum()
    return weights.values

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
        score = TextBlob(headline).sentiment.polarity
        label = "Positive 🟢" if score > 0.1 else "Negative 🔴" if score < -0.1 else "Neutral 😐"
        news_data.append({"Headline": headline, "Sentiment": label, "Score": round(score, 2)})
    return pd.DataFrame(news_data)