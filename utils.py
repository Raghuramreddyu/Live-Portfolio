import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.optimize import minimize
from datetime import datetime, timedelta

# --- ADVANCED AI IMPORTS ---
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
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
        instrument_key = INSTRUMENT_MAP.get(symbol)
        if not instrument_key: return pd.DataFrame()

        # LOGIC 1: LIVE INTRADAY
        if interval == "1minute":
            url = f"https://api.upstox.com/v2/historical-candle/intraday/{instrument_key}/{interval}"
            try:
                response = requests.get(url, headers=self.headers)
                data = response.json()
                
                if response.status_code != 200 or not data.get("data", {}).get("candles"):
                    to_date = datetime.now().strftime("%Y-%m-%d")
                    from_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
                    url = f"https://api.upstox.com/v2/historical-candle/{instrument_key}/{interval}/{to_date}/{from_date}"
                    response = requests.get(url, headers=self.headers)
            except:
                pass 

        # LOGIC 2: LONG TERM HISTORY
        else:
            to_date = datetime.now().strftime("%Y-%m-%d")
            from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            url = f"https://api.upstox.com/v2/historical-candle/{instrument_key}/{interval}/{to_date}/{from_date}"
            response = requests.get(url, headers=self.headers)

        # DATA PROCESSING
        try:
            if response.status_code == 200:
                data = response.json()
                if data.get("data", {}).get("candles"):
                    candles = data["data"]["candles"]
                    df = pd.DataFrame(candles, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI'])
                    
                    # FIX 1: Strip timezones to prevent Streamlit Table crashes
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                    if df['Timestamp'].dt.tz is not None:
                        df['Timestamp'] = df['Timestamp'].dt.tz_localize(None)
                        
                    df.set_index('Timestamp', inplace=True)
                    df.sort_index(inplace=True)
                    return df
        except:
            pass
            
        return self._generate_fallback_data(interval)

    def _generate_fallback_data(self, interval):
        freq = 'D' if interval == 'day' else '1min'
        periods = 100
        dates = pd.date_range(end=datetime.now(), periods=periods, freq=freq)
        df = pd.DataFrame(index=dates)
        
        # FIX 2: Explicitly name the index so ML model doesn't throw a KeyError
        df.index.name = 'Timestamp'
        
        base = 2000
        df['Close'] = base + np.cumsum(np.random.randn(periods) * 10)
        df['Open'] = df['Close'] + np.random.uniform(-5, 5, periods)
        df['High'] = df['Close'] + 10
        df['Low'] = df['Close'] - 10
        df['Volume'] = np.random.randint(10000, 50000, periods)
        return df

# --- ADVANCED AI PREDICTOR ---
def predict_future_prices(df, days_to_predict=90, model_type="Linear Regression"):
    if df.empty: return None, None
    df_t = df.reset_index()
    
    if 'Timestamp' not in df_t.columns and 'index' in df_t.columns:
        df_t.rename(columns={'index': 'Timestamp'}, inplace=True)
        
    # FIX 3: Safe conversion of dates to numerical values
    X = df_t['Timestamp'].apply(lambda x: x.timestamp()).values.reshape(-1, 1)
    y = df_t['Close'].values
    
    if "Polynomial" in model_type:
        model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    elif "Random Forest" in model_type:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = LinearRegression()
    
    model.fit(X, y)
    
    last_dt = df_t['Timestamp'].iloc[-1]
    fut_dates = [last_dt + timedelta(days=i) for i in range(1, days_to_predict + 1)]
    fut_X = np.array([d.timestamp() for d in fut_dates]).reshape(-1, 1)
    
    return fut_dates, model.predict(fut_X)

# --- OPTIMIZATION STRATEGIES ---
def optimize_portfolio(prices_df):
    returns = prices_df.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(mean_returns)
    def neg_sharpe(weights):
        p_ret = np.sum(mean_returns * weights) * 252
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        if p_vol == 0: return 0 
        return -(p_ret - 0.07) / p_vol
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.05, 0.4) for _ in range(num_assets))
    try:
        res = minimize(neg_sharpe, num_assets*[1./num_assets], method='SLSQP', bounds=bounds, constraints=constraints)
        return res.x
    except:
        return np.array([1./num_assets]*num_assets)

def optimize_risk_parity(prices_df):
    # FIX 4: Prevent Division by Zero NaN crash in pie chart
    vol = prices_df.pct_change().dropna().std().replace(0, np.nan)
    inv_vol = 1.0 / vol
    return (inv_vol / inv_vol.sum()).fillna(1.0/len(vol)).values

def optimize_minimum_variance(prices_df):
    returns = prices_df.pct_change().dropna()
    cov = returns.cov() * 252
    num_assets = len(prices_df.columns)
    def obj(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    try:
        res = minimize(obj, num_assets*[1./num_assets], method='SLSQP', bounds=bounds, constraints=constraints)
        return res.x
    except:
        return np.array([1./num_assets]*num_assets)

def optimize_equal_weight(prices_df):
    n = len(prices_df.columns)
    return np.array([1.0/n] * n)

def optimize_hrp_simplified(prices_df):
    # FIX 5: Prevent NaN crash
    corr = prices_df.pct_change().dropna().corr().mean().replace(0, np.nan)
    inv_corr = 1.0 / corr
    return (inv_corr / inv_corr.sum()).fillna(1.0/len(corr)).values

# --- ANALYTICS ---
def analyze_stock(df, symbol):
    if df.empty: return {"symbol": symbol, "price": 0, "signal": "N/A"}
    curr = df['Close'].iloc[-1]
    df['S20'] = df['Close'].rolling(20).mean()
    df['S50'] = df['Close'].rolling(50).mean()
    sig = "BUY" if df['S20'].iloc[-1] > df['S50'].iloc[-1] else "SELL"
    return {"symbol": symbol, "price": curr, "signal": sig}

def backtest_strategy(df, initial_capital=100000):
    if len(df) < 55: return None, initial_capital
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

def analyze_news_sentiment(symbol):
    headlines = [f"{symbol} growth peaks", f"{symbol} faces hurdle", f"{symbol} dividend out"]
    data = []
    for h in headlines:
        score = TextBlob(h).sentiment.polarity
        sent = "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"
        data.append({"Headline": h, "Sentiment": sent, "Score": score})
    return pd.DataFrame(data)