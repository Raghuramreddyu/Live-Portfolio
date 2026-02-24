import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime

# --- 1. CONFIGURATION & IMPORTS ---
st.set_page_config(page_title="AI Portfolio Agent Pro", layout="wide")

# Import custom modules
from config import UPSTOX_CONFIG, INSTRUMENT_MAP
from utils import (
    check_password, DataProvider, predict_future_prices, 
    optimize_portfolio, optimize_risk_parity, optimize_minimum_variance,
    optimize_equal_weight, optimize_hrp_simplified,
    analyze_stock, backtest_strategy, analyze_news_sentiment
)

# Import Optional Advanced Modules
try:
    from generative import StressTestLab, AICoPilot, AlgoBattle
    GENERATIVE_AVAILABLE = True
except ImportError:
    GENERATIVE_AVAILABLE = False

try:
    from fundamentals import FundamentalAgent
    FUNDAMENTALS_AVAILABLE = True
except ImportError:
    FUNDAMENTALS_AVAILABLE = False

# --- 2. AUTHENTICATION & SETUP ---
if not check_password(): st.stop()

# --- 3. UI SIDEBAR ---
st.title("🤖 AI Portfolio Agent Pro")
st.markdown("---")

st.sidebar.header("1. Portfolio Setup")
available_stocks = list(INSTRUMENT_MAP.keys())
selected_tickers = st.sidebar.multiselect("Select Stocks", available_stocks, default=["RELIANCE", "TCS"])

portfolio_data = []
if selected_tickers:
    st.sidebar.subheader("2. Purchase Details")
    for ticker in selected_tickers:
        c1, c2 = st.sidebar.columns(2)
        qty = c1.number_input(f"{ticker} Qty", 1, 1000, 10, key=f"q_{ticker}")
        price = c2.number_input(f"{ticker} Avg Price", 1.0, 10000.0, 2000.0, key=f"p_{ticker}")
        portfolio_data.append({"Symbol": ticker, "Qty": qty, "BuyPrice": price})

if st.sidebar.button("🚀 Run AI Analysis"):
    st.session_state["analyzed"] = True

# --- 4. MAIN APPLICATION LOGIC ---
if st.session_state.get("analyzed", False) and selected_tickers:
    data_provider = DataProvider(UPSTOX_CONFIG)
    price_history_df = pd.DataFrame()
    
    with st.spinner('Fetching Data & Running Models...'):
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📊 Portfolio", "🧠 Optimizer", "🔮 3-Month AI", 
            "⏮️ Backtest", "📰 Sentiment", "📈 Pro Charts", "🧪 Labs", "🏥 Fundamentals"
        ])

        # --- TAB 1: PORTFOLIO SUMMARY ---
        with tab1:
            st.subheader("Live Performance")
            summary_rows = []
            total_inv, total_val = 0, 0
            
            for item in portfolio_data:
                symbol = item['Symbol']
                df = data_provider.fetch_ohlc(symbol, days=1, interval="1minute") 
                
                analysis = analyze_stock(df, symbol)
                curr_price = analysis['price']
                val = curr_price * item['Qty']
                inv = item['BuyPrice'] * item['Qty']
                total_inv += inv; total_val += val
                
                summary_rows.append({
                    "Stock": symbol, "Qty": item['Qty'], "LTP": round(curr_price, 2),
                    "Value": round(val, 2), "P/L": round(val - inv, 2),
                    "Signal": analysis['signal']
                })
                if not df.empty: price_history_df[symbol] = df['Close']

            st.metric("Total Portfolio Value", f"₹{total_val:,.2f}", f"₹{total_val - total_inv:,.2f} Total P/L")
            st.dataframe(pd.DataFrame(summary_rows).style.applymap(
                lambda x: 'color: red' if isinstance(x, (int, float)) and x < 0 else 'color: green' if isinstance(x, (int, float)) and x > 0 else '',
                subset=['P/L']
            ))

        # --- TAB 2: MULTI-STRATEGY OPTIMIZER ---
        with tab2:
            st.subheader("🧠 Multi-Strategy Portfolio Optimizer")
            opt_method = st.selectbox("Select Optimization Strategy", [
                "Markowitz (Max Sharpe Ratio)", 
                "Risk Parity (Inverse Volatility)",
                "Minimum Variance (Lowest Risk)",
                "Equal Weight (Standard)",
                "Cluster Risk (Correlation-Based)"
            ])
            
            if not price_history_df.empty:
                clean_prices = price_history_df.ffill().dropna()
                
                # Dynamic Logic Selection
                if opt_method == "Markowitz (Max Sharpe Ratio)":
                    opt_weights = optimize_portfolio(clean_prices)
                elif opt_method == "Risk Parity (Inverse Volatility)":
                    opt_weights = optimize_risk_parity(clean_prices)
                elif opt_method == "Minimum Variance (Lowest Risk)":
                    opt_weights = optimize_minimum_variance(clean_prices)
                elif opt_method == "Equal Weight (Standard)":
                    opt_weights = optimize_equal_weight(clean_prices)
                else:
                    opt_weights = optimize_hrp_simplified(clean_prices)

                col_opt1, col_opt2 = st.columns(2)
                with col_opt1:
                    fig_pie = px.pie(names=clean_prices.columns, values=opt_weights, hole=0.4, 
                                     title=f"Target Allocation: {opt_method}")
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                with col_opt2:
                    # Calculate Current Allocation Percentages
                    current_weights = []
                    total_v = sum([r['Value'] for r in summary_rows])
                    for ticker in clean_prices.columns:
                        matched_val = next((r['Value'] for r in summary_rows if r['Stock'] == ticker), 0)
                        current_weights.append(matched_val / total_v if total_v > 0 else 0)

                    df_compare = pd.DataFrame({
                        "Stock": clean_prices.columns,
                        "Current %": [round(w*100, 1) for w in current_weights],
                        "Optimal %": [round(w*100, 1) for w in opt_weights]
                    })
                    fig_bar = px.bar(df_compare.melt(id_vars='Stock'), x='Stock', y='value', 
                                     color='variable', barmode='group', title="Allocation Shift (Current vs Optimal)")
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                descriptions = {
                    "Markowitz (Max Sharpe Ratio)": "Best for high growth. Balances risk and return using the Sharpe Ratio.",
                    "Risk Parity (Inverse Volatility)": "Ensures every stock contributes equal risk to the portfolio.",
                    "Minimum Variance (Lowest Risk)": "Mathematically finds the combination with the absolute lowest price swings.",
                    "Equal Weight (Standard)": "Classic 1/N allocation. Simple and often the most reliable benchmark.",
                    "Cluster Risk (Correlation-Based)": "Uses correlation data to ensure you aren't over-exposed to one sector (e.g. IT)."
                }
                st.info(f"**Strategy Insight:** {descriptions[opt_method]}")

        # --- TAB 3: AI PREDICTOR ---
        with tab3:
            st.subheader("🔮 90-Day Trend Forecast")
            target_ml = st.selectbox("Predict Stock", selected_tickers, key="ml_sel")
            if target_ml:
                df_daily = data_provider.fetch_ohlc(target_ml, days=365, interval="day")
                if not df_daily.empty:
                    dates, preds = predict_future_prices(df_daily)
                    fig_ml = go.Figure()
                    fig_ml.add_trace(go.Scatter(x=df_daily.index, y=df_daily['Close'], name='Historical'))
                    fig_ml.add_trace(go.Scatter(x=dates, y=preds, name='AI Forecast', line=dict(dash='dash', color='orange')))
                    st.plotly_chart(fig_ml, use_container_width=True)

        # --- TAB 6: 3-MONTH PRO CHARTS ---
        with tab6:
            st.subheader("📈 Professional 90-Day High-Res Chart")
            target_chart = st.selectbox("Select Chart", selected_tickers, key="chart_sel")
            if target_chart:
                with st.spinner(f"Aggregating 3 months of 1-minute data..."):
                    df_c = data_provider.fetch_ohlc(target_chart, days=90, interval="1minute")
                
                if not df_c.empty:
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
                    fig.add_trace(go.Candlestick(x=df_c.index, open=df_c['Open'], high=df_c['High'], low=df_c['Low'], close=df_c['Close'], name="Price"), row=1, col=1)
                    
                    df_c['SMA_20'] = df_c['Close'].rolling(20).mean()
                    fig.add_trace(go.Scatter(x=df_c.index, y=df_c['SMA_20'], name="SMA 20", line=dict(color='yellow')), row=1, col=1)
                    
                    fig.add_trace(go.Bar(x=df_c.index, y=df_c['Volume'], name="Volume", marker_color='teal'), row=2, col=1)
                    fig.update_layout(height=700, xaxis_rangeslider_visible=False, template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)

        # --- TABS 4, 5, 7, 8: REMAINING UTILITIES ---
        with tab4:
            st.subheader("⏮️ Backtest Results")
            target_bt = st.selectbox("Select for Backtest", selected_tickers, key="bt_sel")
            if target_bt:
                df_bt = data_provider.fetch_ohlc(target_bt, days=200, interval="day")
                trades, f_val = backtest_strategy(df_bt)
                st.metric("Backtest Result (100k start)", f"₹{f_val:,.2f}", f"{((f_val-100000)/100000)*100:.2f}%")
                if trades: st.dataframe(pd.DataFrame(trades))

        with tab5:
            st.subheader("📰 Market Sentiment Audit")
            target_news = st.selectbox("News Symbol", selected_tickers, key="news_sel")
            if target_news:
                st.write(analyze_news_sentiment(target_news))

        with tab7:
            st.subheader("🧪 AI Labs")
            if GENERATIVE_AVAILABLE:
                st.success("Gen-AI Features Loaded.")
            else: st.warning("generative.py module missing.")

        with tab8:
            st.subheader("🏥 Fundamental Health Check")
            if FUNDAMENTALS_AVAILABLE:
                target_f = st.selectbox("Audit Stock", selected_tickers, key="fund_sel")
                if st.button("Generate Health Card"):
                    st.json(FundamentalAgent.get_health_card(target_f))
            else: st.warning("fundamentals.py module missing.")

else:
    st.info("👈 Set up your portfolio in the sidebar and click 'Run AI Analysis' to begin.")