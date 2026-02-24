import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

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

# Optional Advanced Modules (Error Handling included)
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
if not UPSTOX_CONFIG["api_key"] or not UPSTOX_CONFIG["access_token"]:
    st.error("🚨 Critical Error: API Credentials not found in config.")
    st.stop()

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
        qty = c1.number_input(f"{ticker} Qty", min_value=1, max_value=10000, value=10, key=f"q_{ticker}")
        price = c2.number_input(f"{ticker} Avg Price", min_value=1.0, max_value=10000000.0, value=2000.0, key=f"p_{ticker}")
        portfolio_data.append({"Symbol": ticker, "Qty": qty, "BuyPrice": price})

if st.sidebar.button("🚀 Run AI Analysis"):
    st.session_state["analyzed"] = True

# --- 4. MAIN APPLICATION LOGIC ---
if st.session_state.get("analyzed", False) and selected_tickers:
    data_provider = DataProvider(UPSTOX_CONFIG)
    price_history_df = pd.DataFrame()
    
    # FIX: TABS ARE NOW OUTSIDE THE SPINNER
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Portfolio", "🧠 Optimizer", "🔮 3-Month AI", 
        "⏮️ Backtest", "📰 Sentiment", "📈 Pro Charts", "🧪 Labs", "🏥 Fundamentals"
    ])

    # --- TAB 1: PORTFOLIO SUMMARY ---
    with tab1:
        st.subheader("Live Performance")
        with st.spinner("Fetching Live Market Data..."):
            summary_rows = []
            total_inv, total_val = 0, 0
            
            for item in portfolio_data:
                symbol = item['Symbol']
                df = data_provider.fetch_ohlc(symbol, days=30, interval="1minute") 
                
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

        st.metric("Total P/L", f"₹{total_val - total_inv:,.2f}", f"{((total_val-total_inv)/total_inv)*100:.2f}%")
        st.dataframe(pd.DataFrame(summary_rows).style.map(
            lambda x: 'color: red' if isinstance(x, (int, float)) and x < 0 else 'color: green' if isinstance(x, (int, float)) and x > 0 else '',
            subset=['P/L']
        ))

    # --- TAB 2: MULTI-STRATEGY OPTIMIZER ---
    with tab2:
        st.subheader("🧠 Multi-Strategy Portfolio Optimizer")
        opt_method = st.selectbox("Select Strategy", [
            "Markowitz (Max Sharpe Ratio)", "Risk Parity (Inverse Volatility)",
            "Minimum Variance", "Equal Weight", "Cluster Risk"
        ])
        
        if not price_history_df.empty:
            clean_prices = price_history_df.ffill().dropna()
            if not clean_prices.empty:
                if opt_method == "Markowitz (Max Sharpe Ratio)":
                    opt_weights = optimize_portfolio(clean_prices)
                elif opt_method == "Risk Parity (Inverse Volatility)":
                    opt_weights = optimize_risk_parity(clean_prices)
                elif opt_method == "Minimum Variance":
                    opt_weights = optimize_minimum_variance(clean_prices)
                elif opt_method == "Equal Weight":
                    opt_weights = optimize_equal_weight(clean_prices)
                else:
                    opt_weights = optimize_hrp_simplified(clean_prices)

                col_opt1, col_opt2 = st.columns(2)
                with col_opt1:
                    st.plotly_chart(px.pie(names=clean_prices.columns, values=opt_weights, hole=0.4), use_container_width=True)
                with col_opt2:
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
                    st.plotly_chart(px.bar(df_compare.melt(id_vars='Stock'), x='Stock', y='value', color='variable', barmode='group'), use_container_width=True)
            else: st.warning("Not enough data history to optimize.")

    # --- TAB 3: AI-DRIVEN 3-MONTH FORECAST ---
    with tab3:
        st.subheader("🔮 90-Day Trend Forecast")
        c1, c2 = st.columns([1, 2])
        with c1:
            target_ml = st.selectbox("Predict Stock", selected_tickers, key="tab3_sel")
        with c2:
            model_choice = st.selectbox("AI Strategy Model", [
                "Linear Regression (Straight Line)", 
                "Polynomial Regression (Curved)", 
                "Random Forest (Complex Pattern)"
            ])

        if target_ml:
            with st.spinner("Training AI Model..."):
                df_daily = data_provider.fetch_ohlc(target_ml, days=365, interval="day")
                if not df_daily.empty:
                    future_dates, preds = predict_future_prices(df_daily, days_to_predict=90, model_type=model_choice)
                    
                    current_price = df_daily['Close'].iloc[-1]
                    future_price = preds[-1]
                    pct_change = ((future_price - current_price) / current_price) * 100
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Current Price", f"₹{current_price:,.2f}")
                    m2.metric("Predicted Price", f"₹{future_price:,.2f}")
                    m3.metric("Expected P/L", f"{pct_change:+.2f}%", delta_color="normal")

                    fig_ml = go.Figure()
                    fig_ml.add_trace(go.Scatter(x=df_daily.index, y=df_daily['Close'], name='History (365d)'))
                    fig_ml.add_trace(go.Scatter(x=future_dates, y=preds, name=f'Forecast ({model_choice})', line=dict(dash='dash', color='orange', width=2)))
                    fig_ml.update_layout(template="plotly_dark", height=500, title=f"Projection using {model_choice}")
                    st.plotly_chart(fig_ml, use_container_width=True)
                    
                    verdict = "PROFIT 🚀" if pct_change > 0 else "LOSS 🔻"
                    st.success(f"**AI Verdict:** The {model_choice} model predicts a {verdict} of **{pct_change:.2f}%**.")

    # --- TAB 4: BACKTESTING ---
    with tab4:
        st.subheader("⏮️ Strategy Backtesting Engine")
        target_bt = st.selectbox("Select Stock to Backtest", selected_tickers, key="bt_select")
        if target_bt:
            df_bt = data_provider.fetch_ohlc(target_bt, days=200, interval="day")
            trades, final_val = backtest_strategy(df_bt)
            if final_val:
                c1, c2 = st.columns(2)
                c1.metric("Initial Capital", "₹100,000")
                ret = ((final_val - 100000)/100000)*100
                c2.metric("Final Value", f"₹{final_val:,.2f}", f"{ret:.2f}%")
                if trades: st.dataframe(pd.DataFrame(trades), use_container_width=True)

    # --- TAB 5: SENTIMENT ---
    with tab5:
        st.subheader("📰 AI News Sentiment Analysis")
        target_news = st.selectbox("Select Stock for News", selected_tickers, key="news_select")
        if target_news:
            news_df = analyze_news_sentiment(target_news)
            if not news_df.empty:
                for idx, row in news_df.iterrows():
                    st.markdown(f"""
                    <div style='padding:10px; border-left: 5px solid {"green" if row["Score"] > 0 else "red"}; background-color: #1e1e1e; margin-bottom:10px;'>
                        <strong>{row['Headline']}</strong><br>Mood: {row['Sentiment']}
                    </div>""", unsafe_allow_html=True)

    # --- TAB 6: PRO CHARTS ---
    with tab6:
        st.subheader("📈 Professional 90-Day High-Res Chart")
        target_chart = st.selectbox("Select Chart", selected_tickers, key="chart_sel")
        if target_chart:
            df_c = data_provider.fetch_ohlc(target_chart, days=90, interval="day")
            
            if not df_c.empty:
                df_c['SMA_20'] = df_c['Close'].rolling(window=20).mean()
                df_c['SMA_50'] = df_c['Close'].rolling(window=50).mean()

                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
                fig.add_trace(go.Candlestick(x=df_c.index, open=df_c['Open'], high=df_c['High'], low=df_c['Low'], close=df_c['Close'], name="Price"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_c.index, y=df_c['SMA_20'], name="SMA 20", line=dict(color='orange', width=1.5)), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_c.index, y=df_c['SMA_50'], name="SMA 50", line=dict(color='cyan', width=1.5)), row=1, col=1)
                fig.add_trace(go.Bar(x=df_c.index, y=df_c['Volume'], name="Volume", marker_color='teal'), row=2, col=1)
                
                fig.update_layout(height=700, xaxis_rangeslider_visible=False, template="plotly_dark", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

    # --- TAB 7: LABS ---
    with tab7:
        st.subheader("🧪 Advanced AI Labs")
        if GENERATIVE_AVAILABLE:
            col_lab1, col_lab2 = st.columns(2)
            with col_lab1:
                st.markdown("### 💥 Stress Test Simulator")
                scenario = st.selectbox("Crash Scenario", list(StressTestLab.SCENARIOS.keys()))
                if st.button("RUN STRESS TEST"):
                    result = StressTestLab.run_simulation(total_val, scenario)
                    st.error(f"Projected Loss: -₹{result['loss_amount']:,.2f}")
                    st.metric("Simulated Value", f"₹{result['simulated_value']:,.2f}", delta=f"-{result['drop_percentage']}%", delta_color="inverse")
            with col_lab2:
                st.markdown("### 🤖 Hedge Fund Co-Pilot")
                user_q = st.text_input("Ask AI about portfolio:")
                if st.button("Get Insight"):
                    with st.spinner("AI is thinking..."):
                        agent = AICoPilot()
                        st.info(agent.generate_insight(portfolio_data, user_q))
            
            st.markdown("---")
            if st.button("START ALGO BATTLE"):
                battle_stock = selected_tickers[0] if selected_tickers else "RELIANCE"
                df_battle = data_provider.fetch_ohlc(battle_stock, days=365, interval="day")
                fig_battle, winner, v1, v2 = AlgoBattle.run_battle(df_battle)
                st.plotly_chart(fig_battle, use_container_width=True)
        else: st.warning("generative.py module missing.")

    # --- TAB 8: FUNDAMENTALS ---
    with tab8:
        st.subheader("🏥 Fundamental Health Check")
        if FUNDAMENTALS_AVAILABLE:
            target_f = st.selectbox("Audit Stock", selected_tickers, key="fund_sel")
            if st.button("Generate Health Card"):
                with st.spinner("Auditing financials..."):
                    health = FundamentalAgent.get_health_card(target_f)
                    if health["success"]:
                        st.metric("AI Verdict", health["verdict"])
                        st.write(health["summary"])
                        st.json(health)
        else: st.warning("fundamentals.py module missing.")

else:
    st.info("👈 Set up your portfolio and click 'Run AI Analysis' to begin.")