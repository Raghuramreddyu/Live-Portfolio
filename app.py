import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# --- 1. CONFIGURATION & IMPORTS ---
st.set_page_config(page_title="AI Portfolio Agent Pro", layout="wide")

# Import custom modules
from config import UPSTOX_CONFIG, INSTRUMENT_MAP
from utils import (
    check_password, DataProvider, predict_future_prices, 
    optimize_portfolio, analyze_stock, backtest_strategy, analyze_news_sentiment
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
if not UPSTOX_CONFIG["api_key"] or not UPSTOX_CONFIG["access_token"]:
    st.error("🚨 Critical Error: API Credentials not found in .env file.")
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
            "⏮️ Backtest", "📰 Sentiment", "📈 Pro Charts", "⚔️ Labs", "🏥 Fundamentals"
        ])

        # --- 1. PORTFOLIO SUMMARY ---
        with tab1:
            st.subheader("Live Performance")
            summary_rows = []
            total_inv, total_val = 0, 0
            
            for item in portfolio_data:
                symbol = item['Symbol']
                # Live status needs 1min data
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
            st.dataframe(pd.DataFrame(summary_rows).style.applymap(
                lambda x: 'color: red' if isinstance(x, (int, float)) and x < 0 else 'color: green' if isinstance(x, (int, float)) and x > 0 else '',
                subset=['P/L']
            ))

        # --- 2. OPTIMIZER ---
        with tab2:
            st.subheader("🧠 Portfolio Optimization (Markowitz Model)")
            if not price_history_df.empty:
                clean_prices = price_history_df.ffill().dropna()
                if not clean_prices.empty:
                    opt_weights = optimize_portfolio(clean_prices)
                    col_opt1, col_opt2 = st.columns(2)
                    with col_opt1:
                        fig_pie = px.pie(names=clean_prices.columns, values=opt_weights, hole=0.4, title="Target Allocation")
                        st.plotly_chart(fig_pie, use_container_width=True)
                    with col_opt2:
                        current_weights = []
                        total_v = sum([r['Value'] for r in summary_rows])
                        for ticker in clean_prices.columns:
                            matched_val = 0
                            for row in summary_rows:
                                if row['Stock'] == ticker: matched_val = row['Value']; break
                            current_weights.append(matched_val / total_v if total_v > 0 else 0)
                        
                        df_compare = pd.DataFrame({
                            "Stock": clean_prices.columns,
                            "Current %": [round(w*100,1) for w in current_weights],
                            "Optimal %": [round(w*100,1) for w in opt_weights]
                        })
                        df_melt = df_compare.melt(id_vars='Stock', var_name='Type', value_name='Percentage')
                        fig_bar = px.bar(df_melt, x='Stock', y='Percentage', color='Type', barmode='group',
                                         title="Action Plan: Current vs Optimal",
                                         color_discrete_map={"Current %": "gray", "Optimal %": "#00CC96"})
                        st.plotly_chart(fig_bar, use_container_width=True)
                        st.info("💡 **Insight:** Green > Gray = BUY more. Gray > Green = SELL some.")
                else: st.warning("Insufficient data.")

        # --- 3. AI PREDICTOR ---
        with tab3:
            st.subheader("🔮 3-Month Trend Forecast (Daily Data)")
            target_ml = st.selectbox("Select Stock to Predict", selected_tickers, key="ml_select")
            if target_ml:
                df_daily = data_provider.fetch_ohlc(target_ml, days=365, interval="day")
                if not df_daily.empty:
                    dates, preds = predict_future_prices(df_daily, days_to_predict=90)
                    fig_ml = go.Figure()
                    fig_ml.add_trace(go.Scatter(x=df_daily.index, y=df_daily['Close'], name='Past Year', line=dict(color='#1f77b4')))
                    fig_ml.add_trace(go.Scatter(x=dates, y=preds, name='Next 3 Months (AI)', line=dict(color='#ff7f0e', width=3, dash='dash')))
                    fig_ml.update_layout(title=f"90-Day Forecast for {target_ml}", xaxis_title="Date", yaxis_title="Price (₹)", hovermode="x unified")
                    st.plotly_chart(fig_ml, use_container_width=True)
                    start_p = preds[0]; end_p = preds[-1]
                    change = ((end_p - start_p) / start_p) * 100
                    color = "green" if change > 0 else "red"
                    st.markdown(f"### Predicted Move: <span style='color:{color}'>{change:+.2f}%</span> over 90 days", unsafe_allow_html=True)

        # --- 4. BACKTESTING ---
        with tab4:
            st.subheader("⏮️ Strategy Backtesting Engine")
            target_bt = st.selectbox("Select Stock to Backtest", selected_tickers, key="bt_select")
            if target_bt:
                df_bt = data_provider.fetch_ohlc(target_bt, days=100, interval="day")
                trades, final_val = backtest_strategy(df_bt)
                if trades is not None:
                    c1, c2 = st.columns(2)
                    c1.metric("Initial Capital", "₹100,000")
                    ret = ((final_val - 100000)/100000)*100
                    c2.metric("Final Capital", f"₹{final_val:,.2f}", f"{ret:.2f}%")
                    if trades: st.dataframe(pd.DataFrame(trades))
                else: st.warning("Not enough data to backtest.")

        # --- 5. SENTIMENT ---
        with tab5:
            st.subheader("📰 AI News Sentiment Analysis")
            target_news = st.selectbox("Select Stock for News", selected_tickers, key="news_select")
            if target_news:
                news_df = analyze_news_sentiment(target_news)
                for idx, row in news_df.iterrows():
                    st.markdown(f"<div style='padding:10px; border:1px solid #ddd; border-radius:5px; margin-bottom:10px;'><strong>{row['Headline']}</strong><br>Sentiment: {row['Sentiment']} (Score: {row['Score']})</div>", unsafe_allow_html=True)

        # --- 6. PRO CHARTS ---
        with tab6:
            st.subheader("📈 Professional Technical Chart")
            target_chart = st.selectbox("Analyze Stock", selected_tickers, key="chart_select")
            if target_chart:
                df_c = data_provider.fetch_ohlc(target_chart, days=100, interval="day")
                if not df_c.empty:
                    df_c['SMA_50'] = df_c['Close'].rolling(window=50).mean()
                    df_c['SMA_20'] = df_c['Close'].rolling(window=20).mean()
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
                    fig.add_trace(go.Candlestick(x=df_c.index, open=df_c['Open'], high=df_c['High'], low=df_c['Low'], close=df_c['Close'], name="OHLC"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df_c.index, y=df_c['SMA_20'], line=dict(color='orange', width=1), name="SMA 20"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df_c.index, y=df_c['SMA_50'], line=dict(color='blue', width=1), name="SMA 50"), row=1, col=1)
                    colors = ['red' if row['Open'] - row['Close'] >= 0 else 'green' for index, row in df_c.iterrows()]
                    fig.add_trace(go.Bar(x=df_c.index, y=df_c['Volume'], marker_color=colors, name="Volume"), row=2, col=1)
                    fig.update_layout(title=f"{target_chart} Daily Chart", xaxis_rangeslider_visible=False, height=600, showlegend=True, hovermode="x unified", yaxis_title="Price", yaxis2_title="Volume")
                    st.plotly_chart(fig, use_container_width=True)

        # --- 7. ADVANCED LABS ---
        with tab7:
            st.subheader("🧪 Generative AI & Stress Testing Labs")
            if GENERATIVE_AVAILABLE:
                col_lab1, col_lab2 = st.columns(2)
                with col_lab1:
                    st.markdown("### 💥 Stress Test Simulator")
                    scenario = st.selectbox("Choose Crash Scenario", list(StressTestLab.SCENARIOS.keys()))
                    if st.button("RUN STRESS TEST"):
                        result = StressTestLab.run_simulation(total_val, scenario)
                        if result:
                            st.error(f"⚠️ Projected Loss: -₹{result['loss_amount']:,.2f} ({result['drop_percentage']:.1f}%)")
                            st.metric("Portfolio Value After Crash", f"₹{result['simulated_value']:,.2f}", delta=f"-{result['drop_percentage']:.1f}%", delta_color="inverse")
                            st.caption(result['description'])
                with col_lab2:
                    st.markdown("### 🤖 Hedge Fund Co-Pilot (Groq AI)")
                    user_q = st.text_input("Ask the AI agent about your portfolio:")
                    if st.button("Ask Agent"):
                        agent = AICoPilot()
                        with st.spinner("Agent is thinking..."):
                            response = agent.generate_insight(portfolio_data, user_q)
                            st.info(response)
                    st.markdown("---")
                    st.markdown("#### 💡 AI Buy Recommendations")
                    if st.button("Suggest New Stocks to Buy"):
                        agent = AICoPilot()
                        with st.spinner("Scanning market..."):
                            recommendation = agent.recommend_new_stocks(portfolio_data)
                            st.success("AI Recommendation Generated:")
                            st.markdown(recommendation)
                st.markdown("---")
                st.markdown("### ⚔️ Algo-Battle Arena")
                battle_stock = st.selectbox("Select Stock for Battle", selected_tickers, key="battle_stock")
                if st.button("FIGHT!"):
                    df_battle = data_provider.fetch_ohlc(battle_stock, days=365, interval="day")
                    fig_battle, winner, v1, v2 = AlgoBattle.run_battle(df_battle)
                    if fig_battle:
                        st.plotly_chart(fig_battle, use_container_width=True)
                        if winner == "Golden Cross": st.success(f"🏆 Active Trading beat Buy & Hold by ₹{v2-v1:,.2f}!")
                        else: st.warning(f"🐢 Buy & Hold beat Active Trading by ₹{v1-v2:,.2f}!")
            else:
                st.error("Generative module not loaded. Check 'generative.py'.")
        
        # --- 8. FUNDAMENTALS ---
        with tab8:
            st.subheader("🏥 Fundamental Health Check")
            target_fund = st.selectbox("Select Stock to Audit", selected_tickers, key="fund_select")
            if FUNDAMENTALS_AVAILABLE and target_fund:
                if st.button("Analyze Financial Health"):
                    with st.spinner(f"Auditing {target_fund}'s Balance Sheet..."):
                        health = FundamentalAgent.get_health_card(target_fund)
                        if health["success"]:
                            c1, c2 = st.columns([1, 2])
                            with c1:
                                st.metric("AI Verdict", health["verdict"])
                                st.markdown(f"**Sector:** {health['sector']}")
                                st.markdown(f"**Industry:** {health['industry']}")
                                st.markdown(f"[Official Website]({health['website']})")
                            with c2:
                                st.markdown(f"### Business Summary")
                                st.write(health['summary'])
                            st.markdown("---")
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Market Cap", f"₹{health['market_cap_cr']:,} Cr")
                            pe = health['pe_ratio']; pe_delta = "High (Exp.)" if pe > 30 else "Low (Value)"
                            m2.metric("P/E Ratio", pe, delta=pe_delta, delta_color="inverse")
                            roe = health['roe_pct']
                            m3.metric("ROE (Profitability)", f"{roe}%", delta="Great" if roe > 15 else "Low")
                            debt = health['debt_to_equity']
                            m4.metric("Debt/Equity", debt, delta="High Risk" if debt > 100 else "Safe", delta_color="inverse")
                        else: st.error(f"Could not fetch data: {health.get('error')}")
            else:
                st.warning("Fundamentals module not loaded. Check 'fundamentals.py'.")

else:
    st.info("👈 Please Login and Click 'Run AI Analysis' to start.")