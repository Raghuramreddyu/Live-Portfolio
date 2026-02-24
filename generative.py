import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from dotenv import load_dotenv

# Try importing Groq, handle error if not installed
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

load_dotenv()

# ==========================================
# 1. STRESS TEST LAB 🧪
# ==========================================
class StressTestLab:
    SCENARIOS = {
        "2008 Financial Crisis": {"drop": -0.35, "desc": "Global banking collapse scenario"},
        "2020 COVID Crash": {"drop": -0.28, "desc": "Pandemic-induced panic selling"},
        "2000 Dot-Com Bubble": {"drop": -0.40, "desc": "Tech sector valuation correction"},
        "mild_recession": {"drop": -0.15, "desc": "Standard economic slowdown"}
    }

    @staticmethod
    def run_simulation(portfolio_value, scenario_key):
        scenario = StressTestLab.SCENARIOS.get(scenario_key)
        if not scenario: return None
        
        drop_pct = scenario['drop']
        simulated_value = portfolio_value * (1 + drop_pct)
        loss = portfolio_value - simulated_value
        
        return {
            "scenario": scenario_key,
            "description": scenario['desc'],
            "original_value": portfolio_value,
            "simulated_value": simulated_value,
            "loss_amount": loss,
            "drop_percentage": drop_pct * 100
        }

# ==========================================
# 2. GENERATIVE AI CO-PILOT 🤖
# ==========================================
class AICoPilot:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = None
        if GROQ_AVAILABLE and self.api_key:
            self.client = Groq(api_key=self.api_key)

    def generate_insight(self, portfolio_data, user_question=None):
        if not self.client:
            return "⚠️ AI Error: Please add GROQ_API_KEY to .env to enable the Co-Pilot."

        holdings_str = ", ".join([f"{item['Symbol']} ({item['Qty']} qty)" for item in portfolio_data])
        system_prompt = "You are a ruthless hedge fund manager AI. Be concise, direct, and slightly cynical."
        
        if user_question:
            user_content = f"My Portfolio: {holdings_str}. User Question: {user_question}"
        else:
            user_content = f"Review this portfolio: {holdings_str}. Identify one major risk and one opportunity."

        try:
            return self._call_groq(system_prompt, user_content)
        except Exception as e:
            return f"⚠️ AI Error: {str(e)}"

    def recommend_new_stocks(self, portfolio_data):
        if not self.client:
            return "⚠️ AI Error: Please add GROQ_API_KEY to .env."

        holdings_str = ", ".join([f"{item['Symbol']}" for item in portfolio_data])
        system_prompt = "You are an expert Indian Stock Market Advisor. Your goal is to suggest diversification."
        
        user_content = f"""
        Current Portfolio: {holdings_str}
        Task:
        1. Identify the sectors currently represented.
        2. Suggest 2 NEW stocks (NSE Tickers) from DIFFERENT sectors to improve diversification.
        3. For each suggestion, explain clearly WHY it makes the portfolio better.
        Format:
        * **[Stock Symbol]**: [Reason]
        * **[Stock Symbol]**: [Reason]
        """

        try:
            return self._call_groq(system_prompt, user_content)
        except Exception as e:
            return f"⚠️ AI Error: {str(e)}"

    def _call_groq(self, sys_prompt, user_prompt):
        completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b-versatile", 
        )
        return completion.choices[0].message.content

# ==========================================
# 3. ALGO-BATTLE ARENA ⚔️
# ==========================================
class AlgoBattle:
    @staticmethod
    def run_battle(df, initial_capital=100000):
        # FIX: Return empty go.Figure() instead of None to prevent Streamlit crash
        if df.empty or len(df) < 55:
            fig = go.Figure()
            fig.update_layout(title="Not enough data to run Algo Battle.")
            return fig, "N/A", 0, 0

        df = df.copy()
        start_price = df['Close'].iloc[0]
        end_price = df['Close'].iloc[-1]
        units = initial_capital / start_price
        final_bh = units * end_price
        df['Equity_BuyHold'] = units * df['Close']

        # Strategy B: Golden Cross
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        capital = initial_capital
        position = 0
        equity_curve = []
        
        for i in range(len(df)):
            price = df['Close'].iloc[i]
            sma20 = df['SMA_20'].iloc[i]
            sma50 = df['SMA_50'].iloc[i]
            
            if i > 50:
                if sma20 > sma50 and position == 0:
                    position = capital / price 
                    capital = 0
                elif sma20 < sma50 and position > 0:
                    capital = position * price 
                    position = 0
            equity_curve.append(capital + (position * price))
            
        df['Equity_GoldenCross'] = equity_curve
        final_gc = equity_curve[-1]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Equity_BuyHold'], mode='lines', name=f'Buy & Hold (₹{final_bh:,.0f})', line=dict(color='grey', dash='dash')))
        fig.add_trace(go.Scatter(x=df.index, y=df['Equity_GoldenCross'], mode='lines', name=f'Golden Cross (₹{final_gc:,.0f})', line=dict(color='orange', width=2)))
        
        winner = "Golden Cross" if final_gc > final_bh else "Buy & Hold"
        fig.update_layout(title=f"⚔️ Battle Result: {winner} Wins!", xaxis_title="Time", yaxis_title="Portfolio Value (₹)")
        
        return fig, winner, final_bh, final_gc