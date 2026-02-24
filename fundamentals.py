import yfinance as yf

class FundamentalAgent:
    """
    Fetches financial health metrics (P/E, ROE, Debt) using Yahoo Finance.
    Independent of Upstox API.
    """
    
    @staticmethod
    def get_health_card(symbol):
        """
        Returns a dictionary of key fundamental metrics.
        Handles the '.NS' suffix for NSE stocks automatically.
        """
        try:
            # 1. Handle Symbol Format (e.g., RELIANCE -> RELIANCE.NS)
            if not symbol.endswith(".NS"):
                ns_symbol = f"{symbol}.NS"
            else:
                ns_symbol = symbol
                
            stock = yf.Ticker(ns_symbol)
            info = stock.info
            
            # 2. Extract Data (Safely handle missing keys and None values)
            market_cap = info.get("marketCap") or 0
            pe_ratio = info.get("trailingPE") or 0
            roe = info.get("returnOnEquity") or 0
            debt_to_equity = info.get("debtToEquity") or 0
            sector = info.get("sector", "Unknown")
            industry = info.get("industry", "Unknown")
            website = info.get("website", "#")
            summary = info.get("longBusinessSummary", "No summary available.")
            
            # 3. Automatic "Verdict" based on simple rules
            verdict = "NEUTRAL"
            if pe_ratio > 0 and pe_ratio < 25 and roe > 0.15:
                verdict = "UNDERVALUED (Good to Buy)"
            elif pe_ratio > 50:
                verdict = "OVERVALUED (Expensive)"
            
            return {
                "success": True,
                "symbol": symbol,
                "sector": sector,
                "industry": industry,
                "market_cap_cr": round(market_cap / 10000000, 2), # Convert to Crores
                "pe_ratio": round(pe_ratio, 2),
                "roe_pct": round(roe * 100, 2),
                "debt_to_equity": round(debt_to_equity, 2),
                "summary": summary[:400] + "...", # Truncate long text
                "website": website,
                "verdict": verdict
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def get_peer_comparison(symbol):
        """
        Fetches a quick comparison if available.
        """
        return "Peer comparison requires a paid data subscription."