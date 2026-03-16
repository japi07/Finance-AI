"""OpenBB data fetching layer with yfinance fallback."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    from openbb import obb
    OPENBB_AVAILABLE = True
except ImportError:
    OPENBB_AVAILABLE = False


def get_price_history(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical price data."""
    if OPENBB_AVAILABLE:
        try:
            result = obb.equity.price.historical(
                symbol, start_date=start_date, end_date=end_date, provider="yfinance"
            )
            df = result.to_dataframe().reset_index()
            df.columns = [c.lower() for c in df.columns]
            return df
        except Exception:
            pass

    if YFINANCE_AVAILABLE:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        df.rename(columns={"date": "date", "open": "open", "high": "high",
                            "low": "low", "close": "close", "volume": "volume"}, inplace=True)
        return df

    return pd.DataFrame()


def get_income_statement(symbol: str, period: str = "annual", limit: int = 5) -> pd.DataFrame:
    """Fetch income statement data."""
    if OPENBB_AVAILABLE:
        try:
            result = obb.equity.fundamental.income(
                symbol, period=period, limit=limit, provider="yfinance"
            )
            return result.to_dataframe()
        except Exception:
            pass

    if YFINANCE_AVAILABLE:
        ticker = yf.Ticker(symbol)
        if period == "annual":
            df = ticker.income_stmt
        else:
            df = ticker.quarterly_income_stmt
        if df is not None and not df.empty:
            return df.T
    return pd.DataFrame()


def get_balance_sheet(symbol: str, period: str = "annual", limit: int = 5) -> pd.DataFrame:
    """Fetch balance sheet data."""
    if OPENBB_AVAILABLE:
        try:
            result = obb.equity.fundamental.balance(
                symbol, period=period, limit=limit, provider="yfinance"
            )
            return result.to_dataframe()
        except Exception:
            pass

    if YFINANCE_AVAILABLE:
        ticker = yf.Ticker(symbol)
        if period == "annual":
            df = ticker.balance_sheet
        else:
            df = ticker.quarterly_balance_sheet
        if df is not None and not df.empty:
            return df.T
    return pd.DataFrame()


def get_cash_flow(symbol: str, period: str = "annual", limit: int = 5) -> pd.DataFrame:
    """Fetch cash flow statement data."""
    if OPENBB_AVAILABLE:
        try:
            result = obb.equity.fundamental.cash(
                symbol, period=period, limit=limit, provider="yfinance"
            )
            return result.to_dataframe()
        except Exception:
            pass

    if YFINANCE_AVAILABLE:
        ticker = yf.Ticker(symbol)
        if period == "annual":
            df = ticker.cashflow
        else:
            df = ticker.quarterly_cashflow
        if df is not None and not df.empty:
            return df.T
    return pd.DataFrame()


def get_company_overview(symbol: str) -> dict:
    """Fetch company profile and key metrics."""
    info = {}
    if OPENBB_AVAILABLE:
        try:
            result = obb.equity.profile(symbol, provider="yfinance")
            df = result.to_dataframe()
            if not df.empty:
                info.update(df.iloc[0].to_dict())
        except Exception:
            pass

    if YFINANCE_AVAILABLE:
        ticker = yf.Ticker(symbol)
        yf_info = ticker.info or {}
        mapping = {
            "longName": "name",
            "sector": "sector",
            "industry": "industry",
            "longBusinessSummary": "description",
            "marketCap": "market_cap",
            "trailingPE": "pe_ratio",
            "forwardPE": "forward_pe",
            "priceToBook": "pb_ratio",
            "enterpriseValue": "enterprise_value",
            "totalRevenue": "revenue",
            "grossProfits": "gross_profit",
            "ebitda": "ebitda",
            "totalDebt": "total_debt",
            "totalCash": "total_cash",
            "beta": "beta",
            "52WeekHigh": "week_52_high",
            "52WeekLow": "week_52_low",
            "dividendYield": "dividend_yield",
            "returnOnEquity": "roe",
            "returnOnAssets": "roa",
            "debtToEquity": "debt_to_equity",
            "currentRatio": "current_ratio",
            "revenueGrowth": "revenue_growth",
            "earningsGrowth": "earnings_growth",
            "profitMargins": "profit_margin",
            "operatingMargins": "operating_margin",
            "grossMargins": "gross_margin",
            "freeCashflow": "free_cash_flow",
            "sharesOutstanding": "shares_outstanding",
            "currentPrice": "current_price",
            "targetMeanPrice": "analyst_target_price",
        }
        for yf_key, our_key in mapping.items():
            val = yf_info.get(yf_key) or yf_info.get(yf_key.replace("52", "fiftyTwo"))
            if val is not None:
                info[our_key] = val
        if "symbol" not in info:
            info["symbol"] = symbol

    return info


def get_peers(symbol: str) -> list[str]:
    """Get peer/competitor tickers."""
    peers_map = {
        "AAPL": ["MSFT", "GOOGL", "META", "AMZN"],
        "MSFT": ["AAPL", "GOOGL", "AMZN", "META"],
        "GOOGL": ["META", "MSFT", "AAPL", "AMZN"],
        "AMZN": ["MSFT", "GOOGL", "AAPL", "WMT"],
        "META": ["GOOGL", "SNAP", "TWTR", "PINS"],
        "TSLA": ["F", "GM", "RIVN", "NIO"],
        "NVDA": ["AMD", "INTC", "QCOM", "AVGO"],
        "JPM": ["BAC", "WFC", "C", "GS"],
        "JNJ": ["PFE", "MRK", "ABT", "BMY"],
    }
    if symbol.upper() in peers_map:
        return peers_map[symbol.upper()]

    if YFINANCE_AVAILABLE:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        sector = info.get("sector", "")
        industry = info.get("industry", "")
        # Return generic tech peers as default
        return ["MSFT", "AAPL", "GOOGL", "AMZN"]

    return ["MSFT", "AAPL", "GOOGL", "AMZN"]


def get_analyst_estimates(symbol: str) -> dict:
    """Get analyst price targets and recommendations."""
    result = {"target_price": None, "recommendation": None, "num_analysts": None}
    if YFINANCE_AVAILABLE:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        result["target_price"] = info.get("targetMeanPrice")
        result["target_high"] = info.get("targetHighPrice")
        result["target_low"] = info.get("targetLowPrice")
        result["recommendation"] = info.get("recommendationKey", "").upper()
        result["num_analysts"] = info.get("numberOfAnalystOpinions")
    return result
