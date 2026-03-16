"""Stock Screener page — filter stocks by fundamental criteria."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.openbb_client import get_company_overview
from components.charts import COLORS, LAYOUT_DEFAULTS

st.set_page_config(page_title="Stock Screener", page_icon="🔍", layout="wide")

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #0e1117; }
    [data-testid="stSidebar"] { background-color: #161b27; }
</style>
""", unsafe_allow_html=True)

st.title("🔍 Stock Screener")
st.markdown("*Screen stocks across sectors by fundamental metrics*")
st.divider()

# Predefined universe by sector
STOCK_UNIVERSE = {
    "Technology": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "AMD", "INTC", "QCOM", "AVGO"],
    "Healthcare": ["JNJ", "UNH", "PFE", "MRK", "ABT", "AMGN", "BMY", "GILD", "CVS", "CI"],
    "Financials": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "AXP", "CB", "USB"],
    "Consumer": ["TSLA", "AMZN", "HD", "MCD", "NKE", "SBUX", "TGT", "COST", "WMT", "LOW"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "VLO", "PSX", "OXY", "HAL"],
}

with st.sidebar:
    st.markdown("### Screener Filters")
    selected_sectors = st.multiselect(
        "Sectors", list(STOCK_UNIVERSE.keys()), default=["Technology"]
    )
    max_pe = st.slider("Max P/E Ratio", 5.0, 100.0, 40.0, 1.0)
    min_market_cap_b = st.slider("Min Market Cap ($B)", 1.0, 500.0, 10.0, 1.0)
    min_margin = st.slider("Min Net Margin (%)", -20.0, 50.0, 0.0, 1.0)
    max_debt_equity = st.slider("Max Debt/Equity", 0.0, 10.0, 3.0, 0.1)
    run_screen = st.button("🔍 Run Screener", type="primary", use_container_width=True)

if run_screen and selected_sectors:
    tickers = []
    for sector in selected_sectors:
        tickers.extend(STOCK_UNIVERSE.get(sector, []))

    progress = st.progress(0, text="Loading company data…")
    results = []

    for i, ticker in enumerate(tickers):
        progress.progress((i + 1) / len(tickers), text=f"Loading {ticker}…")
        try:
            info = get_company_overview(ticker)
            if not info:
                continue
            pe = info.get("pe_ratio")
            mc = info.get("market_cap", 0) or 0
            margin = info.get("profit_margin", 0) or 0
            de = info.get("debt_to_equity", 0) or 0
            price = info.get("current_price")
            rev_growth = info.get("revenue_growth", 0) or 0
            roe = info.get("roe", 0) or 0

            # Apply filters
            if mc < min_market_cap_b * 1e9:
                continue
            if pe is not None and pe > max_pe:
                continue
            if margin * 100 < min_margin:
                continue
            if de > max_debt_equity:
                continue

            results.append({
                "Symbol": ticker,
                "Name": info.get("name", ticker),
                "Sector": info.get("sector", "N/A"),
                "Price": price,
                "Market Cap ($B)": mc / 1e9,
                "P/E": pe,
                "Net Margin": margin,
                "Revenue Growth": rev_growth,
                "ROE": roe,
                "Debt/Equity": de,
                "Beta": info.get("beta"),
                "Div Yield": info.get("dividend_yield"),
            })
        except Exception:
            continue

    progress.empty()

    if not results:
        st.warning("No stocks matched your criteria. Try relaxing the filters.")
    else:
        st.success(f"Found **{len(results)}** stocks matching your criteria.")
        df = pd.DataFrame(results)

        # Format for display
        def fmt(val, mult=1, suffix="", prefix=""):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return "N/A"
            return f"{prefix}{val*mult:.1f}{suffix}"

        display_df = df.copy()
        display_df["Market Cap ($B)"] = display_df["Market Cap ($B)"].apply(lambda x: f"${x:.1f}B")
        display_df["Net Margin"] = display_df["Net Margin"].apply(lambda x: fmt(x, 100, "%"))
        display_df["Revenue Growth"] = display_df["Revenue Growth"].apply(lambda x: fmt(x, 100, "%"))
        display_df["ROE"] = display_df["ROE"].apply(lambda x: fmt(x, 100, "%"))
        display_df["Div Yield"] = display_df["Div Yield"].apply(lambda x: fmt(x, 100, "%") if x else "N/A")
        display_df["Price"] = display_df["Price"].apply(lambda x: f"${x:.2f}" if x else "N/A")
        display_df["P/E"] = display_df["P/E"].apply(lambda x: f"{x:.1f}x" if x else "N/A")
        display_df["Beta"] = display_df["Beta"].apply(lambda x: f"{x:.2f}" if x else "N/A")
        display_df["Debt/Equity"] = display_df["Debt/Equity"].apply(lambda x: f"{x:.2f}" if x else "N/A")

        st.dataframe(display_df.set_index("Symbol"), use_container_width=True)

        # Scatter: Market Cap vs P/E
        st.divider()
        st.markdown("### Valuation Scatter")
        scatter_df = df.dropna(subset=["P/E", "Net Margin", "Market Cap ($B)"])
        if not scatter_df.empty:
            fig = px.scatter(
                scatter_df,
                x="P/E", y="Net Margin",
                size="Market Cap ($B)",
                color="Sector",
                hover_name="Name",
                hover_data=["Symbol", "Revenue Growth", "ROE"],
                title="P/E vs Net Margin (bubble = market cap)",
                template="plotly_dark",
            )
            fig.update_layout(**LAYOUT_DEFAULTS)
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Configure your filters in the sidebar and click **Run Screener** to begin.")
    st.markdown("""
    ### Available Metrics
    | Metric | Description |
    |--------|-------------|
    | P/E Ratio | Price-to-Earnings — valuation multiple |
    | Market Cap | Total market capitalization |
    | Net Margin | Net income as % of revenue |
    | Revenue Growth | YoY revenue growth rate |
    | ROE | Return on Equity |
    | Debt/Equity | Leverage ratio |
    | Beta | Market sensitivity |
    | Dividend Yield | Annual dividend / price |
    """)
