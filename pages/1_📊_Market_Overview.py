"""Market Overview page — macro indicators and sector performance."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.openbb_client import get_price_history
from components.charts import COLORS, LAYOUT_DEFAULTS

st.set_page_config(page_title="Market Overview", page_icon="🌐", layout="wide")

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #0e1117; }
    [data-testid="stSidebar"] { background-color: #161b27; }
</style>
""", unsafe_allow_html=True)

st.title("🌐 Market Overview")
st.markdown("*Major indices and sector performance at a glance*")
st.divider()

INDICES = {
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "Dow Jones": "^DJI",
    "Russell 2000": "^RUT",
    "VIX": "^VIX",
}

SECTORS = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Consumer Disc.": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Communication": "XLC",
}


@st.cache_data(ttl=3600)
def load_index_data(symbols: dict, days: int = 365):
    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    data = {}
    for name, sym in symbols.items():
        df = get_price_history(sym, start, end)
        if not df.empty:
            close = "close" if "close" in df.columns else df.columns[-1]
            data[name] = df.set_index(df.columns[0])[close]
    return data


with st.spinner("Loading market data…"):
    index_data = load_index_data(INDICES)
    sector_data = load_index_data(SECTORS, days=90)

# ── Index Performance Cards
st.markdown("### Major Indices")
cols = st.columns(len(INDICES))
for i, (name, _) in enumerate(INDICES.items()):
    series = index_data.get(name)
    with cols[i]:
        if series is not None and len(series) > 1:
            current = series.iloc[-1]
            prev = series.iloc[-2]
            chg = (current - prev) / prev
            ytd_chg = (current - series.iloc[0]) / series.iloc[0]
            color = "#00c853" if chg >= 0 else "#f44336"
            sign = "▲" if chg >= 0 else "▼"
            st.markdown(f"""
            <div style="background:#1e2130;border-radius:8px;padding:14px;text-align:center;border-left:4px solid {color};">
                <div style="color:#9aa0b4;font-size:11px;text-transform:uppercase">{name}</div>
                <div style="color:#fafafa;font-size:20px;font-weight:700">{current:,.0f}</div>
                <div style="color:{color};font-size:13px">{sign} {abs(chg)*100:.2f}% today</div>
                <div style="color:#9aa0b4;font-size:11px">YTD: {ytd_chg*100:+.1f}%</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:#1e2130;border-radius:8px;padding:14px;text-align:center;">
                <div style="color:#9aa0b4">{name}</div>
                <div style="color:#fafafa">N/A</div>
            </div>""", unsafe_allow_html=True)

st.divider()

# ── Indices Line Chart
st.markdown("### Index Performance (1 Year, Normalized)")
if index_data:
    fig = go.Figure()
    colors_list = px.colors.qualitative.Set1
    for i, (name, series) in enumerate(index_data.items()):
        if name == "VIX" or series is None or len(series) < 2:
            continue
        normalized = series / series.iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=series.index, y=normalized,
            name=name, mode="lines",
            line=dict(color=colors_list[i % len(colors_list)], width=2),
        ))
    fig.add_hline(y=100, line_dash="dot", line_color="rgba(255,255,255,0.3)")
    fig.update_layout(
        yaxis_title="Indexed to 100",
        **LAYOUT_DEFAULTS,
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Sector Performance
st.markdown("### Sector Performance (90 Days)")
if sector_data:
    sector_returns = {}
    for name, series in sector_data.items():
        if series is not None and len(series) > 1:
            ret = (series.iloc[-1] - series.iloc[0]) / series.iloc[0]
            sector_returns[name] = ret

    if sector_returns:
        sorted_sectors = sorted(sector_returns.items(), key=lambda x: x[1], reverse=True)
        names = [s[0] for s in sorted_sectors]
        rets = [s[1] * 100 for s in sorted_sectors]
        bar_colors = [COLORS["green"] if r >= 0 else COLORS["red"] for r in rets]

        fig_sec = go.Figure(go.Bar(
            x=rets, y=names,
            orientation="h",
            marker_color=bar_colors,
            text=[f"{r:+.1f}%" for r in rets],
            textposition="outside",
        ))
        fig_sec.update_layout(
            title="Sector ETF Returns (90 Days)",
            xaxis_title="Return (%)",
            **LAYOUT_DEFAULTS,
        )
        st.plotly_chart(fig_sec, use_container_width=True)

st.divider()
st.caption("Data sourced via yfinance. Past performance does not indicate future results.")
