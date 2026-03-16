"""Portfolio Analyzer page — weights, returns, and risk metrics."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.openbb_client import get_price_history
from components.charts import COLORS, LAYOUT_DEFAULTS

st.set_page_config(page_title="Portfolio Analyzer", page_icon="📐", layout="wide")

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #0e1117; }
    [data-testid="stSidebar"] { background-color: #161b27; }
</style>
""", unsafe_allow_html=True)

st.title("📐 Portfolio Analyzer")
st.markdown("*Analyze a multi-asset portfolio: returns, risk, correlation, and efficient frontier*")
st.divider()

DEFAULT_PORTFOLIO = "AAPL 0.25, MSFT 0.20, GOOGL 0.15, AMZN 0.15, NVDA 0.10, JPM 0.10, JNJ 0.05"

with st.sidebar:
    st.markdown("### Portfolio Builder")
    portfolio_input = st.text_area(
        "Enter holdings (TICKER WEIGHT, …)",
        value=DEFAULT_PORTFOLIO,
        height=180,
        help="Format: TICKER WEIGHT per line or comma-separated. Weights are normalized.",
    )
    period_days = st.selectbox("Analysis Period", [90, 180, 365, 730], index=2,
                                format_func=lambda x: f"{x} days")
    benchmark = st.selectbox("Benchmark", ["^GSPC", "^IXIC", "^DJI"], index=0,
                              format_func=lambda x: {"^GSPC": "S&P 500", "^IXIC": "NASDAQ", "^DJI": "Dow Jones"}.get(x, x))
    rf_rate = st.slider("Risk-Free Rate (Annual)", 0.01, 0.07, 0.045, 0.005, format="%.3f")
    run_btn = st.button("▶ Analyze Portfolio", type="primary", use_container_width=True)


def parse_portfolio(text: str) -> dict:
    holdings = {}
    for item in text.replace("\n", ",").split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split()
        if len(parts) >= 2:
            ticker = parts[0].upper()
            try:
                weight = float(parts[1])
                holdings[ticker] = weight
            except ValueError:
                continue
    # Normalize weights
    total = sum(holdings.values())
    if total > 0:
        holdings = {k: v / total for k, v in holdings.items()}
    return holdings


if run_btn:
    holdings = parse_portfolio(portfolio_input)
    if not holdings:
        st.error("Could not parse portfolio. Use format: TICKER WEIGHT")
        st.stop()

    st.markdown(f"**Portfolio:** {len(holdings)} holdings | Period: {period_days} days")

    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=period_days)).strftime("%Y-%m-%d")

    progress = st.progress(0, text="Loading price data…")
    price_series = {}
    for i, ticker in enumerate(list(holdings.keys()) + [benchmark]):
        progress.progress((i + 1) / (len(holdings) + 1), text=f"Loading {ticker}…")
        df = get_price_history(ticker, start, end)
        if not df.empty:
            close = "close" if "close" in df.columns else df.columns[-1]
            date_col = df.columns[0]
            s = df.set_index(date_col)[close].dropna()
            s.index = pd.to_datetime(s.index)
            price_series[ticker] = s
    progress.empty()

    bench_series = price_series.pop(benchmark, None)
    holdings_with_data = {k: v for k, v in holdings.items() if k in price_series}
    if not holdings_with_data:
        st.error("Could not load price data for any holdings.")
        st.stop()

    # Build aligned returns dataframe
    prices_df = pd.DataFrame(price_series).dropna()
    returns_df = prices_df.pct_change().dropna()

    # Portfolio returns
    weights = pd.Series({k: holdings_with_data.get(k, 0) for k in prices_df.columns})
    weights = weights / weights.sum()
    portfolio_returns = (returns_df * weights).sum(axis=1)

    # Benchmark returns
    bench_returns = None
    if bench_series is not None:
        bench_series = bench_series.reindex(returns_df.index).pct_change().dropna()
        bench_returns = bench_series

    # ── Weights Pie
    st.divider()
    st.markdown("### Portfolio Weights")
    p1, p2 = st.columns([1, 2])
    with p1:
        fig_pie = px.pie(
            values=list(weights.values),
            names=list(weights.index),
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_pie.update_layout(**LAYOUT_DEFAULTS, title="Allocation")
        st.plotly_chart(fig_pie, use_container_width=True)

    with p2:
        # Cumulative returns
        cum_port = (1 + portfolio_returns).cumprod()
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=cum_port.index, y=cum_port * 100 - 100,
            name="Portfolio", line=dict(color=COLORS["primary"], width=2),
        ))
        if bench_returns is not None and len(bench_returns) > 0:
            cum_bench = (1 + bench_returns).cumprod()
            fig_cum.add_trace(go.Scatter(
                x=cum_bench.index, y=cum_bench * 100 - 100,
                name=benchmark, line=dict(color=COLORS["secondary"], width=2, dash="dot"),
            ))
        fig_cum.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
        fig_cum.update_layout(
            title="Cumulative Returns (%)", yaxis_title="Return (%)",
            **LAYOUT_DEFAULTS,
        )
        st.plotly_chart(fig_cum, use_container_width=True)

    # ── Risk Metrics
    st.divider()
    st.markdown("### Risk & Performance Metrics")

    annual_ret = portfolio_returns.mean() * 252
    annual_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe = (annual_ret - rf_rate) / annual_vol if annual_vol > 0 else 0

    cumulative = (1 + portfolio_returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    total_return = cumulative.iloc[-1] - 1

    # Beta and alpha
    beta_val, alpha_val = None, None
    if bench_returns is not None and len(bench_returns) > 5:
        aligned = pd.concat([portfolio_returns, bench_returns], axis=1).dropna()
        if len(aligned) > 5:
            cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
            beta_val = cov[0, 1] / cov[1, 1]
            alpha_val = (annual_ret - rf_rate) - beta_val * (bench_returns.mean() * 252 - rf_rate)

    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
    metrics = [
        ("Total Return", f"{total_return*100:+.1f}%", "green" if total_return > 0 else "red"),
        ("Annual Return", f"{annual_ret*100:.1f}%", "green" if annual_ret > 0 else "red"),
        ("Annual Volatility", f"{annual_vol*100:.1f}%", ""),
        ("Sharpe Ratio", f"{sharpe:.2f}", "green" if sharpe > 1 else "orange" if sharpe > 0 else "red"),
        ("Max Drawdown", f"{max_drawdown*100:.1f}%", "red"),
        ("Beta", f"{beta_val:.2f}" if beta_val else "N/A", ""),
    ]
    for col, (label, value, color) in zip([mc1, mc2, mc3, mc4, mc5, mc6], metrics):
        with col:
            st.markdown(f"""
            <div style="background:#1e2130;border-radius:8px;padding:12px;
                        border-left:4px solid {'#00c853' if color=='green' else '#f44336' if color=='red' else '#ff9800' if color=='orange' else '#1f77b4'};">
                <div style="color:#9aa0b4;font-size:11px;text-transform:uppercase">{label}</div>
                <div style="color:#fafafa;font-size:20px;font-weight:700">{value}</div>
            </div>""", unsafe_allow_html=True)

    # ── Correlation heatmap
    if len(prices_df.columns) > 1:
        st.divider()
        st.markdown("### Correlation Matrix")
        corr = returns_df.corr()
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdYlGn",
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            showscale=True,
        ))
        fig_corr.update_layout(title="Asset Correlation", **LAYOUT_DEFAULTS)
        st.plotly_chart(fig_corr, use_container_width=True)

    # ── Drawdown chart
    st.divider()
    st.markdown("### Drawdown")
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown * 100,
        fill="tozeroy", name="Drawdown",
        line=dict(color=COLORS["red"], width=1),
        fillcolor="rgba(244,67,54,0.3)",
    ))
    fig_dd.update_layout(
        yaxis_title="Drawdown (%)", **LAYOUT_DEFAULTS
    )
    st.plotly_chart(fig_dd, use_container_width=True)

else:
    st.info("Configure your portfolio in the sidebar and click **Analyze Portfolio** to begin.")
    st.markdown("""
    ### How to Use
    1. Enter your holdings in the text box: `AAPL 0.25, MSFT 0.20, …`
    2. Weights are automatically normalized to sum to 100%
    3. Select your analysis period and benchmark
    4. Click **Analyze Portfolio**

    ### Metrics Calculated
    - **Total & Annualized Returns** — absolute and time-adjusted performance
    - **Sharpe Ratio** — risk-adjusted return vs. risk-free rate
    - **Max Drawdown** — largest peak-to-trough decline
    - **Beta** — sensitivity relative to benchmark
    - **Correlation Matrix** — cross-asset diversification view
    """)
