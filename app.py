"""
Finance-AI Analytics Dashboard
Powered by OpenBB · Built with Streamlit
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

from data.openbb_client import (
    get_price_history,
    get_income_statement,
    get_balance_sheet,
    get_cash_flow,
    get_company_overview,
    get_peers,
    get_analyst_estimates,
)
from models.dcf import (
    dcf_valuation,
    capm_cost_of_equity,
    calculate_wacc,
    sensitivity_analysis,
    monte_carlo_dcf,
)
from models.valuation import (
    calculate_financial_ratios,
    sector_comps_valuation,
    comps_valuation,
    SECTOR_MULTIPLES,
)
from components.charts import (
    price_chart,
    dcf_waterfall_chart,
    sensitivity_heatmap,
    monte_carlo_histogram,
    football_field,
    revenue_earnings_chart,
    peer_multiples_bar,
    COLORS,
)

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Finance-AI Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #0e1117; }
    [data-testid="stSidebar"] { background-color: #161b27; }
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 8px;
    }
    .metric-card.green { border-left-color: #00c853; }
    .metric-card.red   { border-left-color: #f44336; }
    .metric-card.orange{ border-left-color: #ff9800; }
    .metric-label { color: #9aa0b4; font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }
    .metric-value { color: #fafafa; font-size: 24px; font-weight: 700; }
    .metric-sub   { color: #9aa0b4; font-size: 12px; }
    .section-header {
        color: #fafafa;
        font-size: 20px;
        font-weight: 700;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 6px;
        margin-bottom: 16px;
    }
    .tag {
        display: inline-block;
        background: #1f77b4;
        color: white;
        border-radius: 4px;
        padding: 2px 10px;
        font-size: 12px;
        margin-right: 6px;
    }
    .valuation-box {
        background: #1e2130;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
        margin-bottom: 8px;
    }
    .val-label { color: #9aa0b4; font-size: 11px; }
    .val-price { color: #fafafa; font-size: 22px; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ────────────────────────────────────────────────────────────────
def fmt_number(val, prefix="", suffix="", decimals=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    if abs(val) >= 1e12:
        return f"{prefix}{val/1e12:.{decimals}f}T{suffix}"
    if abs(val) >= 1e9:
        return f"{prefix}{val/1e9:.{decimals}f}B{suffix}"
    if abs(val) >= 1e6:
        return f"{prefix}{val/1e6:.{decimals}f}M{suffix}"
    return f"{prefix}{val:.{decimals}f}{suffix}"


def pct(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val*100:.1f}%"


def metric_card(label, value, sub=None, color=""):
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    st.markdown(f"""
    <div class="metric-card {color}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {sub_html}
    </div>""", unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_company_data(symbol: str):
    info = get_company_overview(symbol)
    return info


@st.cache_data(ttl=3600, show_spinner=False)
def load_price_data(symbol: str, period_days: int):
    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=period_days)).strftime("%Y-%m-%d")
    return get_price_history(symbol, start, end)


@st.cache_data(ttl=3600, show_spinner=False)
def load_financials(symbol: str):
    income = get_income_statement(symbol)
    balance = get_balance_sheet(symbol)
    cashflow = get_cash_flow(symbol)
    return income, balance, cashflow


@st.cache_data(ttl=3600, show_spinner=False)
def load_peers_data(symbol: str):
    peers = get_peers(symbol)
    peers_info = {}
    for p in peers:
        try:
            peers_info[p] = get_company_overview(p)
        except Exception:
            pass
    return peers, peers_info


# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Finance-AI")
    st.markdown("*Powered by [OpenBB](https://openbb.co)*")
    st.divider()

    symbol = st.text_input("Ticker Symbol", value="AAPL", max_chars=10).upper().strip()
    period_map = {"1 Month": 30, "3 Months": 90, "6 Months": 180,
                  "1 Year": 365, "2 Years": 730, "5 Years": 1825}
    period_label = st.selectbox("Price History Period", list(period_map.keys()), index=3)
    period_days = period_map[period_label]

    st.divider()
    st.markdown("### DCF Assumptions")
    risk_free = st.slider("Risk-Free Rate", 0.01, 0.08, 0.045, 0.005, format="%.3f")
    erp = st.slider("Equity Risk Premium", 0.03, 0.10, 0.055, 0.005, format="%.3f")
    terminal_growth = st.slider("Terminal Growth Rate", 0.01, 0.05, 0.025, 0.005, format="%.3f")
    forecast_years = st.slider("Forecast Years", 3, 10, 5)

    st.divider()
    run_analysis = st.button("🔍 Run Analysis", use_container_width=True, type="primary")

    st.divider()
    st.caption("Data sourced via OpenBB / yfinance. For informational purposes only.")


# ─── Main ───────────────────────────────────────────────────────────────────
if not symbol:
    st.info("Enter a ticker symbol in the sidebar to begin.")
    st.stop()

# Load data with spinners
with st.spinner(f"Loading data for {symbol}…"):
    info = load_company_data(symbol)
    price_df = load_price_data(symbol, period_days)
    income_df, balance_df, cashflow_df = load_financials(symbol)
    peers, peers_info = load_peers_data(symbol)
    analyst = get_analyst_estimates(symbol)

if not info:
    st.error(f"Could not load data for '{symbol}'. Check the ticker and try again.")
    st.stop()

current_price = info.get("current_price") or (
    float(price_df["close"].iloc[-1]) if not price_df.empty else None
)

# ─── Company Header ─────────────────────────────────────────────────────────
name = info.get("name", symbol)
sector = info.get("sector", "N/A")
industry = info.get("industry", "N/A")
market_cap = info.get("market_cap")

st.markdown(f"# {name} &nbsp; `{symbol}`", unsafe_allow_html=False)
col_tags = st.columns([1, 1, 1, 4])
with col_tags[0]:
    st.markdown(f'<span class="tag">{sector}</span>', unsafe_allow_html=True)
with col_tags[1]:
    st.markdown(f'<span class="tag" style="background:#9467bd">{industry}</span>', unsafe_allow_html=True)
with col_tags[2]:
    if market_cap:
        st.markdown(f'<span class="tag" style="background:#2ca02c">Market Cap: {fmt_number(market_cap, "$")}</span>', unsafe_allow_html=True)

st.divider()

# ─── Key Metrics Row ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Key Metrics</div>', unsafe_allow_html=True)
m1, m2, m3, m4, m5, m6 = st.columns(6)

price_color = ""
if current_price and analyst.get("target_price"):
    upside = (analyst["target_price"] - current_price) / current_price
    price_color = "green" if upside > 0 else "red"

with m1:
    metric_card("Current Price", fmt_number(current_price, "$"), color=price_color)
with m2:
    metric_card("P/E Ratio", fmt_number(info.get("pe_ratio"), decimals=1) if info.get("pe_ratio") else "N/A")
with m3:
    metric_card("EV/EBITDA", fmt_number(
        (info.get("enterprise_value") or 0) / (info.get("ebitda") or 1), decimals=1
    ) if info.get("enterprise_value") and info.get("ebitda") else "N/A")
with m4:
    metric_card("Revenue", fmt_number(info.get("revenue"), "$"))
with m5:
    metric_card("EBITDA", fmt_number(info.get("ebitda"), "$"))
with m6:
    if analyst.get("target_price") and current_price:
        upside = (analyst["target_price"] - current_price) / current_price
        color = "green" if upside > 0 else "red"
        metric_card("Analyst Target", fmt_number(analyst["target_price"], "$"),
                    sub=f"Upside: {pct(upside)}", color=color)
    else:
        metric_card("Free Cash Flow", fmt_number(info.get("free_cash_flow"), "$"))


# ─── Tabs ───────────────────────────────────────────────────────────────────
tab_price, tab_financials, tab_dcf, tab_comps, tab_ratios = st.tabs([
    "📊 Price & Technical",
    "📋 Financials",
    "💰 DCF Valuation",
    "🏢 Comparable Cos.",
    "📐 Ratios & Analysis",
])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — PRICE & TECHNICAL
# ══════════════════════════════════════════════════════════════════
with tab_price:
    if price_df.empty:
        st.warning("No price data available.")
    else:
        fig = price_chart(price_df, symbol)
        st.plotly_chart(fig, use_container_width=True)

        # Stats row
        close_col = "close" if "close" in price_df.columns else price_df.columns[-1]
        prices = price_df[close_col].dropna()
        if len(prices) > 1:
            st.markdown('<div class="section-header">Price Statistics</div>', unsafe_allow_html=True)
            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
            with sc1:
                metric_card("52-Week High", fmt_number(prices.max(), "$"), color="green")
            with sc2:
                metric_card("52-Week Low", fmt_number(prices.min(), "$"), color="red")
            with sc3:
                ret = (prices.iloc[-1] / prices.iloc[0] - 1)
                metric_card("Period Return", pct(ret), color="green" if ret > 0 else "red")
            with sc4:
                daily_ret = prices.pct_change().dropna()
                vol = daily_ret.std() * np.sqrt(252)
                metric_card("Annualized Volatility", pct(vol))
            with sc5:
                if len(daily_ret) > 0:
                    sharpe = (daily_ret.mean() * 252 - risk_free) / (daily_ret.std() * np.sqrt(252))
                    metric_card("Sharpe Ratio", f"{sharpe:.2f}")

        # Returns distribution
        if len(prices) > 30:
            st.markdown('<div class="section-header">Returns Distribution</div>', unsafe_allow_html=True)
            daily_ret = prices.pct_change().dropna() * 100
            fig_ret = go.Figure()
            fig_ret.add_trace(go.Histogram(
                x=daily_ret, nbinsx=50, name="Daily Returns",
                marker_color=COLORS["primary"], opacity=0.8,
            ))
            fig_ret.add_vline(x=0, line_dash="solid", line_color="white")
            mean_ret = daily_ret.mean()
            fig_ret.add_vline(x=mean_ret, line_dash="dash", line_color=COLORS["secondary"],
                              annotation_text=f"Mean: {mean_ret:.2f}%")
            fig_ret.update_layout(
                title="Daily Returns Distribution (%)",
                xaxis_title="Daily Return (%)", yaxis_title="Frequency",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#fafafa"), margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_ret, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# TAB 2 — FINANCIALS
# ══════════════════════════════════════════════════════════════════
with tab_financials:
    if income_df.empty and balance_df.empty and cashflow_df.empty:
        st.warning("Financial statement data not available for this ticker.")
    else:
        f1, f2 = st.columns([2, 1])
        with f1:
            if not income_df.empty:
                st.markdown('<div class="section-header">Income Statement</div>', unsafe_allow_html=True)
                fig_rev = revenue_earnings_chart(income_df, symbol)
                st.plotly_chart(fig_rev, use_container_width=True)

        with f2:
            # Margin trend
            if not income_df.empty:
                margin_cols = {
                    "Gross Margin": ["Gross Profit", "Total Revenue"],
                    "Operating Margin": ["Operating Income", "Total Revenue"],
                    "Net Margin": ["Net Income", "Total Revenue"],
                }
                margin_data = {}
                for margin_name, (num_col, denom_col) in margin_cols.items():
                    num_candidates = [c for c in income_df.columns if num_col.lower() in c.lower()]
                    den_candidates = [c for c in income_df.columns if denom_col.lower() in c.lower()]
                    if num_candidates and den_candidates:
                        n = income_df[num_candidates[0]]
                        d = income_df[den_candidates[0]]
                        margin_data[margin_name] = (n / d * 100).round(1)

                if margin_data:
                    st.markdown('<div class="section-header">Margin Trends</div>', unsafe_allow_html=True)
                    fig_margin = go.Figure()
                    colors_list = [COLORS["success"], COLORS["primary"], COLORS["secondary"]]
                    for i, (name_m, series) in enumerate(margin_data.items()):
                        fig_margin.add_trace(go.Scatter(
                            x=series.index, y=series.values,
                            name=name_m, mode="lines+markers",
                            line=dict(color=colors_list[i], width=2),
                        ))
                    fig_margin.update_layout(
                        yaxis_title="Margin (%)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#fafafa"),
                        margin=dict(l=20, r=20, t=40, b=20),
                    )
                    st.plotly_chart(fig_margin, use_container_width=True)

        # Balance sheet highlights
        if not balance_df.empty:
            st.markdown('<div class="section-header">Balance Sheet Highlights</div>', unsafe_allow_html=True)
            b1, b2, b3, b4 = st.columns(4)
            def get_bs_val(df, keywords):
                for kw in keywords:
                    cols = [c for c in df.columns if kw.lower() in c.lower()]
                    if cols:
                        return df[cols[0]].iloc[-1]
                return None

            total_assets = get_bs_val(balance_df, ["Total Assets"])
            total_liab = get_bs_val(balance_df, ["Total Liab"])
            total_equity = get_bs_val(balance_df, ["Total Stockholder", "Total Equity"])
            cash = get_bs_val(balance_df, ["Cash And Cash", "Cash"])

            with b1: metric_card("Total Assets", fmt_number(total_assets, "$"))
            with b2: metric_card("Total Liabilities", fmt_number(total_liab, "$"))
            with b3: metric_card("Shareholders' Equity", fmt_number(total_equity, "$"))
            with b4: metric_card("Cash & Equivalents", fmt_number(cash, "$"), color="green")

        # Raw tables toggle
        with st.expander("📄 View Raw Financial Statements"):
            if not income_df.empty:
                st.subheader("Income Statement")
                st.dataframe(income_df.style.format("{:,.0f}"), use_container_width=True)
            if not balance_df.empty:
                st.subheader("Balance Sheet")
                st.dataframe(balance_df.style.format("{:,.0f}"), use_container_width=True)
            if not cashflow_df.empty:
                st.subheader("Cash Flow Statement")
                st.dataframe(cashflow_df.style.format("{:,.0f}"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# TAB 3 — DCF VALUATION
# ══════════════════════════════════════════════════════════════════
with tab_dcf:
    st.markdown('<div class="section-header">Discounted Cash Flow Model</div>', unsafe_allow_html=True)

    # Pull key inputs from info
    fcf_raw = info.get("free_cash_flow")
    total_debt_raw = info.get("total_debt", 0) or 0
    total_cash_raw = info.get("total_cash", 0) or 0
    shares_raw = info.get("shares_outstanding", 1e9) or 1e9
    beta_raw = info.get("beta", 1.0) or 1.0
    ev_raw = info.get("enterprise_value", 0) or 0

    net_debt_default = total_debt_raw - total_cash_raw

    dcf_col1, dcf_col2 = st.columns([1, 2])

    with dcf_col1:
        st.markdown("**Model Inputs**")
        base_fcf = st.number_input(
            "Base Free Cash Flow ($M)",
            value=round((fcf_raw or 5000) / 1e6, 1),
            step=100.0,
            format="%.1f",
            help="Last twelve months free cash flow",
        ) * 1e6

        beta = st.number_input("Beta", value=round(beta_raw, 2), step=0.05, min_value=0.1)
        cost_equity = capm_cost_of_equity(beta, risk_free, erp)
        st.info(f"CAPM Cost of Equity: **{cost_equity:.2%}**  \n(rf={risk_free:.2%} + β×ERP)")

        cost_debt = st.slider("Pre-tax Cost of Debt", 0.02, 0.12, 0.05, 0.005, format="%.3f")
        tax_rate = st.slider("Tax Rate", 0.10, 0.35, 0.21, 0.01, format="%.2f")

        total_equity_val = (market_cap or ev_raw * 0.7)
        wacc = calculate_wacc(total_equity_val, total_debt_raw, cost_equity, cost_debt, tax_rate)
        st.success(f"Calculated WACC: **{wacc:.2%}**")

        net_debt = st.number_input(
            "Net Debt ($M)",
            value=round(net_debt_default / 1e6, 1),
            step=100.0,
            format="%.1f",
        ) * 1e6
        shares = st.number_input(
            "Shares Outstanding (M)",
            value=round(shares_raw / 1e6, 1),
            step=10.0,
            format="%.1f",
        ) * 1e6

        st.markdown("**Revenue Growth Assumptions**")
        growth_rates = []
        default_growth = [0.15, 0.12, 0.10, 0.08, 0.06]
        for i in range(forecast_years):
            default = default_growth[i] if i < len(default_growth) else 0.05
            g = st.slider(f"Year {i+1} FCF Growth", -0.20, 0.60, default, 0.01, format="%.0%%")
            growth_rates.append(g)

    with dcf_col2:
        # Run DCF
        try:
            result = dcf_valuation(
                base_fcf=base_fcf,
                growth_rates=growth_rates,
                wacc=wacc,
                terminal_growth_rate=terminal_growth,
                net_debt=net_debt,
                shares_outstanding=shares,
            )

            # Summary cards
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                metric_card("Enterprise Value", fmt_number(result["enterprise_value"], "$"))
            with r2:
                metric_card("Equity Value", fmt_number(result["equity_value"], "$"))
            with r3:
                implied = result["implied_price"]
                color = "green" if (current_price and implied > current_price) else "red"
                metric_card("Implied Price", fmt_number(implied, "$"), color=color)
            with r4:
                if current_price and implied:
                    upside = (implied - current_price) / current_price
                    metric_card("Upside / Downside", pct(upside),
                                sub=f"vs. current ${current_price:.1f}",
                                color="green" if upside > 0 else "red")

            st.info(f"Terminal Value is **{result['tv_pct_of_ev']:.0%}** of Enterprise Value")

            # Waterfall chart
            fig_wf = dcf_waterfall_chart(result)
            st.plotly_chart(fig_wf, use_container_width=True)

            # Year-by-year table
            with st.expander("📊 Year-by-Year Cash Flow Projection"):
                yd = result["yearly_details"].copy()
                yd["FCF"] = yd["FCF"].apply(lambda x: fmt_number(x, "$"))
                yd["PV of FCF"] = yd["PV of FCF"].apply(lambda x: fmt_number(x, "$"))
                yd["Discount Factor"] = yd["Discount Factor"].round(4)
                st.dataframe(yd, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"DCF calculation error: {e}")
            result = None

    # ── Sensitivity Analysis
    st.markdown('<div class="section-header">Sensitivity Analysis</div>', unsafe_allow_html=True)
    wacc_range = np.arange(max(0.05, wacc - 0.03), wacc + 0.04, 0.01).tolist()
    tgr_range = np.arange(max(0.005, terminal_growth - 0.015), terminal_growth + 0.02, 0.005).tolist()

    with st.spinner("Running sensitivity analysis…"):
        try:
            sens_df = sensitivity_analysis(
                base_fcf, growth_rates, net_debt, shares,
                wacc_range=wacc_range, tgr_range=tgr_range,
            )
            fig_heat = sensitivity_heatmap(sens_df, current_price)
            st.plotly_chart(fig_heat, use_container_width=True)
        except Exception as e:
            st.warning(f"Sensitivity analysis unavailable: {e}")

    # ── Monte Carlo
    st.markdown('<div class="section-header">Monte Carlo Simulation</div>', unsafe_allow_html=True)
    mc_col1, mc_col2 = st.columns([1, 3])
    with mc_col1:
        mean_g = st.slider("Mean Annual FCF Growth", 0.0, 0.30, 0.10, 0.01, format="%.0%%")
        std_g = st.slider("Std Dev of Growth", 0.01, 0.20, 0.05, 0.01, format="%.0%%")
        n_sims = st.selectbox("Simulations", [1000, 2500, 5000, 10000], index=2)

    with mc_col2:
        with st.spinner("Running Monte Carlo…"):
            try:
                mc = monte_carlo_dcf(
                    base_fcf, mean_g, std_g, forecast_years, wacc,
                    terminal_growth, net_debt, shares, int(n_sims),
                )
                fig_mc = monte_carlo_histogram(mc, current_price)
                st.plotly_chart(fig_mc, use_container_width=True)

                mc2a, mc2b, mc2c, mc2d = st.columns(4)
                with mc2a: metric_card("Median Price", fmt_number(mc["median"], "$"), color="green")
                with mc2b: metric_card("Mean Price", fmt_number(mc["mean"], "$"))
                with mc2c: metric_card("5th Percentile", fmt_number(mc["p5"], "$"), color="red")
                with mc2d: metric_card("95th Percentile", fmt_number(mc["p95"], "$"), color="green")
            except Exception as e:
                st.warning(f"Monte Carlo unavailable: {e}")


# ══════════════════════════════════════════════════════════════════
# TAB 4 — COMPARABLE COMPANIES
# ══════════════════════════════════════════════════════════════════
with tab_comps:
    st.markdown('<div class="section-header">Comparable Company Analysis</div>', unsafe_allow_html=True)

    # Build peer metrics table
    target_row = {
        "Symbol": symbol,
        "Name": info.get("name", symbol),
        "Market Cap": info.get("market_cap"),
        "EV": info.get("enterprise_value"),
        "Revenue": info.get("revenue"),
        "EBITDA": info.get("ebitda"),
        "P/E": info.get("pe_ratio"),
        "EV/EBITDA": (
            (info.get("enterprise_value") or 0) / (info.get("ebitda") or 1)
            if info.get("enterprise_value") and info.get("ebitda") else None
        ),
        "EV/Revenue": (
            (info.get("enterprise_value") or 0) / (info.get("revenue") or 1)
            if info.get("enterprise_value") and info.get("revenue") else None
        ),
        "Gross Margin": info.get("gross_margin"),
        "Net Margin": info.get("profit_margin"),
        "ROE": info.get("roe"),
        "Beta": info.get("beta"),
    }

    rows = [target_row]
    for peer_sym, peer_info in peers_info.items():
        rows.append({
            "Symbol": peer_sym,
            "Name": peer_info.get("name", peer_sym),
            "Market Cap": peer_info.get("market_cap"),
            "EV": peer_info.get("enterprise_value"),
            "Revenue": peer_info.get("revenue"),
            "EBITDA": peer_info.get("ebitda"),
            "P/E": peer_info.get("pe_ratio"),
            "EV/EBITDA": (
                (peer_info.get("enterprise_value") or 0) / (peer_info.get("ebitda") or 1)
                if peer_info.get("enterprise_value") and peer_info.get("ebitda") else None
            ),
            "EV/Revenue": (
                (peer_info.get("enterprise_value") or 0) / (peer_info.get("revenue") or 1)
                if peer_info.get("enterprise_value") and peer_info.get("revenue") else None
            ),
            "Gross Margin": peer_info.get("gross_margin"),
            "Net Margin": peer_info.get("profit_margin"),
            "ROE": peer_info.get("roe"),
            "Beta": peer_info.get("beta"),
        })

    peer_df = pd.DataFrame(rows).set_index("Symbol")

    # Style table: highlight target row
    def highlight_target(row):
        return ["background-color: rgba(31,119,180,0.2); font-weight: bold"
                if row.name == symbol else "" for _ in row]

    # Format large numbers
    format_dict = {
        "Market Cap": lambda x: fmt_number(x, "$") if pd.notna(x) else "N/A",
        "EV": lambda x: fmt_number(x, "$") if pd.notna(x) else "N/A",
        "Revenue": lambda x: fmt_number(x, "$") if pd.notna(x) else "N/A",
        "EBITDA": lambda x: fmt_number(x, "$") if pd.notna(x) else "N/A",
        "P/E": lambda x: f"{x:.1f}x" if pd.notna(x) else "N/A",
        "EV/EBITDA": lambda x: f"{x:.1f}x" if pd.notna(x) else "N/A",
        "EV/Revenue": lambda x: f"{x:.1f}x" if pd.notna(x) else "N/A",
        "Gross Margin": lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A",
        "Net Margin": lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A",
        "ROE": lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A",
        "Beta": lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
    }

    st.dataframe(
        peer_df.style.apply(highlight_target, axis=1).format(format_dict),
        use_container_width=True,
    )

    # Comparison charts
    st.markdown('<div class="section-header">Multiple Comparisons</div>', unsafe_allow_html=True)
    comp_c1, comp_c2 = st.columns(2)

    numeric_peer_df = pd.DataFrame(rows).set_index("Symbol")
    with comp_c1:
        fig_pe = peer_multiples_bar(numeric_peer_df, "P/E", highlight=symbol)
        st.plotly_chart(fig_pe, use_container_width=True)

        fig_evebitda = peer_multiples_bar(numeric_peer_df, "EV/EBITDA", highlight=symbol)
        st.plotly_chart(fig_evebitda, use_container_width=True)

    with comp_c2:
        fig_evrev = peer_multiples_bar(numeric_peer_df, "EV/Revenue", highlight=symbol)
        st.plotly_chart(fig_evrev, use_container_width=True)

        fig_roe = peer_multiples_bar(numeric_peer_df, "ROE", highlight=symbol)
        st.plotly_chart(fig_roe, use_container_width=True)

    # Sector comps valuation
    st.markdown('<div class="section-header">Sector Multiples Valuation</div>', unsafe_allow_html=True)
    sector_for_val = info.get("sector", "Default") or "Default"
    if sector_for_val not in SECTOR_MULTIPLES:
        sector_for_val = "Default"

    revenue = info.get("revenue", 0) or 0
    ebitda = info.get("ebitda", 0) or 0
    net_income = info.get("free_cash_flow", 0) or 0  # proxy
    book_val = (info.get("market_cap", 0) or 0) / max(info.get("pb_ratio") or 3, 0.1)
    net_debt_comp = (info.get("total_debt", 0) or 0) - (info.get("total_cash", 0) or 0)
    shares_comp = info.get("shares_outstanding", 1e9) or 1e9

    if revenue > 0 or ebitda > 0:
        comp_result = sector_comps_valuation(
            sector_for_val, revenue, ebitda, net_income, book_val, net_debt_comp, shares_comp
        )
        sv_cols = st.columns(len(comp_result["valuations"]) + 1)
        for i, (method, vals) in enumerate(comp_result["valuations"].items()):
            with sv_cols[i]:
                price_imp = vals.get("implied_price", 0)
                st.markdown(f"""
                <div class="valuation-box">
                    <div class="val-label">{method}</div>
                    <div class="val-price">{fmt_number(price_imp, "$")}</div>
                    <div class="val-label">{vals['multiple']}x multiple</div>
                </div>""", unsafe_allow_html=True)
        with sv_cols[-1]:
            mean_p = comp_result.get("mean_implied_price")
            st.markdown(f"""
            <div class="valuation-box" style="border: 2px solid #1f77b4;">
                <div class="val-label">Average (Sector Comps)</div>
                <div class="val-price">{fmt_number(mean_p, "$")}</div>
                <div class="val-label">vs. current {fmt_number(current_price, "$")}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("Insufficient financial data for comps valuation.")


# ══════════════════════════════════════════════════════════════════
# TAB 5 — RATIOS & ANALYSIS
# ══════════════════════════════════════════════════════════════════
with tab_ratios:
    st.markdown('<div class="section-header">Financial Ratios Dashboard</div>', unsafe_allow_html=True)

    ratios = calculate_financial_ratios(info)

    if ratios:
        categories = {
            "Valuation": ["P/E Ratio", "Forward P/E", "P/B Ratio", "EV/EBITDA", "EV/Revenue"],
            "Profitability": ["Gross Margin", "Operating Margin", "Net Margin", "ROE", "ROA"],
            "Leverage": ["Debt/Equity", "Current Ratio"],
            "Growth": ["Revenue Growth", "Earnings Growth"],
            "Market": ["Beta", "Dividend Yield"],
        }

        for cat_name, cat_keys in categories.items():
            cat_ratios = {k: v for k, v in ratios.items() if k in cat_keys}
            if not cat_ratios:
                continue

            st.markdown(f"**{cat_name}**")
            cols = st.columns(min(len(cat_ratios), 5))
            for i, (k, v) in enumerate(cat_ratios.items()):
                with cols[i % 5]:
                    if "Margin" in k or "Growth" in k or "Yield" in k or k in ["ROE", "ROA"]:
                        formatted = pct(v)
                        color = "green" if v > 0 else "red"
                    else:
                        formatted = f"{v:.2f}"
                        color = ""
                    metric_card(k, formatted, color=color)
            st.markdown("")

    # Company description
    desc = info.get("description")
    if desc:
        st.markdown('<div class="section-header">Business Overview</div>', unsafe_allow_html=True)
        with st.expander("Read full description", expanded=True):
            st.write(desc[:2000] + ("..." if len(desc or "") > 2000 else ""))

    # Analyst summary
    if any(analyst.values()):
        st.markdown('<div class="section-header">Analyst Consensus</div>', unsafe_allow_html=True)
        an1, an2, an3, an4 = st.columns(4)
        rec = analyst.get("recommendation", "N/A")
        rec_color = {"BUY": "green", "STRONG_BUY": "green", "HOLD": "orange",
                     "SELL": "red", "UNDERPERFORM": "red"}.get(rec, "")
        with an1: metric_card("Recommendation", rec or "N/A", color=rec_color)
        with an2: metric_card("Target Price", fmt_number(analyst.get("target_price"), "$"))
        with an3: metric_card("High Target", fmt_number(analyst.get("target_high"), "$"), color="green")
        with an4: metric_card("Low Target", fmt_number(analyst.get("target_low"), "$"), color="red")

    # Football field summary
    if current_price:
        st.markdown('<div class="section-header">Valuation Summary (Football Field)</div>', unsafe_allow_html=True)
        ff_data = {}

        if analyst.get("target_low") and analyst.get("target_high") and analyst.get("target_price"):
            ff_data["Analyst Consensus"] = {
                "low": analyst["target_low"],
                "mid": analyst["target_price"],
                "high": analyst["target_high"],
            }

        if revenue > 0 and sector_for_val in SECTOR_MULTIPLES:
            mults = SECTOR_MULTIPLES[sector_for_val]
            # Use ±20% around sector median multiple
            ev_rev_mult = mults["ev_revenue"]
            eq_low = (ev_rev_mult * 0.8 * revenue - net_debt_comp) / shares_comp if shares_comp else 0
            eq_mid = (ev_rev_mult * revenue - net_debt_comp) / shares_comp if shares_comp else 0
            eq_high = (ev_rev_mult * 1.2 * revenue - net_debt_comp) / shares_comp if shares_comp else 0
            if eq_low > 0:
                ff_data["EV/Revenue (Sector)"] = {"low": eq_low, "mid": eq_mid, "high": eq_high}

        if result:  # DCF result from earlier tab
            implied_p = result.get("implied_price", 0)
            if implied_p > 0:
                ff_data["DCF (Base Case)"] = {
                    "low": implied_p * 0.8,
                    "mid": implied_p,
                    "high": implied_p * 1.2,
                }

        if ff_data:
            fig_ff = football_field(ff_data, current_price)
            st.plotly_chart(fig_ff, use_container_width=True)
        else:
            st.info("Insufficient data for football field chart.")
