"""Plotly chart components for the analytics dashboard."""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "success": "#2ca02c",
    "danger": "#d62728",
    "accent": "#9467bd",
    "bg": "#0e1117",
    "card": "#1e2130",
    "text": "#fafafa",
    "muted": "#6c757d",
    "green": "#00c853",
    "red": "#f44336",
}

LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=COLORS["text"], family="Inter, sans-serif"),
    margin=dict(l=20, r=20, t=40, b=20),
)


def price_chart(df: pd.DataFrame, symbol: str, show_volume: bool = True) -> go.Figure:
    """Candlestick / OHLCV chart with optional volume bars."""
    if df.empty:
        return go.Figure()

    date_col = "date" if "date" in df.columns else df.columns[0]
    close_col = next((c for c in ["close", "Close"] if c in df.columns), None)
    if close_col is None:
        return go.Figure()

    has_ohlc = all(c in df.columns for c in ["open", "high", "low", "close"])

    if show_volume and "volume" in df.columns:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.75, 0.25],
        )
        row_price = 1
    else:
        fig = make_subplots(rows=1, cols=1)
        row_price = 1

    if has_ohlc:
        fig.add_trace(
            go.Candlestick(
                x=df[date_col],
                open=df["open"], high=df["high"],
                low=df["low"], close=df["close"],
                name=symbol,
                increasing_line_color=COLORS["green"],
                decreasing_line_color=COLORS["red"],
            ),
            row=row_price, col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(x=df[date_col], y=df[close_col], name=symbol,
                       line=dict(color=COLORS["primary"], width=2)),
            row=row_price, col=1,
        )

    # 50-day MA
    if len(df) >= 50:
        df = df.copy()
        df["ma50"] = df[close_col].rolling(50).mean()
        fig.add_trace(
            go.Scatter(x=df[date_col], y=df["ma50"], name="50-day MA",
                       line=dict(color=COLORS["secondary"], width=1, dash="dot")),
            row=row_price, col=1,
        )

    # Volume
    if show_volume and "volume" in df.columns:
        colors = [
            COLORS["green"] if c >= o else COLORS["red"]
            for c, o in zip(df.get("close", df[close_col]), df.get("open", df[close_col]))
        ]
        fig.add_trace(
            go.Bar(x=df[date_col], y=df["volume"], name="Volume",
                   marker_color=colors, opacity=0.5),
            row=2, col=1,
        )

    fig.update_layout(
        title=f"{symbol} — Price History",
        xaxis_rangeslider_visible=False,
        showlegend=True,
        **LAYOUT_DEFAULTS,
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    return fig


def dcf_waterfall_chart(dcf_result: dict) -> go.Figure:
    """Waterfall chart showing DCF build-up to enterprise value."""
    details = dcf_result.get("yearly_details", pd.DataFrame())
    pv_tv = dcf_result.get("pv_terminal_value", 0)

    labels = list(details["Year"]) + ["Terminal Value", "Enterprise Value"]
    values = list(details["PV of FCF"]) + [pv_tv, None]

    # Waterfall
    measure = ["relative"] * (len(details) + 1) + ["total"]
    y_values = list(details["PV of FCF"]) + [pv_tv, dcf_result.get("enterprise_value", 0)]

    fig = go.Figure(go.Waterfall(
        name="DCF",
        orientation="v",
        measure=measure,
        x=labels,
        y=y_values,
        connector={"line": {"color": "rgba(255,255,255,0.2)"}},
        increasing={"marker": {"color": COLORS["success"]}},
        totals={"marker": {"color": COLORS["primary"]}},
        texttemplate="%{y:$.2s}",
        textposition="outside",
    ))

    fig.update_layout(
        title="DCF Valuation Waterfall",
        yaxis_title="Value ($)",
        **LAYOUT_DEFAULTS,
    )
    return fig


def sensitivity_heatmap(df: pd.DataFrame, current_price: float = None) -> go.Figure:
    """Heatmap for DCF sensitivity analysis (WACC vs TGR)."""
    z = df.values.astype(float)

    annotations = []
    for i, row in enumerate(df.index):
        for j, col in enumerate(df.columns):
            val = z[i, j]
            if not np.isnan(val):
                color = "white"
                if current_price:
                    color = COLORS["green"] if val >= current_price else COLORS["red"]
                annotations.append(dict(
                    x=col, y=row, text=f"${val:.0f}",
                    showarrow=False,
                    font=dict(color=color, size=11),
                ))

    fig = go.Figure(go.Heatmap(
        z=z,
        x=df.columns.tolist(),
        y=df.index.tolist(),
        colorscale="RdYlGn",
        showscale=True,
        colorbar=dict(title="Price ($)"),
    ))
    fig.update_layout(
        title="Sensitivity Analysis: Implied Share Price",
        xaxis_title="Terminal Growth Rate",
        yaxis_title="WACC",
        annotations=annotations,
        **LAYOUT_DEFAULTS,
    )
    return fig


def monte_carlo_histogram(mc_result: dict, current_price: float = None) -> go.Figure:
    """Histogram of Monte Carlo DCF simulation results."""
    prices = mc_result.get("prices", [])
    if len(prices) == 0:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=prices,
        nbinsx=60,
        name="Simulated Prices",
        marker_color=COLORS["primary"],
        opacity=0.7,
    ))

    for pct, label, color in [
        ("p5", "5th pct", COLORS["red"]),
        ("median", "Median", COLORS["success"]),
        ("p95", "95th pct", COLORS["secondary"]),
    ]:
        val = mc_result.get(pct)
        if val:
            fig.add_vline(x=val, line_dash="dash", line_color=color,
                          annotation_text=f"{label}: ${val:.1f}",
                          annotation_font_color=color)

    if current_price:
        fig.add_vline(x=current_price, line_dash="solid", line_color="white",
                      annotation_text=f"Current: ${current_price:.1f}",
                      annotation_font_color="white")

    fig.update_layout(
        title="Monte Carlo DCF Simulation",
        xaxis_title="Implied Share Price ($)",
        yaxis_title="Frequency",
        **LAYOUT_DEFAULTS,
    )
    return fig


def football_field(valuations_summary: dict, current_price: float = None) -> go.Figure:
    """
    Football field chart showing valuation ranges per method.
    valuations_summary: {method: {"low": x, "mid": y, "high": z}}
    """
    methods = list(valuations_summary.keys())
    lows = [v["low"] for v in valuations_summary.values()]
    mids = [v["mid"] for v in valuations_summary.values()]
    highs = [v["high"] for v in valuations_summary.values()]

    fig = go.Figure()

    # Range bars
    for i, method in enumerate(methods):
        low, mid, high = lows[i], mids[i], highs[i]
        fig.add_trace(go.Bar(
            name=method,
            x=[high - low],
            y=[method],
            orientation="h",
            base=[low],
            marker_color=px.colors.qualitative.Set2[i % 8],
            opacity=0.7,
            showlegend=True,
        ))
        # Mid point marker
        fig.add_trace(go.Scatter(
            x=[mid], y=[method],
            mode="markers",
            marker=dict(color="white", size=10, symbol="diamond"),
            showlegend=False,
            name=f"{method} mid",
        ))

    if current_price:
        fig.add_vline(x=current_price, line_dash="solid", line_color=COLORS["secondary"],
                      annotation_text=f"Current ${current_price:.1f}",
                      annotation_font_color=COLORS["secondary"])

    fig.update_layout(
        title="Football Field Valuation Summary",
        xaxis_title="Implied Share Price ($)",
        barmode="overlay",
        **LAYOUT_DEFAULTS,
    )
    return fig


def ratios_radar(ratios_by_company: dict, categories: list[str]) -> go.Figure:
    """Radar/spider chart comparing financial ratios across companies."""
    fig = go.Figure()
    colors = list(px.colors.qualitative.Set1)

    for i, (company, ratios) in enumerate(ratios_by_company.items()):
        values = [ratios.get(cat, 0) or 0 for cat in categories]
        values += [values[0]]  # close polygon
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill="toself",
            name=company,
            line_color=colors[i % len(colors)],
            opacity=0.6,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, gridcolor="rgba(255,255,255,0.1)")),
        title="Financial Ratios Comparison",
        **LAYOUT_DEFAULTS,
    )
    return fig


def revenue_earnings_chart(income_df: pd.DataFrame, symbol: str) -> go.Figure:
    """Bar + line chart of revenue and net income over time."""
    if income_df.empty:
        return go.Figure()

    rev_cols = [c for c in income_df.columns if "revenue" in c.lower() or "Revenue" in c]
    ni_cols = [c for c in income_df.columns if "net income" in c.lower() or "Net Income" in c]

    if not rev_cols and not ni_cols:
        return go.Figure()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if rev_cols:
        rev_col = rev_cols[0]
        fig.add_trace(
            go.Bar(x=income_df.index, y=income_df[rev_col] / 1e9,
                   name="Revenue (B)", marker_color=COLORS["primary"], opacity=0.8),
            secondary_y=False,
        )

    if ni_cols:
        ni_col = ni_cols[0]
        fig.add_trace(
            go.Scatter(x=income_df.index, y=income_df[ni_col] / 1e9,
                       name="Net Income (B)", line=dict(color=COLORS["success"], width=3)),
            secondary_y=True,
        )

    fig.update_layout(
        title=f"{symbol} — Revenue & Net Income",
        **LAYOUT_DEFAULTS,
    )
    fig.update_yaxes(title_text="Revenue ($B)", secondary_y=False,
                     gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(title_text="Net Income ($B)", secondary_y=True,
                     gridcolor="rgba(255,255,255,0.05)")
    return fig


def peer_multiples_bar(peers_metrics: pd.DataFrame, metric: str, highlight: str = None) -> go.Figure:
    """Bar chart comparing a financial metric across peers."""
    if peers_metrics.empty or metric not in peers_metrics.columns:
        return go.Figure()

    df = peers_metrics.dropna(subset=[metric]).sort_values(metric, ascending=False)
    colors = [
        COLORS["secondary"] if sym == highlight else COLORS["primary"]
        for sym in df.index
    ]

    fig = go.Figure(go.Bar(
        x=df.index,
        y=df[metric],
        marker_color=colors,
        text=df[metric].round(2),
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Peer Comparison — {metric}",
        yaxis_title=metric,
        **LAYOUT_DEFAULTS,
    )
    return fig
