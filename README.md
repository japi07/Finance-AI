# Finance-AI Analytics Dashboard

A comprehensive financial analytics dashboard powered by [OpenBB](https://openbb.co) and built with [Streamlit](https://streamlit.io).

## Features

### Main Dashboard (`app.py`)
- **Price & Technical** — Candlestick chart with 50-day MA, volume, returns distribution, volatility, and Sharpe ratio
- **Financials** — Income statement, balance sheet, cash flow visualization, and margin trend charts
- **DCF Valuation** — Full discounted cash flow model with CAPM WACC, sensitivity heatmap (WACC × TGR), and Monte Carlo simulation
- **Comparable Companies** — Peer multiples table (P/E, EV/EBITDA, EV/Revenue), sector comps valuation, and bar chart comparisons
- **Ratios & Analysis** — Financial ratios dashboard, analyst consensus, football field valuation summary

### Additional Pages
- 📊 **Market Overview** — Major indices (S&P 500, NASDAQ, Dow, Russell 2000, VIX), normalized performance chart, sector ETF heatmap
- 🔍 **Stock Screener** — Filter 50+ stocks across 5 sectors by P/E, market cap, net margin, debt/equity; scatter plot visualization
- 📐 **Portfolio Analyzer** — Multi-asset portfolio returns, Sharpe ratio, max drawdown, beta, alpha, correlation matrix

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data | [OpenBB](https://openbb.co) + [yfinance](https://pypi.org/project/yfinance/) fallback |
| Models | Custom DCF (Monte Carlo, sensitivity), Comps valuation |
| UI | [Streamlit](https://streamlit.io) |
| Charts | [Plotly](https://plotly.com) |

## Valuation Models

### DCF (Discounted Cash Flow)
- CAPM-derived cost of equity: `rf + β × ERP`
- WACC calculation with debt tax shield
- Multi-year FCF projection with user-defined growth rates
- Gordon Growth Model terminal value
- **Sensitivity analysis**: 5×4 WACC × terminal growth rate heatmap
- **Monte Carlo simulation**: 5,000 paths with normal FCF growth distribution

### Comparable Company Analysis
- Peer multiples: EV/Revenue, EV/EBITDA, P/E, P/B
- Sector median multiples database (11 sectors)
- Football field chart combining DCF + comps + analyst targets

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Usage

1. Enter a ticker symbol (e.g., `AAPL`, `MSFT`, `TSLA`) in the sidebar
2. Adjust DCF assumptions (risk-free rate, ERP, terminal growth, forecast years)
3. Click **Run Analysis**
4. Navigate tabs to explore Price, Financials, DCF, Comps, and Ratios

## Data Sources

Data is fetched via the **OpenBB SDK** with **yfinance** as a fallback provider. No API keys are required for basic usage.

## Disclaimer

This dashboard is for **informational and educational purposes only**. It does not constitute financial advice. Always conduct your own research before making investment decisions.
