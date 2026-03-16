"""Generate a self-contained HTML dashboard for a given ticker."""
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

sys.path.insert(0, "/home/user/Finance-AI")
from data.openbb_client import (
    get_price_history, get_income_statement, get_balance_sheet,
    get_company_overview, get_peers, get_analyst_estimates,
)
from models.dcf import dcf_valuation, capm_cost_of_equity, calculate_wacc, sensitivity_analysis, monte_carlo_dcf
from models.valuation import calculate_financial_ratios, sector_comps_valuation, SECTOR_MULTIPLES
from components.charts import (
    price_chart, dcf_waterfall_chart, sensitivity_heatmap,
    monte_carlo_histogram, football_field, revenue_earnings_chart,
    peer_multiples_bar, COLORS,
)

SYMBOL = sys.argv[1].upper() if len(sys.argv) > 1 else "AAPL"
print(f"Generating dashboard for {SYMBOL}...")

# ── Fetch data
info = get_company_overview(SYMBOL)
end = datetime.today().strftime("%Y-%m-%d")
start = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")
price_df = get_price_history(SYMBOL, start, end)
income_df = get_income_statement(SYMBOL)
peers = get_peers(SYMBOL)
analyst = get_analyst_estimates(SYMBOL)
peers_info = {}
for p in peers[:4]:
    try:
        peers_info[p] = get_company_overview(p)
    except Exception:
        pass

current_price = info.get("current_price") or (
    float(price_df["close"].iloc[-1]) if not price_df.empty else None
)

def fmt(val, prefix="", suffix="", d=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    if abs(val) >= 1e12: return f"{prefix}{val/1e12:.{d}f}T{suffix}"
    if abs(val) >= 1e9:  return f"{prefix}{val/1e9:.{d}f}B{suffix}"
    if abs(val) >= 1e6:  return f"{prefix}{val/1e6:.{d}f}M{suffix}"
    return f"{prefix}{val:.{d}f}{suffix}"

def pct(v):
    if v is None: return "N/A"
    return f"{v*100:.1f}%"

# ── Build all figures
figs = {}

# 1. Price chart
figs["price"] = price_chart(price_df, SYMBOL)

# 2. Revenue & earnings
figs["revenue"] = revenue_earnings_chart(income_df, SYMBOL)

# 3. DCF
fcf = info.get("free_cash_flow") or 5e9
total_debt = info.get("total_debt", 0) or 0
total_cash = info.get("total_cash", 0) or 0
shares = info.get("shares_outstanding", 1e9) or 1e9
beta = info.get("beta", 1.0) or 1.0
market_cap = info.get("market_cap") or 1e12

net_debt = total_debt - total_cash
cost_equity = capm_cost_of_equity(beta, 0.045, 0.055)
wacc = calculate_wacc(market_cap, total_debt, cost_equity, 0.05, 0.21)
growth_rates = [0.15, 0.12, 0.10, 0.08, 0.06]

dcf_result = dcf_valuation(fcf, growth_rates, wacc, 0.025, net_debt, shares)
figs["dcf_waterfall"] = dcf_waterfall_chart(dcf_result)

wacc_range = np.arange(max(0.05, wacc-0.03), wacc+0.04, 0.01).tolist()
tgr_range  = np.arange(0.010, 0.046, 0.005).tolist()
sens_df = sensitivity_analysis(fcf, growth_rates, net_debt, shares, wacc_range, tgr_range)
figs["sensitivity"] = sensitivity_heatmap(sens_df, current_price)

mc = monte_carlo_dcf(fcf, 0.10, 0.05, 5, wacc, 0.025, net_debt, shares, 5000)
figs["monte_carlo"] = monte_carlo_histogram(mc, current_price)

# 4. Peer comparison
rows = [{
    "Symbol": SYMBOL, "P/E": info.get("pe_ratio"),
    "EV/EBITDA": (info.get("enterprise_value") or 0)/(info.get("ebitda") or 1) if info.get("ebitda") else None,
    "EV/Revenue": (info.get("enterprise_value") or 0)/(info.get("revenue") or 1) if info.get("revenue") else None,
    "Net Margin": info.get("profit_margin"), "ROE": info.get("roe"),
}]
for sym, pi in peers_info.items():
    rows.append({
        "Symbol": sym, "P/E": pi.get("pe_ratio"),
        "EV/EBITDA": (pi.get("enterprise_value") or 0)/(pi.get("ebitda") or 1) if pi.get("ebitda") else None,
        "EV/Revenue": (pi.get("enterprise_value") or 0)/(pi.get("revenue") or 1) if pi.get("revenue") else None,
        "Net Margin": pi.get("profit_margin"), "ROE": pi.get("roe"),
    })
peer_df = pd.DataFrame(rows).set_index("Symbol")
figs["pe_bar"]      = peer_multiples_bar(peer_df, "P/E", SYMBOL)
figs["evebitda_bar"] = peer_multiples_bar(peer_df, "EV/EBITDA", SYMBOL)
figs["evrev_bar"]   = peer_multiples_bar(peer_df, "EV/Revenue", SYMBOL)
figs["roe_bar"]     = peer_multiples_bar(peer_df, "ROE", SYMBOL)

# 5. Football field
ff_data = {}
if analyst.get("target_low") and analyst.get("target_high") and analyst.get("target_price"):
    ff_data["Analyst Consensus"] = {"low": analyst["target_low"], "mid": analyst["target_price"], "high": analyst["target_high"]}
implied = dcf_result.get("implied_price", 0)
if implied > 0:
    ff_data["DCF (Base Case)"] = {"low": implied*0.8, "mid": implied, "high": implied*1.2}
sector = info.get("sector", "Default") or "Default"
rev = info.get("revenue", 0) or 0
if rev > 0 and sector in SECTOR_MULTIPLES:
    m = SECTOR_MULTIPLES[sector]["ev_revenue"]
    eq_mid = (m * rev - net_debt) / shares if shares else 0
    if eq_mid > 0:
        ff_data["EV/Revenue (Sector)"] = {"low": eq_mid*0.8, "mid": eq_mid, "high": eq_mid*1.2}
if ff_data:
    figs["football_field"] = football_field(ff_data, current_price)

# ── Render all figures to HTML divs (shared plotly.js)
from plotly.io import to_html
import plotly

plotlyjs = f'<script src="https://cdn.plot.ly/plotly-{plotly.__version__}.min.js"></script>'

def fig_div(fig, height=420):
    return to_html(fig, full_html=False, include_plotlyjs=False,
                   config={"responsive": True, "displayModeBar": False},
                   default_height=height)

# ── Build HTML
name  = info.get("name", SYMBOL)
mc_str = fmt(market_cap, "$")
pe_str = f'{info.get("pe_ratio"):.1f}x' if info.get("pe_ratio") else "N/A"
ev_ebitda = (info.get("enterprise_value") or 0)/(info.get("ebitda") or 1) if info.get("ebitda") else None
ev_ebitda_str = f'{ev_ebitda:.1f}x' if ev_ebitda else "N/A"
rev_str = fmt(info.get("revenue"), "$")
ebitda_str = fmt(info.get("ebitda"), "$")
price_str = fmt(current_price, "$")
implied_str = fmt(implied, "$")
upside = (implied - current_price) / current_price if implied and current_price else None
upside_str = pct(upside) if upside else "N/A"
upside_color = "#00c853" if upside and upside > 0 else "#f44336"
wacc_str = f"{wacc:.2%}"
tgr_str = "2.50%"

# Key ratios
ratios = calculate_financial_ratios(info)

ratios_html = ""
for k, v in list(ratios.items())[:12]:
    if "Margin" in k or "Growth" in k or "Yield" in k or k in ["ROE","ROA"]:
        val_str = pct(v)
        col = "#00c853" if v > 0 else "#f44336"
    else:
        val_str = f"{v:.2f}"
        col = "#e0e0e0"
    ratios_html += f"""
    <div class="kpi-card">
      <div class="kpi-label">{k}</div>
      <div class="kpi-value" style="color:{col}">{val_str}</div>
    </div>"""

# Analyst block
rec = analyst.get("recommendation","")
rec_color = {"BUY":"#00c853","STRONG_BUY":"#00c853","HOLD":"#ff9800","SELL":"#f44336","UNDERPERFORM":"#f44336"}.get(rec,"#9aa0b4")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{SYMBOL} — Finance-AI Dashboard</title>
{plotlyjs}
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0e1117;color:#fafafa;font-family:'Segoe UI',Arial,sans-serif;padding:0}}
  a{{color:#4db8ff}}
  .topbar{{background:linear-gradient(135deg,#1a237e,#0d47a1);padding:18px 32px;display:flex;align-items:center;justify-content:space-between}}
  .topbar h1{{font-size:22px;font-weight:700;letter-spacing:.02em}}
  .topbar .sub{{font-size:13px;color:#90caf9;margin-top:4px}}
  .tag{{display:inline-block;background:rgba(255,255,255,.15);border-radius:4px;padding:2px 10px;font-size:12px;margin-right:6px}}
  .container{{max-width:1400px;margin:0 auto;padding:24px 20px}}
  .kpi-row{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:28px}}
  .kpi-card{{background:#1e2130;border-radius:10px;padding:16px;border-left:4px solid #1f77b4}}
  .kpi-label{{color:#9aa0b4;font-size:11px;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px}}
  .kpi-value{{font-size:22px;font-weight:700;color:#fafafa}}
  .kpi-sub{{font-size:11px;color:#9aa0b4;margin-top:4px}}
  .section{{margin-bottom:36px}}
  .section-title{{font-size:17px;font-weight:700;border-bottom:2px solid #1f77b4;padding-bottom:6px;margin-bottom:16px;color:#e8eaf6}}
  .grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
  .grid-3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}}
  .card{{background:#1e2130;border-radius:10px;padding:16px}}
  .tabs{{display:flex;gap:0;margin-bottom:24px;border-bottom:2px solid #1f77b4}}
  .tab{{padding:10px 22px;cursor:pointer;font-size:14px;font-weight:600;color:#9aa0b4;border-bottom:3px solid transparent;margin-bottom:-2px;transition:.2s}}
  .tab.active,.tab:hover{{color:#fafafa;border-bottom-color:#1f77b4}}
  .tab-content{{display:none}}.tab-content.active{{display:block}}
  .info-box{{background:#1e2130;border-radius:8px;padding:16px;margin-bottom:16px;font-size:14px;line-height:1.6;color:#cfd8dc}}
  .model-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px}}
  .val-box{{background:#1e2130;border-radius:8px;padding:14px;text-align:center}}
  .val-label{{color:#9aa0b4;font-size:11px;text-transform:uppercase;margin-bottom:6px}}
  .val-price{{font-size:20px;font-weight:700}}
  table{{width:100%;border-collapse:collapse;font-size:13px}}
  th{{background:#161b27;padding:10px 12px;text-align:left;color:#9aa0b4;font-weight:600;border-bottom:1px solid #2a2f3e}}
  td{{padding:9px 12px;border-bottom:1px solid #1e2130;color:#e0e0e0}}
  tr:hover td{{background:#1e2130}}
  tr.highlight td{{background:rgba(31,119,180,.18);font-weight:600}}
  footer{{text-align:center;padding:24px;color:#9aa0b4;font-size:12px;border-top:1px solid #1e2130;margin-top:40px}}
  @media(max-width:768px){{.grid-2,.grid-3,.model-row{{grid-template-columns:1fr}}.kpi-row{{grid-template-columns:repeat(2,1fr)}}}}
</style>
</head>
<body>

<div class="topbar">
  <div>
    <h1>📈 {name} &nbsp;<span style="font-weight:400;font-size:16px">({SYMBOL})</span></h1>
    <div class="sub">
      <span class="tag">{info.get('sector','N/A')}</span>
      <span class="tag">{info.get('industry','N/A')}</span>
      &nbsp; Finance-AI Analytics Dashboard &middot; Generated {datetime.today().strftime('%B %d, %Y')}
    </div>
  </div>
  <div style="text-align:right">
    <div style="font-size:28px;font-weight:700">{price_str}</div>
    <div style="font-size:12px;color:#90caf9">Current Price</div>
  </div>
</div>

<div class="container">

<!-- KPI Row -->
<div class="kpi-row" style="margin-top:20px">
  <div class="kpi-card"><div class="kpi-label">Market Cap</div><div class="kpi-value">{mc_str}</div></div>
  <div class="kpi-card"><div class="kpi-label">P/E Ratio</div><div class="kpi-value">{pe_str}</div></div>
  <div class="kpi-card"><div class="kpi-label">EV/EBITDA</div><div class="kpi-value">{ev_ebitda_str}</div></div>
  <div class="kpi-card"><div class="kpi-label">Revenue (TTM)</div><div class="kpi-value">{rev_str}</div></div>
  <div class="kpi-card"><div class="kpi-label">EBITDA</div><div class="kpi-value">{ebitda_str}</div></div>
  <div class="kpi-card" style="border-left-color:{upside_color}"><div class="kpi-label">DCF Implied Price</div><div class="kpi-value" style="color:{upside_color}">{implied_str}</div><div class="kpi-sub">Upside: {upside_str}</div></div>
  <div class="kpi-card"><div class="kpi-label">WACC</div><div class="kpi-value">{wacc_str}</div></div>
  <div class="kpi-card" style="border-left-color:{rec_color}"><div class="kpi-label">Analyst Rec.</div><div class="kpi-value" style="color:{rec_color}">{rec or 'N/A'}</div><div class="kpi-sub">Target: {fmt(analyst.get('target_price'),'$')}</div></div>
</div>

<!-- Tabs -->
<div class="tabs">
  <div class="tab active" onclick="showTab('price')">📊 Price &amp; Technical</div>
  <div class="tab" onclick="showTab('financials')">📋 Financials</div>
  <div class="tab" onclick="showTab('dcf')">💰 DCF Valuation</div>
  <div class="tab" onclick="showTab('comps')">🏢 Comparables</div>
  <div class="tab" onclick="showTab('ratios')">📐 Ratios &amp; Analysis</div>
</div>

<!-- TAB: PRICE -->
<div id="tab-price" class="tab-content active">
  <div class="section">
    <div class="section-title">Price History — {SYMBOL} (1 Year)</div>
    <div class="card">{fig_div(figs['price'], 480)}</div>
  </div>
</div>

<!-- TAB: FINANCIALS -->
<div id="tab-financials" class="tab-content">
  <div class="section">
    <div class="section-title">Revenue &amp; Net Income</div>
    <div class="card">{fig_div(figs['revenue'], 400)}</div>
  </div>
</div>

<!-- TAB: DCF -->
<div id="tab-dcf" class="tab-content">
  <div class="section">
    <div class="section-title">DCF Model Assumptions</div>
    <div class="info-box">
      <b>Base FCF:</b> {fmt(fcf,'$')} &nbsp;|&nbsp;
      <b>WACC:</b> {wacc_str} (CAPM: rf=4.5% + β{beta:.2f}×ERP5.5%) &nbsp;|&nbsp;
      <b>Growth:</b> 15%→12%→10%→8%→6% &nbsp;|&nbsp;
      <b>Terminal Growth:</b> {tgr_str} &nbsp;|&nbsp;
      <b>TV % of EV:</b> {dcf_result['tv_pct_of_ev']:.0%}
    </div>
    <div class="model-row">
      <div class="val-box"><div class="val-label">Enterprise Value</div><div class="val-price">{fmt(dcf_result['enterprise_value'],'$')}</div></div>
      <div class="val-box"><div class="val-label">Equity Value</div><div class="val-price">{fmt(dcf_result['equity_value'],'$')}</div></div>
      <div class="val-box"><div class="val-label">Implied Price</div><div class="val-price" style="color:{upside_color}">{implied_str}</div></div>
      <div class="val-box"><div class="val-label">Upside / Downside</div><div class="val-price" style="color:{upside_color}">{upside_str}</div></div>
    </div>
    <div class="card">{fig_div(figs['dcf_waterfall'], 380)}</div>
  </div>
  <div class="section">
    <div class="section-title">Sensitivity Analysis (WACC × Terminal Growth Rate)</div>
    <div class="card">{fig_div(figs['sensitivity'], 400)}</div>
  </div>
  <div class="section">
    <div class="section-title">Monte Carlo Simulation (5,000 paths)</div>
    <div class="card">{fig_div(figs['monte_carlo'], 380)}</div>
    <div class="model-row" style="margin-top:12px">
      <div class="val-box"><div class="val-label">Median Price</div><div class="val-price" style="color:#00c853">{fmt(mc['median'],'$')}</div></div>
      <div class="val-box"><div class="val-label">5th Percentile</div><div class="val-price" style="color:#f44336">{fmt(mc['p5'],'$')}</div></div>
      <div class="val-box"><div class="val-label">95th Percentile</div><div class="val-price" style="color:#00c853">{fmt(mc['p95'],'$')}</div></div>
      <div class="val-box"><div class="val-label">Std Deviation</div><div class="val-price">{fmt(mc['std'],'$')}</div></div>
    </div>
  </div>
</div>

<!-- TAB: COMPS -->
<div id="tab-comps" class="tab-content">
  <div class="section">
    <div class="section-title">Peer Comparison Table</div>
    <div class="card" style="overflow-x:auto">
      <table>
        <tr><th>Symbol</th><th>P/E</th><th>EV/EBITDA</th><th>EV/Revenue</th><th>Net Margin</th><th>ROE</th></tr>
        {''.join(
          f'<tr class="{"highlight" if r["Symbol"]==SYMBOL else ""}"><td><b>{r["Symbol"]}</b></td>'
          f'<td>{f"{r[chr(80)+chr(47)+chr(69)]:.1f}x" if r.get("P/E") else "N/A"}</td>'
          f'<td>{f"{r.get(chr(69)+chr(86)+chr(47)+chr(69)+chr(66)+chr(73)+chr(84)+chr(68)+chr(65)):.1f}x" if r.get("EV/EBITDA") else "N/A"}</td>'
          f'<td>{f"{r.get(chr(69)+chr(86)+chr(47)+chr(82)+chr(101)+chr(118)+chr(101)+chr(110)+chr(117)+chr(101)):.1f}x" if r.get("EV/Revenue") else "N/A"}</td>'
          f'<td>{pct(r.get("Net Margin"))}</td>'
          f'<td>{pct(r.get("ROE"))}</td></tr>'
          for r in rows
        )}
      </table>
    </div>
  </div>
  <div class="section">
    <div class="section-title">Multiple Comparisons</div>
    <div class="grid-2">
      <div class="card">{fig_div(figs['pe_bar'], 320)}</div>
      <div class="card">{fig_div(figs['evebitda_bar'], 320)}</div>
      <div class="card">{fig_div(figs['evrev_bar'], 320)}</div>
      <div class="card">{fig_div(figs['roe_bar'], 320)}</div>
    </div>
  </div>
</div>

<!-- TAB: RATIOS -->
<div id="tab-ratios" class="tab-content">
  <div class="section">
    <div class="section-title">Financial Ratios</div>
    <div class="kpi-row">{ratios_html}</div>
  </div>
  {'<div class="section"><div class="section-title">Valuation Summary — Football Field</div><div class="card">' + fig_div(figs["football_field"], 360) + '</div></div>' if "football_field" in figs else ""}
  <div class="section">
    <div class="section-title">Business Overview</div>
    <div class="info-box">{(info.get('description') or 'No description available.')[:1500]}</div>
  </div>
</div>

</div><!-- /container -->

<footer>
  Finance-AI Analytics Dashboard &middot; Powered by OpenBB &amp; yfinance &middot;
  For informational purposes only — not financial advice.
</footer>

<script>
function showTab(name) {{
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.target.classList.add('active');
}}
</script>
</body>
</html>"""

out = f"/home/user/Finance-AI/{SYMBOL}_dashboard.html"
with open(out, "w") as f:
    f.write(html)
print(f"Done! Saved to: {out}")
print(f"  Current Price : {price_str}")
print(f"  DCF Implied   : {implied_str}  (upside: {upside_str})")
print(f"  WACC          : {wacc_str}")
print(f"  Monte Carlo   : median={fmt(mc['median'],'$')}  p5={fmt(mc['p5'],'$')}  p95={fmt(mc['p95'],'$')}")
