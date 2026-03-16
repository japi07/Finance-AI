"""Comparable company analysis and multiples-based valuation."""
import pandas as pd
import numpy as np
from typing import Optional


SECTOR_MULTIPLES = {
    "Technology": {"ev_revenue": 8.0, "ev_ebitda": 25.0, "pe": 30.0, "pb": 7.0},
    "Healthcare": {"ev_revenue": 4.5, "ev_ebitda": 18.0, "pe": 22.0, "pb": 4.0},
    "Financial Services": {"ev_revenue": 3.0, "ev_ebitda": 12.0, "pe": 15.0, "pb": 1.8},
    "Consumer Discretionary": {"ev_revenue": 2.5, "ev_ebitda": 14.0, "pe": 20.0, "pb": 4.5},
    "Consumer Staples": {"ev_revenue": 2.0, "ev_ebitda": 13.0, "pe": 18.0, "pb": 5.5},
    "Energy": {"ev_revenue": 1.5, "ev_ebitda": 8.0, "pe": 14.0, "pb": 2.0},
    "Industrials": {"ev_revenue": 2.2, "ev_ebitda": 12.0, "pe": 18.0, "pb": 3.5},
    "Materials": {"ev_revenue": 1.8, "ev_ebitda": 9.0, "pe": 16.0, "pb": 2.5},
    "Real Estate": {"ev_revenue": 6.0, "ev_ebitda": 18.0, "pe": 35.0, "pb": 2.2},
    "Utilities": {"ev_revenue": 2.5, "ev_ebitda": 11.0, "pe": 17.0, "pb": 1.9},
    "Communication Services": {"ev_revenue": 5.0, "ev_ebitda": 15.0, "pe": 22.0, "pb": 4.0},
    "Default": {"ev_revenue": 3.0, "ev_ebitda": 14.0, "pe": 20.0, "pb": 3.0},
}


def comps_valuation(
    peers_data: list[dict],
    target_revenue: float,
    target_ebitda: float,
    target_earnings: float,
    target_book_value: float,
    net_debt: float,
    shares_outstanding: float,
) -> dict:
    """
    Comparable company analysis (comps).
    peers_data: list of dicts with keys: symbol, ev, revenue, ebitda, earnings, book_value
    """
    if not peers_data:
        return {}

    ev_revenue_multiples = []
    ev_ebitda_multiples = []
    pe_multiples = []
    pb_multiples = []

    for peer in peers_data:
        ev = peer.get("enterprise_value", 0)
        rev = peer.get("revenue", 0)
        ebitda = peer.get("ebitda", 0)
        earnings = peer.get("net_income", 0)
        book = peer.get("book_value", 0)
        market_cap = peer.get("market_cap", 0)

        if rev and rev > 0 and ev and ev > 0:
            ev_revenue_multiples.append(ev / rev)
        if ebitda and ebitda > 0 and ev and ev > 0:
            ev_ebitda_multiples.append(ev / ebitda)
        if earnings and earnings > 0 and market_cap and market_cap > 0:
            pe_multiples.append(market_cap / earnings)
        if book and book > 0 and market_cap and market_cap > 0:
            pb_multiples.append(market_cap / book)

    def median_safe(lst):
        return float(np.median(lst)) if lst else None

    med_ev_rev = median_safe(ev_revenue_multiples)
    med_ev_ebitda = median_safe(ev_ebitda_multiples)
    med_pe = median_safe(pe_multiples)
    med_pb = median_safe(pb_multiples)

    implied_prices = []
    valuations = {}

    if med_ev_rev and target_revenue > 0:
        ev_implied = med_ev_rev * target_revenue
        eq_implied = ev_implied - net_debt
        price_implied = eq_implied / shares_outstanding if shares_outstanding > 0 else 0
        valuations["EV/Revenue"] = {
            "multiple": round(med_ev_rev, 2),
            "ev": ev_implied,
            "equity_value": eq_implied,
            "implied_price": price_implied,
        }
        implied_prices.append(price_implied)

    if med_ev_ebitda and target_ebitda > 0:
        ev_implied = med_ev_ebitda * target_ebitda
        eq_implied = ev_implied - net_debt
        price_implied = eq_implied / shares_outstanding if shares_outstanding > 0 else 0
        valuations["EV/EBITDA"] = {
            "multiple": round(med_ev_ebitda, 2),
            "ev": ev_implied,
            "equity_value": eq_implied,
            "implied_price": price_implied,
        }
        implied_prices.append(price_implied)

    if med_pe and target_earnings > 0:
        market_cap_implied = med_pe * target_earnings
        price_implied = market_cap_implied / shares_outstanding if shares_outstanding > 0 else 0
        valuations["P/E"] = {
            "multiple": round(med_pe, 2),
            "market_cap": market_cap_implied,
            "implied_price": price_implied,
        }
        implied_prices.append(price_implied)

    if med_pb and target_book_value > 0:
        market_cap_implied = med_pb * target_book_value
        price_implied = market_cap_implied / shares_outstanding if shares_outstanding > 0 else 0
        valuations["P/B"] = {
            "multiple": round(med_pb, 2),
            "market_cap": market_cap_implied,
            "implied_price": price_implied,
        }
        implied_prices.append(price_implied)

    mean_implied = float(np.mean(implied_prices)) if implied_prices else None
    return {
        "valuations": valuations,
        "mean_implied_price": mean_implied,
        "multiples": {
            "ev_revenue": med_ev_rev,
            "ev_ebitda": med_ev_ebitda,
            "pe": med_pe,
            "pb": med_pb,
        },
    }


def sector_comps_valuation(
    sector: str,
    target_revenue: float,
    target_ebitda: float,
    target_earnings: float,
    target_book_value: float,
    net_debt: float,
    shares_outstanding: float,
) -> dict:
    """Use sector median multiples when peer data is unavailable."""
    mults = SECTOR_MULTIPLES.get(sector, SECTOR_MULTIPLES["Default"])
    implied_prices = []
    valuations = {}

    if target_revenue > 0:
        ev = mults["ev_revenue"] * target_revenue
        eq = ev - net_debt
        price = eq / shares_outstanding if shares_outstanding > 0 else 0
        valuations["EV/Revenue"] = {"multiple": mults["ev_revenue"], "implied_price": price}
        implied_prices.append(price)

    if target_ebitda > 0:
        ev = mults["ev_ebitda"] * target_ebitda
        eq = ev - net_debt
        price = eq / shares_outstanding if shares_outstanding > 0 else 0
        valuations["EV/EBITDA"] = {"multiple": mults["ev_ebitda"], "implied_price": price}
        implied_prices.append(price)

    if target_earnings > 0:
        market_cap = mults["pe"] * target_earnings
        price = market_cap / shares_outstanding if shares_outstanding > 0 else 0
        valuations["P/E"] = {"multiple": mults["pe"], "implied_price": price}
        implied_prices.append(price)

    if target_book_value > 0:
        market_cap = mults["pb"] * target_book_value
        price = market_cap / shares_outstanding if shares_outstanding > 0 else 0
        valuations["P/B"] = {"multiple": mults["pb"], "implied_price": price}
        implied_prices.append(price)

    return {
        "valuations": valuations,
        "mean_implied_price": float(np.mean(implied_prices)) if implied_prices else None,
        "sector": sector,
    }


def football_field_chart(valuations: dict) -> pd.DataFrame:
    """
    Build a football field summary DataFrame from multiple valuation methods.
    valuations: {method_name: {"low": x, "mid": y, "high": z}}
    """
    rows = []
    for method, vals in valuations.items():
        rows.append({
            "Method": method,
            "Low": vals.get("low", 0),
            "Mid": vals.get("mid", 0),
            "High": vals.get("high", 0),
        })
    return pd.DataFrame(rows)


def calculate_financial_ratios(info: dict) -> dict:
    """Derive and organize key financial ratios from company info dict."""
    ratios = {}

    # Valuation ratios
    ratios["P/E Ratio"] = info.get("pe_ratio")
    ratios["Forward P/E"] = info.get("forward_pe")
    ratios["P/B Ratio"] = info.get("pb_ratio")
    ratios["EV/EBITDA"] = (
        info.get("enterprise_value") / info.get("ebitda")
        if info.get("enterprise_value") and info.get("ebitda")
        else None
    )
    ratios["EV/Revenue"] = (
        info.get("enterprise_value") / info.get("revenue")
        if info.get("enterprise_value") and info.get("revenue")
        else None
    )

    # Profitability
    ratios["Gross Margin"] = info.get("gross_margin")
    ratios["Operating Margin"] = info.get("operating_margin")
    ratios["Net Margin"] = info.get("profit_margin")
    ratios["ROE"] = info.get("roe")
    ratios["ROA"] = info.get("roa")

    # Leverage
    ratios["Debt/Equity"] = info.get("debt_to_equity")
    ratios["Current Ratio"] = info.get("current_ratio")

    # Growth
    ratios["Revenue Growth"] = info.get("revenue_growth")
    ratios["Earnings Growth"] = info.get("earnings_growth")

    # Dividend
    ratios["Dividend Yield"] = info.get("dividend_yield")
    ratios["Beta"] = info.get("beta")

    return {k: v for k, v in ratios.items() if v is not None}
