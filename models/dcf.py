"""Discounted Cash Flow (DCF) valuation model."""
import numpy as np
import pandas as pd
from typing import Optional


def calculate_wacc(
    equity_value: float,
    debt_value: float,
    cost_of_equity: float,
    cost_of_debt: float,
    tax_rate: float = 0.21,
) -> float:
    """Calculate Weighted Average Cost of Capital."""
    total = equity_value + debt_value
    if total == 0:
        return cost_of_equity
    weight_equity = equity_value / total
    weight_debt = debt_value / total
    return weight_equity * cost_of_equity + weight_debt * cost_of_debt * (1 - tax_rate)


def capm_cost_of_equity(
    beta: float,
    risk_free_rate: float = 0.045,
    equity_risk_premium: float = 0.055,
) -> float:
    """Calculate cost of equity using CAPM: rf + beta * (rm - rf)."""
    return risk_free_rate + beta * equity_risk_premium


def project_fcf(
    base_fcf: float,
    growth_rates: list[float],
) -> list[float]:
    """Project Free Cash Flows for N years given growth rates."""
    fcfs = []
    current = base_fcf
    for rate in growth_rates:
        current = current * (1 + rate)
        fcfs.append(current)
    return fcfs


def terminal_value(
    final_fcf: float,
    wacc: float,
    terminal_growth_rate: float = 0.025,
) -> float:
    """Gordon Growth Model terminal value."""
    if wacc <= terminal_growth_rate:
        raise ValueError("WACC must exceed terminal growth rate")
    return final_fcf * (1 + terminal_growth_rate) / (wacc - terminal_growth_rate)


def dcf_valuation(
    base_fcf: float,
    growth_rates: list[float],
    wacc: float,
    terminal_growth_rate: float = 0.025,
    net_debt: float = 0.0,
    shares_outstanding: float = 1.0,
) -> dict:
    """
    Full DCF valuation.
    Returns enterprise value, equity value, implied share price, and yearly details.
    """
    projected_fcfs = project_fcf(base_fcf, growth_rates)
    n_years = len(growth_rates)

    # Discount each FCF
    pv_fcfs = []
    for i, fcf in enumerate(projected_fcfs):
        pv = fcf / (1 + wacc) ** (i + 1)
        pv_fcfs.append(pv)

    sum_pv_fcfs = sum(pv_fcfs)

    # Terminal value
    tv = terminal_value(projected_fcfs[-1], wacc, terminal_growth_rate)
    pv_tv = tv / (1 + wacc) ** n_years

    enterprise_value = sum_pv_fcfs + pv_tv
    equity_value = enterprise_value - net_debt
    implied_price = equity_value / shares_outstanding if shares_outstanding > 0 else 0

    yearly_details = pd.DataFrame({
        "Year": [f"Y{i+1}" for i in range(n_years)],
        "FCF": projected_fcfs,
        "PV of FCF": pv_fcfs,
        "Discount Factor": [1 / (1 + wacc) ** (i + 1) for i in range(n_years)],
    })

    return {
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "implied_price": implied_price,
        "sum_pv_fcfs": sum_pv_fcfs,
        "terminal_value": tv,
        "pv_terminal_value": pv_tv,
        "tv_pct_of_ev": pv_tv / enterprise_value if enterprise_value else 0,
        "yearly_details": yearly_details,
        "wacc": wacc,
        "terminal_growth_rate": terminal_growth_rate,
    }


def sensitivity_analysis(
    base_fcf: float,
    growth_rates: list[float],
    net_debt: float,
    shares_outstanding: float,
    wacc_range: list[float],
    tgr_range: list[float],
) -> pd.DataFrame:
    """
    Sensitivity table: implied share price for WACC x Terminal Growth Rate combinations.
    """
    rows = []
    for wacc in wacc_range:
        row = {}
        for tgr in tgr_range:
            if wacc <= tgr:
                row[f"{tgr:.1%}"] = None
            else:
                result = dcf_valuation(
                    base_fcf, growth_rates, wacc, tgr, net_debt, shares_outstanding
                )
                row[f"{tgr:.1%}"] = round(result["implied_price"], 2)
        rows.append(row)

    df = pd.DataFrame(rows, index=[f"{w:.1%}" for w in wacc_range])
    df.index.name = "WACC \\ TGR"
    return df


def monte_carlo_dcf(
    base_fcf: float,
    mean_growth: float,
    std_growth: float,
    n_years: int,
    wacc: float,
    terminal_growth_rate: float,
    net_debt: float,
    shares_outstanding: float,
    n_simulations: int = 5000,
) -> dict:
    """
    Monte Carlo simulation of DCF valuation.
    Returns distribution statistics for implied share price.
    """
    np.random.seed(42)
    prices = []
    for _ in range(n_simulations):
        growth_rates = list(np.random.normal(mean_growth, std_growth, n_years))
        # Clamp growth rates to avoid extreme values
        growth_rates = [max(-0.5, min(g, 1.0)) for g in growth_rates]
        try:
            result = dcf_valuation(
                base_fcf, growth_rates, wacc, terminal_growth_rate,
                net_debt, shares_outstanding
            )
            prices.append(result["implied_price"])
        except ValueError:
            continue

    prices = np.array(prices)
    return {
        "mean": np.mean(prices),
        "median": np.median(prices),
        "std": np.std(prices),
        "p5": np.percentile(prices, 5),
        "p25": np.percentile(prices, 25),
        "p75": np.percentile(prices, 75),
        "p95": np.percentile(prices, 95),
        "prices": prices,
    }
