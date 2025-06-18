#!/usr/bin/env python
"""
ratio_library.py
────────────────
Defines
  • ratio_funcs            – base (~400) ratio callables
  • derived_ratio_funcs    – (optional) ratios that reference earlier ratios
  • _ensure_core_columns   – one-shot helper that fabricates prerequisite
                             fields such as NetAssets, “days” metrics, etc.

The YAML mapping 〈ratio_domain_stage_map.yaml〉 is *not* loaded here; it
is consumed directly by pipeline stages that need it.
"""
from __future__ import annotations

from pathlib import Path
from typing  import Callable, Dict, List

import numpy as np
import pandas as pd
from  scipy.stats import skew as _scipy_skew   # noqa: N812  (keep scipy alias)

# ─────────────────────────────── helpers ──────────────────────────────
def safe_div(a, b):
    """Vectorised division → NaN where the denominator is 0 / invalid."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    if isinstance(out, pd.Series):
        return out.replace([np.inf, -np.inf], np.nan)
    return np.where(np.isfinite(out), out, np.nan)


def slope(series: pd.Series, window: int) -> float:
    """OLS slope of the most-recent *window* observations (NaN if < 2)."""
    y = series.dropna().tail(window)
    if len(y) < 2:
        return np.nan
    x = np.arange(len(y), dtype=float)
    try:
        m, _ = np.polyfit(x, y.to_numpy(float), 1)
        return m
    except Exception:            # numerical failure
        return np.nan


def skew(series: pd.Series, window: int) -> pd.Series:
    """Rolling, bias-corrected skew."""
    return series.rolling(window, min_periods=3).apply(
        lambda x: _scipy_skew(x, bias=False) if x.notna().sum() >= 3 else np.nan,
        raw=False,
    )


def winsor(s: pd.Series, pct: float = 0.01) -> pd.Series:
    """Two-sided winsorisation at *pct* and 1-*pct* quantiles."""
    if not 0 < pct < 0.5:
        raise ValueError("pct must be between 0 and 0.5")
    lo, hi = s.quantile(pct), s.quantile(1 - pct)
    return s.clip(lo, hi)


def _to_series(x, index):
    """Return *x* as float64 Series aligned to *index*."""
    if isinstance(x, pd.Series):
        return x.reindex(index).astype("float64")

    if isinstance(x, pd.DataFrame):
        nums = [c for c in x if pd.api.types.is_numeric_dtype(x[c])]
        if not nums:
            raise ValueError("DataFrame result has no numeric columns")
        first = x[nums[0]]
        if len(nums) == 1 or all(first.equals(x[c]) for c in nums[1:]):
            return first.reindex(index).astype("float64")
        return x[nums].mean(axis=1).reindex(index).astype("float64")

    if np.isscalar(x):
        return pd.Series(float(x), index=index, dtype="float64")

    raise ValueError(f"unsupported output type: {type(x)}")


# ────────────────────── pre-derived “core” columns ────────────────────
def _ensure_core_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create / fill convenience columns that **many** ratios rely on.
    Call it once in Stage-03 right after loading the raw financials.
    """
    # –– Balance-sheet base ––––––––––––––––––––––––––––––––––––––––
    if "NetAssets" not in df:
        df["NetAssets"] = df["TotalAssets"] - df["TotalLiabilitiesAsReported"]

    # –– Efficiency “days” metrics ––––––––––––––––––––––––––––––––
    day_pairs = {
        "DaysSalesOutstanding":     ("AccountsReceivable", "TotalRevenue"),
        "DaysInventoryOutstanding": ("Inventory",          "CostOfRevenue"),
        "DaysPayablesOutstanding":  ("AccountsPayable",    "CostOfRevenue"),
    }
    for new, (num, den) in day_pairs.items():
        if new not in df:
            df[new] = 365 * safe_div(df[num], df[den])

    if "OperatingCycle" not in df:
        df["OperatingCycle"] = (
            df["DaysSalesOutstanding"]
            + df["DaysInventoryOutstanding"]
            - df["DaysPayablesOutstanding"]
        )

    # –– Alternate variants ––––––––––––––––––––––––––––––––––––––––
    ratio_pairs = {
        "AccountsReceivableDays": ("AccountsReceivable", "TotalRevenue"),
        "InventoryDays":          ("Inventory",          "CostOfRevenue"),
        "AccountsPayableDays":    ("AccountsPayable",    "CostOfRevenue"),
    }
    for new, (num, den) in ratio_pairs.items():
        if new not in df:
            df[new] = 365 * safe_div(df[num], df[den])

    # –– Misc defaults ––––––––––––––––––––––––––––––––––––––––––––
    if "CommonStockRepurchased" not in df:
        df["CommonStockRepurchased"] = df.get("CommonStockPayments", 0.0)

    if "TotalDeposits" not in df:
        df["TotalDeposits"] = 0.0

    return df


# ───────────────────────── ratio definitions ──────────────────────────
# (Only a **tiny** excerpt shown here.  Keep your full ~400-entry dict.)
ratio_funcs: Dict[str, Callable] = {

     #### PREPARE STAGES
    # ───────────────────────────────────── PHYSICAL · PREPARE
    "Cash_to_Total_Assets":  lambda C: safe_div(C["Cash"].fillna(0), C["TotalAssets"]),

    "Cash_to_Total_Assets_Alt":
        lambda C: safe_div(C["CashAndCashEquivalents"].fillna(0), C["TotalAssets"]),

    "Current_Ratio":         lambda C: safe_div(C["CurrentAssets"], C["CurrentLiabilities"]),

    "Quick_Ratio":           lambda C: safe_div(C["CurrentAssets"] - C["Inventory"],
                                                C["CurrentLiabilities"]),

    "Cash_Ratio":            lambda C: safe_div(C["CashAndCashEquivalents"],
                                                C["CurrentLiabilities"]),

    "Operating_Cash_Flow_Ratio":            # rolling mean done inside group-apply
        lambda C: safe_div(
            C["OperatingCashFlow"],
            C.groupby("Symbol").apply(
                lambda g: g["CurrentLiabilities"].rolling(2).mean()
            ).reset_index(level=0, drop=True)
        ),

    "NetWorkingCapital_to_Assets":
        lambda C: safe_div(C["CurrentAssets"] - C["CurrentLiabilities"], C["TotalAssets"]),

    "Cash_Conversion_Cycle":
        lambda C: C["AccountsReceivableDays"] + C["InventoryDays"] - C["AccountsPayableDays"],

    "Operating_CF_to_Debt":                  # rolling mean inside group-apply
        lambda C: safe_div(
            C["OperatingCashFlow"],
            C.groupby("Symbol").apply(
                lambda g: g["TotalDebt"].rolling(2).mean()
            ).reset_index(level=0, drop=True)
        ),

    "NetDebt_to_OCF":
        lambda C: safe_div(C["TotalDebt"] - C["CashAndCashEquivalents"], C["OperatingCashFlow"]),

    "DaysPayablesOutstanding":
        lambda C: safe_div(C["AccountsPayable"], C["CostOfRevenue"] / 365),

    "WorkingCapital_to_Sales":
        lambda C: safe_div(C["CurrentAssets"] - C["CurrentLiabilities"], C["TotalRevenue"]),

    "CashEquivalents_to_CurrentLiab":
        lambda C: safe_div(C["CashAndCashEquivalents"], C["CurrentLiabilities"]),

    "AccountsReceivable_Turnover":
        lambda C: safe_div(
            C["TotalRevenue"],
            C.groupby("Symbol").apply(
                lambda g: g["AccountsReceivable"].rolling(2).mean()
            ).reset_index(level=0, drop=True)
        ),

    "Inventory_Turnover":
        lambda C: safe_div(
            C["CostOfRevenue"],
            C.groupby("Symbol").apply(
                lambda g: g["Inventory"].rolling(2).mean()
            ).reset_index(level=0, drop=True)
        ),

    "Net_Operating_WC_to_Assets":
        lambda C: safe_div(C["AccountsReceivable"] + C["Inventory"] - C["AccountsPayable"],
                           C["TotalAssets"]),

    "CashFlow_to_Debt":
        lambda C: safe_div(
            C["OperatingCashFlow"],
            C.groupby("Symbol").apply(
                lambda g: g["TotalDebt"].rolling(2).mean()
            ).reset_index(level=0, drop=True)
        ),

    "CashFlow_to_Assets":    lambda C: safe_div(C["OperatingCashFlow"], C["TotalAssets"]),
    "FCF_to_Assets":         lambda C: safe_div(C["FreeCashFlow"], C["TotalAssets"]),

    "CashFlow_Coverage_of_Interest":
        lambda C: safe_div(C["OperatingCashFlow"], C["InterestExpense"]),

    "OCF_to_NetIncome":      lambda C: safe_div(C["OperatingCashFlow"], C["NetIncome"]),

    "DaysSalesOutstanding":
        lambda C: safe_div(C["AccountsReceivable"], safe_div(C["TotalRevenue"], 365)),

    "DaysInventoryOutstanding":
        lambda C: safe_div(C["Inventory"], safe_div(C["CostOfRevenue"], 365)),

    "OperatingCycle":
        lambda C: safe_div(C["AccountsReceivable"], safe_div(C["TotalRevenue"], 365)) + \
                  safe_div(C["Inventory"], safe_div(C["CostOfRevenue"], 365)),

    "BeginningCash_to_Sales":
        lambda C: safe_div(C["BeginningCashPosition"], C["TotalRevenue"]),

    "RetainedEarnings_to_Assets":
        lambda C: safe_div(C["RetainedEarnings"], C["TotalAssets"]),

    "Deposits_to_Assets":    lambda C: safe_div(C["TotalDeposits"], C["TotalAssets"]),

    "Cash_Burn_Duration":
        lambda C: safe_div(C["CashAndCashEquivalents"], safe_div(C["OperatingExpense"], 365)),

    # (WorkingCapital_Days_Trend moved to derived_ratio_funcs)

    "Cash_Return_on_Assets":
        lambda C: safe_div(C["OperatingCashFlow"], C["TotalAssets"]),

    # (CashConversion_Ratio_3yrAvg now in derived_ratio_funcs)

    "FCF_Margin_3yrAvg":
        lambda C: C.groupby("Symbol").apply(
            lambda g: safe_div(g["FreeCashFlow"].rolling(3, 1).mean(),
                               g["TotalRevenue"].rolling(3, 1).mean())
        ).reset_index(level=0, drop=True),

    "FCF_Yield_on_Assets":
        lambda C: safe_div(C["FreeCashFlow"], C["TotalAssets"]),

    "Operating_Efficiency_Ratio":
        lambda C: safe_div(C["OperatingExpense"], C["TotalRevenue"]),

    "WorkingCapital_Turnover":
        lambda C: safe_div(C["TotalRevenue"], C["CurrentAssets"] - C["CurrentLiabilities"]),

    "Liquidity_Runway_Days":
        lambda C: safe_div(C["CashAndCashEquivalents"], C["OperatingExpense"] / 365),

    # ─────────────────────────────────── INFORMATION · PREPARE
    "Accrual_Ratio":
        lambda C: safe_div(
            C["NetIncome"] - C["OperatingCashFlow"],
            C.groupby("Symbol").apply(lambda g: g["TotalAssets"].rolling(2).mean())
             .reset_index(level=0, drop=True)
        ),

    "Cash_Earnings_Ratio":   lambda C: safe_div(C["OperatingCashFlow"], C["NetIncome"]),

    "Net_Operating_Accruals":
        lambda C: safe_div(C["NetIncome"] - C["OperatingCashFlow"] - C["Depreciation"],
                           C["TotalAssets"]),

    "Dechow_Dichev_AQ":
        lambda C: safe_div(ratio_funcs["Accrual_Ratio"](C), np.abs(C["OperatingCashFlow"])),

    "Percent_Accruals":
        lambda C: safe_div(C["NetIncome"] - C["OperatingCashFlow"], C["TotalRevenue"]),

    "Revenue_Quality":
        lambda C: safe_div(C["AccountsReceivable"].diff(), C["TotalRevenue"].diff()),

    "Revenue_Quality_Delta_AR":
        lambda C: safe_div(C["AccountsReceivable"].diff(), C["TotalRevenue"].diff()),

    "WC_Accrual_Ratio":
        lambda C: safe_div(
            C["AccountsReceivable"].diff() + C["Inventory"].diff() - C["AccountsPayable"].diff(),
            C["TotalRevenue"]
        ),

    # (DSO_Trend_3yr & Inventory_Inflation_3yr moved to derived_ratio_funcs)

    "ROA_3yrAvg":
        lambda C: safe_div(
            C.groupby("Symbol")["NetIncome"].transform(lambda x: x.rolling(3, 1).mean()),
            C.groupby("Symbol")["TotalAssets"].transform(lambda x: x.rolling(3, 1).mean())
        ),

    "ROE_3yrAvg":
        lambda C: safe_div(
            C.groupby("Symbol")["NetIncome"].transform(lambda x: x.rolling(3, 1).mean()),
            C.groupby("Symbol")["TotalEquity"].transform(lambda x: x.rolling(3, 1).mean())
        ),

    "GrossMargin_3yrAvg":
        lambda C: safe_div(
            C.groupby("Symbol")["GrossProfit"].transform(lambda x: x.rolling(3, 1).mean()),
            C.groupby("Symbol")["TotalRevenue"].transform(lambda x: x.rolling(3, 1).mean())
        ),

    "EBITDA_Margin_3yrAvg":
        lambda C: safe_div(
            C.groupby("Symbol")["EBITDA"].transform(lambda x: x.rolling(3, 1).mean()),
            C.groupby("Symbol")["TotalRevenue"].transform(lambda x: x.rolling(3, 1).mean())
        ),

    "OCF_Margin_3yrAvg":
        lambda C: C.groupby("Symbol").apply(
            lambda g: safe_div(g["OperatingCashFlow"].rolling(3, 1).mean(),
                               g["TotalRevenue"].rolling(3, 1).mean())
        ).reset_index(level=0, drop=True),

    "Accruals_to_Sales":
        lambda C: safe_div(C["NetIncome"] - C["OperatingCashFlow"], C["TotalRevenue"]),

    # ─────────────────────────────── COGNITIVE · PREPARE
    "R_D_Growth_Rate":  lambda C: C.groupby("Symbol")["ResearchAndDevelopment"].pct_change(),
    "R_D_to_Opex":      lambda C: safe_div(C["ResearchAndDevelopment"], C["OperatingExpense"]),
    "RnD_Plus_CapEx_Intensity":
        lambda C: safe_div(C["ResearchAndDevelopment"] + C["CapitalExpenditure"],
                           C["TotalRevenue"]),
    "R&D_Intensity":    lambda C: safe_div(C["ResearchAndDevelopment"], C["TotalRevenue"]),
    "R_D_3yr_CAGR":     lambda C: C.groupby("Symbol")["ResearchAndDevelopment"].pct_change(3),
    "Innovation_Ratio": lambda C: safe_div(C["ResearchAndDevelopment"], C["GrossProfit"]),
    "R_D_Growth":       lambda C: C.groupby("Symbol")["ResearchAndDevelopment"].pct_change(),

    # ─────────────────────────────── SOCIAL · PREPARE
    "Interest_Coverage_Ratio":
        lambda C: safe_div(C["EarningBeforeInterestAndTax"], C["InterestExpense"]),

    "Cash_Interest_Coverage_Ratio":
        lambda C: safe_div(C["OperatingCashFlow"], C["InterestExpense"]),

    "EBITDA_Interest_Coverage":
        lambda C: safe_div(C["EBITDA"], C["InterestExpense"]),

    "DSCR":
        lambda C: safe_div(C["EBITDA"] - C["CapitalExpenditure"],
                           C["InterestExpense"] + C.get("RepaymentOfDebt", 0)),

    "DSCR_Alt":
        lambda C: safe_div(C["OperatingCashFlow"],
                           C["InterestExpense"] + C.get("RepaymentOfDebt", 0)),

    "Debt_to_Assets":  lambda C: safe_div(C["TotalDebt"], C["TotalAssets"]),
    "EquityRatio":     lambda C: safe_div(C["TotalEquity"], C["TotalAssets"]),

    "LongTerm_Debt_to_Equity":
        lambda C: safe_div(C["LongTermDebt"], C["TotalEquity"]),

    "Net_Debt_to_Equity":
        lambda C: safe_div(C["TotalDebt"] - C["Cash"], C["TotalEquity"]),

    "Financial_Leverage":
        lambda C: safe_div(C["TotalAssets"], C["TotalEquity"]),

    "Times_Interest_Earned":
        lambda C: safe_div(C["EarningBeforeInterestAndTax"], C["InterestExpense"]),

    "Debt_Maturity_Split":
        lambda C: safe_div(C["LongTermDebt"], C["TotalDebt"]),

    "InterestCoverage_Cushion":
        lambda C: safe_div(C["EBITDA"], C["InterestExpense"].replace(0, np.nan)),

    "Net_Leverage_Trend_3yr":
        lambda C: slope(((C["TotalDebt"] - C["CashAndCashEquivalents"]) / C["EBITDA"]), 3),

    "Debt_to_Capital":
        lambda C: safe_div(C["TotalDebt"], C["TotalDebt"] + C["TotalEquity"]),

    "Short_Term_Debt_Ratio":
        lambda C: safe_div(C.get("CurrentDebt",
                                 C["TotalDebt"] - C["LongTermDebt"]),
                           C["TotalDebt"]),

    "Interest_Burden_Ratio":
        lambda C: safe_div(C["InterestExpense"], C["EarningBeforeInterestAndTax"]),


    #### ABSORB STAGES
    # ───────────────────────────────────── PHYSICAL · ABSORB
    "Asset_Turnover_Ratio":
        lambda C: safe_div(
            C["TotalRevenue"],
            C.groupby("Symbol")["TotalAssets"].rolling(2).mean().reset_index(level=0, drop=True)
        ),

    "Gross_Profit_Margin":  lambda C: safe_div(C["GrossProfit"], C["TotalRevenue"]),

    "ROA":
        lambda C: safe_div(
            C["NetIncome"],
            C.groupby("Symbol")["TotalAssets"].rolling(2).mean().reset_index(level=0, drop=True)
        ),

    "Operating_Margin":       lambda C: safe_div(C["OperatingIncome"], C["TotalRevenue"]),
    "EBITDA_Margin":          lambda C: safe_div(C["EBITDA"], C["TotalRevenue"]),
    "Net_Income_Margin":      lambda C: safe_div(C["NetIncome"], C["TotalRevenue"]),
    "OperatingIncome_Margin": lambda C: safe_div(C["OperatingIncome"], C["TotalRevenue"]),
    "Cost_of_Revenue_Ratio":  lambda C: safe_div(C["CostOfRevenue"], C["TotalRevenue"]),
    "Operating_Expense_Ratio":lambda C: safe_div(C["OperatingExpense"], C["TotalRevenue"]),
    "SG_A_to_Revenue":        lambda C: safe_div(C["SellingGeneralAndAdministration"], C["TotalRevenue"]),
    "EBT_Margin":             lambda C: safe_div(C["PretaxIncome"], C["TotalRevenue"]),

    # avoids Series.groupby inside rolling calc
    "ROCE":
        lambda C: safe_div(
            C["EarningBeforeInterestAndTax"],
            C.groupby("Symbol")
             .apply(lambda g: (g["TotalAssets"] - g["CurrentLiabilities"]).rolling(2).mean())
             .reset_index(level=0, drop=True)
        ),

    "GrossMargin_Stability":
        lambda C: C.groupby("Symbol").apply(
            lambda g: safe_div(g["GrossProfit"], g["TotalRevenue"]).rolling(5, 3).std()
        ).reset_index(level=0, drop=True),

    "Fixed_Asset_Turnover":
        lambda C: safe_div(
            C["TotalRevenue"],
            C.groupby("Symbol")
             .apply(lambda g: (g["TotalAssets"] - g["CurrentAssets"]).rolling(2).mean())
             .reset_index(level=0, drop=True)
        ),

    "Inventory_to_Assets":    lambda C: safe_div(C["Inventory"], C["TotalAssets"]),
    "Receivables_to_Assets":  lambda C: safe_div(C["AccountsReceivable"], C["TotalAssets"]),
    "Payables_to_Assets":     lambda C: safe_div(C["AccountsPayable"], C["TotalAssets"]),
    "Operating_Return_on_Opex": lambda C: safe_div(C["OperatingIncome"], C["OperatingExpense"]),
    "CashFlow_Margin":          lambda C: safe_div(C["OperatingCashFlow"], C["TotalRevenue"]),
    "CashConversionEfficiency": lambda C: safe_div(C["OperatingCashFlow"], C["TotalRevenue"]),
    "Operating_Leverage":
        lambda C: safe_div(C["OperatingIncome"].pct_change(5),
                           C["TotalRevenue"].pct_change(5)),

    "InvestedCapital_Turnover":
        lambda C: safe_div(
            C["TotalRevenue"],
            C.groupby("Symbol")["InvestedCapital"].rolling(2).mean().reset_index(level=0, drop=True)
        ),

    "FCF_Margin":      lambda C: safe_div(C["FreeCashFlow"], C["TotalRevenue"]),
    "Asset_Age_Ratio": lambda C: safe_div(C["AccumulatedDepreciation"],
                                          C["TotalAssets"] - C["CurrentAssets"]),

    # ─────────────────────────────────── INFORMATION · ABSORB
    "Earnings_Volatility":
        lambda C: C.groupby("Symbol")["NetIncome"].transform(
            lambda x: safe_div(x.rolling(5, 2).std(), x.rolling(5, 2).mean())
        ),
    "EBITDA_Volatility":
        lambda C: C.groupby("Symbol")["EBITDA"].transform(
            lambda x: safe_div(x.rolling(5, 2).std(), x.rolling(5, 2).mean())
        ),
    "Earnings_Volatility_AbsMean":
        lambda C: C.groupby("Symbol")["NetIncome"].transform(
            lambda x: safe_div(
                x.rolling(5, 2).std(),
                x.rolling(5, 2).apply(lambda y: np.abs(y.mean()))
            )
        ),
    "EBITDA_CV":
        lambda C: C.groupby("Symbol")["EBITDA"].transform(
            lambda x: safe_div(x.rolling(5, 2).std(), x.rolling(5, 2).mean())
        ),
    "NetIncome_StDev_3yr":  lambda C: C.groupby("Symbol")["NetIncome"].transform(lambda x: x.rolling(3, 2).std()),
    "EBITDA_StDev_3yr":     lambda C: C.groupby("Symbol")["EBITDA"].transform(lambda x: x.rolling(3, 2).std()),
    "Revenue_StDev_3yr":    lambda C: C.groupby("Symbol")["TotalRevenue"].transform(lambda x: x.rolling(3, 2).std()),
    "OCF_StDev_3yr":        lambda C: C.groupby("Symbol")["OperatingCashFlow"].transform(lambda x: x.rolling(3, 2).std()),

    "ROA_StDev_5yr":
        lambda C: C.groupby("Symbol").apply(
            lambda g: safe_div(g["NetIncome"], g["TotalAssets"]).rolling(5, 2).std()
        ).reset_index(level=0, drop=True),

    "EBITDA_Volatility_5yr":
        lambda C: C.groupby("Symbol")["EBITDA"].transform(
            lambda x: safe_div(x.rolling(5, 2).std(), x.rolling(5, 2).mean())
        ),
    "Earnings_Volatility_5yr":
        lambda C: C.groupby("Symbol")["NetIncome"].transform(
            lambda x: safe_div(x.rolling(5, 2).std(), x.rolling(5, 2).mean())
        ),
    "OperatingCF_Volatility_5yr":
        lambda C: C.groupby("Symbol")["OperatingCashFlow"].transform(
            lambda x: safe_div(x.rolling(5, 2).std(), x.rolling(5, 2).mean())
        ),
    "Sales_Volatility_5yr":
        lambda C: C.groupby("Symbol")["TotalRevenue"].transform(
            lambda x: safe_div(x.rolling(5, 2).std(), x.rolling(5, 2).mean())
        ),
    "FCF_Volatility_3yr":
        lambda C: C.groupby("Symbol")["FreeCashFlow"].transform(lambda x: x.rolling(3, 2).std()),

    "GrossMargin_Volatility_5yr":
        lambda C: C.groupby("Symbol").apply(
            lambda g: safe_div(
                (g["GrossProfit"] / g["TotalRevenue"]).rolling(5, 2).std(),
                (g["GrossProfit"] / g["TotalRevenue"]).rolling(5, 2).mean()
            )
        ).reset_index(level=0, drop=True),

    "GrossMargin_Volatility_3yr":
        lambda C: C.groupby("Symbol").apply(
            lambda g: safe_div(
                (g["GrossProfit"] / g["TotalRevenue"]).rolling(3, 2).std(),
                (g["GrossProfit"] / g["TotalRevenue"]).rolling(3, 2).mean()
            )
        ).reset_index(level=0, drop=True),

    # ───────────────────────────────────── COGNITIVE · ABSORB
    "Gross_Profitability_Alt": lambda C: safe_div(C["GrossProfit"], C["NetAssets"]),
    "GrossProfit_to_Equity":   lambda C: safe_div(C["GrossProfit"], C["TotalEquity"]),
    "Price_Premium_Index":
        lambda C: safe_div(
            ratio_funcs["Gross_Profit_Margin"](C),
            safe_div(C["GrossProfit"], C["TotalRevenue"])
              .groupby(C["SectorName"]).transform("median")
        ),
    "Price_Cost_PassThrough":
        lambda C: C.groupby("Symbol").apply(
            lambda g: (g["GrossProfit"] / g["TotalRevenue"]).pct_change() -
                      g["CostOfRevenue"].pct_change()
        ).reset_index(level=0, drop=True),

    "R&D_Payoff_Ratio":
        lambda C: safe_div(C["GrossProfit"].diff(), C["ResearchAndDevelopment"].shift(1)),

    # ─────────────────────────────────────── SOCIAL · ABSORB
    "LT_Debt_to_EBITDA":             lambda C: safe_div(C["LongTermDebt"], C["EBITDA"]),
    "InterestBearingDebt_to_EBITDA": lambda C: safe_div(C["TotalDebt"], C["EBITDA"]),
    "Net_Debt_to_EBITDA":            lambda C: safe_div(C["TotalDebt"] - C["CashAndCashEquivalents"], C["EBITDA"]),
    "LT_Debt_to_TotalDebt":          lambda C: safe_div(C["LongTermDebt"], C["TotalDebt"]),

    "ETR_Volatility_3yr":
        lambda C: C.groupby("Symbol")["EffectiveTaxRateAsReported"].transform(
            lambda x: safe_div(x.rolling(3, 2).std(), x.rolling(5, 2).mean())
        ),

    "InterestExpense_to_Sales": lambda C: safe_div(C["InterestExpense"], C["TotalRevenue"]),

    "InterestCoverage_Volatility_5yr":
        lambda C: C.groupby("Symbol").apply(
            lambda g: safe_div(
                (g["EarningBeforeInterestAndTax"] / g["InterestExpense"]).rolling(5, 2).std(),
                (g["EarningBeforeInterestAndTax"] / g["InterestExpense"]).rolling(5, 2).mean()
            )
        ).reset_index(level=0, drop=True),

    "InterestCoverage_Volatility_3yr":
        lambda C: C.groupby("Symbol").apply(
            lambda g: safe_div(
                (g["EarningBeforeInterestAndTax"] / g["InterestExpense"]).rolling(3, 2).std(),
                (g["EarningBeforeInterestAndTax"] / g["InterestExpense"]).rolling(3, 2).mean()
            )
        ).reset_index(level=0, drop=True),

    "Implied_Credit_Spread":
        lambda C: safe_div(
            C["InterestExpense"],
            C.groupby("Symbol")["TotalDebt"].rolling(2).mean().reset_index(level=0, drop=True)
        ),

    "OpLev_Risk":
        lambda C: C.groupby("Symbol").apply(
            lambda g: (g["EarningBeforeInterestAndTax"] / g["TotalRevenue"]).rolling(5).std()
        ).reset_index(level=0, drop=True) * (C["TotalDebt"] - C["CashAndCashEquivalents"]),

    "Interest_Burden_Absorb": lambda C: safe_div(C["InterestExpense"], C["EarningBeforeInterestAndTax"]),

    "Net_Debt_Increase_Rate":
        lambda C: safe_div(
            (C["TotalDebt"] - C["CashAndCashEquivalents"]).diff(),
            (C["TotalDebt"] - C["CashAndCashEquivalents"]).shift(1)
        ),

    "Net_Debt_to_EBITDA_Trend_3yr":
        lambda C: C.groupby("Symbol").apply(
            lambda g: slope((g["TotalDebt"] - g["CashAndCashEquivalents"]) / g["EBITDA"], 3)
        ).reset_index(level=0, drop=True),

    "Net_Debt_to_EBITDA_Trend_5yr":
        lambda C: C.groupby("Symbol").apply(
            lambda g: slope((g["TotalDebt"] - g["CashAndCashEquivalents"]) / g["EBITDA"], 5)
        ).reset_index(level=0, drop=True),

    "Interest_to_OCF":    lambda C: safe_div(C["InterestExpense"], C["OperatingCashFlow"]),
    "Interest_to_EBITDA": lambda C: safe_div(C["InterestExpense"], C["EBITDA"]),
    "Interest_to_EBIT":   lambda C: safe_div(C["InterestExpense"], C["EarningBeforeInterestAndTax"]),

    "Dividend_Suspension_Flag":
        lambda C: ((C["CashDividendsPaid"].shift(1) > 0) &
                   (C["CashDividendsPaid"] == 0)).astype(int),

  

    #### RECOVER STAGES
    # ───────────────────────────────────── PHYSICAL · RECOVER
    "ROE":
        lambda C: safe_div(
            C["NetIncome"],
            C.groupby("Symbol")["TotalEquity"].rolling(2).mean().reset_index(level=0, drop=True)
        ),

    "Debt_to_Equity":      lambda C: safe_div(C["TotalDebt"], C["TotalEquity"]),
    "Debt_to_Equity_Liab": lambda C: safe_div(C["TotalLiabilitiesAsReported"], C["TotalEquity"]),
    "EPS":                 lambda C: safe_div(C["NetIncome"], C["BasicAverageShares"]),

    "Cash_Dividends_to_Net_Income":
        lambda C: safe_div(C["CashDividendsPaid"], C["NetIncome"]),

    "Return_on_Tangible_Equity":
        lambda C: safe_div(
            C["NetIncome"],
            C.groupby("Symbol")
             .apply(lambda g: (g["TotalEquity"] - g["Goodwill"]).rolling(2).mean())
             .reset_index(level=0, drop=True)
        ),

    "Total_Payout_Ratio":
        lambda C: safe_div(C["CashDividendsPaid"] + C["CommonStockRepurchased"],
                           C["NetIncome"]),

    "NetDebt_PayDown_Rate":
        lambda C: safe_div(-(C["TotalDebt"] - C["CashAndCashEquivalents"]).diff(), C["TotalDebt"]),

    "Incremental_ROIC":
        lambda C: safe_div(
            C["NetIncome"].diff() + C["InterestExpense"].diff() * (1 - 0.30),
            (C["TotalDebt"] + C["TotalEquity"] - C["Cash"]).diff()
        ),

    "EBITDA_DropThrough":
        lambda C: safe_div(C["EBITDA"].diff(), C["TotalRevenue"].diff()),

    "OperatingLeverage_Slope":
        lambda C: C.groupby("Symbol").apply(
            lambda g: slope(
                safe_div(g["OperatingIncome"].pct_change(), g["TotalRevenue"].pct_change()), 5
            )
        ).reset_index(level=0, drop=True),

    "Revenue_Recovery_Rate":
        lambda C: safe_div(C.groupby("Symbol")["TotalRevenue"].shift(-1), C["TotalRevenue"]),

    "Retention_to_Growth":
        lambda C: safe_div(C["RetainedEarnings"].diff(), C["TotalRevenue"].diff()),

    "CapEx_vs_Revenue_Rebound":
        lambda C: safe_div(
            C.groupby("Symbol")["CapitalExpenditure"].pct_change(),
            C.groupby("Symbol")["TotalRevenue"].pct_change()
        ),

    # ─────────────────────────────────── INFORMATION · RECOVER
    "Advertising_to_Sales":
        lambda C: safe_div(C["SellingAndMarketingExpense"].fillna(0), C["TotalRevenue"]),

    "Marketing_Efficiency_Ratio":
        lambda C: safe_div(C["TotalRevenue"] - C["CostOfRevenue"] - C["SellingAndMarketingExpense"],
                           C["SellingAndMarketingExpense"]),

    "Combined_SellingExpense_to_Sales":
        lambda C: safe_div(C["SellingAndMarketingExpense"], C["TotalRevenue"]),

    "SGA_to_Sales":
        lambda C: safe_div(C["SellingGeneralAndAdministration"], C["TotalRevenue"]),

    "Advertising_to_Sales_3yrCAGR":
        lambda C: ratio_funcs["Advertising_to_Sales"](C) \
                    .groupby(C["Symbol"]).pct_change(3),

    "OperatingMargin_Delta":
        lambda C: safe_div(C["OperatingIncome"], C["TotalRevenue"]) - \
                  safe_div(C["OperatingIncome"].shift(1), C["TotalRevenue"].shift(1)),

    "OperatingMargin_Slope_5yr":
        lambda C: C.groupby("Symbol").apply(
            lambda g: slope(safe_div(g["OperatingIncome"], g["TotalRevenue"]), 5)
        ).reset_index(level=0, drop=True),

    "OperatingMargin_Slope_3yr":
        lambda C: C.groupby("Symbol").apply(
            lambda g: slope(safe_div(g["OperatingIncome"], g["TotalRevenue"]), 3)
        ).reset_index(level=0, drop=True),

    "EBITDA_Margin_StdDev_5yr":
        lambda C: C.groupby("Symbol").apply(
            lambda g: safe_div(g["EBITDA"], g["TotalRevenue"]).rolling(5, 2).std()
        ).reset_index(level=0, drop=True),

    "EBITDA_Margin_StdDev_3yr":
        lambda C: C.groupby("Symbol").apply(
            lambda g: safe_div(g["EBITDA"], g["TotalRevenue"]).rolling(3, 2).std()
        ).reset_index(level=0, drop=True),

    "EPS_Volatility_5yr":
        lambda C: C.groupby("Symbol")["BasicEPS"].transform(
            lambda x: safe_div(x.rolling(5, 2).std(), x.rolling(5, 2).mean().abs())
        ),

    "EPS_Volatility_3yr":
        lambda C: C.groupby("Symbol")["BasicEPS"].transform(
            lambda x: safe_div(x.rolling(3, 2).std(), x.rolling(3, 2).mean().abs())
        ),

    "EPS_Growth":    lambda C: safe_div(C["BasicEPS"].diff(), C["BasicEPS"].shift(1)),
    "EPS_Stability":
        lambda C: C.groupby("Symbol")["BasicEPS"].transform(
            lambda x: safe_div(x.rolling(5, 2).std(), x.rolling(5, 2).mean().abs())
        ),

    "ROE_StdDev_5yr":
        lambda C: C.groupby("Symbol").apply(
            lambda g: safe_div(g["NetIncome"],
                               g["TotalEquity"].rolling(2).mean()).rolling(5, 2).std()
        ).reset_index(level=0, drop=True),

    "ROE_StdDev_3yr":
        lambda C: C.groupby("Symbol").apply(
            lambda g: safe_div(g["NetIncome"],
                               g["TotalEquity"].rolling(2).mean()).rolling(3, 2).std()
        ).reset_index(level=0, drop=True),

    "Advertising_ROI":
        lambda C: C.groupby("Symbol").apply(
            lambda g: safe_div(g["TotalRevenue"].diff(), g["SellingAndMarketingExpense"])
        ).reset_index(level=0, drop=True),

    # ───────────────────────────────────── COGNITIVE · RECOVER
    "NOPAT_Alt":
        lambda C: safe_div(
            C["NetIncome"] + C["InterestExpense"] * (1 - C["EffectiveTaxRateAsReported"]),
            C["TotalRevenue"]
        ),

    "NOPAT_to_Avg_InvestedCapital":
        lambda C: safe_div(
            C["NetIncome"] + C["InterestExpense"] * (1 - 0.30),
            (C["TotalDebt"] + C["TotalEquity"] - C["Cash"]).rolling(2).mean()
        ),

    "ROIC":   # base ROIC (slopes & moving-stats live in derived_ratio_funcs)
        lambda C: safe_div(
            C["EarningBeforeInterestAndTax"],
            C.groupby("Symbol").apply(
                lambda g: (g["TotalDebt"] + g["TotalEquity"] - g["Cash"]).rolling(2).mean()
            ).reset_index(level=0, drop=True)
        ),

    "ROIC_5yr": lambda C: C.groupby("Symbol")["ROIC"].transform(lambda x: x.rolling(5, 1).mean()),
    "ROIC_3yr": lambda C: C.groupby("Symbol")["ROIC"].transform(lambda x: x.rolling(3, 1).mean()),

    "ROIC_Spread_vs_Sector":
        lambda C: ratio_funcs["ROIC"](C) - \
                  ratio_funcs["ROIC"](C).groupby(C["SectorName"]).transform("median"),

    "CapEx_Efficiency":
        lambda C: safe_div(C["EarningBeforeInterestAndTax"].diff(), C["CapitalExpenditure"]),

    "EVA_Margin":
        lambda C: safe_div(
            ratio_funcs["NOPAT_Alt"](C) -
            0.08 * (C["TotalDebt"] + C["TotalEquity"] - C["Cash"]),
            C["TotalRevenue"]
        ),

    # ─────────────────────────────────────── SOCIAL · RECOVER
    "Equity_Issuance_Rate":
        lambda C: safe_div(
            C["IssuanceOfCapitalStock"],
            C.groupby("Symbol")["TotalEquity"].rolling(2).mean().reset_index(level=0, drop=True)
        ),

    "Share_Issuance_Rate":
        lambda C: safe_div(C["BasicAverageShares"].diff(), C["BasicAverageShares"].shift(1)),

    "Share_Dilution_3yrChg":
        lambda C: safe_div(C["BasicAverageShares"],
                           C.groupby("Symbol")["BasicAverageShares"].shift(3)) - 1,

    "Debt_Issuance_Rate":
        lambda C: safe_div(C["IssuanceOfDebt"], C["TotalDebt"]),

    "Equity_to_CapEx_Financing":
        lambda C: safe_div(C["IssuanceOfCapitalStock"], C["CapitalExpenditure"]),

    "Debt_Service_Coverage_Recover":
        lambda C: safe_div(C["OperatingCashFlow"],
                           C["InterestExpense"] + C["RepaymentOfDebt"]),

    "Dividend_Reinstatement_Flag":
        lambda C: ((C["CashDividendsPaid"].shift(1) == 0) &
                   (C["CashDividendsPaid"] > 0)).astype(int),


#### ADAPT STAGES
# ───────────────────────────────────── PHYSICAL · ADAPT
"CapEx_to_Sales":
    lambda C: safe_div(C["CapitalExpenditure"], C["TotalRevenue"]),

"CapEx_to_Depreciation":
    lambda C: safe_div(C["CapitalExpenditure"], C["Depreciation"]),

"CapEx_plus_RnD_to_Sales":
    lambda C: safe_div(C["CapitalExpenditure"] + C.get("ResearchAndDevelopment", 0),
                       C["TotalRevenue"]),

"CapEx_GrowthRate":
    lambda C: C.groupby("Symbol")["CapitalExpenditure"].pct_change(),

# vectorised version – no DataFrame output risk
"CapEx_Dep_Growth":
    lambda C: safe_div(C["CapitalExpenditure"], C["Depreciation"])  \
                 .groupby(C["Symbol"]).pct_change(),

"Maintenance_CapEx_Ratio":
    lambda C: safe_div(C["CapitalExpenditure"], C["Depreciation"]),

"FreeCashFlow_to_InvestedCapital":
    lambda C: safe_div(
        C["FreeCashFlow"],
        C.groupby("Symbol")["InvestedCapital"].rolling(2).mean().reset_index(level=0, drop=True)
    ),

"CapitalisedSoftware_to_Assets":
    lambda C: safe_div(C.get("GoodwillAndOtherIntangibleAssets", 0), C["TotalAssets"]),

"Reinvestment_Rate":
    lambda C: safe_div(C["CapitalExpenditure"] - C["Depreciation"], C["OperatingCashFlow"]),

"CapEx_Variability_5yr":
    lambda C: C.groupby("Symbol")["CapitalExpenditure"].transform(
        lambda x: safe_div(x.rolling(5, 2).std(), x.rolling(5, 2).mean())
    ),

"CapEx_Variability_3yr":
    lambda C: C.groupby("Symbol")["CapitalExpenditure"].transform(
        lambda x: safe_div(x.rolling(3, 2).std(), x.rolling(3, 2).mean())
    ),

"OCF_to_CapEx":
    lambda C: safe_div(C["OperatingCashFlow"], C["CapitalExpenditure"]),

"OCF_FreeCash_Cushion":
    lambda C: safe_div(
        C["OperatingCashFlow"] - C["CapitalExpenditure"] - C["CashDividendsPaid"],
        C["TotalRevenue"]
    ),

"Net_Investing_Flag":
    lambda C: (C["CapitalExpenditure"] > C["Depreciation"]).astype(int),

# ─────────────────────────────────── INFORMATION · ADAPT
"Retention_Ratio":
    lambda C: 1 - safe_div(C["CashDividendsPaid"], C["NetIncome"]),

"Gross_Profitability":
    lambda C: safe_div(C["GrossProfit"], C["TotalAssets"]),

"NOPAT_Margin":
    lambda C: safe_div(C["NetIncome"] + C["InterestExpense"] * (1 - 0.30), C["TotalRevenue"]),

"Sustainable_Growth_Rate":
    lambda C: safe_div(
        C["NetIncome"],
        C.groupby("Symbol")["TotalEquity"].rolling(2).mean().reset_index(level=0, drop=True)
    ) * (1 - safe_div(C["CashDividendsPaid"], C["NetIncome"])),

"Revenue_CAGR_5yr":
    lambda C: safe_div(C["TotalRevenue"],
                       C.groupby("Symbol")["TotalRevenue"].shift(5)) ** 0.2 - 1,

"Revenue_CAGR_3yr":
    lambda C: safe_div(C["TotalRevenue"],
                       C.groupby("Symbol")["TotalRevenue"].shift(3)) ** (1/3) - 1,

"Revenue_Growth":     lambda C: C.groupby("Symbol")["TotalRevenue"].pct_change(),
"NetIncome_Growth":   lambda C: C.groupby("Symbol")["NetIncome"].pct_change(),
"EBITDA_Growth":      lambda C: C.groupby("Symbol")["EBITDA"].pct_change(),
"Assets_Growth":      lambda C: C.groupby("Symbol")["TotalAssets"].pct_change(),
"Equity_Growth":      lambda C: C.groupby("Symbol")["TotalEquity"].pct_change(),
"OCF_Growth":         lambda C: C.groupby("Symbol")["OperatingCashFlow"].pct_change(),
"FCF_Growth":         lambda C: C.groupby("Symbol")["FreeCashFlow"].pct_change(),

"FCF_Growth_3yrCAGR":
    lambda C: safe_div(C["FreeCashFlow"],
                       C.groupby("Symbol")["FreeCashFlow"].shift(3)) ** (1/3) - 1,

"OCF_CAGR_5yr":
    lambda C: safe_div(C["OperatingCashFlow"],
                       C.groupby("Symbol")["OperatingCashFlow"].shift(5)) ** 0.2 - 1,

"OCF_CAGR_3yr":
    lambda C: safe_div(C["OperatingCashFlow"],
                       C.groupby("Symbol")["OperatingCashFlow"].shift(3)) ** (1/3) - 1,

"Dividend_Growth":
    lambda C: C.groupby("Symbol")["DividendPerShare"].pct_change(),

"Dividend_Growth_Alt":
    lambda C: C.groupby("Symbol")["CashDividendsPaid"].pct_change(),

"CapEx_Growth":
    lambda C: C.groupby("Symbol")["CapitalExpenditure"].pct_change(),

"GrossProfit_Growth":
    lambda C: C.groupby("Symbol")["GrossProfit"].pct_change(),

"OCF_Volatility_3yr":
    lambda C: C.groupby("Symbol")["OperatingCashFlow"].transform(
        lambda x: safe_div(x.rolling(3, 2).std(), x.rolling(3, 2).mean())
    ),

"CashFlow_Skewness":
    lambda C: C.groupby("Symbol")["OperatingCashFlow"].transform(lambda x: skew(x, 5)),

"FCF_Volatility_5yr":
    lambda C: C.groupby("Symbol")["FreeCashFlow"].transform(
        lambda x: safe_div(x.rolling(5, 2).std(), x.rolling(5, 2).mean())
    ),

"OCF_Margin":
    lambda C: safe_div(C["OperatingCashFlow"], C["TotalRevenue"]),

"RnD_Capitalised_Share":
    lambda C: safe_div(C.get("PurchaseOfIntangibles", 0),
                       C.get("ResearchAndDevelopment", np.nan)),

# ───────────────────────────────────── COGNITIVE · ADAPT
"Market_Share_Ratio":
    lambda C: safe_div(C["TotalRevenue"],
                       C.groupby("SectorName")["TotalRevenue"].transform("sum")),

"Relative_Revenue_Growth_vs_Sector":
    lambda C: C.groupby("Symbol")["TotalRevenue"].pct_change() - \
              C.groupby("SectorName")["TotalRevenue"].pct_change(),

"Market_Share_Revenue_Change":
    lambda C: safe_div(C["TotalRevenue"],
                       C.groupby("SectorName")["TotalRevenue"].transform("sum")),

"Market_Share_EBITDA_Change":
    lambda C: safe_div(C["EBITDA"],
                       C.groupby("SectorName")["EBITDA"].transform("sum")),

"Relative_EBITDA_Growth_vs_Sector":
    lambda C: C.groupby("Symbol")["EBITDA"].pct_change() - \
              C.groupby("SectorName")["EBITDA"].pct_change(),

"Relative_OperatingIncome_Growth_vs_Sector":
    lambda C: C.groupby("Symbol")["OperatingIncome"].pct_change() - \
              C.groupby("SectorName")["OperatingIncome"].pct_change(),

"Market_Share_OperatingIncome_Change":
    lambda C: safe_div(C["OperatingIncome"],
                       C.groupby("SectorName")["OperatingIncome"].transform("sum")),

"Rev_CAGR_vs_Sector":
    lambda C: C.groupby("SectorName")["TotalRevenue"].transform(lambda x: x.pct_change(3)),

"Relative_Revenue_Growth_Sector":
    lambda C: C.groupby("SectorName")["TotalRevenue"].pct_change(),

"Relative_EBITDA_Growth_Sector":
    lambda C: C.groupby("SectorName")["EBITDA"].pct_change(),

"Market_Share_of_Revenue":
    lambda C: safe_div(C["TotalRevenue"],
                       C.groupby("SectorName")["TotalRevenue"].transform("sum")),

"Revenue_Sector_Share_Growth":
    lambda C: ratio_funcs["Market_Share_of_Revenue"](C).groupby(C["Symbol"]).pct_change(),

"EBITDA_Sector_Share":
    lambda C: safe_div(C["EBITDA"],
                       C.groupby("SectorName")["EBITDA"].transform("sum")),

"Relative_OperatingIncome_Growth_Sector":
    lambda C: C.groupby("SectorName")["OperatingIncome"].pct_change(),

"Relative_NetIncome_Growth_Sector":
    lambda C: C.groupby("SectorName")["NetIncome"].pct_change(),

"NetIncome_Sector_Share":
    lambda C: safe_div(C["NetIncome"],
                       C.groupby("SectorName")["NetIncome"].transform("sum")),

# avoid Series.groupby("Symbol") inside .apply for rolling mean
"Sales_to_TotalAssets":
    lambda C: safe_div(
        C["TotalRevenue"],
        C.groupby("Symbol")["TotalAssets"].rolling(2).mean().reset_index(level=0, drop=True)
    ),

"Sales_to_Marketing_Leverage":
    lambda C: C.groupby("Symbol").apply(
        lambda g: safe_div(g["TotalRevenue"].pct_change(),
                           g["SellingAndMarketingExpense"].pct_change())
    ).reset_index(level=0, drop=True),

"GrossProfit_to_Marketing_Leverage":
    lambda C: C.groupby("Symbol").apply(
        lambda g: safe_div(g["GrossProfit"].pct_change(),
                           g["SellingAndMarketingExpense"].pct_change())
    ).reset_index(level=0, drop=True),

"Marketing_Intensity":
    lambda C: safe_div(C["SellingAndMarketingExpense"], C["TotalRevenue"]),

"GrossMargin_Slope_5yr":
    lambda C: C.groupby("Symbol").apply(
        lambda g: slope(safe_div(g["GrossProfit"], g["TotalRevenue"]), 5)
    ).reset_index(level=0, drop=True),

"GrossMargin_Slope_3yr":
    lambda C: C.groupby("Symbol").apply(
        lambda g: slope(safe_div(g["GrossProfit"], g["TotalRevenue"]), 3)
    ).reset_index(level=0, drop=True),

"Price_Realisation_Index":
    lambda C: C.groupby("Symbol").apply(
        lambda g: (safe_div(g["GrossProfit"], g["TotalRevenue"]).diff()) -
                  safe_div(g["CostOfRevenue"].diff(), g["TotalRevenue"].shift(1))
    ).reset_index(level=0, drop=True),

# ─────────────────────────────────────── SOCIAL · ADAPT
"DPS_to_EPS":
    lambda C: safe_div(C["DividendPerShare"], C.get("BasicEPS", np.nan)),

"Dividend_Payout_Ratio":
    lambda C: safe_div(C["CashDividendsPaid"], C["NetIncome"]),

"FCF_Payout_Ratio":
    lambda C: safe_div(C["CashDividendsPaid"], C["FreeCashFlow"]),

"Dividend_Stability_Index":
    lambda C: C.groupby("Symbol")["CashDividendsPaid"].transform(
        lambda x: x.notna().rolling(10, 1).mean()
    ),

"Dividend_Yield_on_FCF":
    lambda C: safe_div(C["CashDividendsPaid"], C["FreeCashFlow"]),

"Dividend_Coverage":
    lambda C: safe_div(C["OperatingCashFlow"], C["CashDividendsPaid"]),

"Dividend_Coverage_FCF":
    lambda C: safe_div(C["OperatingCashFlow"] - C["CapitalExpenditure"],
                       C["CashDividendsPaid"]),

"Dividend_Payout_CV":
    lambda C: C.groupby("Symbol").apply(
        lambda g: safe_div(
            safe_div(g["CashDividendsPaid"], g["NetIncome"]).rolling(5, 2).std(),
            safe_div(g["CashDividendsPaid"], g["NetIncome"]).rolling(5, 2).mean()
        )
    ).reset_index(level=0, drop=True),

"Share_Count_Reduction_YoY":
    lambda C: -C.groupby("Symbol")["BasicAverageShares"].pct_change(),

"Net_Buyback_to_FCF":
    lambda C: safe_div(C.get("CommonStockRepurchased", 0) -
                       C.get("IssuanceOfCapitalStock", 0),
                       C["FreeCashFlow"]),

"Dividend_Payout_Flexibility":
    lambda C: 1 - C.groupby("Symbol")["CashDividendsPaid"].transform(
        lambda x: safe_div(x.rolling(5, 2).std(), x.rolling(5, 2).mean())
    ),

"Dividend_Cut_Flag":
    lambda C: (C["CashDividendsPaid"].diff() < 0).astype(int),

"Buyback_Coverage_OCF":
    lambda C: safe_div(C.get("CommonStockRepurchased", 0) -
                       C.get("IssuanceOfCapitalStock", 0),
                       C["OperatingCashFlow"]),
}

# ───────────────────── post-hoc derived ratios ─────────────────────
# If Stage-03 imported ratio_library before these definitions ran we
# still need derived_ratio_funcs to exist, so create / reuse it safely.
if "derived_ratio_funcs" not in globals():
    derived_ratio_funcs: Dict[str, Callable] = {}

def _log1p_signed(s: pd.Series) -> pd.Series:
    """
    Signed log-transform that handles negative inputs:
        f(x) = sign(x) · log1p(|x|)
    """
    return np.sign(s) * np.log1p(np.abs(s))

def _sqrt_signed(s: pd.Series) -> pd.Series:
    """
    Signed square-root transform that handles negative inputs:
        f(x) = sign(x) · sqrt(|x|)
    """
    return np.sign(s) * np.sqrt(np.abs(s))

# Attach log / sqrt variants for key profitability ratios.
# -- add more bases here whenever needed --
for _base in ("ROE", "ROA", "ROIC"):
    if _base in ratio_funcs:
        _f = ratio_funcs[_base]
        derived_ratio_funcs[f"Log_{_base}"]  = lambda C, fn=_f: _log1p_signed(fn(C))
        derived_ratio_funcs[f"Sqrt_{_base}"] = lambda C, fn=_f: _sqrt_signed(fn(C))

# ───────────────────────── contract footer ──────────────────────────
# Guarantee the three public symbols so all pipeline stages can do
#   `from ratio_library import …`  without defensive guards.

if "ratio_funcs" not in globals():
    raise RuntimeError("ratio_library must define `ratio_funcs`")

if "ratio_domain_stage_map" not in globals():
    ratio_domain_stage_map: Dict[str, List[str] | str] = {}

__all__: List[str] = [
    # helpers
    "safe_div", "slope", "skew", "winsor", "_to_series", "_ensure_core_columns",
    # contract objects
    "ratio_funcs", "derived_ratio_funcs", "ratio_domain_stage_map",
]
