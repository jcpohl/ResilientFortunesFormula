"""
ratio_library.py
────────────────
This module defines:
  • ratio_funcs:          a dictionary of ~400 first-pass ratio lambdas
  • derived_ratio_funcs:  a dictionary of derived ratios that reference earlier ratios
  • Helper functions:     safe_div, winsor, slope, skew, and _to_series

Note: The ratio mapping (DOMAIN-STAGE MAP) now lives in 
      ratio_domain_stage_map.yaml.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import skew as _scipy_skew

# ── Try to import shared helpers; if unavailable, define fallbacks ─────────
try:
    from utils import safe_div, slope, skew
except ImportError:
    def safe_div(a, b):
        """Vectorised division – returns NaN where denominator is 0 or invalid."""
        with np.errstate(divide="ignore", invalid="ignore"):
            out = a / b
        if isinstance(out, pd.Series):
            return out.replace([np.inf, -np.inf], np.nan)
        out = np.where(np.isfinite(out), out, np.nan)
        return out

    def slope(series: pd.Series, window: int) -> float:
        """
        Computes the OLS slope of the last `window` non-NaN observations.
        Returns NaN if there are fewer than 2 usable points.
        """
        y = series.dropna().tail(window)
        n = len(y)
        if n < 2:
            return np.nan
        x = np.arange(n, dtype=float)
        try:
            m, _ = np.polyfit(x, y.values.astype(float), 1)
            return m
        except Exception:
            return np.nan

    def skew(series: pd.Series, window: int) -> pd.Series:
        """Calculates rolling (bias-corrected) skewness over `window` observations."""
        return series.rolling(window, min_periods=3).apply(
            lambda x: _scipy_skew(x, bias=False) if x.notna().sum() >= 3 else np.nan,
            raw=False
        )

def winsor(s: pd.Series, pct: float = 0.01) -> pd.Series:
    """
    Two-sided winsorisation at the `pct` and 1-`pct` quantiles.
    """
    if not 0 < pct < 0.5:
        raise ValueError("pct must be between 0 and 0.5")
    lo, hi = s.quantile(pct), s.quantile(1 - pct)
    return s.clip(lo, hi)

def _to_series(x, index):
    """
    Coerces any ratio output into a float Series aligned to index.
    Handles scalar, DataFrame (1- or 2-column), or Series outputs.
    """
    if isinstance(x, pd.Series):
        return x.reindex(index).astype("float64")
    if isinstance(x, pd.DataFrame):
        dup_cols = [c for c in x.columns if x[c].equals(x.index)]
        if dup_cols:
            x = x.drop(columns=dup_cols)
        num_cols = [c for c in x.columns if pd.api.types.is_numeric_dtype(x[c])]
        if not num_cols:
            raise ValueError("output has no numeric columns")
        if len(num_cols) == 1:
            return x[num_cols[0]].reindex(index).astype("float64")
        first = x[num_cols[0]]
        if all(first.equals(x[c]) for c in num_cols[1:]):
            return first.reindex(index).astype("float64")
        return x[num_cols].mean(axis=1).reindex(index).astype("float64")
    if np.isscalar(x):
        return pd.Series(float(x), index=index, dtype="float64")
    raise ValueError(f"unsupported output type: {type(x)}")
    
# ─────────────────────────────────────────────────────────────────────
# 1 · PRIMARY RATIO FUNCTIONS
#    (copy–paste the giant block you already have)
# ─────────────────────────────────────────────────────────────────────

ratio_funcs = {

    #### PREPARE STAGES
         # ───────────────────────────────────── PHYSICAL · PREPARE
    "Cash_to_Total_Assets":            lambda C: safe_div(C["Cash"].fillna(0), C["TotalAssets"]),
    "Cash_to_Total_Assets_Alt":        lambda C: safe_div(C["CashAndCashEquivalents"].fillna(0), C["TotalAssets"]),
    "Current_Ratio":                   lambda C: safe_div(C["CurrentAssets"], C["CurrentLiabilities"]),
    "Quick_Ratio":                     lambda C: safe_div(C["CurrentAssets"] - C["Inventory"], C["CurrentLiabilities"]),
    "Cash_Ratio":                      lambda C: safe_div(C["CashAndCashEquivalents"], C["CurrentLiabilities"]),
    "Operating_Cash_Flow_Ratio":       lambda C: safe_div(C["OperatingCashFlow"], C["CurrentLiabilities"]),
    "NetWorkingCapital_to_Assets":     lambda C: safe_div(C["CurrentAssets"] - C["CurrentLiabilities"], C["TotalAssets"]),
    "Cash_Conversion_Cycle":           lambda C: C["AccountsReceivableDays"] + C["InventoryDays"] - C["AccountsPayableDays"],
    "Operating_CF_to_Debt":            lambda C: safe_div(C["OperatingCashFlow"], C["TotalDebt"]),
    "NetDebt_to_OCF":                  lambda C: safe_div(C["TotalDebt"] - C["CashAndCashEquivalents"], C["OperatingCashFlow"]),
    "DaysPayablesOutstanding":         lambda C: safe_div(C["AccountsPayable"], safe_div(C["CostOfRevenue"], 365)),
    "WorkingCapital_to_Sales":         lambda C: safe_div(C["CurrentAssets"] - C["CurrentLiabilities"], C["TotalRevenue"]),
    "CashEquivalents_to_CurrentLiab":  lambda C: safe_div(C["CashAndCashEquivalents"], C["CurrentLiabilities"]),
    "AccountsReceivable_Turnover":     lambda C: safe_div(C["TotalRevenue"], C["AccountsReceivable"]),
    "Inventory_Turnover":              lambda C: safe_div(C["CostOfRevenue"], C["Inventory"]),
    "Net_Operating_WC_to_Assets":      lambda C: safe_div(C["AccountsReceivable"] + C["Inventory"] - C["AccountsPayable"], C["TotalAssets"]),
    "CashFlow_to_Debt":                lambda C: safe_div(C["OperatingCashFlow"], C["TotalDebt"]),
    "CashFlow_to_Assets":              lambda C: safe_div(C["OperatingCashFlow"], C["TotalAssets"]),
    "FCF_to_Assets":                   lambda C: safe_div(C["FreeCashFlow"], C["TotalAssets"]),
    "CashFlow_Coverage_of_Interest":   lambda C: safe_div(C["OperatingCashFlow"], C["InterestExpense"]),
    "OCF_to_NetIncome":                lambda C: safe_div(C["OperatingCashFlow"], C["NetIncome"]),
    "DaysSalesOutstanding":            lambda C: safe_div(C["AccountsReceivable"], safe_div(C["TotalRevenue"], 365)),
    "DaysInventoryOutstanding":        lambda C: safe_div(C["Inventory"], safe_div(C["CostOfRevenue"], 365)),
    "OperatingCycle":                  lambda C: safe_div(C["AccountsReceivable"], safe_div(C["TotalRevenue"], 365)) + safe_div(C["Inventory"], safe_div(C["CostOfRevenue"], 365)),
    "BeginningCash_to_Sales":          lambda C: safe_div(C["BeginningCashPosition"], C["TotalRevenue"]),
    "RetainedEarnings_to_Assets":      lambda C: safe_div(C["RetainedEarnings"], C["TotalAssets"]),
    "Deposits_to_Assets":              lambda C: safe_div(C.get("TotalDeposits", 0), C["TotalAssets"]),
    "Cash_Burn_Duration":              lambda C: safe_div(C["CashAndCashEquivalents"], safe_div(C["CostOfRevenue"] + C["SellingGeneralAndAdministration"], 365)),
    "WorkingCapital_Days_Trend":       lambda C: C.groupby(C["Symbol"]).apply(lambda g: slope(g["DaysSalesOutstanding"] + g["DaysInventoryOutstanding"] - g["DaysPayablesOutstanding"], 3)).reset_index(level=0, drop=True),
    "Cash_Return_on_Assets":           lambda C: safe_div(C["OperatingCashFlow"], C["TotalAssets"]),
    "CashConversion_Ratio_3yrAvg":     lambda C: C.groupby(C["Symbol"]).apply(lambda g: safe_div((g["OperatingCashFlow"]/g["NetIncome"]).rolling(3,1).mean(), 1)).reset_index(level=0, drop=True),
    "FCF_Margin_3yrAvg":               lambda C: C.groupby(C["Symbol"])["FreeCashFlow"].transform(lambda x: safe_div(x.rolling(3,1).mean(), C["TotalRevenue"].rolling(3,1).mean())),
    "FCF_Yield_on_Assets":             lambda C: safe_div(C["FreeCashFlow"], C["TotalAssets"]),
    "Operating_Efficiency_Ratio":      lambda C: safe_div(C["OperatingExpense"], C["TotalRevenue"]),
    "WorkingCapital_Turnover":         lambda C: safe_div(C["TotalRevenue"], C["CurrentAssets"] - C["CurrentLiabilities"]),

    # ─────────────────────────────────── INFORMATION · PREPARE
    "Accrual_Ratio":                   lambda C: safe_div(C["NetIncome"] - C["OperatingCashFlow"], C["TotalAssets"]),
    "Sloan_Accrual_Measure":           lambda C: safe_div(C["NetIncome"] - C["OperatingCashFlow"], C["TotalAssets"]),
    "Cash_Earnings_Ratio":             lambda C: safe_div(C["OperatingCashFlow"], C["NetIncome"]),
    "Net_Operating_Accruals":          lambda C: safe_div(C["NetIncome"] - C["OperatingCashFlow"] - C["Depreciation"], C["TotalAssets"]),
    "Dechow_Dichev_AQ":                lambda C: safe_div(ratio_funcs["Accrual_Ratio"](C), np.abs(C["OperatingCashFlow"])),
    "Percent_Accruals":                lambda C: safe_div(C["NetIncome"] - C["OperatingCashFlow"], C["TotalRevenue"]),
    "Revenue_Quality":                 lambda C: safe_div(C["AccountsReceivable"].diff(), C["TotalRevenue"].diff()),
    "Revenue_Quality_Delta_AR":        lambda C: safe_div(C["AccountsReceivable"].diff(), C["TotalRevenue"].diff()),
    "WC_Accrual_Ratio":                lambda C: safe_div((C["AccountsReceivable"].diff() + C["Inventory"].diff() - C["AccountsPayable"].diff()), C["TotalRevenue"]),
    "DSO_Trend_3yr": lambda C: C.groupby("Symbol")["DaysSalesOutstanding"].transform(lambda x: slope(x, 3)),
    "Inventory_Inflation_3yr": lambda C: C.groupby("Symbol")["DaysInventoryOutstanding"].transform(lambda x: slope(x, 3)),
    "ROA_3yrAvg":                      lambda C: safe_div(C.groupby(C["Symbol"])["NetIncome"].transform(lambda x: x.rolling(3, 1).mean()), C.groupby(C["Symbol"])["TotalAssets"].transform(lambda x: x.rolling(3, 1).mean())),
    "ROE_3yrAvg":                      lambda C: safe_div(C.groupby(C["Symbol"])["NetIncome"].transform(lambda x: x.rolling(3, 1).mean()), C.groupby(C["Symbol"])["TotalEquity"].transform(lambda x: x.rolling(3, 1).mean())),
    "GrossMargin_3yrAvg":              lambda C: safe_div(C.groupby(C["Symbol"])["GrossProfit"].transform(lambda x: x.rolling(3, 1).mean()), C.groupby(C["Symbol"])["TotalRevenue"].transform(lambda x: x.rolling(3, 1).mean())),
    "EBITDA_Margin_3yrAvg":            lambda C: safe_div(C.groupby(C["Symbol"])["EBITDA"].transform(lambda x: x.rolling(3, 1).mean()), C.groupby(C["Symbol"])["TotalRevenue"].transform(lambda x: x.rolling(3, 1).mean())),
    "OCF_Margin_3yrAvg":               lambda C: safe_div(C.groupby(C["Symbol"])["OperatingCashFlow"].transform(lambda x: x.rolling(3, 1).mean()), C.groupby(C["Symbol"])["TotalRevenue"].transform(lambda x: x.rolling(3, 1).mean())),
    "Accruals_to_Sales":               lambda C: safe_div(C["NetIncome"] - C["OperatingCashFlow"], C["TotalRevenue"]),

    # ───────────────────────────────────── COGNITIVE · PREPARE
    "R_D_Growth_Rate":                 lambda C: C["ResearchAndDevelopment"].pct_change(),
    "R_D_to_Opex":                     lambda C: safe_div(C["ResearchAndDevelopment"], C["OperatingExpense"]),
    "RnD_Plus_CapEx_Intensity":        lambda C: safe_div(C["ResearchAndDevelopment"] + C["CapitalExpenditure"], C["TotalRevenue"]),
    "R&D_Intensity":                   lambda C: safe_div(C["ResearchAndDevelopment"], C["TotalRevenue"]),
    "R_D_3yr_CAGR":                    lambda C: C.groupby(C["Symbol"])["ResearchAndDevelopment"].pct_change(periods=3),
    "Innovation_Ratio":                lambda C: safe_div(C["ResearchAndDevelopment"], C["GrossProfit"]),
    "R_D_Growth":                      lambda C: C.groupby(C["Symbol"])["ResearchAndDevelopment"].pct_change(),

    # ─────────────────────────────────────── SOCIAL · PREPARE
    "Interest_Coverage_Ratio":         lambda C: safe_div(C["EarningBeforeInterestAndTax"], C["InterestExpense"]),
    "Cash_Interest_Coverage_Ratio":    lambda C: safe_div(C["OperatingCashFlow"], C["InterestExpense"]),
    "EBITDA_Interest_Coverage":        lambda C: safe_div(C["EBITDA"], C["InterestExpense"]),
    "DSCR":                            lambda C: safe_div(C["EBITDA"] - C["CapitalExpenditure"], C["InterestExpense"] + C.get("DebtRepayment", 0)),
    "Debt_to_Assets":                  lambda C: safe_div(C["TotalDebt"], C["TotalAssets"]),
    "EquityRatio":                     lambda C: safe_div(C["TotalEquity"], C["TotalAssets"]),
    "LongTerm_Debt_to_Equity":         lambda C: safe_div(C["LongTermDebt"], C["TotalEquity"]),
    "Net_Debt_to_Equity":              lambda C: safe_div(C["TotalDebt"] - C["Cash"], C["TotalEquity"]),
    "Financial_Leverage":              lambda C: safe_div(C["TotalAssets"], C["TotalEquity"]),
    "Times_Interest_Earned":           lambda C: safe_div(C["EarningBeforeInterestAndTax"], C["InterestExpense"]),
    "Debt_Maturity_Split":             lambda C: safe_div(C["LongTermDebt"], C["TotalDebt"]),
    "InterestCoverage_Cushion": lambda C: safe_div(
    C["EBITDA"].fillna(C.groupby("SectorName")["EBITDA"].transform("median"))
      - C["CapitalExpenditure"].fillna(
            C.groupby("SectorName")["CapitalExpenditure"].transform("median")),
    C["InterestExpense"].fillna(
            C.groupby("SectorName")["InterestExpense"].transform("median"))
),


    #### ABSORB STAGES

    # ───────────────────────────────────── PHYSICAL · ABSORB
    "Asset_Turnover_Ratio":            lambda C: safe_div(C["TotalRevenue"], C["TotalAssets"]),
    "Gross_Profit_Margin":             lambda C: safe_div(C["GrossProfit"], C["TotalRevenue"]),
    "ROA":                             lambda C: safe_div(C["NetIncome"], C["TotalAssets"]),
    "Operating_Margin":                lambda C: safe_div(C["OperatingIncome"], C["TotalRevenue"]),
    "EBITDA_Margin":                   lambda C: safe_div(C["EBITDA"], C["TotalRevenue"]),
    "Net_Income_Margin":               lambda C: safe_div(C["NetIncome"], C["TotalRevenue"]),
    "OperatingIncome_Margin":          lambda C: safe_div(C["OperatingIncome"], C["TotalRevenue"]),
    "Cost_of_Revenue_Ratio":           lambda C: safe_div(C["CostOfRevenue"], C["TotalRevenue"]),
    "Operating_Expense_Ratio":         lambda C: safe_div(C["OperatingExpense"], C["TotalRevenue"]),
    "SG_A_to_Revenue":                 lambda C: safe_div(C["SellingGeneralAndAdministration"], C["TotalRevenue"]),
    "EBT_Margin":                      lambda C: safe_div(C["PretaxIncome"], C["TotalRevenue"]),
    "ROCE":                            lambda C: safe_div(C["EarningBeforeInterestAndTax"], C["TotalAssets"] - C["CurrentLiabilities"]),
    "GrossMargin_Stability":           lambda C: (safe_div(C["GrossProfit"], C["TotalRevenue"]).groupby(C["Symbol"]).transform(lambda x: x.rolling(5, 3).std())),
    "Fixed_Asset_Turnover":            lambda C: safe_div(C["TotalRevenue"], C["TotalAssets"] - C["CurrentAssets"]),
    "Inventory_to_Assets":             lambda C: safe_div(C["Inventory"], C["TotalAssets"]),
    "Receivables_to_Assets":           lambda C: safe_div(C["AccountsReceivable"], C["TotalAssets"]),
    "Payables_to_Assets":              lambda C: safe_div(C["AccountsPayable"], C["TotalAssets"]),
    "Operating_Return_on_Opex":        lambda C: safe_div(C["OperatingIncome"], C["OperatingExpense"]),
    "CashFlow_Margin":                 lambda C: safe_div(C["OperatingCashFlow"], C["TotalRevenue"]),
    "CashConversionEfficiency":        lambda C: safe_div(C["OperatingCashFlow"], C["TotalRevenue"]),
    "Operating_Leverage":              lambda C: safe_div(C["OperatingIncome"].pct_change(5), C["TotalRevenue"].pct_change(5)),
    "InvestedCapital_Turnover":        lambda C: safe_div(C["TotalRevenue"], C["InvestedCapital"]),
    "FCF_Margin":                      lambda C: safe_div(C["FreeCashFlow"], C["TotalRevenue"]),

    # ─────────────────────────────────── INFORMATION · ABSORB
    "Earnings_Volatility":             lambda C: C.groupby(C["Symbol"])["NetIncome"].transform(lambda x: safe_div(x.rolling(5, 2).std(), x.rolling(5, 2).mean())),
    "EBITDA_Volatility":               lambda C: C.groupby(C["Symbol"])["EBITDA"].transform(lambda x: safe_div(x.rolling(5, 2).std(), x.rolling(5, 2).mean())),
    "Earnings_Volatility_AbsMean":     lambda C: C.groupby(C["Symbol"])["NetIncome"].transform(lambda x: safe_div(x.rolling(5, 2).std(), x.rolling(5, 2).apply(lambda y: np.abs(y.mean())))),
    "EBITDA_CV":                       lambda C: C.groupby(C["Symbol"])["EBITDA"].transform(lambda x: safe_div(x.rolling(5, 2).std(), x.rolling(5, 2).mean())),
    "NetIncome_StDev_3yr":             lambda C: C.groupby(C["Symbol"])["NetIncome"].transform(lambda x: x.rolling(3, 2).std()),
    "EBITDA_StDev_3yr":                lambda C: C.groupby(C["Symbol"])["EBITDA"].transform(lambda x: x.rolling(3, 2).std()),
    "Revenue_StDev_3yr":               lambda C: C.groupby(C["Symbol"])["TotalRevenue"].transform(lambda x: x.rolling(3, 2).std()),
    "OCF_StDev_3yr":                   lambda C: C.groupby(C["Symbol"])["OperatingCashFlow"].transform(lambda x: x.rolling(3, 2).std()),
    "ROA_StDev_5yr":                   lambda C: C.groupby(C["Symbol"]).apply(lambda g: safe_div(g["NetIncome"], g["TotalAssets"]).rolling(5, 2).std()).reset_index(level=0, drop=True),
    "EBITDA_Volatility_5yr":           lambda C: C.groupby(C["Symbol"])["EBITDA"].transform(lambda x: safe_div(x.rolling(5, 2).std(), x.rolling(5, 2).mean())),
    "Earnings_Volatility_5yr":         lambda C: C.groupby(C["Symbol"])["NetIncome"].transform(lambda x: safe_div(x.rolling(5, 2).std(), x.rolling(5, 2).mean())),
    "OperatingCF_Volatility_5yr":      lambda C: C.groupby(C["Symbol"])["OperatingCashFlow"].transform(lambda x: safe_div(x.rolling(5, 2).std(), x.rolling(5, 2).mean())),
    "Sales_Volatility_5yr":            lambda C: C.groupby(C["Symbol"])["TotalRevenue"].transform(lambda x: safe_div(x.rolling(5, 2).std(), x.rolling(5, 2).mean())),
    "FCF_Volatility_3yr":              lambda C: C.groupby(C["Symbol"])["FreeCashFlow"].transform(lambda x: x.rolling(3, 2).std()),
    "GrossMargin_Volatility":          lambda C: C.groupby(C["Symbol"])["GrossProfit"].transform(lambda x: safe_div((x/C["TotalRevenue"]).rolling(5,2).std(), (x/C["TotalRevenue"]).rolling(5,2).mean())),

    # ───────────────────────────────────── COGNITIVE · ABSORB
    "Gross_Profitability_Alt":         lambda C: safe_div(C["GrossProfit"], C["NetAssets"]),
    "GrossProfit_to_Equity":           lambda C: safe_div(C["GrossProfit"], C["TotalEquity"]),
    "Price_Premium_Index":             lambda C: safe_div(ratio_funcs["Gross_Profit_Margin"](C), safe_div(C["GrossProfit"], C["TotalRevenue"]).groupby(C["SectorName"]).transform("median")),

    # ─────────────────────────────────────── SOCIAL · ABSORB
    "LT_Debt_to_EBITDA":               lambda C: safe_div(C["LongTermDebt"], C["EBITDA"]),
    "InterestBearingDebt_to_EBITDA":   lambda C: safe_div(C["TotalDebt"], C["EBITDA"]),
    "Net_Debt_to_EBITDA":              lambda C: safe_div(C["TotalDebt"] - C["CashAndCashEquivalents"], C["EBITDA"]),
    "LT_Debt_to_TotalDebt":            lambda C: safe_div(C["LongTermDebt"], C["TotalDebt"]),
    "Effective_Tax_Rate":              lambda C: safe_div(C["IncomeTaxExpense"], C["PretaxIncome"]),
    "InterestExpense_to_Sales":        lambda C: safe_div(C["InterestExpense"], C["TotalRevenue"]),
    "InterestCoverage_Volatility_5yr": lambda C: C.groupby(C["Symbol"]).apply(lambda g: safe_div((g["EarningBeforeInterestAndTax"]/g["InterestExpense"]).rolling(5, 2).std(), (g["EarningBeforeInterestAndTax"]/g["InterestExpense"]).rolling(5, 2).mean())).reset_index(level=0, drop=True),
    "InterestCoverage_Volatility_3yr": lambda C: C.groupby(C["Symbol"]).apply(lambda g: safe_div((g["EarningBeforeInterestAndTax"]/g["InterestExpense"]).rolling(3, 2).std(), (g["EarningBeforeInterestAndTax"]/g["InterestExpense"]).rolling(3, 2).mean())).reset_index(level=0, drop=True),


    #### RECOVER STAGES

    # ───────────────────────────────────── PHYSICAL · RECOVER
    "ROE":                             lambda C: safe_div(C["NetIncome"], C["TotalEquity"]),
    "Debt_to_Equity":                  lambda C: safe_div(C["TotalDebt"], C["TotalEquity"]),
    "Debt_to_Equity_Liab":             lambda C: safe_div(C["TotalLiabilitiesAsReported"], C["TotalEquity"]),
    "EPS":                             lambda C: safe_div(C["NetIncome"], C["BasicAverageShares"]),
    "Cash_Dividends_to_Net_Income":    lambda C: safe_div(C["CashDividendsPaid"], C["NetIncome"]),
    "Return_on_Tangible_Equity":       lambda C: safe_div(C["NetIncome"], C["TotalEquity"] - C["Goodwill"]),
    "Total_Payout_Ratio":              lambda C: safe_div(C.get("CashDividendsPaid", 0) + C.get("CommonStockRepurchased", 0), C["NetIncome"]),
    "NetDebt_PayDown_Rate":            lambda C: safe_div(-(C["TotalDebt"] - C["CashAndCashEquivalents"]).diff(), C["TotalDebt"]),
    "Incremental_ROIC":                lambda C: safe_div(C["NetIncome"].diff() + C["InterestExpense"].diff() * (1 - 0.30), (C["TotalDebt"] + C["TotalEquity"] - C["Cash"]).diff()),
    "ROIC_Slope_5yr":                  lambda C: C.groupby(C["Symbol"])["ROIC"].transform(lambda x: slope(x, 5)),
    "ROIC_Slope_3yr":                  lambda C: C.groupby(C["Symbol"])["ROIC"].transform(lambda x: slope(x, 3)),
    "ROIC_Trend_5yr_Slope":            lambda C: C.groupby(C["Symbol"])["ROIC"].transform(lambda x: slope(x, 5)),
    "ROIC_Trend_3yr_Slope":            lambda C: C.groupby(C["Symbol"])["ROIC"].transform(lambda x: slope(x, 3)),
    "ROIC_3yr_Avg":                    lambda C: C.groupby(C["Symbol"])["ROIC"].transform(lambda x: x.rolling(3, 1).mean()),
    "ROIC_5yr_Median":                 lambda C: C.groupby(C["Symbol"])["ROIC"].transform(lambda x: x.rolling(5, 1).median()),
    "ROIC_3yr_Median":                 lambda C: C.groupby(C["Symbol"])["ROIC"].transform(lambda x: x.rolling(3, 1).median()),
    "EBITDA_DropThrough":              lambda C: safe_div(C["EBITDA"].diff(), C["TotalRevenue"].diff()),
    "OperatingLeverage_Slope":         lambda C: C.groupby(C["Symbol"]).apply(lambda g: slope((g["OperatingIncome"].pct_change()) / (g["TotalRevenue"].pct_change()), 5)).reset_index(level=0, drop=True),
    "Revenue_Recovery_Rate":           lambda C: safe_div(C.groupby(C["Symbol"])["TotalRevenue"].shift(-1), C["TotalRevenue"]),
    "Retention_to_Growth":             lambda C: safe_div(C["RetainedEarnings"], C["TotalRevenue"].diff()),

    # ─────────────────────────────────── INFORMATION · RECOVER
    "Advertising_to_Sales":            lambda C: safe_div(C["SellingAndMarketingExpense"].fillna(0), C["TotalRevenue"]),
    "Marketing_Efficiency_Ratio":      lambda C: safe_div(C["TotalRevenue"] - C["CostOfRevenue"] - C["SellingAndMarketingExpense"], C["SellingAndMarketingExpense"]),
    "Combined_SellingExpense_to_Sales":lambda C: safe_div(C["SellingAndMarketingExpense"], C["TotalRevenue"]),
    "SGA_to_Sales":                    lambda C: safe_div(C["SellingGeneralAndAdministration"], C["TotalRevenue"]),
    "Advertising_to_Sales_3yrCAGR":    lambda C: ratio_funcs["Advertising_to_Sales"](C).groupby(C["Symbol"]).pct_change(3),
    "OperatingMargin_Delta":           lambda C: (C["OperatingIncome"]/C["TotalRevenue"]) - (C["OperatingIncome"].shift(1)/C["TotalRevenue"].shift(1)),
    "OperatingMargin_Slope_5yr":       lambda C: C.groupby(C["Symbol"]).apply(lambda g: slope(g["OperatingIncome"]/g["TotalRevenue"], 5)).reset_index(level=0, drop=True),
    "OperatingMargin_Slope_3yr":       lambda C: C.groupby(C["Symbol"]).apply(lambda g: slope(g["OperatingIncome"]/g["TotalRevenue"], 3)).reset_index(level=0, drop=True),
    "EBITDA_Margin_StdDev_5yr":        lambda C: C.groupby(C["Symbol"])["EBITDA"].transform(lambda x: (x/C["TotalRevenue"]).rolling(5,2).std()),
    "EBITDA_Margin_StdDev_3yr":        lambda C: C.groupby(C["Symbol"])["EBITDA"].transform(lambda x: (x/C["TotalRevenue"]).rolling(3,2).std()),
    "EPS_Volatility_5yr":              lambda C: C.groupby(C["Symbol"])["BasicEPS"].transform(lambda x: safe_div(x.rolling(5,2).std(), x.rolling(5,2).mean().abs())),
    "EPS_Volatility_3yr":              lambda C: C.groupby(C["Symbol"])["BasicEPS"].transform(lambda x: safe_div(x.rolling(3,2).std(), x.rolling(3,2).mean().abs())),
    "EPS_Growth":                      lambda C: safe_div(C["BasicEPS"].diff(), C["BasicEPS"].shift(1)),
    "EPS_Stability":                   lambda C: C.groupby(C["Symbol"])["BasicEPS"].transform(lambda x: safe_div(x.rolling(5,2).std(), x.rolling(5,2).mean().abs())),
    "ROE_StdDev_5yr":                  lambda C: C.groupby(C["Symbol"])["NetIncome"].transform(lambda x: safe_div(x.rolling(5,2).std(), C["TotalEquity"].rolling(5,2).mean())),
    "ROE_StdDev_3yr":                  lambda C: C.groupby(C["Symbol"])["NetIncome"].transform(lambda x: safe_div(x.rolling(3,2).std(), C["TotalEquity"].rolling(3,2).mean())),

    # ───────────────────────────────────── COGNITIVE · RECOVER
    "NOPAT_Alt":                       lambda C: safe_div(C["NetIncome"] + C["InterestExpense"] * (1 - C.get("EffectiveTaxRateAsReported", 0.30)), C["TotalRevenue"]),
    "NOPAT_to_Avg_InvestedCapital":    lambda C: safe_div(C["NetIncome"] + C["InterestExpense"] * (1 - 0.30), (C["TotalDebt"] + C["TotalEquity"] - C["Cash"]).rolling(2).mean()),
    "ROIC":                            lambda C: safe_div(C["EarningBeforeInterestAndTax"], C["TotalDebt"] + C["TotalEquity"] - C["Cash"]),

    # ─────────────────────────────────────── SOCIAL · RECOVER
    "Equity_Issuance_Rate":            lambda C: safe_div(C["IssuanceOfCapitalStock"], C["TotalEquity"]),
    "Share_Issuance_Rate":             lambda C: safe_div(C["BasicAverageShares"].diff(), C["BasicAverageShares"].shift(1)),
    "Share_Dilution_3yrChg":           lambda C: safe_div(C["BasicAverageShares"], C.groupby(C["Symbol"])["BasicAverageShares"].shift(3)) - 1,
    
    #### ADAPT STAGES

     # ───────────────────────────────────── PHYSICAL · ADAPT
    "CapEx_to_Sales":                  lambda C: safe_div(C["CapitalExpenditure"], C["TotalRevenue"]),
    "CapEx_to_Depreciation":           lambda C: safe_div(C["CapitalExpenditure"], C["Depreciation"]),
    "CapEx_plus_RnD_to_Sales":         lambda C: safe_div(C["CapitalExpenditure"] + C["ResearchAndDevelopment"], C["TotalRevenue"]),
    "CapEx_GrowthRate":                lambda C: C.groupby(C["Symbol"])["CapitalExpenditure"].pct_change(),
    "CapEx_Dep_Growth":                lambda C: safe_div(C["CapitalExpenditure"], C["Depreciation"]).pct_change(),
    "Maintenance_CapEx_Ratio":         lambda C: safe_div(C["CapitalExpenditure"], C["Depreciation"]),
    "FreeCashFlow_to_InvestedCapital": lambda C: safe_div(C["FreeCashFlow"], C["InvestedCapital"]),
    "CapitalisedSoftware_to_Assets":   lambda C: safe_div(C.get("SoftwareIntangibles", 0), C["TotalAssets"]),
    "Acquisitions_to_OCF":             lambda C: safe_div(C.get("AcquisitionsCashOutflow", 0), C["OperatingCashFlow"]),
    "Reinvestment_Rate":               lambda C: safe_div(C["CapitalExpenditure"] - C["Depreciation"], C["OperatingCashFlow"]),
    "CapEx_Variability_5yr":           lambda C: C.groupby(C["Symbol"])["CapitalExpenditure"].transform(lambda x: safe_div(x.rolling(5, 2).std(), x.rolling(5, 2).mean())),
    "CapEx_Variability_3yr":           lambda C: C.groupby(C["Symbol"])["CapitalExpenditure"].transform(lambda x: safe_div(x.rolling(3, 2).std(), x.rolling(3, 2).mean())),
    "OCF_to_CapEx":                    lambda C: safe_div(C["OperatingCashFlow"], C["CapitalExpenditure"]),
    "OCF_FreeCash_Cushion":            lambda C: safe_div(C["OperatingCashFlow"] - C["CapitalExpenditure"] - C["CashDividendsPaid"], C["TotalRevenue"]),

    # ─────────────────────────────────── INFORMATION · ADAPT
    "Retention_Ratio":                 lambda C: 1 - safe_div(C["CashDividendsPaid"], C["NetIncome"]),
    "Gross_Profitability":             lambda C: safe_div(C["GrossProfit"], C["TotalAssets"]),
    "NOPAT_Margin":                    lambda C: safe_div(C["NetIncome"] + C["InterestExpense"] * (1 - 0.30), C["TotalRevenue"]),
    "Sustainable_Growth_Rate":         lambda C: safe_div(C["NetIncome"], C["TotalEquity"]) * (1 - safe_div(C["CashDividendsPaid"], C["NetIncome"])),
    "Revenue_CAGR_5yr":                lambda C: (safe_div(C["TotalRevenue"], C.groupby(C["Symbol"])["TotalRevenue"].shift(5)))**0.2 - 1,
    "Revenue_CAGR_3yr":                lambda C: (safe_div(C["TotalRevenue"], C.groupby(C["Symbol"])["TotalRevenue"].shift(3)))**(1/3) - 1,
    "Revenue_Growth":                  lambda C: C.groupby(C["Symbol"])["TotalRevenue"].pct_change(),
    "NetIncome_Growth":                lambda C: C.groupby(C["Symbol"])["NetIncome"].pct_change(),
    "EBITDA_Growth":                   lambda C: C.groupby(C["Symbol"])["EBITDA"].pct_change(),
    "Assets_Growth":                   lambda C: C.groupby(C["Symbol"])["TotalAssets"].pct_change(),
    "Equity_Growth":                   lambda C: C.groupby(C["Symbol"])["TotalEquity"].pct_change(),
    "OCF_Growth":                      lambda C: C.groupby(C["Symbol"])["OperatingCashFlow"].pct_change(),
    "FCF_Growth":                      lambda C: safe_div(C["FreeCashFlow"].diff(), C["FreeCashFlow"].shift(1)),
    "FCF_Growth_3yrCAGR":              lambda C: (safe_div(C["FreeCashFlow"], C.groupby(C["Symbol"])["FreeCashFlow"].shift(3)))**(1/3) - 1,
    "OCF_CAGR_5yr":                    lambda C: safe_div(C["OperatingCashFlow"], C.groupby(C["Symbol"])["OperatingCashFlow"].shift(5))**0.2 - 1,
    "OCF_CAGR_3yr":                    lambda C: safe_div(C["OperatingCashFlow"], C.groupby(C["Symbol"])["OperatingCashFlow"].shift(3))**(1/3) - 1,
    "Dividend_Growth":                 lambda C: C.groupby(C["Symbol"])["DividendPerShare"].pct_change(),
    "Dividend_Growth_Alt":             lambda C: C.groupby(C["Symbol"])["CashDividendsPaid"].pct_change(),
    "CapEx_Growth":                    lambda C: C.groupby(C["Symbol"])["CapitalExpenditure"].pct_change(),
    "GrossProfit_Growth":              lambda C: C.groupby(C["Symbol"])["GrossProfit"].pct_change(),
    "OCF_Volatility_3yr":              lambda C: C.groupby(C["Symbol"])["OperatingCashFlow"].transform(lambda x: safe_div(x.rolling(3,2).std(), x.rolling(3,2).mean())),
    "CashFlow_Skewness":               lambda C: C.groupby(C["Symbol"])["OperatingCashFlow"].transform(lambda x: skew(x, 5)),
    "FCF_Volatility_5yr":              lambda C: C.groupby(C["Symbol"])["FreeCashFlow"].transform(lambda x: safe_div(x.rolling(5,2).std(), x.rolling(5,2).mean())),
    "OCF_Margin":                      lambda C: safe_div(C["OperatingCashFlow"], C["TotalRevenue"]),

    # ───────────────────────────────────── COGNITIVE · ADAPT
    "Market_Share_Ratio":              lambda C: safe_div(C["TotalRevenue"], C.groupby(C["SectorName"])["TotalRevenue"].transform("sum")),
    "Relative_Revenue_Growth_vs_Sector": lambda C: C.groupby(C["SectorName"])["TotalRevenue"].pct_change() - C.groupby(C["Symbol"])["TotalRevenue"].pct_change(),
    "Market_Share_Revenue_Change":     lambda C: safe_div(C["TotalRevenue"], C.groupby(C["SectorName"])["TotalRevenue"].transform("sum")),
    "Market_Share_EBITDA_Change":      lambda C: safe_div(C["EBITDA"], C.groupby(C["SectorName"])["EBITDA"].transform("sum")),
    "Relative_EBITDA_Growth_vs_Sector": lambda C: C.groupby(C["SectorName"])["EBITDA"].pct_change() - C.groupby(C["Symbol"])["EBITDA"].pct_change(),
    "Relative_OperatingIncome_Growth_vs_Sector": lambda C: C.groupby(C["SectorName"])["OperatingIncome"].pct_change() - C.groupby(C["Symbol"])["OperatingIncome"].pct_change(),
    "Market_Share_OperatingIncome_Change": lambda C: safe_div(C["OperatingIncome"], C.groupby(C["SectorName"])["OperatingIncome"].transform("sum")),
    "Rev_CAGR_vs_Sector":              lambda C: C.groupby(C["SectorName"])["TotalRevenue"].transform(lambda x: x.pct_change(periods=3)),
    "Relative_Revenue_Growth_Sector":  lambda C: C.groupby(C["SectorName"])["TotalRevenue"].pct_change(),
    "Relative_EBITDA_Growth_Sector":   lambda C: C.groupby(C["SectorName"])["EBITDA"].pct_change(),
    "Market_Share_of_Revenue":         lambda C: safe_div(C["TotalRevenue"], C.groupby(C["SectorName"])["TotalRevenue"].transform("sum")),
    "Revenue_Sector_Share_Growth":     lambda C: ratio_funcs["Market_Share_of_Revenue"](C).groupby(C["Symbol"]).pct_change(),
    "EBITDA_Sector_Share":             lambda C: safe_div(C["EBITDA"], C.groupby(C["SectorName"])["EBITDA"].transform("sum")),
    "Relative_OperatingIncome_Growth_Sector": lambda C: C.groupby(C["SectorName"])["OperatingIncome"].pct_change(),
    "Relative_NetIncome_Growth_Sector":       lambda C: C.groupby(C["SectorName"])["NetIncome"].pct_change(),
    "NetIncome_Sector_Share":                 lambda C: safe_div(C["NetIncome"], C.groupby(C["SectorName"])["NetIncome"].transform("sum")),
    "Sales_to_TotalAssets":            lambda C: safe_div(C["TotalRevenue"], C["TotalAssets"]),
    "Sales_to_Marketing_Leverage":     lambda C: safe_div(C["TotalRevenue"].pct_change(), C["SellingAndMarketingExpense"].pct_change()),
    "GrossMargin_Slope_5yr":           lambda C: C.groupby(C["Symbol"]).apply(lambda g: slope(g["GrossProfit"]/g["TotalRevenue"], 5)).reset_index(level=0, drop=True),
    "GrossMargin_Slope_3yr":           lambda C: C.groupby(C["Symbol"]).apply(lambda g: slope(g["GrossProfit"]/g["TotalRevenue"], 3)).reset_index(level=0, drop=True),
    "Price_Realisation_Index":         lambda C: (C["GrossProfit"]/C["TotalRevenue"]).diff() - safe_div(C["CostOfRevenue"].diff(), C["TotalRevenue"].shift(1)),

    # ─────────────────────────────────────── SOCIAL · ADAPT
    "DPS_to_EPS":                      lambda C: safe_div(C["DividendPerShare"], C.get("BasicEPS", pd.Series(np.nan))),
    "Dividend_Payout_Ratio":           lambda C: safe_div(C["CashDividendsPaid"], C["NetIncome"]),
    "FCF_Payout_Ratio":                lambda C: safe_div(C["CashDividendsPaid"], C["FreeCashFlow"]),
    "Dividend_Stability_Index":        lambda C: C.groupby(C["Symbol"])["CashDividendsPaid"].transform(lambda x: x.notna().rolling(10, 1).sum() / 10),
    "Dividend_Yield_on_FCF":           lambda C: safe_div(C["CashDividendsPaid"], C["FreeCashFlow"]),
    "Dividend_Coverage":               lambda C: safe_div(C["OperatingCashFlow"], C["CashDividendsPaid"]),
    "Dividend_Coverage_FCF":           lambda C: safe_div(C["OperatingCashFlow"] - C["CapitalExpenditure"], C["CashDividendsPaid"]),
    "Dividend_Payout_CV":              lambda C: C.groupby(C["Symbol"]).apply(lambda g: safe_div((g["CashDividendsPaid"]/g["NetIncome"]).rolling(5,2).std(), (g["CashDividendsPaid"]/g["NetIncome"]).rolling(5,2).mean())).reset_index(level=0, drop=True),
    "Share_Count_Reduction_YoY":       lambda C: -safe_div(C["BasicAverageShares"].diff(), C["BasicAverageShares"].shift(1)),
    "Share_Dilution_3yrChg":           lambda C: safe_div(C["BasicAverageShares"], C.groupby(C["Symbol"])["BasicAverageShares"].shift(3)) - 1,
    "Net_Buyback_to_FCF":              lambda C: safe_div(C.get("CommonStockRepurchased", 0) - C.get("IssuanceOfCapitalStock", 0), C["FreeCashFlow"]),
}

derived_ratio_funcs = {
    # ── ratios that need ROIC ──
    "ROIC_Slope_5yr"        : ratio_funcs.pop("ROIC_Slope_5yr"),
    "ROIC_Slope_3yr"        : ratio_funcs.pop("ROIC_Slope_3yr"),
    "ROIC_Trend_5yr_Slope"  : ratio_funcs.pop("ROIC_Trend_5yr_Slope"),
    "ROIC_Trend_3yr_Slope"  : ratio_funcs.pop("ROIC_Trend_3yr_Slope"),
    "ROIC_3yr_Avg"          : ratio_funcs.pop("ROIC_3yr_Avg"),
    "ROIC_5yr_Median"       : ratio_funcs.pop("ROIC_5yr_Median"),
    "ROIC_3yr_Median"       : ratio_funcs.pop("ROIC_3yr_Median"),

    # ── ratios that need Days-based helpers ──
    "WorkingCapital_Days_Trend": ratio_funcs.pop("WorkingCapital_Days_Trend"),
    "DSO_Trend_3yr"            : ratio_funcs.pop("DSO_Trend_3yr"),
    "Inventory_Inflation_3yr"  : ratio_funcs.pop("Inventory_Inflation_3yr"),

    # ── others that reference earlier ratios ──
    "CashConversion_Ratio_3yrAvg" : ratio_funcs.pop("CashConversion_Ratio_3yrAvg"),
    "Revenue_Sector_Share_Growth" : ratio_funcs.pop("Revenue_Sector_Share_Growth"),
}
# expose a convenience list of all first-pass ratio names
ratio_names = list(ratio_funcs.keys())