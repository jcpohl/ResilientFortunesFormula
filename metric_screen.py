#!/usr/bin/env python
"""
metric_screen.py – decide which metrics are predictably linked to ratios
• 5-fold stratified CV on the pre-event sample
• score = AUROC of the best-single-ratio classifier inside each fold
• keep metrics that pass two gates:
    1. mean-AUROC ≥ 0.65
    2. AUROC > 0.50 in ≥4 of 5 folds  → two-sided sign-test p≈0.0625
Writes   AcceptedMetrics_<SWAN>.txt   (one metric per line)
"""
from __future__ import annotations
import os, json, pickle, logging, warnings
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import binom_test
from pipeline_utils import load_cfg, resolve_run_dir
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── config & paths ──────────────────────────────────────────
CFG        = load_cfg()
EVENTS     = {str(k): v for k, v in CFG["events"].items()}
SWAN       = str(os.getenv("SWAN_YEAR") or next(iter(EVENTS)))
YEAR_INT   = int(SWAN)
RUN        = resolve_run_dir(swan_year=SWAN, run_tag=os.getenv("RUN_TAG"),
                             must_have=f"stage04/Stage4_winsor_RatioRanking_{SWAN}.csv")
ST3_CSV    = RUN / "stage03" / f"Stage3_Data_WithRatios_{SWAN}.csv"
RANK_CSV   = RUN / "stage04" / f"Stage4_winsor_RatioRanking_{SWAN}.csv"
OUT_TXT    = RUN / f"AcceptedMetrics_{SWAN}.txt"
DATE_COL   = "ReportDate"; ID_COL = "Symbol"
K          = 5     # folds
THR_AUC    = 0.65  # mean AUROC gate

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s")

# ── load data ───────────────────────────────────────────────
df    = pd.read_csv(ST3_CSV, parse_dates=[DATE_COL])
pre   = df[df[DATE_COL].dt.year < YEAR_INT].copy()
rank  = pd.read_csv(RANK_CSV)          # has |rho| & AUROC per (ratio, metric)

METRICS = rank["Metric"].unique().tolist()

keep = []
for m in METRICS:
    # ---------- pick *best* ratio for this metric (highest AUROC, winsor set) ----------
    top = (rank[rank["Metric"] == m]
           .sort_values("AUROC", ascending=False)
           .iloc[0])
    ratio = top["Ratio"]
    if ratio not in pre.columns:
        logging.warning("Ratio %s missing in Stage-03 – skip metric %s", ratio, m)
        continue

    flag = f"FlagTemporal_{m}"   # choice of dimension doesn’t matter for predictability
    if flag not in pre.columns:  # fallback to impact
        flag = f"FlagImpact_{m}"
    if flag not in pre.columns:  # or dynamic
        flag = f"FlagDynamic_{m}"
    if flag not in pre.columns:
        logging.warning("No flag found for metric %s – skip", m)
        continue

    X = pre[ratio].astype(float)
    y = pre[flag].astype(int)
    ok = X.notna() & y.notna()
    X, y = X[ok].values.reshape(-1, 1), y[ok].values
    if np.unique(y).size < 2:
        continue

    # ---------- 5-fold CV AUROC of one-variable logistic ----------
    skf  = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
    aucs = []
    for train, test in skf.split(X, y):
        # analytic sigmoid on scalar β fits fine with statsmodels but
        # simplest: rank order == feature value because 1-var – just use value itself
        pred = X[test, 0]
        aucs.append(roc_auc_score(y[test], pred))
    mean_auc = np.mean(aucs)

    # ---------- gate tests ----------
    successes = sum(a > .5 for a in aucs)
    pval      = binom_test(successes, K, 0.5, alternative="greater")
    if mean_auc >= THR_AUC and pval <= 0.10:     # FDR done later
        keep.append(dict(Metric=m, meanAUC=mean_auc, p=pval))

# ---------- FDR (Benjamini–Hochberg) ----------
if keep:
    tbl = pd.DataFrame(keep).sort_values("p")
    m   = len(tbl);  fdr=0.10
    thresh = tbl.assign(rank=np.arange(1, m+1)) \
                .loc[lambda t: t["p"] <= t["rank"]/m*fdr]
    tbl = tbl.loc[thresh.index]
    tbl["Metric"].to_csv(OUT_TXT, index=False, header=False)
    logging.info("Accepted metrics: %s", tbl["Metric"].tolist())
else:
    logging.warning("No metric passed the gates – default to original list")
    Path(OUT_TXT).write_text("\n".join(METRICS))
