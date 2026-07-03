from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
SEED = 20260703
THREADS = 1

NUMERIC_FEATURES = [
    "bp",
    "ep",
    "sp",
    "cfp",
    "reversal_5d",
    "momentum_60d",
    "volatility_20d",
    "volatility_60d",
    "skewness_60d",
    "turnover_rate",
    "avg_money_20d",
    "money_volatility_20d",
    "avg_volume_20d",
    "log_total_mv",
    "log_circ_mv",
    "roa",
    "gross_profit_margin",
    "net_profit_margin",
    "revenue_yoy",
    "operating_cash_flow_to_total_assets",
    "debt_to_assets",
]

LGB_CONFIGS = [
    {"n_estimators": 80, "learning_rate": 0.05, "num_leaves": 15, "max_depth": 3, "min_child_samples": 40, "subsample": 0.9, "colsample_bytree": 0.9, "reg_alpha": 0.0, "reg_lambda": 1.0},
    {"n_estimators": 120, "learning_rate": 0.03, "num_leaves": 15, "max_depth": 3, "min_child_samples": 40, "subsample": 0.9, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 2.0},
    {"n_estimators": 80, "learning_rate": 0.05, "num_leaves": 31, "max_depth": 4, "min_child_samples": 30, "subsample": 0.8, "colsample_bytree": 0.9, "reg_alpha": 0.0, "reg_lambda": 1.0},
    {"n_estimators": 120, "learning_rate": 0.03, "num_leaves": 31, "max_depth": 4, "min_child_samples": 30, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0},
    {"n_estimators": 160, "learning_rate": 0.02, "num_leaves": 31, "max_depth": 4, "min_child_samples": 50, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 2.0},
    {"n_estimators": 100, "learning_rate": 0.04, "num_leaves": 63, "max_depth": 5, "min_child_samples": 40, "subsample": 0.75, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.5},
    {"n_estimators": 140, "learning_rate": 0.025, "num_leaves": 63, "max_depth": 5, "min_child_samples": 60, "subsample": 0.75, "colsample_bytree": 0.75, "reg_alpha": 0.1, "reg_lambda": 2.0},
    {"n_estimators": 80, "learning_rate": 0.06, "num_leaves": 15, "max_depth": 2, "min_child_samples": 60, "subsample": 1.0, "colsample_bytree": 0.9, "reg_alpha": 0.0, "reg_lambda": 3.0},
    {"n_estimators": 100, "learning_rate": 0.04, "num_leaves": 31, "max_depth": 3, "min_child_samples": 80, "subsample": 0.9, "colsample_bytree": 0.7, "reg_alpha": 0.2, "reg_lambda": 2.0},
    {"n_estimators": 140, "learning_rate": 0.025, "num_leaves": 31, "max_depth": 4, "min_child_samples": 80, "subsample": 0.7, "colsample_bytree": 0.7, "reg_alpha": 0.2, "reg_lambda": 3.0},
    {"n_estimators": 180, "learning_rate": 0.02, "num_leaves": 15, "max_depth": 3, "min_child_samples": 50, "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.1, "reg_lambda": 1.5},
    {"n_estimators": 120, "learning_rate": 0.035, "num_leaves": 47, "max_depth": 5, "min_child_samples": 70, "subsample": 0.85, "colsample_bytree": 0.75, "reg_alpha": 0.2, "reg_lambda": 2.5},
]


def add_industry_dummies(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    dummies = pd.get_dummies(panel["sw_l1_name"].fillna("Unknown"), prefix="ind", dtype=float)
    out = pd.concat([panel.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    return out, NUMERIC_FEATURES + list(dummies.columns)


def safe_corr(x: pd.Series, y: pd.Series, spearman: bool = False) -> float:
    ok = x.notna() & y.notna()
    if ok.sum() < 5:
        return np.nan
    xv = x[ok].to_numpy(dtype=float)
    yv = y[ok].to_numpy(dtype=float)
    if spearman:
        xv = rankdata(xv)
        yv = rankdata(yv)
    if np.nanstd(xv) == 0 or np.nanstd(yv) == 0:
        return np.nan
    return float(np.corrcoef(xv, yv)[0, 1])


def weekly_metrics(frame: pd.DataFrame, pred_col: str = "prediction") -> pd.DataFrame:
    rows = []
    for date, group in frame.groupby("signal_date"):
        rows.append({
            "signal_date": date,
            "ic": safe_corr(group[pred_col], group["fwd_ret_5d"], spearman=False),
            "rankic": safe_corr(group[pred_col], group["fwd_ret_5d"], spearman=True),
            "n": len(group),
        })
    return pd.DataFrame(rows).sort_values("signal_date")


def summarize_ic(metrics: pd.DataFrame) -> dict[str, float]:
    ic = metrics["ic"].dropna()
    ric = metrics["rankic"].dropna()
    return {
        "mean_ic": float(ic.mean()) if len(ic) else np.nan,
        "std_ic": float(ic.std(ddof=1)) if len(ic) > 1 else np.nan,
        "icir_raw": float(ic.mean() / ic.std(ddof=1)) if len(ic) > 1 and ic.std(ddof=1) else np.nan,
        "icir_annualized": float(ic.mean() / ic.std(ddof=1) * math.sqrt(52)) if len(ic) > 1 and ic.std(ddof=1) else np.nan,
        "mean_rankic": float(ric.mean()) if len(ric) else np.nan,
        "std_rankic": float(ric.std(ddof=1)) if len(ric) > 1 else np.nan,
        "rankicir_raw": float(ric.mean() / ric.std(ddof=1)) if len(ric) > 1 and ric.std(ddof=1) else np.nan,
        "rankicir_annualized": float(ric.mean() / ric.std(ddof=1) * math.sqrt(52)) if len(ric) > 1 and ric.std(ddof=1) else np.nan,
        "positive_rankic_rate": float((ric > 0).mean()) if len(ric) else np.nan,
    }


def make_lgb(config: dict):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        objective="regression",
        random_state=SEED,
        n_jobs=THREADS,
        verbose=-1,
        **config,
    )
