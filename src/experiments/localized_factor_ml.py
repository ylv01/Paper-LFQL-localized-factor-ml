from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.lfql_common import LGB_CONFIGS, ROOT, SEED, THREADS, add_industry_dummies, make_lgb, safe_corr, summarize_ic, weekly_metrics


OUT_DIR = ROOT / "outputs" / "localized_factor_ml"
TABLE_DIR = OUT_DIR / "tables"
REPORT_DIR = OUT_DIR / "reports"
MODEL_DIR = OUT_DIR / "models"
PRED_DIR = OUT_DIR / "predictions"
LOG_DIR = OUT_DIR / "logs"
CONFIGS_DIR = ROOT / "configs"
PANEL_PATH = ROOT / "data" / "processed" / "feature_panel.parquet"

P_GRID = [10, 20, 30, 40, 50, 60, 70, 80, 90]
TRAIN_HALF_WIDTH = 0.15
CORE_HALF_WIDTH = 0.03
BOOTSTRAP_REPS = 2000
BOOTSTRAP_BLOCK = 4

CONDITION_FACTORS = [
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
FACTOR_GROUPS = {
    "valuation": ["bp", "ep", "sp", "cfp"],
    "momentum_reversal": ["reversal_5d", "momentum_60d"],
    "risk_volatility": ["volatility_20d", "volatility_60d", "skewness_60d"],
    "liquidity_activity": ["turnover_rate", "avg_money_20d", "money_volatility_20d", "avg_volume_20d"],
    "size": ["log_total_mv", "log_circ_mv"],
    "quality_growth": ["roa", "gross_profit_margin", "net_profit_margin", "revenue_yoy", "operating_cash_flow_to_total_assets", "debt_to_assets"],
}
GROUP_BY_FACTOR = {factor: group for group, factors in FACTOR_GROUPS.items() for factor in factors}


def ensure_dirs() -> None:
    for path in [TABLE_DIR, REPORT_DIR, MODEL_DIR, PRED_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    stamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{stamp} - {message}"
    print(line, flush=True)
    with (LOG_DIR / "run_log.txt").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def maybe_write_data_manifests() -> None:
    data_dir = ROOT / "data"
    files = sorted(path for path in data_dir.glob("*.csv") if path.is_file())
    rows = []
    sums = []
    for path in files:
        digest = sha256_file(path)
        rows.append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": digest})
        sums.append(f"{digest}  {path.name}")
    contract = data_dir / "data_contract.json"
    if contract.exists():
        rows.append({"path": "data/data_contract.json", "bytes": contract.stat().st_size, "sha256": sha256_file(contract)})
    processed = data_dir / "processed" / "feature_panel.parquet"
    if processed.exists():
        rows.append({"path": "data/processed/feature_panel.parquet", "bytes": processed.stat().st_size, "sha256": sha256_file(processed)})
    pd.DataFrame(rows).to_csv(data_dir / "file_inventory.csv", index=False)
    (data_dir / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    readme = (
        "# Data\n\n"
        "This directory tracks lightweight reference inputs and metadata. Large raw data exports "
        "and processed panels are local inputs and are ignored by Git.\n"
    )
    (data_dir / "README.md").write_text(readme, encoding="utf-8")

def load_panel() -> tuple[pd.DataFrame, list[str]]:
    panel = pd.read_parquet(PANEL_PATH)
    for col in ["signal_date", "trade_date", "label_end_date"]:
        panel[col] = pd.to_datetime(panel[col])
    missing = [feature for feature in NUMERIC_FEATURES if feature not in panel.columns]
    if missing:
        raise RuntimeError(f"feature_panel missing model features: {missing}")
    panel, feature_cols = add_industry_dummies(panel)
    dummy_cols = [c for c in feature_cols if c.startswith("ind_")]
    if feature_cols[:21] != NUMERIC_FEATURES or len(dummy_cols) != 32 or len(feature_cols) != 53:
        raise RuntimeError(f"Expected 21 numeric + 32 industry dummies = 53 features, got {len(feature_cols)}")
    return panel, feature_cols


def split_panel(panel: pd.DataFrame):
    a = panel[panel["period"].eq("A")].copy()
    b = panel[panel["period"].eq("B")].copy()
    c = panel[panel["period"].eq("C")].copy()
    c_train = c[c["signal_date"] <= pd.Timestamp("2023-09-30")].copy()
    c_val = c[c["signal_date"] >= pd.Timestamp("2023-10-01")].copy()
    d = panel[panel["period"].eq("D")].copy()
    if min(len(a), len(b), len(c), len(c_train), len(c_val), len(d)) == 0:
        raise RuntimeError("A/B/C/C-train/C-val/D split contains an empty block.")
    return a, b, c, c_train, c_val, d


def q2021(a_panel: pd.DataFrame, factor: str, q: float) -> float:
    values = a_panel[factor].replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        raise RuntimeError(f"No A-period valid values for {factor}")
    if q <= 0:
        return float(values.min())
    if q >= 1:
        return float(values.max())
    return float(values.quantile(q))


def frozen_threshold_grid(a_panel: pd.DataFrame, factor: str) -> tuple[pd.DataFrame, dict[int, float]]:
    rows = []
    thresholds = {}
    values = a_panel[factor].replace([np.inf, -np.inf], np.nan).dropna()
    for p in P_GRID:
        val = q2021(a_panel, factor, p / 100)
        thresholds[p] = val
        rows.append({
            "condition_factor": factor,
            "percentile": p,
            "frozen_threshold_raw_value": val,
            "a_valid_count": int(values.count()),
            "a_mean": float(values.mean()),
            "a_std": float(values.std(ddof=1)),
        })
    return pd.DataFrame(rows), thresholds


def threshold_search_2022(b_panel: pd.DataFrame, factor: str, thresholds: dict[int, float]) -> tuple[pd.DataFrame, dict]:
    rows = []
    for p, threshold in thresholds.items():
        weekly = []
        high_cov = []
        low_cov = []
        rankics = []
        ics = []
        for _, group in b_panel.groupby("signal_date"):
            valid = group[group[factor].notna()]
            if len(valid) < 10:
                continue
            high = valid[valid[factor] > threshold]
            low = valid[valid[factor] <= threshold]
            high_cov.append(len(high) / len(valid))
            low_cov.append(len(low) / len(valid))
            if len(high) and len(low):
                weekly.append(float(high["fwd_ret_5d"].mean() - low["fwd_ret_5d"].mean()))
            rankics.append(safe_corr(valid[factor], valid["fwd_ret_5d"], spearman=True))
            ics.append(safe_corr(valid[factor], valid["fwd_ret_5d"], spearman=False))
        ls = pd.Series(weekly, dtype=float).dropna()
        mean_ls = float(ls.mean()) if len(ls) else np.nan
        std_ls = float(ls.std(ddof=1)) if len(ls) > 1 else np.nan
        ir_raw = float(mean_ls / std_ls) if std_ls and pd.notna(std_ls) else np.nan
        rows.append({
            "condition_factor": factor,
            "threshold_percentile": p,
            "threshold_raw_value": threshold,
            "high_group_coverage": float(np.nanmean(high_cov)) if high_cov else np.nan,
            "low_group_coverage": float(np.nanmean(low_cov)) if low_cov else np.nan,
            "mean_ls_return": mean_ls,
            "std_ls_return": std_ls,
            "ls_ir_raw": ir_raw,
            "ls_ir_annualized": ir_raw * math.sqrt(52) if pd.notna(ir_raw) else np.nan,
            "positive_ls_rate": float((ls > 0).mean()) if len(ls) else np.nan,
            "mean_rankic_raw_factor": float(pd.Series(rankics, dtype=float).mean()),
            "mean_ic_raw_factor": float(pd.Series(ics, dtype=float).mean()),
        })
    grid = pd.DataFrame(rows)
    if grid["ls_ir_raw"].notna().sum() == 0:
        raise RuntimeError(f"All threshold candidates have invalid LS IR for {factor}.")
    grid["balance_gap"] = (grid["high_group_coverage"] - grid["low_group_coverage"]).abs()
    grid = grid.sort_values(["ls_ir_raw", "mean_ls_return", "balance_gap", "threshold_percentile"], ascending=[False, False, True, True])
    selected_index = grid.index[0]
    grid["selected_flag"] = grid.index == selected_index
    selected = grid.loc[selected_index].to_dict()
    return grid.sort_values("threshold_percentile").drop(columns=["balance_gap"]), selected


def interval_bounds(a_panel: pd.DataFrame, factor: str, pstar: float, half_width: float) -> tuple[float, float, float, float]:
    lower_p = max(0.0, pstar - half_width)
    upper_p = min(1.0, pstar + half_width)
    lower_raw = q2021(a_panel, factor, lower_p)
    upper_raw = q2021(a_panel, factor, upper_p)
    if lower_raw > upper_raw:
        lower_raw, upper_raw = upper_raw, lower_raw
    if lower_raw == upper_raw:
        raise RuntimeError(f"Degenerate interval for {factor} at width {half_width}")
    return lower_p, upper_p, lower_raw, upper_raw


def interval_mask(frame: pd.DataFrame, factor: str, lower: float, upper: float) -> pd.Series:
    return frame[factor].between(min(lower, upper), max(lower, upper), inclusive="both")


def fit_lgb(train: pd.DataFrame, val: pd.DataFrame, full: pd.DataFrame, feature_cols: list[str]) -> dict:
    start = time.perf_counter()
    best_summary = None
    best_id = None
    fits = 0
    for i, config in enumerate(LGB_CONFIGS):
        model = make_lgb(config)
        model.fit(train[feature_cols], train["fwd_ret_5d"])
        fits += 1
        pred = val[["signal_date", "fwd_ret_5d"]].copy()
        pred["prediction"] = model.predict(val[feature_cols])
        summary = summarize_ic(weekly_metrics(pred))
        if best_summary is None or (pd.notna(summary["rankicir_raw"]) and summary["rankicir_raw"] > best_summary["rankicir_raw"]):
            best_summary = summary
            best_id = i
    if best_id is None:
        raise RuntimeError("No valid LightGBM config produced validation metrics.")
    final_start = time.perf_counter()
    final_model = make_lgb(LGB_CONFIGS[best_id])
    final_model.fit(full[feature_cols], full["fwd_ret_5d"])
    fits += 1
    final_fit_seconds = time.perf_counter() - final_start
    return {
        "model": final_model,
        "best_config_id": best_id,
        "validation": best_summary,
        "model_fits": fits,
        "tuning_time_seconds": time.perf_counter() - start - final_fit_seconds,
        "final_fit_time_seconds": final_fit_seconds,
    }


def moving_block_bootstrap(diff: pd.Series) -> dict:
    values = diff.dropna().to_numpy(dtype=float)
    n = len(values)
    if n < BOOTSTRAP_BLOCK * 3:
        return {"mean": np.nan, "ci": "", "pvalue": np.nan}
    rng = np.random.default_rng(SEED)
    starts = np.arange(0, n - BOOTSTRAP_BLOCK + 1)
    reps = []
    for _ in range(BOOTSTRAP_REPS):
        blocks = []
        while len(blocks) * BOOTSTRAP_BLOCK < n:
            s = rng.choice(starts)
            blocks.append(values[s:s + BOOTSTRAP_BLOCK])
        reps.append(np.concatenate(blocks)[:n].mean())
    reps = np.asarray(reps)
    lo, hi = np.quantile(reps, [0.025, 0.975])
    pvalue = 2 * min((reps <= 0).mean(), (reps >= 0).mean())
    return {"mean": float(values.mean()), "ci": f"[{lo:.6f}, {hi:.6f}]", "pvalue": float(min(pvalue, 1.0))}


def eval_pred(pred: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    metrics = weekly_metrics(pred)
    summary = summarize_ic(metrics)
    summary["date_count"] = int(metrics["rankic"].notna().sum())
    summary["prediction_rows"] = int(len(pred))
    return summary, metrics


def save_model(model, out: Path, params: dict, metadata: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    booster = model.booster_
    names = booster.feature_name()
    booster.save_model(out / "lightgbm_booster.txt")
    (out / "feature_names.json").write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame({"feature_name": names, "importance_gain": booster.feature_importance(importance_type="gain")}).to_csv(out / "feature_importance_gain.csv", index=False)
    pd.DataFrame({"feature_name": names, "importance_split": booster.feature_importance(importance_type="split")}).to_csv(out / "feature_importance_split.csv", index=False)
    (out / "best_params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    (out / "training_log.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def rolling_rows(pred: pd.DataFrame, factor: str, method: str, universe: str) -> pd.DataFrame:
    metrics = weekly_metrics(pred)
    counts = pred.groupby("signal_date").size()
    metrics["condition_factor"] = factor
    metrics["method"] = method
    metrics["evaluation_universe"] = universe
    metrics["candidate_count"] = metrics["signal_date"].map(counts).fillna(0).astype(int)
    return metrics[["condition_factor", "method", "signal_date", "evaluation_universe", "candidate_count", "ic", "rankic"]]


def main() -> int:
    started = time.perf_counter()
    ensure_dirs()
    maybe_write_data_manifests()
    protocol_hash = sha256_text((CONFIGS_DIR / "localized_factor_ml_config.yaml").read_text(encoding="utf-8"))
    config_hash = sha256_text((CONFIGS_DIR / "feature_set.yaml").read_text(encoding="utf-8"))
    panel, feature_cols = load_panel()
    dummy_cols = [c for c in feature_cols if c.startswith("ind_")]
    a, b, c, c_train, c_val, d = split_panel(panel)
    log(f"Loaded panel rows={len(panel)} final condition factors={len(CONDITION_FACTORS)}")

    b0_fit = fit_lgb(c_train, c_val, c, feature_cols)
    save_model(
        b0_fit["model"],
        MODEL_DIR / "global_ml",
        LGB_CONFIGS[b0_fit["best_config_id"]],
        {"model": "B0_Global_ML", "validation": b0_fit["validation"], "protocol_hash": protocol_hash, "config_hash": config_hash},
    )
    b0_d = d[["signal_date", "code", "fwd_ret_5d"]].copy()
    b0_d["prediction"] = b0_fit["model"].predict(d[feature_cols])
    b0_d.to_parquet(PRED_DIR / "global_ml_d_predictions.parquet", index=False)

    threshold_rows = []
    search_rows = []
    summary_rows = []
    audit_rows = []
    rolling = []

    for factor in CONDITION_FACTORS:
        item_start = time.perf_counter()
        log(f"Final LFQL factor={factor}")
        grid_a, thresholds = frozen_threshold_grid(a, factor)
        threshold_rows.append(grid_a)
        search_b, selected = threshold_search_2022(b, factor, thresholds)
        search_rows.append(search_b)
        pstar = float(selected["threshold_percentile"]) / 100.0
        train_lp, train_up, train_lo, train_hi = interval_bounds(a, factor, pstar, TRAIN_HALF_WIDTH)
        core_lp, core_up, core_lo, core_hi = interval_bounds(a, factor, pstar, CORE_HALF_WIDTH)
        train_local = c_train[interval_mask(c_train, factor, train_lo, train_hi)].copy()
        val_local = c_val[interval_mask(c_val, factor, train_lo, train_hi)].copy()
        full_local = c[interval_mask(c, factor, train_lo, train_hi)].copy()
        d_self = d[interval_mask(d, factor, train_lo, train_hi)].copy()
        d_core = d[interval_mask(d, factor, core_lo, core_hi)].copy()
        if min(len(train_local), len(val_local), len(full_local), len(d_self), len(d_core)) < 100:
            raise RuntimeError(f"Insufficient local rows for {factor}")
        fit = fit_lgb(train_local, val_local, full_local, feature_cols)
        save_model(
            fit["model"],
            MODEL_DIR / factor,
            LGB_CONFIGS[fit["best_config_id"]],
            {
                "condition_factor": factor,
                "train_half_width": TRAIN_HALF_WIDTH,
                "core_evaluation_half_width": CORE_HALF_WIDTH,
                "train_interval": [train_lo, train_hi],
                "core_interval": [core_lo, core_hi],
                "validation": fit["validation"],
                "protocol_hash": protocol_hash,
                "config_hash": config_hash,
            },
        )
        self_pred = d_self[["signal_date", "code", "fwd_ret_5d"]].copy()
        self_pred["prediction"] = fit["model"].predict(d_self[feature_cols])
        core_pred = d_core[["signal_date", "code", "fwd_ret_5d"]].copy()
        core_pred["prediction"] = fit["model"].predict(d_core[feature_cols])
        self_pred.to_parquet(PRED_DIR / f"{factor}_self_local_d_predictions.parquet", index=False)
        core_pred.to_parquet(PRED_DIR / f"{factor}_core_d_predictions.parquet", index=False)

        b0_self = self_pred[["signal_date", "code", "fwd_ret_5d"]].merge(b0_d[["signal_date", "code", "prediction"]], on=["signal_date", "code"], how="inner")
        b0_core = core_pred[["signal_date", "code", "fwd_ret_5d"]].merge(b0_d[["signal_date", "code", "prediction"]], on=["signal_date", "code"], how="inner")
        if len(b0_self) != len(self_pred) or len(b0_core) != len(core_pred):
            raise RuntimeError(f"B0 alignment failed for {factor}")
        self_summary, self_metrics = eval_pred(self_pred)
        core_summary, core_metrics = eval_pred(core_pred)
        b0_self_summary, b0_self_metrics = eval_pred(b0_self)
        b0_core_summary, b0_core_metrics = eval_pred(b0_core)
        self_boot = moving_block_bootstrap(self_metrics.set_index("signal_date")["rankic"] - b0_self_metrics.set_index("signal_date")["rankic"])
        core_boot = moving_block_bootstrap(core_metrics.set_index("signal_date")["rankic"] - b0_core_metrics.set_index("signal_date")["rankic"])

        rolling.append(rolling_rows(core_pred, factor, "LFQL", "core_width_003"))
        rolling.append(rolling_rows(b0_core, factor, "B0_Global_ML", "core_width_003"))
        rolling.append(rolling_rows(self_pred, factor, "LFQL", "self_local_width_015"))
        rolling.append(rolling_rows(b0_self, factor, "B0_Global_ML", "self_local_width_015"))

        summary_rows.append({
            "condition_factor": factor,
            "factor_group": GROUP_BY_FACTOR[factor],
            "status": "PASS",
            "chosen_threshold_percentile": selected["threshold_percentile"],
            "chosen_threshold_raw_value": selected["threshold_raw_value"],
            "train_half_width": TRAIN_HALF_WIDTH,
            "core_evaluation_half_width": CORE_HALF_WIDTH,
            "c_train_rows": int(len(train_local)),
            "c_local_coverage": float(len(full_local) / len(c)),
            "d_core_mean_ic": core_summary["mean_ic"],
            "d_core_icir_raw": core_summary["icir_raw"],
            "d_core_mean_rankic": core_summary["mean_rankic"],
            "d_core_rankicir_raw": core_summary["rankicir_raw"],
            "d_core_b0_mean_rankic": b0_core_summary["mean_rankic"],
            "d_core_b0_rankicir_raw": b0_core_summary["rankicir_raw"],
            "d_core_lfql_minus_b0_mean_rankic": core_boot["mean"],
            "d_core_lfql_minus_b0_mean_rankic_ci_95": core_boot["ci"],
            "d_core_lfql_minus_b0_mean_rankic_pvalue": core_boot["pvalue"],
            "d_self_mean_ic": self_summary["mean_ic"],
            "d_self_icir_raw": self_summary["icir_raw"],
            "d_self_mean_rankic": self_summary["mean_rankic"],
            "d_self_rankicir_raw": self_summary["rankicir_raw"],
            "d_self_b0_mean_rankic": b0_self_summary["mean_rankic"],
            "d_self_b0_rankicir_raw": b0_self_summary["rankicir_raw"],
            "d_self_lfql_minus_b0_mean_rankic": self_boot["mean"],
            "d_self_lfql_minus_b0_mean_rankic_ci_95": self_boot["ci"],
            "d_self_lfql_minus_b0_mean_rankic_pvalue": self_boot["pvalue"],
            "model_fits": fit["model_fits"],
            "configurations_considered": len(LGB_CONFIGS),
            "total_runtime_seconds": time.perf_counter() - item_start,
            "peak_memory_mb": np.nan,
        })
        audit_rows.append({
            "condition_factor": factor,
            "lfql_numeric_feature_count": len(NUMERIC_FEATURES),
            "lfql_industry_dummy_count": len(dummy_cols),
            "lfql_total_feature_count": len(feature_cols),
            "b0_total_feature_count": len(feature_cols),
            "condition_factor_used_as_ml_input": factor in feature_cols,
            "lfql_b0_feature_columns_identical": True,
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(TABLE_DIR / "factor_summary.csv", index=False)
    agg = {
        "factor_count": int(len(summary)),
        "positive_d_core_delta_count": int((summary["d_core_lfql_minus_b0_mean_rankic"] > 0).sum()),
        "positive_d_core_delta_share": float((summary["d_core_lfql_minus_b0_mean_rankic"] > 0).mean()),
        "mean_d_core_delta_rankic": float(summary["d_core_lfql_minus_b0_mean_rankic"].mean()),
        "median_d_core_delta_rankic": float(summary["d_core_lfql_minus_b0_mean_rankic"].median()),
        "positive_d_self_delta_count": int((summary["d_self_lfql_minus_b0_mean_rankic"] > 0).sum()),
        "positive_d_self_delta_share": float((summary["d_self_lfql_minus_b0_mean_rankic"] > 0).mean()),
        "mean_d_self_delta_rankic": float(summary["d_self_lfql_minus_b0_mean_rankic"].mean()),
        "median_d_self_delta_rankic": float(summary["d_self_lfql_minus_b0_mean_rankic"].median()),
        "mean_d_core_rankic": float(summary["d_core_mean_rankic"].mean()),
        "median_d_core_rankic": float(summary["d_core_mean_rankic"].median()),
        "mean_d_self_rankic": float(summary["d_self_mean_rankic"].mean()),
        "median_d_self_rankic": float(summary["d_self_mean_rankic"].median()),
    }
    pd.DataFrame([agg]).to_csv(TABLE_DIR / "factor_aggregate.csv", index=False)
    pd.concat(rolling, ignore_index=True).to_csv(TABLE_DIR / "rolling_metrics.csv", index=False)
    pd.concat(threshold_rows, ignore_index=True).to_csv(TABLE_DIR / "threshold_grid_2021.csv", index=False)
    pd.concat(search_rows, ignore_index=True).to_csv(TABLE_DIR / "threshold_search_2022.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(TABLE_DIR / "model_feature_audit.csv", index=False)

    report = [
        "# Localized Factor ML Report",
        "",
        "This run uses 21 numeric factors with a fixed local training half-width of +/-15 percentage points.",
        "",
        "The fixed protocol is: 21 condition factors, +/-15% local training width, +/-3% core evaluation width, and a 53-column LightGBM design matrix for both LFQL and Global ML.",
        "",
        "Future confirmation should freeze this protocol before testing on a new external market or future unseen period.",
        "",
        f"Factor count: {len(summary)}",
        f"Positive core LFQL-B0 delta count: {agg['positive_d_core_delta_count']}",
        f"Mean core LFQL-B0 RankIC delta: {agg['mean_d_core_delta_rankic']:.6f}",
    ]
    (REPORT_DIR / "localized_factor_ml_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    release_manifest = {
        "factor_count": 21,
        "numeric_feature_count": len(NUMERIC_FEATURES),
        "industry_dummy_count": len(dummy_cols),
        "total_feature_count": len(feature_cols),
        "train_half_width": TRAIN_HALF_WIDTH,
        "core_width": CORE_HALF_WIDTH,
        "data_reproducible": True,
        "protocol_status": "exploratory",
    }
    manifest = {
        "status": "complete",
        "version": "localized_factor_ml",
        "runtime_seconds": time.perf_counter() - started,
        "python": sys.version,
        "platform": platform.platform(),
        "processor_count": os.cpu_count(),
        "lightgbm": __import__("lightgbm").__version__,
        "seed": SEED,
        "threads": THREADS,
        "condition_factor_count": len(CONDITION_FACTORS),
        "train_half_width": TRAIN_HALF_WIDTH,
        "core_evaluation_half_width": CORE_HALF_WIDTH,
        "protocol_status": "exploratory",
        "protocol_hash": protocol_hash,
        "config_hash": config_hash,
    }
    (LOG_DIR / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (LOG_DIR / "release_manifest.json").write_text(json.dumps(release_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log("localized_factor_ml complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
