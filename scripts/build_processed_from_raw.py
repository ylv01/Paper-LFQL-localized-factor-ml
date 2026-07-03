from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "processed"

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

PANEL_COLUMNS = [
    "code", "signal_date", "trade_date", "label_end_date", "period", "entry_open",
    "exit_close", "fwd_ret_5d", "pe_ttm", "pb", "ps_ttm", "pcf_ttm", "total_mv",
    "circ_mv", "year", *NUMERIC_FEATURES, "sw_l1_name", "bp_percentile",
]


def read_csv(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(DATA / name, **kwargs)


def read_dates(name: str, columns: list[str], **kwargs) -> pd.DataFrame:
    df = read_csv(name, usecols=columns, **kwargs)
    for col in columns:
        if col.endswith("date") or col in {"date", "asof_date", "ann_date", "end_date"}:
            df[col] = pd.to_datetime(df[col])
    return df


def add_period(signal_date: pd.Series) -> pd.Series:
    year = signal_date.dt.year
    return np.select(
        [year == 2021, year == 2022, year == 2023, year >= 2024],
        ["A", "B", "C", "D"],
        default="OUT",
    )


def build_label_dates(rebalance: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    open_days = calendar.loc[calendar["is_open"] == 1, "date"].sort_values().reset_index(drop=True)
    day_to_pos = {day: idx for idx, day in enumerate(open_days)}

    def fifth_session(trade_date: pd.Timestamp) -> pd.Timestamp | pd.NaT:
        pos = day_to_pos.get(trade_date)
        if pos is None or pos + 4 >= len(open_days):
            return pd.NaT
        return open_days.iloc[pos + 4]

    out = rebalance[["signal_date", "trade_date"]].copy()
    out["label_end_date"] = out["trade_date"].map(fifth_session)
    return out.dropna(subset=["label_end_date"])


def build_base_panel() -> pd.DataFrame:
    calendar = read_dates("calendar.csv", ["date", "is_open"])
    rebalance = read_dates("rebalance_calendar.csv", ["signal_date", "trade_date"])
    rebalance = rebalance[
        (rebalance["signal_date"] >= "2021-01-01")
        & (rebalance["signal_date"] <= "2026-07-01")
    ].copy()
    rebalance = build_label_dates(rebalance, calendar)

    members = read_dates("csi1000_membership.csv", ["date", "code", "is_member"])
    members = members[members["is_member"].eq(True) | members["is_member"].eq(1)].copy()
    members = members.rename(columns={"date": "signal_date"})
    members = members.merge(rebalance, on="signal_date", how="inner")

    bar_pre = read_dates("daily_bar_pre.csv", ["date", "code", "open", "close", "pre_close"])
    entry = bar_pre[["date", "code", "open"]].rename(
        columns={"date": "trade_date", "open": "entry_open"}
    )
    exit_ = bar_pre[["date", "code", "close"]].rename(
        columns={"date": "label_end_date", "close": "exit_close"}
    )
    panel = members.merge(entry, on=["trade_date", "code"], how="left")
    panel = panel.merge(exit_, on=["label_end_date", "code"], how="left")
    panel["fwd_ret_5d"] = panel["exit_close"] / panel["entry_open"] - 1.0
    return panel.dropna(subset=["entry_open", "exit_close", "fwd_ret_5d"])


def add_raw_fundamental_features(panel: pd.DataFrame) -> pd.DataFrame:
    basic = read_dates(
        "daily_basic.csv",
        ["date", "code", "pe_ttm", "pb", "ps_ttm", "pcf_ttm", "total_mv", "circ_mv", "turnover_rate"],
    ).rename(columns={"date": "signal_date"})
    panel = panel.merge(basic, on=["signal_date", "code"], how="left")

    fin = read_dates(
        "financial_pit.csv",
        [
            "asof_date", "code", "roa", "net_profit_margin", "gross_profit_margin",
            "inc_revenue_year_on_year", "total_assets", "total_liability",
            "net_operate_cash_flow",
        ],
    ).rename(columns={"asof_date": "signal_date"})
    panel = panel.merge(fin, on=["signal_date", "code"], how="left")

    industry = read_dates("industry_pit.csv", ["asof_date", "code", "sw_l1_name"]).rename(
        columns={"asof_date": "signal_date"}
    )
    panel = panel.merge(industry, on=["signal_date", "code"], how="left")

    panel["bp"] = np.where(panel["pb"] > 0, 1.0 / panel["pb"], np.nan)
    panel["ep"] = np.where(panel["pe_ttm"] > 0, 1.0 / panel["pe_ttm"], np.nan)
    panel["sp"] = np.where(panel["ps_ttm"] > 0, 1.0 / panel["ps_ttm"], np.nan)
    panel["cfp"] = np.where(panel["pcf_ttm"] > 0, 1.0 / panel["pcf_ttm"], np.nan)
    panel["log_total_mv"] = np.where(panel["total_mv"] > 0, np.log(panel["total_mv"]), np.nan)
    panel["log_circ_mv"] = np.where(panel["circ_mv"] > 0, np.log(panel["circ_mv"]), np.nan)
    panel["revenue_yoy"] = panel["inc_revenue_year_on_year"]
    panel["operating_cash_flow_to_total_assets"] = (
        panel["net_operate_cash_flow"] / panel["total_assets"].replace(0, np.nan)
    )
    panel["debt_to_assets"] = panel["total_liability"] / panel["total_assets"].replace(0, np.nan)
    return panel


def rolling_last_on_signal(rolling: pd.DataFrame, signal_dates: pd.Series) -> pd.DataFrame:
    signal_index = pd.DataFrame({"signal_date": pd.to_datetime(signal_dates.unique())}).sort_values("signal_date")
    frames = []
    for code, group in rolling.groupby("code", sort=False):
        merged = pd.merge_asof(
            signal_index,
            group.sort_values("date"),
            left_on="signal_date",
            right_on="date",
            direction="backward",
        )
        merged["code"] = code
        frames.append(merged.drop(columns=["date"]))
    return pd.concat(frames, ignore_index=True)


def add_technical_features(panel: pd.DataFrame) -> pd.DataFrame:
    signal_dates = panel["signal_date"].drop_duplicates()

    pre = read_dates("daily_bar_pre.csv", ["date", "code", "close", "pre_close"]).sort_values(["code", "date"])
    pre["ret"] = pre["close"] / pre["pre_close"] - 1.0
    grouped = pre.groupby("code", group_keys=False)
    tech = pre[["date", "code"]].copy()
    tech["reversal_5d"] = -grouped["close"].pct_change(5)
    tech["momentum_60d"] = grouped["close"].pct_change(60)
    tech["volatility_20d"] = grouped["ret"].rolling(20, min_periods=10).std().reset_index(level=0, drop=True)
    tech["volatility_60d"] = grouped["ret"].rolling(60, min_periods=20).std().reset_index(level=0, drop=True)
    tech["skewness_60d"] = grouped["ret"].rolling(60, min_periods=20).skew().reset_index(level=0, drop=True)

    bar = read_dates("daily_bar.csv", ["date", "code", "money", "volume"]).sort_values(["code", "date"])
    bar_grouped = bar.groupby("code", group_keys=False)
    liq = bar[["date", "code"]].copy()
    liq["avg_money_20d"] = bar_grouped["money"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
    liq["money_volatility_20d"] = bar_grouped["money"].rolling(20, min_periods=10).std().reset_index(level=0, drop=True)
    liq["avg_volume_20d"] = bar_grouped["volume"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)

    tech_signal = rolling_last_on_signal(tech, signal_dates)
    liq_signal = rolling_last_on_signal(liq, signal_dates)
    panel = panel.merge(tech_signal, on=["signal_date", "code"], how="left")
    panel = panel.merge(liq_signal, on=["signal_date", "code"], how="left")
    return panel


def winsorize_impute(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel[NUMERIC_FEATURES] = panel[NUMERIC_FEATURES].replace([np.inf, -np.inf], np.nan)
    for _, idx in panel.groupby("signal_date", sort=False).groups.items():
        block = panel.loc[idx, NUMERIC_FEATURES]
        lower = block.quantile(0.01)
        upper = block.quantile(0.99)
        clipped = block.clip(lower=lower, upper=upper, axis=1)
        med = clipped.median()
        panel.loc[idx, NUMERIC_FEATURES] = clipped.fillna(med)
    panel["bp_percentile"] = panel.groupby("signal_date")["bp"].rank(method="average", pct=True)
    return panel


def build_feature_panel() -> pd.DataFrame:
    panel = build_base_panel()
    panel = add_raw_fundamental_features(panel)
    panel = add_technical_features(panel)
    panel["signal_date"] = pd.to_datetime(panel["signal_date"])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel["label_end_date"] = pd.to_datetime(panel["label_end_date"])
    panel["year"] = panel["signal_date"].dt.year
    panel["period"] = add_period(panel["signal_date"])
    panel = panel[panel["period"].ne("OUT")].copy()
    panel = winsorize_impute(panel)
    panel = panel.sort_values(["signal_date", "code"]).reset_index(drop=True)
    return panel[PANEL_COLUMNS]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build processed LFQL feature panel from raw CSV files.")
    parser.add_argument("--force", action="store_true", help="Overwrite data/processed/feature_panel.parquet if present.")
    args = parser.parse_args()

    out_path = OUT / "feature_panel.parquet"
    if out_path.exists() and not args.force:
        print(f"{out_path.relative_to(ROOT)} already exists; use --force to rebuild")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    panel = build_feature_panel()
    panel.to_parquet(out_path, index=False)

    manifest = {
        "feature_panel": str(out_path.relative_to(ROOT)).replace("\\", "/"),
        "rows": int(len(panel)),
        "columns": list(panel.columns),
        "signal_date_min": str(panel["signal_date"].min().date()),
        "signal_date_max": str(panel["signal_date"].max().date()),
        "note": "Generated from raw CSV files by scripts/build_processed_from_raw.py.",
    }
    (OUT / "feature_panel_build_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out_path.relative_to(ROOT)} rows={len(panel)} cols={len(panel.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
