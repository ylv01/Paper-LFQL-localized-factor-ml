from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_TRACKED_DATA = [
    "calendar.csv",
    "rebalance_calendar.csv",
    "stock_master.csv",
    "benchmark.csv",
    "data_contract.json",
    "data_manifest.csv",
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def main() -> int:
    problems = []
    notes = []
    data_dir = ROOT / "data"
    for name in REQUIRED_TRACKED_DATA:
        if not (data_dir / name).exists():
            problems.append(f"missing data/{name}")
    if len(NUMERIC_FEATURES) != 21:
        problems.append("numeric feature count is not 21")

    panel_path = data_dir / "processed" / "feature_panel.parquet"
    panel_available = panel_path.exists()
    if panel_available:
        panel = pd.read_parquet(panel_path)
        missing = [c for c in NUMERIC_FEATURES if c not in panel.columns]
        if missing:
            problems.append(f"feature_panel missing numeric features: {missing}")
        dummy_count = pd.get_dummies(panel["sw_l1_name"].fillna("Unknown"), prefix="ind", dtype=float).shape[1]
        if dummy_count != 32:
            problems.append(f"expected 32 industry dummies, got {dummy_count}")
        if len(NUMERIC_FEATURES) + dummy_count != 53:
            problems.append("total feature count is not 53")
    else:
        notes.append("processed panel not present; build it from local raw data before running the experiment")

    sums = data_dir / "SHA256SUMS.txt"
    if sums.exists():
        for line in sums.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected, name = line.split(maxsplit=1)
            actual = sha256_file(data_dir / name)
            if actual != expected:
                problems.append(f"sha256 mismatch: data/{name}")

    result = {"status": "PASS" if not problems else "FAILED", "problems": problems, "notes": notes}
    print(json.dumps(result, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
