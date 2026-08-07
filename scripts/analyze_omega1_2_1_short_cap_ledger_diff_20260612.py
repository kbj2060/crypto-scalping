#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tmp/causal_regen_20260516/omega1_2_1_horizon_short_cap_fine_20260612"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega1_2_1_short_cap_ledger_diff_20260612"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(type(obj).__name__)


def _load(split: str, candidate_id: int) -> pd.DataFrame:
    path = SRC / f"{split}_candidate{candidate_id}_ledger.csv"
    df = pd.read_csv(path)
    df["entry_time"] = pd.to_datetime(df["entry_time"], errors="raise")
    df["exit_time"] = pd.to_datetime(df["exit_time"], errors="raise")
    df["candidate_id"] = int(candidate_id)
    return df


def _summary(df: pd.DataFrame) -> dict[str, Any]:
    ret = pd.to_numeric(df["net_trade_return_pct"], errors="raise")
    hold = pd.to_numeric(df["hold_bars"], errors="raise")
    return {
        "trades": int(len(df)),
        "sum_trade_pct": float(ret.sum()),
        "mean_trade_pct": float(ret.mean()) if len(ret) else 0.0,
        "median_trade_pct": float(ret.median()) if len(ret) else 0.0,
        "win_rate": float(ret.gt(0).mean()) if len(ret) else 0.0,
        "avg_hold": float(hold.mean()) if len(hold) else 0.0,
        "max_hold": int(hold.max()) if len(hold) else 0,
        "reasons": df["exit_reason"].astype(str).value_counts().to_dict(),
        "sides": df["side"].astype(str).value_counts().to_dict(),
    }


def _diff(split: str, aggressive_id: int = 3, stable_id: int = 21) -> dict[str, Any]:
    aggressive = _load(split, aggressive_id)
    stable = _load(split, stable_id)
    key = "entry_time"
    merged = stable.merge(
        aggressive,
        on=key,
        how="outer",
        suffixes=("_stable", "_aggressive"),
        indicator=True,
    ).sort_values(key)
    both = merged[merged["_merge"].eq("both")].copy()
    both["delta_trade_pct"] = pd.to_numeric(both["net_trade_return_pct_aggressive"], errors="raise") - pd.to_numeric(both["net_trade_return_pct_stable"], errors="raise")
    both["delta_hold"] = pd.to_numeric(both["hold_bars_aggressive"], errors="raise") - pd.to_numeric(both["hold_bars_stable"], errors="raise")
    changed = both[(both["delta_trade_pct"].abs() > 1e-9) | (both["delta_hold"].abs() > 0)]
    only_stable = merged[merged["_merge"].eq("left_only")]
    only_aggressive = merged[merged["_merge"].eq("right_only")]
    return {
        "split": split,
        "stable_summary": _summary(stable),
        "aggressive_summary": _summary(aggressive),
        "shared_trades": int(len(both)),
        "changed_shared_trades": int(len(changed)),
        "only_stable": int(len(only_stable)),
        "only_aggressive": int(len(only_aggressive)),
        "changed_rows": changed[
            [
                "entry_time",
                "side_stable",
                "exit_time_stable",
                "exit_time_aggressive",
                "exit_reason_stable",
                "exit_reason_aggressive",
                "net_trade_return_pct_stable",
                "net_trade_return_pct_aggressive",
                "delta_trade_pct",
                "hold_bars_stable",
                "hold_bars_aggressive",
                "delta_hold",
            ]
        ].to_dict(orient="records"),
        "only_stable_rows": only_stable[[c for c in only_stable.columns if c.endswith("_stable") or c == "entry_time"]].to_dict(orient="records"),
        "only_aggressive_rows": only_aggressive[[c for c in only_aggressive.columns if c.endswith("_aggressive") or c == "entry_time"]].to_dict(orient="records"),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "model_id": "omega1_2_1_short_cap_ledger_diff_20260612",
        "stable": "short_cap2000_min0.035",
        "aggressive": "short_cap1760_min0.035",
        "purpose": "Trade-level dependency analysis between stable and aggressive short-cap candidates.",
        "validation": _diff("validation"),
        "oos": _diff("oos"),
    }
    report_path = OUT_DIR / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "validation_changed": report["validation"]["changed_shared_trades"], "oos_changed": report["oos"]["changed_shared_trades"], "oos": report["oos"]}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
