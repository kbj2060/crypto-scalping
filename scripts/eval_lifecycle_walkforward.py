#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_lifecycle_ai_stress import LIFECYCLE_CONFIGS  # noqa: E402
from scripts.run_lifecycle_manager_grid import MODEL_COLS, backtest_lifecycle  # noqa: E402
from scripts.train_eval_fully_learned_governor import backtest_policy  # noqa: E402


DEFAULT_POLICY = ROOT / "data/ensemble/supervised/fully_learned_ai_combo_grid/patchtst__tide__dlinear.pkl"
DEFAULT_LIFECYCLE = ROOT / "data/ensemble/supervised/lifecycle_manager/patchtst_tide_dlinear_lifecycle_exit.pkl"
DEFAULT_TRAIN_CSV = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL_CSV = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_REPORT_OUT = ROOT / "data/ensemble/reports/lifecycle_walkforward_patchtst_tide_dlinear_2026.json"


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise KeyError(f"{path} missing timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return df.reset_index(drop=True)


def _compact(bt: dict[str, Any]) -> dict[str, Any]:
    return {
        k: bt.get(k)
        for k in (
            "pnl",
            "mdd",
            "trades",
            "wr",
            "trades_per_day",
            "long_entries",
            "short_entries",
            "avg_notional",
            "avg_leverage",
            "lifecycle_exits",
            "scale_downs",
            "scale_ups",
        )
        if k in bt
    }


def _audit(train_df: pd.DataFrame, eval_df: pd.DataFrame, lifecycle: dict[str, Any]) -> dict[str, Any]:
    train_ts = pd.to_datetime(train_df["timestamp"], errors="coerce")
    eval_ts = pd.to_datetime(eval_df["timestamp"], errors="coerce")
    feature_cols = list(lifecycle.get("feature_cols", []))
    suspicious = [
        c
        for c in feature_cols
        if any(token in str(c).lower() for token in ("future", "target", "label", "candidate", "exit_idx", "realized"))
    ]
    overlap = set(train_ts.astype("int64").tolist()) & set(eval_ts.astype("int64").tolist())
    return {
        "train_rows": int(len(train_df)),
        "eval_rows": int(len(eval_df)),
        "train_start": str(train_ts.min()),
        "train_end": str(train_ts.max()),
        "eval_start": str(eval_ts.min()),
        "eval_end": str(eval_ts.max()),
        "timestamp_overlap_rows": int(len(overlap)),
        "train_duplicate_timestamps": int(train_ts.duplicated().sum()),
        "eval_duplicate_timestamps": int(eval_ts.duplicated().sum()),
        "lifecycle_feature_count": int(len(feature_cols)),
        "expected_feature_count": int(len(MODEL_COLS)),
        "feature_count_match": bool(len(feature_cols) == len(MODEL_COLS)),
        "suspicious_lifecycle_features": suspicious,
        "sample_meta": lifecycle.get("sample_meta", {}),
    }


def _evaluate_slice(
    df: pd.DataFrame,
    policy: dict[str, Any],
    exit_model: Any,
    *,
    fee: float,
    slip: float,
    min_age: int,
    max_notional: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "entry_only": _compact(backtest_policy(df, policy, fee=fee, slip=slip)),
        "lifecycle": {},
    }
    for name, cfg in LIFECYCLE_CONFIGS.items():
        row["lifecycle"][name] = _compact(
            backtest_lifecycle(
                df,
                policy,
                exit_model,
                fee=fee,
                slip=slip,
                min_age=min_age,
                max_notional=max_notional,
                **cfg,
            )
        )
    return row


def _segments(df: pd.DataFrame, freq: str, *, min_rows: int) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    for key, idx in df.groupby(ts.dt.to_period(freq)).groups.items():
        part = df.loc[list(idx)].sort_values("timestamp").reset_index(drop=True)
        if len(part) >= int(min_rows):
            out[str(key)] = part
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward lifecycle manager validation on 2026 OOS slices.")
    p.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    p.add_argument("--lifecycle", type=Path, default=DEFAULT_LIFECYCLE)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--min-age", type=int, default=3)
    p.add_argument("--max-notional", type=float, default=3.60)
    p.add_argument("--min-week-rows", type=int, default=1000)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    policy = joblib.load(args.policy)
    lifecycle = joblib.load(args.lifecycle)
    exit_model = lifecycle["model"]
    train_df = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    report: dict[str, Any] = {
        "type": "lifecycle_walkforward_patchtst_tide_dlinear_2026",
        "policy": str(args.policy),
        "lifecycle": str(args.lifecycle),
        "train_csv": str(args.train_csv),
        "eval_csv": str(args.eval_csv),
        "audit": _audit(train_df, eval_df, lifecycle),
        "full_eval": _evaluate_slice(
            eval_df,
            policy,
            exit_model,
            fee=float(args.fee),
            slip=float(args.slip),
            min_age=int(args.min_age),
            max_notional=float(args.max_notional),
        ),
        "monthly": {},
        "weekly": {},
    }
    for name, part in _segments(eval_df, "M", min_rows=2000).items():
        report["monthly"][name] = _evaluate_slice(
            part,
            policy,
            exit_model,
            fee=float(args.fee),
            slip=float(args.slip),
            min_age=int(args.min_age),
            max_notional=float(args.max_notional),
        )
    for name, part in _segments(eval_df, "W", min_rows=int(args.min_week_rows)).items():
        report["weekly"][name] = _evaluate_slice(
            part,
            policy,
            exit_model,
            fee=float(args.fee),
            slip=float(args.slip),
            min_age=int(args.min_age),
            max_notional=float(args.max_notional),
        )
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.report_out),
                "audit": report["audit"],
                "full_eval": report["full_eval"],
                "monthly_keys": list(report["monthly"]),
                "weekly_keys": list(report["weekly"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
