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

from scripts.run_lifecycle_manager_grid import MODEL_COLS, backtest_lifecycle  # noqa: E402
from scripts.train_eval_fully_learned_governor import backtest_policy  # noqa: E402


DEFAULT_POLICY = ROOT / "data/ensemble/supervised/fully_learned_ai_combo_grid/patchtst__tide__dlinear.pkl"
DEFAULT_LIFECYCLE = ROOT / "data/ensemble/supervised/lifecycle_manager/patchtst_tide_dlinear_lifecycle_exit.pkl"
DEFAULT_TRAIN_CSV = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL_CSV = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_REPORT_OUT = ROOT / "data/ensemble/reports/lifecycle_guard_search_patchtst_tide_dlinear_2026.json"


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


def _configs() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [
        {
            "name": "scaledown_control",
            "exit_threshold": 0.85,
            "scale_threshold": 0.65,
            "scale_multiplier": 0.50,
            "scale_up_threshold": None,
            "scale_up_multiplier": 1.0,
        },
        {
            "name": "scaleup_control",
            "exit_threshold": 0.85,
            "scale_threshold": None,
            "scale_multiplier": 1.0,
            "scale_up_threshold": 0.10,
            "scale_up_multiplier": 1.25,
        },
    ]
    for dd in (0.04, 0.06, 0.08, 0.10):
        out.append(
            {
                "name": f"scaleup_guard_acctdd_{dd:.2f}",
                "exit_threshold": 0.85,
                "scale_threshold": None,
                "scale_multiplier": 1.0,
                "scale_up_threshold": 0.10,
                "scale_up_multiplier": 1.25,
                "scale_up_max_account_dd": dd,
            }
        )
    for pos_dd in (0.010, 0.020, 0.035):
        out.append(
            {
                "name": f"scaleup_guard_posdd_{pos_dd:.3f}",
                "exit_threshold": 0.85,
                "scale_threshold": None,
                "scale_multiplier": 1.0,
                "scale_up_threshold": 0.10,
                "scale_up_multiplier": 1.25,
                "scale_up_max_position_drawdown": pos_dd,
            }
        )
    for min_unreal in (0.004, 0.008, 0.012):
        out.append(
            {
                "name": f"scaleup_min_unreal_{min_unreal:.3f}",
                "exit_threshold": 0.85,
                "scale_threshold": None,
                "scale_multiplier": 1.0,
                "scale_up_threshold": 0.10,
                "scale_up_multiplier": 1.25,
                "scale_up_min_unrealized": min_unreal,
            }
        )
    for conf in (0.55, 0.65, 0.75):
        out.append(
            {
                "name": f"scaleup_regime_conf_{conf:.2f}",
                "exit_threshold": 0.85,
                "scale_threshold": None,
                "scale_multiplier": 1.0,
                "scale_up_threshold": 0.10,
                "scale_up_multiplier": 1.25,
                "scale_up_min_regime_confidence": conf,
            }
        )
    out.append(
        {
            "name": "scaleup_block_regime_disagree",
            "exit_threshold": 0.85,
            "scale_threshold": None,
            "scale_multiplier": 1.0,
            "scale_up_threshold": 0.10,
            "scale_up_multiplier": 1.25,
            "scale_up_block_regime_disagree": True,
        }
    )
    out.append(
        {
            "name": "hybrid_scaledown_scaleup_guarded",
            "exit_threshold": 0.85,
            "scale_threshold": 0.65,
            "scale_multiplier": 0.50,
            "scale_up_threshold": 0.10,
            "scale_up_multiplier": 1.20,
            "scale_up_max_account_dd": 0.06,
            "scale_up_max_position_drawdown": 0.020,
            "scale_up_min_unrealized": 0.004,
            "scale_up_min_regime_confidence": 0.55,
            "scale_up_block_regime_disagree": True,
        }
    )
    for up_mult in (1.15, 1.25, 1.35):
        for account_dd in (0.04, 0.06, 0.08):
            for pos_dd in (0.015, 0.025):
                for min_unreal in (0.003, 0.006):
                    out.append(
                        {
                            "name": (
                                f"hybrid_ext_up{up_mult:.2f}_acct{account_dd:.2f}"
                                f"_pos{pos_dd:.3f}_u{min_unreal:.3f}"
                            ),
                            "exit_threshold": 0.85,
                            "scale_threshold": 0.65,
                            "scale_multiplier": 0.50,
                            "scale_up_threshold": 0.10,
                            "scale_up_multiplier": up_mult,
                            "scale_up_max_account_dd": account_dd,
                            "scale_up_max_position_drawdown": pos_dd,
                            "scale_up_min_unrealized": min_unreal,
                            "scale_up_min_regime_confidence": 0.55,
                            "scale_up_block_regime_disagree": True,
                        }
                    )
    return out


def _audit(train_df: pd.DataFrame, eval_df: pd.DataFrame, lifecycle: dict[str, Any]) -> dict[str, Any]:
    train_ts = pd.to_datetime(train_df["timestamp"], errors="coerce")
    eval_ts = pd.to_datetime(eval_df["timestamp"], errors="coerce")
    overlap = set(train_ts.astype("int64").tolist()) & set(eval_ts.astype("int64").tolist())
    feature_cols = list(lifecycle.get("feature_cols", []))
    impossible = [
        c
        for c in feature_cols
        if any(token in str(c).lower() for token in ("future", "target", "label", "candidate", "exit_idx"))
    ]
    live_state_cols = [c for c in feature_cols if c in {"lc_unrealized", "lc_peak_unrealized", "lc_drawdown_from_peak"}]
    return {
        "train_rows": int(len(train_df)),
        "eval_rows": int(len(eval_df)),
        "train_range": [str(train_ts.min()), str(train_ts.max())],
        "eval_range": [str(eval_ts.min()), str(eval_ts.max())],
        "timestamp_overlap_rows": int(len(overlap)),
        "train_duplicate_timestamps": int(train_ts.duplicated().sum()),
        "eval_duplicate_timestamps": int(eval_ts.duplicated().sum()),
        "feature_count": int(len(feature_cols)),
        "expected_feature_count": int(len(MODEL_COLS)),
        "feature_count_match": bool(len(feature_cols) == len(MODEL_COLS)),
        "impossible_feature_flags": impossible,
        "live_state_features": live_state_cols,
        "sample_meta": lifecycle.get("sample_meta", {}),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Search guarded lifecycle scale-up/down policies.")
    p.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    p.add_argument("--lifecycle", type=Path, default=DEFAULT_LIFECYCLE)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--min-age", type=int, default=3)
    p.add_argument("--max-notional", type=float, default=3.60)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    policy = joblib.load(args.policy)
    lifecycle = joblib.load(args.lifecycle)
    exit_model = lifecycle["model"]
    train_df = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    rows: list[dict[str, Any]] = []
    baseline = _compact(backtest_policy(eval_df, policy, fee=float(args.fee), slip=float(args.slip)))
    for cfg in _configs():
        params = {k: v for k, v in cfg.items() if k != "name"}
        bt = backtest_lifecycle(
            eval_df,
            policy,
            exit_model,
            fee=float(args.fee),
            slip=float(args.slip),
            min_age=int(args.min_age),
            max_notional=float(args.max_notional),
            **params,
        )
        rows.append({"name": cfg["name"], "config": cfg, "eval": _compact(bt)})
    ranked_pnl = sorted(rows, key=lambda r: float(r["eval"].get("pnl") or -1e18), reverse=True)
    ranked_mdd_safe = sorted(
        rows,
        key=lambda r: (
            float(r["eval"].get("pnl") or -1e18)
            if float(r["eval"].get("mdd") or -1e18) >= float(baseline["mdd"])
            else -1e18
        ),
        reverse=True,
    )
    report = {
        "type": "lifecycle_guard_search_patchtst_tide_dlinear_2026",
        "policy": str(args.policy),
        "lifecycle": str(args.lifecycle),
        "audit": _audit(train_df, eval_df, lifecycle),
        "baseline": baseline,
        "grid": rows,
        "ranked_by_pnl": [{"name": r["name"], **r["eval"]} for r in ranked_pnl],
        "ranked_by_pnl_with_mdd_not_worse_than_baseline": [
            {"name": r["name"], **r["eval"]}
            for r in ranked_mdd_safe
            if float(r["eval"].get("mdd") or -1e18) >= float(baseline["mdd"])
        ],
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.report_out),
                "audit": report["audit"],
                "baseline": baseline,
                "top_pnl": report["ranked_by_pnl"][:5],
                "top_mdd_safe": report["ranked_by_pnl_with_mdd_not_worse_than_baseline"][:5],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
