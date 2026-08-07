#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import (  # noqa: E402
    FullyLearnedGovernorConfig,
    build_training_set,
    train_policy,
)
from scripts.eval_lifecycle_ai_stress import AI_GROUPS, _stress_frame  # noqa: E402
from scripts.train_eval_fully_learned_governor import backtest_policy  # noqa: E402


DEFAULT_TRAIN_CSV = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL_CSV = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_MODEL_OUT = ROOT / "data/ensemble/supervised/fully_learned_ai_dropout/patchtst_tide_dlinear_dropout.pkl"
DEFAULT_REPORT_OUT = ROOT / "data/ensemble/reports/fully_learned_ai_dropout_patchtst_tide_dlinear_2026.json"


def _float_tuple(value: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if value is None or str(value).strip() == "":
        return default
    return tuple(float(x.strip()) for x in str(value).split(",") if x.strip())


def _int_tuple(value: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None or str(value).strip() == "":
        return default
    return tuple(int(x.strip()) for x in str(value).split(",") if x.strip())


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return df.reset_index(drop=True)


def _apply_train_dropout(
    train_df: pd.DataFrame,
    *,
    seed: int,
    group_prob: float,
    all_prob: float,
    stale_prob: float,
    stale_bars: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    out = train_df.copy()
    meta: dict[str, Any] = {
        "group_prob": float(group_prob),
        "all_prob": float(all_prob),
        "stale_prob": float(stale_prob),
        "stale_bars": int(stale_bars),
        "masked": {},
    }
    all_mask = rng.random(len(out)) < float(all_prob)
    for group, cols in AI_GROUPS.items():
        present = [c for c in cols if c in out.columns]
        if not present:
            continue
        group_mask = (rng.random(len(out)) < float(group_prob)) | all_mask
        stale_mask = rng.random(len(out)) < float(stale_prob)
        for col in present:
            s = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
            stale_values = s.shift(int(stale_bars)).ffill().fillna(0.0)
            vals = s.to_numpy(dtype=np.float64, copy=True)
            vals[stale_mask] = stale_values.to_numpy(dtype=np.float64, copy=False)[stale_mask]
            vals[group_mask] = 0.0
            out[col] = vals
        meta["masked"][group] = {
            "cols": present,
            "zero_rate": float(group_mask.mean()),
            "stale_rate": float(stale_mask.mean()),
        }
    return out, meta


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
        )
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/evaluate fully learned governor with AI feature dropout on train rows.")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    p.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUT)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    p.add_argument("--validation-start", default="2025-10-01")
    p.add_argument("--group-prob", type=float, default=0.18)
    p.add_argument("--all-prob", type=float, default=0.04)
    p.add_argument("--stale-prob", type=float, default=0.08)
    p.add_argument("--stale-bars", type=int, default=288)
    p.add_argument("--stride-bars", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--max-train-horizon-bars", type=int, default=288)
    p.add_argument("--adverse-penalty", type=float, default=2.45)
    p.add_argument("--size-penalty", type=float, default=0.180)
    p.add_argument("--hold-penalty", type=float, default=0.042)
    p.add_argument("--turnover-bonus", type=float, default=0.0012)
    p.add_argument("--cash-score", type=float, default=0.020)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--notional-buckets", type=str, default="0.20,0.32,0.50,0.75,1.05,1.45,2.00,2.70,3.60")
    p.add_argument("--leverage-buckets", type=str, default=None)
    p.add_argument("--take-profit-buckets", type=str, default="0.007,0.011,0.018,0.030,0.050,0.090,0.180,0.450,0.900")
    p.add_argument("--stop-loss-buckets", type=str, default="0.004,0.006,0.009,0.014,0.022,0.035,0.055")
    p.add_argument("--max-hold-buckets", type=str, default="6,12,24,48,96,192,288")
    p.add_argument("--cooldown-buckets", type=str, default="0,1,3,6,12,24,48")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    split_ts = pd.Timestamp(args.validation_start)
    train_df = train_all[train_all["timestamp"] < split_ts].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= split_ts].reset_index(drop=True)
    train_dropout, dropout_meta = _apply_train_dropout(
        train_df,
        seed=int(args.seed),
        group_prob=float(args.group_prob),
        all_prob=float(args.all_prob),
        stale_prob=float(args.stale_prob),
        stale_bars=int(args.stale_bars),
    )
    cfg = FullyLearnedGovernorConfig(
        notional_buckets=_float_tuple(args.notional_buckets, FullyLearnedGovernorConfig.notional_buckets),
        leverage_buckets=_float_tuple(args.leverage_buckets, FullyLearnedGovernorConfig.leverage_buckets),
        take_profit_buckets=_float_tuple(args.take_profit_buckets, FullyLearnedGovernorConfig.take_profit_buckets),
        stop_loss_buckets=_float_tuple(args.stop_loss_buckets, FullyLearnedGovernorConfig.stop_loss_buckets),
        max_hold_buckets=_int_tuple(args.max_hold_buckets, FullyLearnedGovernorConfig.max_hold_buckets),
        cooldown_buckets=_int_tuple(args.cooldown_buckets, FullyLearnedGovernorConfig.cooldown_buckets),
        max_train_horizon_bars=int(args.max_train_horizon_bars),
        adverse_penalty=float(args.adverse_penalty),
        size_penalty=float(args.size_penalty),
        hold_penalty=float(args.hold_penalty),
        turnover_bonus=float(args.turnover_bonus),
        cash_score=float(args.cash_score),
        fee=float(args.fee),
        slip=float(args.slip),
    )
    x, y, training_meta = build_training_set(
        train_dropout,
        cfg=cfg,
        stride_bars=int(args.stride_bars),
        batch_size=int(args.batch_size),
    )
    bundle = train_policy(x, y, cfg=cfg, random_state=int(args.seed))
    bundle["train_csv"] = str(args.train_csv)
    bundle["training_meta"] = training_meta
    bundle["ai_dropout_meta"] = dropout_meta
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_out)

    stress_modes = [
        "normal",
        "patchtst_zero",
        "tide_zero",
        "dlinear_zero",
        "patchtst__tide_zero",
        "patchtst__dlinear_zero",
        "tide__dlinear_zero",
        "all_ai_zero",
        "patchtst_stale_1d",
        "tide_stale_1d",
        "dlinear_stale_1d",
    ]
    stress_results: dict[str, Any] = {}
    for mode in stress_modes:
        stressed, meta = _stress_frame(eval_df, mode)
        stress_results[mode] = {
            "stress": meta,
            "eval": _compact(backtest_policy(stressed, bundle, fee=float(args.fee), slip=float(args.slip))),
        }

    report = {
        "type": "fully_learned_ai_dropout_patchtst_tide_dlinear_2026",
        "train_csv": str(args.train_csv),
        "eval_csv": str(args.eval_csv),
        "model_out": str(args.model_out),
        "config": asdict(cfg),
        "ai_dropout_meta": dropout_meta,
        "training_meta": training_meta,
        "label_distribution": bundle.get("label_distribution", {}),
        "train_dropout_backtest": _compact(backtest_policy(train_dropout, bundle, fee=float(args.fee), slip=float(args.slip))),
        "validation_original": _compact(backtest_policy(val_df, bundle, fee=float(args.fee), slip=float(args.slip))),
        "stress_results": stress_results,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.report_out),
                "validation_original": report["validation_original"],
                "normal_eval": stress_results["normal"]["eval"],
                "all_ai_zero_eval": stress_results["all_ai_zero"]["eval"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
