#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.microstructure_wnc_sleeve import (
    MicrostructureSleeveConfig,
    backtest_microstructure_sleeve,
    predict_microstructure_proba,
    required_columns,
    train_microstructure_classifier,
)


DEFAULT_TRAIN_CSV = ROOT / "data" / "ensemble" / "event_driven" / "trade_candidates_v1_oof_regime_v3.csv"
DEFAULT_EVAL_CSV = ROOT / "data" / "ensemble" / "event_driven" / "trade_candidates_2026_oofdet_router_regime_v3_manifest_policy.csv"
DEFAULT_MODEL_OUT = ROOT / "data" / "ensemble" / "supervised" / "microstructure_wnc_sleeve_v2.pkl"
DEFAULT_REPORT_OUT = ROOT / "data" / "ensemble" / "reports" / "microstructure_wnc_sleeve_v2_turbo10x_oos_2026.json"


PROFILE_CONFIGS = {
    "v1": MicrostructureSleeveConfig(),
    "production_5x": MicrostructureSleeveConfig(
        entry_confidence=0.37959911614824376,
        entry_gap=0.3273389977834895,
        max_hold_bars=30,
        stop_loss=0.007647931125556956,
        take_profit=0.02608703214690002,
        trailing_stop=0.00859599753819247,
        max_notional_exposure=5.0,
        max_leverage=5.0,
        cooldown_bars=0,
        whipsaw_notional_mult=1.9454359357639504,
        chop_notional_mult=1.3334158280759751,
        normal_notional_mult=2.047679763051344,
        portfolio_soft_drawdown=0.22224971377661384,
        portfolio_hard_drawdown=0.4994007139121906,
        portfolio_min_drawdown_scale=0.6266004508398677,
    ),
    "turbo_10x": MicrostructureSleeveConfig(
        entry_confidence=0.3870777030038392,
        entry_gap=0.26260183981817936,
        max_hold_bars=20,
        stop_loss=0.00996410283391025,
        take_profit=0.04464206326407899,
        trailing_stop=0.007980417832712935,
        max_notional_exposure=10.0,
        max_leverage=10.0,
        cooldown_bars=1,
        whipsaw_notional_mult=2.166737899985807,
        chop_notional_mult=1.5484842405099886,
        normal_notional_mult=2.454551388321664,
        portfolio_soft_drawdown=0.17319296249630803,
        portfolio_hard_drawdown=0.45753785481698284,
        portfolio_min_drawdown_scale=0.8073887198792993,
    ),
}


def _load(path: Path, feature_mode: str) -> pd.DataFrame:
    if feature_mode == "full":
        return pd.read_csv(path)
    cols = set(required_columns())
    return pd.read_csv(path, usecols=lambda c: c in cols)


def _config_from_args(args: argparse.Namespace) -> MicrostructureSleeveConfig:
    cfg = PROFILE_CONFIGS[str(args.profile)]
    overrides = {
        "entry_confidence": args.entry_confidence,
        "entry_gap": args.entry_gap,
        "max_hold_bars": args.max_hold_bars,
        "stop_loss": args.stop_loss,
        "take_profit": args.take_profit,
        "trailing_stop": args.trailing_stop,
        "max_notional_exposure": args.max_notional_exposure,
        "max_leverage": args.max_leverage,
        "whipsaw_notional_mult": args.whipsaw_notional_mult,
        "chop_notional_mult": args.chop_notional_mult,
        "normal_notional_mult": args.normal_notional_mult,
    }
    clean = {k: v for k, v in overrides.items() if v is not None}
    return replace(cfg, **clean)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the whipsaw/normal/chop microstructure sleeve.")
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUT)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--feature-mode", choices=["minimal", "full"], default="full")
    parser.add_argument("--model-mode", choices=["hgb", "ensemble"], default="ensemble")
    parser.add_argument("--profile", choices=sorted(PROFILE_CONFIGS), default="turbo_10x")
    parser.add_argument("--entry-confidence", type=float, default=None)
    parser.add_argument("--entry-gap", type=float, default=None)
    parser.add_argument("--max-hold-bars", type=int, default=None)
    parser.add_argument("--stop-loss", type=float, default=None)
    parser.add_argument("--take-profit", type=float, default=None)
    parser.add_argument("--trailing-stop", type=float, default=None)
    parser.add_argument("--max-notional-exposure", type=float, default=None)
    parser.add_argument("--max-leverage", type=float, default=None)
    parser.add_argument("--whipsaw-notional-mult", type=float, default=None)
    parser.add_argument("--chop-notional-mult", type=float, default=None)
    parser.add_argument("--normal-notional-mult", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_df = _load(args.train_csv, args.feature_mode)
    eval_df = _load(args.eval_csv, args.feature_mode)
    model, feature_cols, train_summary = train_microstructure_classifier(
        train_df,
        feature_mode=str(args.feature_mode),
        model_mode=str(args.model_mode),
    )
    eval_frame, proba, classes = predict_microstructure_proba(model, eval_df, feature_cols)
    cfg = _config_from_args(args)
    result = backtest_microstructure_sleeve(eval_frame, proba, classes, cfg)

    bundle = {
        "model": model,
        "feature_cols": feature_cols,
        "classes": classes,
        "config": result.config,
        "train_summary": train_summary,
        "target_regimes": ["whipsaw", "normal", "chop"],
        "model_type": "microstructure_wnc_sleeve_v2",
        "profile": str(args.profile),
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_out)

    report = {
        "type": "microstructure_wnc_sleeve_v2",
        "train_csv": str(args.train_csv),
        "eval_csv": str(args.eval_csv),
        "model_out": str(args.model_out),
        "feature_mode": str(args.feature_mode),
        "model_mode": str(args.model_mode),
        "profile": str(args.profile),
        "train_summary": train_summary,
        "config": asdict(cfg),
        "result": result.asdict(),
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report["result"][k] for k in ["total_return_pct", "mdd_pct", "trades", "trades_per_day", "win_rate", "regime_entries"]}, ensure_ascii=False))
    print(args.report_out)


if __name__ == "__main__":
    main()
