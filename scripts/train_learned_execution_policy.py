#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.learned_execution_policy import (  # noqa: E402
    LearnedExecutionConfig,
    build_execution_training_set,
    predict_learned_execution,
    train_learned_execution_policy,
)
from ensemble.macro_trend_sleeve import MacroTrendSleeveConfig  # noqa: E402


DEFAULT_TRAIN_CSV = ROOT / "tmp/pipeline_audit_causal_regime/trade_candidates_2025_causal_regime.csv"
DEFAULT_EVAL_CSV = ROOT / "data/ensemble/event_driven/trade_candidates_2026_causal_regime_predicted_45m_hgb_telemetry.csv"
DEFAULT_MODEL_OUT = ROOT / "data/ensemble/supervised/learned_execution_policy_v1.pkl"
DEFAULT_REPORT_OUT = ROOT / "data/ensemble/reports/learned_execution_policy_v1_train_report.json"


def _close_array(frame: pd.DataFrame) -> np.ndarray:
    return (
        pd.to_numeric(frame["close"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .to_numpy(dtype=np.float64)
    )


def _precompute_macro(close: np.ndarray, cfg: MacroTrendSleeveConfig) -> tuple[np.ndarray, np.ndarray]:
    n = len(close)
    look = max(1, int(cfg.lookback_bars))
    mom = np.full(n, np.nan, dtype=np.float64)
    if n > look:
        mom[look:] = close[look:] / np.maximum(close[:-look], 1e-12) - 1.0
    desired = np.zeros(n, dtype=np.int8)
    desired[mom > float(cfg.threshold)] = 1
    desired[mom < -float(cfg.threshold)] = -1
    confirmed = np.zeros(n, dtype=np.int8)
    current = 0
    pending = 0
    pending_count = 0
    update = max(1, int(cfg.update_bars))
    persist = max(1, int(cfg.persist_updates))
    for i, raw in enumerate(desired):
        if i <= max(look, int(cfg.min_history_bars)):
            confirmed[i] = 0
            continue
        if i % update == 0:
            raw_i = int(raw)
            if raw_i == current:
                pending = 0
                pending_count = 0
            elif raw_i == pending:
                pending_count += 1
            else:
                pending = raw_i
                pending_count = 1
            if pending_count >= persist:
                current = raw_i
                pending = 0
                pending_count = 0
        sig = int(current)
        if sig == 0 and bool(cfg.bootstrap_current) and np.isfinite(mom[i]) and abs(float(mom[i])) > float(cfg.threshold):
            sig = 1 if float(mom[i]) > 0.0 else -1
        confirmed[i] = sig
    return confirmed, np.nan_to_num(mom, nan=0.0, posinf=0.0, neginf=0.0)


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return df.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train learned TP/leverage/notional execution policy from 2025 macro candidates.")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    p.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUT)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    p.add_argument("--candidate-stride-bars", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=768)
    p.add_argument("--macro-lookback-bars", type=int, default=6048)
    p.add_argument("--macro-threshold", type=float, default=0.05)
    p.add_argument("--macro-persist-updates", type=int, default=5)
    p.add_argument("--macro-update-bars", type=int, default=288)
    p.add_argument("--max-train-horizon-bars", type=int, default=864)
    p.add_argument("--stop-loss", type=float, default=0.030)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    macro_cfg = MacroTrendSleeveConfig(
        lookback_bars=int(args.macro_lookback_bars),
        threshold=float(args.macro_threshold),
        persist_updates=int(args.macro_persist_updates),
        update_bars=int(args.macro_update_bars),
        min_history_bars=int(args.macro_lookback_bars),
        bootstrap_current=True,
    )
    train_signal, train_mom = _precompute_macro(_close_array(train), macro_cfg)
    cfg = LearnedExecutionConfig(
        max_train_horizon_bars=int(args.max_train_horizon_bars),
        stop_loss=float(args.stop_loss),
    )
    x, y, meta = build_execution_training_set(
        train,
        close=_close_array(train),
        macro_signal=train_signal,
        macro_momentum=train_mom,
        cfg=cfg,
        candidate_stride_bars=int(args.candidate_stride_bars),
        batch_size=int(args.batch_size),
    )
    bundle = train_learned_execution_policy(x, y, cfg=cfg, random_state=int(args.random_state))
    bundle["train_csv"] = str(args.train_csv)
    bundle["macro_config"] = {
        "lookback_bars": int(args.macro_lookback_bars),
        "threshold": float(args.macro_threshold),
        "persist_updates": int(args.macro_persist_updates),
        "update_bars": int(args.macro_update_bars),
    }
    bundle["training_meta"] = meta
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_out)

    eval_signal, eval_mom = _precompute_macro(_close_array(eval_df), macro_cfg)
    idx = np.flatnonzero(eval_signal != 0)
    idx = idx[idx < len(eval_df)]
    sample_idx = idx[:: max(1, int(args.candidate_stride_bars))]
    decisions = []
    for i in sample_idx[:2000]:
        dec = predict_learned_execution(
            bundle,
            eval_df.iloc[int(i)],
            source="macro",
            side=int(eval_signal[int(i)]),
            macro_momentum=float(eval_mom[int(i)]),
        )
        decisions.append(dec.asdict())
    dec_df = pd.DataFrame(decisions)
    report = {
        "type": "learned_execution_policy_v1",
        "train_csv": str(args.train_csv),
        "eval_csv": str(args.eval_csv),
        "model_out": str(args.model_out),
        "macro_config": bundle["macro_config"],
        "training_meta": meta,
        "label_distribution": bundle["label_distribution"],
        "eval_candidate_count": int(len(idx)),
        "eval_decision_sample_count": int(len(dec_df)),
        "eval_decision_summary": {
            col: {
                "mean": float(dec_df[col].mean()),
                "p25": float(dec_df[col].quantile(0.25)),
                "p50": float(dec_df[col].quantile(0.50)),
                "p75": float(dec_df[col].quantile(0.75)),
            }
            for col in ["notional_exposure", "leverage", "take_profit", "max_hold_bars", "quality_score", "confidence"]
            if col in dec_df.columns and len(dec_df)
        },
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"model": str(args.model_out), "report": str(args.report_out), "training_meta": meta, "eval_summary": report["eval_decision_summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
