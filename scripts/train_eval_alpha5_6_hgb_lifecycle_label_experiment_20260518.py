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
    ACTION_CASH,
    ACTION_LONG,
    ACTION_SHORT,
    FullyLearnedGovernorConfig,
    build_training_set,
    train_policy,
)
from scripts.train_eval_alpha5_3_hmm_dqn_router_parent_20260517 import (  # noqa: E402
    CLEAN4_PREFIX,
    DEFAULT_CLEAN4_REPORT,
    DEFAULT_EVAL,
    DEFAULT_PREPROCESS_MANIFEST,
    DEFAULT_TRAIN,
    REGIMES,
    ROUTER_COLS,
    _verify_state24_sticky090_inputs,
)
from scripts.train_eval_alpha5_4_single_conditioned_dqn_20260518 import (  # noqa: E402
    _feature_cols as _alpha5_feature_cols,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import (  # noqa: E402
    _json_default,
    _read,
    backtest_policy_frame,
)


MODEL_ID = "alpha5_6_hgb_lifecycle_label_experiment_state24_sticky090_20260518"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_6_hgb_lifecycle_label_experiment_20260518"


def _split(frame: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    out = frame.copy()
    if start:
        out = out[out["timestamp"] >= pd.Timestamp(start)]
    if end:
        out = out[out["timestamp"] < pd.Timestamp(end)]
    return out.reset_index(drop=True)


def _days(df: pd.DataFrame) -> float:
    return max((df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 86400.0, 1e-8)


def _score(metrics: dict[str, Any]) -> float:
    c1 = metrics["cost1"]
    c2 = metrics["cost2"]
    c3 = metrics["cost3"]
    return (
        float(c1["pnl"])
        + 0.50 * float(c2["pnl"])
        + 0.25 * float(c3["pnl"])
        - 0.35 * abs(float(c1["mdd"]))
        - 2.0 * max(0.0, float(c1["trades_per_day"]) - 8.0)
    )


def _cfg_l1() -> FullyLearnedGovernorConfig:
    return FullyLearnedGovernorConfig(
        notional_buckets=(0.20, 0.32, 0.50, 0.75, 1.05, 1.45, 2.00, 2.70, 3.60),
        leverage_buckets=(1.5, 2.0, 3.0, 4.0, 5.0),
        take_profit_buckets=(0.007, 0.011, 0.018, 0.030, 0.050, 0.090, 0.180, 0.450, 0.900),
        stop_loss_buckets=(0.004, 0.006, 0.009, 0.014, 0.022, 0.035, 0.055),
        max_hold_buckets=(6, 12, 24, 48, 96, 192, 288),
        cooldown_buckets=(0, 1, 3, 6, 12, 24, 48),
        max_train_horizon_bars=288,
        adverse_penalty=2.45,
        size_penalty=0.180,
        hold_penalty=0.042,
        turnover_bonus=0.0012,
        cash_score=0.020,
    )


def _cfg_l3() -> FullyLearnedGovernorConfig:
    return FullyLearnedGovernorConfig(
        notional_buckets=(0.10, 0.16, 0.24, 0.34, 0.48, 0.68, 0.95, 1.30),
        leverage_buckets=(1.5, 2.0, 3.0),
        take_profit_buckets=(0.006, 0.010, 0.018, 0.030, 0.050, 0.090, 0.180),
        stop_loss_buckets=(0.004, 0.006, 0.010, 0.016, 0.024, 0.035),
        max_hold_buckets=(6, 12, 24, 48, 96, 192, 288),
        cooldown_buckets=(0, 1, 3, 6, 12, 24),
        max_train_horizon_bars=288,
        adverse_penalty=2.70,
        size_penalty=0.260,
        hold_penalty=0.050,
        turnover_bonus=0.0010,
        cash_score=0.026,
    )


def _variant_specs() -> dict[str, tuple[FullyLearnedGovernorConfig, bool]]:
    return {
        "L1_base_h288": (_cfg_l1(), False),
        "L2_regime_aware_h288": (_cfg_l1(), True),
        "L3_alpha5_slim_h288": (_cfg_l3(), False),
    }


def _valid_indices(n_rows: int, horizon: int, stride: int) -> np.ndarray:
    return np.arange(0, max(0, int(n_rows) - int(horizon) - 1), max(1, int(stride)), dtype=np.int64)


def _label_counts(values: np.ndarray) -> dict[str, int]:
    names = {
        "action": {ACTION_CASH: "cash", ACTION_LONG: "long", ACTION_SHORT: "short"},
        "notional": None,
        "leverage": None,
        "take_profit": None,
        "stop_loss": None,
        "max_hold": None,
        "cooldown": None,
    }
    _ = names
    out: dict[str, int] = {}
    for k, v in pd.Series(values).value_counts().sort_index().items():
        out[str(int(k))] = int(v)
    return out


def _qa_labels(y: dict[str, np.ndarray], frame: pd.DataFrame, valid_idx: np.ndarray) -> dict[str, Any]:
    action = np.asarray(y["action"], dtype=np.int64)
    quality = np.asarray(y["quality"], dtype=np.float64)
    probs = frame.iloc[valid_idx][ROUTER_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    regime_idx = np.argmax(probs, axis=1) if len(probs) else np.zeros(0, dtype=np.int64)
    by_regime: dict[str, Any] = {}
    for i, name in enumerate(REGIMES):
        mask = regime_idx == i
        if not np.any(mask):
            by_regime[name] = {"n": 0}
            continue
        by_regime[name] = {
            "n": int(mask.sum()),
            "action_counts": {
                "cash": int(np.sum(action[mask] == ACTION_CASH)),
                "long": int(np.sum(action[mask] == ACTION_LONG)),
                "short": int(np.sum(action[mask] == ACTION_SHORT)),
            },
            "quality_mean": float(np.mean(quality[mask])),
            "quality_p90": float(np.quantile(quality[mask], 0.90)),
        }
    return {
        "rows": int(len(action)),
        "action_counts": {
            "cash": int(np.sum(action == ACTION_CASH)),
            "long": int(np.sum(action == ACTION_LONG)),
            "short": int(np.sum(action == ACTION_SHORT)),
        },
        "bucket_counts": {
            key: _label_counts(np.asarray(y[key]))
            for key in ("notional", "leverage", "take_profit", "stop_loss", "max_hold", "cooldown")
        },
        "quality": {
            "mean": float(np.mean(quality)),
            "p10": float(np.quantile(quality, 0.10)),
            "p50": float(np.quantile(quality, 0.50)),
            "p90": float(np.quantile(quality, 0.90)),
            "p95": float(np.quantile(quality, 0.95)),
        },
        "by_regime": by_regime,
    }


def _apply_regime_hurdle(y: dict[str, np.ndarray], frame: pd.DataFrame, valid_idx: np.ndarray) -> dict[str, Any]:
    action = np.asarray(y["action"], dtype=np.int64).copy()
    quality = np.asarray(y["quality"], dtype=np.float64).copy()
    probs = frame.iloc[valid_idx][ROUTER_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    regime_idx = np.argmax(probs, axis=1)
    before = {
        "cash": int(np.sum(action == ACTION_CASH)),
        "long": int(np.sum(action == ACTION_LONG)),
        "short": int(np.sum(action == ACTION_SHORT)),
    }
    bull = regime_idx == REGIMES.index("bull")
    bear = regime_idx == REGIMES.index("bear")
    chop = regime_idx == REGIMES.index("chop")
    whipsaw = regime_idx == REGIMES.index("whipsaw")
    block = np.zeros(len(action), dtype=bool)
    block |= whipsaw & (action != ACTION_CASH) & (quality < 0.055)
    block |= chop & (action != ACTION_CASH) & (quality < 0.032)
    block |= bull & (action == ACTION_SHORT) & (quality < 0.040)
    block |= bear & (action == ACTION_LONG) & (quality < 0.040)
    for key in ("notional", "leverage", "take_profit", "stop_loss", "max_hold", "cooldown"):
        arr = np.asarray(y[key]).copy()
        arr[block] = 0
        y[key] = arr
    action[block] = ACTION_CASH
    quality[block] = np.minimum(quality[block], 0.0)
    y["action"] = action
    y["quality"] = quality
    after = {
        "cash": int(np.sum(action == ACTION_CASH)),
        "long": int(np.sum(action == ACTION_LONG)),
        "short": int(np.sum(action == ACTION_SHORT)),
    }
    return {"blocked_rows": int(block.sum()), "before": before, "after": after}


def _feature_contract(train_df: pd.DataFrame, eval_df: pd.DataFrame, top_k: int, include_future: bool) -> list[str]:
    cols = _alpha5_feature_cols(
        train_df,
        eval_df,
        include_future_regime_pred=bool(include_future),
        feature_top_k=int(top_k),
        feature_select_horizon=48,
    )
    feature_cols = ["side_hint"] + [c for c in cols if c != "side_hint"]
    return list(dict.fromkeys(feature_cols))


def _metrics_for_bundle(frame: pd.DataFrame, bundle: dict[str, Any], cfg: FullyLearnedGovernorConfig) -> dict[str, Any]:
    return {
        f"cost{mult}": backtest_policy_frame(
            frame,
            bundle,
            fee=float(cfg.fee) * float(mult),
            slip=float(cfg.slip) * float(mult),
            record_trades=False,
        )
        for mult in (1, 2, 3)
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Alpha5.6 HGB lifecycle label experiment on Regime4 state24 data.")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--manifest", type=Path, default=DEFAULT_PREPROCESS_MANIFEST)
    p.add_argument("--clean4-report", type=Path, default=DEFAULT_CLEAN4_REPORT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--train-end", default="2025-10-01")
    p.add_argument("--val-start", default="2025-10-01")
    p.add_argument("--val-end", default="2026-01-01")
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--feature-top-k", type=int, default=64)
    p.add_argument("--include-future-regime-pred", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--variants", default="L1_base_h288,L2_regime_aware_h288,L3_alpha5_slim_h288")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    np.random.seed(int(args.seed))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_train = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    train_df = _split(raw_train, None, args.train_end)
    val_df = _split(raw_train, args.val_start, args.val_end)
    audit = _verify_state24_sticky090_inputs(raw_train, eval_df, args.manifest, args.clean4_report)
    feature_cols = _feature_contract(raw_train, eval_df, int(args.feature_top_k), bool(args.include_future_regime_pred))
    forbidden_legacy = [c for c in feature_cols if c.startswith("clean_regime_2024_unsup_v4_")]
    if forbidden_legacy:
        raise RuntimeError("legacy clean_regime_2024_unsup_v4 features selected: " + ",".join(forbidden_legacy[:20]))
    print(
        json.dumps(
            {
                "stage": "start",
                "model_id": MODEL_ID,
                "train_rows": len(train_df),
                "validation_rows": len(val_df),
                "oos_rows": len(eval_df),
                "feature_count": len(feature_cols),
                "clean4_feature_count": int(sum(c.startswith(CLEAN4_PREFIX) for c in feature_cols)),
                "future_regime4_feature_count": int(sum(c.startswith("regime4_pred_") for c in feature_cols)),
                "audit": {
                    "expected_model_found_in_manifest": audit.get("expected_model_found_in_manifest"),
                    "legacy_v4_count": audit.get("legacy_v4_count"),
                    "router_missing": audit.get("router_missing"),
                },
            },
            ensure_ascii=False,
            default=_json_default,
        ),
        flush=True,
    )

    variants = [v.strip() for v in str(args.variants).split(",") if v.strip()]
    specs = _variant_specs()
    experiments: list[dict[str, Any]] = []
    for i, name in enumerate(variants):
        if name not in specs:
            raise ValueError(f"unknown variant: {name}")
        cfg, regime_aware = specs[name]
        print(json.dumps({"stage": "build_labels", "variant": name, "cfg": asdict(cfg)}, ensure_ascii=False), flush=True)
        x_train, y_train, train_meta = build_training_set(
            train_df,
            cfg=cfg,
            stride_bars=int(args.stride),
            batch_size=512,
            feature_cols=feature_cols,
        )
        train_valid_idx = _valid_indices(len(train_df), int(cfg.max_train_horizon_bars), int(args.stride))
        regime_patch = None
        if regime_aware:
            regime_patch = _apply_regime_hurdle(y_train, train_df, train_valid_idx)
        train_qa = _qa_labels(y_train, train_df, train_valid_idx)
        print(json.dumps({"stage": "label_qa", "variant": name, "train_qa": train_qa, "regime_patch": regime_patch}, ensure_ascii=False, default=_json_default), flush=True)

        print(json.dumps({"stage": "train_hgb_policy", "variant": name, "rows": len(x_train)}, ensure_ascii=False), flush=True)
        bundle = train_policy(x_train, y_train, cfg=cfg, random_state=int(args.seed) + i * 100, feature_cols=feature_cols)
        bundle["alpha5_6_variant"] = name
        bundle["alpha5_6_label_meta"] = {"train_meta": train_meta, "train_qa": train_qa, "regime_patch": regime_patch}

        val_metrics = _metrics_for_bundle(val_df, bundle, cfg)
        oos_metrics = _metrics_for_bundle(eval_df, bundle, cfg)
        score = _score(val_metrics)
        out_path = args.out_dir / f"{name}_parent.joblib"
        joblib.dump(bundle, out_path)
        row = {
            "name": name,
            "score": float(score),
            "train_meta": train_meta,
            "train_qa": train_qa,
            "regime_patch": regime_patch,
            "validation_metrics": val_metrics,
            "oos_metrics": oos_metrics,
            "artifact": str(out_path),
        }
        experiments.append(row)
        print(
            json.dumps(
                {
                    "stage": "variant_complete",
                    "variant": name,
                    "score": float(score),
                    "validation_cost1": val_metrics["cost1"],
                    "oos_cost1": oos_metrics["cost1"],
                    "artifact": str(out_path),
                },
                ensure_ascii=False,
                default=_json_default,
            ),
            flush=True,
        )

    best = max(experiments, key=lambda r: float(r["score"]))
    summary = {
        "model_id": MODEL_ID,
        "design": "Lifecycle multi-head HGB label experiment for Alpha5.6. Labels include action, quality, notional, leverage, TP, SL, max_hold, and cooldown; inputs use Regime4 state24 sticky090 features and optional future regime predictor features.",
        "train_csv": str(args.train_csv),
        "eval_csv": str(args.eval_csv),
        "split": {
            "train": [str(train_df["timestamp"].iloc[0]), str(train_df["timestamp"].iloc[-1])],
            "validation": [str(val_df["timestamp"].iloc[0]), str(val_df["timestamp"].iloc[-1])],
            "oos": [str(eval_df["timestamp"].iloc[0]), str(eval_df["timestamp"].iloc[-1])],
        },
        "feature_contract": {
            "feature_cols": feature_cols,
            "feature_count": len(feature_cols),
            "clean4_feature_count": int(sum(c.startswith(CLEAN4_PREFIX) for c in feature_cols)),
            "future_regime4_feature_count": int(sum(c.startswith("regime4_pred_") for c in feature_cols)),
            "legacy_clean_v4_count": int(sum(c.startswith("clean_regime_2024_unsup_v4_") for c in feature_cols)),
        },
        "state24_sticky090_audit": audit,
        "experiments": experiments,
        "best": best,
        "artifacts": {
            "out_dir": str(args.out_dir),
            "summary": str(args.out_dir / "alpha5_6_hgb_lifecycle_label_experiment_summary.json"),
        },
    }
    report_path = args.out_dir / "alpha5_6_hgb_lifecycle_label_experiment_summary.json"
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    grid_path = args.out_dir / "alpha5_6_hgb_lifecycle_label_experiment_grid.csv"
    pd.DataFrame(
        [
            {
                "name": r["name"],
                "score": r["score"],
                "val_cost1_pnl": r["validation_metrics"]["cost1"]["pnl"],
                "val_cost1_mdd": r["validation_metrics"]["cost1"]["mdd"],
                "val_cost1_trades_day": r["validation_metrics"]["cost1"]["trades_per_day"],
                "oos_cost1_pnl": r["oos_metrics"]["cost1"]["pnl"],
                "oos_cost1_mdd": r["oos_metrics"]["cost1"]["mdd"],
                "oos_cost1_trades_day": r["oos_metrics"]["cost1"]["trades_per_day"],
                "oos_cost2_pnl": r["oos_metrics"]["cost2"]["pnl"],
                "oos_cost3_pnl": r["oos_metrics"]["cost3"]["pnl"],
                "artifact": r["artifact"],
            }
            for r in experiments
        ]
    ).to_csv(grid_path, index=False)
    print(json.dumps({"stage": "complete", "summary": str(report_path), "grid": str(grid_path), "best": best["name"]}, ensure_ascii=False, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
