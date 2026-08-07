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

from ensemble.fully_learned_governor_policy import build_training_set, prepare_features  # noqa: E402
from scripts.train_eval_alpha5_3_hmm_dqn_router_parent_20260517 import (  # noqa: E402
    CLEAN4_PREFIX,
    DEFAULT_CLEAN4_REPORT,
    DEFAULT_PREPROCESS_MANIFEST,
    _verify_state24_sticky090_inputs,
)
from scripts.train_eval_alpha5_5_lgbm_supervised_parent_20260518 import (  # noqa: E402
    _backtest_actions,
    _decide_actions,
    _predict_proba_3,
)
from scripts.train_eval_alpha5_8_hgb_action_feature_contract_compare_20260518 import (  # noqa: E402
    ALPHA4_PARENT,
    _alpha4_mapped_features,
)
from scripts.tune_alpha5_9_hgb_action_master_20260518 import (  # noqa: E402
    HGBSpec,
    _cfgs,
    _fit_hgb,
    _grid,
    _hgb_specs,
    _score,
    _valid_indices,
    _weights,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _json_default, _read  # noqa: E402


MODEL_ID = "alpha5_10_hgb_legacy_v4_vs_regime4_20260518"
DEFAULT_LEGACY_TRAIN = ROOT / "tmp/causal_regen_20260516/alpha4_3_hgb_atr3_tp_sl_action_score_20260517/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_LEGACY_EVAL = ROOT / "tmp/causal_regen_20260516/alpha4_3_hgb_atr3_tp_sl_action_score_20260517/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_REGIME4_TRAIN = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_REGIME4_EVAL = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2026_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_10_hgb_legacy_v4_vs_regime4_20260518"


def _split(frame: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    out = frame.copy()
    if start:
        out = out[out["timestamp"] >= pd.Timestamp(start)]
    if end:
        out = out[out["timestamp"] < pd.Timestamp(end)]
    return out.reset_index(drop=True)


def _x(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return prepare_features(frame, side_hint=0, close=_close(frame), feature_cols=cols)


def _legacy_v4_parent_features(train: pd.DataFrame, eval_df: pd.DataFrame) -> list[str]:
    parent = joblib.load(ALPHA4_PARENT)
    common = set(train.columns) & set(eval_df.columns)
    out: list[str] = []
    for col in parent["feature_cols"]:
        name = str(col)
        if name in out:
            continue
        if name in common or name == "side_hint" or name.startswith(("mom_", "abs_mom_")):
            out.append(name)
    return out


def _feature_cols(track: str, train: pd.DataFrame, eval_df: pd.DataFrame) -> list[str]:
    if track == "legacy_v4_parent":
        return _legacy_v4_parent_features(train, eval_df)
    if track == "regime4_mapped_core":
        return _alpha4_mapped_features(train, eval_df, include_future=False)
    if track == "regime4_mapped_future":
        return _alpha4_mapped_features(train, eval_df, include_future=True)
    raise ValueError(f"unknown track: {track}")


def _metrics(frame: pd.DataFrame, proba: np.ndarray, prob: float, margin: float, fee: float, slip: float, exposure: float, max_hold: int) -> dict[str, Any]:
    actions = _decide_actions(proba, prob, margin)
    return {
        f"cost{m}": _backtest_actions(
            frame,
            actions,
            fee=float(fee) * float(m),
            slip=float(slip) * float(m),
            unit_exposure=float(exposure),
            max_hold_bars=int(max_hold),
        )
        for m in (1, 2, 3)
    }


def _assert_same_clock(a: pd.DataFrame, b: pd.DataFrame, name: str) -> None:
    if len(a) != len(b):
        raise RuntimeError(f"{name} row mismatch: {len(a)} != {len(b)}")
    if not a["timestamp"].reset_index(drop=True).equals(b["timestamp"].reset_index(drop=True)):
        raise RuntimeError(f"{name} timestamp mismatch")


def main() -> None:
    p = argparse.ArgumentParser(description="Compare legacy clean_regime v4 versus regime4 HGB action master under identical tuning.")
    p.add_argument("--legacy-train-csv", type=Path, default=DEFAULT_LEGACY_TRAIN)
    p.add_argument("--legacy-eval-csv", type=Path, default=DEFAULT_LEGACY_EVAL)
    p.add_argument("--regime4-train-csv", type=Path, default=DEFAULT_REGIME4_TRAIN)
    p.add_argument("--regime4-eval-csv", type=Path, default=DEFAULT_REGIME4_EVAL)
    p.add_argument("--manifest", type=Path, default=DEFAULT_PREPROCESS_MANIFEST)
    p.add_argument("--clean4-report", type=Path, default=DEFAULT_CLEAN4_REPORT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--train-end", default="2025-10-01")
    p.add_argument("--val-start", default="2025-10-01")
    p.add_argument("--val-end", default="2026-01-01")
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--tracks", default="legacy_v4_parent,regime4_mapped_core,regime4_mapped_future")
    p.add_argument("--label-cfgs", default="unit_c014_a20_h025,unit_c018_a22_h030,unit_c022_a25_h035,size_l1_c020")
    p.add_argument("--weight-modes", default="balanced,quality")
    p.add_argument("--prob-thresholds", default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.93,0.95")
    p.add_argument("--margin-thresholds", default="0.00,0.03,0.05,0.08,0.12,0.16,0.20")
    p.add_argument("--max-hold-bars", type=int, default=96)
    p.add_argument("--unit-exposure", type=float, default=1.0)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--seed", type=int, default=51001)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    legacy_all = _read(args.legacy_train_csv)
    legacy_eval = _read(args.legacy_eval_csv)
    regime4_all = _read(args.regime4_train_csv)
    regime4_eval = _read(args.regime4_eval_csv)
    _assert_same_clock(legacy_all, regime4_all, "train")
    _assert_same_clock(legacy_eval, regime4_eval, "eval")

    frames = {
        "legacy_v4_parent": {
            "train_all": legacy_all,
            "eval": legacy_eval,
        },
        "regime4_mapped_core": {
            "train_all": regime4_all,
            "eval": regime4_eval,
        },
        "regime4_mapped_future": {
            "train_all": regime4_all,
            "eval": regime4_eval,
        },
    }
    for payload in frames.values():
        payload["train"] = _split(payload["train_all"], None, args.train_end)
        payload["val"] = _split(payload["train_all"], args.val_start, args.val_end)

    audit = _verify_state24_sticky090_inputs(regime4_all, regime4_eval, args.manifest, args.clean4_report)
    cfgs = _cfgs()
    selected_cfgs = [x.strip() for x in str(args.label_cfgs).split(",") if x.strip()]
    tracks = [x.strip() for x in str(args.tracks).split(",") if x.strip()]
    weight_modes = [x.strip() for x in str(args.weight_modes).split(",") if x.strip()]
    hgb_specs = _hgb_specs()

    print(
        json.dumps(
            {
                "stage": "start",
                "model_id": MODEL_ID,
                "tracks": tracks,
                "label_cfgs": selected_cfgs,
                "weight_modes": weight_modes,
                "hgb_specs": [asdict(s) for s in hgb_specs],
                "rows": {
                    "train": len(frames["legacy_v4_parent"]["train"]),
                    "validation": len(frames["legacy_v4_parent"]["val"]),
                    "oos": len(legacy_eval),
                },
                "audit": {
                    "expected_model_found_in_manifest": audit.get("expected_model_found_in_manifest"),
                    "legacy_v4_count_in_regime4": audit.get("legacy_v4_count"),
                },
            },
            ensure_ascii=False,
            default=_json_default,
        ),
        flush=True,
    )

    label_payloads: dict[str, tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]] = {}
    label_cols = _alpha4_mapped_features(regime4_all, regime4_eval, include_future=False)
    for cfg_name in selected_cfgs:
        cfg = cfgs[cfg_name]
        _, y, train_meta = build_training_set(frames["regime4_mapped_core"]["train"], cfg=cfg, stride_bars=int(args.stride), batch_size=512, feature_cols=label_cols)
        valid_idx = _valid_indices(len(frames["regime4_mapped_core"]["train"]), int(cfg.max_train_horizon_bars), int(args.stride))
        action = np.asarray(y["action"], dtype=np.int64)
        quality = np.asarray(y["quality"], dtype=np.float64)
        report = {
            "rows": int(len(action)),
            "action_counts": {"cash": int(np.sum(action == 0)), "long": int(np.sum(action == 1)), "short": int(np.sum(action == 2))},
            "trade_ratio": float(np.mean(action != 0)),
            "quality_mean": float(np.mean(quality)),
            "quality_p95": float(np.quantile(quality, 0.95)),
        }
        label_payloads[cfg_name] = (y, valid_idx, report)
        print(json.dumps({"stage": "label_built", "label_cfg": cfg_name, "label_report": report}, ensure_ascii=False, default=_json_default), flush=True)

    feature_cache: dict[str, tuple[list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    for track in tracks:
        payload = frames[track]
        cols = _feature_cols(track, payload["train_all"], payload["eval"])
        feature_cache[track] = (cols, _x(payload["train"], cols), _x(payload["val"], cols), _x(payload["eval"], cols))
        print(
            json.dumps(
                {
                    "stage": "features_ready",
                    "track": track,
                    "feature_count": len(cols),
                    "legacy_v4_count": int(sum(c.startswith("clean_regime_2024_unsup_v4_") for c in cols)),
                    "regime4_count": int(sum(c.startswith(CLEAN4_PREFIX) for c in cols)),
                    "future_pred_count": int(sum(c.startswith("regime4_pred_") for c in cols)),
                    "has_tp_sl_action_score": "tp_sl_action_score" in cols,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    rows: list[dict[str, Any]] = []
    total = len(selected_cfgs) * len(tracks) * len(weight_modes) * len(hgb_specs)
    done = 0
    for cfg_i, cfg_name in enumerate(selected_cfgs):
        y, valid_idx, label_report = label_payloads[cfg_name]
        y_action = np.asarray(y["action"], dtype=np.int64)
        for track_i, track in enumerate(tracks):
            cols, x_train_full, x_val, x_eval = feature_cache[track]
            x_train = x_train_full.iloc[valid_idx].reset_index(drop=True)
            val_frame = frames[track]["val"]
            eval_frame = frames[track]["eval"]
            for weight_i, weight_mode in enumerate(weight_modes):
                sample_weight = _weights(y, weight_mode)
                for spec_i, spec in enumerate(hgb_specs):
                    done += 1
                    print(json.dumps({"stage": "fit", "done": done, "total": total, "label_cfg": cfg_name, "track": track, "weight_mode": weight_mode, "hgb": spec.name}, ensure_ascii=False), flush=True)
                    model = _fit_hgb(x_train, y_action, sample_weight, spec, int(args.seed) + cfg_i * 1000 + track_i * 200 + weight_i * 50 + spec_i)
                    val_proba = _predict_proba_3(model, x_val)
                    eval_proba = _predict_proba_3(model, x_eval)
                    best: dict[str, Any] | None = None
                    for prob in _grid(args.prob_thresholds):
                        for margin in _grid(args.margin_thresholds):
                            val_metrics = _metrics(val_frame, val_proba, prob, margin, args.fee, args.slip, args.unit_exposure, args.max_hold_bars)
                            score = _score(val_metrics)
                            if best is None or score > float(best["score"]):
                                best = {
                                    "label_cfg": cfg_name,
                                    "track": track,
                                    "weight_mode": weight_mode,
                                    "hgb": asdict(spec),
                                    "prob_threshold": float(prob),
                                    "margin_threshold": float(margin),
                                    "score": float(score),
                                    "validation_metrics": val_metrics,
                                }
                    assert best is not None
                    oos_metrics = _metrics(eval_frame, eval_proba, best["prob_threshold"], best["margin_threshold"], args.fee, args.slip, args.unit_exposure, args.max_hold_bars)
                    artifact = args.out_dir / f"{cfg_name}_{track}_{weight_mode}_{spec.name}_action_hgb.joblib"
                    joblib.dump(
                        {
                            "model_id": MODEL_ID,
                            "model": model,
                            "feature_cols": cols,
                            "label_cfg_name": cfg_name,
                            "track": track,
                            "weight_mode": weight_mode,
                            "hgb": asdict(spec),
                            "label_report": label_report,
                            "selected_thresholds": {
                                "prob_threshold": best["prob_threshold"],
                                "margin_threshold": best["margin_threshold"],
                                "max_hold_bars": int(args.max_hold_bars),
                            },
                        },
                        artifact,
                    )
                    row = {
                        **best,
                        "oos_metrics": oos_metrics,
                        "label_report": label_report,
                        "feature_count": len(cols),
                        "legacy_v4_count": int(sum(c.startswith("clean_regime_2024_unsup_v4_") for c in cols)),
                        "regime4_count": int(sum(c.startswith(CLEAN4_PREFIX) for c in cols)),
                        "future_pred_count": int(sum(c.startswith("regime4_pred_") for c in cols)),
                        "artifact": str(artifact),
                    }
                    rows.append(row)
                    print(
                        json.dumps(
                            {
                                "stage": "candidate",
                                "label_cfg": cfg_name,
                                "track": track,
                                "weight_mode": weight_mode,
                                "hgb": spec.name,
                                "score": best["score"],
                                "selected": {"prob": best["prob_threshold"], "margin": best["margin_threshold"]},
                                "val_cost1": best["validation_metrics"]["cost1"],
                                "oos_cost1": oos_metrics["cost1"],
                            },
                            ensure_ascii=False,
                            default=_json_default,
                        ),
                        flush=True,
                    )

    best = max(rows, key=lambda r: float(r["score"]))
    summary = {
        "model_id": MODEL_ID,
        "design": "Identical HGB action-only tuning comparison between legacy clean_regime_2024_unsup_v4 parent contract and regime4 mapped contracts.",
        "legacy_csv": {"train": str(args.legacy_train_csv), "eval": str(args.legacy_eval_csv)},
        "regime4_csv": {"train": str(args.regime4_train_csv), "eval": str(args.regime4_eval_csv)},
        "state24_sticky090_audit": audit,
        "experiments": rows,
        "best": best,
        "top20": sorted(rows, key=lambda r: float(r["score"]), reverse=True)[:20],
    }
    summary_path = args.out_dir / "alpha5_10_hgb_legacy_v4_vs_regime4_summary.json"
    grid_path = args.out_dir / "alpha5_10_hgb_legacy_v4_vs_regime4_grid.csv"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "label_cfg": r["label_cfg"],
                "track": r["track"],
                "weight_mode": r["weight_mode"],
                "hgb_name": r["hgb"]["name"],
                "score": r["score"],
                "prob_threshold": r["prob_threshold"],
                "margin_threshold": r["margin_threshold"],
                "feature_count": r["feature_count"],
                "legacy_v4_count": r["legacy_v4_count"],
                "regime4_count": r["regime4_count"],
                "future_pred_count": r["future_pred_count"],
                "label_trade_ratio": r["label_report"]["trade_ratio"],
                "val_cost1_pnl": r["validation_metrics"]["cost1"]["pnl"],
                "val_cost1_mdd": r["validation_metrics"]["cost1"]["mdd"],
                "val_cost1_trades": r["validation_metrics"]["cost1"]["trades"],
                "val_cost1_tpd": r["validation_metrics"]["cost1"]["trades_per_day"],
                "oos_cost1_pnl": r["oos_metrics"]["cost1"]["pnl"],
                "oos_cost1_mdd": r["oos_metrics"]["cost1"]["mdd"],
                "oos_cost1_trades": r["oos_metrics"]["cost1"]["trades"],
                "oos_cost1_tpd": r["oos_metrics"]["cost1"]["trades_per_day"],
                "oos_cost2_pnl": r["oos_metrics"]["cost2"]["pnl"],
                "oos_cost3_pnl": r["oos_metrics"]["cost3"]["pnl"],
                "artifact": r["artifact"],
            }
            for r in rows
        ]
    ).sort_values("score", ascending=False).to_csv(grid_path, index=False)
    print(
        json.dumps(
            {
                "stage": "complete",
                "summary": str(summary_path),
                "grid": str(grid_path),
                "best": {
                    "track": best["track"],
                    "label_cfg": best["label_cfg"],
                    "weight_mode": best["weight_mode"],
                    "hgb": best["hgb"]["name"],
                    "score": best["score"],
                    "oos_cost1": best["oos_metrics"]["cost1"],
                },
            },
            ensure_ascii=False,
            default=_json_default,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
