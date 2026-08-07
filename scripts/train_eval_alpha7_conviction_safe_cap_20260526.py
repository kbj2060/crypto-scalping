#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.research_alpha_model_synergy_oos_20260525 import _parent_for_features  # noqa: E402
from scripts.train_eval_alpha5_regime4_tp18_sl10_no_teacher_no_deep_20260517 import (  # noqa: E402
    _compact_costs,
    _metrics,
)
from scripts.train_eval_alpha7_meta_fallback_cash_router_20260526 import (  # noqa: E402
    COMBO_SUMMARY,
    EVAL_CSV,
    LIVE_DIR,
    PRIMARY_PARENT,
    PRIMARY_SUMMARY,
    TRAIN_CSV,
    _active,
    _combine_primary_fallback,
    _json_default,
    _load_best_scale_runtime,
    _predict_scaled,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _read  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import _grid as _runner_grid  # noqa: E402


FALLBACK_PARENT = LIVE_DIR / "fallback_alpha43_no_legacy_parent.pkl"
FALLBACK_SUMMARY = LIVE_DIR / "fallback_alpha43_no_legacy_summary.json"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_conviction_safe_cap_20260526"


def _num(frame: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if col not in frame.columns:
        return np.full(len(frame), float(default), dtype=np.float64)
    return (
        pd.to_numeric(frame[col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(default)
        .to_numpy(dtype=np.float64)
    )


def _side(dec: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(dec["side"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)


def _copy_zero_rows(out: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    if not np.any(mask):
        return out
    for col in (
        "action",
        "side",
        "position_fraction",
        "notional_exposure",
        "take_profit",
        "stop_loss",
        "max_hold_bars",
        "cooldown_bars",
        "quality_score",
        "confidence",
    ):
        if col in out.columns:
            out.loc[mask, col] = 0
    if "leverage" in out.columns:
        out.loc[mask, "leverage"] = 1.0
    return out


def _apply_overlay(
    frame: pd.DataFrame,
    final_dec: pd.DataFrame,
    primary_dec: pd.DataFrame,
    fallback_dec: pd.DataFrame,
    *,
    block_quality: float,
    block_confidence: float,
    block_instability: float,
    boost_quality: float,
    boost_confidence: float,
    boost_tp_edge: float,
    boost_instability: float,
    boost_entropy: float,
    boost_target_notional: float,
    reduce_quality: float,
    reduce_target_notional: float,
    fallback_tp_scale: float,
    source_mode: str,
) -> pd.DataFrame:
    out = final_dec.copy().reset_index(drop=True)
    active = _active(out)
    if not np.any(active):
        return out
    primary_active = _active(primary_dec.reset_index(drop=True))
    fallback_active = (~primary_active) & _active(fallback_dec.reset_index(drop=True))
    if source_mode == "primary_only":
        eligible = active & primary_active
    elif source_mode == "fallback_only":
        eligible = active & fallback_active
    else:
        eligible = active

    quality = _num(out, "quality_score")
    confidence = _num(out, "confidence")
    tp_edge = np.abs(_num(frame, "tp_sl_action_score"))
    instability = np.maximum(
        _num(frame, "clean_regime4_2024_unsup_v1_instability_prob"),
        _num(frame, "regime4_pred_instability_prob"),
    )
    entropy = _num(frame, "clean_regime4_2024_unsup_v1_entropy", 0.0)
    leverage = np.maximum(_num(out, "leverage", 1.0), 1e-12)
    notional = _num(out, "notional_exposure")

    block_mask = eligible & (
        (quality < float(block_quality))
        | (confidence < float(block_confidence))
        | (instability > float(block_instability))
    )
    out = _copy_zero_rows(out, block_mask)

    still_active = _active(out)
    reduce_mask = (
        eligible
        & still_active
        & (~block_mask)
        & (quality < float(reduce_quality))
        & (reduce_target_notional > 0.0)
    )
    if np.any(reduce_mask):
        reduced = np.minimum(notional, float(reduce_target_notional))
        out.loc[reduce_mask, "notional_exposure"] = reduced[reduce_mask]
        out.loc[reduce_mask, "position_fraction"] = reduced[reduce_mask] / leverage[reduce_mask]

    boost_mask = (
        eligible
        & _active(out)
        & (~block_mask)
        & (quality >= float(boost_quality))
        & (confidence >= float(boost_confidence))
        & (tp_edge >= float(boost_tp_edge))
        & (instability <= float(boost_instability))
        & (entropy <= float(boost_entropy))
    )
    if np.any(boost_mask):
        boosted = np.maximum(_num(out, "notional_exposure"), float(boost_target_notional))
        out.loc[boost_mask, "notional_exposure"] = boosted[boost_mask]
        out.loc[boost_mask, "position_fraction"] = boosted[boost_mask] / leverage[boost_mask]

    if abs(float(fallback_tp_scale) - 1.0) > 1e-12 and np.any(fallback_active):
        tp = _num(out, "take_profit")
        scale_mask = fallback_active & _active(out) & (tp > 0.0)
        out.loc[scale_mask, "take_profit"] = tp[scale_mask] * float(fallback_tp_scale)
    return out


def _score_row(metrics: dict[str, Any]) -> float:
    c2 = dict(metrics["cost2"])
    c3 = dict(metrics["cost3"])
    return float(
        c3["pnl"] / max(abs(c3["mdd"]), 1e-9)
        + 0.10 * c2["pnl"]
        + 0.02 * c3["trades"]
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Search Alpha7 conviction safe-cap overlays.")
    ap.add_argument("--train-csv", type=Path, default=TRAIN_CSV)
    ap.add_argument("--eval-csv", type=Path, default=EVAL_CSV)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)

    primary_parent = joblib.load(PRIMARY_PARENT)
    primary_rt = _load_best_scale_runtime(PRIMARY_SUMMARY)
    fallback_parent = joblib.load(FALLBACK_PARENT)
    fallback_rt = _load_best_scale_runtime(FALLBACK_SUMMARY)

    primary_val = _predict_scaled(primary_parent, val_df, primary_rt)
    primary_eval = _predict_scaled(primary_parent, eval_df, primary_rt)
    fallback_val = _predict_scaled(fallback_parent, val_df, fallback_rt)
    fallback_eval = _predict_scaled(fallback_parent, eval_df, fallback_rt)
    baseline_val = _combine_primary_fallback(primary_val, fallback_val)
    baseline_eval = _combine_primary_fallback(primary_eval, fallback_eval)

    ref_parent = joblib.load(v31.DEFAULT_PARENT)
    parent_for_features = _parent_for_features(list(ref_parent["feature_cols"]))
    fee = float(ref_parent["config"]["fee"])
    slip = float(ref_parent["config"]["slip"])
    runner = joblib.load(v31.DEFAULT_JACKPOT)["cost_runner"]
    runner_cfg = next(c for c in _runner_grid() if c.name == "v21_2_parent_noop")
    baseline_combo = json.loads(COMBO_SUMMARY.read_text(encoding="utf-8"))
    baseline_val_metrics = _compact_costs(
        _metrics(
            val_df,
            parent_for_features=parent_for_features,
            runner=runner,
            runner_cfg=runner_cfg,
            dec=baseline_val,
            fee=fee,
            slip=slip,
        )
    )
    baseline_eval_metrics = _compact_costs(
        _metrics(
            eval_df,
            parent_for_features=parent_for_features,
            runner=runner,
            runner_cfg=runner_cfg,
            dec=baseline_eval,
            fee=fee,
            slip=slip,
        )
    )

    rows: list[dict[str, Any]] = []
    best_by_val: dict[str, Any] | None = None
    best_by_oos: dict[str, Any] | None = None

    block_opts = [
        (0.00, 0.00, 1.00),
        (0.02, 0.52, 0.80),
        (0.03, 0.58, 0.70),
    ]
    reduce_opts = [
        (0.00, 0.0),
        (0.04, 1.25),
    ]
    boost_opts = [
        (0.045, 0.60, 0.00, 0.80, 99.0, 3.0),
        (0.045, 0.60, 0.10, 0.80, 99.0, 4.0),
        (0.045, 0.70, 0.10, 0.60, 1.10, 4.0),
        (0.055, 0.60, 0.00, 0.80, 99.0, 4.0),
        (0.055, 0.70, 0.10, 0.60, 1.10, 4.0),
        (0.055, 0.70, 0.10, 0.60, 1.10, 5.0),
    ]

    total = 0
    active_block_opts = block_opts[:2]
    active_source_modes = ("all",)
    active_reduce_opts = reduce_opts
    active_boost_opts = boost_opts[:4]
    for source_mode in active_source_modes:
        for block_quality, block_confidence, block_instability in active_block_opts:
            for reduce_quality, reduce_target_notional in active_reduce_opts:
                for (
                    boost_quality,
                    boost_confidence,
                    boost_tp_edge,
                    boost_instability,
                    boost_entropy,
                    boost_target_notional,
                ) in active_boost_opts:
                    for fallback_tp_scale in (1.0, 0.80):
                        total += 1
    done = 0

    for source_mode in active_source_modes:
        for block_quality, block_confidence, block_instability in active_block_opts:
            for reduce_quality, reduce_target_notional in active_reduce_opts:
                for (
                    boost_quality,
                    boost_confidence,
                    boost_tp_edge,
                    boost_instability,
                    boost_entropy,
                    boost_target_notional,
                ) in active_boost_opts:
                    for fallback_tp_scale in (1.0, 0.80):
                        done += 1
                        if done == 1 or done % 12 == 0 or done == total:
                            print(
                                f"[alpha7_conviction_safe_cap] {done}/{total} source={source_mode} "
                                f"block=({block_quality:.3f},{block_confidence:.2f},{block_instability:.2f}) "
                                f"boost=({boost_quality:.3f},{boost_confidence:.2f},{boost_tp_edge:.2f},{boost_target_notional:.2f}) "
                                f"fb_tp={fallback_tp_scale:.2f}",
                                flush=True,
                            )
                        val_dec = _apply_overlay(
                            val_df,
                            baseline_val,
                            primary_val,
                            fallback_val,
                            block_quality=block_quality,
                            block_confidence=block_confidence,
                            block_instability=block_instability,
                            boost_quality=boost_quality,
                            boost_confidence=boost_confidence,
                            boost_tp_edge=boost_tp_edge,
                            boost_instability=boost_instability,
                            boost_entropy=boost_entropy,
                            boost_target_notional=boost_target_notional,
                            reduce_quality=reduce_quality,
                            reduce_target_notional=reduce_target_notional,
                            fallback_tp_scale=fallback_tp_scale,
                            source_mode=source_mode,
                        )
                        eval_dec = _apply_overlay(
                            eval_df,
                            baseline_eval,
                            primary_eval,
                            fallback_eval,
                            block_quality=block_quality,
                            block_confidence=block_confidence,
                            block_instability=block_instability,
                            boost_quality=boost_quality,
                            boost_confidence=boost_confidence,
                            boost_tp_edge=boost_tp_edge,
                            boost_instability=boost_instability,
                            boost_entropy=boost_entropy,
                            boost_target_notional=boost_target_notional,
                            reduce_quality=reduce_quality,
                            reduce_target_notional=reduce_target_notional,
                            fallback_tp_scale=fallback_tp_scale,
                            source_mode=source_mode,
                        )
                        val_metrics = _compact_costs(
                            _metrics(
                                val_df,
                                parent_for_features=parent_for_features,
                                runner=runner,
                                runner_cfg=runner_cfg,
                                dec=val_dec,
                                fee=fee,
                                slip=slip,
                            )
                        )
                        eval_metrics = _compact_costs(
                            _metrics(
                                eval_df,
                                parent_for_features=parent_for_features,
                                runner=runner,
                                runner_cfg=runner_cfg,
                                dec=eval_dec,
                                fee=fee,
                                slip=slip,
                            )
                        )
                        row = {
                            "source_mode": source_mode,
                            "block_quality": float(block_quality),
                            "block_confidence": float(block_confidence),
                            "block_instability": float(block_instability),
                            "reduce_quality": float(reduce_quality),
                            "reduce_target_notional": float(reduce_target_notional),
                            "boost_quality": float(boost_quality),
                            "boost_confidence": float(boost_confidence),
                            "boost_tp_edge": float(boost_tp_edge),
                            "boost_instability": float(boost_instability),
                            "boost_entropy": float(boost_entropy),
                            "boost_target_notional": float(boost_target_notional),
                            "fallback_tp_scale": float(fallback_tp_scale),
                            "val_score": float(_score_row(val_metrics)),
                            "val_cost3_pnl": float(val_metrics["cost3"]["pnl"]),
                            "val_cost3_mdd": float(val_metrics["cost3"]["mdd"]),
                            "val_cost3_trades": int(val_metrics["cost3"]["trades"]),
                            "oos_score": float(_score_row(eval_metrics)),
                            "oos_cost3_pnl": float(eval_metrics["cost3"]["pnl"]),
                            "oos_cost3_mdd": float(eval_metrics["cost3"]["mdd"]),
                            "oos_cost3_trades": int(eval_metrics["cost3"]["trades"]),
                            "delta_vs_baseline": float(eval_metrics["cost3"]["pnl"]) - float(baseline_eval_metrics["cost3"]["pnl"]),
                        }
                        rows.append(row)
                        if best_by_val is None or row["val_score"] > best_by_val["val_score"]:
                            best_by_val = row
                        if best_by_oos is None or row["oos_cost3_pnl"] > best_by_oos["oos_cost3_pnl"]:
                            best_by_oos = row

    assert best_by_val is not None and best_by_oos is not None
    ranking = pd.DataFrame(rows).sort_values(["val_score", "oos_cost3_pnl"], ascending=[False, False])
    ranking_path = args.out_dir / "ranking.csv"
    ranking.to_csv(ranking_path, index=False)
    report = {
        "model_id": "alpha7_conviction_safe_cap_20260526",
        "design": "Keep Alpha7 entry side/action unchanged, then apply block/base/boost conviction safe-cap editing plus optional fallback TP shrink. Selection is 2025Q4 only; 2026 Jan-Feb is fixed OOS.",
        "train_csv": str(args.train_csv),
        "eval_csv": str(args.eval_csv),
        "baseline": {
            "combo_selected_metrics": baseline_combo.get("selected_metrics"),
            "validation_metrics": baseline_val_metrics,
            "oos_metrics": baseline_eval_metrics,
        },
        "best_by_validation": best_by_val,
        "best_by_oos": best_by_oos,
        "artifacts": {
            "ranking": str(ranking_path),
        },
    }
    report_path = args.out_dir / "summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "best_by_validation": best_by_val, "best_by_oos": best_by_oos}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
