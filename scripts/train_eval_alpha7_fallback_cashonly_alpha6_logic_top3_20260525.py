#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
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

from ensemble.fully_learned_governor_policy import (  # noqa: E402
    ACTION_CASH,
    FullyLearnedGovernorConfig,
    build_training_set,
    predict_policy_frame,
    train_policy,
)
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.alpha7_experiment_config import get_live_baseline  # noqa: E402
from scripts.alpha6_catboost_entry_quality_exit_policy_20260522 import (  # noqa: E402
    EQEConfig,
    _apply_label_preset,
    _build_entry_labels,
)
from scripts.research_alpha_model_synergy_oos_20260525 import _parent_for_features  # noqa: E402
from scripts.train_eval_alpha5_regime4_tp18_sl10_no_teacher_no_deep_20260517 import (  # noqa: E402
    BASE_PARENT,
    _compact_costs,
    _metrics,
    _read,
    _score,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import _grid as _runner_grid  # noqa: E402


BASELINE = get_live_baseline()
LIVE_DIR = BASELINE.live_dir
TRAIN_CSV = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv"
EVAL_CSV = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2026_regime4_state24_sticky090_tp18_sl10_fixed.csv"
PRIMARY_PARENT = BASELINE.primary_parent
PRIMARY_SUMMARY = BASELINE.primary_summary
FALLBACK_PARENT = BASELINE.fallback_parent
FALLBACK_SUMMARY = BASELINE.fallback_summary
COMBO_SUMMARY = BASELINE.combo_summary
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_fallback_cashonly_alpha6_logic_top3_20260525"
DEFAULT_PRESETS = ("current_quality", "high_precision_robust", "short_horizon_robust")


def _alpha7_valid_idx(frame: pd.DataFrame, *, stride: int, max_horizon: int) -> np.ndarray:
    return np.arange(0, max(0, len(frame) - int(max_horizon) - 1), max(1, int(stride)), dtype=np.int64)


def _active(dec: pd.DataFrame) -> np.ndarray:
    action = pd.to_numeric(dec["action"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    side = pd.to_numeric(dec["side"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    return (action != ACTION_CASH) & (side != 0)


def _copy_rows(target: pd.DataFrame, source: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    out = target.copy()
    for col in source.columns:
        out.loc[mask, col] = source.loc[mask, col].to_numpy()
    return out


def _combine_primary_fallback(primary: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    primary = primary.reset_index(drop=True)
    fallback = fallback.reset_index(drop=True)
    return _copy_rows(primary, fallback, ~_active(primary) & _active(fallback))


def _load_best_scale_runtime(summary_path: Path) -> alpha2.Alpha2Runtime | None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    target = summary.get("best_by_selection")
    experiments = summary.get("experiments", [])
    if isinstance(target, dict):
        rt = target.get("selected_parent_scale_runtime")
        if rt:
            return alpha2.Alpha2Runtime(
                name=str(rt["name"]),
                confidence=float(rt["confidence"]),
                parent_notional_scale=float(rt["parent_notional_scale"]),
                max_notional=float(rt["max_notional"]),
            )
    for exp in experiments:
        if target is not None and exp.get("name") != target:
            continue
        rt = exp.get("selected_parent_scale_runtime")
        if rt:
            return alpha2.Alpha2Runtime(
                name=str(rt["name"]),
                confidence=float(rt["confidence"]),
                parent_notional_scale=float(rt["parent_notional_scale"]),
                max_notional=float(rt["max_notional"]),
            )
    return None


def _scale_decisions(dec: pd.DataFrame, rt: alpha2.Alpha2Runtime | None) -> pd.DataFrame:
    if rt is None:
        return dec.reset_index(drop=True)
    return alpha2._scale_parent_notional(dec, rt).reset_index(drop=True)


def _max_hold_bucket_index(horizon: int, buckets: tuple[int, ...]) -> int:
    for idx, bucket in enumerate(buckets):
        if int(horizon) <= int(bucket):
            return idx
    return len(buckets) - 1


def _build_cashonly_training_set(
    *,
    train_df: pd.DataFrame,
    x_train: pd.DataFrame,
    native_y: dict[str, np.ndarray],
    candidate_idx: np.ndarray,
    primary_train_dec: pd.DataFrame,
    preset: str,
    stride: int,
    batch_size: int,
    session_topk: int,
    max_hold_buckets: tuple[int, ...],
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    eqe_cfg = _apply_label_preset(EQEConfig(), preset)
    alpha6_valid, alpha6_y, alpha6_meta = _build_entry_labels(
        train_df,
        eqe_cfg,
        stride_bars=int(stride),
        batch_size=int(batch_size),
        adaptive_sampling=False,
        label_preset=str(preset),
        session_topk=int(session_topk),
    )
    alpha6_pos = pd.Series(np.arange(len(alpha6_valid), dtype=np.int64), index=alpha6_valid)
    lookup = alpha6_pos.reindex(candidate_idx)
    if lookup.isna().any():
        missing_idx = int(candidate_idx[np.flatnonzero(lookup.isna().to_numpy())[0]])
        raise ValueError(f"alpha6 label logic missing candidate index {missing_idx} for preset={preset}")
    take = lookup.to_numpy(dtype=np.int64)
    primary_busy = _active(primary_train_dec.iloc[candidate_idx].reset_index(drop=True))
    cash_mask = ~primary_busy
    if not np.any(cash_mask):
        raise ValueError(f"no primary cash candidates for preset={preset}")

    x_cash = x_train.iloc[cash_mask].reset_index(drop=True)
    y = {k: np.asarray(v)[cash_mask].copy() for k, v in native_y.items()}
    action = alpha6_y["action"][take][cash_mask].astype(np.int64)
    quality = alpha6_y["quality"][take][cash_mask].astype(np.float64)
    target_horizon = alpha6_y["target_horizon"][take][cash_mask].astype(np.int64)
    hold_idx = np.asarray(
        [_max_hold_bucket_index(int(v), max_hold_buckets) if int(v) > 0 else 0 for v in target_horizon],
        dtype=np.int64,
    )
    y["action"] = action
    y["quality"] = quality
    trade = action != 0
    y["max_hold"][trade] = hold_idx[trade]
    meta = {
        "preset": str(preset),
        "alpha6_cfg": {
            "max_train_horizon_bars": int(eqe_cfg.max_train_horizon_bars),
            "score_horizons": list(int(v) for v in eqe_cfg.score_horizons),
            "min_net_edge": float(eqe_cfg.min_net_edge),
            "dynamic_min_edge_atr_frac": float(eqe_cfg.dynamic_min_edge_atr_frac),
            "direction_margin": float(eqe_cfg.direction_margin),
            "mae_penalty_lambda": float(eqe_cfg.mae_penalty_lambda),
            "path_vol_penalty_lambda": float(eqe_cfg.path_vol_penalty_lambda),
            "hold_penalty": float(eqe_cfg.hold_penalty),
        },
        "alpha6_label_meta": alpha6_meta,
        "candidate_rows": int(len(candidate_idx)),
        "primary_busy_rows": int(primary_busy.sum()),
        "primary_cash_rows": int(cash_mask.sum()),
        "cash_train_rows": int(len(x_cash)),
        "cash_trade_rows": int(trade.sum()),
        "cash_trade_ratio": float(np.mean(trade.astype(np.float64))),
        "action_distribution_cash_only": pd.Series(action).value_counts().sort_index().to_dict(),
        "quality_mean": float(np.mean(quality)),
        "quality_p95": float(np.quantile(quality, 0.95)),
        "target_horizon_distribution": pd.Series(target_horizon[trade]).value_counts().sort_index().to_dict(),
    }
    return x_cash, y, meta


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train fallback-only cash-region parents using Alpha6 label logic on Alpha7 live contract.")
    ap.add_argument("--train-csv", type=Path, default=TRAIN_CSV)
    ap.add_argument("--eval-csv", type=Path, default=EVAL_CSV)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--presets", default=",".join(DEFAULT_PRESETS))
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--seed", type=int, default=72526)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--session-topk", type=int, default=2)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    presets = [v.strip() for v in str(args.presets).split(",") if v.strip()]

    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    cutoff = pd.Timestamp("2025-10-01")
    train_df = train_all[train_all["timestamp"] < cutoff].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= cutoff].reset_index(drop=True)

    primary_bundle = joblib.load(PRIMARY_PARENT)
    fallback_ref = joblib.load(FALLBACK_PARENT)
    label_cfg = FullyLearnedGovernorConfig(**dict(joblib.load(BASE_PARENT)["config"]))
    feature_cols = list(fallback_ref["feature_cols"])
    x_train, y_train_native, train_meta = build_training_set(
        train_df,
        cfg=label_cfg,
        stride_bars=int(args.stride),
        batch_size=int(args.batch_size),
        feature_cols=feature_cols,
    )
    candidate_idx = _alpha7_valid_idx(train_df, stride=int(args.stride), max_horizon=int(label_cfg.max_train_horizon_bars))
    if len(candidate_idx) != len(x_train):
        raise ValueError("fallback candidate contract mismatch")

    primary_scale_rt = _load_best_scale_runtime(PRIMARY_SUMMARY)
    fallback_scale_baseline = _load_best_scale_runtime(FALLBACK_SUMMARY)
    primary_train_dec = _scale_decisions(predict_policy_frame(primary_bundle, train_df, close=_close(train_df)), primary_scale_rt)
    primary_val_dec = _scale_decisions(predict_policy_frame(primary_bundle, val_df, close=_close(val_df)), primary_scale_rt)
    primary_eval_dec = _scale_decisions(predict_policy_frame(primary_bundle, eval_df, close=_close(eval_df)), primary_scale_rt)

    noop_runner = joblib.load(v31.DEFAULT_JACKPOT)["cost_runner"]
    noop_cfg = next(c for c in _runner_grid() if c.name == "v21_2_parent_noop")
    ref_parent = _parent_for_features(list(joblib.load(v31.DEFAULT_PARENT)["feature_cols"]))
    fee = float(joblib.load(v31.DEFAULT_PARENT)["config"]["fee"])
    slip = float(joblib.load(v31.DEFAULT_PARENT)["config"]["slip"])
    baseline_combo = json.loads(COMBO_SUMMARY.read_text(encoding="utf-8"))

    baseline_fallback_dec = _scale_decisions(predict_policy_frame(fallback_ref, eval_df, close=_close(eval_df)), fallback_scale_baseline)
    baseline_combo_eval = _combine_primary_fallback(primary_eval_dec, baseline_fallback_dec)
    baseline_combo_metrics = _compact_costs(
        _metrics(eval_df, parent_for_features=ref_parent, runner=noop_runner, runner_cfg=noop_cfg, dec=baseline_combo_eval, fee=fee, slip=slip)
    )

    ranking_rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for i, preset in enumerate(presets):
        x_cash, y_train, label_meta = _build_cashonly_training_set(
            train_df=train_df,
            x_train=x_train,
            native_y=y_train_native,
            candidate_idx=candidate_idx,
            primary_train_dec=primary_train_dec,
            preset=preset,
            stride=int(args.stride),
            batch_size=int(args.batch_size),
            session_topk=int(args.session_topk),
            max_hold_buckets=tuple(int(v) for v in label_cfg.max_hold_buckets),
        )
        parent = train_policy(
            x_cash,
            y_train,
            cfg=label_cfg,
            random_state=int(args.seed) + i * 17,
            feature_cols=feature_cols,
        )
        variant_dir = args.out_dir / preset
        variant_dir.mkdir(parents=True, exist_ok=True)
        parent_path = variant_dir / "fallback_parent.pkl"
        joblib.dump(parent, parent_path)

        base_val_fb = predict_policy_frame(parent, val_df, close=_close(val_df)).reset_index(drop=True)
        base_eval_fb = predict_policy_frame(parent, eval_df, close=_close(eval_df)).reset_index(drop=True)

        rows: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        for rt in alpha2._runtimes():
            val_fb = _scale_decisions(base_val_fb, rt)
            eval_fb = _scale_decisions(base_eval_fb, rt)
            val_final = _combine_primary_fallback(primary_val_dec, val_fb)
            eval_final = _combine_primary_fallback(primary_eval_dec, eval_fb)
            val_metrics = _compact_costs(
                _metrics(val_df, parent_for_features=ref_parent, runner=noop_runner, runner_cfg=noop_cfg, dec=val_final, fee=fee, slip=slip)
            )
            eval_metrics = _compact_costs(
                _metrics(eval_df, parent_for_features=ref_parent, runner=noop_runner, runner_cfg=noop_cfg, dec=eval_final, fee=fee, slip=slip)
            )
            score = _score(val_metrics)
            row = {
                "preset": preset,
                "scale_runtime": rt.__dict__,
                "selection_score": float(score),
                "validation_metrics": val_metrics,
                "oos_metrics": eval_metrics,
                "fallback_used_rows_val": int((~_active(primary_val_dec) & _active(val_fb)).sum()),
                "fallback_used_rows_oos": int((~_active(primary_eval_dec) & _active(eval_fb)).sum()),
            }
            rows.append(
                {
                    "preset": preset,
                    "scale_runtime": rt.name,
                    "confidence": float(rt.confidence),
                    "parent_notional_scale": float(rt.parent_notional_scale),
                    "max_notional": float(rt.max_notional),
                    "selection_score": float(score),
                    "val_cost3_pnl": float(val_metrics["cost3"]["pnl"]),
                    "val_cost3_mdd": float(val_metrics["cost3"]["mdd"]),
                    "val_cost3_trades": int(val_metrics["cost3"]["trades"]),
                    "oos_cost3_pnl": float(eval_metrics["cost3"]["pnl"]),
                    "oos_cost3_mdd": float(eval_metrics["cost3"]["mdd"]),
                    "oos_cost3_trades": int(eval_metrics["cost3"]["trades"]),
                    "fallback_used_rows_oos": int(row["fallback_used_rows_oos"]),
                }
            )
            if best is None or float(score) > float(best["selection_score"]):
                best = row
        assert best is not None
        pd.DataFrame(rows).sort_values("selection_score", ascending=False).to_csv(variant_dir / f"{preset}_grid.csv", index=False)
        report = {
            "model_id": f"alpha7_fallback_cashonly_alpha6_logic_{preset}_20260525",
            "preset": preset,
            "design": "Primary Alpha7 is fixed. Fallback parent is trained only on candidate rows where the fixed primary is CASH. Fallback feature contract remains the 61-feature alpha43_no_legacy contract. Labels come from Alpha6 label-generation logic on those cash-only candidates.",
            "train_csv": str(args.train_csv),
            "eval_csv": str(args.eval_csv),
            "feature_contract": {
                "feature_count": int(len(feature_cols)),
                "feature_cols": feature_cols,
            },
            "train_meta": train_meta,
            "label_meta": label_meta,
            "primary_runtime": primary_scale_rt.__dict__ if primary_scale_rt is not None else None,
            "baseline": {
                "combo_selected_metrics": baseline_combo.get("selected_metrics"),
                "combo_manifest_metrics": baseline_combo_metrics,
            },
            "best_by_selection": best,
            "artifacts": {
                "parent": str(parent_path),
                "grid": str(variant_dir / f"{preset}_grid.csv"),
            },
        }
        report_path = variant_dir / f"{preset}_summary.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest.append({"preset": preset, "report": str(report_path)})
        ranking_rows.append(
            {
                "preset": preset,
                "selection_score": float(best["selection_score"]),
                "cost1_pnl": float(best["oos_metrics"]["cost1"]["pnl"]),
                "cost2_pnl": float(best["oos_metrics"]["cost2"]["pnl"]),
                "cost3_pnl": float(best["oos_metrics"]["cost3"]["pnl"]),
                "cost3_mdd": float(best["oos_metrics"]["cost3"]["mdd"]),
                "cost3_trades": int(best["oos_metrics"]["cost3"]["trades"]),
                "cost3_wr": float(best["oos_metrics"]["cost3"]["wr"]),
                "fallback_used_rows_oos": int(best["fallback_used_rows_oos"]),
                "delta_vs_baseline_cost3_pnl": float(best["oos_metrics"]["cost3"]["pnl"]) - float(baseline_combo_metrics["cost3"]["pnl"]),
                "report": str(report_path),
            }
        )
        print(
            json.dumps(
                {
                    "preset": preset,
                    "cost3_pnl": best["oos_metrics"]["cost3"]["pnl"],
                    "cost3_mdd": best["oos_metrics"]["cost3"]["mdd"],
                    "cost3_trades": best["oos_metrics"]["cost3"]["trades"],
                    "delta_vs_baseline": float(best["oos_metrics"]["cost3"]["pnl"]) - float(baseline_combo_metrics["cost3"]["pnl"]),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    ranking = pd.DataFrame(ranking_rows).sort_values(["cost3_pnl", "selection_score"], ascending=[False, False]).reset_index(drop=True)
    ranking_path = args.out_dir / "ranking.csv"
    ranking.to_csv(ranking_path, index=False)
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(ranking.to_string(index=False))
    print(f"ranking_csv={ranking_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
