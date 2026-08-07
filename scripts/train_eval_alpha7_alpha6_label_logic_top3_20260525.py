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
    FullyLearnedGovernorConfig,
    build_training_set,
    predict_policy_frame,
    train_policy,
)
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts.alpha6_catboost_entry_quality_exit_policy_20260522 import (  # noqa: E402
    EQEConfig,
    _apply_label_preset,
    _build_entry_labels,
)
from scripts.eval_alpha3_ft_transformer_mtl_parent_v2_20260515 import ft_v1  # noqa: E402
from scripts.train_eval_alpha5_regime4_tp18_sl10_no_teacher_no_deep_20260517 import (  # noqa: E402
    BASE_PARENT,
    DEFAULT_EVAL,
    DEFAULT_TRAIN,
    _close,
    _compact_costs,
    _feature_cols,
    _metrics,
    _read,
    _scale_decisions,
    _score,
    _select_runner,
)
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import _grid as _runner_grid  # noqa: E402


MODEL_ID = "alpha7_alpha6_label_logic_top3_20260525"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_alpha6_label_logic_top3_20260525"
DEFAULT_PRESETS = ("current_quality", "high_precision_robust", "short_horizon_robust")


def _alpha7_valid_idx(frame: pd.DataFrame, *, stride: int, max_horizon: int) -> np.ndarray:
    return np.arange(0, max(0, len(frame) - int(max_horizon) - 1), max(1, int(stride)), dtype=np.int64)


def _max_hold_bucket_index(horizon: int, buckets: tuple[int, ...]) -> int:
    for idx, bucket in enumerate(buckets):
        if int(horizon) <= int(bucket):
            return idx
    return len(buckets) - 1


def _override_entry_quality_from_alpha6_logic(
    *,
    train_df: pd.DataFrame,
    native_y: dict[str, np.ndarray],
    alpha7_valid: np.ndarray,
    preset: str,
    stride: int,
    batch_size: int,
    session_topk: int,
    max_hold_buckets: tuple[int, ...],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
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
    lookup = alpha6_pos.reindex(alpha7_valid)
    if lookup.isna().any():
        missing_idx = int(alpha7_valid[np.flatnonzero(lookup.isna().to_numpy())[0]])
        raise ValueError(f"alpha6 label logic missing alpha7 candidate index {missing_idx} for preset={preset}")
    take = lookup.to_numpy(dtype=np.int64)
    y = {k: np.asarray(v).copy() for k, v in native_y.items()}
    action = alpha6_y["action"][take].astype(np.int64)
    quality = alpha6_y["quality"][take].astype(np.float64)
    target_horizon = alpha6_y["target_horizon"][take].astype(np.int64)
    hold_idx = np.asarray(
        [_max_hold_bucket_index(int(v), max_hold_buckets) if int(v) > 0 else 0 for v in target_horizon],
        dtype=np.int64,
    )
    y["action"] = action
    y["quality"] = quality
    trade = action != 0
    y["max_hold"][trade] = hold_idx[trade]
    return y, {
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
        "alpha7_candidate_rows": int(len(alpha7_valid)),
        "alpha6_candidate_rows": int(len(alpha6_valid)),
        "match_coverage": float(len(alpha7_valid) / max(len(alpha7_valid), 1)),
        "action_distribution": pd.Series(action).value_counts().sort_index().to_dict(),
        "target_horizon_distribution": pd.Series(target_horizon[trade]).value_counts().sort_index().to_dict(),
        "quality_mean": float(np.mean(quality)),
        "quality_p95": float(np.quantile(quality, 0.95)),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train/evaluate Alpha7 using Alpha6 top-3 label-generation logic on Alpha7 frames.")
    ap.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    ap.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--presets", default=",".join(DEFAULT_PRESETS))
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--seed", type=int, default=5517)
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

    base_parent_ref = joblib.load(BASE_PARENT)
    label_cfg = FullyLearnedGovernorConfig(**dict(base_parent_ref["config"]))
    feature_cols = _feature_cols(train_all, eval_df)
    x_train, y_train_native, train_meta = build_training_set(
        train_df,
        cfg=label_cfg,
        stride_bars=int(args.stride),
        batch_size=int(args.batch_size),
        feature_cols=feature_cols,
    )
    alpha7_valid = _alpha7_valid_idx(train_df, stride=int(args.stride), max_horizon=int(label_cfg.max_train_horizon_bars))
    if len(alpha7_valid) != len(x_train):
        raise ValueError("alpha7 candidate contract mismatch")

    parent_ref = joblib.load(ft_v1.v31.DEFAULT_PARENT)
    fee = float(dict(parent_ref["config"])["fee"])
    slip = float(dict(parent_ref["config"])["slip"])
    parent_for_features = copy.deepcopy(parent_ref)
    parent_for_features["feature_cols"] = list(feature_cols)
    noop_runner = joblib.load(ft_v1.v31.DEFAULT_JACKPOT)["cost_runner"]
    noop_cfg = next(c for c in _runner_grid() if c.name == "v21_2_parent_noop")

    ranking_rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for i, preset in enumerate(presets):
        y_train, label_meta = _override_entry_quality_from_alpha6_logic(
            train_df=train_df,
            native_y=y_train_native,
            alpha7_valid=alpha7_valid,
            preset=preset,
            stride=int(args.stride),
            batch_size=int(args.batch_size),
            session_topk=int(args.session_topk),
            max_hold_buckets=tuple(int(v) for v in label_cfg.max_hold_buckets),
        )
        parent = train_policy(
            x_train,
            y_train,
            cfg=label_cfg,
            random_state=int(args.seed) + i * 17,
            feature_cols=feature_cols,
        )
        variant_dir = args.out_dir / preset
        variant_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(parent, variant_dir / "parent.pkl")

        base_train_dec = predict_policy_frame(parent, train_df, close=_close(train_df))
        base_val_dec = predict_policy_frame(parent, val_df, close=_close(val_df))
        base_eval_dec = predict_policy_frame(parent, eval_df, close=_close(eval_df))

        experiments: list[dict[str, Any]] = []
        grid_rows: list[dict[str, Any]] = []
        raw_result, raw_rows = _select_runner(
            name="parent_direct_raw_no_teacher",
            train_df=train_df,
            val_df=val_df,
            eval_df=eval_df,
            parent_for_features=parent_for_features,
            train_dec=base_train_dec,
            val_dec=base_val_dec,
            eval_dec=base_eval_dec,
            fee=fee,
            slip=slip,
            out_dir=variant_dir / "runners",
        )
        experiments.append(raw_result)
        grid_rows.extend(raw_rows)

        best_scale: tuple[alpha2.Alpha2Runtime, dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None
        for rt in alpha2._runtimes():
            train_dec = _scale_decisions(base_train_dec, rt)
            val_dec = _scale_decisions(base_val_dec, rt)
            eval_dec = _scale_decisions(base_eval_dec, rt)
            val_metrics = _metrics(
                val_df,
                parent_for_features=parent_for_features,
                runner=noop_runner,
                runner_cfg=noop_cfg,
                dec=val_dec,
                fee=fee,
                slip=slip,
            )
            score = _score(val_metrics)
            grid_rows.append(
                {
                    "candidate": "parent_direct_scaled_no_teacher",
                    "stage": "scale_runtime",
                    "preset": preset,
                    **rt.__dict__,
                    "score": score,
                    "val_cost1_pnl": val_metrics["cost1"]["pnl"],
                    "val_cost1_mdd": val_metrics["cost1"]["mdd"],
                    "val_cost2_pnl": val_metrics["cost2"]["pnl"],
                    "val_cost3_pnl": val_metrics["cost3"]["pnl"],
                    "val_trades": val_metrics["cost1"]["trades"],
                }
            )
            if best_scale is None or score > best_scale[1]["score"]:
                best_scale = (rt, {"score": score, "metrics": val_metrics}, train_dec, val_dec, eval_dec)
        assert best_scale is not None
        scale_rt, scale_selection, scale_train_dec, scale_val_dec, scale_eval_dec = best_scale
        scaled_result, scaled_rows = _select_runner(
            name="parent_direct_scaled_no_teacher",
            train_df=train_df,
            val_df=val_df,
            eval_df=eval_df,
            parent_for_features=parent_for_features,
            train_dec=scale_train_dec,
            val_dec=scale_val_dec,
            eval_dec=scale_eval_dec,
            fee=fee,
            slip=slip,
            out_dir=variant_dir / "runners",
        )
        scaled_result["selected_parent_scale_runtime"] = scale_rt.__dict__
        scaled_result["scale_selection_metrics"] = scale_selection["metrics"]
        experiments.append(scaled_result)
        grid_rows.extend(scaled_rows)

        best = max(experiments, key=lambda e: float(e["selection_score"]))
        report = {
            "model_id": f"{MODEL_ID}_{preset}",
            "preset": preset,
            "design": "Alpha7 architecture retrained on Alpha7 fixed sticky trade-candidate frames. Only action/quality labels are replaced with Alpha6 label-generation logic; Alpha7 native risk heads (notional/leverage/tp/sl/max_hold/cooldown) remain, with max_hold optionally nudged from Alpha6 target_horizon buckets.",
            "train_csv": str(args.train_csv),
            "eval_csv": str(args.eval_csv),
            "feature_contract": {
                "feature_count": int(len(feature_cols)),
                "feature_cols": feature_cols,
            },
            "train_meta": train_meta,
            "label_meta": label_meta,
            "validation_split": {
                "train": [str(train_df["timestamp"].iloc[0]), str(train_df["timestamp"].iloc[-1])],
                "validation": [str(val_df["timestamp"].iloc[0]), str(val_df["timestamp"].iloc[-1])],
                "oos_eval": [str(eval_df["timestamp"].iloc[0]), str(eval_df["timestamp"].iloc[-1])],
            },
            "experiments": [
                {
                    "name": e["name"],
                    "selection_score": float(e["selection_score"]),
                    "selected_runner_config": e["selected_runner_config"],
                    "selected_parent_scale_runtime": e.get("selected_parent_scale_runtime"),
                    "validation_metrics": _compact_costs(e["validation_metrics"]),
                    "oos_metrics": _compact_costs(e["metrics"]),
                }
                for e in experiments
            ],
            "best_by_selection": {
                "name": best["name"],
                "selection_score": float(best["selection_score"]),
                "selected_runner_config": best["selected_runner_config"],
                "selected_parent_scale_runtime": best.get("selected_parent_scale_runtime"),
                "validation_metrics": _compact_costs(best["validation_metrics"]),
                "oos_metrics": _compact_costs(best["metrics"]),
            },
        }
        report_path = variant_dir / f"{preset}_summary.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        pd.DataFrame(grid_rows).to_csv(variant_dir / f"{preset}_grid.csv", index=False)
        manifest.append({"preset": preset, "report": str(report_path)})
        ranking_rows.append(
            {
                "preset": preset,
                "best_name": best["name"],
                "selection_score": float(best["selection_score"]),
                "cost1_pnl": float(best["metrics"]["cost1"]["pnl"]),
                "cost2_pnl": float(best["metrics"]["cost2"]["pnl"]),
                "cost3_pnl": float(best["metrics"]["cost3"]["pnl"]),
                "cost3_mdd": float(best["metrics"]["cost3"]["mdd"]),
                "cost3_trades": int(best["metrics"]["cost3"]["trades"]),
                "cost3_wr": float(best["metrics"]["cost3"]["wr"]),
                "quality_mean": float(label_meta["quality_mean"]),
                "quality_p95": float(label_meta["quality_p95"]),
                "report": str(report_path),
            }
        )
        print(
            json.dumps(
                {
                    "preset": preset,
                    "best_name": best["name"],
                    "cost3_pnl": best["metrics"]["cost3"]["pnl"],
                    "cost3_mdd": best["metrics"]["cost3"]["mdd"],
                    "cost3_trades": best["metrics"]["cost3"]["trades"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    ranking = pd.DataFrame(ranking_rows).sort_values(["cost3_pnl", "selection_score"], ascending=[False, False]).reset_index(drop=True)
    ranking_path = args.out_dir / "ranking.csv"
    ranking.to_csv(ranking_path, index=False)
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(ranking.to_string(index=False))
    print(f"ranking_csv={ranking_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
