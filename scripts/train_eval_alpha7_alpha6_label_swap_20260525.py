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
)
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts.eval_alpha3_ft_transformer_mtl_parent_v2_20260515 import ft_v1  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import _grid as _runner_grid  # noqa: E402


DEFAULT_LABEL_ROOT = ROOT / "tmp/causal_regen_20260516/alpha6_target_mode_abc_gpu_rapid_20260523"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_alpha6_label_swap_20260525"
DEFAULT_VARIANTS = (
    "current_quality",
    "density_balanced",
    "regime_conditional",
    "perturbation_robust",
    "adverse_conformal",
    "sam_conformal",
    "high_precision_robust",
    "turnover_balanced_robust",
    "scalp_short_horizon_hreg",
    "short_horizon_robust_hreg",
)


def _quantile_map(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(target, dtype=np.float64)
    if src.size == 0 or dst.size == 0:
        raise ValueError("empty quality mapping arrays")
    if np.nanmax(src) - np.nanmin(src) < 1e-12:
        return np.full(src.shape, float(np.nanmedian(dst)), dtype=np.float64)
    src_sorted = np.sort(src)
    dst_sorted = np.sort(dst)
    return np.interp(src, src_sorted, dst_sorted).astype(np.float64)


def _max_hold_bucket_index(horizon: int, buckets: tuple[int, ...]) -> int:
    for idx, bucket in enumerate(buckets):
        if int(horizon) <= int(bucket):
            return idx
    return len(buckets) - 1


def _override_train_labels(
    *,
    candidate_ts: pd.Series,
    native_y: dict[str, np.ndarray],
    label_csv: Path,
    cutoff: pd.Timestamp,
    max_hold_buckets: tuple[int, ...],
    min_match_coverage: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    alpha6 = pd.read_csv(
        label_csv,
        usecols=["timestamp", "action", "quality", "target_bucket", "target_horizon"],
        parse_dates=["timestamp"],
    )
    alpha6 = alpha6[alpha6["timestamp"] < cutoff].copy()
    if alpha6["timestamp"].duplicated().any():
        dup = alpha6.loc[alpha6["timestamp"].duplicated(), "timestamp"].iloc[0]
        raise ValueError(f"duplicate alpha6 timestamps in {label_csv}: {dup}")
    candidates = pd.DataFrame({"timestamp": candidate_ts.to_numpy(), "native_idx": np.arange(len(candidate_ts), dtype=np.int64)})
    merged = candidates.merge(alpha6, on="timestamp", how="inner")
    coverage = float(len(merged)) / float(len(candidates))
    if coverage < float(min_match_coverage):
        raise ValueError(f"alpha6 label coverage too low for {label_csv}: {coverage:.4f}")
    sel = merged["native_idx"].to_numpy(dtype=np.int64)
    out = {k: np.asarray(v)[sel].copy() for k, v in native_y.items()}
    action = merged["action"].to_numpy(dtype=np.int64)
    if not np.isin(action, [0, 1, 2]).all():
        raise ValueError(f"invalid action values in {label_csv}")
    out["action"] = action
    mapped_quality = _quantile_map(
        merged["quality"].to_numpy(dtype=np.float64),
        np.asarray(native_y["quality"], dtype=np.float64)[sel],
    )
    out["quality"] = mapped_quality
    hold_idx = np.asarray(
        [_max_hold_bucket_index(int(v), max_hold_buckets) for v in merged["target_horizon"].to_numpy(dtype=np.int64)],
        dtype=np.int64,
    )
    trade_mask = action != 0
    out["max_hold"][trade_mask] = hold_idx[trade_mask]
    return sel, out, {
        "label_csv": str(label_csv),
        "matched_rows": int(len(merged)),
        "candidate_rows": int(len(candidates)),
        "match_coverage": coverage,
        "range": [str(merged["timestamp"].iloc[0]), str(merged["timestamp"].iloc[-1])],
        "action_distribution": pd.Series(action).value_counts().sort_index().to_dict(),
        "target_bucket_distribution": pd.Series(merged["target_bucket"]).value_counts().sort_index().to_dict(),
        "target_horizon_distribution": pd.Series(merged["target_horizon"]).value_counts().sort_index().to_dict(),
        "quality_raw_mean": float(merged["quality"].mean()),
        "quality_raw_p95": float(merged["quality"].quantile(0.95)),
        "quality_mapped_mean": float(np.mean(mapped_quality)),
        "quality_mapped_p95": float(np.quantile(mapped_quality, 0.95)),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Retrain Alpha7 parent with Alpha6 label presets on matched candidate timestamps.")
    ap.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    ap.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    ap.add_argument("--label-root", type=Path, default=DEFAULT_LABEL_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--seed", type=int, default=5517)
    ap.add_argument("--min-match-coverage", type=float, default=0.99)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    variants = [v.strip() for v in str(args.variants).split(",") if v.strip()]
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    cutoff = pd.Timestamp("2025-10-01")
    train_df = train_all[train_all["timestamp"] < cutoff].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= cutoff].reset_index(drop=True)
    base_parent_ref = joblib.load(BASE_PARENT)
    label_cfg = FullyLearnedGovernorConfig(**dict(base_parent_ref["config"]))
    feature_cols = _feature_cols(train_all, eval_df)
    x_train_native, y_train_native, train_meta = build_training_set(
        train_df,
        cfg=label_cfg,
        stride_bars=int(args.stride),
        batch_size=512,
        feature_cols=feature_cols,
    )
    valid_idx = np.arange(0, max(0, len(train_df) - int(label_cfg.max_train_horizon_bars) - 1), max(1, int(args.stride)), dtype=np.int64)
    if len(valid_idx) != len(x_train_native):
        raise ValueError("candidate timestamp contract mismatch")
    candidate_ts = train_df.iloc[valid_idx]["timestamp"].reset_index(drop=True)
    parent_ref = joblib.load(ft_v1.v31.DEFAULT_PARENT)
    fee = float(dict(parent_ref["config"])["fee"])
    slip = float(dict(parent_ref["config"])["slip"])
    parent_for_features = copy.deepcopy(parent_ref)
    parent_for_features["feature_cols"] = list(feature_cols)
    noop_runner = joblib.load(ft_v1.v31.DEFAULT_JACKPOT)["cost_runner"]
    noop_cfg = next(c for c in _runner_grid() if c.name == "v21_2_parent_noop")
    scale_rt = next(rt for rt in alpha2._runtimes() if rt.name == "noflip_c0.56_parent_scale1.00")

    ranking_rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for i, variant in enumerate(variants):
        label_csv = args.label_root / variant / "current_tail111_train_labels.csv"
        if not label_csv.exists():
            raise FileNotFoundError(label_csv)
        sel, y_train, label_meta = _override_train_labels(
            candidate_ts=candidate_ts,
            native_y=y_train_native,
            label_csv=label_csv,
            cutoff=cutoff,
            max_hold_buckets=tuple(int(v) for v in label_cfg.max_hold_buckets),
            min_match_coverage=float(args.min_match_coverage),
        )
        x_train = x_train_native.iloc[sel].reset_index(drop=True)
        parent = train_policy(
            x_train,
            y_train,
            cfg=label_cfg,
            random_state=int(args.seed) + i * 17,
            feature_cols=feature_cols,
        )
        variant_dir = args.out_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(parent, variant_dir / "parent.pkl")

        base_val_dec = predict_policy_frame(parent, val_df, close=_close(val_df))
        base_eval_dec = predict_policy_frame(parent, eval_df, close=_close(eval_df))
        val_dec = _scale_decisions(base_val_dec, scale_rt)
        eval_dec = _scale_decisions(base_eval_dec, scale_rt)
        val_metrics = _metrics(val_df, parent_for_features=parent_for_features, runner=noop_runner, runner_cfg=noop_cfg, dec=val_dec, fee=fee, slip=slip)
        eval_metrics = _metrics(eval_df, parent_for_features=parent_for_features, runner=noop_runner, runner_cfg=noop_cfg, dec=eval_dec, fee=fee, slip=slip)
        selection_score = _score(val_metrics)
        report = {
            "model_id": f"alpha7_alpha6_label_swap_{variant}_20260525",
            "variant": variant,
            "design": "Alpha7 architecture with Alpha6 entry/quality labels mapped onto matched Alpha7 candidate timestamps; Alpha7 native risk heads retained; max_hold bucket overridden from Alpha6 target_horizon. Runtime wrapper is fixed to current Alpha7 primary: v21_2_parent_noop + noflip_c0.56_parent_scale1.00.",
            "train_csv": str(args.train_csv),
            "eval_csv": str(args.eval_csv),
            "alpha6_label_meta": label_meta,
            "feature_contract": {
                "feature_count": int(len(feature_cols)),
                "feature_cols": feature_cols,
            },
            "train_meta": train_meta,
            "runtime_wrapper": {
                "runner_config": noop_cfg.name,
                "scale_runtime": scale_rt.__dict__,
            },
            "validation_metrics": _compact_costs(val_metrics),
            "oos_metrics": _compact_costs(eval_metrics),
            "selection_score": float(selection_score),
        }
        report_path = variant_dir / f"{variant}_summary.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest.append({"variant": variant, "report": str(report_path)})
        ranking_rows.append(
            {
                "variant": variant,
                "selection_score": float(selection_score),
                "cost1_pnl": float(eval_metrics["cost1"]["pnl"]),
                "cost2_pnl": float(eval_metrics["cost2"]["pnl"]),
                "cost3_pnl": float(eval_metrics["cost3"]["pnl"]),
                "cost3_mdd": float(eval_metrics["cost3"]["mdd"]),
                "cost3_trades": int(eval_metrics["cost3"]["trades"]),
                "cost3_wr": float(eval_metrics["cost3"]["wr"]),
                "match_coverage": float(label_meta["match_coverage"]),
                "quality_mapped_mean": float(label_meta["quality_mapped_mean"]),
                "report": str(report_path),
            }
        )
        print(
            json.dumps(
                {
                    "variant": variant,
                    "cost3_pnl": eval_metrics["cost3"]["pnl"],
                    "cost3_mdd": eval_metrics["cost3"]["mdd"],
                    "cost3_trades": eval_metrics["cost3"]["trades"],
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
    print(json.dumps({"ranking": str(ranking_path), "manifest": str(manifest_path)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
