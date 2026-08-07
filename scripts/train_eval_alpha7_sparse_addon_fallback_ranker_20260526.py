#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
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
    _score,
)
from scripts.train_eval_alpha7_meta_fallback_cash_router_20260526 import (  # noqa: E402
    COMBO_SUMMARY,
    EVAL_CSV,
    PRIMARY_PARENT,
    PRIMARY_SUMMARY,
    TRAIN_CSV,
    _active,
    _build_meta_features,
    _candidate_specs,
    _combine_primary_fallback,
    _copy_rows,
    _empty_dec_like,
    _json_default,
    _load_best_scale_runtime,
    _predict_scaled,
    _trade_reward,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _read  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import _grid as _runner_grid  # noqa: E402


DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_sparse_addon_fallback_ranker_20260526"


def _addon_candidates():
    specs = _candidate_specs()
    baseline = specs[0]
    alts = specs[1:]
    return baseline, alts


def _build_addon_features(
    frame: pd.DataFrame,
    primary_dec: pd.DataFrame,
    base_dec: pd.DataFrame,
    alt_specs: list[Any],
    alt_decs: list[pd.DataFrame],
) -> pd.DataFrame:
    all_specs = [type("Spec", (), {"name": "base"})()] + alt_specs
    all_decs = [base_dec] + alt_decs
    feat = _build_meta_features(frame, primary_dec, all_specs, all_decs)
    base_q = feat["base_quality"].to_numpy(dtype=np.float64)
    base_c = feat["base_confidence"].to_numpy(dtype=np.float64)
    for spec in alt_specs:
        feat[f"{spec.name}_quality_minus_base"] = feat[f"{spec.name}_quality"] - base_q
        feat[f"{spec.name}_confidence_minus_base"] = feat[f"{spec.name}_confidence"] - base_c
        feat[f"{spec.name}_edge_rank_proxy"] = feat[f"{spec.name}_quality"] * feat[f"{spec.name}_confidence"]
    edge_cols = [f"{spec.name}_edge_rank_proxy" for spec in alt_specs]
    feat["alt_edge_rank_top"] = feat[edge_cols].max(axis=1)
    feat["alt_edge_rank_std"] = feat[edge_cols].std(axis=1)
    return feat


def _label_addon(
    frame: pd.DataFrame,
    primary_dec: pd.DataFrame,
    base_dec: pd.DataFrame,
    alt_specs: list[Any],
    alt_decs: list[pd.DataFrame],
    *,
    min_edge: float,
    gap_min: float,
    min_confidence: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    primary_cash = ~_active(primary_dec)
    base_cash = ~_active(base_dec)
    mask = primary_cash & base_cash
    n = len(frame)
    y_gate = np.zeros(n, dtype=np.int64)
    y_choice = np.zeros(n, dtype=np.int64)
    y_quality = np.zeros(n, dtype=np.float64)
    for i in range(max(0, n - 98)):
        if not mask[i]:
            continue
        rewards: list[float] = []
        for dec in alt_decs:
            row = dec.iloc[i]
            if int(row["action"]) == 0 or int(row["side"]) == 0 or float(row["confidence"]) < float(min_confidence):
                rewards.append(0.0)
                continue
            rewards.append(_trade_reward(frame, dec, i, fee=0.0004, slip=0.00015))
        if not rewards:
            continue
        best_idx = int(np.argmax(rewards))
        best = float(rewards[best_idx])
        second = float(sorted(rewards, reverse=True)[1]) if len(rewards) > 1 else 0.0
        if best > float(min_edge) and (best - second) > float(gap_min):
            y_gate[i] = 1
            y_choice[i] = best_idx + 1
            y_quality[i] = best
    meta = {
        "mask_rows": int(mask.sum()),
        "trade_rows": int((y_gate == 1).sum()),
        "trade_ratio": float(np.mean(y_gate[mask])) if np.any(mask) else 0.0,
        "choice_distribution": pd.Series(y_choice[mask]).value_counts().sort_index().to_dict(),
        "mean_quality": float(np.mean(y_quality[mask])) if np.any(mask) else 0.0,
    }
    return mask, y_gate, y_choice, y_quality, meta


def _fit_gate_model(x: pd.DataFrame, y: np.ndarray, *, seed: int) -> Any:
    model = lgb.LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        n_estimators=240,
        learning_rate=0.025,
        max_depth=3,
        num_leaves=7,
        min_child_samples=120,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_alpha=4.0,
        reg_lambda=16.0,
        random_state=seed,
        verbosity=-1,
    )
    model.fit(x, y)
    return model


def _fit_choice_model(x: pd.DataFrame, y: np.ndarray, *, seed: int) -> Any:
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=int(np.max(y)) + 1,
        class_weight="balanced",
        n_estimators=220,
        learning_rate=0.025,
        max_depth=3,
        num_leaves=7,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_alpha=4.0,
        reg_lambda=16.0,
        random_state=seed,
        verbosity=-1,
    )
    model.fit(x, y)
    return model


def _fit_quality_model(x: pd.DataFrame, y: np.ndarray, *, seed: int) -> Any:
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=180,
        learning_rate=0.025,
        max_depth=2,
        num_leaves=5,
        min_child_samples=100,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_alpha=6.0,
        reg_lambda=20.0,
        random_state=seed,
        verbosity=-1,
    )
    model.fit(x, y)
    return model


def _proba_full(model: Any, x: pd.DataFrame, class_count: int) -> np.ndarray:
    proba = np.asarray(model.predict_proba(x), dtype=np.float64)
    out = np.zeros((len(x), class_count), dtype=np.float64)
    for j, cls in enumerate(np.asarray(model.classes_, dtype=np.int64)):
        if 0 <= int(cls) < class_count:
            out[:, int(cls)] = proba[:, j]
    return out


def _build_addon_decisions(
    template: pd.DataFrame,
    mask: np.ndarray,
    alt_decs: list[pd.DataFrame],
    gate_prob: np.ndarray,
    choice_proba: np.ndarray,
    choice_cls: np.ndarray,
    quality_pred: np.ndarray,
    *,
    gate_min: float,
    choice_min: float,
    quality_min: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    out = _empty_dec_like(template)
    counts = {"cash": 0}
    for i in range(len(out)):
        if not mask[i]:
            counts["cash"] += 1
            continue
        cls = int(choice_cls[i])
        if gate_prob[i] < float(gate_min) or cls <= 0 or cls > len(alt_decs):
            counts["cash"] += 1
            continue
        if float(choice_proba[i, cls]) < float(choice_min) or float(quality_pred[i]) < float(quality_min):
            counts["cash"] += 1
            continue
        chosen = alt_decs[cls - 1]
        if not _active(chosen.iloc[[i]]).item():
            counts["cash"] += 1
            continue
        for col in out.columns:
            out.iat[i, out.columns.get_loc(col)] = chosen.iat[i, chosen.columns.get_loc(col)]
        key = f"candidate_{cls}"
        counts[key] = counts.get(key, 0) + 1
    return out, counts


def _overlay_addon(base_dec: pd.DataFrame, addon_dec: pd.DataFrame) -> pd.DataFrame:
    return _copy_rows(base_dec, addon_dec, _active(addon_dec))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train sparse add-on fallback ranker on Alpha7 current-fallback cash region.")
    ap.add_argument("--train-csv", type=Path, default=TRAIN_CSV)
    ap.add_argument("--eval-csv", type=Path, default=EVAL_CSV)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--seed", type=int, default=52627)
    ap.add_argument("--label-min-edge", type=float, default=0.00035)
    ap.add_argument("--label-gap-min", type=float, default=0.00010)
    ap.add_argument("--label-min-confidence", type=float, default=0.56)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    cutoff = pd.Timestamp("2025-10-01")
    train_df = train_all[train_all["timestamp"] < cutoff].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= cutoff].reset_index(drop=True)

    primary_parent = joblib.load(PRIMARY_PARENT)
    primary_rt = _load_best_scale_runtime(PRIMARY_SUMMARY)
    primary_train = _predict_scaled(primary_parent, train_df, primary_rt)
    primary_val = _predict_scaled(primary_parent, val_df, primary_rt)
    primary_eval = _predict_scaled(primary_parent, eval_df, primary_rt)

    base_spec, alt_specs = _addon_candidates()
    specs = [base_spec] + alt_specs
    train_decs = []
    val_decs = []
    eval_decs = []
    for spec in specs:
        parent = joblib.load(spec.parent)
        rt = _load_best_scale_runtime(spec.summary)
        train_decs.append(_predict_scaled(parent, train_df, rt))
        val_decs.append(_predict_scaled(parent, val_df, rt))
        eval_decs.append(_predict_scaled(parent, eval_df, rt))
    base_train, alt_train = train_decs[0], train_decs[1:]
    base_val, alt_val = val_decs[0], val_decs[1:]
    base_eval, alt_eval = eval_decs[0], eval_decs[1:]

    x_train = _build_addon_features(train_df, primary_train, base_train, alt_specs, alt_train)
    x_val = _build_addon_features(val_df, primary_val, base_val, alt_specs, alt_val)
    x_eval = _build_addon_features(eval_df, primary_eval, base_eval, alt_specs, alt_eval)

    train_mask, y_gate_train, y_choice_train, y_quality_train, label_meta_train = _label_addon(
        train_df,
        primary_train,
        base_train,
        alt_specs,
        alt_train,
        min_edge=float(args.label_min_edge),
        gap_min=float(args.label_gap_min),
        min_confidence=float(args.label_min_confidence),
    )
    val_mask, y_gate_val, y_choice_val, y_quality_val, label_meta_val = _label_addon(
        val_df,
        primary_val,
        base_val,
        alt_specs,
        alt_val,
        min_edge=float(args.label_min_edge),
        gap_min=float(args.label_gap_min),
        min_confidence=float(args.label_min_confidence),
    )
    eval_mask = (~_active(primary_eval)) & (~_active(base_eval))

    train_rows = int(train_mask.sum())
    train_trade_rows = int((y_gate_train == 1).sum())
    if train_trade_rows < 100:
        raise RuntimeError(f"too few add-on labels: {train_trade_rows}")

    gate_model = _fit_gate_model(x_train.loc[train_mask].reset_index(drop=True), y_gate_train[train_mask], seed=int(args.seed))
    choice_train_mask = train_mask & (y_gate_train == 1)
    choice_model = _fit_choice_model(x_train.loc[choice_train_mask].reset_index(drop=True), y_choice_train[choice_train_mask], seed=int(args.seed) + 17)
    quality_model = _fit_quality_model(x_train.loc[choice_train_mask].reset_index(drop=True), y_quality_train[choice_train_mask], seed=int(args.seed) + 29)

    gate_val = np.asarray(gate_model.predict_proba(x_val), dtype=np.float64)[:, 1]
    gate_eval = np.asarray(gate_model.predict_proba(x_eval), dtype=np.float64)[:, 1]
    class_count = len(alt_specs) + 1
    choice_val_proba = _proba_full(choice_model, x_val, class_count)
    choice_eval_proba = _proba_full(choice_model, x_eval, class_count)
    choice_val_cls = np.argmax(choice_val_proba, axis=1).astype(np.int64)
    choice_eval_cls = np.argmax(choice_eval_proba, axis=1).astype(np.int64)
    quality_val_pred = np.asarray(quality_model.predict(x_val), dtype=np.float64)
    quality_eval_pred = np.asarray(quality_model.predict(x_eval), dtype=np.float64)

    ref_parent = _parent_for_features(list(joblib.load(v31.DEFAULT_PARENT)["feature_cols"]))
    fee = float(joblib.load(v31.DEFAULT_PARENT)["config"]["fee"])
    slip = float(joblib.load(v31.DEFAULT_PARENT)["config"]["slip"])
    noop_runner = joblib.load(v31.DEFAULT_JACKPOT)["cost_runner"]
    noop_cfg = next(c for c in _runner_grid() if c.name == "v21_2_parent_noop")
    baseline_combo = json.loads(COMBO_SUMMARY.read_text(encoding="utf-8"))
    baseline_metrics = _compact_costs(
        _metrics(
            eval_df,
            parent_for_features=ref_parent,
            runner=noop_runner,
            runner_cfg=noop_cfg,
            dec=_combine_primary_fallback(primary_eval, base_eval),
            fee=fee,
            slip=slip,
        )
    )

    active_quality = quality_val_pred[val_mask & (choice_val_cls != 0)]
    if len(active_quality) == 0:
        raise RuntimeError("no active validation add-on candidates")
    quality_grid = sorted(set(float(x) for x in np.quantile(active_quality, [0.25, 0.40, 0.55, 0.70, 0.85])))
    gate_grid = [0.45, 0.55, 0.65, 0.75]
    choice_grid = [0.35, 0.45, 0.55, 0.65]

    grid_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for gate_min in gate_grid:
        for choice_min in choice_grid:
            for quality_min in quality_grid:
                addon_val_dec, val_counts = _build_addon_decisions(
                    base_val,
                    val_mask,
                    alt_val,
                    gate_val,
                    choice_val_proba,
                    choice_val_cls,
                    quality_val_pred,
                    gate_min=gate_min,
                    choice_min=choice_min,
                    quality_min=quality_min,
                )
                addon_eval_dec, eval_counts = _build_addon_decisions(
                    base_eval,
                    eval_mask,
                    alt_eval,
                    gate_eval,
                    choice_eval_proba,
                    choice_eval_cls,
                    quality_eval_pred,
                    gate_min=gate_min,
                    choice_min=choice_min,
                    quality_min=quality_min,
                )
                final_val = _overlay_addon(base_val, addon_val_dec)
                final_eval = _overlay_addon(base_eval, addon_eval_dec)
                val_metrics = _compact_costs(
                    _metrics(
                        val_df,
                        parent_for_features=ref_parent,
                        runner=noop_runner,
                        runner_cfg=noop_cfg,
                        dec=_combine_primary_fallback(primary_val, final_val),
                        fee=fee,
                        slip=slip,
                    )
                )
                eval_metrics = _compact_costs(
                    _metrics(
                        eval_df,
                        parent_for_features=ref_parent,
                        runner=noop_runner,
                        runner_cfg=noop_cfg,
                        dec=_combine_primary_fallback(primary_eval, final_eval),
                        fee=fee,
                        slip=slip,
                    )
                )
                row = {
                    "gate_min": float(gate_min),
                    "choice_min": float(choice_min),
                    "quality_min": float(quality_min),
                    "selection_score": float(_score(val_metrics)),
                    "val_cost3_pnl": float(val_metrics["cost3"]["pnl"]),
                    "val_cost3_mdd": float(val_metrics["cost3"]["mdd"]),
                    "val_cost3_trades": int(val_metrics["cost3"]["trades"]),
                    "oos_cost3_pnl": float(eval_metrics["cost3"]["pnl"]),
                    "oos_cost3_mdd": float(eval_metrics["cost3"]["mdd"]),
                    "oos_cost3_trades": int(eval_metrics["cost3"]["trades"]),
                    "oos_cost3_wr": float(eval_metrics["cost3"]["wr"]),
                    "delta_vs_baseline": float(eval_metrics["cost3"]["pnl"]) - float(baseline_metrics["cost3"]["pnl"]),
                    "val_counts": val_counts,
                    "eval_counts": eval_counts,
                }
                grid_rows.append(row)
                if best is None or row["selection_score"] > best["selection_score"]:
                    best = row
    assert best is not None

    addon_val_dec, best_val_counts = _build_addon_decisions(
        base_val,
        val_mask,
        alt_val,
        gate_val,
        choice_val_proba,
        choice_val_cls,
        quality_val_pred,
        gate_min=float(best["gate_min"]),
        choice_min=float(best["choice_min"]),
        quality_min=float(best["quality_min"]),
    )
    addon_eval_dec, best_eval_counts = _build_addon_decisions(
        base_eval,
        eval_mask,
        alt_eval,
        gate_eval,
        choice_eval_proba,
        choice_eval_cls,
        quality_eval_pred,
        gate_min=float(best["gate_min"]),
        choice_min=float(best["choice_min"]),
        quality_min=float(best["quality_min"]),
    )
    final_val = _overlay_addon(base_val, addon_val_dec)
    final_eval = _overlay_addon(base_eval, addon_eval_dec)
    best_val_metrics = _compact_costs(
        _metrics(
            val_df,
            parent_for_features=ref_parent,
            runner=noop_runner,
            runner_cfg=noop_cfg,
            dec=_combine_primary_fallback(primary_val, final_val),
            fee=fee,
            slip=slip,
        )
    )
    best_eval_metrics = _compact_costs(
        _metrics(
            eval_df,
            parent_for_features=ref_parent,
            runner=noop_runner,
            runner_cfg=noop_cfg,
            dec=_combine_primary_fallback(primary_eval, final_eval),
            fee=fee,
            slip=slip,
        )
    )

    grid_path = args.out_dir / "grid.csv"
    pd.DataFrame(grid_rows).sort_values(["selection_score", "oos_cost3_pnl"], ascending=[False, False]).to_csv(grid_path, index=False)
    artifact = {
        "feature_cols": list(x_train.columns),
        "alt_candidate_names": [spec.name for spec in alt_specs],
        "gate_model": gate_model,
        "choice_model": choice_model,
        "quality_model": quality_model,
        "gate_min": float(best["gate_min"]),
        "choice_min": float(best["choice_min"]),
        "quality_min": float(best["quality_min"]),
    }
    artifact_path = args.out_dir / "sparse_addon_ranker.joblib"
    joblib.dump(artifact, artifact_path)

    report = {
        "model_id": "alpha7_sparse_addon_fallback_ranker_20260526",
        "design": "Current fallback alpha43_no_legacy is preserved. A sparse add-on ranker is only allowed on rows where the primary and current fallback are both CASH. It uses a binary gate, then chooses one of three alternate fallback candidates.",
        "train_csv": str(args.train_csv),
        "eval_csv": str(args.eval_csv),
        "baseline": {
            "combo_selected_metrics": baseline_combo.get("selected_metrics"),
            "current_fallback_combo_metrics": baseline_metrics,
        },
        "labeling": {
            "train": label_meta_train,
            "validation": label_meta_val,
            "label_min_edge": float(args.label_min_edge),
            "label_gap_min": float(args.label_gap_min),
            "label_min_confidence": float(args.label_min_confidence),
        },
        "feature_contract": {
            "feature_count": int(len(x_train.columns)),
            "feature_cols": list(x_train.columns),
        },
        "best_by_selection": {
            **best,
            "val_metrics": best_val_metrics,
            "oos_metrics": best_eval_metrics,
            "best_val_counts": best_val_counts,
            "best_eval_counts": best_eval_counts,
        },
        "artifacts": {
            "ranker": str(artifact_path),
            "grid": str(grid_path),
        },
    }
    report_path = args.out_dir / "summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "best_gate_min": best["gate_min"],
                "best_choice_min": best["choice_min"],
                "best_quality_min": best["quality_min"],
                "oos_cost3_pnl": best_eval_metrics["cost3"]["pnl"],
                "oos_cost3_mdd": best_eval_metrics["cost3"]["mdd"],
                "oos_cost3_trades": best_eval_metrics["cost3"]["trades"],
                "delta_vs_baseline": float(best_eval_metrics["cost3"]["pnl"]) - float(baseline_metrics["cost3"]["pnl"]),
            },
            ensure_ascii=False,
            default=_json_default,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
