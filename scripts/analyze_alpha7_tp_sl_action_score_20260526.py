#!/usr/bin/env python3
from __future__ import annotations

import copy
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

from ensemble.fully_learned_governor_policy import ACTION_CASH, FullyLearnedGovernorConfig, build_training_set, predict_policy_frame, train_policy  # noqa: E402
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.ablate_alpha4_3_legacy_regime_block_20260517 import ALPHA4_3_PARENT, DEFAULT_EVAL as FB_EVAL_CSV, DEFAULT_TRAIN as FB_TRAIN_CSV  # noqa: E402
from scripts.alpha7_experiment_config import get_live_baseline  # noqa: E402
from scripts.research_alpha_model_synergy_oos_20260525 import _parent_for_features  # noqa: E402
from scripts.train_eval_alpha5_regime4_tp18_sl10_no_teacher_no_deep_20260517 import BASE_PARENT, _compact_costs, _metrics, _score, _scale_decisions, _select_runner  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _json_default, _read  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import _grid as _runner_grid  # noqa: E402


BASELINE = get_live_baseline()
LIVE_DIR = BASELINE.live_dir
PRIMARY_PARENT = BASELINE.primary_parent
PRIMARY_SUMMARY = BASELINE.primary_summary
FALLBACK_PARENT = BASELINE.fallback_parent
FALLBACK_SUMMARY = BASELINE.fallback_summary
PRIMARY_TRAIN_CSV = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv"
PRIMARY_EVAL_CSV = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2026_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_tp_sl_action_score_audit_20260526"
SPLIT_TS = pd.Timestamp("2025-10-01")
TP_COL = "tp_sl_action_score"


def _active(dec: pd.DataFrame) -> pd.Series:
    action = pd.to_numeric(dec["action"], errors="coerce").fillna(0).astype(int)
    side = pd.to_numeric(dec["side"], errors="coerce").fillna(0).astype(int)
    return (action != ACTION_CASH) & (side != 0)


def _copy_rows(target: pd.DataFrame, source: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    out = target.copy()
    for col in source.columns:
        out.loc[mask, col] = source.loc[mask, col].to_numpy()
    return out


def _combine_primary_fallback(primary: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    primary = primary.reset_index(drop=True)
    fallback = fallback.reset_index(drop=True)
    mask = (~_active(primary)) & _active(fallback)
    return _copy_rows(primary, fallback, mask)


def _load_best_scale_runtime(summary_path: Path) -> alpha2.Alpha2Runtime | None:
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    target = summary.get("best_by_selection")
    for exp in summary.get("experiments", []):
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


def _predict_scaled(parent: dict[str, Any], df: pd.DataFrame, rt: alpha2.Alpha2Runtime | None) -> pd.DataFrame:
    dec = predict_policy_frame(parent, df, close=_close(df)).reset_index(drop=True)
    if rt is not None:
        dec = alpha2._scale_parent_notional(dec, rt).reset_index(drop=True)
    return dec


def _combo_metrics(df: pd.DataFrame, dec: pd.DataFrame, *, fee_mult: float = 1.0) -> dict[str, Any]:
    ref_parent = _parent_for_features(list(joblib.load(v31.DEFAULT_PARENT)["feature_cols"]))
    fee = float(joblib.load(v31.DEFAULT_PARENT)["config"]["fee"]) * float(fee_mult)
    slip = float(joblib.load(v31.DEFAULT_PARENT)["config"]["slip"]) * float(fee_mult)
    noop_runner = joblib.load(v31.DEFAULT_JACKPOT)["cost_runner"]
    noop_cfg = next(c for c in _runner_grid() if c.name == "v21_2_parent_noop")
    return _compact_costs(
        _metrics(
            df.reset_index(drop=True),
            parent_for_features=ref_parent,
            runner=noop_runner,
            runner_cfg=noop_cfg,
            dec=dec.reset_index(drop=True),
            fee=fee,
            slip=slip,
        )
    )


def _decision_delta(base: pd.DataFrame, alt: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in ["action", "side", "notional", "leverage", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars", "quality_score"]:
        if col not in base.columns or col not in alt.columns:
            continue
        a = pd.to_numeric(base[col], errors="coerce").to_numpy(dtype=np.float64)
        b = pd.to_numeric(alt[col], errors="coerce").to_numpy(dtype=np.float64)
        mask = np.isfinite(a) & np.isfinite(b)
        if not np.any(mask):
            continue
        if col in {"quality_score", "notional", "leverage", "take_profit", "stop_loss"}:
            out[col] = {
                "mean_abs_diff": float(np.mean(np.abs(a[mask] - b[mask]))),
                "max_abs_diff": float(np.max(np.abs(a[mask] - b[mask]))),
                "changed_rate": float(np.mean(np.abs(a[mask] - b[mask]) > 1e-12)),
            }
        else:
            out[col] = {
                "changed_rate": float(np.mean(a[mask] != b[mask])),
            }
    return out


def _head_sensitivity(parent: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    cols = list(parent["feature_cols"])
    base = df.reindex(columns=cols).copy()
    alt = base.copy()
    if TP_COL not in alt.columns:
        raise ValueError(f"missing {TP_COL} in frame")
    alt[TP_COL] = 0.0
    report: dict[str, Any] = {}
    for head in [
        "action_model",
        "quality_model",
        "notional_model",
        "leverage_model",
        "take_profit_model",
        "stop_loss_model",
        "max_hold_model",
        "cooldown_model",
    ]:
        model = parent[head]
        if head == "quality_model":
            pred = np.asarray(model.predict(base), dtype=np.float64)
            pred_alt = np.asarray(model.predict(alt), dtype=np.float64)
            report[head] = {
                "mean_abs_diff": float(np.mean(np.abs(pred - pred_alt))),
                "max_abs_diff": float(np.max(np.abs(pred - pred_alt))),
                "sign_flip_rate": float(np.mean(np.sign(pred) != np.sign(pred_alt))),
            }
            continue
        pred = np.asarray(model.predict(base))
        pred_alt = np.asarray(model.predict(alt))
        head_report = {
            "class_changed_rate": float(np.mean(pred != pred_alt)),
        }
        if hasattr(model, "predict_proba"):
            proba = np.asarray(model.predict_proba(base), dtype=np.float64)
            proba_alt = np.asarray(model.predict_proba(alt), dtype=np.float64)
            head_report["mean_abs_proba_diff"] = float(np.mean(np.abs(proba - proba_alt)))
            head_report["max_abs_proba_diff"] = float(np.max(np.abs(proba - proba_alt)))
        report[head] = head_report
    return report


def _train_parent(
    *,
    train_all: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
    out_dir: Path,
) -> tuple[dict[str, Any], alpha2.Alpha2Runtime | None, dict[str, Any]]:
    train_df = train_all[train_all["timestamp"] < SPLIT_TS].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
    parent_ref = joblib.load(v31.DEFAULT_PARENT)
    base_parent_ref = joblib.load(BASE_PARENT)
    label_cfg = FullyLearnedGovernorConfig(**dict(base_parent_ref["config"]))
    fee = float(dict(parent_ref["config"])["fee"])
    slip = float(dict(parent_ref["config"])["slip"])

    x_train, y_train, train_meta = build_training_set(
        train_df,
        cfg=label_cfg,
        stride_bars=6,
        batch_size=512,
        feature_cols=feature_cols,
    )
    parent = train_policy(x_train, y_train, cfg=label_cfg, random_state=int(seed), feature_cols=feature_cols)
    joblib.dump(parent, out_dir / "parent.pkl")

    parent_for_features = copy.deepcopy(parent_ref)
    parent_for_features["feature_cols"] = list(feature_cols)

    base_train_dec = predict_policy_frame(parent, train_df, close=_close(train_df))
    base_val_dec = predict_policy_frame(parent, val_df, close=_close(val_df))
    base_eval_dec = predict_policy_frame(parent, eval_df, close=_close(eval_df))

    experiments: list[dict[str, Any]] = []
    raw_result, _ = _select_runner(
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
        out_dir=out_dir / "runners",
    )
    experiments.append(raw_result)

    best_scale: tuple[alpha2.Alpha2Runtime, dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None
    noop_runner = joblib.load(v31.DEFAULT_JACKPOT)["cost_runner"]
    noop_cfg = next(c for c in _runner_grid() if c.name == "v21_2_parent_noop")
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
        if best_scale is None or score > best_scale[1]["score"]:
            best_scale = (rt, {"score": score, "metrics": val_metrics}, train_dec, val_dec, eval_dec)
    assert best_scale is not None
    scale_rt, scale_selection, scale_train_dec, scale_val_dec, scale_eval_dec = best_scale
    scaled_result, _ = _select_runner(
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
        out_dir=out_dir / "runners",
    )
    scaled_result["selected_parent_scale_runtime"] = asdict(scale_rt)
    scaled_result["scale_selection_metrics"] = scale_selection["metrics"]
    experiments.append(scaled_result)

    best = max(experiments, key=lambda e: float(e["selection_score"]))
    selected_rt = None
    if best["name"] == "parent_direct_scaled_no_teacher":
        selected_parent_rt = best.get("selected_parent_scale_runtime")
        if selected_parent_rt:
            selected_rt = alpha2.Alpha2Runtime(
                name=str(selected_parent_rt["name"]),
                confidence=float(selected_parent_rt["confidence"]),
                parent_notional_scale=float(selected_parent_rt["parent_notional_scale"]),
                max_notional=float(selected_parent_rt["max_notional"]),
            )
    summary = {
        "feature_count": len(feature_cols),
        "contains_tp_sl_action_score": TP_COL in feature_cols,
        "best_by_selection": best["name"],
        "selection_score": float(best["selection_score"]),
        "selected_validation_metrics": _compact_costs(best["validation_metrics"]),
        "selected_metrics": _compact_costs(best["metrics"]),
        "train_meta": train_meta,
        "experiments": experiments,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return parent, selected_rt, summary


def parse_args() -> Any:
    import argparse

    ap = argparse.ArgumentParser(description="Audit Alpha7 tp_sl_action_score dependence and no-tp retrain impact.")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    primary_parent = joblib.load(PRIMARY_PARENT)
    fallback_parent = joblib.load(FALLBACK_PARENT)
    primary_rt = _load_best_scale_runtime(PRIMARY_SUMMARY)
    fallback_rt = _load_best_scale_runtime(FALLBACK_SUMMARY)
    primary_summary = json.loads(PRIMARY_SUMMARY.read_text(encoding="utf-8"))

    primary_train = _read(PRIMARY_TRAIN_CSV)
    primary_eval = _read(PRIMARY_EVAL_CSV)
    fallback_train = _read(FB_TRAIN_CSV)
    fallback_eval = _read(FB_EVAL_CSV)

    live_primary_eval_dec = _predict_scaled(primary_parent, primary_eval, primary_rt)
    live_fallback_eval_dec = _predict_scaled(fallback_parent, primary_eval, fallback_rt)
    live_combo_eval_dec = _combine_primary_fallback(live_primary_eval_dec, live_fallback_eval_dec)
    live_combo_metrics = _combo_metrics(primary_eval, live_combo_eval_dec)

    primary_eval_zero = primary_eval.copy()
    primary_eval_zero[TP_COL] = 0.0
    live_primary_eval_zero_dec = _predict_scaled(primary_parent, primary_eval_zero, primary_rt)
    live_fallback_eval_zero_dec = _predict_scaled(fallback_parent, primary_eval_zero, fallback_rt)
    live_combo_eval_zero_dec = _combine_primary_fallback(live_primary_eval_zero_dec, live_fallback_eval_zero_dec)
    live_combo_zero_metrics = _combo_metrics(primary_eval_zero, live_combo_eval_zero_dec)

    primary_head_sensitivity = _head_sensitivity(primary_parent, primary_eval)
    fallback_head_sensitivity = _head_sensitivity(fallback_parent, primary_eval)

    primary_no_tp_cols = [c for c in list(primary_summary["feature_contract"]["feature_cols"]) if c != TP_COL]
    fallback_no_tp_cols = [c for c in list(fallback_parent["feature_cols"]) if c != TP_COL]

    primary_no_tp_dir = args.out_dir / "primary_no_tp"
    fallback_no_tp_dir = args.out_dir / "fallback_no_tp"
    primary_no_tp_dir.mkdir(parents=True, exist_ok=True)
    fallback_no_tp_dir.mkdir(parents=True, exist_ok=True)

    retrained_primary_parent, retrained_primary_rt, retrained_primary_summary = _train_parent(
        train_all=primary_train,
        eval_df=primary_eval,
        feature_cols=primary_no_tp_cols,
        seed=5517,
        out_dir=primary_no_tp_dir,
    )
    retrained_fallback_parent, retrained_fallback_rt, retrained_fallback_summary = _train_parent(
        train_all=fallback_train,
        eval_df=fallback_eval,
        feature_cols=fallback_no_tp_cols,
        seed=4317,
        out_dir=fallback_no_tp_dir,
    )

    retrained_primary_eval_dec = _predict_scaled(retrained_primary_parent, primary_eval_zero, retrained_primary_rt)
    retrained_fallback_eval_dec = _predict_scaled(retrained_fallback_parent, primary_eval_zero, retrained_fallback_rt)
    retrained_combo_eval_dec = _combine_primary_fallback(retrained_primary_eval_dec, retrained_fallback_eval_dec)
    retrained_combo_metrics = _combo_metrics(primary_eval_zero, retrained_combo_eval_dec)

    report = {
        "model_id": "alpha7_tp_sl_action_score_audit_20260526",
        "scope": "Alpha7 live primary+fallback tp_sl_action_score dependence, zero-ablation, and no-tp retrain backtest.",
        "lineage": {
            "primary_parent": str(PRIMARY_PARENT),
            "fallback_parent": str(FALLBACK_PARENT),
            "primary_train_csv": str(PRIMARY_TRAIN_CSV),
            "primary_eval_csv": str(PRIMARY_EVAL_CSV),
            "fallback_train_csv": str(FB_TRAIN_CSV),
            "fallback_eval_csv": str(FB_EVAL_CSV),
        },
        "feature_contract": {
            "primary_contains_tp_sl_action_score": TP_COL in primary_parent["feature_cols"],
            "fallback_contains_tp_sl_action_score": TP_COL in fallback_parent["feature_cols"],
            "primary_feature_count": len(primary_parent["feature_cols"]),
            "fallback_feature_count": len(fallback_parent["feature_cols"]),
        },
        "current_live_combo": {
            "oos_costs": live_combo_metrics,
            "zeroed_tp_sl_action_score_oos_costs": live_combo_zero_metrics,
            "delta_cost3_pnl": float(live_combo_zero_metrics["cost3"]["pnl"]) - float(live_combo_metrics["cost3"]["pnl"]),
            "decision_delta_primary": _decision_delta(live_primary_eval_dec, live_primary_eval_zero_dec),
            "decision_delta_fallback": _decision_delta(live_fallback_eval_dec, live_fallback_eval_zero_dec),
            "decision_delta_final": _decision_delta(live_combo_eval_dec, live_combo_eval_zero_dec),
        },
        "head_sensitivity": {
            "primary": primary_head_sensitivity,
            "fallback": fallback_head_sensitivity,
        },
        "no_tp_retrain": {
            "primary": {
                "summary": retrained_primary_summary,
                "selected_scale_runtime": asdict(retrained_primary_rt) if retrained_primary_rt is not None else None,
            },
            "fallback": {
                "summary": retrained_fallback_summary,
                "selected_scale_runtime": asdict(retrained_fallback_rt) if retrained_fallback_rt is not None else None,
            },
            "alpha7_combo_oos_costs": retrained_combo_metrics,
            "delta_vs_current_live_combo_cost3": float(retrained_combo_metrics["cost3"]["pnl"]) - float(live_combo_metrics["cost3"]["pnl"]),
        },
        "rename_proposal": {
            "current_name": TP_COL,
            "proposed_name": "legacy_signed_path_edge_score",
            "reason": "The feature is not the live TP/SL executor. It is a legacy signed path-edge input where positive favors long, negative favors short, and zero can be a valid no-edge state.",
        },
    }

    report_path = args.out_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "current_cost3": live_combo_metrics["cost3"],
                "zeroed_cost3": live_combo_zero_metrics["cost3"],
                "retrained_no_tp_cost3": retrained_combo_metrics["cost3"],
            },
            ensure_ascii=False,
            default=_json_default,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
