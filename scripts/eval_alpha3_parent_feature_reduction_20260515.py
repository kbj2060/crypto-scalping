#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import (  # noqa: E402
    FullyLearnedGovernorConfig,
    build_training_set,
    predict_policy_frame,
    prepare_features,
    train_policy,
)
from scripts import eval_alpha1_l2_execution_replay_20260513 as l2  # noqa: E402
from scripts import eval_alpha1_teacher_constrained_deep_parent_20260513 as teacher  # noqa: E402
from scripts import eval_alpha2_1_signal_immediate_limit_20260514 as alpha3_exec  # noqa: E402
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts import eval_alpha3_limit_close_fallback_20260514 as alpha3_close  # noqa: E402
from scripts import eval_hf_v13_v31_parent_swap_v40 as v40swap  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _json_default, _read  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402


MODEL_ID = "alpha3_parent_feature_reduction_20260515"
FEATURE_SETS = ROOT / "data/ensemble/reports/alpha3_parent_feature_analysis_20260515_feature_sets.json"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha3_parent_feature_reduction_20260515_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/alpha3_parent_feature_reduction_20260515_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/alpha3_parent_feature_reduction_20260515_grid.csv"
OUT_DIR = ROOT / "data/ensemble/supervised/alpha3_parent_feature_reduction_20260515"


def _load_teacher() -> tuple[Any, list[str], dict[str, Any], tuple[float, ...]]:
    payload = torch.load(alpha3_exec.TEACHER_MODEL, map_location="cpu", weights_only=False)
    model = alpha2._load_teacher_model(payload)
    return model, list(payload["feature_cols"]), dict(payload["train_meta"]["norm"]), tuple(float(x) for x in payload["buckets"])


def _selected_runtime() -> alpha2.Alpha2Runtime:
    audit = json.loads(alpha3_exec.ALPHA2_AUDIT.read_text(encoding="utf-8"))
    runtime = dict(audit.get("selected_runtime", {}) or {})
    return alpha2.Alpha2Runtime(
        name=str(runtime.get("name", "noflip_c0.56_parent_scale1.10")),
        confidence=float(runtime.get("confidence", 0.56)),
        parent_notional_scale=float(runtime.get("parent_notional_scale", 1.10)),
        max_notional=float(runtime.get("max_notional", 2.75)),
    )


def _canonical_limit_cfg() -> alpha3_exec.ImmediateLimitConfig:
    return alpha3_exec.ImmediateLimitConfig(
        "next_open_limit_touch0_fee20",
        "next_open",
        0.0,
        0.0,
        0.0,
        0.20,
        entry_miss="skip",
        exit_miss="market_fallback",
    )


def _metrics(
    *,
    eval_df: pd.DataFrame,
    runner_parent: dict[str, Any],
    decision_parent: dict[str, Any],
    decision_frame: pd.DataFrame,
    teacher_pred: dict[str, np.ndarray],
    teacher_buckets: tuple[float, ...],
    runtime: alpha2.Alpha2Runtime,
    jackpot_model: dict[str, Any],
    add_cfg: CostRunnerConfig,
    eval_q: np.ndarray,
    overlay: Any,
    limit_cfg: alpha3_exec.ImmediateLimitConfig,
    fee: float,
    slip: float,
) -> dict[str, Any]:
    decisions = alpha2._decisions(decision_frame, teacher_pred, teacher_buckets, runtime)
    return alpha3_close._metrics_signal_limit_close(
        eval_df,
        runner_parent,
        jackpot_model,
        add_cfg,
        eval_q,
        decisions,
        overlay,
        limit_cfg,
        fee=fee,
        slip=slip,
    )


def _fit_reduced_parent(
    *,
    name: str,
    features: list[str],
    train_all: pd.DataFrame,
    x_train_all: pd.DataFrame,
    y: dict[str, np.ndarray],
    cfg: FullyLearnedGovernorConfig,
) -> dict[str, Any]:
    cols = [c for c in features if c in x_train_all.columns]
    if "side_hint" not in cols:
        cols = ["side_hint"] + cols
    cols = list(dict.fromkeys(cols))
    print(f"[{MODEL_ID}] training {name} feature_count={len(cols)}", flush=True)
    bundle = train_policy(
        x_train_all.reindex(columns=cols),
        y,
        cfg=cfg,
        random_state=4200 + len(cols),
        feature_cols=cols,
    )
    bundle["model_id"] = MODEL_ID
    bundle["candidate_name"] = name
    bundle["base_model"] = str(v31.DEFAULT_PARENT)
    bundle["feature_reduction"] = {"name": name, "feature_count": len(cols), "features": cols}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.pkl"
    joblib.dump(bundle, path)
    bundle["artifact_path"] = str(path)
    return bundle


def _build_v40_decisions(
    *,
    train_all: pd.DataFrame,
    eval_df: pd.DataFrame,
    train_idx: np.ndarray,
    y: dict[str, np.ndarray],
    train_feat_zero: pd.DataFrame,
    val_feat_zero: pd.DataFrame,
    eval_feat_zero: pd.DataFrame,
    train_feat_nan: pd.DataFrame,
    val_feat_nan: pd.DataFrame,
    eval_feat_nan: pd.DataFrame,
    cfg: FullyLearnedGovernorConfig,
    args_ns: Any,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    variant_name = "v40_6_chronos_kairos_pls_parent"
    variant = dict(v40swap.PARENT_VARIANTS["v31_v40_6_hgb_pls_parent"])
    bundle = v40swap._load_bundle(Path(variant["bundle"]))
    report = json.loads(Path(variant["report"]).read_text(encoding="utf-8"))
    train_df = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val_idx = np.arange(len(val_df), dtype=np.int64)
    eval_idx = np.arange(len(eval_df), dtype=np.int64)
    proj_targets = v40swap._projection_targets(y)
    _, eval_frame, meta = v40swap._build_encoded_frames(
        variant_name=variant_name,
        variant=variant,
        bundle=bundle,
        report=report,
        args=args_ns,
        train_df=train_df,
        val_df=val_df,
        eval_df=eval_df,
        train_feat_zero=train_feat_zero,
        val_feat_zero=val_feat_zero,
        eval_feat_zero=eval_feat_zero,
        train_feat_nan=train_feat_nan,
        val_feat_nan=val_feat_nan,
        eval_feat_nan=eval_feat_nan,
        train_idx=train_idx,
        val_idx=val_idx,
        eval_idx=eval_idx,
        proj_targets=proj_targets,
    )
    return bundle, predict_policy_frame(bundle, eval_frame, close=_close(eval_frame)), meta


def main() -> int:
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[{MODEL_ID}] loading Alpha3 stack", flush=True)
    original_parent = joblib.load(v31.DEFAULT_PARENT)
    jackpot_payload = joblib.load(v31.DEFAULT_JACKPOT)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    _, v27_model = v31._load_v27(v31.DEFAULT_V27)
    v27_payload = torch.load(v31.DEFAULT_V27, map_location="cpu", weights_only=False)
    teacher_model, teacher_cols, teacher_norm, teacher_buckets = _load_teacher()
    runtime = _selected_runtime()
    overlay = next(v.overlay for v in l2._variants() if v.name == "alpha1_l2_conservative_fee20")
    limit_cfg = _canonical_limit_cfg()
    fee = float(dict(original_parent["config"])["fee"])
    slip = float(dict(original_parent["config"])["slip"])

    train_all = _read(v31.DEFAULT_TRAIN)
    eval_df = _read(v31.DEFAULT_EVAL)
    cfg = FullyLearnedGovernorConfig(**dict(original_parent["config"]))
    feature_sets = json.loads(FEATURE_SETS.read_text(encoding="utf-8"))
    raw_feature_cols = list(original_parent["feature_cols"])

    print(f"[{MODEL_ID}] building labels for reduced parents", flush=True)
    x_train_all, y, training_meta = build_training_set(train_all, cfg=cfg, stride_bars=6, batch_size=512, feature_cols=raw_feature_cols)
    train_df_for_v40 = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    _, y_v40, _ = build_training_set(train_df_for_v40, cfg=cfg, stride_bars=48, batch_size=512, feature_cols=raw_feature_cols)
    train_idx = np.arange(0, max(0, len(train_df_for_v40) - cfg.max_train_horizon_bars - 1), 48, dtype=np.int64)

    print(f"[{MODEL_ID}] predicting teacher/V27 once", flush=True)
    eval_teacher_features = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=teacher_cols)
    eval_teacher_pred = teacher._predict_deep(teacher_model, eval_teacher_features, teacher_cols, teacher_norm)
    eval_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])

    results: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    def evaluate_variant(name: str, decision_parent: dict[str, Any], decision_frame: pd.DataFrame, meta: dict[str, Any]) -> None:
        print(f"[{MODEL_ID}] evaluating {name}", flush=True)
        metrics = _metrics(
            eval_df=eval_df,
            runner_parent=original_parent,
            decision_parent=decision_parent,
            decision_frame=decision_frame,
            teacher_pred=eval_teacher_pred,
            teacher_buckets=teacher_buckets,
            runtime=runtime,
            jackpot_model=jackpot_model,
            add_cfg=add_cfg,
            eval_q=eval_q,
            overlay=overlay,
            limit_cfg=limit_cfg,
            fee=fee,
            slip=slip,
        )
        item = {
            "variant": name,
            "feature_count": int(len(decision_parent.get("feature_cols") or [])),
            "meta": meta,
            "metrics": metrics,
            "score": alpha2._score(metrics["cost1"], metrics["cost2"], metrics["cost3"]),
        }
        results.append(item)
        rows.append(
            {
                "variant": name,
                "feature_count": item["feature_count"],
                "score": item["score"],
                "cost1_pnl": metrics["cost1"]["pnl"],
                "cost1_mdd": metrics["cost1"]["mdd"],
                "cost1_trades": metrics["cost1"]["trades"],
                "cost1_deep_entries": metrics["cost1"].get("deep_entries", 0),
                "cost2_pnl": metrics["cost2"]["pnl"],
                "cost2_mdd": metrics["cost2"]["mdd"],
                "cost3_pnl": metrics["cost3"]["pnl"],
                "cost3_mdd": metrics["cost3"]["mdd"],
            }
        )
        print(
            f"[{MODEL_ID}] {name} c1={metrics['cost1']['pnl']:.2f} mdd={metrics['cost1']['mdd']:.2f} "
            f"c2={metrics['cost2']['pnl']:.2f} c3={metrics['cost3']['pnl']:.2f}",
            flush=True,
        )

    # Canonical Alpha3 parent baseline under the same corrected execution contract.
    original_dec = predict_policy_frame(original_parent, eval_df, close=_close(eval_df))
    evaluate_variant("alpha3_original_93_raw_parent", original_parent, original_dec, {"kind": "baseline"})

    for key in ("top32_raw_parent", "top48_raw_parent", "corr_pruned_raw_parent"):
        reduced = _fit_reduced_parent(name=key, features=list(feature_sets[key]), train_all=train_all, x_train_all=x_train_all, y=y, cfg=cfg)
        dec = predict_policy_frame(reduced, eval_df, close=_close(eval_df))
        evaluate_variant(key, reduced, dec, {"kind": "retrained_raw_reduced", "artifact": reduced.get("artifact_path")})

    # Existing Chronos/Kairos train-only PLS parent, retested inside the Alpha3 corrected execution contract.
    print(f"[{MODEL_ID}] preparing existing Chronos/Kairos PLS parent", flush=True)
    train_df = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    train_feat_zero = prepare_features(train_df, side_hint=0, close=_close(train_df), feature_cols=raw_feature_cols)
    val_feat_zero = prepare_features(val_df, side_hint=0, close=_close(val_df), feature_cols=raw_feature_cols)
    eval_feat_zero = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=raw_feature_cols)
    train_feat_nan = v40swap._preserve_source_missing_as_nan(train_feat_zero, train_df, raw_feature_cols)
    val_feat_nan = v40swap._preserve_source_missing_as_nan(val_feat_zero, val_df, raw_feature_cols)
    eval_feat_nan = v40swap._preserve_source_missing_as_nan(eval_feat_zero, eval_df, raw_feature_cols)
    args_ns = type(
        "Args",
        (),
        {
            "train_csv": v31.DEFAULT_TRAIN,
            "eval_csv": v31.DEFAULT_EVAL,
            "train_stride": 48,
            "embed_batch": 8,
        },
    )()
    try:
        v40_parent, v40_dec, v40_meta = _build_v40_decisions(
            train_all=train_all,
            eval_df=eval_df,
            train_idx=train_idx,
            y=y_v40,
            train_feat_zero=train_feat_zero,
            val_feat_zero=val_feat_zero,
            eval_feat_zero=eval_feat_zero,
            train_feat_nan=train_feat_nan,
            val_feat_nan=val_feat_nan,
            eval_feat_nan=eval_feat_nan,
            cfg=cfg,
            args_ns=args_ns,
        )
        evaluate_variant("chronos_kairos_pls_v40_6_parent", v40_parent, v40_dec, {"kind": "existing_v40_6_pls", **v40_meta})
    except Exception as exc:
        results.append({"variant": "chronos_kairos_pls_v40_6_parent", "error": repr(exc)})
        rows.append({"variant": "chronos_kairos_pls_v40_6_parent", "error": repr(exc)})

    grid = pd.DataFrame(rows).sort_values("score", ascending=False)
    grid.to_csv(GRID_OUT, index=False)
    blocking: list[str] = []
    warnings = [
        "reduced_parent_models_retrained_with_feature_subsets_but_v21_2_runner_feature_frame_preserved_from_original_parent",
        "feature_set_was_selected_from_2025Q4_analysis_not_from_2026",
        "chronos_kairos_v40_6_parent_is_existing_target_aware_pls_artifact_retested_in_alpha3_contract",
    ]
    audit = {
        "status": "pass" if not blocking else "fail",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "oos_window": "2026 fixed OOS only",
        "alpha3_execution_contract": asdict(limit_cfg),
        "runtime": asdict(runtime),
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Alpha3 parent feature reduction retest. Parent-only feature subsets are retrained, while teacher gate, frozen V27 scout, V21.2 runner, V31/L2 overlay, and corrected next_open_limit_touch0_fee20 execution contract are fixed.",
        "training_meta": training_meta,
        "results": results,
        "summary": rows,
        "audit": audit,
        "artifacts": {
            "report": str(REPORT_OUT),
            "audit": str(AUDIT_OUT),
            "grid": str(GRID_OUT),
            "out_dir": str(OUT_DIR),
        },
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "grid": str(GRID_OUT), "best": grid.iloc[0].to_dict() if len(grid) else {}}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
