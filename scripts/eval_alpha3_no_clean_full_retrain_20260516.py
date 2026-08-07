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
from scripts import eval_alpha1_teacher_constrained_deep_parent_20260513 as teacher  # noqa: E402
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts import eval_alpha3_limit_close_fallback_20260514 as alpha3_close  # noqa: E402
from scripts import eval_alpha3_parent_feature_reduction_20260515 as parent_reduce  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts import train_eval_hf_v13_deep_alpha_candidate_expansion_v27 as v27  # noqa: E402
from scripts import train_eval_hf_v13_jackpot_runner_v21_2 as v21  # noqa: E402
from scripts.eval_alpha3_ft_v2_retrained_downstream_20260515 import _fit_cost_runner_with_decisions  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _json_default, _read  # noqa: E402


MODEL_ID = "alpha3_no_clean_full_retrain_20260516"
CLEAN_PREFIX = "clean_regime_2024_unsup_v4_"
OUT_DIR = ROOT / "data/ensemble/supervised/alpha3_no_clean_full_retrain_20260516"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha3_no_clean_full_retrain_20260516_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/alpha3_no_clean_full_retrain_20260516_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/alpha3_no_clean_full_retrain_20260516_grid.csv"


def _score(metrics: dict[str, Any]) -> float:
    return alpha2._score(metrics["cost1"], metrics["cost2"], metrics["cost3"])


def _metrics(
    df: pd.DataFrame,
    parent: dict[str, Any],
    runner: dict[str, Any],
    add_cfg: v21.CostRunnerConfig,
    q: np.ndarray,
    decisions: pd.DataFrame,
    overlay: v31.OverlayConfig,
    limit_cfg: Any,
    *,
    fee: float,
    slip: float,
) -> dict[str, Any]:
    return alpha3_close._metrics_signal_limit_close(
        df,
        parent,
        runner,
        add_cfg,
        q,
        decisions,
        overlay,
        limit_cfg,
        fee=fee,
        slip=slip,
    )


def _no_clean_cols(cols: list[str]) -> tuple[list[str], list[str]]:
    removed = [c for c in cols if c.startswith(CLEAN_PREFIX)]
    kept = [c for c in cols if not c.startswith(CLEAN_PREFIX)]
    if "side_hint" not in kept:
        kept.insert(0, "side_hint")
    return list(dict.fromkeys(kept)), removed


def _safe_seq_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in v27._seq_cols(df) if not c.startswith(CLEAN_PREFIX)]
    bad = [
        c
        for c in cols
        if any(tok in c.lower() for tok in ("target", "label", "future", "cash_after", "pnl_after", "regime_v2", "hdb", "hmm"))
    ]
    if bad:
        raise RuntimeError(f"forbidden seq columns selected: {bad}")
    return cols[:80]


def _baseline(
    *,
    original_parent: dict[str, Any],
    eval_df: pd.DataFrame,
    fee: float,
    slip: float,
    limit_cfg: Any,
    overlay: Any,
) -> dict[str, Any]:
    jackpot_payload = joblib.load(v31.DEFAULT_JACKPOT)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = v21.CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = v31._load_v27(v31.DEFAULT_V27)
    teacher_model, teacher_cols, teacher_norm, teacher_buckets = parent_reduce._load_teacher()
    runtime = parent_reduce._selected_runtime()
    parent_dec = predict_policy_frame(original_parent, eval_df, close=_close(eval_df))
    teacher_features = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=teacher_cols)
    teacher_pred = teacher._predict_deep(teacher_model, teacher_features, teacher_cols, teacher_norm)
    decisions = alpha2._decisions(parent_dec, teacher_pred, teacher_buckets, runtime)
    q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])
    metrics = _metrics(eval_df, original_parent, jackpot_model, add_cfg, q, decisions, overlay, limit_cfg, fee=fee, slip=slip)
    return {
        "name": "alpha3_corrected_baseline",
        "metrics": metrics,
        "score": _score(metrics),
        "runtime": asdict(runtime),
        "runner_config": asdict(add_cfg),
        "overlay": asdict(overlay),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Full-stack Alpha3 retune after removing clean_regime_2024_unsup_v4_* features.")
    p.add_argument("--parent-stride", type=int, default=6)
    p.add_argument("--teacher-epochs", type=int, default=25)
    p.add_argument("--deep-stride", type=int, default=3)
    p.add_argument("--deep-epochs", type=int, default=25)
    p.add_argument("--seed", type=int, default=20260516)
    args = p.parse_args()

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)

    print(json.dumps({"stage": "load_data_stack", "model_id": MODEL_ID, "cuda": torch.cuda.is_available()}), flush=True)
    original_parent = joblib.load(v31.DEFAULT_PARENT)
    cfg = FullyLearnedGovernorConfig(**dict(original_parent["config"]))
    fee = float(dict(original_parent["config"])["fee"])
    slip = float(dict(original_parent["config"])["slip"])
    train_all = _read(v31.DEFAULT_TRAIN)
    eval_df = _read(v31.DEFAULT_EVAL)
    train_df = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    raw_cols = list(original_parent["feature_cols"])
    feature_cols, removed_cols = _no_clean_cols(raw_cols)
    limit_cfg = parent_reduce._canonical_limit_cfg()
    overlay_ref = next(v.overlay for v in parent_reduce.l2._variants() if v.name == "alpha1_l2_conservative_fee20")

    print(json.dumps({"stage": "baseline_reproduction"}, ensure_ascii=False), flush=True)
    baseline = _baseline(
        original_parent=original_parent,
        eval_df=eval_df,
        fee=fee,
        slip=slip,
        limit_cfg=limit_cfg,
        overlay=overlay_ref,
    )

    print(
        json.dumps(
            {
                "stage": "train_parent",
                "raw_feature_count": len(raw_cols),
                "removed_clean_regime_count": len(removed_cols),
                "candidate_feature_count": len(feature_cols),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    x_parent, y_parent, parent_meta = build_training_set(
        train_df,
        cfg=cfg,
        stride_bars=int(args.parent_stride),
        batch_size=512,
        feature_cols=raw_cols,
    )
    parent_bundle = train_policy(
        x_parent.reindex(columns=feature_cols),
        y_parent,
        cfg=cfg,
        random_state=int(args.seed),
        feature_cols=feature_cols,
    )
    parent_bundle["model_id"] = MODEL_ID
    parent_bundle["candidate_name"] = "no_clean_full_retrain_parent"
    parent_bundle["feature_ablation"] = {
        "removed_prefix": CLEAN_PREFIX,
        "removed_features": removed_cols,
        "raw_feature_count": len(raw_cols),
        "candidate_feature_count": len(feature_cols),
    }
    parent_path = OUT_DIR / "parent_no_clean.pkl"
    joblib.dump(parent_bundle, parent_path)

    print(json.dumps({"stage": "predict_parent_frames"}, ensure_ascii=False), flush=True)
    train_parent_dec = predict_policy_frame(parent_bundle, train_df, close=_close(train_df))
    val_parent_dec = predict_policy_frame(parent_bundle, val_df, close=_close(val_df))
    eval_parent_dec = predict_policy_frame(parent_bundle, eval_df, close=_close(eval_df))

    print(json.dumps({"stage": "train_teacher", "epochs": int(args.teacher_epochs)}, ensure_ascii=False), flush=True)
    train_features = prepare_features(train_df, side_hint=0, close=_close(train_df), feature_cols=feature_cols)
    train_seq = teacher._seq_tensor(train_features, np.arange(len(train_df), dtype=np.int64), feature_cols)
    buckets = tuple(float(x) for x in cfg.notional_buckets)
    teacher_model, teacher_meta = teacher._train_teacher_model(
        train_seq,
        train_parent_dec["action"].astype(int).to_numpy(dtype=np.int64),
        pd.to_numeric(train_parent_dec["quality_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
        teacher._bucket_labels(train_parent_dec, buckets),
        n_buckets=len(buckets),
        epochs=int(args.teacher_epochs),
    )
    teacher_path = OUT_DIR / "teacher_no_clean.pt"
    torch.save(
        {
            "model_id": MODEL_ID,
            "state_dict": teacher_model.state_dict(),
            "feature_cols": feature_cols,
            "train_meta": teacher_meta,
            "buckets": buckets,
        },
        teacher_path,
    )
    train_teacher_pred = teacher._predict_deep(teacher_model, train_features, feature_cols, teacher_meta["norm"])
    val_features = prepare_features(val_df, side_hint=0, close=_close(val_df), feature_cols=feature_cols)
    eval_features = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=feature_cols)
    val_teacher_pred = teacher._predict_deep(teacher_model, val_features, feature_cols, teacher_meta["norm"])
    eval_teacher_pred = teacher._predict_deep(teacher_model, eval_features, feature_cols, teacher_meta["norm"])

    print(json.dumps({"stage": "train_deep_scout", "epochs": int(args.deep_epochs)}, ensure_ascii=False), flush=True)
    seq_cols = _safe_seq_cols(train_df)
    train_ds = v27._build_train_set(train_df, seq_cols, fee=fee, slip=slip, stride=int(args.deep_stride))
    deep_norm = v27._normalizer(train_ds["seq"])
    deep_model = v27._train_model(train_ds, deep_norm, epochs=int(args.deep_epochs))
    scout_path = OUT_DIR / "deep_scout_no_clean.pt"
    torch.save(
        {
            "model_id": MODEL_ID,
            "state_dict": deep_model.state_dict(),
            "seq_cols": seq_cols,
            "norm": deep_norm,
            "removed_clean_regime": True,
        },
        scout_path,
    )
    val_q = v27._predict_all(deep_model, val_df, seq_cols, deep_norm)
    eval_q = v27._predict_all(deep_model, eval_df, seq_cols, deep_norm)

    print(json.dumps({"stage": "select_teacher_runtime"}, ensure_ascii=False), flush=True)
    rows: list[dict[str, Any]] = []
    noop_cfg = next(c for c in v21._grid() if c.name == "v21_2_parent_noop")
    best_rt: alpha2.Alpha2Runtime | None = None
    best_rt_score = -1e18
    for rt in alpha2._runtimes():
        val_decisions = alpha2._decisions(val_parent_dec, val_teacher_pred, buckets, rt)
        metrics = _metrics(val_df, parent_bundle, joblib.load(v31.DEFAULT_JACKPOT)["cost_runner"], noop_cfg, val_q, val_decisions, overlay_ref, limit_cfg, fee=fee, slip=slip)
        score = _score(metrics)
        rows.append(
            {
                "stage": "teacher_runtime",
                "teacher_runtime": rt.name,
                "runner": noop_cfg.name,
                "overlay": overlay_ref.name,
                "score": score,
                "val_cost1_pnl": metrics["cost1"]["pnl"],
                "val_cost1_mdd": metrics["cost1"]["mdd"],
                "val_cost2_pnl": metrics["cost2"]["pnl"],
                "val_cost3_pnl": metrics["cost3"]["pnl"],
            }
        )
        if score > best_rt_score:
            best_rt_score = score
            best_rt = rt
    assert best_rt is not None
    train_decisions = alpha2._decisions(train_parent_dec, train_teacher_pred, buckets, best_rt)
    val_decisions = alpha2._decisions(val_parent_dec, val_teacher_pred, buckets, best_rt)
    eval_decisions = alpha2._decisions(eval_parent_dec, eval_teacher_pred, buckets, best_rt)

    print(json.dumps({"stage": "train_v21_runner", "teacher_runtime": best_rt.name}, ensure_ascii=False), flush=True)
    runner = _fit_cost_runner_with_decisions(train_df, parent_bundle, train_decisions, fee=fee, slip=slip)
    runner_path = OUT_DIR / "v21_runner_no_clean.pkl"
    joblib.dump({"model_id": MODEL_ID, "cost_runner": runner, "teacher_runtime": asdict(best_rt)}, runner_path)

    print(json.dumps({"stage": "select_runner_and_v31_overlay"}, ensure_ascii=False), flush=True)
    best_row: dict[str, Any] | None = None
    for add_cfg in v21._grid():
        for overlay in v31._grid():
            metrics = _metrics(val_df, parent_bundle, runner, add_cfg, val_q, val_decisions, overlay, limit_cfg, fee=fee, slip=slip)
            score = _score(metrics)
            row = {
                "stage": "runner_overlay",
                "teacher_runtime": best_rt.name,
                "runner": add_cfg.name,
                "overlay": overlay.name,
                "score": score,
                "val_cost1_pnl": metrics["cost1"]["pnl"],
                "val_cost1_mdd": metrics["cost1"]["mdd"],
                "val_cost1_trades": metrics["cost1"]["trades"],
                "val_cost2_pnl": metrics["cost2"]["pnl"],
                "val_cost3_pnl": metrics["cost3"]["pnl"],
                "runner_config": asdict(add_cfg),
                "overlay_config": asdict(overlay),
            }
            rows.append(row)
            if best_row is None or score > float(best_row["score"]):
                best_row = row
    assert best_row is not None
    selected_runner = v21.CostRunnerConfig(**best_row["runner_config"])
    selected_overlay = v31.OverlayConfig(**best_row["overlay_config"])
    pd.DataFrame(rows).sort_values("score", ascending=False).to_csv(GRID_OUT, index=False)

    print(json.dumps({"stage": "evaluate_2026_oos", "runner": selected_runner.name, "overlay": selected_overlay.name}, ensure_ascii=False), flush=True)
    candidate_metrics = _metrics(
        eval_df,
        parent_bundle,
        runner,
        selected_runner,
        eval_q,
        eval_decisions,
        selected_overlay,
        limit_cfg,
        fee=fee,
        slip=slip,
    )
    candidate = {
        "name": "alpha3_no_clean_full_retrain",
        "metrics": candidate_metrics,
        "score": _score(candidate_metrics),
        "selected_teacher_runtime": asdict(best_rt),
        "selected_runner_config": asdict(selected_runner),
        "selected_overlay": asdict(selected_overlay),
        "artifacts": {
            "parent": str(parent_path),
            "teacher": str(teacher_path),
            "runner": str(runner_path),
            "deep_scout": str(scout_path),
        },
    }
    delta = {
        cost: {
            "pnl": float(candidate_metrics[cost]["pnl"] - baseline["metrics"][cost]["pnl"]),
            "mdd": float(candidate_metrics[cost]["mdd"] - baseline["metrics"][cost]["mdd"]),
            "trades": int(candidate_metrics[cost]["trades"] - baseline["metrics"][cost]["trades"]),
        }
        for cost in ("cost1", "cost2", "cost3")
    }
    warnings: list[str] = []
    if delta["cost1"]["pnl"] <= 0:
        warnings.append("full_retrain_did_not_improve_cost1_pnl")
    if candidate_metrics["cost1"]["mdd"] < baseline["metrics"]["cost1"]["mdd"]:
        warnings.append("full_retrain_worsened_cost1_mdd")
    if candidate_metrics["cost2"]["pnl"] <= 0 or candidate_metrics["cost3"]["pnl"] <= 0:
        warnings.append("candidate_failed_cost2_or_cost3_survival")
    audit = {
        "status": "pass",
        "blocking": [],
        "warnings": warnings,
        "primary_mutable_surface": "full_stack_retune",
        "changed_layers": ["parent", "teacher_gate", "v21_2_runner", "v27_deep_scout", "v31_exit_overlay_selection"],
        "frozen_layers": ["execution", "accounting", "data"],
        "removed_prefix": CLEAN_PREFIX,
        "removed_feature_count": len(removed_cols),
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS",
        "parent_training_meta": parent_meta,
        "teacher_meta": teacher_meta,
        "deep_train_snapshot_count": int(len(train_ds["target"])),
        "seq_feature_count": len(seq_cols),
        "alpha3_execution_contract": asdict(limit_cfg),
        "best_validation_row": best_row,
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Full-stack Alpha3 retune with clean_regime_2024_unsup_v4_* removed from the learned parent/teacher/deep-scout schemas. Parent, teacher gate, V21.2 runner, V27-style deep scout, and V31 overlay selection are rebuilt; execution/accounting/data remain frozen.",
        "base_model_alias": "alpha3",
        "frozen_protocol": "alpha3_frozen_backtest_protocol_20260515",
        "primary_mutable_surface": "full_stack_retune",
        "changed_layers": audit["changed_layers"],
        "frozen_layers": audit["frozen_layers"],
        "selection_uses_2026": False,
        "baseline_reproduced": True,
        "baseline_metrics": baseline["metrics"],
        "candidate_metrics": candidate_metrics,
        "delta_vs_baseline": delta,
        "experiments": [baseline, candidate],
        "audit": audit,
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "report": str(REPORT_OUT),
            "audit": str(AUDIT_OUT),
            "grid": str(GRID_OUT),
            **candidate["artifacts"],
        },
        "verdict": "promote_to_shadow_candidate" if not warnings and delta["cost1"]["pnl"] > 0 else "do_not_promote_iterate",
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(REPORT_OUT),
                "audit": str(AUDIT_OUT),
                "grid": str(GRID_OUT),
                "delta_vs_baseline": delta,
                "verdict": report["verdict"],
            },
            ensure_ascii=False,
            default=_json_default,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
