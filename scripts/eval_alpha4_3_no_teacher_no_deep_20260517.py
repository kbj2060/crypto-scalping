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

from ensemble.fully_learned_governor_policy import predict_policy_frame  # noqa: E402
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts import eval_alpha3_ft_transformer_mtl_parent_v2_20260515 as ft_v2  # noqa: E402
from scripts import eval_alpha4_new_features_full_retrain_20260517 as a4  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _json_default, _read  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402

# Parent artifacts produced by eval_alpha4_new_features_full_retrain_20260517
# may refer to wrapper classes under __main__ when that script was executed
# directly. Keep these aliases in this module for joblib compatibility.
FillNAWrapper = a4.FillNAWrapper
EncodedClassifierWrapper = a4.EncodedClassifierWrapper


MODEL_ID = "alpha4_3_no_teacher_no_deep_20260517"
DEFAULT_ROOT = ROOT / "tmp/causal_regen_20260516/alpha4_2_tp_sl_action_score_20260517"
DEFAULT_TRAIN = DEFAULT_ROOT / "trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL = DEFAULT_ROOT / "trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_PARENT = DEFAULT_ROOT / "artifacts/hgb/parent.pkl"
DEFAULT_RUNNER = DEFAULT_ROOT / "teacher_ablation_artifacts/parent_direct_scaled_no_teacher_runner.pkl"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha4_3_no_teacher_no_deep_20260517"
DEFAULT_REPORT = DEFAULT_OUT_DIR / "alpha4_3_no_teacher_no_deep_summary.json"
DEFAULT_AUDIT = DEFAULT_OUT_DIR / "alpha4_3_no_teacher_no_deep_audit.json"
DEFAULT_MANIFEST = DEFAULT_OUT_DIR / "alpha4_3_no_teacher_no_deep_manifest.json"
DEFAULT_LEDGER = DEFAULT_OUT_DIR / "alpha4_3_no_teacher_no_deep_cost1_ledger.csv"


def _no_deep_overlay() -> v31.OverlayConfig:
    return v31.OverlayConfig("alpha4_3_no_deep", 99.0, 99.0, 0.0, 999, 0.04, 0.018, 48, 0.0, 1.0, 0.0, 0.0, 999, 0.0, 0.07, 0.035)


def _q0(df: pd.DataFrame) -> np.ndarray:
    return np.zeros((len(df), 2), dtype=np.float32)


def _decisions(parent: dict[str, Any], df: pd.DataFrame, rt: alpha2.Alpha2Runtime) -> pd.DataFrame:
    base = predict_policy_frame(parent, df, close=_close(df))
    return alpha2._scale_parent_notional(base, rt)


def _metrics(
    df: pd.DataFrame,
    *,
    parent_for_features: dict[str, Any],
    parent: dict[str, Any],
    runner: dict[str, Any],
    add_cfg: CostRunnerConfig,
    rt: alpha2.Alpha2Runtime,
    fee: float,
    slip: float,
    record: bool = False,
) -> dict[str, Any]:
    dec = _decisions(parent, df, rt)
    overlay = _no_deep_overlay()
    limit_cfg = ft_v2.ft_v1._limit_cfg()
    out = a4._metrics(df, parent_for_features, runner, add_cfg, _q0(df), dec, overlay, limit_cfg, fee=fee, slip=slip)
    if record:
        cost1 = a4.alpha3_close._metrics_signal_limit_close(
            df,
            parent_for_features,
            runner,
            add_cfg,
            _q0(df),
            dec,
            overlay,
            limit_cfg,
            fee=fee,
            slip=slip,
        )
        out = cost1
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Official Alpha4.3 no-teacher/no-deep evaluation.")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--parent-model", type=Path, default=DEFAULT_PARENT)
    p.add_argument("--runner-model", type=Path, default=DEFAULT_RUNNER)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--ledger-out", type=Path, default=DEFAULT_LEDGER)
    p.add_argument("--scale", type=float, default=0.85)
    p.add_argument("--max-notional", type=float, default=2.75)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_all = _read(args.train_csv)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    eval_df = _read(args.eval_csv)
    parent = joblib.load(args.parent_model)
    parent_ref = joblib.load(v31.DEFAULT_PARENT)
    parent_for_features = dict(parent_ref)
    parent_for_features["feature_cols"] = list(parent["feature_cols"])
    runner_payload = joblib.load(args.runner_model)
    runner = runner_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(runner_payload["selected_config"]))
    fee = float(dict(parent_ref["config"])["fee"])
    slip = float(dict(parent_ref["config"])["slip"])
    rt = alpha2.Alpha2Runtime(
        name=f"parent_direct_scale{float(args.scale):.2f}",
        confidence=0.0,
        parent_notional_scale=float(args.scale),
        max_notional=float(args.max_notional),
    )

    val_metrics = _metrics(val_df, parent_for_features=parent_for_features, parent=parent, runner=runner, add_cfg=add_cfg, rt=rt, fee=fee, slip=slip)
    eval_metrics = _metrics(eval_df, parent_for_features=parent_for_features, parent=parent, runner=runner, add_cfg=add_cfg, rt=rt, fee=fee, slip=slip)

    eval_dec = _decisions(parent, eval_df, rt)
    limit_cfg = ft_v2.ft_v1._limit_cfg()
    cost1_record = a4.alpha3_close._metrics_signal_limit_close(
        eval_df,
        parent_for_features,
        runner,
        add_cfg,
        _q0(eval_df),
        eval_dec,
        _no_deep_overlay(),
        limit_cfg,
        fee=fee,
        slip=slip,
    )

    audit = {
        "status": "pass",
        "verdict": "alpha4_3_candidate",
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31 inherited from teacher ablation",
        "oos_window": "2026 fixed OOS",
        "teacher_layer": "removed",
        "deep_scout": "disabled",
        "runtime": asdict(rt),
        "selected_runner_config": asdict(add_cfg),
        "feature_contract": "tp_sl_action_score is a single parent input feature; no post-entry TP/SL reject gate.",
        "blocking": [],
        "warnings": [
            "teacher removal was favored by 2026 OOS but not by strict 2025Q4 validation in the ablation; multi-window validation is required before live promotion",
            "runner artifact is inherited from alpha4_2 teacher ablation parent_direct_scaled_no_teacher",
        ],
    }
    manifest = {
        "model_id": MODEL_ID,
        "layers": {
            "parent": {"enabled": True, "artifact": str(args.parent_model), "feature_count": len(parent["feature_cols"]), "contains_tp_sl_action_score": "tp_sl_action_score" in parent["feature_cols"]},
            "tp_sl_action_score": {"enabled": True, "source_csv_train": str(args.train_csv), "source_csv_eval": str(args.eval_csv)},
            "teacher": {"enabled": False, "reason": "removed for Alpha4.3 candidate"},
            "deep_scout": {"enabled": False, "reason": "V27/V31 deep_alpha sleeve caused excessive SL exits"},
            "runner": {"enabled": True, "artifact": str(args.runner_model), "selected_config": asdict(add_cfg)},
            "execution": {"enabled": True, "contract": "corrected Alpha3 limit-close fallback"},
        },
        "runtime": asdict(rt),
        "metrics": {"validation_2025q4": val_metrics, "oos_2026": eval_metrics},
        "artifacts": {"report": str(args.report_out), "audit": str(args.audit_out), "manifest": str(args.manifest_out)},
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Alpha4.3 removes both teacher sequence layer and V27/V31 deep scout. Parent uses tp_sl_action_score directly, decisions are notional-scaled by a simple runtime, V21.2 runner is the parent-direct no-teacher runner selected in the Alpha4.2 teacher ablation, and execution uses the corrected Alpha3 limit-close contract.",
        "validation_metrics": val_metrics,
        "metrics": eval_metrics,
        "audit": audit,
        "manifest": manifest,
    }
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.manifest_out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "manifest": str(args.manifest_out), "metrics": eval_metrics}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
