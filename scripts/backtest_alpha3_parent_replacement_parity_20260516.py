#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble import fully_learned_governor_policy as flg  # noqa: E402
from ensemble.fully_learned_governor_policy import (  # noqa: E402
    ACTION_LONG,
    FullyLearnedGovernorConfig,
    prepare_features,
)
from scripts import backtest_alpha3_runtime_native_20260515 as native  # noqa: E402
from scripts import eval_alpha3_ft_parent_redesign_20260515 as ft_grouped  # noqa: E402
from scripts import eval_alpha3_ft_transformer_mtl_parent_v2_20260515 as ft_v2  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.eval_hf_v13_deep_tabular_parent_mdd_20260514 import (  # noqa: E402
    RuntimeConfig,
    _decisions_from_outputs,
    _normalise_apply,
    _predict_outputs,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _json_default  # noqa: E402


MODEL_ID = "alpha3_parent_replacement_parity_20260516"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/alpha3_parent_replacement_parity_20260516_summary.json"
DEFAULT_BASE_LEDGER = ROOT / "data/ensemble/reports/alpha3_parent_replacement_parity_20260516_baseline_ledger.csv"
DEFAULT_CANDIDATE_LEDGER = ROOT / "data/ensemble/reports/alpha3_parent_replacement_parity_20260516_candidate_ledger.csv"
DEFAULT_BASE_REPORT = ROOT / "data/ensemble/reports/alpha3_parent_replacement_parity_20260516_baseline.json"
DEFAULT_CANDIDATE_REPORT = ROOT / "data/ensemble/reports/alpha3_parent_replacement_parity_20260516_candidate.json"
CANONICAL_LEDGER = ROOT / "data/ensemble/reports/alpha3_csv_canonical_aligned_native_1m_20260516_ledger_corrected.csv"
FT_GROUPED_MODEL = ROOT / "data/ensemble/supervised/alpha3_ft_parent_redesign_20260515/ft_grouped_hgb_surrogate.pt"
FT_V2_MODEL = ROOT / "data/ensemble/supervised/alpha3_ft_transformer_mtl_parent_v2_20260515/ft_transformer_mtl_parent_v2.pt"
FT_V2_REPORT = ROOT / "data/ensemble/reports/alpha3_ft_transformer_mtl_parent_v2_20260515_summary.json"


def _base_args(report_out: Path, ledger_out: Path, args: argparse.Namespace) -> Namespace:
    return Namespace(
        eval_csv=Path(args.eval_csv),
        report_out=Path(report_out),
        ledger_out=Path(ledger_out),
        start_index=int(args.start_index),
        end_index=int(args.end_index),
        max_bars=None,
        with_m7=False,
        progress=int(args.progress),
        v31_config_json="",
        v31_name="",
        accelerated_cache=False,
        alpha3_csv_execution_parity=True,
        alpha3_cost_mult=1.0,
        alpha3_maker_fee_mult=0.20,
        alpha3_entry_miss="skip",
        alpha3_exit_miss="market_fallback",
        alpha3_csv_state_parity=True,
        alpha3_csv_decision_parity=False,
        alpha3_csv_mark_parity=True,
        alpha3_csv_cooldown_parity_env=True,
        alpha3_csv_loop_parity=True,
        compare_csv_ledger=CANONICAL_LEDGER,
    )


def _load_ft_grouped_parent(device: str, batch_size: int) -> dict[str, Any]:
    ckpt = torch.load(FT_GROUPED_MODEL, map_location="cpu", weights_only=False)
    feature_cols = list(ckpt["feature_cols"])
    label_cfg = FullyLearnedGovernorConfig(**dict(ckpt["label_config"]))
    runtime_parent = joblib.load(v31.DEFAULT_PARENT)
    model = ft_grouped.GroupedFTParent(len(feature_cols), label_cfg, d_model=80, n_layers=3)
    model.load_state_dict(ckpt["state_dict"])
    dummy_y = {
        "action": np.asarray([ACTION_LONG], dtype=np.int64),
        "quality": np.asarray([0.0], dtype=np.float32),
        "notional": np.asarray([0], dtype=np.int64),
        "leverage": np.asarray([0], dtype=np.int64),
        "take_profit": np.asarray([0], dtype=np.int64),
        "stop_loss": np.asarray([0], dtype=np.int64),
        "max_hold": np.asarray([0], dtype=np.int64),
        "cooldown": np.asarray([0], dtype=np.int64),
    }
    parent = ft_grouped._parent_bundle(model, ckpt["normalizer"], label_cfg, feature_cols, device, batch_size, dummy_y)
    return ft_grouped._with_runtime_overlay(parent, runtime_parent)


def _load_ft_v2_runtime() -> RuntimeConfig:
    report = json.loads(FT_V2_REPORT.read_text(encoding="utf-8"))
    candidate = next((x for x in report.get("experiments", []) if str(x.get("name", "")).startswith("alpha3_ft_transformer_mtl_parent_v2::")), None)
    runtime = dict((candidate or {}).get("runtime", {}) or {})
    if not runtime:
        runtime = {
            "name": "ft_mtl_replace_c0.30_q-0.020_s0.90_cap2.10_u0.040",
            "model_key": "ft_transformer_mtl",
            "mode": "replace",
            "confidence": 0.30,
            "quality_floor": -0.020,
            "notional_scale": 0.90,
            "max_notional": 2.10,
            "uncertainty_max": 0.040,
        }
    return RuntimeConfig(**runtime)


def _ft_v2_decisions(df: pd.DataFrame, device: str, batch_size: int) -> pd.DataFrame:
    ckpt = torch.load(FT_V2_MODEL, map_location="cpu", weights_only=False)
    feature_cols = list(ckpt["feature_cols"])
    cfg = FullyLearnedGovernorConfig(**dict(ckpt["config"]))
    model = ft_v2.FTTransformerParentV2(len(feature_cols), cfg, d_model=80, n_layers=3)
    model.load_state_dict(ckpt["state_dict"])
    features = prepare_features(df, side_hint=0, close=_close(df), feature_cols=feature_cols)
    x = _normalise_apply(features, ckpt["normalizer"])
    dev = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    outputs = _predict_outputs(model, x, None, dev, int(batch_size), mc_passes=5)
    return _decisions_from_outputs(outputs, cfg, _load_ft_v2_runtime(), df.index)


def _candidate_predictor(candidate: str, device: str, batch_size: int):
    original_predict = flg.predict_policy_frame
    grouped_parent: dict[str, Any] | None = None
    cache: dict[tuple[str, int, str, str], pd.DataFrame] = {}
    if candidate == "ft_grouped_hgb_surrogate":
        grouped_parent = _load_ft_grouped_parent(device, batch_size)

    def patched_predict_policy_frame(parent: dict[str, Any], df: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        is_alpha3_hgb_parent = isinstance(parent, dict) and str(parent.get("model_type", "")).startswith("fully_learned_governor")
        if not is_alpha3_hgb_parent:
            return original_predict(parent, df, *args, **kwargs)
        if len(df) == 0:
            return original_predict(parent, df, *args, **kwargs)
        key = (
            str(candidate),
            int(len(df)),
            str(pd.Timestamp(df["timestamp"].iloc[0])) if "timestamp" in df.columns else "",
            str(pd.Timestamp(df["timestamp"].iloc[-1])) if "timestamp" in df.columns else "",
        )
        if key not in cache:
            if candidate == "ft_grouped_hgb_surrogate":
                assert grouped_parent is not None
                cache[key] = original_predict(grouped_parent, df, *args, **kwargs).reset_index(drop=True)
            elif candidate == "ft_mtl_v2":
                cache[key] = _ft_v2_decisions(df.reset_index(drop=True), device, batch_size).reset_index(drop=True)
            else:
                raise ValueError(f"unknown candidate: {candidate}")
        return cache[key].copy()

    return original_predict, patched_predict_policy_frame


def _metric_delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    cm = dict(candidate.get("metrics", {}) or {})
    bm = dict(baseline.get("metrics", {}) or {})
    return {
        "return_pct": float(cm.get("return_pct", 0.0) - bm.get("return_pct", 0.0)),
        "max_drawdown_pct": float(cm.get("max_drawdown_pct", 0.0) - bm.get("max_drawdown_pct", 0.0)),
        "closed_trades": int(cm.get("closed_trades", 0) - bm.get("closed_trades", 0)),
        "deep_entries": int(cm.get("deep_entries", 0) - bm.get("deep_entries", 0)),
    }


def _action_counts(report: dict[str, Any]) -> dict[str, Any]:
    p = dict(report.get("parity_compare", {}) or {})
    m = dict(report.get("metrics", {}) or {})
    return {
        "action_events": p.get("candidate_action_events"),
        "event_counts": p.get("candidate_event_counts"),
        "runner_actions": m.get("runner_actions", {}),
        "route_counts": m.get("route_counts", {}),
        "first_action_diff_vs_baseline": p.get("first_action_diff"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Retest Alpha3 parent replacement under the 2026-05-16 CSV/native parity baseline.")
    parser.add_argument("--candidate", choices=("ft_grouped_hgb_surrogate", "ft_mtl_v2"), default="ft_grouped_hgb_surrogate")
    parser.add_argument("--eval-csv", type=Path, default=native.DEFAULT_EVAL_CSV)
    parser.add_argument("--start-index", type=int, default=6999)
    parser.add_argument("--end-index", type=int, default=15638)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--progress", type=int, default=0)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--baseline-report-out", type=Path, default=DEFAULT_BASE_REPORT)
    parser.add_argument("--candidate-report-out", type=Path, default=DEFAULT_CANDIDATE_REPORT)
    parser.add_argument("--baseline-ledger-out", type=Path, default=DEFAULT_BASE_LEDGER)
    parser.add_argument("--candidate-ledger-out", type=Path, default=DEFAULT_CANDIDATE_LEDGER)
    args = parser.parse_args()

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    baseline_args = _base_args(args.baseline_report_out, args.baseline_ledger_out, args)
    print(json.dumps({"stage": "baseline_parity_gate", "candidate": args.candidate}), flush=True)
    baseline = native.run(baseline_args)
    baseline_compare = dict(baseline.get("parity_compare", {}) or {})
    final_pnl_diff = baseline_compare.get("final_pnl_diff_pct", 999.0)
    baseline_pass = bool(
        baseline_compare.get("action_events_match")
        and baseline_compare.get("first_action_diff") is None
        and final_pnl_diff is not None
        and abs(float(final_pnl_diff)) <= 1e-8
    )
    if not baseline_pass:
        raise RuntimeError(f"baseline parity gate failed: {baseline_compare}")

    print(json.dumps({"stage": "candidate_parent_only_retest", "candidate": args.candidate}), flush=True)
    original_predict, patched_predict = _candidate_predictor(args.candidate, str(args.device), int(args.batch_size))
    flg.predict_policy_frame = patched_predict
    try:
        candidate_args = _base_args(args.candidate_report_out, args.candidate_ledger_out, args)
        candidate = native.run(candidate_args)
    finally:
        flg.predict_policy_frame = original_predict

    delta = _metric_delta(candidate, baseline)
    warnings: list[str] = []
    blockers: list[str] = []
    if delta["return_pct"] <= 0:
        warnings.append("candidate_did_not_improve_one_month_return_under_fixed_parity_loop")
    if float(candidate["metrics"]["max_drawdown_pct"]) > float(baseline["metrics"]["max_drawdown_pct"]) + 1e-9:
        warnings.append("candidate_worsened_one_month_mdd_under_fixed_parity_loop")
    if str(args.candidate) == "ft_grouped_hgb_surrogate":
        warnings.append("candidate_origin_report_was_parent_plus_downstream_retune; this run isolates parent_only with frozen downstream")
    summary = {
        "model_id": MODEL_ID,
        "created_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "base_model_alias": "alpha3",
        "frozen_protocol": "alpha3_csv_native_backtest_parity_20260516",
        "primary_mutable_surface": "parent_only",
        "changed_layers": ["parent"],
        "frozen_layers": ["teacher_gate", "v21_2_runner", "v27_deep_scout", "v31_exit", "execution", "accounting", "data"],
        "candidate": str(args.candidate),
        "baseline_parity_gate_passed": baseline_pass,
        "baseline_metrics": baseline.get("metrics", {}),
        "candidate_metrics": candidate.get("metrics", {}),
        "delta_vs_baseline": delta,
        "baseline_action_summary": _action_counts(baseline),
        "candidate_action_summary": _action_counts(candidate),
        "selection_uses_2026": False,
        "selection_window": "candidate artifact selected before this retest; this retest uses fixed 2026 one-month window only for evaluation",
        "oos_window": "2026-01-25 07:15:00..2026-02-24 07:10:00",
        "warnings": warnings,
        "red_team_blockers": blockers,
        "verdict": "promote_to_shadow_candidate" if not blockers and delta["return_pct"] > 0 and float(candidate["metrics"]["max_drawdown_pct"]) <= float(baseline["metrics"]["max_drawdown_pct"]) else "do_not_promote_iterate",
        "artifacts": {
            "summary": str(args.report_out),
            "baseline_report": str(args.baseline_report_out),
            "baseline_ledger": str(args.baseline_ledger_out),
            "candidate_report": str(args.candidate_report_out),
            "candidate_ledger": str(args.candidate_ledger_out),
            "canonical_compare_ledger": str(CANONICAL_LEDGER),
        },
    }
    args.report_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "candidate": args.candidate, "delta": delta, "verdict": summary["verdict"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
