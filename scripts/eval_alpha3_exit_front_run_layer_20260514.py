#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import predict_policy_frame, prepare_features  # noqa: E402
from scripts import eval_alpha1_l2_execution_replay_20260513 as l2  # noqa: E402
from scripts import eval_alpha1_teacher_constrained_deep_parent_20260513 as teacher  # noqa: E402
from scripts import eval_alpha2_1_signal_immediate_limit_20260514 as alpha3  # noqa: E402
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _read  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402


MODEL_ID = "alpha3_exit_front_run_layer_20260514"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha3_exit_front_run_layer_20260514_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/alpha3_exit_front_run_layer_20260514_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/alpha3_exit_front_run_layer_20260514_grid.csv"


def _score(metrics: dict[str, Any]) -> float:
    return alpha2._score(metrics["cost1"], metrics["cost2"], metrics["cost3"])


def _selected_alpha3_runtime() -> alpha2.Alpha2Runtime:
    audit = json.loads(alpha3.ALPHA2_AUDIT.read_text(encoding="utf-8"))
    runtime = dict(audit.get("selected_runtime", {}) or {})
    return alpha2.Alpha2Runtime(
        name=str(runtime.get("name", "noflip_c0.56_parent_scale1.10")),
        confidence=float(runtime.get("confidence", 0.56)),
        parent_notional_scale=float(runtime.get("parent_notional_scale", 1.10)),
        max_notional=float(runtime.get("max_notional", 2.75)),
    )


def _configs() -> list[alpha3.ImmediateLimitConfig]:
    rows: list[alpha3.ImmediateLimitConfig] = []
    for exit_offset in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
        for penetration in (0.0, 0.25, 0.5, 1.0):
            rows.append(
                alpha3.ImmediateLimitConfig(
                    name=f"entry2_exit{exit_offset:g}_pen{penetration:g}_fee20",
                    anchor="next_open",
                    entry_offset_bps=2.0,
                    exit_offset_bps=float(exit_offset),
                    penetration_bps=float(penetration),
                    maker_fee_mult=0.20,
                    entry_miss="market_fallback",
                    exit_miss="market_fallback",
                )
            )
    return rows


def _load_fixed_stack() -> dict[str, Any]:
    rt = _selected_alpha3_runtime()
    parent = joblib.load(v31.DEFAULT_PARENT)
    jackpot_payload = joblib.load(v31.DEFAULT_JACKPOT)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = v31._load_v27(v31.DEFAULT_V27)
    teacher_payload = torch.load(alpha3.TEACHER_MODEL, map_location="cpu", weights_only=False)
    teacher_model = alpha2._load_teacher_model(teacher_payload)
    selected_variant = next(v for v in l2._variants() if v.name == "alpha1_l2_conservative_fee20")
    return {
        "runtime": rt,
        "parent": parent,
        "jackpot_model": jackpot_model,
        "add_cfg": add_cfg,
        "v27_payload": v27_payload,
        "v27_model": v27_model,
        "teacher_payload": teacher_payload,
        "teacher_model": teacher_model,
        "overlay": selected_variant.overlay,
        "selected_l2_variant": selected_variant,
        "fee": float(dict(parent["config"])["fee"]),
        "slip": float(dict(parent["config"])["slip"]),
    }


def _decisions_and_q(df: pd.DataFrame, stack: dict[str, Any]) -> tuple[pd.DataFrame, Any]:
    teacher_payload = stack["teacher_payload"]
    feature_cols = list(teacher_payload["feature_cols"])
    base_dec = predict_policy_frame(stack["parent"], df, close=_close(df))
    features = prepare_features(df, side_hint=0, close=_close(df), feature_cols=feature_cols)
    pred = teacher._predict_deep(
        stack["teacher_model"],
        features,
        feature_cols,
        dict(dict(teacher_payload["train_meta"])["norm"]),
    )
    decisions = alpha2._decisions(
        base_dec,
        pred,
        tuple(float(x) for x in teacher_payload["buckets"]),
        stack["runtime"],
    )
    q = v31._predict_all(
        stack["v27_model"],
        df,
        stack["v27_payload"]["seq_cols"],
        stack["v27_payload"]["norm"],
    )
    return decisions, q


def main() -> int:
    print(f"[{MODEL_ID}] loading fixed Alpha3 stack", flush=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    stack = _load_fixed_stack()
    train_all = _read(v31.DEFAULT_TRAIN)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    eval_df = _read(v31.DEFAULT_EVAL)

    print(f"[{MODEL_ID}] rebuilding decisions and V27 q", flush=True)
    val_dec, val_q = _decisions_and_q(val_df, stack)
    eval_dec, eval_q = _decisions_and_q(eval_df, stack)

    print(f"[{MODEL_ID}] selecting exit placement on 2025Q4", flush=True)
    rows: list[dict[str, Any]] = []
    best_cfg: alpha3.ImmediateLimitConfig | None = None
    best_score = -1e18
    for cfg in _configs():
        metrics = alpha3._metrics_signal_limit(
            val_df,
            stack["parent"],
            stack["jackpot_model"],
            stack["add_cfg"],
            val_q,
            val_dec,
            stack["overlay"],
            cfg,
            fee=stack["fee"],
            slip=stack["slip"],
        )
        score = _score(metrics)
        rows.append(
            {
                **asdict(cfg),
                "selection_score": score,
                "val_cost1_pnl": metrics["cost1"]["pnl"],
                "val_cost1_mdd": metrics["cost1"]["mdd"],
                "val_cost1_trades": metrics["cost1"]["trades"],
                "val_cost2_pnl": metrics["cost2"]["pnl"],
                "val_cost3_pnl": metrics["cost3"]["pnl"],
                "val_cost1_route_counts": json.dumps(metrics["cost1"].get("route_counts", {}), sort_keys=True),
            }
        )
        if score > best_score:
            best_score = score
            best_cfg = cfg
            print(
                f"[{MODEL_ID}] new best {cfg.name} score={score:.2f} "
                f"c1={metrics['cost1']['pnl']:.2f} mdd={metrics['cost1']['mdd']:.2f} "
                f"c2={metrics['cost2']['pnl']:.2f} c3={metrics['cost3']['pnl']:.2f}",
                flush=True,
            )
    assert best_cfg is not None
    pd.DataFrame(rows).sort_values("selection_score", ascending=False).to_csv(GRID_OUT, index=False)

    print(f"[{MODEL_ID}] fixed 2026 OOS", flush=True)
    baseline_cfg = alpha3.ImmediateLimitConfig(
        "next_open_limit_offset2_entry_fallback_fee20",
        "next_open",
        2.0,
        2.0,
        0.5,
        0.20,
        entry_miss="market_fallback",
        exit_miss="market_fallback",
    )
    taker = alpha2._metrics(
        eval_df,
        stack["parent"],
        stack["jackpot_model"],
        stack["add_cfg"],
        eval_q,
        eval_dec,
        l2._variants()[0],
        fee=stack["fee"],
        slip=stack["slip"],
    )
    old_l2 = alpha2._metrics(
        eval_df,
        stack["parent"],
        stack["jackpot_model"],
        stack["add_cfg"],
        eval_q,
        eval_dec,
        stack["selected_l2_variant"],
        fee=stack["fee"],
        slip=stack["slip"],
    )
    baseline = alpha3._metrics_signal_limit(
        eval_df,
        stack["parent"],
        stack["jackpot_model"],
        stack["add_cfg"],
        eval_q,
        eval_dec,
        stack["overlay"],
        baseline_cfg,
        fee=stack["fee"],
        slip=stack["slip"],
    )
    candidate = alpha3._metrics_signal_limit(
        eval_df,
        stack["parent"],
        stack["jackpot_model"],
        stack["add_cfg"],
        eval_q,
        eval_dec,
        stack["overlay"],
        best_cfg,
        fee=stack["fee"],
        slip=stack["slip"],
    )
    experiments = [
        {"name": "alpha2_1_next_open_taker_control", "metrics": taker, "score": _score(taker)},
        {"name": "alpha2_1_old_l2_replay_fee20_control", "metrics": old_l2, "score": _score(old_l2)},
        {"name": f"alpha3_baseline::{baseline_cfg.name}", "config": asdict(baseline_cfg), "metrics": baseline, "score": _score(baseline)},
        {"name": f"alpha3_exit_front_run::{best_cfg.name}", "config": asdict(best_cfg), "metrics": candidate, "score": _score(candidate)},
    ]
    for exp in experiments:
        m = exp["metrics"]
        print(
            f"[{MODEL_ID}] {exp['name']} c1={m['cost1']['pnl']:.2f} mdd={m['cost1']['mdd']:.2f} "
            f"c2={m['cost2']['pnl']:.2f} c3={m['cost3']['pnl']:.2f}",
            flush=True,
        )

    warnings = [
        "signal_limit_fill_uses_5m_high_low_touch_proxy_not_queue_fill",
        "real_l2_queue_and_partial_fill_require_forward_shadow_validation",
        "exit_offset_changes_execution_price_model_and_must_be_live_shadow_validated",
    ]
    if candidate["cost1"]["pnl"] <= baseline["cost1"]["pnl"]:
        warnings.append("candidate_did_not_improve_alpha3_cost1_pnl")
    if candidate["cost1"]["mdd"] < baseline["cost1"]["mdd"]:
        warnings.append("candidate_worsened_alpha3_cost1_mdd")
    audit = {
        "status": "pass",
        "verdict": "shadow_promote_candidate" if not any(w.startswith("candidate_") for w in warnings) else "iterate",
        "blocking": [],
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS only after selection",
        "selected_config": asdict(best_cfg),
        "baseline_config": asdict(baseline_cfg),
        "frozen_decision_stack": {
            "runtime": asdict(stack["runtime"]),
            "hgb_parent": str(v31.DEFAULT_PARENT),
            "teacher": str(alpha3.TEACHER_MODEL),
            "v27_deep_scout": str(v31.DEFAULT_V27),
            "v21_2_jackpot": str(v31.DEFAULT_JACKPOT),
        },
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Alpha3 model-wide exit placement layer. The Alpha3 decision stack is frozen. Entry remains next-open post-only limit with 2bps passive offset and market fallback, while the shared historical touch penetration is selected with the exit placement config. Reduce-only exit placement is selected separately, allowing exits to use a different passive offset before market fallback.",
        "experiments": experiments,
        "selection_grid": str(GRID_OUT),
        "audit": audit,
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "grid": str(GRID_OUT), "selected": best_cfg.name}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
