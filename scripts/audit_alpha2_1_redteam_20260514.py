#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import predict_policy_frame, prepare_features  # noqa: E402
from scripts import eval_alpha1_l2_execution_replay_20260513 as l2  # noqa: E402
from scripts import eval_alpha1_teacher_constrained_deep_parent_20260513 as teacher  # noqa: E402
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _feature_cols, _read  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402


MODEL_ID = "alpha2_1_redteam_audit_20260514"
TEACHER_MODEL = ROOT / "data/ensemble/supervised/alpha1_l2_teacher_deep_parent_20260514/teacher_deep_parent_l2_replay.pt"
ALPHA2_REPORT = ROOT / "data/ensemble/reports/alpha2_teacher_l2_runtime_sweep_20260514_summary.json"
ALPHA2_AUDIT = ROOT / "data/ensemble/reports/alpha2_teacher_l2_runtime_sweep_20260514_audit.json"
OUT_REPORT = ROOT / "data/ensemble/reports/alpha2_1_redteam_audit_20260514.json"
OUT_MD = ROOT / "docs/experiments/alpha2_1_redteam_audit_20260514.md"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return dict(json.load(f))


def _extract_live_constants() -> dict[str, Any]:
    text = (ROOT / "trading_bot.py").read_text(encoding="utf-8")

    def f(name: str, default: float) -> float:
        m = re.search(rf"{re.escape(name)}\s*=\s*([0-9.]+)", text)
        return float(m.group(1)) if m else float(default)

    return {
        "model_id_literal_present": "alpha2_1_teacher_l2_runtime_sweep_20260514" in text,
        "teacher_path_literal_present": "alpha1_l2_teacher_deep_parent_20260514/teacher_deep_parent_l2_replay.pt" in text,
        "confidence": f("FINAL_GOVERNOR_ALPHA2_1_CONFIDENCE", -1.0),
        "parent_notional_scale": f("FINAL_GOVERNOR_ALPHA2_1_PARENT_NOTIONAL_SCALE", -1.0),
        "max_notional": f("FINAL_GOVERNOR_ALPHA2_1_MAX_NOTIONAL", -1.0),
        "requires_audit_pass": "alpha2_1_audit_not_pass" in text,
        "blocks_selection_uses_2026": "alpha2_1_audit_selection_uses_2026" in text,
        "no_flip_runtime": '"allow_flip": False' in text,
        "parent_scale_applied": "alpha2_1_parent_notional_scale" in text and "parent_notional_after" in text,
    }


def _live_execution_contract() -> dict[str, Any]:
    from trading_bot_modules import binance_execution as be

    return {
        "alpha14_router_enabled_default": bool(be.BINANCE_EXECUTION_ALPHA14_ROUTER_ENABLE),
        "maker_reduce_only_enabled_default": bool(be.BINANCE_EXECUTION_MAKER_REDUCE_ONLY_ENABLE),
        "maker_fallback_market_default": bool(be.BINANCE_EXECUTION_MAKER_FALLBACK_MARKET),
        "maker_wait_sec_default": float(be.BINANCE_EXECUTION_MAKER_WAIT_SEC),
        "maker_max_spread_bps_default": float(be.BINANCE_EXECUTION_MAKER_MAX_SPREAD_BPS),
        "maker_min_imbalance_default": float(be.BINANCE_EXECUTION_MAKER_MIN_IMBALANCE),
        "maker_min_microprice_edge_bps_default": float(be.BINANCE_EXECUTION_MAKER_MIN_MICROPRICE_EDGE_BPS),
    }


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    return alpha2._score(c1, c2, c3)


def main() -> int:
    print(f"[{MODEL_ID}] loading reports and artifacts", flush=True)
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)

    report = _load_json(ALPHA2_REPORT)
    audit = _load_json(ALPHA2_AUDIT)
    teacher_payload = torch.load(TEACHER_MODEL, map_location="cpu", weights_only=False)

    runtime = dict(audit.get("selected_runtime", {}) or {})
    selected_variant = dict(audit.get("selected_variant", {}) or {})
    rt = alpha2.Alpha2Runtime(
        name=str(runtime.get("name", "noflip_c0.56_parent_scale1.10")),
        confidence=float(runtime.get("confidence", 0.56)),
        parent_notional_scale=float(runtime.get("parent_notional_scale", 1.10)),
        max_notional=float(runtime.get("max_notional", 2.75)),
    )

    print(f"[{MODEL_ID}] rebuilding 2026 Alpha2.1 decisions", flush=True)
    parent = joblib.load(v31.DEFAULT_PARENT)
    jackpot_payload = joblib.load(v31.DEFAULT_JACKPOT)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = v31._load_v27(v31.DEFAULT_V27)
    fee = float(dict(parent["config"])["fee"])
    slip = float(dict(parent["config"])["slip"])
    train_all = _read(v31.DEFAULT_TRAIN)
    eval_df = _read(v31.DEFAULT_EVAL)
    train_fit = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    contract_features = _feature_cols(train_all, eval_df)
    parent_audit = _audit_contract(train_all, eval_df, list(parent.get("feature_cols") or []))

    teacher_model = alpha2._load_teacher_model(teacher_payload)
    feature_cols = list(teacher_payload["feature_cols"])
    norm = dict(dict(teacher_payload["train_meta"])["norm"])
    buckets = tuple(float(x) for x in teacher_payload["buckets"])
    eval_dec = predict_policy_frame(parent, eval_df, close=_close(eval_df))
    eval_features = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=contract_features)
    eval_pred = teacher._predict_deep(teacher_model, eval_features, feature_cols, norm)
    alpha21_dec = alpha2._decisions(eval_dec, eval_pred, buckets, rt)
    eval_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])

    print(f"[{MODEL_ID}] re-running taker and L2 fee stress", flush=True)
    variant_results: list[dict[str, Any]] = []
    for variant in l2._variants():
        metrics = alpha2._metrics(eval_df, parent, jackpot_model, add_cfg, eval_q, alpha21_dec, variant, fee=fee, slip=slip)
        variant_results.append(
            {
                "variant": variant.name,
                "layer": variant.layer,
                "sniper_fee_mult": float(variant.sniper_fee_mult),
                "sniper_slip_mult": float(variant.sniper_slip_mult),
                "score": _score(metrics["cost1"], metrics["cost2"], metrics["cost3"]),
                "metrics": metrics,
            }
        )
        print(
            f"[{MODEL_ID}] {variant.name} c1={metrics['cost1']['pnl']:.2f} "
            f"mdd={metrics['cost1']['mdd']:.2f} c2={metrics['cost2']['pnl']:.2f} c3={metrics['cost3']['pnl']:.2f}",
            flush=True,
        )

    fee20 = next(x for x in variant_results if x["variant"] == "alpha1_l2_conservative_fee20")
    taker = next(x for x in variant_results if x["variant"] == "alpha1_taker_baseline")
    live_constants = _extract_live_constants()
    live_exec = _live_execution_contract()
    l2_stats = l2._live_l2_stats()

    findings: list[dict[str, Any]] = []
    blocking_live: list[str] = []
    warnings: list[str] = []

    if audit.get("status") != "pass":
        blocking_live.append("stored_alpha2_1_audit_not_pass")
    if bool(audit.get("selection_uses_2026", False)):
        blocking_live.append("stored_alpha2_1_selection_uses_2026")
    if parent_audit.get("status") != "pass":
        blocking_live.append("parent_feature_contract_failed")

    maker_routes = dict(fee20["metrics"]["cost1"].get("route_counts", {})).get("conservative_maker_replay", 0)
    fallback_routes = dict(fee20["metrics"]["cost1"].get("route_counts", {})).get("l2_replay_taker_fallback", 0)
    route_total = max(int(maker_routes) + int(fallback_routes), 1)
    maker_ratio = float(maker_routes / route_total)

    if not l2_stats.get("usable_for_replay", False):
        blocking_live.append("l2_forward_snapshots_insufficient_for_live_promotion")
        findings.append(
            {
                "severity": "HIGH",
                "id": "synthetic_l2_replay_not_validated",
                "detail": "Alpha2.1 PnL depends on conservative_l2_replay, but live orderbook snapshots are not yet sufficient for real fill validation.",
                "evidence": l2_stats,
            }
        )

    if maker_ratio >= 0.50:
        findings.append(
            {
                "severity": "HIGH",
                "id": "pnl_dominated_by_synthetic_maker_fills",
                "detail": "Cost1 replay gives maker-like fills with reduced fee and zero slippage on a majority of route events.",
                "maker_ratio_cost1": maker_ratio,
                "route_counts_cost1": fee20["metrics"]["cost1"].get("route_counts", {}),
            }
        )

    if bool(live_exec["maker_reduce_only_enabled_default"]) is False:
        blocking_live.append("backtest_live_exit_route_parity_failed_reduce_only_maker_disabled")
        findings.append(
            {
                "severity": "HIGH",
                "id": "backtest_live_exit_route_mismatch",
                "detail": "Backtest L2 replay can apply maker-like fills to exits. Live default routes reduce-only exits to market because maker_reduce_only is disabled.",
                "live_execution_contract": live_exec,
            }
        )

    stored = next(e for e in report.get("experiments", []) if str(e.get("name", "")).startswith("alpha2_1::"))
    stored_c1 = float(stored["metrics"]["cost1"]["pnl"])
    recomputed_c1 = float(fee20["metrics"]["cost1"]["pnl"])
    if abs(stored_c1 - recomputed_c1) > 1e-6:
        blocking_live.append("recomputed_metric_mismatch")
        findings.append(
            {
                "severity": "CRITICAL",
                "id": "recomputed_metric_mismatch",
                "stored_cost1_pnl": stored_c1,
                "recomputed_cost1_pnl": recomputed_c1,
            }
        )

    if "alpha2_1_runtime_sweep_did_not_beat_alpha2_reference" in audit.get("warnings", []):
        warnings.append("alpha2_1_did_not_beat_alpha2_reference_combined_score")
        findings.append(
            {
                "severity": "MEDIUM",
                "id": "alpha2_1_not_best_combined_score",
                "detail": "Alpha2.1 improved cost1/MDD but had worse combined score than Alpha2 reference due to weaker cost2/cost3.",
            }
        )

    if not all(
        [
            live_constants["model_id_literal_present"],
            live_constants["teacher_path_literal_present"],
            abs(live_constants["confidence"] - rt.confidence) < 1e-12,
            abs(live_constants["parent_notional_scale"] - rt.parent_notional_scale) < 1e-12,
            abs(live_constants["max_notional"] - rt.max_notional) < 1e-12,
            live_constants["requires_audit_pass"],
            live_constants["blocks_selection_uses_2026"],
        ]
    ):
        blocking_live.append("live_alpha2_1_runtime_constant_or_guard_mismatch")

    raw_taker_vs_l2 = {
        "taker_cost1_pnl": taker["metrics"]["cost1"]["pnl"],
        "taker_cost1_mdd": taker["metrics"]["cost1"]["mdd"],
        "l2_fee20_cost1_pnl": fee20["metrics"]["cost1"]["pnl"],
        "l2_fee20_cost1_mdd": fee20["metrics"]["cost1"]["mdd"],
        "pnl_lift_from_l2_fee20": float(fee20["metrics"]["cost1"]["pnl"] - taker["metrics"]["cost1"]["pnl"]),
    }

    if raw_taker_vs_l2["pnl_lift_from_l2_fee20"] > abs(raw_taker_vs_l2["taker_cost1_pnl"]) * 0.25:
        findings.append(
            {
                "severity": "HIGH",
                "id": "execution_assumption_large_pnl_lift",
                "detail": "L2 fee20 replay materially changes PnL versus the same Alpha2.1 decisions under taker execution.",
                **raw_taker_vs_l2,
            }
        )

    verdict = "fail_live_promotion" if blocking_live else "pass_shadow_only"
    out = {
        "model_id": MODEL_ID,
        "subject": "Alpha2.1 teacher + L2 runtime sweep",
        "status": "pass_shadow_audit",
        "verdict": verdict,
        "blocking_for_live_promotion": blocking_live,
        "warnings": warnings + list(audit.get("warnings", [])) + list(parent_audit.get("warnings", [])),
        "findings": findings,
        "artifact_hashes": {
            "teacher_model_sha256": _sha256(TEACHER_MODEL),
            "summary_sha256": _sha256(ALPHA2_REPORT),
            "audit_sha256": _sha256(ALPHA2_AUDIT),
        },
        "split_contract": {
            "teacher_train_window": "2025-01-01..2025-09-30",
            "selection_window": str(audit.get("selection_window")),
            "oos_window": str(audit.get("oos_window")),
            "selection_uses_2026": bool(audit.get("selection_uses_2026", False)),
            "train_range": [str(train_all["timestamp"].iloc[0]), str(train_all["timestamp"].iloc[-1])],
            "train_fit_range": [str(train_fit["timestamp"].iloc[0]), str(train_fit["timestamp"].iloc[-1])],
            "validation_range": [str(val["timestamp"].iloc[0]), str(val["timestamp"].iloc[-1])],
            "eval_range": [str(eval_df["timestamp"].iloc[0]), str(eval_df["timestamp"].iloc[-1])],
        },
        "parent_feature_audit": parent_audit,
        "teacher_payload": {
            "model_id": teacher_payload.get("model_id"),
            "feature_count": len(feature_cols),
            "bucket_count": len(buckets),
            "train_meta_keys": sorted(dict(teacher_payload.get("train_meta", {})).keys()),
        },
        "selected_runtime": runtime,
        "selected_variant": selected_variant,
        "live_constants": live_constants,
        "live_execution_contract": live_exec,
        "l2_stats": l2_stats,
        "same_decision_execution_sensitivity": variant_results,
        "raw_taker_vs_l2": raw_taker_vs_l2,
        "recommendation": (
            "Do not treat Alpha2.1 + L2 fee20 PnL as clean live-promotable PnL until real L2 decision snapshots validate maker fill rates "
            "and live reduce-only exit routing matches the replay contract. Shadow is acceptable; live notional should stay conservative."
        ),
    }
    OUT_REPORT.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")

    md = f"""# Alpha2.1 Red Team Audit 2026-05-14

## Verdict

`{verdict}`

Alpha2.1 can remain a shadow/aggressive research branch, but the current +718.70% PnL should not be treated as clean live-promotable PnL.

## Main Findings

- L2 snapshots usable for replay: `{bool(l2_stats.get('usable_for_replay', False))}`; rows: `{int(l2_stats.get('rows', 0) or 0)}`.
- Cost1 route maker ratio under selected replay: `{maker_ratio:.2%}`.
- Live reduce-only maker exits enabled by default: `{bool(live_exec['maker_reduce_only_enabled_default'])}`.
- Same Alpha2.1 decisions under taker execution: `{taker['metrics']['cost1']['pnl']:.2f}%` PnL / `{taker['metrics']['cost1']['mdd']:.2f}%` MDD.
- Same Alpha2.1 decisions under selected L2 fee20 replay: `{fee20['metrics']['cost1']['pnl']:.2f}%` PnL / `{fee20['metrics']['cost1']['mdd']:.2f}%` MDD.

## Blocking For Live Promotion

{chr(10).join(f'- `{x}`' for x in blocking_live) if blocking_live else '- none'}

## Recommendation

Use Alpha2.1 only as shadow or very conservative live sizing until real L2 fill statistics validate the synthetic maker replay assumptions and live exit routing matches backtest routing.
"""
    OUT_MD.write_text(md, encoding="utf-8")
    print(json.dumps({"report": str(OUT_REPORT), "md": str(OUT_MD), "verdict": verdict}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
