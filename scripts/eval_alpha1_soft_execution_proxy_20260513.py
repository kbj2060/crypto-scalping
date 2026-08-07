#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import predict_policy_frame  # noqa: E402
from scripts import eval_alpha1_rl_exit_and_sizing_20260513 as alpha1  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts import eval_hf_v13_v31_frozen_parent_layer_ablation_v45 as v45  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _read  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402


MODEL_ID = "alpha1_soft_execution_proxy_20260513"
OUT_DIR = ROOT / "data/ensemble/supervised/alpha1_soft_execution_proxy_20260513"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha1_soft_execution_proxy_20260513_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/alpha1_soft_execution_proxy_20260513_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/alpha1_soft_execution_proxy_20260513_grid.csv"


def _variants() -> list[v45.LayerVariant]:
    base = alpha1.ALPHA1_CFG
    rows = [v45.LayerVariant("alpha1_baseline_taker", "baseline", base)]
    for flow_th in (0.10, 0.16, 0.22, 0.30):
        for fee_mult, slip_mult, label in ((0.80, 0.65, "conservative"), (0.70, 0.50, "balanced"), (0.60, 0.35, "aggressive")):
            rows.append(
                v45.LayerVariant(
                    f"alpha1_soft_exec_{label}_flow{flow_th:.2f}",
                    "soft_execution_proxy",
                    base,
                    execution_sniper=True,
                    sniper_flow_th=flow_th,
                    sniper_fee_mult=fee_mult,
                    sniper_slip_mult=slip_mult,
                )
            )
    return rows


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1.get("trades", 0)) < 20:
        return -1e9 + float(c1.get("pnl", 0.0))
    return float(c1["pnl"] + 0.45 * c2["pnl"] + 0.25 * c3["pnl"] - 0.25 * abs(c1["mdd"]))


def _run(df, bundle, jackpot_model, add_cfg, q, dec, variant, fee: float, slip: float) -> dict[str, Any]:
    return {
        f"cost{mult}": v45.backtest_variant(df, bundle, jackpot_model, add_cfg, q, variant, fee=fee, slip=slip, cost_mult=float(mult), decisions=dec)
        for mult in (1, 2, 3)
    }


def main() -> int:
    print(f"[{MODEL_ID}] loading alpha1 stack", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(v31.DEFAULT_PARENT)
    jackpot_payload = joblib.load(v31.DEFAULT_JACKPOT)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = v31._load_v27(v31.DEFAULT_V27)
    base_cfg = dict(bundle["config"])
    fee = float(base_cfg["fee"])
    slip = float(base_cfg["slip"])
    train_all = _read(v31.DEFAULT_TRAIN)
    eval_df = _read(v31.DEFAULT_EVAL)
    val = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    audit_contract = _audit_contract(train_all, eval_df, list(bundle.get("feature_cols") or []))
    print(f"[{MODEL_ID}] predicting frozen parent/V27", flush=True)
    val_q = v31._predict_all(v27_model, val, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])
    val_dec = predict_policy_frame(bundle, val, close=_close(val))
    eval_dec = predict_policy_frame(bundle, eval_df, close=_close(eval_df))

    rows: list[dict[str, Any]] = []
    selected = None
    best_score = -1e18
    for variant in _variants():
        vm = _run(val, bundle, jackpot_model, add_cfg, val_q, val_dec, variant, fee, slip)
        score = _score(vm["cost1"], vm["cost2"], vm["cost3"])
        row = {
            "name": variant.name,
            "layer": variant.layer,
            "selection_score": score,
            "val_cost1_pnl": vm["cost1"]["pnl"],
            "val_cost1_mdd": vm["cost1"]["mdd"],
            "val_cost1_trades": vm["cost1"]["trades"],
            "val_cost2_pnl": vm["cost2"]["pnl"],
            "val_cost3_pnl": vm["cost3"]["pnl"],
            "sniper_flow_th": variant.sniper_flow_th,
            "sniper_fee_mult": variant.sniper_fee_mult,
            "sniper_slip_mult": variant.sniper_slip_mult,
        }
        rows.append(row)
        if score > best_score:
            best_score = score
            selected = variant
    assert selected is not None
    pd.DataFrame(rows).sort_values("selection_score", ascending=False).to_csv(GRID_OUT, index=False)
    print(f"[{MODEL_ID}] selected {selected.name}", flush=True)

    experiments = []
    for variant in (_variants()[0], selected):
        metrics = _run(eval_df, bundle, jackpot_model, add_cfg, eval_q, eval_dec, variant, fee, slip)
        experiments.append({"name": "alpha1" if variant.name == "alpha1_baseline_taker" else f"alpha1.4::{variant.name}", "variant": asdict(variant), "metrics": metrics, "score": _score(metrics["cost1"], metrics["cost2"], metrics["cost3"])})
        print(f"[{MODEL_ID}] {variant.name} cost1={metrics['cost1']['pnl']:.2f} mdd={metrics['cost1']['mdd']:.2f} cost2={metrics['cost2']['pnl']:.2f} cost3={metrics['cost3']['pnl']:.2f}", flush=True)

    manifest_path = OUT_DIR / "soft_execution_proxy_manifest.json"
    manifest = {"model_id": MODEL_ID, "selected_variant": asdict(selected), "parent_frozen": True, "v27_frozen": True, "v21_2_frozen": True, "alpha1_deep_notional": alpha1.ALPHA1_CFG.notional}
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    best = max(experiments, key=lambda e: e["score"])
    blocking = list(audit_contract.get("blocking", []))
    warnings = list(audit_contract.get("warnings", []))
    warnings.append("soft_execution_proxy_uses_ohlcv_micro_proxy_not_live_l2_orderbook")
    alpha_metrics = experiments[0]["metrics"]
    if best["name"] != "alpha1" and best["metrics"]["cost1"]["pnl"] <= alpha_metrics["cost1"]["pnl"]:
        warnings.append("soft_execution_proxy_did_not_beat_alpha1_cost1")
    if best["metrics"]["cost2"]["pnl"] <= alpha_metrics["cost2"]["pnl"]:
        warnings.append("soft_execution_proxy_did_not_beat_alpha1_cost2")
    if best["metrics"]["cost3"]["pnl"] <= alpha_metrics["cost3"]["pnl"]:
        warnings.append("soft_execution_proxy_did_not_beat_alpha1_cost3")
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": "promote" if not blocking and best["name"] != "alpha1" and best["metrics"]["cost1"]["pnl"] > alpha_metrics["cost1"]["pnl"] and best["metrics"]["cost2"]["pnl"] > alpha_metrics["cost2"]["pnl"] else "iterate",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS only after selection",
        "parent_frozen": True,
        "v27_entry_frozen": True,
        "v21_2_model_frozen": True,
        "selected_variant": asdict(selected),
        "feature_audit": audit_contract,
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Alpha1.4 soft execution proxy. Parent, V21.2, V27, and V31 are frozen; only entry/exit fee and slippage route use a flow/liquidity OHLCV proxy to apply maker-like cost relief when microstructure is favorable.",
        "experiments": experiments,
        "audit": audit,
        "artifacts": {"manifest": str(manifest_path), "report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "grid": str(GRID_OUT)},
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "best": best["name"], "verdict": audit["verdict"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
