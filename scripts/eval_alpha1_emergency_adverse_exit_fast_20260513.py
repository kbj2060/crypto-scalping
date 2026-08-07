#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import predict_policy_frame  # noqa: E402
from scripts import eval_alpha1_emergency_adverse_exit_20260513 as full  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _read  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402


MODEL_ID = "alpha1_emergency_adverse_exit_fast_20260513"
OUT_DIR = ROOT / "data/ensemble/supervised/alpha1_emergency_adverse_exit_fast_20260513"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha1_emergency_adverse_exit_fast_20260513_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/alpha1_emergency_adverse_exit_fast_20260513_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/alpha1_emergency_adverse_exit_fast_20260513_grid.csv"


def _configs() -> list[full.Runtime]:
    return [
        full.Runtime("alpha1_4_identity", "none", 999, 99.0, -99.0, 99.0, 99.0, False, 0),
        full.Runtime("deep_both_mae012", "deep", 2, 0.012, -0.0066, 0.28, 0.014, True, 6),
        full.Runtime("deep_both_mae018", "deep", 2, 0.018, -0.0099, 0.28, 0.014, True, 6),
        full.Runtime("deep_both_strict", "deep", 3, 0.018, -0.0099, 0.40, 0.020, True, 6),
        full.Runtime("deep_or_strict", "deep", 3, 0.018, -0.0135, 0.50, 0.025, False, 6),
        full.Runtime("all_both_mae012", "all", 2, 0.012, -0.0066, 0.28, 0.014, True, 6),
        full.Runtime("all_both_mae018", "all", 2, 0.018, -0.0099, 0.28, 0.014, True, 6),
        full.Runtime("all_both_strict", "all", 3, 0.018, -0.0099, 0.40, 0.020, True, 6),
    ]


def _metrics(df, bundle, jackpot_model, add_cfg, q, dec, rt, fee, slip):
    return {
        f"cost{m}": full.backtest_emergency(df, bundle, jackpot_model, add_cfg, q, rt, fee=fee, slip=slip, cost_mult=float(m), decisions=dec)
        for m in (1, 2, 3)
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
    val_q = v31._predict_all(v27_model, val, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])
    val_dec = predict_policy_frame(bundle, val, close=_close(val))
    eval_dec = predict_policy_frame(bundle, eval_df, close=_close(eval_df))
    rows = []
    selected = None
    best_score = -1e18
    for rt in _configs():
        vm = _metrics(val, bundle, jackpot_model, add_cfg, val_q, val_dec, rt, fee, slip)
        score = full._score(vm["cost1"], vm["cost2"], vm["cost3"])
        rows.append({**asdict(rt), "selection_score": score, "val_cost1_pnl": vm["cost1"]["pnl"], "val_cost1_mdd": vm["cost1"]["mdd"], "val_cost2_pnl": vm["cost2"]["pnl"], "val_cost3_pnl": vm["cost3"]["pnl"], "val_emergency_exits": sum(v for k, v in vm["cost1"]["exits"].items() if "emergency" in k)})
        if score > best_score:
            best_score = score
            selected = rt
    assert selected is not None
    pd.DataFrame(rows).sort_values("selection_score", ascending=False).to_csv(GRID_OUT, index=False)
    experiments = []
    for name, rt in (("alpha1.4", _configs()[0]), (f"alpha1.5::{selected.name}", selected)):
        metrics = _metrics(eval_df, bundle, jackpot_model, add_cfg, eval_q, eval_dec, rt, fee, slip)
        experiments.append({"name": name, "config": asdict(rt), "metrics": metrics, "score": full._score(metrics["cost1"], metrics["cost2"], metrics["cost3"])})
        print(f"[{MODEL_ID}] {name} cost1={metrics['cost1']['pnl']:.2f} mdd={metrics['cost1']['mdd']:.2f} cost2={metrics['cost2']['pnl']:.2f} cost3={metrics['cost3']['pnl']:.2f}", flush=True)
    best = max(experiments, key=lambda e: e["score"])
    alpha14 = experiments[0]["metrics"]
    blocking = list(audit_contract.get("blocking", []))
    warnings = list(audit_contract.get("warnings", []))
    warnings.append("execution_component_is_ohlcv_proxy_not_live_l2_orderbook")
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": "promote" if not blocking and best["name"] != "alpha1.4" and best["metrics"]["cost1"]["mdd"] > alpha14["cost1"]["mdd"] and best["metrics"]["cost1"]["pnl"] >= alpha14["cost1"]["pnl"] * 0.90 and best["metrics"]["cost2"]["pnl"] > 0.0 else "iterate",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS only after selection",
        "selected_config": asdict(selected),
        "parent_frozen": True,
        "v27_entry_frozen": True,
        "v21_2_model_frozen": True,
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Fast pass for local emergency adverse-flow exit on top of alpha1.4. No entry blocking; only selected post-entry loss acceleration cases can close early.",
        "experiments": experiments,
        "audit": audit,
        "artifacts": {"report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "grid": str(GRID_OUT)},
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "best": best["name"], "verdict": audit["verdict"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
