#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import predict_policy_frame  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _read  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402


MODEL_ID = "hf_v13_v31_winner_addon_reallocator_v54_1_20260513"
DEFAULT_PARENT = v31.DEFAULT_PARENT
DEFAULT_JACKPOT = v31.DEFAULT_JACKPOT
DEFAULT_V27 = v31.DEFAULT_V27
DEFAULT_TRAIN = v31.DEFAULT_TRAIN
DEFAULT_EVAL = v31.DEFAULT_EVAL
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_v31_winner_addon_reallocator_v54_1_20260513"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_v31_winner_addon_reallocator_v54_1_20260513_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_v31_winner_addon_reallocator_v54_1_20260513_audit.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/hf_v13_v31_winner_addon_reallocator_v54_1_20260513_grid.csv"

V31_BASELINE = {
    "cost1": {"pnl": 277.0679629973942, "mdd": -31.74},
    "cost2": {"pnl": 112.79326141840412, "mdd": -31.46},
    "cost3": {"pnl": 20.933695032758784, "mdd": -43.09},
}


@dataclass(frozen=True)
class AddonConfig:
    name: str
    min_unrealized: float
    min_bars: int
    jackpot_p: float
    jackpot_q90: float
    bad_cap: float
    cost3_floor: float
    full_add_frac: float
    max_total_mult: float
    max_entry_notional: float
    dd_block: float
    deep_mult: float


def _configs() -> list[AddonConfig]:
    rows: list[AddonConfig] = []
    i = 0
    for min_u in (0.004, 0.0075, 0.010):
        for frac, total in ((0.30, 1.50), (0.45, 1.75), (0.60, 2.00), (0.80, 2.35)):
            for p, bad in ((0.20, 0.50), (0.25, 0.45), (0.35, 0.35)):
                rows.append(AddonConfig(f"v54_1_add{i}", min_u, 3, p, 0.015, bad, 0.40, frac, total, 4.14, 0.30, 1.0))
                i += 1
    for min_u in (0.0075, 0.010):
        rows.append(AddonConfig(f"v54_1_deepboost_{min_u:.4f}", min_u, 3, 0.25, 0.015, 0.45, 0.40, 0.45, 1.75, 4.14, 0.25, 1.15))
    rows.append(AddonConfig("v54_1_baseline", 0.004, 3, 0.20, 0.015, 0.50, 0.40, 0.20, 1.35, 2.75, 0.30, 1.0))
    return rows


def _overlay_with_deep_mult(overlay: v31.OverlayConfig, mult: float) -> v31.OverlayConfig:
    if mult == 1.0:
        return overlay
    return replace(
        overlay,
        name=f"{overlay.name}_deepx{mult:.2f}",
        notional=float(overlay.notional) * float(mult),
        base_tp=float(overlay.base_tp) * float(mult),
        base_sl=float(overlay.base_sl) * float(mult),
        tp_cap=float(overlay.tp_cap) * float(mult),
        sl_cap=float(overlay.sl_cap) * float(mult),
    )


def _addon_cfg(base: CostRunnerConfig, cfg: AddonConfig) -> CostRunnerConfig:
    return replace(
        base,
        name=f"{base.name}_{cfg.name}",
        jackpot_p=float(cfg.jackpot_p),
        jackpot_q90=float(cfg.jackpot_q90),
        bad_cap=float(cfg.bad_cap),
        min_unrealized=float(cfg.min_unrealized),
        min_bars_since_entry=int(cfg.min_bars),
        full_add_frac=float(cfg.full_add_frac),
        half_add_frac=0.0,
        max_total_mult=float(cfg.max_total_mult),
        max_entry_notional=float(cfg.max_entry_notional),
        dd_block=float(cfg.dd_block),
    )


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1.get("trades", 0)) < 20:
        return -1e9 + float(c1.get("pnl", 0.0))
    return float(c1["pnl"] + 0.35 * c2["pnl"] + 0.15 * c3["pnl"] - 0.25 * abs(c1["mdd"]) + 0.50 * c1.get("runner_actions", {}).get("v21_add_on", 0))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V54.1 V31 winner-only add-on reallocator.")
    p.add_argument("--parent-model", type=Path, default=DEFAULT_PARENT)
    p.add_argument("--jackpot-model", type=Path, default=DEFAULT_JACKPOT)
    p.add_argument("--v27-model", type=Path, default=DEFAULT_V27)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--grid-out", type=Path, default=DEFAULT_GRID)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print(f"[{MODEL_ID}] loading models/data", flush=True)
    parent = joblib.load(args.parent_model)
    jackpot_payload = joblib.load(args.jackpot_model)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg0 = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = v31._load_v27(args.v27_model)
    base = dict(parent["config"])
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    val = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    feature_audit = _audit_contract(train_all, eval_df, list(parent.get("feature_cols") or []))
    val_q = v31._predict_all(v27_model, val, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])
    val_dec = predict_policy_frame(parent, val, close=_close(val))
    eval_dec = predict_policy_frame(parent, eval_df, close=_close(eval_df))
    overlays = [o for o in v31._grid() if o.name in {"v31_notional1_time_decay", "v31_tight_after_24", "v31_ref", "v31_precision"}]
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for cfg in _configs():
        add_cfg = _addon_cfg(add_cfg0, cfg)
        for overlay0 in overlays:
            overlay = _overlay_with_deep_mult(overlay0, cfg.deep_mult)
            print(f"[{MODEL_ID}] validation cfg={cfg.name} overlay={overlay.name}", flush=True)
            v1 = v31.backtest(val, parent, jackpot_model, add_cfg, val_q, overlay, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=1.0, decisions=val_dec)
            v2 = v31.backtest(val, parent, jackpot_model, add_cfg, val_q, overlay, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=2.0, decisions=val_dec)
            v3 = v31.backtest(val, parent, jackpot_model, add_cfg, val_q, overlay, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=3.0, decisions=val_dec)
            row = {"config": asdict(cfg), "overlay": asdict(overlay), "validation_cost1": v1, "validation_cost2": v2, "validation_cost3": v3, "selection_score": _score(v1, v2, v3)}
            rows.append(row)
            if best is None or row["selection_score"] > best["selection_score"]:
                best = row
    assert best is not None
    selected_cfg = AddonConfig(**best["config"])
    selected_overlay = v31.OverlayConfig(**best["overlay"])
    selected_add = _addon_cfg(add_cfg0, selected_cfg)
    metrics: dict[str, Any] = {}
    ledgers: dict[str, str] = {}
    for mult in (1, 2, 3):
        r = v31.backtest(eval_df, parent, jackpot_model, selected_add, eval_q, selected_overlay, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=float(mult), decisions=eval_dec, record=(mult == 1))
        if mult == 1:
            ledger = pd.DataFrame(r.pop("trade_records", []))
            lp = args.report_out.with_name(args.report_out.stem + "_cost1_ledger.csv")
            lp.parent.mkdir(parents=True, exist_ok=True)
            ledger.to_csv(lp, index=False)
            ledgers["cost1"] = str(lp)
        metrics[f"cost{mult}"] = r
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "v54_1_winner_addon_reallocator_manifest.json"
    manifest_path.write_text(json.dumps({"model_id": MODEL_ID, "selected_config": asdict(selected_cfg), "selected_overlay": asdict(selected_overlay), "selected_add_config": asdict(selected_add)}, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    pd.DataFrame(
        [
            {
                **{f"cfg_{k}": v for k, v in r["config"].items()},
                "overlay_name": r["overlay"]["name"],
                "selection_score": r["selection_score"],
                "val_pnl": r["validation_cost1"]["pnl"],
                "val_mdd": r["validation_cost1"]["mdd"],
                "val_trades": r["validation_cost1"]["trades"],
                "val_addons": r["validation_cost1"].get("runner_actions", {}).get("v21_add_on", 0),
                "val_c2_pnl": r["validation_cost2"]["pnl"],
                "val_c3_pnl": r["validation_cost3"]["pnl"],
            }
            for r in rows
        ]
    ).to_csv(args.grid_out, index=False)
    blocking = list(feature_audit.get("blocking", []))
    warnings = list(feature_audit.get("warnings", []))
    if metrics["cost1"]["pnl"] <= V31_BASELINE["cost1"]["pnl"]:
        warnings.append("oos_cost1_did_not_beat_v31")
    if metrics["cost1"]["pnl"] <= 500:
        warnings.append("target_pnl_500_not_reached")
    if metrics["cost2"]["pnl"] <= 0:
        warnings.append("cost2_not_survived")
    if metrics["cost3"]["pnl"] <= -50:
        warnings.append("cost3_catastrophic")
    verdict = "promote" if not blocking and metrics["cost1"]["pnl"] > 500 and metrics["cost2"]["pnl"] > 0 else "iterate"
    audit = {"status": "pass" if not blocking else "fail", "verdict": verdict, "blocking": blocking, "warnings": warnings, "selection_uses_2026": False, "selection_window": "2025-10-01..2025-12-31", "oos_window": "2026 fixed OOS only after selection", "policy": "v31_winner_addon_reallocator_v54_1", "entry_owner_frozen": True, "initial_parent_notional_unchanged": True, "feature_audit": feature_audit, "selected_config": asdict(selected_cfg), "selected_overlay": asdict(selected_overlay), "metrics": metrics, "baseline_v31": V31_BASELINE}
    report = {"model_id": MODEL_ID, "design": "V54.1 keeps initial V31 entry sizing unchanged and only increases same-side V21.2 add-ons after a position is already profitable. This is the failure branch after V54 initial notional amplification worsened MDD.", "selected_config": asdict(selected_cfg), "selected_overlay": asdict(selected_overlay), "selection_result": best, "metrics": metrics, "audit": audit, "artifacts": {"manifest": str(manifest_path), "report": str(args.report_out), "audit": str(args.audit_out), "grid": str(args.grid_out), "ledgers": ledgers}}
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "manifest": str(manifest_path), "selected_config": asdict(selected_cfg), "selected_overlay": asdict(selected_overlay), "metrics": metrics, "verdict": verdict}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
