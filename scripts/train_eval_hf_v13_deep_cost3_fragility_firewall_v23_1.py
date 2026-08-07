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

from scripts import train_eval_hf_v13_deep_jackpot_sequence_verifier_v23 as v23
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _read
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig
from ensemble.fully_learned_governor_policy import predict_policy_frame


MODEL_ID = "hf_v13_deep_cost3_fragility_firewall_v23_1_20260511"
DEFAULT_OUT_DIR = v23.ROOT / "data/ensemble/supervised/hf_v13_deep_cost3_fragility_firewall_v23_1_20260511"
DEFAULT_REPORT = v23.ROOT / "data/ensemble/reports/hf_v13_deep_cost3_fragility_firewall_v23_1_20260511_summary.json"
DEFAULT_AUDIT = v23.ROOT / "data/ensemble/reports/hf_v13_deep_cost3_fragility_firewall_v23_1_20260511_audit.json"
DEFAULT_GRID = v23.ROOT / "data/ensemble/reports/hf_v13_deep_cost3_fragility_firewall_v23_1_20260511_grid.csv"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _grid() -> list[v23.VerifierConfig]:
    rows: list[v23.VerifierConfig] = []
    for fragile_th in (0.55, 0.65, 0.75):
        for cost3_floor in (-0.006, -0.003, 0.0):
            for reduce_frac in (0.0, 0.10, 0.15):
                rows.append(
                    v23.VerifierConfig(
                        name=f"v23_1_cost3_firewall_f{fragile_th:.2f}_c3{cost3_floor:.3f}_r{reduce_frac:.2f}",
                        fragile_th=fragile_th,
                        edge_th=cost3_floor,
                        reduce_frac=reduce_frac,
                        q10_floor=-0.006,
                    )
                )
    return rows


def _cost3_firewall_action(pred: dict[str, float], cfg: v23.VerifierConfig) -> tuple[str, float]:
    # Negative-only firewall: preserve V21.2 full add-on unless the deep head
    # sees a cost3-fragile add-on. This avoids the V23 MVP's alpha-killing bias.
    fragile = pred["p_cost3_fragile"] >= cfg.fragile_th and pred["delta_cost3"] <= cfg.edge_th
    if not fragile:
        return "full", 0.20
    if cfg.reduce_frac <= 1e-12:
        return "reject", 0.0
    return "reduce", cfg.reduce_frac


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1["trades"]) < 20:
        return -1e9 + float(c1["pnl"])
    cost3_bonus = max(0.0, float(c3["pnl"]) - v23.V21_2_COST3)
    return float(float(c1["pnl"]) + 0.42 * float(c2["pnl"]) + 0.25 * float(c3["pnl"]) + 0.35 * cost3_bonus - 0.20 * abs(float(c1["mdd"])))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V23.1 deep cost3 fragility firewall for V21.2 jackpot add-ons.")
    p.add_argument("--parent-model", type=Path, default=v23.DEFAULT_PARENT)
    p.add_argument("--jackpot-model", type=Path, default=v23.DEFAULT_JACKPOT)
    p.add_argument("--train-csv", type=Path, default=v23.DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=v23.DEFAULT_EVAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--grid-out", type=Path, default=DEFAULT_GRID)
    p.add_argument("--epochs", type=int, default=80)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bundle = joblib.load(args.parent_model)
    jackpot_payload = joblib.load(args.jackpot_model)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    base = dict(bundle["config"])
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    train = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    seq_cols = v23._select_seq_cols(train_all)
    feature_audit = _audit_contract(train_all, eval_df, list(bundle.get("feature_cols") or []))
    forbidden_cols = [c for c in seq_cols if any(tok in c.lower() for tok in v23.FORBIDDEN)]

    train_ds = v23._collect_snapshots(train, bundle, jackpot_model, add_cfg, seq_cols, fee=float(base["fee"]), slip=float(base["slip"]))
    norm = v23._normalizers(train_ds["seq"], train_ds["ctx"])
    verifier = v23._train_model(train_ds, norm, epochs=args.epochs)
    val_dec = predict_policy_frame(bundle, val, close=_close(val))
    eval_dec = predict_policy_frame(bundle, eval_df, close=_close(eval_df))

    old_action = v23._verifier_action
    v23._verifier_action = _cost3_firewall_action
    try:
        rows: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        for cfg in _grid():
            v1 = v23.backtest(val, bundle, jackpot_model, verifier, norm, add_cfg, cfg, seq_cols, fee=float(base["fee"]), slip=float(base["slip"]), decisions=val_dec, cost_mult=1.0)
            v2 = v23.backtest(val, bundle, jackpot_model, verifier, norm, add_cfg, cfg, seq_cols, fee=float(base["fee"]), slip=float(base["slip"]), decisions=val_dec, cost_mult=2.0)
            v3 = v23.backtest(val, bundle, jackpot_model, verifier, norm, add_cfg, cfg, seq_cols, fee=float(base["fee"]), slip=float(base["slip"]), decisions=val_dec, cost_mult=3.0)
            row = {"config": asdict(cfg), "validation_cost1": v1, "validation_cost2": v2, "validation_cost3": v3, "selection_score": _score(v1, v2, v3)}
            rows.append(row)
            if best is None or row["selection_score"] > best["selection_score"]:
                best = row
        selected = v23.VerifierConfig(**best["config"])
        metrics: dict[str, Any] = {}
        ledgers: dict[str, str] = {}
        for mult in (1, 2, 3):
            r = v23.backtest(eval_df, bundle, jackpot_model, verifier, norm, add_cfg, selected, seq_cols, fee=float(base["fee"]), slip=float(base["slip"]), decisions=eval_dec, cost_mult=float(mult), record=(mult == 1))
            if mult == 1:
                ledger = pd.DataFrame(r.pop("trade_records", []))
                lp = args.report_out.with_name(args.report_out.stem + "_cost1_ledger.csv")
                lp.parent.mkdir(parents=True, exist_ok=True)
                ledger.to_csv(lp, index=False)
                ledgers["cost1"] = str(lp)
            metrics[f"cost{mult}"] = r
    finally:
        v23._verifier_action = old_action

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "v23_1_deep_cost3_fragility_firewall.pt"
    torch.save({"model_id": MODEL_ID, "state_dict": verifier.state_dict(), "seq_cols": seq_cols, "ctx_cols": v23.CTX_COLS, "norm": norm, "selected_config": asdict(selected), "add_config": asdict(add_cfg), "parent_model": str(args.parent_model), "jackpot_model": str(args.jackpot_model)}, model_path)
    manifest_path = args.out_dir / "feature_manifest.json"
    manifest_path.write_text(json.dumps({"seq_cols": seq_cols, "ctx_cols": v23.CTX_COLS, "seq_len": v23.SEQ_LEN, "forbidden_cols": forbidden_cols}, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.grid_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{**{f"cfg_{k}": v for k, v in r["config"].items()}, "score": r["selection_score"], "val_pnl": r["validation_cost1"]["pnl"], "val_mdd": r["validation_cost1"]["mdd"], "val_trades": r["validation_cost1"]["trades"], "val_c2_pnl": r["validation_cost2"]["pnl"], "val_c3_pnl": r["validation_cost3"]["pnl"], "val_actions": json.dumps(r["validation_cost1"].get("runner_actions", {}), ensure_ascii=False)} for r in rows]).to_csv(args.grid_out, index=False)

    blocking: list[str] = []
    warnings: list[str] = []
    if feature_audit["status"] != "pass":
        blocking.extend(feature_audit["blocking"])
    if forbidden_cols:
        blocking.append(f"forbidden_sequence_columns={forbidden_cols}")
    warnings.extend(feature_audit.get("warnings", []))
    if metrics["cost1"]["pnl"] <= v23.V21_2_COST1:
        warnings.append("oos_cost1_did_not_beat_v21_2")
    if metrics["cost2"]["pnl"] <= 0.0:
        warnings.append("cost2_not_survived")
    if metrics["cost3"]["pnl"] <= 0.0:
        warnings.append("cost3_not_survived")
    verdict = "promote" if not blocking and metrics["cost1"]["pnl"] > v23.V21_2_COST1 and metrics["cost2"]["pnl"] > 0.0 and metrics["cost3"]["pnl"] > 0.0 else "iterate"
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": verdict,
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS only after selection",
        "policy": "negative_only_deep_cost3_firewall",
        "forbidden_sequence_columns": forbidden_cols,
        "train_snapshot_count": int(len(train_ds["target"])),
        "feature_audit": feature_audit,
        "selected_config": asdict(selected),
        "metrics": metrics,
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Negative-only deep cost3 fragility firewall. V21.2 full add-on is preserved by default; the deep head only reduces or rejects add-ons predicted to be cost3 fragile.",
        "parent_model": str(args.parent_model),
        "jackpot_model": str(args.jackpot_model),
        "model": str(model_path),
        "feature_manifest": str(manifest_path),
        "split_policy": "Verifier trained on 2025 Jan-Sep; firewall selected on 2025 Oct-Dec; 2026 fixed OOS after selection only.",
        "selected_config": asdict(selected),
        "selection_result": best,
        "metrics": metrics,
        "audit": audit,
        "artifacts": {"model": str(model_path), "manifest": str(manifest_path), "report": str(args.report_out), "audit": str(args.audit_out), "grid": str(args.grid_out), "ledgers": ledgers},
    }
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "model": str(model_path), "selected": asdict(selected), "metrics": metrics, "verdict": verdict}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
