#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _read, backtest_policy_frame  # noqa: E402


MODEL_ID = "hf_v13_clean_regime_validation_selected_exposure_20260511"
DEFAULT_BASE_MODEL = ROOT / "data/ensemble/supervised/hf_v13_clean_regime_repro_20260511/v13_clean_regime_h288.pkl"
DEFAULT_TRAIN = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_clean_regime_validation_selected_exposure_20260511"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_clean_regime_validation_selected_exposure_20260511_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_clean_regime_validation_selected_exposure_20260511_audit.json"


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


def _candidate_grid() -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for notional_scale in (0.85, 1.00, 1.05, 1.10, 1.15, 1.20):
        for margin_cap in (1.00, 1.05, 1.10):
            rows.append({"notional_bucket_scale": float(notional_scale), "max_margin_fraction": float(margin_cap)})
    return rows


def _apply_overlay(bundle: dict[str, Any], overlay: dict[str, float]) -> dict[str, Any]:
    out = copy.deepcopy(bundle)
    cfg = dict(out.get("config", {}))
    base_buckets = list(cfg.get("notional_buckets", []))
    # The base model may already be unscaled; always scale from its own stored buckets.
    cfg["notional_buckets"] = [float(x) * float(overlay["notional_bucket_scale"]) for x in base_buckets]
    cfg["max_margin_fraction"] = float(overlay["max_margin_fraction"])
    out["config"] = cfg
    out["model_id"] = MODEL_ID
    out["base_model"] = out.get("base_model") or "base_v13_clean_regime_h288"
    out["overlay"] = dict(overlay)
    out["selection_policy"] = "overlay selected on 2025-10-01..2025-12-31 validation only; 2026 never used for selection"
    return out


def _score(cost1: dict[str, Any], cost2: dict[str, Any], cost3: dict[str, Any]) -> float:
    trades = int(cost1.get("trades", 0))
    if trades < 20:
        return -1e9 + float(cost1.get("pnl", -9999))
    pnl = float(cost1["pnl"])
    mdd = abs(float(cost1["mdd"]))
    cost2_pnl = float(cost2["pnl"])
    cost3_pnl = float(cost3["pnl"])
    tpd = float(cost1["trades_per_day"])
    penalty = 0.0
    if cost2_pnl < 0.0:
        penalty += abs(cost2_pnl) * 1.5
    if cost3_pnl < 0.0:
        penalty += abs(cost3_pnl) * 2.0
    return float(pnl + 0.30 * cost2_pnl + 0.15 * cost3_pnl - 1.20 * mdd + 0.20 * min(tpd, 4.0) - penalty)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select v13 clean-regime exposure overlay on 2025 validation only, then evaluate fixed 2026 OOS.")
    p.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    base_bundle = joblib.load(args.base_model)
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    validation = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    feature_cols = list(base_bundle.get("feature_cols") or [])
    feature_audit = _audit_contract(train_all, eval_df, feature_cols)

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for overlay in _candidate_grid():
        bundle = _apply_overlay(base_bundle, overlay)
        cfg = dict(bundle["config"])
        v1 = backtest_policy_frame(validation, bundle, fee=float(cfg["fee"]), slip=float(cfg["slip"]))
        v2 = backtest_policy_frame(validation, bundle, fee=float(cfg["fee"]) * 2.0, slip=float(cfg["slip"]) * 2.0)
        v3 = backtest_policy_frame(validation, bundle, fee=float(cfg["fee"]) * 3.0, slip=float(cfg["slip"]) * 3.0)
        row = {
            "overlay": overlay,
            "validation_cost1": v1,
            "validation_cost2": v2,
            "validation_cost3": v3,
            "selection_score": _score(v1, v2, v3),
        }
        rows.append(row)
        if best is None or row["selection_score"] > best["selection_score"]:
            best = row
    if best is None:
        raise RuntimeError("no selected overlay")

    selected_bundle = _apply_overlay(base_bundle, best["overlay"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "v13_clean_regime_validation_selected_exposure.pkl"
    joblib.dump(selected_bundle, model_path)
    cfg = dict(selected_bundle["config"])
    metrics: dict[str, Any] = {}
    ledgers: dict[str, str] = {}
    for mult in (1, 2, 3):
        result = backtest_policy_frame(eval_df, selected_bundle, fee=float(cfg["fee"]) * mult, slip=float(cfg["slip"]) * mult, record_trades=(mult == 1))
        if mult == 1:
            ledger = pd.DataFrame(result.pop("trade_records", []))
            ledger_path = args.report_out.with_name(args.report_out.stem + "_cost1_ledger.csv")
            ledger_path.parent.mkdir(parents=True, exist_ok=True)
            ledger.to_csv(ledger_path, index=False)
            ledgers["cost1"] = str(ledger_path)
        metrics[f"cost{mult}"] = result

    grid_path = args.report_out.with_name(args.report_out.stem + "_validation_grid.json")
    grid_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    blocking = []
    warnings = []
    if feature_audit["status"] != "pass":
        blocking.extend(feature_audit["blocking"])
    warnings.extend(feature_audit.get("warnings", []))
    if int(feature_audit.get("train_eval_timestamp_overlap", 0)) != 0:
        blocking.append("train_eval_timestamp_overlap")
    if metrics["cost1"]["pnl"] < 100.0:
        warnings.append("oos_cost1_below_target_100pct")
    if metrics["cost2"]["pnl"] < 0.0 or metrics["cost3"]["pnl"] < 0.0:
        warnings.append("cost_stress_survival_incomplete")

    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": "promote_candidate" if not blocking and metrics["cost1"]["pnl"] >= 100.0 and metrics["cost2"]["pnl"] > 0.0 and metrics["cost3"]["pnl"] > 0.0 else "iterate",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed, used only after overlay selection",
        "selected_overlay": best["overlay"],
        "feature_audit": feature_audit,
        "metrics": metrics,
    }
    report = {
        "model_id": MODEL_ID,
        "design": "V13 clean-regime HF core with exposure overlay selected only on 2025 validation, then fixed 2026 OOS evaluation.",
        "base_model": str(args.base_model),
        "model": str(model_path),
        "train_csv": str(args.train_csv),
        "eval_csv": str(args.eval_csv),
        "split_policy": "Base core trained on 2025 Jan-Sep; overlay selected on 2025 Oct-Dec; 2026 fixed OOS not used for selection.",
        "selected_overlay": best["overlay"],
        "selection_score": best["selection_score"],
        "selection_result": {k: v for k, v in best.items() if k != "selection_score"},
        "metrics": metrics,
        "audit": audit,
        "artifacts": {
            "model": str(model_path),
            "report": str(args.report_out),
            "audit": str(args.audit_out),
            "validation_grid": str(grid_path),
            "ledgers": ledgers,
        },
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "model": str(model_path), "selected_overlay": best["overlay"], "metrics": metrics, "verdict": audit["verdict"]}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
