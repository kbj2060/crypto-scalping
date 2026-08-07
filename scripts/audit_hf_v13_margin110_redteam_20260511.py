#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

from scripts.train_eval_hf_clean_regime_core_loop_20260511 import (  # noqa: E402
    CLEAN_PREFIX,
    _audit_contract,
    _read,
    backtest_policy_frame,
)


DEFAULT_MODEL = ROOT / "data/ensemble/supervised/hf_v13_clean_regime_margin110_20260511/v13_clean_regime_margin110.pkl"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_clean_regime_margin110_20260511_summary.json"
DEFAULT_LEDGER = ROOT / "data/ensemble/reports/hf_v13_clean_regime_margin110_20260511_summary_cost1_ledger.csv"
DEFAULT_OUT = ROOT / "data/ensemble/reports/hf_v13_clean_regime_margin110_20260511_redteam_full_audit.json"
DEFAULT_TRAIN = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"


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


def _metric_delta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    keys = ("pnl", "mdd", "trades", "trades_per_day", "avg_notional", "avg_leverage")
    out = {}
    for key in keys:
        if key in a and key in b:
            out[key] = float(a[key]) - float(b[key])
    return out


def _same_bar_delta_minutes(a: pd.Series, b: pd.Series) -> pd.Series:
    return (pd.to_datetime(a, errors="coerce") - pd.to_datetime(b, errors="coerce")).dt.total_seconds() / 60.0


def _audit_ledger_prices(ledger: pd.DataFrame, eval_df: pd.DataFrame, fee: float, slip: float) -> dict[str, Any]:
    blocking: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}
    if ledger.empty:
        return {"status": "fail", "blocking": ["empty_ledger"], "warnings": [], "checks": {}}
    idx = eval_df.set_index("timestamp")
    work = ledger.copy()
    for col in ("entry_signal_timestamp", "entry_fill_timestamp", "exit_signal_timestamp", "exit_fill_timestamp"):
        work[col] = pd.to_datetime(work[col], errors="coerce")
    entry_delta = _same_bar_delta_minutes(work["entry_fill_timestamp"], work["entry_signal_timestamp"])
    exit_delta = _same_bar_delta_minutes(work["exit_fill_timestamp"], work["exit_signal_timestamp"])
    bad_entry_delta = work.loc[entry_delta != 5.0]
    bad_exit_delta = work.loc[exit_delta != 5.0]
    if not bad_entry_delta.empty:
        blocking.append(f"entry_fill_not_next_5m:{len(bad_entry_delta)}")
    if not bad_exit_delta.empty:
        blocking.append(f"exit_fill_not_next_5m:{len(bad_exit_delta)}")

    entry_price_errors = []
    for row in work.itertuples(index=False):
        try:
            open_px = float(idx.loc[row.entry_fill_timestamp, "open"])
        except Exception:
            entry_price_errors.append({"time": str(row.entry_fill_timestamp), "reason": "missing_eval_open"})
            continue
        side = 1 if row.side == "LONG" else -1
        expected = open_px * (1.0 + slip if side > 0 else 1.0 - slip)
        if abs(float(row.entry_price) - expected) > max(1e-8, abs(expected) * 1e-9):
            entry_price_errors.append({"time": str(row.entry_fill_timestamp), "ledger": float(row.entry_price), "expected": expected})
    if entry_price_errors:
        blocking.append(f"entry_price_not_next_open_with_slip:{len(entry_price_errors)}")
    checks["entry_price_error_sample"] = entry_price_errors[:5]
    checks["entry_fill_delta_min_counts"] = entry_delta.value_counts(dropna=False).to_dict()
    checks["exit_fill_delta_min_counts"] = exit_delta.value_counts(dropna=False).to_dict()
    checks["max_position_fraction"] = float(pd.to_numeric(work["position_fraction"], errors="coerce").max())
    checks["rows_position_fraction_gt_1"] = int((pd.to_numeric(work["position_fraction"], errors="coerce") > 1.0 + 1e-12).sum())
    checks["entry_fee_pct_matches"] = bool(np.allclose(
        pd.to_numeric(work["fee_entry_pct"], errors="coerce").to_numpy(float),
        pd.to_numeric(work["notional_exposure"], errors="coerce").to_numpy(float) * fee * 100.0,
        rtol=1e-10,
        atol=1e-10,
    ))
    checks["exit_fee_pct_matches"] = bool(np.allclose(
        pd.to_numeric(work["fee_exit_pct"], errors="coerce").to_numpy(float),
        pd.to_numeric(work["notional_exposure"], errors="coerce").to_numpy(float) * fee * 100.0,
        rtol=1e-10,
        atol=1e-10,
    ))
    if not checks["entry_fee_pct_matches"]:
        blocking.append("entry_fee_pct_mismatch")
    if not checks["exit_fee_pct_matches"]:
        blocking.append("exit_fee_pct_mismatch")
    if checks["rows_position_fraction_gt_1"] > 0:
        warnings.append("position_fraction_above_1_intentional_margin_overlay")
    return {"status": "pass" if not blocking else "fail", "blocking": blocking, "warnings": warnings, "checks": checks}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full red-team audit for hf_v13 clean regime margin110 candidate.")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    bundle = joblib.load(args.model)
    train = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    ledger = pd.read_csv(args.ledger) if args.ledger.exists() else pd.DataFrame()
    cfg = dict(bundle.get("config", {}))
    fee = float(cfg.get("fee", 0.0005))
    slip = float(cfg.get("slip", 0.0002))
    feature_cols = list(bundle.get("feature_cols") or [])

    blocking: list[str] = []
    warnings: list[str] = []
    feature_audit = _audit_contract(train, eval_df, feature_cols)
    if feature_audit["status"] != "pass":
        blocking.extend(feature_audit["blocking"])
    if feature_audit.get("warnings"):
        warnings.extend(feature_audit["warnings"])

    recomputed: dict[str, Any] = {}
    metric_deltas: dict[str, Any] = {}
    for mult in (1, 2, 3):
        key = f"cost{mult}"
        r = backtest_policy_frame(eval_df, bundle, fee=fee * mult, slip=slip * mult, record_trades=(mult == 1))
        r.pop("trade_records", None)
        recomputed[key] = r
        metric_deltas[key] = _metric_delta(r, report["metrics"][key])
        if any(abs(v) > 1e-8 for v in metric_deltas[key].values()):
            blocking.append(f"{key}_report_recompute_mismatch")

    ledger_audit = _audit_ledger_prices(ledger, eval_df, fee, slip)
    if ledger_audit["status"] != "pass":
        blocking.extend(ledger_audit["blocking"])
    warnings.extend(ledger_audit["warnings"])

    report_audit = report.get("audit", {}) if isinstance(report.get("audit"), dict) else {}
    selection_uses_2026 = report_audit.get("selection_uses_2026")
    if selection_uses_2026 is None:
        selection_policy = str(bundle.get("selection_policy") or report.get("split_policy") or "")
        selection_uses_2026 = "2026 never used for selection" not in selection_policy and "2026 fixed OOS not used for selection" not in selection_policy

    leakage_checks = {
        "train_eval_timestamp_overlap": feature_audit["train_eval_timestamp_overlap"],
        "train_range": feature_audit["train_range"],
        "eval_range": feature_audit["eval_range"],
        "model_train_csv": bundle.get("train_csv"),
        "model_eval_csv": bundle.get("eval_csv"),
        "report_split_policy": report.get("split_policy"),
        "selection_uses_2026": bool(selection_uses_2026),
        "selection_window": report_audit.get("selection_window"),
        "oos_window": report_audit.get("oos_window"),
    }
    if bool(selection_uses_2026):
        blocking.append("oos_parameter_selection_leak:overlay_selected_on_2026_results")

    regime_cols = [c for c in feature_cols if "regime" in c.lower()]
    non_clean_regime_cols = [c for c in regime_cols if not c.startswith(CLEAN_PREFIX) and not c.startswith(("patchtst_", "ai_"))]
    if non_clean_regime_cols:
        blocking.append("non_clean_regime_feature_cols:" + ",".join(non_clean_regime_cols[:20]))

    audit = {
        "status": "fail" if blocking else "pass",
        "verdict": "reject_until_reselected_without_2026_oos" if blocking else "pass",
        "blocking": sorted(set(blocking)),
        "warnings": sorted(set(warnings)),
        "model": str(args.model),
        "report": str(args.report),
        "ledger": str(args.ledger),
        "feature_audit": feature_audit,
        "ledger_audit": ledger_audit,
        "leakage_checks": leakage_checks,
        "recomputed_metrics": recomputed,
        "reported_metrics": report.get("metrics", {}),
        "metric_deltas": metric_deltas,
        "feature_contract": {
            "feature_count": len(feature_cols),
            "clean_regime_feature_count": len([c for c in feature_cols if c.startswith(CLEAN_PREFIX)]),
            "regime_named_feature_cols": regime_cols,
            "non_clean_regime_cols": non_clean_regime_cols,
        },
        "accounting_notes": [
            "Entry fee is charged at entry as cash * fee * notional.",
            "Exit fee is charged at exit as pre-exit cash * fee * notional.",
            "Entry/exit slippage is applied to fill prices.",
            "Stops and take-profits are evaluated on notional-scaled unrealized return, matching the HF labeler contract.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "status": audit["status"], "verdict": audit["verdict"], "blocking": audit["blocking"], "warnings": audit["warnings"]}, ensure_ascii=False))
    return 1 if audit["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
