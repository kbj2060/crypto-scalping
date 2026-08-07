#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "omega2_1_label_atr1_h24_hgb_12seed_ensemble_thr055"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / "omega2_1_hgb_12seed_redteam_audit_20260614"
MANIFEST = ROOT / "data/ensemble/supervised" / MODEL_ID / "candidate_manifest.json"
BUNDLE = ROOT / "data/ensemble/supervised" / MODEL_ID / "omega2_1_hgb_12seed_cash_sleeve.joblib"
FREEZE_REPORT = ROOT / "tmp/causal_regen_20260516" / "omega2_1_cash_sleeve_freeze_verify_20260609" / "report.json"
ACCOUNTING_AUDIT = ROOT / "tmp/causal_regen_20260516" / "omega2_1_hgb_scale25_levexp_accounting_audit_20260609" / "report.json"
CONTRACT = ROOT / "docs/model_contracts/omega2_1_hgb_12seed_cash_sleeve_20260609_contract.md"

FORBIDDEN_PREFIXES = ("clean_regime4_", "regime4_pred_", "teacher_", "exit_head_")
FORBIDDEN_EXACT = {"tp_sl_action_score"}


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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _forbidden(cols: list[str]) -> list[str]:
    return [
        str(c)
        for c in cols
        if str(c) in FORBIDDEN_EXACT or any(str(c).startswith(prefix) for prefix in FORBIDDEN_PREFIXES)
    ]


def _find_corrected_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    text = json.dumps(payload, ensure_ascii=False)
    # Keep the extraction deliberately fail-fast: if this report format changes,
    # the audit should be updated rather than guessing.
    if "33.87790064270535" not in text:
        raise RuntimeError("corrected Omega2.1 true-leverage metric not found in accounting audit report")

    def walk(x: Any) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if isinstance(x, dict):
            if "pnl" in x and "mdd" in x and "trades" in x:
                out.append(x)
            for v in x.values():
                out.extend(walk(v))
        elif isinstance(x, list):
            for v in x:
                out.extend(walk(v))
        return out

    candidates = walk(payload)
    target = [c for c in candidates if abs(float(c.get("pnl", 0.0)) - 33.87790064270535) < 1e-9]
    if not target:
        raise RuntimeError("corrected Omega2.1 metric block missing")
    return target[0]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _read_json(MANIFEST)
    freeze = _read_json(FREEZE_REPORT)
    accounting = _read_json(ACCOUNTING_AUDIT)
    bundle = joblib.load(BUNDLE)

    manifest_features = list(manifest["feature_cols"])
    bundle_features = list(bundle["feature_cols"])
    if manifest_features != bundle_features:
        raise RuntimeError("manifest/bundle feature contract mismatch")
    if str(bundle.get("model_id")) != MODEL_ID:
        raise RuntimeError(f"bundle model id mismatch: {bundle.get('model_id')}")
    if _forbidden(bundle_features):
        raise RuntimeError(f"forbidden bundle feature columns: {_forbidden(bundle_features)}")

    corrected = _find_corrected_metrics(accounting)
    legacy = manifest["metrics"]["oos_full_train_parity"]
    contract_text = CONTRACT.read_text(encoding="utf-8")
    contract_deprecated = "deprecated_historical_reference_only_accounting_invalid_true_leverage" in contract_text

    report = {
        "audit_id": "omega2_1_hgb_12seed_redteam_audit_20260614",
        "model_id": MODEL_ID,
        "verdict": "deprecated_historical_reference_only_accounting_invalid_true_leverage",
        "summary": {
            "manifest_bundle_feature_match": True,
            "forbidden_feature_contract": "pass",
            "direct_feature_forbidden_count": 0,
            "contract_already_deprecated": bool(contract_deprecated),
            "live_promotable": False,
        },
        "legacy_reported_oos": {
            "pnl_pct": float(legacy["oos_pnl"]),
            "mdd_pct": float(legacy["oos_mdd"]),
            "wr": float(legacy["oos_wr"]),
            "trades": int(legacy["oos_trades"]),
            "fallback_entries": int(legacy.get("oos_fallback_entries", 0)),
            "primary_takeovers": int(legacy.get("oos_primary_takeovers", 0)),
            "accounting": "invalid_for_true_leverage_promotion_notional_exposure_only",
        },
        "corrected_true_leverage_oos": {
            "pnl_pct": float(corrected["pnl"]),
            "mdd_pct": float(corrected["mdd"]),
            "wr": float(corrected.get("wr", corrected.get("win_rate", 0.0))),
            "trades": int(corrected["trades"]),
            "accounting": "effective_exposure_equals_notional_times_leverage",
        },
        "reason": [
            "Fallback risk stores notional=0.30 and leverage=2.0, but legacy replay accounted PnL/fees/MDD on notional only.",
            "Current Omega accounting contract requires effective_exposure = notional * leverage.",
            "Under corrected true-leverage accounting, headline OOS changed from +102.61% / MDD -8.11% to +33.88% / MDD -23.98%.",
            "Therefore the artifact is historical reference only unless rebuilt and re-evaluated under true-leverage accounting.",
        ],
        "source_artifacts": {
            "manifest": str(MANIFEST),
            "bundle": str(BUNDLE),
            "freeze_report": str(FREEZE_REPORT),
            "accounting_audit": str(ACCOUNTING_AUDIT),
            "contract": str(CONTRACT),
        },
        "freeze_report_snapshot": {
            "oos_full_train_parity": freeze.get("oos_full_train_parity"),
            "reference_selection": freeze.get("reference_selection"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(OUT_DIR / "report.json"),
        "verdict": report["verdict"],
        "legacy_oos": report["legacy_reported_oos"],
        "corrected_oos": report["corrected_true_leverage_oos"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
