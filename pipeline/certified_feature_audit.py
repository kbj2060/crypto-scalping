from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

AI_CONTRACT_DIRS = (
    ROOT / "data/nf_patchtst",
    ROOT / "data/nf_tide",
    ROOT / "data/nf_dlinear",
    ROOT / "data/nf_timesnet",
)

LEGACY_REGIME_EXACT = {
    "regime_bull",
    "regime_bear",
    "regime_chop",
    "regime_whipsaw",
    "regime_normal",
    "regime_trending",
    "regime_break",
    "cvp_regime",
    "m7_hdb_label",
    "m7_hdb_prob",
}

FORBIDDEN_FRAGMENTS = (
    "legacy",
    "regime_v2",
    "hdbscan",
    "hmm_",
    "future",
    "target",
    "label",
    "realized",
    "trade_pnl",
    "cash_after",
)


def audit_ai_contracts(contract_dirs: tuple[Path, ...] = AI_CONTRACT_DIRS) -> dict[str, Any]:
    blocking: list[str] = []
    evidence: dict[str, Any] = {}
    for directory in contract_dirs:
        path = directory / "specialist_contract.json"
        key = str(directory.relative_to(ROOT))
        if not path.exists():
            blocking.append(f"missing_ai_contract:{key}")
            continue
        obj = json.loads(path.read_text(encoding="utf-8"))
        prov = obj.get("provenance", {})
        row = {
            "certified": bool(obj.get("artifact_training_provenance_certified")),
            "data_path": prov.get("data_path"),
            "expected_year": prov.get("expected_year"),
            "actual_years": prov.get("actual_years"),
            "timestamp_start": prov.get("timestamp_start"),
            "timestamp_end": prov.get("timestamp_end"),
            "rows_after_limit": prov.get("rows_after_limit"),
        }
        evidence[key] = row
        if not row["certified"]:
            blocking.append(f"ai_not_certified:{key}")
        if row["expected_year"] != 2024 or row["actual_years"] != [2024]:
            blocking.append(f"ai_not_2024_only:{key}")
    return {"status": "pass" if not blocking else "fail", "blocking": blocking, "evidence": evidence}


def forbidden_columns(columns: list[str], *, clean_prefix: str) -> list[str]:
    bad: list[str] = []
    for col in columns:
        lower = col.lower()
        if lower.startswith(clean_prefix):
            continue
        if col in LEGACY_REGIME_EXACT:
            bad.append(col)
            continue
        if any(fragment in lower for fragment in FORBIDDEN_FRAGMENTS):
            bad.append(col)
    return sorted(set(bad))


def audit_frame_contract(
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    clean_prefix: str,
    require_m7: bool = True,
    require_ai: bool = True,
) -> dict[str, Any]:
    blocking: list[str] = []
    warnings: list[str] = []
    missing = [c for c in feature_cols if c not in frame.columns]
    if missing:
        blocking.append("missing_feature_cols:" + ",".join(missing[:30]))
    bad = forbidden_columns(feature_cols, clean_prefix=clean_prefix)
    if bad:
        blocking.append("forbidden_model_feature_cols:" + ",".join(bad[:30]))
    if require_m7 and not any(c.startswith("m7_") for c in feature_cols):
        blocking.append("missing_m7_features")
    if require_ai and not any(c.startswith("ai_") or c.startswith("patchtst_") or c in {"pred_patchtst", "conf_patchtst"} for c in feature_cols):
        blocking.append("missing_ai_features")
    if not any(c.startswith(clean_prefix) for c in feature_cols):
        blocking.append("missing_clean_regime_features")
    ts = pd.to_datetime(frame.get("timestamp"), errors="coerce")
    if ts.isna().any():
        blocking.append("timestamp_nulls_present")
    if ts.duplicated().any():
        blocking.append("timestamp_duplicates_present")
    if not ts.is_monotonic_increasing:
        warnings.append("timestamps_not_monotonic_increasing")
    return {
        "status": "pass" if not blocking else "fail",
        "blocking": blocking,
        "warnings": warnings,
        "feature_count": len(feature_cols),
        "forbidden_feature_cols": bad,
    }

