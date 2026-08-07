#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data/ensemble/reports"
AUDIT_OUT = OUT_DIR / "m7_teacher_live_provenance_20260527_audit.json"
LIVE_OUT = OUT_DIR / "m7_teacher_live_candidate_20260527.json"

M7_FILES = {
    "m7_2025": ROOT / "data/splits/year_oos/rl_training_2025_m7.csv",
    "m7_2026": ROOT / "data/splits/year_oos/rl_training_2026_m7_supervised_redesign_clean.csv",
}

MODEL_META = {
    "trend_xgb": ROOT / "data/ensemble/supervised/trend_xgb.json",
    "entry_price_model": ROOT / "data/ensemble/supervised/entry_price_model.json",
    "multi_target_lgbm": ROOT / "data/ensemble/supervised/multi_target_lgbm.json",
    "quantile_forest": ROOT / "data/ensemble/supervised/quantile_forest.json",
    "gmm_volatility": ROOT / "data/ensemble/unsupervised/gmm_volatility.json",
    "isolation_forest": ROOT / "data/ensemble/unsupervised/isolation_forest.json",
    "vae_anomaly": ROOT / "data/ensemble/unsupervised/vae_anomaly.json",
}

PIPELINE_FILES = {
    "year_split_train": ROOT / "scripts/pipeline_year_split_train.py",
    "augment_m7_dataset": ROOT / "pipeline/augment_m7_dataset.py",
    "seven_model_ensemble": ROOT / "ensemble/seven_model_ensemble.py",
    "teacher_features": ROOT / "pipeline/teacher_meta_side_features.py",
    "certified_teacher_moe": ROOT / "ensemble/certified_teacher_regime_moe.py",
    "teacher_builder_2025": ROOT / "scripts/build_certified_teacher_features_2025.py",
    "teacher_builder_2026": ROOT / "scripts/build_certified_teacher_features_2026.py",
}

OPERATOR_ATTESTATION = {
    "ai_features": "ai_*, pred_patchtst, conf_patchtst were trained on 2024 and scored on 2025.",
    "m7_features": "m7_* were trained on 2024 and scored on 2025; 2026 uses transform/score artifacts without 2026 fitting.",
    "recorded_from": "user confirmation in active Codex thread on 2026-05-27",
}

TEACHER_UPSTREAM_COLS = {
    "ai_dir_p_up",
    "ai_dir_p_down",
    "pred_patchtst",
    "conf_patchtst",
    "m7_trend_xgb_up",
    "m7_trend_xgb_dn",
    "m7_prob_up",
    "m7_prob_dn",
    "m7_confidence",
    "m7_q10",
    "m7_q50",
    "m7_q90",
    "ai_adverse_risk",
    "m7_tail_risk",
}


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _inspect_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        if "timestamp" not in header:
            return {"exists": True, "sha256": _sha256(path), "error": "missing timestamp", "columns": len(header)}
        rows = 0
        first_ts = None
        last_ts = None
        previous_ts = None
        monotonic = True
        duplicate_count = 0
        seen: set[str] = set()
        for row in reader:
            ts = str(row.get("timestamp", "")).strip()
            if rows == 0:
                first_ts = ts
            if previous_ts is not None and ts <= previous_ts:
                monotonic = False
            if ts in seen:
                duplicate_count += 1
            seen.add(ts)
            previous_ts = ts
            last_ts = ts
            rows += 1
    m7_cols = [c for c in header if c.startswith("m7_")]
    teacher_inputs_present = sorted(c for c in header if c in TEACHER_UPSTREAM_COLS)
    target_named_m7 = sorted(c for c in m7_cols if "target" in c.lower())
    return {
        "exists": True,
        "path": _rel(path),
        "sha256": _sha256(path),
        "rows": rows,
        "columns": len(header),
        "timestamp_start": first_ts,
        "timestamp_end": last_ts,
        "timestamp_monotonic_strict": monotonic,
        "timestamp_duplicate_count": duplicate_count,
        "m7_column_count": len(m7_cols),
        "m7_columns": m7_cols,
        "teacher_upstream_columns_present": teacher_inputs_present,
        "target_named_m7_outputs": target_named_m7,
    }


def _inspect_model_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    obj = json.loads(path.read_text(encoding="utf-8"))
    model_ref = obj.get("model_path")
    model_path = (path.parent / model_ref).resolve() if model_ref else None
    feature_cols = obj.get("feature_cols") or []
    return {
        "exists": True,
        "path": _rel(path),
        "model_path": _rel(model_path) if model_path else None,
        "model_exists": bool(model_path and model_path.exists()),
        "feature_count": len(feature_cols) if isinstance(feature_cols, list) else None,
        "embedded_provenance_present": bool(obj.get("provenance") or obj.get("artifact_training_provenance_certified")),
    }


def _pipeline_checks() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    year_split = _read_text(PIPELINE_FILES["year_split_train"])
    augment = _read_text(PIPELINE_FILES["augment_m7_dataset"])
    teacher_moe = _read_text(PIPELINE_FILES["certified_teacher_moe"])
    teacher_features = _read_text(PIPELINE_FILES["teacher_features"])

    checks["year_split_train_defaults"] = {
        "sup_year_2024_default": "--sup-year" in year_split and "default=2024" in year_split,
        "rl_year_2025_default": "--rl-year" in year_split and "default=2025" in year_split,
        "trains_on_sup_split_before_augment": "feat_sup_path" in year_split
        and "rl_sup_path" in year_split
        and "train_all_ensemble" in year_split,
        "augments_rl_year_split": "feat_rl_path" in year_split
        and "rl_base_path" in year_split
        and "pipeline/augment_m7_dataset.py" in year_split,
    }
    checks["augment_m7_merge"] = {
        "uses_exact_timestamp_merge": ".merge(" in augment and "on=timestamp_col" in augment and "how=\"left\"" in augment,
        "uses_merge_asof": "merge_asof" in augment,
        "uses_bfill": ".bfill" in augment or "bfill(" in augment,
        "uses_future_shift_minus": "shift(-" in augment,
    }
    checks["teacher_merge"] = {
        "uses_exact_timestamp_merge": "out.merge(add[[\"timestamp\"] + cols], on=\"timestamp\", how=\"left\"" in teacher_moe,
        "uses_merge_asof": "merge_asof" in teacher_moe,
        "uses_bfill": ".bfill" in teacher_moe or "bfill(" in teacher_moe,
    }
    teacher_feature_body = "\n".join(
        line for line in teacher_features.splitlines() if not line.strip().startswith("from __future__")
    ).lower()
    checks["teacher_feature_formula"] = {
        "deterministic_from_ai_m7": all(token in teacher_features for token in ("ai_dir_p_up", "m7_trend_xgb_up", "m7_q90")),
        "direct_future_or_label_terms": [
            token
            for token in ("future", "label", "realized", "pnl", "forward", "target")
            if token in teacher_feature_body
        ],
    }
    return checks


def _status(audit: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    blocking: list[str] = []
    warnings: list[str] = []

    for key, row in audit["m7_files"].items():
        if not row.get("exists"):
            blocking.append(f"missing_m7_file:{key}")
            continue
        if not row.get("timestamp_monotonic_strict"):
            blocking.append(f"m7_timestamps_not_strictly_monotonic:{key}")
        if int(row.get("timestamp_duplicate_count") or 0) > 0:
            blocking.append(f"m7_timestamp_duplicates:{key}")
        if int(row.get("m7_column_count") or 0) == 0:
            blocking.append(f"missing_m7_columns:{key}")
        if key == "m7_2025" and not str(row.get("timestamp_start", "")).startswith("2025-"):
            blocking.append("m7_2025_start_not_2025")
        if key == "m7_2026" and not str(row.get("timestamp_start", "")).startswith("2026-"):
            blocking.append("m7_2026_start_not_2026")
        if row.get("target_named_m7_outputs"):
            warnings.append(f"{key}: target-named m7 outputs are model predictions, not direct labels: {row['target_named_m7_outputs']}")

    for key, row in audit["model_meta"].items():
        if not row.get("exists"):
            blocking.append(f"missing_model_meta:{key}")
            continue
        if not row.get("model_exists"):
            blocking.append(f"missing_model_binary:{key}")
        if not row.get("embedded_provenance_present"):
            warnings.append(f"{key}: embedded model provenance absent; covered by external audit + operator attestation")

    pipe = audit["pipeline_checks"]
    if not all(pipe["year_split_train_defaults"].values()):
        blocking.append("year_split_train_contract_incomplete")
    if pipe["augment_m7_merge"]["uses_merge_asof"] or pipe["augment_m7_merge"]["uses_bfill"] or pipe["augment_m7_merge"]["uses_future_shift_minus"]:
        blocking.append("augment_m7_has_forward_fill_or_future_join_pattern")
    if pipe["teacher_merge"]["uses_merge_asof"] or pipe["teacher_merge"]["uses_bfill"]:
        blocking.append("teacher_merge_has_forward_join_pattern")
    if pipe["teacher_feature_formula"]["direct_future_or_label_terms"]:
        blocking.append("teacher_formula_contains_forbidden_future_or_label_terms")

    status = "pass" if not blocking else "fail"
    return status, blocking, warnings


def main() -> int:
    audit: dict[str, Any] = {
        "audit_id": "m7_teacher_live_provenance_20260527",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "M7 provenance for teacher_* active/live eligibility",
        "operator_attestation": OPERATOR_ATTESTATION,
        "m7_files": {key: _inspect_csv(path) for key, path in M7_FILES.items()},
        "model_meta": {key: _inspect_model_meta(path) for key, path in MODEL_META.items()},
        "pipeline_checks": _pipeline_checks(),
        "decision_rule": [
            "teacher_* must be deterministic transforms of point-in-time AI/M7 predictions.",
            "M7 files must be timestamp-monotonic, duplicate-free, and exact-year scored outputs.",
            "M7 generation must use 2024 train split before 2025 augmentation by pipeline contract.",
            "No merge_asof/bfill/future-shift pattern may appear in active teacher materialization path.",
        ],
    }
    status, blocking, warnings = _status(audit)
    audit["status"] = status
    audit["blocking"] = blocking
    audit["warnings"] = warnings
    audit["active_live_candidate_decision"] = (
        "promote_teacher_features_with_m7" if status == "pass" else "do_not_promote"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    live = {
        "candidate": "teacher_features_with_m7_upstream",
        "status": "active_live_candidate" if status == "pass" else "blocked",
        "audit": _rel(AUDIT_OUT),
        "allowed_feature_prefixes": ["teacher_", "m7_", "ai_"],
        "conditions": [
            "2025 teacher_* uses 2024-trained AI/M7 scores.",
            "2026 teacher_* uses transform-only/OOS AI/M7 scores.",
            "Future M7 rebuilds must rerun this audit and keep timestamp-exact joins.",
        ],
        "blocking": blocking,
        "warnings": warnings,
    }
    LIVE_OUT.write_text(json.dumps(live, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": status, "audit": _rel(AUDIT_OUT), "live": _rel(LIVE_OUT)}, ensure_ascii=False))
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
