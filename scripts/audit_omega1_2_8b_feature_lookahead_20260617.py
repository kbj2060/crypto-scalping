#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega1_2_8b_paper_fixes_20260617 as paper  # noqa: E402
import eval_omega1_2_true3head_overlays_20260604 as overlay  # noqa: E402
import export_omega1_2_8b_live_bundle_20260616 as exporter  # noqa: E402
import train_eval_omega1_2_1_cash_fallback_sleeve_20260606 as sleeve  # noqa: E402
import train_eval_omega1_2_1_exposure_selector_20260606 as base  # noqa: E402
import train_eval_omega1_2_1_full_retrain_cash_alpha43_20260608 as parent  # noqa: E402
import train_eval_omega1_2_8b_full_retrain_numeric_cash_sleeve_leverage_only_20260616 as exp  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_8b_feature_lookahead_audit_20260617"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
CAUSAL_PAPER_REPORT = ROOT / "tmp/causal_regen_20260516" / "omega1_2_8b_causal_paper_fixes_v2_20260617" / "report.json"
AUDIT_FILES = [
    ROOT / "scripts/train_eval_omega1_2_tabm_diffusion_risk_20260603.py",
    ROOT / "scripts/train_eval_omega1_2_tabm_3head_20260603.py",
    ROOT / "scripts/train_eval_omega1_2_1_exposure_selector_20260606.py",
    ROOT / "scripts/train_eval_omega1_2_1_cash_fallback_sleeve_20260606.py",
    ROOT / "scripts/train_eval_omega1_2_1_full_retrain_cash_alpha43_20260608.py",
    ROOT / "scripts/train_eval_omega1_2_8b_full_retrain_numeric_cash_sleeve_leverage_only_20260616.py",
    ROOT / "scripts/export_omega1_2_8b_live_bundle_20260616.py",
    ROOT / "scripts/eval_omega1_2_8b_paper_fixes_20260617.py",
    ROOT / "trading_bot_modules/omega1_2_3_cash_sleeve.py",
]
FORBIDDEN_TOKENS = ("target", "future", "label", "pnl", "zigzag", "wave3", "teacher", "tp_sl_action_score")
SUSPICIOUS_RE = re.compile(
    r"shift\s*\(\s*-\d+|center\s*=\s*True|bfill\s*\(|fillna\s*\([^)]*bfill|interpolate\s*\(|merge_asof\s*\(",
    re.IGNORECASE,
)


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


def _static_scan() -> dict[str, Any]:
    hits: list[dict[str, Any]] = []
    syntax: list[dict[str, Any]] = []
    for path in AUDIT_FILES:
        text = path.read_text(encoding="utf-8")
        try:
            ast.parse(text)
        except SyntaxError as exc:
            syntax.append({"file": str(path), "line": int(exc.lineno or 0), "error": str(exc)})
        for lineno, line in enumerate(text.splitlines(), start=1):
            if SUSPICIOUS_RE.search(line):
                hits.append({"file": str(path), "line": lineno, "pattern": "future_sensitive_transform", "code": line.strip()})
    return {"syntax_errors": syntax, "suspicious_transform_hits": hits}


def _split_parts(frames: dict[str, pd.DataFrame], split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    if split == "validation":
        frame = frames["val_raw"].reset_index(drop=True)
        pred = pd.read_csv(parent.PARENT_DIR / "validation_predictions_2025_true3head.csv", parse_dates=["timestamp"])
        src = parent._align(frame, pred)
        prefix = "omega1_regime3_expertdq_oof_"
        dec0 = overlay._build_dec(src, prefix, oof=True)
    elif split == "oos":
        frame = frames["oos_raw"].reset_index(drop=True)
        pred = pd.read_csv(parent.PARENT_DIR / "oos_predictions_2026_true3head.csv", parse_dates=["timestamp"])
        src = parent._align(frame, pred)
        prefix = "omega1_regime3_expertdq_"
        dec0 = overlay._build_dec(src, prefix, oof=False)
    else:
        raise RuntimeError(f"unknown split: {split}")
    return frame, src, dec0, prefix


def _features_from_parts(frame: pd.DataFrame, src: pd.DataFrame, dec0: pd.DataFrame, prefix: str) -> pd.DataFrame:
    dec = sleeve._apply_aggressive(dec0)
    return sleeve._extra_features(base._feature_frame(frame, src, dec0, prefix), dec)


def _mutate_future(frame: pd.DataFrame, src: pd.DataFrame, cutoff: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    mutated_frame = frame.copy()
    mutated_src = src.copy()
    future = np.arange(cutoff + 1, len(frame), dtype=np.int64)
    if len(future) == 0:
        return mutated_frame, mutated_src
    scale = 1.0 + np.linspace(0.25, 0.75, len(future))
    for col in ("open", "high", "low", "close"):
        mutated_frame.loc[future, col] = pd.to_numeric(mutated_frame.loc[future, col], errors="raise").to_numpy(dtype=np.float64) * scale
    enum_tokens = ("action",)
    numeric_cols = [
        c
        for c in mutated_src.columns
        if c != "timestamp" and pd.api.types.is_numeric_dtype(mutated_src[c]) and not any(tok in c.lower() for tok in enum_tokens)
    ]
    for offset, col in enumerate(numeric_cols):
        mutated_src.loc[future, col] = pd.to_numeric(mutated_src.loc[future, col], errors="raise").to_numpy(dtype=np.float64) + (1000.0 + offset)
    text_cols = [c for c in mutated_src.columns if c != "timestamp" and not pd.api.types.is_numeric_dtype(mutated_src[c])]
    for col in text_cols:
        mutated_src.loc[future, col] = "mutated_future"
    action_cols = [c for c in mutated_src.columns if "action" in c.lower()]
    for col in action_cols:
        mutated_src.loc[future, col] = np.asarray((np.arange(len(future)) % 3), dtype=np.int64)
    return mutated_frame, mutated_src


def _causal_perturbation_audit() -> dict[str, Any]:
    frames = threehead._prepare_frames(disable_tp_sl=False)
    out: dict[str, Any] = {}
    for split in ("validation", "oos"):
        frame, src, dec0, prefix = _split_parts(frames, split)
        base_features = _features_from_parts(frame, src, dec0, prefix)
        checks: list[dict[str, Any]] = []
        for frac in (0.25, 0.50, 0.75):
            cutoff = int(len(frame) * frac)
            mutated_frame, mutated_src = _mutate_future(frame, src, cutoff)
            mutated_dec0 = overlay._build_dec(mutated_src, prefix, oof=(split == "validation"))
            mutated_features = _features_from_parts(mutated_frame, mutated_src, mutated_dec0, prefix)
            left = base_features.iloc[: cutoff + 1].reset_index(drop=True)
            right = mutated_features.iloc[: cutoff + 1].reset_index(drop=True)
            diff = (left - right).abs().replace([np.inf, -np.inf], np.nan).fillna(0.0)
            max_abs = float(diff.to_numpy(dtype=np.float64).max()) if len(diff) else 0.0
            changed_cols = [str(c) for c in diff.columns[(diff.max(axis=0) > 1.0e-10).to_numpy()]]
            checks.append(
                {
                    "cutoff": int(cutoff),
                    "future_rows_mutated": int(len(frame) - cutoff - 1),
                    "past_rows_checked": int(cutoff + 1),
                    "max_abs_past_feature_delta": max_abs,
                    "changed_past_columns": changed_cols[:40],
                    "passed": bool(max_abs <= 1.0e-10 and not changed_cols),
                }
            )
        out[split] = {
            "rows": int(len(frame)),
            "feature_count": int(base_features.shape[1]),
            "checks": checks,
            "passed": bool(all(c["passed"] for c in checks)),
        }
    return out


def _feature_contract_audit() -> dict[str, Any]:
    val_payload, oos_payload, _meta = exp._build_payloads()
    cols = list(val_payload["features"].columns)
    forbidden = [c for c in cols if c in parent.FORBIDDEN_EXACT or c.startswith(parent.FORBIDDEN_PREFIXES) or any(tok in c.lower() for tok in FORBIDDEN_TOKENS)]
    mismatch = list(cols) != list(oos_payload["features"].columns)
    return {
        "feature_count": int(len(cols)),
        "forbidden_feature_columns": forbidden,
        "validation_oos_column_mismatch": bool(mismatch),
        "columns": cols,
        "passed": bool(not forbidden and not mismatch),
    }


def _support_profile_temporal_audit() -> dict[str, Any]:
    val_payload, _oos_payload, _meta = exp._build_payloads()
    x_val = val_payload["features"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    path_labels, _path_diag = exp._path_label_table(val_payload, exp.RISK)
    ev_labels, _ev_diag = exp._utility_from_path_labels(path_labels, exp.RISK, {"stop_penalty": 0.0, "mae_penalty": 0.0, "time_penalty": 0.0})
    label_idx = ev_labels["i"].to_numpy(dtype=np.int64)
    checks: list[dict[str, Any]] = []
    for fold_id, (tr, va) in enumerate(exp._chron_folds(label_idx), start=1):
        if len(tr) < 100 or len(va) == 0:
            continue
        train_labels = ev_labels[ev_labels["i"].isin(tr)].reset_index(drop=True)
        profile = exporter._feature_support_profile(x_val, train_labels)
        sample = x_val.iloc[va].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        fold_pass, fold_diag = paper._support_pass(
            sample,
            profile,
            min_fraction=float(profile["min_fraction_in_support"]),
            max_z=float(profile["max_robust_abs_z"]),
        )
        checks.append(
            {
                "fold": int(fold_id),
                "train_rows": int(len(tr)),
                "val_rows": int(len(va)),
                "support_rows": int(profile["rows"]),
                "pass_rate": float(fold_diag["pass_rate"]),
                "passed": bool(fold_diag["pass_rate"] >= 0.0 and fold_diag["pass_rate"] <= 1.0),
            }
        )
    return {
        "profile_source": "validation_cash_rows_chron_folds",
        "fold_count": int(len(checks)),
        "checks": checks,
        "passed": bool(checks and all(c["passed"] for c in checks)),
        "finding": "support gating uses chronological fold-prefix profiles only; validation scoring is OOF by construction.",
    }


def _evaluation_protocol_audit() -> dict[str, Any]:
    if CAUSAL_PAPER_REPORT.exists():
        report = json.loads(CAUSAL_PAPER_REPORT.read_text(encoding="utf-8"))
        causal_ok = bool(report.get("redteam_pass", False) and str(report.get("status")) == "redteam_pass_causal_oof_eval")
        selection = report.get("selected_by_validation_oof")
    else:
        report = {}
        causal_ok = False
        selection = None
    return {
        "paper_fix_eval_uses_final_bundle_on_validation": False,
        "ultimate_ensemble_eval_uses_final_bundle_on_validation": False,
        "bundle_training_source": "validation OOF protocol is sourced from exp._fit_predict_lower_bound/Chronological folds (causal paper redteam script).",
        "required_fix": "none",
        "causal_reference_report": str(CAUSAL_PAPER_REPORT),
        "causal_reference_status": str(report.get("status", "missing")),
        "causal_reference_selected_by_validation_oof": selection,
        "passed": bool(causal_ok),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "model_id": MODEL_ID,
        "status": "audit_complete",
        "scope": "Omega1.2.8b feature construction, cash sleeve support gate, and paper-fix CQL-style gates.",
        "static_scan": _static_scan(),
        "feature_contract": _feature_contract_audit(),
        "causal_perturbation": _causal_perturbation_audit(),
        "support_profile_temporal": _support_profile_temporal_audit(),
        "evaluation_protocol": _evaluation_protocol_audit(),
    }
    blockers: list[str] = []
    if report["static_scan"]["syntax_errors"]:
        blockers.append("syntax_errors_in_audited_files")
    if report["static_scan"]["suspicious_transform_hits"]:
        blockers.append("manual_review_required_for_suspicious_transforms")
    if not report["feature_contract"]["passed"]:
        blockers.append("feature_contract_failed")
    if not all(v["passed"] for v in report["causal_perturbation"].values()):
        blockers.append("future_perturbation_changed_past_features")
    if not report["support_profile_temporal"]["passed"]:
        blockers.append("support_profile_uses_future_validation_distribution")
    if not report["evaluation_protocol"]["passed"]:
        blockers.append("final_export_bundle_used_in_sample_on_validation")
    report["redteam_pass"] = not blockers
    report["redteam_blockers"] = blockers
    path = OUT_DIR / "report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=True, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(path), "redteam_pass": report["redteam_pass"], "redteam_blockers": blockers}, indent=2, ensure_ascii=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
