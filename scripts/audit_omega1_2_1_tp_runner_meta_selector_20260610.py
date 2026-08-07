#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "omega1_2_1_tp_runner_meta_selector_20260610"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
BUNDLE_PATH = ROOT / "data/ensemble/supervised" / MODEL_ID / "tp_runner_meta_selector.joblib"
SCRIPT_PATH = ROOT / "scripts/train_eval_omega1_2_1_tp_runner_meta_selector_20260610.py"
AUDIT_JSON = OUT_DIR / "redteam_accounting_audit.json"
AUDIT_MD = OUT_DIR / "redteam_accounting_audit.md"

FORBIDDEN_PREFIXES = (
    "teacher_",
    "clean_regime4_",
    "clean_regime_2024_unsup_v4_",
    "regime4_pred_",
)
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


def _forbidden(cols: list[str]) -> list[str]:
    return [c for c in cols if c in FORBIDDEN_EXACT or any(c.startswith(p) for p in FORBIDDEN_PREFIXES)]


def _ledger_audit(path: Path, *, expected_pnl: float | None) -> dict[str, Any]:
    df = pd.read_csv(path)
    if df.empty:
        return {"path": str(path), "rows": 0, "status": "empty"}
    cash_after_values = pd.to_numeric(df["cash_after"], errors="raise").to_numpy(dtype=np.float64)
    reported_final_cash = float(cash_after_values[-1])
    reported_pnl = float((reported_final_cash - 1.0) * 100.0)
    expected_cash = None if expected_pnl is None else float(1.0 + float(expected_pnl) / 100.0)
    expected_err = None if expected_cash is None else abs(reported_final_cash - expected_cash)
    monotonic_rows = bool(np.all(np.isfinite(cash_after_values)))
    status = "pass"
    if not monotonic_rows:
        status = "fail"
    if expected_err is not None and expected_err > 1e-10:
        status = "fail"
    return {
        "path": str(path),
        "rows": int(len(df)),
        "reported_final_cash": reported_final_cash,
        "reported_final_pnl": reported_pnl,
        "expected_pnl": None if expected_pnl is None else float(expected_pnl),
        "expected_cash_abs_error": None if expected_err is None else float(expected_err),
        "cash_after_values_finite": monotonic_rows,
        "note": "net_trade_return_pct is entry_equity-relative; ledger rows do not contain enough entry-fee detail for simple pct compounding replay.",
        "exit_reasons": df["exit_reason"].value_counts().to_dict(),
        "runner_extensions": {
            "sum": int(pd.to_numeric(df.get("runner_extensions", 0), errors="coerce").fillna(0).sum()),
            "max": int(pd.to_numeric(df.get("runner_extensions", 0), errors="coerce").fillna(0).max()),
        },
        "status": status,
    }


def _event_audit(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    if df.empty:
        return {"path": str(path), "events": 0, "status": "empty"}
    labels = pd.to_numeric(df["label"], errors="raise")
    allowed = pd.to_numeric(df["allowed"], errors="raise")
    edge = pd.to_numeric(df["edge"], errors="raise")
    return {
        "path": str(path),
        "events": int(len(df)),
        "positive_labels": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "allowed": int(allowed.sum()),
        "allowed_rate": float(allowed.mean()),
        "mean_edge": float(edge.mean()),
        "median_edge": float(edge.median()),
        "max_edge": float(edge.max()),
        "min_edge": float(edge.min()),
        "status": "pass" if len(df) >= 20 else "low_sample",
    }


def _source_selection_audit() -> dict[str, Any]:
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    oos_sort = 'sort_values(["oos_pnl_median", "val_pnl_median", "score"]' in text
    saves_bundle = "joblib.dump(bundle" in text
    return {
        "script": str(SCRIPT_PATH),
        "uses_oos_for_primary_ranking": bool(oos_sort),
        "saves_live_bundle": bool(saves_bundle),
        "status": "fail" if oos_sort else "pass",
    }


def main() -> int:
    if not OUT_DIR.exists():
        raise RuntimeError(f"missing output dir: {OUT_DIR}")
    report = json.loads((OUT_DIR / "report.json").read_text(encoding="utf-8"))
    ranking = pd.read_csv(OUT_DIR / "meta_selector_seed_ranking.csv")
    detail = pd.read_csv(OUT_DIR / "meta_selector_seed_detail.csv")
    bundle = joblib.load(BUNDLE_PATH) if BUNDLE_PATH.exists() else {}
    feature_cols = list(bundle.get("feature_cols") or [])
    forbidden = _forbidden(feature_cols)

    expected_pnl_by_ledger = {
        "validation_baseline_ledger": float(report["baseline"]["validation"]["pnl"]),
        "oos_baseline_ledger": float(report["baseline"]["oos"]["pnl"]),
        "validation_best_seed_ledger": float(bundle.get("selected_metrics", {}).get("validation", {}).get("val_pnl")),
        "oos_best_seed_ledger": float(bundle.get("selected_metrics", {}).get("oos", {}).get("oos_pnl")),
    }
    ledgers = {}
    for p in sorted(OUT_DIR.glob("*ledger.csv")):
        ledgers[p.stem] = _ledger_audit(p, expected_pnl=expected_pnl_by_ledger.get(p.stem))
    events = {
        p.stem: _event_audit(p)
        for p in sorted(OUT_DIR.glob("*validation_tp_events.csv"))
    }
    source = _source_selection_audit()
    top_oos = ranking.iloc[0].to_dict() if len(ranking) else {}
    top_val = (
        ranking.sort_values(["val_pnl_median", "val_pnl_min", "oos_pnl_median"], ascending=False)
        .iloc[0]
        .to_dict()
        if len(ranking)
        else {}
    )
    selected = {
        "bundle_path": str(BUNDLE_PATH),
        "model_id": bundle.get("model_id", ""),
        "status": bundle.get("status", ""),
        "selector_kind": bundle.get("selector_kind", ""),
        "selector_seed": bundle.get("selector_seed", None),
        "proba_min": bundle.get("proba_min", None),
        "template": bundle.get("template", {}),
        "feature_cols": feature_cols,
        "forbidden_features": forbidden,
    }

    findings: list[dict[str, Any]] = []
    if source["uses_oos_for_primary_ranking"]:
        findings.append(
            {
                "severity": "P1",
                "title": "OOS is used for selector/model ranking",
                "detail": "The saved bundle is selected from ranking sorted by oos_pnl_median first. OOS remains useful as research feedback, but the reported OOS uplift is not an untouched holdout result.",
            }
        )
    min_events = min((e.get("events", 0) for e in events.values()), default=0)
    if min_events <= 20:
        findings.append(
            {
                "severity": "P1",
                "title": "Training sample is too small for live promotion",
                "detail": f"Validation TP-hit event samples are only {min_events}. This is acceptable for shadow research, not for active TP extension.",
            }
        )
    if forbidden:
        findings.append(
            {
                "severity": "P0",
                "title": "Forbidden features found",
                "detail": forbidden,
            }
        )
    if any(v.get("status") == "fail" for v in ledgers.values()):
        findings.append(
            {
                "severity": "P0",
                "title": "Ledger cash compounding mismatch",
                "detail": {k: v for k, v in ledgers.items() if v.get("status") == "fail"},
            }
        )
    findings.append(
        {
            "severity": "P2",
            "title": "Live parity still unproven",
            "detail": "The shadow logger is installed, but no live TP-hit shadow rows have been collected yet. Live MAE is approximated in shadow logging until full in-position MAE tracking is added.",
        }
    )

    audit = {
        "model_id": MODEL_ID,
        "baseline": report.get("baseline", {}),
        "selected_bundle": selected,
        "top_by_oos_ranking": top_oos,
        "top_by_validation_ranking": top_val,
        "seed_detail_rows": int(len(detail)),
        "ledger_audit": ledgers,
        "event_audit": events,
        "source_selection_audit": source,
        "findings": findings,
        "verdict": {
            "accounting": "pass" if not any(v.get("status") == "fail" for v in ledgers.values()) else "fail",
            "feature_contract": "pass" if not forbidden else "fail",
            "promotion": "blocked",
            "reason": "OOS selection contamination and small TP-hit sample; keep shadow-only.",
        },
    }
    AUDIT_JSON.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")

    lines = [
        f"# {MODEL_ID} Red-Team / Accounting Audit",
        "",
        "## Verdict",
        "",
        f"- Accounting: `{audit['verdict']['accounting']}`",
        f"- Feature contract: `{audit['verdict']['feature_contract']}`",
        f"- Promotion: `{audit['verdict']['promotion']}`",
        f"- Reason: {audit['verdict']['reason']}",
        "",
        "## Findings",
        "",
    ]
    for f in findings:
        lines.append(f"- `{f['severity']}` {f['title']}: {f['detail']}")
    lines += [
        "",
        "## Selected Bundle",
        "",
        f"- Path: `{BUNDLE_PATH}`",
        f"- Template: `{selected['template']}`",
        f"- Selector: `{selected['selector_kind']}` seed `{selected['selector_seed']}` proba_min `{selected['proba_min']}`",
        f"- Forbidden features: `{forbidden}`",
        "",
        "## Ledger Audit",
        "",
    ]
    for name, item in ledgers.items():
        lines.append(
            f"- `{name}`: status `{item.get('status')}`, rows `{item.get('rows')}`, "
            f"reported_pnl `{item.get('reported_final_pnl', 0.0):.6f}%`, expected_err `{item.get('expected_cash_abs_error', 0.0) or 0.0:.3e}`"
        )
    lines += ["", "## Event Audit", ""]
    for name, item in events.items():
        lines.append(
            f"- `{name}`: events `{item.get('events')}`, positive `{item.get('positive_labels')}`, "
            f"allowed `{item.get('allowed')}`, mean_edge `{item.get('mean_edge', 0.0):.6f}`"
        )
    AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"audit_json": str(AUDIT_JSON), "audit_md": str(AUDIT_MD), "verdict": audit["verdict"], "findings": findings}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
