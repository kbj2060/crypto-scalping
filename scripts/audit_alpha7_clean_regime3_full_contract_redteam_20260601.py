#!/usr/bin/env python3
from __future__ import annotations

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

from scripts.analyze_alpha7_tp_sl_action_score_20260526 import SPLIT_TS, _combo_metrics, _json_default
from scripts.retrain_alpha7_active_clean_regime3_full_contract_moe_20260601 import _load_frames_with_full_regime3


MODEL_ID = "alpha7_active_clean_regime3_full_contract_moe_20260601"
MODEL_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_active_clean_regime3_full_contract_moe_20260601"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_active_clean_regime3_full_contract_redteam_20260601"
DOC_PATH = ROOT / "docs/audits/alpha7_active_clean_regime3_full_contract_redteam_20260601.md"

FORBIDDEN_PREFIXES = ("clean_regime4_", "regime4_pred_", "regime3_pred_", "teacher_", "a5dir_")
FORBIDDEN_EXACT = {"tp_sl_action_score"}
FORBIDDEN_TOKENS = ("tp_sl_action_score", "label", "target", "realized", "pnl", "wave3", "zigzag")
PARENT_PATHS = [
    MODEL_DIR / "baseline_regime3_full/primary_no_tp/parent.pkl",
    MODEL_DIR / "baseline_regime3_full/fallback_clean/parent.pkl",
    MODEL_DIR / "bull_regime3_full/primary_no_tp/parent.pkl",
    MODEL_DIR / "bull_regime3_full/fallback_clean/parent.pkl",
    MODEL_DIR / "bear_regime3_full/primary_no_tp/parent.pkl",
    MODEL_DIR / "bear_regime3_full/fallback_clean/parent.pkl",
    MODEL_DIR / "chop_regime3_full/primary_no_tp/parent.pkl",
    MODEL_DIR / "chop_regime3_full/fallback_clean/parent.pkl",
]


def _issue(severity: str, title: str, detail: str, **extra: Any) -> dict[str, Any]:
    return {"severity": severity, "title": title, "detail": detail, **extra}


def _active(dec: pd.DataFrame) -> np.ndarray:
    action = pd.to_numeric(dec["action"], errors="raise").to_numpy(dtype=np.int64)
    side = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
    return (action != 0) & (side != 0)


def _forbidden(cols: list[str]) -> list[str]:
    return [
        c for c in cols
        if c in FORBIDDEN_EXACT
        or c.startswith(FORBIDDEN_PREFIXES)
        or any(tok in c.lower() for tok in FORBIDDEN_TOKENS)
    ]


def _audit_features() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}
    for path in PARENT_PATHS:
        if not path.exists():
            issues.append(_issue("P0", "missing parent artifact", str(path)))
            continue
        model = joblib.load(path)
        cols = list(model.get("feature_cols", []))
        bad = _forbidden(cols)
        if bad:
            issues.append(_issue("P0", "forbidden feature in strict full-Regime3 contract", ", ".join(bad[:40]), artifact=str(path)))
        stats[str(path)] = {
            "feature_count": len(cols),
            "regime3_count": sum(c.startswith("regime3_") for c in cols),
            "forbidden_count": len(bad),
            "first_30_features": cols[:30],
        }
    return issues, stats


def _audit_decisions(dec: pd.DataFrame, tag: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    numeric = ["action", "side", "notional_exposure", "position_fraction", "leverage", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars", "quality_score", "confidence", "router_confidence"]
    for col in numeric:
        if col not in dec.columns:
            issues.append(_issue("P0", f"{tag} missing decision column", col))
            continue
        values = pd.to_numeric(dec[col], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            issues.append(_issue("P0", f"{tag} non-finite decision column", col))
    active = _active(dec)
    side = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
    expert = dec["router_expert"].astype(str).to_numpy()
    if bool((active & (expert == "bull") & (side < 0)).any()):
        issues.append(_issue("P0", f"{tag} bull expert emitted short", str(int((active & (expert == "bull") & (side < 0)).sum()))))
    if bool((active & (expert == "bear") & (side > 0)).any()):
        issues.append(_issue("P0", f"{tag} bear expert emitted long", str(int((active & (expert == "bear") & (side > 0)).sum()))))
    return issues, {
        "rows": int(len(dec)),
        "active_rows": int(active.sum()),
        "policy_counts": {str(k): int(v) for k, v in dec["router_expert"].value_counts().to_dict().items()},
        "max_notional_exposure": float(pd.to_numeric(dec["notional_exposure"], errors="raise").max()),
        "max_position_fraction": float(pd.to_numeric(dec["position_fraction"], errors="raise").max()),
        "max_leverage": float(pd.to_numeric(dec["leverage"], errors="raise").max()),
    }


def _compare_metrics(name: str, got: dict[str, Any], expected: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for cost in ("cost1", "cost2", "cost3"):
        for key in ("pnl", "mdd", "trades", "trades_per_day", "wr", "avg_notional"):
            a = got[cost][key]
            b = expected[cost][key]
            if key == "trades":
                ok = int(a) == int(b)
            else:
                ok = abs(float(a) - float(b)) <= 1e-7
            if not ok:
                issues.append(_issue("P0", f"{name} metric mismatch", f"{cost}.{key}: recomputed={a}, report={b}"))
    return issues


def _write_md(payload: dict[str, Any]) -> None:
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    issue_lines = [f"- **{i['severity']} {i['title']}**: {i['detail']}" for i in payload["issues"]] or ["- No issues."]
    md = f"""# Alpha7 Clean Regime3 Full-Contract Red-Team Audit 2026-06-01

## Verdict

{payload['verdict']}

## Findings

{chr(10).join(issue_lines)}

## Recomputed Metrics

- Validation Cost3: PnL `{payload['metrics_recomputed']['validation']['cost3']['pnl']:.6f}`, MDD `{payload['metrics_recomputed']['validation']['cost3']['mdd']:.6f}`, trades `{payload['metrics_recomputed']['validation']['cost3']['trades']}`, WR `{payload['metrics_recomputed']['validation']['cost3']['wr']:.6f}`
- OOS Cost3: PnL `{payload['metrics_recomputed']['oos']['cost3']['pnl']:.6f}`, MDD `{payload['metrics_recomputed']['oos']['cost3']['mdd']:.6f}`, trades `{payload['metrics_recomputed']['oos']['cost3']['trades']}`, WR `{payload['metrics_recomputed']['oos']['cost3']['wr']:.6f}`

## Artifacts

- JSON report: `{OUT_DIR / 'report.json'}`
- Model report: `{MODEL_DIR / 'report.json'}`
"""
    DOC_PATH.write_text(md, encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_report = json.loads((MODEL_DIR / "report.json").read_text(encoding="utf-8"))
    ranking = pd.read_csv(MODEL_DIR / "ranking_validation_only.csv")
    val_dec = pd.read_csv(MODEL_DIR / "validation_decisions.csv", low_memory=False).reset_index(drop=True)
    oos_dec = pd.read_csv(MODEL_DIR / "oos_2026_decisions.csv", low_memory=False).reset_index(drop=True)
    issues: list[dict[str, Any]] = []
    if any(c.startswith("oos_") for c in ranking.columns):
        issues.append(_issue("P1", "OOS metrics leaked into validation-only ranking", str(MODEL_DIR / "ranking_validation_only.csv")))
    train_all, eval_df, overlay = _load_frames_with_full_regime3()
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
    if len(val_df) != len(val_dec):
        issues.append(_issue("P0", "validation row mismatch", f"frame={len(val_df)}, decisions={len(val_dec)}"))
    if len(eval_df) != len(oos_dec):
        issues.append(_issue("P0", "oos row mismatch", f"frame={len(eval_df)}, decisions={len(oos_dec)}"))
    val_metrics = _combo_metrics(val_df, val_dec)
    oos_metrics = _combo_metrics(eval_df, oos_dec)
    issues.extend(_compare_metrics("validation", val_metrics, model_report["selected"]["validation"]))
    issues.extend(_compare_metrics("oos", oos_metrics, model_report["selected"]["oos"]))
    feature_issues, feature_stats = _audit_features()
    issues.extend(feature_issues)
    val_issues, val_sanity = _audit_decisions(val_dec, "validation")
    oos_issues, oos_sanity = _audit_decisions(oos_dec, "oos")
    issues.extend(val_issues)
    issues.extend(oos_issues)
    if float(val_metrics["cost3"]["pnl"]) <= 0.0:
        issues.append(_issue("P2", "validation Cost3 PnL is non-positive", "Clean full-Regime3 candidate should not be promoted live despite OOS strength until validation instability is resolved."))
    hard = [i for i in issues if i["severity"] in {"P0", "P1"}]
    verdict = "PASS_WITH_WARNINGS: strict contract passed, but validation performance warning remains." if not hard else "FAIL: full-Regime3 candidate has P0/P1 issues."
    payload = {
        "model_id": MODEL_ID,
        "verdict": verdict,
        "issues": issues,
        "metrics_recomputed": {"validation": val_metrics, "oos": oos_metrics},
        "decision_sanity": {"validation": val_sanity, "oos": oos_sanity},
        "feature_contract": feature_stats,
        "overlay": overlay,
        "selection_ranking_columns": list(ranking.columns),
        "artifacts": {"model_report": str(MODEL_DIR / "report.json"), "audit_report": str(OUT_DIR / "report.json")},
    }
    (OUT_DIR / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    _write_md(payload)
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "doc": str(DOC_PATH), "verdict": verdict, "issues": pd.Series([i["severity"] for i in issues]).value_counts().to_dict()}, ensure_ascii=False, indent=2))
    return 1 if hard else 0


if __name__ == "__main__":
    raise SystemExit(main())
