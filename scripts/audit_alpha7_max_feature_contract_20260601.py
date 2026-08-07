#!/usr/bin/env python3
from __future__ import annotations

import hashlib
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
from scripts.retrain_alpha7_active_max_feature_contract_moe_20260601 import _load_frames_max


MODEL_ID = "alpha7_active_max_feature_contract_moe_20260601"
MODEL_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_active_max_feature_contract_moe_20260601"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_active_max_feature_contract_audit_20260601"
DOC_PATH = ROOT / "docs/audits/alpha7_active_max_feature_contract_audit_20260601.md"

FORBIDDEN_PREFIXES = ("teacher_", "a5dir_", "clean_regime4_", "regime4_pred_", "regime3_pred_")
FORBIDDEN_TOKENS = ("target", "future", "pnl", "wave3", "zigzag", "tp_sl_action_score")
SAFE_EXACT = {"realized_vol_ratio", "realized_skewness", "m7_hdb_label"}
PARENT_PATHS = [
    MODEL_DIR / "baseline_max/primary_max/parent.pkl",
    MODEL_DIR / "baseline_max/fallback_max/parent.pkl",
    MODEL_DIR / "bull_max/primary_max/parent.pkl",
    MODEL_DIR / "bull_max/fallback_max/parent.pkl",
    MODEL_DIR / "bear_max/primary_max/parent.pkl",
    MODEL_DIR / "bear_max/fallback_max/parent.pkl",
    MODEL_DIR / "chop_max/primary_max/parent.pkl",
    MODEL_DIR / "chop_max/fallback_max/parent.pkl",
]
HASH_COLS = ("timestamp", "open", "high", "low", "close", "volume")


def _issue(severity: str, title: str, detail: str, **extra: Any) -> dict[str, Any]:
    return {"severity": severity, "title": title, "detail": detail, **extra}


def _source_hash(frame: pd.DataFrame) -> pd.Series:
    cols = [c for c in HASH_COLS if c in frame.columns]
    data = frame[cols].copy()
    for col in cols:
        data[col] = data[col].astype(str)
    return data.apply(lambda row: hashlib.sha256("|".join(row.tolist()).encode("utf-8")).hexdigest()[:24], axis=1)


def _attach_timestamp(dec: pd.DataFrame, frame: pd.DataFrame, tag: str) -> pd.DataFrame:
    if len(dec) != len(frame):
        raise RuntimeError(f"{tag}: row mismatch {len(dec)} != {len(frame)}")
    out = dec.copy().reset_index(drop=True)
    out.insert(0, "timestamp", pd.to_datetime(frame["timestamp"]).astype(str).reset_index(drop=True))
    out.insert(1, "source_row_hash", _source_hash(frame).to_numpy())
    return out


def _forbidden(cols: list[str]) -> list[str]:
    bad: list[str] = []
    for col in cols:
        if col in SAFE_EXACT:
            continue
        low = col.lower()
        if col.startswith(FORBIDDEN_PREFIXES) or any(tok in low for tok in FORBIDDEN_TOKENS):
            bad.append(col)
    return bad


def _audit_features() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}
    for path in PARENT_PATHS:
        if not path.exists():
            issues.append(_issue("P0", "missing parent artifact", str(path)))
            continue
        cols = list(joblib.load(path).get("feature_cols", []))
        bad = _forbidden(cols)
        if bad:
            issues.append(_issue("P0", "forbidden feature in max contract", ", ".join(bad[:80]), artifact=str(path), count=len(bad)))
        stats[str(path)] = {
            "feature_count": len(cols),
            "forbidden_count": len(bad),
            "regime3_count": sum(c.startswith("regime3_") for c in cols),
            "m7_count": sum(c.startswith("m7_") for c in cols),
            "ai_tsfm_count": sum(c.startswith(("ai_", "patchtst_", "tide_", "dlinear_", "pred_patchtst", "conf_patchtst")) for c in cols),
        }
    return issues, stats


def _active(dec: pd.DataFrame) -> np.ndarray:
    action = pd.to_numeric(dec["action"], errors="raise").to_numpy(dtype=np.int64)
    side = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
    return (action != 0) & (side != 0)


def _audit_decisions(dec: pd.DataFrame, tag: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    active = _active(dec)
    side = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
    expert = dec["router_expert"].astype(str).to_numpy()
    if bool((active & (expert == "bull") & (side < 0)).any()):
        issues.append(_issue("P0", f"{tag} bull emitted short", str(int((active & (expert == "bull") & (side < 0)).sum()))))
    if bool((active & (expert == "bear") & (side > 0)).any()):
        issues.append(_issue("P0", f"{tag} bear emitted long", str(int((active & (expert == "bear") & (side > 0)).sum()))))
    for col in ["action", "side", "notional_exposure", "position_fraction", "leverage", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars", "quality_score", "confidence", "router_confidence"]:
        values = pd.to_numeric(dec[col], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            issues.append(_issue("P0", f"{tag} non-finite decision col", col))
    return issues, {
        "rows": int(len(dec)),
        "active_rows": int(active.sum()),
        "policy_counts": {str(k): int(v) for k, v in dec["router_expert"].value_counts().to_dict().items()},
        "max_notional_exposure": float(pd.to_numeric(dec["notional_exposure"], errors="raise").max()),
        "max_position_fraction": float(pd.to_numeric(dec["position_fraction"], errors="raise").max()),
        "max_leverage": float(pd.to_numeric(dec["leverage"], errors="raise").max()),
    }


def _monthly(val_frame: pd.DataFrame, val_dec: pd.DataFrame) -> list[dict[str, Any]]:
    months = pd.to_datetime(val_frame["timestamp"]).dt.to_period("M").astype(str)
    out = []
    for month, idx in pd.Series(np.arange(len(months))).groupby(months).groups.items():
        loc = np.asarray(list(idx), dtype=np.int64)
        costs = _combo_metrics(val_frame.iloc[loc].reset_index(drop=True), val_dec.iloc[loc].reset_index(drop=True))
        c = costs["cost3"]
        out.append({"month": str(month), "rows": int(len(loc)), "cost3_pnl": float(c["pnl"]), "cost3_mdd": float(c["mdd"]), "cost3_trades": int(c["trades"]), "cost3_wr": float(c["wr"])})
    return out


def _write_md(payload: dict[str, Any]) -> None:
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    issue_lines = [f"- **{i['severity']} {i['title']}**: {i['detail']}" for i in payload["issues"]] or ["- No issues."]
    month_lines = [f"- `{m['month']}`: pnl `{m['cost3_pnl']:.2f}%`, mdd `{m['cost3_mdd']:.2f}%`, trades `{m['cost3_trades']}`, WR `{m['cost3_wr']:.3f}`" for m in payload["monthly_validation_stability"]]
    md = f"""# Alpha7 Max Feature Contract Audit 2026-06-01

## Verdict

{payload['verdict']}

## Findings

{chr(10).join(issue_lines)}

## Monthly Validation Stability

{chr(10).join(month_lines)}

## Timestamped Decisions

- Validation: `{OUT_DIR / 'validation_decisions_timestamped.csv'}`
- OOS: `{OUT_DIR / 'oos_2026_decisions_timestamped.csv'}`
"""
    DOC_PATH.write_text(md, encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = json.loads((MODEL_DIR / "report.json").read_text(encoding="utf-8"))
    ranking = pd.read_csv(MODEL_DIR / "ranking_validation_only.csv")
    issues: list[dict[str, Any]] = []
    if any(c.startswith("oos_") for c in ranking.columns):
        issues.append(_issue("P1", "OOS metrics leaked into validation-only ranking", str(MODEL_DIR / "ranking_validation_only.csv")))
    train_all, eval_df, overlay = _load_frames_max()
    val_frame = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
    val_dec = pd.read_csv(MODEL_DIR / "validation_decisions.csv", low_memory=False).reset_index(drop=True)
    oos_dec = pd.read_csv(MODEL_DIR / "oos_2026_decisions.csv", low_memory=False).reset_index(drop=True)
    val_ts = _attach_timestamp(val_dec, val_frame, "validation")
    oos_ts = _attach_timestamp(oos_dec, eval_df, "oos")
    val_ts.to_csv(OUT_DIR / "validation_decisions_timestamped.csv", index=False)
    oos_ts.to_csv(OUT_DIR / "oos_2026_decisions_timestamped.csv", index=False)
    val_metrics = _combo_metrics(val_frame, val_dec)
    oos_metrics = _combo_metrics(eval_df, oos_dec)
    feature_issues, feature_stats = _audit_features()
    issues.extend(feature_issues)
    val_issues, val_sanity = _audit_decisions(val_dec, "validation")
    oos_issues, oos_sanity = _audit_decisions(oos_dec, "oos")
    issues.extend(val_issues)
    issues.extend(oos_issues)
    months = _monthly(val_frame, val_dec)
    neg = [m for m in months if m["cost3_pnl"] <= 0.0]
    if neg:
        issues.append(_issue("P1", "validation has non-positive monthly Cost3 blocks", ", ".join(m["month"] for m in neg), months=neg))
    hard = [i for i in issues if i["severity"] in {"P0", "P1"}]
    verdict = "PASS" if not hard else "FAIL: max feature candidate has hard audit issues."
    payload = {
        "model_id": MODEL_ID,
        "verdict": verdict,
        "issues": issues,
        "metrics_recomputed": {"validation": val_metrics, "oos": oos_metrics},
        "reported_selected": report["selected"],
        "feature_contract": feature_stats,
        "decision_sanity": {"validation": val_sanity, "oos": oos_sanity},
        "monthly_validation_stability": months,
        "overlay": overlay,
        "timestamped_artifacts": {"validation": str(OUT_DIR / "validation_decisions_timestamped.csv"), "oos": str(OUT_DIR / "oos_2026_decisions_timestamped.csv")},
    }
    (OUT_DIR / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    _write_md(payload)
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "doc": str(DOC_PATH), "verdict": verdict, "issues": pd.Series([i["severity"] for i in issues]).value_counts().to_dict()}, ensure_ascii=False, indent=2))
    return 1 if hard else 0


if __name__ == "__main__":
    raise SystemExit(main())
