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
from scripts.train_alpha7_regime3_current_moe_feature_variants_20260601 import _load_frames_with_risk


MODEL_ID = "alpha7_active_clean_contract_moe_20260601"
MODEL_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_active_clean_contract_moe_20260601"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_active_clean_contract_hard_audit_20260601"
DOC_PATH = ROOT / "docs/audits/alpha7_active_clean_contract_hard_audit_20260601.md"

PARENT_PATHS = [
    MODEL_DIR / "baseline_clean/primary_no_tp/parent.pkl",
    MODEL_DIR / "baseline_clean/fallback_clean/parent.pkl",
    MODEL_DIR / "bull_practical_clean/primary_no_tp/parent.pkl",
    MODEL_DIR / "bull_practical_clean/fallback_clean/parent.pkl",
    MODEL_DIR / "bear_risk_clean/primary_no_tp/parent.pkl",
    MODEL_DIR / "bear_risk_clean/fallback_clean/parent.pkl",
    MODEL_DIR / "chop_practical_clean/primary_no_tp/parent.pkl",
    MODEL_DIR / "chop_practical_clean/fallback_clean/parent.pkl",
]

CANONICAL_DENY_PREFIXES = (
    "teacher_",
    "a5dir_",
    "clean_regime4_",
    "regime4_pred_",
    "regime3_pred_",
)
CANONICAL_DENY_TOKENS = (
    "label",
    "target",
    "future",
    "realized",
    "pnl",
    "wave3",
    "zigzag",
    "tp_sl_action_score",
)
PROVENANCE_REVIEW_PREFIXES = (
    "m7_",
    "patchtst_",
    "ai_dir_",
    "ai_",
    "tide_",
    "dlinear_",
)
HASH_COL_CANDIDATES = ("timestamp", "open", "high", "low", "close", "volume")


def _issue(severity: str, title: str, detail: str, **extra: Any) -> dict[str, Any]:
    return {"severity": severity, "title": title, "detail": detail, **extra}


def _source_hash(frame: pd.DataFrame) -> pd.Series:
    cols = [c for c in HASH_COL_CANDIDATES if c in frame.columns]
    if "timestamp" not in cols:
        raise RuntimeError("source frame missing timestamp for row hash")
    data = frame[cols].copy()
    for col in cols:
        data[col] = data[col].astype(str)
    return data.apply(lambda row: hashlib.sha256("|".join(row.tolist()).encode("utf-8")).hexdigest()[:24], axis=1)


def _attach_timestamp(dec: pd.DataFrame, frame: pd.DataFrame, *, tag: str) -> pd.DataFrame:
    if len(dec) != len(frame):
        raise RuntimeError(f"{tag}: decision/frame row mismatch: {len(dec)} != {len(frame)}")
    out = dec.copy().reset_index(drop=True)
    ts = pd.to_datetime(frame["timestamp"]).reset_index(drop=True)
    if "timestamp" in out.columns:
        existing = pd.to_datetime(out["timestamp"]).reset_index(drop=True)
        if not existing.equals(ts):
            raise RuntimeError(f"{tag}: existing decision timestamp does not match source frame")
    out.insert(0, "timestamp", ts.astype(str))
    out.insert(1, "source_row_hash", _source_hash(frame).to_numpy())
    return out


def _forbidden_cols(cols: list[str]) -> list[str]:
    out: list[str] = []
    for col in cols:
        low = col.lower()
        if col.startswith(CANONICAL_DENY_PREFIXES) or any(tok in low for tok in CANONICAL_DENY_TOKENS):
            out.append(col)
    return out


def _audit_features() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}
    for path in PARENT_PATHS:
        if not path.exists():
            issues.append(_issue("P0", "missing parent artifact", str(path)))
            continue
        model = joblib.load(path)
        cols = list(model.get("feature_cols", []))
        bad = _forbidden_cols(cols)
        if bad:
            issues.append(_issue("P0", "canonical deny-list violation", ", ".join(bad[:60]), artifact=str(path), count=len(bad)))
        review = [c for c in cols if c.startswith(PROVENANCE_REVIEW_PREFIXES)]
        stats[str(path)] = {
            "feature_count": len(cols),
            "denied_count": len(bad),
            "denied_cols": bad,
            "provenance_review_count": len(review),
            "provenance_review_cols": review,
        }
    return issues, stats


def _monthly_stability(val_frame: pd.DataFrame, val_dec: pd.DataFrame) -> list[dict[str, Any]]:
    tmp = val_frame[["timestamp"]].copy().reset_index(drop=True)
    tmp["month"] = pd.to_datetime(tmp["timestamp"]).dt.to_period("M").astype(str)
    out: list[dict[str, Any]] = []
    for month, idx in tmp.groupby("month").groups.items():
        loc = np.asarray(list(idx), dtype=np.int64)
        frame_m = val_frame.iloc[loc].reset_index(drop=True)
        dec_m = val_dec.iloc[loc].reset_index(drop=True)
        costs = _combo_metrics(frame_m, dec_m)
        c3 = costs["cost3"]
        out.append({
            "month": str(month),
            "rows": int(len(frame_m)),
            "cost3_pnl": float(c3["pnl"]),
            "cost3_mdd": float(c3["mdd"]),
            "cost3_trades": int(c3["trades"]),
            "cost3_wr": float(c3["wr"]),
            "cost3_trades_per_day": float(c3["trades_per_day"]),
        })
    return out


def _write_md(payload: dict[str, Any]) -> None:
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    issue_lines = [f"- **{i['severity']} {i['title']}**: {i['detail']}" for i in payload["issues"]] or ["- No issues."]
    months = payload["monthly_validation_stability"]
    month_lines = [
        f"- `{m['month']}`: pnl `{m['cost3_pnl']:.2f}%`, mdd `{m['cost3_mdd']:.2f}%`, trades `{m['cost3_trades']}`, WR `{m['cost3_wr']:.3f}`"
        for m in months
    ]
    md = f"""# Alpha7 Clean Contract Hard Audit 2026-06-01

## Verdict

{payload['verdict']}

## Findings

{chr(10).join(issue_lines)}

## Monthly Validation Stability

{chr(10).join(month_lines)}

## Timestamped Decisions

- Validation: `{OUT_DIR / 'validation_decisions_timestamped.csv'}`
- OOS: `{OUT_DIR / 'oos_2026_decisions_timestamped.csv'}`

## JSON

- `{OUT_DIR / 'report.json'}`
"""
    DOC_PATH.write_text(md, encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_all, eval_df, overlay = _load_frames_with_risk()
    val_frame = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
    val_dec = pd.read_csv(MODEL_DIR / "validation_decisions.csv", low_memory=False).reset_index(drop=True)
    oos_dec = pd.read_csv(MODEL_DIR / "oos_2026_decisions.csv", low_memory=False).reset_index(drop=True)
    val_ts = _attach_timestamp(val_dec, val_frame, tag="validation")
    oos_ts = _attach_timestamp(oos_dec, eval_df, tag="oos")
    val_ts.to_csv(OUT_DIR / "validation_decisions_timestamped.csv", index=False)
    oos_ts.to_csv(OUT_DIR / "oos_2026_decisions_timestamped.csv", index=False)
    issues, feature_stats = _audit_features()
    monthly = _monthly_stability(val_frame, val_dec)
    neg_months = [m for m in monthly if float(m["cost3_pnl"]) <= 0.0]
    if neg_months:
        issues.append(_issue("P1", "validation has non-positive monthly Cost3 blocks", ", ".join(m["month"] for m in neg_months), months=neg_months))
    hard = [i for i in issues if i["severity"] in {"P0", "P1"}]
    verdict = "PASS" if not hard else "FAIL: canonical feature or monthly stability hard checks failed."
    payload = {
        "model_id": MODEL_ID,
        "verdict": verdict,
        "issues": issues,
        "feature_contract": feature_stats,
        "monthly_validation_stability": monthly,
        "timestamped_artifacts": {
            "validation": str(OUT_DIR / "validation_decisions_timestamped.csv"),
            "oos": str(OUT_DIR / "oos_2026_decisions_timestamped.csv"),
        },
        "overlay": overlay,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    _write_md(payload)
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "doc": str(DOC_PATH), "verdict": verdict, "issue_counts": pd.Series([i["severity"] for i in issues]).value_counts().to_dict()}, ensure_ascii=False, indent=2))
    return 1 if hard else 0


if __name__ == "__main__":
    raise SystemExit(main())
