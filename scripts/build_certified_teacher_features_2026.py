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

from ensemble.certified_teacher_regime_moe import CLEAN_PREFIX, append_clean_regime, load_csv, merge_teacher_sources  # noqa: E402
from pipeline.certified_feature_audit import audit_ai_contracts  # noqa: E402
from pipeline.teacher_meta_side_features import append_side_teacher_features  # noqa: E402


PASSTHROUGH_M7_CONTEXT_COLS = [
    "garch_vol_z",
    "liquidity_vacuum",
    "execution_quality",
    "jump_z",
    "jump_flag",
    "evt_tail_flag",
    "evt_excess_z",
    "funding_abs",
    "funding_pressure",
    "crowding_pressure",
    "whale_conviction",
]


def _noninformative(series: pd.Series) -> bool:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if not s.notna().any():
        return True
    return int(s.nunique(dropna=True)) <= 1 or float((s.fillna(0.0) == 0.0).mean()) >= 0.999


def merge_context_from_m7(base: pd.DataFrame, m7: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in PASSTHROUGH_M7_CONTEXT_COLS if c in m7.columns]
    if not cols:
        return base
    renamed = m7[["timestamp"] + cols].rename(columns={c: f"{c}__m7ctx" for c in cols})
    out = base.merge(renamed, on="timestamp", how="left", validate="one_to_one").sort_values("timestamp").reset_index(drop=True)
    for col in cols:
        src = f"{col}__m7ctx"
        if src not in out.columns:
            continue
        m7_s = pd.to_numeric(out[src], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if col not in out.columns or _noninformative(out[col]):
            out[col] = m7_s
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").where(lambda s: s.notna(), m7_s)
        out.drop(columns=[src], inplace=True)
    return out


def assert_context_health(frame: pd.DataFrame) -> None:
    for col in ("garch_vol_z",):
        if col not in frame.columns:
            raise ValueError(f"missing critical context feature: {col}")
        if _noninformative(frame[col]):
            raise ValueError(f"noninformative critical context feature: {col}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build 2026 certified teacher feature matrix using frozen 2024 clean-regime predictor.")
    p.add_argument("--base-2026", type=Path, default=ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv")
    p.add_argument("--ai-2026", type=Path, default=ROOT / "data/tmp/unified_build_ckpt_2026/03_after_ai.csv")
    p.add_argument("--m7-2026", type=Path, default=ROOT / "data/splits/year_oos/rl_training_2026_m7_supervised_redesign_clean.csv")
    p.add_argument("--regime", type=Path, default=ROOT / "data/ensemble/supervised/certified_teacher_regime_moe_v1/clean_regime_2024_unsup_v4.pkl")
    p.add_argument("--out", type=Path, default=ROOT / "data/ensemble/supervised/certified_teacher_regime_moe_v1/features_2026.csv")
    p.add_argument("--audit-out", type=Path, default=ROOT / "data/ensemble/reports/certified_teacher_regime_moe_v1_features_2026_audit.json")
    return p.parse_args()


def _load_regime(path: Path) -> dict[str, Any]:
    if not path.exists():
        bundled = path.with_name("model.pkl")
        if bundled.exists():
            path = bundled
    payload = joblib.load(path)
    if isinstance(payload, dict) and isinstance(payload.get("regime"), dict):
        payload = payload["regime"]
    if not isinstance(payload, dict) or not {"feature_cols", "preprocess", "model"}.issubset(payload):
        raise ValueError(f"invalid clean regime artifact: {path}")
    return payload


def main() -> int:
    args = parse_args()
    ai_audit = audit_ai_contracts()
    regime = _load_regime(args.regime)
    base = load_csv(args.base_2026)
    ai = load_csv(args.ai_2026)
    m7 = load_csv(args.m7_2026)
    base = merge_context_from_m7(base, m7)
    frame = merge_teacher_sources(base, ai, m7)
    frame = append_clean_regime(frame, regime)
    frame = append_side_teacher_features(frame)
    assert_context_health(frame)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    audit = {
        "status": ai_audit["status"],
        "blocking": list(ai_audit["blocking"]),
        "ai_contracts": ai_audit["evidence"],
        "output": str(args.out),
        "rows": int(len(frame)),
        "timestamp_start": str(frame["timestamp"].iloc[0]),
        "timestamp_end": str(frame["timestamp"].iloc[-1]),
        "clean_prefix": CLEAN_PREFIX,
        "clean_regime_cols": [c for c in frame.columns if c.startswith(CLEAN_PREFIX)],
        "note": "2026 is transform-only OOS materialization. No fit or selection allowed.",
    }
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": audit["status"], "out": str(args.out), "rows": len(frame)}, ensure_ascii=False))
    return 0 if audit["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
