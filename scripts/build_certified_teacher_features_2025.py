#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.certified_teacher_regime_moe import (  # noqa: E402
    CLEAN_PREFIX,
    append_clean_regime,
    fit_clean_regime_predictor,
    load_csv,
    merge_teacher_sources,
)
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


def merge_context_from_m7(base, m7):
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
    p = argparse.ArgumentParser(description="Build 2025 certified teacher feature matrix.")
    p.add_argument("--state-2024", type=Path, default=ROOT / "data/splits/year_oos/training_features_2024.csv")
    p.add_argument("--base-2025", type=Path, default=ROOT / "data/splits/year_oos/training_features_2025.csv")
    p.add_argument("--ai-2025", type=Path, default=ROOT / "data/tmp/unified_build_ckpt/03_after_ai.csv")
    p.add_argument("--m7-2025", type=Path, default=ROOT / "data/splits/year_oos/rl_training_2025_m7.csv")
    p.add_argument("--out", type=Path, default=ROOT / "data/ensemble/supervised/certified_teacher_regime_moe_v1/features_2025.csv")
    p.add_argument("--regime-out", type=Path, default=ROOT / "data/ensemble/supervised/certified_teacher_regime_moe_v1/clean_regime_2024_unsup_v4.pkl")
    p.add_argument("--audit-out", type=Path, default=ROOT / "data/ensemble/reports/certified_teacher_regime_moe_v1_features_2025_audit.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ai_audit = audit_ai_contracts()
    y2024 = load_csv(args.state_2024)
    base = load_csv(args.base_2025)
    ai = load_csv(args.ai_2025)
    m7 = load_csv(args.m7_2025)
    base = merge_context_from_m7(base, m7)
    train_teacher = merge_teacher_sources(y2024, None, None)
    if args.ai_2025.exists():
        train_teacher = merge_teacher_sources(train_teacher, ai=None, m7=None)
    regime = fit_clean_regime_predictor(merge_teacher_sources(y2024, None, None))
    frame = merge_teacher_sources(base, ai, m7)
    frame = append_clean_regime(frame, regime)
    frame = append_side_teacher_features(frame)
    assert_context_health(frame)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.regime_out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    joblib.dump(regime, args.regime_out)
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
        "note": "2025 is model-training materialization, not OOS.",
    }
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": audit["status"], "out": str(args.out), "rows": len(frame)}, ensure_ascii=False))
    return 0 if audit["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
