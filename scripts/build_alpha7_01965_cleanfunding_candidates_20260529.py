#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


MODEL_ID = "alpha7_01965_cleanfunding_candidates_20260529"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

OLD_TRAIN = ROOT / "tmp/causal_regen_20260516/alpha7_1_01965_v2only_tp_sl_action_score_20260528/trade_candidates_2025_alpha6_current_tail111_exact.csv"
OLD_EVAL = ROOT / "tmp/causal_regen_20260516/alpha7_1_01965_v2only_tp_sl_action_score_20260528/trade_candidates_2026_alpha6_current_tail111_exact.csv"

CLEAN_UNIFIED_2025 = ROOT / "tmp/causal_regen_20260516/funding_clean_retrain_20260529/rl_training_2025_unified_cleanfunding.csv"
CLEAN_FEATURES_2026 = ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"
CLEAN_M7_2025 = ROOT / "tmp/causal_regen_20260516/splits/rl_training_2025_m7.csv"
CLEAN_M7_2026 = ROOT / "tmp/causal_regen_20260516/splits/rl_training_2026_m7.csv"
REGIME_PRED_2025 = ROOT / "data/ensemble/supervised/regime4_pred_tft_h12_nomdjd_all74_20260517/training_features_2025_regime4_pred_tft_vsn_selected.csv"
REGIME_PRED_2026 = ROOT / "data/ensemble/supervised/regime4_pred_tft_h12_nomdjd_all74_20260517/training_features_2026_rebuilt_regime4_pred_tft_vsn_selected.csv"

FORBIDDEN_PREFIXES = ("clean_regime_2024_unsup_v4_", "clean_regime4_2024_unsup_v1_")
REQUIRED_V2_PREFIX = "clean_regime4_state24_sticky090_v2_"
FUNDING_DERIVED_COLS = [
    "last_funding_rate",
    "funding_abs",
    "funding_pressure",
    "funding_roc_12",
    "funding_roc_48",
    "funding_roc_288",
    "funding_z_score",
    "funding_price_divergence",
    "long_squeeze_risk",
    "short_squeeze_risk",
    "squeeze_power",
    "mta_funding",
    "ou_funding_z",
    "crowding_pressure",
]


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise RuntimeError(f"missing timestamp: {path}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    if df["timestamp"].duplicated().any():
        raise RuntimeError(f"duplicate timestamps in {path}")
    return df.sort_values("timestamp").reset_index(drop=True)


def _audit_frame(df: pd.DataFrame, *, name: str) -> dict[str, Any]:
    bad = [c for c in df.columns if str(c).startswith(FORBIDDEN_PREFIXES)]
    if bad:
        raise RuntimeError(f"{name} contains forbidden legacy columns: {bad[:20]}")
    v2 = [c for c in df.columns if str(c).startswith(REQUIRED_V2_PREFIX)]
    reg = [c for c in df.columns if str(c).startswith("regime4_pred_")]
    if len(v2) < 20:
        raise RuntimeError(f"{name} has insufficient v2 regime columns: {len(v2)}")
    if len(reg) < 12:
        raise RuntimeError(f"{name} has insufficient regime4_pred columns: {len(reg)}")
    return {
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "v2_regime_cols": int(len(v2)),
        "regime4_pred_cols": int(len(reg)),
        "forbidden_legacy_cols": 0,
        "funding_cols": [c for c in df.columns if "funding" in str(c)],
        "m7_cols": [c for c in df.columns if str(c).startswith("m7_")],
        "ai_cols": [c for c in df.columns if str(c).startswith(("ai_", "patchtst", "tide", "dlinear"))],
    }


def _overlay_exact(base: pd.DataFrame, source: pd.DataFrame, *, source_name: str, columns: list[str] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    cols = [c for c in source.columns if c != "timestamp" and c in base.columns]
    if columns is not None:
        want = [c for c in columns if c in base.columns]
        missing = [c for c in want if c not in source.columns]
        if missing:
            raise RuntimeError(f"{source_name} missing required overlay columns: {missing[:20]}")
        cols = want
    if not cols:
        raise RuntimeError(f"{source_name} has no overlay columns")
    src = source[["timestamp", *cols]].copy()
    merged = base[["timestamp"]].merge(src, on="timestamp", how="left", validate="one_to_one", indicator=True)
    if not (merged["_merge"] == "both").all():
        miss = merged.loc[merged["_merge"] != "both", "timestamp"].head(5).astype(str).tolist()
        raise RuntimeError(f"{source_name} overlay has missing timestamps: {miss}")
    merged = merged.drop(columns=["_merge"])
    out = base.copy()
    changed: dict[str, Any] = {}
    for c in cols:
        old = pd.to_numeric(out[c], errors="coerce")
        new = pd.to_numeric(merged[c], errors="coerce")
        numeric = old.notna().any() or new.notna().any()
        if numeric:
            diff = (old.astype(float) - new.astype(float)).abs()
            changed[c] = {"max_abs_diff": float(diff.max(skipna=True) or 0.0), "diff_count": int((diff > 1e-12).sum())}
        out[c] = merged[c].to_numpy()
    return out, {"source": source_name, "column_count": int(len(cols)), "columns": cols, "diff": changed}


def _build(base_path: Path, *, split: str, unified: Path | None, clean_features: Path | None, m7_path: Path, regime_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    base = _read(base_path)
    overlays: list[dict[str, Any]] = []
    if "tp_sl_action_score" in base.columns:
        base = base.drop(columns=["tp_sl_action_score"])
    if unified is not None:
        uni = _read(unified)
        base, info = _overlay_exact(base, uni, source_name=f"{split}_clean_unified")
        overlays.append(info)
    if clean_features is not None:
        feat = _read(clean_features)
        base, info = _overlay_exact(base, feat, source_name=f"{split}_clean_funding_derived", columns=FUNDING_DERIVED_COLS)
        overlays.append(info)
    m7 = _read(m7_path)
    m7_cols = [c for c in m7.columns if c != "timestamp" and c in base.columns and (c.startswith("m7_") or c == "sig_ai_squeeze")]
    base, info = _overlay_exact(base, m7, source_name=f"{split}_clean_m7", columns=m7_cols)
    overlays.append(info)
    regime = _read(regime_path)
    regime_cols = [c for c in regime.columns if c.startswith("regime4_pred_")]
    base, info = _overlay_exact(base, regime, source_name=f"{split}_clean_regime4_pred", columns=regime_cols)
    overlays.append(info)
    audit = {"split": split, "base": str(base_path), "overlays": overlays, "frame": _audit_frame(base, name=split)}
    return base, audit


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train, train_audit = _build(OLD_TRAIN, split="train_2025", unified=CLEAN_UNIFIED_2025, clean_features=None, m7_path=CLEAN_M7_2025, regime_path=REGIME_PRED_2025)
    eval_df, eval_audit = _build(OLD_EVAL, split="eval_2026", unified=None, clean_features=CLEAN_FEATURES_2026, m7_path=CLEAN_M7_2026, regime_path=REGIME_PRED_2026)
    train_path = OUT_DIR / OLD_TRAIN.name
    eval_path = OUT_DIR / OLD_EVAL.name
    train.to_csv(train_path, index=False)
    eval_df.to_csv(eval_path, index=False)
    audit = {
        "model_id": MODEL_ID,
        "status": "candidate_csv_cleanfunding_rebuilt",
        "selection_uses_2026": False,
        "train_csv": str(train_path),
        "eval_csv": str(eval_path),
        "source_old_train": str(OLD_TRAIN),
        "source_old_eval": str(OLD_EVAL),
        "clean_sources": {
            "clean_unified_2025": str(CLEAN_UNIFIED_2025),
            "clean_features_2026": str(CLEAN_FEATURES_2026),
            "clean_m7_2025": str(CLEAN_M7_2025),
            "clean_m7_2026": str(CLEAN_M7_2026),
            "regime_pred_2025": str(REGIME_PRED_2025),
            "regime_pred_2026": str(REGIME_PRED_2026),
        },
        "train": train_audit,
        "eval": eval_audit,
    }
    audit_path = OUT_DIR / "candidate_cleanfunding_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"train_csv": str(train_path), "eval_csv": str(eval_path), "audit": str(audit_path)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
