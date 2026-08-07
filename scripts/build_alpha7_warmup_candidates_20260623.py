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

from features.engineering import QuantSignalFeatures  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


MODEL_ID = "alpha7_01965_cleanfunding_candidates_warmup_20260623"
SOURCE_ID = "alpha7_01965_cleanfunding_candidates_20260529"
SOURCE_DIR = ROOT / "tmp/causal_regen_20260516" / SOURCE_ID
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

TRAIN_CSV = SOURCE_DIR / "trade_candidates_2025_alpha6_current_tail111_exact.csv"
EVAL_CSV = SOURCE_DIR / "trade_candidates_2026_alpha6_current_tail111_exact.csv"
WARMUP_ROWS = 2500
WARMUP_COLS = [
    "turtle_signal",
    "dual_momentum",
    "mean_reversion_z",
    "breakout_strength",
    "volume_profile_signal",
    "fibonacci_level",
]
RAW_REQUIRED_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    missing = [c for c in RAW_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"{path} missing required columns: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    if df["timestamp"].duplicated().any():
        raise RuntimeError(f"{path} has duplicate timestamps")
    return df.sort_values("timestamp").reset_index(drop=True)


def _zero_ratio(s: pd.Series, rows: int) -> float:
    head = pd.to_numeric(s.head(min(int(rows), len(s))), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(head) == 0:
        return 0.0
    return float((head.abs() <= 1.0e-12).mean())


def _col_audit(before: pd.Series, after: pd.Series) -> dict[str, Any]:
    before_num = pd.to_numeric(before, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    after_num = pd.to_numeric(after, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    diff = (after_num - before_num).abs()
    changed = diff > 1.0e-12
    return {
        "before_zero_ratio_first_1d": _zero_ratio(before_num, 288),
        "after_zero_ratio_first_1d": _zero_ratio(after_num, 288),
        "before_zero_ratio_first_7d": _zero_ratio(before_num, 2016),
        "after_zero_ratio_first_7d": _zero_ratio(after_num, 2016),
        "before_zero_ratio_first_30d": _zero_ratio(before_num, 8640),
        "after_zero_ratio_first_30d": _zero_ratio(after_num, 8640),
        "changed_rows": int(changed.sum()),
        "max_abs_diff": float(diff.max()) if len(diff) else 0.0,
    }


def _manual_signal_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = [c for c in RAW_REQUIRED_COLS if c != "timestamp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"manual signal recompute missing columns: {missing}")
    return QuantSignalFeatures(df.copy()).add_all_signals()


def main() -> int:
    train = _read(TRAIN_CSV)
    eval_df = _read(EVAL_CSV)
    missing_warmup = [c for c in WARMUP_COLS if c not in eval_df.columns]
    if missing_warmup:
        raise RuntimeError(f"eval csv missing warmup columns: {missing_warmup}")
    if train["timestamp"].iloc[-1] >= eval_df["timestamp"].iloc[0]:
        raise RuntimeError("train/eval timestamps are not strictly chronological")
    if len(train) < WARMUP_ROWS:
        raise RuntimeError(f"train rows {len(train)} < warmup rows {WARMUP_ROWS}")

    warm = train.tail(WARMUP_ROWS).copy()
    combined = pd.concat([warm, eval_df], ignore_index=True)
    recomputed = _manual_signal_frame(combined)
    eval_recomputed = recomputed.iloc[len(warm) :].reset_index(drop=True)
    if not eval_df["timestamp"].reset_index(drop=True).equals(eval_recomputed["timestamp"].reset_index(drop=True)):
        raise RuntimeError("warmup recompute timestamp alignment failed")

    patched = eval_df.copy()
    audit_cols: dict[str, Any] = {}
    for col in WARMUP_COLS:
        audit_cols[col] = _col_audit(patched[col], eval_recomputed[col])
        patched[col] = pd.to_numeric(eval_recomputed[col], errors="raise").to_numpy(dtype=np.float64)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUT_DIR / TRAIN_CSV.name
    eval_path = OUT_DIR / EVAL_CSV.name
    train.to_csv(train_path, index=False)
    patched.to_csv(eval_path, index=False)
    audit = {
        "model_id": MODEL_ID,
        "status": "candidate_csv_with_2025_tail_warmup_for_2026_manual_signals",
        "source_id": SOURCE_ID,
        "warmup_rows": WARMUP_ROWS,
        "warmup_columns": WARMUP_COLS,
        "selection_uses_2026": False,
        "train_csv": str(train_path),
        "eval_csv": str(eval_path),
        "source_train_csv": str(TRAIN_CSV),
        "source_eval_csv": str(EVAL_CSV),
        "train_rows": int(len(train)),
        "eval_rows": int(len(eval_df)),
        "eval_start": str(eval_df["timestamp"].iloc[0]),
        "eval_end": str(eval_df["timestamp"].iloc[-1]),
        "column_audit": audit_cols,
    }
    audit_path = OUT_DIR / "candidate_warmup_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"train_csv": str(train_path), "eval_csv": str(eval_path), "audit": str(audit_path)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
