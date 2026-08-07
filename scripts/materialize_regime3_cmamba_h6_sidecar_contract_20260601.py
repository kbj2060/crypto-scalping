#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "data/ensemble/supervised/regime3_cryptomamba_pred_h6_nocurrent_20260531"
OUT_DIR = ROOT / "data/ensemble/supervised/regime3_cryptomamba_h6_sidecar_20260601"

FILES = {
    "2024": "training_features_2024_regime3_cryptomamba_pred_h6_nocurrent_20260531.csv",
    "2025": "training_features_2025_regime3_cryptomamba_pred_h6_nocurrent_20260531.csv",
    "2026": "training_features_2026_rebuilt_regime3_cryptomamba_pred_h6_nocurrent_20260531.csv",
}

RENAME_MAP = {
    "regime3_cmamba_h6_future_bull_prob": "regime3_cmamba_h6_sidecar_bull_prob",
    "regime3_cmamba_h6_future_bear_prob": "regime3_cmamba_h6_sidecar_bear_prob",
    "regime3_cmamba_h6_future_chop_prob": "regime3_cmamba_h6_sidecar_chop_prob",
    "regime3_cmamba_h6_future_pred_id": "regime3_cmamba_h6_sidecar_class_id",
    "regime3_cmamba_h6_future_pred_name": "regime3_cmamba_h6_sidecar_class_name",
    "regime3_cmamba_h6_confidence": "regime3_cmamba_h6_sidecar_confidence",
    "regime3_cmamba_h6_transition_prob": "regime3_cmamba_h6_sidecar_transition_prob",
    "regime3_cmamba_h6_stability_score": "regime3_cmamba_h6_sidecar_stability_score",
}

REQUIRED_OUTPUT_COLS = ["timestamp", *RENAME_MAP.values()]


def _check_contract(df: pd.DataFrame, *, tag: str) -> None:
    missing = [c for c in RENAME_MAP if c not in df.columns]
    if missing:
        raise RuntimeError(f"{tag}: source missing required columns: {missing}")
    forbidden = [c for c in df.columns if "future" in c.lower()]
    if forbidden:
        # Source can contain old names; this check is for documenting the input only.
        pass


def _write_one(year: str, src_name: str) -> dict[str, Any]:
    src = SRC_DIR / src_name
    if not src.exists():
        raise FileNotFoundError(src)
    df = pd.read_csv(src, parse_dates=["timestamp"], low_memory=False)
    if "timestamp" not in df.columns:
        raise RuntimeError(f"{src}: missing timestamp")
    _check_contract(df, tag=year)
    out = df[["timestamp", *RENAME_MAP.keys()]].rename(columns=RENAME_MAP).copy()
    if any("future" in c.lower() for c in out.columns):
        raise RuntimeError(f"{year}: renamed sidecar still contains future in output columns")
    if out["timestamp"].duplicated().any():
        raise RuntimeError(f"{year}: duplicate timestamps in output")
    out = out.sort_values("timestamp").reset_index(drop=True)
    out_name = src_name.replace("regime3_cryptomamba_pred_h6_nocurrent_20260531", "regime3_cryptomamba_h6_sidecar_20260601")
    out_path = OUT_DIR / out_name
    out.to_csv(out_path, index=False)
    return {
        "year": year,
        "source": str(src),
        "output": str(out_path),
        "rows": int(len(out)),
        "first_timestamp": str(out["timestamp"].iloc[0]) if len(out) else None,
        "last_timestamp": str(out["timestamp"].iloc[-1]) if len(out) else None,
        "columns": list(out.columns),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [_write_one(year, name) for year, name in FILES.items()]
    report = {
        "contract": "Regime3 CryptoMamba h6 sidecar renamed contract. This is a materialized contract file, not a runtime alias. Columns avoid the word 'future' because values are model sidecar outputs, not future labels.",
        "source_dir": str(SRC_DIR),
        "output_dir": str(OUT_DIR),
        "rename_map": RENAME_MAP,
        "required_output_cols": REQUIRED_OUTPUT_COLS,
        "outputs": outputs,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "outputs": outputs}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
