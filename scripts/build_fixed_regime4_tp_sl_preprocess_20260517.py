#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_regime_pred_moe_20260517 import _json_default  # noqa: E402


MODEL_ID = "fixed_regime4_tp18_sl10_preprocess_20260517"
CLEAN4_PREFIX = "clean_regime4_2024_unsup_v1_"
PRED4_PREFIX = "regime4_pred_"
OLD_CLEAN_PREFIX = "clean_regime_2024_unsup_v4_"
REGIMES = ("bull", "bear", "chop", "whipsaw")

DEFAULT_TRAIN_CSV = ROOT / "tmp/causal_regen_20260516/alpha4_2_tp_sl_action_score_20260517/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL_CSV = ROOT / "tmp/causal_regen_20260516/alpha4_2_tp_sl_action_score_20260517/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_CLEAN4_2025 = ROOT / "data/ensemble/supervised/clean_regime4_raw_state12_v1_20260517/training_features_2025_clean_regime4_raw_state12_v1.csv"
DEFAULT_PRED4_2025 = ROOT / "data/ensemble/supervised/regime4_pred_tft_h12_nomdjd_all74_20260517/training_features_2025_regime4_pred_tft_vsn_selected.csv"
DEFAULT_CLEAN4_2026 = ROOT / "data/ensemble/supervised/clean_regime4_raw_state12_v1_20260517/training_features_2026_rebuilt_clean_regime4_raw_state12_v1.csv"
DEFAULT_PRED4_2026 = ROOT / "data/ensemble/supervised/regime4_pred_tft_h12_nomdjd_all74_20260517/training_features_2026_rebuilt_regime4_pred_tft_vsn_selected.csv"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/fixed_regime4_tp18_sl10_preprocess_20260517"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build canonical fixed Regime4 + TP 1.8% / SL 1.0% preprocessing frames.")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    p.add_argument("--clean4-2025", type=Path, default=DEFAULT_CLEAN4_2025)
    p.add_argument("--pred4-2025", type=Path, default=DEFAULT_PRED4_2025)
    p.add_argument("--clean4-2026", type=Path, default=DEFAULT_CLEAN4_2026)
    p.add_argument("--pred4-2026", type=Path, default=DEFAULT_PRED4_2026)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--train-out", type=Path, default=None)
    p.add_argument("--eval-out", type=Path, default=None)
    p.add_argument("--manifest-out", type=Path, default=None)
    p.add_argument("--strict-eval-regime", action="store_true", help="fail when 2026 Regime4 sidecars are missing")
    return p.parse_args()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"{path} missing timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _merge_sidecar(base: pd.DataFrame, sidecar: pd.DataFrame, prefix: str, source: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    cols = [c for c in sidecar.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(f"{source} has no columns with prefix {prefix}")
    overlap = set(base.columns) & set(cols)
    if overlap:
        raise ValueError(f"{source} would overwrite existing columns: {sorted(overlap)[:8]}")
    out = base.merge(sidecar[["timestamp"] + cols], on="timestamp", how="left")
    missing_rows = int(out[cols].isna().any(axis=1).sum())
    tail_rows_dropped = 0
    if missing_rows:
        missing_mask = out[cols].isna().any(axis=1)
        miss_idx = np.flatnonzero(missing_mask.to_numpy())
        is_tail_suffix = bool(
            len(miss_idx) > 0
            and np.array_equal(miss_idx, np.arange(int(miss_idx[0]), len(out)))
        )
        if is_tail_suffix:
            tail_rows_dropped = int(len(miss_idx))
            out = out.loc[~missing_mask].reset_index(drop=True)
        else:
            raise ValueError(f"{source} failed timestamp alignment; missing rows={missing_rows}")
    return out.sort_values("timestamp").reset_index(drop=True), {
        "source": str(source),
        "rows": int(len(sidecar)),
        "columns": cols,
        "column_count": int(len(cols)),
        "sha256": _sha256(source),
        "tail_rows_dropped": tail_rows_dropped,
    }


def _prob_audit(frame: pd.DataFrame, prefix: str) -> dict[str, Any]:
    prob_cols = [f"{prefix}{name}_prob" for name in REGIMES]
    missing = [c for c in prob_cols if c not in frame.columns]
    if missing:
        return {"status": "fail", "missing": missing}
    sums = frame[prob_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    return {
        "status": "pass",
        "probability_columns": prob_cols,
        "prob_sum_min": float(sums.min()),
        "prob_sum_max": float(sums.max()),
        "nan_count": int(frame[prob_cols].isna().sum().sum()),
    }


def _frame_audit(frame: pd.DataFrame) -> dict[str, Any]:
    if "tp_sl_action_score" not in frame.columns:
        raise ValueError("fixed TP/SL source missing tp_sl_action_score")
    tp = pd.to_numeric(frame["tp_sl_action_score"], errors="coerce")
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "range": [str(frame["timestamp"].iloc[0]), str(frame["timestamp"].iloc[-1])],
        "tp_sl_action_score": {
            "nan_count": int(tp.isna().sum()),
            "mean": float(tp.mean()),
            "std": float(tp.std()),
            "zero_rate": float((tp.fillna(0.0) == 0.0).mean()),
            "positive_rate": float((tp > 0.0).mean()),
            "negative_rate": float((tp < 0.0).mean()),
        },
        "clean_regime4": _prob_audit(frame, CLEAN4_PREFIX),
        "future_regime4": _prob_audit(frame, PRED4_PREFIX),
    }


def _build_one(base_path: Path, clean_path: Path, pred_path: Path, out_path: Path) -> dict[str, Any]:
    base = _read(base_path)
    old_clean_cols = [c for c in base.columns if c.startswith(OLD_CLEAN_PREFIX)]
    if old_clean_cols:
        base = base.drop(columns=old_clean_cols)
    clean = _read(clean_path)
    pred = _read(pred_path)
    if "tp_sl_action_score" not in base.columns:
        raise ValueError(f"{base_path} is not the fixed TP/SL 1.8/1.0 preprocessed source; missing tp_sl_action_score")
    out, clean_meta = _merge_sidecar(base, clean, CLEAN4_PREFIX, clean_path)
    out, pred_meta = _merge_sidecar(out, pred, PRED4_PREFIX, pred_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return {
        "output": str(out_path),
        "output_sha256": _sha256(out_path),
        "source": {"path": str(base_path), "sha256": _sha256(base_path)},
        "dropped_legacy_columns": old_clean_cols,
        "sidecars": {"clean_regime4": clean_meta, "future_regime4": pred_meta},
        "audit": _frame_audit(out),
    }


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_out = args.train_out or args.out_dir / "trade_candidates_2025_regime4_tp18_sl10_fixed.csv"
    eval_out = args.eval_out or args.out_dir / "trade_candidates_2026_regime4_tp18_sl10_fixed.csv"
    manifest_out = args.manifest_out or args.out_dir / "fixed_regime4_tp18_sl10_preprocess_manifest.json"

    train = _build_one(args.train_csv, args.clean4_2025, args.pred4_2025, train_out)
    eval_result: dict[str, Any] | None = None
    eval_missing = [str(p) for p in (args.clean4_2026, args.pred4_2026) if not p.exists()]
    if eval_missing:
        if args.strict_eval_regime:
            raise FileNotFoundError("missing 2026 Regime4 sidecars: " + ", ".join(eval_missing))
    else:
        eval_result = _build_one(args.eval_csv, args.clean4_2026, args.pred4_2026, eval_out)

    manifest = {
        "model_id": MODEL_ID,
        "fixed_preprocessing_contract": {
            "regime_taxonomy": list(REGIMES),
            "current_regime_prefix": CLEAN4_PREFIX,
            "future_regime_prefix": PRED4_PREFIX,
            "tp_sl_feature": "tp_sl_action_score",
            "tp": 0.018,
            "sl": 0.010,
            "tp_sl_horizon_bars": 48,
            "tp_sl_entry_reference": "next_bar_open",
            "same_bar_tp_sl_tie": "sl_wins",
            "timestamp_join": "exact_left_join_no_missing_rows",
            "risk_off_transition_classes": "disabled",
            "risk_off_transition_auxiliary_features": "enabled",
            "current_regime_auxiliary_features": [
                "factor_trend",
                "factor_flow",
                "factor_vol",
                "factor_crowding",
                "factor_liquidity",
                "trend_bias",
                "risk_off_prob",
                "transition_risk",
            ],
            "normal_class": "disabled",
        },
        "train": train,
        "eval": eval_result,
        "warnings": [
            "2026 output is only written when both current and future Regime4 2026 sidecars exist.",
            "This fixes preprocessing inputs; it does not promote any downstream model.",
        ],
    }
    manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_out), "train": train["output"], "eval": None if eval_result is None else eval_result["output"], "eval_missing": eval_missing}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
