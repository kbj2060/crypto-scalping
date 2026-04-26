#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
for _p in (_ROOT_DIR, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ensemble.supervised.train_entry_price_model import EntryPriceBrain


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate entry-price model on OOS split")
    p.add_argument("--data-path", required=True)
    p.add_argument("--rl-path", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--opportunity-bps", type=float, default=8.0)
    p.add_argument("--output-path", default="")
    return p.parse_args()


def load_frames(data_path: str, rl_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    feat_df = pd.read_csv(data_path)
    rl_df = pd.read_csv(rl_path)
    for df in (feat_df, rl_df):
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return feat_df, rl_df


def align_work_frame(feat_df: pd.DataFrame, rl_df: pd.DataFrame) -> pd.DataFrame:
    work_df = rl_df.copy()
    extra_cols = [c for c in feat_df.columns if c not in work_df.columns and c != "timestamp"]
    if extra_cols:
        work_df = work_df.merge(feat_df[["timestamp"] + extra_cols], on="timestamp", how="left")
    return work_df


def _bps(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64) * 1e4


def _safe_mean(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.nanmean(arr))


def _safe_median(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.nanmedian(arr))


def main() -> int:
    args = parse_args()
    feat_df, rl_df = load_frames(args.data_path, args.rl_path)
    work_df = align_work_frame(feat_df, rl_df)

    brain = EntryPriceBrain.load(args.model_path)
    preds = brain.predict_batch_from_df(work_df)
    out = work_df.copy()
    out["pred_entry_long_offset"] = np.asarray(preds["entry_long_offset"], dtype=np.float64)
    out["pred_entry_short_offset"] = np.asarray(preds["entry_short_offset"], dtype=np.float64)

    close = pd.to_numeric(out["close"], errors="coerce").to_numpy(dtype=np.float64)
    low_fut = (
        pd.to_numeric(feat_df["low"], errors="coerce")
        .rolling(args.horizon, min_periods=args.horizon)
        .min()
        .shift(-args.horizon)
        .reindex(out.index)
        .to_numpy(dtype=np.float64)
    )
    high_fut = (
        pd.to_numeric(feat_df["high"], errors="coerce")
        .rolling(args.horizon, min_periods=args.horizon)
        .max()
        .shift(-args.horizon)
        .reindex(out.index)
        .to_numpy(dtype=np.float64)
    )

    pred_long_offset = np.clip(np.nan_to_num(out["pred_entry_long_offset"].to_numpy(dtype=np.float64), nan=0.0), -0.02, 0.0)
    pred_short_offset = np.clip(np.nan_to_num(out["pred_entry_short_offset"].to_numpy(dtype=np.float64), nan=0.0), 0.0, 0.02)
    reco_long = close * (1.0 + pred_long_offset)
    reco_short = close * (1.0 + pred_short_offset)

    valid = np.isfinite(close) & np.isfinite(low_fut) & np.isfinite(high_fut) & (close > 0.0)
    valid_long = valid & (low_fut > 0.0)
    valid_short = valid & (high_fut > 0.0)

    oracle_long_bps = _bps((close - low_fut) / close)
    oracle_short_bps = _bps((high_fut - close) / close)
    reco_long_improve_bps = _bps((close - reco_long) / close)
    reco_short_improve_bps = _bps((reco_short - close) / close)
    long_shortfall_oracle_bps = _bps((reco_long - low_fut) / close)
    short_shortfall_oracle_bps = _bps((high_fut - reco_short) / close)
    long_fill = low_fut <= reco_long
    short_fill = high_fut >= reco_short

    long_mae_bps = _bps(np.abs((low_fut / close - 1.0) - pred_long_offset))
    short_mae_bps = _bps(np.abs((high_fut - close) / close - pred_short_offset))

    opp_th = float(args.opportunity_bps)
    long_opp = valid_long & (oracle_long_bps >= opp_th)
    short_opp = valid_short & (oracle_short_bps >= opp_th)

    result = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
        "data_path": args.data_path,
        "rl_path": args.rl_path,
        "model_path": args.model_path,
        "horizon": int(args.horizon),
        "opportunity_bps": opp_th,
        "test_rows": int(np.sum(valid)),
        "long_mae_bps": _safe_mean(long_mae_bps[valid_long]),
        "short_mae_bps": _safe_mean(short_mae_bps[valid_short]),
        "long_fill_rate": _safe_mean(long_fill[valid_long].astype(np.float64)),
        "short_fill_rate": _safe_mean(short_fill[valid_short].astype(np.float64)),
        "avg_long_improve_bps_all": _safe_mean(reco_long_improve_bps[valid_long]),
        "avg_short_improve_bps_all": _safe_mean(reco_short_improve_bps[valid_short]),
        "avg_long_improve_bps_filled": _safe_mean(reco_long_improve_bps[valid_long & long_fill]),
        "avg_short_improve_bps_filled": _safe_mean(reco_short_improve_bps[valid_short & short_fill]),
        "avg_long_shortfall_to_oracle_bps": _safe_mean(long_shortfall_oracle_bps[valid_long]),
        "avg_short_shortfall_to_oracle_bps": _safe_mean(short_shortfall_oracle_bps[valid_short]),
        "avg_oracle_long_bps": _safe_mean(oracle_long_bps[valid_long]),
        "avg_oracle_short_bps": _safe_mean(oracle_short_bps[valid_short]),
        "median_long_improve_bps_filled": _safe_median(reco_long_improve_bps[valid_long & long_fill]),
        "median_short_improve_bps_filled": _safe_median(reco_short_improve_bps[valid_short & short_fill]),
        "long_samples_over8bps": int(np.sum(long_opp)),
        "short_samples_over8bps": int(np.sum(short_opp)),
        "avg_long_reco_bps_on_oppty": _safe_mean(reco_long_improve_bps[long_opp]),
        "avg_short_reco_bps_on_oppty": _safe_mean(reco_short_improve_bps[short_opp]),
        "fill_long_on_oppty": _safe_mean(long_fill[long_opp].astype(np.float64)),
        "fill_short_on_oppty": _safe_mean(short_fill[short_opp].astype(np.float64)),
    }

    output_path = args.output_path.strip()
    if not output_path:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = f"data/ensemble/metrics/entry_price_oos_{stamp}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
