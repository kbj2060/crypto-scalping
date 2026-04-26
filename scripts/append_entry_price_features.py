#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
for _p in (_ROOT_DIR, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ensemble.supervised.train_entry_price_model import EntryPriceBrain


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Append entry/tp/sl features to RL csv")
    p.add_argument("--rl-path", required=True)
    p.add_argument("--feature-path", required=True)
    p.add_argument("--model-path", default="data/ensemble/supervised/entry_price_model.json")
    p.add_argument("--output-path", default="")
    return p.parse_args()


def _load_work_frame(rl_path: str, feature_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rl_df = pd.read_csv(rl_path)
    feat_df = pd.read_csv(feature_path)
    for df in (rl_df, feat_df):
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    work_df = rl_df.copy()
    extra_cols = [c for c in feat_df.columns if c not in work_df.columns and c != "timestamp"]
    if extra_cols:
        work_df = work_df.merge(feat_df[["timestamp"] + extra_cols], on="timestamp", how="left")
    return rl_df, work_df


def _trend_dir_from_rl(df: pd.DataFrame) -> np.ndarray:
    xgb_cols = ["m7_trend_xgb_dn", "m7_trend_xgb_fl", "m7_trend_xgb_up"]
    if all(c in df.columns for c in xgb_cols):
        xgb_probs = df[xgb_cols].to_numpy(dtype=np.float64)
        return np.argmax(np.nan_to_num(xgb_probs, nan=1.0 / 3.0), axis=1)
    return np.ones(len(df), dtype=np.int64)


def main() -> int:
    args = parse_args()
    rl_df, work_df = _load_work_frame(args.rl_path, args.feature_path)
    brain = EntryPriceBrain.load(args.model_path)
    preds = brain.predict_from_df(work_df)

    close = pd.to_numeric(work_df["close"], errors="coerce").ffill().bfill().to_numpy(dtype=np.float64)
    close = np.maximum(close, 1e-8)
    q10 = pd.to_numeric(rl_df.get("m7_q10", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    q90 = pd.to_numeric(rl_df.get("m7_q90", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    action = pd.to_numeric(rl_df.get("m7_action", 0.0), errors="coerce").fillna(0.0).round().clip(-1, 1).to_numpy(dtype=np.int64)
    direction = _trend_dir_from_rl(rl_df)

    entry_long_offset = np.clip(np.asarray(preds["entry_long_offset"], dtype=np.float64), -0.02, 0.0)
    entry_short_offset = np.clip(np.asarray(preds["entry_short_offset"], dtype=np.float64), 0.0, 0.02)
    entry_long_price = close * (1.0 + entry_long_offset)
    entry_short_price = close * (1.0 + entry_short_offset)

    tp_floor = 8e-4
    sl_floor = 6e-4
    ref_side = np.where(action != 0, action, np.where(direction == 2, 1, np.where(direction == 0, -1, 0)))
    tp_offset = np.where(
        ref_side > 0,
        np.maximum(q90, tp_floor),
        np.where(ref_side < 0, np.minimum(q10, -tp_floor), 0.0),
    )
    sl_offset = np.where(
        ref_side > 0,
        np.minimum(q10, -sl_floor),
        np.where(ref_side < 0, np.maximum(q90, sl_floor), 0.0),
    )
    tp_price = close * (1.0 + tp_offset)
    sl_price = close * (1.0 + sl_offset)

    out_df = rl_df.drop(
        columns=[
            c
            for c in [
                "m7_entry_long_offset",
                "m7_entry_short_offset",
                "m7_entry_long_price",
                "m7_entry_short_price",
                "m7_tp_offset",
                "m7_sl_offset",
                "m7_tp_price",
                "m7_sl_price",
            ]
            if c in rl_df.columns
        ],
        errors="ignore",
    ).copy()
    out_df["m7_entry_long_offset"] = entry_long_offset.astype(np.float32)
    out_df["m7_entry_short_offset"] = entry_short_offset.astype(np.float32)
    out_df["m7_entry_long_price"] = entry_long_price.astype(np.float32)
    out_df["m7_entry_short_price"] = entry_short_price.astype(np.float32)
    out_df["m7_tp_offset"] = tp_offset.astype(np.float32)
    out_df["m7_sl_offset"] = sl_offset.astype(np.float32)
    out_df["m7_tp_price"] = tp_price.astype(np.float32)
    out_df["m7_sl_price"] = sl_price.astype(np.float32)

    output_path = args.output_path.strip() or args.rl_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"saved={output_path} rows={len(out_df)} cols={len(out_df.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
