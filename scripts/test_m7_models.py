#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
for _p in (_ROOT_DIR, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ensemble.seven_model_ensemble import SevenModelEnsemble
from features.engineering import FeatureEngineer
from strategies.elite_builder import EliteSignals, row_to_market_row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end smoke/integrity test for M7 models")
    p.add_argument("--split-dir", default="data/splits/year_oos")
    p.add_argument("--year", type=int, default=2025)
    p.add_argument("--batch-tail", type=int, default=4096)
    p.add_argument("--live-tail", type=int, default=1024)
    return p.parse_args()


def _assert_no_nan(df: pd.DataFrame, name: str) -> None:
    bad = [c for c in df.columns if df[c].isna().any()]
    if bad:
        raise RuntimeError(f"{name}: output contains NaN columns: {', '.join(bad[:10])}")


def _add_pred_conf_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = pd.to_numeric(out["close"], errors="coerce").ffill().fillna(0.0)
    r = close.pct_change().fillna(0.0)
    base_conf = r.abs().rolling(12, min_periods=1).mean().clip(0.0, 1.0)
    out["pred_patchtst"] = np.tanh(r * 150.0)
    out["conf_patchtst"] = base_conf
    out["pred_chronos"] = np.tanh(r.rolling(3, min_periods=1).mean() * 180.0)
    out["conf_chronos"] = base_conf.rolling(3, min_periods=1).mean().clip(0.0, 1.0)
    out["pred_tide"] = np.tanh(r.rolling(6, min_periods=1).mean() * 220.0)
    out["conf_tide"] = base_conf.rolling(6, min_periods=1).mean().clip(0.0, 1.0)
    return out


def _inject_elite_signals_tail(df: pd.DataFrame, tail: int) -> pd.DataFrame:
    out = df.copy()
    ext = EliteSignals()
    start = max(0, len(out) - int(tail))
    records = out.to_dict("records")
    smf = pd.to_numeric(out.get("smart_money_flow", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0)
    smf_std = smf.rolling(window=576, min_periods=10).std().fillna(smf.expanding(min_periods=1).std()).fillna(1.0)
    for i in range(start, len(records)):
        cur = row_to_market_row(records[i])
        prev = row_to_market_row(records[i - 1]) if i > 0 else cur
        _smf_std = float(smf_std.iloc[i])
        if not np.isfinite(_smf_std) or _smf_std <= 0.0:
            _smf_std = 1e-8
        sigs = ext.compute_all(current=cur, prev=prev, smf_std=_smf_std)
        for k, v in sigs.items():
            if isinstance(k, str) and k.startswith("sig_"):
                out.at[i, k] = float(v)
    return out


def main() -> int:
    args = parse_args()
    split_dir = Path(args.split_dir)
    feat_path = split_dir / f"training_features_{args.year}.csv"
    rl_base_path = split_dir / f"rl_base_{args.year}.csv"
    if not feat_path.exists() or not rl_base_path.exists():
        raise FileNotFoundError(f"missing split files: {feat_path} / {rl_base_path}")

    print("[TEST] loading M7 ensemble")
    m7 = SevenModelEnsemble(strict=True)

    print("[TEST] batch inference on year_oos merged frame")
    feat_df = pd.read_csv(feat_path)
    rl_df = pd.read_csv(rl_base_path)
    for df in (feat_df, rl_df):
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    extra = [c for c in feat_df.columns if c not in rl_df.columns and c != "timestamp"]
    work = rl_df.merge(feat_df[["timestamp"] + extra], on="timestamp", how="left")
    work = _add_pred_conf_signals(work)
    work = _inject_elite_signals_tail(work, tail=max(args.batch_tail, args.live_tail))
    batch_in = work.tail(int(args.batch_tail)).copy()
    batch_out = m7.predict_batch(batch_in)
    if batch_out.empty:
        raise RuntimeError("batch output is empty")
    _assert_no_nan(batch_out, "batch")
    print(f"[OK] batch rows={len(batch_out)} cols={len(batch_out.columns)}")

    print("[TEST] live-like inference on local test feed")
    eth = pd.read_csv("data/test/eth_test_data.csv")
    btc = pd.read_csv("data/test/btc_test_data.csv")
    live = FeatureEngineer().process(eth, btc)
    live = _add_pred_conf_signals(live)
    live = _inject_elite_signals_tail(live, tail=int(args.live_tail))
    last = m7.predict_last(live.tail(max(512, int(args.live_tail))))
    if not last:
        raise RuntimeError("predict_last returned empty")
    if any(not np.isfinite(float(v)) for v in last.values()):
        raise RuntimeError("predict_last contains non-finite values")
    print(f"[OK] live last keys={len(last)} action={last.get('m7_action')}")

    print("[DONE] M7 model test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
