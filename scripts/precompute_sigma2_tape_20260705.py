#!/usr/bin/env python3
"""Sigma2 decision tape: predictions over context + validation (2025-07-01..12-31) + soft
window (2026-01..02) + untouched fresh window (2026-03-02..06-30). One tape per seed
(--suffix seedA/seedB). Features: year_oos 2025 + regime3 overlays (May-built, identical to
training-side sources) concatenated with the extended 2026 frame parquet (overlay columns
already merged there, extension reproducibility-verified).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from train_sigma2_seq_zigzag_20260705 import MODEL_ID, OVERLAYS, STRING_COLS, Sigma2GRU, gather_windows  # noqa: E402

FEATURES_2025 = ROOT / "data/splits/year_oos/training_features_2025.csv"
EXTENDED_2026 = ROOT / "tmp/causal_regen_20260516/extended_eval_frame_nom7_20260704/frame.parquet"
CONTEXT_START = pd.Timestamp("2025-06-25")
DEFAULT_THRESHOLD = 0.45
ATR_WINDOW = 192


def _atr_pct(frame: pd.DataFrame, window: int = ATR_WINDOW) -> np.ndarray:
    high = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=np.float64)
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=np.float64)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    atr = pd.Series(tr).rolling(window=window, min_periods=1).mean().to_numpy(dtype=np.float64)
    return atr / np.maximum(close, 1e-12)


def load_2025() -> pd.DataFrame:
    frame = pd.read_csv(FEATURES_2025, parse_dates=["timestamp"], low_memory=False)
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    for _name, (dir_path, pattern) in OVERLAYS.items():
        overlay = pd.read_csv(dir_path / pattern.format(year=2025), parse_dates=["timestamp"], low_memory=False)
        cols = [c for c in overlay.columns if c != "timestamp" and c not in STRING_COLS]
        frame = frame.merge(overlay[["timestamp", *cols]], on="timestamp", how="left", validate="one_to_one")
    return frame


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", required=True)
    args = ap.parse_args()

    bundle_path = ROOT / "tmp/causal_regen_20260516" / f"{MODEL_ID}_{args.suffix}" / "sigma2_bundle.pt"
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    cols = list(bundle["feature_cols"])
    window = int(bundle["window"])
    mean = np.asarray(bundle["mean"], dtype=np.float32)
    std = np.asarray(bundle["std"], dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Sigma2GRU(len(cols)).to(device)
    model.load_state_dict(bundle["state_dict"])
    model.eval()

    f25 = load_2025()
    f26 = pd.read_parquet(EXTENDED_2026)
    f26["timestamp"] = pd.to_datetime(f26["timestamp"])
    combined = pd.concat([f25, f26], ignore_index=True, sort=False)
    combined = combined[combined["timestamp"] >= CONTEXT_START]
    combined = combined.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    missing = [c for c in cols if c not in combined.columns]
    if missing:
        raise RuntimeError(f"missing feature cols: {missing[:10]}")

    feat = combined[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    feat_z = np.clip((feat - mean[None, :]) / std[None, :], -10.0, 10.0).astype(np.float32)

    out_idx = np.arange(window - 1, len(combined), dtype=np.int64)
    probs = np.zeros((len(out_idx), 3), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(out_idx), 4096):
            bidx = out_idx[start : start + 4096]
            xb = torch.from_numpy(gather_windows(feat_z, bidx, window)).to(device)
            probs[start : start + len(bidx)] = torch.softmax(model(xb), dim=-1).cpu().numpy()

    sub = combined.iloc[out_idx].reset_index(drop=True)
    n = len(sub)
    atr = _atr_pct(combined)[out_idx]
    dir_action = probs.argmax(axis=1)
    qual = np.where(dir_action > 0, probs[np.arange(n), dir_action], probs[:, 0])
    final_action = np.where((dir_action != 0) & (qual >= DEFAULT_THRESHOLD), dir_action, 0)
    side = np.where(final_action == 1, 1, np.where(final_action == 2, -1, 0))

    out = pd.DataFrame(
        {
            "i": out_idx,
            "timestamp": sub["timestamp"],
            "open": sub["open"].astype(np.float64),
            "high": sub["high"].astype(np.float64),
            "low": sub["low"].astype(np.float64),
            "close": sub["close"].astype(np.float64),
            "jump_flag": pd.to_numeric(sub.get("jump_flag", 0.0), errors="coerce").fillna(0.0),
            "evt_tail_flag": pd.to_numeric(sub.get("evt_tail_flag", 0.0), errors="coerce").fillna(0.0),
            "jump_z": pd.to_numeric(sub.get("jump_z", 0.0), errors="coerce").fillna(0.0),
            "atr_pct": atr,
            "primary_action": final_action,
            "primary_side": side,
            "primary_expert": "sigma2",
            "primary_route_confidence": 1.0,
            "primary_route_margin": 1.0,
            "primary_dir_p_cash": probs[:, 0],
            "primary_dir_p_long": probs[:, 1],
            "primary_dir_p_short": probs[:, 2],
            "primary_quality_p_cash": probs[:, 0],
            "primary_quality_p_long": probs[:, 1],
            "primary_quality_p_short": probs[:, 2],
            "primary_quality_score": np.where(final_action != 0, qual, 0.0),
            "primary_confidence": probs.max(axis=1),
            "fallback_action": 0,
            "fallback_side": 0,
            "fallback_expert": "none",
            "fallback_route_confidence": 0.0,
            "fallback_route_margin": 0.0,
            "fallback_dir_p_cash": 1.0,
            "fallback_dir_p_long": 0.0,
            "fallback_dir_p_short": 0.0,
            "fallback_quality_p_cash": 1.0,
            "fallback_quality_p_long": 0.0,
            "fallback_quality_p_short": 0.0,
            "fallback_quality_score": 0.0,
            "fallback_confidence": 0.0,
        }
    )
    out_path = ROOT / "tmp/causal_regen_20260516" / f"sigma2_tape_{args.suffix}_20260705" / "tape.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"wrote {len(out)} rows to {out_path} ({out['timestamp'].min()}..{out['timestamp'].max()})", flush=True)
    print(f"primary_side nonzero pct: {(out['primary_side'] != 0).mean():.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
