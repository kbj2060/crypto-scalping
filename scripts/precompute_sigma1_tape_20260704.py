#!/usr/bin/env python3
"""Build a decision tape (schema-compatible with replay_omega6_v2_variants_20260704.run_variant)
from the trained Sigma1 GRU bundle, covering context + validation (2025-10..12) + OOS
(2026-01..02-28, data limit).

Causality: the prediction at bar i uses the trailing WINDOW-bar feature window ending at i,
built from year_oos 2025+2026 features concatenated chronologically (2025-12-31 23:55 ->
2026-01-01 00:00 is a real contiguous market boundary, unlike the 2024->2025 training-file
boundary, so windows may span it). fallback_* columns are zeroed -- Sigma1 is a single model,
no fallback chain.
"""

from __future__ import annotations

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

from train_sigma1_seq_barrier_20260704 import FEATURE_FILES, Sigma1GRU, gather_windows  # noqa: E402

BUNDLE_PATH = ROOT / "tmp/causal_regen_20260516/sigma1_seq_barrier_20260704/sigma1_bundle.pt"
OUT_PATH = ROOT / "tmp/causal_regen_20260516/sigma1_decision_tape_20260704/tape.parquet"
CONTEXT_START = pd.Timestamp("2025-09-28")  # a few days of context before VAL_START for persistence warm-up
DEFAULT_THRESHOLD = 0.45


def _atr_pct(frame: pd.DataFrame, window: int = 192) -> np.ndarray:
    high = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=np.float64)
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=np.float64)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    atr = pd.Series(tr).rolling(window=window, min_periods=1).mean().to_numpy(dtype=np.float64)
    return atr / np.maximum(close, 1e-12)


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torch.load(BUNDLE_PATH, map_location="cpu", weights_only=False)
    cols = list(bundle["feature_cols"])
    window = int(bundle["window"])
    mean = np.asarray(bundle["mean"], dtype=np.float32)
    std = np.asarray(bundle["std"], dtype=np.float32)
    model = Sigma1GRU(len(cols)).to(device)
    model.load_state_dict(bundle["state_dict"])
    model.eval()

    frames = []
    for year in (2025, 2026):
        f = pd.read_csv(FEATURE_FILES[year], parse_dates=["timestamp"], low_memory=False)
        f = f.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        frames.append(f)
    combined = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)

    feat = combined[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    feat_z = np.clip((feat - mean[None, :]) / std[None, :], -10.0, 10.0).astype(np.float32)
    atr = _atr_pct(combined)

    out_mask = (combined["timestamp"] >= CONTEXT_START).to_numpy()
    out_idx = np.flatnonzero(out_mask)
    out_idx = out_idx[out_idx >= window - 1]

    dir_probs = np.zeros((len(out_idx), 3), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(out_idx), 4096):
            bidx = out_idx[start : start + 4096]
            xb = torch.from_numpy(gather_windows(feat_z, bidx, window)).to(device)
            probs = torch.softmax(model(xb), dim=-1).cpu().numpy().astype(np.float64)
            dir_probs[start : start + len(bidx)] = probs

    sub = combined.iloc[out_idx].reset_index(drop=True)
    n = len(sub)
    dir_action = dir_probs.argmax(axis=1)
    qual_for_action = np.where(dir_action > 0, dir_probs[np.arange(n), dir_action], dir_probs[:, 0])
    final_action = np.where((dir_action != 0) & (qual_for_action >= DEFAULT_THRESHOLD), dir_action, 0)
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
            "atr_pct": atr[out_idx],
            "primary_action": final_action,
            "primary_side": side,
            "primary_expert": "sigma1",
            "primary_route_confidence": 1.0,
            "primary_route_margin": 1.0,
            "primary_dir_p_cash": dir_probs[:, 0],
            "primary_dir_p_long": dir_probs[:, 1],
            "primary_dir_p_short": dir_probs[:, 2],
            "primary_quality_p_cash": dir_probs[:, 0],
            "primary_quality_p_long": dir_probs[:, 1],
            "primary_quality_p_short": dir_probs[:, 2],
            "primary_quality_score": np.where(final_action != 0, qual_for_action, 0.0),
            "primary_confidence": dir_probs.max(axis=1),
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
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"wrote {len(out)} rows to {OUT_PATH} ({out['timestamp'].min()}..{out['timestamp'].max()})", flush=True)
    print(f"primary_side nonzero pct at default threshold: {(out['primary_side'] != 0).mean():.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
