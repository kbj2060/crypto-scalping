"""Stage 1 rolling-window stress test (user request, following the same methodology that made
the event gate ([[project-btc-event-gate-stage1-stable-lift-20260804]]) trustworthy): replay the
ALREADY-TRAINED (frozen, no retraining) Rho1 panel-pretrained / BTC-only / EWMA-benchmark
comparison from eval_rho1_btc_oos_20260804.py across 8 overlapping windows, to see whether the
single-split ~0.66% panel-vs-btconly edge survives across time or was itself a single-window
artifact.

IMPORTANT SCOPING CONSTRAINT (documented, not hidden): both models were trained on data strictly
before 2025-09-01 (VAL start). Unlike the event gate's rolling test (which could span the full
2024-2026 range because its GMM/IF detectors were unsupervised and not "trained on a fixed cutoff"
in the same sense), replaying THESE supervised models on pre-2025-09-01 data would be replaying
them on their own training set -- not a valid OOS check. So the 8 windows here are restricted to
the genuinely-held-out span, 2025-09-01 through the latest available data (~2026-08-04), using
4-month-wide windows with a 1-month stride:
  W1 2025-09~2026-01, W2 2025-10~2026-02, ..., W8 2026-04~2026-08
The first ~3 windows partially overlap the VAL period (2025-09-01..2025-12-31), which was used
for checkpoint/early-stopping selection -- a much milder form of look-ahead than training on it
directly, but still worth flagging separately from the pure-OOS windows (W5-W8, entirely within
2026-01-01 onward).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_rho1_panel_backbone_20260804 as base  # noqa: E402
from eval_rho1_btc_oos_20260804 import PANEL_CKPT, BTCONLY_CKPT, predict_model, pinball_np  # noqa: E402

WINDOWS = [
    ("W1", "2025-09-01", "2026-01-01"),
    ("W2", "2025-10-01", "2026-02-01"),
    ("W3", "2025-11-01", "2026-03-01"),
    ("W4", "2025-12-01", "2026-04-01"),
    ("W5", "2026-01-01", "2026-05-01"),
    ("W6", "2026-02-01", "2026-06-01"),
    ("W7", "2026-03-01", "2026-07-01"),
    ("W8", "2026-04-01", "2026-08-01"),
]


def load_btc_all():
    df = pd.read_parquet(base.FEATURES_DIR / "BTCUSDT.parquet")
    ts = df["timestamp"]
    close = df["close"].to_numpy(dtype=np.float64)
    realized_vol_288 = df["realized_vol_288"].to_numpy(dtype=np.float64)
    fwd_ret = np.log(np.roll(close, -base.HORIZON_H) / close)
    fwd_ret[-base.HORIZON_H:] = np.nan

    X = df[base.FEATURE_COLS].to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -20.0, 20.0)
    return ts, X, fwd_ret, realized_vol_288


def block_bootstrap_ci(diff: np.ndarray, block: int = 288, n_boot: int = 1000, seed: int = 0):
    n = len(diff)
    n_blocks = max(n // block, 1)
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        block_ids = rng.integers(0, n_blocks, size=n_blocks)
        sample = np.concatenate([diff[i * block:(i + 1) * block] for i in block_ids])
        boot_means[b] = sample.mean()
    return np.percentile(boot_means, [2.5, 97.5])


def main():
    print("loading BTC full series + models...", flush=True)
    ts, X, fwd_ret, realized_vol_288 = load_btc_all()
    n = len(ts)

    ckpt_panel_full = torch.load(PANEL_CKPT, map_location="cpu", weights_only=False)
    btc_sym_id = ckpt_panel_full["symbol_to_id"]["BTCUSDT"]

    rows = []
    for name, start, end in WINDOWS:
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        mask = ((ts >= start_ts) & (ts < end_ts)).to_numpy() & ~np.isnan(fwd_ret)
        idxs = np.arange(base.WINDOW_L, n - base.HORIZON_H)
        idxs = idxs[mask[idxs]]
        if len(idxs) < 500:
            print(f"{name}: too few samples ({len(idxs)}), skipping")
            continue

        windows = np.stack([X[i - base.WINDOW_L:i] for i in idxs])
        targets_raw = fwd_ret[idxs]
        vol_at_entry = realized_vol_288[idxs]
        scale = vol_at_entry * math.sqrt(base.HORIZON_H)

        pred_panel_norm, _ = predict_model(PANEL_CKPT, windows, sym_id=btc_sym_id)
        pred_btconly_norm, _ = predict_model(BTCONLY_CKPT, windows, sym_id=0)
        pred_panel_raw = pred_panel_norm * scale[:, None]
        pred_btconly_raw = pred_btconly_norm * scale[:, None]
        pred_ewma_raw = np.stack([stats.norm.ppf(q, loc=0.0, scale=scale) for q in base.QUANTILES], axis=1)

        pb_panel = pinball_np(targets_raw, pred_panel_raw, base.QUANTILES).mean()
        pb_btconly = pinball_np(targets_raw, pred_btconly_raw, base.QUANTILES).mean()
        pb_ewma = pinball_np(targets_raw, pred_ewma_raw, base.QUANTILES).mean()

        q = np.array(base.QUANTILES)[None, :]
        pb_panel_row = np.maximum(q * (targets_raw[:, None] - pred_panel_raw),
                                   (q - 1) * (targets_raw[:, None] - pred_panel_raw)).mean(axis=1)
        pb_btconly_row = np.maximum(q * (targets_raw[:, None] - pred_btconly_raw),
                                     (q - 1) * (targets_raw[:, None] - pred_btconly_raw)).mean(axis=1)
        diff = pb_panel_row - pb_btconly_row
        ci_lo, ci_hi = block_bootstrap_ci(diff)

        panel_vs_btconly_pct = 100 * (pb_panel - pb_btconly) / pb_btconly
        panel_vs_ewma_pct = 100 * (pb_panel - pb_ewma) / pb_ewma
        rows.append({
            "window": name, "start": start, "end": end, "n": len(idxs),
            "pb_panel": pb_panel, "pb_btconly": pb_btconly, "pb_ewma": pb_ewma,
            "panel_vs_btconly_pct": panel_vs_btconly_pct, "panel_vs_ewma_pct": panel_vs_ewma_pct,
            "ci_lo": ci_lo, "ci_hi": ci_hi,
            "significant": bool(ci_lo > 0 or ci_hi < 0),
        })
        print(f"{name} [{start}..{end}) n={len(idxs):6d}  panel={pb_panel:.6f} btconly={pb_btconly:.6f} "
              f"ewma={pb_ewma:.6f}  panel_vs_btconly={panel_vs_btconly_pct:+.2f}%  "
              f"panel_vs_ewma={panel_vs_ewma_pct:+.2f}%  boot_CI=[{ci_lo:.6f},{ci_hi:.6f}] "
              f"{'SIG' if (ci_lo > 0 or ci_hi < 0) else 'ns'}", flush=True)

    out = pd.DataFrame(rows)
    out_path = ROOT / "tmp/rho1_rolling_window_20260804.csv"
    out.to_csv(out_path, index=False)

    n_panel_better = (out["panel_vs_btconly_pct"] < 0).sum()
    n_sig = out["significant"].sum()
    n_sig_and_panel_better = ((out["panel_vs_btconly_pct"] < 0) & out["significant"]).sum()
    print(f"\n=== summary across {len(out)} windows ===")
    print(f"panel beats btconly (any margin): {n_panel_better}/{len(out)}")
    print(f"bootstrap-significant (either direction): {n_sig}/{len(out)}")
    print(f"significant AND panel better: {n_sig_and_panel_better}/{len(out)}")
    print(f"panel_vs_btconly_pct range: [{out['panel_vs_btconly_pct'].min():.2f}%, {out['panel_vs_btconly_pct'].max():.2f}%]")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
