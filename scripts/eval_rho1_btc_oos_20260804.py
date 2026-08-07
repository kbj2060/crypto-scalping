"""Stage 1 decisive comparison (docs/btc_panel_crossasset_architecture_design_20260804.md):
BTC OOS (2026-01-01..2026-04-01, canonical split) pinball loss for three candidates on the SAME
task (predict the 7-quantile distribution of the 4h-forward log return):

1. Rho1 panel-pretrained backbone (data/panel/ckpt/rho1_panel_backbone_best.pt) -- trained on all
   60 pooled symbols.
2. Same architecture trained on BTCUSDT alone (data/panel/ckpt/rho1_btconly_backbone_best.pt) --
   isolates whether pooling (H2) helps at all, holding architecture/hyperparameters fixed.
3. A zero-parameter EWMA/causal-vol Gaussian benchmark: assume the forward return is
   Normal(0, sigma_t^2) with sigma_t = BTC's own trailing realized_vol_288 (already an input
   feature, already causal) scaled to the horizon -- the standard "no-model" volatility
   benchmark this literature (2507.07296) uses TSFMs to try to beat.

All three are scored on the SAME raw (non-normalized) forward log-return target in price-move
space, using pinball loss at the same 7 quantile levels, so they are directly comparable.
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

PANEL_CKPT = base.CKPT_DIR / "rho1_panel_backbone_best.pt"
BTCONLY_CKPT = base.CKPT_DIR / "rho1_btconly_backbone_best.pt"


def pinball_np(target: np.ndarray, pred: np.ndarray, quantiles: list[float]) -> np.ndarray:
    """target: (n,), pred: (n, n_q) -> per-quantile mean pinball loss, shape (n_q,)"""
    q = np.array(quantiles)[None, :]
    errors = target[:, None] - pred
    loss = np.maximum(q * errors, (q - 1) * errors)
    return loss.mean(axis=0)


def load_btc_oos_windows():
    df = pd.read_parquet(base.FEATURES_DIR / "BTCUSDT.parquet")
    ts = df["timestamp"]
    close = df["close"].to_numpy(dtype=np.float64)
    realized_vol_288 = df["realized_vol_288"].to_numpy(dtype=np.float64)
    fwd_ret = np.log(np.roll(close, -base.HORIZON_H) / close)
    fwd_ret[-base.HORIZON_H:] = np.nan

    X = df[base.FEATURE_COLS].to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -20.0, 20.0)

    oos_end = pd.Timestamp("2026-04-01")  # canonical OOS window used throughout this repo
    mask = ((ts >= base.OOS_START) & (ts < oos_end)).to_numpy() & ~np.isnan(fwd_ret)
    n = len(df)
    idxs = np.arange(base.WINDOW_L, n - base.HORIZON_H)
    idxs = idxs[mask[idxs]]

    windows = np.stack([X[i - base.WINDOW_L:i] for i in idxs])
    targets_raw = fwd_ret[idxs]  # raw price-move log return
    vol_at_entry = realized_vol_288[idxs]  # causal, known at prediction time
    return windows, targets_raw, vol_at_entry, ts.iloc[idxs].reset_index(drop=True)


def predict_model(ckpt_path: Path, windows: np.ndarray, sym_id: int = None, n_quantiles: int = None) -> np.ndarray:
    ckpt = torch.load(ckpt_path, map_location=base.DEVICE, weights_only=False)
    n_symbols = len(ckpt["symbol_to_id"])
    if n_quantiles is None:
        n_quantiles = len(ckpt.get("quantiles", base.QUANTILES))
    model = base.Rho1Backbone(n_features=len(base.FEATURE_COLS), n_symbols=n_symbols,
                               n_quantiles=n_quantiles).to(base.DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    if sym_id is None:
        sym_id = ckpt["symbol_to_id"]["BTCUSDT"]

    preds = []
    bs = 2048
    with torch.no_grad():
        for i in range(0, len(windows), bs):
            batch = torch.from_numpy(windows[i:i + bs]).to(base.DEVICE)
            sid = torch.full((len(batch),), sym_id, dtype=torch.long, device=base.DEVICE)
            pred = model(batch, sid).cpu().numpy()
            preds.append(pred)
    return np.concatenate(preds, axis=0), ckpt


def main():
    print("loading BTC OOS windows...", flush=True)
    windows, targets_raw, vol_at_entry, ts = load_btc_oos_windows()
    print(f"n_oos_windows={len(windows)}, range {ts.iloc[0]}..{ts.iloc[-1]}", flush=True)

    # --- 1. Rho1 panel-pretrained ---
    ckpt_panel_full = torch.load(PANEL_CKPT, map_location="cpu", weights_only=False)
    btc_sym_id = ckpt_panel_full["symbol_to_id"]["BTCUSDT"]
    pred_panel_norm, _ = predict_model(PANEL_CKPT, windows, sym_id=btc_sym_id)
    scale = vol_at_entry * math.sqrt(base.HORIZON_H)
    pred_panel_raw = pred_panel_norm * scale[:, None]

    # --- 2. BTC-only same-architecture baseline ---
    pred_btconly_norm, _ = predict_model(BTCONLY_CKPT, windows, sym_id=0)
    pred_btconly_raw = pred_btconly_norm * scale[:, None]

    # --- 3. EWMA/causal-vol Gaussian benchmark (zero-parameter, "no model") ---
    ewma_sigma = vol_at_entry * math.sqrt(base.HORIZON_H)
    pred_ewma_raw = np.stack([stats.norm.ppf(q, loc=0.0, scale=ewma_sigma) for q in base.QUANTILES], axis=1)

    results = {}
    for name, pred in [("rho1_panel_pretrained", pred_panel_raw),
                       ("rho1_btconly_baseline", pred_btconly_raw),
                       ("ewma_gaussian_benchmark", pred_ewma_raw)]:
        per_q = pinball_np(targets_raw, pred, base.QUANTILES)
        results[name] = {"mean_pinball": float(per_q.mean()), "per_quantile": per_q.tolist()}

    print("\n=== BTC OOS (2026-01-01..2026-04-01) pinball loss, raw price-move space ===")
    print(f"{'model':30s} {'mean_pinball':>14s}  per-quantile [{','.join(str(q) for q in base.QUANTILES)}]")
    for name, r in results.items():
        pq = ", ".join(f"{v:.5f}" for v in r["per_quantile"])
        print(f"{name:30s} {r['mean_pinball']:14.6f}  [{pq}]")

    panel_vs_btconly = 100 * (results["rho1_panel_pretrained"]["mean_pinball"] - results["rho1_btconly_baseline"]["mean_pinball"]) / results["rho1_btconly_baseline"]["mean_pinball"]
    panel_vs_ewma = 100 * (results["rho1_panel_pretrained"]["mean_pinball"] - results["ewma_gaussian_benchmark"]["mean_pinball"]) / results["ewma_gaussian_benchmark"]["mean_pinball"]
    btconly_vs_ewma = 100 * (results["rho1_btconly_baseline"]["mean_pinball"] - results["ewma_gaussian_benchmark"]["mean_pinball"]) / results["ewma_gaussian_benchmark"]["mean_pinball"]
    print(f"\npanel vs btconly: {panel_vs_btconly:+.2f}% (negative = panel better)")
    print(f"panel vs ewma:    {panel_vs_ewma:+.2f}% (negative = panel beats zero-param benchmark)")
    print(f"btconly vs ewma:  {btconly_vs_ewma:+.2f}% (negative = btconly beats zero-param benchmark)")

    # --- bootstrap CI on the per-window mean-pinball difference, to check whether the
    # improvement above is distinguishable from resampling noise (block bootstrap, since
    # adjacent 5m-bar windows are highly autocorrelated -- a naive iid bootstrap would
    # understate the true uncertainty) ---
    rng = np.random.default_rng(0)
    block = 288  # 24h blocks
    n = len(targets_raw)
    n_blocks = n // block
    pb_panel_per_row = np.maximum(np.array(base.QUANTILES)[None, :] * (targets_raw[:, None] - pred_panel_raw),
                                   (np.array(base.QUANTILES)[None, :] - 1) * (targets_raw[:, None] - pred_panel_raw)).mean(axis=1)
    pb_btconly_per_row = np.maximum(np.array(base.QUANTILES)[None, :] * (targets_raw[:, None] - pred_btconly_raw),
                                     (np.array(base.QUANTILES)[None, :] - 1) * (targets_raw[:, None] - pred_btconly_raw)).mean(axis=1)
    diff = pb_panel_per_row - pb_btconly_per_row  # negative = panel better, per-row
    boot_means = []
    for _ in range(2000):
        block_ids = rng.integers(0, n_blocks, size=n_blocks)
        sample = np.concatenate([diff[b * block:(b + 1) * block] for b in block_ids])
        boot_means.append(sample.mean())
    boot_means = np.array(boot_means)
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    print(f"\nblock-bootstrap 95% CI on (panel - btconly) mean pinball diff: [{ci_lo:.6f}, {ci_hi:.6f}] "
          f"(entirely negative = panel reliably better; straddles 0 = not distinguishable from noise)")


if __name__ == "__main__":
    main()
