#!/usr/bin/env python3
"""Research probe (2026-08-10, literature follow-up idea C): Chronos time-series foundation
model, zero-shot, on ETH 5m log-price -- the third literature lead from the user's question
about recent papers crossing the train/OOS generalization gap. Chronos is open-weight (no
TabPFN-style license wall) and is the one TSFM the 2026 "Re(Visiting) TSFMs in Finance" paper
found shows genuine (if economically limited) improvement from fine-tuning on financial data.

This script does ZERO-SHOT only (no fine-tuning) as the first, cheaper test: feed a trailing
context window of ETH's own log-price history, get Chronos's probabilistic forecast for the next
`HORIZON` bars, and extract two signals per literature convention:
  median direction: sign(median forecast at horizon end - last context value)
  quantile skew:    compare the forecast distribution's upper vs lower tail magnitude (matches
                    tonight's own idea #14 quantile-regression-skew construction, but from a
                    pretrained foundation model instead of a model trained from scratch here)
Evaluated with this project's own corrected methodology: always_long/always_short baseline,
per-bar tp_move/sl_move payoff, on genuinely held-out DEV/VAL/OOS splits (subsampled every
`STRIDE` bars purely for Chronos inference-time tractability, not for any statistical reason).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.architecture_workbench import effect_size_report  # noqa: E402

ETH_PRICE_PATH = ROOT / "data/eth_5m_1year.csv"
ETH_LABEL_PATH = ROOT / "data/splits/year_oos/eth_5m_tripbarrier_tradeoutcome_labels_20260808.parquet"
CONTEXT_LEN = 256
HORIZON = 48
NUM_SAMPLES = 30
STRIDE = 12  # evaluate every 12th eligible bar (1 hour spacing) -- inference-tractability only
BATCH_SIZE = 32
ROUND_TRIP_COST = 0.0006
DEV_START, DEV_END = "2025-07-01 00:00:00", "2025-08-31 23:55:00"
VAL_START, VAL_END = "2025-09-01 00:00:00", "2025-12-31 23:55:00"
OOS_START = "2026-01-01 00:00:00"


def load_merged() -> pd.DataFrame:
    price = pd.read_csv(ETH_PRICE_PATH, usecols=["timestamp", "close"], parse_dates=["timestamp"])
    labels = pd.read_parquet(ETH_LABEL_PATH)
    labels["timestamp"] = pd.to_datetime(labels["timestamp"])
    merged = price.merge(labels, on="timestamp", how="inner", validate="one_to_one")
    merged = merged.dropna(subset=["tp_move", "sl_move"]).sort_values("timestamp").reset_index(drop=True)
    return merged


def bar_payoff(cls, outcome, tp, sl, cost):
    pl = np.where(outcome == 1, tp, np.where(outcome == 2, -sl, 0.0))
    ps = np.where(outcome == 2, tp, np.where(outcome == 1, -sl, 0.0))
    r = np.where(cls == 1, pl, np.where(cls == 2, ps, 0.0))
    r = np.where(cls != 0, r - cost, 0.0)
    return r


def favored_direction_payoff(outcome, tp, sl, cost):
    n = len(outcome)
    long_net = bar_payoff(np.ones(n), outcome, tp, sl, cost)
    short_net = bar_payoff(np.full(n, 2), outcome, tp, sl, cost)
    return (long_net, "always_long") if long_net.sum() >= short_net.sum() else (short_net, "always_short")


def run_chronos_batch(pipe, contexts: list[np.ndarray], device: str):
    ctx_tensor = torch.tensor(np.stack(contexts), dtype=torch.float32)
    forecast = pipe.predict(ctx_tensor, prediction_length=HORIZON, num_samples=NUM_SAMPLES)
    return forecast.numpy()  # (batch, num_samples, horizon)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    pipe = ChronosPipeline.from_pretrained("amazon/chronos-t5-small", device_map=device, torch_dtype=torch.bfloat16)

    merged = load_merged()
    log_close = np.log(merged["close"].to_numpy())
    ts = merged["timestamp"]
    outcome_all, tp_all, sl_all = merged["trade_outcome_action"].to_numpy(), merged["tp_move"].to_numpy(), merged["sl_move"].to_numpy()

    def eligible_indices(start, end):
        mask = (ts >= start) & (ts <= end)
        idx = np.where(mask.to_numpy())[0]
        idx = idx[idx >= CONTEXT_LEN]
        idx = idx[idx < len(merged) - HORIZON]
        return idx[::STRIDE]

    for split_name, start, end in [("DEV", DEV_START, DEV_END), ("VAL", VAL_START, VAL_END), ("OOS", OOS_START, str(ts.max()))]:
        idx = eligible_indices(start, end)
        print(f"\n=== {split_name}: {len(idx)} subsampled bars (stride={STRIDE}) ===")
        t0 = time.time()
        median_cls = np.zeros(len(idx), dtype=np.int64)
        skew_cls = np.zeros(len(idx), dtype=np.int64)
        for b in range(0, len(idx), BATCH_SIZE):
            batch_idx = idx[b:b + BATCH_SIZE]
            contexts = [log_close[i - CONTEXT_LEN:i] for i in batch_idx]
            forecast = run_chronos_batch(pipe, contexts, device)  # (batch, samples, horizon)
            last_ctx = log_close[batch_idx]
            median_end = np.median(forecast[:, :, -1], axis=1)
            median_cls[b:b + BATCH_SIZE] = np.where(median_end > last_ctx, 1, np.where(median_end < last_ctx, 2, 0))
            q85 = np.quantile(forecast[:, :, -1], 0.85, axis=1)
            q15 = np.quantile(forecast[:, :, -1], 0.15, axis=1)
            up_mag = np.maximum(q85 - last_ctx, 0)
            down_mag = np.maximum(last_ctx - q15, 0)
            skew_cls[b:b + BATCH_SIZE] = np.where(up_mag > down_mag, 1, np.where(down_mag > up_mag, 2, 0))
            if b % (BATCH_SIZE * 10) == 0:
                print(f"  {b}/{len(idx)} done, {time.time() - t0:.1f}s elapsed")
        print(f"  total: {time.time() - t0:.1f}s for {len(idx)} series")

        outcome, tp, sl = outcome_all[idx], tp_all[idx], sl_all[idx]
        baseline_payoff, baseline_name = favored_direction_payoff(outcome, tp, sl, ROUND_TRIP_COST)
        median_payoff = bar_payoff(median_cls, outcome, tp, sl, ROUND_TRIP_COST)
        skew_payoff = bar_payoff(skew_cls, outcome, tp, sl, ROUND_TRIP_COST)
        print(f"  baseline({baseline_name}) sum={baseline_payoff.sum():.4f}  n={len(idx)}")
        print(f"  Chronos median-direction: n_trades={(median_cls != 0).sum()}  sum={median_payoff.sum():.4f}")
        print(f"  Chronos quantile-skew:    n_trades={(skew_cls != 0).sum()}  sum={skew_payoff.sum():.4f}")
        if (median_cls != 0).sum() >= 3:
            r = effect_size_report(median_payoff[median_cls != 0], baseline_payoff, label_a="chronos_median", label_b="baseline")
            print(f"  effect size (chronos_median vs baseline): welch_t={r['welch_t']:.4f} p_mean={r['p_mean']:.4f}")
        if (skew_cls != 0).sum() >= 3:
            r2 = effect_size_report(skew_payoff[skew_cls != 0], baseline_payoff, label_a="chronos_skew", label_b="baseline")
            print(f"  effect size (chronos_skew vs baseline): welch_t={r2['welch_t']:.4f} p_mean={r2['p_mean']:.4f}")


if __name__ == "__main__":
    main()
