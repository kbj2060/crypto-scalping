#!/usr/bin/env python3
"""Research probe (2026-08-09, overnight loop idea #8): binary skip-filter reformulation.

Ideas #6-#7 failed because a 3-way reclassification model conflates "skip a bad bar" (filtering
-- the one thing this project has shown can work, e.g. h48qual) with "re-pick the opposite
direction" (prediction -- never works here). This probe isolates the filtering question: given
the period's known favored direction (always_short for ETH, per
project-baseline-must-be-always-long-short-not-zero-20260809), train a BINARY classifier that
predicts whether THIS bar's favored-direction bet will WIN or LOSE, using a broad feature set
(BTC lead-lag returns + OFI + price-volume signature terms, i.e. every feature family tried
tonight, combined instead of tested in isolation -- if none of them carry information alone,
maybe a genuinely different question about the SAME features does). Skip the bottom-quantile
bars by predicted win-probability; keep the rest on the favored direction.

Gates:
  1. falsification_audit: real skip-filter's DEV sum vs a null built by skipping a RANDOM,
     same-size subset of bars instead of the model-chosen ones, repeated many times.
  2. Filtered sum must beat the always-favored-direction baseline sum on DEV.
  3. VAL must also beat baseline.
  Only then: partial OOS.
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.selection_stats import falsification_audit  # noqa: E402

ETH_PRICE_PATH = ROOT / "data/eth_5m_1year.csv"
ETH_LABEL_PATH = ROOT / "data/splits/year_oos/eth_5m_tripbarrier_tradeoutcome_labels_20260808.parquet"
BTC_PRICE_PATH = ROOT / "data/btc_5m_1year.csv"
WINDOWS = [3, 6, 12, 24, 48]
SKIP_QUANTILE = 0.30  # skip the worst 30% of bars by predicted win-probability
ROUND_TRIP_COST = 0.0006
TRAIN_END = "2025-06-30 23:55:00"
DEV_START, DEV_END = "2025-07-01 00:00:00", "2025-08-31 23:55:00"
VAL_START, VAL_END = "2025-09-01 00:00:00", "2025-12-31 23:55:00"
OOS_START = "2026-01-01 00:00:00"


def load_merged() -> pd.DataFrame:
    eth_price = pd.read_csv(ETH_PRICE_PATH, usecols=["timestamp", "close", "volume", "taker_buy_base"], parse_dates=["timestamp"])
    btc_price = pd.read_csv(BTC_PRICE_PATH, usecols=["timestamp", "close"], parse_dates=["timestamp"]).rename(columns={"close": "btc_close"})
    labels = pd.read_parquet(ETH_LABEL_PATH)
    labels["timestamp"] = pd.to_datetime(labels["timestamp"])
    merged = eth_price.merge(btc_price, on="timestamp", how="inner").merge(labels, on="timestamp", how="inner", validate="one_to_one")
    merged = merged.dropna(subset=["tp_move", "sl_move"]).sort_values("timestamp").reset_index(drop=True)
    return merged


def build_features(df: pd.DataFrame) -> list[str]:
    cols = []
    log_eth = np.log(df["close"].to_numpy())
    log_btc = np.log(df["btc_close"].to_numpy())
    ofi = np.where(df["volume"] > 0, 2 * df["taker_buy_base"] / df["volume"] - 1.0, 0.0)
    for w in WINDOWS:
        eth_ret = log_eth - np.roll(log_eth, w); eth_ret[:w] = np.nan
        btc_ret = log_btc - np.roll(log_btc, w); btc_ret[:w] = np.nan
        ofi_mean = pd.Series(ofi).rolling(w, min_periods=w).mean().to_numpy()
        df[f"eth_ret_w{w}"], df[f"btc_ret_w{w}"], df[f"ofi_w{w}"] = eth_ret, btc_ret, ofi_mean
        cols += [f"eth_ret_w{w}", f"btc_ret_w{w}", f"ofi_w{w}"]
    return cols


def favored_direction_and_payoff(outcome, tp, sl, cost):
    n = len(outcome)
    long_net = np.where(outcome == 1, tp, np.where(outcome == 2, -sl, 0.0)) - cost
    short_net = np.where(outcome == 2, tp, np.where(outcome == 1, -sl, 0.0)) - cost
    if long_net.sum() >= short_net.sum():
        return long_net, "always_long", 1
    return short_net, "always_short", 2


def win_label_for_direction(outcome: np.ndarray, direction: int) -> np.ndarray:
    """1 if betting `direction` wins this bar's barrier race outright, 0 otherwise
    (loss or timeout are both 0 -- a binary 'was this a clean win' target)."""
    return (outcome == direction).astype(int)


def main() -> None:
    merged = load_merged()
    feat_cols = build_features(merged)
    merged = merged.dropna(subset=feat_cols).reset_index(drop=True)
    ts = merged["timestamp"]
    print(f"Merged rows: {len(merged)} ({ts.min()} .. {ts.max()})")

    train_df = merged[ts <= TRAIN_END]
    dev_df = merged[(ts >= DEV_START) & (ts <= DEV_END)]
    val_df = merged[(ts >= VAL_START) & (ts <= VAL_END)]
    oos_df = merged[ts >= OOS_START]
    print(f"TRAIN={len(train_df)} DEV={len(dev_df)} VAL={len(val_df)} OOS(partial)={len(oos_df)}")

    train_outcome, train_tp, train_sl = train_df["trade_outcome_action"].to_numpy(), train_df["tp_move"].to_numpy(), train_df["sl_move"].to_numpy()
    _, _, train_direction = favored_direction_and_payoff(train_outcome, train_tp, train_sl, ROUND_TRIP_COST)
    print(f"Favored direction on TRAIN: {'long' if train_direction == 1 else 'short'}")

    train_win = win_label_for_direction(train_outcome, train_direction)
    model = lgb.LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.05,
                               min_child_samples=200, random_state=270705, verbosity=-1)
    model.fit(train_df[feat_cols], train_win)

    def evaluate(split_df, name):
        # Use the TRAIN-frozen favored direction on every split -- re-picking long/short per
        # split with that split's own hindsight outcome (as ideas #5-#7 did for the *baseline
        # only*, harmlessly there) would be inconsistent here with a model trained to predict
        # win-probability for one specific fixed direction.
        outcome, tp, sl = split_df["trade_outcome_action"].to_numpy(), split_df["tp_move"].to_numpy(), split_df["sl_move"].to_numpy()
        direction = train_direction
        baseline_name = "always_long" if direction == 1 else "always_short"
        baseline_payoff = (np.where(outcome == 1, tp, np.where(outcome == 2, -sl, 0.0)) - ROUND_TRIP_COST
                            if direction == 1 else
                            np.where(outcome == 2, tp, np.where(outcome == 1, -sl, 0.0)) - ROUND_TRIP_COST)
        win_prob = model.predict_proba(split_df[feat_cols])[:, 1]
        threshold = np.quantile(win_prob, SKIP_QUANTILE)
        keep = win_prob > threshold
        filtered_payoff = np.where(keep, baseline_payoff, 0.0)
        return baseline_payoff, baseline_name, filtered_payoff, keep, win_prob

    dev_baseline, dev_baseline_name, dev_filtered, dev_keep, dev_prob = evaluate(dev_df, "DEV")
    print(f"\nDEV baseline({dev_baseline_name}) sum={dev_baseline.sum():.4f}  "
          f"filtered sum={dev_filtered.sum():.4f}  kept={int(dev_keep.sum())}/{len(dev_keep)}")

    rng = np.random.default_rng(20260809)
    n_null = 1000
    n_skip = int((~dev_keep).sum())
    null_sums = np.empty(n_null)
    for i in range(n_null):
        idx = rng.choice(len(dev_baseline), size=n_skip, replace=False)
        rand_keep = np.ones(len(dev_baseline), dtype=bool)
        rand_keep[idx] = False
        null_sums[i] = np.where(rand_keep, dev_baseline, 0.0).sum()
    real_sum = dev_filtered.sum()
    percentile = float((null_sums < real_sum).mean())
    print(f"\n=== GATE 1: real filtered sum ({real_sum:.4f}) vs {n_null} random-same-size-skip draws "
          f"(mean={null_sums.mean():.4f}, p95={np.percentile(null_sums, 95):.4f}) -> percentile={percentile:.3f} ===")
    if percentile < 0.95:
        print("\n[STOP] GATE 1 FAILED -- the model's chosen skips are not better than skipping "
              "the same NUMBER of bars at random. No VAL/OOS spent.")
        return
    print("\n[GATE 1 PASSED]")

    print(f"\n=== GATE 2: DEV filtered sum ({dev_filtered.sum():.4f}) vs baseline ({dev_baseline.sum():.4f}) ===")
    if dev_filtered.sum() <= dev_baseline.sum():
        print("\n[STOP] GATE 2 FAILED -- filtering does not beat just always betting the favored side.")
        return
    print("\n[GATE 2 PASSED] Proceeding to VAL.")

    val_baseline, val_baseline_name, val_filtered, val_keep, _ = evaluate(val_df, "VAL")
    print(f"\n=== VAL: baseline({val_baseline_name}) sum={val_baseline.sum():.4f}  "
          f"filtered sum={val_filtered.sum():.4f}  kept={int(val_keep.sum())}/{len(val_keep)} ===")
    if val_filtered.sum() <= val_baseline.sum():
        print("\n[STOP] VAL FAILED. No OOS spent.")
        return
    print("\n[VAL PASSED] Proceeding to the one, final, partial-OOS look.")

    oos_baseline, oos_baseline_name, oos_filtered, oos_keep, _ = evaluate(oos_df, "OOS")
    print(f"\n=== Partial OOS: baseline({oos_baseline_name}) sum={oos_baseline.sum():.4f}  "
          f"filtered sum={oos_filtered.sum():.4f}  kept={int(oos_keep.sum())}/{len(oos_keep)} ===")


if __name__ == "__main__":
    main()
