#!/usr/bin/env python3
"""v8 (2026-08-30, user request): widen v7b's reversal time axis by 1.5x -- same 50/50 fast:full
ratio the v3->v4->v6 lineage always used, just scaled up one more notch (30/60min -> 45/90min).

Motivated by the OOS false-positive/true-negative example charts showing a recurring pattern: a
genuine-looking reversal arriving late (around +35 to +55min) that v7b's 30min-fast/60min-full
window doesn't credit -- FP#8 (proba=0.78) missed a rally at +35-40min, TN#8 (proba=0.32) shows one
at +50-55min right at the window boundary. This ALSO matches the liquidation/reversal literature
review's own finding (docs/liquidation_spike_v_rebound_entry_exit_literature_review_20260830.md)
that Osler(2005)'s stop-loss cascades run "hours, not days" and Bremer&Sweeney(1991)'s reversal
window is ~2 days -- both suggest the payoff window for this kind of reversal may be longer than
30-60min.

Everything else is UNCHANGED from v7b: same sweep-event population (reused from
eth_5m_sweep_v_rebound_labels.csv, not recomputed), same ATR_MULT=1.5 (close-confirmed fast move),
same GIVEBACK T_SUSTAIN=0.20, same CHOP threshold (fast_move < 1.0x ATR), same Tier0+rsi features,
same TabPFN 4-seed VAL/OOS/holdout methodology as research_eth_sweep_v_rebound_label_v7b_
comparison_20260830.py -- only FAST_BARS/LOOKAHEAD_BARS change, so any AUC delta is attributable to
the window widening alone, not a confound.

One exploratory comparison run (not a promotion decision) -- reuses the holdout window like every
prior v3->v4->v5->v6->v7->v7b iteration in this specific sub-project already has (this sub-project
does not treat holdout as single-exposure the way other signals' final promotion tests do; that
repeated-reuse is a known, already-flagged tension, not new here).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tabpfn import TabPFNClassifier

ROOT = Path(__file__).resolve().parents[1]
IMPL_SCRIPT = ROOT / "scripts/build_eth_5m_sweep_followthrough_v2_labels_20260829.py"
SOURCE = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
LABEL_DIR = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829"
CURRENT_LABEL_CSV = LABEL_DIR / "eth_5m_sweep_v_rebound_labels.csv"
# NOTE: the full, UNFILTERED Tier0 feature set (all 14,259 sweep events) -- NOT the "_v7b_"
# suffixed one, which is already restricted to v7b's own 5,933-row V자반등/지지횡보 population.
# Starting from that filtered file would silently inner-join away any event v7b excluded but
# v8's wider window now classifies validly, biasing the comparison toward v7b's own population.
TIER0_FULL_CSV = LABEL_DIR / "eth_5m_sweep_v_rebound_features_tier0.csv"
OUT_DIR = ROOT / "tmp/eth_sweep_v_rebound_label_v8_wider_window_20260830"

NEW_FAST_BARS = 9        # 45min (v7b: 6/30min) -- 1.5x wider, same 50% ratio
NEW_LOOKAHEAD_BARS = 18  # 90min (v7b: 12/60min)
ATR_MULT = 1.5
T_SUSTAIN = 0.20
CHOP_ATR_MULT = 1.0

VAL_START, VAL_END = pd.Timestamp("2025-09-01", tz="UTC"), pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
OOS_START, OOS_END = pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-03-31 23:59:59", tz="UTC")
HOLDOUT_START, HOLDOUT_END = pd.Timestamp("2026-04-01", tz="UTC"), pd.Timestamp("2026-08-28 23:59:59", tz="UTC")
LABEL_WINDOW = pd.Timedelta(minutes=NEW_LOOKAHEAD_BARS * 5)
SEEDS = [20260829, 141592, 271828, 577215]

TIER0 = [
    "is_downside", "sweep_penetration_atr", "atr", "atr_percentile_864",
    "range_width_pct", "hour_utc", "weekday", "delta_z", "flow_aligned_delta_z",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48", "vol_z",
    "lower_wick_ratio", "upper_wick_ratio", "bb_pctb", "adx14", "pdi", "ndi", "bb_width_pctile",
]
FEATURES = TIER0 + ["rsi"]

# v7b's own already-established numbers (research_eth_sweep_v_rebound_label_v7b_comparison_20260830.py)
V7B_BASELINE = {"val_auc": 0.7342, "oos_auc": 0.7621, "holdout_auc": 0.7788}


def load_impl():
    spec = importlib.util.spec_from_file_location("sweep_impl_v8_20260830", IMPL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_rsi() -> pd.DataFrame:
    frames = []
    for y in ("2024", "2025", "2026_rebuilt"):
        f = pd.read_csv(ROOT / f"data/splits/year_oos/training_features_{y}.csv", usecols=["timestamp", "rsi"])
        f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True)
        frames.append(f)
    return pd.concat(frames, ignore_index=True).drop_duplicates("timestamp")


def build_v8_labels(impl) -> pd.DataFrame:
    frame = impl.load_5m(SOURCE)
    current = pd.read_csv(CURRENT_LABEL_CSV)
    n_before = len(current)
    current = current[current["candidate_index"] + NEW_LOOKAHEAD_BARS < len(frame)].reset_index(drop=True)
    print(f"events with full {NEW_LOOKAHEAD_BARS * 5}min of future data: {len(current)}/{n_before}")

    fast_move_atr_mult, close_attempted, giveback_ratio = [], [], []
    for _, event in current.iterrows():
        idx = int(event["candidate_index"])
        row = frame.iloc[idx]
        future = frame.iloc[idx + 1: idx + NEW_LOOKAHEAD_BARS + 1]
        fast_future = future.iloc[:NEW_FAST_BARS]
        atr = event["atr"]  # pre-sweep, unaffected by window size, reused as-is
        if event["side"] == "downside":
            sweep_extreme = row["low"]
            fast_close_move = fast_future["close"].max() - sweep_extreme
            peak = future["high"].max()
            end = future["close"].iloc[-1]
            giveback = peak - end
        else:
            sweep_extreme = row["high"]
            fast_close_move = sweep_extreme - fast_future["close"].min()
            peak = future["low"].min()
            end = future["close"].iloc[-1]
            giveback = end - peak
        total_move = abs(peak - sweep_extreme)
        fast_move_atr_mult.append(float(fast_close_move / atr))
        close_attempted.append(bool(fast_close_move >= ATR_MULT * atr))
        giveback_ratio.append(float(giveback / total_move) if total_move > 1e-12 else np.nan)

    current["fast_move_atr_mult"] = fast_move_atr_mult
    current["close_attempted"] = close_attempted
    current["giveback_ratio_v8"] = giveback_ratio

    v_rebound = current["close_attempted"] & (current["giveback_ratio_v8"] <= T_SUSTAIN)
    support_chop = current["fast_move_atr_mult"] < CHOP_ATR_MULT
    new_label = pd.Series(np.nan, index=current.index)
    new_label[v_rebound] = 1
    new_label[support_chop] = 0

    n = len(current)
    print(f"V자반등(1): {int((new_label == 1).sum())} ({(new_label == 1).mean():.1%})  "
          f"지지/횡보(0): {int((new_label == 0).sum())} ({(new_label == 0).mean():.1%})  "
          f"제외: {int(new_label.isna().sum())} ({new_label.isna().mean():.1%})  (n={n})")

    current["label_v8"] = new_label
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    current.to_csv(OUT_DIR / "events_with_v8_labels.csv", index=False)
    return current[["candidate_index", "side", "label_v8"]]


def main() -> int:
    impl = load_impl()
    v8_labels = build_v8_labels(impl)

    tier0 = pd.read_csv(TIER0_FULL_CSV)
    tier0["timestamp"] = pd.to_datetime(tier0["timestamp"], utc=True)
    print(f"tier0(full) side values: {tier0['side'].unique()}, v8_labels side values: {v8_labels['side'].unique()}")

    merged = tier0.drop(columns=["label"]).merge(v8_labels, on=["candidate_index", "side"], how="inner")
    merged = merged.rename(columns={"label_v8": "label"}).dropna(subset=["label"]).reset_index(drop=True)
    merged["label"] = merged["label"].astype(int)
    print(f"tier0 rows: {len(tier0)}  merged+labeled(v8) rows: {len(merged)}")

    rsi = load_rsi()
    df = merged.merge(rsi, on="timestamp", how="left").dropna(subset=FEATURES + ["label"]).reset_index(drop=True)
    ts = df["timestamp"]

    train = df.loc[ts < VAL_START].reset_index(drop=True)
    window_end = ts + LABEL_WINDOW
    val = df.loc[(ts >= VAL_START) & (ts <= VAL_END) & (window_end < OOS_START)].reset_index(drop=True)
    oos = df.loc[(ts >= OOS_START) & (ts <= OOS_END)].reset_index(drop=True)
    holdout = df.loc[(ts >= HOLDOUT_START) & (ts <= HOLDOUT_END)].reset_index(drop=True)
    print(f"train n={len(train)}  val n={len(val)}  oos n={len(oos)}  holdout n={len(holdout)}")
    print(f"label rates: train={train['label'].mean():.4f} val={val['label'].mean():.4f} "
          f"oos={oos['label'].mean():.4f} holdout={holdout['label'].mean():.4f}")

    train.to_csv(OUT_DIR / "tabpfn_train_context_v8_20260830.csv", index=False)

    seed_rows = []
    for seed in SEEDS:
        clf = TabPFNClassifier(device="cuda", random_state=seed)
        clf.fit(train[FEATURES], train["label"].to_numpy())
        row = {"seed": seed}
        for split_name, split in (("val", val), ("oos", oos), ("holdout", holdout)):
            proba = clf.predict_proba(split[FEATURES])[:, 1]
            row[f"{split_name}_auc"] = round(float(roc_auc_score(split["label"], proba)), 4)
        seed_rows.append(row)
        print(f"  seed={seed}: val_auc={row['val_auc']:.4f} oos_auc={row['oos_auc']:.4f} holdout_auc={row['holdout_auc']:.4f}")

    table = pd.DataFrame(seed_rows)
    result = {
        "val_auc": table["val_auc"].mean(), "val_auc_std": table["val_auc"].std(ddof=1),
        "oos_auc": table["oos_auc"].mean(), "oos_auc_std": table["oos_auc"].std(ddof=1),
        "holdout_auc": table["holdout_auc"].mean(), "holdout_auc_std": table["holdout_auc"].std(ddof=1),
    }
    print(f"\n=== v8 (45min/90min window) ===")
    print(f"  VAL {result['val_auc']:.4f}+/-{result['val_auc_std']:.4f}  "
          f"OOS {result['oos_auc']:.4f}+/-{result['oos_auc_std']:.4f}  "
          f"HOLDOUT {result['holdout_auc']:.4f}+/-{result['holdout_auc_std']:.4f}")
    print(f"\n=== v7b baseline (30min/60min window, already established) ===")
    print(f"  VAL {V7B_BASELINE['val_auc']:.4f}  OOS {V7B_BASELINE['oos_auc']:.4f}  "
          f"HOLDOUT {V7B_BASELINE['holdout_auc']:.4f}")
    print(f"\n=== delta (v8 - v7b) ===")
    print(f"  VAL {result['val_auc'] - V7B_BASELINE['val_auc']:+.4f}  "
          f"OOS {result['oos_auc'] - V7B_BASELINE['oos_auc']:+.4f}  "
          f"HOLDOUT {result['holdout_auc'] - V7B_BASELINE['holdout_auc']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
