#!/usr/bin/env python3
"""Build the v7b V자반등 trade-candidate population for the economic/cost-gate re-test (per
docs/homer/README.md + memory eth_v_rebound_trailing_stop_costgate_marginal_20260830's explicit
instruction: "재라벨링이 확정되면 이 진단(SL-race부터)을 새 라벨로 반드시 재실행해야 한다").

Runs the SAME TabPFN(Tier0+rsi) v7b model used for the label-comparison AUC test on VAL+OOS
(out-of-sample, 4-seed averaged proba -- NOT ground truth labels, this is what a live deployment
would actually see), keeps only events the model actually CALLS V자반등 (proba>=0.5) -- the
economically-relevant population, matching how the original v4 cost-gate test and the prior
V_REBOUND trailing-stop memory both scoped their candidate set (a directional call, not a passive
classification exercise). Reserved holdout is NOT touched here (VAL+OOS only, matching every
other cost-gate test in this lineage).

Entry convention (verbatim from eth_v_rebound_trailing_stop_costgate_marginal_20260830 memory,
confirmed against build_eth_5m_liquidity_sweep_v_rebound_labels_20260829.py / live_eth_sweep_v_
rebound_signal_20260829.py): entry = NEXT 5m bar's OPEN (not the sweep bar's own close) -- this
project's standard causal_futures_backtest convention. side: downside sweep + V자반등 call =
LONG, upside sweep + V자반등 call = SHORT.

Pulls 200 bars of forward OHLC per candidate (~16.7h) -- generous buffer for whatever SL/ARM/Trail
widths get tried; the label's own window was only 60min but a trailing stop can in principle stay
open much longer if price drifts favorably without ever triggering the trail.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier

ROOT = Path(__file__).resolve().parents[1]
LABEL_DIR = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829"
KLINES = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
OUT_DIR = ROOT / "data/research/eth_sweep_v_rebound_v7b_costgate_20260830"

VAL_START, VAL_END = pd.Timestamp("2025-09-01", tz="UTC"), pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
OOS_START, OOS_END = pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-03-31 23:59:59", tz="UTC")
LABEL_WINDOW = pd.Timedelta(minutes=60)
SEEDS = [20260829, 141592, 271828, 577215]
FORWARD_BARS = 200

TIER0 = [
    "is_downside", "sweep_penetration_atr", "atr", "atr_percentile_864",
    "range_width_pct", "hour_utc", "weekday", "delta_z", "flow_aligned_delta_z",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48", "vol_z",
    "lower_wick_ratio", "upper_wick_ratio", "bb_pctb", "adx14", "pdi", "ndi", "bb_width_pctile",
]
FEATURES = TIER0 + ["rsi"]


def load_rsi() -> pd.DataFrame:
    frames = []
    for y in ("2024", "2025", "2026_rebuilt"):
        f = pd.read_csv(ROOT / f"data/splits/year_oos/training_features_{y}.csv", usecols=["timestamp", "rsi"])
        f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True)
        frames.append(f)
    return pd.concat(frames, ignore_index=True).drop_duplicates("timestamp")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(LABEL_DIR / "tabpfn_train_context_frozen_v7b_20260830.csv")
    print(f"train n={len(train)}")

    tier0 = pd.read_csv(LABEL_DIR / "eth_5m_sweep_v_rebound_features_tier0_v7b_20260830.csv")
    tier0["timestamp"] = pd.to_datetime(tier0["timestamp"], utc=True)
    rsi = load_rsi()
    df = tier0.merge(rsi, on="timestamp", how="left").dropna(subset=FEATURES + ["label"]).reset_index(drop=True)
    ts = df["timestamp"]
    window_end = ts + LABEL_WINDOW
    val = df.loc[(ts >= VAL_START) & (ts <= VAL_END) & (window_end < OOS_START)].copy()
    oos = df.loc[(ts >= OOS_START) & (ts <= OOS_END)].copy()
    print(f"val n={len(val)}  oos n={len(oos)}")

    probas = {"val": [], "oos": []}
    for seed in SEEDS:
        clf = TabPFNClassifier(device="cuda", random_state=seed)
        clf.fit(train[FEATURES], train["label"].to_numpy())
        probas["val"].append(clf.predict_proba(val[FEATURES])[:, 1])
        probas["oos"].append(clf.predict_proba(oos[FEATURES])[:, 1])
        print(f"  seed={seed} done")

    val["model_proba"] = np.mean(probas["val"], axis=0)
    oos["model_proba"] = np.mean(probas["oos"], axis=0)
    val["split"] = "val"
    oos["split"] = "oos"
    combined = pd.concat([val, oos], ignore_index=True)

    called = combined[combined["model_proba"] >= 0.5].copy()
    print(f"V자반등 called (proba>=0.5): {len(called)} / {len(combined)} "
          f"(val {int((called['split']=='val').sum())}, oos {int((called['split']=='oos').sum())})")
    print(f"  of these, actually label==1 (precision): {called['label'].mean():.4f}")

    kl = pd.read_csv(KLINES, usecols=["timestamp", "open", "high", "low", "close"])
    kl["timestamp"] = pd.to_datetime(kl["timestamp"], utc=True)
    kl = kl.dropna().sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    ts_to_idx = pd.Series(kl.index.to_numpy(), index=kl["timestamp"].to_numpy())

    rows = []
    for _, ev in called.iterrows():
        idx = ts_to_idx.get(ev["timestamp"])
        if idx is None or idx + FORWARD_BARS + 1 >= len(kl):
            continue
        entry_bar = kl.iloc[idx + 1]
        side = "long" if ev["is_downside"] == 1 else "short"
        fwd = kl.iloc[idx + 1: idx + 1 + FORWARD_BARS][["timestamp", "open", "high", "low", "close"]]
        rows.append({
            "candidate_index": int(ev["candidate_index"]), "sweep_ts": ev["timestamp"], "split": ev["split"],
            "side": side, "model_proba": float(ev["model_proba"]), "label": int(ev["label"]),
            "atr": float(ev["atr"]), "entry_ts": entry_bar["timestamp"], "entry_price": float(entry_bar["open"]),
            "fwd_open": fwd["open"].tolist(), "fwd_high": fwd["high"].tolist(),
            "fwd_low": fwd["low"].tolist(), "fwd_close": fwd["close"].tolist(),
        })
    result = pd.DataFrame(rows)
    print(f"final candidates with full forward data: {len(result)}")
    result.to_pickle(OUT_DIR / "v7b_costgate_candidates.pkl")
    result.drop(columns=["fwd_open", "fwd_high", "fwd_low", "fwd_close"]).to_csv(
        OUT_DIR / "v7b_costgate_candidates_summary.csv", index=False)
    print(f"saved to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
