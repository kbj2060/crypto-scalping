#!/usr/bin/env python3
"""ZDC(완전wick) 9트리거 통합모델의 trade-candidate population 빌드 -- wick-앵커판
(research_eth_v_rebound_multitrigger_zigzag_direction_costgate_candidates_20260901.py)과 완전히
동일한 구조(FORWARD_BARS=400, VAL+OOS proba>=0.5 호출모집단, entry=트리거idx+1봉 open) --
비교가 목적이므로 FEATURES_CSV 경로 외 전부 고정.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier

ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_zigzag_direction_full_wick_20260901/eth_5m_v_rebound_multitrigger_zigzag_direction_full_wick_features_tier0.csv"
KLINES = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
OUT_DIR = ROOT / "data/research/eth_v_rebound_multitrigger_zigzag_direction_full_wick_costgate_20260901"

VAL_START, VAL_END = pd.Timestamp("2025-09-01", tz="UTC"), pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
OOS_START, OOS_END = pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-03-31 23:59:59", tz="UTC")
HOLDOUT_START = pd.Timestamp("2026-04-01", tz="UTC")
LABEL_WINDOW = pd.Timedelta(hours=24)
SEEDS = [20260829, 141592, 271828, 577215]
FORWARD_BARS = 400

FEATURE_COLUMNS = [
    "is_downside", "sweep_penetration_atr", "atr", "atr_percentile_864",
    "range_width_pct", "hour_utc", "weekday",
    "delta_z", "flow_aligned_delta_z",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48",
    "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
    "adx14", "pdi", "ndi", "bb_width_pctile", "rsi",
]


def embargoed_split(df: pd.DataFrame) -> dict:
    ts = df["timestamp"]
    window_end = ts + LABEL_WINDOW
    return {
        "train": df.loc[(ts < VAL_START) & (window_end < VAL_START)],
        "val": df.loc[(ts >= VAL_START) & (ts <= VAL_END) & (window_end < OOS_START)],
        "oos": df.loc[(ts >= OOS_START) & (ts <= OOS_END) & (ts < HOLDOUT_START)],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURES_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["hit_bool"] = df["hit"].astype(str).map({"True": True, "False": False})
    df = df[df["hit_bool"].isin([True, False])].copy()
    df["label"] = df["hit_bool"].astype(int)
    df = df.dropna(subset=FEATURE_COLUMNS + ["label"]).reset_index(drop=True)

    parts = embargoed_split(df)
    train = parts["train"]
    val, oos = parts["val"].copy(), parts["oos"].copy()
    print(f"train n={len(train)}  val n={len(val)}  oos n={len(oos)}")

    over_limit = len(train) > 10000
    probas = {"val": [], "oos": []}
    for seed in SEEDS:
        clf = TabPFNClassifier(device="cuda", random_state=seed, ignore_pretraining_limits=over_limit)
        clf.fit(train[FEATURE_COLUMNS], train["label"].to_numpy())
        probas["val"].append(clf.predict_proba(val[FEATURE_COLUMNS])[:, 1])
        probas["oos"].append(clf.predict_proba(oos[FEATURE_COLUMNS])[:, 1])
        print(f"  seed={seed} done", flush=True)

    val["model_proba"] = np.mean(probas["val"], axis=0)
    oos["model_proba"] = np.mean(probas["oos"], axis=0)
    val["split"] = "val"
    oos["split"] = "oos"
    combined = pd.concat([val, oos], ignore_index=True)

    called = combined[combined["model_proba"] >= 0.5].copy()
    print(f"ZDC(full_wick) called (proba>=0.5): {len(called)} / {len(combined)} "
          f"(val {int((called['split']=='val').sum())}, oos {int((called['split']=='oos').sum())})")
    print(f"  of these, actually label==1 (precision): {called['label'].mean():.4f}")

    kl = pd.read_csv(KLINES, usecols=["timestamp", "open", "high", "low", "close"])
    kl["timestamp"] = pd.to_datetime(kl["timestamp"], utc=True)
    kl = kl.dropna().sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    ts_to_idx = pd.Series(kl.index.to_numpy(), index=kl["timestamp"].to_numpy())

    rows = []
    n_buffer_short = 0
    for _, ev in called.iterrows():
        pos = ts_to_idx.get(ev["timestamp"])
        if pos is None:
            continue
        if pos + FORWARD_BARS + 1 >= len(kl):
            n_buffer_short += 1
            continue
        entry_bar = kl.iloc[pos + 1]
        side = "long" if ev["is_downside"] == 1 else "short"
        fwd = kl.iloc[pos + 1: pos + 1 + FORWARD_BARS][["timestamp", "open", "high", "low", "close"]]
        rows.append({
            "idx": int(ev["idx"]), "trigger_ts": ev["timestamp"], "split": ev["split"],
            "side": side, "model_proba": float(ev["model_proba"]), "label": int(ev["label"]),
            "atr": float(ev["atr"]), "entry_ts": entry_bar["timestamp"], "entry_price": float(entry_bar["open"]),
            "fwd_open": fwd["open"].tolist(), "fwd_high": fwd["high"].tolist(),
            "fwd_low": fwd["low"].tolist(), "fwd_close": fwd["close"].tolist(),
        })
    result = pd.DataFrame(rows)
    print(f"final candidates with full forward data: {len(result)} "
          f"(dropped {n_buffer_short} for insufficient trailing klines near dataset end)")
    result.to_pickle(OUT_DIR / "zdc_fullwick_costgate_candidates.pkl")
    result.drop(columns=["fwd_open", "fwd_high", "fwd_low", "fwd_close"]).to_csv(
        OUT_DIR / "zdc_fullwick_costgate_candidates_summary.csv", index=False)
    print(f"saved to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
