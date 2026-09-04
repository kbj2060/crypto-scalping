#!/usr/bin/env python3
"""Pilot: does pooling a SECOND, differently-triggered evidence-signal's events (dalton_rule2_
balance_edge -- price near its own 48-bar range edge in a low-vol regime, a DIFFERENT mechanism
than liquidity_sweep's breach-and-reclaim) into the TRAINING set help or dilute the deployed
liquidity_sweep V_REBOUND model? User's question: would pooling ground truth across multiple
evidence signals improve accuracy (either as one pooled model, or per-signal models to ensemble).
This project's own strong prior (top-6 confluence backtests, sweep_flow_combo/smt_flow_combo
dropped for the same reason) is that STACKING signals tends to dilute rather than strengthen --
this pilot tests that prior directly rather than assuming it, scoped to just ONE additional
signal (dalton_rule2_balance_edge, chosen because it's the only OTHER of the 8 dashboard evidence
signals with a natural "level" concept -- the others are oscillator/volume/divergence-based with
no level to test a "V_REBOUND"-style outcome against).

Trigger definition reused VERBATIM (not reimplemented) from analyze_eth_amt_vsa_footprint_ifvg_
component_evidence_20260815.py::add_amt_features (balance_edge_low/high) -- note this uses an
UNSHIFTED range_low/high (includes the current bar, per that script's own original design, not
"fixed" to match liquidity_sweep's shifted convention). The OUTCOME formula (30min/6bar-sustain/
1.5xATR) is kept IDENTICAL to the deployed V_REBOUND label, using this repo's own `atr` (from
add_causal_columns, not dalton_rule2's own atr_price) -- isolates the test to "does the trigger
type matter", not "did the outcome definition also change".

Feature harmonization caveat (this is a pilot, not a polished feature): dalton_rule2 events don't
"penetrate" a level the way a sweep does (no breach required, just edge-proximity) -- the
sweep_penetration_atr analog here is (row_extreme - range_edge)/atr, a non-negative "distance from
the range's own edge", not a like-for-like value with a real sweep's overshoot-beyond-level
penetration. Flagged, not hidden.

Evaluation: trains on POOLED (liquidity_sweep TRAIN + dalton_rule2 TRAIN) events, evaluates on the
UNCHANGED liquidity_sweep-only VAL/OOS (the actual deployed use case) -- directly answers "does
adding this other signal's data help or hurt the model we actually ship", not a different question.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tabpfn import TabPFNClassifier

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

TIER0_CSV = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829/eth_5m_sweep_v_rebound_features_tier0.csv"
TIER0_BUILDER = ROOT / "scripts/build_eth_5m_sweep_v_rebound_features_tier0_20260829.py"
AMT_SCRIPT = ROOT / "scripts/analyze_eth_amt_vsa_footprint_ifvg_component_evidence_20260815.py"
RSI_SOURCES = [ROOT / f"data/splits/year_oos/training_features_{y}.csv" for y in ("2024", "2025", "2026_rebuilt")]

VAL_START, VAL_END = pd.Timestamp("2025-09-01", tz="UTC"), pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
OOS_START, OOS_END = pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-03-31 23:59:59", tz="UTC")
LABEL_WINDOW = pd.Timedelta(minutes=30)
LOOKAHEAD_BARS = 6
V_REBOUND_ATR_MULT = 1.5
SEEDS = [20260829, 141592, 271828, 577215]

TIER0 = [
    "is_downside", "sweep_penetration_atr", "atr", "atr_percentile_864",
    "range_width_pct", "hour_utc", "weekday", "delta_z", "flow_aligned_delta_z",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48", "vol_z",
    "lower_wick_ratio", "upper_wick_ratio", "bb_pctb", "adx14", "pdi", "ndi", "bb_width_pctile",
]
FEATURES = TIER0 + ["rsi"]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_dalton_events(tier0_mod, sweep_impl) -> pd.DataFrame:
    """Builds dalton_rule2_balance_edge trigger events with a V_REBOUND-style outcome and the
    same Tier0 feature schema, using verbatim add_amt_features() for the trigger."""
    frame = tier0_mod.build_indicator_frame(sweep_impl)  # has atr_pct/atr_price/adx14/etc. already
    amt_mod = load_module(AMT_SCRIPT, "amt_features_dalton_pilot_20260829")
    frame = amt_mod.add_sweep(frame)
    frame = amt_mod.add_amt_features(frame)

    causal = sweep_impl.add_causal_columns(sweep_impl.load_5m(tier0_mod.SOURCE))
    frame["atr"] = causal["atr"]
    frame["atr_percentile_864"] = frame["atr"].rolling(864, min_periods=864).rank(pct=True)
    frame["hour_utc"] = frame["timestamp"].dt.hour
    frame["weekday"] = frame["timestamp"].dt.weekday

    low, high, close = frame["low"].to_numpy(), frame["high"].to_numpy(), frame["close"].to_numpy()
    atr = frame["atr"].to_numpy()
    range_low, range_high = frame["low"].rolling(48, min_periods=48).min().to_numpy(), frame["high"].rolling(48, min_periods=48).max().to_numpy()
    balance_low = frame["balance_edge_low"].fillna(False).to_numpy()
    balance_high = frame["balance_edge_high"].fillna(False).to_numpy()
    n = len(frame)

    rows = []
    for idx in range(48, n - LOOKAHEAD_BARS):
        a = atr[idx]
        if not np.isfinite(a) or a <= 0:
            continue
        future_close = close[idx + 1: idx + LOOKAHEAD_BARS + 1]
        future_high = high[idx + 1: idx + LOOKAHEAD_BARS + 1]
        future_low = low[idx + 1: idx + LOOKAHEAD_BARS + 1]

        if balance_low[idx] and np.isfinite(range_low[idx]):
            level = range_low[idx]
            move = float(future_high.max() - low[idx])
            confirmed = bool((future_close > level).all())
            label = int(move >= V_REBOUND_ATR_MULT * a and confirmed)
            rows.append({"row_idx": idx, "is_downside": 1, "label": label, "level": level, "atr": a,
                         "penetration": max(low[idx] - level, 0.0) / a, "range_width": range_high[idx] - level})
        if balance_high[idx] and np.isfinite(range_high[idx]):
            level = range_high[idx]
            move = float(high[idx] - future_low.min())
            confirmed = bool((future_close < level).all())
            label = int(move >= V_REBOUND_ATR_MULT * a and confirmed)
            rows.append({"row_idx": idx, "is_downside": 0, "label": label, "level": level, "atr": a,
                         "penetration": max(level - high[idx], 0.0) / a, "range_width": level - range_low[idx]})

    events = pd.DataFrame(rows)
    feat = frame.iloc[events["row_idx"].to_numpy()].reset_index(drop=True)
    out = pd.DataFrame({
        "timestamp": feat["timestamp"].to_numpy(),
        "label": events["label"].to_numpy(),
        "is_downside": events["is_downside"].to_numpy(),
        "sweep_penetration_atr": events["penetration"].to_numpy(),
        "atr": events["atr"].to_numpy(),
        "atr_percentile_864": feat["atr_percentile_864"].to_numpy(),
        "range_width_pct": (events["range_width"] / feat["close"].to_numpy()).to_numpy(),
        "hour_utc": feat["hour_utc"].to_numpy(), "weekday": feat["weekday"].to_numpy(),
    })
    delta_z = feat["delta_z"].to_numpy(dtype=float)
    out["delta_z"] = delta_z
    out["flow_aligned_delta_z"] = np.where(events["is_downside"].to_numpy() == 1, delta_z, -delta_z)
    for col in ["p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48", "vol_z",
                "lower_wick_ratio", "upper_wick_ratio", "bb_pctb", "adx14", "pdi", "ndi", "bb_width_pctile"]:
        out[col] = feat[col].to_numpy()
    return out


def main() -> int:
    tier0_mod = load_module(TIER0_BUILDER, "tier0_dalton_pilot_20260829")
    sweep_impl = tier0_mod.load_sweep_impl()

    print("building dalton_rule2_balance_edge events (verbatim trigger, V_REBOUND-style outcome)...")
    dalton = build_dalton_events(tier0_mod, sweep_impl)
    dalton["timestamp"] = pd.to_datetime(dalton["timestamp"], utc=True)
    print(f"  {len(dalton)} dalton events, label rate {dalton['label'].mean():.4f}")

    sweep_tier0 = pd.read_csv(TIER0_CSV)
    sweep_tier0["timestamp"] = pd.to_datetime(sweep_tier0["timestamp"], utc=True)

    frames = []
    for p in RSI_SOURCES:
        f = pd.read_csv(p, usecols=["timestamp", "rsi"])
        f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True)
        frames.append(f)
    rsi = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp")

    sweep_df = sweep_tier0.merge(rsi, on="timestamp", how="left").dropna(subset=FEATURES + ["label"]).reset_index(drop=True)
    dalton_df = dalton.merge(rsi, on="timestamp", how="left").dropna(subset=FEATURES + ["label"]).reset_index(drop=True)

    ts_s, we_s = sweep_df["timestamp"], sweep_df["timestamp"] + LABEL_WINDOW
    sweep_train = sweep_df.loc[(ts_s < VAL_START) & (we_s < VAL_START)]
    sweep_val = sweep_df.loc[(ts_s >= VAL_START) & (ts_s <= VAL_END) & (we_s < OOS_START)]
    sweep_oos = sweep_df.loc[(ts_s >= OOS_START) & (ts_s <= OOS_END)]

    ts_d, we_d = dalton_df["timestamp"], dalton_df["timestamp"] + LABEL_WINDOW
    dalton_train = dalton_df.loc[(ts_d < VAL_START) & (we_d < VAL_START)]
    print(f"sweep_only train n={len(sweep_train)}  dalton train n={len(dalton_train)}  "
          f"pooled train n={len(sweep_train) + len(dalton_train)}")
    print(f"VAL n={len(sweep_val)}  OOS n={len(sweep_oos)}  (unchanged -- liquidity_sweep only, the deployed eval population)")

    pooled_train = pd.concat([sweep_train, dalton_train], ignore_index=True)

    results = {}
    for variant_name, train_df in (("sweep_only (current deployed)", sweep_train), ("pooled (sweep+dalton_rule2)", pooled_train)):
        seed_rows = []
        for seed in SEEDS:
            clf = TabPFNClassifier(device="cuda", random_state=seed)
            clf.fit(train_df[FEATURES], train_df["label"].to_numpy())
            row = {"seed": seed}
            for name, split in (("val", sweep_val), ("oos", sweep_oos)):
                proba = clf.predict_proba(split[FEATURES])[:, 1]
                row[f"{name}_auc"] = round(float(roc_auc_score(split["label"], proba)), 4)
            seed_rows.append(row)
            print(f"  [{variant_name}] seed={seed}: val_auc={row['val_auc']:.4f} oos_auc={row['oos_auc']:.4f}")
        table = pd.DataFrame(seed_rows)
        results[variant_name] = {
            "val_auc_mean": round(float(table["val_auc"].mean()), 4), "val_auc_std": round(float(table["val_auc"].std(ddof=1)), 4),
            "oos_auc_mean": round(float(table["oos_auc"].mean()), 4), "oos_auc_std": round(float(table["oos_auc"].std(ddof=1)), 4),
        }

    print("\n=== SUMMARY: does pooling dalton_rule2_balance_edge help the liquidity_sweep model? ===")
    for name, r in results.items():
        print(f"  {name:32s} VAL {r['val_auc_mean']:.4f}+/-{r['val_auc_std']:.4f}  OOS {r['oos_auc_mean']:.4f}+/-{r['oos_auc_std']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
