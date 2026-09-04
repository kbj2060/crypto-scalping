#!/usr/bin/env python3
"""2026-08-27: the trend_bucket thresholds (0.10 / 0.995 / 0.9995) were hand-picked by eyeballing a
10-decile table, not searched for. This does a proper fine-grained sweep: a sliding-window smoothed
big-miss-rate curve over the FULL [0,1] trend_score range (not just deciles), to find where risk
genuinely transitions, then checks whether a revised threshold separates risk better than the
current ad hoc ones."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
from live_evidence_signal_dashboard_20260823 import compute_signals, SIGNAL_ORDER  # noqa: E402
from research_eth_funding_crossasset_combo_signal_20260825 import load_funding_z  # noqa: E402
from retrain_clean_regime_hmm_raw_state12_20260517 import _with_raw_state12  # noqa: E402

BTC_PATH = ROOT / "data" / "btc_5m_1year.csv"
REGIME_MODEL_PATH = ROOT / "tmp" / "eth_regime_gbm3_independent_20260826" / "model.joblib"
K_30M = 6
BIG_MISS_PCT = 0.5

payload = joblib.load(REGIME_MODEL_PATH)
MODEL_COLS = payload["feature_cols"]
MODEL_MED = pd.Series(payload["feature_medians"])
MODEL = payload["model"]
MODEL_CLASSES = list(payload["classes"])


def compute_frame(base_csv: Path) -> pd.DataFrame:
    raw = pd.read_csv(base_csv, low_memory=False)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw = raw.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    btc = pd.read_csv(BTC_PATH, usecols=["timestamp", "high", "low"], parse_dates=["timestamp"])
    funding = load_funding_z()
    base_cols = ["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"]
    sig = compute_signals(raw[base_cols].copy(), btc_df=btc, funding_df=funding)
    feats = _with_raw_state12(raw)
    for c in MODEL_COLS:
        if c not in feats.columns:
            feats[c] = MODEL_MED.get(c, 0.0)
    x = feats[MODEL_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(MODEL_MED).fillna(0.0)
    proba = MODEL.predict_proba(x)
    prob_df = pd.DataFrame(proba, columns=MODEL_CLASSES)
    sig["trend_score"] = (1.0 - prob_df["chop"]).to_numpy()
    close = raw["close"].to_numpy()
    fwd = np.full(len(sig), np.nan)
    fwd[:-K_30M] = close[K_30M:] / close[:-K_30M] - 1.0
    sig["fwd_ret_30m"] = fwd
    return sig


print("Building frames...")
frames = [compute_frame(gate.sweep.BASE_2025), compute_frame(gate.sweep.BASE_2026)]
all_fr = pd.concat(frames, ignore_index=True)

SIGNAL_NAMES = [name for name, _ in SIGNAL_ORDER]
rows = []
for name in SIGNAL_NAMES:
    for side in ("top", "bottom"):
        col = f"{side}_{name}"
        fired = all_fr[col].fillna(False).to_numpy() & all_fr["fwd_ret_30m"].notna().to_numpy()
        sub = all_fr.loc[fired, ["trend_score", "fwd_ret_30m"]].copy()
        sub["side"] = side
        rows.append(sub)
pooled = pd.concat(rows, ignore_index=True)
pooled["rets_pct"] = pooled["fwd_ret_30m"] * 100.0
pooled["is_big_miss"] = np.where(pooled["side"] == "top", pooled["rets_pct"] > BIG_MISS_PCT, pooled["rets_pct"] < -BIG_MISS_PCT)
pooled = pooled.sort_values("trend_score").reset_index(drop=True)
print(f"n = {len(pooled)}")

WINDOW = 2500
STEP = 500
print(f"\n=== Sliding-window big-miss-rate curve (window={WINDOW}, step={STEP}) across FULL [0,1] range ===")
bm = pooled["is_big_miss"].to_numpy()
ts = pooled["trend_score"].to_numpy()
curve = []
for start in range(0, len(pooled) - WINDOW, STEP):
    seg = bm[start:start + WINDOW]
    seg_ts = ts[start:start + WINDOW]
    curve.append((seg_ts.min(), seg_ts.mean(), seg_ts.max(), seg.mean(), len(seg)))
curve_df = pd.DataFrame(curve, columns=["ts_min", "ts_mean", "ts_max", "big_miss_rate", "n"])
pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 140)
print(curve_df.to_string(index=False))

print("\n=== Zoomed: fine percentile cuts in the LOW range (0 to 0.3) ===")
low = pooled[pooled["trend_score"] < 0.3]
low_bins = pd.cut(low["trend_score"], bins=np.arange(0, 0.31, 0.02))
print(low.groupby(low_bins, observed=True)["is_big_miss"].agg(["size", "mean"]).to_string())

print("\n=== Zoomed: fine percentile cuts in the HIGH range (0.95 to 1.0) ===")
high = pooled[pooled["trend_score"] >= 0.95]
edges = [0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.9998, 0.9999, 0.99995, 1.0]
high_bins = pd.cut(high["trend_score"], bins=edges)
print(high.groupby(high_bins, observed=True)["is_big_miss"].agg(["size", "mean"]).to_string())

print("\n=== Threshold search: for candidate high-thresholds, compare below-vs-above big-miss-rate gap ===")
candidates = [0.90, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.9998, 0.9999]
for c in candidates:
    below = pooled[pooled["trend_score"] < c]
    above = pooled[pooled["trend_score"] >= c]
    if len(below) < 500 or len(above) < 500:
        continue
    print(f"threshold={c}: below(n={len(below)})={below['is_big_miss'].mean()*100:.2f}%  "
          f"above(n={len(above)})={above['is_big_miss'].mean()*100:.2f}%  gap={((above['is_big_miss'].mean()-below['is_big_miss'].mean())*100):+.2f}pp")
