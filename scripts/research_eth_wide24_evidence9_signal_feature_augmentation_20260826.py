#!/usr/bin/env python3
"""Round 3 of the wide24 feature-augmentation follow-ups: add the ACTUAL 9 live dashboard
evidence signals (not just their raw ingredient columns, which was round 2) as new HMM features.
Reuses scripts.live_evidence_signal_dashboard_20260823.compute_signals() VERBATIM -- the exact
pure function the dashboard itself calls, no formula reconstruction -- fed with historical klines/
funding fetched fresh from Binance's public API so it can be run over 2025-12-29..2026-08-19
(needs ~3 days/864 bars of lookback before 2026-01-01 to warm up orthogonal_combo's percentile
window), instead of only "right now" like the live dashboard does.

New features added on top of the wide24 24: bottom_votes, top_votes, net_score (bottom_votes -
top_votes) -- the 3 aggregate columns compute_signals() itself produces summarizing all 9 signals'
current *_active state, matching what the live dashboard treats as its headline summary stat
([[eth_evidence_signal_indicator_cooking_research_20260825]]: vote count has real monotonic lift).
Individual per-signal active columns (18 more) were NOT added -- round 1/2 both showed piling on
many sparse/redundant columns makes things worse, and votes/net_score already summarize them.

Same TRAIN=2026-01-01..2026-06-30 / EVAL=2026-07-01..2026-08-19 split as round 2, same states=24/
sticky=0.90/seed=7529/label as always, for direct comparability with round 2's wide24 baseline
number (0.7669) on the identical window.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, log_loss
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.retrain_clean_regime_hmm_20260517 import GaussianStateModel  # noqa: E402
from scripts.retrain_clean_regime_hmm_raw_state12_20260517 import STATE12_COLS, _with_raw_state12  # noqa: E402
from scripts.live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402

CLASSES3 = ["bull", "bear", "chop"]
LABEL_CFG = {"trend_adx_min": 16.0, "weak_adx_max": 12.0, "slope_min": 0.00003, "tight_bb_max": 0.006}
WIDE24_EXTRA_COLS = ["volatility_z", "rsi", "macd_hist", "bb_width_z", "hma_slope", "wick_ratio",
                      "mtf_trend_1h", "mtf_trend_4h", "breakout_strength", "mean_reversion_z",
                      "ofi_acceleration", "taker_acceleration"]
WIDE24_COLS = list(STATE12_COLS) + WIDE24_EXTRA_COLS
EVIDENCE9_EXTRA_COLS = ["bottom_votes", "top_votes", "net_score"]
FEATURE_SETS = {"wide24": WIDE24_COLS, "wide24_evidence9": WIDE24_COLS + EVIDENCE9_EXTRA_COLS}
STATES, N_ITER, SEED, STICKY = 24, 22, 7529, 0.90
FETCH_START = pd.Timestamp("2025-12-29", tz="UTC")   # buffer before TRAIN_START for 864-bar warmup
FETCH_END = pd.Timestamp("2026-08-20", tz="UTC")
TRAIN_START = pd.Timestamp("2026-01-01")
EVAL_START = pd.Timestamp("2026-07-01")

FUTURES_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"


def fetch_klines_range(symbol, start, end):
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
            "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    out = []
    cur = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while cur < end_ms:
        params = {"symbol": symbol, "interval": "5m", "limit": 1500, "startTime": cur, "endTime": end_ms}
        r = requests.get(FUTURES_KLINES_URL, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        out.extend(data)
        last_open = data[-1][0]
        if last_open <= cur:
            break
        cur = last_open + 5 * 60 * 1000
        time.sleep(0.12)
        if len(data) < 1500:
            break
    df = pd.DataFrame(out, columns=cols)
    for c in ("open", "high", "low", "close", "volume", "taker_buy_base"):
        df[c] = df[c].astype(np.float64)
    df["timestamp"] = pd.to_datetime(df["open_time"].astype(np.int64), unit="ms", utc=True)
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def fetch_funding_range(symbol, start, end):
    out = []
    cur = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while cur < end_ms:
        params = {"symbol": symbol, "startTime": cur, "endTime": end_ms, "limit": 1000}
        r = requests.get(FUNDING_URL, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        out.extend(data)
        last_ts = int(data[-1]["fundingTime"])
        if last_ts <= cur:
            break
        cur = last_ts + 1
        time.sleep(0.12)
        if len(data) < 1000:
            break
    df = pd.DataFrame(out)
    df["calc_time"] = pd.to_datetime(df["fundingTime"].astype(np.int64), unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype(np.float64)
    df = df.drop_duplicates("calc_time").sort_values("calc_time").reset_index(drop=True)
    FUNDING_Z_WINDOW, FUNDING_Z_MIN_PERIODS = 90, 30
    mean = df["fundingRate"].rolling(FUNDING_Z_WINDOW, min_periods=FUNDING_Z_MIN_PERIODS).mean()
    std = df["fundingRate"].rolling(FUNDING_Z_WINDOW, min_periods=FUNDING_Z_MIN_PERIODS).std()
    df["funding_z"] = (df["fundingRate"] - mean) / std.replace(0.0, np.nan)
    return df[["calc_time", "funding_z"]]


def _num(frame, col, default=0.0):
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _adx(high, low, close, period=14):
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    up = high.diff()
    down = -low.diff()
    pdm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    ndm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    pdi = 100.0 * pdm.ewm(span=period, adjust=False).mean() / (atr + 1e-12)
    ndi = 100.0 * ndm.ewm(span=period, adjust=False).mean() / (atr + 1e-12)
    dx = 100.0 * (pdi - ndi).abs() / (pdi + ndi + 1e-12)
    return dx.ewm(span=period, adjust=False).mean()


def _current_labels3_thresholded(frame, cfg):
    close = _num(frame, "close")
    high = _num(frame, "high")
    low = _num(frame, "low")
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema_slope = (ema21 - ema21.shift(5)) / (close * 5.0 + 1e-12)
    adx = _num(frame, "adx_14", np.nan)
    if adx.isna().all():
        adx = _adx(high, low, close)
    bb_width = _num(frame, "bb_width", np.nan)
    if bb_width.isna().all():
        sma20 = close.rolling(20, min_periods=5).mean()
        bb_width = 2.0 * close.rolling(20, min_periods=5).std() / (sma20 + 1e-12)
    labels = np.full(len(frame), 2, dtype=np.int64)
    trending = adx.fillna(0.0).to_numpy() >= float(cfg["trend_adx_min"])
    slope = ema_slope.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    slope_min = float(cfg["slope_min"])
    labels[trending & (slope > slope_min)] = 0
    labels[trending & (slope < -slope_min)] = 1
    labels[(adx.fillna(0.0).to_numpy() < float(cfg["weak_adx_max"])) | (bb_width.fillna(0.0).to_numpy() < float(cfg["tight_bb_max"]))] = 2
    return labels


def _with_features(frame, cols):
    out = _with_raw_state12(frame.copy())
    for col in cols:
        if col not in out.columns:
            raise ValueError(f"missing feature column: {col}")
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _fit_obs(train, pred, cols):
    x_train_raw = train[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    medians = x_train_raw.median(numeric_only=True).fillna(0.0)
    x_train = x_train_raw.fillna(medians).fillna(0.0)
    x_pred = pred[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0)
    scaler = RobustScaler(quantile_range=(5.0, 95.0))
    return scaler.fit_transform(x_train), scaler.transform(x_pred), scaler, medians


def _state_class_matrix(state_prob, y, smoothing=0.02):
    mat = np.full((state_prob.shape[1], len(CLASSES3)), smoothing, dtype=np.float64)
    for cls in range(len(CLASSES3)):
        mat[:, cls] += state_prob[y == cls].sum(axis=0) / max(int((y == cls).sum()), 1)
    mat /= np.clip(mat.sum(axis=1, keepdims=True), 1e-300, None)
    return mat


def _class_proba(state_prob, state_class):
    proba = state_prob @ state_class
    proba /= np.clip(proba.sum(axis=1, keepdims=True), 1e-300, None)
    return proba


def _run_lengths(pred):
    if len(pred) == 0:
        return []
    lengths, start = [], 0
    for i in range(1, len(pred)):
        if pred[i] != pred[i - 1]:
            lengths.append(i - start)
            start = i
    lengths.append(len(pred) - start)
    return lengths


def _eval(y, proba):
    proba = np.asarray(proba, dtype=np.float64)
    proba /= np.clip(proba.sum(axis=1, keepdims=True), 1e-12, None)
    pred = np.argmax(proba, axis=1)
    cm = confusion_matrix(y, pred, labels=list(range(len(CLASSES3))))
    recalls = {}
    for i, name in enumerate(CLASSES3):
        denom = cm[i].sum()
        recalls[name] = None if denom == 0 else float(cm[i, i] / denom)
    runs = _run_lengths(pred)
    return {
        "rows": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, proba, labels=list(range(len(CLASSES3))))),
        "recall": recalls,
        "true_counts": {CLASSES3[i]: int((y == i).sum()) for i in range(len(CLASSES3))},
        "flip_rate": float(np.mean(pred[1:] != pred[:-1])) if len(pred) > 1 else 0.0,
        "mean_state_duration_bars": float(np.mean(runs)) if runs else 0.0,
    }


def train_one(work_raw, feature_set_name, cols):
    work = _with_features(work_raw, cols)
    train_mask = (work["timestamp"] >= TRAIN_START) & (work["timestamp"] < EVAL_START)
    eval_mask = work["timestamp"] >= EVAL_START
    train_part = work.loc[train_mask].copy()
    eval_part = work.loc[eval_mask].copy()
    train_obs, eval_obs, _, _ = _fit_obs(train_part, eval_part, cols)
    model = GaussianStateModel(STATES, N_ITER, SEED, sticky=STICKY).fit(train_obs)
    y_train = _current_labels3_thresholded(train_part, LABEL_CFG)
    y_eval = _current_labels3_thresholded(eval_part, LABEL_CFG)
    state_class = _state_class_matrix(model.filter_proba(train_obs), y_train)
    eval_proba = _class_proba(model.filter_proba(eval_obs), state_class)
    report = _eval(y_eval, eval_proba)
    report["feature_set"] = feature_set_name
    report["n_features"] = len(cols)
    report["train_rows"] = int(len(train_part))
    report["train_range"] = [str(train_part["timestamp"].min()), str(train_part["timestamp"].max())]
    report["eval_range"] = [str(eval_part["timestamp"].min()), str(eval_part["timestamp"].max())]
    return report


def main():
    print("fetching ETH klines (fresh, public API)...", flush=True)
    eth = fetch_klines_range("ETHUSDT", FETCH_START, FETCH_END)
    print(f"  {len(eth)} bars {eth['timestamp'].min()} .. {eth['timestamp'].max()}")

    print("fetching BTC klines (fresh, public API)...", flush=True)
    btc = fetch_klines_range("BTCUSDT", FETCH_START, FETCH_END)
    print(f"  {len(btc)} bars")

    print("fetching funding-rate history (fresh, public API)...", flush=True)
    funding = fetch_funding_range("ETHUSDT", FETCH_START, FETCH_END)
    print(f"  {len(funding)} funding events")

    print("running compute_signals() (verbatim live dashboard function)...", flush=True)
    sig = compute_signals(eth, btc_df=btc, funding_df=funding)
    sig["timestamp"] = sig["timestamp"].dt.tz_localize(None)
    print(f"  computed, columns include: bottom_votes/top_votes/net_score present = "
          f"{{'bottom_votes','top_votes','net_score'}}.issubset(sig.columns) = "
          f"{set(['bottom_votes','top_votes','net_score']).issubset(sig.columns)}")

    print("loading training_features_2026_rebuilt.csv (UTC) ...", flush=True)
    tf = pd.read_csv(ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv", low_memory=False)
    tf["timestamp"] = pd.to_datetime(tf["timestamp"])
    tf = tf.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    print(f"  {len(tf)} rows, {tf['timestamp'].min()} .. {tf['timestamp'].max()} (UTC)")

    work = tf.merge(sig[["timestamp"] + EVIDENCE9_EXTRA_COLS], on="timestamp", how="left")
    eval_window_mask = work["timestamp"] >= TRAIN_START
    for c in EVIDENCE9_EXTRA_COLS:
        n_missing = int(work.loc[eval_window_mask, c].isna().sum())
        n_total = int(eval_window_mask.sum())
        print(f"  {c}: {n_missing}/{n_total} missing within test window "
              f"(mean={work.loc[eval_window_mask, c].mean():.3f})")

    results = {}
    for name, cols in FEATURE_SETS.items():
        print(f"\n=== training {name} ({len(cols)} features) ===", flush=True)
        r = train_one(work, name, cols)
        results[name] = r
        print({k: v for k, v in r.items() if k not in ("train_range", "eval_range")})

    print("\n\n==== SUMMARY ====")
    for name, r in results.items():
        print(f"{name:20s} balanced_accuracy={r['balanced_accuracy']:.4f}  accuracy={r['accuracy']:.4f}  "
              f"log_loss={r['log_loss']:.4f}  flip_rate={r['flip_rate']:.4f}  "
              f"recall={ {k: (round(v,3) if v is not None else None) for k,v in r['recall'].items()} }  "
              f"train_rows={r['train_rows']}  eval_rows={r['rows']}")

    import json
    out_path = ROOT / "tmp/eth_wide24_evidence9_augmented_regime_test_20260826.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "train_start": str(TRAIN_START), "eval_start": str(EVAL_START),
        "hyperparams": {"states": STATES, "n_iter": N_ITER, "seed": SEED, "sticky": STICKY},
        "label_config": LABEL_CFG,
        "new_columns": EVIDENCE9_EXTRA_COLS,
        "results": results,
    }, indent=2, default=str))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
