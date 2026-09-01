#!/usr/bin/env python3
"""Read-only 3-class (bull/bear/chop) regime signal for BTC -- 2026-09-02.

WHY THIS FILE EXISTS. Until now BTC had NO dashboard regime classifier at all: the Snapshot tab's
BTC ribbon was a hard-coded grey "model not available" band (app.js renderCandleSvg, gated on
activeSnapshotAsset === "eth"). That guard was added 2026-08-31 to stop ETH's GBM3 ribbon being
drawn over BTC candles -- see memory eth-dashboard-btc-regime-classifier-not-trained-todo-20260831.
This module is the BTC-native scorer that guard was waiting for.

CONTRACT: returns exactly the same dict shape as compute_regime_gbm3_signal(), so dashboard/server.py
can cache and serve it the same way.

⭐THE CROSS-ASSET COLUMN NAMING TRAP (train/inference parity). FeatureEngineer hardcodes its
cross-asset column names as close_btc / volume_btc / quote_volume_btc regardless of which asset is
actually passed. The canonical BTC training file (data/splits/year_oos/btc_features_2024_2026.csv)
was built with BTC as the SUBJECT and ETH as the CROSS-ASSET -- verified directly: its `close` is
~42,437 while its `close_btc` is ~2,290 on the same 2024-01 bar, i.e. `close_btc` holds ETH's price.
btc_corr_60 averages 0.797 there, not ~1.0, confirming it is a genuine cross-asset correlation and
not a self-reference. This module therefore feeds ETH into the *_btc slots, exactly as training saw
it. Feeding BTC into its own cross-asset slots would silently produce a self-correlation of ~1.0 on
every live bar and a train/inference mismatch of the kind CLAUDE.md's position-feature parity
contract exists to prevent.

Fetch/merge logic reused verbatim from live_regime_wide24_signal_20260826.py / the ETH GBM3 scorer
(same DAYS_BACK=15 warmup, same Binance endpoints, same causal merge_asof direction="backward"),
with SYMBOL/CROSS_SYMBOL swapped. Label and model provenance live in the artifact's own payload."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from live_regime_wide24_signal_20260826 import (  # noqa: E402
    DAYS_BACK, _fetch_data_api, _fetch_funding, _fetch_klines,
)
from retrain_clean_regime_hmm_raw_state12_20260517 import _with_raw_state12  # noqa: E402

SYMBOL = "BTCUSDT"          # subject asset
CROSS_SYMBOL = "ETHUSDT"    # cross-asset -> fills the *_btc columns, see the docstring trap note
HISTORY_BARS_RETURNED = 120  # matches the ETH regime scorers
MODEL_PATH = ROOT / "tmp/btc_regime_s24k3_20260902/model.joblib"
CLASSES3 = ["bull", "bear", "chop"]


def _empty(err: str | None) -> dict[str, Any]:
    return {"warmed_up": False, "error": err, "latest_bar_utc": None, "bull_prob": None,
            "bear_prob": None, "chop_prob": None, "confidence": None, "history": []}


def compute_regime_btc_signal() -> dict[str, Any]:
    """Same return contract as compute_regime_gbm3_signal(). Never raises -- degrades to
    warmed_up=False so the dashboard ribbon falls back to its waiting state."""
    try:
        from features.engineering import FeatureEngineer

        now = pd.Timestamp.now("UTC").tz_localize(None)
        end_ms = int(now.timestamp() * 1000)
        start_ms = int((now - pd.Timedelta(days=DAYS_BACK)).timestamp() * 1000)

        subj_kline = _fetch_klines(SYMBOL, start_ms, end_ms)
        cross_kline = _fetch_klines(CROSS_SYMBOL, start_ms, end_ms)
        oi = _fetch_data_api("/futures/data/openInterestHist", SYMBOL, start_ms, end_ms,
                             {"sumOpenInterestValue": "sum_open_interest_value"})
        top_ratio = _fetch_data_api("/futures/data/topLongShortPositionRatio", SYMBOL, start_ms, end_ms,
                                    {"longShortRatio": "sum_toptrader_long_short_ratio"})
        acct_ratio = _fetch_data_api("/futures/data/globalLongShortAccountRatio", SYMBOL, start_ms, end_ms,
                                     {"longShortRatio": "count_long_short_ratio"})
        funding = _fetch_funding(SYMBOL, start_ms, end_ms)

        raw = subj_kline.copy()
        for extra in (oi, top_ratio, acct_ratio):
            raw = pd.merge_asof(raw.sort_values("timestamp"), extra.sort_values("timestamp"),
                                on="timestamp", direction="backward")
        raw = pd.merge_asof(raw.sort_values("timestamp"), funding, on="timestamp", direction="backward")

        cross = cross_kline.rename(columns={"close": "close_btc", "volume": "volume_btc",
                                            "quote_volume": "quote_volume_btc"})
        raw = raw.merge(cross[["timestamp", "close_btc", "volume_btc", "quote_volume_btc"]],
                        on="timestamp", how="left")
        raw = raw.dropna(subset=["close_btc", "sum_open_interest_value", "last_funding_rate",
                                 "sum_toptrader_long_short_ratio", "count_long_short_ratio"]).reset_index(drop=True)
        if raw.empty:
            return _empty("no_rows_after_merge")

        subj_cols = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume",
                     "trades", "taker_buy_base", "taker_buy_quote", "sum_open_interest_value",
                     "sum_toptrader_long_short_ratio", "count_long_short_ratio", "last_funding_rate"]
        cross_cols = ["timestamp", "close_btc", "volume_btc", "quote_volume_btc"]
        feats = FeatureEngineer().process(raw[subj_cols].copy(), raw[cross_cols].copy())
        # same 8 state7_*/state12_* derivation the training pipeline applies -- omitting this is the
        # exact 2026-08-26 bug that silently pushed those columns onto TRAIN-median fallbacks
        feats = _with_raw_state12(feats)

        payload = joblib.load(MODEL_PATH)
        cols = payload["feature_cols"]
        med = pd.Series(payload["feature_medians"])
        for c in [c for c in cols if c not in feats.columns]:
            feats[c] = med.get(c, 0.0)
        x = feats[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
        proba = payload["model"].predict_proba(x)

        out = pd.DataFrame({"timestamp": feats["timestamp"].reset_index(drop=True)})
        for i, name in enumerate(CLASSES3):
            out[f"{name}_prob"] = proba[:, i]
        out["confidence"] = proba.max(axis=1)
        out = out.dropna().tail(HISTORY_BARS_RETURNED).reset_index(drop=True)
        if out.empty:
            return _empty("no_valid_regime_rows")

        history = [{"ts_ms": int(r.timestamp.value // 1_000_000), "bull_prob": float(r.bull_prob),
                    "bear_prob": float(r.bear_prob), "chop_prob": float(r.chop_prob),
                    "confidence": float(r.confidence)} for r in out.itertuples()]
        last = history[-1]
        return {"warmed_up": True, "error": None,
                "latest_bar_utc": out["timestamp"].iloc[-1].isoformat() + "Z",
                "bull_prob": last["bull_prob"], "bear_prob": last["bear_prob"],
                "chop_prob": last["chop_prob"], "confidence": last["confidence"],
                "history": history}
    except Exception as e:  # noqa: BLE001 -- fetch/model errors all degrade the same way
        return _empty(f"regime_btc_error: {e}")


if __name__ == "__main__":
    import json
    r = compute_regime_btc_signal()
    print(json.dumps({k: v for k, v in r.items() if k != "history"}, indent=2))
    print(f"history rows: {len(r['history'])}")
