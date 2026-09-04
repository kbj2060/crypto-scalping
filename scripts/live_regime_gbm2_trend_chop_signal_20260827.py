#!/usr/bin/env python3
"""Read-only 2-class (trend vs chop) regime signal for the Snapshot tab's liquidation-map chart,
2026-08-27. Replaces live_regime_gbm3_signal_20260826.py's bull/bear/chop model with
eth_regime_gbm2_trend_chop_20260827 -- see scripts/train_eth_regime_gbm2_trend_chop_20260827.py's
module docstring for why this is 2-class (not a 3rd "transition" class -- explicitly rejected by
docs/model_contracts/regime3_whipsaw_risk_policy_20260529.md and
docs/active_live/regime3_policy_20260530.md) and why the training target itself is a k_bars=12
(1h) debounced RegimeEngine label rather than the raw per-bar one.

Two independent stability layers, both already baked into the model artifact -- this script does
not choose either, it only reads and applies what training already selected:
  1. The model was TRAINED to predict an already-debounced target, so its own predictions are
     structurally biased toward fewer flips than a model trained on the raw label would produce.
  2. `payload["hysteresis_config"]` (k_bars/band, selected on VAL by the training script) is applied
     on top of the model's live probability output as a light secondary smoothing pass -- this
     mainly filters residual boundary noise in the model's own predict_proba, it is not the primary
     source of stability.

Fetch/merge/feature logic reused verbatim from live_regime_gbm3_signal_20260826.py (itself reused
from live_regime_wide24_signal_20260826.py) -- no new data sources."""
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
    SYMBOL, BTC_SYMBOL, DAYS_BACK, _fetch_klines, _fetch_data_api, _fetch_funding,
)
from retrain_clean_regime_hmm_raw_state12_20260517 import _with_raw_state12  # noqa: E402
from train_eth_regime_gbm2_trend_chop_20260827 import _apply_hysteresis  # noqa: E402

HISTORY_BARS_RETURNED = 120  # matches live_regime_gbm3_signal_20260826.py
MODEL_PATH = ROOT / "tmp/eth_regime_gbm2_trend_chop_20260827/model.joblib"
CLASSES2 = ["chop", "trend"]

_EMPTY: dict[str, Any] = {
    "warmed_up": False, "error": None, "latest_bar_utc": None,
    "trend_prob": None, "chop_prob": None, "confidence": None,
    "raw_state": None, "confirmed_state": None, "bars_since_confirm": None, "history": [],
}


def compute_regime_gbm2_trend_chop_signal() -> dict[str, Any]:
    """Returns {"warmed_up", "error", "latest_bar_utc", "trend_prob", "chop_prob", "confidence",
    "raw_state", "confirmed_state", "bars_since_confirm",
    "history": [{"ts_ms", "trend_prob", "chop_prob", "confidence", "raw_state", "confirmed_state"}, ...]}.
    Deliberately drops bull_prob/bear_prob (GBM3's fields) rather than faking zeros -- this model
    cannot provide direction, only trend-vs-chop. Fully stateless: recomputes raw+confirmed states
    from scratch over the whole DAYS_BACK fetch window every call (no persisted state needed -- the
    largest k_bars in use, either the label's 12 or the serving hysteresis's small k, settles long
    before the last HISTORY_BARS_RETURNED bars). Never raises -- degrades to warmed_up=False."""
    try:
        from features.engineering import FeatureEngineer

        now = pd.Timestamp.now("UTC").tz_localize(None)
        end_ms = int(now.timestamp() * 1000)
        start_ms = int((now - pd.Timedelta(days=DAYS_BACK)).timestamp() * 1000)

        eth_kline = _fetch_klines(SYMBOL, start_ms, end_ms)
        btc_kline = _fetch_klines(BTC_SYMBOL, start_ms, end_ms)
        oi = _fetch_data_api("/futures/data/openInterestHist", SYMBOL, start_ms, end_ms,
                              {"sumOpenInterestValue": "sum_open_interest_value"})
        top_ratio = _fetch_data_api("/futures/data/topLongShortPositionRatio", SYMBOL, start_ms, end_ms,
                                     {"longShortRatio": "sum_toptrader_long_short_ratio"})
        acct_ratio = _fetch_data_api("/futures/data/globalLongShortAccountRatio", SYMBOL, start_ms, end_ms,
                                      {"longShortRatio": "count_long_short_ratio"})
        funding = _fetch_funding(SYMBOL, start_ms, end_ms)

        raw = eth_kline.copy()
        for extra in (oi, top_ratio, acct_ratio):
            raw = pd.merge_asof(raw.sort_values("timestamp"), extra.sort_values("timestamp"), on="timestamp", direction="backward")
        raw = pd.merge_asof(raw.sort_values("timestamp"), funding, on="timestamp", direction="backward")

        btc = btc_kline.rename(columns={"close": "close_btc", "volume": "volume_btc", "quote_volume": "quote_volume_btc"})
        raw = raw.merge(btc[["timestamp", "close_btc", "volume_btc", "quote_volume_btc"]], on="timestamp", how="left")
        raw = raw.dropna(subset=["close_btc", "sum_open_interest_value", "last_funding_rate",
                                  "sum_toptrader_long_short_ratio", "count_long_short_ratio"]).reset_index(drop=True)
        if raw.empty:
            return {**_EMPTY, "error": "no_rows_after_merge"}

        eth_raw_cols = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume",
                         "trades", "taker_buy_base", "taker_buy_quote",
                         "sum_open_interest_value", "sum_toptrader_long_short_ratio",
                         "count_long_short_ratio", "last_funding_rate"]
        btc_raw_cols = ["timestamp", "close_btc", "volume_btc", "quote_volume_btc"]
        eth_df = raw[eth_raw_cols].copy()
        btc_df = raw[btc_raw_cols].copy()

        feats = FeatureEngineer().process(eth_df, btc_df)
        feats = _with_raw_state12(feats)

        payload = joblib.load(MODEL_PATH)
        cols = payload["feature_cols"]
        med = pd.Series(payload["feature_medians"])
        missing = [c for c in cols if c not in feats.columns]
        for c in missing:
            feats[c] = med.get(c, 0.0)
        x = feats[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
        proba = payload["model"].predict_proba(x)

        out = pd.DataFrame({"timestamp": feats["timestamp"].reset_index(drop=True)})
        out["trend_prob"] = proba[:, CLASSES2.index("trend")]
        out["chop_prob"] = proba[:, CLASSES2.index("chop")]
        out["confidence"] = proba.max(axis=1)
        out = out.dropna().reset_index(drop=True)
        if out.empty:
            return {**_EMPTY, "error": "no_valid_regime_rows"}

        hcfg = payload["hysteresis_config"]
        raw_codes = (out["trend_prob"].to_numpy() >= 0.5).astype(int)
        confirmed_codes = _apply_hysteresis(out["trend_prob"].to_numpy(), hcfg["k_bars"], hcfg["band"])
        out["raw_state"] = [CLASSES2[i] for i in raw_codes]
        out["confirmed_state"] = [CLASSES2[i] for i in confirmed_codes]

        # bars_since_confirm: how many bars the *confirmed* state has held its current value
        bars_since_confirm = 1
        for i in range(len(confirmed_codes) - 2, -1, -1):
            if confirmed_codes[i] == confirmed_codes[-1]:
                bars_since_confirm += 1
            else:
                break

        out = out.tail(HISTORY_BARS_RETURNED).reset_index(drop=True)
        history = [
            {
                "ts_ms": int(row.timestamp.value // 1_000_000),
                "trend_prob": float(row.trend_prob),
                "chop_prob": float(row.chop_prob),
                "confidence": float(row.confidence),
                "raw_state": row.raw_state,
                "confirmed_state": row.confirmed_state,
            }
            for row in out.itertuples()
        ]
        last = history[-1]
        return {
            "warmed_up": True, "error": None,
            "latest_bar_utc": out["timestamp"].iloc[-1].isoformat() + "Z",
            "trend_prob": last["trend_prob"], "chop_prob": last["chop_prob"], "confidence": last["confidence"],
            "raw_state": last["raw_state"], "confirmed_state": last["confirmed_state"],
            "bars_since_confirm": bars_since_confirm,
            "history": history,
        }
    except Exception as e:  # noqa: BLE001 -- upstream fetch/model errors all degrade the same way
        return {**_EMPTY, "error": f"regime_gbm2_error: {e}"}


if __name__ == "__main__":
    import json as _json
    result = compute_regime_gbm2_trend_chop_signal()
    history = result.pop("history", [])
    print(_json.dumps(result, indent=2))
    print(f"history: {len(history)} bars")
