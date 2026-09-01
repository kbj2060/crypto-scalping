#!/usr/bin/env python3
"""Read-only 3-class (bull/bear/chop) regime signal for the Snapshot tab's liquidation-map chart,
2026-08-26. Replaces live_regime_wide24_signal_20260826.py's wide24 HMM+linear-calibration model
with an independently-trained HistGradientBoostingClassifier (see memory
eth_regime_hierarchical_whipsaw_circularity_rejected_20260826 for the full audit trail this model
descends from -- a whipsaw sub-class was investigated and found to add too much false-positive
noise for a dashboard chip at every feature set tried; dropping it and going back to 3-class with
this richer independently-trained model gave OOS balanced_accuracy=0.9189 on 2026-07-01~08-19,
vs 0.7691 for the original wide24 HMM baseline -- both val (0.9292) and OOS (0.9189) agree closely,
no overfitting signal).

Model artifact: tmp/eth_regime_gbm3_independent_20260826/model.joblib (feature_cols/feature_medians/
model/classes). Trained on the full postfix-clean TRAIN file (2024-01-01~2026-06-30); the 24 wide24
feature_cols are a subset of this model's 136 features, plus ~112 more (CVD, funding, cross-asset
BTC, compression, VWAP, wick/sweep, volume-profile, etc.) already computed by
features.engineering.FeatureEngineer().process() -- no extra data sources needed beyond what the
wide24 signal already fetches.

2026-09-02 LABEL CHANGE (S12_K3) -- model config, the 136 feature_cols and this file's whole
fetch/score path are UNCHANGED; only the trained artifact (and therefore the target label) differs.
The new label is the scale-parameterized 3-class family at S=12 (1h efficiency-ratio legs, 2h
direction anchor) with a K=3 (15min) confirm, replacing RegimeEngine's 2h/4h-scale raw label:
    er_12=|c-c[-12]|/sum|diff|(12), er_24 likewise; net_24=c-c[-24]; slope_12=EMA(c,12).pct_change()
    trend=(er_12>=T1)|(er_24>=T2); bull=trend&net_24>0&slope_12>0; bear=mirror; chop=rest; debounce K=3
T1/T2 are percentile-matched on TRAIN ONLY to the old label's own firing rates, so class shares stay
comparable (bull .228 / bear .214 / chop .558).

⚠️ CLASSIFICATION ACCURACY REGRESSES AND THAT IS ACCEPTED, NOT OVERLOOKED: OOS bal_acc 0.8550 vs
0.9108, chop precision 0.8670 vs 0.9202. It ships because the two things this label is actually for
both improve -- evidence-signal gate quality (predicted-chop conditional lift +9.8% pooled, 14 of 16
signal-side cells positive, vs the old label's -0.8% and 6 of 14, where it was actively HARMFUL to
taker_delta_z_climax at -0.301 OOS) and display stability (predicted flip_rate 0.0965 vs 0.1803,
median state 7 bars vs 3). User reviewed the side-by-side regime charts and approved on 2026-09-02.
The OOS window used for those figures (2026-07-01~08-19) had already been consumed by ~8+ prior
regime rounds -- a research/dev score, NOT a single-touch Fresh-Forward result.
Full study chain: docs/experiments/eth_regime_scalping_label_geometry_20260902.md (Phase 1),
eth_regime_label_conditional_lift_20260902.md (Phase 2), eth_regime_s12k3_label_train_20260902.md
(Phase 3/3b). Trainer: scripts/train_eth_regime_s12k3_20260902.py.

Fetch/merge logic reused verbatim from live_regime_wide24_signal_20260826.py (same DAYS_BACK=15
warmup rationale, same Binance endpoints, same causal merge_asof direction="backward")."""
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

HISTORY_BARS_RETURNED = 120  # matches live_regime_wide24_signal_20260826.py
# 2026-09-02: label swapped to S12_K3 (see the "2026-09-02 LABEL CHANGE" block in the
# module docstring). ROLLBACK = point this single line back at
# tmp/eth_regime_gbm3_independent_20260826/model.joblib, which is still present on the
# server untouched; nothing else in this file or in dashboard/server.py depends on which
# of the two artifacts is loaded (identical payload schema and class order).
MODEL_PATH = ROOT / "tmp/eth_regime_s12k3_20260902/model.joblib"
CLASSES3 = ["bull", "bear", "chop"]


def compute_regime_gbm3_signal() -> dict[str, Any]:
    """Returns {"warmed_up", "error", "latest_bar_utc", "bull_prob", "bear_prob", "chop_prob",
    "confidence", "history": [{"ts_ms", "bull_prob", "bear_prob", "chop_prob", "confidence"}, ...]}
    -- same contract as compute_regime_wide24_signal(), drop-in swap for dashboard/server.py.
    Never raises -- degrades to warmed_up=False."""
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
            return {"warmed_up": False, "error": "no_rows_after_merge", "latest_bar_utc": None,
                    "bull_prob": None, "bear_prob": None, "chop_prob": None, "confidence": None, "history": []}

        eth_raw_cols = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume",
                         "trades", "taker_buy_base", "taker_buy_quote",
                         "sum_open_interest_value", "sum_toptrader_long_short_ratio",
                         "count_long_short_ratio", "last_funding_rate"]
        btc_raw_cols = ["timestamp", "close_btc", "volume_btc", "quote_volume_btc"]
        eth_df = raw[eth_raw_cols].copy()
        btc_df = raw[btc_raw_cols].copy()

        feats = FeatureEngineer().process(eth_df, btc_df)
        # model.feature_cols includes 8 state7_*/state12_* columns that FeatureEngineer().process()
        # doesn't produce on its own -- _with_raw_state12() derives them from columns FeatureEngineer
        # already computed (mtf_trend_1h/4h, hma_slope, breakout_strength, chop_index, etc.), same as
        # the training pipeline (scripts/train_and_persist_gbm3_final.py) and the wide24 signal's own
        # _transform(). 2026-08-26 bugfix: this call was missing, so those 8 columns silently fell
        # into the "missing -> fill with TRAIN median" fallback below instead of their real live
        # values -- no error, just quietly-wrong features on every live prediction until now.
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
        for i, name in enumerate(CLASSES3):
            out[f"{name}_prob"] = proba[:, i]
        out["confidence"] = proba.max(axis=1)
        out = out.dropna().tail(HISTORY_BARS_RETURNED).reset_index(drop=True)
        if out.empty:
            return {"warmed_up": False, "error": "no_valid_regime_rows", "latest_bar_utc": None,
                    "bull_prob": None, "bear_prob": None, "chop_prob": None, "confidence": None, "history": []}

        history = [
            {
                "ts_ms": int(row.timestamp.value // 1_000_000),
                "bull_prob": float(row.bull_prob),
                "bear_prob": float(row.bear_prob),
                "chop_prob": float(row.chop_prob),
                "confidence": float(row.confidence),
            }
            for row in out.itertuples()
        ]
        last = history[-1]
        return {
            "warmed_up": True, "error": None,
            "latest_bar_utc": out["timestamp"].iloc[-1].isoformat() + "Z",
            "bull_prob": last["bull_prob"], "bear_prob": last["bear_prob"],
            "chop_prob": last["chop_prob"], "confidence": last["confidence"],
            "history": history,
        }
    except Exception as e:  # noqa: BLE001 -- upstream fetch/model errors all degrade the same way
        return {"warmed_up": False, "error": f"regime_gbm3_error: {e}", "latest_bar_utc": None,
                "bull_prob": None, "bear_prob": None, "chop_prob": None, "confidence": None, "history": []}


if __name__ == "__main__":
    import json as _json
    result = compute_regime_gbm3_signal()
    history = result.pop("history", [])
    print(_json.dumps(result, indent=2))
    print(f"history: {len(history)} bars")
