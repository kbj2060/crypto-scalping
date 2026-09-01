#!/usr/bin/env python3
"""Read-only wide24 HMM regime signal (bull/bear/chop probability per 5-min bar) for the Snapshot
tab's liquidation-map chart, 2026-08-26.

Loads the CONFIRMED states=24/sticky=0.90/seed=7529 wide24 HMM artifact (N=5 real-random-seed
robust, contamination-bug-fixed refit -- see docs/model_contracts/
ilias_eth_human_direction_risk_management_contract_20260817.md "레짐 분류기 계약" section and
memory eth_regime_classifier_wide24_vs_jm_sjm_investigation_20260821), not whatever
DEFAULT_CURRENT_REGIME_PATH in trading_bot_modules/odyssey_regime3_live.py currently resolves to --
that live-bot path was last confirmed (2026-08-26) to still point at an older 2026-05-30
balancedish/state12-default-hyperparams artifact, and dashboard vs. live-bot cutover are
deliberately decoupled here (see memory
eth_wide24_hmm_live_artifact_vs_confirmed_research_discrepancy_20260826 -- "대시보드용으로는 별
문제 없음, 대시보드는 라이브 봇과 별개로 CONFIRMED 아티팩트를 직접 로드해서 씀").

Ported from the session scratchpad script that first produced this chart (fetch/merge/transform
logic reproduced verbatim, only the CSV-writing main() replaced with a dict-returning function).
Runs the EXACT production feature pipeline (features.engineering.FeatureEngineer().process(), same
as scripts/build_eth_panel_for_regime_comparison_20260808.py) over freshly-fetched Binance data --
DAYS_BACK=15 is kept unchanged from the validated script rather than shortened for speed, since
some of wide24's 24 hand-crafted features may need multi-day rolling windows to warm up correctly;
shrinking it risks silently feeding the model out-of-distribution inputs it was never validated
against. This makes a full recompute take on the order of 10-20s (mostly paginated-fetch network
time), which is why dashboard/server.py caches this behind a multi-minute TTL rather than calling
it per-request -- regime is also inherently slow-moving (sticky=0.90 means the HMM itself resists
flipping bar-to-bar), so a coarse cache does not meaningfully stale the reading.

_transform() is not imported from scripts/experiment_regime3_current_hmm_wide24_20260529.py --
that module's import chain pulls in scripts.train_regime3_hmm_mamba_20260529, which requires
mamba_ssm (GPU-only, not installed on this dev machine). Reproduced verbatim below from that
module's lines 282-298, minus the _eval(y, proba) accuracy-scoring tail (not needed for live
inference, and that's the part that actually needs the mamba-gated label import)."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

FAPI = "https://fapi.binance.com"
SYMBOL = "ETHUSDT"
BTC_SYMBOL = "BTCUSDT"
DAYS_BACK = 15  # kept equal to the validated script -- see module docstring, do not shrink casually
HISTORY_BARS_RETURNED = 120  # 10h of 5-min bars -- comfortably covers the 6h/72-candle chart window
MODEL_PATH = (ROOT / "tmp/eth_hmm_wide24_resweep_train2026h1_20260821"
              / "postfix_recheck_states24_sticky0.90_seed7529/models/regime3_current_sensitive_v2_hmm_wide24_2024.joblib")
CLASSES3 = ["bull", "bear", "chop"]


def _class_proba(state_prob: np.ndarray, state_class: np.ndarray) -> np.ndarray:
    proba = state_prob @ state_class
    proba /= np.clip(proba.sum(axis=1, keepdims=True), 1e-300, None)
    return proba


def _transform(payload: dict, frame: pd.DataFrame) -> pd.DataFrame:
    from retrain_clean_regime_hmm_raw_state12_20260517 import _with_raw_state12
    cols = payload["feature_cols"]
    out = _with_raw_state12(frame.copy())
    for col in cols:
        if col not in out.columns:
            raise ValueError(f"missing current HMM feature column: {col}")
    med = pd.Series(payload["feature_medians"])
    x_raw = out[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    obs = payload["scaler"].transform(x_raw)
    proba = _class_proba(payload["model"].filter_proba(obs), payload["state_class_matrix"])
    result = pd.DataFrame({"timestamp": out["timestamp"].reset_index(drop=True)})
    prefix = f"{payload['prefix_stem']}_{payload['feature_set']}_"
    for i, name in enumerate(CLASSES3):
        result[f"{prefix}{name}_prob"] = proba[:, i]
    sp = np.sort(proba, axis=1)
    result[f"{prefix}confidence"] = sp[:, -1]
    return result


def _fetch_klines(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    out: list = []
    cur = start_ms
    while cur < end_ms:
        params = {"symbol": symbol, "interval": "5m", "limit": 1500, "startTime": cur, "endTime": end_ms}
        r = requests.get(f"{FAPI}/fapi/v1/klines", params=params, timeout=15)
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
    cols = ["timestamp", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    df = pd.DataFrame(out, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True).dt.tz_localize(None)
    for c in ("open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"):
        df[c] = df[c].astype(float)
    df["trades"] = df["trades"].astype(int)
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def _fetch_data_api(path: str, symbol: str, start_ms: int, end_ms: int, field_map: dict) -> pd.DataFrame:
    out: list = []
    cur = start_ms
    while cur < end_ms:
        params = {"symbol": symbol, "period": "5m", "limit": 500, "startTime": cur, "endTime": end_ms}
        r = requests.get(f"{FAPI}{path}", params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        out.extend(data)
        last_ts = int(data[-1]["timestamp"])
        if last_ts <= cur:
            break
        cur = last_ts + 1
        time.sleep(0.15)
        if len(data) < 500:
            break
    df = pd.DataFrame(out)
    if df.empty:
        raise RuntimeError(f"{path} returned no data")
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(np.int64), unit="ms", utc=True).dt.tz_localize(None)
    df = df.rename(columns=field_map)
    value_cols = list(field_map.values())
    for c in value_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[["timestamp"] + value_cols]
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def _fetch_funding(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    out: list = []
    cur = start_ms
    while cur < end_ms:
        params = {"symbol": symbol, "startTime": cur, "endTime": end_ms, "limit": 1000}
        r = requests.get(f"{FAPI}/fapi/v1/fundingRate", params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        out.extend(data)
        last_ts = int(data[-1]["fundingTime"])
        if last_ts <= cur:
            break
        cur = last_ts + 1
        time.sleep(0.15)
        if len(data) < 1000:
            break
    df = pd.DataFrame(out)
    df["timestamp"] = pd.to_datetime(df["fundingTime"].astype(np.int64), unit="ms", utc=True).dt.tz_localize(None)
    df["last_funding_rate"] = df["fundingRate"].astype(float)
    return df[["timestamp", "last_funding_rate"]].drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def compute_regime_wide24_signal() -> dict[str, Any]:
    """Returns {"warmed_up", "error", "latest_bar_utc", "bull_prob", "bear_prob", "chop_prob",
    "confidence", "history": [{"ts_ms", "bull_prob", "bear_prob", "chop_prob", "confidence"}, ...]}
    (history oldest-to-newest, up to HISTORY_BARS_RETURNED bars). Never raises -- degrades to
    warmed_up=False, same contract as compute_liquidation_5m_signal()/compute_oi_delta_signal()."""
    try:
        from features.engineering import FeatureEngineer

        now = pd.Timestamp.utcnow().tz_localize(None)
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

        payload = joblib.load(MODEL_PATH)
        sidecar = _transform(payload, feats)
        prefix = f"{payload['prefix_stem']}_{payload['feature_set']}_"
        out = sidecar[["timestamp", f"{prefix}bull_prob", f"{prefix}bear_prob", f"{prefix}chop_prob", f"{prefix}confidence"]].copy()
        out.columns = ["timestamp", "bull_prob", "bear_prob", "chop_prob", "confidence"]
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
        return {"warmed_up": False, "error": f"regime_wide24_error: {e}", "latest_bar_utc": None,
                "bull_prob": None, "bear_prob": None, "chop_prob": None, "confidence": None, "history": []}


if __name__ == "__main__":
    import json as _json
    result = compute_regime_wide24_signal()
    history = result.pop("history", [])
    print(_json.dumps(result, indent=2))
    print(f"history: {len(history)} bars")
