#!/usr/bin/env python3
"""Read-only 3-class (bull/bear/chop) regime signal for **XRP** -- 2026-09-03.

`live_regime_btc_signal_20260902.py`의 자산 상수만 바꾼 포팅. 반환 dict 모양은 동일하므로
`dashboard/server.py`가 BTC판과 같은 방식으로 캐시/서빙한다.

## ⭐교차자산 컬럼 명명 함정 (학습/추론 파리티)

`FeatureEngineer`는 교차자산 컬럼을 `close_btc`/`volume_btc`/`quote_volume_btc`로 **하드코딩**한다.
어떤 자산을 넣든 이름은 `_btc`다. 자산마다 그 슬롯에 들어가는 게 다르다:

    ETH 캐노니컬  -> `_btc` 슬롯에 BTC
    BTC 캐노니컬  -> `_btc` 슬롯에 **ETH**  (live_regime_btc_signal_20260902.py)
    XRP 캐노니컬  -> `_btc` 슬롯에 **BTC**  (build_xrp_raw_frame_20260903.py)

그래서 여기 `CROSS_SYMBOL`은 BTCUSDT다. 학습 때 본 것과 같은 것을 넣지 않으면
CLAUDE.md의 position-feature 파리티 계약이 막으려는 바로 그 불일치가 조용히 생긴다.

## 모델

`tmp/xrp_regime_s96k9_20260903/model.joblib` (**S96_K9**, 2026-09-03 교체).

⚠️**2026-09-03 교체**: 이전 `S48_K6`은 격자 상단 경계에서 뽑힌 값이었다
(`SCALES=(6,12,24,48)`->S48, `DEBOUNCES=(1,3,6)`->K6, **둘 다 격자 끝**. 게다가 `DEBOUNCES`는
ETH 스크립트에서 import한 것이고 K=12 배제 근거도 ETH의 lock-up 관측이었다).
격자를 S->192 / K->12로 넓히니 `S96_K9`가 네 축 전부에서 이겼다 --
Phase2 15/16 vs 10/16, Phase3 bal_acc 0.8676 vs 0.8364, Phase3b 13/16 OOS +0.1437 vs 10/16 +0.0466,
플리커 0.0358 vs 0.0476. ⭐이 저장소에서 드물게 **학습가능성 트레이드오프가 없다**.

승격 근거는 **미사용 창 단일 노출**(2026-04-01~06-30, 사전등록 3기준 전부 통과):
bal_acc 0.8439>=0.8219 / pred_flip 0.0339<=0.0513 / chop_recall 0.9173>=0.85. 그 창은 소진됐다.
감사 전문: docs/experiments/xrp_tuning_gap_grid_boundary_audit_20260903.md

⚠️스케일이 4시간 -> **8시간**이 됐다(ETH S12=1h, BTC S24=2h 중 가장 느림).
XRP는 스케일·디바운스 두 축 모두 "느릴수록 좋다"가 일관되게 나온 자산이다.
"""

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

SYMBOL = "XRPUSDT"          # subject asset
CROSS_SYMBOL = "BTCUSDT"    # cross-asset -> fills the *_btc columns, see the docstring trap note
HISTORY_BARS_RETURNED = 120  # matches the ETH regime scorers
MODEL_PATH = ROOT / "tmp/xrp_regime_s96k9_20260903/model.joblib"   # 2026-09-03 S48_K6에서 교체
CLASSES3 = ["bull", "bear", "chop"]


def _empty(err: str | None) -> dict[str, Any]:
    return {"warmed_up": False, "error": err, "latest_bar_utc": None, "bull_prob": None,
            "bear_prob": None, "chop_prob": None, "confidence": None, "history": []}


def compute_regime_xrp_signal() -> dict[str, Any]:
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
        return _empty(f"regime_xrp_error: {e}")


if __name__ == "__main__":
    import json
    r = compute_regime_xrp_signal()
    print(json.dumps({k: v for k, v in r.items() if k != "history"}, indent=2))
    print(f"history rows: {len(r['history'])}")
