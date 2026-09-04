#!/usr/bin/env python3
"""배포된 레짐 스코어러가 **파생 500행 제약**에 영향받는가 (2026-09-03).

입증된 결함: 바이낸스 `/futures/data/*`는 `startTime`을 무시하고 **최근 500행(약 41시간)만**
준다. 진입 모델에서 그 결과 FeatureEngineer 입력이 4,319봉 → 499봉으로 잘렸고 **136피쳐 중
13개가 어긋났다**(`funding_pressure`는 상관 −0.526으로 부호까지 뒤집힘). 4,000봉을 주면
불일치 0이었다.

⚠️`live_regime_gbm3_signal_20260826.py`가 **같은 `_fetch_data_api`를 쓴다.** 그쪽은 마지막
봉만 채점하므로 영향이 다를 수 있지만, 같은 136피쳐를 쓰므로 확인이 필요하다.
(진입 모델 문서에 "미측정"으로 남겨뒀던 항목이다.)

방법: 같은 최근 봉들에 대해 레짐 예측을 두 경로로 만들어 비교한다.
  A 라이브 그대로   -- API 파생만(≈500행)
  B 이력 보강       -- 일별 metrics 덤프로 메운 긴 이력(진입 모델의 `_deriv_history`)
클래스 일치율과 확률 차이를 본다. **불일치가 크면 배포 중인 레짐 신호가 지금 어긋나 있다.**
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CLASSES3 = ["bull", "bear", "chop"]


def log(m): print(f"[regaudit] {m}", flush=True)


def main() -> int:
    import live_eth_entry_limit_fade_signal_20260903 as S
    from live_regime_wide24_signal_20260826 import (
        DAYS_BACK, _fetch_klines, _fetch_data_api, _fetch_funding, SYMBOL, BTC_SYMBOL)
    from live_regime_gbm3_signal_20260826 import MODEL_PATH
    from features.engineering import FeatureEngineer
    from retrain_clean_regime_hmm_raw_state12_20260517 import _with_raw_state12

    now = pd.Timestamp.now("UTC").tz_localize(None)
    end_ms = int(now.timestamp() * 1000)
    start_ms = int((now - pd.Timedelta(days=DAYS_BACK)).timestamp() * 1000)
    eth = S._drop_forming(_fetch_klines(SYMBOL, start_ms, end_ms))
    btc = S._drop_forming(_fetch_klines(BTC_SYMBOL, start_ms, end_ms))
    fund = _fetch_funding(SYMBOL, start_ms, end_ms)
    api = _fetch_data_api("/futures/data/openInterestHist", SYMBOL, start_ms, end_ms,
                          {"sumOpenInterestValue": "sum_open_interest_value"})
    for path, fm in (("/futures/data/topLongShortPositionRatio",
                      {"longShortRatio": "sum_toptrader_long_short_ratio"}),
                     ("/futures/data/globalLongShortAccountRatio",
                      {"longShortRatio": "count_long_short_ratio"})):
        api = api.merge(_fetch_data_api(path, SYMBOL, start_ms, end_ms, fm),
                        on="timestamp", how="outer")
    hist = S._deriv_history(now - pd.Timedelta(days=DAYS_BACK), now)
    log(f"klines {len(eth):,}봉 · API 파생 {len(api):,}행 · 덤프 이력 {len(hist):,}행")

    DERIV = ["sum_open_interest_value", "sum_toptrader_long_short_ratio",
             "count_long_short_ratio", "last_funding_rate"]

    def build(met):
        m2 = met.dropna(subset=S.MET_COLS).drop_duplicates(
            "timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
        raw = eth.copy()
        for extra in (m2, fund):
            raw = pd.merge_asof(raw.sort_values("timestamp"), extra.sort_values("timestamp"),
                                on="timestamp", direction="backward")
        b = btc.rename(columns={"close": "close_btc", "volume": "volume_btc",
                                "quote_volume": "quote_volume_btc"})
        raw = raw.merge(b[["timestamp", "close_btc", "volume_btc", "quote_volume_btc"]],
                        on="timestamp", how="left")
        raw = raw.dropna(subset=["close_btc"] + DERIV).reset_index(drop=True)
        ed = raw[["timestamp", "open", "high", "low", "close", "volume", "quote_volume",
                  "trades", "taker_buy_base", "taker_buy_quote"] + DERIV].copy()
        bd = raw[["timestamp", "close_btc", "volume_btc", "quote_volume_btc"]].copy()
        return _with_raw_state12(FeatureEngineer().process(ed, bd)), len(ed)

    fa, na = build(api)
    fb, nb = build(pd.concat([hist, api], ignore_index=True))
    log(f"A 라이브(API만) FeatureEngineer 입력 {na:,}봉 → 출력 {len(fa):,}행")
    log(f"B 이력 보강      FeatureEngineer 입력 {nb:,}봉 → 출력 {len(fb):,}행")

    src = joblib.load(MODEL_PATH)
    cols, med, mdl = src["feature_cols"], src["feature_medians"], src["model"]

    def score(F):
        X = pd.DataFrame({c: (F[c] if c in F.columns else np.nan) for c in cols})
        X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        for c in cols:
            X[c] = X[c].fillna(med.get(c, 0.0))
        return mdl.predict_proba(X[cols])

    m = fa[["timestamp"]].merge(fb[["timestamp"]], on="timestamp", how="inner")
    A = fa[fa.timestamp.isin(m.timestamp)].reset_index(drop=True)
    B = fb[fb.timestamp.isin(m.timestamp)].reset_index(drop=True)
    pa, pb = score(A), score(B)
    ca, cb = pa.argmax(1), pb.argmax(1)
    agree = float((ca == cb).mean())
    log(f"\n겹치는 봉 {len(A):,} · ⭐클래스 일치율 **{agree*100:.2f}%**")
    print(f"\n{'클래스':>8s}{'A 라이브':>10s}{'B 보강':>10s}{'평균|Δp|':>12s}{'최대|Δp|':>12s}")
    for i, nm in enumerate(CLASSES3):
        d = np.abs(pa[:, i] - pb[:, i])
        print(f"{nm:>8s}{float((ca==i).mean()):10.3f}{float((cb==i).mean()):10.3f}"
              f"{d.mean():12.4f}{d.max():12.4f}")
    print(f"\n⭐**최신 봉**(라이브가 실제로 내보내는 값)")
    print(f"  A 라이브: " + " ".join(f"{n} {pa[-1,i]:.4f}" for i, n in enumerate(CLASSES3))
          + f"  → {CLASSES3[ca[-1]]}")
    print(f"  B 보강  : " + " ".join(f"{n} {pb[-1,i]:.4f}" for i, n in enumerate(CLASSES3))
          + f"  → {CLASSES3[cb[-1]]}")
    print(f"  {'⚠️최신 봉 클래스 불일치' if ca[-1]!=cb[-1] else '✅최신 봉 클래스 일치'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
