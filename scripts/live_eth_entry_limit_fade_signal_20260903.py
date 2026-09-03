#!/usr/bin/env python3
"""ETH 지정가 페이드 **진입 신호** -- 동결 v1(`eth_entry_limit_fade_v1_20260903`) 라이브 채점.

동결 정책 v3(`eth_entry_limit_fade_v3_l3arm1_20260903`):
    8종 raw 트리거 → **신호방향 지정가만**(arm1) depth 3.0×ATR · 대기 6봉 · 미체결 취소
    → TabPFN 분류 5멤버 확률 > p_thr → 4슬롯 → 트레일링 SL3.0/ARM1.0/Trail0.1

⚠️⚠️**v3는 승격 후보가 아니다.** 2026-09-03 전수조사에서 v1/v2의 라벨이 미래참조로 확인됐고
(체결 봉의 **체결 이전** 고가를 진입 후 이익으로 크레딧, 전체 후보 PF 2.86→0.95), 정직한
라벨에서는 **트리거가 무작위 봉보다 못하고**(VAL +1.48 vs +2.87) 독립 일수가 42~45일뿐이라
확립이 불가능했다. v3의 목적은 **정직한 기반 위에서 전진 데이터를 모으는 것**뿐이다.
전문: `docs/experiments/eth_entry_intrabar_fill_bar_credit_artifact_20260903.md`

⭐v2 대비 바뀐 것: ①L3 정직 라벨로 학습 ②**arm1(신호방향)만 제출** -- 역방향 팔이
아티팩트의 최대 수혜자였다 ③임계값을 arm1 TRAIN 분포에서 유도

이 모듈은 **후보를 만들고 채점만 한다.** 가상 원장·체결 판정은 섀도우 러너가 맡는다.
⚠️주문 함수를 import하지 않는다.

## ⭐형성 중 봉 (이 저장소의 반복 함정)

`_fetch_klines`가 저장소에 **둘** 있고 동작이 다르다:
  · `live_eth_sweep_v_rebound_signal_20260829._fetch_klines` -- 형성 봉을 버린다
  · `live_regime_wide24_signal_20260826._fetch_klines` -- **버리지 않는다** (endTime 없이 부르면
    바이낸스가 진행 중 캔들을 마지막 원소로 준다)
여기서는 후자(레짐 패널에 필요한 장기 페치)를 쓰므로 **직접 버린다**(`_drop_forming`).

## ⭐트리거는 RAW 컬럼이다

`compute_signals()`의 `bottom_{name}`/`top_{name}`을 쓴다 -- `_active`(sustain 롤링)가 아니다.
동결 모델은 causal fires(= raw 단일봉 발동)로 학습됐다.
그리고 `cluster_dedup`은 **쓰지 않는다** -- 앵커 선택이 군집의 미래를 봐야 하므로 미래참조다
(2026-09-02 감사, docs/experiments/eth_entry_limit_fade_model_20260903.md).

## 피쳐 161개

  base 22 (`build_indicator_frame`, Tier0) + arm/sig_id/atr_pct/depth 4
  + R136 (GBM3 레짐 모델의 feature_cols 136, 이름충돌은 `_r136` 접미사, 그 모델의 median으로 채움)
라이브 조립 경로는 `FeatureEngineer().process(eth_df, btc_df)` + `_with_raw_state12()`이며
2026-09-03 파리티 검증에서 156/156 불일치 0으로 확인됐다
(`scripts/verify_eth_entry_live_feature_parity_20260903.py`).
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ART_DIR = ROOT / "tmp/eth_entry_limit_fade_v3_l3arm1_20260903"
GBM3_PATH = ROOT / "tmp/eth_regime_s12k3_20260902/model.joblib"
SYMBOL, BTC_SYMBOL = "ETHUSDT", "BTCUSDT"

_ART: dict[str, Any] | None = None
_CARD: dict[str, Any] | None = None


def _art() -> tuple[dict, dict]:
    global _ART, _CARD
    if _ART is None:
        _ART = joblib.load(ART_DIR / "model.joblib")
        _CARD = json.loads((ART_DIR / "model_card.json").read_text())
    return _ART, _CARD


def _drop_forming(df: pd.DataFrame) -> pd.DataFrame:
    """진행 중 캔들 제거. 레짐 쪽 `_fetch_klines`는 이걸 안 한다(모듈 docstring 참조)."""
    if "close_time" not in df.columns or df.empty:
        return df
    ct = pd.to_datetime(df["close_time"])
    now = pd.Timestamp.utcnow().tz_localize(None)
    return df[ct <= now].reset_index(drop=True)


def _tz_aware(df: pd.DataFrame) -> pd.DataFrame:
    """`compute_signals()`는 대시보드 관행(tz-aware UTC)을 따르고, 레짐 페처는 tz-naive를 준다.
    두 관행이 한 프로세스에 공존하므로 경계에서 명시적으로 맞춘다."""
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    return d


MET_DIR = ROOT / "binance_data/metrics"
MET_URL = "https://data.binance.vision/data/futures/um/daily/metrics/{sym}/{sym}-metrics-{day}.zip"
WARMUP_BARS = 4000       # 실측 수렴점: 4,000봉(14일)에서 136피쳐 불일치 0 · 예측상관 1.0000
MET_COLS = ["sum_open_interest_value", "sum_toptrader_long_short_ratio", "count_long_short_ratio"]


def _metrics_day(day: str) -> pd.DataFrame | None:
    """일별 metrics 덤프 1일치(5분 288행). 없으면 공개 저장소에서 받아 캐시한다."""
    import zipfile
    f = MET_DIR / f"{SYMBOL}-metrics-{day}.zip"
    if not f.exists():
        import urllib.request
        try:
            MET_DIR.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(MET_URL.format(sym=SYMBOL, day=day), f)
        except Exception:                                          # noqa: BLE001
            if f.exists():
                f.unlink(missing_ok=True)
            return None                                            # 당일/전일은 아직 미공개(404)
    try:
        with zipfile.ZipFile(f) as z:
            d = pd.read_csv(z.open(z.namelist()[0]))
    except Exception:                                              # noqa: BLE001
        return None
    d["timestamp"] = pd.to_datetime(d["create_time"])
    return d[["timestamp"] + MET_COLS].sort_values("timestamp")


def _deriv_history(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """⭐파생거래소 3열의 **긴 이력**.

    바이낸스 `/futures/data/*` 엔드포인트는 `startTime`을 무시하고 **최근 500행(약 41시간)만**
    돌려준다 -- 페이지네이션 코드가 있어도 두 번째 호출이 같은 구간을 반환해 루프가 끝난다.
    그 500행으로 FeatureEngineer를 돌리면 136피쳐 중 13개가 어긋나고(`funding_pressure`는
    부호까지 뒤집힌다) 예측 상관이 0.9909로 내려간다. 4,000봉을 주면 불일치 0·상관 1.0000이다
    (2026-09-03 실측, docs/experiments/eth_entry_limit_fade_model_20260903.md).

    그래서 일별 덤프(T+1 공개)로 과거를 메우고 최근 구간만 API로 덮는다.
    """
    days = pd.date_range(start.normalize(), end.normalize(), freq="D")
    frames = [d for d in (_metrics_day(f"{x:%Y-%m-%d}") for x in days) if d is not None]
    if not frames:
        return pd.DataFrame(columns=["timestamp"] + MET_COLS)
    h = pd.concat(frames, ignore_index=True)
    return h.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def _assemble() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(kl, btc_kl, eth_df, btc_df) -- 전부 완결 봉. 파생거래소 4열은 eth_df에 병합돼 있다."""
    from live_regime_wide24_signal_20260826 import (
        DAYS_BACK, _fetch_klines, _fetch_data_api, _fetch_funding)

    now = pd.Timestamp.now("UTC").tz_localize(None)
    end_ms = int(now.timestamp() * 1000)
    start_ms = int((now - pd.Timedelta(days=DAYS_BACK)).timestamp() * 1000)

    eth = _drop_forming(_fetch_klines(SYMBOL, start_ms, end_ms))
    btc = _drop_forming(_fetch_klines(BTC_SYMBOL, start_ms, end_ms))
    # 최근 구간(API, 약 41시간) -- 덤프가 아직 안 나온 당일/전일을 덮는다
    api = _fetch_data_api("/futures/data/openInterestHist", SYMBOL, start_ms, end_ms,
                          {"sumOpenInterestValue": "sum_open_interest_value"})
    for path, fm in (("/futures/data/topLongShortPositionRatio",
                      {"longShortRatio": "sum_toptrader_long_short_ratio"}),
                     ("/futures/data/globalLongShortAccountRatio",
                      {"longShortRatio": "count_long_short_ratio"})):
        api = api.merge(_fetch_data_api(path, SYMBOL, start_ms, end_ms, fm),
                        on="timestamp", how="outer")
    hist = _deriv_history(now - pd.Timedelta(days=DAYS_BACK), now)
    met = pd.concat([hist, api], ignore_index=True)
    met = met.dropna(subset=MET_COLS).drop_duplicates(
        "timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
    fund = _fetch_funding(SYMBOL, start_ms, end_ms)

    raw = eth.copy()
    for extra in (met, fund):
        raw = pd.merge_asof(raw.sort_values("timestamp"), extra.sort_values("timestamp"),
                            on="timestamp", direction="backward")
    b = btc.rename(columns={"close": "close_btc", "volume": "volume_btc",
                            "quote_volume": "quote_volume_btc"})
    raw = raw.merge(b[["timestamp", "close_btc", "volume_btc", "quote_volume_btc"]],
                    on="timestamp", how="left")
    DERIV = ["sum_open_interest_value", "sum_toptrader_long_short_ratio",
             "count_long_short_ratio", "last_funding_rate"]
    raw = raw.dropna(subset=["close_btc"] + DERIV).reset_index(drop=True)
    eth_df = raw[["timestamp", "open", "high", "low", "close", "volume", "quote_volume",
                  "trades", "taker_buy_base", "taker_buy_quote"] + DERIV].copy()
    btc_df = raw[["timestamp", "close_btc", "volume_btc", "quote_volume_btc"]].copy()
    return eth, btc, eth_df, btc_df


def _feature_frame(kl: pd.DataFrame, eth_df: pd.DataFrame, btc_df: pd.DataFrame,
                   feat_cols: list[str]) -> pd.DataFrame:
    """라이브 161피쳐 프레임(arm/sig_id/depth 제외 -- 후보별로 채운다)."""
    from features.engineering import FeatureEngineer
    from retrain_clean_regime_hmm_raw_state12_20260517 import _with_raw_state12
    from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import build_indicator_frame

    live = _with_raw_state12(FeatureEngineer().process(eth_df, btc_df))
    live = live.loc[:, ~pd.Index(live.columns).duplicated()]

    # R136: GBM3 레짐 모델의 컬럼을 그 모델 median으로 채우고, 이름 충돌은 `_r136`으로 (b6와 동일)
    src = joblib.load(GBM3_PATH)
    cols = list(dict.fromkeys(src["feature_cols"]))
    med = src["feature_medians"]
    ind = build_indicator_frame(kl)
    ind["timestamp"] = kl["timestamp"].to_numpy()
    base_names = set(ind.columns)
    x = live[["timestamp"] + [c for c in cols if c in live.columns]].copy()
    for cc in cols:
        if cc not in x.columns:
            x[cc] = med.get(cc, 0.0)
        x[cc] = pd.to_numeric(x[cc], errors="coerce").replace(
            [np.inf, -np.inf], np.nan).fillna(med.get(cc, 0.0))
    dup = [cc for cc in cols if cc in base_names]
    x = x.rename(columns={cc: cc + "_r136" for cc in dup})
    keep = ["timestamp"] + [(cc + "_r136" if cc in dup else cc) for cc in cols]
    F = ind.merge(x[keep], on="timestamp", how="left")
    for c in feat_cols:
        if c not in F.columns and c not in ("arm", "sig_id", "depth"):
            F[c] = np.nan
    return F


def compute_entry_signal(score_tabpfn: bool = False) -> dict[str, Any]:
    """후보 팔 + 채점. 절대 raise하지 않는다 -- warmed_up=False로 degrade한다."""
    empty: dict[str, Any] = {"warmed_up": False, "error": None, "last_closed_bar_utc": None,
                             "close": None, "candidates": [], "bars": []}
    try:
        A, CARD = _art()
        FE_COLS, POL = A["feature_cols"], CARD["policy"]
        CODE, HZ = CARD["signal_code_map"], CARD["signal_horizons"]
        depth = float(POL["depth_atr"])

        kl, btc_kl, eth_df, btc_df = _assemble()
        if len(kl) < 1000:
            return {**empty, "error": f"insufficient_bars:{len(kl)}"}
        # ⭐워밍업이 모자라면 조용히 상수 피쳐로 채점하지 말고 명시적으로 실패한다.
        if len(eth_df) < WARMUP_BARS:
            return {**empty, "error": f"insufficient_warmup:{len(eth_df)}<{WARMUP_BARS}"}

        # ⚠️`compute_signals`에는 BTC **klines 원본**을 준다 -- smt_divergence가 BTC OHLC를 쓴다.
        # (`btc_df`는 FeatureEngineer용 3열짜리라 여기엔 부족하다.)
        from live_evidence_signal_dashboard_20260823 import compute_signals, fetch_funding_safe
        sig = compute_signals(_tz_aware(kl), _tz_aware(btc_kl), fetch_funding_safe())
        F = _feature_frame(kl, eth_df, btc_df, FE_COLS)

        # ⭐이벤트 봉 = 마지막 **완결** 봉. 신호·피쳐·종가를 전부 이 한 봉에서 읽는다.
        ts = kl["timestamp"].iloc[-1]
        if F["timestamp"].iloc[-1] != ts:
            return {**empty, "error": "feature_bar_misaligned"}
        row = F.iloc[-1]
        srow = sig.iloc[-1]
        close = float(kl["close"].iloc[-1])
        atr_pct = float(row.get("atr_pct", np.nan))
        if not np.isfinite(atr_pct) or atr_pct <= 0:
            return {**empty, "error": "atr_pct_invalid", "last_closed_bar_utc": str(ts)}

        cands = []
        for name in CODE:
            for side in ("bottom", "top"):
                col = f"{side}_{name}"
                if col not in srow.index or not bool(srow[col]):
                    continue
                # ⭐v3: **arm1(신호방향)만** 제출한다. 역방향(arm0)은 L0 아티팩트의 최대
                # 수혜자였고, 정직한 L3에서는 3창 전부 손실이다(OOS −11.33bp).
                for arm in (1,):
                    sd = (1 if side == "bottom" else -1) * (1 if arm else -1)
                    lim = close * (1 - depth * atr_pct) if sd > 0 else close * (1 + depth * atr_pct)
                    cands.append({"signal": name, "side": side, "arm": arm, "sd": int(sd),
                                  "limit": float(lim), "atr_pct": atr_pct,
                                  "horizon": int(HZ[name]), "sig_id": int(CODE[name])})
        if not cands:
            return {**empty, "warmed_up": True, "last_closed_bar_utc": str(ts), "close": close,
                    "bars": _bars_out(kl)}

        X = pd.DataFrame([{**{c: row.get(c, np.nan) for c in FE_COLS},
                           "arm": c_["arm"], "sig_id": c_["sig_id"], "depth": depth}
                          for c_ in cands])[FE_COLS]
        X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        X = X.fillna(pd.Series(A["feature_medians"]))
        # HGB는 **대조**(v3에서 주 채점자는 TabPFN)
        pred = np.mean([m.predict(X) for m in A["hgb_models"]], axis=0)
        for c_, p in zip(cands, pred):
            c_["pred_hgb"] = float(p)
            c_["pass_hgb"] = bool(p > float(POL["hgb_tau"]))
        if score_tabpfn:
            _score_tabpfn(X, cands, float(POL["p_threshold"]))
        # ⚠️TabPFN 채점이 없으면 제출하지 않는다 -- v3의 주 채점자가 없는 상태이기 때문.
        for c_ in cands:
            c_["pass_tau"] = bool(c_.get("pred_tabpfn") is not None
                                  and c_["pred_tabpfn"] > float(POL["p_threshold"]))
        return {"warmed_up": True, "error": None, "last_closed_bar_utc": str(ts),
                "close": close, "candidates": cands, "bars": _bars_out(kl)}
    except Exception as e:                                          # noqa: BLE001
        return {**empty, "error": f"{type(e).__name__}: {e}"}


def _bars_out(kl: pd.DataFrame, n: int = 120) -> list[dict]:
    t = kl.tail(n)
    return [{"timestamp_utc": str(a), "open": float(b), "high": float(c),
             "low": float(d), "close": float(e)}
            for a, b, c, d, e in zip(t["timestamp"], t["open"], t["high"], t["low"], t["close"])]


def _score_tabpfn(X: pd.DataFrame, cands: list[dict], tau: float) -> None:
    """대조 채점. ⚠️한 호출에 약 69초(2026-09-03 실측) -- 후보 전부를 **한 번에** 배치한다.
    실패해도 HGB 판정에 영향을 주지 않는다(pred_tabpfn=None)."""
    try:
        import os
        env = ROOT / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("TABPFN_TOKEN="):
                    os.environ["TABPFN_TOKEN"] = line.split("=", 1)[1].strip().strip('"')
        from tabpfn import TabPFNClassifier
        C = _tabpfn_ctx()
        p = np.mean([m.predict_proba(X.to_numpy())[:, 1] for m in C], axis=0)
        for c_, v in zip(cands, p):
            c_["pred_tabpfn"] = float(v)
    except Exception as e:                                          # noqa: BLE001
        for c_ in cands:
            c_["pred_tabpfn"] = None
            c_["tabpfn_error"] = f"{type(e).__name__}: {e}"


_TP: list | None = None


def _tabpfn_ctx() -> list:
    """5멤버를 **상주**시킨다(적합 4.18초, 상주 1.065GB -- 매 호출 재적합하지 않는다)."""
    global _TP
    if _TP is None:
        raise RuntimeError("tabpfn_context_not_built")            # 러너가 build_tabpfn()으로 준비
    return _TP


def build_tabpfn(sub: int = 18000) -> int:
    """섀도우 러너 기동 시 1회. ⭐v3는 **컨텍스트를 아티팩트가 들고 있으므로** fills.csv를
    다시 읽지 않는다 -- 동결과 정확히 같은 바이트에서 멤버를 세운다(재적재 검증이 이를 확인)."""
    global _TP
    import os
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("TABPFN_TOKEN="):
                os.environ["TABPFN_TOKEN"] = line.split("=", 1)[1].strip().strip('"')
    from tabpfn import TabPFNClassifier
    A, _ = _art()
    loc = {int(v): i for i, v in enumerate(A["context_index"])}
    _TP = []
    for s in A["seeds"]:
        rs = np.random.default_rng(s).choice(A["context_index"], size=sub, replace=False)
        sel = np.array([loc[int(v)] for v in rs])
        m = TabPFNClassifier(device="cuda", random_state=s)
        m.fit(A["context_X"][sel], A["context_y"][sel])
        _TP.append(m)
    return len(_TP)


if __name__ == "__main__":
    out = compute_entry_signal()
    print(json.dumps({k: v for k, v in out.items() if k != "bars"},
                     ensure_ascii=False, indent=2, default=str))
