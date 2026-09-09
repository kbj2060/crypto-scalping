#!/usr/bin/env python3
"""ETH **24시간 변동성 전망** 라이브 계산 (2026-09-10) — 대시보드 「위험도」 지표용, 읽기 전용.

"앞으로 24시간 실현변동성이 지금의 1.3배 이상으로 확장되는가"를 안정/주의/위험 3등급으로 준다.
**방향도 수익도 예측하지 않는다.** 사람이 크기·손절폭·관망을 정할 때 쓰는 맥락이다.

정보원  Binance 5분봉(과거 변동성 HAR) + **Deribit DVOL**(30일 내재변동성, 새 정보원)
        ⭐화면의 다른 변동성 지표는 전부 과거만 본다 -- 미래지향 입력은 이게 처음이다.
아티팩트 data/live/eth_vol_forecast_artifact/{model.joblib,meta.json} (TRAIN ≤2026-03-31 적합)
성적    AUC TRAIN .773 / OOS .811 / 🔒홀드아웃 .836 · 위험 등급 홀드아웃 정밀도 .787(기저 .191)

⚠️어휘는 「위험도」 그룹 규약(안정/주의/위험)을 따른다 -- 방향 신호가 아니므로 롱/숏을 쓰지 않는다.
⚠️색: 위험/주의 = warn(주황) · 안정 = neutral(회색). 새 색을 만들지 않는다(표시 규약 §2).
"""
from __future__ import annotations
import sys, time
from pathlib import Path
from typing import Any
import numpy as np, pandas as pd, requests

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data/live/eth_vol_forecast_artifact"
KLINES = "https://fapi.binance.com/fapi/v1/klines"
DVOL_URL = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
SYMBOL = "ETHUSDT"
FETCH_BARS = 3000            # 168시간 HAR 창(2016봉) + 띠 48시간 + 여유 (2페이지)
HISTORY_BARS = 48            # 대시보드 띠 관례(시간봉 48개 = 2일)
FEATS = ["l_rv1", "l_rv24", "l_rv168", "l_dvol", "vrp"]
_CACHE: dict[str, Any] = {}


def load_artifact() -> dict | None:
    if "art" in _CACHE:
        return _CACHE["art"]
    try:
        import joblib, json
        _CACHE["art"] = {"m": joblib.load(ART / "model.joblib"),
                         "meta": json.loads((ART / "meta.json").read_text())}
    except Exception:
        return None
    return _CACHE["art"]


def _fetch_klines() -> pd.DataFrame | None:
    frames, end = [], None
    for _ in range(2):
        params = {"symbol": SYMBOL, "interval": "5m", "limit": 1500}
        if end is not None:
            params["endTime"] = end
        try:
            r = requests.get(KLINES, params=params, timeout=15); r.raise_for_status()
            d = r.json()
        except Exception:
            return None
        if not d:
            break
        f = pd.DataFrame(d, columns=["open_time", "open", "high", "low", "close", "volume",
                                     "close_time", "qv", "trades", "tb", "tq", "ig"])
        frames.append(f); end = int(f["open_time"].iloc[0]) - 1
        if len(frames) * 1500 >= FETCH_BARS:
            break
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True)
    d["timestamp"] = pd.to_datetime(d["open_time"], unit="ms")
    for c in ("open", "close"):
        d[c] = d[c].astype(float)
    d = d.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return d.iloc[:-1]                    # 형성 중인 봉은 버린다


def _fetch_dvol(hours: int = 240) -> pd.DataFrame | None:
    end = int(time.time() * 1000); start = end - hours * 3600 * 1000
    try:
        r = requests.get(DVOL_URL, params={"currency": "ETH", "start_timestamp": start,
                                           "end_timestamp": end, "resolution": "3600"}, timeout=20)
        r.raise_for_status()
        d = pd.DataFrame(r.json()["result"]["data"], columns=["ts", "o", "h", "l", "close"])
    except Exception:
        return None
    if d.empty:
        return None
    d["timestamp"] = pd.to_datetime(d["ts"], unit="ms")
    return d[["timestamp", "close"]].rename(columns={"close": "dvol"}).sort_values("timestamp")


def build_features(kl: pd.DataFrame, dv: pd.DataFrame) -> pd.DataFrame | None:
    """5분봉 + DVOL -> 시간봉 피쳐. **학습 빌더와 같은 식이어야 한다.**

    ⚠️파리티 하네스는 이 함수를 그대로 부른다 -- 하네스가 식을 다시 쓰면 하네스만 낡아서
      불일치를 못 잡는다(2026-09-10 실제로 그렇게 한 번 헛돌았다).
    🔴min_periods 는 **창 전체**다. 절반만 요구하면 라이브에서 창이 덜 찼을 때 조용히 부분
      창으로 계산돼 학습과 갈라진다(rv168 이 134시간 중 84시간, 확률 7.6e-3 어긋났다).
    """
    p = kl.set_index("timestamp")["close"]
    r5 = np.log(p).diff()
    ann = np.sqrt(288 * 365) * 100

    def rvw(h):
        return (r5.rolling(h * 12, min_periods=h * 12).std() * ann).resample("1h").last()
    hh = pd.DataFrame({"rv1": rvw(1), "rv24": rvw(24), "rv168": rvw(168)}).dropna()
    d = hh.join(dv.set_index("timestamp")["dvol"], how="inner").dropna()
    if d.empty:
        return d
    d["vrp"] = d["dvol"] - d["rv24"]
    for c in ("rv1", "rv24", "rv168", "dvol"):
        d[f"l_{c}"] = np.log(d[c].clip(lower=1e-6))
    return d


def _empty(err: str) -> dict:
    return {"available": False, "error": err, "tone": "neutral", "subText": "데이터 없음",
            "grade": None, "proba": None, "history": [], "times": []}


def compute_eth_vol_forecast() -> dict:
    art = load_artifact()
    if art is None:
        return _empty("artifact_missing")
    try:
        kl = _fetch_klines()
        if kl is None or len(kl) < 2100:
            return _empty("price_fetch_failed")
        dv = _fetch_dvol()
        if dv is None or dv.empty:
            return _empty("dvol_fetch_failed")
        d = build_features(kl, dv)
        if d is None or d.empty:
            return _empty("join_empty")
        M = art["m"]; meta = art["meta"]
        X = d[FEATS].to_numpy(float)
        pr = M["clf"].predict_proba((X - M["mu"]) / M["sd"])[:, 1]
        d["p"] = pr
        cuts = meta["cuts"]
        gr = np.where(pr >= cuts["위험"], "위험", np.where(pr >= cuts["주의"], "주의", "안정"))
        d["grade"] = gr
        last = d.iloc[-1]
        tone = "warn" if last["grade"] in ("위험", "주의") else "neutral"
        tail = d.iloc[-HISTORY_BARS:]
        return {
            "available": True, "rule_id": meta.get("rule_id"),
            "tone": tone, "subText": str(last["grade"]),
            "grade": str(last["grade"]), "proba": round(float(last["p"]), 4),
            "dvol": round(float(last["dvol"]), 2), "rv24": round(float(last["rv24"]), 2),
            "vrp": round(float(last["vrp"]), 2),
            "rv_fwd_pred": round(float(np.exp(M["reg"].predict(X[-1:])[0])), 2),
            "latest_ts_utc": pd.Timestamp(d.index[-1]).tz_localize("UTC").isoformat(),
            "history": [("warn" if g in ("위험", "주의") else "neutral") for g in tail["grade"]],
            "times": [pd.Timestamp(t).tz_localize("UTC").isoformat() for t in tail.index],
            "auc": meta.get("auc"), "precision_holdout": meta.get("precision_holdout"),
            "base_rate_holdout": meta.get("base_rate_holdout"),
            "horizon_hours": meta.get("horizon_hours"), "expand_k": meta.get("expand_k"),
            "holdout_span": meta.get("holdout_span"),
        }
    except Exception as e:  # noqa: BLE001 -- 대시보드 사이클을 절대 깨지 않는다
        return _empty(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    import json
    print(json.dumps(compute_eth_vol_forecast(), ensure_ascii=False, indent=1)[:1400])
