#!/usr/bin/env python3
"""ETH **극점 탐지기** 라이브 계산 (2026-09-09) — 대시보드 특화감지기 칩용, 읽기 전용.

"이 봉이 ±60분 국소 극점일 확률"을 등급(강/중/약)으로 준다. **매매 트리거가 아니다.**

아티팩트  data/live/eth_extreme_detector_artifact/{model.joblib,meta.json}
모집단    증거신호 8종 중 하나라도 발동한 봉(측면별) -- 발동이 없는 봉은 채점 대상이 아니다
라벨      i+1..i+12 에서 봉 i 의 저점/고점을 깨지 않는가 (피쳐는 봉 i 까지, 사건 라벨 경계 계약)
게이트    🔴강한 추세 구간(ret144 7일 롤링 분위 ≥0.80 / ≤0.20)에서는 콜을 억제한다.
         표본외 161일 실측: 강한상승 천장 콜 정밀도 51.6% 인데 적중 +10.8 / 빗나감 -60.7bp.
         게이트 없음 -3.36bp → 중립 구간만 +2.27bp.

⚠️어휘: 이 칩은 **사건의 측면**(바닥/천장)을 말하는 자리라 증거신호 어휘를 쓴다
   (`바닥 발동`/`천장 발동`/`미발동`). 포지션 방향을 말하는 다른 특화감지기의 롱/숏과 다르다
   -- 대시보드 신호 표시 규약 §1 의 구분을 그대로 따른 것이다.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
from typing import Any
import numpy as np, pandas as pd, requests

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path: sys.path.insert(0, str(_p))
from live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402
import build_eth_anchor_label_dataset_20260907 as B  # noqa: E402

ART = ROOT / "data/live/eth_extreme_detector_artifact"
KLINES = "https://fapi.binance.com/fapi/v1/klines"
SYMBOL, BTC_SYMBOL = "ETHUSDT", "BTCUSDT"
FETCH_BARS = 3000          # 2페이지 -- 추세분위(2016봉 롤링)와 atr 분위에 필요
HISTORY_BARS = 48          # 대시보드 띠 관례
W = 12                     # 라벨 창(±60분)
TREND_W, RANK_W = 144, 2016
GATE_HI, GATE_LO = 0.80, 0.20
_CACHE: dict[str, Any] = {}

BASE_COLS = ["p_fast", "p_slow", "delta_z", "vol_z", "lower_wick_ratio", "upper_wick_ratio",
             "ret3_z", "atr_pct", "dem", "kalman_dev_z"]
FEATS = (BASE_COLS + ["atr_pctile"]
         + [f"{k}{w}" for w in (12, 48, 144)
            for k in ("ret", "pos_in_range", "dist_lo", "dist_hi")]
         + ["btc_ret12", "btc_ret48", "eth_btc_div", "hour", "weekday"]
         + [f"f_{s}" for s in B.SIGNALS] + ["n_signals", "is_bottom"])


def _feature_frame(sig: pd.DataFrame, btc: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    ts = pd.to_datetime(sig["timestamp"]); n = len(sig)
    hi_ = sig.high.to_numpy(float); lo_ = sig.low.to_numpy(float)
    cl = sig.close.to_numpy(float); atr = sig.atr_pct.to_numpy(float)
    btc_cl = btc.set_index("timestamp")["close"].reindex(pd.DatetimeIndex(ts)).ffill().to_numpy(float)
    S = pd.DataFrame(index=range(n))
    for c in BASE_COLS:
        S[c] = sig[c].to_numpy(float)
    c_s = pd.Series(cl)
    S["atr_pctile"] = pd.Series(atr).rolling(RANK_W, min_periods=500).rank(pct=True).to_numpy()
    for w in (12, 48, 144):
        S[f"ret{w}"] = (c_s / c_s.shift(w) - 1).to_numpy() / np.maximum(atr, 1e-9)
        rmin = pd.Series(lo_).rolling(w).min().to_numpy(); rmax = pd.Series(hi_).rolling(w).max().to_numpy()
        S[f"pos_in_range{w}"] = (cl - rmin) / np.maximum(rmax - rmin, 1e-9)
        S[f"dist_lo{w}"] = (cl - rmin) / np.maximum(cl * atr, 1e-9)
        S[f"dist_hi{w}"] = (rmax - cl) / np.maximum(cl * atr, 1e-9)
    b_s = pd.Series(btc_cl)
    S["btc_ret12"] = (b_s / b_s.shift(12) - 1).to_numpy() / np.maximum(atr, 1e-9)
    S["btc_ret48"] = (b_s / b_s.shift(48) - 1).to_numpy() / np.maximum(atr, 1e-9)
    S["eth_btc_div"] = S["ret12"] - S["btc_ret12"]
    S["hour"] = ts.dt.hour.to_numpy(); S["weekday"] = ts.dt.weekday.to_numpy()
    for s in B.SIGNALS:
        S[f"f_{s}"] = 0.0
    tq = pd.Series(S["ret144"].to_numpy()).rolling(RANK_W, min_periods=500).rank(pct=True).to_numpy()
    return S, tq


def build_rows(sig: pd.DataFrame, btc: pd.DataFrame, with_label: bool = False) -> pd.DataFrame:
    """발동 봉만 모아 피쳐/라벨 프레임을 만든다. 라벨은 i+1 부터만 본다(경계 계약)."""
    S, tq = _feature_frame(sig, btc)
    ts = pd.to_datetime(sig["timestamp"]); n = len(sig)
    hi_ = sig.high.to_numpy(float); lo_ = sig.low.to_numpy(float)
    out = []
    for sd, long in (("bottom", True), ("top", False)):
        fires = {s: sig[f"{sd}_{s}"].fillna(False).to_numpy(bool) for s in B.SIGNALS}
        anyf = np.zeros(n, bool); cnt = np.zeros(n, int)
        for s in B.SIGNALS:
            anyf |= fires[s]; cnt += fires[s].astype(int)
        idx = np.flatnonzero(anyf); idx = idx[idx >= 900]
        if not len(idx):
            continue
        X = S.iloc[idx].copy()
        for s in B.SIGNALS:
            X[f"f_{s}"] = fires[s][idx].astype(float)
        X["n_signals"] = cnt[idx].astype(float); X["is_bottom"] = 1.0 if long else 0.0
        X["_i"] = idx; X["_ts"] = ts.iloc[idx].to_numpy(); X["_long"] = long; X["_tq"] = tq[idx]
        X["_names"] = [",".join(s for s in B.SIGNALS if fires[s][i]) for i in idx]
        if with_label:
            y = np.full(len(idx), -1, dtype=int)
            okm = idx + W < n
            if long:
                y[okm] = (np.array([lo_[i + 1:i + 1 + W].min() for i in idx[okm]]) >= lo_[idx[okm]]).astype(int)
            else:
                y[okm] = (np.array([hi_[i + 1:i + 1 + W].max() for i in idx[okm]]) <= hi_[idx[okm]]).astype(int)
            X["_y"] = y
        out.append(X)
    if not out:
        return pd.DataFrame(columns=FEATS + ["_i", "_ts", "_long", "_tq", "_names"])
    A = pd.concat(out, ignore_index=True).sort_values("_ts").reset_index(drop=True)
    A[FEATS] = A[FEATS].replace([np.inf, -np.inf], np.nan)
    return A.dropna(subset=FEATS + ["_tq"]).reset_index(drop=True)


def grade_of(p: np.ndarray, cuts: dict) -> np.ndarray:
    return np.where(p >= cuts["강"], "강", np.where(p >= cuts["중"], "중",
                    np.where(p >= cuts["약"], "약", "-")))


def gated_of(tq: np.ndarray, long: np.ndarray) -> np.ndarray:
    """강한 추세 구간이면 억제. 측면 무관 -- 강한상승/강한하락 둘 다에서 죽인다."""
    return (tq >= GATE_HI) | (tq <= GATE_LO)


def load_artifact() -> dict | None:
    if "art" in _CACHE:
        return _CACHE["art"]
    try:
        import joblib, json
        meta = json.loads((ART / "meta.json").read_text())
        _CACHE["art"] = {"models": joblib.load(ART / "model.joblib"), "meta": meta}
    except Exception:
        return None
    return _CACHE["art"]


def _fetch(symbol: str) -> pd.DataFrame | None:
    """3000봉을 두 번에 나눠 받는다(호출당 1500 상한). 형성 중인 봉은 버린다."""
    frames, end = [], None
    for _ in range(2):
        params = {"symbol": symbol, "interval": "5m", "limit": 1500}
        if end is not None:
            params["endTime"] = end
        try:
            r = requests.get(KLINES, params=params, timeout=15); r.raise_for_status()
            d = r.json()
        except Exception:
            return None
        if not d:
            break
        frames.append(d); end = int(d[0][0]) - 1
    if not frames:
        return None
    rows = [x for f in reversed(frames) for x in f]
    kl = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume",
                                     "close_time", "qv", "trades", "taker_buy_base", "tq_", "ig"])
    for c in ("open", "high", "low", "close", "volume", "taker_buy_base"):
        kl[c] = kl[c].astype(float)
    kl["timestamp"] = pd.to_datetime(kl["open_time"], unit="ms")
    kl = kl[kl["close_time"] < int(time.time() * 1000)]
    return kl.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def _empty(err: str) -> dict:
    return {"available": False, "error": err, "tone": "neutral", "subText": "데이터 없음",
            "grade": None, "proba": None, "history": [], "times": []}


def compute_eth_extreme_detector() -> dict:
    """대시보드 페이로드. 절대 예외를 올리지 않는다."""
    art = load_artifact()
    if art is None:
        return _empty("artifact_missing")
    try:
        kl = _fetch(SYMBOL)
        if kl is None or len(kl) < 1200:
            return _empty("price_fetch_failed")
        btc = _fetch(BTC_SYMBOL)
        if btc is None:
            return _empty("btc_fetch_failed")
        sig = compute_signals(kl, btc_df=btc, funding_df=None)
        A = build_rows(sig, btc)
        if A.empty:
            return _empty("no_fires")
        meta = art["meta"]
        P = np.zeros(len(A))
        for m in art["models"]:
            P += m.predict_proba(A[FEATS])[:, 1] / len(art["models"])
        A = A.assign(p=P)
        A["grade"] = grade_of(A.p.to_numpy(), meta["cuts"])
        A["gated"] = gated_of(A._tq.to_numpy(), A._long.to_numpy())
        live = A[(A.grade != "-") & (~A.gated)]
        ts_all = pd.to_datetime(sig["timestamp"])
        last_ts = ts_all.iloc[-1]
        # 봉별 톤: 같은 봉에 양측이 있으면 확률이 높은 쪽
        by_ts: dict[Any, dict] = {}
        # ⚠️itertuples 는 밑줄로 시작하는 컬럼을 _1,_2 로 바꾼다 -- 미리 개명하고 돈다
        for r in live.rename(columns={"_ts": "ts_", "_long": "long_",
                                      "_names": "names_"}).itertuples():
            k = r.ts_
            if k not in by_ts or r.p > by_ts[k]["p"]:
                by_ts[k] = {"p": float(r.p), "long": bool(r.long_), "grade": r.grade,
                            "names": r.names_}
        history, times = [], []
        for t in ts_all.iloc[-HISTORY_BARS:]:
            times.append(pd.Timestamp(t).isoformat())
            b = by_ts.get(t)
            history.append(("good" if b["long"] else "bad") if b else "neutral")
        cur = by_ts.get(last_ts)
        gated_now = bool(A[A._ts == last_ts].gated.any()) if (A._ts == last_ts).any() else False
        if cur:
            tone = "good" if cur["long"] else "bad"
            sub = f"{'바닥' if cur['long'] else '천장'} 발동"
        else:
            tone, sub = "neutral", "미발동"
        return {
            "available": True, "rule_id": meta.get("rule_id"), "tone": tone, "subText": sub,
            "grade": cur["grade"] if cur else None,
            "proba": round(cur["p"], 4) if cur else None,
            "signals": cur["names"] if cur else None,
            "gated_now": gated_now, "trend_q": round(float(A[A._ts == last_ts]._tq.iloc[0]), 3)
                          if (A._ts == last_ts).any() else None,
            "latest_ts_utc": pd.Timestamp(last_ts).isoformat(),
            "history": history, "times": times,
            "precision": meta.get("precision"), "per_day": meta.get("per_day"),
            "base_rate": meta.get("base_rate"), "auc_oos": meta.get("auc_oos"),
            "suppressed_per_day": meta.get("gated_suppressed_per_day"),
            "oos_span": meta.get("oos_span"),
        }
    except Exception as e:  # noqa: BLE001 -- 대시보드 사이클을 절대 깨지 않는다
        return _empty(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    import json
    print(json.dumps(compute_eth_extreme_detector(), ensure_ascii=False, indent=1)[:2000])
