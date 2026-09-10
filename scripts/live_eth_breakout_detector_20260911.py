#!/usr/bin/env python3
"""ETH **횡보→추세 전환** 경보·탐지 라이브 계산 (2026-09-11) — 읽기 전용.

사용자 실계좌 손실 패턴에서 출발: 압축 구간 상단에서 숏 페이드 → 상방 돌파 → -535.96 USDT
(전체 손실의 95.7%). **방향은 예측하지 않는다** — 돌파가 온다/왔다만 말한다.
방향 축은 이 저장소에서 닫혔다(MFE-MAE rho 0.000) — 여기서 다시 열지 않는다.

경보(예측)  피쳐 3종을 **각자 독립 신호등**으로 켠다(합치지 않는다 — 지평이 달라 뜻이 다르다).
            압축 구간에서만 감시. 새 진입 억제용.
탐지(즉시)  피쳐 2종(거래대금·체결속도) z288 q90 **AND**. 보유 중 반대 방향 돌파면 즉시 청산용.

성적(ETH 2026-01~09, 압축 봉 29,778 · 전환 431건 · 앞만 보는 타깃 · 순환이동 귀무):
  경보  3봉지속 z2016 q99 → 앞 2시간 상위5% (전역분위 연구 7.71x / **인과 임계 5.75x**)
        전반 8.56x / 후반 8.58x · 63셀 전부 2x 이상 · 셀 순위 상관 +0.708
  경보  인과 임계로 재현: 3봉지속 5.75x(하루 1.6회) · 체결속도 3.65x · 거래대금 3.06x
        경보 ON 시 앞 2시간 상위5% 확률 28.75%(기저 5%) · 실제 선행 중앙 15분
  탐지  2종 AND q90 → 포착률 98.4% · 지연 1봉(5분) · **진행률 6.08%** · 헛발동 39.8회/일
        (진행률 = 감지 시점 이동폭 / 전체 이동폭. 사용자 실제 사고는 15% 였다)
        volexp 포함 3종이면 진행률 11.51% · 헛발동 28.2회 — 조용해지는 대가가 지연이다

⚠️리드타임을 주장하지 않는다 — 앞선 측정의 "-25분"은 탐색 창 폭의 산물이었다.
⚠️입력은 공개 kline 의 n(체결 건수)·quote_volume 뿐이다. 호가는 기여 없음(166건 검정).
⚠️단일 자산·단일 연도. 전환 정의(압축<0.7 → 확장>=1.8)는 이 저장소 것이다.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

KLINES = "https://fapi.binance.com/fapi/v1/klines"
SYMBOL = "ETHUSDT"
FETCH_BARS = 4200            # z2016(7일) + 압축분위 창 + 여유
COMPRESS = 0.70              # volexp < 0.70 = 압축(횡보) — 이 구간에서만 감시
QWIN = 2016                  # 임계 분위를 재는 후행 창(인과)
# 경보: (피쳐, z창, 평활봉, 분위, 최적지평 표기, 실측 lift)
# lift 는 **인과 임계 백테스트** 값이다(전역 분위 연구값 7.71/3.85/3.70 보다 낮다).
ALERT = [("체결속도 3봉지속", "n", 2016, 3, 0.99, "앞 2시간", 5.75),
         ("체결속도", "n", 864, 1, 0.99, "앞 5~15분", 3.65),
         ("거래대금", "qv", 2016, 1, 0.99, "앞 30분", 3.06)]
# 탐지: 2종 AND · z288 · q90
# volexp 는 뺐다 — 전환의 **정의**(volexp>=1.8 교차)에 쓰인 양이라 기여가 정의상 보장된
# 것이지 정보가 아니고, 넣으면 진행률이 6.08% → 11.51% 로 느려진다(12봉 롤링이라 구조적 지연).
DETECT = [("거래대금", "qv", 288, 0.90), ("체결속도", "n", 288, 0.90)]
HIST_BARS = 48               # 화면 띠 길이(4시간) — 다른 특화감지기와 같은 눈금
_CACHE: dict[str, Any] = {}


def _fetch(limit: int = FETCH_BARS) -> pd.DataFrame:
    out, end = [], None
    while len(out) < limit:
        p = {"symbol": SYMBOL, "interval": "5m", "limit": 1500}
        if end is not None:
            p["endTime"] = end
        r = requests.get(KLINES, params=p, timeout=15)
        r.raise_for_status()
        k = r.json()
        if not k:
            break
        out = k + out
        end = k[0][0] - 1
        if len(k) < 1500:
            break
    d = pd.DataFrame(out, columns=["ot", "o", "h", "l", "c", "v", "ct", "qv", "n", "tbv", "tbq", "x"])
    d["timestamp"] = pd.to_datetime(d.ot, unit="ms")
    for cc in ("o", "h", "l", "c", "qv"):
        d[cc] = d[cc].astype(float)
    d["n"] = d["n"].astype(int)
    return d.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def compute_signals(d: pd.DataFrame) -> dict[str, Any]:
    """kline 프레임 하나로 경보·탐지를 계산한다. 마지막 행이 현재 봉이다."""
    c = d["c"].to_numpy(float)
    lr = np.diff(np.log(c), prepend=np.log(c[0]))
    volexp = (pd.Series(lr).rolling(12).std() / pd.Series(lr).rolling(288).std()).to_numpy()
    comp = volexp < COMPRESS
    # 탐지 감시창: 돌파 봉은 이미 volexp 가 올라 압축이 아니다 — 직전 1시간 내 압축이면 감시한다
    watch = pd.Series(comp).rolling(12, min_periods=1).max().to_numpy() == 1

    def _z(col: str, w: int) -> np.ndarray:
        s = pd.to_numeric(d[col], errors="coerce")
        return ((s - s.rolling(w).mean()) / s.rolling(w).std()).to_numpy()

    def _thr_at(x: np.ndarray, q: float, j: int) -> float:
        """봉 j 시점의 임계 — 압축 봉만 모아 후행 창에서 분위를 낸다. 미래를 안 본다.

        띠(history)도 이 함수로 그린다. 지금 임계로 과거를 칠하면 임계가 움직일 때마다
        과거 칸 색이 바뀐다 — 사용자가 본 적 없는 그림이 된다.
        """
        m = comp[:j + 1] & np.isfinite(x[:j + 1])
        v = x[:j + 1][m][-QWIN:]
        return float(np.nanquantile(v, q)) if len(v) >= 200 else np.inf

    def _thr(x: np.ndarray, q: float) -> float:
        return _thr_at(x, q, len(d) - 1)

    i = len(d) - 1
    out: dict[str, Any] = {"timestamp": str(d["timestamp"].iloc[i]), "close": float(c[i]),
                           "volexp": float(volexp[i]) if np.isfinite(volexp[i]) else None,
                           "compressed": bool(comp[i]), "watch": bool(watch[i])}
    alerts = []
    for label, col, w, smooth, q, horizon, lift in ALERT:
        x = _z(col, w)
        if smooth > 1:
            x = pd.Series(x).rolling(smooth).min().to_numpy()
        thr = _thr(x, q)
        on = bool(comp[i] and np.isfinite(x[i]) and x[i] >= thr)
        alerts.append({"name": label, "horizon": horizon, "lift": lift, "on": on,
                       "z": None if not np.isfinite(x[i]) else round(float(x[i]), 3),
                       "threshold": None if not np.isfinite(thr) else round(thr, 3)})
    dets = []
    for label, col, w, q in DETECT:
        x = _z(col, w)
        thr = _thr(x, q)
        on = bool(watch[i] and np.isfinite(x[i]) and x[i] >= thr)
        dets.append({"name": label, "on": on,
                     "z": None if not np.isfinite(x[i]) else round(float(x[i]), 3),
                     "threshold": None if not np.isfinite(thr) else round(thr, 3)})
    # 경보는 신호등 3개다 — 합치지 않는다. 각자 지평이 달라 뜻이 다르다.
    out["alert"] = {"lights": alerts, "lit": sum(a["on"] for a in alerts)}
    out["detect"] = {"signals": dets, "count": sum(x["on"] for x in dets),
                     "on": all(x["on"] for x in dets)}      # AND
    out["state"] = ("돌파 진행" if out["detect"]["on"] else
                    (f"경보 {out['alert']['lit']}등" if out["alert"]["lit"] else
                     ("횡보 감시" if out["compressed"] else "감시 밖")))

    # ── 화면 계약 (규약 §1~3). 톤은 4색 안에서만 쓴다.
    #    탐지=bad · 경보=warn · 그 외=neutral. 이 행에는 방향 축이 자체가 없어서
    #    (방향은 예측하지 않는다) 빨강이 «숏»으로 읽힐 여지가 없다 — 두 단계를 색으로
    #    가르지 않으면 경보→탐지 경계가 띠에서 사라진다(규약 §5-5: 이유 없는 경계 금지).
    alert_v, detect_v = [], []
    for label, col, w, smooth, q, horizon, lift in ALERT:
        x = _z(col, w)
        if smooth > 1:
            x = pd.Series(x).rolling(smooth).min().to_numpy()
        alert_v.append(x)
    for label, col, w, q in DETECT:
        detect_v.append(_z(col, w))
    hist, times = [], []
    for j in range(max(i - HIST_BARS + 1, 0), i + 1):
        lit_j = sum(bool(comp[j] and np.isfinite(x[j]) and x[j] >= _thr_at(x, a[4], j))
                    for x, a in zip(alert_v, ALERT))
        det_j = all(bool(watch[j] and np.isfinite(x[j]) and x[j] >= _thr_at(x, dd[3], j))
                    for x, dd in zip(detect_v, DETECT))
        hist.append("bad" if det_j else ("warn" if lit_j else "neutral"))
        times.append(str(pd.Timestamp(d["timestamp"].iloc[j]).tz_localize("UTC").isoformat()))
    out["tone"] = hist[-1]
    out["subText"] = ("돌파 발동" if out["detect"]["on"] else
                      ("돌파 경보" if out["alert"]["lit"] else "미발동"))
    out["history"], out["times"] = hist, times
    return out


def get_signals(ttl: int = 60) -> dict[str, Any]:
    now = time.time()
    if _CACHE.get("t", 0) + ttl > now and "v" in _CACHE:
        return _CACHE["v"]
    try:
        v = compute_signals(_fetch())
        v["ok"] = True
    except Exception as e:                                   # 신뢰경계: 외부 API
        v = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    _CACHE.update(t=now, v=v)
    return v


def _mk(n: int = 3400, quiet: int = 120, seed: int = 0) -> pd.DataFrame:
    """뒤쪽 quiet 봉만 조용하게 만들어 volexp<0.7(압축)을 성립시킨다."""
    rng = np.random.default_rng(seed)
    sd = np.r_[np.full(n - quiet, 1.0), np.full(quiet, 0.15)]
    c = 2000 + np.cumsum(rng.normal(0, 1, n) * sd)
    cnt = rng.normal(5000, 250, n)
    return pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=n, freq="5min"),
                         "o": c, "h": c + 0.5, "l": c - 0.5, "c": c,
                         "qv": cnt * 1000.0, "n": cnt.astype(int)})


def _self_check() -> None:
    d = _mk()
    base = compute_signals(d)
    assert base["compressed"] is True, base                     # 뒤쪽이 조용하므로 압축
    assert base["alert"]["lit"] == 0, base["alert"]             # 평탄하면 아무 등도 안 켜진다
    assert base["detect"]["on"] is False, base["detect"]

    # 체결 건수만 급등 — 거래대금은 그대로다. 탐지는 2종 **AND** 라 켜지면 안 된다.
    # (첫 판에서는 n·qv 를 같이 올려 «경보» 시나리오가 실은 탐지까지 켜고 있었다 —
    #  그래서 AND 게이트를 한 번도 시험하지 못했다. 한 쪽만 올려야 그게 검사가 된다.)
    a = d.copy()
    a.loc[a.index[-3:], "n"] = 60000
    ra = compute_signals(a)
    assert ra["compressed"] is True, ra                         # 가격이 안 움직였으니 여전히 압축
    assert ra["alert"]["lit"] >= 2, ra["alert"]                 # 체결속도 계열 등이 켜진다
    assert ra["detect"]["count"] == 1, ra["detect"]             # 한 쪽만 — AND 미성립
    assert ra["detect"]["on"] is False, ra["detect"]

    b = a.copy()                                                # 거래대금까지 + 가격 돌파
    b.loc[b.index[-3:], "qv"] = 6.0e7
    b.loc[b.index[-1], ["o", "h", "l", "c"]] = float(b["c"].iloc[-1]) + 14.0
    rb = compute_signals(b)
    assert rb["compressed"] is False, rb                        # 돌파 봉은 압축이 아니고
    assert rb["watch"] is True, rb                              # 감시창 안이며
    assert rb["detect"]["on"] is True, rb["detect"]             # 2종 AND 가 켜진다
    assert rb["state"] == "돌파 진행", rb["state"]

    # 화면 계약: 톤 4색 안 · 띠 길이 · 마지막 칸이 현재 톤 · 세 단계가 색으로 갈린다
    for r, want_tone, want_sub in ((base, "neutral", "미발동"),
                                   (ra, "warn", "돌파 경보"), (rb, "bad", "돌파 발동")):
        assert r["tone"] == want_tone, (r["state"], r["tone"])
        assert r["subText"] == want_sub, r["subText"]
        assert len(r["history"]) == len(r["times"]) == HIST_BARS, len(r["history"])
        assert r["history"][-1] == r["tone"], (r["history"][-1], r["tone"])
        assert set(r["history"]) <= {"good", "bad", "warn", "neutral"}, set(r["history"])
    print("self-check OK  (평탄→무발동 · 체결만급등→경보(AND 미성립) · 거래대금+돌파→탐지 · 화면 계약)")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    else:
        import json
        print(json.dumps(get_signals(), ensure_ascii=False, indent=2))
