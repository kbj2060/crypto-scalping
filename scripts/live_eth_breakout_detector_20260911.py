#!/usr/bin/env python3
"""코인별 RVOL(오늘 거래량 «적음/보통/많음») 라이브 계산 (2026-09-11 추세 전환 경보·탐지 → 2026-10-18 RVOL 전용) — 읽기 전용.

🔴2026-10-18 «전환 예고»(경보기 HGB 5시드)·«전환 탐지»(거래대금·체결속도 z288 q90 AND) 제거 (사용자 «RVOL은 유지하고
  전환 탐지는 제거해줘. 워커 3개를 고쳐서 RVOL만 유지하게»). 재검증(scripts/research_eth_prewarn_lagging_verify_20261018.py, 2025-09~2026-09-28):
  예고가 켜지는 첫 봉 96% 가 거래대금 이미 상위 20% · 85% 는 직전 30분 안 탐지 이미 발동 = 급증 «뒤»에 켜졌고,
  앞 1h 큰 움직임 AUC .682 < 직전 1h 가격폭 단독 .770 · 조용했던 큰 움직임 시작의 8% 만 미리 켜졌다.
  되살리려면 git 이력(이 파일 2026-10-18 이전판 + data/live/eth_breakout_prewarn_artifact).
파일 이름·워커·supervisor·crontab 은 그대로 둔다(경로를 바꾸면 @reboot 줄 3개와 상태 파일 이름이 같이 바뀐다).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
KLINES = "https://fapi.binance.com/fapi/v1/klines"
SYMBOL = __import__("os").getenv("BO_SYMBOL", "ETHUSDT").upper()   # 코인별 워커(ETH·SOL·XRP)
FETCH_BARS = 4200            # RVOL 기준선 14일(4032봉) + 여유

# ── RVOL (2026-09-23, 사용자 「거래대금 대신 rvol 은 어때?」) ─────────────────────────
# 여기서 계산하는 이유: RVOL 은 «같은 시각의 평소»가 필요해 최소 7일치 5분봉이 있어야 하는데,
# 대시보드 evidence_signal_cache 는 1500봉(5.2일)뿐이다. 이 워커는 이미 FETCH_BARS 를 받는다.
# 🔴기준선 길이가 변별력을 지배한다(ETH 5m 4.7년, 앞 30분 레인지 상위25% AUC):
#   4일 .6678 · 5일 .6732 · 7일 .6807 · 14일 .7055 · 28일 .7251 (거래대금 z288 은 .6535)
#   => 5.2일짜리 서버 캐시로는 현행 대비 이득이 거의 없다. 14일이 FETCH_BARS(4200=14.6일) 상한.
RVOL_BASE_DAYS = 14
RVOL_BARS = 150              # 차트 캔들 100 + 여유. 띠(HIST_BARS=48)와는 다른 용도라 따로 둔다.
# 🔴선에 쓰는 창은 **1시간(12봉)**이다(2026-09-23 사용자 「5분봉이라 계산이 너무 튈 것 같다」).
#   실측으로 직관이 맞았고, 5분은 «튄다»만이 아니라 **변별력도 최하위**였다
#   (앞 30분 레인지 상위25% AUC / 봉폭 통제 후 / 봉간 |Δ|>0.5 비율):
#     5분 .6985 / **.4767**(동전 이하) / **41.9%**   15분 .7043 / .5226 / 18.9%
#     30분 .7055 / .5429 / 8.9%                      **1시간 .7053 / .5598 / 3.2%**
#     2시간 .7024 / .5698 / 0.8%                      4시간 .6961 / .5758 / 0.1%
#   창을 넓힐수록 «이 봉이 굵다»의 재진술에서 벗어나 봉폭 통제 후 점수가 오른다.
#   🔴2026-09-23 **5분판(`bar5`)은 제거했다.** 유일한 쓸모가 «흡수» 판독이었는데 재보니
#   흡수가 신호가 아니었다 -- 같은 거래량 수준에서 델타 低(흡수) vs 高(추종)의 앞 30분
#   되돌림이 **50.3~51.6% 대 52.4~55.3%** 로 둘 다 동전이고, 델타 임계를 q40->q10 으로
#   조여도 레인지 배수가 1.19->1.20 으로 붙박이다(효과는 전부 «거래량이 많다» 하나에서 나온다).
#   5분판 자체도 봉 폭 통제 후 AUC .4767 로 동전 이하였다. 사용자 결정으로 화면에서 뺀다.
#   세션과의 중복은 0.357 -> 0.571 로 오르지만 양방향 증분은 살아 있다
#   (세션->1시간 +0.0097 · 1시간->세션 +0.0251).
RVOL_LINE_BARS = 12
# 세션 «많다/적다» 경계 — 임의 상수가 아니라 **분위**다(ETH 5m 4.7년 세션 RVOL 분포).
#   q25 0.704 · q50 0.982 · q75 1.373. 이 경계로 가른 날의 앞 24h 레인지 중앙값은
#   적음 378bp / 보통 451 / 많음 516 으로 단조다 -- 표시에 뜻이 있다(신호는 아니다).
RVOL_SESSION_LO, RVOL_SESSION_HI = 0.70, 1.37
# 주말 보정(2026-10-05 사용자 «주말에 너무 작게», B안: 계산은 그대로·주말만 주말 분포로 읽는다).
#   14일 기준선의 10일이 평일이라 주말 세션 RVOL 은 구조적으로 눌린다 -- ETH 5m 2025-01~2026-09 주말 중앙 0.553
#   (토 0.52 · 일 0.58 · 시각별 0.46~0.65), 주말이 «적음» 65%. 주말 안 순위 정보는 그대로라(다음 24h 레인지 순위상관
#   주말 +0.39 vs 평일 +0.19) 값은 두고 «주말 평소 대비 배수 = 값/0.553»과 주말 q25/q75 경계로 라벨만 다시 붙인다.
#   ponytail: 상수 표다 -- 주말 수준이 2022~24 0.63 -> 2025~ 0.55 로 움직였으니 분기마다 같은 측정으로 갱신.
RVOL_WEEKEND_MED = 0.553
RVOL_WEEKEND_LO, RVOL_WEEKEND_HI = 0.40, 0.82
_CACHE: dict[str, Any] = {}


def session_view(sv: float, weekend: bool) -> dict[str, Any]:
    """세션 RVOL -> 라벨·경계·기준. 주말이면 주말 분포 경계와 «주말 평소 대비» 배수."""
    lo, hi = (RVOL_WEEKEND_LO, RVOL_WEEKEND_HI) if weekend else (RVOL_SESSION_LO, RVOL_SESSION_HI)
    ok = np.isfinite(sv)
    return {"session_label": None if not ok else "적음" if sv < lo else "많음" if sv > hi else "보통",
            "session_bounds": [lo, hi],
            "session_basis": "weekend" if weekend else "all",
            "session_weekend": round(float(sv) / RVOL_WEEKEND_MED, 3) if ok and weekend else None}


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
    # 2026-09-30 검증: 워커는 봉 마감 +20초에 도는데 바이낸스는 마감 ~5초 뒤 **새로 열린 봉**을 마지막 행으로 붙인다 --
    #   «지금» 판정·경보 확률·전진 로그가 20초치 봉으로 계산되고 있었다(실측 18:30:20 에 18:30 봉, 거래대금 z −0.69).
    #   종료 시각이 아직 안 지난 봉은 버린다 = 마지막 행이 **방금 마감된 봉**.
    d = d[pd.to_numeric(d["ct"]) < time.time() * 1000]
    d["timestamp"] = pd.to_datetime(d.ot, unit="ms")
    for cc in ("o", "h", "l", "c", "qv", "tbq"):   # tbq 누락 시 피쳐가 터진다
        d[cc] = d[cc].astype(float)
    d["n"] = d["n"].astype(int)
    return d.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def _rvol(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """RVOL = 지금 거래대금 / «같은 5분 슬롯의 후행 RVOL_BASE_DAYS 일 중앙값». 둘을 돌려준다.

    ⚠️`.shift()` 는 슬롯 그룹 **안에서** 미는 것이라 하루 전으로 민다 — 기준선이 자기 자신을
      포함하지 않는다(미래참조 방지). 중앙값을 쓰는 이유는 한 번의 스파이크가 «평소»를
      들어올리지 않게 하기 위해서다.
    둘째 값은 **세션 누적** RVOL(오늘 UTC 00:00 부터 지금까지 누적 / 평소 같은 시점 누적).
      봉 RVOL 과 달리 «현재 봉 폭»을 통제해도 변별력이 남는 유일한 거래량 지표다
      (봉폭 십분위 안 AUC .5815 vs 봉 RVOL .4935 — 누적이라 현재 봉과 구조적으로 독립).
    """
    ts = pd.to_datetime(d["timestamp"])
    slot = (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy()
    qv = pd.to_numeric(d["qv"], errors="coerce")
    med = lambda x: x.groupby(slot).transform(
        lambda z: z.rolling(RVOL_BASE_DAYS, min_periods=5).median().shift())
    roll = qv.rolling(RVOL_LINE_BARS).sum()           # 선 = 최근 1시간 누적
    line = roll / med(roll).replace(0, np.nan)
    cum = qv.groupby(ts.dt.floor("D").to_numpy()).cumsum()
    sess = cum / med(cum).replace(0, np.nan)
    return line.to_numpy(float), sess.to_numpy(float)


def compute_signals(d: pd.DataFrame) -> dict[str, Any]:
    """kline 프레임 하나로 RVOL 을 계산한다. 마지막 행이 방금 마감된 봉이다."""
    i = len(d) - 1
    # ── RVOL — 사분면 행의 «거래대금» 선을 대체할 값 (2026-09-23) · 표시값 전용
    rl, rs = _rvol(d)
    lo_r = max(i - RVOL_BARS + 1, 0)
    r3 = lambda v: None if not np.isfinite(v) else round(float(v), 3)
    sv = rs[i]
    return {
        "timestamp": str(d["timestamp"].iloc[i]), "close": float(d["c"].iloc[i]),
        "rvol": {
            "base_days": RVOL_BASE_DAYS,
            "line_minutes": RVOL_LINE_BARS * 5,
            "bar": [r3(rl[j]) for j in range(lo_r, i + 1)],          # 선 = 1시간 롤링
            "times": [str(pd.Timestamp(d["timestamp"].iloc[j]).tz_localize("UTC").isoformat())
                      for j in range(lo_r, i + 1)],
            "session": r3(sv),
            # 라벨은 워커가 붙인다 -- 경계가 바뀌면 화면 두 곳이 아니라 여기 한 곳만 고친다.
            # 세션은 UTC 하루라 주말 = 그 봉의 UTC 요일 토·일.
            **session_view(sv, pd.Timestamp(d["timestamp"].iloc[i]).dayofweek >= 5),
        },
    }


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


def _self_check() -> None:
    n = 19 * 288
    ts = pd.date_range("2026-01-05", periods=n, freq="5min")             # 월요일 시작 · 19일 → 마지막 봉 = 금요일(평일 경계)
    c = 2000 + np.cumsum(np.random.default_rng(0).normal(0, 1, n))
    d = pd.DataFrame({"timestamp": ts, "c": c, "qv": np.full(n, 1000.0)})
    r = compute_signals(d)
    assert set(r) == {"timestamp", "close", "rvol"}, sorted(r)           # 예고·탐지 키는 없다
    rv = r["rvol"]
    assert abs(rv["session"] - 1.0) < 1e-9 and rv["session_label"] == "보통", rv
    assert len(rv["bar"]) == len(rv["times"]) == RVOL_BARS and abs(rv["bar"][-1] - 1.0) < 1e-9, rv["bar"][-3:]
    d2 = d.copy(); d2.loc[d2.index[-36:], "qv"] = 6000.0                 # 마지막 3시간만 6배 → 하루 누적 1.63배 > 1.37 «많음»
    assert compute_signals(d2)["rvol"]["session_label"] == "많음"
    print("self-check OK  (RVOL 전용 · 예고·탐지 키 없음 · 평소 1.0 «보통» · 급증 «많음»)")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    else:
        import json
        print(json.dumps(get_signals(), ensure_ascii=False, indent=2))
