"""큰 스윙 선(2026-11-15) -- 1시간·30분봉의 «아직 안 깨진 큰 스윙 고점·저점»과 마지막 구조 돌파.

정의 = 의미 검증 연구(scripts/research_smc_ict_meaning_20261115.py pivots·structure, 브랜치 claude/fujimoto-126-trading-analysis-724dd5)의 인과판 그대로:
  스윙 = 앞뒤 50봉 최고(최저) · 봉 p 의 스윙은 p+50 봉 마감에 알려진다 · 종가가 가장 최근 안 깨진 스윙을 넘으면 돌파(직전 추세 반대면 CHoCH, 아니면 BOS).
검증(docs/experiments/choch_swing_2atr_confirm_20261115.md): 돌파 뒤 깬 쪽 2 ATR 이 반대 2 ATR 보다 먼저 -- 1시간 52.8% vs 무작위 48.8% · 30분 53.1% vs 49.2%
(4코인 2022~2026-11 · BOS 도 같은 크기 = «추세 전환» 정보 아님 · 5분은 +1.3pp 로 작음). 표시 전용 참고선.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

N = 50


def _pivots(x: np.ndarray, high: bool) -> np.ndarray:
    n = len(x); out = np.zeros(n, bool)
    for p in range(N, n - N):
        w = x[p - N:p + N + 1]
        out[p] = x[p] == (w.max() if high else w.min())
    return out


def swing_state(h: Sequence[float], l: Sequence[float], c: Sequence[float], close_ms: Sequence[int]) -> dict[str, Any]:
    """완결 봉(오래된 것부터) → 지금 안 깨진 큰 스윙 고점·저점(값·피벗 봉 끝 시각·알려진 시각)과 마지막 돌파."""
    h, l, c = (np.asarray(v, float) for v in (h, l, c))
    t = [int(v) for v in close_ms]
    ph, pl = _pivots(h, True), _pivots(l, False)
    top = btm = None; trend = 0; last = None
    for i in range(len(c)):
        p = i - N
        if p >= 0:
            if ph[p]:
                top = {"price": float(h[p]), "pivot_t": t[p], "known_t": t[i]}
            if pl[p]:
                btm = {"price": float(l[p]), "pivot_t": t[p], "known_t": t[i]}
        if top and c[i] > top["price"]:
            last = {"dir": 1, "kind": "CHoCH" if trend == -1 else "BOS", "level": top["price"], "t": t[i]}
            top, trend = None, 1
        if btm and c[i] < btm["price"]:
            last = {"dir": -1, "kind": "CHoCH" if trend == 1 else "BOS", "level": btm["price"], "t": t[i]}
            btm, trend = None, -1
    return {"hi": top, "lo": btm, "last_break": last, "trend": trend, "asof_t": t[-1] if t else None}


if __name__ == "__main__":
    # 자체 점검: 연구 structure 와 같은 손계산 봉(N 을 2 로 낮춰) -- 고점 10(봉 2) 확정 봉 4 → 봉 6 종가 11.5 > 10 = BOS(추세 0) · 뒤 하방 돌파 = CHoCH
    N = 2
    h = [5, 8, 10, 7, 6, 7, 12, 11, 9, 4, 3, 5.]; l = [x - 1 for x in h]; c = [x - 0.5 for x in h]; l[4] = 4.0
    s = swing_state(h, l, c, list(range(len(h))))
    assert s["last_break"]["dir"] == -1 and s["last_break"]["kind"] == "CHoCH", s
    s6 = swing_state(h[:7], l[:7], c[:7], list(range(7)))
    assert s6["last_break"] == {"dir": 1, "kind": "BOS", "level": 10.0, "t": 6} and s6["hi"] is None, s6
    s5 = swing_state(h[:6], l[:6], c[:6], list(range(6)))
    assert s5["hi"] == {"price": 10.0, "pivot_t": 2, "known_t": 4} and s5["last_break"] is None, s5   # 아직 안 깨짐 = 선
    print("swing_lines selfcheck OK", s["lo"], s["hi"])
