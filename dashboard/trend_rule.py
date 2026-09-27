"""일 단위 추세 규칙(5기간 묶음) — 30분 카드의 «위:아래 방향» 자리를 대신한다 (2026-09-28 사용자 지시).

신호 = mean(sign(C[t] - C[t-L]) for L in 7·14·28·56·90) ∈ {±0.2, ±0.6, ±1}. C = 완결된 UTC 일봉 종가.
근거: ETH 현물 2017-11~2026-08 샤프 1.33(396 조합 중 1등, 워크포워드 2020~ 내내 선택) · 기본 7-14-21-28 은 1.04.
🔴이 설정은 표본 안에서 고른 것이다(PBO 0.63) — 기본값보다 낫다는 통계적 근거는 없고 «해마다 고른» 대안이다.
확률은 싣지 않는다: 신호 단계별 다음 7일 상승 확률이 23→37% 로만 움직여 과잉 정밀이 된다.

뒤집힘 가격 = 다음 결정(다음 UTC 00시)에서 그 기간 표를 가르는 기준 = C[n-L] (지금 표는 C[n-1] vs C[n-1-L]).
권장 크기 = 신호 × 연 50% ÷ 최근 20일 변동성(연율), ±2배 상한 — 백테스트와 같은 식.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

LOOKBACKS = (7, 14, 28, 56, 90)
VOL_WINDOW = 20
TARGET_VOL = 0.50
MAX_LEVERAGE = 2.0


def trend_payload(closes: Sequence[float], close_times_ms: Sequence[int]) -> dict[str, Any]:
    """완결된 일봉 종가(오래된 것부터)와 그 종가 시각(ms, 봉 끝) -> 카드 payload. 모자라면 ok=False + 이유."""
    c = [float(x) for x in closes]
    n = len(c)
    need = max(LOOKBACKS) + 2
    if n < need or any(not (x > 0) for x in c):
        return {"ok": False, "reason": f"일봉이 모자랍니다 ({n}/{need})"}
    votes = []
    for L in LOOKBACKS:
        ref_now = c[n - 1 - L]
        votes.append({"L": L, "up": c[n - 1] > ref_now, "ref_now": ref_now, "flip_price": c[n - L]})
    signal = sum(1 if v["up"] else -1 for v in votes) / len(votes)
    rets = [math.log(c[i] / c[i - 1]) for i in range(n - VOL_WINDOW, n)]
    mu = sum(rets) / len(rets)
    vol_ann = math.sqrt(sum((r - mu) ** 2 for r in rets) / (len(rets) - 1)) * math.sqrt(365)
    size = max(-MAX_LEVERAGE, min(MAX_LEVERAGE, signal * TARGET_VOL / vol_ann)) if vol_ann > 0 else 0.0

    def sig_at(i: int) -> int:
        return sum(1 if c[i] > c[i - L] else -1 for L in LOOKBACKS)

    # 추세 나이 = 신호 부호가 지금과 같은 채로 이어진 완결 일수
    sgn = 1 if signal > 0 else -1
    age = 0
    for i in range(n - 1, max(LOOKBACKS) - 1, -1):
        if (1 if sig_at(i) > 0 else -1) != sgn:
            break
        age += 1
    return {"ok": True, "signal": round(signal, 1), "ups": sum(v["up"] for v in votes), "n_votes": len(votes),
            "votes": votes, "close": c[n - 1], "asof_ms": int(close_times_ms[-1]), "vol_ann": round(vol_ann, 4),
            "size": round(size, 3), "age_days": age, "lookbacks": list(LOOKBACKS)}


if __name__ == "__main__":
    # 자체 점검: 꾸준한 상승 → +1 · 뒤집힘 가격 = C[n-L] · 크기 부호 · 7일 표만 뒤집힘 · 부족하면 ok=False
    up = [100 * 1.01 ** i for i in range(120)]
    p = trend_payload(up, list(range(120)))
    assert p["ok"] and p["signal"] == 1.0 and p["ups"] == 5 and p["size"] > 0, p
    assert all(v["flip_price"] == up[120 - v["L"]] for v in p["votes"]), p["votes"]
    assert p["age_days"] >= 29, p["age_days"]
    q = trend_payload(list(reversed(up)), list(range(120)))
    assert q["signal"] == -1.0 and -MAX_LEVERAGE <= q["size"] < 0, q
    mix = up[:115] + [up[114] * 0.99 ** k for k in range(1, 6)]            # 마지막 5일 하락: 7일 표만 뒤집힘
    m = trend_payload(mix, list(range(120)))
    assert [v["up"] for v in m["votes"]] == [False, True, True, True, True] and m["signal"] == 0.6, m
    assert not trend_payload(up[:50], list(range(50)))["ok"]
    print("trend_rule selfcheck OK", {k: m[k] for k in ("signal", "ups", "size", "age_days")})
