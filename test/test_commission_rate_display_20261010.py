"""2026-10-10 주문 심볼 실수수료율 → 화면 «예상 비용». 가짜 commissionRate 응답만 쓴다(🔴실 API 호출 0 -- 로컬=서버 공인 IP).

· 심볼별 요율(USDC 0/0.0004 · USDT 0.0002/0.0005)이 계획 비용에 맞게 들어가는가
· 1시간 안에는 재호출하지 않는가 · 실패면 지난 성공값(stale) · 이력도 없으면 표준 2/5bp(fallback, «추정»)
· 판정 상수(지평 권고·처방 비용)는 요율과 무관한가
"""
import asyncio
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import scripts.live_manual_peg_execute_20260912 as ex  # noqa: E402
from scripts.live_eth_trade_plan_20260913 import execution_plan, plan_now  # noqa: E402

RATES = {"ETHUSDC": {"makerCommissionRate": "0.000000", "takerCommissionRate": "0.000400"},
         "ETHUSDT": {"makerCommissionRate": "0.000200", "takerCommissionRate": "0.000500"}}
calls, fail = [], {"on": False}


async def fake_signed(session, method, path, params, key, secret, offset):
    assert (method, path) == ("GET", "/fapi/v1/commissionRate"), (method, path)   # 다른 경로(주문)는 절대 안 탄다
    calls.append(params["symbol"])
    return {"__error__": "418 banned"} if fail["on"] else RATES[params["symbol"]]


async def fake_offset(session):
    return 0


def test_commission_rate_cache_and_plan_cost():
    orig = ex.signed, ex._clock_offset
    ex.signed, ex._clock_offset = fake_signed, fake_offset
    try:
        _check()
    finally:   # 같은 프로세스의 다른 시험이 가짜 서명을 물려받지 않게
        ex.signed, ex._clock_offset = orig
        ex._fee_cache.clear()


def _check():
    os.environ.setdefault("BINANCE_API_KEY", "x")
    os.environ.setdefault("BINANCE_SECRET_KEY", "y")
    ex._fee_cache.clear()
    run = lambda s: asyncio.run(ex.commission_bp(None, s))  # noqa: E731

    usdc, usdt = run("ETHUSDC"), run("ETHUSDT")
    assert (usdc["maker_bp"], usdc["taker_bp"], usdc["source"]) == (0.0, 4.0, "live"), usdc
    assert (usdt["maker_bp"], usdt["taker_bp"], usdt["source"]) == (2.0, 5.0, "live"), usdt
    run("ETHUSDC"); run("ETHUSDT")
    assert calls == ["ETHUSDC", "ETHUSDT"], calls          # 1시간 안 재호출 0

    # 비용: USDT 요율이면 옛 상수 그대로, USDC 는 수수료 차이만 바뀐다
    c_t = execution_plan(12.0, fees=usdt)["cost"]
    c_0 = execution_plan(12.0)["cost"]
    assert c_t["round_trip_bp"] == c_0["round_trip_bp"] == 5.88 and c_t["total_bp"] == c_0["total_bp"], (c_t, c_0)
    assert c_t["fee"]["promo"] is False and c_0["fee"]["source"] == "fallback"
    e_c = execution_plan(12.0, fees=usdc)
    c_c = e_c["cost"]
    assert c_c["round_trip_bp"] == 1.88, c_c                # 5.88 − 2×(2.0−0)
    assert c_c["fee"] == {"symbol": "ETHUSDC", "maker_bp": 0.0, "taker_bp": 4.0, "source": "live", "promo": True}, c_c
    assert (e_c["entry"]["maker_bp"], e_c["entry"]["taker_bp"]) == (0.0, 4.0)
    assert abs(c_c["total_bp"] - (c_t["total_bp"] - 4.0 + c_c["stop_hit_rate"] * 1.0)) < 0.011, (c_c, c_t)

    # 판정·크기 쪽(지평 권고·처방 비용)은 요율과 무관 -- 표준 상수
    live = {str(h): {"LONG": {"safe_mae_pct": m}, "SHORT": {"safe_mae_pct": m}}
            for h, m in ((60, 1.8), (120, 2.4), (240, 3.4), (480, 7.0), (1440, 23.0))}
    kw = dict(side="LONG", equity=1000.0, existing_notional=0.0, unrealized_pnl=0.0, risk_table=live,
              vol_bpm=12.0, cap_x=8.0, atr_pct=0.0004, hold_min=240)
    a, b = plan_now(**kw, fees=usdc), plan_now(**kw)
    assert a["hold"] == b["hold"] and a["prescription"] == b["prescription"] and a["size"] == b["size"]
    assert a["execution"]["cost"]["round_trip_bp"] != b["execution"]["cost"]["round_trip_bp"]

    # 실패: 만료 뒤 조회 실패 → 지난 성공값(stale) · 이력 없는 심볼 → 표준 2/5bp(fallback) · 실패는 5분 캐시
    fail["on"] = True
    ex._fee_cache["ETHUSDC"] = (ex._fee_cache["ETHUSDC"][0] - 3601, 0.0, 4.0, "live")
    s = run("ETHUSDC")
    assert (s["maker_bp"], s["taker_bp"], s["source"]) == (0.0, 4.0, "stale"), s
    f = run("SOLUSDC")
    assert (f["maker_bp"], f["taker_bp"], f["source"]) == (2.0, 5.0, "fallback"), f
    n = len(calls)
    run("ETHUSDC"); run("SOLUSDC")
    assert len(calls) == n, calls                          # 실패 직후 재호출 0
    assert execution_plan(12.0, fees=f)["cost"]["fee"]["promo"] is False   # 추정값엔 프로모션 표시 없음


if __name__ == "__main__":
    test_commission_rate_cache_and_plan_cost()
    print("통과")
