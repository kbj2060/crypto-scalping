#!/usr/bin/env python3
"""리테일/중형/고래 분리의 자체점검 (2026-09-19).

확인하는 것 셋:
  1. **경계가 하나뿐인가** — 화면(dashboard/server.py)과 저장(체결 테이프 수집기)이 같은
     상수를 쓴다. 두 벌이 되면 「그날 화면의 고래」와 「DB 의 고래」가 조용히 갈라진다.
  2. 부등호 방향 — 리테일은 `< $10k`, 고래는 `>= $100k`. 경계값 자신이 어디에 속하는지까지.
  3. 셀 6칸의 대수 — 고래·리테일은 총량의 **부분집합**이고 중형은 뺄셈으로만 나온다.
     화면(app.js supplyFlowOfBar)이 이 뺄셈을 그대로 하므로 부호 규약이 여기서 굳는다.

python test/test_footprint_whale_split_20260919.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard import server as dash  # noqa: E402
from scripts.live_trade_tape_collector_20260916 import (  # noqa: E402
    RETAIL_MAX_USD,
    WHALE_MIN_USD,
    TakerOrderAggregator,
    TapeBuffer,
)


def test_one_definition_of_whale() -> None:
    """대시보드는 경계를 새로 만들지 않고 수집기 것을 가져다 쓴다."""
    assert dash.RETAIL_MAX_USD is RETAIL_MAX_USD
    assert dash.WHALE_MIN_USD is WHALE_MIN_USD
    assert RETAIL_MAX_USD < WHALE_MIN_USD, "중형이 들어설 자리가 있어야 한다"


def classify(notional: float) -> str:
    if notional < RETAIL_MAX_USD:
        return "retail"
    if notional >= WHALE_MIN_USD:
        return "whale"
    return "mid"


def test_boundaries_belong_to_the_right_side() -> None:
    assert classify(RETAIL_MAX_USD - 0.01) == "retail"
    assert classify(RETAIL_MAX_USD) == "mid", "$10k 딱 맞으면 리테일이 아니다"
    assert classify(WHALE_MIN_USD - 0.01) == "mid"
    assert classify(WHALE_MIN_USD) == "whale", "$100k 딱 맞으면 고래다"


def test_buffer_matches_that_classification() -> None:
    """수집기 버퍼가 위 분류와 같은 답을 내는가 — 부등호를 두 곳에 적었으니 대조한다."""
    price = 2500.0
    for notional in (1_000, 9_999, 10_000, 50_000, 99_999, 100_000, 500_000):
        buf = TapeBuffer(0.1)
        qty = notional / price
        buf.add(1_000_000, price, qty, sell=False)          # 총량
        buf.add_order(1_000_000, price, qty, sell=False)    # 크기 구간
        buf.add(1_001_000, price, 0.001, sell=False)   # 다음 초 -- 위를 닫는다
        row = buf.take_closed()[0]
        total, retail, whale, whale_n = row[2], row[8], row[10], row[12]
        want = classify(notional)
        assert abs(total - qty) < 1e-9, (notional, row)
        assert (retail > 0) == (want == "retail"), (notional, want, row)
        assert (whale > 0) == (want == "whale"), (notional, want, row)
        assert whale_n == (1 if want == "whale" else 0), (notional, row)
        # 중형은 칸이 없다 -- 뺄셈으로만 나온다.
        assert (round(total - retail - whale, 9) > 0) == (want == "mid"), (notional, row)


def test_cell_algebra_and_signs() -> None:
    """셀 = [매수, 매도, 고래매수, 고래매도, 리테일매수, 리테일매도]."""
    buy, sell = 100.0, 40.0
    w_buy, w_sell = 70.0, 5.0
    r_buy, r_sell = 12.0, 30.0
    whale = w_buy - w_sell
    retail = r_buy - r_sell
    mid = (buy - w_buy - r_buy) - (sell - w_sell - r_sell)
    assert whale == 65.0 and retail == -18.0 and mid == 13.0
    # 세 순수급의 합이 전체 델타여야 한다. 어느 하나를 잘못 빼면 여기서 깨진다.
    assert whale + retail + mid == buy - sell
    # 괴리(부호 갈림)가 실제로 잡히는 조합인지. 고래 매수 × 리테일 매도 = 흡수.
    assert (whale > 0) != (retail > 0)


def test_sweep_is_one_whale_not_many_minnows() -> None:
    """🔴이 저장소가 2026-09-19 에 실제로 틀렸던 자리.

    큰 주문 하나가 호가를 쓸면 `@trade` 는 작은 체결 여러 건으로 보고한다. 되묶지 않으면
    $250k 고래 하나가 리테일 25건이 된다 -- 같은 11.3초 구간 실측에서 고래 물량 비중이
    aggTrade 기준 37.4% vs @trade 기준 9.6% 로 갈렸고, 총 명목은 동일했다.
    """
    price, ts = 2500.0, 1_000
    fills = [4.0] * 25          # 각 $10,000 -> 낱개로는 전부 «중형»
    total = sum(fills)
    assert price * fills[0] < WHALE_MIN_USD, "낱개는 고래가 아니어야 시험이 성립한다"
    assert price * total >= WHALE_MIN_USD, "합치면 고래여야 한다"

    naive, aggregated = TapeBuffer(0.1), TapeBuffer(0.1)
    orders = TakerOrderAggregator()
    for q in fills:
        naive.add(ts, price, q, sell=False)
        naive.add_order(ts, price, q, sell=False)       # 되묶지 않은 «틀린» 방식
        aggregated.add(ts, price, q, sell=False)
        done = orders.add(price, q, ts, sell=False)
        assert done is None, "같은 (가격·방향·ms) 는 한 주문이다"
    last = orders.take()
    aggregated.add_order(last[2], last[0], last[1], last[3])
    for buf in (naive, aggregated):
        buf.add(ts + 1_000, price, 0.001, sell=False)   # 초를 닫는다

    n_row, a_row = naive.take_closed()[0], aggregated.take_closed()[0]
    assert n_row[2] == a_row[2] == total, "총량은 어느 쪽이든 같아야 한다"
    assert n_row[10] == 0.0, ("되묶지 않으면 고래가 0이다", n_row)
    assert a_row[10] == total, ("되묶으면 통째로 고래다", a_row)
    assert a_row[12] == 1, ("고래 «건수»는 25가 아니라 1이다", a_row)


def test_counts_close_under_subtraction() -> None:
    """건수 뺄셈이 «주문 단위 안에서» 닫히는가 — 2026-09-19 에 세 번 걸려 닫은 자리.

    `buy_n`(개별 체결)과 `*_n`(테이커 주문)은 단위가 달라 **서로 빼면 안 된다**. 대신
    주문 단위 안에서는 중형이 정확히 뺄셈으로 나와야 한다.
    """
    price = 2500.0
    buf = TapeBuffer(0.1)
    plan = [(1_000, "retail"), (50_000, "mid"), (60_000, "mid"), (200_000, "whale")]
    for notional, _want in plan:
        buf.add_order(1_000_000, price, notional / price, sell=False)
        # 같은 주문이 체결 3건으로 쪼개져 들어왔다고 치자 -- 체결 건수는 주문 수와 달라진다.
        for _ in range(3):
            buf.add(1_000_000, price, notional / price / 3, sell=False)
    buf.add(1_001_000, price, 0.001, sell=False)
    row = buf.take_closed()[0]

    fills_n, orders_n = row[4], row[16]
    retail_n, whale_n = row[14], row[12]
    assert orders_n == 4, ("주문 4건", row)
    assert fills_n == 12, ("체결 12건 -- 주문 수와 다르다", row)
    assert retail_n == 1 and whale_n == 1, (retail_n, whale_n)
    assert orders_n - retail_n - whale_n == 2, ("중형 주문 2건이 뺄셈으로 나와야 한다", row)
    # 쓸어담기 깊이: 주문 하나가 평균 몇 체결로 갈렸나.
    assert fills_n / orders_n == 3.0


if __name__ == "__main__":
    test_one_definition_of_whale()
    test_boundaries_belong_to_the_right_side()
    test_buffer_matches_that_classification()
    test_cell_algebra_and_signs()
    test_sweep_is_one_whale_not_many_minnows()
    test_counts_close_under_subtraction()
    print("ok — 단일 경계 · 부등호 · 버퍼 대조 · 셀 대수 · 되묶기 · 건수 뺄셈 6건 통과")
