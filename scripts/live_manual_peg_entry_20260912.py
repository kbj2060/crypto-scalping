"""대시보드 수동 진입 — peg 지정가 + 크기 상한 (2026-09-12, 사용자 요청).

왜 있나: 실계좌 19왕복에서 **손실은 방향이 아니라 크기에서 왔다** — 크기–수익 상관 −0.494,
한 건(중앙 명목의 5.2배·28.4배 레버리지)이 이익 합 전부보다 큰 −548.88 을 냈다. 크기를
버튼이 정하면 그 순간의 재량이 사라진다. 근거: research_sizing_counterfactual_two_ledgers_20260912.py

**봇과 완전히 분리된 경로다.** trading_bot 의 BinanceFuturesExecutionAdapter 를 안 쓰는 이유 둘:
  (1) 그쪽은 `_ensure_one_way_mode` 로 **헤지 모드를 거부**한다. 이 계좌는 헤지 모드다.
  (2) 플래그가 전역이라 BINANCE_EXECUTION_DRY_RUN 을 끄면 **봇도 같이 실거래로 간다**.
      오메가의 자기 판단 실행은 승인된 적 없는 별개 축이다.
그래서 자체 플래그 `DASHBOARD_MANUAL_EXEC_ENABLED`(기본 false)를 쓴다.

1단계(현재): 게이트가 닫혀 있어 **주문을 보내지 않는다**. 보낼 주문을 그대로 돌려준다.
2단계: 같은 코드가 같은 계획으로 실제 전송한다 -- 미리보기가 실제 경로를 통과하므로
       "미리보기는 됐는데 실주문은 다르다"가 생기지 않는다.

체결 정책(사용자 선택 b): peg post-only(GTX) → `FALLBACK_SEC` 안에 미체결이면 테이커 전환.
실측(1,340 legs): peg 2.76bp/leg 100% 체결, 타임아웃 폴백분 평균 14.82bp.
"""
from __future__ import annotations

import os
from typing import Any

FAPI = "https://fapi.binance.com"
FALLBACK_SEC = 120.0          # 섀도우 워커의 MAKER_SHADOW_TIMEOUT_S 와 같은 값
_filters: dict[str, dict[str, float]] = {}


def exec_enabled() -> bool:
    """실주문 게이트. **코드 기본값이 아니라 서버 .env 가 정답**이라 매번 읽는다
    (feedback_verify_live_flags_from_server_env_not_code_defaults_20260904)."""
    return os.getenv("DASHBOARD_MANUAL_EXEC_ENABLED", "").strip().lower() in ("1", "true", "yes")


async def load_filters(session, symbol: str) -> dict[str, float]:
    """거래소의 수량/가격 격자. 캐시한다 -- 심볼 규격은 안 변한다."""
    if symbol in _filters:
        return _filters[symbol]
    async with session.get(f"{FAPI}/fapi/v1/exchangeInfo") as response:
        info = await response.json()
    for item in info.get("symbols", []):
        if item.get("symbol") != symbol:
            continue
        out = {"step": 0.001, "tick": 0.01, "min_qty": 0.0, "min_notional": 0.0}
        for f in item.get("filters", []):
            kind = f.get("filterType")
            if kind == "LOT_SIZE":
                out["step"], out["min_qty"] = float(f["stepSize"]), float(f["minQty"])
            elif kind == "PRICE_FILTER":
                out["tick"] = float(f["tickSize"])
            elif kind in ("MIN_NOTIONAL", "NOTIONAL"):
                out["min_notional"] = float(f.get("notional") or f.get("minNotional") or 0.0)
        _filters[symbol] = out
        return out
    raise RuntimeError(f"symbol not in exchangeInfo: {symbol}")


def _floor_to(value: float, grid: float) -> float:
    """격자에 **내림**. 반올림하면 상한을 넘길 수 있고, 상한은 넘지 않는 게 요점이다."""
    if grid <= 0:
        return value
    return int(value / grid + 1e-9) * grid


def build_entry_plan(*, side: str, best_bid: float, best_ask: float, recommended_qty: float,
                     cap_notional: float | None, filters: dict[str, float],
                     symbol: str = "ETHUSDT") -> dict[str, Any]:
    """보낼 주문 하나를 만든다. **순수 함수** -- 네트워크도 시계도 안 본다(그래야 검사가 된다).

    peg 는 «내가 메이커로 남는 가격»이다: 롱은 최우선 매수호가, 숏은 최우선 매도호가.
    한 틱 더 공격적으로 가면 테이커가 되어 post-only(GTX)가 거부한다.
    """
    if side not in ("LONG", "SHORT"):
        raise ValueError(f"side must be LONG/SHORT, got {side!r}")
    if not (best_bid > 0 and best_ask > 0 and best_ask >= best_bid):
        raise ValueError(f"bad book: bid={best_bid} ask={best_ask}")

    price = _floor_to(best_bid, filters["tick"]) if side == "LONG" else best_ask
    notes: list[str] = []
    qty = max(0.0, float(recommended_qty))

    if cap_notional and qty * price > cap_notional:
        qty = cap_notional / price
        notes.append(f"상한 적용: 명목 {cap_notional:,.0f} USDT")
    qty = _floor_to(qty, filters["step"])

    blocked = None
    if qty < filters["min_qty"] or qty <= 0:
        blocked = f"수량이 최소 {filters['min_qty']} 미만"
    elif filters["min_notional"] and qty * price < filters["min_notional"]:
        blocked = f"명목이 최소 {filters['min_notional']:,.0f} USDT 미만"

    return {
        "symbol": symbol,
        "side": "BUY" if side == "LONG" else "SELL",
        "positionSide": side,          # 헤지 모드라 필수. 빼면 롱/숏이 서로를 상계한다.
        "type": "LIMIT",
        "timeInForce": "GTX",          # post-only -- 테이커가 되면 체결 대신 거부된다
        "price": round(price, 8),
        "quantity": round(qty, 8),
        "notional_usdt": round(qty * price, 2),
        "fallback_after_sec": FALLBACK_SEC,
        "fallback": "taker",           # 사용자 선택 b: 미체결이면 테이커 전환
        "cap_notional_usdt": cap_notional,
        "notes": notes,
        "blocked": blocked,
        "dry_run": not exec_enabled(),
    }


def _self_check() -> None:
    f = {"step": 0.001, "tick": 0.01, "min_qty": 0.001, "min_notional": 20.0}

    plan = build_entry_plan(side="LONG", best_bid=2470.00, best_ask=2470.01,
                            recommended_qty=3.4, cap_notional=None, filters=f)
    assert plan["side"] == "BUY" and plan["positionSide"] == "LONG", plan
    assert plan["price"] == 2470.00, plan          # 롱 peg = 최우선 매수호가
    assert plan["timeInForce"] == "GTX" and plan["blocked"] is None, plan

    plan = build_entry_plan(side="SHORT", best_bid=2470.00, best_ask=2470.01,
                            recommended_qty=3.4, cap_notional=None, filters=f)
    assert plan["side"] == "SELL" and plan["price"] == 2470.01, plan   # 숏 peg = 최우선 매도호가

    # 상한: 명목 15,201 → 6.154 ETH. 내림이라 상한을 **넘지 않는다**.
    plan = build_entry_plan(side="LONG", best_bid=2470.00, best_ask=2470.01,
                            recommended_qty=14.273, cap_notional=15201.0, filters=f)
    assert plan["quantity"] == 6.154, plan
    assert plan["notional_usdt"] <= 15201.0, plan
    assert plan["notes"], "상한을 걸었으면 이유를 남긴다"

    # 상한 아래면 건드리지 않는다
    plan = build_entry_plan(side="LONG", best_bid=2470.00, best_ask=2470.01,
                            recommended_qty=2.0, cap_notional=15201.0, filters=f)
    assert plan["quantity"] == 2.0 and not plan["notes"], plan

    # 최소 명목 미달은 조용히 보내지 않고 막는다
    plan = build_entry_plan(side="LONG", best_bid=2470.00, best_ask=2470.01,
                            recommended_qty=0.005, cap_notional=None, filters=f)
    assert plan["blocked"] and "명목" in plan["blocked"], plan

    for bad in (dict(side="FLAT", best_bid=1.0, best_ask=1.0),
                dict(side="LONG", best_bid=0.0, best_ask=1.0),
                dict(side="LONG", best_bid=2.0, best_ask=1.0)):
        try:
            build_entry_plan(recommended_qty=1.0, cap_notional=None, filters=f, **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"막았어야 한다: {bad}")

    print("통과 6/6 — 수동 진입 계획 계약 유지")


if __name__ == "__main__":
    _self_check()
