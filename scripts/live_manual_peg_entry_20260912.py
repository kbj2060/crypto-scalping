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

import math
import os
from typing import Any

FAPI = "https://fapi.binance.com"
FALLBACK_SEC = 120.0          # 진입 기본값. 섀도우 워커의 MAKER_SHADOW_TIMEOUT_S 와 같은 값

# ── 청산 마감을 변동성에 연동한다(2026-09-13, 사용자 요청) ──────────────────────
# 근거: 섀도우 7,982legs 를 1분봉 실현변동성과 조인해 보니 **체결 시간이 변동성에 강하게
# 반비례**한다(순위상관 -0.429). 변동성 5분위별 중앙 체결시간 15.2 / 5.4 / 4.5 / 3.2 / 1.8초.
# 그래서 고정 120초는 빠른 장에서 필요량의 60배를 기다리는 셈이고, 청산에서 그 초과분은
# **포지션을 든 채 노출되는 시간**이다.
# 곡선은 적합해서 뽑았다: 중앙체결시간 ≈ 52.0 / vol^1.472. 마감은 그 ~29배(체결 분포의
# 중앙 4.1초 대비 99% 82.4초 = 20배에 여유를 더한 값)로 잡아 K=1500 을 쓴다.
# 실측 검증: K=1500 에서 메이커 체결 중 **0.03%(2건)** 만 잘리고, 그 2건은 원래 8.203bp 를
# 내던 건이라 테이커 5.0bp 로 바뀌면 오히려 미세하게 유리하다(전체 평균 -0.0008bp).
EXIT_DEADLINE_K = 1500.0
EXIT_DEADLINE_EXP = 1.472
EXIT_DEADLINE_MIN = 15.0
EXIT_DEADLINE_MAX = 120.0
EXIT_VOL_WINDOW = 30          # 1분봉 개수. 수익률 30개를 쓰므로 종가는 31개 필요하다.

# 🔴이 값을 넘으면 **지정가를 아예 안 걸고 시장가로 닫는다**(사용자 요청).
# 정직하게: 비용으로는 손해다. 직전 1분이 29.5bp 움직인 상위 1% 구간에서도 peg 는
# 0.5초에 3.26bp 로 체결돼 테이커 5.0bp 보다 싸다(비용-변동성 순위상관 +0.026 = 사실상 0).
# 그럼에도 두는 이유는 비용이 아니라 **확실성**이다 -- 급락 중에 «닫혔나»를 확인하느라
# 초를 세는 상황을 없애는 값이고, 그 대가가 약 1.7bp 다. 30 은 1분봉 869일 기준 99.3분위
# (하루 약 9.7분)라 평상시엔 발동하지 않는다.
EXIT_TAKER_VOL_BPM = 30.0
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
                     symbol: str = "ETHUSDT", existing_notional: float = 0.0,
                     equity: float = 0.0, leverage: float = 0.0) -> dict[str, Any]:
    """보낼 주문 하나를 만든다. **순수 함수** -- 네트워크도 시계도 안 본다(그래야 검사가 된다).

    peg 는 «내가 메이커로 남는 가격»이다: 롱은 최우선 매수호가, 숏은 최우선 매도호가.
    한 틱 더 공격적으로 가면 테이커가 되어 post-only(GTX)가 거부한다.

    🔴상한은 **합산 포지션**에 건다(2026-09-12 수정). 주문 하나에만 걸면 나눠 넣어서
    얼마든지 넘길 수 있는데, 막아야 할 대상은 −548.88 을 만든 그 왕복의 `max_qty` 14.273 --
    즉 **왕복 중 최대 포지션**이지 주문 크기가 아니다. 실제로 기존 숏 3.025(7,642)가 열린
    상태에서 2.693(6,803)을 더하면 14,444 로 상한 13,579 를 넘고 있었다.

    equity/leverage 는 **화면 설명용**이다. 교차 마진이라 청산 거리는 대략
    `순자산 / 총명목` 이고(실측 1,065/7,642 = 13.9% vs 거래소 13.48%), 설정 레버리지는
    거기 안 들어간다 -- 그건 잠기는 증거금만 정한다.
    """
    if side not in ("LONG", "SHORT"):
        raise ValueError(f"side must be LONG/SHORT, got {side!r}")
    if not (best_bid > 0 and best_ask > 0 and best_ask >= best_bid):
        raise ValueError(f"bad book: bid={best_bid} ask={best_ask}")

    price = _floor_to(best_bid, filters["tick"]) if side == "LONG" else best_ask
    notes: list[str] = []
    qty = max(0.0, float(recommended_qty))
    existing_notional = max(0.0, float(existing_notional))

    if cap_notional:
        room = cap_notional - existing_notional
        if room <= 0:
            qty = 0.0
            notes.append(f"기존 포지션 {existing_notional:,.0f} USDT 가 이미 상한을 채웠습니다")
        elif qty * price > room:
            qty = room / price
            notes.append(f"상한까지 남은 여유 {room:,.0f} USDT 로 줄였습니다"
                         f" (기존 {existing_notional:,.0f} + 신규 = 상한 {cap_notional:,.0f})")
    qty = _floor_to(qty, filters["step"])

    blocked = None
    if qty < filters["min_qty"] or qty <= 0:
        blocked = ("상한 여유가 없습니다" if cap_notional and existing_notional >= cap_notional
                   else f"수량이 최소 {filters['min_qty']} 미만")
    elif filters["min_notional"] and qty * price < filters["min_notional"]:
        blocked = f"명목이 최소 {filters['min_notional']:,.0f} USDT 미만"

    notional = qty * price
    total_notional = notional + existing_notional
    margin = notional / leverage if leverage else 0.0
    return {
        "symbol": symbol,
        "side": "BUY" if side == "LONG" else "SELL",
        "positionSide": side,          # 헤지 모드라 필수. 빼면 롱/숏이 서로를 상계한다.
        "type": "LIMIT",
        "timeInForce": "GTX",          # post-only -- 테이커가 되면 체결 대신 거부된다
        "price": round(price, 8),
        "quantity": round(qty, 8),
        "notional_usdt": round(notional, 2),
        # ── 화면이 «이게 무슨 뜻인지»를 말할 수 있게 하는 값들 ────────────────────
        "margin_usdt": round(margin, 2),                       # 실제로 잠기는 현금
        "margin_pct_of_equity": round(100 * margin / equity, 1) if equity else None,
        "leverage": leverage or None,                          # 거래소 설정값
        "existing_notional_usdt": round(existing_notional, 2),
        "total_notional_usdt": round(total_notional, 2),
        "effective_leverage": round(total_notional / equity, 2) if equity else None,
        # 교차 마진 근사. 거래소 실제 청산가와 0.5pp 안쪽으로 맞았다(실측).
        "liq_distance_pct": round(100 * equity / total_notional, 1) if equity and total_notional else None,
        "cap_used_pct": round(100 * total_notional / cap_notional, 0) if cap_notional else None,
        "fallback_after_sec": FALLBACK_SEC,
        "fallback": "taker",           # 사용자 선택 b: 미체결이면 테이커 전환
        "cap_notional_usdt": cap_notional,
        "notes": notes,
        "blocked": blocked,
        "dry_run": not exec_enabled(),
    }


def realized_vol_bpm(closes: list[float]) -> float | None:
    """1분봉 종가들로 실현변동성(bp/√분). 표본이 모자라면 None -- **0 이 아니다**.
    0 으로 돌리면 «변동성 없음»이 되어 마감이 최대로 늘어나는데, 실제로는 «모른다»다."""
    px = [float(c) for c in closes if c and float(c) > 0]
    if len(px) < 10:
        return None
    rets = [math.log(px[i] / px[i - 1]) for i in range(1, len(px))]
    n = len(rets)
    mean = sum(rets) / n
    var = sum((r - mean) ** 2 for r in rets) / (n - 1)
    return math.sqrt(var) * 1e4


def exit_deadline_sec(vol_bpm: float | None) -> float:
    """변동성이 클수록 짧은 마감. 변동성을 모르면 보수적으로 최대값을 쓴다.

    빠른 장이 위험해서가 아니라 **빠른 장에서는 어차피 즉시 체결되기 때문**이다 --
    남는 시간은 이득 없이 노출만 늘린다. 상단 주석의 실측 근거 참조."""
    if vol_bpm is None or vol_bpm <= 0:
        return EXIT_DEADLINE_MAX
    raw = EXIT_DEADLINE_K / (vol_bpm ** EXIT_DEADLINE_EXP)
    return min(EXIT_DEADLINE_MAX, max(EXIT_DEADLINE_MIN, raw))


def build_exit_plan(*, position_side: str, position_qty: float, best_bid: float, best_ask: float,
                    filters: dict[str, float], symbol: str = "ETHUSDT",
                    entry_price: float = 0.0, mark_price: float = 0.0,
                    vol_bpm: float | None = None) -> dict[str, Any]:
    """열린 포지션 하나를 **메이커로 닫는** 주문을 만든다. 순수 함수 -- build_entry_plan 과 같다.

    🔴`reduceOnly` 를 보내지 않는다. 이 계좌는 헤지 모드고 바이낸스는 헤지 모드에서
    reduceOnly 를 거부한다(-1106). 헤지 모드의 청산은 «측면 반전 + **같은** positionSide»
    이고, 안전장치는 플래그가 아니라 **수량**이다 -- 그래서 호출부가 방금 읽은 포지션
    수량을 넘겨야 하고 여기서 그 위로는 절대 안 올린다.

    메이커 가격은 진입의 거울이다: 롱을 닫으면 SELL 이라 최우선 **매도**호가,
    숏을 닫으면 BUY 라 최우선 **매수**호가. 한 틱이라도 더 공격적이면 GTX 가 거부한다.
    """
    if position_side not in ("LONG", "SHORT"):
        raise ValueError(f"position_side must be LONG/SHORT, got {position_side!r}")
    if not (best_bid > 0 and best_ask > 0 and best_ask >= best_bid):
        raise ValueError(f"bad book: bid={best_bid} ask={best_ask}")

    closing_long = position_side == "LONG"
    price = best_ask if closing_long else _floor_to(best_bid, filters["tick"])
    qty = _floor_to(max(0.0, float(position_qty)), filters["step"])
    deadline = exit_deadline_sec(vol_bpm)
    # 극단 변동성이면 지정가를 걸지 않는다. 비용이 아니라 확실성을 사는 선택이라
    # 화면에 이유를 그대로 남긴다(조용히 시장가로 바꾸지 않는다).
    market = vol_bpm is not None and vol_bpm >= EXIT_TAKER_VOL_BPM

    blocked = None
    if qty <= 0 or qty < filters["min_qty"]:
        blocked = "닫을 포지션이 없습니다" if position_qty <= 0 else \
                  f"남은 수량이 최소 {filters['min_qty']} 미만이라 지정가로 못 닫습니다"
    elif filters["min_notional"] and qty * price < filters["min_notional"]:
        # 지정가로는 못 보낸다. 조용히 시장가로 바꾸지 않고 화면에 그대로 말한다.
        blocked = (f"명목 {qty * price:,.0f} 가 최소 {filters['min_notional']:,.0f} USDT 미만입니다"
                   " -- 이 크기는 시장가로 닫아야 합니다")

    notional = qty * price
    move = ((price - entry_price) / entry_price if closing_long else
            (entry_price - price) / entry_price) if entry_price > 0 else None
    return {
        "symbol": symbol,
        "side": "SELL" if closing_long else "BUY",
        "positionSide": position_side,   # 반전하지 않는다 -- 반전하면 청산이 아니라 신규 진입이다
        "type": "MARKET" if market else "LIMIT",
        **({} if market else {"timeInForce": "GTX", "price": round(price, 8)}),
        "quantity": round(qty, 8),
        "notional_usdt": round(notional, 2),
        "reference_price": round(price, 8),   # 시장가일 때도 화면이 «대략 얼마»를 말할 수 있게
        "position_side": position_side,
        "position_qty": round(float(position_qty), 8),
        "entry_price": entry_price or None,
        "mark_price": mark_price or None,
        # 이 가격에 닫으면 수수료 전 몇 % 인가. 화면이 «왜 지금 닫나»를 말할 수 있게 한다.
        "exit_move_pct": round(100 * move, 3) if move is not None else None,
        "fallback_after_sec": round(deadline, 1),
        "fallback": "taker",
        "repeg": not market,             # 진입과 다르다 -- 위 주석 참조
        "vol_bpm": round(vol_bpm, 2) if vol_bpm is not None else None,
        "market_reason": (f"변동성 {vol_bpm:.1f} ≥ {EXIT_TAKER_VOL_BPM:.0f} bp/√분 — "
                          "지정가를 걸지 않고 즉시 시장가로 닫습니다"
                          f" (비용은 메이커보다 약 1.7bp 비쌉니다)") if market else None,
        "notes": [],
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

    # 🔴상한은 **합산**에 건다. 실제로 겪은 값으로 검사한다: 기존 7,642 가 열린 채
    # 6,803 을 더하면 14,444 로 13,579 를 넘었는데 예전 코드는 그냥 통과시켰다.
    plan = build_entry_plan(side="SHORT", best_bid=2526.08, best_ask=2526.09,
                            recommended_qty=2.693, cap_notional=13579.31, filters=f,
                            existing_notional=7641.6, equity=1064.69, leverage=30.0)
    assert plan["total_notional_usdt"] <= 13579.31 + 1e-6, plan
    assert plan["quantity"] < 2.693, "기존 포지션만큼 줄어야 한다"
    assert plan["notes"], "줄였으면 이유를 남긴다"

    # 기존 포지션이 이미 상한을 채웠으면 아예 막는다
    plan = build_entry_plan(side="SHORT", best_bid=2526.08, best_ask=2526.09,
                            recommended_qty=2.0, cap_notional=13579.31, filters=f,
                            existing_notional=14000.0, equity=1064.69, leverage=30.0)
    assert plan["quantity"] == 0.0 and plan["blocked"], plan

    # 화면 설명값: 증거금은 **설정 레버리지**로 나눈 값, 청산 근사는 순자산/총명목
    plan = build_entry_plan(side="SHORT", best_bid=2526.08, best_ask=2526.09,
                            recommended_qty=2.693, cap_notional=None, filters=f,
                            existing_notional=0.0, equity=1064.69, leverage=30.0)
    assert abs(plan["margin_usdt"] - plan["notional_usdt"] / 30.0) < 0.01, plan
    assert plan["effective_leverage"] == round(plan["notional_usdt"] / 1064.69, 2), plan
    # 순자산 1,064.69 / 명목 6,802 ≈ 15.6% -- 거래소 실측(13.48% @ 7,642)과 같은 눈금
    assert 10.0 < plan["liq_distance_pct"] < 20.0, plan
    assert plan["margin_pct_of_equity"] is not None, plan

    # equity/leverage 를 모르면 설명값은 None 이지 0 이 아니다(0 은 "레버리지 0배"로 읽힌다)
    plan = build_entry_plan(side="LONG", best_bid=2470.00, best_ask=2470.01,
                            recommended_qty=1.0, cap_notional=None, filters=f)
    assert plan["effective_leverage"] is None and plan["liq_distance_pct"] is None, plan

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

    # ── 청산 계획 ────────────────────────────────────────────────────────────
    f = {"step": 0.001, "tick": 0.01, "min_qty": 0.001, "min_notional": 20.0}
    x = build_exit_plan(position_side="LONG", position_qty=2.693, best_bid=2470.00,
                        best_ask=2470.01, filters=f, entry_price=2400.0)
    assert x["side"] == "SELL" and x["positionSide"] == "LONG", x
    assert x["price"] == 2470.01, x                      # 롱 청산 = 최우선 매도호가
    assert "reduceOnly" not in x, "헤지 모드에서 reduceOnly 는 -1106 로 거부된다"
    assert x["quantity"] == 2.693 and x["blocked"] is None, x
    assert abs(x["exit_move_pct"] - 2.917) < 0.01, x

    x = build_exit_plan(position_side="SHORT", position_qty=3.025, best_bid=2470.00,
                        best_ask=2470.01, filters=f, entry_price=2500.0)
    assert x["side"] == "BUY" and x["positionSide"] == "SHORT", x
    assert x["price"] == 2470.00, x                      # 숏 청산 = 최우선 매수호가
    assert x["exit_move_pct"] > 0, "진입보다 싸게 되사면 이익이다"

    # 수량은 포지션 위로 절대 안 올라간다(헤지 모드엔 reduceOnly 가 없어 이게 유일한 안전장치)
    x = build_exit_plan(position_side="LONG", position_qty=0.0015, best_bid=2470.00,
                        best_ask=2470.01, filters=f)
    assert x["quantity"] == 0.001 and "최소" in (x["blocked"] or ""), x
    x = build_exit_plan(position_side="LONG", position_qty=0.0,
                        best_bid=2470.00, best_ask=2470.01, filters=f)
    assert x["blocked"] == "닫을 포지션이 없습니다", x

    # ── 변동성 연동 마감 ─────────────────────────────────────────────────────
    assert exit_deadline_sec(None) == EXIT_DEADLINE_MAX, "모르면 보수적으로 최대"
    assert exit_deadline_sec(0.0) == EXIT_DEADLINE_MAX
    prev = 1e9
    for v in (2.0, 5.25, 9.97, 17.74, 30.0, 100.0):
        t = exit_deadline_sec(v)
        assert t <= prev, f"변동성이 오르면 마감은 줄어야 한다: {v}"
        assert EXIT_DEADLINE_MIN <= t <= EXIT_DEADLINE_MAX, t
        prev = t
    # 실측 적합점(위 주석의 표와 같은 값이어야 한다)
    assert exit_deadline_sec(5.25) == EXIT_DEADLINE_MAX, exit_deadline_sec(5.25)
    assert abs(exit_deadline_sec(9.97) - 50.8) < 1.0, exit_deadline_sec(9.97)
    assert abs(exit_deadline_sec(17.74) - 21.8) < 1.0, exit_deadline_sec(17.74)
    assert exit_deadline_sec(30.0) == EXIT_DEADLINE_MIN

    # 변동성 추정: 일정한 값이면 0, 표본 부족이면 None(0 이 아니다)
    assert realized_vol_bpm([100.0] * 31) == 0.0
    assert realized_vol_bpm([100.0, 101.0]) is None, "표본 부족은 None"
    assert realized_vol_bpm([]) is None
    noisy = realized_vol_bpm([100.0 * (1.001 ** (i % 2)) for i in range(31)])
    assert noisy and noisy > 0

    # 극단 변동성이면 시장가 계획이 되고 지정가 필드가 사라진다
    m = build_exit_plan(position_side="LONG", position_qty=2.0, best_bid=2470.00,
                        best_ask=2470.01, filters=f, vol_bpm=45.0)
    assert m["type"] == "MARKET" and "price" not in m and "timeInForce" not in m, m
    assert m["repeg"] is False and m["market_reason"], m
    assert m["reference_price"] == 2470.01, "시장가여도 화면이 쓸 참조가는 남긴다"

    # 보통 변동성이면 지정가 + 줄어든 마감
    n = build_exit_plan(position_side="LONG", position_qty=2.0, best_bid=2470.00,
                        best_ask=2470.01, filters=f, vol_bpm=12.0)
    assert n["type"] == "LIMIT" and n["timeInForce"] == "GTX", n
    assert n["repeg"] is True and n["market_reason"] is None
    assert EXIT_DEADLINE_MIN <= n["fallback_after_sec"] < EXIT_DEADLINE_MAX, n["fallback_after_sec"]
    assert n["vol_bpm"] == 12.0

    print("통과 33/33 — 진입 계획 + 합산 상한 + 화면 설명값 + 청산 계획 + 변동성 연동 마감")


if __name__ == "__main__":
    _self_check()
