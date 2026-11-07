"""단일자산 모드: 주문 심볼의 증거금 풀 선택 자체점검 (2026-11-05).
실행: python test/test_pool_balance_20261105.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard.orders import pool_balance, pool_positions

A = {"balance": {"margin": 1500.0}, "balances": {"USDC": {"margin": 1500.0}, "USDT": {"margin": 300.0}},
     "positions": [{"symbol": "ETHUSDC", "notional": -900.0}, {"symbol": "SOXLUSDT", "notional": 100.0}]}
assert pool_balance(A, "SOXLUSDT")["margin"] == 300.0 and pool_balance(A, "ETHUSDC")["margin"] == 1500.0
assert [p["symbol"] for p in pool_positions(A, "SOXLUSDT")] == ["SOXLUSDT"]            # ETHUSDC 숏은 USDT 풀 상한에 안 센다
assert pool_balance({**A, "balances": {"USDC": {"margin": 1500.0}}}, "SOXLUSDT") == {}   # USDT 지갑 없음 = 0 (USDC 로 떨어지지 않는다)
OLD = {"balance": {"margin": 7.0}, "positions": A["positions"]}                         # balances 없는 옛 응답 = 종전 그대로
assert pool_balance(OLD, "SOXLUSDT")["margin"] == 7.0 and len(pool_positions(OLD, "SOXLUSDT")) == 2
assert len(pool_positions({**A, "multi_assets_margin": True}, "SOXLUSDT")) == 2          # 멀티에셋 = 풀 하나
print("ok — 5건 통과")

# 2026-11-07 진입 수량은 그 풀의 가용 잔고 안이어야 한다 -- 순자산 100% 를 그대로 쓰면 거래소가 -2019 로 거부하고 SL/TP 도 안 걸린다
from scripts.live_manual_peg_entry_20260912 import MARGIN_BUFFER, build_entry_plan

F = {"tick": 0.01, "step": 0.001, "min_qty": 0.001, "min_notional": 5.0}
def plan(**kw):
    base = dict(side="SHORT", best_bid=2241.0, best_ask=2241.12, recommended_qty=0.0, cap_notional=None, filters=F,
                symbol="ETHUSDC", equity=993.05, leverage=10.0, fraction=1.0, fraction_of_equity=True, order_leverage=10.0)
    return build_entry_plan(**{**base, **kw})
need = lambda p: p["quantity"] * p["price"] / 10.0
assert need(plan()) > 993.05 * (1 - MARGIN_BUFFER), "가용 잔고를 안 주면 옛 동작 그대로"
p = plan(available_margin=993.05)                                                       # 11-07 16:29 거부된 주문(4.43개, 증거금 ≈ 가용 전부)
assert need(p) <= 993.05 * (1 - MARGIN_BUFFER) and p["capped"] and p["notes"] and not p["blocked"], p
p = plan(symbol="SNDKUSDT", equity=686.0, available_margin=384.7)                       # 다른 포지션이 증거금을 쓰는 풀: 순자산 686 · 가용 385
assert need(p) <= 384.7 * (1 - MARGIN_BUFFER) and p["capped"], p
p = plan(fraction=0.25, available_margin=993.05)                                        # 가용 안이면 요청 그대로, 줄였다고 말하지 않는다
assert abs(need(p) - 993.05 * 0.25) < 0.3 and not p["capped"], p
p = plan(available_margin=0.0)                                                          # 가용 0 = 막고 사유를 말한다(0개 주문을 내지 않는다)
assert p["quantity"] == 0 and "가용" in p["blocked"], p
print("ok — 가용 잔고 상한 5건 통과")
