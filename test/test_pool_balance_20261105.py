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
