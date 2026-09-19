"""notional_sum 의 «범위» 자체점검 (2026-09-19).

고친 버그: 진입 미리보기의 상한이 **수동 주문 심볼 하나만** 세고 있었다. 봇이 ETHUSDT 를
들고 있는데 수동으로 ETHUSDC 를 미리보면 기존 노출이 0 으로 잡혀 상한이 그만큼 헐거워진다.
교차 마진에서 청산거리 = 순자산 / **총명목**이므로 상한은 계좌 전체를 봐야 한다.
투영(청산거리 before/after)은 반대로 심볼 안에서만 성립하므로 그쪽은 심볼별 합을 그대로 쓴다.

실행: python test/test_notional_sum_scope_20260919.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard.server import notional_sum

POS = [
    {"symbol": "ETHUSDT", "side": "LONG", "notional": 1000.0},   # 봇
    {"symbol": "ETHUSDT", "side": "SHORT", "notional": -400.0},  # 헤지 다리(절대값으로 센다)
    {"symbol": "ETHUSDC", "side": "LONG", "notional": 250.0},    # 수동
    {"symbol": "BTCUSDT", "side": "LONG", "notional": 700.0},
    {"symbol": "SOLUSDT", "side": "LONG"},                       # notional 누락
    {"symbol": "XRPUSDT", "side": "LONG", "notional": None},
]

assert notional_sum(POS) == 2350.0, notional_sum(POS)            # 계좌 전체 = 상한이 봐야 할 값
assert notional_sum(POS, "ETHUSDC") == 250.0                     # 수동 심볼만
assert notional_sum(POS, "ETHUSDT") == 1400.0                    # 롱·숏 절대값 합(상쇄 없음)
assert notional_sum(POS, "없는심볼") == 0.0
assert notional_sum([]) == 0.0 and notional_sum(None) == 0.0     # 조회 실패/빈 계좌
# 🔴이 한 줄이 이번 버그다: 심볼별 합은 계좌 전체보다 **작다**. 상한에 심볼별을 쓰면 안 된다.
assert notional_sum(POS, "ETHUSDC") < notional_sum(POS)

print("ok — 6건 통과")
