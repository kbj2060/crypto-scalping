"""주식 선물(SNDK·SOXL)은 USDC 시장이 없어 주문 심볼 = 시세 심볼(USDT). 2026-11-02"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard import orders

for a, s in (("sndk", "SNDKUSDT"), ("soxl", "SOXLUSDT")):
    assert orders.MANUAL_EXEC_SYMBOLS[a] == orders.MARKET_SYMBOLS[a] == s
    assert orders.exec_asset_of(s) == a
print("ok")
