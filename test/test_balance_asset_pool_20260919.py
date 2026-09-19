"""잔고를 «어느 자산 지갑»에서 읽는가 자체점검 (2026-09-19).

고친 버그: 단일자산 담보 모드에서 /fapi/v2/account 의 최상위 total* 합계는 **USDT 전용**이다.
현금이 USDC 에만 있으면 그 필드가 전부 0 으로 내려오고, 대시보드는 «잔고 0»을 띄운다.
거기서 파생되는 순자산(equity)이 0 이라 진입 상한·증거금 표시까지 같이 0 이 된다.
2026-09-19 실측: multiAssetsMargin=False · totalWalletBalance 0.00 · assets 의 USDC 1,472.07.

실행: python test/test_balance_asset_pool_20260919.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.live_binance_account_20260910 import pick_balance

# 실측 모양 그대로: 최상위는 0, 돈은 USDC 에 있다.
SINGLE = {
    "multiAssetsMargin": False,
    "totalWalletBalance": "0.00000000", "totalMarginBalance": "0.00000000",
    "totalInitialMargin": "0.00000000", "totalMaintMargin": "0.00000000",
    "availableBalance": "0.00000000", "totalUnrealizedProfit": "0.00000000",
    "assets": [
        {"asset": "USDT", "walletBalance": "0", "availableBalance": "0", "marginBalance": "0"},
        {"asset": "USDC", "walletBalance": "1472.06877318", "availableBalance": "1472.06877318",
         "marginBalance": "1472.06877318", "initialMargin": "0", "maintMargin": "0",
         "unrealizedProfit": "0"},
    ],
}
bal, listed = pick_balance(SINGLE, "USDC")
assert round(bal["wallet"], 2) == 1472.07, bal          # 🔴이 줄이 이번 버그다
assert round(bal["margin"], 2) == 1472.07, bal          # equity -> 진입 상한이 여기서 나온다
assert [a["asset"] for a in listed] == ["USDC"]         # 잔액 0 인 USDT 는 목록에서 빠진다

# USDT 를 고르면 0 이 맞다 -- 그 지갑엔 실제로 없다(«0 이 틀렸다»가 아니라 «지갑을 잘못 봤다»).
assert pick_balance(SINGLE, "USDT")[0]["wallet"] == 0.0

# 멀티에셋 모드면 최상위 합계가 계좌 전체다.
MULTI = {**SINGLE, "multiAssetsMargin": True, "totalWalletBalance": "1500.5",
         "totalMarginBalance": "1500.5", "availableBalance": "1400.0",
         "totalUnrealizedProfit": "0", "totalInitialMargin": "100.5", "totalMaintMargin": "5"}
assert pick_balance(MULTI, "USDC")[0]["wallet"] == 1500.5

# 그 자산 항목 자체가 없으면 최상위로 떨어지되, 목록은 그대로 실어 보낸다.
bal2, listed2 = pick_balance(SINGLE, "BUSD")
assert bal2["wallet"] == 0.0 and [a["asset"] for a in listed2] == ["USDC"]

print("ok — 6건 통과")
