"""시장 맥락 코인별 묶음 폭 (2026-10-04 검증에서 잡은 버그).

    python3 -m pytest -q test/test_market_ctx_coin_scale_20261004.py

ETH 달러 폭($5·$2.5)을 XRP($1.5)에 쓰면 청산가가 0·5 로 반올림됐다. 서버는 풋프린트 칸 폭 비율(_px_scale)로 줄인다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard import market_ctx as m  # noqa: E402

ETH_BUCKET, XRP_BUCKET, SOL_BUCKET = 0.5, 0.0003, 0.02   # server.py FLOW_SPECS


def test_xrp_hl_levels_stay_near_price():
    lv = m.hl_liq_levels([(10, 1.4969), (-5, 1.5031)], 1.5, m.HL_LIQ_BIN_USD * XRP_BUCKET / ETH_BUCKET)
    assert [x["px"] for x in lv["below"]] == [1.497] and [x["px"] for x in lv["above"]] == [1.503], lv


def test_sol_liq_profile_bins_are_relative():
    prof = m.liq_profile([(120.04, 100, True), (120.31, 50, False)], m.LIQ_PROFILE_BIN_USD * SOL_BUCKET / ETH_BUCKET)
    assert [r[0] for r in prof] == [120.0, 120.3], prof


def test_eth_unchanged():
    assert m.hl_liq_levels([(10, 2650.0), (-5, 2752.4)], 2700.0) == {
        "below": [{"px": 2650.0, "usd": 27000, "n": 1}], "above": [{"px": 2750.0, "usd": 13500, "n": 1}]}
    assert m.liq_profile([(2683.7, 100, True)]) == [[2682.5, 100, 0]]
