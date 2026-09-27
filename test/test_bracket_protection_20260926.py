"""브래킷 보호 사고 (2026-09-26).   python -m pytest -q test/test_bracket_protection_20260926.py

사고: IP 밴 중 잔고는 읽고 포지션 조회만 실패 → fetch_account 가 ok=True·positions=[] → 브래킷 감시가 «포지션이
사라졌다»로 읽고 **열린 ETH 숏의 익절·비상 스탑을 스스로 지웠다**. 청산 계획도 같은 입력을 no_position 으로 읽었다.
① 포지션 조회 실패는 계좌 실패다(«없음»이 아니다). ② 무장 상태 키는 «심볼:측면»(코인이 늘면 측면만으론 덮인다).
③ 방향: 롱 = 익절 저항2·손절 지지1, 숏 = 익절 지지2·손절 저항1 (step=0). 2026-09-27 부터 세 코인 모두 step=1.
④ SL 판정 = 무장 뒤 마감된 5분봉 종가가 레벨을 넘었나(호가 터치 아님, 2026-09-27).
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.live_binance_account_20260910 as acct  # noqa: E402
from scripts.live_manual_peg_entry_20260912 import (  # noqa: E402
    bracket_action, bracket_key, bracket_rekey, build_bracket_plan)


BALANCE = {"assets": [], "multiAssetsMargin": False, "totalWalletBalance": "100", "totalUnrealizedProfit": "0",
           "totalMarginBalance": "100", "availableBalance": "100", "totalInitialMargin": "0", "totalMaintMargin": "0"}


def test_position_lookup_failure_is_account_failure(monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY", "k")
    monkeypatch.setenv("BINANCE_SECRET_KEY", "s")

    async def fake_offset(_s):
        return 0

    async def fake_get(_s, path, *_a):
        if path == "/fapi/v2/account":
            return BALANCE
        return {"__error__": "418 [-1003] Way too many requests; IP banned until 1790435099030"}
    monkeypatch.setattr(acct, "_clock_offset", fake_offset)
    monkeypatch.setattr(acct, "_get", fake_get)
    out = asyncio.run(acct.fetch_account(None, ["ETHUSDC"]))
    assert out["ok"] is False and "positionRisk" in out["error"], out
    # 그 결과로 감시가 «정리»(익절·비상 스탑 삭제)로 가면 안 된다 -- ok=False 면 hold
    armed = {"symbol": "ETHUSDC", "armed_at": 0.0, "sl_price": 2702.76}
    assert bracket_action(armed, "SHORT", out.get("positions") or [], bool(out.get("ok")),
                          10_000.0, (9_999.0, 2688.0)) == "hold"


def test_state_key_is_symbol_and_side_and_legacy_keys_migrate():
    assert bracket_key("SOLUSDC", "LONG") == "SOLUSDC:LONG"
    legacy = {"SHORT": {"symbol": "ETHUSDC", "sl_price": 2702.76, "armed_at": 1.0},
              "SOLUSDC:LONG": {"symbol": "SOLUSDC", "side": "LONG", "armed_at": 2.0}}
    got = bracket_rekey(legacy)
    assert set(got) == {"ETHUSDC:SHORT", "SOLUSDC:LONG"}, got
    assert got["ETHUSDC:SHORT"]["side"] == "SHORT" and got["ETHUSDC:SHORT"]["sl_price"] == 2702.76
    both = {bracket_key("ETHUSDC", "LONG"): {}, bracket_key("SOLUSDC", "LONG"): {}}
    assert len(both) == 2, "ETH 롱과 SOL 롱이 한 키로 덮였다"


def test_long_tp_resistance2_sl_support1_short_mirrored():
    sup, res = [2670.45, 2667.76, 2597.77], [2702.76, 2705.45, 2708.14]
    f = {"tick": 0.01}
    lg = build_bracket_plan(position_side="LONG", support_levels=sup, resistance_levels=res,
                            ref_price=2688.0, basis=1.0, filters=f)
    assert (lg["tp_name"], lg["tp_level"], lg["sl_name"], lg["sl_level"]) == ("저항2", 2705.45, "지지1", 2670.45)
    assert lg["backstop_price"] < lg["sl_price"] < 2688.0 < lg["tp_price"], lg
    sh = build_bracket_plan(position_side="SHORT", support_levels=sup, resistance_levels=res,
                            ref_price=2688.0, basis=1.0, filters=f)
    assert (sh["tp_name"], sh["tp_level"], sh["sl_name"], sh["sl_level"]) == ("지지2", 2667.76, "저항1", 2702.76)
    assert sh["tp_price"] < 2688.0 < sh["sl_price"] < sh["backstop_price"], sh


def test_step1_moves_both_legs_one_level_further():
    """step=1(2026-09-27 부터 ETH·SOL·XRP 전부): 롱 = 익절 저항3·손절 지지2, 숏 = 익절 지지3·손절 저항2."""
    sup, res = [2670.45, 2667.76, 2597.77], [2702.76, 2705.45, 2708.14]
    kw = dict(support_levels=sup, resistance_levels=res, ref_price=2688.0, basis=1.0, filters={"tick": 0.01}, step=1)
    lg = build_bracket_plan(position_side="LONG", **kw)
    assert (lg["tp_name"], lg["tp_level"], lg["sl_name"], lg["sl_level"]) == ("저항3", 2708.14, "지지2", 2667.76)
    sh = build_bracket_plan(position_side="SHORT", **kw)
    assert (sh["tp_name"], sh["tp_level"], sh["sl_name"], sh["sl_level"]) == ("지지3", 2597.77, "저항2", 2705.45)
    short2 = build_bracket_plan(position_side="LONG", **{**kw, "resistance_levels": res[:2]})
    assert short2["tp_price"] is None and short2["sl_level"] == 2667.76, "레벨이 모자라면 그 다리만 비운다"


def test_sl_fires_on_closed_5m_bar_close_not_wick():
    """꼬리만 넘은 봉은 hold, 종가가 넘은 봉만 fire. 무장 전에 마감된 봉은 무시. sl_level(USDT) 우선."""
    armed = {"symbol": "ETHUSDC", "armed_at": 1_000.0, "sl_price": 2701.5, "sl_level": 2702.76}
    pos = [{"symbol": "ETHUSDC", "side": "SHORT", "qty": -1.0}]
    act = lambda bar, side="SHORT", a=armed: bracket_action(a, side, pos if side == "SHORT" else
                                                             [{**pos[0], "side": side}], True, 2_000.0, bar)
    assert act((1_300.0, 2702.0)) == "hold", "종가가 레벨 아래 -- 꼬리가 넘었어도 안 나간다"
    assert act((1_300.0, 2702.76)) == "fire"
    assert act((1_300.0, 2702.0 + 1.0)) == "fire"
    assert act((900.0, 2800.0)) == "hold", "무장 전에 마감된 봉"
    assert act(None) == "hold", "봉을 못 읽으면 hold"
    lg = {"symbol": "ETHUSDC", "armed_at": 1_000.0, "sl_price": 2669.0, "sl_level": 2670.45}
    assert act((1_300.0, 2670.46), "LONG", lg) == "hold" and act((1_300.0, 2670.0), "LONG", lg) == "fire"
    legacy = {"symbol": "ETHUSDC", "armed_at": 1_000.0, "sl_price": 2701.5}   # sl_level 없는 옛 무장
    assert act((1_300.0, 2701.6), "SHORT", legacy) == "fire" and act((1_300.0, 2701.4), "SHORT", legacy) == "hold"
