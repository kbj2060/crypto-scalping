"""들어간 포지션 손절선을 지금가 반대쪽에 놓으면 «올라/내려 닿으면» 손절(2026-11-10). 실행: python -m pytest -q test/test_line_sl_watch_20261110.py"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.live_manual_peg_entry_20260912 import bracket_action, bracket_sl_hit, line_bracket, line_hit  # noqa: E402

F = {"tick": 0.01}


def lb(side, tp, sl, watch=True):
    return line_bracket(position_side=side, ref_level=100.0, tp_level=tp, sl_level=sl, basis=1.0, filters=F, sl_watch=watch)


def test_long_sl_above_price_fires_only_when_price_rises_to_it() -> None:
    b = lb("LONG", 103.0, 101.0)
    assert b["sl_up"] is True
    assert line_hit("LONG", b, (99.0, 100.9)) is None          # 아래로 빠져도 안 쏜다(옛 판이면 바로 손절)
    assert line_hit("LONG", b, (100.2, 101.0)) == "sl"         # 올라 닿으면
    armed = {**b, "armed_at": 0.0}
    assert bracket_sl_hit("LONG", (10.0, 101.2), armed) and not bracket_sl_hit("LONG", (10.0, 99.0), armed)


def test_short_sl_below_price_fires_when_price_falls_to_it() -> None:
    b = lb("SHORT", 97.0, 99.0)
    assert b["sl_up"] is False
    assert line_hit("SHORT", b, (99.1, 101.0)) is None
    assert line_hit("SHORT", b, (99.0, 99.8)) == "sl"


def test_normal_side_and_defaults_unchanged() -> None:
    assert lb("LONG", 103.0, 98.0)["sl_up"] is False and lb("SHORT", 97.0, 102.0)["sl_up"] is True
    assert line_hit("LONG", {"sl_level": 98.0, "tp_level": 103.0, "tp_resting": True}, (97.9, 100.0)) == "sl"   # 옛 무장(sl_up 없음)
    assert line_hit("SHORT", {"sl_level": 102.0, "sl_up": None, "tp_resting": True}, (99.0, 102.0)) == "sl"    # None = 측면 기본
    with pytest.raises(ValueError):   # 진입(sl_watch 아님)은 예전처럼 손절이 손실 쪽이어야 한다
        lb("LONG", 103.0, 101.0, watch=False)
    with pytest.raises(ValueError):   # 익절은 여전히 수익 쪽
        lb("LONG", 99.0, 101.0)


def test_bracket_action_fires_on_flipped_touch() -> None:
    armed = {"symbol": "ETHUSDC", "armed_at": 0.0, "lines": True, **lb("LONG", 103.0, 101.0)}
    pos = [{"symbol": "ETHUSDC", "side": "LONG", "qty": 1}]
    assert bracket_action(armed, "LONG", pos, True, 10.0, None, (99.0, 100.5)) == "hold"
    assert bracket_action(armed, "LONG", pos, True, 10.0, None, (100.0, 101.0)) == "fire"
