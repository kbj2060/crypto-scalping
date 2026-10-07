"""4분할 규칙 진입 계획(build_rule_ladder) -- 순수 함수.   python -m pytest -q test/test_rule_ladder_20261008.py
네트워크·주문 없음."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.live_manual_peg_entry_20260912 import RULE_UNIT_X, build_rule_ladder  # noqa: E402

F = {"tick": 0.01, "step": 0.001, "min_qty": 0.001, "min_notional": 20.0}


def plan(side="LONG", equity=2641.58, price=2722.36, sigma=153.66, mult=0.52, cap=None):
    return build_rule_ladder(side=side, equity=equity, price=price, sigma24_bp=sigma, mult=mult, cap_notional=cap, filters=F)


def test_reproduces_research_trip_20261006():
    """10-07 연구의 그 왕복: 지갑 $2,642 · σ 154bp · m 0.52 → 1분할 지갑의 1.62배, 손절 2,596.87, 손실 ≈ 26%."""
    p = plan()
    assert p["available"] and p["binding"] == "rule", p
    assert [t["offset_bp"] for t in p["tranches"]] == [0, 44, 88, 132]
    assert abs(p["tranches"][0]["qty"] * p["tranches"][0]["price"] / 2641.58 - RULE_UNIT_X * 0.52) < 0.01
    assert p["tranches"][1]["price"] == 2710.38 and p["tranches"][3]["price"] == 2686.42
    assert p["stop_price"] == 2596.86                                      # 연구 2596.87 을 틱 아래로(먼 쪽)
    assert 24 < p["loss_pct"] < 28 and p["liq_price"] < p["stop_price"] * 0.9


def test_cap_binds_and_liq_guard_binds():
    hi = plan(mult=1.5, cap=None)                                          # m 1.5 · 같은 σ → 청산이 손절보다 먼저 → 축소
    assert hi["binding"] == "liq" and hi["total_x"] < hi["rule_x"], hi
    assert hi["liq_price"] < hi["stop_price"] * (1 - 0.004), hi           # 축소 뒤엔 손절이 먼저(여유 포함)
    capped = plan(mult=0.52, cap=2641.58 * 4)
    assert capped["binding"] == "cap" and capped["total_x"] <= 4.0 + 1e-9, capped


@pytest.mark.parametrize("mult,sigma", [(0.25, 80), (0.52, 154), (1.0, 295), (2.0, 600), (3.9, 1200)])
def test_liquidation_always_beyond_stop(mult, sigma):
    for side in ("LONG", "SHORT"):
        p = plan(side=side, mult=mult, sigma=sigma)
        s = 1 if side == "LONG" else -1
        assert p["available"], p
        assert s * (p["stop_price"] - p["liq_price"]) > 0, p               # 롱: 손절 > 청산 · 숏: 손절 < 청산
        assert p["loss_pct"] < 100


def test_short_mirrors_long():
    lg, sh = plan(side="LONG"), plan(side="SHORT")
    assert sh["tranches"][1]["price"] > sh["tranches"][0]["price"] > lg["tranches"][1]["price"]
    assert abs((sh["stop_price"] - 2722.36) - (2722.36 - lg["stop_price"])) < 0.05
    assert abs(sh["total_x"] - lg["total_x"]) < 0.02 and abs(sh["loss_pct"] - lg["loss_pct"]) < 0.5


def test_blocks_dust_and_missing_inputs():
    assert plan(equity=5.0)["blocked"]                                      # 1분할이 최소 명목 미만
    assert plan(sigma=0)["available"] is False and plan(mult=0)["available"] is False
    with pytest.raises(ValueError):
        plan(side="BUY")
