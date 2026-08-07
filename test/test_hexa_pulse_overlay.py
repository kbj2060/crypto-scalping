import pytest

from trading_bot_modules.hexa_pulse_overlay import decide_overlay


def test_overlay_delays_hostile_parent_entry_without_reversing() -> None:
    decision = decide_overlay(
        parent_position=1,
        overlay_position=0,
        score=-0.20,
        toxicity=0.1,
        tail_risk=0.0,
        available=True,
    )
    assert decision.position == 0
    assert decision.action == "DELAY"


@pytest.mark.parametrize("parent", [-1, 1])
def test_overlay_allows_non_hostile_parent_entry(parent: int) -> None:
    decision = decide_overlay(
        parent_position=parent,
        overlay_position=0,
        score=0.0,
        toxicity=0.1,
        tail_risk=0.0,
        available=True,
    )
    assert decision.position == parent
    assert decision.action == "ENTER"


def test_overlay_exits_on_opposition() -> None:
    decision = decide_overlay(
        parent_position=-1,
        overlay_position=-1,
        score=0.16,
        toxicity=0.1,
        tail_risk=0.0,
        available=True,
    )
    assert decision.position == 0
    assert decision.reason == "HEXA_OPPOSITION_EXIT"


def test_overlay_risk_block_exits_but_never_reverses() -> None:
    decision = decide_overlay(
        parent_position=1,
        overlay_position=1,
        score=0.9,
        toxicity=0.81,
        tail_risk=0.0,
        available=True,
    )
    assert decision.position == 0
    assert decision.reason == "TOXICITY_BLOCK"


def test_parent_cash_has_final_authority() -> None:
    decision = decide_overlay(
        parent_position=0,
        overlay_position=-1,
        score=-0.9,
        toxicity=0.0,
        tail_risk=0.0,
        available=True,
    )
    assert decision.position == 0
    assert decision.reason == "PARENT_CASH"
