import pytest

from trading_bot_modules.microstructure_operating_controller import (
    ControllerConfig,
    decide_entry,
    decide_exit,
    fragility_score,
    margin_for_entry,
    opportunity_score,
)


def test_opportunity_is_movement_magnitude_not_direction() -> None:
    quiet = opportunity_score(
        trade_notional=0.1, whale_activity=0.1, energy=0.1, queue_collapse=0.1
    )
    active = opportunity_score(
        trade_notional=0.9, whale_activity=0.9, energy=0.9, queue_collapse=0.9
    )
    assert active > quiet
    assert active == pytest.approx(0.9)


def test_fragility_increases_with_adverse_inputs() -> None:
    normal = fragility_score(
        model_risk=0.1, toxicity=0.1, queue_collapse=0.1, aftershock=0.0, spoofing=0.0
    )
    fragile = fragility_score(
        model_risk=0.9, toxicity=0.9, queue_collapse=0.9, aftershock=1.0, spoofing=1.0
    )
    assert fragile > normal


def test_entry_never_invents_a_direction_and_blocks_low_opportunity() -> None:
    blocked = decide_entry(side=-1, opportunity=0.2, risk=0.1, alignment=0.0)
    assert not blocked.allow
    assert blocked.notional == 0.0
    assert blocked.reason == "INSUFFICIENT_MOVEMENT"


def test_high_fragility_blocks_even_when_opportunity_is_high() -> None:
    blocked = decide_entry(side=1, opportunity=0.9, risk=0.9, alignment=0.8)
    assert not blocked.allow
    assert blocked.reason == "FRAGILITY_BLOCK"


def test_margin_contract_uses_fixed_leverage_once() -> None:
    config = ControllerConfig(base_margin_fraction=0.30, leverage=3.0)
    margin, notional = margin_for_entry(0.9, 0.1, config)
    assert config.min_margin_fraction <= margin <= config.base_margin_fraction
    assert notional == pytest.approx(margin * 3.0)


def test_execution_mode_and_exit_urgency_are_separate() -> None:
    entry = decide_entry(side=1, opportunity=1.0, risk=0.2, alignment=1.0)
    assert entry.allow
    assert entry.execution_mode == "MARKETABLE_LIMIT"
    urgent = decide_exit(opportunity=0.9, risk=0.9, alignment=0.8)
    assert urgent.action == "URGENT_EXIT"
    passive = decide_exit(opportunity=0.1, risk=0.1, alignment=-0.8)
    assert passive.action == "PASSIVE_EXIT"


def test_invalid_normalized_inputs_fail_fast() -> None:
    with pytest.raises(ValueError):
        opportunity_score(
            trade_notional=1.1, whale_activity=0.0, energy=0.0, queue_collapse=0.0
        )


def test_evaluation_runner_has_no_execution_capability() -> None:
    source = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "scripts/evaluate_microstructure_operating_controller_20260719.py"
    ).read_text()
    assert '"order_submission_supported": False' in source
    assert '"activation_allowed": False' in source
    assert "create_order" not in source
    assert "place_order" not in source
    assert "binance_execution" not in source
