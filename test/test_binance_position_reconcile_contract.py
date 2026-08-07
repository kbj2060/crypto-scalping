from __future__ import annotations

from trading_bot_modules.position_sync import (
    classify_account_position_snapshot,
    exchange_position_went_flat,
)


def test_unavailable_position_query_is_not_flat() -> None:
    state, position = classify_account_position_snapshot(
        {"position_query_ok": False, "position": None}
    )
    assert state == "unavailable"
    assert position is None
    assert not exchange_position_went_flat(state, "LONG")


def test_successful_empty_position_query_is_flat() -> None:
    state, position = classify_account_position_snapshot(
        {"position_query_ok": True, "position": None}
    )
    assert state == "flat"
    assert position is None
    assert exchange_position_went_flat(state, "LONG")


def test_successful_open_position_query_preserves_position() -> None:
    expected = {"type": "SHORT", "entry_price": 1900.0}
    state, position = classify_account_position_snapshot(
        {"position_query_ok": True, "position": expected}
    )
    assert state == "open"
    assert position == expected
    assert not exchange_position_went_flat(state, "SHORT")
