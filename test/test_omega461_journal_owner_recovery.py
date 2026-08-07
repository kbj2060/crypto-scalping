from types import SimpleNamespace

import trading_bot


def test_omega461_open_journal_is_not_recovered_as_lifecycle(monkeypatch) -> None:
    trade_id = "trade-omega461-open"
    monkeypatch.setattr(
        trading_bot,
        "_load_trade_journal_rows",
        lambda _path: [
            {
                "trade_id": trade_id,
                "kind": "OPEN",
                "source": f"{trading_bot.OMEGA4_6_1_OWNER}|zig075",
                "model_id": trading_bot.OMEGA4_6_1_MODEL_ID,
                "take_profit": 0.075,
                "stop_loss": 0.04,
            }
        ],
    )
    runtime = SimpleNamespace(
        active_lifecycle_v1_effective_notional=0.0,
        active_lifecycle_v1_take_profit=0.0,
        active_lifecycle_v1_stop_loss=0.0,
        active_lifecycle_v1_max_hold_bars=0,
    )
    router = SimpleNamespace(pos="SHORT", open_trade_id=trade_id)

    recovered = trading_bot.FinalGovernorRuntime._recover_lifecycle_v1_state_from_open_journal(
        runtime, router, "unknown"
    )

    assert recovered is False
