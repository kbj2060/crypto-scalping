from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_watchdog_uses_explicit_cycle_heartbeat() -> None:
    runtime_config = (ROOT / "trading_bot_modules/runtime_config.py").read_text()
    trading_bot = (ROOT / "trading_bot.py").read_text()
    supervisor = (ROOT / "scripts/supervise_trading_bot.sh").read_text()

    assert "live.trading_bot_decision_heartbeat.v1" in trading_bot
    assert "DATA_PIPELINE_DECISION_HEARTBEAT_PATH" in runtime_config
    assert "data/live/trading_bot_decision_heartbeat.json" in supervisor


def test_skipped_next_open_cycle_still_records_orderbook() -> None:
    source = (ROOT / "trading_bot.py").read_text()
    heartbeat = source.index('"status": "cycle_input_ready"')
    skip = source.index('"record_reason": "next_open_execution_skipped"')
    skipped_return = source.index("return", skip)

    assert heartbeat < skip < skipped_return
