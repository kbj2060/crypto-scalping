from dataclasses import replace

from trading_bot_modules.binance_execution import BinanceFuturesExecutionAdapter
from trading_bot_modules.binance_runtime_config import BinanceAccountConfig, BinanceExecutionConfig


class _Fetcher:
    account_exchange = object()
    account_enabled = True
    account_testnet = False
    account_symbol = "ETH/USDT:USDT"


def test_account_config_reads_position_sync_default_from_account_flag(monkeypatch):
    monkeypatch.setenv("BINANCE_ACCOUNT_ENABLED", "true")
    monkeypatch.delenv("BINANCE_POSITION_SYNC_ENABLED", raising=False)

    config = BinanceAccountConfig.from_env()

    assert config.enabled is True
    assert config.position_sync_enabled is True


def test_mainnet_execution_fails_closed_without_confirmation(monkeypatch):
    monkeypatch.setenv("BINANCE_EXECUTION_ENABLED", "true")
    monkeypatch.setenv("BINANCE_EXECUTION_DRY_RUN", "false")
    monkeypatch.setenv("BINANCE_EXECUTION_REQUIRE_TESTNET", "false")
    monkeypatch.delenv("BINANCE_EXECUTION_CONFIRM_LIVE", raising=False)

    adapter = BinanceFuturesExecutionAdapter(_Fetcher())

    assert adapter.enabled is False


def test_explicit_mainnet_confirmation_keeps_execution_enabled(monkeypatch):
    monkeypatch.setenv("BINANCE_EXECUTION_ENABLED", "true")
    monkeypatch.setenv("BINANCE_EXECUTION_DRY_RUN", "false")
    monkeypatch.setenv("BINANCE_EXECUTION_REQUIRE_TESTNET", "false")
    monkeypatch.setenv("BINANCE_EXECUTION_CONFIRM_LIVE", "I_UNDERSTAND_REAL_ORDERS")

    adapter = BinanceFuturesExecutionAdapter(_Fetcher())

    assert adapter.enabled is True


def test_adapter_uses_injected_config_instead_of_process_environment(monkeypatch):
    monkeypatch.setenv("BINANCE_EXECUTION_ENABLED", "true")
    config = replace(BinanceExecutionConfig.from_env(), enabled=False)

    adapter = BinanceFuturesExecutionAdapter(_Fetcher(), config=config)

    assert adapter.enabled is False
