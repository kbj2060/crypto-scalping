from __future__ import annotations

import os
from dataclasses import dataclass


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class BinanceAccountConfig:
    enabled: bool
    position_sync_enabled: bool
    testnet: bool
    symbol: str

    @classmethod
    def from_env(cls) -> "BinanceAccountConfig":
        enabled = _env_flag("BINANCE_ACCOUNT_ENABLED", False)
        return cls(
            enabled=enabled,
            position_sync_enabled=_env_flag("BINANCE_POSITION_SYNC_ENABLED", enabled),
            testnet=_env_flag("BINANCE_ACCOUNT_TESTNET", _env_flag("BINANCE_TESTNET", False)),
            symbol=os.getenv("BINANCE_ACCOUNT_SYMBOL", ""),
        )


@dataclass(frozen=True)
class BinanceExecutionConfig:
    enabled: bool
    dry_run: bool
    require_testnet: bool
    confirm_live: str
    symbol: str
    audit_path: str
    margin_mode: str
    set_leverage: bool
    max_exchange_leverage: int
    min_notional_usdt: float
    max_target_notional_usdt: float
    rebalance_tolerance_usdt: float
    alpha14_router_enabled: bool
    maker_reduce_only_enabled: bool
    maker_entry_fallback_market: bool
    maker_exit_fallback_market: bool
    maker_wait_sec: float
    maker_book_depth: int
    maker_max_spread_bps: float
    maker_min_imbalance: float
    maker_min_microprice_edge_bps: float
    maker_entry_offset_bps: float
    maker_exit_offset_bps: float
    resting_tpsl_enabled: bool
    promotion_manifest_required: bool
    promotion_manifest_path: str

    @classmethod
    def from_env(cls) -> "BinanceExecutionConfig":
        return cls(
            enabled=_env_flag("BINANCE_EXECUTION_ENABLED", False),
            dry_run=_env_flag("BINANCE_EXECUTION_DRY_RUN", True),
            require_testnet=_env_flag("BINANCE_EXECUTION_REQUIRE_TESTNET", True),
            confirm_live=os.getenv("BINANCE_EXECUTION_CONFIRM_LIVE", ""),
            symbol=os.getenv("BINANCE_EXECUTION_SYMBOL", ""),
            audit_path=os.getenv("BINANCE_EXECUTION_AUDIT_PATH", "data/live/binance_execution_audit.jsonl"),
            margin_mode=os.getenv("BINANCE_EXECUTION_MARGIN_MODE", "isolated").strip().lower(),
            set_leverage=_env_flag("BINANCE_EXECUTION_SET_LEVERAGE", True),
            max_exchange_leverage=int(float(os.getenv("BINANCE_EXECUTION_MAX_EXCHANGE_LEVERAGE", "5"))),
            min_notional_usdt=float(os.getenv("BINANCE_EXECUTION_MIN_NOTIONAL_USDT", "5.0")),
            max_target_notional_usdt=float(os.getenv("BINANCE_EXECUTION_MAX_TARGET_NOTIONAL_USDT", "0.0")),
            rebalance_tolerance_usdt=float(os.getenv("BINANCE_EXECUTION_REBALANCE_TOLERANCE_USDT", "10.0")),
            alpha14_router_enabled=_env_flag("BINANCE_EXECUTION_ALPHA14_ROUTER_ENABLE", True),
            maker_reduce_only_enabled=_env_flag("BINANCE_EXECUTION_MAKER_REDUCE_ONLY_ENABLE", True),
            maker_entry_fallback_market=_env_flag("BINANCE_EXECUTION_MAKER_ENTRY_FALLBACK_MARKET", False),
            maker_exit_fallback_market=_env_flag("BINANCE_EXECUTION_MAKER_EXIT_FALLBACK_MARKET", True),
            maker_wait_sec=float(os.getenv("BINANCE_EXECUTION_MAKER_WAIT_SEC", "2.0")),
            maker_book_depth=int(float(os.getenv("BINANCE_EXECUTION_MAKER_BOOK_DEPTH", "20"))),
            maker_max_spread_bps=float(os.getenv("BINANCE_EXECUTION_MAKER_MAX_SPREAD_BPS", "4.0")),
            maker_min_imbalance=float(os.getenv("BINANCE_EXECUTION_MAKER_MIN_IMBALANCE", "0.05")),
            maker_min_microprice_edge_bps=float(os.getenv("BINANCE_EXECUTION_MAKER_MIN_MICROPRICE_EDGE_BPS", "0.0")),
            maker_entry_offset_bps=float(os.getenv("BINANCE_EXECUTION_MAKER_ENTRY_OFFSET_BPS", "0.0")),
            maker_exit_offset_bps=float(os.getenv("BINANCE_EXECUTION_MAKER_EXIT_OFFSET_BPS", "0.0")),
            resting_tpsl_enabled=_env_flag("BINANCE_EXECUTION_RESTING_TPSL_ENABLE", True),
            promotion_manifest_required=_env_flag(
                "FINAL_GOVERNOR_OMEGA4_6_1_ENABLE", False
            ),
            promotion_manifest_path=os.getenv(
                "FINAL_GOVERNOR_OMEGA4_6_1_MANIFEST_PATH",
                "docs/model_contracts/CURRENT_LIVE_MANIFEST.json",
            ),
        )
