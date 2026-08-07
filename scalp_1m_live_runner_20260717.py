"""Standalone live runner for the ETH 1m scalping strategy, on a FULLY SEPARATE Binance account
from the existing Omega4.6.1 (5m) swing bot -- follows the same "independent process" pattern as
run_live_collectors.py, not trading_bot.py's main loop. Nothing in trading_bot.py is touched or
depends on this file; the only shared code is an additive, backward-compatible extension to
BinanceLiveFetcher (optional account_api_key/account_api_secret params, unused unless passed).

Sizing (per project decision, 2026-07-17): PER_TRADE_PCT=0.50, MAX_TOTAL_EXPOSURE_PCT=0.50 (env-
overridable via SCALP_PER_TRADE_PCT / SCALP_MAX_TOTAL_EXPOSURE_PCT) -- effectively single-position
sizing on this dedicated sub-account. NOTE: backtested/walk-forward-validated only up to 20% in
scripts/simulate_exposure_capped_scalp_1m_20260717.py; 50% was checked once more on the same clean
7-fold data (see that script's CONFIGS) but is a materially more aggressive choice than what this
session's own analysis concluded was "credible" (1%/5%) -- proceeding per explicit user direction,
not this assistant's recommendation.

STATUS / WHAT STILL NEEDS HARDENING BEFORE REAL CAPITAL (read before flipping SCALP_ACCOUNT_ENABLED
or SCALP_EXECUTION_ENABLED to True):
  - Live feature computation (FeatureEngineer.process on a rolling buffer) mirrors
    build_features_1m_20260716.py's causal logic but has NOT been live-tested against a real
    streaming feed -- verify a handful of live-computed feature rows against a batch rebuild
    before trusting it (same method as verify_scalp_1m_no_lookahead_20260717.py).
  - Position/order reconciliation on restart (what happens if this process crashes mid-position)
    is NOT implemented -- trading_bot.py has extensive reconciliation logic
    (position_sync.py/state_transition_gate.py) that this script does not reuse.
  - TP/SL are placed once at entry (via place_tp_sl_orders) using the model's own tp_move/sl_move
    output; the horizon-based forced-exit-at-20min-if-neither-hit (part of the backtested label
    logic) is NOT implemented here yet -- currently a position with no TP/SL fill just rests
    until one triggers, which does not match the backtest's HORIZON=20min cutoff behavior.
  - No dry-run/paper-trading validation period has been run yet.
Treat this as a first-pass implementation, not a finished, live-ready system.

Usage: python scalp_1m_live_runner_20260717.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import pickle
import signal
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from features.engineering import FeatureEngineer
from features.schema import prune_to_active_feature_keep
from build_scalp_1m_tb_labels_20260716 import ATR_LOOKBACK, TP_ATR_MULT, SL_ATR_MULT, TP_BOUNDS, SL_BOUNDS
from trading_bot_modules.binance_live_fetcher import BinanceLiveFetcher
from trading_bot_modules.binance_execution import BinanceFuturesExecutionAdapter
from trading_bot_modules.binance_runtime_config import BinanceAccountConfig, BinanceExecutionConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ScalpLiveRunner")

ROOT = _ROOT
MODEL_ID = "scalp_1m_eth_live_v1_20260717"
MODEL_PATH = os.path.join(ROOT, "data", "ensemble", "ckpt", f"{MODEL_ID}.pkl")

CLASS_ORDER = ["CASH", "LONG", "SHORT"]
BUFFER_MIN_BARS = 400  # >> the largest rolling window used by FeatureEngineer (288 bars) + margin
POLL_INTERVAL_SEC = 60


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass
class ScalpConfig:
    account_enabled: bool
    account_testnet: bool
    symbol: str
    execution_enabled: bool
    execution_dry_run: bool
    per_trade_pct: float
    max_total_exposure_pct: float
    confidence_threshold: float
    api_key: str
    api_secret: str

    @classmethod
    def from_env(cls) -> "ScalpConfig":
        return cls(
            account_enabled=_env_flag("SCALP_ACCOUNT_ENABLED", False),
            account_testnet=_env_flag("SCALP_ACCOUNT_TESTNET", True),
            symbol=os.getenv("SCALP_ACCOUNT_SYMBOL", "ETH/USDT:USDT"),
            execution_enabled=_env_flag("SCALP_EXECUTION_ENABLED", False),
            execution_dry_run=_env_flag("SCALP_EXECUTION_DRY_RUN", True),
            per_trade_pct=_env_float("SCALP_PER_TRADE_PCT", 0.05),
            max_total_exposure_pct=_env_float("SCALP_MAX_TOTAL_EXPOSURE_PCT", 0.05),
            confidence_threshold=_env_float("SCALP_CONFIDENCE_THRESHOLD", 0.55),
            api_key=os.getenv("SCALP_ACCOUNT_API_KEY", "").strip(),
            api_secret=os.getenv("SCALP_ACCOUNT_SECRET_KEY", "").strip(),
        )


class OpenPosition:
    __slots__ = ("side", "entry_price", "tp_price", "sl_price", "notional_pct", "opened_at")

    def __init__(self, side: str, entry_price: float, tp_price: float, sl_price: float, notional_pct: float, opened_at: float):
        self.side = side
        self.entry_price = entry_price
        self.tp_price = tp_price
        self.sl_price = sl_price
        self.notional_pct = notional_pct
        self.opened_at = opened_at


class ScalpLiveRunner:
    def __init__(self, cfg: ScalpConfig):
        self.cfg = cfg
        self.model = None
        self.feat_cols: list[str] = []
        self.open_positions: list[OpenPosition] = []
        self._stop = asyncio.Event()

        account_config = BinanceAccountConfig(
            enabled=cfg.account_enabled,
            position_sync_enabled=cfg.account_enabled,
            testnet=cfg.account_testnet,
            symbol=cfg.symbol,
        )
        self.fetcher = BinanceLiveFetcher(
            symbol="ETHUSDT",
            timeframe="1m",
            account_symbol=cfg.symbol,
            account_config=account_config,
            account_api_key=cfg.api_key or None,
            account_api_secret=cfg.api_secret or None,
        )
        exec_config = BinanceExecutionConfig(
            enabled=cfg.execution_enabled,
            dry_run=cfg.execution_dry_run,
            require_testnet=True,
            confirm_live="",
            symbol=cfg.symbol,
            audit_path=os.path.join(ROOT, "data", "live", "scalp_1m_execution_audit.jsonl"),
            margin_mode="isolated",
            set_leverage=True,
            max_exchange_leverage=3,
            min_notional_usdt=5.0,
            max_target_notional_usdt=0.0,
            rebalance_tolerance_usdt=1.0,
            alpha14_router_enabled=False,
            maker_reduce_only_enabled=True,
            maker_entry_fallback_market=False,
            maker_exit_fallback_market=True,
            maker_wait_sec=180.0,  # matches the backtest's FILL_LOOKAHEAD=3min
            maker_book_depth=20,
            maker_max_spread_bps=4.0,
            maker_min_imbalance=0.0,
            maker_min_microprice_edge_bps=0.0,
            maker_entry_offset_bps=1.0,  # matches the backtest's OFFSET=1bp passive limit
            maker_exit_offset_bps=0.0,
            resting_tpsl_enabled=True,
        )
        self.executor = BinanceFuturesExecutionAdapter(self.fetcher, config=exec_config)

    def load_model(self):
        with open(MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
        self.model = bundle["model"]
        self.feat_cols = bundle["feature_cols"]
        logger.info(f"Loaded model {MODEL_ID}: {len(self.feat_cols)} features")

    async def fetch_recent_candles(self) -> pd.DataFrame:
        """Fetches enough recent 1m ETH candles for the feature pipeline's rolling windows."""
        ohlcv = await self.fetcher.exchange.fetch_ohlcv(self.fetcher.symbol.replace("USDT", "/USDT"),
                                                          timeframe="1m", limit=BUFFER_MIN_BARS)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["quote_volume"] = df["volume"] * df["close"]
        df["trades"] = 0
        df["taker_buy_base"] = 0.0
        df["taker_buy_quote"] = 0.0
        return df

    async def fetch_recent_btc_candles(self) -> pd.DataFrame:
        """5m BTC, matching the backtest's cross-asset input choice (BTC 1m history was skipped as
        unnecessary -- see build_features_1m_20260716.py)."""
        ohlcv = await self.fetcher.exchange.fetch_ohlcv("BTC/USDT", timeframe="5m", limit=200)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["quote_volume"] = df["volume"] * df["close"]
        return df[["timestamp", "close", "volume", "quote_volume"]]

    def compute_latest_features(self, eth_df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.Series | None:
        engineer = FeatureEngineer(candle_minutes=1, keep_only_active=True, include_entry_price=False)
        result = engineer.process(eth_df.copy(), btc_df.copy())
        result = prune_to_active_feature_keep(result, include_entry_price=False, include_m7_artifacts=True,
                                               extra_keep=["timestamp", "close"])
        if result.empty:
            return None
        return result.iloc[-1]

    @staticmethod
    def compute_atr_pct(eth_df: pd.DataFrame, lookback: int = ATR_LOOKBACK) -> float:
        """Exact same formula as build_scalp_1m_tb_labels_20260716.py::_atr_pct, evaluated at the
        latest bar -- keeps live TP/SL sizing consistent with what was backtested."""
        prev_close = eth_df["close"].shift(1)
        tr = pd.concat([
            eth_df["high"] - eth_df["low"],
            (eth_df["high"] - prev_close).abs(),
            (eth_df["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(lookback, min_periods=lookback).mean()
        atr_pct = (atr / eth_df["close"]).iloc[-1]
        return float(atr_pct) if np.isfinite(atr_pct) else 0.0

    def decide(self, row: pd.Series, atr_pct: float) -> tuple[str, float, float, float]:
        """Returns (side, confidence, tp_move, sl_move). side is one of CLASS_ORDER."""
        X = row[self.feat_cols].fillna(0.0).to_numpy(dtype=np.float32).reshape(1, -1)
        proba = self.model.predict_proba(X)[0]
        classes = self.model.classes_
        max_idx = int(np.argmax(proba))
        side = classes[max_idx] if proba[max_idx] >= self.cfg.confidence_threshold else "CASH"
        tp_move = float(np.clip(atr_pct * TP_ATR_MULT, *TP_BOUNDS))
        sl_move = float(np.clip(atr_pct * SL_ATR_MULT, *SL_BOUNDS))
        return str(side), float(proba[max_idx]), tp_move, sl_move

    def current_exposure(self) -> float:
        return sum(p.notional_pct for p in self.open_positions)

    async def maybe_enter(self, side: str, confidence: float, price: float, tp_move: float, sl_move: float):
        if side == "CASH":
            return
        if self.current_exposure() + self.cfg.per_trade_pct > self.cfg.max_total_exposure_pct + 1e-9:
            logger.info(f"Skip {side} signal (conf={confidence:.3f}): exposure cap reached "
                        f"({self.current_exposure():.0%}/{self.cfg.max_total_exposure_pct:.0%})")
            return
        final_action = 1 if side == "LONG" else 2
        result = await self.executor.execute_to_target(
            final_action=final_action,
            target_exposure=self.cfg.per_trade_pct,
            target_exec_leverage=1.0,
            current_price=price,
            timestamp_kst=pd.Timestamp.now(),
            decision_info={"model": MODEL_ID, "confidence": confidence, "side": side},
        )
        if not result.get("ok"):
            logger.warning(f"Entry not placed: {result.get('status')}")
            return
        tpsl = await self.executor.place_tp_sl_orders(
            side=side, entry_price=price, take_profit=tp_move, stop_loss=sl_move,
            reason_prefix="scalp_1m",
        )
        self.open_positions.append(OpenPosition(side, price, tpsl.get("tp_price", 0.0),
                                                  tpsl.get("sl_price", 0.0), self.cfg.per_trade_pct, time.time()))
        logger.info(f"Entered {side} @ {price} conf={confidence:.3f} tp={tpsl.get('tp_price')} "
                    f"sl={tpsl.get('sl_price')} exposure_now={self.current_exposure():.0%}")

    async def run_cycle(self):
        eth_df = await self.fetch_recent_candles()
        btc_df = await self.fetch_recent_btc_candles()
        row = self.compute_latest_features(eth_df, btc_df)
        if row is None:
            logger.warning("No feature row produced this cycle (insufficient warmup data?)")
            return
        atr_pct = self.compute_atr_pct(eth_df)
        side, confidence, tp_move, sl_move = self.decide(row, atr_pct)
        price = float(row["close"])
        logger.info(f"Cycle @ {row['timestamp']}: side={side} confidence={confidence:.3f} price={price}")
        await self.maybe_enter(side, confidence, price, tp_move, sl_move)
        # TODO (see module docstring): reconcile self.open_positions against exchange-reported
        # fills (TP/SL may have triggered since the last cycle) and enforce the 20min horizon
        # forced-exit if neither TP nor SL has triggered by then.

    async def main_loop(self):
        self.load_model()
        logger.info(f"Scalp live runner starting: account_enabled={self.cfg.account_enabled} "
                    f"execution_enabled={self.cfg.execution_enabled} dry_run={self.cfg.execution_dry_run} "
                    f"per_trade={self.cfg.per_trade_pct:.0%} max_exposure={self.cfg.max_total_exposure_pct:.0%}")
        if not self.cfg.account_enabled:
            logger.warning("SCALP_ACCOUNT_ENABLED=False -- running in decision-only mode, no orders will be placed")
        while not self._stop.is_set():
            now = time.time()
            next_boundary = now - (now % POLL_INTERVAL_SEC) + POLL_INTERVAL_SEC
            while time.time() < next_boundary and not self._stop.is_set():
                await asyncio.sleep(1)
            if self._stop.is_set():
                break
            try:
                await self.run_cycle()
            except Exception:
                logger.exception("Cycle failed")

    def stop(self):
        self._stop.set()


async def main():
    cfg = ScalpConfig.from_env()
    runner = ScalpLiveRunner(cfg)

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, runner.stop)
        except NotImplementedError:
            pass  # Windows doesn't support add_signal_handler for these

    await runner.main_loop()


if __name__ == "__main__":
    asyncio.run(main())
