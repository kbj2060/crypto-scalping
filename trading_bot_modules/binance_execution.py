import asyncio
import json
import logging
import math
import os
import time
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from trading_bot_modules.binance_runtime_config import BinanceExecutionConfig
from trading_bot_modules.omega4_6_1_runtime_contract import (
    require_execution_promotion_manifest,
)

load_dotenv()

logger = logging.getLogger("LiveBot")


def _safe_float(v, d: float = 0.0) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else float(d)
    except Exception:
        return float(d)


def _append_jsonl(path: str, payload: dict) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


class BinanceFuturesExecutionAdapter:
    """Explicitly gated Binance USDT-M futures execution bridge."""

    def __init__(self, fetcher: Any, config: BinanceExecutionConfig | None = None):
        config = config or BinanceExecutionConfig.from_env()
        self.fetcher = fetcher
        self.exchange = fetcher.account_exchange
        self.requested_enabled = bool(config.enabled)
        self.enabled = bool(config.enabled)
        self.disabled_reason = "" if self.enabled else "configured_off"
        self.disabled_at = "" if self.enabled else pd.Timestamp.now(tz="Asia/Seoul").isoformat()
        self.last_error = ""
        self.last_error_at = ""
        self.dry_run = bool(config.dry_run)
        self.symbol = str(config.symbol or fetcher.account_symbol)
        self.require_testnet = bool(config.require_testnet)
        self.audit_path = str(config.audit_path)
        self.margin_mode = str(config.margin_mode or "").lower()
        self.set_leverage_enabled = bool(config.set_leverage)
        self.max_exchange_leverage = int(max(1, config.max_exchange_leverage))
        self.min_notional_usdt = float(max(0.0, config.min_notional_usdt))
        self.max_target_notional_usdt = float(max(0.0, config.max_target_notional_usdt))
        self.rebalance_tolerance_usdt = float(max(0.0, config.rebalance_tolerance_usdt))
        self.alpha14_router_enabled = bool(config.alpha14_router_enabled)
        self.maker_reduce_only_enabled = bool(config.maker_reduce_only_enabled)
        self.maker_entry_fallback_market = bool(config.maker_entry_fallback_market)
        self.maker_exit_fallback_market = bool(config.maker_exit_fallback_market)
        self.maker_fallback_market = bool(self.maker_entry_fallback_market or self.maker_exit_fallback_market)
        self.maker_wait_sec = float(max(0.0, config.maker_wait_sec))
        self.maker_book_depth = int(max(5, config.maker_book_depth))
        self.maker_max_spread_bps = float(max(0.0, config.maker_max_spread_bps))
        self.maker_min_imbalance = float(max(0.0, config.maker_min_imbalance))
        self.maker_min_microprice_edge_bps = float(max(0.0, config.maker_min_microprice_edge_bps))
        self.maker_entry_offset_bps = float(max(0.0, config.maker_entry_offset_bps))
        self.maker_exit_offset_bps = float(max(0.0, config.maker_exit_offset_bps))
        self.resting_tpsl_enabled = bool(config.resting_tpsl_enabled)
        self._markets_loaded = False
        self._one_way_checked = False
        if self.enabled and not self._ready():
            logger.warning("SYSTEM binance_execution=OFF reason=account_not_ready")
            self._disable("account_not_ready")
        if self.enabled and not self.dry_run and self.require_testnet and not bool(fetcher.account_testnet):
            logger.warning("SYSTEM binance_execution=OFF reason=mainnet_requires_explicit_override")
            self._disable("mainnet_requires_explicit_override")
        if self.enabled and not self.dry_run and not bool(fetcher.account_testnet):
            confirm = str(config.confirm_live or "").strip()
            if confirm != "I_UNDERSTAND_REAL_ORDERS":
                logger.warning("SYSTEM binance_execution=OFF reason=missing_live_confirmation")
                self._disable("missing_live_confirmation")
        if self.enabled and bool(config.promotion_manifest_required):
            try:
                require_execution_promotion_manifest(config.promotion_manifest_path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                logger.error(
                    "SYSTEM binance_execution=OFF reason=promotion_manifest_failed err=%s",
                    exc,
                )
                self._disable("promotion_manifest_failed", str(exc))

    def _disable(self, reason: str, error: str = "") -> None:
        self.enabled = False
        self.disabled_reason = str(reason or "disabled")
        self.disabled_at = pd.Timestamp.now(tz="Asia/Seoul").isoformat()
        if error:
            self.last_error = str(error)
            self.last_error_at = self.disabled_at

    def _record_runtime_issue(self, reason: str, error: str = "") -> None:
        current_error = str(error or reason or "execution_issue")
        if current_error != self.last_error or not self.last_error_at:
            self.last_error_at = pd.Timestamp.now(tz="Asia/Seoul").isoformat()
        self.last_error = current_error

    def _clear_runtime_issue(self) -> None:
        if self.enabled:
            self.last_error = ""
            self.last_error_at = ""

    def _mark_result_issue(self, result: dict, reason: str, error: str = "") -> None:
        self._record_runtime_issue(reason, error)
        result.update(
            {
                "last_error": str(self.last_error),
                "last_error_at": str(self.last_error_at),
            }
        )

    def _ready(self) -> bool:
        return bool(self.exchange is not None and self.fetcher.account_enabled)

    def status(self) -> dict:
        return {
            "requested_enabled": bool(self.requested_enabled),
            "enabled": bool(self.enabled),
            "health": "ready" if self.enabled else ("blocked" if self.requested_enabled else "disabled"),
            "disabled_reason": str(self.disabled_reason),
            "disabled_at": str(self.disabled_at),
            "last_error": str(self.last_error),
            "last_error_at": str(self.last_error_at),
            "dry_run": bool(self.dry_run),
            "testnet": bool(self.fetcher.account_testnet),
            "symbol": str(self.symbol),
            "audit_path": str(self.audit_path),
            "require_testnet": bool(self.require_testnet),
            "min_notional_usdt": float(self.min_notional_usdt),
            "max_target_notional_usdt": float(self.max_target_notional_usdt),
            "rebalance_tolerance_usdt": float(self.rebalance_tolerance_usdt),
            "alpha14_router_enabled": bool(self.alpha14_router_enabled),
            "maker_wait_sec": float(self.maker_wait_sec),
            "maker_fallback_market": bool(self.maker_fallback_market),
            "maker_entry_fallback_market": bool(self.maker_entry_fallback_market),
            "maker_exit_fallback_market": bool(self.maker_exit_fallback_market),
            "maker_reduce_only_enabled": bool(self.maker_reduce_only_enabled),
            "maker_entry_offset_bps": float(self.maker_entry_offset_bps),
            "maker_exit_offset_bps": float(self.maker_exit_offset_bps),
            "resting_tpsl_enabled": bool(self.resting_tpsl_enabled),
        }

    async def _call(self, label: str, fn):
        return await self.fetcher._call_with_retry(label, fn)

    async def _ensure_markets(self) -> None:
        if self._markets_loaded or self.exchange is None:
            return
        await self._call("binance_execution.load_markets", lambda: self.exchange.load_markets())
        self._markets_loaded = True

    async def _ensure_one_way_mode(self) -> None:
        if self._one_way_checked or self.exchange is None:
            return
        try:
            method = getattr(self.exchange, "fapiPrivateGetPositionSideDual", None)
            if method is None:
                method = getattr(self.exchange, "fapiPrivate_get_position_side_dual", None)
            if method is None:
                raise RuntimeError("position side mode endpoint unavailable")
            payload = await self._call("binance_execution.position_side_dual", lambda: method())
            dual = str(dict(payload or {}).get("dualSidePosition", "false")).lower() == "true"
            if dual:
                raise RuntimeError("hedge_mode_enabled")
            self._one_way_checked = True
        except Exception as e:
            raise RuntimeError(f"one_way_mode_check_failed:{e}") from e

    def _market(self) -> dict:
        if self.exchange is None:
            return {}
        try:
            return dict(self.exchange.market(self.symbol) or {})
        except Exception:
            return {}

    def _amount_to_precision(self, amount: float) -> float:
        amount = abs(float(amount or 0.0))
        if self.exchange is None:
            return amount
        try:
            return float(self.exchange.amount_to_precision(self.symbol, amount))
        except Exception:
            return amount

    def _price_to_precision(self, price: float) -> float:
        price = float(price or 0.0)
        if self.exchange is None:
            return price
        try:
            return float(self.exchange.price_to_precision(self.symbol, price))
        except Exception:
            return price

    def _min_order_notional(self) -> float:
        market = self._market()
        limits = dict(market.get("limits", {}) or {})
        cost = dict(limits.get("cost", {}) or {})
        return float(max(self.min_notional_usdt, _safe_float(cost.get("min", 0.0), 0.0)))

    @staticmethod
    def _action_side(action: int) -> str:
        if int(action) == 1:
            return "LONG"
        if int(action) == 2:
            return "SHORT"
        return "NONE"

    @staticmethod
    def _signed_amount(position: dict | None) -> float:
        pos = dict(position or {})
        side = str(pos.get("type", "") or "").upper()
        contracts = _safe_float(pos.get("contracts", 0.0), 0.0)
        if side == "LONG":
            return abs(contracts)
        if side == "SHORT":
            return -abs(contracts)
        return 0.0

    @staticmethod
    def _safe_order(order: dict | None) -> dict:
        row = dict(order or {})
        out = {}
        for key in ("id", "clientOrderId", "symbol", "type", "side", "amount", "price", "average", "filled", "remaining", "cost", "status", "timestamp", "datetime"):
            if key in row:
                out[key] = row.get(key)
        info = dict(row.get("info", {}) or {})
        for key in ("orderId", "clientOrderId", "symbol", "status", "executedQty", "avgPrice", "cumQuote", "reduceOnly"):
            if key in info and key not in out:
                out[key] = info.get(key)
        return out

    async def _append_audit(self, payload: dict) -> None:
        row = dict(payload or {})
        row.setdefault("schema_version", "binance_execution_audit.v1")
        row.setdefault("created_at", pd.Timestamp.utcnow().isoformat())
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _append_jsonl, self.audit_path, row)

    async def _fetch_balance_equity(self) -> float:
        if self.exchange is None:
            return 0.0
        bal = await self._call(
            "binance_execution.fetch_balance[futures]",
            lambda: self.exchange.fetch_balance(params={"type": "future"}),
        )
        total = dict((bal or {}).get("total", {}) or {})
        # USDT-M futures: account balance/margin settles in USDT.
        return _safe_float(total.get("USDT", 0.0), 0.0)

    async def _fetch_position(self, *, require_ok: bool = True) -> dict | None:
        if self.exchange is None:
            if require_ok:
                raise RuntimeError("position_fetch_unavailable")
            return None
        try:
            positions = await self._call(
                f"binance_execution.fetch_positions[{self.symbol}]",
                lambda: self.exchange.fetch_positions([self.symbol]),
            )
            active: list[dict] = []
            for p in positions or []:
                info = dict(p.get("info", {}) or {})
                pos_side = str(info.get("positionSide", p.get("positionSide", "")) or "").upper()
                if pos_side not in {"", "BOTH"}:
                    raise RuntimeError(f"hedge_position_side_detected:{pos_side}")
                contracts = _safe_float(p.get("contracts", info.get("positionAmt", 0.0)), 0.0)
                position_amt = _safe_float(info.get("positionAmt", contracts), contracts)
                size = position_amt if abs(position_amt) >= abs(contracts) else contracts
                if abs(size) <= 1e-12:
                    continue
                side = str(p.get("side") or "").upper()
                if side not in {"LONG", "SHORT"}:
                    side = "LONG" if size > 0.0 else "SHORT"
                entry = _safe_float(p.get("entryPrice", p.get("entry_price", info.get("entryPrice", 0.0))), 0.0)
                notional = abs(_safe_float(p.get("notional", info.get("notional", 0.0)), 0.0))
                leverage = _safe_float(p.get("leverage", info.get("leverage", 0.0)), 0.0)
                active.append({
                    "type": side,
                    "entry_price": float(entry),
                    "contracts": float(abs(size)),
                    "signed_contracts": float(size),
                    "notional": float(notional),
                    "leverage": float(leverage),
                    "source": "binance_execution",
                })
            if len(active) > 1:
                raise RuntimeError(f"multiple_active_positions:{len(active)}")
            return active[0] if active else None
        except Exception as e:
            logger.warning("SYSTEM binance_execution position=BAD reason=%s", e)
            if require_ok:
                raise
        return None

    async def _configure_symbol(self, leverage: float) -> dict:
        out = {"margin_mode": "", "leverage": 0, "warnings": []}
        if self.exchange is None:
            return out
        await self._ensure_one_way_mode()
        if self.margin_mode in {"isolated", "cross"}:
            try:
                await self._call(
                    f"binance_execution.set_margin_mode[{self.margin_mode}]",
                    lambda: self.exchange.set_margin_mode(self.margin_mode, self.symbol),
                )
                out["margin_mode"] = self.margin_mode
            except Exception as e:
                raise RuntimeError(f"set_margin_mode_failed:{e}") from e
        if self.set_leverage_enabled:
            lev = int(max(1, min(self.max_exchange_leverage, math.ceil(float(leverage or 1.0)))))
            try:
                await self._call(
                    f"binance_execution.set_leverage[{lev}]",
                    lambda: self.exchange.set_leverage(lev, self.symbol),
                )
                out["leverage"] = int(lev)
            except Exception as e:
                raise RuntimeError(f"set_leverage_failed:{e}") from e
        return out

    def _position_matches_target(self, position: dict | None, target_side: str, target_notional: float, price: float) -> tuple[bool, dict]:
        pos = dict(position or {})
        actual_signed = float(self._signed_amount(pos))
        target_notional = float(max(0.0, target_notional))
        price = float(max(price, 1e-12))
        target_amount = target_notional / price
        if target_side == "LONG":
            target_signed = target_amount
        elif target_side == "SHORT":
            target_signed = -target_amount
        else:
            target_signed = 0.0
        delta_notional = abs(actual_signed - target_signed) * price
        tolerance = max(float(self.rebalance_tolerance_usdt), self._min_order_notional())
        side_ok = (
            (target_side == "NONE" and abs(actual_signed) <= 1e-12)
            or (target_side == "LONG" and actual_signed > 0.0)
            or (target_side == "SHORT" and actual_signed < 0.0)
        )
        notional_ok = delta_notional <= tolerance
        return bool(side_ok and notional_ok), {
            "target_side": str(target_side),
            "target_signed_amount": float(target_signed),
            "actual_signed_amount": float(actual_signed),
            "delta_notional_usdt": float(delta_notional),
            "tolerance_usdt": float(tolerance),
            "side_ok": bool(side_ok),
            "notional_ok": bool(notional_ok),
        }

    async def _submit_market_order(self, *, side: str, amount: float, reduce_only: bool, reason: str) -> dict:
        amount_precise = self._amount_to_precision(amount)
        if amount_precise <= 0.0:
            raise ValueError(f"order amount rounded to zero: raw={amount}")
        params = {
            "reduceOnly": bool(reduce_only),
            "newClientOrderId": f"cbot_{time.time_ns() % 10_000_000_000}_{len(reason) % 1000}",
        }
        if self.dry_run:
            return {
                "dry_run": True,
                "symbol": self.symbol,
                "type": "market",
                "side": side,
                "amount": float(amount_precise),
                "reduceOnly": bool(reduce_only),
                "reason": str(reason),
            }
        order = await self._call(
            f"binance_execution.create_order[{side}:{amount_precise}]",
            lambda: self.exchange.create_order(self.symbol, "market", side, amount_precise, None, params),
        )
        out = self._safe_order(order)
        out.update({"dry_run": False, "reduceOnly": bool(reduce_only), "reason": str(reason)})
        return out

    @staticmethod
    def _tp_sl_prices(*, side: str, entry_price: float, take_profit: float, stop_loss: float) -> tuple[float, float]:
        """Raw price-move fractions -> absolute trigger prices. Mirrors the LONG/SHORT price-move
        convention already used for `take_profit_price`/`stop_price` elsewhere (e.g.
        `_omega461_shadow_price` in trading_bot.py and `thresholdPrice` in dashboard/live/app.js)
        -- single formula, don't re-derive it differently here."""
        entry = float(entry_price)
        tp = float(max(0.0, take_profit))
        sl = float(max(0.0, stop_loss))
        if str(side).upper() == "LONG":
            tp_price = entry * (1.0 + tp) if tp > 0.0 else 0.0
            sl_price = entry * max(0.0, 1.0 - sl) if sl > 0.0 else 0.0
        else:
            tp_price = entry * max(0.0, 1.0 - tp) if tp > 0.0 else 0.0
            sl_price = entry * (1.0 + sl) if sl > 0.0 else 0.0
        return float(tp_price), float(sl_price)

    async def _submit_stop_order(self, *, side: str, stop_price: float, order_type: str, reason: str) -> dict:
        """Places a closePosition=true STOP_MARKET/TAKE_PROFIT_MARKET order. `side` is the side
        that CLOSES the position (opposite of the position's own side). No quantity/reduceOnly --
        both are mutually exclusive with closePosition on Binance USDS-M futures."""
        stop_price_precise = self._price_to_precision(stop_price)
        if stop_price_precise <= 0.0:
            raise ValueError(f"stop price rounded to zero: raw={stop_price}")
        params = {
            "stopPrice": stop_price_precise,
            "closePosition": True,
            "workingType": "MARK_PRICE",
            "newClientOrderId": f"cbot_tpsl_{time.time_ns() % 10_000_000_000}_{len(reason) % 1000}",
        }
        if self.dry_run:
            return {
                "dry_run": True,
                "symbol": self.symbol,
                "type": str(order_type),
                "side": side,
                "stopPrice": float(stop_price_precise),
                "closePosition": True,
                "reason": str(reason),
                "id": f"dry_{order_type.lower()}_{time.time_ns() % 10_000_000_000}",
            }
        order = await self._call(
            f"binance_execution.create_stop_order[{order_type}:{side}@{stop_price_precise}]",
            lambda: self.exchange.create_order(self.symbol, order_type, side, None, None, params),
        )
        out = self._safe_order(order)
        out.update({"dry_run": False, "type": str(order_type), "reason": str(reason)})
        return out

    async def _submit_reduce_limit_order(self, *, side: str, amount: float, price: float, reason: str) -> dict:
        """Places a plain GTC reduceOnly LIMIT order (not postOnly). Used for take-profit: unlike
        stop-loss, a LONG's TP sits *above* the current price, so a resting sell limit there is a
        genuine maker order that only fills once price actually rises to meet it -- no stopPrice
        trigger needed. (The same trick doesn't work for stop-loss: a sell limit *below* the
        current price would be immediately marketable and fill the instant it's placed.) If price
        has already passed the limit price by the time this is submitted, it fills immediately as
        a taker at the limit price or better -- not an error, just an immediate fill."""
        amount_precise = self._amount_to_precision(amount)
        price_precise = self._price_to_precision(price)
        if amount_precise <= 0.0:
            raise ValueError(f"order amount rounded to zero: raw={amount}")
        if price_precise <= 0.0:
            raise ValueError(f"limit price rounded to zero: raw={price}")
        params = {
            "reduceOnly": True,
            "timeInForce": "GTC",
            "newClientOrderId": f"cbot_tp_{time.time_ns() % 10_000_000_000}_{len(reason) % 1000}",
        }
        if self.dry_run:
            return {
                "dry_run": True,
                "symbol": self.symbol,
                "type": "limit",
                "side": side,
                "amount": float(amount_precise),
                "price": float(price_precise),
                "reduceOnly": True,
                "reason": str(reason),
                "id": f"dry_tp_limit_{time.time_ns() % 10_000_000_000}",
                "status": "open",
            }
        order = await self._call(
            f"binance_execution.create_reduce_limit_order[{side}:{amount_precise}@{price_precise}]",
            lambda: self.exchange.create_order(self.symbol, "limit", side, amount_precise, price_precise, params),
        )
        out = self._safe_order(order)
        out.update({"dry_run": False, "type": "limit", "reason": str(reason)})
        return out

    @staticmethod
    def _is_immediate_trigger_error(e: Exception) -> bool:
        msg = str(e).lower()
        return "-2021" in msg or "immediately trigger" in msg

    async def place_tp_sl_orders(self, *, side: str, entry_price: float, take_profit: float,
                                  stop_loss: float, reason_prefix: str, amount: float = 0.0) -> dict:
        """Places a resting take-profit LIMIT order + a STOP_MARKET stop-loss right after an entry
        fill, so a barrier touch fills immediately at the exchange instead of waiting for the
        bot's next decision cycle.

        TP is a plain reduceOnly LIMIT order (see _submit_reduce_limit_order) -- for a LONG it
        sits above the current price as a genuine maker order, no trigger needed, no slippage.
        SL cannot use the same trick (a sell limit below market would fill immediately on
        placement), so it stays a STOP_MARKET conditional order: guaranteed to fill once
        triggered, at the cost of market-order slippage.

        Binance USDS-M futures has no native OCO, so the two orders are independent; the caller is
        responsible for cancelling the untouched leg once the other fills or the position is
        closed by other means (see cancel_tp_sl_orders).

        If SL's trigger price is already on the wrong side of the current mark price (e.g. entry
        filled late enough that price already blew through the barrier), Binance rejects the
        conditional order with -2021 "would immediately trigger"; the fallback is an immediate
        market close (that leg has effectively already been hit) -- this requires `amount` (the
        position size). TP doesn't need this fallback: if price already passed the TP limit price,
        the LIMIT order just fills immediately as a taker instead of being rejected."""
        result = {"tp_order_id": "", "sl_order_id": "", "tp_price": 0.0, "sl_price": 0.0, "errors": []}
        if not self.resting_tpsl_enabled:
            return result
        tp_price, sl_price = self._tp_sl_prices(
            side=side, entry_price=entry_price, take_profit=take_profit, stop_loss=stop_loss
        )
        close_side = "sell" if str(side).upper() == "LONG" else "buy"
        closed_immediately = False
        if tp_price > 0.0:
            if float(amount) <= 0.0:
                result["errors"].append("take_profit_order_failed:amount_required_for_limit_order")
                logger.warning("SYSTEM binance_execution tp_order=BAD reason=amount_required_for_limit_order")
            else:
                try:
                    tp_order = await self._submit_reduce_limit_order(
                        side=close_side, amount=float(amount), price=tp_price,
                        reason=f"{reason_prefix}_take_profit",
                    )
                    result["tp_order_id"] = str(tp_order.get("id") or "")
                    result["tp_price"] = float(tp_price)
                    if str(tp_order.get("status", "")).lower() in {"closed", "filled"}:
                        # Price had already passed the TP limit price by the time we placed it --
                        # it filled immediately as a taker. The position is already closed; don't
                        # also place a STOP_MARKET SL against a position that no longer exists.
                        result["tp_immediate_fill"] = tp_order
                        closed_immediately = True
                except Exception as e:
                    result["errors"].append(f"take_profit_order_failed:{e}")
                    logger.warning("SYSTEM binance_execution tp_order=BAD reason=%s", e)
        if sl_price > 0.0 and not closed_immediately:
            try:
                sl_order = await self._submit_stop_order(
                    side=close_side, stop_price=sl_price, order_type="STOP_MARKET",
                    reason=f"{reason_prefix}_stop_loss",
                )
                result["sl_order_id"] = str(sl_order.get("id") or "")
                result["sl_price"] = float(sl_price)
            except Exception as e:
                if self._is_immediate_trigger_error(e) and float(amount) > 0.0:
                    logger.warning(
                        "SYSTEM binance_execution sl_order immediate_trigger -> market_fallback amount=%s", amount
                    )
                    try:
                        fallback = await self._submit_market_order(
                            side=close_side, amount=float(amount), reduce_only=True,
                            reason=f"{reason_prefix}_stop_loss_immediate_trigger_fallback",
                        )
                        result["sl_immediate_fill"] = fallback
                        closed_immediately = True
                        if result.get("tp_order_id"):
                            cancel_res = await self.cancel_tp_sl_orders(
                                tp_order_id=str(result["tp_order_id"]), sl_order_id=""
                            )
                            result["tp_cancelled_after_sl_immediate_fill"] = cancel_res
                    except Exception as e2:
                        result["errors"].append(f"stop_loss_immediate_fallback_failed:{e2}")
                        logger.warning("SYSTEM binance_execution sl_immediate_fallback=BAD reason=%s", e2)
                else:
                    result["errors"].append(f"stop_loss_order_failed:{e}")
                    logger.warning("SYSTEM binance_execution sl_order=BAD reason=%s", e)
        result["closed_immediately"] = bool(closed_immediately)
        return result

    async def cancel_tp_sl_orders(self, *, tp_order_id: str, sl_order_id: str) -> dict:
        """Cancels whichever resting TP/SL order ids are non-empty. Tolerates 'unknown order'
        errors -- the order may have already filled or been cancelled."""
        result = {"tp_cancelled": False, "sl_cancelled": False, "errors": []}
        if self.dry_run or self.exchange is None:
            result["tp_cancelled"] = bool(tp_order_id)
            result["sl_cancelled"] = bool(sl_order_id)
            return result
        for label, order_id, key in (("tp", tp_order_id, "tp_cancelled"), ("sl", sl_order_id, "sl_cancelled")):
            if not order_id:
                continue
            try:
                await self._call(
                    f"binance_execution.cancel_order[{label}:{order_id}]",
                    lambda oid=order_id: self.exchange.cancel_order(oid, self.symbol),
                )
                result[key] = True
            except Exception as e:
                msg = str(e).lower()
                if "unknown order" in msg or "order does not exist" in msg or "-2011" in msg:
                    result[key] = True
                else:
                    result["errors"].append(f"{label}_cancel_failed:{e}")
                    logger.warning("SYSTEM binance_execution cancel_%s_order=BAD reason=%s", label, e)
        return result

    async def poll_tp_sl_orders(self, *, tp_order_id: str, sl_order_id: str) -> dict:
        """Fetches current status of the resting TP/SL orders for reconciliation (e.g. detecting
        a fill that happened between decision cycles or while the process was down)."""
        result = {"tp": None, "sl": None}
        if self.exchange is None:
            return result
        for label, order_id, key in (("tp", tp_order_id, "tp"), ("sl", sl_order_id, "sl")):
            if not order_id:
                continue
            try:
                order = await self._call(
                    f"binance_execution.fetch_order[{label}:{order_id}]",
                    lambda oid=order_id: self.exchange.fetch_order(oid, self.symbol),
                )
                result[key] = self._safe_order(order)
            except Exception as e:
                logger.warning("SYSTEM binance_execution fetch_%s_order=BAD reason=%s", label, e)
        return result

    @staticmethod
    def _book_summary(orderbook: dict | None) -> dict:
        ob = dict(orderbook or {})
        bids = [[_safe_float(x[0]), _safe_float(x[1])] for x in list(ob.get("bids") or []) if len(x) >= 2]
        asks = [[_safe_float(x[0]), _safe_float(x[1])] for x in list(ob.get("asks") or []) if len(x) >= 2]
        best_bid = bids[0][0] if bids else 0.0
        best_ask = asks[0][0] if asks else 0.0
        mid = (best_bid + best_ask) / 2.0 if best_bid > 0 and best_ask > 0 else 0.0
        bid_notional_5 = sum(float(p) * float(q) for p, q in bids[:5])
        ask_notional_5 = sum(float(p) * float(q) for p, q in asks[:5])
        imbalance_5 = (
            (bid_notional_5 - ask_notional_5) / max(abs(bid_notional_5) + abs(ask_notional_5), 1e-12)
            if (bid_notional_5 or ask_notional_5)
            else 0.0
        )
        bid_qty = bids[0][1] if bids else 0.0
        ask_qty = asks[0][1] if asks else 0.0
        microprice = (
            (best_ask * bid_qty + best_bid * ask_qty) / max(bid_qty + ask_qty, 1e-12)
            if best_bid > 0 and best_ask > 0
            else 0.0
        )
        return {
            "best_bid": float(best_bid),
            "best_ask": float(best_ask),
            "mid": float(mid),
            "spread_bps": float((best_ask - best_bid) / mid * 10000.0) if mid > 0 else 9999.0,
            "imbalance_5": float(imbalance_5),
            "microprice_edge_bps": float((microprice - mid) / mid * 10000.0) if mid > 0 else 0.0,
        }

    async def _fetch_orderbook_summary(self) -> dict:
        if self.exchange is None:
            return {}
        ob = await self._call(
            f"binance_execution.fetch_order_book[{self.symbol}]",
            lambda: self.exchange.fetch_order_book(self.symbol, limit=int(self.maker_book_depth)),
        )
        return self._book_summary(ob)

    def _orderbook_from_decision(self, decision_info: dict | None) -> dict:
        info = dict(decision_info or {})
        snap = dict(info.get("orderbook_snapshot", {}) or {})
        if not snap:
            snap = dict(dict(info.get("sleeve_trace", {}) or {}).get("orderbook_snapshot", {}) or {})
        if not snap or not bool(snap.get("recorded", False)):
            return {}
        return {
            "best_bid": _safe_float(snap.get("best_bid", 0.0), 0.0),
            "best_ask": _safe_float(snap.get("best_ask", 0.0), 0.0),
            "spread_bps": _safe_float(snap.get("spread_bps", 9999.0), 9999.0),
            "imbalance_5": _safe_float(snap.get("imbalance_5", 0.0), 0.0),
            "microprice_edge_bps": _safe_float(snap.get("microprice_edge_bps", 0.0), 0.0),
        }

    def _fallback_enabled(self, reduce_only: bool) -> bool:
        return bool(self.maker_exit_fallback_market if bool(reduce_only) else self.maker_entry_fallback_market)

    def _no_maker_route(self, *, reduce_only: bool, reason: str, **extra) -> dict:
        route = {
            "router": "alpha3_corrected_next_open_limit_touch0",
            "enabled": bool(self.alpha14_router_enabled),
            "route": "market" if bool(reduce_only) and self._fallback_enabled(True) else "skip",
            "reason": str(reason),
            "book": {},
            "fallback": "market" if bool(reduce_only) and self._fallback_enabled(True) else "none",
        }
        route.update(extra)
        return route

    async def _route_order(self, *, side: str, reduce_only: bool, decision_info: dict | None) -> dict:
        route = {
            "router": "alpha3_corrected_next_open_limit_touch0",
            "enabled": bool(self.alpha14_router_enabled),
            "route": "skip",
            "reason": "router_disabled",
            "book": {},
            "fallback": "none",
        }
        if not self.alpha14_router_enabled:
            return self._no_maker_route(reduce_only=reduce_only, reason="router_disabled")
        if bool(reduce_only) and not self.maker_reduce_only_enabled:
            return self._no_maker_route(reduce_only=reduce_only, reason="reduce_only_maker_disabled")
        book = self._orderbook_from_decision(decision_info)
        if not book:
            try:
                book = await self._fetch_orderbook_summary()
            except Exception as exc:
                return self._no_maker_route(reduce_only=reduce_only, reason=f"orderbook_unavailable:{exc}")
        route["book"] = dict(book)
        best_bid = _safe_float(book.get("best_bid", 0.0), 0.0)
        best_ask = _safe_float(book.get("best_ask", 0.0), 0.0)
        spread_bps = _safe_float(book.get("spread_bps", 9999.0), 9999.0)
        side_sign = 1.0 if str(side).lower() == "buy" else -1.0
        flow_score = side_sign * _safe_float(book.get("imbalance_5", 0.0), 0.0)
        micro_edge = side_sign * _safe_float(book.get("microprice_edge_bps", 0.0), 0.0)
        if best_bid <= 0.0 or best_ask <= 0.0:
            return self._no_maker_route(reduce_only=reduce_only, reason="bad_book", book=dict(book))
        offset_bps = float(self.maker_exit_offset_bps if bool(reduce_only) else self.maker_entry_offset_bps)
        if str(side).lower() == "buy":
            maker_price = best_bid * (1.0 - offset_bps / 10000.0)
        else:
            maker_price = best_ask * (1.0 + offset_bps / 10000.0)
        route.update({
            "route": "post_only_limit",
            "reason": "book_favorable",
            "maker_price": float(self._price_to_precision(maker_price)),
            "flow_score": float(flow_score),
            "micro_edge_bps": float(micro_edge),
            "spread_bps": float(spread_bps),
            "offset_bps": float(offset_bps),
            "fallback": "market" if self._fallback_enabled(bool(reduce_only)) else "none",
        })
        return route

    async def _submit_limit_order(self, *, side: str, amount: float, price: float, reduce_only: bool, reason: str, route: dict) -> dict:
        amount_precise = self._amount_to_precision(amount)
        price_precise = self._price_to_precision(price)
        if amount_precise <= 0.0:
            raise ValueError(f"order amount rounded to zero: raw={amount}")
        if price_precise <= 0.0:
            raise ValueError(f"limit price rounded to zero: raw={price}")
        params = {
            "reduceOnly": bool(reduce_only),
            "timeInForce": "GTX",
            "postOnly": True,
            "newClientOrderId": f"cbot_m_{time.time_ns() % 10_000_000_000}_{len(reason) % 1000}",
        }
        if self.dry_run:
            return {
                "dry_run": True,
                "symbol": self.symbol,
                "type": "limit",
                "timeInForce": "GTX",
                "postOnly": True,
                "side": side,
                "amount": float(amount_precise),
                "price": float(price_precise),
                "reduceOnly": bool(reduce_only),
                "reason": str(reason),
                "execution_route": dict(route),
            }
        order = await self._call(
            f"binance_execution.create_limit_post_only[{side}:{amount_precise}@{price_precise}]",
            lambda: self.exchange.create_order(self.symbol, "limit", side, amount_precise, price_precise, params),
        )
        out = self._safe_order(order)
        out.update({"dry_run": False, "reduceOnly": bool(reduce_only), "reason": str(reason), "execution_route": dict(route)})
        return out

    @staticmethod
    def _wait_until_bar_close_sec(timestamp_kst) -> float:
        try:
            base_ts = pd.Timestamp(timestamp_kst)
            if base_ts.tzinfo is not None:
                base_ts = base_ts.tz_convert("Asia/Seoul").tz_localize(None)
            close_ts = base_ts + pd.Timedelta(minutes=5)
            now_kst = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None)
            return float(max(0.0, (close_ts - now_kst).total_seconds()))
        except Exception:
            return 0.0

    async def _wait_or_cancel_limit(
        self,
        order: dict,
        *,
        amount: float,
        side: str,
        reduce_only: bool,
        reason: str,
        wait_sec: float,
    ) -> dict:
        oid = str(order.get("id") or order.get("orderId") or "")
        if not oid:
            return {"status": "unknown", "filled": 0.0, "remaining": float(amount), "order": dict(order)}
        wait_sec = float(max(0.0, wait_sec))
        deadline = time.monotonic() + wait_sec
        poll_sec = float(min(1.0, max(0.1, wait_sec / 30.0 if wait_sec > 0.0 else 0.1)))
        latest = dict(order)
        while time.monotonic() < deadline:
            await asyncio.sleep(poll_sec)
            latest = self._safe_order(await self._call(
                f"binance_execution.fetch_order[{oid}]",
                lambda: self.exchange.fetch_order(oid, self.symbol),
            ))
            status = str(latest.get("status", "")).lower()
            if status in {"closed", "filled"}:
                return {
                    "status": "filled",
                    "filled": _safe_float(latest.get("filled", amount), amount),
                    "remaining": 0.0,
                    "order": latest,
                    "wait_sec": float(wait_sec),
                }
        try:
            cancelled = self._safe_order(await self._call(
                f"binance_execution.cancel_order[{oid}]",
                lambda: self.exchange.cancel_order(oid, self.symbol),
            ))
            latest.update({"cancel_result": cancelled})
        except Exception as exc:
            latest.update({"cancel_error": str(exc)})
        filled = _safe_float(latest.get("filled", 0.0), 0.0)
        remaining = max(0.0, float(amount) - filled)
        return {
            "status": "cancelled_unfilled" if filled <= 1e-12 else "cancelled_partial",
            "filled": float(filled),
            "remaining": float(remaining),
            "order": latest,
            "wait_sec": float(wait_sec),
        }

    async def _submit_routed_order(
        self,
        *,
        side: str,
        amount: float,
        reduce_only: bool,
        reason: str,
        decision_info: dict | None,
        timestamp_kst,
    ) -> dict:
        route = await self._route_order(side=side, reduce_only=reduce_only, decision_info=decision_info)
        if str(route.get("route")) != "post_only_limit":
            if str(route.get("route")) == "skip":
                return {
                    "dry_run": bool(self.dry_run),
                    "symbol": self.symbol,
                    "type": "skipped",
                    "side": side,
                    "amount": float(amount),
                    "reduceOnly": bool(reduce_only),
                    "reason": str(reason),
                    "execution_route": dict(route),
                    "unfilled_without_fallback": True,
                }
            out = await self._submit_market_order(side=side, amount=amount, reduce_only=reduce_only, reason=reason)
            out["execution_route"] = dict(route)
            return out
        limit_order = await self._submit_limit_order(
            side=side,
            amount=amount,
            price=float(route.get("maker_price", 0.0)),
            reduce_only=reduce_only,
            reason=reason,
            route=route,
        )
        if self.dry_run:
            return limit_order
        wait_sec = self._wait_until_bar_close_sec(timestamp_kst)
        wait_result = await self._wait_or_cancel_limit(
            limit_order,
            amount=amount,
            side=side,
            reduce_only=reduce_only,
            reason=reason,
            wait_sec=wait_sec,
        )
        if str(wait_result.get("status")) == "filled":
            limit_order["maker_wait_result"] = dict(wait_result)
            return limit_order
        remaining = _safe_float(wait_result.get("remaining", 0.0), 0.0)
        if remaining > 1e-12 and self._fallback_enabled(bool(reduce_only)):
            fallback = await self._submit_market_order(
                side=side,
                amount=remaining,
                reduce_only=reduce_only,
                reason=f"{reason}|maker_fallback_market",
            )
            return {
                "dry_run": False,
                "symbol": self.symbol,
                "type": "routed",
                "side": side,
                "amount": float(amount),
                "reduceOnly": bool(reduce_only),
                "reason": str(reason),
                "execution_route": dict(route),
                "maker_order": dict(limit_order),
                "maker_wait_result": dict(wait_result),
                "fallback_order": dict(fallback),
            }
        limit_order["maker_wait_result"] = dict(wait_result)
        limit_order["unfilled_without_fallback"] = True
        return limit_order

    async def execute_to_target(
        self,
        *,
        final_action: int,
        target_exposure: float,
        target_exec_leverage: float,
        current_price: float,
        timestamp_kst,
        decision_info: dict | None = None,
        existing_tp_sl_order_ids: dict | None = None,
    ) -> dict:
        result = {
            **self.status(),
            "ok": True,
            "blocking": False,
            "status": "disabled" if not self.enabled else "ready",
            "orders": [],
            "warnings": [],
            "target": {},
            "current": {},
            "tp_sl": {},
        }
        if not self.enabled:
            return result
        if not self._ready():
            result.update({"ok": False, "blocking": True, "status": "account_not_ready"})
            self._mark_result_issue(result, "account_not_ready")
            return result
        price = float(current_price or 0.0)
        if price <= 0.0:
            result.update({"ok": False, "blocking": True, "status": "bad_price"})
            self._mark_result_issue(result, "bad_price")
            return result
        try:
            await self._ensure_markets()
            equity = await self._fetch_balance_equity()
            current_pos = await self._fetch_position(require_ok=True)
            current_signed = self._signed_amount(current_pos)
            target_side = self._action_side(int(final_action))
            # A resting TP/SL order can fill between decision cycles; if the caller still passes
            # ids for orders tied to a position that is now flat on the exchange, the caller's
            # local state hasn't reconciled that fill yet. Treating this as a fresh entry would
            # silently reopen the just-closed position. Block instead and let the caller's
            # reconcile path (which owns cancelling/clearing those ids) catch up first.
            _existing_ids = dict(existing_tp_sl_order_ids or {})
            _had_resting_orders = bool(_existing_ids.get("tp_order_id") or _existing_ids.get("sl_order_id"))
            if _had_resting_orders and abs(current_signed) <= 1e-12 and target_side != "NONE":
                result.update({"ok": False, "blocking": True, "status": "position_closed_externally_pending_reconcile"})
                result["current"] = dict(current_pos or {"type": "NONE", "contracts": 0.0, "signed_contracts": 0.0})
                self._mark_result_issue(result, "position_closed_externally_pending_reconcile")
                return result
            target_exposure = float(max(0.0, target_exposure if int(final_action) in (1, 2) else 0.0))
            target_notional = float(max(0.0, equity * target_exposure))
            if self.max_target_notional_usdt > 0.0 and target_notional > self.max_target_notional_usdt:
                result["warnings"].append(
                    f"target_notional_capped:{target_notional:.4f}->{self.max_target_notional_usdt:.4f}"
                )
                target_notional = float(self.max_target_notional_usdt)
                target_exposure = float(target_notional / max(equity, 1e-12))
            target_amount = float(target_notional / max(price, 1e-12))
            target_signed = 0.0
            if target_side == "LONG":
                target_signed = target_amount
            elif target_side == "SHORT":
                target_signed = -target_amount
            min_notional = self._min_order_notional()
            tolerance = max(float(self.rebalance_tolerance_usdt), min_notional)
            result["target"] = {
                "action": int(final_action),
                "side": target_side,
                "exposure": float(target_exposure),
                "notional_usdt": float(target_notional),
                "amount": float(target_amount),
                "exchange_leverage_target": float(target_exec_leverage),
            }
            result["current"] = dict(current_pos or {"type": "NONE", "contracts": 0.0, "signed_contracts": 0.0})
            if target_side != "NONE":
                cfg = (
                    {"dry_run": True, "margin_mode": self.margin_mode, "leverage": int(max(1, min(self.max_exchange_leverage, math.ceil(float(target_exec_leverage or 1.0))))), "warnings": []}
                    if self.dry_run
                    else await self._configure_symbol(float(target_exec_leverage or 1.0))
                )
                result["exchange_config"] = dict(cfg)
                result["warnings"].extend(list(cfg.get("warnings", []) or []))

            orders_to_place: list[dict] = []
            cur_abs = abs(float(current_signed))
            tgt_abs = abs(float(target_signed))
            if tgt_abs <= 1e-12:
                if cur_abs > 1e-12:
                    close_side = "sell" if current_signed > 0.0 else "buy"
                    orders_to_place.append({"side": close_side, "amount": cur_abs, "reduce_only": True, "reason": "close_to_flat"})
            elif abs(float(np.sign(current_signed)) - float(np.sign(target_signed))) > 1e-12 and cur_abs > 1e-12:
                close_side = "sell" if current_signed > 0.0 else "buy"
                open_side = "buy" if target_signed > 0.0 else "sell"
                orders_to_place.append({"side": close_side, "amount": cur_abs, "reduce_only": True, "reason": "close_before_reverse"})
                orders_to_place.append({"side": open_side, "amount": tgt_abs, "reduce_only": False, "reason": "open_after_reverse"})
            else:
                delta = float(target_signed - current_signed)
                delta_notional = abs(delta) * price
                if delta_notional > tolerance:
                    reduce_resize = bool(cur_abs > 1e-12 and tgt_abs < cur_abs)
                    if delta > 0.0:
                        orders_to_place.append({"side": "buy", "amount": abs(delta), "reduce_only": reduce_resize, "reason": "rebalance_buy"})
                    elif delta < 0.0:
                        orders_to_place.append({"side": "sell", "amount": abs(delta), "reduce_only": reduce_resize, "reason": "rebalance_sell"})
            executable_orders = []
            skipped_orders = []
            for order in orders_to_place:
                notional = float(order["amount"]) * price
                if notional + 1e-9 < min_notional and not bool(order.get("reduce_only", False)):
                    skipped_orders.append({**order, "notional_usdt": notional, "skip_reason": "below_min_notional"})
                    continue
                executable_orders.append({**order, "notional_usdt": notional})
            result["planned_orders"] = executable_orders
            result["skipped_orders"] = skipped_orders
            if not executable_orders and skipped_orders and tgt_abs > 1e-12 and cur_abs <= 1e-12:
                result.update({"ok": False, "blocking": True, "status": "below_min_notional"})
                await self._append_audit(
                    {
                        "ts": str(timestamp_kst),
                        "status": "below_min_notional",
                        "ok": False,
                        "dry_run": bool(self.dry_run),
                        "symbol": str(self.symbol),
                        "target": dict(result["target"]),
                        "current": dict(result["current"]),
                        "planned_orders": [],
                        "skipped_orders": list(skipped_orders),
                    }
                )
                self._mark_result_issue(result, "below_min_notional")
                return result
            closing_orders = {"close_to_flat", "close_before_reverse"}
            if any(str(o.get("reason", "")) in closing_orders for o in executable_orders):
                existing_ids = dict(existing_tp_sl_order_ids or {})
                if existing_ids.get("tp_order_id") or existing_ids.get("sl_order_id"):
                    result["tp_sl_cancel"] = await self.cancel_tp_sl_orders(
                        tp_order_id=str(existing_ids.get("tp_order_id", "")),
                        sl_order_id=str(existing_ids.get("sl_order_id", "")),
                    )
            for order in executable_orders:
                submitted = await self._submit_routed_order(
                    side=str(order["side"]),
                    amount=float(order["amount"]),
                    reduce_only=bool(order.get("reduce_only", False)),
                    reason=str(order.get("reason", "")),
                    decision_info=decision_info,
                    timestamp_kst=timestamp_kst,
                )
                result["orders"].append(submitted)
            entry_skip = any(
                str(order.get("type")) == "skipped" and not bool(order.get("reduceOnly", False))
                for order in result["orders"]
            )
            result["status"] = (
                "dry_run"
                if self.dry_run
                else ("entry_maker_miss_skipped" if entry_skip else ("submitted" if result["orders"] else "no_order_needed"))
            )
            result["post_position"] = await self._fetch_position(require_ok=True) if not self.dry_run else result["current"]
            if not self.dry_run and entry_skip:
                result.update({"ok": False, "blocking": True})
                result["post_position_verification"] = {
                    "entry_miss_skip_contract": True,
                    "target_not_verified_after_skipped_entry": True,
                    "target_side": str(target_side),
                    "target_notional_usdt": float(target_notional),
                }
            elif not self.dry_run:
                matched, verification = self._position_matches_target(
                    result.get("post_position"),
                    str(target_side),
                    float(target_notional),
                    float(price),
                )
                result["post_position_verification"] = dict(verification)
                if not matched:
                    result.update({"ok": False, "blocking": True, "status": "post_position_mismatch"})
            else:
                result["post_position_verification"] = {"dry_run": True}
            prior_side = "LONG" if current_signed > 1e-12 else ("SHORT" if current_signed < -1e-12 else "NONE")
            entered_new_side = (
                target_side in {"LONG", "SHORT"} and tgt_abs > 1e-12 and target_side != prior_side
            )
            if entered_new_side and bool(result.get("ok", False)):
                info = dict(decision_info or {})
                post_pos = dict(result.get("post_position") or {})
                entry_price_for_tpsl = float(post_pos.get("entry_price") or price)
                amount_for_tpsl = float(post_pos.get("contracts") or tgt_abs)
                result["tp_sl"] = await self.place_tp_sl_orders(
                    side=str(target_side),
                    entry_price=entry_price_for_tpsl,
                    take_profit=_safe_float(info.get("take_profit", 0.0), 0.0),
                    stop_loss=_safe_float(info.get("stop_loss", 0.0), 0.0),
                    reason_prefix=str(info.get("position_reason", "entry")),
                    amount=amount_for_tpsl,
                )
            await self._append_audit(
                {
                    "ts": str(timestamp_kst),
                    "status": result["status"],
                    "ok": bool(result["ok"]),
                    "dry_run": bool(self.dry_run),
                    "symbol": str(self.symbol),
                    "decision_source": str(dict(decision_info or {}).get("source", "")),
                    "decision_reason": str(dict(decision_info or {}).get("position_reason", "")),
                    "target": dict(result["target"]),
                    "current": dict(result["current"]),
                    "planned_orders": list(executable_orders),
                    "skipped_orders": list(skipped_orders),
                    "orders": list(result["orders"]),
                    "post_position": dict(result.get("post_position") or {}),
                    "post_position_verification": dict(result.get("post_position_verification") or {}),
                    "tp_sl": dict(result.get("tp_sl") or {}),
                    "tp_sl_cancel": dict(result.get("tp_sl_cancel") or {}),
                    "warnings": list(result["warnings"]),
                }
            )
            if bool(result.get("blocking", False)):
                self._mark_result_issue(result, str(result.get("status", "execution_blocked")))
            else:
                self._clear_runtime_issue()
                result.update({"last_error": "", "last_error_at": ""})
            return result
        except Exception as e:
            result.update({"ok": False, "blocking": True, "status": "execution_error", "error": str(e)})
            self._mark_result_issue(result, "execution_error", str(e))
            await self._append_audit(
                {
                    "ts": str(timestamp_kst),
                    "status": "execution_error",
                    "ok": False,
                    "dry_run": bool(self.dry_run),
                    "symbol": str(self.symbol),
                    "error": str(e),
                    "target": dict(result.get("target", {}) or {}),
                    "current": dict(result.get("current", {}) or {}),
                }
            )
            logger.warning("SYSTEM binance_execution=BAD reason=%s", e)
            return result
