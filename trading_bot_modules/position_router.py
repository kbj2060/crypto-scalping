from __future__ import annotations

import json
import logging
import os
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd

from trading_bot_modules.live_io import _atomic_write_json
from trading_bot_modules.omega5_live import OMEGA5_MODEL_ID, OMEGA5_OWNER, OMEGA5_SOURCE_MODEL_ID
from trading_bot_modules.position_accounting import _safe_float

logger = logging.getLogger("LiveBot")


class Colors:
    GREEN, RED, YELLOW, CYAN, BLUE, MAGENTA, DIM, RESET, BOLD = (
        "\033[92m", "\033[91m", "\033[93m", "\033[96m",
        "\033[94m", "\033[95m", "\033[2m", "\033[0m", "\033[1m",
    )


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


FINAL_GOVERNOR_RECONCILE_DEFAULT_EXPOSURE = float(
    os.getenv("FINAL_GOVERNOR_RECONCILE_DEFAULT_EXPOSURE", "1.0")
)


class GovernorPositionRouter:
    def __init__(self):
        self.pos: str | None = None
        self.entry_price: float = 0.0
        self.hold_count: int = 0
        self.position_fraction: float = 0.0
        self.execution_leverage: float = 1.0
        self.current_leverage: float = 0.0
        self.exposure_cap: float = float(os.getenv("GOVERNOR_EXPOSURE_CAP", "5.0"))
        self.peak_equity: float = 1.0
        self.cur_equity: float = 1.0
        self.position_realized_pnl_frac: float = 0.0
        self.last_resize_realized_pnl_frac: float = 0.0
        self.last_realized_pnl: float | None = None
        self.last_closed_hold_count: int = 0
        self._open_trade_diag: dict | None = None
        self.open_trade_id: str = ""
        self.opened_at: str = ""
        self.decision_at: str = ""
        self.entry_price_source: str = ""
        self.entry_decision_price: float = 0.0
        self.exchange_entry_price: float = 0.0
        self.entry_execution_liquidity: str = ""
        self.entry_execution_route: str = ""
        self.entry_execution_order_type: str = ""
        self.open_model_version: str = ""
        self.open_model_id: str = ""
        self.open_model_path: str = ""
        self.open_model_sleeve: str = ""
        self.open_source: str = ""
        self.strategy_state: dict[str, object] = {}
        self._pending_entry_price_source: str = "decision_current_price"
        self._pending_entry_decision_price: float = 0.0
        self._pending_entry_execution_liquidity: str = ""
        self._pending_entry_execution_route: str = ""
        self._pending_entry_execution_order_type: str = ""
        self.trade_history: deque[dict] = deque(maxlen=2000)
        self.recent_realized: deque[float] = deque(maxlen=20)
        self.loss_streak: int = 0
        self.cooldown_bars_left: int = 0
        self.trend_mismatch_streak: int = 0
        self.position_exit_streak: int = 0
        self.adaptive_enter_offset: float = 0.0
        self.adaptive_agreement_offset: float = 0.0

        self.min_live_kelly = float(os.getenv("FUSE_MIN_LIVE_KELLY", "0.04"))
        self.governor_hard_stop = float(os.getenv("GOVERNOR_HARD_STOP", "0.025"))
        self.governor_max_hold = int(os.getenv("GOVERNOR_MAX_HOLD", "36"))
        self.governor_trail_arm = float(os.getenv("GOVERNOR_TRAIL_ARM", "0.012"))
        self.governor_trail_gap = float(os.getenv("GOVERNOR_TRAIL_GAP", "0.008"))
        self.governor_vol_scale_enable = _env_flag("GOVERNOR_VOL_SCALE_ENABLE", True)
        self.governor_cooldown_enable = _env_flag("GOVERNOR_COOLDOWN_ENABLE", False)
        self.governor_trend_exit_enable = _env_flag("GOVERNOR_TREND_EXIT_ENABLE", False)
        self.governor_trend_exit_hold_bars = int(os.getenv("GOVERNOR_TREND_EXIT_HOLD_BARS", "24"))
        self.governor_trend_exit_confirm_bars = int(os.getenv("GOVERNOR_TREND_EXIT_CONFIRM_BARS", "2"))
        self.governor_trend_exit_score = float(os.getenv("GOVERNOR_TREND_EXIT_SCORE", "0.20"))
        self.governor_trend_exit_quality = float(os.getenv("GOVERNOR_TREND_EXIT_QUALITY", "0.000"))
        self.governor_vae_block_ratio = float(os.getenv("GOVERNOR_VAE_BLOCK_RATIO", "1.35"))

        self.hibernation_enable = _env_flag("GOVERNOR_HIBERNATION_ENABLE", True)
        self.hibernation_score_th = float(os.getenv("GOVERNOR_HIBERNATION_SCORE_TH", "0.85"))
        self.entry_reco_enable = _env_flag("GOVERNOR_ENTRY_RECO_ENABLE", False)
        self.enforce_bar_fill_price = _env_flag("GOVERNOR_ENFORCE_BAR_FILL_PRICE", True)
        self.entry_reco_min_strength = float(os.getenv("GOVERNOR_ENTRY_RECO_MIN_STRENGTH", "0.55"))
        self.entry_reco_min_quality = float(os.getenv("GOVERNOR_ENTRY_RECO_MIN_QUALITY", "-0.002"))
        self.entry_reco_max_offset = float(os.getenv("GOVERNOR_ENTRY_RECO_MAX_OFFSET", "0.0045"))
        self.entry_reco_price_buffer = float(os.getenv("GOVERNOR_ENTRY_RECO_PRICE_BUFFER", "0.0002"))
        self.trade_fee = float(os.getenv("LIVE_FEE_RATE", "0.0005"))
        self.trade_slip = float(os.getenv("LIVE_SLIP_RATE", "0.0002"))
        self.taker_fee = float(os.getenv("LIVE_TAKER_FEE_RATE", str(self.trade_fee)))
        self.maker_fee = float(os.getenv("LIVE_MAKER_FEE_RATE", str(self.trade_fee * float(os.getenv("LIVE_MAKER_FEE_MULT", "0.20")))))
        self.live_state_path = os.getenv("GOVERNOR_LIVE_STATE_PATH", "data/ensemble/governor_live_state.json")
        
        self.adaptive_gate_enable = _env_flag("GOVERNOR_ADAPTIVE_GATE_ENABLE", True)
        self.adaptive_gate_pnl_window = int(os.getenv("GOVERNOR_ADAPTIVE_GATE_PNL_WINDOW", "8"))
        self.adaptive_gate_enter_step = float(os.getenv("GOVERNOR_ADAPTIVE_GATE_ENTER_STEP", "0.01"))
        self.adaptive_gate_agreement_step = float(os.getenv("GOVERNOR_ADAPTIVE_GATE_AGREEMENT_STEP", "0.01"))
        self.adaptive_gate_loosen_step = float(os.getenv("GOVERNOR_ADAPTIVE_GATE_LOOSEN_STEP", "0.02"))
        self.adaptive_gate_enter_min = float(os.getenv("GOVERNOR_ADAPTIVE_GATE_ENTER_MIN", "-0.18"))
        self.adaptive_gate_enter_max = float(os.getenv("GOVERNOR_ADAPTIVE_GATE_ENTER_MAX", "0.08"))
        self.adaptive_gate_agreement_min = float(os.getenv("GOVERNOR_ADAPTIVE_GATE_AGREEMENT_MIN", "-0.14"))
        self.adaptive_gate_agreement_max = float(os.getenv("GOVERNOR_ADAPTIVE_GATE_AGREEMENT_MAX", "0.08"))
        self.adaptive_gate_flat_bars = int(os.getenv("GOVERNOR_ADAPTIVE_GATE_FLAT_BARS", "10"))
        self.adaptive_gate_loss_streak_th = int(os.getenv("GOVERNOR_ADAPTIVE_GATE_LOSS_STREAK_TH", "4"))
        self.adaptive_gate_bad_pnl_cut = float(os.getenv("GOVERNOR_ADAPTIVE_GATE_BAD_PNL_CUT", "-0.015"))
        self.adaptive_gate_good_pnl_cut = float(os.getenv("GOVERNOR_ADAPTIVE_GATE_GOOD_PNL_CUT", "0.006"))
        self.adaptive_flat_cycles: int = 0

        self.step_stop_enable = _env_flag("GOVERNOR_STEP_STOP_ENABLE", True)
        self.step_stop_levels: list[tuple[float, float]] = [
            (0.020, 0.012), (0.015, 0.007), (0.010, 0.003), (0.006, 0.000),
        ]

        self._load_live_state()

    def _set_position_sizing(
        self,
        exposure: float | None = None,
        fraction: float | None = None,
        leverage_mult: float | None = None,
    ) -> None:
        if exposure is not None:
            exp = float(np.clip(float(exposure), 0.0, self.exposure_cap))
            frac = float(np.clip(min(exp, 1.0), 0.0, 1.0))
            lev_mult = float(np.clip(exp / max(frac, 1e-8), 1.0, self.exposure_cap)) if frac > 0.0 else 1.0
        else:
            frac = float(np.clip(float(fraction if fraction is not None else self.position_fraction), 0.0, 1.0))
            lev_mult = float(np.clip(float(leverage_mult if leverage_mult is not None else self.execution_leverage), 1.0, self.exposure_cap))
            exp = float(np.clip(frac * lev_mult, 0.0, self.exposure_cap))
        self.position_fraction = frac
        self.execution_leverage = lev_mult
        self.current_leverage = exp

    @staticmethod
    def _new_trade_id(ts_like=None) -> str:
        if ts_like is None:
            ts = pd.Timestamp.utcnow()
        else:
            ts = pd.Timestamp(ts_like)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC")
        else:
            ts = ts.tz_localize("UTC")
        return ts.strftime("trade-%Y%m%dT%H%M%S.%fZ")

    def _clear_open_model_identity(self) -> None:
        self.open_model_version = ""
        self.open_model_id = ""
        self.open_model_path = ""
        self.open_model_sleeve = ""
        self.open_source = ""

    def _set_open_model_identity(self, model_context: dict | None, *, source: str = "") -> None:
        ctx = dict(model_context or {})
        model_id = str(ctx.get("model_id", "") or "").strip()
        model_version = str(ctx.get("model_version", "") or "").strip()
        if not model_id:
            raise RuntimeError("open_trade_model_identity_missing:model_id")
        self.open_model_id = model_id
        self.open_model_version = model_version
        self.open_model_path = str(ctx.get("model_path", "") or "")
        self.open_model_sleeve = str(ctx.get("model_sleeve", "") or "")
        self.open_source = str(source or ctx.get("source", "") or "")

    def _apply_open_model_identity_to_audit(self, audit: dict, snapshot: dict, *, kind: str) -> dict:
        out = dict(audit or {})
        snap = dict(snapshot or {})
        open_model_id = str(snap.get("open_model_id", self.open_model_id) or "").strip()
        open_model_version = str(snap.get("open_model_version", self.open_model_version) or "").strip()
        open_model_path = str(snap.get("open_model_path", self.open_model_path) or "")
        open_model_sleeve = str(snap.get("open_model_sleeve", self.open_model_sleeve) or "")
        if not open_model_id:
            trade_id = str(snap.get("trade_id", self.open_trade_id) or "")
            raise RuntimeError(f"{str(kind).lower()}_trade_open_model_identity_missing:trade_id={trade_id}")
        prefix = str(kind).lower()
        out[f"{prefix}_decision_model_id"] = str(out.get("model_id", "") or "")
        out[f"{prefix}_decision_model_version"] = str(out.get("model_version", "") or "")
        out[f"{prefix}_decision_model_path"] = str(out.get("model_path", "") or "")
        out[f"{prefix}_decision_model_sleeve"] = str(out.get("model_sleeve", "") or "")
        out["model_id"] = open_model_id
        out["model_version"] = open_model_version
        out["model_path"] = open_model_path
        out["model_sleeve"] = open_model_sleeve
        out["open_model_id"] = open_model_id
        out["open_model_version"] = open_model_version
        out["open_model_path"] = open_model_path
        out["open_model_sleeve"] = open_model_sleeve
        out["open_source"] = str(snap.get("open_source", self.open_source) or "")
        return out

    def position_snapshot(self) -> dict:
        pos = self.pos if self.pos in {"LONG", "SHORT"} else None
        return {
            "pos": pos,
            "entry_price": float(self.entry_price or 0.0),
            "hold_bars": int(self.hold_count or 0),
            "position_fraction": float(self.position_fraction or 0.0),
            "margin_fraction": float(self.position_fraction or 0.0),
            "execution_leverage": float(self.execution_leverage or 1.0),
            "notional_exposure": float(self.current_leverage or 0.0),
            "total_exposure": float(self.current_leverage or 0.0),
            "position_realized_pnl_frac": float(self.position_realized_pnl_frac or 0.0),
            "trade_id": str(self.open_trade_id or ""),
            "opened_at": str(self.opened_at or ""),
            "decision_at": str(self.decision_at or ""),
            "entry_price_source": str(self.entry_price_source or ""),
            "entry_decision_price": float(self.entry_decision_price or 0.0),
            "exchange_entry_price": float(self.exchange_entry_price or 0.0),
            "entry_execution_liquidity": str(self.entry_execution_liquidity or ""),
            "entry_execution_route": str(self.entry_execution_route or ""),
            "entry_execution_order_type": str(self.entry_execution_order_type or ""),
            "open_model_version": str(self.open_model_version or ""),
            "open_model_id": str(self.open_model_id or ""),
            "open_model_path": str(self.open_model_path or ""),
            "open_model_sleeve": str(self.open_model_sleeve or ""),
            "open_source": str(self.open_source or ""),
        }

    @staticmethod
    def _is_real_execution_liquidity(liquidity: str | None) -> bool:
        s = str(liquidity or "").strip().lower()
        return bool(s and "dry_run" not in s and "shadow" not in s and "miss" not in s)

    def _fee_rate_for_liquidity(self, liquidity: str | None, *, default_synthetic: bool = True) -> tuple[float, str]:
        s = str(liquidity or "").strip().lower()
        if self._is_real_execution_liquidity(s):
            if "maker" in s and "taker" not in s:
                return float(self.maker_fee), "maker"
            if "maker" in s and "taker" in s:
                # Without fill-level quantity splits, use the conservative taker
                # rate for mixed maker->taker routes.
                return float(self.taker_fee), "maker_taker_conservative_taker"
            if "taker" in s or "market" in s:
                return float(self.taker_fee), "taker"
        if default_synthetic:
            return float(self.trade_fee), "synthetic_default"
        return 0.0, "none"

    def _trade_math(
        self,
        side: str,
        entry_price: float,
        exit_price: float,
        exposure: float,
        *,
        entry_liquidity: str | None = None,
        exit_liquidity: str | None = None,
    ) -> dict:
        side_u = str(side or "").upper()
        entry = float(entry_price or 0.0)
        exit_raw = float(exit_price or 0.0)
        lev = float(np.clip(float(exposure or 0.0), 0.0, self.exposure_cap))
        if side_u not in {"LONG", "SHORT"} or entry <= 0.0 or exit_raw <= 0.0:
            return {
                "entry_exec_price": 0.0,
                "exit_exec_price": 0.0,
                "gross_return_frac": 0.0,
                "entry_fee_rate": 0.0,
                "exit_fee_rate": 0.0,
                "roundtrip_fee_rate": 0.0,
                "fee_model": "invalid",
                "pnl_frac": 0.0,
                "pnl_pct": 0.0,
            }
        entry_fee, entry_fee_model = self._fee_rate_for_liquidity(entry_liquidity)
        exit_fee, exit_fee_model = self._fee_rate_for_liquidity(exit_liquidity)
        entry_is_real = self._is_real_execution_liquidity(entry_liquidity)
        exit_is_real = self._is_real_execution_liquidity(exit_liquidity)
        if side_u == "LONG":
            entry_exec = entry if entry_is_real else entry * (1.0 + self.trade_slip)
            exit_exec = exit_raw if exit_is_real else exit_raw * (1.0 - self.trade_slip)
            gross = (exit_exec - entry_exec) / max(entry_exec, 1e-8)
        else:
            entry_exec = entry if entry_is_real else entry * (1.0 - self.trade_slip)
            exit_exec = exit_raw if exit_is_real else exit_raw * (1.0 + self.trade_slip)
            gross = (entry_exec - exit_exec) / max(abs(entry_exec), 1e-8)
        pnl_frac = float(gross * lev - ((entry_fee + exit_fee) * lev))
        return {
            "entry_exec_price": float(entry_exec),
            "exit_exec_price": float(exit_exec),
            "gross_return_frac": float(gross),
            "entry_fee_rate": float(entry_fee),
            "exit_fee_rate": float(exit_fee),
            "roundtrip_fee_rate": float(entry_fee + exit_fee),
            "entry_fee_model": str(entry_fee_model),
            "exit_fee_model": str(exit_fee_model),
            "fee_model": f"{entry_fee_model}+{exit_fee_model}",
            "fee_cost_frac": float((entry_fee + exit_fee) * lev),
            "pnl_frac": pnl_frac,
            "pnl_pct": float(pnl_frac * 100.0),
        }

    @staticmethod
    def _journal_event_recorded_at() -> str:
        return pd.Timestamp.now(tz="Asia/Seoul").isoformat()

    @staticmethod
    def _order_fill_price(order: dict | None) -> float:
        row = dict(order or {})
        info = dict(row.get("info", {}) or {})
        for key in ("average", "avgPrice", "price"):
            val = _safe_float(row.get(key, info.get(key, 0.0)), 0.0)
            if val > 0.0:
                return float(val)
        return 0.0

    @staticmethod
    def _order_execution_liquidity(order: dict | None) -> str:
        row = dict(order or {})
        route = dict(row.get("execution_route", {}) or {})
        order_type = str(row.get("type", "") or "").lower()
        route_name = str(route.get("route", "") or "").lower()
        tif = str(row.get("timeInForce", row.get("time_in_force", "")) or "").upper()
        post_only = bool(row.get("postOnly", row.get("post_only", False)))
        dry_run = bool(row.get("dry_run", False))
        has_maker_attempt = bool(row.get("maker_order")) or post_only or tif == "GTX" or route_name == "post_only_limit" or order_type == "limit"
        has_taker_attempt = bool(row.get("fallback_order")) or order_type == "market" or route_name == "market"

        # Dry-run/shadow orders are route simulations, not exchange-confirmed
        # fills. Keep them visible, but do not label them as real maker/taker.
        if dry_run:
            if has_maker_attempt and has_taker_attempt:
                return "maker_taker_dry_run"
            if has_maker_attempt:
                return "maker_dry_run"
            if has_taker_attempt:
                return "taker_dry_run"
            return ""

        fallback = dict(row.get("fallback_order", {}) or {})
        wait = dict(row.get("maker_wait_result", {}) or {})
        maker_filled_qty = _safe_float(wait.get("filled", 0.0), 0.0)
        maker_status = str(wait.get("status", "") or "").lower()
        maker_filled = bool(has_maker_attempt and (maker_status == "filled" or maker_filled_qty > 1e-12))
        fallback_filled = bool(fallback) and not bool(fallback.get("unfilled_without_fallback", False))

        if maker_filled and fallback_filled:
            return "maker_taker"
        if maker_filled:
            return "maker"
        if fallback_filled or (has_taker_attempt and not has_maker_attempt):
            return "taker"
        if has_maker_attempt and bool(row.get("unfilled_without_fallback", False)):
            return "maker_miss"
        return ""

    @staticmethod
    def _merge_execution_liquidity(values: list[str]) -> str:
        vals = {str(v or "").lower() for v in values if str(v or "").strip()}
        if "maker_taker_dry_run" in vals or ("maker_dry_run" in vals and "taker_dry_run" in vals):
            return "maker_taker_dry_run"
        if "maker_dry_run" in vals:
            return "maker_dry_run"
        if "taker_dry_run" in vals:
            return "taker_dry_run"
        if "maker_miss" in vals and "taker" not in vals:
            return "maker_miss"
        if "maker_taker" in vals or ("maker" in vals and "taker" in vals):
            return "maker_taker"
        if "maker" in vals:
            return "maker"
        if "taker" in vals:
            return "taker"
        return ""

    @classmethod
    def _execution_route_summary(cls, orders: list[dict], *, reduce_only: bool) -> dict:
        liqs: list[str] = []
        routes: list[str] = []
        types: list[str] = []
        for order in orders:
            row = dict(order or {})
            if bool(row.get("reduceOnly", row.get("reduce_only", False))) != bool(reduce_only):
                continue
            liq = cls._order_execution_liquidity(row)
            if liq:
                liqs.append(liq)
            route = dict(row.get("execution_route", {}) or {})
            route_name = str(route.get("route", "") or "")
            if route_name:
                routes.append(route_name)
            typ = str(row.get("type", "") or "")
            if typ:
                types.append(typ)
            if row.get("maker_order"):
                types.append("limit")
            if row.get("fallback_order"):
                types.append("market")
        return {
            "liquidity": cls._merge_execution_liquidity(liqs),
            "route": "|".join(dict.fromkeys(routes)),
            "order_type": "|".join(dict.fromkeys(types)),
        }

    @classmethod
    def _exchange_execution_audit_fields(cls, live_execution: dict | None, *, kind: str) -> dict:
        live = dict(live_execution or {})
        orders = list(live.get("orders", []) or [])
        out = {
            "exchange_execution_enabled": bool(live.get("enabled", False)),
            "exchange_execution_dry_run": bool(live.get("dry_run", True)),
            "exchange_execution_status": str(live.get("status", "")),
            "exchange_order_count": int(len(orders)),
            "exchange_fill_price_source": "",
            "exchange_entry_price": 0.0,
            "exchange_exit_price": 0.0,
            "entry_execution_liquidity": "",
            "entry_execution_route": "",
            "entry_execution_order_type": "",
            "exit_execution_liquidity": "",
            "exit_execution_route": "",
            "exit_execution_order_type": "",
        }
        entry_route = cls._execution_route_summary(orders, reduce_only=False)
        exit_route = cls._execution_route_summary(orders, reduce_only=True)
        out.update(
            {
                "entry_execution_liquidity": str(entry_route.get("liquidity", "")),
                "entry_execution_route": str(entry_route.get("route", "")),
                "entry_execution_order_type": str(entry_route.get("order_type", "")),
                "exit_execution_liquidity": str(exit_route.get("liquidity", "")),
                "exit_execution_route": str(exit_route.get("route", "")),
                "exit_execution_order_type": str(exit_route.get("order_type", "")),
            }
        )
        post = dict(live.get("post_position", {}) or {})
        post_entry = _safe_float(post.get("entry_price", 0.0), 0.0)
        if str(kind).upper() in {"OPEN", "RESIZE"} and post_entry > 0.0:
            out["exchange_entry_price"] = float(post_entry)
            out["exchange_fill_price_source"] = "post_position.entry_price"

        for order in orders:
            row = dict(order or {})
            fill = cls._order_fill_price(row)
            if fill <= 0.0:
                continue
            reduce_only = bool(row.get("reduceOnly", row.get("reduce_only", False)))
            if str(kind).upper() == "CLOSE" and reduce_only:
                out["exchange_exit_price"] = float(fill)
                out["exchange_fill_price_source"] = "reduce_order.fill_price"
            elif str(kind).upper() == "OPEN" and not reduce_only:
                out["exchange_entry_price"] = float(fill)
                out["exchange_fill_price_source"] = "open_order.fill_price"
            elif str(kind).upper() == "RESIZE":
                key = "exchange_exit_price" if reduce_only else "exchange_entry_price"
                out[key] = float(fill)
                out["exchange_fill_price_source"] = "resize_order.fill_price"
        return out

    @staticmethod
    def _journal_jsonable(value):
        if isinstance(value, dict):
            return {str(k): GovernorPositionRouter._journal_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [GovernorPositionRouter._journal_jsonable(v) for v in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return [GovernorPositionRouter._journal_jsonable(v) for v in value.tolist()]
        if isinstance(value, (pd.Timestamp, datetime)):
            return str(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    @classmethod
    def _omega5_journal_audit_fields(cls, ctx: dict) -> dict:
        trace = dict(ctx.get("sleeve_trace", {}) or {})
        if not trace:
            return {}
        model_id = str(trace.get("model_id", "") or "")
        source_model_id = str(trace.get("source_model_id", "") or "")
        if model_id != OMEGA5_MODEL_ID and source_model_id != OMEGA5_SOURCE_MODEL_ID:
            return {}
        parent_trace = dict(trace.get("parent_trace", {}) or {})
        first_veto = dict(trace.get("first_veto", {}) or {})
        second_veto = dict(trace.get("second_veto", {}) or {})
        event_risk = dict(trace.get("event_risk", {}) or {})
        parent_action = int(trace.get("parent_action", 0) or 0)
        parent_side = int(trace.get("parent_side", 0) or 0)
        return {
            "parent_action": parent_action,
            "parent_side": parent_side,
            "omega5_trace_present": True,
            "omega5_trace_schema_version": "omega5.trade_journal_trace.v1",
            "omega5_sizing_trace": cls._journal_jsonable(trace),
            "omega5_model_id": model_id,
            "omega5_model_version": str(trace.get("model_version", "") or ""),
            "omega5_source_model_id": source_model_id,
            "omega5_parent_model_id": str(trace.get("parent_model_id", "") or ""),
            "omega5_reference_model_id": str(trace.get("reference_model_id", "") or ""),
            "omega5_reason": str(trace.get("omega5_reason", "") or ""),
            "omega5_parent_action": parent_action,
            "omega5_parent_side": parent_side,
            "omega5_parent_notional_exposure": _safe_float(trace.get("parent_notional_exposure", 0.0), 0.0),
            "omega5_parent_quality_score": _safe_float(trace.get("parent_quality_score", 0.0), 0.0),
            "omega5_parent_confidence": _safe_float(trace.get("parent_confidence", 0.0), 0.0),
            "omega5_reference_exposure_factor": _safe_float(trace.get("reference_exposure_factor", 0.0), 0.0),
            "omega5_reference_notional_exposure": _safe_float(trace.get("reference_notional_exposure", 0.0), 0.0),
            "omega5_exposure_factor": _safe_float(trace.get("exposure_factor", 0.0), 0.0),
            "omega5_final_notional_exposure": _safe_float(trace.get("notional_exposure", 0.0), 0.0),
            "omega5_margin_fraction": _safe_float(trace.get("position_fraction", 0.0), 0.0),
            "omega5_leverage": _safe_float(trace.get("leverage", 0.0), 0.0),
            "omega5_leverage_cap": _safe_float(trace.get("leverage_cap", 0.0), 0.0),
            "omega5_max_margin_fraction": _safe_float(trace.get("max_margin_fraction", 0.0), 0.0),
            "omega5_tp_price_move": _safe_float(trace.get("tp_price_move", 0.0), 0.0),
            "omega5_sl_price_move": _safe_float(trace.get("sl_price_move", 0.0), 0.0),
            "omega5_take_profit": _safe_float(trace.get("take_profit", 0.0), 0.0),
            "omega5_stop_loss": _safe_float(trace.get("stop_loss", 0.0), 0.0),
            "omega5_max_hold_bars": int(trace.get("max_hold_bars", 0) or 0),
            "omega5_first_veto_feature": str(first_veto.get("feature", "") or ""),
            "omega5_first_veto_op": str(first_veto.get("op", "") or ""),
            "omega5_first_veto_threshold": _safe_float(first_veto.get("threshold", 0.0), 0.0),
            "omega5_first_veto_value": _safe_float(trace.get("first_veto_value", 0.0), 0.0),
            "omega5_second_veto_feature": str(second_veto.get("feature", "") or ""),
            "omega5_second_veto_op": str(second_veto.get("op", "") or ""),
            "omega5_second_veto_threshold": _safe_float(second_veto.get("threshold", 0.0), 0.0),
            "omega5_second_veto_value": _safe_float(trace.get("second_veto_value", 0.0), 0.0),
            "omega5_event_risk_policy_id": str(event_risk.get("policy_id", "") or ""),
            "omega5_event_risk_macro_entry_veto": bool(event_risk.get("macro_entry_veto", False)),
            "omega5_event_risk_macro_event_names": "|".join(
                str(x) for x in list(event_risk.get("macro_event_names", []) or [])
            ),
            "omega5_event_risk_shock_haircut": bool(event_risk.get("shock_haircut", False)),
            "omega5_event_risk_notional_scale": _safe_float(trace.get("event_risk_notional_scale", 1.0), 1.0),
            "omega5_pre_event_risk_notional_exposure": _safe_float(
                trace.get("pre_event_risk_notional_exposure", 0.0),
                0.0,
            ),
            "omega5_event_risk_jump_z": _safe_float(event_risk.get("jump_z", 0.0), 0.0),
            "omega5_event_risk_ret_1h_past": _safe_float(event_risk.get("ret_1h_past", 0.0), 0.0),
            "omega5_event_risk_ret_4h_past": _safe_float(event_risk.get("ret_4h_past", 0.0), 0.0),
            "omega5_parent_ledger_replay_used": bool(parent_trace.get("ledger_replay_used", True)),
            "omega5_source_policy_interval_adapter": bool(parent_trace.get("source_policy_interval_adapter", False)),
            "omega5_reference_policy_entry_event_adapter": bool(
                parent_trace.get("reference_policy_entry_event_adapter", False)
            ),
            "omega5_source_parent_policy_row": int(parent_trace.get("source_parent_policy_row", -1) or -1),
            "omega5_source_parent_policy_artifact": str(parent_trace.get("source_parent_policy_artifact", "") or ""),
            "omega5_source_parent_live_native_adapter": bool(
                parent_trace.get("source_parent_live_native_adapter", False)
            ),
            "omega5_source_parent_predictive_artifact": str(
                parent_trace.get("source_parent_predictive_artifact", "") or ""
            ),
            "omega5_source_parent_component_report": str(
                parent_trace.get("source_parent_component_report", "") or ""
            ),
            "omega5_source_parent_component_bundle": str(
                parent_trace.get("source_parent_component_bundle", "") or ""
            ),
            "omega5_source_parent_component_sidecar": str(
                parent_trace.get("source_parent_component_sidecar", "") or ""
            ),
            "omega5_source_parent_loss_governor_scale": _safe_float(
                parent_trace.get("loss_governor_scale", 0.0),
                0.0,
            ),
            "omega5_source_parent_policy_entry_timestamp": str(
                parent_trace.get("source_parent_policy_entry_timestamp", "") or ""
            ),
            "omega5_source_parent_policy_exit_timestamp": str(
                parent_trace.get("source_parent_policy_exit_timestamp", "") or ""
            ),
            "omega5_reference_policy_row": int(parent_trace.get("reference_policy_row", -1) or -1),
            "omega5_reference_policy_artifact": str(parent_trace.get("reference_policy_artifact", "") or ""),
            "omega5_reference_policy_entry_timestamp": str(
                parent_trace.get("reference_policy_entry_timestamp", "") or ""
            ),
            "omega5_reference_policy_exit_timestamp": str(
                parent_trace.get("reference_policy_exit_timestamp", "") or ""
            ),
            "omega5_reference_policy_reason": str(parent_trace.get("reference_policy_reason", "") or ""),
            "omega5_reference_policy_raw_exit_price_move": _safe_float(
                parent_trace.get("reference_policy_raw_exit_price_move", 0.0),
                0.0,
            ),
            "omega5_reference_policy_net_per_notional": _safe_float(
                parent_trace.get("reference_policy_net_per_notional", 0.0),
                0.0,
            ),
            "omega5_reference_policy_roundtrip_cost": _safe_float(
                parent_trace.get("reference_policy_roundtrip_cost", 0.0),
                0.0,
            ),
        }

    @classmethod
    def _journal_audit_fields(cls, audit_context: dict | None, *, kind: str) -> dict:
        ctx = dict(audit_context or {})
        out = {
            "audit_schema_version": "trade_journal.audit.v2",
            "ledger_ts_kind": str(ctx.get("ledger_ts_kind", "decision_bar")),
            "decision_made_at_kst": str(ctx.get("decision_made_at_kst", "")),
            "decision_bar_ts": str(ctx.get("decision_bar_ts", "")),
            "decision_bar_utc": str(ctx.get("decision_bar_utc", "")),
            "decision_bar_open": _safe_float(ctx.get("decision_bar_open", 0.0), 0.0),
            "decision_bar_high": _safe_float(ctx.get("decision_bar_high", 0.0), 0.0),
            "decision_bar_low": _safe_float(ctx.get("decision_bar_low", 0.0), 0.0),
            "decision_bar_close": _safe_float(ctx.get("decision_bar_close", ctx.get("decision_price", 0.0)), 0.0),
            "decision_bar_volume": _safe_float(ctx.get("decision_bar_volume", 0.0), 0.0),
            "decision_bar_is_complete": bool(ctx.get("decision_bar_is_complete", False)),
            "decision_price": _safe_float(ctx.get("decision_price", 0.0), 0.0),
            "decision_price_source": str(ctx.get("decision_price_source", "eth_buffer.close[-1]")),
            "execution_bar_ts": str(ctx.get("execution_bar_ts", "")),
            "execution_bar_utc": str(ctx.get("execution_bar_utc", "")),
            "execution_bar_open": _safe_float(ctx.get("execution_bar_open", ctx.get("execution_price", 0.0)), 0.0),
            "execution_bar_high": _safe_float(ctx.get("execution_bar_high", ctx.get("execution_price", 0.0)), 0.0),
            "execution_bar_low": _safe_float(ctx.get("execution_bar_low", ctx.get("execution_price", 0.0)), 0.0),
            "execution_bar_close": _safe_float(ctx.get("execution_bar_close", ctx.get("execution_price", 0.0)), 0.0),
            "execution_bar_volume": _safe_float(ctx.get("execution_bar_volume", 0.0), 0.0),
            "execution_bar_is_current": bool(ctx.get("execution_bar_is_current", False)),
            "execution_price": _safe_float(ctx.get("execution_price", ctx.get("decision_price", 0.0)), 0.0),
            "execution_price_source": str(ctx.get("execution_price_source", "")),
            "execution_delay_sec": _safe_float(ctx.get("execution_delay_sec", 0.0), 0.0),
            "execution_delay_late": bool(ctx.get("execution_delay_late", False)),
            "execution_delay_mode": str(ctx.get("execution_delay_mode", "")),
            "ai_timing": dict(ctx.get("ai_timing", {}) or {}),
            "model_version": str(ctx.get("model_version", "")),
            "model_id": str(ctx.get("model_id", "")),
            "model_path": str(ctx.get("model_path", "")),
            "model_sleeve": str(ctx.get("model_sleeve", "")),
            "scout_prob": _safe_float(ctx.get("scout_prob", 0.0), 0.0),
            "scout_frac": _safe_float(ctx.get("scout_frac", 0.0), 0.0),
            "scout_probability_threshold": _safe_float(ctx.get("scout_probability_threshold", 0.0), 0.0),
            "scout_cost_pass": bool(ctx.get("scout_cost_pass", False)),
            "learned_config": dict(ctx.get("learned_config", {}) or {}),
            "take_profit": _safe_float(ctx.get("take_profit", 0.0), 0.0),
            "stop_loss": _safe_float(ctx.get("stop_loss", 0.0), 0.0),
            "max_hold_bars": int(ctx.get("max_hold_bars", 0) or 0),
            "max_hold_remaining_bars": int(ctx.get("max_hold_remaining_bars", 0) or 0),
            "take_profit_price": _safe_float(ctx.get("take_profit_price", 0.0), 0.0),
            "stop_price": _safe_float(ctx.get("stop_price", 0.0), 0.0),
            "trailing_stop_price": _safe_float(ctx.get("trailing_stop_price", 0.0), 0.0),
            "effective_take_profit": _safe_float(ctx.get("effective_take_profit", ctx.get("take_profit", 0.0)), 0.0),
            "effective_stop_loss": _safe_float(ctx.get("effective_stop_loss", ctx.get("stop_loss", 0.0)), 0.0),
            "v31_q_long": _safe_float(ctx.get("v31_q_long", 0.0), 0.0),
            "v31_q_short": _safe_float(ctx.get("v31_q_short", 0.0), 0.0),
            "v31_q_long_raw": _safe_float(ctx.get("v31_q_long_raw", 0.0), 0.0),
            "v31_q_short_raw": _safe_float(ctx.get("v31_q_short_raw", 0.0), 0.0),
            "v31_edge": _safe_float(ctx.get("v31_edge", 0.0), 0.0),
            "v31_margin": _safe_float(ctx.get("v31_margin", 0.0), 0.0),
            "v31_raw_margin": _safe_float(ctx.get("v31_raw_margin", 0.0), 0.0),
            "v31_selected_side": str(ctx.get("v31_selected_side", "")),
            "v31_pass_gate": bool(ctx.get("v31_pass_gate", False)),
            "v31_guard_reason": str(ctx.get("v31_guard_reason", "")),
            "v31_transition_risk": _safe_float(ctx.get("v31_transition_risk", 0.0), 0.0),
            "parent_action": int(ctx.get("parent_action", 0) or 0),
            "parent_side": int(ctx.get("parent_side", 0) or 0),
            "omega5_source_roundtrip_cost": _safe_float(ctx.get("omega5_source_roundtrip_cost", 0.0), 0.0),
            "omega5_source_exit_reason": str(ctx.get("omega5_source_exit_reason", "") or ""),
            "omega5_source_exit_price_move": _safe_float(ctx.get("omega5_source_exit_price_move", 0.0), 0.0),
            "teacher_gate_result": str(ctx.get("teacher_gate_result", "")),
            "teacher_pred_action": int(ctx.get("teacher_pred_action", 0) or 0),
            "teacher_confidence": _safe_float(ctx.get("teacher_confidence", 0.0), 0.0),
            "teacher_quality": _safe_float(ctx.get("teacher_quality", 0.0), 0.0),
            "teacher_keep_parent": bool(ctx.get("teacher_keep_parent", False)),
        }
        out.update(cls._omega5_journal_audit_fields(ctx))
        out.update(cls._exchange_execution_audit_fields(ctx.get("live_execution"), kind=kind))
        return out

    def build_open_trade_payload(
        self,
        snapshot: dict,
        timestamp_kst,
        event: str,
        regime_name: str,
        source: str = "",
        reason: str = "",
        audit_context: dict | None = None,
    ) -> dict:
        snap = dict(snapshot or {})
        side = str(snap.get("pos") or "").upper()
        entry_price = float(snap.get("entry_price", 0.0) or 0.0)
        exposure = float(snap.get("notional_exposure", snap.get("total_exposure", 0.0)) or 0.0)
        event_recorded_at = self._journal_event_recorded_at()
        audit = self._journal_audit_fields(audit_context, kind="OPEN")
        self._set_open_model_identity(audit, source=source)
        snap["open_model_id"] = self.open_model_id
        snap["open_model_version"] = self.open_model_version
        snap["open_model_path"] = self.open_model_path
        snap["open_model_sleeve"] = self.open_model_sleeve
        snap["open_source"] = self.open_source
        audit["open_model_id"] = self.open_model_id
        audit["open_model_version"] = self.open_model_version
        audit["open_model_path"] = self.open_model_path
        audit["open_model_sleeve"] = self.open_model_sleeve
        audit["open_source"] = self.open_source
        if self.pos in {"LONG", "SHORT"}:
            self._save_live_state()
        if not str(audit.get("entry_execution_liquidity", "") or ""):
            audit["entry_execution_liquidity"] = str(snap.get("entry_execution_liquidity", "") or "")
        math = self._trade_math(
            side,
            entry_price,
            entry_price,
            exposure,
            entry_liquidity=str(audit.get("entry_execution_liquidity", "") or ""),
            exit_liquidity="",
        )
        entry_exec_price = float(math.get("entry_exec_price", 0.0))
        return {
            "schema_version": "trade_journal.v1",
            "ts": str(timestamp_kst),
            "kind": "OPEN",
            "event": str(event),
            "side": side,
            "trade_id": str(snap.get("trade_id", "") or ""),
            "decision_at": str(snap.get("decision_at", "") or str(timestamp_kst)),
            "opened_at": str(snap.get("opened_at", "") or str(timestamp_kst)),
            "actual_opened_at": str(snap.get("opened_at", "") or event_recorded_at),
            "event_recorded_at": str(event_recorded_at),
            "entry_price": float(entry_price),
            "entry_price_source": str(snap.get("entry_price_source", "") or "decision_current_price"),
            "entry_decision_price": float(snap.get("entry_decision_price", audit.get("decision_price", 0.0)) or 0.0),
            "entry_exec_price": float(entry_exec_price),
            "entry_exec_price_kind": "synthetic_fee_slippage_model",
            "synthetic_entry_exec_price": float(entry_exec_price),
            "entry_fee_rate": float(math.get("entry_fee_rate", 0.0)),
            "entry_fee_model": str(math.get("entry_fee_model", "")),
            "exit_fee_rate": float(math.get("exit_fee_rate", 0.0)),
            "exit_fee_model": str(math.get("exit_fee_model", "")),
            "roundtrip_fee_rate": float(math.get("roundtrip_fee_rate", 0.0)),
            "fee_model": str(math.get("fee_model", "")),
            "fee_cost_frac": float(math.get("fee_cost_frac", 0.0)),
            "hold_bars": int(snap.get("hold_bars", 0) or 0),
            "position_fraction": float(snap.get("position_fraction", 0.0) or 0.0),
            "margin_fraction": float(snap.get("margin_fraction", snap.get("position_fraction", 0.0)) or 0.0),
            "execution_leverage": float(snap.get("execution_leverage", 1.0) or 1.0),
            "notional_exposure": float(exposure),
            "total_exposure": float(exposure),
            "regime": str(regime_name),
            "source": str(source),
            "reason": str(reason),
            **audit,
        }

    def build_close_trade_payload(
        self,
        snapshot: dict,
        current_price: float,
        timestamp_kst,
        event: str,
        regime_name: str,
        source: str = "",
        reason: str = "",
        next_side: str | None = None,
        audit_context: dict | None = None,
    ) -> dict:
        snap = dict(snapshot or {})
        side = str(snap.get("pos") or "").upper()
        entry_price = float(snap.get("entry_price", 0.0) or 0.0)
        exposure = float(snap.get("notional_exposure", snap.get("total_exposure", 0.0)) or 0.0)
        event_recorded_at = self._journal_event_recorded_at()
        audit = self._journal_audit_fields(audit_context, kind="CLOSE")
        audit = self._apply_open_model_identity_to_audit(audit, snap, kind="CLOSE")
        if not str(audit.get("entry_execution_liquidity", "") or ""):
            audit["entry_execution_liquidity"] = str(snap.get("entry_execution_liquidity", "") or "")
        if not str(audit.get("entry_execution_route", "") or ""):
            audit["entry_execution_route"] = str(snap.get("entry_execution_route", "") or "")
        if not str(audit.get("entry_execution_order_type", "") or ""):
            audit["entry_execution_order_type"] = str(snap.get("entry_execution_order_type", "") or "")
        math = self._trade_math(
            side,
            entry_price,
            float(current_price or 0.0),
            exposure,
            entry_liquidity=str(audit.get("entry_execution_liquidity", "") or ""),
            exit_liquidity=str(audit.get("exit_execution_liquidity", "") or ""),
        )
        source_roundtrip_cost = _safe_float(audit.get("omega5_source_roundtrip_cost", 0.0), 0.0)
        is_omega5_source = (
            str(source or "").startswith(f"{OMEGA5_OWNER}|")
            or str(audit.get("model_id", "") or "") == OMEGA5_MODEL_ID
            or str(audit.get("open_model_id", "") or "") == OMEGA5_MODEL_ID
        )
        if is_omega5_source and source_roundtrip_cost > 0.0 and exposure > 0.0:
            gross = float(math.get("gross_return_frac", 0.0) or 0.0)
            pnl_frac = float((gross - source_roundtrip_cost) * exposure)
            math.update(
                {
                    "roundtrip_fee_rate": float(source_roundtrip_cost),
                    "entry_fee_rate": float(source_roundtrip_cost / 2.0),
                    "exit_fee_rate": float(source_roundtrip_cost / 2.0),
                    "entry_fee_model": "omega5_source_cost",
                    "exit_fee_model": "omega5_source_cost",
                    "fee_model": "omega5_source_roundtrip_cost",
                    "fee_cost_frac": float(source_roundtrip_cost * exposure),
                    "pnl_frac": pnl_frac,
                    "pnl_pct": float(pnl_frac * 100.0),
                }
            )
        entry_exec_price = float(math.get("entry_exec_price", 0.0))
        exit_exec_price = float(math.get("exit_exec_price", 0.0))
        return {
            "schema_version": "trade_journal.v1",
            "ts": str(timestamp_kst),
            "kind": "CLOSE",
            "event": str(event),
            "side": side,
            "trade_id": str(snap.get("trade_id", "") or ""),
            "decision_at": str(snap.get("decision_at", "") or ""),
            "opened_at": str(snap.get("opened_at", "") or ""),
            "closed_at": str(timestamp_kst),
            "actual_opened_at": str(snap.get("opened_at", "") or ""),
            "actual_closed_at": str(event_recorded_at),
            "event_recorded_at": str(event_recorded_at),
            "next_side": (str(next_side).upper() if next_side else None),
            "entry_price": float(entry_price),
            "entry_price_source": str(snap.get("entry_price_source", "") or ""),
            "entry_decision_price": float(snap.get("entry_decision_price", 0.0) or 0.0),
            "entry_exec_price": float(entry_exec_price),
            "entry_exec_price_kind": "synthetic_fee_slippage_model",
            "synthetic_entry_exec_price": float(entry_exec_price),
            "exit_price": float(current_price or 0.0),
            "exit_price_source": str(audit.get("execution_price_source", "") or "decision_current_price"),
            "exit_exec_price": float(exit_exec_price),
            "exit_exec_price_kind": "synthetic_fee_slippage_model",
            "synthetic_exit_exec_price": float(exit_exec_price),
            "gross_return_frac": float(math.get("gross_return_frac", 0.0)),
            "entry_fee_rate": float(math.get("entry_fee_rate", 0.0)),
            "entry_fee_model": str(math.get("entry_fee_model", "")),
            "exit_fee_rate": float(math.get("exit_fee_rate", 0.0)),
            "exit_fee_model": str(math.get("exit_fee_model", "")),
            "roundtrip_fee_rate": float(math.get("roundtrip_fee_rate", 0.0)),
            "fee_model": str(math.get("fee_model", "")),
            "fee_cost_frac": float(math.get("fee_cost_frac", 0.0)),
            "pnl_frac": float(math.get("pnl_frac", 0.0)),
            "pnl_pct": float(math.get("pnl_pct", 0.0)),
            "remaining_position_pnl_frac": float(math.get("pnl_frac", 0.0)),
            "position_realized_pnl_frac_before_close": float(snap.get("position_realized_pnl_frac", 0.0) or 0.0),
            "total_position_pnl_frac_est": float((snap.get("position_realized_pnl_frac", 0.0) or 0.0) + float(math.get("pnl_frac", 0.0))),
            "hold_bars": int(snap.get("hold_bars", 0) or 0),
            "position_fraction": float(snap.get("position_fraction", 0.0) or 0.0),
            "margin_fraction": float(snap.get("margin_fraction", snap.get("position_fraction", 0.0)) or 0.0),
            "execution_leverage": float(snap.get("execution_leverage", 1.0) or 1.0),
            "notional_exposure": float(exposure),
            "total_exposure": float(exposure),
            "regime": str(regime_name),
            "source": str(source),
            "reason": str(reason),
            **audit,
        }

    def build_resize_trade_payload(
        self,
        prev_snapshot: dict,
        new_snapshot: dict,
        current_price: float,
        timestamp_kst,
        regime_name: str,
        source: str = "",
        reason: str = "",
        audit_context: dict | None = None,
    ) -> dict:
        prev_snap = dict(prev_snapshot or {})
        new_snap = dict(new_snapshot or {})
        side = str(new_snap.get("pos") or prev_snap.get("pos") or "").upper()
        prev_exposure = float(prev_snap.get("notional_exposure", prev_snap.get("total_exposure", 0.0)) or 0.0)
        new_exposure = float(new_snap.get("notional_exposure", new_snap.get("total_exposure", 0.0)) or 0.0)
        prev_fraction = float(prev_snap.get("position_fraction", 0.0) or 0.0)
        new_fraction = float(new_snap.get("position_fraction", 0.0) or 0.0)
        prev_exec_lev = float(prev_snap.get("execution_leverage", 1.0) or 1.0)
        new_exec_lev = float(new_snap.get("execution_leverage", 1.0) or 1.0)
        entry_price = float(new_snap.get("entry_price", prev_snap.get("entry_price", 0.0)) or 0.0)
        delta_exposure = float(new_exposure - prev_exposure)
        delta_fraction = float(new_fraction - prev_fraction)
        if delta_exposure > 1e-9:
            event = "UPSIZE"
        elif delta_exposure < -1e-9:
            event = "DOWNSIZE"
        else:
            event = "REBALANCE"
        event_recorded_at = self._journal_event_recorded_at()
        audit = self._journal_audit_fields(audit_context, kind="RESIZE")
        identity_snap = dict(prev_snap)
        identity_snap.update({k: v for k, v in new_snap.items() if v not in (None, "")})
        audit = self._apply_open_model_identity_to_audit(audit, identity_snap, kind="RESIZE")
        if not str(audit.get("entry_execution_liquidity", "") or ""):
            audit["entry_execution_liquidity"] = str(prev_snap.get("entry_execution_liquidity", new_snap.get("entry_execution_liquidity", "")) or "")
        mark = self._trade_math(
            side,
            entry_price,
            float(current_price or 0.0),
            prev_exposure,
            entry_liquidity=str(audit.get("entry_execution_liquidity", "") or ""),
            exit_liquidity="",
        )
        resize_math = (
            self._trade_math(
                side,
                entry_price,
                float(current_price or 0.0),
                abs(delta_exposure),
                entry_liquidity=str(audit.get("entry_execution_liquidity", "") or ""),
                exit_liquidity=str(audit.get("exit_execution_liquidity", "") or ""),
            )
            if delta_exposure < -1e-9
            else {}
        )
        resize_pnl_frac = float(resize_math.get("pnl_frac", self.last_resize_realized_pnl_frac or 0.0) or 0.0)
        resize_entry_exec_price = float(resize_math.get("entry_exec_price", 0.0))
        resize_exit_exec_price = float(resize_math.get("exit_exec_price", 0.0))
        return {
            "schema_version": "trade_journal.v1",
            "ts": str(timestamp_kst),
            "kind": "RESIZE",
            "event": event,
            "side": side,
            "trade_id": str(new_snap.get("trade_id", prev_snap.get("trade_id", "")) or ""),
            "decision_at": str(new_snap.get("decision_at", prev_snap.get("decision_at", "")) or ""),
            "opened_at": str(new_snap.get("opened_at", prev_snap.get("opened_at", "")) or ""),
            "actual_opened_at": str(new_snap.get("opened_at", prev_snap.get("opened_at", "")) or ""),
            "actual_resized_at": str(event_recorded_at),
            "event_recorded_at": str(event_recorded_at),
            "mark_price": float(current_price or 0.0),
            "entry_price": float(entry_price),
            "entry_price_source": str(new_snap.get("entry_price_source", prev_snap.get("entry_price_source", "")) or ""),
            "entry_decision_price": float(new_snap.get("entry_decision_price", prev_snap.get("entry_decision_price", 0.0)) or 0.0),
            "gross_return_frac_mark": float(mark.get("gross_return_frac", 0.0)),
            "mark_pnl_frac_prev_exposure": float(mark.get("pnl_frac", 0.0)),
            "mark_pnl_pct_prev_exposure": float(mark.get("pnl_pct", 0.0)),
            "resize_entry_exec_price": float(resize_entry_exec_price),
            "resize_exit_exec_price": float(resize_exit_exec_price),
            "resize_exec_price_kind": "synthetic_fee_slippage_model",
            "synthetic_resize_entry_exec_price": float(resize_entry_exec_price),
            "synthetic_resize_exit_exec_price": float(resize_exit_exec_price),
            "resize_gross_return_frac": float(resize_math.get("gross_return_frac", 0.0)),
            "entry_fee_rate": float(resize_math.get("entry_fee_rate", 0.0)),
            "entry_fee_model": str(resize_math.get("entry_fee_model", "")),
            "exit_fee_rate": float(resize_math.get("exit_fee_rate", 0.0)),
            "exit_fee_model": str(resize_math.get("exit_fee_model", "")),
            "roundtrip_fee_rate": float(resize_math.get("roundtrip_fee_rate", 0.0)),
            "fee_model": str(resize_math.get("fee_model", "")),
            "fee_cost_frac": float(resize_math.get("fee_cost_frac", 0.0)),
            "pnl_frac": float(resize_pnl_frac),
            "pnl_pct": float(resize_pnl_frac * 100.0),
            "resize_realized_pnl_frac": float(resize_pnl_frac),
            "resize_realized_pnl_pct": float(resize_pnl_frac * 100.0),
            "position_realized_pnl_frac": float(new_snap.get("position_realized_pnl_frac", 0.0) or 0.0),
            "costs_recognized_in_strategy_equity": bool(abs(float(self.last_resize_realized_pnl_frac or 0.0)) > 1e-12),
            "hold_bars": int(new_snap.get("hold_bars", prev_snap.get("hold_bars", 0)) or 0),
            "prev_position_fraction": prev_fraction,
            "new_position_fraction": new_fraction,
            "prev_margin_fraction": float(prev_snap.get("margin_fraction", prev_fraction) or 0.0),
            "new_margin_fraction": float(new_snap.get("margin_fraction", new_fraction) or 0.0),
            "delta_position_fraction": delta_fraction,
            "prev_execution_leverage": prev_exec_lev,
            "new_execution_leverage": new_exec_lev,
            "prev_notional_exposure": prev_exposure,
            "new_notional_exposure": new_exposure,
            "delta_notional_exposure": delta_exposure,
            "prev_total_exposure": prev_exposure,
            "new_total_exposure": new_exposure,
            "delta_total_exposure": delta_exposure,
            "regime": str(regime_name),
            "source": str(source),
            "reason": str(reason),
            **audit,
        }

    def record_outcome(self, realized_pnl_pct: float):
        pnl = float(realized_pnl_pct)
        self.last_realized_pnl = None
        self.recent_realized.append(pnl)
        self.loss_streak = 0 if pnl > 0 else (self.loss_streak + 1)
        self._save_live_state()
        self._open_trade_diag = None

    def update_adaptive_gate(self, final_action: int, in_position: bool) -> tuple[float, float]:
        if not self.adaptive_gate_enable:
            self.adaptive_enter_offset = 0.0
            self.adaptive_agreement_offset = 0.0
            return 0.0, 0.0

        if in_position:
            self.adaptive_flat_cycles = 0
        elif int(final_action) == 0:
            self.adaptive_flat_cycles += 1
        else:
            self.adaptive_flat_cycles = 0

        window = max(1, int(self.adaptive_gate_pnl_window))
        recent_vals = list(self.recent_realized)[-window:]
        recent_pnl_sum = float(sum(recent_vals)) if recent_vals else 0.0

        enter_offset = 0.0
        agreement_offset = 0.0
        if self.loss_streak >= max(1, self.adaptive_gate_loss_streak_th) or recent_pnl_sum <= self.adaptive_gate_bad_pnl_cut:
            enter_offset += float(self.adaptive_gate_enter_step)
            agreement_offset += float(self.adaptive_gate_agreement_step)
        elif self.cooldown_bars_left == 0 and self.loss_streak == 0 and recent_pnl_sum >= self.adaptive_gate_good_pnl_cut:
            enter_offset -= float(self.adaptive_gate_loosen_step)
            agreement_offset -= float(self.adaptive_gate_loosen_step)

        if self.pos is None and self.adaptive_flat_cycles >= max(1, self.adaptive_gate_flat_bars):
            enter_offset -= float(self.adaptive_gate_loosen_step)
            agreement_offset -= float(self.adaptive_gate_loosen_step * 0.5)

        self.adaptive_enter_offset = float(np.clip(enter_offset, self.adaptive_gate_enter_min, self.adaptive_gate_enter_max))
        self.adaptive_agreement_offset = float(np.clip(agreement_offset, self.adaptive_gate_agreement_min, self.adaptive_gate_agreement_max))
        return self.adaptive_enter_offset, self.adaptive_agreement_offset

    def _open_position(
        self,
        side: str,
        entry_px: float,
        decision_at=None,
        leverage: float | None = None,
        fraction: float | None = None,
        leverage_mult: float | None = None,
    ) -> None:
        self._clear_open_model_identity()
        self.pos = side
        self.entry_price = float(max(entry_px, 0.0))
        self.hold_count = 0
        if fraction is not None or leverage_mult is not None:
            self._set_position_sizing(fraction=fraction, leverage_mult=leverage_mult)
        else:
            self._set_position_sizing(exposure=(leverage if leverage is not None else self.current_leverage))
        self.peak_equity = self.cur_equity = 1.0
        self.position_realized_pnl_frac = 0.0
        self.last_resize_realized_pnl_frac = 0.0
        self.last_realized_pnl = None
        self.trend_mismatch_streak = 0
        self.position_exit_streak = 0
        self.open_trade_id = self._new_trade_id()
        self.opened_at = pd.Timestamp.now(tz="Asia/Seoul").isoformat()
        self.decision_at = str(decision_at or self.opened_at)
        self.entry_price_source = str(self._pending_entry_price_source or "decision_current_price")
        self.entry_decision_price = float(self._pending_entry_decision_price or entry_px or 0.0)
        self.exchange_entry_price = 0.0
        self.entry_execution_liquidity = str(self._pending_entry_execution_liquidity or "")
        self.entry_execution_route = str(self._pending_entry_execution_route or "")
        self.entry_execution_order_type = str(self._pending_entry_execution_order_type or "")

    def _choose_entry_price(
        self,
        final_action: int,
        current_price: float,
        trend_signal: dict | None = None,
    ) -> float:
        px = max(float(current_price), 0.0)
        out = px
        source = "decision_current_price"
        ts = trend_signal if isinstance(trend_signal, dict) else {}
        if self.enforce_bar_fill_price:
            self._pending_entry_price_source = "bar_fill_price"
            self._pending_entry_decision_price = float(px)
            return out
        if self.entry_reco_enable and px > 0.0 and ts:
            strength = float(ts.get("strength", 0.0) or 0.0)
            quality = float(ts.get("m7_quality_pred", 0.0) or 0.0)
            if strength >= self.entry_reco_min_strength and quality >= self.entry_reco_min_quality:
                if final_action == 1:
                    reco_px = float(ts.get("m7_entry_long_price", 0.0) or 0.0)
                    reco_off = abs(float(ts.get("m7_entry_long_offset", 0.0) or 0.0))
                    if reco_px > 0.0 and reco_px <= px * (1.0 + self.entry_reco_price_buffer) and reco_off <= self.entry_reco_max_offset:
                        out = min(out, reco_px)
                        source = "m7_entry_long_price" if out == reco_px else source
                elif final_action == 2:
                    reco_px = float(ts.get("m7_entry_short_price", 0.0) or 0.0)
                    reco_off = abs(float(ts.get("m7_entry_short_offset", 0.0) or 0.0))
                    if reco_px > 0.0 and reco_px >= px * (1.0 - self.entry_reco_price_buffer) and reco_off <= self.entry_reco_max_offset:
                        out = max(out, reco_px)
                        source = "m7_entry_short_price" if out == reco_px else source
        self._pending_entry_price_source = str(source)
        self._pending_entry_decision_price = float(px)
        return out

    def _update_pos(
        self,
        final_action: int,
        current_price: float,
        timestamp_kst=None,
        leverage: float | None = None,
        fraction: float | None = None,
        leverage_mult: float | None = None,
        trend_signal: dict | None = None,
        entry_price_source_override: str | None = None,
        entry_decision_price_override: float | None = None,
        entry_execution_liquidity_override: str | None = None,
        entry_execution_route_override: str | None = None,
        entry_execution_order_type_override: str | None = None,
    ):
        entry_px = self._choose_entry_price(final_action, current_price, trend_signal)
        if entry_price_source_override:
            self._pending_entry_price_source = str(entry_price_source_override)
            if str(entry_price_source_override) in {
                "next_bar_open",
                "scheduled_next_bar_open",
                "bar_fill_price",
                "decision_bar",
                "execution_bar",
            }:
                entry_px = max(float(current_price), 0.0)
        if entry_decision_price_override is not None:
            self._pending_entry_decision_price = float(entry_decision_price_override or 0.0)
        if entry_execution_liquidity_override is not None:
            self._pending_entry_execution_liquidity = str(entry_execution_liquidity_override or "")
        if entry_execution_route_override is not None:
            self._pending_entry_execution_route = str(entry_execution_route_override or "")
        if entry_execution_order_type_override is not None:
            self._pending_entry_execution_order_type = str(entry_execution_order_type_override or "")
        if final_action == 1 and self.pos == "SHORT":
            if self.entry_price > 0 and current_price > 0: self.cur_equity = 1.0 + self._net_pnl_frac(current_price)
            realized = float(self.cur_equity - 1.0)
            closed_hold = int(self.hold_count)
            self._open_position("LONG", entry_px, timestamp_kst, leverage, fraction=fraction, leverage_mult=leverage_mult)
            self.last_realized_pnl = realized
            self.last_closed_hold_count = closed_hold
            self._save_live_state()
            return
        if final_action == 2 and self.pos == "LONG":
            if self.entry_price > 0 and current_price > 0: self.cur_equity = 1.0 + self._net_pnl_frac(current_price)
            realized = float(self.cur_equity - 1.0)
            closed_hold = int(self.hold_count)
            self._open_position("SHORT", entry_px, timestamp_kst, leverage, fraction=fraction, leverage_mult=leverage_mult)
            self.last_realized_pnl = realized
            self.last_closed_hold_count = closed_hold
            self._save_live_state()
            return
        if final_action == 1 and self.pos is None:
            self._open_position("LONG", entry_px, timestamp_kst, leverage, fraction=fraction, leverage_mult=leverage_mult)
            self._save_live_state()
        elif final_action == 2 and self.pos is None:
            self._open_position("SHORT", entry_px, timestamp_kst, leverage, fraction=fraction, leverage_mult=leverage_mult)
            self._save_live_state()
        elif final_action == 0 and self.pos is not None:
            if self.entry_price > 0 and current_price > 0: self.cur_equity = 1.0 + self._net_pnl_frac(current_price)
            self.last_realized_pnl = float(self.cur_equity - 1.0)
            self.last_closed_hold_count = int(self.hold_count)
            self.pos, self.entry_price, self.hold_count = None, 0.0, 0
            self.open_trade_id = ""
            self.opened_at = ""
            self.decision_at = ""
            self.entry_price_source = ""
            self.entry_decision_price = 0.0
            self.exchange_entry_price = 0.0
            self.entry_execution_liquidity = ""
            self.entry_execution_route = ""
            self.entry_execution_order_type = ""
            self._clear_open_model_identity()
            self._set_position_sizing(exposure=0.0)
            self.position_realized_pnl_frac = 0.0
            self.last_resize_realized_pnl_frac = 0.0
            self.peak_equity = 1.0
            self.cur_equity = 1.0
            self.trend_mismatch_streak = 0
            self.position_exit_streak = 0
            self._save_live_state()
        elif self.pos is not None and self.entry_price > 0 and current_price > 0:
            self.hold_count += 1
            prev_exposure = float(self.current_leverage or 0.0)
            prev_entry = float(self.entry_price or 0.0)
            self.last_resize_realized_pnl_frac = 0.0
            if fraction is not None or leverage_mult is not None:
                self._set_position_sizing(fraction=fraction, leverage_mult=leverage_mult)
            elif leverage is not None:
                self._set_position_sizing(exposure=leverage)
            new_exposure = float(self.current_leverage or 0.0)
            delta_exposure = float(new_exposure - prev_exposure)
            if delta_exposure < -1e-9:
                closed_exposure = float(abs(delta_exposure))
                resize_math = self._trade_math(self.pos, prev_entry, current_price, closed_exposure)
                resize_realized = float(resize_math.get("pnl_frac", 0.0) or 0.0)
                self.last_resize_realized_pnl_frac = resize_realized
                self.position_realized_pnl_frac += resize_realized
            elif delta_exposure > 1e-9 and prev_exposure > 1e-12 and new_exposure > 1e-12:
                added_px = float(entry_px if entry_px > 0.0 else current_price)
                self.entry_price = float(
                    ((prev_entry * prev_exposure) + (added_px * delta_exposure)) / max(new_exposure, 1e-12)
                )
                self.entry_price_source = f"weighted_average:{self._pending_entry_price_source or 'decision_current_price'}"
                self.entry_decision_price = float(current_price or 0.0)
            self.cur_equity = 1.0 + self._net_pnl_frac(current_price)
            self.peak_equity = max(self.peak_equity, self.cur_equity)
            self.last_realized_pnl = None
            self._save_live_state()
        elif self.pos is None and final_action == 0:
            self._save_live_state()

    def _load_live_state(self) -> None:
        path = self.live_state_path
        if not path or not os.path.exists(path): return
        try:
            with open(path, "r", encoding="utf-8") as f: data = json.load(f)
            self.pos = data.get("pos")
            self.entry_price = float(data.get("entry_price", 0.0))
            self.hold_count = int(data.get("hold_count", 0))
            self.open_trade_id = str(data.get("open_trade_id", "") or "")
            self.opened_at = str(data.get("opened_at", "") or "")
            self.decision_at = str(data.get("decision_at", "") or "")
            self.entry_price_source = str(data.get("entry_price_source", "") or "")
            self.entry_decision_price = float(data.get("entry_decision_price", 0.0) or 0.0)
            self.exchange_entry_price = float(data.get("exchange_entry_price", 0.0) or 0.0)
            self.entry_execution_liquidity = str(data.get("entry_execution_liquidity", "") or "")
            self.entry_execution_route = str(data.get("entry_execution_route", "") or "")
            self.entry_execution_order_type = str(data.get("entry_execution_order_type", "") or "")
            self.open_model_version = str(data.get("open_model_version", "") or "")
            self.open_model_id = str(data.get("open_model_id", "") or "")
            self.open_model_path = str(data.get("open_model_path", "") or "")
            self.open_model_sleeve = str(data.get("open_model_sleeve", "") or "")
            self.open_source = str(data.get("open_source", "") or "")
            strategy_state = data.get("strategy_state", {})
            if not isinstance(strategy_state, dict):
                raise TypeError("strategy_state must be a JSON object")
            self.strategy_state = dict(strategy_state)
            saved_fraction = data.get("position_fraction", None)
            saved_exec_lev = data.get("execution_leverage", None)
            saved_exposure = data.get("current_exposure", data.get("current_leverage", 0.0))
            if saved_fraction is not None or saved_exec_lev is not None:
                self._set_position_sizing(
                    fraction=float(saved_fraction if saved_fraction is not None else 0.0),
                    leverage_mult=float(saved_exec_lev if saved_exec_lev is not None else 1.0),
                )
            else:
                self._set_position_sizing(exposure=float(saved_exposure or 0.0))
            self.peak_equity = float(max(data.get("peak_equity", 1.0), 1e-8))
            self.cur_equity = float(max(data.get("cur_equity", 1.0), 1e-8))
            self.position_realized_pnl_frac = float(data.get("position_realized_pnl_frac", 0.0) or 0.0)
            self.last_resize_realized_pnl_frac = float(data.get("last_resize_realized_pnl_frac", 0.0) or 0.0)
            self.last_realized_pnl = data.get("last_realized_pnl", None)
            self.last_closed_hold_count = int(data.get("last_closed_hold_count", 0))
            self.loss_streak = int(data.get("loss_streak", 0))
            self.cooldown_bars_left = int(data.get("cooldown_bars_left", 0))
            self.trend_mismatch_streak = int(data.get("trend_mismatch_streak", 0))
            self.position_exit_streak = int(data.get("position_exit_streak", 0))
            self.adaptive_flat_cycles = int(data.get("adaptive_flat_cycles", 0))
            self.recent_realized = deque([float(x) for x in data.get("recent_realized", [])], maxlen=20)
            self.trade_history = deque(data.get("trade_history", []), maxlen=2000)
            if self.pos in {"LONG", "SHORT"} and self.opened_at:
                try:
                    opened = pd.Timestamp(self.opened_at)
                    now = pd.Timestamp.now(tz="Asia/Seoul")
                    if opened.tzinfo is None:
                        opened = opened.tz_localize("Asia/Seoul")
                    else:
                        opened = opened.tz_convert("Asia/Seoul")
                    elapsed_bars = int(max(0, np.floor((now - opened).total_seconds() / 300.0)))
                    if elapsed_bars > self.hold_count:
                        logger.warning(
                            "SYSTEM live_state_hold_count_recovered opened_at=%s saved=%d elapsed=%d",
                            str(self.opened_at),
                            int(self.hold_count),
                            int(elapsed_bars),
                        )
                        self.hold_count = int(elapsed_bars)
                except Exception as e:
                    logger.warning("SYSTEM live_state_hold_count_recovery_failed: %s", e)
        except Exception as e:
            raise RuntimeError(f"governor_live_state_load_failed:{path}") from e

    def _save_live_state(self) -> None:
        path = self.live_state_path
        if not path: return
        try:
            payload = {
                "pos": self.pos, "entry_price": self.entry_price, "hold_count": self.hold_count,
                "open_trade_id": self.open_trade_id, "opened_at": self.opened_at, "decision_at": self.decision_at,
                "entry_price_source": self.entry_price_source,
                "entry_decision_price": self.entry_decision_price,
                "exchange_entry_price": self.exchange_entry_price,
                "entry_execution_liquidity": self.entry_execution_liquidity,
                "entry_execution_route": self.entry_execution_route,
                "entry_execution_order_type": self.entry_execution_order_type,
                "open_model_version": self.open_model_version,
                "open_model_id": self.open_model_id,
                "open_model_path": self.open_model_path,
                "open_model_sleeve": self.open_model_sleeve,
                "open_source": self.open_source,
                "strategy_state": self._journal_jsonable(self.strategy_state),
                "current_exposure": self.current_leverage, "current_leverage": self.current_leverage,
                "position_fraction": self.position_fraction, "execution_leverage": self.execution_leverage,
                "peak_equity": self.peak_equity,
                "cur_equity": self.cur_equity,
                "position_realized_pnl_frac": self.position_realized_pnl_frac,
                "last_resize_realized_pnl_frac": self.last_resize_realized_pnl_frac,
                "last_realized_pnl": self.last_realized_pnl,
                "last_closed_hold_count": self.last_closed_hold_count, "loss_streak": self.loss_streak,
                "cooldown_bars_left": self.cooldown_bars_left, "trend_mismatch_streak": self.trend_mismatch_streak,
                "position_exit_streak": self.position_exit_streak, "adaptive_flat_cycles": self.adaptive_flat_cycles,
                "recent_realized": list(self.recent_realized), "trade_history": list(self.trade_history),
                "saved_at": pd.Timestamp.utcnow().tz_localize(None).isoformat(),
            }
            _atomic_write_json(path, payload)
        except Exception as e:
            raise RuntimeError(f"governor_live_state_save_failed:{path}") from e

    def _force_close_record(self, price: float, reason: str = "") -> None:
        if self.pos is None: return
        pnl = self._mark_pnl_frac(price) if price > 0 else 0.0
        trade_math = self._trade_math(
            self.pos,
            self.entry_price,
            price,
            self.current_leverage,
            entry_liquidity=self.entry_execution_liquidity,
            exit_liquidity="",
        )
        self.trade_history.append({
            "ts": datetime.utcnow().isoformat(timespec="seconds"), "side": self.pos,
            "entry": self.entry_price, "exit": price, "hold": self.hold_count,
            "trade_id": self.open_trade_id,
            "decision_at": self.decision_at,
            "opened_at": self.opened_at,
            "actual_opened_at": self.opened_at,
            "entry_price": self.entry_price,
            "entry_price_source": self.entry_price_source,
            "entry_decision_price": float(self.entry_decision_price or 0.0),
            "exchange_entry_price": float(self.exchange_entry_price or 0.0),
            "entry_exec_price": float(trade_math.get("entry_exec_price", 0.0)),
            "entry_exec_price_kind": "synthetic_fee_slippage_model",
            "synthetic_entry_exec_price": float(trade_math.get("entry_exec_price", 0.0)),
            "exit_price": float(price),
            "exit_exec_price": float(trade_math.get("exit_exec_price", 0.0)),
            "exit_exec_price_kind": "synthetic_fee_slippage_model",
            "synthetic_exit_exec_price": float(trade_math.get("exit_exec_price", 0.0)),
            "entry_fee_rate": float(trade_math.get("entry_fee_rate", 0.0)),
            "entry_fee_model": str(trade_math.get("entry_fee_model", "")),
            "exit_fee_rate": float(trade_math.get("exit_fee_rate", 0.0)),
            "exit_fee_model": str(trade_math.get("exit_fee_model", "")),
            "roundtrip_fee_rate": float(trade_math.get("roundtrip_fee_rate", 0.0)),
            "fee_model": str(trade_math.get("fee_model", "")),
            "fee_cost_frac": float(trade_math.get("fee_cost_frac", 0.0)),
            "position_fraction": float(self.position_fraction),
            "margin_fraction": float(self.position_fraction),
            "execution_leverage": float(self.execution_leverage),
            "notional_exposure": float(self.current_leverage),
            "total_exposure": float(self.current_leverage),
            "pnl_frac": float(pnl), "pnl": round(pnl, 6), "reason": reason or "FORCE_CLOSE", "liq_forced": True,
        })
        self.recent_realized.append(pnl)
        self.loss_streak = 0 if pnl > 0 else (self.loss_streak + 1)
        logger.warning("🚨 FORCE_CLOSE 기록 | pos=%s entry=%.4f exit=%.4f pnl=%.4f%% 사유=%s",
                       self.pos, self.entry_price, price, pnl * 100, reason)
        self.pos, self.entry_price, self.hold_count, self._open_trade_diag = None, 0.0, 0, None
        self.open_trade_id = ""
        self.opened_at = ""
        self.decision_at = ""
        self.entry_price_source = ""
        self.entry_decision_price = 0.0
        self.exchange_entry_price = 0.0
        self._set_position_sizing(exposure=0.0)
        self.position_realized_pnl_frac = 0.0
        self.last_resize_realized_pnl_frac = 0.0

    def _mark_pnl_frac(self, current_price: float, exposure: float | None = None) -> float:
        if self.pos is None or self.entry_price <= 0.0 or current_price <= 0.0: return 0.0
        # `current_leverage` is the total account exposure used by the strategy.
        lev = float(np.clip(self.current_leverage if exposure is None else exposure, 0.0, self.exposure_cap))
        entry_is_real = self._is_real_execution_liquidity(self.entry_execution_liquidity)
        if self.pos == "LONG":
            entry_exec = self.entry_price if entry_is_real else self.entry_price * (1.0 + self.trade_slip)
            exit_exec = current_price * (1.0 - self.trade_slip)
            gross = (exit_exec - entry_exec) / max(entry_exec, 1e-8)
        else:
            entry_exec = self.entry_price if entry_is_real else self.entry_price * (1.0 - self.trade_slip)
            exit_exec = current_price * (1.0 + self.trade_slip)
            gross = (entry_exec - exit_exec) / max(abs(entry_exec), 1e-8)
        entry_fee, _ = self._fee_rate_for_liquidity(self.entry_execution_liquidity)
        exit_fee, _ = self._fee_rate_for_liquidity("")
        return float(gross * lev - ((entry_fee + exit_fee) * lev))

    def _net_pnl_frac(self, current_price: float) -> float:
        if self.pos is None:
            return 0.0
        return float(self.position_realized_pnl_frac + self._mark_pnl_frac(current_price))

    def unrealized_pnl(self, current_price: float) -> float:
        return self._net_pnl_frac(current_price) * 100.0

    def decrement_cooldown(self) -> None:
        if self.cooldown_bars_left > 0: self.cooldown_bars_left -= 1

    def long_trend_score(self, processed_df: pd.DataFrame, trend_signal: dict | None) -> float:
        last_row = processed_df.iloc[-1]
        def _sf(v, d: float = 0.0) -> float:
            try: return float(v)
            except Exception: return float(d)
        trend_1h = _sf(last_row.get("mtf_trend_1h", 0.0))
        trend_4h = _sf(last_row.get("mtf_trend_4h", 0.0))
        closes = processed_df["close"].tail(12).astype(float).values if "close" in processed_df.columns else np.array([], dtype=float)
        ret_12 = ((closes[-1] / closes[0]) - 1.0) if len(closes) >= 2 and abs(closes[0]) > 1e-8 else 0.0

        mtf_edge = float(np.tanh((trend_1h + trend_4h + ret_12 * 80.0) / 2.4))
        return float(np.clip(mtf_edge, -1.0, 1.0))

    def update_trend_mismatch(self, processed_df: pd.DataFrame, trend_signal: dict | None) -> tuple[bool, float, str]:
        if not self.governor_trend_exit_enable or self.pos is None:
            self.trend_mismatch_streak = 0
            return False, 0.0, ""

        score = self.long_trend_score(processed_df, trend_signal)
        quality = float(trend_signal.get("m7_quality_pred", 0.0)) if isinstance(trend_signal, dict) else 0.0

        mismatch, reason = False, ""
        if self.hold_count >= max(1, self.governor_trend_exit_hold_bars):
            if self.pos == "LONG" and score <= -abs(self.governor_trend_exit_score) and quality <= self.governor_trend_exit_quality:
                mismatch, reason = True, "GOVERNOR_M7_LONG_MISMATCH"
            elif self.pos == "SHORT" and score >= abs(self.governor_trend_exit_score) and quality >= -self.governor_trend_exit_quality:
                mismatch, reason = True, "GOVERNOR_M7_SHORT_MISMATCH"

        self.trend_mismatch_streak = (self.trend_mismatch_streak + 1) if mismatch else 0
        return (self.trend_mismatch_streak >= max(1, self.governor_trend_exit_confirm_bars)), score, reason

    def reconcile_external_position(
        self,
        pos_type: str | None,
        entry_price: float,
        leverage: float = 0.0,
        notional: float = 0.0,
        account_equity: float = 0.0,
        notional_exposure: float = 0.0,
        current_price: float = 0.0,
        tp_sl_fill_info: dict | None = None,
        timestamp_kst=None,
        regime_name: str = "",
        governor_source: str = "",
    ) -> None:
        """`tp_sl_fill_info` (optional) is the {"tp": order_or_None, "sl": order_or_None} result of
        polling a resting exchange TP/SL order pair (see BinanceFuturesExecutionAdapter.
        poll_tp_sl_orders). When the exchange position just went LONG/SHORT -> flat and a filled
        leg is identified, this builds a proper CLOSE trade_journal payload (reason/exit price
        from the fill) via build_close_trade_payload instead of silently discarding the closed
        trade -- otherwise a fill that lands between decision cycles (or while the process is
        down) would leave no journal record and no realized-PnL bookkeeping at all. The payload is
        stashed on self._last_reconcile_close_payload for the async caller to append to
        trade_journal.jsonl (file I/O must happen in the async loop, not here)."""
        self._last_reconcile_close_payload = None
        ext_pos = pos_type if pos_type in {"LONG", "SHORT"} else None
        ext_entry = float(entry_price) if entry_price and entry_price > 0 else 0.0
        # Preserve internal exposure semantics. Accept direct exposure snapshots in 0..cap,
        # but ignore exchange-style high x-leverage values unless the internal model already stores them.
        restored_exposure = float(notional_exposure or 0.0)
        if restored_exposure <= 0.0 and float(notional or 0.0) > 0.0 and float(account_equity or 0.0) > 0.0:
            restored_exposure = float(notional) / max(float(account_equity), 1e-8)
        if restored_exposure > 0.0:
            ext_lev = float(np.clip(restored_exposure, 0.0, self.exposure_cap))
        elif 0.0 < float(leverage or 0.0) <= self.exposure_cap:
            ext_lev = float(leverage)
        elif self.current_leverage > 0.0:
            ext_lev = float(self.current_leverage)
        else:
            ext_lev = float(np.clip(FINAL_GOVERNOR_RECONCILE_DEFAULT_EXPOSURE, 0.0, self.exposure_cap))
        if ext_pos is None:
            if self.pos is not None:
                self.last_closed_hold_count = int(self.hold_count)
                fill = dict(tp_sl_fill_info or {})
                tp_order = dict(fill.get("tp") or {}) or None
                sl_order = dict(fill.get("sl") or {}) or None
                filled_leg, filled_order = None, None
                for leg, order in (("take_profit", tp_order), ("stop_loss", sl_order)):
                    if order and str(order.get("status", "")).lower() in {"closed", "filled"}:
                        filled_leg, filled_order = leg, order
                        break
                close_price = float(current_price or self.entry_price or 0.0)
                close_reason = "exchange_position_closed_externally"
                if filled_order is not None:
                    close_reason = f"omega4_6_1_exchange_{filled_leg}"
                    fill_price = _safe_float(filled_order.get("average", filled_order.get("price", 0.0)), 0.0)
                    if fill_price > 0.0:
                        close_price = fill_price
                if close_price > 0.0:
                    ts = timestamp_kst or pd.Timestamp.now(tz="Asia/Seoul")
                    try:
                        snapshot = self.position_snapshot()
                        payload = self.build_close_trade_payload(
                            snapshot=snapshot,
                            current_price=close_price,
                            timestamp_kst=ts,
                            event="reconcile_close",
                            regime_name=str(regime_name or ""),
                            source=str(governor_source or "reconcile"),
                            reason=close_reason,
                            next_side=None,
                        )
                        realized = float(payload.get("pnl_frac", 0.0))
                        self.record_outcome(realized)
                        self.append_trade_history(ts, realized, payload=payload)
                        self._last_reconcile_close_payload = payload
                    except Exception:
                        # build_close_trade_payload failed (e.g. missing audit/feature fields), but the
                        # exchange has already confirmed this position is gone -- self.pos is still reset
                        # to flat below regardless. Record a minimal fallback payload from _mark_pnl_frac
                        # (independent of the failing rich-payload path) so the realized PnL/trade is not
                        # silently dropped from bookkeeping; degraded_reconcile_payload marks it for audit.
                        logger.exception("SYSTEM reconcile_close_payload_build_failed trade_id=%s", self.open_trade_id)
                        try:
                            realized = self._mark_pnl_frac(close_price)
                            fallback_payload = {
                                "trade_id": self.open_trade_id,
                                "side": self.pos,
                                "entry_price": self.entry_price,
                                "exit_price": close_price,
                                "pnl_frac": realized,
                                "ts": str(ts),
                                "event": "reconcile_close",
                                "reason": f"{close_reason}|degraded_reconcile_payload",
                                "source": str(governor_source or "reconcile"),
                            }
                            self.record_outcome(realized)
                            self.append_trade_history(ts, realized, payload=fallback_payload)
                            self._last_reconcile_close_payload = fallback_payload
                        except Exception:
                            logger.exception(
                                "SYSTEM reconcile_close_fallback_payload_failed trade_id=%s -- realized PnL for this "
                                "close was NOT recorded", self.open_trade_id
                            )
                self.pos, self.entry_price, self.hold_count, self.peak_equity, self.cur_equity = None, 0.0, 0, 1.0, 1.0
                self.open_trade_id = ""
                self.opened_at = ""
                self.decision_at = ""
                self.entry_price_source = ""
                self.entry_decision_price = 0.0
                self.exchange_entry_price = 0.0
                self._set_position_sizing(exposure=0.0)
                self._save_live_state()
            return
        if self.pos != ext_pos or abs(self.entry_price - ext_entry) > 1e-6:
            self.pos, self.entry_price, self.hold_count, self.peak_equity, self.cur_equity = ext_pos, ext_entry, 0, 1.0, 1.0
            self.open_trade_id = self._new_trade_id()
            self.opened_at = pd.Timestamp.now(tz="Asia/Seoul").isoformat()
            self.decision_at = self.opened_at
            self.entry_price_source = "exchange_position_reconcile"
            self.entry_decision_price = float(ext_entry or 0.0)
            self.exchange_entry_price = float(ext_entry or 0.0)
            self._set_position_sizing(exposure=ext_lev)
            self._save_live_state()
        elif 0.0 < ext_lev <= self.exposure_cap and abs(self.current_leverage - ext_lev) > 1e-9:
            self._set_position_sizing(exposure=ext_lev)
            self._save_live_state()

    def append_trade_history(self, timestamp_kst, pnl_frac: float, payload: dict | None = None) -> None:
        ts_str = timestamp_kst.isoformat() if hasattr(timestamp_kst, "isoformat") else str(timestamp_kst)
        row = {
            "ts": ts_str,
            "pnl_frac": float(pnl_frac),
            "hold_bars": int(self.last_closed_hold_count),
        }
        extra = dict(payload or {})
        for key, val in extra.items():
            if key not in row:
                row[key] = val
        self.trade_history.append(row)
        self._save_live_state()

    def performance_metrics(self, now_kst) -> dict:
        if not self.trade_history:
            return {"pnl_24h": 0.0, "wr_24h": 0.0, "trades_24h": 0, "mdd_24h": 0.0, "pnl_7d": 0.0, "wr_7d": 0.0, "trades_7d": 0, "mdd_7d": 0.0, "pnl_all": 0.0, "wr_all": 0.0, "trades_all": 0, "mdd_all": 0.0, "pnl_24h_sum": 0.0, "pnl_7d_sum": 0.0, "pnl_all_sum": 0.0, "cooldown_bars_left": int(self.cooldown_bars_left)}
        def _to_utc_ts(value, default: str = "2000-01-01") -> pd.Timestamp:
            ts = pd.Timestamp(value if value not in (None, "") else default)
            if ts.tzinfo is None:
                # Old live-state rows were written as KST wall-clock strings without tz.
                ts = ts.tz_localize("Asia/Seoul")
            else:
                ts = ts.tz_convert("Asia/Seoul")
            return ts.tz_convert("UTC")

        now_ts = _to_utc_ts(now_kst, default=pd.Timestamp.now(tz="Asia/Seoul").isoformat())
        def _window(hours: int):
            cutoff = now_ts - pd.Timedelta(hours=hours)
            rows = []
            for r in self.trade_history:
                try:
                    if _to_utc_ts(r.get("ts", "2000-01-01")) >= cutoff:
                        rows.append(r)
                except Exception:
                    continue
            if not rows: return 0.0, 0.0, 0, 0.0, 0.0
            pnls = [float(x.get("pnl_frac", 0.0)) for x in rows]
            eq = np.cumprod([1.0 + p for p in pnls])
            peak = np.maximum.accumulate(eq) if len(eq) else np.asarray([1.0], dtype=float)
            drawdown = ((eq / np.maximum(peak, 1e-12)) - 1.0) if len(eq) else np.asarray([0.0], dtype=float)
            return (
                float(eq[-1] - 1.0) * 100.0 if len(eq) else 0.0,
                100.0 * sum(1 for p in pnls if p > 0) / len(pnls),
                len(pnls),
                float(drawdown.min() * 100.0) if len(drawdown) else 0.0,
                float(sum(pnls)) * 100.0,
            )
        p7, w7, t7, m7, p7_sum = _window(24 * 7)
        p24, w24, t24, m24, p24_sum = _window(24)
        pall_sum = float(sum(float(x.get("pnl_frac", 0.0)) for x in self.trade_history)) * 100.0
        wall = 100.0 * sum(1 for x in self.trade_history if float(x.get("pnl_frac", 0.0)) > 0) / len(self.trade_history)
        all_pnls = [float(x.get("pnl_frac", 0.0)) for x in self.trade_history]
        all_eq = np.cumprod([1.0 + p for p in all_pnls]) if all_pnls else np.asarray([1.0], dtype=float)
        pall = float(all_eq[-1] - 1.0) * 100.0 if len(all_eq) else 0.0
        all_peak = np.maximum.accumulate(all_eq) if len(all_eq) else np.asarray([1.0], dtype=float)
        all_dd = ((all_eq / np.maximum(all_peak, 1e-12)) - 1.0) if len(all_eq) else np.asarray([0.0], dtype=float)
        return {
            "pnl_24h": p24, "wr_24h": w24, "trades_24h": t24, "mdd_24h": m24,
            "pnl_7d": p7, "wr_7d": w7, "trades_7d": t7, "mdd_7d": m7,
            "pnl_all": pall, "wr_all": wall, "trades_all": len(self.trade_history), "mdd_all": float(all_dd.min() * 100.0) if len(all_dd) else 0.0,
            "pnl_24h_sum": p24_sum, "pnl_7d_sum": p7_sum, "pnl_all_sum": pall_sum,
            "cooldown_bars_left": int(self.cooldown_bars_left),
        }

    def performance_summary(self, now_kst) -> str:
        m = self.performance_metrics(now_kst)
        return f"perf 24h pnl:{m['pnl_24h']:+.2f}% wr:{m['wr_24h']:.0f}% | 7d pnl:{m['pnl_7d']:+.2f}% wr:{m['wr_7d']:.0f}% | all pnl:{m['pnl_all']:+.2f}% cd:{m['cooldown_bars_left']}"

    def print_meta_dashboard(self, result: dict, current_price: float = 0.0):
        C = Colors
        fa = int(result.get("final_action", 0))
        src = str(result.get("source", "N/A"))
        fa_arrow = {0: "─", 1: "▲", 2: "▼"}.get(fa, "?")
        fa_color = {0: C.YELLOW, 1: C.GREEN, 2: C.RED}.get(fa, C.RESET)
        fa_word = {0: "HOLD", 1: "LONG", 2: "SHORT"}.get(fa, "?")

        print(f" {fa_color}{C.BOLD}{fa_arrow}{fa_arrow}  {fa_word}{C.RESET}  score={float(result.get('rl_score', 0.0)):.3f}  Kelly={float(result.get('unified_kelly', 0.0)):.3f}  source: {C.CYAN}{src}{C.RESET}")
        print(f"  {C.CYAN}• RISK{C.RESET}    step_stop={'ON' if self.step_stop_enable else 'OFF'}  trail={self.governor_trail_arm:.3f}/{self.governor_trail_gap:.3f}  max_hold={self.governor_max_hold}  vol_scale={'ON' if self.governor_vol_scale_enable else 'OFF'}  cooldown={self.cooldown_bars_left}")

        if self.pos is not None:
            unr = self.unrealized_pnl(current_price)
            pos_color = C.GREEN if self.pos == "LONG" else C.RED
            unr_color = C.GREEN if unr > 0 else (C.RED if unr < 0 else C.YELLOW)
            print(f"  {pos_color}● 포지션{C.RESET}  {pos_color}{self.pos}{C.RESET}  진입가={self.entry_price:.2f}  미실현={unr_color}{unr:+.2f}%{C.RESET}  보유={self.hold_count}봉")


def _reset_virtual_router_state(router: GovernorPositionRouter) -> None:
    router.pos = None
    router.entry_price = 0.0
    router.hold_count = 0
    router.position_fraction = 0.0
    router.execution_leverage = 1.0
    router.current_leverage = 0.0
    router.peak_equity = 1.0
    router.cur_equity = 1.0
    router.last_realized_pnl = None
    router.last_closed_hold_count = 0
    router._open_trade_diag = None
    router.open_trade_id = ""
    router.opened_at = ""
    router.decision_at = ""
    router.strategy_state = {}
    router.trade_history = deque(maxlen=2000)
    router.recent_realized = deque(maxlen=20)
    router.loss_streak = 0
    router.cooldown_bars_left = 0
    router.trend_mismatch_streak = 0
    router.position_exit_streak = 0
    router.adaptive_enter_offset = 0.0
    router.adaptive_agreement_offset = 0.0
    router.adaptive_flat_cycles = 0


def _bootstrap_virtual_router(router: GovernorPositionRouter, live_state_path: str) -> None:
    router.live_state_path = str(live_state_path)
    _reset_virtual_router_state(router)
    if router.live_state_path and os.path.exists(router.live_state_path):
        router._load_live_state()
    else:
        router._save_live_state()


def _decode_exposure_bucket(exposure: float, cap: float = 3.0) -> tuple[float, float]:
    exp = float(np.clip(float(exposure or 0.0), 0.0, max(float(cap), 1.0)))
    if exp <= 1e-12:
        return 0.0, 1.0
    fraction = float(np.clip(min(exp, 1.0), 0.0, 1.0))
    leverage_mult = float(np.clip(exp / max(fraction, 1e-8), 1.0, max(float(cap), 1.0)))
    return fraction, leverage_mult
