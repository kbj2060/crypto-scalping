import numpy as np


def _safe_float(v, d: float = 0.0) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else float(d)
    except Exception:
        return float(d)

def _accounting_equity_from_history(rows) -> float:
    equity = 1.0
    for row in list(rows or []):
        pnl = _safe_float((row or {}).get("pnl_frac", (row or {}).get("pnl", 0.0)), 0.0)
        equity *= max(0.0, 1.0 + pnl)
    return float(equity)

def _price_return_frac(side: str, entry_price: float, exit_price: float) -> float:
    side_u = str(side or "").upper()
    entry = _safe_float(entry_price, 0.0)
    exit_px = _safe_float(exit_price, 0.0)
    if side_u not in {"LONG", "SHORT"} or entry <= 0.0 or exit_px <= 0.0:
        return 0.0
    if side_u == "LONG":
        return float((exit_px - entry) / max(entry, 1e-8))
    return float((entry - exit_px) / max(abs(entry), 1e-8))

def _build_position_accounting_audit_row(
    *,
    trade_row: dict,
    equity_before: float,
    equity_after: float,
    prev_snapshot: dict,
    new_snapshot: dict,
    current_price: float,
    from_pos: str | None,
    to_pos: str | None,
    decision_info: dict | None,
    fee_rate: float,
    slippage_rate: float,
) -> dict:
    row = dict(trade_row or {})
    prev = dict(prev_snapshot or {})
    new = dict(new_snapshot or {})
    info = dict(decision_info or {})
    kind = str(row.get("kind", "")).upper()
    fee_rate = _safe_float(fee_rate, 0.0)
    slip_rate = _safe_float(slippage_rate, 0.0)
    row_entry_fee_rate = _safe_float(row.get("entry_fee_rate", fee_rate), fee_rate)
    row_exit_fee_rate = _safe_float(row.get("exit_fee_rate", fee_rate), fee_rate)

    if kind == "OPEN":
        before_fraction = 0.0
        after_fraction = _safe_float(row.get("position_fraction", row.get("margin_fraction", 0.0)), 0.0)
        before_exec_lev = 1.0
        after_exec_lev = _safe_float(row.get("execution_leverage", 1.0), 1.0)
        before_notional = 0.0
        after_notional = _safe_float(row.get("notional_exposure", row.get("total_exposure", 0.0)), 0.0)
        delta_notional = after_notional
    elif kind == "RESIZE":
        before_fraction = _safe_float(row.get("prev_position_fraction", prev.get("position_fraction", 0.0)), 0.0)
        after_fraction = _safe_float(row.get("new_position_fraction", new.get("position_fraction", 0.0)), 0.0)
        before_exec_lev = _safe_float(row.get("prev_execution_leverage", prev.get("execution_leverage", 1.0)), 1.0)
        after_exec_lev = _safe_float(row.get("new_execution_leverage", new.get("execution_leverage", 1.0)), 1.0)
        before_notional = _safe_float(row.get("prev_notional_exposure", prev.get("notional_exposure", 0.0)), 0.0)
        after_notional = _safe_float(row.get("new_notional_exposure", new.get("notional_exposure", 0.0)), 0.0)
        delta_notional = _safe_float(row.get("delta_notional_exposure", after_notional - before_notional), 0.0)
    else:
        before_fraction = _safe_float(row.get("position_fraction", prev.get("position_fraction", 0.0)), 0.0)
        after_fraction = _safe_float(new.get("position_fraction", 0.0), 0.0)
        before_exec_lev = _safe_float(row.get("execution_leverage", prev.get("execution_leverage", 1.0)), 1.0)
        after_exec_lev = _safe_float(new.get("execution_leverage", 1.0), 1.0)
        before_notional = _safe_float(row.get("notional_exposure", row.get("total_exposure", prev.get("notional_exposure", 0.0))), 0.0)
        after_notional = _safe_float(new.get("notional_exposure", 0.0), 0.0)
        delta_notional = -before_notional

    event_notional_delta_abs = abs(float(delta_notional))
    if kind == "OPEN":
        event_fee_rate = float(row_entry_fee_rate)
    elif kind == "CLOSE":
        event_fee_rate = float(row_exit_fee_rate)
    elif kind == "RESIZE" and float(delta_notional) > 0.0:
        event_fee_rate = float(row_entry_fee_rate)
    elif kind == "RESIZE" and float(delta_notional) < 0.0:
        event_fee_rate = float(row_exit_fee_rate)
    else:
        event_fee_rate = float(fee_rate)
    event_fee_cost_frac = float(event_fee_rate * event_notional_delta_abs)
    event_slippage_cost_est_frac = float(slip_rate * event_notional_delta_abs)
    recognized_pnl_frac = _safe_float(row.get("pnl_frac", 0.0), 0.0) if kind in {"CLOSE", "RESIZE"} else 0.0
    raw_price_return = 0.0
    exec_price_return = 0.0
    raw_return_on_equity = 0.0
    exec_return_on_equity = 0.0
    roundtrip_fee_cost_frac = 0.0
    roundtrip_slippage_cost_frac = 0.0
    if kind == "CLOSE" or (kind == "RESIZE" and bool(row.get("costs_recognized_in_strategy_equity", False))):
        side = str(row.get("side", prev.get("pos", ""))).upper()
        entry_price = _safe_float(row.get("entry_price", prev.get("entry_price", 0.0)), 0.0)
        exit_price = _safe_float(row.get("exit_price", current_price), current_price)
        exposure_for_return = before_notional if kind == "CLOSE" else event_notional_delta_abs
        raw_price_return = _price_return_frac(side, entry_price, exit_price)
        exec_price_return = _safe_float(
            row.get("gross_return_frac", row.get("resize_gross_return_frac", raw_price_return)),
            raw_price_return,
        )
        raw_return_on_equity = float(raw_price_return * exposure_for_return)
        exec_return_on_equity = float(exec_price_return * exposure_for_return)
        roundtrip_slippage_cost_frac = float(max(0.0, raw_return_on_equity - exec_return_on_equity))
        roundtrip_fee_cost_frac = float(max(0.0, exec_return_on_equity - recognized_pnl_frac))

    event_total_cost_est_frac = float(event_fee_cost_frac + event_slippage_cost_est_frac)
    costs_recognized = bool(
        kind == "CLOSE"
        or (kind == "RESIZE" and bool(row.get("costs_recognized_in_strategy_equity", False)))
    )
    cost_adjusted_equity_after_est = float(equity_after if costs_recognized else equity_after * max(0.0, 1.0 - event_total_cost_est_frac))
    return {
        "schema_version": "position_accounting_audit.v1",
        "ts": str(row.get("ts", "")),
        "kind": kind,
        "event": str(row.get("event", "")),
        "trade_id": str(row.get("trade_id", "")),
        "from_pos": from_pos or "NONE",
        "to_pos": to_pos or "NONE",
        "side": str(row.get("side", prev.get("pos", new.get("pos", ""))) or "").upper(),
        "source": str(row.get("source", info.get("source", ""))),
        "reason": str(row.get("reason", info.get("position_reason", ""))),
        "decision_logic": str(info.get("decision_logic", "")),
        "model_version": str(row.get("model_version", info.get("model_version", ""))),
        "model_id": str(row.get("model_id", info.get("model_id", ""))),
        "model_path": str(row.get("model_path", info.get("model_path", ""))),
        "model_sleeve": str(row.get("model_sleeve", info.get("model_sleeve", ""))),
        "scout_prob": _safe_float(row.get("scout_prob", info.get("scout_prob", 0.0)), 0.0),
        "scout_frac": _safe_float(row.get("scout_frac", info.get("scout_frac", 0.0)), 0.0),
        "scout_probability_threshold": _safe_float(row.get("scout_probability_threshold", 0.0), 0.0),
        "scout_cost_pass": bool(row.get("scout_cost_pass", False)),
        "position_signal": str(info.get("position_signal", "")),
        "regime": str(row.get("regime", info.get("regime", ""))),
        "audit_schema_version": str(row.get("audit_schema_version", "")),
        "ledger_ts_kind": str(row.get("ledger_ts_kind", "")),
        "decision_made_at_kst": str(row.get("decision_made_at_kst", "")),
        "decision_bar_ts": str(row.get("decision_bar_ts", "")),
        "decision_bar_utc": str(row.get("decision_bar_utc", "")),
        "decision_bar_open": _safe_float(row.get("decision_bar_open", 0.0), 0.0),
        "decision_bar_high": _safe_float(row.get("decision_bar_high", 0.0), 0.0),
        "decision_bar_low": _safe_float(row.get("decision_bar_low", 0.0), 0.0),
        "decision_bar_close": _safe_float(row.get("decision_bar_close", row.get("decision_price", 0.0)), 0.0),
        "decision_bar_volume": _safe_float(row.get("decision_bar_volume", 0.0), 0.0),
        "decision_bar_is_complete": bool(row.get("decision_bar_is_complete", False)),
        "decision_price": _safe_float(row.get("decision_price", current_price), current_price),
        "decision_price_source": str(row.get("decision_price_source", "")),
        "execution_bar_ts": str(row.get("execution_bar_ts", "")),
        "execution_bar_utc": str(row.get("execution_bar_utc", "")),
        "execution_bar_open": _safe_float(row.get("execution_bar_open", row.get("execution_price", 0.0)), 0.0),
        "execution_bar_high": _safe_float(row.get("execution_bar_high", row.get("execution_price", 0.0)), 0.0),
        "execution_bar_low": _safe_float(row.get("execution_bar_low", row.get("execution_price", 0.0)), 0.0),
        "execution_bar_close": _safe_float(row.get("execution_bar_close", row.get("execution_price", 0.0)), 0.0),
        "execution_bar_volume": _safe_float(row.get("execution_bar_volume", 0.0), 0.0),
        "execution_bar_is_current": bool(row.get("execution_bar_is_current", False)),
        "execution_price": _safe_float(row.get("execution_price", current_price), current_price),
        "execution_price_source": str(row.get("execution_price_source", "")),
        "execution_delay_sec": _safe_float(row.get("execution_delay_sec", 0.0), 0.0),
        "execution_delay_late": bool(row.get("execution_delay_late", False)),
        "execution_delay_mode": str(row.get("execution_delay_mode", "")),
        "ai_timing": dict(row.get("ai_timing", {}) or {}),
        "event_recorded_at": str(row.get("event_recorded_at", "")),
        "actual_opened_at": str(row.get("actual_opened_at", row.get("opened_at", ""))),
        "actual_closed_at": str(row.get("actual_closed_at", "")),
        "actual_resized_at": str(row.get("actual_resized_at", "")),
        "price": _safe_float(current_price, 0.0),
        "entry_price": _safe_float(row.get("entry_price", new.get("entry_price", prev.get("entry_price", 0.0))), 0.0),
        "entry_price_source": str(row.get("entry_price_source", new.get("entry_price_source", prev.get("entry_price_source", "")))),
        "entry_decision_price": _safe_float(row.get("entry_decision_price", new.get("entry_decision_price", prev.get("entry_decision_price", 0.0))), 0.0),
        "entry_exec_price": _safe_float(row.get("entry_exec_price", row.get("resize_entry_exec_price", 0.0)), 0.0),
        "entry_exec_price_kind": str(row.get("entry_exec_price_kind", row.get("resize_exec_price_kind", ""))),
        "entry_execution_liquidity": str(row.get("entry_execution_liquidity", "")),
        "entry_execution_route": str(row.get("entry_execution_route", "")),
        "entry_execution_order_type": str(row.get("entry_execution_order_type", "")),
        "synthetic_entry_exec_price": _safe_float(row.get("synthetic_entry_exec_price", row.get("synthetic_resize_entry_exec_price", row.get("entry_exec_price", 0.0))), 0.0),
        "exit_price": _safe_float(row.get("exit_price", row.get("mark_price", 0.0)), 0.0),
        "exit_exec_price": _safe_float(row.get("exit_exec_price", row.get("resize_exit_exec_price", 0.0)), 0.0),
        "exit_exec_price_kind": str(row.get("exit_exec_price_kind", row.get("resize_exec_price_kind", ""))),
        "exit_execution_liquidity": str(row.get("exit_execution_liquidity", "")),
        "exit_execution_route": str(row.get("exit_execution_route", "")),
        "exit_execution_order_type": str(row.get("exit_execution_order_type", "")),
        "synthetic_exit_exec_price": _safe_float(row.get("synthetic_exit_exec_price", row.get("synthetic_resize_exit_exec_price", row.get("exit_exec_price", 0.0))), 0.0),
        "exchange_execution_enabled": bool(row.get("exchange_execution_enabled", False)),
        "exchange_execution_dry_run": bool(row.get("exchange_execution_dry_run", True)),
        "exchange_execution_status": str(row.get("exchange_execution_status", "")),
        "exchange_order_count": int(_safe_float(row.get("exchange_order_count", 0), 0.0)),
        "exchange_fill_price_source": str(row.get("exchange_fill_price_source", "")),
        "exchange_entry_price": _safe_float(row.get("exchange_entry_price", new.get("exchange_entry_price", prev.get("exchange_entry_price", 0.0))), 0.0),
        "exchange_exit_price": _safe_float(row.get("exchange_exit_price", 0.0), 0.0),
        "recognized_equity_before": float(equity_before),
        "recognized_equity_after": float(equity_after),
        "recognized_equity_delta": float(equity_after - equity_before),
        "recognized_return_frac": float((equity_after / max(equity_before, 1e-12)) - 1.0),
        "recognized_return_pct": float(((equity_after / max(equity_before, 1e-12)) - 1.0) * 100.0),
        "realized_pnl_frac": float(recognized_pnl_frac),
        "realized_pnl_pct": float(recognized_pnl_frac * 100.0),
        "raw_price_return_frac": float(raw_price_return),
        "exec_price_return_frac": float(exec_price_return),
        "raw_return_on_equity_frac": float(raw_return_on_equity),
        "exec_return_on_equity_frac": float(exec_return_on_equity),
        "before_position_fraction": float(before_fraction),
        "after_position_fraction": float(after_fraction),
        "delta_position_fraction": float(after_fraction - before_fraction),
        "before_execution_leverage": float(before_exec_lev),
        "after_execution_leverage": float(after_exec_lev),
        "before_notional_exposure": float(before_notional),
        "after_notional_exposure": float(after_notional),
        "delta_notional_exposure": float(delta_notional),
        "event_notional_delta_abs": float(event_notional_delta_abs),
        "fee_rate": float(fee_rate),
        "event_fee_rate": float(event_fee_rate),
        "entry_fee_rate": float(row_entry_fee_rate),
        "entry_fee_model": str(row.get("entry_fee_model", "")),
        "exit_fee_rate": float(row_exit_fee_rate),
        "exit_fee_model": str(row.get("exit_fee_model", "")),
        "roundtrip_fee_rate": float(_safe_float(row.get("roundtrip_fee_rate", row_entry_fee_rate + row_exit_fee_rate), row_entry_fee_rate + row_exit_fee_rate)),
        "fee_model": str(row.get("fee_model", "")),
        "fee_cost_frac": float(_safe_float(row.get("fee_cost_frac", 0.0), 0.0)),
        "slippage_rate": float(slip_rate),
        "event_fee_cost_frac": float(event_fee_cost_frac),
        "event_slippage_cost_est_frac": float(event_slippage_cost_est_frac),
        "event_total_cost_est_frac": float(event_total_cost_est_frac),
        "roundtrip_fee_cost_frac": float(roundtrip_fee_cost_frac),
        "roundtrip_slippage_cost_frac": float(roundtrip_slippage_cost_frac),
        "roundtrip_total_cost_frac": float(roundtrip_fee_cost_frac + roundtrip_slippage_cost_frac),
        "costs_recognized_in_strategy_equity": bool(costs_recognized),
        "cost_adjusted_equity_after_est": float(cost_adjusted_equity_after_est),
        "hold_bars": int(_safe_float(row.get("hold_bars", new.get("hold_bars", prev.get("hold_bars", 0))), 0.0)),
    }
