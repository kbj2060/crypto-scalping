#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
LIVE_DIR = ROOT / "data" / "live"
TRADE_JOURNAL = LIVE_DIR / "trade_journal.jsonl"
POSITION_AUDIT = LIVE_DIR / "position_accounting_audit.jsonl"
DASHBOARD_EVENTS = LIVE_DIR / "dashboard_events.jsonl"
DASHBOARD_STATE = LIVE_DIR / "dashboard_state.json"
DASHBOARD_STATE_GOVERNOR = LIVE_DIR / "dashboard_state_governor.json"
GOVERNOR_LIVE_STATE = ROOT / "data" / "ensemble" / "governor_live_state.json"
REPORTS_DIR = LIVE_DIR

FEE_RATE = float(os.getenv("LIVE_FEE_RATE", "0.0005"))
SLIP_RATE = float(os.getenv("LIVE_SLIP_RATE", "0.0002"))
TIMEFRAME_MINUTES = 5
KST_OFFSET = pd.Timedelta(hours=9)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(_jsonable(payload), f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return str(value)
    try:
        import numpy as np

        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return [_jsonable(v) for v in value.tolist()]
    except Exception:
        pass
    if isinstance(value, float) and not math.isfinite(value):
        return 0.0
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _parse_kst(value: Any) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is not None:
        return ts.tz_convert("Asia/Seoul").tz_localize(None)
    return ts


def _fmt(ts: pd.Timestamp | None) -> str:
    if ts is None:
        return ""
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _floor_5m(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(ts).floor(f"{TIMEFRAME_MINUTES}min")


def _next_5m_after(ts: pd.Timestamp) -> pd.Timestamp:
    return _floor_5m(ts) + pd.Timedelta(minutes=TIMEFRAME_MINUTES)


def _to_utc_ms_from_kst(ts: pd.Timestamp) -> int:
    utc_naive = pd.Timestamp(ts) - KST_OFFSET
    return int(utc_naive.tz_localize(timezone.utc).timestamp() * 1000)


def _utc_text_from_kst(ts: pd.Timestamp) -> str:
    return _fmt(pd.Timestamp(ts) - KST_OFFSET)


def _event_recorded_time(row: dict[str, Any]) -> pd.Timestamp | None:
    kind = str(row.get("kind", "")).upper()
    if kind == "OPEN":
        keys = ("actual_opened_at", "event_recorded_at", "opened_at", "ts")
    elif kind == "CLOSE":
        keys = ("actual_closed_at", "event_recorded_at", "closed_at", "ts")
    elif kind == "RESIZE":
        keys = ("actual_resized_at", "event_recorded_at", "ts")
    else:
        keys = ("event_recorded_at", "ts")
    for key in keys:
        ts = _parse_kst(row.get(key))
        if ts is not None:
            return ts
    return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def _trade_math(side: str, entry_price: float, exit_price: float, exposure: float) -> dict[str, float]:
    side_u = str(side or "").upper()
    entry = float(entry_price or 0.0)
    exit_raw = float(exit_price or 0.0)
    lev = max(0.0, float(exposure or 0.0))
    if side_u not in {"LONG", "SHORT"} or entry <= 0.0 or exit_raw <= 0.0:
        return {
            "entry_exec_price": 0.0,
            "exit_exec_price": 0.0,
            "gross_return_frac": 0.0,
            "pnl_frac": 0.0,
            "pnl_pct": 0.0,
        }
    if side_u == "LONG":
        entry_exec = entry * (1.0 + SLIP_RATE)
        exit_exec = exit_raw * (1.0 - SLIP_RATE)
        gross = (exit_exec - entry_exec) / max(entry_exec, 1e-8)
    else:
        entry_exec = entry * (1.0 - SLIP_RATE)
        exit_exec = exit_raw * (1.0 + SLIP_RATE)
        gross = (entry_exec - exit_exec) / max(abs(entry_exec), 1e-8)
    pnl_frac = gross * lev - (2.0 * FEE_RATE * lev)
    return {
        "entry_exec_price": float(entry_exec),
        "exit_exec_price": float(exit_exec),
        "gross_return_frac": float(gross),
        "pnl_frac": float(pnl_frac),
        "pnl_pct": float(pnl_frac * 100.0),
    }


def _fetch_eth_5m(start_kst: pd.Timestamp, end_kst: pd.Timestamp) -> dict[pd.Timestamp, dict[str, Any]]:
    start_ms = _to_utc_ms_from_kst(start_kst)
    end_ms = _to_utc_ms_from_kst(end_kst)
    out: dict[pd.Timestamp, dict[str, Any]] = {}
    cur = start_ms
    url = "https://fapi.binance.com/fapi/v1/klines"
    while cur <= end_ms:
        params = {
            "symbol": "ETHUSDT",
            "interval": "5m",
            "startTime": cur,
            "endTime": end_ms,
            "limit": 1500,
        }
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        for row in data:
            utc = pd.to_datetime(int(row[0]), unit="ms")
            kst = utc + KST_OFFSET
            out[pd.Timestamp(kst)] = {
                "timestamp_utc": _fmt(utc),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
            }
        cur = int(data[-1][0]) + TIMEFRAME_MINUTES * 60 * 1000
        time.sleep(0.05)
    return out


def _row_old_execution_ts(row: dict[str, Any]) -> pd.Timestamp | None:
    return _parse_kst(row.get("execution_bar_ts")) or _parse_kst(row.get("ts"))


def _row_decision_bar_ts(row: dict[str, Any], old_exec: pd.Timestamp | None) -> pd.Timestamp | None:
    existing = _parse_kst(row.get("decision_bar_ts"))
    if existing is not None:
        return existing
    if old_exec is not None:
        return old_exec - pd.Timedelta(minutes=TIMEFRAME_MINUTES)
    return None


def _bar(candles: dict[pd.Timestamp, dict[str, Any]], ts: pd.Timestamp | None) -> dict[str, Any] | None:
    if ts is None:
        return None
    return candles.get(pd.Timestamp(ts))


def _apply_bar_contract(
    row: dict[str, Any],
    candles: dict[pd.Timestamp, dict[str, Any]],
    repair_ts: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    out = dict(row)
    old_exec = _row_old_execution_ts(out)
    decision_bar_ts = _row_decision_bar_ts(out, old_exec)
    recorded_ts = _event_recorded_time(out) or old_exec
    if recorded_ts is None:
        return out, {"status": "skipped", "reason": "missing_time"}
    decision_made_at = _floor_5m(recorded_ts)
    new_exec_ts = _next_5m_after(recorded_ts)
    if decision_bar_ts is None:
        decision_bar_ts = decision_made_at - pd.Timedelta(minutes=TIMEFRAME_MINUTES)

    decision_bar = _bar(candles, decision_bar_ts)
    exec_bar = _bar(candles, new_exec_ts)
    if exec_bar is None:
        return out, {
            "status": "skipped",
            "reason": "missing_execution_candle",
            "execute_at": _fmt(new_exec_ts),
        }

    original = {
        "ts": row.get("ts"),
        "decision_at": row.get("decision_at"),
        "opened_at": row.get("opened_at"),
        "closed_at": row.get("closed_at"),
        "actual_opened_at": row.get("actual_opened_at"),
        "actual_closed_at": row.get("actual_closed_at"),
        "event_recorded_at": row.get("event_recorded_at"),
        "entry_price": row.get("entry_price"),
        "exit_price": row.get("exit_price"),
        "price": row.get("price"),
        "ledger_ts_kind": row.get("ledger_ts_kind"),
        "execution_bar_ts": row.get("execution_bar_ts"),
        "execution_price": row.get("execution_price"),
    }
    out["audit_schema_version"] = "trade_journal.audit.v2"
    out["ledger_ts_kind"] = "scheduled_next_bar_open_repaired"
    out["ledger_repair"] = {
        "repaired_at": repair_ts,
        "reason": "live_ai_decision_must_execute_on_next_5m_open",
        "method": "decision_cycle_bar_plus_one_5m_open",
        "original": original,
    }
    out["decision_made_at_kst"] = _fmt(decision_made_at)
    out["decision_cycle_bar_ts"] = _fmt(decision_made_at)
    out["scheduled_execute_at_kst"] = _fmt(new_exec_ts)
    out["decision_bar_ts"] = _fmt(decision_bar_ts)
    out["decision_bar_utc"] = _utc_text_from_kst(decision_bar_ts)
    out["decision_bar_is_complete"] = True
    if decision_bar is not None:
        out["decision_bar_open"] = float(decision_bar["open"])
        out["decision_bar_high"] = float(decision_bar["high"])
        out["decision_bar_low"] = float(decision_bar["low"])
        out["decision_bar_close"] = float(decision_bar["close"])
        out["decision_bar_volume"] = float(decision_bar["volume"])
        out["decision_price"] = float(decision_bar["close"])
    else:
        out["decision_price"] = _safe_float(out.get("decision_price", out.get("entry_decision_price", 0.0)), 0.0)
    out["decision_price_source"] = "eth_buffer.close[-2]_repair"
    out["execution_bar_ts"] = _fmt(new_exec_ts)
    out["execution_bar_utc"] = _utc_text_from_kst(new_exec_ts)
    out["execution_bar_open"] = float(exec_bar["open"])
    out["execution_bar_high"] = float(exec_bar["high"])
    out["execution_bar_low"] = float(exec_bar["low"])
    out["execution_bar_close"] = float(exec_bar["close"])
    out["execution_bar_volume"] = float(exec_bar["volume"])
    out["execution_bar_is_current"] = False
    out["execution_price"] = float(exec_bar["open"])
    out["execution_price_source"] = "scheduled_next_bar_open_repair"
    out["execution_delay_sec"] = 0.0
    out["execution_delay_mode"] = "ledger_repair_scheduled"
    out["ts"] = _fmt(new_exec_ts)
    return out, {"status": "updated", "execute_at": _fmt(new_exec_ts), "execution_price": float(exec_bar["open"])}


def _repair_trade_rows(rows: list[dict[str, Any]], candles: dict[pd.Timestamp, dict[str, Any]], repair_ts: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    repaired: list[dict[str, Any]] = []
    open_by_trade: dict[str, dict[str, Any]] = {}
    stats = {"updated": 0, "skipped": 0, "skips": []}
    for idx, row in enumerate(rows):
        kind = str(row.get("kind", "")).upper()
        out, status = _apply_bar_contract(row, candles, repair_ts)
        if status.get("status") != "updated":
            stats["skipped"] += 1
            stats["skips"].append({"index": idx, **status})
            repaired.append(out)
            continue

        exec_price = _safe_float(out.get("execution_price"), 0.0)
        side = str(out.get("side", "")).upper()
        exposure = _safe_float(out.get("notional_exposure", out.get("total_exposure", 0.0)), 0.0)
        trade_id = str(out.get("trade_id", ""))
        if kind == "OPEN":
            entry = exec_price
            math_row = _trade_math(side, entry, entry, exposure)
            out["decision_at"] = str(out.get("decision_made_at_kst", ""))
            out["opened_at"] = str(out.get("scheduled_execute_at_kst", ""))
            out["actual_opened_at"] = str(out.get("scheduled_execute_at_kst", ""))
            out["event_recorded_at"] = repair_ts
            out["entry_price"] = float(entry)
            out["price"] = float(entry)
            out["entry_price_source"] = "scheduled_next_bar_open_repair"
            out["entry_decision_price"] = _safe_float(out.get("decision_price"), entry)
            out["entry_exec_price"] = float(math_row["entry_exec_price"])
            out["entry_exec_price_kind"] = "synthetic_fee_slippage_model"
            out["synthetic_entry_exec_price"] = float(math_row["entry_exec_price"])
            open_by_trade[trade_id] = dict(out)
        elif kind == "CLOSE":
            open_row = open_by_trade.get(trade_id)
            entry = _safe_float((open_row or {}).get("entry_price", out.get("entry_price")), 0.0)
            entry_decision = _safe_float((open_row or {}).get("entry_decision_price", out.get("entry_decision_price")), 0.0)
            entry_source = str((open_row or {}).get("entry_price_source", out.get("entry_price_source", "")) or "")
            opened_at = str((open_row or {}).get("opened_at", out.get("opened_at", "")) or "")
            decision_at = str((open_row or {}).get("decision_at", out.get("decision_at", "")) or "")
            math_row = _trade_math(side, entry, exec_price, exposure)
            out["decision_at"] = decision_at
            out["opened_at"] = opened_at
            out["actual_opened_at"] = opened_at
            out["closed_at"] = str(out.get("scheduled_execute_at_kst", ""))
            out["actual_closed_at"] = str(out.get("scheduled_execute_at_kst", ""))
            out["event_recorded_at"] = repair_ts
            out["entry_price"] = float(entry)
            out["price"] = float(exec_price)
            out["entry_price_source"] = entry_source
            out["entry_decision_price"] = float(entry_decision)
            out["entry_exec_price"] = float(math_row["entry_exec_price"])
            out["entry_exec_price_kind"] = "synthetic_fee_slippage_model"
            out["synthetic_entry_exec_price"] = float(math_row["entry_exec_price"])
            out["exit_price"] = float(exec_price)
            out["exit_price_source"] = "scheduled_next_bar_open_repair"
            out["exit_exec_price"] = float(math_row["exit_exec_price"])
            out["exit_exec_price_kind"] = "synthetic_fee_slippage_model"
            out["synthetic_exit_exec_price"] = float(math_row["exit_exec_price"])
            out["gross_return_frac"] = float(math_row["gross_return_frac"])
            out["pnl_frac"] = float(math_row["pnl_frac"])
            out["pnl_pct"] = float(math_row["pnl_pct"])
            out["remaining_position_pnl_frac"] = float(math_row["pnl_frac"])
            before_pos = _safe_float(out.get("position_realized_pnl_frac_before_close"), 0.0)
            out["total_position_pnl_frac_est"] = float(before_pos + math_row["pnl_frac"])
        else:
            out["event_recorded_at"] = repair_ts
        stats["updated"] += 1
        repaired.append(out)
    return repaired, stats


def _repair_audit_rows(
    rows: list[dict[str, Any]],
    candles: dict[pd.Timestamp, dict[str, Any]],
    repaired_trade_by_key: dict[tuple[str, str, str], dict[str, Any]],
    repair_ts: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    repaired: list[dict[str, Any]] = []
    stats = {"updated": 0, "skipped": 0, "skips": []}
    equity = _safe_float(rows[0].get("recognized_equity_before"), 1.0) if rows else 1.0
    prev_snapshot: dict[str, Any] = {}
    for idx, row in enumerate(rows):
        kind = str(row.get("kind", "")).upper()
        trade_id = str(row.get("trade_id", ""))
        key = (kind, trade_id, str(row.get("event", "")))
        trade = repaired_trade_by_key.get(key)
        if trade is None:
            out, status = _apply_bar_contract(row, candles, repair_ts)
            if status.get("status") != "updated":
                stats["skipped"] += 1
                stats["skips"].append({"index": idx, **status})
                repaired.append(out)
                continue
        else:
            out = dict(row)
            for k, v in trade.items():
                if k not in {"schema_version"}:
                    out[k] = v
            out["schema_version"] = "position_accounting_audit.v1"
        price = _safe_float(out.get("execution_price", out.get("price", 0.0)), 0.0)
        out["price"] = price
        if kind == "OPEN":
            out["entry_price"] = _safe_float(out.get("entry_price"), price)
            out["entry_exec_price"] = 0.0
            out["exit_price"] = 0.0
            out["exit_exec_price"] = 0.0
            out["recognized_equity_before"] = float(equity)
            out["recognized_equity_after"] = float(equity)
            out["recognized_equity_delta"] = 0.0
            out["recognized_return_frac"] = 0.0
            out["recognized_return_pct"] = 0.0
            out["realized_pnl_frac"] = 0.0
            out["realized_pnl_pct"] = 0.0
            out["raw_price_return_frac"] = 0.0
            out["exec_price_return_frac"] = 0.0
            out["raw_return_on_equity_frac"] = 0.0
            out["exec_return_on_equity_frac"] = 0.0
            prev_snapshot = dict(out)
        elif kind == "CLOSE":
            pnl = _safe_float(out.get("pnl_frac"), 0.0)
            before = float(equity)
            after = float(before * max(0.0, 1.0 + pnl))
            equity = after
            out["recognized_equity_before"] = before
            out["recognized_equity_after"] = after
            out["recognized_equity_delta"] = after - before
            out["recognized_return_frac"] = pnl
            out["recognized_return_pct"] = pnl * 100.0
            out["realized_pnl_frac"] = pnl
            out["realized_pnl_pct"] = pnl * 100.0
            side = str(out.get("side", "")).upper()
            entry = _safe_float(out.get("entry_price"), 0.0)
            exposure = _safe_float(out.get("notional_exposure", out.get("total_exposure", 0.0)), 0.0)
            if side == "LONG" and entry > 0 and price > 0:
                raw = (price - entry) / entry
            elif side == "SHORT" and entry > 0 and price > 0:
                raw = (entry - price) / entry
            else:
                raw = 0.0
            math_row = _trade_math(side, entry, price, exposure)
            out["raw_price_return_frac"] = raw
            out["exec_price_return_frac"] = float(math_row["gross_return_frac"])
            out["raw_return_on_equity_frac"] = raw * exposure
            out["exec_return_on_equity_frac"] = float(math_row["gross_return_frac"]) * exposure
            prev_snapshot = {}
        out["fee_rate"] = _safe_float(out.get("fee_rate"), FEE_RATE)
        out["slippage_rate"] = _safe_float(out.get("slippage_rate"), SLIP_RATE)
        event_notional = abs(_safe_float(out.get("delta_notional_exposure", out.get("event_notional_delta_abs", 0.0)), 0.0))
        if event_notional <= 0:
            event_notional = _safe_float(out.get("event_notional_delta_abs"), 0.0)
        out["event_notional_delta_abs"] = float(abs(event_notional))
        out["event_fee_cost_frac"] = float(abs(event_notional) * FEE_RATE)
        out["event_slippage_cost_est_frac"] = float(abs(event_notional) * SLIP_RATE)
        out["event_total_cost_est_frac"] = float(abs(event_notional) * (FEE_RATE + SLIP_RATE))
        if kind == "CLOSE":
            exposure = _safe_float(out.get("notional_exposure", out.get("total_exposure", 0.0)), 0.0)
            out["roundtrip_fee_cost_frac"] = float(2.0 * FEE_RATE * exposure)
            out["roundtrip_slippage_cost_frac"] = float(2.0 * SLIP_RATE * exposure)
            out["roundtrip_total_cost_frac"] = float(2.0 * (FEE_RATE + SLIP_RATE) * exposure)
            out["costs_recognized_in_strategy_equity"] = True
            out["cost_adjusted_equity_after_est"] = float(out["recognized_equity_after"])
        elif kind == "OPEN":
            out["roundtrip_fee_cost_frac"] = 0.0
            out["roundtrip_slippage_cost_frac"] = 0.0
            out["roundtrip_total_cost_frac"] = 0.0
            out["costs_recognized_in_strategy_equity"] = False
            out["cost_adjusted_equity_after_est"] = float(equity * max(0.0, 1.0 - out["event_total_cost_est_frac"]))
        stats["updated"] += 1
        repaired.append(out)
    return repaired, stats


def _trade_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("kind", "")).upper(), str(row.get("trade_id", "")), str(row.get("event", "")))


def _repair_dashboard_events(rows: list[dict[str, Any]], repaired_trades: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key = {_trade_key(r): r for r in repaired_trades}
    out_rows: list[dict[str, Any]] = []
    stats = {"updated": 0}
    for row in rows:
        out = dict(row)
        close_trade = out.get("close_trade") if isinstance(out.get("close_trade"), dict) else None
        open_trade = out.get("open_trade") if isinstance(out.get("open_trade"), dict) else None
        if close_trade:
            repl = by_key.get(_trade_key(close_trade))
            if repl:
                out["close_trade"] = dict(repl)
                out["ts"] = repl.get("ts", out.get("ts"))
                out["price"] = _safe_float(repl.get("exit_price", repl.get("execution_price", out.get("price"))), 0.0)
                out["pnl_pct"] = _safe_float(repl.get("pnl_pct"), 0.0)
                stats["updated"] += 1
        if open_trade:
            repl = by_key.get(_trade_key(open_trade))
            if repl:
                out["open_trade"] = dict(repl)
                out["ts"] = repl.get("ts", out.get("ts"))
                out["price"] = _safe_float(repl.get("entry_price", repl.get("execution_price", out.get("price"))), 0.0)
                stats["updated"] += 1
        out_rows.append(out)
    return out_rows, stats


def _latest_open_position(trades: list[dict[str, Any]]) -> dict[str, Any] | None:
    open_by_id: dict[str, dict[str, Any]] = {}
    for row in trades:
        kind = str(row.get("kind", "")).upper()
        tid = str(row.get("trade_id", ""))
        if kind == "OPEN":
            open_by_id[tid] = row
        elif kind == "CLOSE":
            open_by_id.pop(tid, None)
    if not open_by_id:
        return None
    return list(open_by_id.values())[-1]


def _repair_runtime_state(trades: list[dict[str, Any]], candles: dict[pd.Timestamp, dict[str, Any]], repair_ts: str) -> dict[str, Any]:
    stats = {"governor_live_state_updated": False, "dashboard_states_updated": []}
    latest_open = _latest_open_position(trades)
    latest_trade_ts = max((_parse_kst(r.get("ts")) for r in trades if _parse_kst(r.get("ts")) is not None), default=None)
    latest_bar_ts = max(candles.keys()) if candles else latest_trade_ts
    latest_price = _safe_float((candles.get(latest_bar_ts, {}) if latest_bar_ts else {}).get("close"), 0.0)

    state = _load_json(GOVERNOR_LIVE_STATE)
    if state and latest_open:
        side = str(latest_open.get("side", "")).upper()
        entry = _safe_float(latest_open.get("entry_price"), 0.0)
        exposure = _safe_float(latest_open.get("notional_exposure", latest_open.get("total_exposure", 0.0)), 0.0)
        mark = _trade_math(side, entry, latest_price, exposure) if latest_price > 0 else {"pnl_frac": 0.0}
        opened_ts = _parse_kst(latest_open.get("opened_at"))
        hold_count = int(max(0, ((latest_bar_ts - opened_ts).total_seconds() // 300))) if latest_bar_ts is not None and opened_ts is not None else int(state.get("hold_count", 0) or 0)
        state["pos"] = side
        state["entry_price"] = entry
        state["open_trade_id"] = str(latest_open.get("trade_id", ""))
        state["opened_at"] = str(latest_open.get("opened_at", ""))
        state["decision_at"] = str(latest_open.get("decision_at", ""))
        state["entry_price_source"] = str(latest_open.get("entry_price_source", ""))
        state["entry_decision_price"] = _safe_float(latest_open.get("entry_decision_price"), 0.0)
        state["current_exposure"] = exposure
        state["current_leverage"] = exposure
        state["position_fraction"] = _safe_float(latest_open.get("position_fraction"), 0.0)
        state["execution_leverage"] = _safe_float(latest_open.get("execution_leverage"), 1.0)
        state["hold_count"] = hold_count
        state["cur_equity"] = float(max(0.0, 1.0 + _safe_float(mark.get("pnl_frac"), 0.0)))
        state["saved_at"] = repair_ts
        repaired_closes = [r for r in trades if str(r.get("kind", "")).upper() == "CLOSE"]
        close_by_id = {str(r.get("trade_id", "")): r for r in repaired_closes}
        history = []
        for item in list(state.get("trade_history", []) or []):
            repl = close_by_id.get(str(item.get("trade_id", "")))
            history.append(dict(repl or item))
        state["trade_history"] = history[-2000:]
        recent = [_safe_float(r.get("pnl_frac"), 0.0) for r in state["trade_history"][-20:]]
        state["recent_realized"] = recent
        _write_json(GOVERNOR_LIVE_STATE, state)
        stats["governor_live_state_updated"] = True

    for path in (DASHBOARD_STATE, DASHBOARD_STATE_GOVERNOR):
        dash = _load_json(path)
        pos = dash.get("position") if isinstance(dash.get("position"), dict) else None
        if dash and pos and latest_open:
            side = str(latest_open.get("side", "")).upper()
            entry = _safe_float(latest_open.get("entry_price"), 0.0)
            exposure = _safe_float(latest_open.get("notional_exposure", latest_open.get("total_exposure", 0.0)), 0.0)
            mark = _trade_math(side, entry, latest_price, exposure) if latest_price > 0 else {"pnl_frac": 0.0, "pnl_pct": 0.0}
            equity = float(max(0.0, 1.0 + _safe_float(mark.get("pnl_frac"), 0.0)))
            pos["current"] = side
            pos["entry_price"] = entry
            pos["decision_at"] = str(latest_open.get("decision_at", ""))
            pos["opened_at"] = str(latest_open.get("opened_at", ""))
            pos["hold_bars"] = int(state.get("hold_count", pos.get("hold_bars", 0)) if state else pos.get("hold_bars", 0))
            pos["position_fraction"] = _safe_float(latest_open.get("position_fraction"), 0.0)
            pos["margin_fraction"] = _safe_float(latest_open.get("margin_fraction", latest_open.get("position_fraction", 0.0)), 0.0)
            pos["execution_leverage"] = _safe_float(latest_open.get("execution_leverage"), 1.0)
            pos["notional_exposure"] = exposure
            pos["total_exposure"] = exposure
            pos["unrealized_pnl_pct"] = _safe_float(mark.get("pnl_pct"), 0.0)
            pos["strategy_equity"] = equity
            pos["deployed_equity"] = equity * pos["position_fraction"]
            pos["gross_exposure_equity"] = equity * exposure
            pos["unrealized_pnl_amount"] = equity * (_safe_float(mark.get("pnl_pct"), 0.0) / 100.0)
            dash["price"] = latest_price or dash.get("price", 0.0)
            dash["updated_at"] = repair_ts
            _write_json(path, dash)
            stats["dashboard_states_updated"].append(str(path))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write repaired ledgers")
    args = ap.parse_args()

    repair_ts = pd.Timestamp.now(tz="Asia/Seoul").isoformat()
    stamp = pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y%m%d_%H%M%S")
    trade_rows = _load_jsonl(TRADE_JOURNAL)
    audit_rows = _load_jsonl(POSITION_AUDIT)
    event_rows = _load_jsonl(DASHBOARD_EVENTS)
    all_times = []
    for row in trade_rows + audit_rows:
        for value in (row.get("ts"), row.get("opened_at"), row.get("closed_at"), row.get("actual_opened_at"), row.get("actual_closed_at"), row.get("event_recorded_at")):
            ts = _parse_kst(value)
            if ts is not None:
                all_times.append(ts)
    if not all_times:
        raise SystemExit("no ledger timestamps found")
    start = min(all_times).floor("D") - pd.Timedelta(hours=1)
    end = max(all_times).ceil("D") + pd.Timedelta(days=1)
    candles = _fetch_eth_5m(start, end)
    if not candles:
        raise SystemExit("failed to fetch ETHUSDT candles")

    repaired_trades, trade_stats = _repair_trade_rows(trade_rows, candles, repair_ts)
    trade_by_key = {_trade_key(r): r for r in repaired_trades}
    repaired_audits, audit_stats = _repair_audit_rows(audit_rows, candles, trade_by_key, repair_ts)
    repaired_events, event_stats = _repair_dashboard_events(event_rows, repaired_trades)
    state_stats: dict[str, Any] = {}

    backup_dir = LIVE_DIR / "backups" / f"scheduled_next_open_ledger_repair_{stamp}"
    report = {
        "repair_ts": repair_ts,
        "apply": bool(args.apply),
        "backup_dir": str(backup_dir),
        "candles": {
            "start_kst": _fmt(min(candles.keys())),
            "end_kst": _fmt(max(candles.keys())),
            "count": len(candles),
        },
        "trade_journal": trade_stats,
        "position_accounting_audit": audit_stats,
        "dashboard_events": event_stats,
        "runtime_state": state_stats,
    }

    if args.apply:
        backup_dir.mkdir(parents=True, exist_ok=True)
        for path in (TRADE_JOURNAL, POSITION_AUDIT, DASHBOARD_EVENTS, DASHBOARD_STATE, DASHBOARD_STATE_GOVERNOR, GOVERNOR_LIVE_STATE):
            if path.exists():
                shutil.copy2(path, backup_dir / path.name)
        _write_jsonl(TRADE_JOURNAL, repaired_trades)
        _write_jsonl(POSITION_AUDIT, repaired_audits)
        _write_jsonl(DASHBOARD_EVENTS, repaired_events)
        state_stats = _repair_runtime_state(repaired_trades, candles, repair_ts)
        report["runtime_state"] = state_stats

    report_path = REPORTS_DIR / f"scheduled_next_open_ledger_repair_report_{stamp}.json"
    _write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
