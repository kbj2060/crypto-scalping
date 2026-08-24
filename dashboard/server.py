from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import sys
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from aiohttp import ClientSession, ClientTimeout, web


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
# Reuses the exact, already-verified signal formulas from the standalone CLI dashboard rather
# than re-deriving them here -- see that module's docstring for formula provenance (each formula
# transcribed verbatim from the 2026-08-14 research scripts). compute_signals/bars_since_last_true
# are pure functions (no I/O, no sleep) so calling them synchronously inside an async handler is
# safe -- this file already does the same for scalp_shadow_payload()'s duckdb reads.
from scripts.live_evidence_signal_dashboard_20260823 import (  # noqa: E402
    FETCH_LIMIT as EVIDENCE_FETCH_LIMIT,
    PCTRANK_WINDOW as EVIDENCE_PCTRANK_WINDOW,
    SIGNAL_ORDER as EVIDENCE_SIGNAL_ORDER,
    bars_since_last_true,
    compute_signals,
)
# OI 급변 model indicator (replaces OBI, 2026-08-24) -- computed here from the poller's own
# duckdb, NOT from trading_bot.py's dashboard_state.json like the other 5 model indicators, so
# it never touches the live bot. See that module's docstring for the vol-lift validation.
from scripts.live_oi_delta_signal_20260824 import compute_oi_delta_signal  # noqa: E402
# Snapshot-tab liquidation map (estimated support/resistance, self-hosted Coinglass-heatmap
# alternative, 2026-08-24) -- discretionary reading aid only, NOT wired to trading_bot.py.
# Event-driven variant (2026-08-24): backtested for touch/hold win-rate only (60-67% vs a
# distance-matched random level over 4.7y), NOT for return/PnL -- see that module's
# compute_event_driven_levels() docstring for the full methodology and caveats.
from scripts.live_liquidation_map_20260824 import compute_event_driven_levels  # noqa: E402

EVIDENCE_SIGNAL_SYMBOL = "ETHUSDT"
EVIDENCE_SIGNAL_INTERVAL = "5m"
EVIDENCE_SIGNAL_CACHE_SECONDS = 60
EVIDENCE_SIGNAL_HISTORY_BARS = 48  # 4h strip for the Snapshot tab's per-bar activity graph
EVIDENCE_SIGNAL_BTC_SYMBOL = "BTCUSDT"  # smt_divergence's cross-asset non-confirmation leg, 2026-08-24
LIQUIDATION_MAP_SYMBOL = "ETHUSDT"
LIQUIDATION_MAP_INTERVAL = "1h"
LIQUIDATION_MAP_FETCH_LIMIT = 1000  # ~41.6 days at 1h -- feeds the event-driven state machine's
                                     # bootstrap+walk-forward (2026-08-24: fixed-7d -> event-driven,
                                     # see live_liquidation_map_20260824.compute_event_driven_levels
                                     # and eth_liquidation_map_event_driven_reset_20260824). Needs
                                     # enough history for both sides' reset points to settle before
                                     # "now" (median reset gap 44-54h), not just the 7d bootstrap window
LIQUIDATION_MAP_CACHE_SECONDS = 300  # structure moves slowly -- no need to recompute every tick

# The 6 model-internal indicators (microstructure/tail_risk) only ever have their LATEST reading
# persisted by trading_bot.py -- no history is stored anywhere. Rather than touch the live bot
# (would need a bot restart, open-position risk) or the browser (resets every page load), this
# dashboard SERVER -- already polling data/live/dashboard_state.json every EVENT_POLL_SECONDS in
# publish_dashboard_events() -- keeps its own small in-memory sample buffer, gated to a much
# coarser interval than that poll. It survives page refreshes and new browser sessions as long as
# THIS SERVER PROCESS stays up; it resets only on a dashboard-server restart (e.g. a deploy), not
# on every page load like the old client-only accumulation did. Raw values only -- the tone/hint
# thresholds stay in app.js (single source of truth), applied to this history same as to the
# live reading, so there is no second copy of that classification logic to drift out of sync.
MODEL_INDICATOR_SAMPLE_SECONDS = 300  # 5 min, matching the evidence-signal strip's bar cadence
MODEL_INDICATOR_HISTORY_MAX = 48  # 4h at the sample interval above -- same window as evidence signals
LIVE_DIR = REPO_ROOT / "data" / "live"
DASHBOARD_DIR = REPO_ROOT / "dashboard" / "live"
# Shadow-only, no order submission -- standalone loop separate from trading_bot.py's
# single-slot BTC shadow (see scripts/run_btc_multislot_shadow_loop_20260807.py).
BTC_MULTISLOT_SHADOW_STATE_PATH = REPO_ROOT / "data" / "ensemble" / "omega4_6_1_btc_multislot_shadow_state_20260807.json"
BTC_MULTISLOT_SHADOW_LEDGER_PATH = REPO_ROOT / "data" / "ensemble" / "omega4_6_1_btc_multislot_shadow_ledger_20260807.csv"
BTC_MULTISLOT_SHADOW_BAR_SECONDS = 300
# Odyssey4: h48qual regime-aware exit-head guard (Odyssey3, unchanged) + zig075 SHORT
# sustained-uptrend entry veto (Odyssey4 #1, CONFIRMED), shadow-only -- supersedes the retired
# eth-jmlam4-shadow (regime3 HMM->JM swap, never reproduced at N=5 seeds) and eth-exithead-shadow
# (h48qual exit_head liveATR relabel baseline, subsequently found NOT robust to walk-forward
# retraining) as of 2026-08-14 (see scripts/live_eth_odyssey4_zig075_entry_veto_shadow_20260814.py
# and docs/model_contracts/odyssey4_eth_entry_veto_baseline_contract_20260814.md).
ETH_ODYSSEY4_SHADOW_STATE_PATH = REPO_ROOT / "data" / "live" / "eth_odyssey4_shadow" / "state.json"
ETH_ODYSSEY4_SHADOW_TRADES_PATH = REPO_ROOT / "data" / "live" / "eth_odyssey4_shadow" / "closed_trades.jsonl"
ETH_ODYSSEY4_SHADOW_BAR_SECONDS = 300
MARKET_SYMBOLS = {"eth": "ETHUSDT", "sol": "SOLUSDT", "btc": "BTCUSDT"}
EVENT_POLL_SECONDS = 2.5
MARKET_HISTORY_CACHE_SECONDS = 300
SCALP_SHADOW_MODEL_ID = "eth_micro_scalp_source_stable_opportunity_moe_v4_20260718"
SCALP_SHADOW_STATE_SCHEMA = "eth_micro_scalp_v4.shadow_bot_step.v1"
SCALP_SHADOW_SUMMARY_SCHEMA = "eth_micro_scalp_v4.shadow_bot.v1"
SCALP_SHADOW_OBSERVER_SCHEMA = "eth_micro_scalp_v3.fresh_forward_observer.v1"
SCALP_SHADOW_FEES_BP = (2.0, 4.5, 5.5, 9.0)
SCALP_SHADOW_DISPLAY_FEE_BP = 4.5
SCALP_SHADOW_ASSETS = {
    "eth": {
        "asset": "eth",
        "model_id": SCALP_SHADOW_MODEL_ID,
        "state_schema": SCALP_SHADOW_STATE_SCHEMA,
        "summary_schema": SCALP_SHADOW_SUMMARY_SCHEMA,
        "observer_schema": SCALP_SHADOW_OBSERVER_SCHEMA,
        "state_file": "eth_micro_scalp_v4_shadow_state.json",
        "database_file": "eth_micro_scalp_v4_shadow.duckdb",
        "symbol": "ETHUSDT",
        "require_asset_contract": False,
    },
    "btc": {
        "asset": "btc",
        "model_id": "btc_micro_scalp_eth_v4_transfer_adapter_v1_20260718",
        "state_schema": "cross_asset_micro_scalp.shadow_bot_step.v1",
        "summary_schema": "cross_asset_micro_scalp.shadow_bot.v1",
        "observer_schema": "cross_asset_micro_scalp.shadow_observer.v1",
        "state_file": "btc_micro_scalp_shadow_state.json",
        "database_file": "btc_micro_scalp_shadow.duckdb",
        "symbol": "BTCUSDT",
        "require_asset_contract": True,
    },
    "sol": {
        "asset": "sol",
        "model_id": "sol_micro_scalp_eth_v4_transfer_adapter_v1_20260718",
        "state_schema": "cross_asset_micro_scalp.shadow_bot_step.v1",
        "summary_schema": "cross_asset_micro_scalp.shadow_bot.v1",
        "observer_schema": "cross_asset_micro_scalp.shadow_observer.v1",
        "state_file": "sol_micro_scalp_shadow_state.json",
        "database_file": "sol_micro_scalp_shadow.duckdb",
        "symbol": "SOLUSDT",
        "require_asset_contract": True,
    },
}
SCALP_REUSE_MODES = {
    "eth_lifecycle": {
        "asset": "eth",
        "mode": "eth_lifecycle",
        "model_id": "eth_micro_scalp_dynamic_lifecycle_shadow_v1_20260718",
        "state_schema": "micro_scalp_reuse.shadow_bot_step.v1",
        "summary_schema": "micro_scalp_reuse.shadow_bot.v1",
        "observer_schema": "micro_scalp_reuse.shadow_observer.v1",
        "state_file": "eth_micro_scalp_lifecycle_shadow_state.json",
        "database_file": "eth_micro_scalp_lifecycle_shadow.duckdb",
        "symbol": "ETHUSDT",
        "require_asset_contract": True,
    },
    "sol_entry": {
        "asset": "sol",
        "mode": "sol_entry",
        "model_id": "sol_micro_scalp_entry_only_shadow_v1_20260718",
        "state_schema": "micro_scalp_reuse.shadow_bot_step.v1",
        "summary_schema": "micro_scalp_reuse.shadow_bot.v1",
        "observer_schema": "micro_scalp_reuse.shadow_observer.v1",
        "state_file": "sol_micro_scalp_entry_shadow_state.json",
        "database_file": "sol_micro_scalp_entry_shadow.duckdb",
        "symbol": "SOLUSDT",
        "require_asset_contract": True,
    },
}


def file_signature(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_mtime_ns, stat.st_size


def make_etag(prefix: str, *parts: object) -> str:
    digest = hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()[:16]
    return f'W/"{prefix}-{digest}"'


def etag_matches(request: web.Request, etag: str) -> bool:
    candidates = request.headers.get("If-None-Match", "")
    return any(candidate.strip() in {"*", etag} for candidate in candidates.split(","))


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def row_ts(row: dict) -> float:
    raw = row.get("closed_at") or row.get("ts") or row.get("opened_at") or ""
    try:
        return time.mktime(time.strptime(str(raw).replace("T", " ")[:19], "%Y-%m-%d %H:%M:%S"))
    except ValueError:
        return 0.0


def strategy_tag(row: dict) -> str:
    basis = f"{row.get('source', '')} {row.get('raw_source', '')}".upper()
    if "SNIPER" in basis:
        return "SNIPER"
    if "TREND" in basis:
        return "TREND"
    if "MICRO" in basis or "WNC" in basis:
        return "MICRO"
    if "GOVERNOR" in basis or "CASH" in basis:
        return "GOVERNOR"
    if "COMPACT" in basis:
        return "COMPACT"
    if "CONTROLLER" in basis:
        return "CONTROLLER"
    return "GOVERNOR"


def pnl_pct(row: dict) -> float:
    if row.get("pnl_pct") is not None:
        return float(row.get("pnl_pct") or 0.0)
    if row.get("pnl_frac") is not None:
        return float(row.get("pnl_frac") or 0.0) * 100.0
    return 0.0


def equity_series(rows: list[dict], source_filter: str) -> list[dict]:
    closes = [r for r in rows if str(r.get("kind", "")).upper() == "CLOSE"]
    if source_filter != "ALL":
        closes = [r for r in closes if strategy_tag(r) == source_filter]
    closes.sort(key=row_ts)

    equity = 1.0
    out = []
    for idx, row in enumerate(closes, start=1):
        trade_pnl = pnl_pct(row)
        equity *= 1.0 + trade_pnl / 100.0
        out.append(
            {
                **row,
                "chart_index": idx,
                "pnl_pct": trade_pnl,
                "equity": equity,
                "cumulative_return_pct": (equity - 1.0) * 100.0,
                "ts": row.get("closed_at") or row.get("ts"),
            }
        )
    return out


def utc_iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        normalized = value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
        return normalized.isoformat(timespec="seconds").replace("+00:00", "Z")
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(" ", "T")
    if text.endswith("Z") or "+" in text[10:]:
        return text
    return text + "Z"


def utc_age_minutes(value: Any, *, offset_seconds: float = 0.0) -> float | None:
    """`offset_seconds` shifts `value` forward before computing age -- use this to convert an
    OHLCV bar's OPEN-time label (the raw kline convention, e.g. the last-bar timestamps this repo's
    shadow scripts store) into its CLOSE time (open + bar duration) for a freshness/staleness
    reading, since a bar is only actually usable/complete once it closes. Without this, "age" reads
    a full bar duration older than the data actually is. Default 0.0 preserves every existing
    caller's behavior exactly."""
    encoded = utc_iso(value)
    if encoded is None:
        return None
    try:
        timestamp = datetime.fromisoformat(encoded.replace("Z", "+00:00"))
    except ValueError:
        return None
    if offset_seconds:
        timestamp = timestamp + timedelta(seconds=offset_seconds)
    return max(0.0, (datetime.now(timezone.utc) - timestamp).total_seconds() / 60.0)


def _evidence_last_fired_ts(series: pd.Series, latest: pd.Series) -> str | None:
    """Exact UTC timestamp of the bar where `series` was last True, derived from
    bars_since_last_true()'s bar-offset via the fixed 5-minute evidence-signal bar spacing --
    None if it never fired in the loaded lookback."""
    bars = bars_since_last_true(series)
    if bars is None:
        return None
    return utc_iso(latest["timestamp"] - pd.Timedelta(minutes=5 * bars))


def _require_scalp_contract(condition: bool, field: str) -> None:
    if not condition:
        raise RuntimeError(f"scalp shadow contract mismatch: {field}")


def scalp_shadow_payload(
    live_dir: Path,
    asset: str = "eth",
    configs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    configs = SCALP_SHADOW_ASSETS if configs is None else configs
    config = configs.get(asset)
    _require_scalp_contract(config is not None, "asset")
    state_path = live_dir / config["state_file"]
    database_path = live_dir / config["database_file"]
    state = load_json(state_path)
    _require_scalp_contract(isinstance(state, dict), "state")
    summary = state.get("summary") or {}
    stream = state.get("stream") or {}
    _require_scalp_contract(state.get("schema_version") == config["state_schema"], "state.schema_version")
    _require_scalp_contract(state.get("model_id") == config["model_id"], "state.model_id")
    _require_scalp_contract(state.get("activation_allowed") is False, "state.activation_allowed")
    _require_scalp_contract(state.get("order_submission_supported") is False, "state.order_submission_supported")
    _require_scalp_contract(summary.get("schema_version") == config["summary_schema"], "summary.schema_version")
    _require_scalp_contract(summary.get("model_id") == config["model_id"], "summary.model_id")
    if config.get("require_asset_contract"):
        _require_scalp_contract(state.get("asset") == config["asset"], "state.asset")
        _require_scalp_contract(summary.get("asset") == config["asset"], "summary.asset")
        _require_scalp_contract(summary.get("symbol") == config["symbol"], "summary.symbol")
        if config.get("mode"):
            _require_scalp_contract(state.get("mode") == config["mode"], "state.mode")
            _require_scalp_contract(summary.get("mode") == config["mode"], "summary.mode")
    _require_scalp_contract(summary.get("performance_eligible") is False, "summary.performance_eligible")
    _require_scalp_contract(summary.get("order_submission_supported") is False, "summary.order_submission_supported")
    _require_scalp_contract(summary.get("fixed_holding_period_used") is False, "summary.fixed_holding_period_used")
    _require_scalp_contract(float(summary.get("unit_notional", 0.0)) == 1.0, "summary.unit_notional")
    _require_scalp_contract(
        summary.get("evidence_class") == "counterfactual completed-close-to-next-completed-close",
        "summary.evidence_class",
    )
    _require_scalp_contract(summary.get("fresh_forward_bar_by_bar") is True, "summary.fresh_forward_bar_by_bar")
    _require_scalp_contract(summary.get("trade_ledgers_used_as_input") is False, "summary.trade_ledgers_used_as_input")
    _require_scalp_contract(summary.get("saved_parent_exit_timestamps_used") is False, "summary.saved_parent_exit_timestamps_used")
    _require_scalp_contract(summary.get("future_rows_used_for_entry") is False, "summary.future_rows_used_for_entry")
    expected_fee_keys = {f"{fee:.2f}bp_per_notional_change" for fee in SCALP_SHADOW_FEES_BP}
    _require_scalp_contract(set((summary.get("fee_scenarios") or {}).keys()) == expected_fee_keys, "summary.fee_scenarios")
    _require_scalp_contract(database_path.exists(), "database")

    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        metadata = connection.execute(
            """
            SELECT schema_version, model_id, model_sha256, fresh_start_utc,
                   order_submission_supported
            FROM observer_metadata WHERE singleton = true
            """
        ).fetchone()
        _require_scalp_contract(metadata is not None, "observer_metadata")
        _require_scalp_contract(metadata[0] == config["observer_schema"], "observer_metadata.schema_version")
        _require_scalp_contract(metadata[1] == config["model_id"], "observer_metadata.model_id")
        _require_scalp_contract(metadata[2] == state.get("model_sha256"), "observer_metadata.model_sha256")
        _require_scalp_contract(bool(metadata[4]) is False, "observer_metadata.order_submission_supported")

        decision_count = int(connection.execute("SELECT count(*) FROM decisions").fetchone()[0])
        latest = connection.execute(
            """
            SELECT timestamp, close, target_position
            FROM decisions ORDER BY timestamp DESC LIMIT 1
            """
        ).fetchone()
        pnl_rows = connection.execute(
            """
            SELECT fee_bp, decision_timestamp, settlement_timestamp,
                   previous_position, position, turnover, gross_return,
                   cost_return, net_return, equity, causal_settlement
            FROM shadow_pnl
            ORDER BY fee_bp, decision_timestamp
            """
        ).fetchall()
        recent_rows = connection.execute(
            """
            SELECT d.timestamp, d.close, d.available, d.previous_position,
                   d.target_position, d.position_change, p.settlement_timestamp,
                   p.net_return, p.equity
            FROM decisions AS d
            LEFT JOIN shadow_pnl AS p
              ON p.decision_timestamp = d.timestamp AND p.fee_bp = ?
            ORDER BY d.timestamp DESC LIMIT 6
            """,
            [SCALP_SHADOW_DISPLAY_FEE_BP],
        ).fetchall()
    finally:
        connection.close()

    by_fee: dict[float, list[tuple[Any, ...]]] = {fee: [] for fee in SCALP_SHADOW_FEES_BP}
    for row in pnl_rows:
        fee = float(row[0])
        _require_scalp_contract(fee in by_fee, "shadow_pnl.fee_bp")
        _require_scalp_contract(bool(row[9]) and bool(row[9] > 0.0), "shadow_pnl.equity")
        _require_scalp_contract(row[10] is True, "shadow_pnl.causal_settlement")
        by_fee[fee].append(row)
    settled_counts = {len(rows) for rows in by_fee.values()}
    _require_scalp_contract(len(settled_counts) == 1, "shadow_pnl.partial_fee_set")
    settled_intervals = next(iter(settled_counts), 0)
    _require_scalp_contract(decision_count == int(summary.get("decision_count", -1)), "summary.decision_count")
    _require_scalp_contract(settled_intervals == int(summary.get("settled_intervals", -1)), "summary.settled_intervals")

    scenarios = []
    for fee in SCALP_SHADOW_FEES_BP:
        rows = by_fee[fee]
        equities = [float(row[9]) for row in rows]
        peak = 1.0
        max_drawdown = 0.0
        for equity in equities:
            peak = max(peak, equity)
            max_drawdown = max(max_drawdown, 1.0 - equity / peak)
        scenarios.append(
            {
                "fee_bp": fee,
                "compounded_return_pct": ((equities[-1] - 1.0) * 100.0) if equities else 0.0,
                "gross_return_pct": sum(float(row[6]) for row in rows) * 100.0,
                "cost_pct": sum(float(row[7]) for row in rows) * 100.0,
                "max_drawdown_pct": max_drawdown * 100.0,
            }
        )

    display_rows = by_fee[SCALP_SHADOW_DISPLAY_FEE_BP]
    displayed = next(row for row in scenarios if row["fee_bp"] == SCALP_SHADOW_DISPLAY_FEE_BP)
    positioned_intervals = sum(1 for row in display_rows if int(row[4]) != 0)
    position_changes = sum(1 for row in display_rows if float(row[5]) > 0.0)
    equity_rows = display_rows[-180:]
    latest_decision_utc = utc_iso(latest[0]) if latest is not None else None
    latest_feature_completed_utc = utc_iso(stream.get("latest_feature_completed_at_utc"))
    return {
        "contract": {
            "asset": config["asset"],
            "mode": config.get("mode"),
            "symbol": config["symbol"],
            "model_id": config["model_id"],
            "model_sha256": state.get("model_sha256"),
            "parent_model_id": summary.get("parent_model_id"),
            "research_policy_enabled": summary.get("research_policy_enabled", True),
            "dynamic_exit_enabled": summary.get("dynamic_exit_enabled"),
            "evidence_class": summary.get("evidence_class"),
            "actual_execution": False,
            "performance_eligible": False,
            "order_submission_supported": False,
            "fixed_holding_period_used": False,
            "unit_notional": float(summary.get("unit_notional")),
            "display_fee_bp": SCALP_SHADOW_DISPLAY_FEE_BP,
        },
        "health": {
            "latest_feature_completed_utc": latest_feature_completed_utc,
            "stream_age_minutes": utc_age_minutes(latest_feature_completed_utc),
            "latest_decision_utc": latest_decision_utc,
        },
        "summary": {
            "decision_count": decision_count,
            "settled_intervals": settled_intervals,
            "unsettled_decisions": max(0, decision_count - settled_intervals),
            "positioned_intervals": positioned_intervals,
            "position_changes": position_changes,
            "high_risk_bars": int(summary.get("high_risk_bars", 0)),
            "dynamic_exit_enabled": summary.get("dynamic_exit_enabled"),
            "pnl_sample_ready": positioned_intervals > 0,
            "current_position": int(latest[2]) if latest is not None else 0,
            "latest_close": float(latest[1]) if latest is not None else None,
            **displayed,
        },
        "fee_scenarios": scenarios,
        "equity": [
            {
                "ts": utc_iso(row[1]),
                "settlement_ts": utc_iso(row[2]),
                "position": int(row[4]),
                "net_return_pct": float(row[8]) * 100.0,
                "equity": float(row[9]),
                "cumulative_return_pct": (float(row[9]) - 1.0) * 100.0,
            }
            for row in equity_rows
        ],
        "recent_decisions": [
            {
                "ts": utc_iso(row[0]),
                "close": float(row[1]),
                "available": bool(row[2]),
                "previous_position": int(row[3]),
                "target_position": int(row[4]),
                "position_change": int(row[5]),
                "settlement_ts": utc_iso(row[6]),
                "net_return_pct": float(row[7]) * 100.0 if row[7] is not None else None,
                "equity": float(row[8]) if row[8] is not None else None,
            }
            for row in recent_rows
        ],
    }


def btc_multislot_shadow_payload() -> dict[str, Any]:
    state = load_json(BTC_MULTISLOT_SHADOW_STATE_PATH) or {}
    slots = state.get("slots") if isinstance(state.get("slots"), list) else []
    last_bar = state.get("last_bar")
    age_minutes = utc_age_minutes(last_bar, offset_seconds=BTC_MULTISLOT_SHADOW_BAR_SECONDS)
    total_trades = 0
    cumulative_return_pct: float | None = None
    recent_trades: list[dict[str, Any]] = []
    equity_curve: list[dict[str, Any]] = []
    if BTC_MULTISLOT_SHADOW_LEDGER_PATH.exists():
        with BTC_MULTISLOT_SHADOW_LEDGER_PATH.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        total_trades = len(rows)
        equity = 1.0
        for row in rows:
            try:
                return_frac = float(row.get("trade_return_net") or 0.0)
            except (TypeError, ValueError):
                continue
            equity *= 1.0 + return_frac
            equity_curve.append(
                {
                    "ts": row.get("exit_timestamp"),
                    "slot": row.get("slot"),
                    "side": row.get("side"),
                    "trade_return_pct": return_frac * 100.0,
                    "cumulative_return_pct": (equity - 1.0) * 100.0,
                }
            )
        cumulative_return_pct = (equity - 1.0) * 100.0
        equity_curve = equity_curve[-500:]
        for row in rows[-5:]:
            try:
                return_pct = float(row["trade_return_net"]) * 100.0
            except (KeyError, TypeError, ValueError):
                return_pct = None
            recent_trades.append(
                {
                    "slot": row.get("slot"),
                    "side": row.get("side"),
                    "entry_timestamp": row.get("entry_timestamp"),
                    "exit_timestamp": row.get("exit_timestamp"),
                    "entry_price": row.get("entry_price"),
                    "exit_price": row.get("exit_price"),
                    "trade_return_pct": return_pct,
                    "reason": row.get("reason"),
                }
            )
        recent_trades.reverse()
    return {
        "last_bar": last_bar,
        "age_minutes": age_minutes,
        "stale": age_minutes is None or age_minutes >= (BTC_MULTISLOT_SHADOW_BAR_SECONDS / 60.0) * 3,
        "bar_seconds": BTC_MULTISLOT_SHADOW_BAR_SECONDS,
        "slot_count": len(slots),
        "open_slots": sum(1 for s in slots if s),
        "slots": slots,
        "total_trades": total_trades,
        "cumulative_return_pct": cumulative_return_pct,
        "recent_trades": recent_trades,
        "equity_curve": equity_curve,
    }


def eth_odyssey4_shadow_payload() -> dict[str, Any]:
    state = load_json(ETH_ODYSSEY4_SHADOW_STATE_PATH) or {}
    last_bar = state.get("last_processed_bar_ts")
    age_minutes = utc_age_minutes(last_bar, offset_seconds=ETH_ODYSSEY4_SHADOW_BAR_SECONDS)
    position = state.get("position") if isinstance(state.get("position"), dict) else None
    trades = parse_jsonl(ETH_ODYSSEY4_SHADOW_TRADES_PATH)
    total_trades = len(trades)
    equity = state.get("equity")
    cumulative_return_pct = (float(equity) - 1.0) * 100.0 if isinstance(equity, (int, float)) else None
    mdd_pct = float(state["mdd"]) * 100.0 if isinstance(state.get("mdd"), (int, float)) else None
    recent_trades: list[dict[str, Any]] = []
    equity_curve: list[dict[str, Any]] = []
    running_equity = 1.0
    for row in trades:
        try:
            return_frac = float(row["trade_return_frac"])
        except (KeyError, TypeError, ValueError):
            continue
        running_equity *= 1.0 + return_frac
        equity_curve.append(
            {
                "ts": row.get("exit_ts"),
                "side": row.get("side"),
                "trade_return_pct": return_frac * 100.0,
                "cumulative_return_pct": (running_equity - 1.0) * 100.0,
            }
        )
    equity_curve = equity_curve[-500:]
    for row in trades[-5:]:
        try:
            return_pct = float(row["trade_return_frac"]) * 100.0
        except (KeyError, TypeError, ValueError):
            return_pct = None
        recent_trades.append(
            {
                "source_component": row.get("source_component"),
                "side": row.get("side"),
                "entry_ts": row.get("entry_ts"),
                "exit_ts": row.get("exit_ts"),
                "entry_price": row.get("entry_price"),
                "exit_price": row.get("exit_price"),
                "trade_return_pct": return_pct,
                "reason": row.get("reason"),
            }
        )
    recent_trades.reverse()
    return {
        "last_bar": last_bar,
        "age_minutes": age_minutes,
        "stale": age_minutes is None or age_minutes >= (ETH_ODYSSEY4_SHADOW_BAR_SECONDS / 60.0) * 3,
        "bar_seconds": ETH_ODYSSEY4_SHADOW_BAR_SECONDS,
        "position_side": (position or {}).get("side", 0),
        "position_source_component": (position or {}).get("source_component"),
        "position": position,
        "total_trades": total_trades,
        "cumulative_return_pct": cumulative_return_pct,
        "mdd_pct": mdd_pct,
        "recent_trades": recent_trades,
        "equity_curve": equity_curve,
        "candidate": state.get("candidate"),
        "order_submission_supported": state.get("order_submission_supported", False),
        "h48qual_guard_active_bars": state.get("h48qual_guard_active_bars", 0),
        "zig075_short_veto_bars": state.get("zig075_short_veto_bars", 0),
        "h48qual_quality_score": state.get("last_h48qual_quality_score"),
        "h48qual_quality_threshold": state.get("last_h48qual_quality_threshold"),
        "zig075_quality_score": state.get("last_zig075_quality_score"),
        "zig075_quality_threshold": state.get("last_zig075_quality_threshold"),
    }


def no_cache(resp: web.StreamResponse) -> web.StreamResponse:
    resp.headers["Cache-Control"] = "no-cache"
    return resp


def json_response(request: web.Request, payload: Any, etag: str) -> web.Response:
    headers = {"ETag": etag, "Cache-Control": "no-cache"}
    if etag_matches(request, etag):
        return web.Response(status=web.HTTPNotModified.status_code, headers=headers)

    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    response = web.Response(body=body, content_type="application/json", headers=headers)
    response.enable_compression()
    return response


def make_app() -> web.Application:
    @web.middleware
    async def static_asset_headers(
        request: web.Request,
        handler: Any,
    ) -> web.StreamResponse:
        response = await handler(request)
        asset_name = Path(request.path).name
        if asset_name in {"app.js", "styles.css"}:
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
            response.enable_compression()
        elif asset_name.endswith(".ttf"):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response

    app = web.Application(middlewares=[static_asset_headers])
    json_cache: dict[Path, tuple[tuple[int, int] | None, Any]] = {}
    trade_cache: dict[str, Any] = {
        "signature": None,
        "rows": [],
        "payloads": {},
    }
    event_clients: set[asyncio.Queue[str]] = set()
    latest_event_state: dict[str, Any] | None = None
    latest_event_tickers: dict[str, dict[str, Any]] = {}
    market_history_cache: dict[str, tuple[float, list[dict[str, float | int]]]] = {}
    market_history_locks = {asset: asyncio.Lock() for asset in MARKET_SYMBOLS}
    evidence_signal_cache: dict[str, Any] = {"ts": 0.0, "payload": None}
    evidence_signal_lock = asyncio.Lock()
    oi_signal_cache: dict[str, Any] = {"ts": 0.0, "payload": None}
    oi_signal_lock = asyncio.Lock()
    liquidation_map_cache: dict[str, Any] = {"ts": 0.0, "payload": None}
    liquidation_map_lock = asyncio.Lock()
    model_indicator_history: deque = deque(maxlen=MODEL_INDICATOR_HISTORY_MAX)
    model_indicator_sample_state: dict[str, float] = {"last_sample_at": 0.0}

    def load_json_cached(path: Path, signature: tuple[int, int] | None = None) -> Any:
        if signature is None:
            signature = file_signature(path)
        cached = json_cache.get(path)
        if cached and cached[0] == signature:
            return cached[1]
        payload = load_json(path)
        json_cache[path] = (signature, payload)
        return payload

    def cached_trade_rows(path: Path, signature: tuple[int, int] | None = None) -> list[dict]:
        if signature is None:
            signature = file_signature(path)
        if trade_cache["signature"] == signature:
            return trade_cache["rows"]

        rows = []
        for row in parse_jsonl(path):
            raw_source = row.get("source", "")
            tagged = {**row, "raw_source": raw_source}
            tagged["source"] = strategy_tag(tagged)
            rows.append(tagged)
        rows.sort(key=row_ts)
        trade_cache.update(signature=signature, rows=rows, payloads={})
        return rows

    def dashboard_state_payload() -> tuple[dict[str, Any], str]:
        state_path = LIVE_DIR / "dashboard_state.json"
        governor_path = LIVE_DIR / "dashboard_state_governor.json"
        dsac_path = LIVE_DIR / "dashboard_state_dsac_compact.json"
        state_sig = file_signature(state_path)
        governor_sig = file_signature(governor_path)
        dsac_sig = file_signature(dsac_path)
        etag = make_etag("state", state_sig, governor_sig, dsac_sig)
        return {
            "state": load_json_cached(state_path, state_sig),
            "compactState": load_json_cached(governor_path, governor_sig) or load_json_cached(dsac_path, dsac_sig),
        }, etag

    async def fetch_market_ticker(session: ClientSession, asset: str, symbol: str) -> tuple[str, dict[str, Any] | None]:
        try:
            async with session.get(
                "https://fapi.binance.com/fapi/v1/ticker/price",
                params={"symbol": symbol},
            ) as response:
                if response.status != web.HTTPOk.status_code:
                    return asset, None
                payload = await response.json()
        except (asyncio.TimeoutError, OSError, ValueError):
            return asset, None
        try:
            price = float(payload["price"])
        except (KeyError, TypeError, ValueError):
            return asset, None
        if price <= 0:
            return asset, None
        return asset, {"symbol": symbol, "price": price, "ts": datetime.now(timezone.utc).isoformat()}

    async def load_market_history(asset: str) -> list[dict[str, float | int]]:
        cached = market_history_cache.get(asset)
        now = time.monotonic()
        if cached and now - cached[0] < MARKET_HISTORY_CACHE_SECONDS:
            return cached[1]
        async with market_history_locks[asset]:
            cached = market_history_cache.get(asset)
            now = time.monotonic()
            if cached and now - cached[0] < MARKET_HISTORY_CACHE_SECONDS:
                return cached[1]
            async with ClientSession(timeout=ClientTimeout(total=3)) as session:
                async with session.get(
                    "https://fapi.binance.com/fapi/v1/klines",
                    params={"symbol": MARKET_SYMBOLS[asset], "interval": "5m", "limit": 100},
                ) as response:
                    if response.status != web.HTTPOk.status_code:
                        raise web.HTTPBadGateway(reason="market_history_upstream_error")
                    rows = await response.json()
            candles = [
                {
                    "time": int(row[0]) // 1000,
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                }
                for row in rows
            ]
            market_history_cache[asset] = (time.monotonic(), candles)
            return candles

    async def load_evidence_signals() -> dict[str, Any]:
        """Informational-only reversal-evidence-signal readout for the Snapshot tab -- NOT a
        trading signal (see docstring in the imported module). Mirrors load_market_history()'s
        cache/lock pattern but needs a much longer klines window (EVIDENCE_FETCH_LIMIT bars, to
        warm up orthogonal_combo's EVIDENCE_PCTRANK_WINDOW-bar percentile-rank window) than the
        chart's own /api/market-history (limit=100), so it gets its own cache rather than sharing
        market_history_cache."""
        now = time.monotonic()
        cached = evidence_signal_cache["payload"]
        if cached is not None and now - evidence_signal_cache["ts"] < EVIDENCE_SIGNAL_CACHE_SECONDS:
            return cached
        async with evidence_signal_lock:
            cached = evidence_signal_cache["payload"]
            if cached is not None and time.monotonic() - evidence_signal_cache["ts"] < EVIDENCE_SIGNAL_CACHE_SECONDS:
                return cached
            async with ClientSession(timeout=ClientTimeout(total=10)) as session:
                async with session.get(
                    "https://fapi.binance.com/fapi/v1/klines",
                    params={
                        "symbol": EVIDENCE_SIGNAL_SYMBOL,
                        "interval": EVIDENCE_SIGNAL_INTERVAL,
                        "limit": EVIDENCE_FETCH_LIMIT,
                    },
                ) as response:
                    if response.status != web.HTTPOk.status_code:
                        raise web.HTTPBadGateway(reason="evidence_signal_upstream_error")
                    raw = await response.json()
            cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
                    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
            df = pd.DataFrame(raw, columns=cols)
            for c in ("open", "high", "low", "close", "volume", "taker_buy_base"):
                df[c] = df[c].astype("float64")
            df["close_time"] = df["close_time"].astype("int64")
            df["timestamp"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
            df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
            now_ms = int(time.time() * 1000)
            if len(df) and int(df.iloc[-1]["close_time"]) >= now_ms:
                df = df.iloc[:-1].reset_index(drop=True)  # drop the still-forming bar

            # BTC leg for smt_divergence (2026-08-24) -- failure here must never take down the
            # ETH-only signals, so it's caught and logged, not raised; compute_signals() degrades
            # smt_divergence to not-fired when btc_df is None.
            btc_df = None
            try:
                async with ClientSession(timeout=ClientTimeout(total=10)) as btc_session:
                    async with btc_session.get(
                        "https://fapi.binance.com/fapi/v1/klines",
                        params={
                            "symbol": EVIDENCE_SIGNAL_BTC_SYMBOL,
                            "interval": EVIDENCE_SIGNAL_INTERVAL,
                            "limit": EVIDENCE_FETCH_LIMIT,
                        },
                    ) as btc_response:
                        if btc_response.status == web.HTTPOk.status_code:
                            braw = await btc_response.json()
                            bdf = pd.DataFrame(braw, columns=cols)
                            for c in ("high", "low"):
                                bdf[c] = bdf[c].astype("float64")
                            bdf["close_time"] = bdf["close_time"].astype("int64")
                            bdf["timestamp"] = pd.to_datetime(bdf["open_time"].astype("int64"), unit="ms", utc=True)
                            bdf = bdf.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
                            if len(bdf) and int(bdf.iloc[-1]["close_time"]) >= now_ms:
                                bdf = bdf.iloc[:-1].reset_index(drop=True)
                            btc_df = bdf[["timestamp", "high", "low"]]
            except Exception as btc_exc:  # noqa: BLE001 -- ETH signals must still render this cycle
                print(f"evidence-signal BTC leg failed (smt_divergence family will read as "
                      f"not-fired this cycle): {btc_exc}", flush=True)

            sig = compute_signals(df, btc_df=btc_df)
            latest = sig.iloc[-1] if len(sig) else None
            warmed_up = latest is not None and pd.notna(latest.get("p_fast")) and pd.notna(latest.get("p_slow"))
            signals_payload = []
            for name, description in EVIDENCE_SIGNAL_ORDER:
                bcol, tcol = f"bottom_{name}", f"top_{name}"
                # _active = 1h sustain window (2026-08-24) -- rolling-max of the raw bcol/tcol
                # firing column, not a new/looser firing condition (see compute_signals()
                # docstring). last_fired_ts always reads the RAW column so it keeps reporting the
                # true original firing bar even while _active keeps the chip lit.
                bacol, tacol = f"{bcol}_active", f"{tcol}_active"
                signals_payload.append({
                    "name": name,
                    "description": description,
                    "bottom_fired": bool(latest[bacol]) if warmed_up else None,
                    "bottom_last_fired_ts": _evidence_last_fired_ts(sig[bcol], latest) if warmed_up else None,
                    "top_fired": bool(latest[tacol]) if warmed_up else None,
                    "top_last_fired_ts": _evidence_last_fired_ts(sig[tcol], latest) if warmed_up else None,
                    # Oldest-to-newest, for the Snapshot tab's activity-strip graph (one cell/bar).
                    "bottom_history": sig[bacol].tail(EVIDENCE_SIGNAL_HISTORY_BARS).fillna(False).astype(bool).tolist() if warmed_up else [],
                    "top_history": sig[tacol].tail(EVIDENCE_SIGNAL_HISTORY_BARS).fillna(False).astype(bool).tolist() if warmed_up else [],
                })
            payload = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "latest_bar_utc": utc_iso(latest["timestamp"]) if latest is not None else None,
                "price": float(latest["close"]) if latest is not None else None,
                "bars_loaded": int(len(sig)),
                "warmed_up": bool(warmed_up),
                "net_score": int(latest["net_score"]) if warmed_up else None,
                "bottom_votes": int(latest["bottom_votes"]) if warmed_up else None,
                "top_votes": int(latest["top_votes"]) if warmed_up else None,
                "signals": signals_payload,
            }
            evidence_signal_cache["ts"] = time.monotonic()
            evidence_signal_cache["payload"] = payload
            return payload

    async def load_oi_signal() -> dict[str, Any]:
        """OI 급변 model indicator -- see scripts/live_oi_delta_signal_20260824.py docstring for
        the vol-lift validation and why this is computed HERE (dashboard-side) rather than by
        trading_bot.py. compute_oi_delta_signal() retries through the same brief read-vs-write
        lock contention window ops_watchdog.py already retries for tail_risk.duckdb (up to ~2.8s
        of blocking time.sleep across attempts) -- run via asyncio.to_thread so that wait never
        stalls this process's event loop (and every other concurrent dashboard request) the way
        calling it inline would."""
        now = time.monotonic()
        cached = oi_signal_cache["payload"]
        if cached is not None and now - oi_signal_cache["ts"] < EVIDENCE_SIGNAL_CACHE_SECONDS:
            return cached
        async with oi_signal_lock:
            cached = oi_signal_cache["payload"]
            if cached is not None and time.monotonic() - oi_signal_cache["ts"] < EVIDENCE_SIGNAL_CACHE_SECONDS:
                return cached
            payload = await asyncio.to_thread(compute_oi_delta_signal)
            oi_signal_cache["ts"] = time.monotonic()
            oi_signal_cache["payload"] = payload
            return payload

    async def load_liquidation_map() -> dict[str, Any]:
        """Snapshot-tab liquidation map (estimated support/resistance) -- see
        scripts/live_liquidation_map_20260824.py docstring for the estimation methodology and its
        caveats. Mirrors load_evidence_signals()'s klines-fetch/cache pattern (own cache, since
        this needs a much longer 1h lookback than the chart's own /api/market-history)."""
        now = time.monotonic()
        cached = liquidation_map_cache["payload"]
        if cached is not None and now - liquidation_map_cache["ts"] < LIQUIDATION_MAP_CACHE_SECONDS:
            return cached
        async with liquidation_map_lock:
            cached = liquidation_map_cache["payload"]
            if cached is not None and time.monotonic() - liquidation_map_cache["ts"] < LIQUIDATION_MAP_CACHE_SECONDS:
                return cached
            async with ClientSession(timeout=ClientTimeout(total=10)) as session:
                async with session.get(
                    "https://fapi.binance.com/fapi/v1/klines",
                    params={
                        "symbol": LIQUIDATION_MAP_SYMBOL,
                        "interval": LIQUIDATION_MAP_INTERVAL,
                        "limit": LIQUIDATION_MAP_FETCH_LIMIT,
                    },
                ) as response:
                    if response.status != web.HTTPOk.status_code:
                        raise web.HTTPBadGateway(reason="liquidation_map_upstream_error")
                    raw = await response.json()
            cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
                    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
            df = pd.DataFrame(raw, columns=cols)
            for c in ("high", "low", "close", "volume"):
                df[c] = df[c].astype("float64")
            df["close_time"] = df["close_time"].astype("int64")
            df["timestamp"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
            df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
            now_ms = int(time.time() * 1000)
            if len(df) and int(df.iloc[-1]["close_time"]) >= now_ms:
                df = df.iloc[:-1].reset_index(drop=True)  # drop the still-forming bar

            current_price = float(df["close"].iloc[-1]) if len(df) else 0.0
            payload = compute_event_driven_levels(df, current_price)
            payload["generated_at"] = datetime.now(timezone.utc).isoformat()
            liquidation_map_cache["ts"] = time.monotonic()
            liquidation_map_cache["payload"] = payload
            return payload

    async def publish_dashboard_events(app: web.Application) -> None:
        nonlocal latest_event_state, latest_event_tickers
        last_state_etag = ""
        timeout = ClientTimeout(total=2)
        async with ClientSession(timeout=timeout) as session:
            while True:
                started = time.monotonic()
                state_payload, state_etag = dashboard_state_payload()
                if started - model_indicator_sample_state["last_sample_at"] >= MODEL_INDICATOR_SAMPLE_SECONDS:
                    model_indicator_sample_state["last_sample_at"] = started
                    raw_state = (state_payload or {}).get("state") or {}
                    model_indicator_history.append({
                        "sampled_at": datetime.now(timezone.utc).isoformat(),
                        "microstructure": raw_state.get("microstructure") or {},
                        "tail_risk": raw_state.get("tail_risk") or {},
                    })
                ticker_rows = await asyncio.gather(
                    *(fetch_market_ticker(session, asset, symbol) for asset, symbol in MARKET_SYMBOLS.items())
                )
                latest_event_tickers = {
                    asset: ticker for asset, ticker in ticker_rows if ticker is not None
                }
                state_changed = state_etag != last_state_etag
                if state_changed:
                    latest_event_state = state_payload
                    last_state_etag = state_etag
                payload = {
                    "state": state_payload if state_changed else None,
                    "tickers": latest_event_tickers,
                }
                encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                for queue in tuple(event_clients):
                    if queue.full():
                        try:
                            queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    queue.put_nowait(encoded)
                await asyncio.sleep(max(0.0, EVENT_POLL_SECONDS - (time.monotonic() - started)))

    async def start_dashboard_events(app: web.Application) -> None:
        app["dashboard_event_task"] = asyncio.create_task(publish_dashboard_events(app))

    async def stop_dashboard_events(app: web.Application) -> None:
        task = app["dashboard_event_task"]
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    def supervised_processes(specs: list[tuple[str, str]]) -> list[dict[str, Any]]:
        pids: dict[str, int] = {}
        for proc_dir in Path("/proc").glob("[0-9]*"):
            if len(pids) == len(specs):
                break
            try:
                cmdline = (proc_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
            except OSError:
                continue
            for name, signature in specs:
                if name not in pids and signature in cmdline:
                    pids[name] = int(proc_dir.name)
        return [
            {"name": name, "status": "RUNNING", "pid": pids[name]}
            if name in pids
            else {"name": name, "status": "STOPPED", "pid": None}
            for name, _ in specs
        ]

    async def index(_: web.Request) -> web.Response:
        raise web.HTTPFound("/dashboard/live/")

    async def dashboard_index(_: web.Request) -> web.FileResponse:
        response = web.FileResponse(DASHBOARD_DIR / "index.html")
        response.enable_compression()
        return no_cache(response)

    async def api_state(request: web.Request) -> web.Response:
        payload, etag = dashboard_state_payload()
        return json_response(request, payload, etag)

    async def api_events(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(
            status=web.HTTPOk.status_code,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
        await response.prepare(request)
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
        event_clients.add(queue)
        initial_payload = {"state": latest_event_state, "tickers": latest_event_tickers}
        try:
            await response.write(f"data: {json.dumps(initial_payload, ensure_ascii=False, separators=(',', ':'))}\n\n".encode("utf-8"))
            while True:
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=20)
                    await response.write(f"data: {payload}\n\n".encode("utf-8"))
                except asyncio.TimeoutError:
                    await response.write(b": keepalive\n\n")
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            event_clients.discard(queue)
        return response

    async def api_market_history(request: web.Request) -> web.Response:
        asset = request.query.get("asset", "").lower()
        if asset not in MARKET_SYMBOLS:
            raise web.HTTPBadRequest(reason="unsupported_market_history_asset")
        candles = await load_market_history(asset)
        return web.json_response({"asset": asset, "candles": candles}, headers={"Cache-Control": "no-cache"})

    async def api_model_indicator_history(request: web.Request) -> web.Response:
        return web.json_response(
            {"samples": list(model_indicator_history), "sample_interval_seconds": MODEL_INDICATOR_SAMPLE_SECONDS},
            headers={"Cache-Control": "no-cache"},
        )

    async def api_evidence_signals(request: web.Request) -> web.Response:
        try:
            payload = await load_evidence_signals()
        except web.HTTPBadGateway:
            return web.json_response(
                {"error": "evidence_signal_upstream_error", "detail": "Binance klines fetch failed."},
                status=web.HTTPBadGateway.status_code,
                headers={"Cache-Control": "no-cache"},
            )
        return web.json_response(payload, headers={"Cache-Control": "no-cache"})

    async def api_oi_signal(request: web.Request) -> web.Response:
        payload = await load_oi_signal()
        return web.json_response(payload, headers={"Cache-Control": "no-cache"})

    async def api_liquidation_map(request: web.Request) -> web.Response:
        try:
            payload = await load_liquidation_map()
        except web.HTTPBadGateway:
            return web.json_response(
                {"error": "liquidation_map_upstream_error", "detail": "Binance klines fetch failed."},
                status=web.HTTPBadGateway.status_code,
                headers={"Cache-Control": "no-cache"},
            )
        return web.json_response(payload, headers={"Cache-Control": "no-cache"})

    async def api_trades(request: web.Request) -> web.Response:
        source_filter = request.query.get("source", "ALL").upper()
        journal_path = LIVE_DIR / "trade_journal.jsonl"
        signature = file_signature(journal_path)
        etag = make_etag("trades", signature, source_filter)
        if etag_matches(request, etag):
            return json_response(request, None, etag)

        rows = cached_trade_rows(journal_path, signature)
        payloads = trade_cache["payloads"]
        if source_filter not in payloads:
            payloads[source_filter] = {
                "rows": rows,
                "equity": equity_series(rows, source_filter),
            }
        return json_response(request, payloads[source_filter], etag)

    async def api_ops_status(request: web.Request) -> web.Response:
        ops_dir = LIVE_DIR / "ops_watchdog"
        health_path = ops_dir / "health_snapshot.json"
        heartbeat_path = ops_dir / "watchdog_heartbeat.json"
        state_path = ops_dir / "state.json"
        health_sig = file_signature(health_path)
        heartbeat_sig = file_signature(heartbeat_path)
        state_sig = file_signature(state_path)
        etag = make_etag("ops-status", health_sig, heartbeat_sig, state_sig)
        if etag_matches(request, etag):
            return json_response(request, None, etag)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "health": load_json_cached(health_path, health_sig) or {},
            "heartbeat": load_json_cached(heartbeat_path, heartbeat_sig) or {},
            # Match the managed process itself, not the supervisor that launched it --
            # these run under systemd now (previously bash _supervise.sh), and matching
            # the old bash wrapper's argv left this permanently reporting STOPPED after
            # the migration even though everything was healthy.
            "supervisors": supervised_processes([
                ("trading_bot", "trading_bot.py"),
                ("ops_watchdog", "ops_watchdog.py"),
            ]),
        }
        return json_response(request, payload, etag)

    async def api_scalp_shadow(request: web.Request) -> web.Response:
        asset = request.query.get("asset", "eth").lower()
        config = SCALP_SHADOW_ASSETS.get(asset)
        if config is None:
            return web.json_response(
                {"error": "unsupported_scalp_shadow_asset"},
                status=web.HTTPBadRequest.status_code,
                headers={"Cache-Control": "no-cache"},
            )
        state_path = LIVE_DIR / config["state_file"]
        database_path = LIVE_DIR / config["database_file"]
        etag = make_etag(
            "scalp-shadow",
            asset,
            file_signature(state_path),
            file_signature(database_path),
        )
        if etag_matches(request, etag):
            return json_response(request, None, etag)
        try:
            payload = scalp_shadow_payload(LIVE_DIR, asset)
        except Exception as exc:
            print(f"Scalp shadow dashboard contract error: {exc}", flush=True)
            return web.json_response(
                {
                    "error": "scalp_shadow_contract_error",
                    "detail": "Scalp shadow data contract is unavailable.",
                },
                status=web.HTTPServiceUnavailable.status_code,
                headers={"Cache-Control": "no-cache"},
            )
        return json_response(request, payload, etag)

    async def api_scalp_reuse_shadow(request: web.Request) -> web.Response:
        mode = request.query.get("mode", "eth_lifecycle").lower()
        config = SCALP_REUSE_MODES.get(mode)
        if config is None:
            return web.json_response(
                {"error": "unsupported_scalp_reuse_mode"},
                status=web.HTTPBadRequest.status_code,
                headers={"Cache-Control": "no-cache"},
            )
        state_path = LIVE_DIR / config["state_file"]
        database_path = LIVE_DIR / config["database_file"]
        etag = make_etag(
            "scalp-reuse-shadow",
            mode,
            file_signature(state_path),
            file_signature(database_path),
        )
        if etag_matches(request, etag):
            return json_response(request, None, etag)
        try:
            payload = scalp_shadow_payload(LIVE_DIR, mode, SCALP_REUSE_MODES)
        except Exception as exc:
            print(f"Scalp reuse shadow dashboard contract error: {exc}", flush=True)
            return web.json_response(
                {
                    "error": "scalp_reuse_shadow_contract_error",
                    "detail": "Scalp reuse shadow data contract is unavailable.",
                },
                status=web.HTTPServiceUnavailable.status_code,
                headers={"Cache-Control": "no-cache"},
            )
        return json_response(request, payload, etag)

    async def api_btc_multislot_shadow(request: web.Request) -> web.Response:
        etag = make_etag(
            "btc-multislot-shadow",
            file_signature(BTC_MULTISLOT_SHADOW_STATE_PATH),
            file_signature(BTC_MULTISLOT_SHADOW_LEDGER_PATH),
        )
        if etag_matches(request, etag):
            return json_response(request, None, etag)
        return json_response(request, btc_multislot_shadow_payload(), etag)

    async def api_eth_odyssey4_shadow(request: web.Request) -> web.Response:
        etag = make_etag(
            "eth-odyssey4-shadow",
            file_signature(ETH_ODYSSEY4_SHADOW_STATE_PATH),
            file_signature(ETH_ODYSSEY4_SHADOW_TRADES_PATH),
        )
        if etag_matches(request, etag):
            return json_response(request, None, etag)
        return json_response(request, eth_odyssey4_shadow_payload(), etag)

    app.router.add_get("/", index)
    app.router.add_get("/dashboard/live", dashboard_index)
    app.router.add_get("/dashboard/live/", dashboard_index)
    app.router.add_get("/api/state", api_state)
    app.router.add_get("/api/events", api_events)
    app.router.add_get("/api/market-history", api_market_history)
    app.router.add_get("/api/evidence-signals", api_evidence_signals)
    app.router.add_get("/api/oi-signal", api_oi_signal)
    app.router.add_get("/api/liquidation-map", api_liquidation_map)
    app.router.add_get("/api/model-indicator-history", api_model_indicator_history)
    app.router.add_get("/api/trades", api_trades)
    app.router.add_get("/api/ops-status", api_ops_status)
    app.router.add_get("/api/scalp-shadow", api_scalp_shadow)
    app.router.add_get("/api/scalp-reuse-shadow", api_scalp_reuse_shadow)
    app.router.add_get("/api/btc-multislot-shadow", api_btc_multislot_shadow)
    app.router.add_get("/api/eth-odyssey4-shadow", api_eth_odyssey4_shadow)
    app.router.add_static("/dashboard/live/", DASHBOARD_DIR, show_index=True)
    app.router.add_static("/data/live/", LIVE_DIR, show_index=False)
    app.on_startup.append(start_dashboard_events)
    app.on_cleanup.append(stop_dashboard_events)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic live dashboard server.")
    parser.add_argument("--host", default=os.getenv("DASHBOARD_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("DASHBOARD_PORT", "8787")))
    args = parser.parse_args()

    print(f"Serving dashboard at http://{args.host}:{args.port}/dashboard/live/", flush=True)
    web.run_app(
        make_app(),
        host=args.host,
        port=args.port,
        print=None,
    )


if __name__ == "__main__":
    main()
