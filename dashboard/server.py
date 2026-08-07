from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
from aiohttp import ClientSession, ClientTimeout, web


REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE_DIR = REPO_ROOT / "data" / "live"
DASHBOARD_DIR = REPO_ROOT / "dashboard" / "live"
# Shadow-only, no order submission -- standalone loop separate from trading_bot.py's
# single-slot BTC shadow (see scripts/run_btc_multislot_shadow_loop_20260807.py).
BTC_MULTISLOT_SHADOW_STATE_PATH = REPO_ROOT / "data" / "ensemble" / "omega4_6_1_btc_multislot_shadow_state_20260807.json"
BTC_MULTISLOT_SHADOW_LEDGER_PATH = REPO_ROOT / "data" / "ensemble" / "omega4_6_1_btc_multislot_shadow_ledger_20260807.csv"
BTC_MULTISLOT_SHADOW_BAR_SECONDS = 300
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


def utc_age_minutes(value: Any) -> float | None:
    encoded = utc_iso(value)
    if encoded is None:
        return None
    try:
        timestamp = datetime.fromisoformat(encoded.replace("Z", "+00:00"))
    except ValueError:
        return None
    return max(0.0, (datetime.now(timezone.utc) - timestamp).total_seconds() / 60.0)


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
    age_minutes = utc_age_minutes(last_bar)
    total_trades = 0
    cumulative_return_pct: float | None = None
    recent_trades: list[dict[str, Any]] = []
    if BTC_MULTISLOT_SHADOW_LEDGER_PATH.exists():
        with BTC_MULTISLOT_SHADOW_LEDGER_PATH.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        total_trades = len(rows)
        equity = 1.0
        for row in rows:
            try:
                equity *= 1.0 + float(row.get("trade_return_net") or 0.0)
            except (TypeError, ValueError):
                continue
        cumulative_return_pct = (equity - 1.0) * 100.0
        for row in rows[-5:]:
            try:
                return_pct = float(row["trade_return_net"]) * 100.0
            except (KeyError, TypeError, ValueError):
                return_pct = None
            recent_trades.append(
                {
                    "slot": row.get("slot"),
                    "side": row.get("side"),
                    "exit_timestamp": row.get("exit_timestamp"),
                    "trade_return_pct": return_pct,
                    "reason": row.get("reason"),
                }
            )
        recent_trades.reverse()
    return {
        "last_bar": last_bar,
        "age_minutes": age_minutes,
        "stale": age_minutes is None or age_minutes >= (BTC_MULTISLOT_SHADOW_BAR_SECONDS / 60.0) * 3,
        "slot_count": len(slots),
        "open_slots": sum(1 for s in slots if s),
        "slots": slots,
        "total_trades": total_trades,
        "cumulative_return_pct": cumulative_return_pct,
        "recent_trades": recent_trades,
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

    async def publish_dashboard_events(app: web.Application) -> None:
        nonlocal latest_event_state, latest_event_tickers
        last_state_etag = ""
        timeout = ClientTimeout(total=2)
        async with ClientSession(timeout=timeout) as session:
            while True:
                started = time.monotonic()
                state_payload, state_etag = dashboard_state_payload()
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

    app.router.add_get("/", index)
    app.router.add_get("/dashboard/live", dashboard_index)
    app.router.add_get("/dashboard/live/", dashboard_index)
    app.router.add_get("/api/state", api_state)
    app.router.add_get("/api/events", api_events)
    app.router.add_get("/api/market-history", api_market_history)
    app.router.add_get("/api/trades", api_trades)
    app.router.add_get("/api/ops-status", api_ops_status)
    app.router.add_get("/api/scalp-shadow", api_scalp_shadow)
    app.router.add_get("/api/scalp-reuse-shadow", api_scalp_reuse_shadow)
    app.router.add_get("/api/btc-multislot-shadow", api_btc_multislot_shadow)
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
