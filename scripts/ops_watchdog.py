#!/usr/bin/env python3
"""24/7 live operations watchdog. It observes only; it never submits orders or repairs data."""
from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import sqlite3
import subprocess
import time
import urllib.request
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
# trading_bot.py loads .env the same way; this script never did, so
# TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID were unset in its process environment and
# every alert has been silently swallowed (telegram_sent=false on every event)
# regardless of severity. Explicit path so it doesn't depend on invocation cwd.
load_dotenv(ROOT / ".env")
LIVE = ROOT / "data" / "live"
OUT = LIVE / "ops_watchdog"
HISTORY = OUT / "history"
KST = ZoneInfo("Asia/Seoul")
SEVERITY = {"OK": 0, "WARN": 1, "CRITICAL": 2, "BLOCKED": 3}


@dataclass
class Check:
    component: str
    status: str
    summary: str
    details: dict[str, Any]


def now_kst() -> datetime:
    return datetime.now(KST)


def iso_now() -> str:
    return now_kst().isoformat()


def parse_kst(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.replace(tzinfo=KST) if parsed.tzinfo is None else parsed.astimezone(KST)


def age_minutes(value: Any) -> float | None:
    stamp = parse_kst(value)
    return None if stamp is None else max(0.0, (now_kst() - stamp).total_seconds() / 60.0)


def age_minutes_utc_naive(value: Any) -> float | None:
    stamp = parse_utc_naive(value)
    return None if stamp is None else max(0.0, (now_kst() - stamp).total_seconds() / 60.0)


def parse_utc_naive(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        stamp = datetime.fromisoformat(text)
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(KST)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def retain_history(days: int = 30) -> None:
    cutoff = time.time() - days * 86400
    for path in HISTORY.glob("*.jsonl"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            pass


def recommended_action(component: str) -> str:
    # 2026-08-03: migrated to systemd (scripts/ops/systemd/*.service) -- journalctl
    # replaces the old logs/supervisor/*.log tail, which no longer receives output.
    if component in {"trading_bot_process", "trading_bot_heartbeat", "decision_snapshot", "data_pipeline", "pipeline_contract"}:
        return "scripts/ops/botctl.sh status; journalctl -u trading-bot.service -n 100 --no-pager"
    if component in {"market_data_sources", "dashboard_state", "execution_contract"}:
        return "scripts/ops/triage.sh"
    if component in {"runtime_resources", "watchdog_storage"}:
        return "scripts/ops/triage.sh; df -h ."
    if component == "btc_multislot_shadow_process":
        return "scripts/ops/botctl.sh status; tail -n 50 logs/supervisor/btc_multislot_shadow_$(date +%Y%m%d).log"
    return "scripts/ops/triage.sh"


def load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "missing"
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"invalid_json:{type(exc).__name__}"
    return (data, None) if isinstance(data, dict) else (None, "json_not_object")


def tail_jsonl(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with path.open("rb") as fh:
            fh.seek(max(0, path.stat().st_size - 65536))
            lines = fh.read().decode("utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return None, "missing"
    except OSError as exc:
        return None, f"read_error:{type(exc).__name__}"
    for line in reversed(lines):
        if line.strip():
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                return None, "invalid_tail_jsonl"
            return (data, None) if isinstance(data, dict) else (None, "tail_not_object")
    return None, "empty"


def stale_status(age: float | None, warn: float, critical: float) -> str:
    if age is None:
        return "BLOCKED"
    if age >= critical:
        return "CRITICAL"
    if age >= warn:
        return "WARN"
    return "OK"


def process_args() -> str:
    try:
        return subprocess.check_output(["ps", "-eo", "args="], text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return ""


def check_process(component: str, signature: str, required: bool = True) -> Check:
    found = signature in process_args()
    if found:
        return Check(component, "OK", "registered process is present", {"signature": signature})
    status = "CRITICAL" if required else "WARN"
    return Check(component, status, "registered process is absent", {"signature": signature})


def check_snapshot() -> Check:
    path = LIVE / "decision_feature_snapshot.jsonl"
    row, error = tail_jsonl(path)
    if error:
        return Check("decision_snapshot", "BLOCKED", "decision snapshot cannot be read", {"path": str(path), "error": error})
    values = row.get("values") if isinstance(row, dict) else None
    market_ts = values.get("timestamp") if isinstance(values, dict) else None
    # The market bar timestamp belongs to the completed bar and is therefore
    # expected to trail the write that records its decision.  Measure the
    # artifact's creation time for liveness; retain market_ts as diagnostics.
    created_at = row.get("created_at") if isinstance(row, dict) else None
    age = age_minutes(created_at)
    return Check("decision_snapshot", stale_status(age, 12, 18), "market snapshot freshness", {
        "path": str(path), "market_ts": market_ts, "created_at": created_at,
        "age_minutes": age, "warn_minutes": 12, "critical_minutes": 18,
    })


def check_heartbeat() -> Check:
    path = LIVE / "trading_bot_decision_heartbeat.json"
    state, error = load_json(path)
    if error:
        return Check("trading_bot_heartbeat", "BLOCKED", "decision heartbeat cannot be read", {"path": str(path), "error": error})
    recorded = state.get("recorded_at_kst")
    age = age_minutes(recorded)
    return Check("trading_bot_heartbeat", stale_status(age, 6, 10), "decision heartbeat freshness", {
        "recorded_at_kst": recorded, "decision_bar_ts": state.get("decision_bar_ts"), "age_minutes": age,
        "warn_minutes": 6, "critical_minutes": 10,
    })


def check_pipeline() -> Check:
    path = LIVE / "data_pipeline_health.json"
    state, error = load_json(path)
    if error:
        return Check("data_pipeline", "BLOCKED", "pipeline health cannot be read", {"path": str(path), "error": error})
    raw_eth = state.get("raw_eth") if isinstance(state.get("raw_eth"), dict) else {}
    last_ts = raw_eth.get("last_ts")
    age = age_minutes(last_ts)
    freshness = stale_status(age, 12, 18)
    reported = str(state.get("status", "OK")).upper()
    status = freshness if SEVERITY[freshness] >= SEVERITY.get(reported, 1) else reported
    if status not in SEVERITY:
        status = "WARN"
    return Check("data_pipeline", status, "pipeline report and raw ETH freshness", {
        "reported_status": reported, "raw_eth_last_ts": last_ts, "age_minutes": age,
        "warnings": state.get("warnings", []), "warn_minutes": 12, "critical_minutes": 18,
    })


def check_dashboard() -> Check:
    path = LIVE / "dashboard_state.json"
    state, error = load_json(path)
    if error:
        return Check("dashboard_state", "BLOCKED", "dashboard state cannot be read", {"path": str(path), "error": error})
    # `cycle_timestamp_kst` changes only with a 5-minute decision cycle. The
    # dashboard shadow loop updates independently every 10 seconds, so using
    # the decision timestamp here creates false dashboard outage alerts.
    stamp = state.get("shadow_updated_at") or state.get("updated_at") or state.get("cycle_timestamp_kst")
    age = age_minutes(stamp)
    return Check("dashboard_state", stale_status(age, 2, 5), "dashboard shadow refresh freshness", {
        "timestamp": stamp, "age_minutes": age, "warn_minutes": 2, "critical_minutes": 5,
    })


def check_pipeline_contract() -> Check:
    path = LIVE / "data_pipeline_health.json"
    state, error = load_json(path)
    if error:
        return Check("pipeline_contract", "BLOCKED", "pipeline contract cannot be read", {"path": str(path), "error": error})
    ai = state.get("ai") if isinstance(state.get("ai"), dict) else {}
    groups = ai.get("groups") if isinstance(ai.get("groups"), list) else []
    errors = ai.get("errors") if isinstance(ai.get("errors"), list) else []
    missing = ai.get("missing_cols") if isinstance(ai.get("missing_cols"), list) else []
    nonfinite = ai.get("nonfinite_cols") if isinstance(ai.get("nonfinite_cols"), list) else []
    valid = (
        state.get("pipeline_stage") == "final_governor_success"
        and groups == ["tide", "dlinear", "patchtst"]
        and not errors and not missing and not nonfinite
        and state.get("signal_align_ok") is True
    )
    return Check(
        "pipeline_contract",
        "OK" if valid else "BLOCKED",
        "pipeline AI and bar contract" if valid else "pipeline AI or bar contract mismatch",
        {
            "pipeline_stage": state.get("pipeline_stage"), "ai_groups": groups,
            "ai_errors": errors, "missing_cols": missing, "nonfinite_cols": nonfinite,
            "signal_align_ok": state.get("signal_align_ok"),
        },
    )


def check_data_sources() -> Check:
    path = LIVE / "dashboard_state.json"
    state, error = load_json(path)
    if error:
        return Check("market_data_sources", "BLOCKED", "market data state cannot be read", {"path": str(path), "error": error})
    micro = state.get("microstructure") if isinstance(state.get("microstructure"), dict) else {}
    tail = state.get("tail_risk") if isinstance(state.get("tail_risk"), dict) else {}
    sources = {
        "depth_websocket": micro.get("depth_connected"),
        "trade_websocket": micro.get("trade_connected"),
        "rest_poll": micro.get("poll_connected"),
        "liquidation_websocket": tail.get("ws_connected"),
    }
    failed = [name for name, connected in sources.items() if connected is not True]
    stale = bool(micro.get("data_stale"))
    try:
        trade_age = max(0.0, float(micro.get("trade_age_sec")))
    except (TypeError, ValueError):
        trade_age = None
    # Depth and REST polling are the core price/contract feeds. A lone stale
    # trade-flow stream is significant, but short exchange stream gaps must not
    # page as CRITICAL. WARN debounce absorbs brief gaps; sustained loss escalates.
    core_failed = any(sources[name] is not True for name in ("depth_websocket", "rest_poll"))
    if core_failed or len(failed) >= 2:
        status = "CRITICAL"
    elif stale:
        status = "CRITICAL" if trade_age is None or trade_age >= 600.0 else "WARN"
    elif failed:
        status = "WARN"
    else:
        status = "OK"
    return Check("market_data_sources", status, "market source connectivity", {
        "sources": sources, "data_stale": stale, "failed_sources": failed,
        "trade_age_sec": trade_age, "trade_critical_after_sec": 600,
    })


def check_runtime_resources() -> Check:
    usage = shutil.disk_usage(ROOT)
    free_gib = usage.free / (1024 ** 3)
    try:
        meminfo = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line)
        available_kib = int(meminfo.get("MemAvailable", "0 kB").split()[0])
        memory_available_gib = available_kib / (1024 ** 2)
    except (OSError, ValueError):
        memory_available_gib = None
    if free_gib < 10 or (memory_available_gib is not None and memory_available_gib < 2):
        status = "CRITICAL"
    elif free_gib < 20 or (memory_available_gib is not None and memory_available_gib < 4):
        status = "WARN"
    else:
        status = "OK"
    return Check("runtime_resources", status, "disk and memory headroom", {
        "disk_free_gib": round(free_gib, 2), "memory_available_gib": None if memory_available_gib is None else round(memory_available_gib, 2),
        "disk_warn_gib": 20, "disk_critical_gib": 10, "memory_warn_gib": 4, "memory_critical_gib": 2,
    })


def check_watchdog_storage() -> Check:
    required = [OUT / "state.json", OUT / "incidents.sqlite", OUT / "watchdog_heartbeat.json"]
    missing = [str(path) for path in required if not path.is_file()]
    writable = os.access(OUT, os.W_OK)
    status = "OK" if writable and not missing else "BLOCKED"
    return Check("watchdog_storage", status, "watchdog state and incident storage", {
        "directory": str(OUT), "writable": writable, "missing_files": missing,
    })


def check_execution_contract() -> Check:
    path = LIVE / "dashboard_state.json"
    state, error = load_json(path)
    if error:
        return Check("execution_contract", "BLOCKED", "execution contract cannot be read", {"path": str(path), "error": error})
    account = state.get("account") if isinstance(state.get("account"), dict) else {}
    alert = state.get("execution_alert") if isinstance(state.get("execution_alert"), dict) else {}
    valid = account.get("enabled") is False and account.get("testnet") is True and alert.get("status") == "disabled"
    return Check("execution_contract", "OK" if valid else "BLOCKED", "shadow execution safety contract", {
        "account_enabled": account.get("enabled"), "testnet": account.get("testnet"), "execution_alert_status": alert.get("status"),
    })


def init_db(path: Path) -> None:
    with sqlite3.connect(path) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY, observed_at_kst TEXT NOT NULL, component TEXT NOT NULL,
            status TEXT NOT NULL, summary TEXT NOT NULL, details_json TEXT NOT NULL,
            notification_kind TEXT NOT NULL, telegram_sent INTEGER NOT NULL)""")


def load_state(path: Path) -> dict[str, Any]:
    state, _ = load_json(path)
    return state or {"schema_version": "ops_watchdog.state.v1", "checks": {}}


def debounce_seconds(status: str) -> float:
    # CRITICAL/BLOCKED must page immediately -- never sit in a pending window.
    # Everything else (entering OR leaving WARN) must hold for a sustained period
    # before it's treated as real, otherwise a metric that oscillates around its
    # own threshold every polling cycle pages an alert+recovered pair every cycle.
    return 0.0 if status in {"CRITICAL", "BLOCKED"} else 120.0


def apply_debounce(previous: dict[str, Any], raw_status: str) -> tuple[str, dict[str, Any]]:
    confirmed = previous.get("status") or raw_status
    if raw_status == confirmed:
        return confirmed, {"pending_status": None, "pending_since_kst": None}
    pending_status = previous.get("pending_status")
    pending_since = parse_kst(previous.get("pending_since_kst"))
    if raw_status != pending_status or pending_since is None:
        pending_status, pending_since = raw_status, now_kst()
    elapsed = (now_kst() - pending_since).total_seconds()
    if elapsed >= debounce_seconds(raw_status):
        return raw_status, {"pending_status": None, "pending_since_kst": None}
    return confirmed, {"pending_status": pending_status, "pending_since_kst": pending_since.isoformat()}


def notification_kind(previous: str, current: str, last_notified: str | None) -> str | None:
    if current == "OK":
        return "recovered" if previous and previous != "OK" else None
    if current != previous:
        return "alert"
    last = parse_kst(last_notified)
    if last is None:
        return "alert"
    repeat_minutes = 30 if current in {"CRITICAL", "BLOCKED"} else 120
    return "reminder" if (now_kst() - last).total_seconds() >= repeat_minutes * 60 else None


def telegram_message(check: Check, kind: str) -> str:
    icon = {"OK": "🟢", "WARN": "🟠", "CRITICAL": "🔴", "BLOCKED": "⛔"}[check.status]
    title = "RECOVERED" if kind == "recovered" else check.status
    fields = check.details
    lines = [f"{icon} <b>[{title}] {html.escape(check.component)}</b>", html.escape(check.summary), f"감지: {iso_now()}"]
    for key in ("market_ts", "recorded_at_kst", "raw_eth_last_ts", "timestamp", "last_processed_bar_ts", "age_minutes", "error", "equity_curve_error"):
        value = fields.get(key)
        if value not in (None, "", []):
            label = "지연(분)" if key == "age_minutes" else key
            lines.append(f"{label}: <code>{html.escape(str(round(value, 1) if isinstance(value, float) else value))}</code>")
    return "\n".join(lines)


def send_telegram(message: str) -> bool:
    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN", ""), os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return False
    body = json.dumps({"chat_id": chat_id, "text": message, "parse_mode": "HTML"}).encode()
    request = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage", data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            response.read()
        return True
    except OSError:
        return False


def ping_deadman_switch() -> None:
    # Optional complementary signal to scripts/ops/watchdog_deadman.sh: an external
    # dead-man's-switch service (e.g. https://healthchecks.io) that pages independently
    # of this process/host if the ping stops arriving. No-op unless the user has set up
    # their own account and put the ping URL in .env -- we never create that account.
    url = os.getenv("HEALTHCHECK_PING_URL", "")
    if not url:
        return
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="GET"), timeout=5) as response:
            response.read()
    except OSError:
        pass


def run_once(dry_run: bool) -> list[Check]:
    OUT.mkdir(parents=True, exist_ok=True)
    state_path, db_path = OUT / "state.json", OUT / "incidents.sqlite"
    init_db(db_path)
    checks = [
        check_process("trading_bot_process", "trading_bot.py"),
        # Shadow-only live-forward A/B loop (2026-08-07) with no order submission --
        # not required for live trading, but if it dies silently the multi-slot
        # promotion gate quietly stops accumulating observations. WARN, not CRITICAL.
        check_process("btc_multislot_shadow_process", "run_btc_multislot_shadow_loop_20260807.py", required=False),
        check_snapshot(), check_heartbeat(), check_pipeline(), check_pipeline_contract(),
        check_data_sources(), check_dashboard(), check_execution_contract(), check_runtime_resources(),
        check_watchdog_storage(),
    ]
    state = load_state(state_path)
    stored = state.setdefault("checks", {})
    effective_checks: list[Check] = []
    with sqlite3.connect(db_path) as con:
        for check in checks:
            previous = stored.get(check.component, {})
            confirmed_status, debounce_fields = apply_debounce(previous, check.status)
            effective = check if confirmed_status == check.status else replace(
                check, status=confirmed_status, details={**check.details, "raw_status": check.status},
            )
            effective_checks.append(effective)
            kind = notification_kind(str(previous.get("status", "")), confirmed_status, previous.get("last_notified_at_kst"))
            sent = False
            if kind:
                message = telegram_message(effective, kind)
                sent = False if dry_run else send_telegram(message)
                print(f"[{kind}] {effective.component} {effective.status}: {effective.summary}")
                con.execute("INSERT INTO incidents(observed_at_kst, component, status, summary, details_json, notification_kind, telegram_sent) VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (iso_now(), effective.component, effective.status, effective.summary, json.dumps(effective.details, default=str), kind, int(sent)))
                append_jsonl(HISTORY / f"events_{now_kst():%Y%m%d}.jsonl", {
                    "observed_at_kst": iso_now(), "kind": kind, "component": effective.component,
                    "status": effective.status, "summary": effective.summary, "telegram_sent": sent,
                    "recommended_action": recommended_action(effective.component), "details": effective.details,
                })
            stored[check.component] = {
                "status": confirmed_status, "raw_status": check.status, "summary": check.summary, "last_seen_at_kst": iso_now(),
                **debounce_fields,
                # Record every attempted notification as the dedupe point. Otherwise a
                # missing Telegram configuration (or a transient send failure) retries
                # the same alert every polling interval instead of the documented cadence.
                "last_notified_at_kst": iso_now() if kind else previous.get("last_notified_at_kst"),
            }
    observed_at = iso_now()
    snapshot = {"schema_version": "ops_watchdog.health.v1", "updated_at_kst": observed_at, "checks": [asdict(c) for c in effective_checks]}
    append_jsonl(HISTORY / f"watchdog_{now_kst():%Y%m%d}.jsonl", snapshot)
    retain_history()
    atomic_json(state_path, state)
    atomic_json(OUT / "health_snapshot.json", snapshot)
    atomic_json(OUT / "watchdog_heartbeat.json", {"recorded_at_kst": iso_now(), "status": "ok", "check_count": len(checks)})
    ping_deadman_switch()
    return effective_checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Never send Telegram messages")
    args = parser.parse_args()
    while True:
        run_once(args.dry_run)
        if args.once:
            return
        time.sleep(max(5.0, args.interval_seconds))


if __name__ == "__main__":
    main()
