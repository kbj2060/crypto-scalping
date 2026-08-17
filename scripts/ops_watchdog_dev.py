#!/usr/bin/env python3
"""24/7 dev-machine data-collector watchdog. Read-only: it only reads DuckDB tables and
process lists, and writes its own state/incident files -- it never touches a collector or
trading process.

Counterpart to scripts/ops_watchdog.py (server-side, trading_bot/dashboard focus). This one
watches the dev-only research collectors registered in this session's data-collector audit:
duckdb_persist_worker (tail_risk + microstructure, ETH/BTC/SOL), the hourly Deribit GEX cron,
and the daily F4-C altdata cron. Process-liveness alone would have missed every incident this
session found (a worker can run and still silently stop writing), so every check here reads
the DuckDB tables directly.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sqlite3
import subprocess
import time
import urllib.request
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
LIVE = ROOT / "data" / "live"
RESEARCH = ROOT / "data" / "research"
OUT = LIVE / "ops_watchdog_dev"
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


def stale_status(age: float | None, warn: float, critical: float) -> str:
    if age is None:
        return "BLOCKED"
    if age >= critical:
        return "CRITICAL"
    if age >= warn:
        return "WARN"
    return "OK"


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


def process_args() -> str:
    try:
        return subprocess.check_output(["ps", "-eo", "args="], text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return ""


def check_process(component: str, signature: str) -> Check:
    found = signature in process_args()
    status = "OK" if found else "CRITICAL"
    summary = "registered process is present" if found else "registered process is absent"
    return Check(component, status, summary, {"signature": signature})


def check_duckdb_table_freshness(component: str, db_path: Path, table: str, ts_column: str,
                                  warn_minutes: float, critical_minutes: float) -> Check:
    """Read-only DuckDB freshness check. Retries a lock-conflict IOException a few times --
    DuckDB briefly refuses a new read-only connection while duckdb_persist_worker.py's own
    writer connection is mid-insert (see scripts/ops_watchdog.py for the server-side incident
    this mirrors) -- before treating it as a real failure."""
    if not db_path.is_file():
        return Check(component, "BLOCKED", "duckdb file is missing", {"path": str(db_path)})
    last_error: duckdb.Error | None = None
    for attempt, delay in enumerate((0.0, 0.4, 0.8, 1.6)):
        if delay:
            time.sleep(delay)
        try:
            con = duckdb.connect(str(db_path), read_only=True)
            try:
                max_ts = con.execute(f"select max(cast({ts_column} as timestamp)) from {table}").fetchone()[0]
            finally:
                con.close()
            last_error = None
            break
        except duckdb.Error as exc:
            last_error = exc
    if last_error is not None:
        return Check(component, "BLOCKED", "duckdb table cannot be read", {
            "path": str(db_path), "table": table, "error": f"{type(last_error).__name__}: {last_error}",
            "attempts": attempt + 1,
        })
    if max_ts is None:
        return Check(component, "BLOCKED", "duckdb table has no rows", {"path": str(db_path), "table": table})
    # every timestamp column checked here is KST wall time whether or not duckdb attaches
    # tzinfo (VARCHAR-cast columns come back naive but are KST strings at the source).
    age = age_minutes(max_ts)
    return Check(component, stale_status(age, warn_minutes, critical_minutes), "duckdb table freshness", {
        "path": str(db_path), "table": table, "latest_ts": str(max_ts), "age_minutes": age,
        "warn_minutes": warn_minutes, "critical_minutes": critical_minutes,
    })


def init_db(path: Path) -> None:
    with sqlite3.connect(path) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY, observed_at_kst TEXT NOT NULL, component TEXT NOT NULL,
            status TEXT NOT NULL, summary TEXT NOT NULL, details_json TEXT NOT NULL,
            notification_kind TEXT NOT NULL, telegram_sent INTEGER NOT NULL)""")


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {"schema_version": "ops_watchdog_dev.state.v1", "checks": {}}


def debounce_seconds(status: str) -> float:
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
    lines = [f"{icon} <b>[DEV][{title}] {html.escape(check.component)}</b>", html.escape(check.summary), f"감지: {iso_now()}"]
    for key in ("latest_ts", "age_minutes", "table", "error", "signature"):
        value = check.details.get(key)
        if value not in (None, "", []):
            label = "지연(분)" if key == "age_minutes" else key
            lines.append(f"{label}: <code>{html.escape(str(round(value, 1) if isinstance(value, float) else value))}</code>")
    return "\n".join(lines)


def send_telegram(message: str) -> bool | None:
    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN", ""), os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return None
    body = json.dumps({"chat_id": chat_id, "text": message, "parse_mode": "HTML"}).encode()
    request = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage", data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            response.read()
        return True
    except OSError:
        return False


def build_checks() -> list[Check]:
    micro_db = LIVE / "microstructure.duckdb"
    tail_db = LIVE / "tail_risk.duckdb"
    gex_db = LIVE / "deribit_gex.duckdb"
    altdata_db = RESEARCH / "altdata.duckdb"
    return [
        check_process("duckdb_persist_worker_process", "duckdb_persist_worker.py"),
        check_duckdb_table_freshness("duckdb_microstructure_1m_eth", micro_db, "microstructure_1m", "ts", 5, 10),
        check_duckdb_table_freshness("duckdb_microstructure_1m_btc", micro_db, "microstructure_1m_btc", "ts", 5, 10),
        check_duckdb_table_freshness("duckdb_microstructure_1m_sol", micro_db, "microstructure_1m_sol", "ts", 5, 10),
        check_duckdb_table_freshness("duckdb_tail_risk_1m_eth", tail_db, "tail_risk_1m", "ts", 5, 10),
        check_duckdb_table_freshness("duckdb_tail_risk_1m_btc", tail_db, "tail_risk_1m_btc", "ts", 5, 10),
        check_duckdb_table_freshness("duckdb_tail_risk_1m_sol", tail_db, "tail_risk_1m_sol", "ts", 5, 10),
        # hourly cron; one missed run is normal, two in a row is not.
        check_duckdb_table_freshness("duckdb_deribit_gex", gex_db, "gex_summary", "recorded_at_utc", 90, 150),
        # daily cron (0 1 * * *); warn/critical give ~1 and ~2 missed days of slack.
        check_duckdb_table_freshness("duckdb_altdata_fear_greed", altdata_db, "fear_greed_index", "recorded_at_utc", 1800, 2880),
        check_duckdb_table_freshness("duckdb_altdata_funding_spread", altdata_db, "cross_exchange_funding_spread", "recorded_at_utc", 1800, 2880),
    ]


def run_once(dry_run: bool) -> list[Check]:
    OUT.mkdir(parents=True, exist_ok=True)
    state_path, db_path = OUT / "state.json", OUT / "incidents.sqlite"
    init_db(db_path)
    checks = build_checks()
    state = load_json(state_path)
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
            sent = None
            if kind:
                message = telegram_message(effective, kind)
                sent = None if dry_run else send_telegram(message)
                print(f"[{kind}] {effective.component} {effective.status}: {effective.summary}")
                con.execute("INSERT INTO incidents(observed_at_kst, component, status, summary, details_json, notification_kind, telegram_sent) VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (iso_now(), effective.component, effective.status, effective.summary, json.dumps(effective.details, default=str), kind, int(bool(sent))))
                append_jsonl(HISTORY / f"events_{now_kst():%Y%m%d}.jsonl", {
                    "observed_at_kst": iso_now(), "kind": kind, "component": effective.component,
                    "status": effective.status, "summary": effective.summary, "telegram_sent": bool(sent),
                    "details": effective.details,
                })
            stored[check.component] = {
                "status": confirmed_status, "raw_status": check.status, "summary": check.summary, "last_seen_at_kst": iso_now(),
                **debounce_fields,
                "last_notified_at_kst": previous.get("last_notified_at_kst") if sent is False else iso_now(),
            }
    observed_at = iso_now()
    snapshot = {"schema_version": "ops_watchdog_dev.health.v1", "updated_at_kst": observed_at, "checks": [asdict(c) for c in effective_checks]}
    append_jsonl(HISTORY / f"watchdog_{now_kst():%Y%m%d}.jsonl", snapshot)
    retain_history()
    atomic_json(state_path, state)
    atomic_json(OUT / "health_snapshot.json", snapshot)
    atomic_json(OUT / "watchdog_heartbeat.json", {"recorded_at_kst": iso_now(), "status": "ok", "check_count": len(checks)})
    return effective_checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Never send Telegram messages")
    args = parser.parse_args()
    while True:
        run_once(args.dry_run)
        if args.once:
            return
        time.sleep(max(15.0, args.interval_seconds))


if __name__ == "__main__":
    main()
