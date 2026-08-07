"""Read-only Prometheus exporter for the crypto-scalping ops watchdog.

Translates data/live/ops_watchdog/health_snapshot.json, the watchdog
heartbeat, and systemd unit state into Prometheus text-exposition format on
GET /metrics. No prometheus_client dependency -- the format is simple enough
to hand-write, and this keeps the project's dependency surface unchanged.

Never touches trading_bot.py's decision path; purely reads existing JSON
state files and shells out to `systemctl is-active` (read-only).

Run: python scripts/ops/prometheus_exporter.py --port 9101
"""
from __future__ import annotations

import argparse
import json
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "data" / "live"
OUT = LIVE / "ops_watchdog"

STATUS_VALUE = {"OK": 1.0, "WARN": 0.5, "CRITICAL": 0.0, "BLOCKED": 0.0}
SYSTEMD_UNITS = ["trading-bot", "tau1-shadow", "ops-watchdog", "btc-multislot-shadow"]


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def systemd_active(unit: str) -> float:
    try:
        out = subprocess.run(
            ["systemctl", "is-active", f"{unit}.service"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return 0.0
    return 1.0 if out == "active" else 0.0


def render_metrics() -> str:
    lines: list[str] = []

    lines.append("# HELP ops_watchdog_check_status 1=OK 0.5=WARN 0=CRITICAL/BLOCKED")
    lines.append("# TYPE ops_watchdog_check_status gauge")
    lines.append("# HELP ops_watchdog_check_age_minutes age_minutes reported by the check, when present")
    lines.append("# TYPE ops_watchdog_check_age_minutes gauge")
    snapshot = load_json(OUT / "health_snapshot.json")
    if isinstance(snapshot, dict):
        for check in snapshot.get("checks", []):
            component = str(check.get("component", "unknown"))
            status = str(check.get("status", "")).upper()
            value = STATUS_VALUE.get(status, 0.0)
            lines.append(f'ops_watchdog_check_status{{component="{component}"}} {value}')
            age = (check.get("details") or {}).get("age_minutes")
            if isinstance(age, (int, float)):
                lines.append(f'ops_watchdog_check_age_minutes{{component="{component}"}} {age}')

    lines.append("# HELP ops_watchdog_heartbeat_age_seconds seconds since the watchdog's own last poll")
    lines.append("# TYPE ops_watchdog_heartbeat_age_seconds gauge")
    heartbeat = load_json(OUT / "watchdog_heartbeat.json")
    if isinstance(heartbeat, dict):
        from datetime import datetime, timezone
        recorded = heartbeat.get("recorded_at_kst")
        try:
            ts = datetime.fromisoformat(str(recorded).replace("Z", "+00:00"))
            age_sec = max(0.0, (datetime.now(timezone.utc) - ts.astimezone(timezone.utc)).total_seconds())
            lines.append(f"ops_watchdog_heartbeat_age_seconds {age_sec:.1f}")
        except ValueError:
            pass

    lines.append("# HELP crypto_scalping_systemd_unit_active 1=active 0=not active")
    lines.append("# TYPE crypto_scalping_systemd_unit_active gauge")
    for unit in SYSTEMD_UNITS:
        lines.append(f'crypto_scalping_systemd_unit_active{{unit="{unit}"}} {systemd_active(unit)}')

    return "\n".join(lines) + "\n"


class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 (stdlib method name)
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return
        body = render_metrics().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 (stdlib signature)
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Prometheus exporter for ops_watchdog state.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9101)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), MetricsHandler)
    print(f"Serving /metrics at http://{args.host}:{args.port}/metrics", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
