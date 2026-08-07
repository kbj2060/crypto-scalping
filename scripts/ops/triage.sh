#!/usr/bin/env bash
# Read-only first-response summary for the live operations watchdog.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

"$PY" - <<'PY' || true
import json
from pathlib import Path

root = Path.cwd()
snapshot = root / "data/live/ops_watchdog/health_snapshot.json"
try:
    data = json.loads(snapshot.read_text(encoding="utf-8"))
except FileNotFoundError:
    print(f"[WARN] health snapshot not found at {snapshot}")
except (json.JSONDecodeError, OSError) as exc:
    print(f"[WARN] health snapshot unreadable ({exc})")
else:
    print(f"watchdog updated: {data.get('updated_at_kst')}")
    for check in data.get("checks", []):
        if check.get("status") != "OK":
            print(f"[{check.get('status')}] {check.get('component')}: {check.get('summary')}")
            print(json.dumps(check.get("details", {}), ensure_ascii=False, sort_keys=True))
PY

echo "--- supervisors ---"
"$ROOT/scripts/ops/botctl.sh" status
echo "--- recent watchdog events ---"
latest_events="$(find "$ROOT/data/live/ops_watchdog/history" -maxdepth 1 -name 'events_*.jsonl' -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
[[ -n "$latest_events" ]] && tail -n 20 "$latest_events" || true
