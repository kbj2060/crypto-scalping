#!/usr/bin/env bash
# Independent dead-man's-switch for ops_watchdog itself.
#
# ops_watchdog.py can only report a problem while it's alive to run its own
# checks -- if its process hangs or the supervisor stops respawning it, nothing
# inside that process can notice. This script is intentionally a *separate*
# process lineage (run from cron, not from _supervise.sh) with no dependency on
# the python/conda environment, so a broken env that silently kills the
# watchdog doesn't also disable this check.
#
# Usage: run periodically from cron, e.g. every 3 minutes:
#   */3 * * * * cd /home/llewyn/crypto-scalping && /bin/bash scripts/ops/watchdog_deadman.sh >> logs/watchdog_deadman_cron.log 2>&1
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

HEARTBEAT="$ROOT/data/live/ops_watchdog/watchdog_heartbeat.json"
MARKER="$ROOT/data/live/.watchdog_deadman_alerted"
STALE_MINUTES=5

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

send_telegram() {
  local text="$1"
  [[ -z "${TELEGRAM_BOT_TOKEN:-}" || -z "${TELEGRAM_CHAT_ID:-}" ]] && return 0
  curl -s -m 8 -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
    -d "chat_id=${TELEGRAM_CHAT_ID}" --data-urlencode "text=${text}" >/dev/null 2>&1 || true
}

# 2026-11-07 data/live·logs 가 /mnt/nvme, 원시 호가가 /mnt/data2 다. 디스크가 빠지면 감시기 자신이 못 알린다.
# 표시는 /tmp 에 둔다(data/live 가 깨진 상태에서도 돌아야 한다). 그래서 crontab 의 로그도 /tmp 로 받는다.
for m in "${DEADMAN_MOUNTS:-/mnt/nvme /mnt/data2}"; do
  for mp in $m; do
    flag="/tmp/.deadman_unmounted_$(basename "$mp")"
    if mountpoint -q "$mp"; then
      [[ -f "$flag" ]] && { rm -f "$flag"; send_telegram "✅ [DEADMAN] $mp mounted again"; }
    elif [[ ! -f "$flag" ]]; then
      echo "[$(date -Iseconds)] DEADMAN: $mp not mounted"
      send_telegram "⛔ [DEADMAN] $mp not mounted -- collectors on it are down. Recovery: wsl_boot_recover.ps1 -MountOnly (docs/data_storage_map_20261107.md)"
      touch "$flag"
    fi
  done
done

recorded_at=""
if [[ -f "$HEARTBEAT" ]]; then
  recorded_at="$(grep -o '"recorded_at_kst"[[:space:]]*:[[:space:]]*"[^"]*"' "$HEARTBEAT" | sed -E 's/.*"([^"]+)"$/\1/')"
fi

status="ok"
if [[ -z "$recorded_at" ]]; then
  status="missing"
else
  recorded_epoch="$(date -d "$recorded_at" +%s 2>/dev/null || echo 0)"
  now_epoch="$(date +%s)"
  if (( recorded_epoch == 0 )); then
    status="unparsable"
  else
    age_min=$(( (now_epoch - recorded_epoch) / 60 ))
    (( age_min >= STALE_MINUTES )) && status="stale"
  fi
fi

if [[ "$status" != "ok" ]]; then
  echo "[$(date -Iseconds)] DEADMAN: ops_watchdog heartbeat status=$status last=${recorded_at:-none}"
  if [[ ! -f "$MARKER" ]]; then
    send_telegram "⛔ [DEADMAN] ops_watchdog heartbeat ${status} (last: ${recorded_at:-none}). It can't self-report if it's hung -- check manually: scripts/ops/botctl.sh status"
    date -Iseconds > "$MARKER"
  fi
else
  if [[ -f "$MARKER" ]]; then
    echo "[$(date -Iseconds)] DEADMAN: ops_watchdog heartbeat recovered"
    send_telegram "🟢 [DEADMAN] ops_watchdog heartbeat recovered."
    rm -f "$MARKER"
  fi
fi
