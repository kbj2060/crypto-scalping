#!/usr/bin/env bash
# Pull-based local deploy watcher (no self-hosted GitHub Actions runner, no
# inbound access to this machine -- GitHub never reaches in, this script
# always reaches out). Polls origin/main, waits for the GitHub Actions CI
# run on the latest commit to succeed, pulls, restarts only the systemd
# units whose source files actually changed, then watches a short health
# window and auto-rolls-back (git reset + redeploy the previous commit) if
# the restarted unit doesn't come back healthy.
#
# Needs a narrowly-scoped sudoers rule to restart systemd units without a
# password -- see scripts/ops/systemd/deploy_watcher_sudoers (install with
# visudo, this script cannot set that up itself: chicken-and-egg, sudo
# itself is what's being granted). Until installed, restarts fail loudly
# via Telegram but the git pull itself still succeeds.
#
# This repo has observed concurrent sessions (other agents/terminals)
# committing to it mid-session -- never touches git state (pull, reset)
# while the working tree is dirty. A concurrent uncommitted edit at deploy
# time aborts the run rather than risking `git reset --hard` over it.
#
# Usage: run periodically from cron, e.g. every 10 minutes:
#   */10 * * * * cd /home/llewyn/crypto-scalping && /bin/bash scripts/ops/deploy_watcher.sh >> logs/deploy_watcher_cron.log 2>&1
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

LOCK="$ROOT/data/live/.deploy_watcher.lock"
exec 9>"$LOCK"
flock -n 9 || { echo "[$(date -Iseconds)] previous run still in flight, skipping"; exit 0; }

GITHUB_REPO="kbj2060/crypto-scalping"
STATE_DIR="$ROOT/data/live/deploy_watcher"
LAST_NOTIFIED_FAILED_SHA="$STATE_DIR/last_notified_failed_sha"
HEALTH_CHECK_WAIT_SECONDS=60

mkdir -p "$STATE_DIR"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

log() { echo "[$(date -Iseconds)] $*"; }

send_telegram() {
  local text="$1"
  [[ -z "${TELEGRAM_BOT_TOKEN:-}" || -z "${TELEGRAM_CHAT_ID:-}" ]] && return 0
  curl -s -m 8 -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
    -d "chat_id=${TELEGRAM_CHAT_ID}" --data-urlencode "text=${text}" >/dev/null 2>&1 || true
}

working_tree_clean() {
  [[ -z "$(git status --porcelain)" ]]
}

# Explicit refspec, not just "origin main": a stray local ref outside
# refs/remotes/ (refs/codex/... -- unrelated tooling debris found in this
# repo, not ours to clean up) makes the bare form choke on a bad object.
git fetch origin refs/heads/main:refs/remotes/origin/main --quiet
LOCAL_SHA="$(git rev-parse HEAD)"
REMOTE_SHA="$(git rev-parse origin/main)"

if [[ "$LOCAL_SHA" == "$REMOTE_SHA" ]]; then
  log "up to date ($LOCAL_SHA), nothing to do"
  exit 0
fi

log "new commit available: $REMOTE_SHA (current: $LOCAL_SHA)"

if ! working_tree_clean; then
  log "working tree dirty (concurrent session?) -- not touching git state this cycle"
  exit 0
fi

# --- CI status for the remote commit, via GitHub's check-runs API ---
auth_header=()
[[ -n "${GITHUB_TOKEN:-}" ]] && auth_header=(-H "Authorization: Bearer ${GITHUB_TOKEN}")

check_runs="$(curl -s -m 15 "${auth_header[@]}" \
  "https://api.github.com/repos/${GITHUB_REPO}/commits/${REMOTE_SHA}/check-runs")"

ci_status="$(echo "$check_runs" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
except Exception:
    print('unknown'); sys.exit(0)
runs = [r for r in data.get('check_runs', []) if r.get('name') == 'syntax-check']
if not runs:
    print('missing')
elif any(r.get('status') != 'completed' for r in runs):
    print('pending')
elif all(r.get('conclusion') == 'success' for r in runs):
    print('success')
else:
    print('failure')
")"

log "CI status for $REMOTE_SHA: $ci_status"

case "$ci_status" in
  pending|missing)
    log "CI not finished yet, waiting for next poll"
    exit 0
    ;;
  failure)
    if [[ "$(cat "$LAST_NOTIFIED_FAILED_SHA" 2>/dev/null)" != "$REMOTE_SHA" ]]; then
      send_telegram "⛔ [DEPLOY] CI failed for ${REMOTE_SHA:0:8} -- not deploying. https://github.com/${GITHUB_REPO}/commit/${REMOTE_SHA}"
      echo "$REMOTE_SHA" > "$LAST_NOTIFIED_FAILED_SHA"
    fi
    exit 0
    ;;
  success)
    log "CI passed, deploying"
    ;;
  *)
    log "unknown CI status ($ci_status), not deploying"
    exit 0
    ;;
esac

# --- which units does this diff actually touch? ---
changed_files="$(git diff --name-only "$LOCAL_SHA" "$REMOTE_SHA")"
affects() { echo "$changed_files" | grep -qE "$1"; }

declare -A UNITS_TO_RESTART=()
affects '^trading_bot\.py$|^trading_bot_modules/' && UNITS_TO_RESTART[trading-bot]=1
affects '^scripts/ops_watchdog\.py$' && UNITS_TO_RESTART[ops-watchdog]=1
affects '^scripts/ops/prometheus_exporter\.py$' && UNITS_TO_RESTART[prometheus-exporter]=1
affects '^scripts/run_btc_multislot_shadow_loop_20260807\.py$' && UNITS_TO_RESTART[btc-multislot-shadow]=1
affects '^scripts/live_sigma6_regime_tiebreak_shadow_20260801\.py$' && UNITS_TO_RESTART[tau1-shadow]=1
affects '^scripts/live_eth_jmlam4_regime_swap_shadow_20260809\.py$' && UNITS_TO_RESTART[eth-jmlam4-shadow]=1
dashboard_changed=0
affects '^dashboard/' && dashboard_changed=1

restart_dashboard() {
  local child_pid_file="$ROOT/data/live/dashboard_external.child.pid"
  [[ -f "$child_pid_file" ]] && kill "$(cat "$child_pid_file")" 2>/dev/null
}

restart_units() {
  # bash exit-code convention: 0 = all restarts succeeded, 1 = at least one failed.
  local failed=0
  for unit in "${!UNITS_TO_RESTART[@]}"; do
    log "restarting ${unit}.service"
    if ! sudo -n /usr/bin/systemctl restart "${unit}.service" 2>&1; then
      log "sudo restart failed for ${unit}.service"
      failed=1
    fi
  done
  [[ "$dashboard_changed" == "1" ]] && { log "restarting dashboard"; restart_dashboard; }
  return $failed
}

if ! working_tree_clean; then
  log "working tree became dirty while checking CI status -- aborting before pull"
  exit 0
fi

if ! git merge --ff-only origin/main >/dev/null 2>&1; then
  send_telegram "⛔ [DEPLOY] fast-forward pull failed at ${REMOTE_SHA:0:8} -- local state diverged, needs manual attention."
  exit 1
fi
log "pulled to $(git rev-parse HEAD)"

if [[ ${#UNITS_TO_RESTART[@]} -eq 0 && "$dashboard_changed" == "0" ]]; then
  log "no deploy-relevant files changed (docs/research/data/etc.), pull only, no restart"
  exit 0
fi

if ! restart_units; then
  send_telegram "⛔ [DEPLOY] pulled ${REMOTE_SHA:0:8} but sudo restart failed -- see scripts/ops/systemd/deploy_watcher_sudoers. Services still running old code."
  exit 1
fi

sleep "$HEALTH_CHECK_WAIT_SECONDS"

healthy=1
for unit in "${!UNITS_TO_RESTART[@]}"; do
  state="$(systemctl is-active "${unit}.service" 2>/dev/null)"
  if [[ "$state" != "active" ]]; then
    log "${unit}.service is '$state' after restart -- regression"
    healthy=0
  fi
done

if [[ "$healthy" == "1" ]]; then
  send_telegram "🟢 [DEPLOY] ${REMOTE_SHA:0:8} deployed OK. Restarted: ${!UNITS_TO_RESTART[*]:-dashboard}"
  log "deploy OK"
  exit 0
fi

log "rolling back to $LOCAL_SHA"
if ! working_tree_clean; then
  send_telegram "⛔ [DEPLOY] ${REMOTE_SHA:0:8} failed its health check but the working tree is now dirty -- cannot safely auto-rollback, needs manual attention NOW."
  exit 1
fi
git reset --hard "$LOCAL_SHA" >/dev/null 2>&1
restart_units
send_telegram "🟠 [DEPLOY] auto-rolled back to ${LOCAL_SHA:0:8} after ${REMOTE_SHA:0:8} failed its post-deploy health check."
