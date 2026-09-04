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
# This repo doubles as a live research working directory -- untracked and
# modified experiment output routinely sits here uncommitted for hours (see
# 2026-08-08 and 2026-08-12 incidents: dirty-tree handling used to refuse to
# deploy at all, which meant any uncommitted research file blocked EVERY
# future deploy indefinitely, including ones completely unrelated to it).
# `git stash push -u` before the merge and `git stash pop` right after is
# fully reversible and never commits or discards that content -- see the
# stash step below for what happens on a pop conflict.
#
# 2026-08-12: those same concurrent sessions have also been observed running
# `git pull` directly on this machine, outside this script. Gating on "does
# HEAD match origin/main" made that silently satisfy the "nothing to do"
# check forever afterward -- the code landed on disk but the affected
# service (trading-bot.service) was never restarted, with no retry and no
# alert. Deploy progress is now tracked independently of HEAD, in
# $STATE_DIR/last_deployed_sha: the last SHA this script itself confirmed
# pulled + restarted + healthy.
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
# Must match dashboard/scripts/supervise_server.sh's own defaults/env names -- the health check
# below and that supervisor have to agree on which port "the dashboard" means.
DASHBOARD_PORT="${DASHBOARD_PORT:-8787}"
DASHBOARD_SIGTERM_GRACE_SECONDS=6   # then SIGKILL; see restart_dashboard()
DASHBOARD_HEALTH_RETRIES=6          # x5s = 30s past the shared health window, for a slow respawn

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

# Is the dashboard actually serving? The supervisor rebinds the port on respawn, and the failure
# mode above is precisely "socket released, never rebound", so port ownership is the signal that
# matters -- not an HTTP body (some endpoints re-fit models on first call and can take tens of
# seconds, which would make a request-based probe flap).
dashboard_listening() {
  ss -ltn "sport = :${DASHBOARD_PORT}" 2>/dev/null | grep -q ":${DASHBOARD_PORT}"
}

# 2026-09-03: 대시보드와 터널에는 **부팅 자동기동이 없었다.** WSL이 굳어 재부팅되자
# supervisor/수집기/섀도우 23개는 살아났는데 대시보드(8787)와 cloudflared만 안 떴고,
# 어느 계층도 그 상태를 보지 않아 사람이 페이지를 열기 전까지 아무도 몰랐다.
#
#   supervise_server.sh  -- server.py가 죽으면 3초 뒤 재기동. 단 **supervisor 자신이 없으면 무력**
#   deploy_watcher.sh    -- dashboard_changed=1 일 때만 restart_dashboard()
#   dashboard_listening  -- 그 같은 if 블록 안이라 배포가 없으면 아예 실행 안 됨
#   watchdog_deadman/ops_watchdog -- 대시보드를 보지 않음
#
# 2026-09-01 사고 후 추가한 health check가 배포 경로에만 걸려 있어 교훈이 절반만 반영돼
# 있었다. 여기서 **배포 여부와 무관하게** 매 사이클 살린다.
# `start_external.sh`/`start_cloudflare_tunnel.sh`는 이미 중복 기동을 스스로 막으므로
# (PID 확인 + 응답 확인 후 조기 종료) 무조건 호출해도 안전하다.
tunnel_running() {
  pgrep -f "[c]loudflared tunnel" >/dev/null 2>&1
}

revive_serving_if_down() {
  if ! dashboard_listening; then
    log "dashboard NOT listening on ${DASHBOARD_PORT} -- starting (no deploy involved)"
    bash "$ROOT/dashboard/scripts/start_external.sh" >/dev/null 2>&1 ||       log "start_external.sh failed"
    sleep 8
    dashboard_listening && log "dashboard revived" ||       { log "dashboard STILL not listening after start"; send_telegram "🔴 [WATCHER] dashboard down and revive failed (port ${DASHBOARD_PORT})"; }
  fi
  if ! tunnel_running; then
    log "cloudflared not running -- starting"
    bash "$ROOT/dashboard/scripts/start_cloudflare_tunnel.sh" >/dev/null 2>&1 ||       log "start_cloudflare_tunnel.sh failed"
    sleep 5
    tunnel_running && log "tunnel revived" ||       { log "tunnel STILL not running"; send_telegram "🔴 [WATCHER] cloudflared down and revive failed"; }
  fi
}

# Explicit refspec, not just "origin main": a stray local ref outside
# refs/remotes/ (refs/codex/... -- unrelated tooling debris found in this
# repo, not ours to clean up) makes the bare form choke on a bad object.
git fetch origin refs/heads/main:refs/remotes/origin/main --quiet
REMOTE_SHA="$(git rev-parse origin/main)"
PRE_RUN_SHA="$(git rev-parse HEAD)"

LAST_DEPLOYED_SHA_FILE="$STATE_DIR/last_deployed_sha"
LAST_DEPLOYED_SHA="$(cat "$LAST_DEPLOYED_SHA_FILE" 2>/dev/null || true)"

# ⭐배포 여부와 무관하게 먼저 살린다. "nothing to do"로 빠져나가기 **전**이어야 한다 --
# 대부분의 사이클이 그 경로로 끝나므로, 뒤에 두면 사실상 실행되지 않는다.
revive_serving_if_down

if [[ "$LAST_DEPLOYED_SHA" == "$REMOTE_SHA" ]]; then
  log "already deployed ($REMOTE_SHA), nothing to do"
  exit 0
fi

log "deploy needed: $REMOTE_SHA (last confirmed deploy: ${LAST_DEPLOYED_SHA:-none})"

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
# No confirmed baseline (first run under this tracking scheme, or the state
# file was lost) -- we don't actually know what SHA the running services
# reflect, so treat every tracked file as changed and restart everything
# mapped below once to establish a known-good baseline, instead of guessing
# from a diff against an unknown starting point.
if [[ -z "$LAST_DEPLOYED_SHA" ]]; then
  changed_files="$(git ls-tree -r --name-only "$REMOTE_SHA")"
else
  changed_files="$(git diff --name-only "$LAST_DEPLOYED_SHA" "$REMOTE_SHA")"
fi
affects() { echo "$changed_files" | grep -qE "$1"; }
# trading_bot_modules/odyssey_* (odyssey_tabm_core.py / odyssey_regime3_live.py /
# odyssey_live_adapter.py, added 2026-08-16) is confirmed NOT imported by trading_bot.py or by
# anything trading_bot.py imports -- it was a standalone dependency chain for the Odyssey4 shadow
# script, which was removed 2026-09-04; the exclusion stays because the premise (trading_bot.py
# never loads these) is unchanged (verified via sys.modules audit + forced-import-blocking tests, see
# docs/experiments/eth_odyssey_live_cleanroom_dependency_rewrite_20260816.md). The plain
# 'trading_bot_modules/' prefix match below would otherwise restart the real trading-bot.service
# on every odyssey_* change, which actually happened twice on 2026-08-16 (harmless -- came back
# healthy both times, deploy_watcher's own health check didn't roll back -- but still an
# unnecessary restart of a live-adjacent service for files it never loads). Exclude just that
# prefix; anything else under trading_bot_modules/ still restarts trading-bot.service by default.
trading_bot_modules_relevant_change() {
  echo "$changed_files" | grep -E '^trading_bot_modules/' | grep -qvE '^trading_bot_modules/odyssey_'
}

declare -A UNITS_TO_RESTART=()
{ affects '^trading_bot\.py$' || trading_bot_modules_relevant_change; } && UNITS_TO_RESTART[trading-bot]=1
affects '^scripts/ops_watchdog\.py$' && UNITS_TO_RESTART[ops-watchdog]=1
affects '^scripts/ops/prometheus_exporter\.py$' && UNITS_TO_RESTART[prometheus-exporter]=1
# tau1-shadow / eth-jmlam4-shadow 재시작 배선은 2026-09-04에 제거했다. 둘 다 inactive+disabled
# 이지만 **유닛 파일은 남아 있다** -- `systemctl restart`는 disabled 유닛도 시작시키므로(disabled는
# 부팅 자동시작만 막는다), 배선을 남겨두면 해당 스크립트를 누가 수정하는 순간 워처가 은퇴한
# 섀도우를 조용히 되살린다. 되살리려면 sudo systemctl enable --now 로 명시적으로 한다.
dashboard_changed=0
affects '^dashboard/' && dashboard_changed=1

# The dashboard is not a systemd unit: dashboard/scripts/supervise_server.sh owns it, records its
# child's pid here, and respawns ~3s after that child exits. So "restarting" it means killing the
# child and letting the supervisor do the rest -- never spawn one here (a manual process would win
# the port race and leave the supervisor stuck logging "port already in use" forever, with an
# unsupervised orphan serving; see feedback_dashboard_server_sigterm_shutdown_hang_20260827).
#
# 2026-09-01: plain `kill` alone is NOT enough. dashboard/server.py has a long-standing symptom
# where SIGTERM releases the listening socket but the process never exits (4 occurrences since
# 2026-08-27, unrelated to uptime). The supervisor blocks on `wait "$child"`, so a hung child means
# it never respawns -- the port stays dead until someone intervenes. That is exactly what happened
# on 2026-09-01 18:10: this function killed the child, the process hung, and the dashboard was down
# ~2min while this script went on to log "deploy OK". Escalate to SIGKILL if it doesn't exit.
restart_dashboard() {
  local child_pid_file="$ROOT/data/live/dashboard_external.child.pid"
  [[ -f "$child_pid_file" ]] || { log "no dashboard child pid file, nothing to restart"; return 0; }
  local pid
  pid="$(cat "$child_pid_file" 2>/dev/null)"
  [[ -n "$pid" ]] || return 0
  kill -0 "$pid" 2>/dev/null || { log "dashboard child $pid already gone"; return 0; }

  kill "$pid" 2>/dev/null
  local waited=0
  while (( waited < DASHBOARD_SIGTERM_GRACE_SECONDS * 2 )); do
    kill -0 "$pid" 2>/dev/null || { log "dashboard child $pid exited on SIGTERM"; return 0; }
    sleep 0.5
    (( waited++ ))
  done
  log "dashboard child $pid still alive ${DASHBOARD_SIGTERM_GRACE_SECONDS}s after SIGTERM -- sending SIGKILL"
  kill -9 "$pid" 2>/dev/null
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

# 2026-09-04: merge-first. This repo doubles as a live research working directory (see header
# comment): the server tree routinely carries 200+ untracked research scripts plus a few modified
# tracked files. Until today every cycle stashed all of them (`git stash push -u`) and popped them
# back after the merge -- 115 of those pops failed between 2026-08-12 and 2026-09-04 (each leaving
# an autostash behind and sending a Telegram, with the git error discarded), because the pop has to
# re-create hundreds of files against a tree that may have moved under it. `git merge --ff-only`
# itself only refuses when a file it must write is locally modified or exists untracked, so the
# stash is needed only in that rare case. Try the merge first; fall back to the autostash only on
# that specific refusal; log the real git error either way. Fully reversible as before: a clean pop
# restores the stash exactly, a conflicted pop leaves it undropped for a human.
stashed=0
merge_out="$(git merge --ff-only origin/main 2>&1)"; merge_rc=$?
if [[ $merge_rc -ne 0 ]]; then
  if grep -qE "would be overwritten by|untracked working tree files" <<<"$merge_out"; then
    n_paths="$(grep -cE '^[[:space:]]+[^[:space:]]' <<<"$merge_out")"
    log "merge refused: ${n_paths} local path(s) would be overwritten -- falling back to autostash"
    stash_msg="deploy_watcher autostash $(date -Iseconds)"
    if git stash push -u -m "$stash_msg" >/dev/null 2>&1; then
      stashed=1
      log "stashed local changes before pulling: $stash_msg"
    else
      log "could not stash local changes"
      send_telegram "⛔ [DEPLOY] could not stash local changes at ${REMOTE_SHA:0:8} -- not deploying, needs manual attention."
      exit 1
    fi
    merge_out="$(git merge --ff-only origin/main 2>&1)"; merge_rc=$?
  fi
  if [[ $merge_rc -ne 0 ]]; then
    log "fast-forward pull failed: ${merge_out//$'\n'/ | }"
    send_telegram "⛔ [DEPLOY] fast-forward pull failed at ${REMOTE_SHA:0:8} -- local state diverged, needs manual attention."
    [[ "$stashed" == "1" ]] && git stash pop >/dev/null 2>&1
    exit 1
  fi
fi
log "pulled to $(git rev-parse HEAD)"

if [[ "$stashed" == "1" ]]; then
  pop_out="$(git stash pop 2>&1)"; pop_rc=$?
  if [[ $pop_rc -eq 0 ]]; then
    log "restored stashed local changes"
  else
    # Conflict between the just-pulled commit and the stashed changes. The stash is untouched
    # (pop only drops on a clean apply), but the working tree may now hold a half-applied
    # conflict, so stop here instead of restarting anything against it. Whoever's changes are
    # stashed resolves this by hand (`git stash list`, `git stash pop`, fix conflicts).
    log "stash pop FAILED (stash kept for manual resolution): ${pop_out//$'\n'/ | }"
    send_telegram "⛔ [DEPLOY] pulled ${REMOTE_SHA:0:8} but restoring stashed local changes conflicted -- work is safe in 'git stash list' but needs manual resolution. NOT restarting services this cycle. git: $(tail -n 1 <<<"$pop_out")"
    exit 1
  fi
fi

if [[ ${#UNITS_TO_RESTART[@]} -eq 0 && "$dashboard_changed" == "0" ]]; then
  log "no deploy-relevant files changed (docs/research/data/etc.), pull only, no restart"
  echo "$REMOTE_SHA" > "$LAST_DEPLOYED_SHA_FILE"
  exit 0
fi

if ! restart_units; then
  send_telegram "⛔ [DEPLOY] pulled ${REMOTE_SHA:0:8} but sudo restart failed -- see scripts/ops/systemd/deploy_watcher_sudoers. Services still running old code, will retry next poll."
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

# 2026-09-01: the dashboard used to be invisible here -- this loop only ever asked systemd, and the
# dashboard is not a unit. So a restart that left it dead still reached "deploy OK" below, wrote
# last_deployed_sha, and sent a green Telegram. That is how the 18:10 outage stayed unnoticed until
# a human happened to load the page. Give it its own check, with retries: the supervisor needs ~3s
# to respawn and the process a few more to bind, and a SIGKILL escalation may have pushed that
# later than the shared 60s window assumes.
if [[ "$dashboard_changed" == "1" ]]; then
  dash_ok=0
  for _ in $(seq "$DASHBOARD_HEALTH_RETRIES"); do
    if dashboard_listening; then dash_ok=1; break; fi
    sleep 5
  done
  if [[ "$dash_ok" == "1" ]]; then
    log "dashboard is listening on ${DASHBOARD_PORT}"
  else
    log "dashboard is NOT listening on ${DASHBOARD_PORT} after restart -- regression"
    healthy=0
  fi
fi

if [[ "$healthy" == "1" ]]; then
  echo "$REMOTE_SHA" > "$LAST_DEPLOYED_SHA_FILE"
  send_telegram "🟢 [DEPLOY] ${REMOTE_SHA:0:8} deployed OK. Restarted: ${!UNITS_TO_RESTART[*]:-dashboard}"
  log "deploy OK"
  exit 0
fi

log "rolling back to $PRE_RUN_SHA"
# Refuses here instead of stashing again: something already went wrong (the
# health check failed), so any dirty state at this point is new since the
# pop above succeeded moments ago, not routine research scratch -- worth a
# human's eyes rather than another automatic step on top of a failure.
if ! working_tree_clean; then
  send_telegram "⛔ [DEPLOY] ${REMOTE_SHA:0:8} failed its health check but the working tree is now dirty -- cannot safely auto-rollback, needs manual attention NOW."
  exit 1
fi
git reset --hard "$PRE_RUN_SHA" >/dev/null 2>&1
restart_units
echo "$PRE_RUN_SHA" > "$LAST_DEPLOYED_SHA_FILE"
send_telegram "🟠 [DEPLOY] auto-rolled back to ${PRE_RUN_SHA:0:8} after ${REMOTE_SHA:0:8} failed its post-deploy health check."
