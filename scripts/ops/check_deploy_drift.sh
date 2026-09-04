#!/usr/bin/env bash
# 배포 드리프트 점검 -- git(main)과 서버에서 실제 서빙 중인 라이브 파일이 어긋났는지 확인한다.
#
# ## 왜 필요한가 (2026-09-01 대시보드 다운 사고)
#
# 이 저장소에는 배포 경로가 둘 있고 서로를 모른다:
#   (1) scripts/ops/handoff.sh push  -- rsync로 서버 파일을 직접 덮어씀. git을 안 거침.
#   (2) scripts/ops/deploy_watcher.sh -- 10분 cron. origin/main을 폴링해
#       `git stash push -u` -> `git merge --ff-only` -> `git stash pop` 사이클을 돈다.
#
# (1)로 배포하고 커밋하지 않으면 서버 워킹트리에 "git이 모르는 서빙 코드"가 남는다.
# 그 상태에서 main이 전진하면 (2)가 그걸 stash했다가 되돌리려다 **같은 줄을 건드린 경우
# stash pop 충돌** -> 서빙 파일에 <<<<<<< 마커가 그대로 박힌다.
#
# 2026-09-01 실제 사고: 이전 세션이 rsync로만 배포한 early_confirmed 기능이 커밋되지 않은
# 채 있었고, PR 머지로 main이 전진하자 정확히 그 줄에서 충돌 -> 이후 대시보드를 재시작하면서
# 크래시 루프 -> 다운. (2026-08-24에도 같은 원인으로 한 번 발생했었다.)
#
# ## 언제 돌리나
#
#   - **main에 머지하기 전** (가장 중요 -- 머지가 watcher를 깨우는 방아쇠다)
#   - handoff.sh push로 라이브 파일을 배포한 직후
#   - 대시보드가 이상할 때 (다른 원인 추정보다 먼저)
#
# ## 사용법
#   bash scripts/ops/check_deploy_drift.sh
#
# 종료코드: 0 = 안전, 1 = 위험(충돌 마커 또는 커밋되지 않은 라이브 파일 존재)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
HANDOFF="$ROOT/scripts/ops/handoff.sh"
JOB="deploy_drift_check_$$"
RC=0

echo "=== 배포 드리프트 점검 ==="
echo

# --- 1. 서버 상태 조회 (읽기 전용) ---
bash "$HANDOFF" launch server "$JOB" -- /bin/bash -c '
cd /home/llewyn/crypto-scalping
echo "SERVER_HEAD=$(git rev-parse --short HEAD)"
echo "SERVER_ORIGIN=$(git rev-parse --short origin/main)"
echo "LAST_DEPLOYED=$(cat data/live/deploy_watcher/last_deployed_sha 2>/dev/null | cut -c1-7)"
echo "--- UNMERGED ---"
git status --short | grep -E "^(UU|AA|U.|.U|DD)" || echo "(none)"
echo "--- DIRTY_LIVE_FILES ---"
# 백업/스냅샷 잔재는 서빙 코드가 아니므로 제외 -- 안 그러면 매번 늑대소년이 된다
# (.bak* / .backup* / *_backup_* / *.orig / *.pre_* 등, 2026-09-01 오탐으로 확인)
git status --short | grep -E "^( M|M |\?\?)" \
  | grep -E "dashboard/|scripts/live_|trading_bot" \
  | grep -vE "\.bak|\.backup|_backup_|\.orig$|\.pre_|\.save$|~$" || echo "(none)"
echo "--- CONFLICT_MARKERS ---"
grep -rl "<<<<<<< \|>>>>>>> Stashed\|Updated upstream" dashboard/live/ scripts/live_*.py 2>/dev/null || echo "(none)"
echo "--- UNTRACKED_SCRIPTS ---"
# 2026-09-04: 워처는 merge-first 라 미추적 파일 자체는 배포를 막지 않지만, **같은 경로를 커밋하면** 머지가
# 거부되고 폴백 stash 의 pop 이 "already exists" 로 실패한다. 개수와 앞 5개를 알려 커밋을 재촉한다.
git status --porcelain --untracked-files=all | grep -E "^\?\? (scripts/.*\.(py|sh)|[^/]+\.py)$" | cut -c4- | sed -n "1,5p"
echo "UNTRACKED_SCRIPT_COUNT=$(git status --porcelain --untracked-files=all | grep -cE "^\?\? (scripts/.*\.(py|sh)|[^/]+\.py)$")"
' >/dev/null 2>&1

for _ in $(seq 1 30); do
  sleep 3
  bash "$HANDOFF" status server "$JOB" 2>&1 | grep -q "STOPPED" && break
done
OUT="$(bash "$HANDOFF" logs server "$JOB" 2>&1)"

# --- 2. 판정 ---
# 2026-09-04: UNTRACKED_SCRIPTS 절이 뒤에 추가된 뒤 '$p'(끝까지)가 그 절을 마커로 오인해 RC=1 오탐 -> 절 경계로 한정
markers="$(echo "$OUT" | sed -n '/--- CONFLICT_MARKERS ---/,/--- UNTRACKED_SCRIPTS ---/p' | grep -vE '^---|^\(none\)$|^$' || true)"
unmerged="$(echo "$OUT" | sed -n '/--- UNMERGED ---/,/--- DIRTY_LIVE_FILES ---/p' | grep -vE '^---|^\(none\)$|^$' || true)"
dirty="$(echo "$OUT" | sed -n '/--- DIRTY_LIVE_FILES ---/,/--- CONFLICT_MARKERS ---/p' | grep -vE '^---|^\(none\)$|^$' || true)"
untracked_n="$(echo "$OUT" | grep -E '^UNTRACKED_SCRIPT_COUNT=' | tail -1 | cut -d= -f2)"
untracked_sample="$(echo "$OUT" | sed -n '/--- UNTRACKED_SCRIPTS ---/,/^UNTRACKED_SCRIPT_COUNT=/p' | grep -vE '^---|^UNTRACKED_SCRIPT_COUNT=|^$' || true)"

echo "$OUT" | grep -E "^(SERVER_HEAD|SERVER_ORIGIN|LAST_DEPLOYED)=" | sed 's/^/  /'
echo

if [[ -n "$markers" ]]; then
  echo "⛔ 충돌 마커가 서빙 파일에 박혀 있습니다 -- 즉시 조치 필요:"
  echo "$markers" | sed 's/^/     /'
  echo "     조치: 로컬 정상본을 handoff.sh push로 재전송 -> 서버에서 git add로 UU 해소"
  RC=1
elif [[ -n "$unmerged" ]]; then
  echo "⛔ 서버에 unmerged 파일이 있습니다 (watcher가 매 사이클 실패 중일 수 있음):"
  echo "$unmerged" | sed 's/^/     /'
  RC=1
fi

if [[ -n "$dirty" ]]; then
  echo "⚠️  git이 모르는 서빙 코드가 서버에 있습니다 -- main 머지 시 stash pop 충돌 위험:"
  echo "$dirty" | sed 's/^/     /'
  echo
  echo "   조치 (권장 순서):"
  echo "     1) 로컬과 md5 대조: 같으면 그 파일을 커밋해 main == 배포본으로 만든다"
  echo "     2) 다르면 서버 쪽이 최신일 수 있다 -- 덮어쓰기 전에 반드시 내용을 먼저 확인"
  echo "     3) 머지를 미룰 수 없다면, 머지 직후 watcher 사이클(최대 10분)을 지켜본다"
  [[ "$RC" == "0" ]] && RC=1
fi

if [[ -n "$untracked_n" && "$untracked_n" != "0" ]]; then
  echo "ℹ️  서버에 git이 모르는 스크립트가 ${untracked_n}개 있습니다 (경고만, 종료코드 무관):"
  echo "$untracked_sample" | sed 's/^/     /'
  echo "     같은 경로를 커밋하면 merge-first 폴백 stash 의 pop 이 실패합니다 -- 서버 사본 기준으로 커밋하고"
  echo "     푸시 직후 서버 사본을 지운 뒤 수동 ff 머지하세요 (2026-09-04 정리 절차, 호메로스 두 채널 계약)."
  echo
fi

if [[ "$RC" == "0" ]]; then
  echo "✅ 안전: 서버에 커밋 안 된 서빙 코드도, 충돌 마커도 없습니다. 머지해도 됩니다."
fi
exit $RC
