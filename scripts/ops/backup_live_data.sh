#!/usr/bin/env bash
# Daily backup of data/live and data/ensemble to a separate drive (D:\, mounted
# at /mnt/d under WSL) and to the dev machine (second copy, see below) -- the only copy of live state/ledgers/duckdb files
# otherwise lives on the single WSL root disk with no off-host redundancy.
#
# Intentionally does NOT use rsync --delete: a file removed locally (accidental
# rm, a bug) must stay in the backup rather than being mirrored away on the
# next run. This means the backup only grows -- acceptable given the current
# ~11.5G source size against hundreds of GB free on the destination drive.
#
# Usage: run daily from cron, e.g.:
#   0 4 * * * cd /home/llewyn/crypto-scalping && /bin/bash scripts/ops/backup_live_data.sh >> logs/backup_live_data_cron.log 2>&1
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${BACKUP_DEST:-/mnt/d/crypto-scalping-backups}"

if [[ -d "$(dirname "$DEST")" ]]; then
  mkdir -p "$DEST/data/live" "$DEST/data/ensemble"
  echo "[$(date -Iseconds)] backup starting -> $DEST"
  for dir in data/live data/ensemble; do
    rsync -a --exclude '*.tmp' "$ROOT/$dir/" "$DEST/$dir/"
  done
  echo "[$(date -Iseconds)] backup done"
else
  echo "[$(date -Iseconds)] backup destination drive not mounted, skipping ($DEST)"
fi

# ── 두 번째 사본: 다른 머신(dev) ──────────────────────────────────────────────────────────
# 2026-10-01: D: 에는 서버 WSL 디스크(ext4.vhdx)도 있어 위 사본은 디스크 고장에 같이 죽는다.
# dev 주소는 공개 저장소에 적지 않는다 -- gitignore 된 handoff.hosts.conf 의 HOSTS[dev] 를 쓴다.
# 🔴쓰는 중인 duckdb 를 rsync 하면 찢긴 사본이 될 수 있다 -> dev 에서 read_only 로 열어 표마다
#   count(*) 를 해 보고, 실패한 파일만 최대 3번 다시 보낸다. (sealed 원시 .gz 는 안 바뀐다.)
# 🔴*.wal 은 보내지 않는다 -- --delete 가 없어 원본이 체크포인트로 지운 옛 wal 이 dev 에 남고, 그걸 새 본체에
#   재생하면 사본이 깨진다. wal 없는 본체는 «마지막 체크포인트 시점»의 온전한 스냅샷이다.
#   rsync 24(전송 중 파일 사라짐)는 시각 파일 gzip 교체 때문에 정상이다.
HOSTS_CONF="$ROOT/scripts/ops/handoff.hosts.conf"
if [[ -f "$HOSTS_CONF" ]]; then
  # shellcheck disable=SC1090
  source "$HOSTS_CONF"
  IFS='|' read -r DEV_SSH _ DEV_CONDA DEV_ENV <<< "${HOSTS[dev]:-}"
fi
if [[ -z "${DEV_SSH:-}" ]]; then
  echo "[$(date -Iseconds)] dev backup skipped: HOSTS[dev] missing in $HOSTS_CONF"
  exit 0
fi
DEV_HOST="${DEV_SSH%:*}"; DEV_PORT="${DEV_SSH##*:}"
DEV_DEST="${BACKUP_DEV_DEST:-backups/crypto-scalping-server}"   # dev 홈 기준, 저장소 밖(워크트리 복제 방지)
DEV_PY="$DEV_CONDA/envs/$DEV_ENV/bin/python"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 -p "$DEV_PORT")

echo "[$(date -Iseconds)] dev backup starting -> $DEV_HOST:$DEV_DEST"
rc=0
for dir in data/live data/ensemble; do
  "${SSH[@]}" "$DEV_HOST" "mkdir -p '$DEV_DEST/$dir'" || { rc=1; continue; }
  rsync -a --exclude '*.tmp' --exclude '*.wal' -e "${SSH[*]}" "$ROOT/$dir/" "$DEV_HOST:$DEV_DEST/$dir/"
  r=$?; [[ $r -ne 0 && $r -ne 24 ]] && rc=1
done
for attempt in 1 2 3; do
  bad="$("${SSH[@]}" "$DEV_HOST" "$DEV_PY - '$DEV_DEST/data/live'" <<'PY'
import sys, pathlib, duckdb
for f in sorted(pathlib.Path(sys.argv[1]).rglob("*.duckdb")):
    try:
        c = duckdb.connect(str(f), read_only=True)
        for (t,) in c.execute("select table_name from duckdb_tables()").fetchall():
            c.execute(f'select count(*) from "{t}"').fetchone()
        c.close()
    except Exception:
        print(f.relative_to(sys.argv[1]))
PY
)" || { echo "[$(date -Iseconds)] dev verify could not run"; rc=1; break; }
  [[ -z "$bad" ]] && break
  echo "[$(date -Iseconds)] dev verify attempt $attempt: torn copies -> resend: $(echo $bad | tr '\n' ' ')"
  while IFS= read -r rel; do
    rsync -a -e "${SSH[*]}" "$ROOT/data/live/$rel" "$DEV_HOST:$DEV_DEST/data/live/$(dirname "$rel")/"
  done <<< "$bad"
  [[ $attempt == 3 ]] && { echo "[$(date -Iseconds)] dev verify FAILED after 3 attempts: $bad"; rc=1; }
done
echo "[$(date -Iseconds)] dev backup done rc=$rc"
exit $rc
