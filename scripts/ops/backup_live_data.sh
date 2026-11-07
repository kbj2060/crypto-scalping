#!/usr/bin/env bash
# Daily backup. 🔴규칙: 원본과 사본은 항상 다른 디스크(2026-11-07 재설계, docs/data_storage_map_20261107.md).
#
#   hot  = NVMe(data/hot·data/live)와 루트 디스크의 data/ 전체(제외 목록 아래) -> 6TB + dev
#   cold = 6TB 가 원본인 orderflow·lake·splits      -> D: + dev
#
# D: 사본: 서버 ext4.vhdx 가 D: 에 있는 동안은 D: 공간이 곧 서버 디스크의 한계라, D: 여유가 300GiB 넘을 때만 돈다
#   (11-07 압축 뒤 vhdx 212GB·여유 548GB. 압축 전에는 여유 102GB 에 7~8GB/일 소진이었다). 안 돌 때 cold 의 둘째 사본은 dev 다.
#
# Intentionally does NOT use rsync --delete: a file removed locally (accidental rm, a bug) must stay in the
# backup rather than being mirrored away on the next run. 그래서 사본은 늘기만 한다 -- 보존 기간이 정해진 것
# (raster 14일)과 다시 받을 수 있는 것(binance_vision)은 처음부터 보내지 않는다.
#
# 실패를 조용히 넘기지 않는다: 결과는 data/live/ops_watchdog/backup_status.json 에 남고 감시기(check_backup)가 읽는다.
#
# Usage: run daily from cron:
#   0 4 * * * cd /home/llewyn/crypto-scalping && /bin/bash scripts/ops/backup_live_data.sh >> logs/backup_live_data_cron.log 2>&1
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA2="${BACKUP_DATA2:-/mnt/data2}"
DEST="${BACKUP_DEST:-$DATA2/crypto-scalping-backups}"
COLD_DEST="${BACKUP_COLD_DEST:-/mnt/d/crypto-scalping-cold}"
VHDX_ON_D="${BACKUP_VHDX_PATH:-/mnt/d/wsl/Ubuntu/ext4.vhdx}"
STATUS="${BACKUP_STATUS:-$ROOT/data/live/ops_watchdog/backup_status.json}"
log() { echo "[$(date -Iseconds)] $*"; }

# 🔴*.wal 은 보내지 않는다 -- --delete 가 없어 원본이 체크포인트로 지운 옛 wal 이 새 본체 옆에 남고, 그 사본을
#   열면 옛 wal 이 **조용히 재생돼 행이 중복**된다(2026-10-16 D: 에 16개 실측·재현).
# --no-links: data/ 안의 6TB 심볼릭 링크(orderflow/*·lake·splits)는 아래 cold 단계가 실물로 보낸다.
HOT_RSYNC=(-a --no-links --exclude '*.tmp' --exclude '*.wal' --exclude '.snap_*' --exclude '*.moved')
# 진행 중 시각 파일(.bt·.jsonl)은 다음날 .gz 가 되므로 보내면 부분 사본만 남는다(ETH 만 57개 795MB 쌓였었다).
COLD_RSYNC=(-a --exclude '*.tmp' --exclude '*.wal' --exclude '.snap_*' --exclude '*.bt' --exclude '*.jsonl')
COLD_MAP=("orderflow:data/live/orderflow" "lake:data/lake" "splits:data/splits")

rs() { rsync "$@"; local r=$?; [[ $r -eq 24 ]] && r=0; return $r; }   # 24 = 전송 중 사라짐(시각 파일 gzip 교체) -- 정상

send_hot() {    # send_hot <받는 쪽 접두어> [rsync 옵션...]  data/live 는 NVMe 링크라 따로(--no-links 가 건너뛴다)
  local prefix="$1" rc=0; shift
  rs "${HOT_RSYNC[@]}" --exclude '/hot/' --exclude '/hot_backup/' --exclude '/live/' --exclude '/tmp/' --exclude '/test/' \
     --exclude '/binance_vision/' --exclude '/orderflow_hdd/' "$@" "$ROOT/data/" "$prefix/data/" || rc=1
  rs "${HOT_RSYNC[@]}" --exclude '/orderflow/raster/' "$@" "$ROOT/data/live/" "$prefix/data/live/" || rc=1
  return $rc
}

send_cold() {   # send_cold <받는 쪽 접두어> [rsync 옵션...]
  local prefix="$1" rc=0 m; shift
  for m in "${COLD_MAP[@]}"; do
    [[ -d "$DATA2/${m%%:*}" ]] || continue
    rs "${COLD_RSYNC[@]}" "$@" "$DATA2/${m%%:*}/" "$prefix/${m##*:}/" || rc=1
  done
  return $rc
}

# 수집기는 시각이 바뀔 때만 압축한다 -- 재시작하면 쓰던 시각 파일이 .jsonl/.bt 로 남아 위 제외 규칙에 걸리고 사본이 안 생긴다
#   (11-07 13개 278MB, 09-17 것까지). 2시간 넘게 안 바뀐 것을 닫는다. 같은 시각 .gz 가 이미 있으면(Pi 완본) 부분본은 치워 둔다 --
#   둘 다 두면 읽는 쪽이 그 시각을 두 번 센다.
seal_stale() {
  local f part
  while IFS= read -r -d '' f; do
    if [[ -e "$f.gz" ]]; then
      part="$DATA2/orderflow_partial/${f#"$DATA2/orderflow/"}"
      mkdir -p "$(dirname "$part")" && mv -n "$f" "$part"
    else
      nice -n 10 gzip "$f" || log "🔴 gzip failed: $f"
    fi
  done < <(find "$DATA2/orderflow/" -type f \( -name '*.jsonl' -o -name '*.bt' \) -mmin +120 -print0 2>/dev/null)
}

hot_6tb=1; cold_d=1; dev=1
mkdir -p "$ROOT/data/live/backups" "$(dirname "$STATUS")"
crontab -l > "$ROOT/data/live/backups/crontab_latest.txt.tmp" 2>/dev/null \
  && mv "$ROOT/data/live/backups/crontab_latest.txt.tmp" "$ROOT/data/live/backups/crontab_latest.txt"

# 🔴mountpoint: 안 붙은 /mnt/data2 는 그냥 빈 폴더라 없는 줄 모르고 루트 디스크에 수백 GB 를 쓰게 된다.
if mountpoint -q "$DATA2" || [[ "${BACKUP_SKIP_MOUNT_CHECK:-0}" == "1" ]]; then
  log "backup starting -> $DEST"
  seal_stale
  hot_6tb=0
  mkdir -p "$DEST/data/hot_backup"
  # hot SQLite 는 쓰는 중인 파일을 rsync 하면 찢긴다 -> 온라인 백업 API 로 한 시점 사본(WAL 이라 수집기 쓰기를 안 막는다).
  for f in "$ROOT"/data/hot/*.sqlite; do
    [[ -f "$f" ]] || continue
    dst="$DEST/data/hot_backup/$(basename "$f")"
    python3 -c 'import sqlite3, sys
s = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True, timeout=60); d = sqlite3.connect(sys.argv[2])
s.backup(d); d.close(); s.close()' "$f" "$dst.tmp" && mv "$dst.tmp" "$dst" \
      || { log "🔴 hot backup failed: $f"; hot_6tb=1; }
  done
  send_hot "$DEST" || hot_6tb=1
  log "backup done rc=$hot_6tb"
else
  log "🔴 $DATA2 not mounted -- 6TB 단계와 cold 단계를 건너뛴다"
fi

d_free_gib=$(df --output=avail -BG "$(dirname "$COLD_DEST")" 2>/dev/null | tail -1 | tr -dc 0-9)
if [[ -e "$VHDX_ON_D" ]] && (( ${d_free_gib:-0} < ${BACKUP_D_MIN_FREE_GIB:-300} )); then
  cold_d='"skipped_vhdx_on_d"'
elif mountpoint -q "$DATA2" && [[ -d "$(dirname "$COLD_DEST")" ]]; then
  log "cold backup starting -> $COLD_DEST"
  send_cold "$COLD_DEST" --no-perms --no-owner --no-group && cold_d=0
  log "cold backup done rc=$cold_d"
fi

# ── 다른 머신(dev) ──────────────────────────────────────────────────────────────────────
# dev 주소는 공개 저장소에 적지 않는다 -- gitignore 된 handoff.hosts.conf 의 HOSTS[dev] 를 쓴다.
# 🔴쓰는 중인 duckdb 를 rsync 하면 찢긴 사본이 될 수 있다 -> dev 에서 read_only 로 열어 표마다
#   count(*) 를 해 보고, 실패한 파일만 최대 3번 다시 보낸다. (sealed 원시 .gz 는 안 바뀐다.)
HOSTS_CONF="$ROOT/scripts/ops/handoff.hosts.conf"
if [[ -f "$HOSTS_CONF" ]]; then
  # shellcheck disable=SC1090
  source "$HOSTS_CONF"
  IFS='|' read -r DEV_SSH _ DEV_CONDA DEV_ENV <<< "${HOSTS[dev]:-}"
fi
if [[ -n "${DEV_SSH:-}" ]]; then
  DEV_HOST="${DEV_SSH%:*}"; DEV_PORT="${DEV_SSH##*:}"
  DEV_DEST="${BACKUP_DEV_DEST:-backups/crypto-scalping-server}"   # dev 홈 기준, 저장소 밖(워크트리 복제 방지)
  DEV_PY="$DEV_CONDA/envs/$DEV_ENV/bin/python"
  SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 -p "$DEV_PORT")
  log "dev backup starting -> $DEV_HOST:$DEV_DEST"
  rc=0
  "${SSH[@]}" "$DEV_HOST" "mkdir -p '$DEV_DEST/data/live/orderflow' '$DEV_DEST/data/lake' '$DEV_DEST/data/splits' '$DEV_DEST/data/hot_backup'" || rc=1
  send_hot "$DEV_HOST:$DEV_DEST" -e "${SSH[*]}" || rc=1
  if mountpoint -q "$DATA2"; then
    send_cold "$DEV_HOST:$DEV_DEST" -e "${SSH[*]}" || rc=1
    rs -a --exclude '*.tmp' -e "${SSH[*]}" "$DEST/data/hot_backup/" "$DEV_HOST:$DEV_DEST/data/hot_backup/" || rc=1
  else
    rc=1
  fi
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
)" || { log "dev verify could not run"; rc=1; break; }
    [[ -z "$bad" ]] && break
    log "dev verify attempt $attempt: torn copies -> resend: $(echo $bad | tr '\n' ' ')"
    while IFS= read -r rel; do
      rsync -a -e "${SSH[*]}" "$ROOT/data/live/$rel" "$DEV_HOST:$DEV_DEST/data/live/$(dirname "$rel")/"
    done <<< "$bad"
    [[ $attempt == 3 ]] && { log "dev verify FAILED after 3 attempts: $bad"; rc=1; }
  done
  dev=$rc
  log "dev backup done rc=$dev"
else
  log "dev backup skipped: HOSTS[dev] missing in $HOSTS_CONF"
fi

printf '{"finished_at": %s, "targets": {"hot_6tb": %s, "cold_d": %s, "dev": %s}}\n' \
  "$(date +%s)" "$hot_6tb" "$cold_d" "$dev" > "$STATUS.tmp" && mv "$STATUS.tmp" "$STATUS"
[[ "$hot_6tb" == 0 && "$dev" == 0 && ( "$cold_d" == 0 || "$cold_d" == '"skipped_vhdx_on_d"' ) ]]
