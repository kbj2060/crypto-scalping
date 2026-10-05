#!/usr/bin/env bash
# 수집 아카이브를 collector(Pi) -> server 로 일일 전송. **Pi 에서 cron 으로 돈다.**
# 기본 KST 13시(UTC 04) -- 2026-09-22 실측 최저 거래량 시각(17.1MB vs 최대 42.8MB).
# 네트워크는 기가비트라 시각이 거의 무관하지만 Pi 디스크가 수집과 덜 겹친다.
#
# 🔴**서버에 이미 있고 크기가 다른 파일은 절대 덮지 않는다.** Pi 와 서버가 같은 상대 경로에
#   쓰기 때문에, 병행 가동 구간에는 같은 시각 파일이 양쪽에 «서로 다른 내용»으로 존재한다
#   (2026-09-22 DRY_RUN 실측: 서버 9,554,943 vs Pi 8,752,413 -- Pi 가 4분 늦게 시작했다).
#   그대로 밀면 서버의 온전한 파일이 부분 파일로 덮인다. 그래서 순서가
#   «조회 -> 신규만 전송 -> 검증 -> 삭제» 다.
# 충돌(2026-10-06~, 전에는 Pi 에 영원히 쌓였다 -- 10-05 실측 354개 5.4GB):
#   크기 차 1% 안(332/354): 둘 다 온전한 한 시간(압축 차이뿐) -> 서버 유지, Pi 사본은 KEEP_HOURS 뒤 삭제.
#   1% 넘게 차이(한쪽에 구멍): 큰 쪽이 data/live 에 남고, 작은 쪽은 서버 data/archive/orderflow_conflict/
#   에 보관한다(작은 쪽에만 있는 구간이 있을 수 있어 아무것도 버리지 않는다).
#
# 🔴닫힌 .gz 만 다룬다(진행 중 시각은 .bt/.jsonl 이라 자연히 제외).
# 🔴삭제는 KEEP_HOURS(기본 48) 보다 오래되고 **검증을 통과한** 것만.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
KEEP_HOURS="${KEEP_HOURS:-48}"
DRY="${DRY_RUN:-0}"
log() { echo "[$(date -Iseconds)] $*"; }

mapfile -t FILES < <(find data/live/orderflow -type f -name '*.gz' | sort)
[[ ${#FILES[@]} -gt 0 ]] || { log "보낼 .gz 없음"; exit 0; }
log "후보 ${#FILES[@]}개 ($(du -ch "${FILES[@]}" 2>/dev/null | tail -1 | cut -f1))"

IFS='|' read -r CONN SREPO _ _ <<< "$(source scripts/ops/handoff.hosts.conf; echo "${HOSTS[server]}")"
IFS=':' read -r SUH SPORT <<< "$CONN"
SSH_O=(-p "$SPORT" -o BatchMode=yes -o ConnectTimeout=10
       -o ControlMaster=auto -o ControlPath="$HOME/.ssh/cm/%C" -o ControlPersist=120)

LIST=""; for f in "${FILES[@]}"; do LIST+="$f	$(stat -c %s "$f")"$'\n'; done

# ── ① 서버 현황 조회 (전송 전) ────────────────────────────────────────────────
PROBE="cd '$SREPO' || exit 1
while IFS=\$(printf '\\t') read -r path want; do
  [ -n \"\$path\" ] || continue
  if [ ! -f \"\$path\" ]; then echo \"NEW|\$path\"
  elif [ \"\$(stat -c %s \"\$path\")\" = \"\$want\" ]; then
    if gzip -t \"\$path\" 2>/dev/null; then echo \"SAME|\$path\"; else echo \"CORRUPT|\$path\"; fi
  elif gzip -t \"\$path\" 2>/dev/null; then echo \"CONFLICT|\$path|\$(stat -c %s \"\$path\")|\$want\"
  else echo \"CORRUPT|\$path\"; fi
done"
BEFORE=$(printf '%s' "$LIST" | timeout 900 ssh "${SSH_O[@]}" "$SUH" "$PROBE" 2>/dev/null)
[[ -n "$BEFORE" ]] || { log "서버 조회 실패 -- 중단"; exit 1; }

mapfile -t NEW  < <(grep '^NEW|'      <<< "$BEFORE" | cut -d'|' -f2)
mapfile -t SAME < <(grep '^SAME|'     <<< "$BEFORE" | cut -d'|' -f2)
nconf=$(grep -c '^CONFLICT|' <<< "$BEFORE"); ncorr=$(grep -c '^CORRUPT|' <<< "$BEFORE")
log "서버 현황: 신규 ${#NEW[@]} · 동일 ${#SAME[@]} · 충돌 $nconf · 서버측손상 $ncorr"
# 충돌 해소(머리말 참고). 서버 쪽은 PROBE 가 gzip -t 통과를 확인했다.
CA=data/archive/orderflow_conflict; BIGGER=(); SMALLER=(); DROP=()
while IFS='|' read -r _ path ssz psz; do
  if ! gzip -t "$path" 2>/dev/null || (( psz*100 <= ssz*101 && ssz*100 <= psz*101 )); then DROP+=("$path")
  elif (( psz > ssz )); then BIGGER+=("$path"); else SMALLER+=("$CA/$path"); fi
done < <(grep '^CONFLICT|' <<< "$BEFORE")
(( nconf > 0 )) && log "충돌 $nconf: 1% 안·Pi 정리 ${#DROP[@]} · Pi 가 커서 서버 교체 ${#BIGGER[@]} · Pi 가 작아 서버 보관함으로 ${#SMALLER[@]}"
if [[ "$DRY" != "1" ]]; then
  for a in "${SMALLER[@]}"; do mkdir -p "$(dirname "$a")" && mv -n "${a#$CA/}" "$a"; done
  NEW+=("${SMALLER[@]}")
fi
if (( ${#BIGGER[@]} > 0 )) && [[ "$DRY" != "1" ]]; then
  printf '%s\n' "${BIGGER[@]}" | timeout 300 ssh "${SSH_O[@]}" "$SUH" "cd '$SREPO' || exit 1
    while read -r p; do d=$CA/\$p; mkdir -p \"\$(dirname \"\$d\")\" && mv -n \"\$p\" \"\$d\"; done" \
    || { log "서버 사본 이동 실패 -- 교체 생략"; BIGGER=(); }
fi
NEW+=("${BIGGER[@]}")
(( ncorr > 0 )) && { log "🔴 서버측 .gz 손상 -- 사람이 봐야 한다:"; grep '^CORRUPT|' <<< "$BEFORE" | head -5 | sed 's/^/    /'; }

# ── ② 신규만 전송 ─────────────────────────────────────────────────────────────
if (( ${#NEW[@]} == 0 )); then log "전송할 신규 없음"
elif [[ "$DRY" == "1" ]]; then log "DRY_RUN -- ${#NEW[@]}개 전송 생략"
else
  # 🔴HANDOFF_RSYNC_Z=0: 이미 gzip 이라 -z 는 CPU 만 쓴다(실측 이득 1.0%/-0.0%).
  HANDOFF_RSYNC_Z=0 bash scripts/ops/handoff.sh push server "${NEW[@]}" >/dev/null 2>&1 \
    || { log "전송 실패 -- 아무것도 지우지 않는다"; exit 1; }
  log "전송 ${#NEW[@]}개 완료"
fi

# ── ③ 전송 후 재검증 (크기 + gzip -t) ─────────────────────────────────────────
AFTER=$(printf '%s' "$LIST" | timeout 900 ssh "${SSH_O[@]}" "$SUH" "$PROBE" 2>/dev/null)
mapfile -t VERIFIED < <(grep '^SAME|' <<< "$AFTER" | cut -d'|' -f2)
log "검증 통과 ${#VERIFIED[@]}개 / 후보 ${#FILES[@]}개"

# ── ④ 검증 통과 + KEEP_HOURS 경과분만 삭제 ────────────────────────────────────
now=$(date +%s); cutoff=$(( now - KEEP_HOURS*3600 )); del=0; kept=0; freed=0
for path in "${VERIFIED[@]}" "${DROP[@]}"; do
  [[ -f "$path" ]] || continue
  mt=$(stat -c %Y "$path") || continue
  if (( mt < cutoff )); then
    freed=$(( freed + $(stat -c %s "$path") ))
    [[ "$DRY" == "1" ]] || rm -f "$path"
    del=$((del+1))
  else kept=$((kept+1)); fi
done
log "삭제 $del개 ($(awk -v b=$freed 'BEGIN{printf "%.1f MB", b/1e6}')) · 최근이라 보존 $kept개 (KEEP_HOURS=$KEEP_HOURS)"
log "Pi 디스크: $(df -h / | awk 'NR==2{print $3"/"$2" ("$5")"}')"
(( ncorr > 0 )) && exit 2 || exit 0
