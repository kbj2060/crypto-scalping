#!/usr/bin/env bash
# 모델 서빙을 서버에서 그 자리에서 대조하고 넘기는 도구 (2026-11-21 · 사용자 «기다리지 말고 바로바로 반영 배포»). 실행은 서버에서 사람이 한다.
#   bash rtmodel/tools/takeover.sh <이름> check|go|back|status|export
#   bash rtmodel/tools/takeover.sh all export-if-stale            # 재학습 cron 뒤를 잇는 줄(아래 crontab 후보)
# 이름: rt_regime · rt_dirfwd · rt_radar · rt_dirpred · rt_sr_lean
#   check : Rust 한 회차를 그림자 폴더($SHADOW_BASE/<이름>, data/live 밖)로 → 파이썬이 같은 시각에 실제로 쓴 줄과 칸별 대조(rtmodel/tools/takeover_cmp.py)
#           → «칸 N · 어긋남 M · 짝 없음 K» 한 줄 + REST 몇 회. 통과면 $SHADOW_BASE/<이름>/check.ok.
#   go    : check.ok(2시간 안) 일 때만 → crontab 백업 → 파이썬 줄 주석(#rt-takeover:<이름># …)·Rust 줄 추가(# rt-takeover:<이름>) → 줄 수 검증
#           → 상주형은 파이썬 감독·워커 종료(/proc/<pid>/cmdline 으로 저장소 경로까지 확인 · pgrep -f 안 씀) → Rust 감독 시작 → 한 회차 지켜봄.
#   back  : 되돌리기(태그 줄만 되돌림 · 상주형은 Rust 를 끄고 파이썬 감독을 다시 켠다).
# 기록 파일은 Rust 가 같은 파일에 이어 쓴다. 끊김·중복 막기: rt_regime = (ts, coin) 중복 제거가 원본에도 Rust 에도 있음 · rt_dirfwd = 회차 시각(00·08·16시 29~40분)에는 go 거부
#   · rt_radar = 분의 20초 뒤에만 바꿈(파이썬 그 분 회차가 끝난 뒤 · 다음 분부터 Rust) + 그림자 1분봉 캐시를 옮겨 첫 회차 31일 백필(1~2분) 없이.
# 바꿀 수 있는 것(자체 시험용): ROOT · CRONTAB_CMD · SHADOW_BASE · PY · NOW_S(지금 epoch 초) · WATCH_S(지켜보는 최대 초)
set -uo pipefail
NAME="${1:?이름}"; CMD="${2:?check|go|back|status|export}"
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT" || exit 2
CRONTAB_CMD="${CRONTAB_CMD:-crontab}"
SHADOW_BASE="${SHADOW_BASE:-/mnt/data2/rtcollect_shadow/model}"
PY="${PY:-${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}}"
WATCH_S="${WATCH_S:-75}"
LIVE="$ROOT/data/live"; MODELS="$LIVE/rtmodel"; BIN="$ROOT/tmp/rtbin"; SH="$SHADOW_BASE/$NAME"
CMP="$ROOT/rtmodel/tools/takeover_cmp.py"
TAG="rt-takeover:$NAME"
now_s() { echo "${NOW_S:-$(date +%s)}"; }
say() { echo "[$NAME $CMD] $*"; }
die() { echo "[$NAME $CMD] 🔴$*" >&2; exit "${2:-1}"; }

ban_check() {   # 공용 밴 파일(scripts/binance_ban_guard.py) -- 밴 중이면 REST 를 늘리지 않는다
  local f="$LIVE/binance_ban_until" u
  [ -f "$f" ] || return 0
  u=$(tr -d '[:space:]' < "$f")
  awk -v u="$u" -v n="$(now_s)" 'BEGIN{exit !(u+0 > n+0)}' && die "바이낸스 밴 중(해제 $(date -u -d "@${u%.*}" +%H:%M:%SZ)) -- REST 를 보내지 않는다" 1
  return 0
}

# ── 이름별 설정
case "$NAME" in
  rt_regime)  PYPAT="scripts/live_regime_band_cron_20261102.py"
              RUST="cd $ROOT && RT_REGIME_OUT=data/live RT_REGIME_TAKEOVER=1 RT_REGIME_MODEL=data/live/rtmodel/regime_band.json RT_REGIME_SRC_DIR=data/live flock -n data/live/.rt_regime.lock nice -n 10 tmp/rtbin/rt_regime >> logs/supervisor/rt_regime_\$(date +\\%Y\\%m\\%d).log 2>&1" ;;
  rt_dirfwd)  PYPAT="live_eth_direction_forward_recorder_20261010.py --record"
              RUST="cd $ROOT && RT_DIRFWD_OUT=data/live RT_DIRFWD_TAKEOVER=1 nice -n 10 tmp/rtbin/rt_dirfwd >> logs/supervisor/rt_dirfwd_\$(date +\\%Y\\%m\\%d).log 2>&1" ;;
  rt_radar)   PYPAT="scripts/live_risk_radar_20261115.py"
              RUST="cd $ROOT && RT_RADAR_OUT=data/live RT_RADAR_TAKEOVER=1 bash scripts/ops/supervisor_rt_radar.sh" ;;
  rt_sr_lean) PYPAT="scripts/ops/supervisor_sr_lean.sh"
              RUST="cd $ROOT && RT_SR_OUT=data/live/sr_lean RT_SR_TAKEOVER=1 RT_SR_MODEL=data/live/rtmodel/sr_lean.json bash scripts/ops/supervisor_rt_sr_lean.sh >> logs/supervisor/rt_sr_lean_reboot.log 2>&1 &" ;;
  rt_dirpred) PYPAT=""   # 파이썬 몫 = 대시보드 서버 안(pred_record_loop) -- 표식 파일로 끈다(아래 MARK)
              RUST="cd $ROOT && RT_DIR_OUT=data/live RT_DIR_TAKEOVER=1 RT_DIR_MODELS=data/live/rtmodel RT_DIR_SRC_DIR=data/live bash scripts/ops/supervisor_rt_dirpred.sh >> logs/supervisor/rt_dirpred_reboot.log 2>&1 &" ;;
  all)        ;;
  *) die "모르는 이름: $NAME" 2 ;;
esac

# ── 모델 JSON 내보내기(재학습 뒤) -- joblib 이 JSON 보다 새것이면 다시
export_one() {
  mkdir -p "$MODELS"
  case "$1" in
    rt_regime)  "$PY" rtmodel/tools/export_regime.py "$LIVE/regime_band_h48_model.joblib" "$MODELS/regime_band.json" ;;
    rt_dirpred) "$PY" rtmodel/tools/export_dir4h.py "$LIVE" "$MODELS" && "$PY" rtmodel/tools/export_dirshort.py "$MODELS" ;;
    rt_sr_lean) "$PY" rtmodel/tools/export_hgb.py "$MODELS" ;;
    *) return 0 ;;   # rt_dirfwd(규칙) · rt_radar(계수 JSON 을 그대로 읽는다)
  esac
}
stale_one() {   # 0 = 낡음(다시 내보내야 함)
  case "$1" in
    rt_regime)  [ ! -f "$MODELS/regime_band.json" ] || [ "$LIVE/regime_band_h48_model.joblib" -nt "$MODELS/regime_band.json" ] ;;
    rt_dirpred) local j; for j in "$LIVE"/*_dir4h_model.joblib "$LIVE"/*dirshort*_model.joblib; do
                  [ -f "$j" ] || continue
                  for m in "$MODELS"/dir4h_*.json "$MODELS"/ds_*.json; do { [ ! -f "$m" ] || [ "$j" -nt "$m" ]; } && return 0; done
                done; return 1 ;;
    rt_sr_lean) [ ! -f "$MODELS/sr_lean.json" ] || [ dashboard/sr_lean_model_20261118.joblib -nt "$MODELS/sr_lean.json" ] ;;
    *) return 1 ;;
  esac
}
if [ "$NAME" = all ]; then
  [ "$CMD" = export-if-stale ] || die "all 은 export-if-stale 만" 2
  for n in rt_regime rt_dirpred rt_sr_lean; do
    if stale_one "$n"; then echo "[$n] 모델 JSON 낡음 → 내보내기"; export_one "$n" || echo "[$n] 🔴내보내기 실패 -- Rust 는 «모델 낡음»으로 멈춘다(옛 값을 내지 않음)"; fi
  done
  exit 0
fi

# ── crontab
cron_get() { $CRONTAB_CMD -l 2>/dev/null || true; }
cron_swap() {   # 파이썬 줄(주석 아닌 · PYPAT 포함) 주석 처리 + Rust 줄(같은 일정) 추가 · 줄 수 검증
  local old new k sched bak
  old=$(cron_get)
  grep -qF "$TAG" <<< "$old" && die "이미 넘긴 상태($TAG 줄 있음) -- back 먼저" 1
  k=$(grep -v '^[[:space:]]*#' <<< "$old" | grep -cF -- "$PYPAT")
  [ "$k" -ge 1 ] || die "crontab 에 파이썬 줄(${PYPAT})이 없다 -- 손으로 확인" 1
  sched=$(grep -v '^[[:space:]]*#' <<< "$old" | grep -F -- "$PYPAT" | head -1 | awk '{ if ($1 ~ /^@/) print $1; else print $1, $2, $3, $4, $5 }')
  mkdir -p "$SH"; bak="$SH/crontab.bak.$(date +%Y%m%d%H%M%S)"; printf '%s\n' "$old" > "$bak"
  new=$(awk -v pat="$PYPAT" -v tag="$TAG" 'index($0, pat) && $0 !~ /^[[:space:]]*#/ { print "#" tag "# " $0; next } { print }' <<< "$old")
  new=$(printf '%s\n%s %s # %s' "$new" "$sched" "$RUST" "$TAG")
  printf '%s\n' "$new" | $CRONTAB_CMD - || die "crontab 설치 실패(백업 $bak)" 1
  local got; got=$(cron_get)
  [ "$(wc -l <<< "$got")" -eq $(( $(wc -l <<< "$old") + 1 )) ] || die "줄 수가 안 맞는다(백업 $bak 로 되돌려라)" 1
  [ "$(grep -c "^#$TAG# " <<< "$got")" -eq "$k" ] && [ "$(grep -c "# $TAG\$" <<< "$got")" -eq 1 ] \
    && [ "$(grep -v '^[[:space:]]*#' <<< "$got" | grep -cF -- "$PYPAT")" -eq 0 ] || die "태그 줄 수가 안 맞는다(백업 $bak 로 되돌려라)" 1
  say "crontab: 파이썬 ${k}줄 주석 · Rust 1줄 추가(일정 $sched) · 백업 $bak"
}
cron_back() {
  local old new
  old=$(cron_get)
  new=$(awk -v tag="$TAG" 'index($0, "#" tag "# ") == 1 { print substr($0, length(tag) + 4); next } index($0, "# " tag) && $0 !~ ("^#" tag "#") { next } { print }' <<< "$old")
  printf '%s\n' "$new" | $CRONTAB_CMD - || die "crontab 설치 실패" 1
  grep -qF "$TAG" <<< "$(cron_get)" && die "태그 줄이 남았다" 1
  say "crontab: 태그 줄 되돌림($(grep -c "$TAG" <<< "$old")줄)"
}

# ── 프로세스(저장소 경로까지 맞는 것만 · /proc/<pid>/cmdline)
pids_argv() {   # $1 = cmdline 에 들어 있어야 할 글자(여러 개면 모두)
  local p c ok s
  for p in /proc/[0-9]*; do
    c=$(tr '\0' ' ' 2>/dev/null < "$p/cmdline") || continue
    [ -n "$c" ] || continue
    ok=1; for s in "$@"; do [[ "$c" == *"$s"* ]] || ok=0; done
    [ "$ok" = 1 ] && [ "${p#/proc/}" != "$$" ] && [ "${p#/proc/}" != "$BASHPID" ] && echo "${p#/proc/}"
  done
}
pids_env() {   # comm = $1 이고 환경 $2=$3 인 것
  local p
  for p in /proc/[0-9]*; do
    [ "$(cat "$p/comm" 2>/dev/null)" = "$1" ] || continue
    [ "$(tr '\0' '\n' 2>/dev/null < "$p/environ" | grep "^$2=" | head -1 | cut -d= -f2-)" = "$3" ] && echo "${p#/proc/}"
  done
}
stop_pids() {   # 감독 먼저, 그다음 워커 -- TERM 뒤 10초 안 죽으면 KILL · 끝나면 다시 확인
  local p i
  for p in "$@"; do say "종료 pid $p: $(tr '\0' ' ' 2>/dev/null < /proc/$p/cmdline | cut -c1-160)"; kill "$p" 2>/dev/null; done
  for i in $(seq 1 20); do local alive=0; for p in "$@"; do [ -d "/proc/$p" ] && alive=1; done; [ "$alive" = 0 ] && return 0; sleep 0.5; done
  for p in "$@"; do [ -d "/proc/$p" ] && { say "KILL pid $p"; kill -9 "$p" 2>/dev/null; }; done
  sleep 0.5; for p in "$@"; do [ -d "/proc/$p" ] && die "pid $p 가 안 죽는다" 1; done; return 0
}

check_ok_fresh() {
  [ -f "$SH/check.ok" ] || die "check.ok 없음 -- check 를 먼저(통과해야 go)" 1
  local t; t=$(head -n 1 "$SH/check.ok" | cut -d' ' -f1)   # 대조 줄이 여러 줄일 수 있다(깨진 줄 알림)
  [ $(( $(now_s) - t )) -le 7200 ] || die "check.ok 가 2시간 넘게 묵었다 -- check 다시" 1
}
record_check() {   # $1 = 대조기 종료코드 · 나머지 = 대조 한 줄
  local rc="$1"; shift
  local msg; msg=$(printf '%s' "$*" | tr '\n' ' ' | tr -s ' ')   # 대조기 출력이 여러 줄(깨진 줄 알림 등)이어도 check.ok 는 한 줄 = «epoch 내용»
  mkdir -p "$SH"
  if [ "$rc" = 0 ]; then echo "$(now_s) $msg" > "$SH/check.ok"; say "통과 · $msg"; else rm -f "$SH/check.ok"; say "🔴불통과 · $msg"; fi
  return "$rc"
}

# ── rt_dirpred: 대시보드 서버의 적중 기록·상대 강도를 끄는 표식 파일(dashboard/pred_record.py rust_on · 서버 재시작 없이 다음 회차부터)
MARK="$MODELS/pred_record_rust.on"
go_dirpred() {
  check_ok_fresh
  grep -qF "pred_record_rust.on" dashboard/pred_record.py 2>/dev/null || die "대시보드에 표식 파일 스위치가 아직 없다(dashboard/pred_record.py rust_on) -- 배포 먼저" 1
  local ph=$(( $(now_s) % 300 ))
  { [ "$ph" -ge 60 ] && [ "$ph" -le 240 ]; } || die "5분 회차 경계 근처(지금 ${ph}초) -- 파이썬 이번 회차(+20초)가 끝나고 다음 회차 전(60~240초)에" 1
  [ -n "$(pids_env rt_dirpred RT_DIR_OUT data/live)" ] && die "rt_dirpred(data/live)가 이미 돈다" 1
  grep -qF "$TAG" <<< "$(cron_get)" && die "이미 넘긴 상태($TAG 줄 있음) -- back 먼저" 1
  # 순서: 표식(파이썬이 이번 순간부터 안 씀) → Rust 시작(첫 덧붙이기 때 파일 꼬리를 읽어 파이썬이 쓴 줄은 다시 안 씀) · 다음 회차가 Rust 첫 회차 = 끊김·중복 없음
  mkdir -p "$MODELS"; date -u +%FT%TZ > "$MARK"; say "표식 $MARK 만듦(파이썬 pred_record_loop·pred_record_safe 쓰기 멈춤 · _xsec 는 xsec_latest.json 을 읽음)"
  setsid nohup bash -c "${RUST%&}" > /dev/null 2>&1 < /dev/null &
  local old bak; old=$(cron_get); bak="$SH/crontab.bak.$(date +%Y%m%d%H%M%S)"; mkdir -p "$SH"; printf '%s\n' "$old" > "$bak"
  printf '%s\n@reboot sleep 120 && %s # %s\n' "$old" "$RUST" "$TAG" | $CRONTAB_CMD - || die "crontab 설치 실패(백업 $bak)" 1
  [ "$(wc -l <<< "$(cron_get)")" -eq $(( $(wc -l <<< "$old") + 1 )) ] || die "crontab 줄 수가 안 맞는다(백업 $bak)" 1
  say "crontab: @reboot Rust 1줄 추가(파이썬 몫은 cron 이 아니라 표식으로 끔) · 백업 $bak"
  local i r=""; for i in $(seq 1 30); do r=$(pids_env rt_dirpred RT_DIR_OUT data/live); [ -n "$r" ] && break; sleep 1; done
  [ -n "$r" ] || die "Rust 워커가 안 떴다 -- back 을 검토(표식이 있어 지금은 아무도 안 쓴다)" 1
  local x="$LIVE/xsec_latest.json" before; before=$(stat -c %Y "$x" 2>/dev/null || echo 0)
  # 첫 회차가 끝나야 xsec_latest.json 을 쓴다(회차 끝 몫). 첫 회차 = 다음 5분 +20초까지 기다림 + 요청 약 760 × 간격(RT_REST_GAP_MS 기본 200ms) + 코인 사이 3초 × 4
  #   + 가중치 문턱 쉼(분마다 최대 60초 -- 11-22 서버: 첫 회차가 쉼으로 360초를 넘겼다). 그래서 한도 = 그 예상 + 여유, 넘겨도 워커가 살아 있으면 «아직 첫 회차 중»으로 가른다.
  local wait=$(( 300 - $(now_s) % 300 + 20 )) gap_ms="${RT_REST_GAP_MS:-200}"
  local lim="${DIRPRED_WATCH_S:-$(( wait + 760 * gap_ms / 1000 + 12 + 240 ))}"
  say "Rust 워커 pid $r · 첫 회차(다음 회차까지 ${wait}초 + 요청 약 760 · 최대 ${lim}초)가 끝나 xsec_latest.json(Rust 만 씀)을 쓰는지 지켜봄"
  if WATCH_S="$lim" watch_file "$x" "$before"; then
    say "✅Rust 회차가 xsec_latest.json·pred_record 를 갱신 -- 화면 «4h 상대 강도»는 이 파일을 읽는다"
  elif [ -n "$(pids_env rt_dirpred RT_DIR_OUT data/live)" ]; then
    say "⏳아직 첫 회차 중(워커 pid $r 살아 있음 · ${lim}초 안에 xsec_latest.json 이 안 바뀜 -- 가중치 문턱 쉼이 길 수 있다) -- 몇 분 뒤 'ls -l data/live/xsec_latest.json'·로그 [gate] 줄로 확인. 실패 아님"
  else
    die "워커가 죽었고 xsec_latest.json 도 안 바뀌었다 -- logs/supervisor/rt_dirpred_*.log 를 보고 back 을 검토" 1
  fi
}
back_dirpred() {
  local r
  r=$(pids_argv "_supervise.sh" "$LIVE/.supervisor_rt_dirpred.lock"); [ -n "$r" ] && stop_pids $r
  r=$(pids_env rt_dirpred RT_DIR_OUT data/live); [ -n "$r" ] && stop_pids $r
  rm -f "$MARK"; say "표식 지움(파이썬이 다음 회차부터 다시 씀 -- 처음 쓸 때 파일 꼬리를 읽어 Rust 가 쓴 줄은 다시 안 씀)"
  cron_back
}

# ── 이름별 check
check_regime() {
  local L="$LIVE/regime_band_forward.jsonl" T TS NOW out
  [ -f "$L" ] || die "파이썬 기록 없음: $L" 1
  stale_one rt_regime && die "모델 JSON 낡음 -- export 먼저" 1
  T=$("$PY" "$CMP" last "$L" ts); TS=$(date -u -d "$T" +%s)
  [ $(( $(now_s) - TS )) -le 3300 ] || die "파이썬 마지막 결정($T UTC)이 55분 넘게 지났다 -- 다음 정시 기록 뒤 바로 다시(REST 최근 3,000봉 창이 그 결정을 덮어야 한다)" 1
  ban_check
  rm -rf "$SH/run"; mkdir -p "$SH/run"
  grep -avF "\"ts\": \"$T\"" "$L" > "$SH/run/regime_band_forward.jsonl"                       # 그 결정 줄만 빼고 이어 쓰기 상태를 그대로
  cp "$LIVE/regime_band_forward_scored.jsonl" "$SH/run/" 2>/dev/null || true
  NOW=$(( (TS + 127) * 1000 ))
  say "Rust 한 회차(결정 $T · REST 8회 = 4코인 × 1500봉 두 쪽)"
  RT_REGIME_OUT="$SH/run" RT_REGIME_MODEL="$MODELS/regime_band.json" RT_REGIME_SRC_DIR="$LIVE" RT_REGIME_NOW_MS=$NOW "$BIN/rt_regime" > "$SH/run/out.txt" 2>&1 || { cat "$SH/run/out.txt"; record_check 1 "실행 실패"; return 1; }
  grep -aF "\"ts\": \"$T\"" "$L" > "$SH/run/py_T.jsonl"; grep -aF "\"ts\": \"$T\"" "$SH/run/regime_band_forward.jsonl" > "$SH/run/rs_T.jsonl"
  out=$("$PY" "$CMP" lines "$SH/run/py_T.jsonl" "$SH/run/rs_T.jsonl" --key ts,coin --text); local rc=$?
  local o2 rc2=0
  if [ -f "$LIVE/regime_band_forward_summary.json" ]; then o2=$("$PY" "$CMP" text "$LIVE/regime_band_forward_summary.json" "$SH/run/regime_band_forward_summary.json"); rc2=$?; o2="요약 $o2"; fi
  record_check $(( rc | rc2 )) "기록 $out · ${o2:-요약 없음} · REST 8"
}
check_dirfwd() {
  local L="$LIVE/eth_direction_forward.jsonl" R out
  [ -f "$L" ] || die "파이썬 기록 없음: $L" 1
  R=$("$PY" "$CMP" last "$L" recorded_ms)
  [ $(( $(now_s) * 1000 - R )) -le 5400000 ] || die "파이썬 마지막 회차가 90분 넘게 지났다(5분봉 320개 창이 그 회차를 못 덮는다) -- 다음 00·08·16시 31분 회차 뒤 바로" 1
  ban_check
  rm -rf "$SH/run"; mkdir -p "$SH/run"
  say "Rust 한 회차(recorded_ms $R · REST 6회 = 3자산 × 일봉 93·5분봉 320)"
  RT_DIRFWD_OUT="$SH/run" RT_DIRFWD_NOW_MS="$R" "$BIN/rt_dirfwd" > "$SH/run/out.txt" 2>&1 || { cat "$SH/run/out.txt"; record_check 1 "실행 실패"; return 1; }
  tail -n 3 "$L" > "$SH/run/py_last.jsonl"
  out=$("$PY" "$CMP" lines "$SH/run/py_last.jsonl" "$SH/run/eth_direction_forward.jsonl" --key sym,date,hour --text); record_check $? "기록 $out · REST 6"
}
check_radar() {
  local J="$LIVE/risk_radar_latest.json" U US out n
  [ -f "$J" ] || die "파이썬 출력 없음: $J" 1
  U=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['updated'])" "$J"); US=$(date -u -d "$U" +%s)
  [ $(( $(now_s) - US )) -le 90 ] || die "파이썬 latest 가 90초 넘게 묵었다($U) -- 크론이 도는지 먼저" 1
  ban_check
  mkdir -p "$SH/run"; rm -f "$SH/run/risk_radar_latest.json" "$SH/run/risk_radar_forward.jsonl"
  n=$(ls "$SH/run/cache" 2>/dev/null | wc -l)
  say "Rust 한 회차(파이썬 updated $U · REST $( [ "$n" -gt 0 ] && echo '코인마다 1분봉 1회 + DVOL' || echo '처음: 코인마다 1분봉 45쪽(31일 백필) + DVOL'))"
  RT_RADAR_OUT="$SH/run" RT_RADAR_NOW_MS=$(( US * 1000 )) "$BIN/rt_radar" > "$SH/run/out.txt" 2>&1
  [ -f "$SH/run/risk_radar_latest.json" ] || { cat "$SH/run/out.txt"; record_check 1 "실행 실패"; return 1; }
  # data_age_s = 벽시계 몫(달라도 됨) · 실수 끝자리는 X@β BLAS 합 순서로 1e-12 상대까지(REMAINING.md «서버 판과의 차이»)
  out=$("$PY" "$CMP" radar "$J" "$SH/run/risk_radar_latest.json" --skip data_age_s,updated --rtol 1e-12); record_check $? "latest $out · 1분봉 캐시 $(ls "$SH/run/cache" | wc -l)파일"
}
check_dirpred() {
  local D="$LIVE/pred_record" T out o2
  [ -d "$D" ] || die "파이썬 기록 없음: $D" 1
  T=$("$PY" "$CMP" maxt "$D" 4)   # 11-22: 서버 기록에 NUL 바이트가 끼어 grep 이 «binary file matches» 로 줄을 못 냈다 -- 깨진 줄은 원본 _tail 처럼 건너뛴다
  [ -n "$T" ] || die "4h 기록 줄이 없다" 1
  [ $(( $(now_s) * 1000 - T )) -le 3600000 ] || die "파이썬 마지막 결정이 1시간 넘게 지났다 -- 서버 5분 루프가 도는지" 1
  ban_check
  rm -rf "$SH/run"; mkdir -p "$SH/run"
  # 원본(서버 pred_record_loop → d4.rest_frames · 쉼 없이 연달아 · 코인 사이 3초)과 같은 첫 회차: 15분·1시간 4코인 + 4h 3코인(ETH·SOL·XRP) = 7묶음
  #   묶음 첫 코인 120회(5분봉 1000×9쪽 + metrics 5종 91쪽 + 19코인 341봉 + 일봉) · 나머지 102회 → 426 + 324 + xsec 9 = 759회
  #   klines 가중치(1000봉 5 · 341봉 2 · 일봉 120 2 · 61봉 1) ≈ 424 + metrics 637회(/futures/data -- 1분 가중치 몫은 확인 못 함 · 최악 1씩이면 +637)
  #   Rust 는 rtmodel::gate 를 지난다: 요청 사이 200ms(≥ 150초) · 응답 가중치 > RT_REST_WEIGHT_LIMIT(1,200) 면 다음 분까지 쉼 · 418/429 면 멈춤
  say "Rust 한 회차(결정 $T · 처음이라 REST 약 760회 · klines 가중치 약 424(+ metrics 637회) · 요청 사이 200ms + 가중치 1,200 넘으면 다음 분까지 쉼 → 3분 남짓)"
  RT_DIR_OUT="$SH/run" RT_DIR_MODELS="$MODELS" RT_DIR_SRC_DIR="$LIVE" RT_DIR_NOW_MS=$(( T + 20000 )) "$BIN/rt_dirpred" > "$SH/run/out.txt" 2>&1 || { cat "$SH/run/out.txt"; record_check 1 "실행 실패"; return 1; }
  local f rc=0 all=""
  for f in "$SH/run/pred_record"/*.jsonl; do
    [ -f "$f" ] || continue
    out=$("$PY" "$CMP" lines "$D/$(basename "$f")" "$f" --key k,t --text) || rc=1
    all="$all $(basename "$f" .jsonl): $out ·"
  done
  record_check "$rc" "$all REST ~760"
}
check_sr_lean() {
  local start
  local pid; pid=$(pids_env rt_sr_lean RT_SR_OUT "$SH/run" | head -n 1)
  if [ -n "$pid" ]; then
    # 그림자를 이 도구 밖에서 띄웠으면 started 가 없다 -- 워커 프로세스가 생긴 시각(/proc/<pid> 의 시각)으로(0 으로 두면 준비 구간까지 견준다)
    start=$(head -n 1 "$SH/started" 2>/dev/null || stat -c %Y "/proc/$pid")
    [ $(( $(now_s) - start )) -ge 3900 ] || die "그림자 준비 중(시작 $(( ($(now_s) - start) / 60 ))분 전 · 1시간 준비 + 5분) -- 그 뒤 check 다시" 3
    # 대조기 종료코드 그대로(0 = 양쪽 다 채점한 id 전부 같음 · 1 = 다른 줄 · 2 = 양쪽 id 없음) · 출력 전부(사건 수·다른 줄 예)를 한 줄로
    local out rc; out=$("$PY" rtmodel/tools/shadow_sr_lean.py "$LIVE/sr_lean" "$SH/run" --since $(( start + 3600 )) 2>&1); rc=$?
    record_check "$rc" "그림자 시작 $(date -u -d "@$start" +%H:%MZ) · 견줌 시작 $(date -u -d "@$(( start + 3600 ))" +%H:%MZ) · $out"
  else
    mkdir -p "$SH/run"; now_s > "$SH/started"
    RT_SR_OUT="$SH/run" RT_SR_MODEL="$MODELS/sr_lean.json" setsid nohup bash scripts/ops/supervisor_rt_sr_lean.sh > /dev/null 2>&1 < /dev/null &
    say "그림자 시작(RT_SR_OUT=$SH/run · 웹소켓 1 + 시작 klines 1회) -- 레벨 5분은 원본도 재시작 뒤 1시간 준비라 그 뒤 check 다시(65분 뒤)"
    return 3
  fi
}

# ── go·back
watch_file() {   # $1 = 파일 · $2 = 이전 mtime → WATCH_S 안에 바뀌면 0
  local i; for i in $(seq 1 "$WATCH_S"); do [ "$(stat -c %Y "$1" 2>/dev/null || echo 0)" != "$2" ] && return 0; sleep 1; done; return 1
}
go_cron() {
  check_ok_fresh
  case "$NAME" in
    rt_dirfwd) local h m; h=$(date -u -d "@$(now_s)" +%-H); m=$(date -u -d "@$(now_s)" +%-M)
               { [ "$h" = 0 ] || [ "$h" = 8 ] || [ "$h" = 16 ]; } && [ "$m" -ge 29 ] && [ "$m" -le 40 ] && die "회차 시각(${h}시 29~40분)이라 지금 바꾸면 그 회차가 두 번 또는 0번 -- 41분 뒤에" 1 ;;
    rt_regime) local m; m=$(date -u -d "@$(now_s)" +%-M); [ "$m" -le 5 ] && die "정시 0~5분(파이썬 회차 중) -- 6분 뒤에" 1 ;;
    rt_radar)  local s; s=$(date -u -d "@$(now_s)" +%-S); [ "$s" -lt 20 ] && { say "분의 20초를 기다림(파이썬 이번 분 회차가 끝난 뒤)"; sleep $(( 20 - s )); }
               [ -d "$SH/run/cache" ] && { mkdir -p "$LIVE/cache"; cp -n "$SH/run/cache"/* "$LIVE/cache/" 2>/dev/null; say "1분봉 캐시 옮김(첫 회차 백필 없음)"; } ;;
  esac
  cron_swap
  case "$NAME" in
    rt_regime) local before; before=$(stat -c %Y "$LIVE/regime_band_forward_summary.json" 2>/dev/null || echo 0)
               bash -c "${RUST}" ; say "Rust 즉시 한 회차 종료코드 $? (같은 결정 시각은 중복 제거로 새 줄 0 · 요약 다시 씀)"
               watch_file "$LIVE/regime_band_forward_summary.json" "$before" && say "✅요약 파일이 Rust 회차로 갱신됨" || die "요약 파일이 안 바뀌었다 -- back 을 검토" 1 ;;
    rt_radar)  local before; before=$(stat -c %Y "$LIVE/risk_radar_latest.json" 2>/dev/null || echo 0)
               say "다음 분 cron 회차를 지켜봄(최대 ${WATCH_S}초)"
               watch_file "$LIVE/risk_radar_latest.json" "$before" && say "✅latest 가 갱신됨(로그 logs/supervisor/rt_radar_$(date +%Y%m%d).log 마지막 줄: $(tail -1 "logs/supervisor/rt_radar_$(date +%Y%m%d).log" 2>/dev/null | cut -c1-120))" || die "latest 가 안 바뀌었다 -- back 을 검토" 1 ;;
    rt_dirfwd) say "다음 회차(00·08·16시 31분) 뒤 확인: bash rtmodel/tools/takeover.sh rt_dirfwd status" ;;
  esac
}
go_sr_lean() {
  check_ok_fresh
  local sh py pw
  sh=$(pids_argv "_supervise.sh" "$LIVE/.supervisor_rt_sr_lean.lock"); [ -n "$sh" ] && stop_pids $sh                       # 그림자 감독(같은 락)
  sh=$(pids_env rt_sr_lean RT_SR_OUT "$SH/run"); [ -n "$sh" ] && stop_pids $sh
  py=$(pids_argv "_supervise.sh" "$LIVE/.supervisor_sr_lean.lock")
  pw=$(pids_argv " -u " "$ROOT/scripts/live_sr_lean_20261118.py" | grep -vxF -f <(printf '%s\n' $py) || true)   # 감독 cmdline 에도 워커 인자가 있다 -- 감독은 뺌
  [ -n "$py$pw" ] || say "⚠️파이썬 감독·워커가 안 보인다(이미 꺼져 있음?)"
  cron_swap
  [ -n "$py" ] && stop_pids $py; [ -n "$pw" ] && stop_pids $pw
  RT_SR_OUT=data/live/sr_lean RT_SR_TAKEOVER=1 RT_SR_MODEL="$MODELS/sr_lean.json" setsid nohup bash scripts/ops/supervisor_rt_sr_lean.sh > /dev/null 2>&1 < /dev/null &
  local i r=""; for i in $(seq 1 "$WATCH_S"); do r=$(pids_env rt_sr_lean RT_SR_OUT data/live/sr_lean); [ -n "$r" ] && break; sleep 1; done
  [ -n "$r" ] || die "Rust 워커가 안 떴다 -- back 을 검토" 1
  local st="$LIVE/sr_lean/ethusdt_state.json" before; before=$(stat -c %Y "$st" 2>/dev/null || echo 0)
  watch_file "$st" "$before" && say "✅Rust 워커 pid $r 가 $st 를 갱신(재시작이라 1시간 준비 -- 원본 재시작과 같다)" || die "상태 파일이 안 바뀌었다 -- back 을 검토" 1
}
back_sr_lean() {
  local r; r=$(pids_argv "_supervise.sh" "$LIVE/.supervisor_rt_sr_lean.lock"); [ -n "$r" ] && stop_pids $r
  r=$(pids_env rt_sr_lean RT_SR_OUT data/live/sr_lean); [ -n "$r" ] && stop_pids $r
  cron_back
  setsid nohup bash scripts/ops/supervisor_sr_lean.sh > /dev/null 2>&1 < /dev/null &
  local i p=""; for i in $(seq 1 "$WATCH_S"); do p=$(pids_argv "_supervise.sh" "$LIVE/.supervisor_sr_lean.lock"); [ -n "$p" ] && break; sleep 1; done
  [ -n "$p" ] && say "✅파이썬 감독 다시 켬(pid $p)" || die "파이썬 감독이 안 떴다" 1
}

case "$NAME:$CMD" in
  *:export) export_one "$NAME" && say "내보냄 → $MODELS" ;;
  rt_regime:check)  check_regime ;;
  rt_dirfwd:check)  check_dirfwd ;;
  rt_radar:check)   check_radar ;;
  rt_dirpred:check) check_dirpred ;;
  rt_sr_lean:check) check_sr_lean ;;
  rt_regime:go|rt_dirfwd:go|rt_radar:go) go_cron ;;
  rt_regime:back|rt_dirfwd:back|rt_radar:back) cron_back ;;
  rt_sr_lean:go)    go_sr_lean ;;
  rt_sr_lean:back)  back_sr_lean ;;
  rt_dirpred:go)    go_dirpred ;;
  rt_dirpred:back)  back_dirpred ;;
  *:status) say "crontab 태그 줄: $(cron_get | grep -cF "$TAG") · check.ok: $(cat "$SH/check.ok" 2>/dev/null || echo 없음)"
            case "$NAME" in rt_dirfwd) tail -n 3 "$LIVE/eth_direction_forward.jsonl" 2>/dev/null | cut -c1-120; tail -n 2 "logs/supervisor/rt_dirfwd_$(date +%Y%m%d).log" 2>/dev/null ;; esac ;;
  *) die "모르는 명령: $CMD" 2 ;;
esac
