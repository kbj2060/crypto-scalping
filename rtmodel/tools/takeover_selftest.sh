#!/usr/bin/env bash
# takeover.sh 자체 시험 (2026-11-21) -- 가짜 저장소·가짜 crontab·가짜 실행 파일·가짜 파이썬 워커(진짜 프로세스 · 가짜 저장소 경로)로 check·go·back.
# 진짜 crontab·진짜 서버 프로세스는 건드리지 않는다(CRONTAB_CMD = 가짜 · 프로세스는 가짜 저장소 경로가 cmdline 에 있는 것만 끈다).
#   bash rtmodel/tools/takeover_selftest.sh
set -uo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
F=$(mktemp -d); R="$F/repo"; L="$R/data/live"
mkdir -p "$R/rtmodel/tools" "$R/scripts/ops" "$R/bin" "$L/rtmodel" "$L/sr_lean" "$R/logs/supervisor" "$R/dashboard"
cp "$SRC/rtmodel/tools/takeover.sh" "$SRC/rtmodel/tools/takeover_cmp.py" "$R/rtmodel/tools/"
cp "$SRC/scripts/ops/_supervise.sh" "$SRC/scripts/ops/supervisor_sr_lean.sh" "$SRC/scripts/ops/supervisor_rt_sr_lean.sh" "$SRC/scripts/ops/supervisor_rt_radar.sh" "$R/scripts/ops/"
echo "# 가짜 워커" > "$R/scripts/live_sr_lean_20261118.py"
PASS=0; FAIL=0
ok() { if eval "$2"; then echo "  ✅ $1"; PASS=$((PASS + 1)); else echo "  🔴 $1"; FAIL=$((FAIL + 1)); fi; }
cleanup() {
  for p in /proc/[0-9]*; do tr '\0' ' ' 2>/dev/null < "$p/cmdline" | grep -qF "$F/" && kill -9 "${p#/proc/}" 2>/dev/null; done
  rm -rf "$F"
}
trap cleanup EXIT

# 가짜 crontab(-l = 보기 · - = 표준입력으로 설치)
cat > "$F/crontab" <<EOF
#!/usr/bin/env bash
case "\$1" in -l) cat "$F/crontab.txt" ;; -) cat > "$F/crontab.txt" ;; *) cp "\$1" "$F/crontab.txt" ;; esac
EOF
cat > "$F/crontab.txt" <<EOF
# 다른 줄
*/5 * * * * cd $R && echo 다른 일
2 * * * * cd $R && python scripts/live_regime_band_cron_20261102.py >> logs/regime.log 2>&1
31 0,8,16 * * * cd $R && python scripts/live_eth_direction_forward_recorder_20261010.py --record >> logs/dirfwd.log 2>&1
@reboot sleep 90 && cd $R && bash scripts/ops/supervisor_sr_lean.sh >> logs/supervisor/sr_lean_reboot.log 2>&1 &
EOF
cp "$F/crontab.txt" "$F/crontab.orig"
# 가짜 파이썬(워커 자리 · cmdline 에 -u 경로가 남는 bash 고리)
printf '#!/usr/bin/env bash\nwhile :; do sleep 1; done\n' > "$F/fakepy"
# 가짜 rt_sr_lean 은 #!/bin/bash -- comm 이 실행 파일 이름(rt_sr_lean)으로 남아야 진짜 Rust 실행 파일처럼 보인다(env 를 거치면 bash 가 된다)
# 가짜 rt_regime: 파이썬이 그 결정 시각에 쓴 줄을 그대로(FAKE_BAD=1 이면 p_trend 를 바꿔) 이어 쓰고 요약을 쓴다
cat > "$R/bin/rt_regime" <<'EOF'
#!/usr/bin/env bash
O="$RT_REGIME_OUT"; P="$(dirname "$0")/../data/live"
T=$(tail -1 "$P/regime_band_forward.jsonl" | sed 's/.*"ts": "\([^"]*\)".*/\1/')
if [ "$O" != "data/live" ]; then
  grep -F "\"ts\": \"$T\"" "$P/regime_band_forward.jsonl" | { if [ "${FAKE_BAD:-}" = 1 ]; then sed 's/"p_trend": 0.4/"p_trend": 0.41/'; else cat; fi; } >> "$O/regime_band_forward.jsonl"
fi
cp "$P/regime_band_forward_summary.json" "$O/regime_band_forward_summary.json.tmp" 2>/dev/null; mv "$O/regime_band_forward_summary.json.tmp" "$O/regime_band_forward_summary.json"
echo '{"recorded": 0}'
EOF
cat > "$R/bin/rt_sr_lean" <<'EOF'
#!/bin/bash
while :; do date +%s.%N > "$RT_SR_OUT/ethusdt_state.json"; sleep 1; done
EOF
chmod +x "$F/crontab" "$F/fakepy" "$R/bin/"*
echo '{}' > "$L/rtmodel/sr_lean.json"; echo '{}' > "$L/rtmodel/regime_band.json"; touch -d '2 days ago' "$L/regime_band_h48_model.joblib"
NOW=$(( $(date +%s) / 3600 * 3600 + 1800 ))                                      # 정시 30분(회차 시각 아님)
T=$(date -u -d "@$(( NOW - 1800 ))" '+%Y-%m-%d %H:%M:%S')
for c in BTC ETH SOL XRP; do echo "{\"ts\": \"2026-01-01 00:00:00\", \"coin\": \"$c\", \"p_trend\": 0.4, \"regime\": \"횡보\"}"; done > "$L/regime_band_forward.jsonl"
for c in BTC ETH SOL XRP; do echo "{\"ts\": \"$T\", \"coin\": \"$c\", \"p_trend\": 0.4, \"regime\": \"횡보\"}"; done >> "$L/regime_band_forward.jsonl"
echo '{"n_scored": 3}' > "$L/regime_band_forward_summary.json"; touch -d '1 hour ago' "$L/regime_band_forward_summary.json"
export ROOT="$R" CRONTAB_CMD="$F/crontab" SHADOW_BASE="$F/shadow" PY=python3 PYTHON_BIN="$F/fakepy" NOW_S="$NOW" WATCH_S=8
TK="bash $R/rtmodel/tools/takeover.sh"

echo "1. rt_regime check(어긋남) → go 거부"
out=$(FAKE_BAD=1 $TK rt_regime check 2>&1); echo "$out"; ok "어긋나면 check 종료코드 ≠ 0 · check.ok 없음" '[ ! -f "$F/shadow/rt_regime/check.ok" ]'
ok "어긋난 칸의 예(파이썬 값 대 Rust 값)가 출력에" '[[ "$out" == *"예: "*"p_trend: 파이썬 0.4 대 Rust 0.41"* ]]'
$TK rt_regime go 2>&1 | tail -1; ok "check.ok 없으면 go 거부 · crontab 그대로" 'cmp -s "$F/crontab.txt" "$F/crontab.orig"'
echo "2. rt_regime check(같음) → go → back"
$TK rt_regime check; ok "같으면 check.ok" '[ -f "$F/shadow/rt_regime/check.ok" ]'
$TK rt_regime go; ok "go: 파이썬 줄 주석 1 · Rust 줄 1 · 줄 수 +1" '[ "$(grep -c "^#rt-takeover:rt_regime# 2 \* \* \* \* cd" "$F/crontab.txt")" = 1 ] && [ "$(grep -c "rt_regime_" "$F/crontab.txt")" = 1 ] && [ $(wc -l < "$F/crontab.txt") = $(( $(wc -l < "$F/crontab.orig") + 1 )) ]'
ok "Rust 줄 일정 = 파이썬 줄 일정(2 * * * *) · 달러·퍼센트 이스케이프" 'grep -q "^2 \* \* \* \* cd .*RT_REGIME_TAKEOVER=1.*\$(date +\\\\%Y\\\\%m\\\\%d).*# rt-takeover:rt_regime$" "$F/crontab.txt"'
$TK rt_regime go 2>&1 | tail -1; ok "두 번 go 거부(이미 넘김 · 태그 줄 그대로 2)" '[ "$(grep -c "rt-takeover:rt_regime" "$F/crontab.txt")" = 2 ]'
$TK rt_regime back; ok "back: crontab 이 원래와 글자까지 같음" 'cmp -s "$F/crontab.txt" "$F/crontab.orig"'
echo "3. rt_dirfwd go: 회차 시각(08:33)이면 거부"
T8=$(( NOW / 86400 * 86400 + 8 * 3600 + 33 * 60 )); mkdir -p "$F/shadow/rt_dirfwd"; echo "$T8 칸 1" > "$F/shadow/rt_dirfwd/check.ok"
NOW_S=$T8 $TK rt_dirfwd go 2>&1 | tail -1
ok "회차 시각이면 crontab 그대로" 'cmp -s "$F/crontab.txt" "$F/crontab.orig"'
echo "4. rt_sr_lean go(파이썬 감독·워커 끔 → Rust 감독·워커) → back"
( cd "$R" && setsid nohup bash scripts/ops/supervisor_sr_lean.sh > /dev/null 2>&1 < /dev/null & )
pyw() { for p in /proc/[0-9]*; do c=$(tr '\0' ' ' 2>/dev/null < "$p/cmdline"); [[ "$c" == *" -u $R/scripts/live_sr_lean_20261118.py"* ]] && echo "${p#/proc/}"; done; }
pys() { for p in /proc/[0-9]*; do c=$(tr '\0' ' ' 2>/dev/null < "$p/cmdline"); [[ "$c" == *"_supervise.sh"*"$L/.supervisor_sr_lean.lock"* ]] && echo "${p#/proc/}"; done; }
rsw() { for p in /proc/[0-9]*; do [ "$(cat "$p/comm" 2>/dev/null)" = rt_sr_lean ] && tr '\0' '\n' 2>/dev/null < "$p/environ" | grep -qx "RT_SR_OUT=data/live/sr_lean" && echo "${p#/proc/}"; done; }
upw() { for i in $(seq 1 20); do [ -n "$(pys)" ] && [ -n "$(pyw)" ] && return 0; sleep 0.5; done; return 1; }
upw; ok "가짜 파이썬 감독·워커가 떠 있음" '[ -n "$(pys)" ] && [ -n "$(pyw)" ]'
mkdir -p "$F/shadow/rt_sr_lean"; echo "$NOW 칸 9" > "$F/shadow/rt_sr_lean/check.ok"
$TK rt_sr_lean go
ok "go: 파이썬 감독·워커 없음" '[ -z "$(pys)" ] && [ -z "$(pyw)" ]'
ok "go: Rust 워커(RT_SR_OUT=data/live/sr_lean) 있음 · @reboot 줄 바뀜" '[ -n "$(rsw)" ] && grep -q "^#rt-takeover:rt_sr_lean# @reboot" "$F/crontab.txt" && grep -q "^@reboot cd .*supervisor_rt_sr_lean.sh.*# rt-takeover:rt_sr_lean$" "$F/crontab.txt"'
$TK rt_sr_lean back; upw
ok "back: Rust 없음 · 파이썬 감독·워커 다시 · crontab 원래대로" '[ -z "$(rsw)" ] && [ -n "$(pys)" ] && [ -n "$(pyw)" ] && cmp -s "$F/crontab.txt" "$F/crontab.orig"'
echo "5. rt_dirpred go = 거부(파이썬 몫이 대시보드 서버 안)"
$TK rt_dirpred go 2>&1 | tail -1; rc=${PIPESTATUS[0]}
ok "종료코드 3 · crontab 그대로" '[ "$rc" = 3 ] && cmp -s "$F/crontab.txt" "$F/crontab.orig"'
echo "6. export-if-stale: joblib 이 JSON 보다 새것이면 내보내기 시도"
sleep 1; touch "$L/regime_band_h48_model.joblib"; out=$(PY=/bin/false $TK all export-if-stale 2>&1); echo "$out"
ok "낡음 감지 → 내보내기 시도 → 실패 문구(옛 값을 내지 않음)" '[[ "$out" == *"[rt_regime] 모델 JSON 낡음"* ]] && [[ "$out" == *"[rt_regime] 🔴내보내기 실패"* ]]'
echo "== 통과 $PASS · 실패 $FAIL"
[ "$FAIL" = 0 ]
