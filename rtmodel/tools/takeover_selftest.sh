#!/usr/bin/env bash
# takeover.sh 자체 시험 (2026-11-21) -- 가짜 저장소·가짜 crontab·가짜 실행 파일·가짜 파이썬 워커(진짜 프로세스 · 가짜 저장소 경로)로 check·go·back.
# 진짜 crontab·진짜 서버 프로세스는 건드리지 않는다(CRONTAB_CMD = 가짜 · 프로세스는 가짜 저장소 경로가 cmdline 에 있는 것만 끈다).
#   bash rtmodel/tools/takeover_selftest.sh
set -uo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
F=$(mktemp -d); R="$F/repo"; L="$R/data/live"
mkdir -p "$R/rtmodel/tools" "$R/scripts/ops" "$R/tmp/rtbin" "$L/rtmodel" "$L/sr_lean" "$R/logs/supervisor" "$R/dashboard"
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
cat > "$R/tmp/rtbin/rt_regime" <<'EOF'
#!/usr/bin/env bash
O="$RT_REGIME_OUT"; P="$(dirname "$0")/../../data/live"
T=$(tail -1 "$P/regime_band_forward.jsonl" | sed 's/.*"ts": "\([^"]*\)".*/\1/')
if [ "$O" != "data/live" ]; then
  grep -F "\"ts\": \"$T\"" "$P/regime_band_forward.jsonl" | { if [ "${FAKE_BAD:-}" = 1 ]; then sed 's/"p_trend": 0.4/"p_trend": 0.41/'; else cat; fi; } >> "$O/regime_band_forward.jsonl"
fi
cp "$P/regime_band_forward_summary.json" "$O/regime_band_forward_summary.json.tmp" 2>/dev/null; mv "$O/regime_band_forward_summary.json.tmp" "$O/regime_band_forward_summary.json"
echo '{"recorded": 0}'
EOF
cat > "$R/tmp/rtbin/rt_sr_lean" <<'EOF'
#!/bin/bash
while :; do date +%s.%N > "$RT_SR_OUT/ethusdt_state.json"; sleep 1; done
EOF
chmod +x "$F/crontab" "$F/fakepy" "$R/tmp/rtbin/"*
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
echo "5. rt_dirpred go·back(표식 파일 · 대시보드 스위치)"
cp "$SRC/scripts/ops/supervisor_rt_dirpred.sh" "$R/scripts/ops/"; RB="$R/tmp/rtbin"
printf '%s\n' '#!/bin/bash' 'while :; do [ -n "$FAKE_SLOW" ] || { echo "{\"t\": 1}" > "$RT_DIR_OUT/xsec_latest.json.tmp"; mv "$RT_DIR_OUT/xsec_latest.json.tmp" "$RT_DIR_OUT/xsec_latest.json"; }; sleep 1; done' > "$RB/rt_dirpred"
chmod +x "$RB/rt_dirpred"; echo '{}' > "$L/rtmodel/dir4h_eth.json"; echo '{}' > "$L/rtmodel/ds_eth.json"
mkdir -p "$F/shadow/rt_dirpred"; echo "$NOW 칸 9" > "$F/shadow/rt_dirpred/check.ok"
$TK rt_dirpred go 2>&1 | tail -1
ok "대시보드에 스위치가 없으면 거부 · 표식 없음 · crontab 그대로" '[ ! -f "$L/rtmodel/pred_record_rust.on" ] && cmp -s "$F/crontab.txt" "$F/crontab.orig"'
echo 'RUST_ON = ROOT / "data" / "live" / "rtmodel" / "pred_record_rust.on"' > "$R/dashboard/pred_record.py"
NOW_S=$(( NOW + 290 )) $TK rt_dirpred go 2>&1 | tail -1
ok "회차 경계 근처(290초)면 거부" '[ ! -f "$L/rtmodel/pred_record_rust.on" ]'
rdp() { for p in /proc/[0-9]*; do [ "$(cat "$p/comm" 2>/dev/null)" = rt_dirpred ] && tr '\0' '\n' 2>/dev/null < "$p/environ" | grep -qx "RT_DIR_OUT=data/live" && echo "${p#/proc/}"; done; }
NOW_S=$(( NOW + 120 )) DIRPRED_WATCH_S=8 $TK rt_dirpred go
ok "go: 표식 · Rust 워커(RT_DIR_OUT=data/live) · xsec_latest.json 갱신 · @reboot 줄 +1" '[ -f "$L/rtmodel/pred_record_rust.on" ] && [ -n "$(rdp)" ] && [ -f "$L/xsec_latest.json" ] && grep -q "^@reboot sleep 120 && cd .*supervisor_rt_dirpred.sh.*# rt-takeover:rt_dirpred$" "$F/crontab.txt" && [ $(wc -l < "$F/crontab.txt") = $(( $(wc -l < "$F/crontab.orig") + 1 )) ]'
$TK rt_dirpred back
ok "back: Rust 없음 · 표식 없음 · crontab 원래대로" '[ -z "$(rdp)" ] && [ ! -f "$L/rtmodel/pred_record_rust.on" ] && cmp -s "$F/crontab.txt" "$F/crontab.orig"'
out=$(FAKE_SLOW=1 NOW_S=$(( NOW + 120 )) DIRPRED_WATCH_S=3 $TK rt_dirpred go 2>&1); rc=$?; echo "$out" | tail -1 | cut -c1-160
ok "첫 회차가 한도를 넘겨도 워커가 살아 있으면 «아직 첫 회차 중»(실패 아님 · 11-22 서버 오경보)" '[ "$rc" = 0 ] && [[ "$out" == *"아직 첫 회차 중"* ]]'
$TK rt_dirpred back > /dev/null 2>&1
echo "6. export-if-stale: joblib 이 JSON 보다 새것이면 내보내기 시도"
sleep 1; touch "$L/regime_band_h48_model.joblib"; out=$(PY=/bin/false $TK all export-if-stale 2>&1); echo "$out"
ok "낡음 감지 → 내보내기 시도 → 실패 문구(옛 값을 내지 않음)" '[[ "$out" == *"[rt_regime] 모델 JSON 낡음"* ]] && [[ "$out" == *"[rt_regime] 🔴내보내기 실패"* ]]'
echo "7. rt_radar 대조: 가격 다른 코인은 마지막 봉 칸만 빼고, 가격 같은 코인은 전부"
printf '%s' '{"updated": "a", "params": "p", "coins": {"ETH": {"ts": "t", "price": 1686.55, "band80_pct": 0.9077, "iv60": true}, "BTC": {"ts": "t", "price": 9.0, "band80_pct": 0.5}}}' > "$F/pa.json"
printf '%s' '{"updated": "b", "params": "p", "coins": {"ETH": {"ts": "t", "price": 1686.59, "band80_pct": 0.9074, "iv60": true}, "BTC": {"ts": "t", "price": 9.0, "band80_pct": 0.5}}}' > "$F/pb.json"
out=$(python3 "$R/rtmodel/tools/takeover_cmp.py" radar "$F/pa.json" "$F/pb.json" --skip updated --rtol 1e-12); rc=$?; echo "$out"
ok "ETH 가격만 다름 → 통과 · 뺀 칸·이유 출력" '[ "$rc" = 0 ] && [[ "$out" == *"가격 다른 코인 ETH(파이썬 1686.55 대 Rust 1686.59)"*"뺌"* ]]'
sed -i 's/"band80_pct": 0.5}}}/"band80_pct": 0.51}}}/' "$F/pb.json"
out=$(python3 "$R/rtmodel/tools/takeover_cmp.py" radar "$F/pa.json" "$F/pb.json" --skip updated --rtol 1e-12); rc=$?
ok "가격 같은 코인(BTC)의 칸이 다르면 불통과 · 예 출력" '[ "$rc" = 1 ] && [[ "$out" == *"BTC.band80_pct: 파이썬 0.5 대 Rust 0.51"* ]]'
sed -i 's/"price": 9.0, "band80_pct": 0.51/"price": 9.1, "band80_pct": 0.51/' "$F/pb.json"
out=$(python3 "$R/rtmodel/tools/takeover_cmp.py" radar "$F/pa.json" "$F/pb.json" --skip updated --rtol 1e-12); rc=$?
ok "모든 코인 가격이 다르면 불통과" '[ "$rc" = 1 ] && [[ "$out" == *"모든 코인 가격이 다름"* ]]'
echo "8. NUL 바이트가 낀 기록(서버 pred_record 실측) -- 대조기는 깨진 줄을 원본 _tail 처럼 건너뛴다"
mkdir -p "$F/nul"; printf '{"k":"4","t":5,"px":1.0}\n\0\0\0{"k":"4","t":9,"px":1.0}\n{"k":"s","t":7,"a":1}\n' > "$F/nul/BTC.jsonl"; printf '{"k":"4","t":6}\n' > "$F/nul/ETH.jsonl"
printf '{"k":"4","t":5,"px":1.0}\n' > "$F/nul/rs.jsonl"
mt=$(python3 "$R/rtmodel/tools/takeover_cmp.py" maxt "$F/nul" 4); out=$(python3 "$R/rtmodel/tools/takeover_cmp.py" lines "$F/nul/BTC.jsonl" "$F/nul/rs.jsonl" --key k,t --text); rc=$?; echo "$out"
ok "maxt = NUL 줄을 건너뛴 가장 큰 4h t(6) · lines 통과 + 깨진 줄 수를 찍음" '[ "$mt" = 6 ] && [ "$rc" = 0 ] && [[ "$out" == *"깨진 줄"*"파이썬 1"* ]]'
echo "9. 여러 줄 check.ok(11-22 서버: 깨진 줄 알림이 앞줄에 붙어 go 가 산술 오류로 죽었다)"
printf '%s\n%s\n' "$NOW  깨진 줄(NUL 등) 파이썬 1 · Rust 0" "칸 65 · 어긋남 0 · 짝 없음 0" > "$F/shadow/rt_regime/check.ok"
$TK rt_regime go > "$F/go9.txt" 2>&1; rc=$?; tail -1 "$F/go9.txt"
ok "옛 여러 줄 check.ok 도 첫 줄 epoch 로 읽어 go 진행(산술 오류 없음)" '[ "$rc" = 0 ] && ! grep -q "syntax error\|산술\|arithmetic" "$F/go9.txt" && grep -q "rt-takeover:rt_regime" "$F/crontab.txt"'
$TK rt_regime back > /dev/null 2>&1
mkdir -p "$F/fake9"; printf '#!/bin/bash\nprintf "  깨진 줄 파이썬 1 · Rust 0\\n칸 3 · 어긋남 0 · 짝 없음 0\\n"\n' > "$F/fake9/cmp.sh"
( SH="$F/shadow/x9" NAME=x CMD=t; source <(sed -n '/^now_s()/p;/^say()/p;/^record_check() {/,/^}/p' "$R/rtmodel/tools/takeover.sh"); out=$(bash "$F/fake9/cmp.sh"); record_check 0 "기록 $out" > /dev/null )
ok "record_check 는 여러 줄 대조 출력을 한 줄로 펴서 check.ok 에 쓴다" '[ "$(wc -l < "$F/shadow/x9/check.ok")" = 1 ] && grep -q "깨진 줄 파이썬 1 · Rust 0 칸 3 · 어긋남 0" "$F/shadow/x9/check.ok"'
echo "10. rt_sr_lean check 둘째 호출(65분 뒤): 대조기 종료코드를 그대로 판정에(전엔 | tail -1 의 종료코드라 불일치도 통과로 적혔다)"
SR="$F/shadow/rt_sr_lean"; mkdir -p "$SR/run"; echo $(( NOW - 4000 )) > "$SR/started"
( cd "$R" && RT_SR_OUT="$SR/run" setsid nohup "$RB/rt_sr_lean" > /dev/null 2>&1 < /dev/null & )
for i in $(seq 1 20); do [ -n "$(for p in /proc/[0-9]*; do [ "$(cat "$p/comm" 2>/dev/null)" = rt_sr_lean ] && tr '\0' '\n' 2>/dev/null < "$p/environ" | grep -qx "RT_SR_OUT=$SR/run" && echo x; done)" ] && break; sleep 0.5; done
printf '%s\n' 'import os, sys' 'print("사건 id: 파이썬 10 · Rust 10 · 양쪽 10")' 'print("양쪽 다 있는 id 의 줄: ev 10 · 결과 9 · 피쳐 10 → 다른 줄", 0 if os.environ.get("FAKE_SR_OK") else 1)' 'sys.exit(0 if os.environ.get("FAKE_SR_OK") else 1)' > "$R/rtmodel/tools/shadow_sr_lean.py"
out=$($TK rt_sr_lean check 2>&1); echo "$out" | tail -1 | cut -c1-200
ok "다른 줄 1 → 불통과 · check.ok 없음 · 출력에 사건 수" '[ ! -f "$SR/check.ok" ] && [[ "$out" == *"불통과"*"다른 줄 1"* ]]'
out=$(FAKE_SR_OK=1 $TK rt_sr_lean check 2>&1); echo "$out" | tail -1 | cut -c1-200
ok "다른 줄 0 → 통과 · check.ok 한 줄 · 견줌 시작 시각 출력" '[ -f "$SR/check.ok" ] && [ "$(wc -l < "$SR/check.ok")" = 1 ] && [[ "$out" == *"통과"*"견줌 시작"*"다른 줄 0"* ]]'
echo "11. 레벨 5분 대조기(shadow_sr_lean.py): 실수 끝자리(≤ 1e-12)는 같음 · 1e-9 · 판정 칸 p · 결과 줄은 다름(양성 대조)"
sr11() {   # $1 = Rust 쪽 s1h · $2 = Rust 쪽 p · $3 = Rust 쪽 결과 y → 종료코드
  local d="$F/sr11"; rm -rf "$d"; mkdir -p "$d/py" "$d/rs"
  printf '%s\n%s\n' '{"k": "ev", "id": "x1", "a": 100, "s1h": 0.012392114619389981, "p": 0.49054, "grp": "mid"}' '{"k": "res", "id": "x1", "y": 1}' > "$d/py/ethusdt.jsonl"
  printf '%s\n%s\n' "{\"k\": \"ev\", \"id\": \"x1\", \"a\": 100, \"s1h\": $1, \"p\": $2, \"grp\": \"mid\"}" "{\"k\": \"res\", \"id\": \"x1\", \"y\": $3}" > "$d/rs/ethusdt.jsonl"
  python3 "$SRC/rtmodel/tools/shadow_sr_lean.py" "$d/py" "$d/rs" > "$d/out.txt" 2>&1
}
sr11 0.012392114619389995 0.49054 1; r1=$?; sr11 0.012392114632 0.49054 1; r2=$?; sr11 0.012392114619389981 0.49055 1; r3=$?; sr11 0.012392114619389981 0.49054 0; r4=$?
echo "  종료코드: 끝자리 1.1e-15 → $r1 · 1e-9 → $r2 · p 다름 → $r3 · 결과 다름 → $r4"
ok "끝자리만 = 0(같음) · 1e-9 = 1 · p 다름 = 1 · 결과 다름 = 1" '[ "$r1" = 0 ] && [ "$r2" = 1 ] && [ "$r3" = 1 ] && [ "$r4" = 1 ]'
echo "== 통과 $PASS · 실패 $FAIL"
[ "$FAIL" = 0 ]
