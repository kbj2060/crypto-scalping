#!/usr/bin/env bash
# 전환 모드(scripts/ops/supervisor_rtserve_live.sh) 끝에서 끝까지 확인 -- 2026-11-28.
# 임시 폴더에 감독 스크립트 둘을 복사해 진짜 rtserve 실행 파일로 띄우고(바깥망 없는 이름공간에서 돌릴 것), 쓰기 주소 4 + 주문 POST 1 이
# 원본과 같은 data/live 파일(acct_notes.json · acct_wrong.json · push_subscriptions.json · order_requests.jsonl)을 쓰는지 본다.
# 글자 일치는 tests/http2_parity.rs(원본 정답지 · readonly 끔)가 이미 본다 -- 여기는 «스크립트 환경 → 실행 파일 → 같은 자리» 사슬.
# 양성 대조: 같은 요청을 병행 스크립트(supervisor_rtserve.sh)로 띄우면 503 넷 · 파일 0 이어야 한다 / 8787 이 듣고 있으면 전환 스크립트가 거절해야 한다.
# 쓰는 법: unshare -rn bash -c 'ip link set lo up; bash rtserve/tools/live_mode_check.sh rtserve/target/release/rtserve'
set -uo pipefail
BIN="$(realpath "$1")"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }

run() {  # $1 = 스크립트 이름 · $2 = 포트 → 응답 코드들을 $T/codes 에
  T="$(mktemp -d)"; mkdir -p "$T/scripts/ops" "$T/data/live" "$T/dashboard/live"
  cp "$SRC/scripts/ops/$1" "$SRC/scripts/ops/_supervise.sh" "$T/scripts/ops/"
  ( RTSERVE_BIN="$BIN" RTSERVE_ADDR="127.0.0.1:$2" RTSERVE_HUB="ws://127.0.0.1:9/rt" RTSERVE_RTORDER="127.0.0.1:9" bash "$T/scripts/ops/$1" & echo $! > "$T/pid" )
  for _ in $(seq 50); do curl -s -o /dev/null "127.0.0.1:$2/dashboard/live/" && break; sleep 0.2; done
  local u="127.0.0.1:$2" j='Content-Type: application/json'
  {
    curl -s -o /dev/null -w '%{http_code}\n' -H "$j" -d '{"date": "2026-11-20", "text": "메모"}' "$u/api/acct-notes"
    curl -s -o /dev/null -w '%{http_code}\n' -H "$j" -d '{"date": "2026-11-20", "text": "가짜 돌파", "tag": "가짜 돌파"}' "$u/api/acct-wrong"
    curl -s -o /dev/null -w '%{http_code}\n' -H "$j" -d '{"subscription": {"endpoint": "https://push.example.invalid/send/A", "keys": {"p256dh": "P1", "auth": "A1"}}, "label": "폰"}' "$u/api/push/subscribe"
    curl -s -o /dev/null -w '%{http_code}\n' -H "$j" -d '{"subscription": {"endpoint": "https://push.example.invalid/send/B"}}' "$u/api/push/unsubscribe"
    curl -s -o /dev/null -w '%{http_code}\n' -X POST "$u/api/manual-entry?side=long"
  } > "$T/codes"
  local p; p="$(cat "$T/pid")"; local c; c="$(pgrep -P "$p")"   # 이 시험이 띄운 감독과 그 자식만(다른 프로세스는 건드리지 않는다)
  kill "$p" $c 2>/dev/null; sleep 0.3
}

run supervisor_rtserve_live.sh 18790
L="$T/data/live"
check "전환: 쓰기 넷 200" '[ "$(head -4 "$T/codes" | tr "\n" " ")" = "200 200 200 200 " ]'
check "전환: 메모 파일" 'grep -q "메모" "$L/acct_notes.json"'
check "전환: 오답 파일" 'grep -q "가짜 돌파" "$L/acct_wrong.json"'
check "전환: 구독 파일" 'grep -q "send/A" "$L/push_subscriptions.json"'
check "전환: 접수 기록(rtorder 없음 = 502 줄)" 'grep -q "/api/manual-entry" "$L/order_requests.jsonl"'
check "전환: 감독 로그 rtserve_live_*" 'ls "$T"/logs/supervisor/rtserve_live_*.log >/dev/null 2>&1'
rm -rf "$T"

run supervisor_rtserve.sh 18791
check "병행(대조): 쓰기 넷 503" '[ "$(head -4 "$T/codes" | tr "\n" " ")" = "503 503 503 503 " ]'
check "병행(대조): data/live 비어 있음" '[ -z "$(ls -A "$T/data/live" | grep -v "^\.supervisor")" ]'
rm -rf "$T"

python3 -m http.server 8787 --bind 127.0.0.1 >/dev/null 2>&1 & HP=$!; sleep 0.5
T="$(mktemp -d)"; mkdir -p "$T/scripts/ops"; cp "$SRC/scripts/ops/supervisor_rtserve_live.sh" "$SRC/scripts/ops/_supervise.sh" "$T/scripts/ops/"
RTSERVE_BIN="$BIN" RTSERVE_ADDR="127.0.0.1:18792" bash "$T/scripts/ops/supervisor_rtserve_live.sh" 2>"$T/err"; rc=$?
check "8787 듣는 중 = 거절" '[ $rc -ne 0 ] && grep -q 8787 "$T/err"'
kill $HP; rm -rf "$T"
exit $fail
