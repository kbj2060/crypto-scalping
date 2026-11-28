#!/usr/bin/env bash
# rtorder_cutover.sh back 이 8787(파이썬 대시보드)이 듣지 않으면 표식을 지우지 않고 거부하는지 (2026-11-28) + 전환 표식이면 조회를 8790 으로.
# 가짜 ROOT 에 스크립트를 복사하고 ss·pgrep·crontab·curl 을 가짜로 -- 프로세스·거래소·crontab 을 건드리지 않는다.
# 실행: bash test/test_rtorder_cutover_back_guard_20261128.sh
set -uo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/ops/rtorder_cutover.sh"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
fail=0
mkdir -p "$TMP/r/scripts/ops" "$TMP/r/data/live" "$TMP/bin"; cp "$SRC" "$TMP/r/scripts/ops/"
cat > "$TMP/bin/ss" <<'EOF'
#!/bin/sh
[ -n "${FAKE_LISTEN:-}" ] && echo "LISTEN 0 128 127.0.0.1:$FAKE_LISTEN 0.0.0.0:*"
exit 0
EOF
printf '#!/bin/sh\nexit 1\n' > "$TMP/bin/pgrep"; printf '#!/bin/sh\nexit 0\n' > "$TMP/bin/crontab"
printf '#!/bin/sh\necho "$@" >> %s/curl.log\necho "{}"\n' "$TMP" > "$TMP/bin/curl"
chmod +x "$TMP/bin/"*
run() { PATH="$TMP/bin:$PATH" CURL="$TMP/bin/curl" FAKE_LISTEN="$1" bash "$TMP/r/scripts/ops/rtorder_cutover.sh" "$2" 2>&1; }
check() { if eval "$2"; then echo "    OK  $1"; else echo "    🔴 $1"; fail=1; fi; }

echo live > "$TMP/r/data/live/rtorder_live.on"
echo "■ back · 8787 안 들림"
out="$(run 8790 back)"; rc=$?
check "거부(종료 1)" '[ "$rc" = 1 ]'
check "문구에 «먼저 파이썬 대시보드를 살린»" 'grep -q "먼저 파이썬 대시보드를 살린" <<<"$out"'
check "표식 그대로 live" '[ "$(cat "$TMP/r/data/live/rtorder_live.on")" = live ]'

echo "■ status · 전환 표식 없음 → 8787 에 묻는다(예전 그대로)"
rm -f "$TMP/r/data/live/rtorder_live.on" "$TMP/curl.log"; run 8787 status >/dev/null
check "manual-entry/status 가 8787" 'grep -q "127.0.0.1:8787/api/manual-entry/status" "$TMP/curl.log" && ! grep -q 8790 "$TMP/curl.log"'
echo "■ status · 전환 표식 있음 → 8790"
echo live > "$TMP/r/data/live/rtserve_live.on"; rm -f "$TMP/curl.log"; run 8790 status >/dev/null
check "manual-entry/status 가 8790" 'grep -q "127.0.0.1:8790/api/manual-entry/status" "$TMP/curl.log" && ! grep -q 8787 "$TMP/curl.log"'

echo; [ "$fail" = 0 ] && echo "통과 5/5" || echo "🔴 실패"; exit "$fail"
