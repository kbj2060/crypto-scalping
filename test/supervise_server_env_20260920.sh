#!/usr/bin/env bash
# supervise_server.sh 의 .env 로딩 계약 (2026-09-20).
#   ① .env 의 키가 **자식 프로세스까지** 간다 -- 이게 안 되면 재부팅한 대시보드가
#      DASHBOARD_MANUAL_EXEC_ENABLED · BINANCE_API_KEY 없이 올라온다.
#   ② 호출자가 명시한 값이 .env 를 **이긴다** -- source 는 무조건 대입이라, 안 막으면
#      .env 의 DASHBOARD_HOST=0.0.0.0 이 start_external.sh 의 127.0.0.1 을 덮어 LAN 에 열린다.
#   bash test/supervise_server_env_20260920.sh
set -uo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/dashboard/scripts/supervise_server.sh"
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
mkdir -p "$T/dashboard/scripts"
cp "$SRC" "$T/dashboard/scripts/supervise_server.sh"

# .env 는 «자식까지 가야 할 키»와 «호출자에게 져야 할 키»를 함께 담는다.
cat > "$T/.env" <<'ENV'
SUPERVISE_TEST_MARKER=from_env
DASHBOARD_HOST=0.0.0.0
ENV

# 진짜 server.py 대신, 받은 인자와 환경을 적고 끝나는 스텁.
cat > "$T/stub.sh" <<'STUB'
#!/usr/bin/env bash
{ echo "ARGV=$*"; echo "MARKER=${SUPERVISE_TEST_MARKER:-<없음>}"; } > "$OUT"
exit 0
STUB
chmod +x "$T/stub.sh"

OUT="$T/seen.txt" PYTHON_BIN="$T/stub.sh" DASHBOARD_HOST=127.0.0.1 \
  DASHBOARD_PORT=48787 DASHBOARD_RESTART_DELAY=1 \
  bash "$T/dashboard/scripts/supervise_server.sh" >/dev/null 2>&1 &
SUP=$!
for _ in $(seq 1 40); do [ -s "$T/seen.txt" ] && break; sleep 0.25; done
kill "$SUP" 2>/dev/null; wait "$SUP" 2>/dev/null
pkill -P "$SUP" 2>/dev/null

ok=1
if [ ! -s "$T/seen.txt" ]; then echo "🔴 스텁이 실행되지 않았다 (supervisor 가 자식을 못 띄움)"; exit 1; fi
cat "$T/seen.txt" | sed 's/^/  /'
grep -q "MARKER=from_env" "$T/seen.txt" \
  && echo "✅ .env 키가 자식까지 간다" \
  || { echo "🔴 .env 키가 자식에 없다 -- 재부팅한 대시보드가 반쪽으로 뜬다"; ok=0; }
grep -q -- "--host 127.0.0.1" "$T/seen.txt" \
  && echo "✅ 호출자의 DASHBOARD_HOST 가 .env(0.0.0.0)를 이긴다" \
  || { echo "🔴 .env 가 호출자를 덮었다 -- LAN 에 열린다"; ok=0; }

# ③ 두 스크립트의 HOST 기본값이 **같아야** 한다. start_external.sh 는 이 값을 명시해서
#    넘기므로, 갈리면 supervisor 쪽 안전 기본값이 그 경로에서 죽는다(2026-09-12 사고가
#    2026-09-20 까지 반쪽으로 남아 있던 이유).
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
a="$(grep -oE 'HOST="\$\{DASHBOARD_HOST:-[^}]*\}"' "$R/dashboard/scripts/supervise_server.sh" | head -1)"
b="$(grep -oE 'HOST="\$\{DASHBOARD_HOST:-[^}]*\}"' "$R/dashboard/scripts/start_external.sh" | head -1)"
if [ -n "$a" ] && [ "$a" = "$b" ]; then echo "✅ 두 스크립트의 HOST 기본값이 같다: $a"
else echo "🔴 HOST 기본값이 갈렸다 -- supervisor=[$a] start_external=[$b]"; ok=0; fi

[ "$ok" = 1 ] && echo "✅ supervise_server.sh .env 계약 OK"
exit $((1 - ok))
