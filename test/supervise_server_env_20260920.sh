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


# ④⑤ supervisor 중복 방지(flock). pid 파일로는 못 막는다 -- start_external.sh 가 그걸 지우고,
#     재기동 3초 창에는 curl 도 실패해서 두 «이미 도는가» 검사가 둘 다 통과한다.
#     2026-09-20 실측: 그렇게 뜬 두 번째 supervisor 가 이틀을 살았다(pid 759).
T2="$(mktemp -d)"; trap 'rm -rf "$T" "$T2"' EXIT
mkdir -p "$T2/dashboard/scripts"
cp "$SRC" "$T2/dashboard/scripts/supervise_server.sh"
: > "$T2/.env"
printf '#!/usr/bin/env bash\nsleep 20\n' > "$T2/stub2.sh"; chmod +x "$T2/stub2.sh"
ENVV=(PYTHON_BIN="$T2/stub2.sh" DASHBOARD_HOST=127.0.0.1 DASHBOARD_PORT=48788 DASHBOARD_RESTART_DELAY=1)
env "${ENVV[@]}" bash "$T2/dashboard/scripts/supervise_server.sh" >/dev/null 2>&1 &
A=$!
sleep 2
env "${ENVV[@]}" timeout 6 bash "$T2/dashboard/scripts/supervise_server.sh" >/dev/null 2>&1
B_RC=$?
ERRLOG="$T2/data/live/dashboard_external.err"
if [ "$B_RC" = 0 ] && grep -q "lock held" "$ERRLOG" 2>/dev/null && kill -0 "$A" 2>/dev/null; then
  echo "✅ 두 번째 supervisor 는 잠금에 막혀 즉시 종료(첫 번째는 그대로 산다)"
else
  echo "🔴 두 번째 supervisor 가 떴다 -- rc=$B_RC, A살아있음=$(kill -0 "$A" 2>/dev/null && echo y || echo n), lock로그=$(grep -c 'lock held' "$ERRLOG" 2>/dev/null)"; ok=0
fi

# ⑤ supervisor 만 죽이고 **자식은 고아로 남겨도** 잠금이 풀려야 한다. 자식이 FD 9 를 물고
#    가면(9>&- 누락) 후임 supervisor 가 영영 못 뜬다 -- _supervise.sh 가 같은 함정을 적어 뒀다.
CHILD="$(cat "$T2/data/live/dashboard_external.child.pid" 2>/dev/null)"
# 🔴SIGKILL 이어야 한다. 평범한 kill 은 supervisor 의 TERM 트랩(stop_child)을 돌려 자식까지
#   같이 죽이므로 «고아 자식» 상황이 아예 안 만들어진다. 그리고 실제로 워처가 보내는 것도
#   SIGKILL 이다(6초 안에 안 죽으면 승격 -- CLAUDE.md 에 기록된 그 동작).
kill -9 "$A" 2>/dev/null; wait "$A" 2>/dev/null
sleep 1
if [ -n "$CHILD" ] && kill -0 "$CHILD" 2>/dev/null; then
  if ( exec 9>"$T2/data/live/dashboard_external.lock"; flock -n 9 ) 2>/dev/null; then
    echo "✅ supervisor 가 죽으면 잠금이 풀린다(자식이 살아 있어도)"
  else
    echo "🔴 자식이 잠금을 물고 있다 -- 후임 supervisor 가 못 뜬다(9>&- 누락)"; ok=0
  fi
  kill "$CHILD" 2>/dev/null
else
  echo "🔴 ⑤를 못 쟀다 -- 고아 자식이 안 만들어졌다(child=[$CHILD])"; ok=0
fi

[ "$ok" = 1 ] && echo "✅ supervise_server.sh .env 계약 OK"
exit $((1 - ok))
