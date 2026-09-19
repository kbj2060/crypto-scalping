#!/usr/bin/env bash
# 대시보드 사이징 상한 무시 스위치를 켜고/끄고 재기동한다 (2026-09-20 사용자 지시).
#
# 실행(dev 에서):
#   bash scripts/ops/handoff.sh launch server sizing_override --sync \
#     scripts/ops/dashboard_sizing_override_20260920.sh -- \
#     bash /home/llewyn/crypto-scalping/scripts/ops/dashboard_sizing_override_20260920.sh 18
#   끄기: 같은 명령에 인자 off
#
# 왜 스크립트인가: auto 모드 분류기가 서버 .env 편집을 막는다(a4_env_apply_20260904.sh 의
# 2026-09-04 기록과 같은 이유). 절차를 코드로 남겨 사용자가 한 번에 돌리게 한다.
#
# 왜 .env 인가: deploy_watcher.sh:62 가 `set -a; source "$ROOT/.env"` 를 한다. 거기 두면
# 워처가 대시보드를 재기동할 때도 값이 따라간다 -- 셸에서 export 만 하면 **다음 워처
# 사이클(최대 5분)에 조용히 꺼진다**(워처는 start_external.sh 를 제 환경으로 부른다).
#
# 🔴서버 실측(2026-09-20 02:3x)에서 확인한 함정 둘:
#   ① supervise_server.sh 가 **두 개** 돈다(449=진짜 부모, 759=2026-09-18 부터 «포트 사용 중»
#      대기 루프). 그리고 data/live/dashboard_external.pid 는 **759(가짜)** 를 가리킨다.
#      pid 파일을 믿고 죽이면 서빙은 그대로고, 759 를 살려두면 내가 포트를 놓는 3초 안에
#      그놈이 **환경변수 없이** 자식을 띄워 스위치가 안 켜진 채 올라온다.
#      ⇒ 포트 소유자의 **부모**를 찾아 죽이고, 그 밖의 supervise_server.sh 도 전부 죽인다.
#   ② start_external.sh 의 HOST 기본값이 **0.0.0.0** 이다(supervisor 는 127.0.0.1).
#      그냥 부르면 LAN 에 열린다 -- supervise_server.sh 의 2026-09-12 주석이 그 사고 기록이다.
#      ⇒ DASHBOARD_HOST 를 **명시**해서 부른다.
set -uo pipefail
cd /home/llewyn/crypto-scalping

VAL="${1:-18}"
KEY=DASHBOARD_SIZING_CAP_EQUITY_X
PORT="${DASHBOARD_PORT:-8787}"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"

# --- 0. 죽이기 전에 디스크가 파싱되는지 먼저 본다(충돌 마커 위에서 재시작 = 크래시 루프) ---
"$PY" -c 'import ast;ast.parse(open("dashboard/server.py").read())' || { echo "🔴 server.py 파싱 실패 -- 중단"; exit 1; }
grep -q 'SIZING_CAP_OVERRIDE_X' dashboard/server.py || { echo "🔴 스위치 코드가 이 배포본에 없다 -- 중단"; exit 1; }

# --- 1. .env ---
cp -p .env ".env.bak_pre_sizing_override_20260920" || { echo "🔴 백업 실패 -- 중단"; exit 1; }
sed -i "/^${KEY}=/d" .env
if [ "$VAL" != "off" ]; then
  [ -n "$(tail -c1 .env)" ] && echo >> .env
  echo "${KEY}=${VAL}" >> .env
fi
echo "--- .env: $(grep -c "^${KEY}=" .env) 줄 · $(grep "^${KEY}=" .env || echo '(없음 = 꺼짐)')"

# --- 2. 정지: 포트 소유자의 부모(진짜 supervisor)와, 그 밖의 supervisor 전부 ---
OWNER="$(ss -ltnp "sport = :$PORT" 2>/dev/null | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1)"
SUPS="$(pgrep -f 'supervise_server\.sh' || true)"
echo "--- 포트 소유자=$OWNER · supervisor 후보=[$(echo $SUPS)]"
for s in $SUPS; do [ "$(cat /proc/$s/comm 2>/dev/null)" = "bash" ] && kill "$s" 2>/dev/null || true; done
sleep 2
for s in $SUPS; do kill -0 "$s" 2>/dev/null && kill -9 "$s" 2>/dev/null || true; done
if [ -n "$OWNER" ] && kill -0 "$OWNER" 2>/dev/null; then
  kill "$OWNER" 2>/dev/null || true
  # dashboard/server.py 는 SIGTERM 에 행이 잦다(소켓은 놓는데 프로세스가 안 죽는다).
  for _ in 1 2 3 4 5 6; do kill -0 "$OWNER" 2>/dev/null || break; sleep 1; done
  kill -0 "$OWNER" 2>/dev/null && kill -9 "$OWNER" 2>/dev/null || true
fi
rm -f data/live/dashboard_external.pid data/live/dashboard_external.child.pid
sleep 1
echo "--- 정지 후: supervisor=[$(pgrep -f 'supervise_server\.sh' | tr '\n' ' ')] 포트=[$(ss -ltn "sport = :$PORT" | tail -n +2 | wc -l)]"

# --- 3. 기동 (HOST 명시 · .env 를 실어서) ---
set -a; . ./.env; set +a
DASHBOARD_HOST=127.0.0.1 DASHBOARD_PORT="$PORT" bash dashboard/scripts/start_external.sh || true

# --- 4. 검증: 프로세스가 아니라 **서빙**과 **실제 적용값**을 본다 ---
sleep 3
echo "--- HTTP=$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 "http://127.0.0.1:$PORT/dashboard/live/")"
echo "--- 바인드: $(ss -ltn "sport = :$PORT" | tail -n +2 | awk '{print $4}')"
echo "--- override_x(미리보기 API 실제값):"
curl -s --max-time 10 "http://127.0.0.1:$PORT/api/manual-entry/preview?side=LONG&pct=10" \
  | "$PY" -c 'import json,sys
try:
    d = json.load(sys.stdin)
except Exception as e:
    print("    파싱 실패:", e); raise SystemExit
cap = (d.get("cap") or {})
print("    ok=%s override_x=%s binding=%s cap_notional=%s"
      % (d.get("ok"), cap.get("override_x"), cap.get("binding"), cap.get("cap_notional_usdt")))'
