#!/usr/bin/env bash
# rtorder(실전송) 죽음 감시 (2026-11-22 · 주문 경로를 Rust 로 넘긴 날). 1분 cron.
#   * * * * * cd <저장소> && bash scripts/ops/rtorder_health.sh >> logs/supervisor/rtorder_health.log 2>&1 #rtorder-health
# 표식(data/live/rtorder_live.on)이 있는 동안 손절 감시는 rtorder 뿐이다. 감독(_supervise.sh)은 죽으면 15초 뒤 다시 띄우지만
# 반복해서 죽으면 300초를 쉬고, 그동안 아무도 모른다 -- 여기서 텔레그램으로 알리고, 감독자까지 없으면 다시 띄운다.
# 고장 = 프로세스 없음 · 문(127.0.0.1:8791 /internal/acct)이 3초 안에 답 없음 · 실전송이 아닌 모드로 떠 있음.
# 허브 가격 끊김은 고장으로 치지 않는다(rtorder 가 거래소 직접 조회로 잇고 스스로 알린다).
# ponytail: 1분 주기라 최대 1분 늦게 안다. 더 빨리 알아야 하면 rtorder 감독을 systemd 유닛(Restart=always + OnFailure 알림)으로 옮긴다.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 2
MARK="$ROOT/data/live/rtorder_live.on"
[ -f "$MARK" ] || exit 0   # 파이썬이 감시 중 -- 볼 것 없음
STATE="$ROOT/data/live/rtorder/.health_bad_since"; LAST="$ROOT/data/live/rtorder/.health_last_alert"
ALERT_EVERY="${RTORDER_HEALTH_ALERT_EVERY:-300}"
DOOR="${RTORDER_HEALTH_DOOR:-http://127.0.0.1:8791}"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi
send_telegram() {
  [[ -z "${TELEGRAM_BOT_TOKEN:-}" || -z "${TELEGRAM_CHAT_ID:-}" ]] && return 0
  curl -s -m 8 -X POST "${RTORDER_HEALTH_TG:-https://api.telegram.org}/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
    -d "chat_id=${TELEGRAM_CHAT_ID}" --data-urlencode "text=$1" >/dev/null 2>&1 || true
}

why=""
pid="$(pgrep -x rtorder | head -n 1)"
if [ -z "$pid" ]; then
  why="프로세스 없음"
elif [ "$(tr '\0' '\n' 2>/dev/null < "/proc/$pid/environ" | grep -c '^RTORDER_MODE=live$')" != 1 ]; then
  why="실전송이 아닌 모드로 떠 있음(표식은 $(tr -d '\n' < "$MARK"))"
elif ! curl -s -m 3 "$DOOR/internal/acct" | grep -q '"seq"'; then
  why="문(8791)이 3초 안에 답하지 않음"
fi

now=$(date +%s)
if [ -z "$why" ]; then
  if [ -f "$STATE" ]; then
    since=$(cat "$STATE" 2>/dev/null || echo "$now")
    echo "[$(date -Iseconds)] 회복 ($((now - since))초 만에)"
    send_telegram "✅ [RTORDER] 주문 프로세스 회복 -- 손절 감시 다시 돈다 (고장 $((now - since))초)"
    rm -f "$STATE" "$LAST"
  fi
  exit 0
fi

mkdir -p "$(dirname "$STATE")"
[ -f "$STATE" ] || echo "$now" > "$STATE"
since=$(cat "$STATE" 2>/dev/null || echo "$now")
echo "[$(date -Iseconds)] 🔴고장: $why (고장 $((now - since))초째)"
# 감독자까지 없으면 다시 띄운다(표식이 있으니 실전송 판이 맞다 · 감독 스크립트가 중복 실행을 스스로 막는다)
if [ -z "$pid" ] && ! pgrep -f '_supervise.sh rtorder' >/dev/null; then
  echo "[$(date -Iseconds)] 감독자도 없음 -- 실전송 감독을 다시 띄운다"
  setsid nohup bash "$ROOT/scripts/ops/supervisor_rtorder_live.sh" > /dev/null 2>> "$ROOT/logs/supervisor/rtorder_reboot.log" < /dev/null &
fi
last=$(cat "$LAST" 2>/dev/null || echo 0)
if [ $((now - last)) -ge "$ALERT_EVERY" ]; then
  echo "$now" > "$LAST"
  send_telegram "🔴 [RTORDER] 주문 프로세스 고장: $why -- 손절 감시가 멈춰 있다 (고장 $((now - since))초째). 되돌리기: bash scripts/ops/rtorder_cutover.sh back"
fi
exit 1
