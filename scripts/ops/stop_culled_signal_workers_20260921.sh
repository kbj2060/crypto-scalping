#!/bin/bash
# 2026-09-21 제거된 신호 4종의 워커 중지. 순서: crontab -> supervisor -> 자식 (호메로스 규약).
set -u
PATS='live_eth_extreme_detector_worker_20260910|live_eth_vol_forecast_worker_20260910|live_evr_gate_worker_20260915|live_eth_sweep_v_rebound_signal_20260829'

echo "=== 1) crontab @reboot 줄 제거 ==="
crontab -l 2>/dev/null > /tmp/cron.before || true
cp /tmp/cron.before /home/llewyn/crypto-scalping/logs/crontab_before_signal_cull_20260921.txt 2>/dev/null || true
grep -vE 'supervisor_extreme_detector_worker|supervisor_vol_forecast_worker|supervisor_evr_gate_worker|supervisor_signal_worker.sh v_rebound' /tmp/cron.before > /tmp/cron.after
echo "  before $(wc -l < /tmp/cron.before)줄 -> after $(wc -l < /tmp/cron.after)줄"
crontab /tmp/cron.after && echo "  crontab 갱신 완료"

echo "=== 2) supervisor(bash) 먼저 ==="
for pid in $(pgrep -f "_supervise.sh" ); do
  cmd=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
  comm=$(cat /proc/$pid/comm 2>/dev/null)
  if echo "$cmd" | grep -qE "$PATS" && [ "$comm" = "bash" ]; then
    echo "  kill supervisor $pid ($comm)"; kill "$pid" 2>/dev/null
  fi
done
sleep 3

echo "=== 3) 자식(python) ==="
for pid in $(pgrep -f "$PATS"); do
  comm=$(cat /proc/$pid/comm 2>/dev/null)
  cmd=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
  case "$comm" in python*) echo "  kill child $pid ($comm)"; kill "$pid" 2>/dev/null ;; esac
done
sleep 3

echo "=== 4) 양쪽 다시 센다 ==="
left=$(pgrep -af "$PATS" | grep -v "stop_workers\|run.sh\|bash -lc" | wc -l)
echo "  남은 프로세스: $left"
pgrep -af "$PATS" | grep -v "stop_workers\|run.sh\|bash -lc" || echo "  (없음)"
echo "=== 5) 남은 crontab 확인 ==="
crontab -l | grep -cE 'extreme_detector|vol_forecast|evr_gate|v_rebound' || echo "  0 (깨끗)"
