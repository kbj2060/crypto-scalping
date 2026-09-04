#!/usr/bin/env bash
# A4 캡 -- trading-bot.service 재시작 후 건강 확인. .env 적용(a4_env_apply_20260904.sh) 뒤에만 실행.
# 코드(포트폴리오 캡 텔레메트리 커밋)가 main에 머지돼 있으면 deploy_watcher가 어차피 재시작하므로 이 스크립트는
# 불필요하다 -- .env만 바꾸고 코드 머지 없이 활성화할 때 쓴다.
#   bash scripts/ops/handoff.sh launch server a4_restart --sync scripts/ops/a4_restart_verify_20260904.sh \
#     -- bash /home/llewyn/crypto-scalping/scripts/ops/a4_restart_verify_20260904.sh
set -u
cd /home/llewyn/crypto-scalping
python -c "import ast; ast.parse(open('trading_bot.py').read())" || { echo "trading_bot.py does NOT parse -- aborting restart"; exit 1; }
git status --short | grep -E '^(UU|AA)' && { echo "unmerged files present -- aborting restart"; exit 1; }
echo "--- restart (sudoers: scripts/ops/systemd/deploy_watcher_sudoers) ---"
sudo -n /usr/bin/systemctl restart trading-bot.service || { echo "sudo restart failed"; exit 1; }
sleep 45
echo "--- state ---"; systemctl is-active trading-bot.service
echo "--- startup lines (portfolio_cap line appears only once the 09-04 code is deployed) ---"
journalctl -u trading-bot.service --since "-2min" --no-pager 2>/dev/null | grep -E "SYSTEM (portfolio_cap|binance_execution|omega4_6_1=|pending_next_open)|Traceback|Error" | tail -n 12
echo "--- other units / dashboard after the 09-03 21:26 reboot ---"
for u in ops-watchdog tau1-shadow btc-multislot-shadow eth-jmlam4-shadow eth-odyssey4-shadow eth-exithead-shadow prometheus-exporter oi-lsratio-worker tail-risk-btc-sol-worker; do
  printf '%-26s %s\n' "$u" "$(systemctl is-active ${u}.service 2>/dev/null)"; done
ss -ltn 2>/dev/null | grep -q ':8787 ' && echo "dashboard: listening on 8787" || echo "dashboard: NOT listening on 8787"
echo DONE_RESTART_VERIFY
