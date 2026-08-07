#!/usr/bin/env bash
# One-shot: reinstall the 4 unit files (adds crash-loop backoff to the existing
# 3, adds the new btc-multislot-shadow unit) and cut the BTC multislot shadow
# loop over from bash supervision (_supervise.sh) to systemd. Run once with sudo:
#   sudo bash scripts/ops/systemd/update_units_20260807.sh
#
# trading-bot.service and tau1-shadow.service are NOT restarted -- daemon-reload
# alone is enough for systemd to apply the new Restart=/RestartSteps= behavior
# on their next crash-restart, with zero interruption to the running decision
# loop. ops-watchdog.service IS restarted (it needs the actual Python code
# change -- the Telegram retry-on-failure fix -- to take effect).
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "must run with sudo (unit install requires root): sudo bash $0" >&2
  exit 1
fi

ROOT="/home/llewyn/crypto-scalping"
UNIT_DIR="$ROOT/scripts/ops/systemd"
RUN_AS="llewyn"

for f in trading-bot.service tau1-shadow.service ops-watchdog.service btc-multislot-shadow.service; do
  install -m 0644 "$UNIT_DIR/$f" "/etc/systemd/system/$f"
  echo "installed /etc/systemd/system/$f"
done

systemctl daemon-reload
echo "--- daemon-reload done (trading-bot/tau1-shadow pick up new backoff config on their next restart, no interruption now) ---"

echo "--- stopping bash-supervised btc_multislot_shadow ---"
sudo -u "$RUN_AS" bash -lc "pkill -f '[s]cripts/ops/_supervise.sh btc_multislot_shadow ' || true; sleep 1; pkill -f run_btc_multislot_shadow_loop_20260807.py || true"
sleep 2

echo "--- enabling + starting btc-multislot-shadow.service ---"
systemctl enable --now btc-multislot-shadow.service

echo "--- restarting ops-watchdog.service (Telegram retry fix) ---"
systemctl restart ops-watchdog.service

sleep 3

echo "--- status ---"
systemctl --no-pager status trading-bot.service tau1-shadow.service ops-watchdog.service btc-multislot-shadow.service
