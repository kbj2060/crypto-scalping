#!/usr/bin/env bash
# One-shot migration from the bash _supervise.sh stack to systemd. Run once with sudo:
#   sudo bash scripts/ops/systemd/install_and_cutover.sh
#
# What it does, in order:
#   1. Installs the 3 unit files to /etc/systemd/system/
#   2. systemctl daemon-reload
#   3. Stops the bash-managed supervisors + their child processes (botctl.sh stop)
#   4. Enables + starts the 3 systemd units
#   5. Prints status so you can eyeball it before walking away
#
# This causes a brief (a few seconds) gap in the trading_bot/tau1_shadow decision
# loop while control hands over -- there is no way to avoid that with a process
# supervisor swap. trading_bot's own order-execution config is unaffected either way.
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "must run with sudo (unit install requires root): sudo bash $0" >&2
  exit 1
fi

ROOT="/home/llewyn/crypto-scalping"
UNIT_DIR="$ROOT/scripts/ops/systemd"
RUN_AS="llewyn"

for f in trading-bot.service tau1-shadow.service ops-watchdog.service; do
  install -m 0644 "$UNIT_DIR/$f" "/etc/systemd/system/$f"
  echo "installed /etc/systemd/system/$f"
done

systemctl daemon-reload

echo "--- stopping bash supervisors ---"
sudo -u "$RUN_AS" bash -lc "cd '$ROOT' && scripts/ops/botctl.sh stop"

sleep 2

echo "--- enabling + starting systemd units ---"
systemctl enable --now trading-bot.service tau1-shadow.service ops-watchdog.service

sleep 3

echo "--- status ---"
systemctl --no-pager status trading-bot.service tau1-shadow.service ops-watchdog.service
