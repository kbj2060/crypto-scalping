#!/usr/bin/env bash
# One-shot: install the eth-jmlam4-shadow unit and cut the ETH JM(lambda=4) regime-swap
# shadow loop over to systemd. Unlike the other shadow bots, this one was never under
# bash supervision (_supervise.sh) or systemd -- it was started by hand (nohup-style) with
# no auto-restart, which is why it drifts stale for minutes at a time with nothing noticing.
# Run once with sudo:
#   sudo bash scripts/ops/systemd/install_eth_jmlam4_shadow_20260810.sh
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "must run with sudo (unit install requires root): sudo bash $0" >&2
  exit 1
fi

ROOT="/home/llewyn/crypto-scalping"
UNIT_DIR="$ROOT/scripts/ops/systemd"
RUN_AS="llewyn"

install -m 0644 "$UNIT_DIR/eth-jmlam4-shadow.service" /etc/systemd/system/eth-jmlam4-shadow.service
echo "installed /etc/systemd/system/eth-jmlam4-shadow.service"

systemctl daemon-reload

echo "--- stopping any stray manually-started instance ---"
sudo -u "$RUN_AS" bash -lc "pkill -f '[l]ive_eth_jmlam4_regime_swap_shadow_20260809.py' || true"
sleep 2

echo "--- enabling + starting eth-jmlam4-shadow.service ---"
systemctl enable --now eth-jmlam4-shadow.service

sleep 3

echo "--- status ---"
systemctl --no-pager status eth-jmlam4-shadow.service
