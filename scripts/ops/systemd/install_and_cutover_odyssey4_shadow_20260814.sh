#!/usr/bin/env bash
# One-shot cutover, requested 2026-08-14: retire the eth-jmlam4-shadow.service (JM regime-swap
# axis -- never reproduced at N=5 seeds, see docs/model_contracts/odyssey2_eth_live_injection_
# contract_20260813.md #3/C3) and eth-exithead-shadow.service (h48qual exit_head liveATR-relabel
# baseline -- subsequently found NOT robust to walk-forward retraining, 0/3 independent folds,
# docs/experiments/eth_omega461_exit_head_liveatr_relabel_walkforward_20260814.md), and install the
# Odyssey4 shadow (h48qual regime-aware exit guard, unchanged, + zig075 SHORT sustained-uptrend
# entry veto, CONFIRMED) in their place. Run once with sudo:
#   sudo bash scripts/ops/systemd/install_and_cutover_odyssey4_shadow_20260814.sh
#
# What it does, in order:
#   1. Installs eth-odyssey4-shadow.service to /etc/systemd/system/
#   2. systemctl daemon-reload
#   3. Stops + disables eth-jmlam4-shadow.service and eth-exithead-shadow.service
#   4. Enables + starts eth-odyssey4-shadow.service
#   5. Prints status of all three so you can eyeball it before walking away
#
# Deliberately does NOT touch eth-regime-aware-exit-guard-shadow (Odyssey3's own shadow, run
# manually / not under this systemd tree at the time of writing) or any live-trading unit
# (trading-bot.service, tau1-shadow.service, ops-watchdog.service, btc-multislot-shadow.service,
# prometheus-exporter.service) -- only the two named-for-removal shadows and the new Odyssey4 one.
#
# This script exists because scripts/ops/deploy_watcher.sh's sudoers rule
# (scripts/ops/systemd/deploy_watcher_sudoers) is DELIBERATELY narrow -- restart of a fixed
# whitelist only, never stop/disable/enable/install -- so this cutover cannot be automated and
# needs a human with real sudo to run it once.
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "must run with sudo (unit install/stop/disable requires root): sudo bash $0" >&2
  exit 1
fi

ROOT="/home/llewyn/crypto-scalping"
UNIT_DIR="$ROOT/scripts/ops/systemd"

echo "--- installing eth-odyssey4-shadow.service ---"
install -m 0644 "$UNIT_DIR/eth-odyssey4-shadow.service" /etc/systemd/system/eth-odyssey4-shadow.service
echo "installed /etc/systemd/system/eth-odyssey4-shadow.service"

systemctl daemon-reload

echo "--- stopping + disabling eth-jmlam4-shadow.service ---"
systemctl disable --now eth-jmlam4-shadow.service || echo "  (already stopped/disabled or unit absent -- continuing)"

echo "--- stopping + disabling eth-exithead-shadow.service ---"
systemctl disable --now eth-exithead-shadow.service || echo "  (already stopped/disabled or unit absent -- continuing)"

sleep 2

echo "--- enabling + starting eth-odyssey4-shadow.service ---"
systemctl enable --now eth-odyssey4-shadow.service

sleep 5

echo "--- status ---"
systemctl --no-pager status eth-jmlam4-shadow.service 2>&1 || true
echo ""
systemctl --no-pager status eth-exithead-shadow.service 2>&1 || true
echo ""
systemctl --no-pager status eth-odyssey4-shadow.service

echo ""
echo "--- odyssey4 shadow first output (journalctl -u eth-odyssey4-shadow -n 30) ---"
journalctl -u eth-odyssey4-shadow --no-pager -n 30 || true
