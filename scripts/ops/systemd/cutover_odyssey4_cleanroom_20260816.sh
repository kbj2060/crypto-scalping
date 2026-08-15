#!/usr/bin/env bash
# In-place cutover, 2026-08-16: point the ALREADY-RUNNING eth-odyssey4-shadow.service at the new
# cleanroom script (live_eth_odyssey4_zig075_entry_veto_shadow_cleanroom_20260816.py) instead of the
# 2026-08-14 one. Same unit name, same WorkingDirectory, same data/live/eth_odyssey4_shadow/state.json
# -- this is a code-only swap, not a new deployment, so the shadow's equity/trade history is NOT
# reset. The two scripts were verified bit-identical on 2,000+ real historical bars before this
# cutover was written (scripts/verify_eth_odyssey4_cleanroom_parity_20260816.py,
# docs/experiments/eth_odyssey_live_cleanroom_dependency_rewrite_20260816.md).
#
# Why this exists: the old script pulled in ~6,850 lines across 8 files to reach ETH Odyssey's real
# decision logic -- including two fully unused SOL/BTC training-script imports, one fully dead
# risk-sidecar import, and a runtime_config.py import that (as a side effect) loads an entirely
# unrelated Omega5 system and can raise at import time on Omega5-only env-var mismatches. The new
# script depends on three new Odyssey-owned modules (trading_bot_modules.odyssey_tabm_core /
# .odyssey_regime3_live / .odyssey_live_adapter) totalling well under 1,000 lines, with zero
# SOL/BTC/Omega5/dead-sidecar imports. Neither trading_bot.py nor trading_bot_modules/
# omega4_6_1_live.py nor runtime_config.py is touched by this cutover -- matches the "live files
# unchanged" principle documented for every Odyssey generation so far.
#
# Run once with sudo:
#   sudo bash scripts/ops/systemd/cutover_odyssey4_cleanroom_20260816.sh
#
# What it does, in order:
#   1. Re-installs eth-odyssey4-shadow.service to /etc/systemd/system/ (only the ExecStart line
#      changed vs. what's already installed -- see git diff of this repo's copy of the unit file)
#   2. systemctl daemon-reload
#   3. Restarts eth-odyssey4-shadow.service (NOT disable/enable -- it's the same unit, already
#      enabled from the 2026-08-14 cutover)
#   4. Prints status + the first ~30 log lines so you can confirm code_lineage=cleanroom_20260816
#      appears in the init log and no [error] lines follow
#
# Rollback (if anything looks wrong): re-run install_and_cutover_odyssey4_shadow_20260814.sh's
# steps manually, or just `git checkout` this repo's eth-odyssey4-shadow.service back to the 08-14
# ExecStart line, re-run steps 1-3 above. state.json is untouched by either direction.
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "must run with sudo (unit install/restart requires root): sudo bash $0" >&2
  exit 1
fi

ROOT="/home/llewyn/crypto-scalping"
UNIT_DIR="$ROOT/scripts/ops/systemd"
NEW_SCRIPT="$ROOT/scripts/live_eth_odyssey4_zig075_entry_veto_shadow_cleanroom_20260816.py"

if [[ ! -f "$NEW_SCRIPT" ]]; then
  echo "expected cleanroom script not found at $NEW_SCRIPT -- did the repo pull land? aborting." >&2
  exit 1
fi

echo "--- installing updated eth-odyssey4-shadow.service (ExecStart -> cleanroom script) ---"
install -m 0644 "$UNIT_DIR/eth-odyssey4-shadow.service" /etc/systemd/system/eth-odyssey4-shadow.service
grep ExecStart /etc/systemd/system/eth-odyssey4-shadow.service

systemctl daemon-reload

echo "--- restarting eth-odyssey4-shadow.service (same unit, code-only swap) ---"
systemctl restart eth-odyssey4-shadow.service

sleep 5

echo "--- status ---"
systemctl --no-pager status eth-odyssey4-shadow.service

echo ""
echo "--- first output after cutover (journalctl -u eth-odyssey4-shadow -n 30) ---"
echo "    check for: code_lineage=cleanroom_20260816 in the [init] line, no [error] lines,"
echo "    same duration_threshold/detector window as before."
journalctl -u eth-odyssey4-shadow --no-pager -n 30 || true
