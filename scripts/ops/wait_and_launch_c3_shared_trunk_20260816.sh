#!/usr/bin/env bash
# Waits for eth_nhits_moderntcn_direction_quality to release the shared server GPU, then execs
# straight into the C3 shared-trunk N>=5-seed run under the SAME pid (handoff.sh launch tracks
# this job by pid; `exec` preserves it, so `handoff.sh status server eth_odyssey4_shared_trunk`
# stays valid across the handoff from waiting to actually training).
set -euo pipefail
WAIT_PIDFILE="tmp/handoff_jobs/eth_nhits_moderntcn_direction_quality/pid"

echo "$(date -u +%FT%TZ) waiting for eth_nhits_moderntcn_direction_quality to free the GPU..."
while [[ -f "$WAIT_PIDFILE" ]]; do
  pid="$(cat "$WAIT_PIDFILE" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    sleep 60
  else
    break
  fi
done

echo "$(date -u +%FT%TZ) GPU free, launching C3 shared-trunk N>=5-seed run"
exec python3 scripts/research_eth_odyssey4_shared_trunk_regime_experts_20260816.py \
  --epochs 28 --n-seeds 5 --mode both --device cuda --feature-pipeline true
