#!/usr/bin/env bash
set -euo pipefail

cd /home/llewyn/crypto-scalping

PY=/home/llewyn/miniconda3/envs/quant_ai/bin/python
SCRIPT=scripts/train_eval_omega1_2_exposure_governor_20260606.py
LOG=logs/omega1_2_exposure_governor_until_1000_20260606.log
mkdir -p logs

run_case() {
  local suffix="$1"
  shift
  local now
  now="$(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "[$now] START $suffix" | tee -a "$LOG"
  "$PY" "$SCRIPT" "$@" --out-suffix "$suffix" --device auto 2>&1 | tee -a "$LOG"
  now="$(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "[$now] END $suffix" | tee -a "$LOG"
}

while [[ "$(date '+%H%M')" < "1000" ]]; do
  run_case loop_hgb_cap090_scale140_edge002_gate_s261020 \
    --max-label-rows 0 --notional-cap 0.9 --scales 1.0,1.1,1.25,1.4 \
    --compensate-sltp-by-notional --min-edge 0.002 --min-prob 0.62 --min-margin 0.18 --seed 261020

  run_case loop_hgb_cap100_scale140_edge002_gate_s261021 \
    --max-label-rows 0 --notional-cap 1.0 --scales 1.0,1.1,1.25,1.4 \
    --compensate-sltp-by-notional --min-edge 0.002 --min-prob 0.62 --min-margin 0.18 --seed 261021

  run_case loop_hgb_cap100_scale155_edge002_gate_s261022 \
    --max-label-rows 0 --notional-cap 1.0 --scales 1.0,1.15,1.3,1.55 \
    --compensate-sltp-by-notional --min-edge 0.002 --min-prob 0.62 --min-margin 0.18 --seed 261022

  run_case loop_hgb_cap100_scale155_edge002_gate_highprob_s261023 \
    --max-label-rows 0 --notional-cap 1.0 --scales 1.0,1.15,1.3,1.55 \
    --compensate-sltp-by-notional --min-edge 0.002 --min-prob 0.68 --min-margin 0.22 --seed 261023

  run_case loop_hgb_cap110_scale140_edge002_gate_s261024 \
    --max-label-rows 0 --notional-cap 1.1 --scales 1.0,1.1,1.25,1.4 \
    --compensate-sltp-by-notional --min-edge 0.002 --min-prob 0.62 --min-margin 0.18 --seed 261024

  run_case loop_hgb_cap100_scale155_edge003_gate_s261025 \
    --max-label-rows 0 --notional-cap 1.0 --scales 1.0,1.15,1.3,1.55 \
    --compensate-sltp-by-notional --min-edge 0.003 --min-prob 0.62 --min-margin 0.18 --seed 261025
done

"$PY" - <<'PY' | tee -a "$LOG"
import json, glob, os
rows=[]
for p in sorted(glob.glob('/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega1_2_exposure_governor_20260606_*/report.json')):
    name=os.path.basename(os.path.dirname(p))
    if not ('loop_' in name or 'full_hgb_cap100_smallscale_edge002_gate' in name or 'full_hgb_cap110_smallscale_edge002_gate' in name):
        continue
    r=json.load(open(p)); val=r['results']['validation']; oos=r['results']['oos']
    rows.append((name,val['pnl'],val['mdd'],val['wr'],val['trades'],oos['pnl'],oos['mdd'],oos['wr'],oos['trades'],p))
print('name,val_pnl,val_mdd,val_wr,val_trades,oos_pnl,oos_mdd,oos_wr,oos_trades,report')
for row in sorted(rows, key=lambda x: (x[5], x[1]), reverse=True):
    print(','.join([row[0], *[f'{x:.6f}' if isinstance(x,float) else str(x) for x in row[1:]]]))
PY
