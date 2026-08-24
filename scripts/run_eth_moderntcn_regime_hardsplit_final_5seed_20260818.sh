#!/usr/bin/env bash
# Thin orchestration wrapper -- runs train_eval_eth_moderntcn_direction_quality_regime_hardsplit_
# 20260818.py --stage final sequentially over N=5 genuinely-random seeds (Seed-Diversity Ensemble
# Promotion Gate: seeds drawn via Python secrets.randbelow, NOT a fixed-interval increment). Sequential
# (not parallel) deliberately -- dev is CPU-only and PyTorch's own per-process thread pool already
# claims multiple cores, so N concurrent full training runs would contend rather than speed things up,
# and today already had 2 server GPU stalls + 2 dev WSL2/Windows restarts -- keeping this simple and
# single-process makes the per-regime resume checkpointing (added to main() the same day) the only
# recovery mechanism needed if it gets interrupted again.
set -uo pipefail
cd /home/kbj20/crypto-scalping
source /home/kbj20/anaconda3/etc/profile.d/conda.sh
conda activate quant_ai

SEEDS=(839864 503468 587472 954073 120968)

for s in "${SEEDS[@]}"; do
  echo "=== [final_runner] seed=$s starting $(date -Iseconds) ==="
  python -u scripts/train_eval_eth_moderntcn_direction_quality_regime_hardsplit_20260818.py \
    --stage final --seed "$s" --device cpu --out-suffix "final_seed${s}"
  status=$?
  echo "=== [final_runner] seed=$s done exit=$status $(date -Iseconds) ==="
done

echo "=== [final_runner] ALL_SEEDS_DONE ==="
