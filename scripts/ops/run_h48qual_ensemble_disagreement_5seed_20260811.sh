#!/usr/bin/env bash
# 3-1 앙상블 불일치 진단용: h384(FINAL12+384bar) v2와 동일 설정(epochs=40/patience, rows=30000)으로
# 5시드 재학습, 이번엔 모델 번들(true_3head_tabm_bundle.pt)을 남겨서 k=8 멤버별 출력을 뽑을 수 있게
# 한다. 원래 v2 스윕/h48orig에도 쓰인 시드라 끝나면 기존 report.json과 대조 검증 가능.
set -euo pipefail
for seed in 260620 481003 26611 903174 155827; do
  echo "=== seed=$seed 시작 $(date -u +%H:%M:%S)UTC ==="
  python scripts/train_eval_omega4_3head_parent72_eth_h48qual_final12_h384_20260811.py \
    --epochs 40 --max-train-rows 30000 --seed "$seed" --device cuda \
    --out-suffix "h48qual_final12_h384_20260811_v2b_e40_r30000_s${seed}"
  echo "=== seed=$seed 종료 $(date -u +%H:%M:%S)UTC ==="
done
echo "=== 전체 완료, 번들 확인 ==="
for seed in 260620 481003 26611 903174 155827; do
  d="tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_h48qual_final12_h384_20260811_v2b_e40_r30000_s${seed}"
  ls "$d"/*.pt 2>/dev/null || echo "NO BUNDLE: $d"
done
