#!/usr/bin/env bash
# Resume of run_a4_fresh_extension_pipeline_20260831.sh from stage 5 onward.
#
# The first attempt (stages 1-4: klines, metrics/funding zips, corrected TOTAL_*_metrics
# reference, ETH canonical feature extension) completed successfully, extending ETH's canonical
# features file all the way to the latest available kline bar (2026-08-31 11:30). Stage 5 (the
# 2026-08-23 OI-future-leak fix) then correctly failed fast: the daily metrics archive only
# publishes through the previous day, so the corrected TOTAL_ETHUSDT_metrics_2024_2026.csv
# reference stops at 2026-08-31 00:00:00 -- about 11.5h short of the canonical file's new tail,
# which exceeds that fix script's deliberate 9h merge_asof tolerance and left ~90 cells
# unfixable, so it aborted rather than silently leaving a gap or ffilling stale OI values.
#
# Fix: truncate every asset's features file to the metrics reference's own actual coverage
# (scripts/truncate_features_to_metrics_safe_cutoff_20260831.py) before each asset's
# corresponding metrics-integrity-fix step. This is a real data-availability boundary (the
# archive publishes with a 1-day lag), not a bug to route around -- "as recent as possible" per
# the task's own instructions now means "through the metrics archive's actual coverage", not
# literally the current wall-clock minute.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
echo "=== A4 pipeline RESUME starting ==="

stage() { echo; echo "STAGE_START=$1 $(date -u +%H:%M:%S)"; }
done_stage() { echo "STAGE_DONE=$1 $(date -u +%H:%M:%S)"; }

# ---------------------------------------------------------------------------
# 5a. Compute the shared metrics-safe cutoff (min of the three TOTAL_*_metrics reference maxima)
#     and truncate ETH's canonical file to it (stage 4 already ran and overshot this boundary).
# ---------------------------------------------------------------------------
stage compute_metrics_safe_cutoff
CUTOFF=$(python3 -c "
import pandas as pd
ends = []
for sym in ('ETHUSDT', 'BTCUSDT', 'SOLUSDT'):
    with open(f'data/TOTAL_{sym}_metrics_2024_2026.csv') as f:
        f.seek(0, 2); size = f.tell(); f.seek(max(0, size - 4096))
        tail = f.read()
    last = [ln for ln in tail.strip().splitlines() if ln.strip()][-1]
    ends.append(pd.Timestamp(last.split(',')[0]))
print(min(ends))
")
echo "METRICS_SAFE_CUTOFF=${CUTOFF}"
python scripts/truncate_features_to_metrics_safe_cutoff_20260831.py \
  --path data/splits/year_oos/training_features_2026_rebuilt.csv --cutoff "${CUTOFF}"
done_stage compute_metrics_safe_cutoff

# ---------------------------------------------------------------------------
# 5b. Retry ETH OI-future-leak fix (stage 5 from the original driver).
# ---------------------------------------------------------------------------
stage eth_oi_futureleak_fix
python scripts/fix_eth_canonical_2026_oi_futureleak_20260823.py
done_stage eth_oi_futureleak_fix

# ---------------------------------------------------------------------------
# 6. SOL: full FeatureEngineer recompute + truncate to the same safe cutoff + split by year.
# ---------------------------------------------------------------------------
stage sol_features_rebuild
python scripts/build_sol_raw_frame_20260707.py
python scripts/build_sol_features_20260707.py
python scripts/truncate_features_to_metrics_safe_cutoff_20260831.py \
  --path data/splits/year_oos/sol_features_2024_2026.csv --cutoff "${CUTOFF}"
python scripts/split_sol_features_by_year_20260707.py
done_stage sol_features_rebuild

# ---------------------------------------------------------------------------
# 7. BTC: same full-recompute + truncate + split pattern.
# ---------------------------------------------------------------------------
stage btc_features_rebuild
python scripts/build_btc_raw_frame_20260708.py
python scripts/build_btc_features_20260708.py
python scripts/truncate_features_to_metrics_safe_cutoff_20260831.py \
  --path data/splits/year_oos/btc_features_2024_2026.csv --cutoff "${CUTOFF}"
python scripts/split_btc_features_by_year_20260708.py
done_stage btc_features_rebuild

# ---------------------------------------------------------------------------
# 8. BTC/SOL metrics-vintage fix (whole-file gate-then-patch, now within the safe cutoff).
# ---------------------------------------------------------------------------
stage btcsol_metrics_vintage_fix
python scripts/fix_btcsol_metrics_vintage_20260823.py
done_stage btcsol_metrics_vintage_fix

# ---------------------------------------------------------------------------
# 9. ETH regime3 wide24 overlay.
# ---------------------------------------------------------------------------
stage eth_wide24_overlay
python scripts/apply_regime3_wide24_sidecar_extended_20260820.py
done_stage eth_wide24_overlay

# ---------------------------------------------------------------------------
# 10. SOL/BTC regime3 wide24 overlay.
# ---------------------------------------------------------------------------
stage solbtc_wide24_overlay
python scripts/extend_regime3_wide24_sol_btc_20260721.py
done_stage solbtc_wide24_overlay

# ---------------------------------------------------------------------------
# 11. SOL/BTC wave3 zigzag direction labels.
# ---------------------------------------------------------------------------
stage direction_labels
python scripts/build_wave3_action_labels_20260531.py \
  --input-2024 data/splits/year_oos/sol_features_2024.csv \
  --input-2025 data/splits/year_oos/sol_features_2025.csv \
  --input-2026 data/splits/year_oos/sol_features_2026.csv \
  --out-dir tmp/causal_regen_20260516/sol_zigzag_action_labels_20260707
python scripts/build_wave3_action_labels_20260531.py \
  --input-2024 data/splits/year_oos/btc_features_2024.csv \
  --input-2025 data/splits/year_oos/btc_features_2025.csv \
  --input-2026 data/splits/year_oos/btc_features_2026.csv \
  --out-dir tmp/causal_regen_20260516/btc_zigzag_action_labels_20260708
done_stage direction_labels

# ---------------------------------------------------------------------------
# 12. BTC h48-conservative triple-barrier + padded quality labels.
# ---------------------------------------------------------------------------
stage btc_quality_labels
python scripts/build_omega1_2_triple_barrier_labels_btc_20260708.py
python scripts/pad_h48_quality_labels_to_zigzag_timestamps_btc_20260708.py
done_stage btc_quality_labels

# ---------------------------------------------------------------------------
# 13. Re-score (NOT retrain) all three frozen parent bundles.
# ---------------------------------------------------------------------------
stage parent_rescoring
python scripts/build_omega4_6_1_extended_parent_predictions_20260706.py
python scripts/rescore_sol_btc_parent_predictions_20260713.py --asset sol --device cuda
python scripts/rescore_sol_btc_parent_predictions_20260713.py --asset btc --device cuda
done_stage parent_rescoring

# ---------------------------------------------------------------------------
# 14. config A (prealloc cap=3.0) vs config B (uncapped) on validation / oos_extended /
#     fresh_window(entry_floor=2026-07-01).
# ---------------------------------------------------------------------------
stage portfolio_replay
python scripts/replay_portfolio_prealloc_eth15x_fresh_confirmation_20260831.py
done_stage portfolio_replay

echo
echo "=== A4 fresh extension pipeline COMPLETE ==="
