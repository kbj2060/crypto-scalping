#!/usr/bin/env bash
# A4 cross-symbol-exposure-cap fresh confirmation (docs/eth_cross_symbol_exposure_cap_design_20260831.md).
# Reconstructs the 2026-07-13 fresh-window-extension procedure
# (docs/model_contracts/portfolio_concurrent_3asset_fresh_window_confirmation_20260713.md) for
# ETH/SOL/BTC, extended through "today" instead of 07-12, PLUS re-applies the 2026-08-23
# metrics-vintage/future-leak integrity fixes (docs/experiments/
# eth_binance_metrics_archive_backfill_and_canonical_divergence_20260823.md) to the newly-extended
# tail -- those fixes did not exist yet on 07-13 and the underlying scripts that would otherwise
# reintroduce the bug (update_features.py, build_{sol,btc}_raw_frame_*.py's load_metrics()) have
# NOT been patched at the source, only the data was patched after the fact on 08-23. Skipping this
# would put a systematic ~5-minute future-reference into every newly-added row's
# sum_open_interest_value/sum_toptrader_long_short_ratio/count_long_short_ratio and ~12-17 derived
# features, violating CLAUDE.md's Fresh-Forward causal-availability rule for exactly the tail this
# task cares about most.
#
# Runs entirely on GPU server via handoff.sh (this repo's GPU-backlog-offload policy). Read-only
# public Binance REST/data.binance.vision endpoints only -- no live-account client touched. No
# retraining anywhere -- every model step is frozen-artifact re-scoring only.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
TODAY="$(date -u +%Y-%m-%d)"
echo "=== A4 fresh extension pipeline starting, TODAY=${TODAY} ==="

stage() { echo; echo "STAGE_START=$1 $(date -u +%H:%M:%S)"; }
done_stage() { echo "STAGE_DONE=$1 $(date -u +%H:%M:%S)"; }

# ---------------------------------------------------------------------------
# 1. Extend raw klines (ETH/BTC/SOL) -- incremental from each file's own last bar.
# ---------------------------------------------------------------------------
stage klines
python scripts/extend_klines_20260713.py --symbol ETHUSDT
python scripts/extend_klines_20260713.py --symbol BTCUSDT
python scripts/extend_klines_20260713.py --symbol SOLUSDT
done_stage klines

# ---------------------------------------------------------------------------
# 2. Extend raw metrics/funding zips (ETH/BTC/SOL). Start conservatively at 2026-01-01: the
#    downloader skips any day/month whose zip already exists, so this is idempotent-cheap, not a
#    full re-download, and removes any risk of mis-computing each asset's true gap start.
# ---------------------------------------------------------------------------
stage metrics_funding_zips
python scripts/download_metrics_funding_generic_20260713.py --symbol ETHUSDT --start 2026-01-01 --end "${TODAY}"
python scripts/download_metrics_funding_generic_20260713.py --symbol BTCUSDT --start 2026-01-01 --end "${TODAY}"
python scripts/download_metrics_funding_generic_20260713.py --symbol SOLUSDT --start 2026-01-01 --end "${TODAY}"
done_stage metrics_funding_zips

# ---------------------------------------------------------------------------
# 3. Refresh the CORRECTED metrics reference (+5min bucket-label convention fix baked in --
#    see docs/experiments/eth_binance_metrics_archive_backfill_and_canonical_divergence_20260823.md
#    section 2/4). This script is already env-parameterized for BTC/SOL backfill (2026-08-23).
# ---------------------------------------------------------------------------
stage total_metrics_reference
METRICS_SYMBOL=ETHUSDT python scripts/download_eth_binance_metrics_archive_20260823.py
METRICS_SYMBOL=BTCUSDT python scripts/download_eth_binance_metrics_archive_20260823.py
METRICS_SYMBOL=SOLUSDT python scripts/download_eth_binance_metrics_archive_20260823.py
done_stage total_metrics_reference

# ---------------------------------------------------------------------------
# 4. ETH: extend the canonical training_features_2026_rebuilt.csv (existing-first merge, new
#    tail only -- see script docstring for why update_features.py cannot be used here).
# ---------------------------------------------------------------------------
stage eth_features_extend
python scripts/build_eth_raw_frame_and_extend_canonical_20260831.py
done_stage eth_features_extend

# ---------------------------------------------------------------------------
# 5. ETH: correct the new tail's OI/long-short-ratio columns + derived features (dynamic
#    WIN_START..file-end window -- naturally covers whatever this session just added).
# ---------------------------------------------------------------------------
stage eth_oi_futureleak_fix
python scripts/fix_eth_canonical_2026_oi_futureleak_20260823.py
done_stage eth_oi_futureleak_fix

# ---------------------------------------------------------------------------
# 6. SOL: full FeatureEngineer recompute (not append-only -- avoids ou_halflife/garch_vol_z
#    rolling-window seeding errors, per task instructions) + split by year.
# ---------------------------------------------------------------------------
stage sol_features_rebuild
python scripts/build_sol_raw_frame_20260707.py
python scripts/build_sol_features_20260707.py
python scripts/split_sol_features_by_year_20260707.py
done_stage sol_features_rebuild

# ---------------------------------------------------------------------------
# 7. BTC: same full-recompute pattern.
# ---------------------------------------------------------------------------
stage btc_features_rebuild
python scripts/build_btc_raw_frame_20260708.py
python scripts/build_btc_features_20260708.py
python scripts/split_btc_features_by_year_20260708.py
done_stage btc_features_rebuild

# ---------------------------------------------------------------------------
# 8. BTC/SOL: correct metrics-vintage mixing across the whole (freshly-recomputed) file using the
#    corrected reference from stage 3 -- re-establishes the 2026-08-23 fix after the full
#    recompute in stages 6/7 would otherwise have silently reintroduced it for ALL historical rows,
#    not just the new tail (whole-file gate-then-patch, not date-limited -- see script body).
# ---------------------------------------------------------------------------
stage btcsol_metrics_vintage_fix
python scripts/fix_btcsol_metrics_vintage_20260823.py
done_stage btcsol_metrics_vintage_fix

# ---------------------------------------------------------------------------
# 9. ETH regime3 wide24 overlay (frozen 2024-trained joblib, causal _transform, no retraining).
#    Must run AFTER stage 5 (consumes the OI-corrected canonical).
# ---------------------------------------------------------------------------
stage eth_wide24_overlay
python scripts/apply_regime3_wide24_sidecar_extended_20260820.py
done_stage eth_wide24_overlay

# ---------------------------------------------------------------------------
# 10. SOL/BTC regime3 wide24 overlay (both assets, one script). Must run AFTER stage 8.
# ---------------------------------------------------------------------------
stage solbtc_wide24_overlay
python scripts/extend_regime3_wide24_sol_btc_20260721.py
done_stage solbtc_wide24_overlay

# ---------------------------------------------------------------------------
# 11. SOL/BTC wave3 zigzag direction labels (identical builder/params both assets already used;
#     all params left at script defaults, which byte-match the existing audit.json for both dirs).
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
# 12. BTC h48-conservative triple-barrier labels, then padded onto the zigzag timestamp index
#     (quality label used only by BTC's parent bundle).
# ---------------------------------------------------------------------------
stage btc_quality_labels
python scripts/build_omega1_2_triple_barrier_labels_btc_20260708.py
python scripts/pad_h48_quality_labels_to_zigzag_timestamps_btc_20260708.py
done_stage btc_quality_labels

# ---------------------------------------------------------------------------
# 13. Re-score (NOT retrain) all three frozen parent bundles on the extended data.
# ---------------------------------------------------------------------------
stage parent_rescoring
python scripts/build_omega4_6_1_extended_parent_predictions_20260706.py
python scripts/rescore_sol_btc_parent_predictions_20260713.py --asset sol --device cuda
python scripts/rescore_sol_btc_parent_predictions_20260713.py --asset btc --device cuda
done_stage parent_rescoring

# ---------------------------------------------------------------------------
# 14. Run config A (prealloc cap=3.0) vs config B (uncapped CURRENT_BASELINE) on
#     validation / oos_extended / fresh_window(entry_floor=2026-07-01).
# ---------------------------------------------------------------------------
stage portfolio_replay
python scripts/replay_portfolio_prealloc_eth15x_fresh_confirmation_20260831.py
done_stage portfolio_replay

echo
echo "=== A4 fresh extension pipeline COMPLETE ==="
