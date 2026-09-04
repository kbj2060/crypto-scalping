#!/usr/bin/env bash
# Third resume: root-caused the pandas quirk properly this time. The bare "YYYY-MM-DD" (no time)
# last-row serialization only happens when a dataframe's FINAL row has time exactly 00:00:00 --
# confirmed by tracing it through TWO different unmodified pre-existing scripts
# (fix_eth_canonical_2026_oi_futureleak_20260823.py's own `df.to_csv()`, then
# build_omega4_6_1_extended_parent_predictions_20260706.py failing to re-read that same file) even
# though scripts/truncate_features_to_metrics_safe_cutoff_20260831.py's own explicit
# .dt.strftime() write was clean. Patching every downstream script's to_csv() call individually
# is impractical (dozens of pre-existing scripts touch these files); the general, robust fix is to
# never let the metrics-safe cutoff itself land exactly on midnight, so no downstream file's last
# row does either. New cutoff = (metrics reference max) - 5 minutes, landing on 23:55:00 instead
# of 00:00:00.
#
# Restores ETH's canonical file from the known-good, correctly-formatted, 11:30-ending backup
# (.bak_pre_metrics_safe_truncate_20260831_v2 -- captured before either truncation attempt,
# confirmed not to have the exact-midnight last row that triggers this) rather than redoing the
# expensive FeatureEngineer recompute. SOL/BTC are rebuilt from scratch again (fast, idempotent,
# and simpler than hunting the right backup generation), then every stage from the metrics-vintage
# fix onward is redone against the corrected (non-midnight) cutoff.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
echo "=== A4 pipeline RESUME-3 starting ==="

stage() { echo; echo "STAGE_START=$1 $(date -u +%H:%M:%S)"; }
done_stage() { echo "STAGE_DONE=$1 $(date -u +%H:%M:%S)"; }

stage restore_and_retruncate_eth_v3
cp data/splits/year_oos/training_features_2026_rebuilt.csv.bak_pre_metrics_safe_truncate_20260831_v2 \
   data/splits/year_oos/training_features_2026_rebuilt.csv
echo "restored ETH canonical from known-good pre-truncation backup (v2)"
tail -3 data/splits/year_oos/training_features_2026_rebuilt.csv | cut -d, -f1

CUTOFF=$(python3 -c "
import pandas as pd
ends = []
for sym in ('ETHUSDT', 'BTCUSDT', 'SOLUSDT'):
    with open(f'data/TOTAL_{sym}_metrics_2024_2026.csv') as f:
        f.seek(0, 2); size = f.tell(); f.seek(max(0, size - 4096))
        tail = f.read()
    last = [ln for ln in tail.strip().splitlines() if ln.strip()][-1]
    ends.append(pd.Timestamp(last.split(',')[0]))
cutoff = min(ends) - pd.Timedelta(minutes=5)
print(cutoff)
")
echo "METRICS_SAFE_CUTOFF (non-midnight, -5min)=${CUTOFF}"
python scripts/truncate_features_to_metrics_safe_cutoff_20260831.py \
  --path data/splits/year_oos/training_features_2026_rebuilt.csv --cutoff "${CUTOFF}" \
  --backup-suffix .bak_pre_metrics_safe_truncate_20260831_v3
python3 -c "
import pandas as pd
df = pd.read_csv('data/splits/year_oos/training_features_2026_rebuilt.csv', usecols=['timestamp'])
df['timestamp'] = pd.to_datetime(df['timestamp'])
print('post-truncate verify OK:', len(df), df['timestamp'].min(), '..', df['timestamp'].max())
assert df['timestamp'].max().strftime('%H:%M:%S') != '00:00:00', 'still landed on midnight!'
"
done_stage restore_and_retruncate_eth_v3

stage eth_oi_futureleak_fix
python scripts/fix_eth_canonical_2026_oi_futureleak_20260823.py
python3 -c "
import pandas as pd
df = pd.read_csv('data/splits/year_oos/training_features_2026_rebuilt.csv', usecols=['timestamp'])
df['timestamp'] = pd.to_datetime(df['timestamp'])
print('post-fix verify OK:', len(df), df['timestamp'].min(), '..', df['timestamp'].max())
"
done_stage eth_oi_futureleak_fix

stage sol_features_rebuild
python scripts/build_sol_raw_frame_20260707.py
python scripts/build_sol_features_20260707.py
python scripts/truncate_features_to_metrics_safe_cutoff_20260831.py \
  --path data/splits/year_oos/sol_features_2024_2026.csv --cutoff "${CUTOFF}"
python3 -c "
import pandas as pd
df = pd.read_csv('data/splits/year_oos/sol_features_2024_2026.csv', usecols=['timestamp'])
df['timestamp'] = pd.to_datetime(df['timestamp'])
print('post-truncate verify OK:', len(df), df['timestamp'].min(), '..', df['timestamp'].max())
"
python scripts/split_sol_features_by_year_20260707.py
done_stage sol_features_rebuild

stage btc_features_rebuild
python scripts/build_btc_raw_frame_20260708.py
python scripts/build_btc_features_20260708.py
python scripts/truncate_features_to_metrics_safe_cutoff_20260831.py \
  --path data/splits/year_oos/btc_features_2024_2026.csv --cutoff "${CUTOFF}"
python3 -c "
import pandas as pd
df = pd.read_csv('data/splits/year_oos/btc_features_2024_2026.csv', usecols=['timestamp'])
df['timestamp'] = pd.to_datetime(df['timestamp'])
print('post-truncate verify OK:', len(df), df['timestamp'].min(), '..', df['timestamp'].max())
"
python scripts/split_btc_features_by_year_20260708.py
done_stage btc_features_rebuild

stage btcsol_metrics_vintage_fix
python scripts/fix_btcsol_metrics_vintage_20260823.py
python3 -c "
import pandas as pd
for f in ('sol_features_2026.csv', 'btc_features_2026.csv'):
    df = pd.read_csv(f'data/splits/year_oos/{f}', usecols=['timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f'post-fix verify OK {f}:', len(df), df['timestamp'].min(), '..', df['timestamp'].max())
"
done_stage btcsol_metrics_vintage_fix

stage eth_wide24_overlay
python scripts/apply_regime3_wide24_sidecar_extended_20260820.py
done_stage eth_wide24_overlay

stage solbtc_wide24_overlay
python scripts/extend_regime3_wide24_sol_btc_20260721.py
done_stage solbtc_wide24_overlay

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

stage btc_quality_labels
python scripts/build_omega1_2_triple_barrier_labels_btc_20260708.py
python scripts/pad_h48_quality_labels_to_zigzag_timestamps_btc_20260708.py
done_stage btc_quality_labels

stage parent_rescoring
python scripts/build_omega4_6_1_extended_parent_predictions_20260706.py
python scripts/rescore_sol_btc_parent_predictions_20260713.py --asset sol --device cuda
python scripts/rescore_sol_btc_parent_predictions_20260713.py --asset btc --device cuda
done_stage parent_rescoring

stage portfolio_replay
python scripts/replay_portfolio_prealloc_eth15x_fresh_confirmation_20260831.py
done_stage portfolio_replay

echo
echo "=== A4 fresh extension pipeline COMPLETE ==="
