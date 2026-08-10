"""Build extended 2026 eval frames (2026-01-01..06-30, vs the legacy candidate CSV's 02-28 cap) for
the pinned102 h48qual/zig075 retrain, two variants: wide24 (live regime3) and JM lambda=4. Verified
2026-08-09: the live bundles' full 102 base_cols are ALL present in
data/splits/year_oos/training_features_2026_rebuilt.csv (through 06-30) once merged with a regime3
overlay -- so this bypasses the legacy action_score/m7/decontam pipeline entirely (those columns are
explicitly excluded from base_cols by the live forbidden_feature_policy, so they were never needed
for this comparison, just inherited by accident when the earlier retrain didn't pin base_cols).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_2026 = ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"
WIDE24_2026 = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/training_features_2026_rebuilt_regime3_current_sensitive_hmm_wide24.csv"
JM_2026 = ROOT / "data/ensemble/supervised/eth_regime3_current_hmm_jmlam4_20260809_2026_maskedname.csv"
OUT_DIR = ROOT / "data/ensemble/supervised/eth_eval_2026_extended_20260809"


def build() -> None:
    # RAW ONLY -- do NOT pre-merge regime3 columns here. The omega pipeline's own
    # _overlay_required() injects regime3_current_sensitive_wide24_* itself from
    # REGIME3_CURRENT_2026 (which the caller controls); pre-merging caused a pandas merge-suffix
    # collision (_x/_y) that broke the pipeline's exact-column-name lookup. Truncate to 06-30 to
    # match the JM regime3 file's actual coverage (raw goes to 07-20, JM regime3 only to 06-30).
    raw = pd.read_csv(RAW_2026, low_memory=False)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw = raw.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    raw = raw[raw["timestamp"] <= "2026-06-30 23:55:00"].reset_index(drop=True)
    out_path = OUT_DIR / "eth_eval_2026_extended_raw.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out_path, index=False)
    print(f"rows={len(raw)} range=({raw['timestamp'].min()}, {raw['timestamp'].max()}) -> {out_path}")


if __name__ == "__main__":
    build()
