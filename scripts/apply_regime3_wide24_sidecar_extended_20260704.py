#!/usr/bin/env python3
"""Apply the EXISTING (frozen, 2024-trained) regime3_current_sensitive_hmm_wide24 joblib to the
extended 2026 features file (now Jan-Jun after the 2026-07-04 data extension), producing the
extended wide24 sidecar the Omega6 frozen winner's L2 routing depends on.

Explicitly does NOT retrain the HMM -- retraining would change regime probabilities everywhere
and silently invalidate the frozen model's routing. Reuses _transform() from
scripts/experiment_regime3_current_hmm_wide24_20260529.py (the original builder).

Safety: before writing, re-applies the joblib to the OLD file range (Jan-Feb) and asserts the
output matches the existing sidecar row-for-row -- proving this apply path reproduces the
original build exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import experiment_regime3_current_hmm_wide24_20260529 as hmm_mod  # noqa: E402

DIR = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
JOBLIB_PATH = DIR / "regime3_current_sensitive_hmm_wide24_2024.joblib"
SIDECAR_PATH = DIR / "training_features_2026_rebuilt_regime3_current_sensitive_hmm_wide24.csv"
FEATURES_2026 = ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"


def main() -> int:
    payload = joblib.load(JOBLIB_PATH)
    frame = hmm_mod._read(FEATURES_2026)
    print(f"extended 2026 features: {len(frame)} rows ({frame['timestamp'].min()}..{frame['timestamp'].max()})", flush=True)

    sidecar_new, _ev = hmm_mod._transform(payload, frame)

    existing = pd.read_csv(SIDECAR_PATH, parse_dates=["timestamp"])
    print(f"existing sidecar: {len(existing)} rows ({existing['timestamp'].min()}..{existing['timestamp'].max()})", flush=True)

    # Reproducibility check on the overlapping (old) range
    merged = existing.merge(sidecar_new, on="timestamp", suffixes=("_old", "_new"), how="inner")
    if len(merged) != len(existing):
        raise RuntimeError(f"overlap mismatch: {len(merged)} joined vs {len(existing)} existing")
    prob_cols = [c for c in existing.columns if c != "timestamp"]
    max_diff = 0.0
    for c in prob_cols:
        diff = float(np.max(np.abs(merged[f"{c}_old"].to_numpy() - merged[f"{c}_new"].to_numpy())))
        max_diff = max(max_diff, diff)
    print(f"reproducibility max abs diff on old range: {max_diff:.3e}", flush=True)
    if max_diff > 1e-6:
        raise RuntimeError(f"apply path does NOT reproduce the existing sidecar (max diff {max_diff}) -- aborting, not overwriting")

    backup = SIDECAR_PATH.with_suffix(".csv.bak_pre_extend_20260704")
    if not backup.exists():
        SIDECAR_PATH.rename(backup)
        print(f"backed up existing sidecar to {backup}", flush=True)
    sidecar_new.to_csv(SIDECAR_PATH, index=False)
    print(f"wrote extended sidecar: {len(sidecar_new)} rows -> {SIDECAR_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
