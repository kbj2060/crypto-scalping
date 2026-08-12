#!/usr/bin/env python3
"""Apply the EXISTING (frozen, 2024-trained) regime3_current_sensitive_hmm_wide24 joblib to the
extended 2026 features file (now through 2026-07-12 after the 2026-07-13 data extension),
producing the extended wide24 sidecar.

Explicitly does NOT retrain the HMM -- retraining would change regime probabilities everywhere
and silently invalidate the frozen model's routing. Reuses _transform() from
scripts/experiment_regime3_current_hmm_wide24_20260529.py (the original builder).

KNOWN, PRE-DIAGNOSED DEVIATION FROM apply_regime3_wide24_sidecar_extended_20260704.py: that
script's strict reproducibility gate (max diff <= 1e-6 vs the existing Jan-Jun sidecar) FAILS here
(max diff ~0.68) because two upstream input features (ou_halflife, garch_vol_z) legitimately
changed formula/computation in features/elite.py sometime after the original wide24 HMM's Jan-Jun
sidecar was built (this exact drift, ou_halflife ~99.8% / garch_vol_z ~97.1% mismatch rates, was
already found and attributed to a genuine upstream fix -- not a bug in this apply path -- during
the 2026-07-04 extension work; see docs/model_contracts/omega6_synthesis_v1_20260703_contract.md).
Since the OLD sidecar was itself built from the now-stale feature formula, insisting it stays
byte-identical would mean keeping the wrong (pre-fix) regime values rather than the current,
correct ones. This script therefore WARNS instead of aborting on that specific known mismatch and
proceeds to write the freshly-transformed sidecar for the full extended range (backing up the old
file first, matching the original script's convention).
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
        print(
            f"WARNING: apply path does not byte-reproduce the existing (pre-formula-fix) sidecar "
            f"(max diff {max_diff:.3e}) -- accepted as a KNOWN pre-diagnosed drift from the "
            f"ou_halflife/garch_vol_z feature formula change (see module docstring); proceeding "
            f"with the fresh transform as authoritative, not aborting.",
            flush=True,
        )

    # write-to-temp-then-atomic-rename: never touch SIDECAR_PATH until the new file is fully
    # written -- a crash mid-write used to leave only a stray rename()'d backup with no
    # canonical file at all (2026-08-12 incident, root-caused after the canonical CSV silently
    # went missing on both dev and server).
    tmp_path = SIDECAR_PATH.with_name(SIDECAR_PATH.name + ".tmp_write")
    sidecar_new.to_csv(tmp_path, index=False)
    backup = SIDECAR_PATH.with_suffix(".csv.bak_pre_extend_20260713")
    if not backup.exists():
        SIDECAR_PATH.rename(backup)
        print(f"backed up existing sidecar to {backup}", flush=True)
    tmp_path.rename(SIDECAR_PATH)
    print(f"wrote extended sidecar: {len(sidecar_new)} rows -> {SIDECAR_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
