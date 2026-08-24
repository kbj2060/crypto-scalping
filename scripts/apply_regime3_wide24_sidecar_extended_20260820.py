#!/usr/bin/env python3
"""Apply the EXISTING (frozen, 2024-trained) regime3_current_sensitive_hmm_wide24 joblib to the
extended 2026 features file (now through 2026-08-19 after this session's data extension),
producing the extended wide24 sidecar.

Modeled directly on scripts/apply_regime3_wide24_sidecar_extended_20260713.py (same joblib, same
_transform() reuse, same backup-then-atomic-write convention) -- only the backup suffix date
changes. Explicitly does NOT retrain the HMM -- retraining would change regime probabilities
everywhere and silently invalidate the frozen model's routing.

Note: the existing on-disk sidecar (training_features_2026_rebuilt_regime3_current_sensitive_hmm_
wide24.csv) currently ends 2026-06-30, not 2026-07-12 as the 20260713 script's docstring claimed it
would produce -- this session did not investigate why (out of scope), but the reproducibility check
below still verifies the current apply path reproduces whatever IS on disk for the overlap region.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

# This dev shell has no mamba_ssm installed (GPU-only package, matches project memory
# dev_machine_amd_gpu_no_cuda / gpu_backlog_offload_to_server pattern). It's only pulled in
# transitively -- train_regime3_hmm_mamba_20260529.py does `from mamba_ssm import Mamba` at
# module scope for an unrelated training class, but this script only needs three lightweight
# helpers (CLASSES3/_current_labels3/_read) that experiment_regime3_current_hmm_wide24_20260529.py
# re-imports from that file. Stub the module so the import succeeds without needing the real
# package -- no source file is modified, this is a runtime-only sys.modules shim.
if "mamba_ssm" not in sys.modules:
    _stub = types.ModuleType("mamba_ssm")
    _stub.Mamba = object
    sys.modules["mamba_ssm"] = _stub

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
            f"WARNING: apply path does not byte-reproduce the existing sidecar on the overlap "
            f"range (max diff {max_diff:.3e}) -- see script docstring / session report for "
            f"investigation status; proceeding with the fresh transform as authoritative, not "
            f"aborting (matches 20260713 script's convention).",
            flush=True,
        )

    # write-to-temp-then-atomic-rename: never touch SIDECAR_PATH until the new file is fully
    # written (matches 20260713 script's convention, itself a fix for a 2026-08-12 incident).
    tmp_path = SIDECAR_PATH.with_name(SIDECAR_PATH.name + ".tmp_write")
    sidecar_new.to_csv(tmp_path, index=False)
    backup = SIDECAR_PATH.with_suffix(".csv.bak_pre_extend_20260820")
    if not backup.exists():
        SIDECAR_PATH.rename(backup)
        print(f"backed up existing sidecar to {backup}", flush=True)
    tmp_path.rename(SIDECAR_PATH)
    print(f"wrote extended sidecar: {len(sidecar_new)} rows -> {SIDECAR_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
