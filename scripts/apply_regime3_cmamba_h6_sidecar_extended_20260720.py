"""Extend the CryptoMamba-h6 regime-transition stability sidecar
(data/ensemble/supervised/regime3_cryptomamba_pred_h6_nocurrent_20260531/) through the newly
extended training_features_2026_rebuilt.csv, using the already-frozen 2024-trained checkpoint
(inference only, no retraining) -- the CryptoMamba analogue of
apply_regime3_wide24_sidecar_extended_20260713.py for the HMM sidecar. No such extend-only script
existed for CryptoMamba prior to this; train_regime3_cryptomamba_pred_20260531.py always retrains
from scratch, so this script reuses its exact feature-prep/predict logic but loads the saved
state_dict/scaler/feature_cols from the .pt checkpoint instead of calling _fit().
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_regime3_cryptomamba_pred_20260531 import (  # noqa: E402
    CryptoMambaRegimePred,
    _add_rolling_stable_features,
    _add_volume_features,
    _current_path,
    _labels,
    _merge_current,
    _output,
    _predict,
    _read,
)

MODEL_PATH = ROOT / "data/ensemble/supervised/regime3_cryptomamba_pred_h6_nocurrent_20260531/regime3_cryptomamba_pred_h6_nocurrent_20260531_2024.pt"
CURRENT_DIR = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
# NOTE: the model's own DEFAULT_TRANSFORMS pointed at
# tmp/causal_regen_20260516/funding_clean_splits_20260528/training_features_2026_rebuilt.csv, a
# frozen snapshot last extended only to 2026-06-30. That snapshot diverges from the live
# data/splits/year_oos/training_features_2026_rebuilt.csv on a handful of columns/rows (median
# abs diff in scaled feature space is 0.0, i.e. the overwhelming majority of rows/cols match
# exactly, but a small number of rows have large outlier diffs on volume/OI-derived columns,
# consistent with the "funding_clean" pass having scrubbed a handful of anomalous raw values).
# Since the live file is the only one with the fresh July rows this script needs, and since
# median-case agreement is exact, this script uses the live file for both the historical
# reproducibility check and the fresh extension -- same "warn on known divergence, don't abort"
# precedent as apply_regime3_wide24_sidecar_extended_20260713.py.
SOURCE = ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"
OUT_DIR = ROOT / "data/ensemble/supervised/regime3_cryptomamba_pred_h6_nocurrent_20260531"
OUT_PATH = OUT_DIR / f"{SOURCE.stem}_regime3_cryptomamba_pred_h6_nocurrent_20260531.csv"


def main() -> int:
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    cols = list(ckpt["feature_cols"])
    medians = pd.Series(ckpt["feature_medians"])
    scaler_mean = np.asarray(ckpt["scaler_mean"], dtype=np.float64)
    scaler_scale = np.asarray(ckpt["scaler_scale"], dtype=np.float64)
    seq_len = int(ckpt["seq_len"])

    model = CryptoMambaRegimePred(
        n_features=len(cols), seq_len=seq_len, d_model=ckpt["d_model"],
        n_cblocks=ckpt["cblocks"], n_cmblocks=ckpt["cmblocks"], d_state=ckpt["d_state"], dropout=0.0,
    )
    model.load_state_dict(ckpt["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    frame = _merge_current(_add_volume_features(_add_rolling_stable_features(_read(SOURCE))), _current_path(CURRENT_DIR, SOURCE))
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise RuntimeError(f"extended source missing checkpoint feature columns: {missing}")

    raw = frame[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    filled = raw.fillna(medians).fillna(0.0)
    x = ((filled.to_numpy(dtype=np.float64) - scaler_mean) / scaler_scale).astype(np.float32)
    x = np.nan_to_num(x)

    future, transition, n = _labels(frame, ckpt["horizon"])
    idx = np.arange(seq_len - 1, len(frame), dtype=np.int64)
    proba = _predict(model, x, idx, seq_len, 1024, device)
    out = _output(frame, idx, proba)

    if OUT_PATH.exists():
        old = pd.read_csv(OUT_PATH, parse_dates=["timestamp"])
        overlap = old["timestamp"].isin(out["timestamp"])
        if overlap.any():
            merged_new = out.set_index("timestamp")
            merged_old = old.set_index("timestamp")
            common_ts = merged_old.index.intersection(merged_new.index)
            check_cols = [c for c in merged_old.columns if merged_old[c].dtype.kind in "fc"]
            max_diff = 0.0
            for c in check_cols:
                d = (merged_old.loc[common_ts, c].astype(float) - merged_new.loc[common_ts, c].astype(float)).abs().max()
                max_diff = max(max_diff, float(d) if pd.notna(d) else 0.0)
            print(f"reproducibility max abs diff on old range: {max_diff:.3e}", flush=True)
        backup = OUT_PATH.with_name(OUT_PATH.name + ".bak_pre_extend_20260720")
        old.to_csv(backup, index=False)
        print(f"backed up existing sidecar to {backup}", flush=True)

    out.to_csv(OUT_PATH, index=False)
    print(f"extended cmamba sidecar: {len(out)} rows ({out['timestamp'].iloc[0]}..{out['timestamp'].iloc[-1]}) -> {OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
