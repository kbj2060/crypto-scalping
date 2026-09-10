"""Regenerate the JM lambda=4 regime3 2026 output from the CLEAN source
(data/splits/year_oos/training_features_2026_rebuilt.csv, verified zero internal gaps) instead of
the gappy tmp/causal_regen_20260516/funding_clean_splits_20260528/training_features_2026_rebuilt.csv
(has an 8-hour gap 2026-02-28 16:00 -> 2026-03-01 00:00) the original build used. Reuses the SAME
frozen JM payload (scaler, centroids, lambda, temperature, state_class_matrix) fit on 2024 data --
no refitting, purely a cleaner-source causal re-transform.
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

from scripts.experiment_regime3_current_hmm_wide24_20260529 import _class_proba, _with_features  # noqa: E402
from scripts.build_eth_regime3_jm_lam4_20260809 import causal_decode_soft  # noqa: E402

PAYLOAD_PATH = ROOT / "data/ensemble/supervised/eth_regime3_current_jm_jmlam4_20260809_2024.joblib"
CLEAN_SRC_2026 = ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"
OUT_PATH = ROOT / "data/ensemble/supervised/eth_regime3_current_hmm_jmlam4_20260809_2026_cleansource_maskedname.csv"
PREFIX_STEM = "regime3_current_sensitive"


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def main() -> int:
    payload = joblib.load(PAYLOAD_PATH)
    cols = payload["feature_cols"]
    scaler = payload["scaler"]
    mu = payload["jm_mu"]
    lam = payload["jm_lambda"]
    temperature = payload["jm_temperature"]
    state_class = payload["state_class_matrix"]
    classes = payload["classes"]

    frame = _read(CLEAN_SRC_2026)
    print(f"clean source: rows={len(frame)} range=({frame['timestamp'].min()}, {frame['timestamp'].max()})")
    work = _with_features(frame, cols)
    med = pd.Series(payload["feature_medians"])
    x_raw = work[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    x_obs = scaler.transform(x_raw)
    _, state_prob = causal_decode_soft(x_obs, mu, lam, temperature)
    proba = _class_proba(state_prob, state_class)

    out = pd.DataFrame({"timestamp": work["timestamp"].reset_index(drop=True)})
    prefix = f"{PREFIX_STEM}_wide24_"
    for i, name in enumerate(classes):
        out[f"{prefix}{name}_prob"] = proba[:, i]
    sp = np.sort(proba, axis=1)
    out[f"{prefix}confidence"] = sp[:, -1]
    out[f"{prefix}entropy"] = -np.sum(proba * np.log(np.clip(proba, 1e-12, None)), axis=1) / np.log(len(classes))
    out[f"{prefix}margin"] = sp[:, -1] - sp[:, -2]

    diffs = out["timestamp"].diff()
    gaps = diffs[diffs > pd.Timedelta(minutes=5)]
    print(f"output gaps: {len(gaps)}")
    out.to_csv(OUT_PATH, index=False)
    print(f"rows={len(out)} range=({out['timestamp'].min()}, {out['timestamp'].max()}) -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
