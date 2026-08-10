"""Regenerate the ETH redesigned-JM regime3 2026 output from the CLEAN 2026 source.

The build used tmp/causal_regen_20260516/funding_clean_splits_20260528/training_features_2026_
rebuilt.csv, which carries an 8-hour hole (2026-02-28 16:00 -> 2026-03-01 00:00, a 96-bar gap) and
stops earlier than the clean file: 51,746 rows vs 57,601. Two problems follow. The causal decode
treats the two bars either side of the hole as adjacent, so one state transition in the OOS window
is scored across a discontinuity; and the risk sidecar expects the clean-source row set, which is
why the 2026-08-09 lambda=4 swap had to do exactly this same regeneration.

Same treatment as that precedent: the frozen 2024-fit payload (winsorisation bounds, fill medians,
scaler, centroids, lambda, temperature, state->class matrix) is reused unchanged. Nothing is
refitted -- this is purely a causal re-transform over a cleaner row set, so the model is identical
and only its input coverage changes.

BTC needs no equivalent: all three BTC year files were checked and have zero internal gaps.
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

from scripts.jm_regime_redesign_lib_20260810 import (  # noqa: E402
    CLASSES3, _class_proba, _read, causal_decode_V, softmax_states,
)
from scripts.sparse_jm_feature_selection_20260810 import candidate_frame  # noqa: E402

TAG = "jmredesign_20260810"
SUP = ROOT / "data/ensemble/supervised"
MODEL = SUP / f"eth_regime3_current_jm_redesign_{TAG}_2024.joblib"
CLEAN_2026 = ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"
OUT = SUP / f"eth_regime3_current_hmm_{TAG}_2026_cleansource_maskedname.csv"


def main() -> None:
    payload = joblib.load(MODEL)
    cols = payload["panel_cols"]
    print(f"[1/3] frozen payload {MODEL.name}: {len(cols)} features, k={payload['jm_k']}, "
          f"lambda={payload['jm_lambda']:.3f}, T={payload['jm_temperature']:.4f}")

    frame = _read(CLEAN_2026)
    ts = frame["timestamp"]
    gaps = ts.diff().dt.total_seconds().div(300.0)
    print(f"[2/3] clean source {CLEAN_2026.name}: {len(frame):,} rows, "
          f"{ts.min():%Y-%m-%d} .. {ts.max():%Y-%m-%d}, "
          f"internal gaps>1 bar: {int((gaps > 1).sum())}")

    panel = candidate_frame(frame)
    missing = [c for c in cols if c not in panel.columns]
    if missing:
        raise SystemExit(f"clean source is missing required features: {missing}")

    medians = pd.Series(payload["feature_medians"])
    lo = pd.Series(payload["winsor_lower"])
    hi = pd.Series(payload["winsor_upper"])
    raw = panel[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = payload["scaler"].transform(
        raw.fillna(medians).fillna(0.0).clip(lower=lo, upper=hi, axis=1)).astype(np.float64)

    V = causal_decode_V(x, payload["jm_mu"], payload["jm_lambda"])
    proba = _class_proba(softmax_states(V, payload["jm_temperature"]),
                         payload["state_class_matrix"])

    prefix = payload["output_column_prefix"]
    out = pd.DataFrame({"timestamp": ts.reset_index(drop=True)})
    for i, name in enumerate(CLASSES3):
        out[f"{prefix}{name}_prob"] = proba[:, i]
    s = np.sort(proba, axis=1)
    out[f"{prefix}confidence"] = s[:, -1]
    out[f"{prefix}entropy"] = (-np.sum(proba * np.log(np.clip(proba, 1e-12, None)), axis=1)
                               / np.log(len(CLASSES3)))
    out[f"{prefix}margin"] = s[:, -1] - s[:, -2]
    out.to_csv(OUT, index=False)

    pred = np.argmax(proba, axis=1)
    share = np.bincount(pred, minlength=3) / len(pred)
    print(f"[3/3] -> {OUT.name}  ({len(out):,} rows, conf mean {out[f'{prefix}confidence'].mean():.3f}, "
          + " ".join(f"{CLASSES3[i]}={share[i]:.3f}" for i in range(3)) + ")")


if __name__ == "__main__":
    main()
