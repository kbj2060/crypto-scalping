#!/usr/bin/env python3
"""One-time migration: re-pickle the regime3 "current" HMM artifact under Odyssey's own vendored
`GaussianStateModel` class instead of the original `scripts.retrain_clean_regime_hmm_20260517`
class, so `trading_bot_modules/odyssey_regime3_live.py` has zero import dependency on any
`scripts/*` training script.

This is a MECHANICAL PARAMETER COPY, not a retrain: it loads the existing fitted model (pi_, A_,
mu_, var_, log_likelihood_ plus the constructor args n_states/n_iter/seed/min_var/sticky), builds a
new instance of the vendored class with those exact same values, and re-dumps the same payload dict
(feature_cols/feature_medians/scaler/state_class_matrix/classes all unchanged) with only `model`
swapped to the new instance. Self-verifies `filter_proba()` is bit-identical between old and new
before writing anything, and again after reading the new file back from disk.

Usage: python scripts/migrate_regime3_hmm_artifact_to_odyssey_native_20260816.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SRC_PATH = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/regime3_current_sensitive_hmm_wide24_2024.joblib"
DST_PATH = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/regime3_current_sensitive_hmm_wide24_2024_odyssey_native.joblib"


def main() -> None:
    # Original class -- only needed here, once, to unpickle the source artifact.
    from scripts.retrain_clean_regime_hmm_20260517 import GaussianStateModel as OriginalGaussianStateModel  # noqa: F401
    from trading_bot_modules.odyssey_regime3_live import GaussianStateModel as OdysseyGaussianStateModel

    print(f"[migrate] loading source artifact: {SRC_PATH}")
    payload = joblib.load(SRC_PATH)
    old_model = payload["model"]
    print(f"[migrate] source model class: {type(old_model).__module__}.{type(old_model).__qualname__}")
    print(f"[migrate] n_states={old_model.n_states} n_iter={old_model.n_iter} seed={old_model.seed} "
          f"min_var={old_model.min_var} sticky={old_model.sticky}")

    new_model = OdysseyGaussianStateModel(
        n_states=int(old_model.n_states), n_iter=int(old_model.n_iter), seed=int(old_model.seed),
        min_var=float(old_model.min_var), sticky=float(old_model.sticky),
    )
    new_model.pi_ = np.array(old_model.pi_, copy=True)
    new_model.A_ = np.array(old_model.A_, copy=True)
    new_model.mu_ = np.array(old_model.mu_, copy=True)
    new_model.var_ = np.array(old_model.var_, copy=True)
    new_model.log_likelihood_ = list(old_model.log_likelihood_)

    # Self-verify BEFORE writing anything: bit-identical filter_proba on a deterministic synthetic
    # observation matrix shaped like the real feature space.
    n_features = old_model.mu_.shape[1]
    rng = np.random.default_rng(20260816)
    x_test = rng.normal(size=(500, n_features))
    old_proba = old_model.filter_proba(x_test)
    new_proba = new_model.filter_proba(x_test)
    if not np.array_equal(old_proba, new_proba):
        max_diff = float(np.max(np.abs(old_proba - new_proba)))
        raise RuntimeError(f"pre-write parity check FAILED: max_abs_diff={max_diff} -- aborting, nothing written")
    print(f"[migrate] pre-write parity check passed: filter_proba bit-identical on {x_test.shape[0]} synthetic rows")

    new_payload = dict(payload)
    new_payload["model"] = new_model
    new_payload["model_class_migrated_from"] = f"{type(old_model).__module__}.{type(old_model).__qualname__}"
    new_payload["migration_script"] = "scripts/migrate_regime3_hmm_artifact_to_odyssey_native_20260816.py"

    DST_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(new_payload, DST_PATH)
    print(f"[migrate] wrote {DST_PATH}")

    # Re-verify by reading the NEW file back from a fresh load (still within this process, but
    # exercises the real joblib.load() path against the file actually on disk).
    reloaded = joblib.load(DST_PATH)
    print(f"[migrate] reloaded model class: {type(reloaded['model']).__module__}.{type(reloaded['model']).__qualname__}")
    reloaded_proba = reloaded["model"].filter_proba(x_test)
    if not np.array_equal(old_proba, reloaded_proba):
        max_diff = float(np.max(np.abs(old_proba - reloaded_proba)))
        raise RuntimeError(f"post-write reload parity check FAILED: max_abs_diff={max_diff}")
    print(f"[migrate] post-write reload parity check passed: filter_proba bit-identical")
    for key in ("feature_cols", "classes", "state_class_matrix"):
        old_v = payload.get(key)
        new_v = reloaded.get(key)
        same = np.array_equal(np.asarray(old_v), np.asarray(new_v)) if key != "classes" else (list(old_v) == list(new_v))
        print(f"[migrate] {key} unchanged: {same}")
    print("[migrate] DONE")


if __name__ == "__main__":
    main()
