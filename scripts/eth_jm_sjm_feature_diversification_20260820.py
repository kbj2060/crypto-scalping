#!/usr/bin/env python3
"""ETH 3-state regime classifier -- Sparse Jump Model (SJM) feature-diversification sweep
(2026-08-20).

Extends scripts/eth_jm_paperfaithful_build_20260820.py's paper-faithful discrete Jump Model
(JM) build to test whether a Sparse Jump Model (SJM) -- the feature-selection variant of the
same model family -- benefits from a diversified 14-feature candidate pool instead of the
original 9.

Why SJM, not a hand-mixed feature set (paper-faithfulness constraint)
-----------------------------------------------------------------------
This sub-investigation operates under an explicit standing user constraint stated verbatim
earlier in the session: "논문을 제대로 코드로 구현하고 절대 논문과 달라서는 안돼" (implement
the paper faithfully in code, it must NEVER differ from the paper). Naively mixing extra
indicators into the original 9-feature build would violate that constraint. Sparse Jump
Models are themselves a specific published method -- Nystrup, Kolm & Lindstrom, "Feature
Selection in Jump Models," Expert Systems with Applications, 2021 -- designed exactly for
"you have a larger candidate feature pool and want the MODEL ITSELF to select the relevant
subset via an L0-constrained (Lasso-relaxed) sparse weighting," rather than a human
hand-picking features. The same reference `jumpmodels` PyPI package (v0.1.1,
author-maintained) that backs the paper-faithful plain-JM build also implements SJM
faithfully at `jumpmodels.sparse_jump.SparseJumpModel` (confirmed real/complete by reading
the installed source directly -- not a stub). This is also the paper's own answer to a
DIFFERENT earlier failure this session: naively feeding a plain (non-sparse) JumpModel a
154-column engineered feature set collapsed to 99.9% single-state / chance-level
balanced_accuracy (diagnosed as likely curse-of-dimensionality). So: use SparseJumpModel,
not a hand-rolled feature-mixing hack.

This script IMPORTS (never modifies) scripts/eth_jm_paperfaithful_build_20260820.py for
`build_features()` (verbatim reuse of the original 9-feature EWM family) and the halflife
grid `HALFLIVES_BARS`, and imports `_read/_labels/_eval/CLASSES3/LABEL_CONFIGS` from
scripts/experiment_regime3_current_hmm_wide24_20260529.py exactly as the original does.

Run with:
    /home/kbj20/anaconda3/envs/quant_ai/bin/python3 scripts/eth_jm_sjm_feature_diversification_20260820.py

Output: tmp/eth_jm_sjm_feature_diversification_20260820/report.json (+ progress on stdout).
Note: a hard ~90-minute wall-clock budget gate runs BEFORE the full max_feats grid (see
Phase 1 below) -- if a calibration probe shows the full grid cannot fit in budget, the script
stops after Phase 1 and report.json contains calibration/timing info only (full_grid_run:
false), not a full grid result.

JUDGMENT CALLS (also written to report.json["judgment_calls"] verbatim):
  1. Realized-vol formula: ewm_vol_hl_t = sqrt(EWM_mean(log_ret^2, halflife=hl)_t), same 3
     halflives as the existing family (72/288/864 bars = 6h/1d/3d, imported from the original
     script's HALFLIVES_BARS, not re-hardcoded). Reuses the exact `log_ret` Series returned by
     build_features() rather than recomputing np.log(close).diff() a second time, guaranteeing
     the realized-vol feature and the fit()'s ret_ser argument are built from one identical
     causal log-return source.
  2. Causality of the new ewm_vol_hl* feature was VERIFIED (not assumed): recomputed on a
     PREFIX-ONLY slice of the series at one arbitrary interior spot-check bar (index 150000,
     which falls inside 2025 -- neither at the TRAIN boundary nor at series start/end) and
     confirmed to match the full-series value exactly (np.allclose, atol=1e-12). See
     report.json["vol_feature_causality_check"].
  3. hurst_48/hurst_288 are pulled directly (NOT recomputed) from the precomputed columns
     already present in the per-year training_features CSVs, via the same direct
     column-selection technique the original script uses for `close` (both read off the same
     concatenated+index-set `full` frame). Checked for NaN/inf BEFORE trusting them, both via
     a standalone pre-flight check and via a runtime assertion embedded in this script
     (report.json["hurst_nan_inf_check"]): VERIFIED zero NaN/inf cells in hurst_48/hurst_288
     across all 3 years -- no floor/fill policy (analogous to SORTINO_DD_FLOOR) was needed.
     Secondary observation (not requiring intervention): hurst_48/hurst_288 show a constant
     exactly-0.5 warmup run at the very start of the 2024 and 2026 files only (23 bars for
     hurst_48, 143 bars for hurst_288) -- absent from the 2025 file, suggesting the per-file
     CSVs may have been generated with differing lookback/cold-start conventions. This is a
     finite, in-distribution placeholder value (unlike the original script's Sortino blowup to
     O(1e8-1e9)), so it does not destabilize TRAIN-fit clipper/scaler statistics -- flagged
     for transparency, not treated as a defect.
  4. Consequence of (3): hurst_48/hurst_288 are COLD-STARTED PER FILE (whatever convention
     produced the source CSVs), unlike the EWM-based features (9 original + 3 new realized-
     vol) which are computed CONTINUOUSLY across the concatenated 2024+2025+2026 series. This
     is a structural consequence of reusing precomputed columns via direct selection rather
     than a deliberate modeling choice -- flagged as an asymmetry (loosely mirroring, but not
     identical in cause to, the original script's own judgment-call-5/6 continuous-vs-per-file
     asymmetry between JM input features and ground truth).
  5. TRAIN window = 2024 ONLY (not 2024+2025). Deliberate COST tradeoff specific to this task:
     SJM's outer coordinate-descent loop re-fits a full n_init_jm=10 JumpModel every outer
     iteration, so doubling TRAIN rows would roughly double an already-expensive run. This is
     despite a separate experiment this session already showing extending TRAIN to 2024+2025
     barely changes plain-JM's 2026 OOS numbers -- stated explicitly per task instructions,
     not silently done.
  6. max_feats grid = {4.0, 8.0, 14.0} exactly as specified (very sparse / moderate /
     ~everything-included sanity upper bound) -- not expanded or narrowed.
  7. The calibration timing probe (max_feats=8.0, n_init_jm=10, max_iter TEMPORARILY capped at
     5) is a throwaway timing measurement only. Even though max_feats=8.0 coincides with one
     of the 3 official grid points, its (possibly non-converged) fit result is NOT reused as
     the official max_feats=8.0 grid result -- max_feats=8.0 is independently refit with the
     full max_iter=30 outer cap in the main grid loop. Small extra compute cost (one extra
     <=5-outer-iteration fit) traded for correctness/clarity.
  8. Go/no-go rule: computed a HARD worst-case wall-clock upper bound = (measured seconds per
     outer iteration from calibration) * (max_iter=30, a STRICT cap in SJM's own
     `while (n_iter < max_iter ...)` loop, confirmed by reading sparse_jump.py) * (3 grid
     points), compared against the ~90-minute budget stated in the task. This is a true upper
     bound, not a soft estimate -- actual runtime will likely be lower since SJM typically
     converges via tol_w before the cap; the calibration run's own converged-before-cap
     outcome is reported as corroborating evidence, not as the deciding number itself.
  9. Evaluation report fields were scoped exactly to what the task's "Report" section listed
     (balanced_accuracy, flip_rate, median_state_duration_bars, recall by class, true_counts,
     pred_counts, per year). The original build script's ADDITIONAL non-causal in-sample
     batch-DP reference block (labels_-based) was intentionally OMITTED here since it was not
     requested for this comparison -- a deliberate scope-narrowing, not an oversight.
  10. Reused the original script's VERIFIED cumret-sort assertion mechanism (ret_*freq
      non-increasing check before trusting the bull/chop/bear index mapping) unchanged for SJM
      fits, since SJM delegates ret_ser-based sorting to the same base.py::sort_param_dict
      code path as plain JM (SJM.fit() calls jm.fit(..., sort_by=sort_by) each outer
      iteration) -- re-verified applicable to SJM by reading sparse_jump.py, not assumed by
      analogy alone.
  11. VERIFIED (not assumed) that `sjm.predict_proba_online(X)` correctly incorporates the
      learned per-feature `feat_weights` automatically: confirmed by reading jump.py that
      `JumpModel.fit(X, feat_weights=...)` stores `self.feat_weights`, and
      `predict_proba_online` -> `check_X_predict_func` -> `check_X_with_feat_weights`
      re-applies that SAME stored `feat_weights` to X before the distance computation. This
      means the raw (clipped+scaled, un-reweighted) 14-column X_eval matrices are passed
      directly into `sjm.predict_proba_online()` below, matching the task's own claim, now
      source-verified rather than trusted at face value.
  12. `sjm.w` (not `sjm.feat_weights`) is reported as the PRIMARY feature-weight vector
      (non-negative, sum-of-squares=1 scale, directly interpretable against the
      max_feats/kappa^2 constraint per the package's own docstring); `sjm.feat_weights`
      (=sqrt(w)) is also reported for completeness. The nonzero/zero pattern is IDENTICAL
      between the two (sqrt(0)=0), so "selected vs excluded" verdicts do not depend on which
      is used.
  13. jump_penalty=50.0 is passed literally, matching the established lambda=50 plain-JM
      convention. SparseJumpModel.init_jm() automatically rescales this internally to
      50.0/sqrt(14)~=13.36 for its internal JumpModel (confirmed by reading sparse_jump.py:
      `jump_penalty = self.jump_penalty / np.sqrt(self.n_features_all)`) -- a package
      behavior, not a manual adjustment made here. Flagged since the EFFECTIVE penalty
      therefore differs numerically from the 9-feature plain-JM run, even though both start
      from the identical literal input "50" the task specified.
  K fixed at 3 (not searched), matching the plain-JM convention. cont=False (discrete SJM
  only) -- continuous/soft SJM is out of scope for this pass, matching the original script's
  own K/cont scoping.
"""
from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# mamba_ssm stub: required transitively by scripts.train_regime3_hmm_mamba_20260529 and
# scripts.experiment_regime3_current_hmm_wide24_20260529 (via retrain_clean_regime_hmm_20260517
# etc.), which we import below ONLY to reuse `_read()` and the ground-truth labeling / eval
# functions. mamba_ssm itself is never called by anything in this script. (Same stub as the
# original build script -- also set up transitively when we import build_features() from it,
# but kept here too so this script is self-contained regardless of import order.)
sys.modules.setdefault("mamba_ssm", types.ModuleType("mamba_ssm"))
sys.modules["mamba_ssm"].Mamba = object

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_regime3_hmm_mamba_20260529 import _read  # noqa: E402
from scripts.experiment_regime3_current_hmm_wide24_20260529 import (  # noqa: E402
    CLASSES3,
    LABEL_CONFIGS,
    _eval,
    _labels,
)
from scripts.eth_jm_paperfaithful_build_20260820 import (  # noqa: E402
    build_features,
    HALFLIVES_BARS,
)

from jumpmodels.sparse_jump import SparseJumpModel  # noqa: E402
from jumpmodels.preprocess import DataClipperStd, StandardScalerPD  # noqa: E402

# ----------------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------------
DATA_DIR = ROOT / "data/splits/year_oos"
CSV_PATHS = {
    "2024": DATA_DIR / "training_features_2024.csv",
    "2025": DATA_DIR / "training_features_2025.csv",
    "2026": DATA_DIR / "training_features_2026_rebuilt.csv",
}
OUT_DIR = ROOT / "tmp/eth_jm_sjm_feature_diversification_20260820"
REPORT_PATH = OUT_DIR / "report.json"

LABEL_MODE = "balancedish_adx16_slope15_bb012"
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES3)}  # bull=0, bear=1, chop=2

N_COMPONENTS = 3
JUMP_PENALTY = 50.0  # matches the established lambda=50 plain-JM convention; see judgment call 13
SEED = 7529
CLIP_MUL = 3.0

MAX_FEATS_GRID = [4.0, 8.0, 14.0]  # see judgment call 6
N_INIT_JM = 10  # matches the plain-JM N_INIT convention, not reduced -- see task cost note
MAX_ITER_JM = 1000
TOL_JM = 1e-8
MAX_ITER_OUTER = 30  # SJM's own default outer coordinate-descent cap, left as default
TOL_W = 1e-4  # SJM's own default

CALIB_MAX_FEATS = 8.0
CALIB_MAX_ITER_OUTER = 5  # TEMPORARY cap, timing probe only -- see judgment call 7
BUDGET_SECONDS = 90 * 60  # ~90-minute wall-clock ceiling stated in the task

CAUSALITY_SPOT_IDX = 150_000  # arbitrary interior index (falls inside 2025); see judgment call 2


# ----------------------------------------------------------------------------------
# NEW feature family: EWM realized volatility (total vol, sibling of the existing
# EWM-downside-deviation family in build_features()).
# ----------------------------------------------------------------------------------


def build_realized_vol_features(close: pd.Series, log_ret: pd.Series) -> pd.DataFrame:
    """NEW (not in the original 9-feature build): EWM realized-volatility features -- the
    natural sibling of the existing EWM downside-deviation family (TOTAL vol instead of
    downside-only vol). Same causal EWM machinery as build_features(), just without the
    downside `.clip(upper=0.0)` restriction:
        ewm_vol_hl_t = sqrt( EWM_mean(log_ret^2, halflife=hl)_t )
    `log_ret` is expected to be the SAME contemporaneous causal log-return Series returned by
    build_features() (reused, not recomputed a second time from `close` -- see judgment call 1).
    Returns a DataFrame [3 cols], indexed identically to `close`.
    """
    sq = log_ret.astype(np.float64) ** 2
    cols: dict[str, pd.Series] = {}
    for hl in HALFLIVES_BARS:
        cols[f"ewm_vol_hl{hl}"] = np.sqrt(sq.ewm(halflife=hl).mean())
    feats = pd.DataFrame(cols, index=close.index)
    assert not feats.isna().any().any(), "unexpected NaN in causal EWM realized-vol features"
    assert np.isfinite(feats.to_numpy()).all(), "unexpected inf in causal EWM realized-vol features"
    return feats


def _verify_vol_feature_causality(
    close: pd.Series, log_ret: pd.Series, full_vol: pd.DataFrame, spot_idx: int
) -> dict[str, Any]:
    """Empirical causality spot-check for build_realized_vol_features(): recompute the
    feature on a PREFIX-ONLY slice of the series (bars 0..spot_idx inclusive) and confirm it
    matches the full-series value at the same bar exactly. pandas .ewm() is causal by
    construction (depends only on past+current rows), but we verify empirically rather than
    assume -- see judgment call 2."""
    assert spot_idx < len(close), f"spot_idx={spot_idx} out of range (len(close)={len(close)})"
    prefix_close = close.iloc[: spot_idx + 1]
    prefix_log_ret = log_ret.iloc[: spot_idx + 1]
    prefix_vol = build_realized_vol_features(prefix_close, prefix_log_ret)
    full_row = full_vol.iloc[spot_idx]
    prefix_row = prefix_vol.iloc[-1]
    match = bool(
        np.allclose(
            full_row.to_numpy(dtype=np.float64), prefix_row.to_numpy(dtype=np.float64), rtol=0.0, atol=1e-12
        )
    )
    result = {
        "spot_idx": int(spot_idx),
        "spot_timestamp": str(close.index[spot_idx]),
        "full_series_value": full_row.to_dict(),
        "prefix_only_value": prefix_row.to_dict(),
        "match": match,
    }
    print(f"[causality-check] {result}", flush=True)
    assert match, "ewm_vol_hl* feature is NOT causal -- prefix-slice recomputation diverged from the full-series value"
    return result


def _fit_with_iter_count(
    sjm: SparseJumpModel, X: pd.DataFrame, ret_ser: pd.Series, sort_by: str = "cumret"
) -> tuple[float, int]:
    """Fit `sjm` and return (wall_clock_seconds, actual_outer_iterations_run). SJM's own
    fit() loop tracks `n_iter` as a local variable only (never saved as an attribute), so we
    capture it by wrapping the instance's `print_log()` method -- confirmed by reading
    sparse_jump.py that `self.print_log(n_iter, BCSS, w)` is called unconditionally every
    outer iteration regardless of verbosity (the `if self.verbose:` gate is INSIDE
    print_log) -- rather than patching the installed package itself."""
    counter = {"n_iter": 0}
    orig_print_log = sjm.print_log

    def _wrapped(n_iter: int, BCSS: Any, w: Any) -> Any:
        counter["n_iter"] = n_iter
        return orig_print_log(n_iter, BCSS, w)

    sjm.print_log = _wrapped  # type: ignore[method-assign]
    t0 = time.time()
    sjm.fit(X, ret_ser=ret_ser, sort_by=sort_by)
    elapsed = time.time() - t0
    return elapsed, counter["n_iter"]


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    if isinstance(o, pd.Series):
        return o.to_dict()
    raise TypeError(f"not JSON serializable: {type(o)}")


JUDGMENT_CALLS = [
    "1. Realized-vol formula: ewm_vol_hl_t = sqrt(EWM_mean(log_ret^2, halflife=hl)_t), same 3 "
    "halflives as the existing family (72/288/864 bars, imported from the original script's "
    "HALFLIVES_BARS). Reuses the exact log_ret Series returned by build_features() rather than "
    "recomputing np.log(close).diff() a second time.",
    "2. Causality of ewm_vol_hl* was VERIFIED (not assumed): recomputed on a PREFIX-ONLY slice "
    "at one arbitrary interior spot-check bar (index 150000, inside 2025) and confirmed to "
    "match the full-series value exactly (np.allclose atol=1e-12). See "
    "vol_feature_causality_check in this report.",
    "3. hurst_48/hurst_288 are pulled directly (NOT recomputed) from precomputed columns "
    "already in the per-year CSVs, via the same direct column-selection technique the original "
    "script uses for close. VERIFIED zero NaN/inf cells across all 3 years (see "
    "hurst_nan_inf_check) -- no floor/fill policy was needed. Secondary observation: constant "
    "exactly-0.5 warmup run at the start of 2024/2026 files only (23 bars hurst_48, 143 bars "
    "hurst_288; absent in 2025), a finite in-range placeholder, not a numerical blowup -- "
    "flagged for transparency, not treated as a defect.",
    "4. Consequence of (3): hurst_48/hurst_288 are COLD-STARTED PER FILE (whatever convention "
    "produced the source CSVs), unlike the EWM-based features (9 original + 3 new realized-vol) "
    "which are computed CONTINUOUSLY across the concatenated 2024+2025+2026 series -- a "
    "structural consequence of reusing precomputed columns, not a deliberate modeling choice.",
    "5. TRAIN window = 2024 ONLY (not 2024+2025). Deliberate COST tradeoff: SJM's outer "
    "coordinate-descent loop re-fits a full n_init_jm=10 JumpModel every outer iteration, so "
    "doubling TRAIN rows would roughly double an already-expensive run -- despite a separate "
    "experiment this session showing extending TRAIN barely changes plain-JM's 2026 OOS numbers.",
    "6. max_feats grid = {4.0, 8.0, 14.0} exactly as specified (very sparse / moderate / "
    "~everything-included sanity upper bound) -- not expanded or narrowed.",
    "7. The calibration timing probe (max_feats=8.0, n_init_jm=10, max_iter TEMPORARILY capped "
    "at 5) is a throwaway timing measurement only. Its (possibly non-converged) result is NOT "
    "reused as the official max_feats=8.0 grid result even though the value coincides -- "
    "max_feats=8.0 is independently refit with the full max_iter=30 cap in the main grid loop.",
    "8. Go/no-go rule: HARD worst-case wall-clock bound = (measured seconds/outer-iteration "
    "from calibration) * (max_iter=30, a STRICT cap in SJM's own while-loop) * (3 grid points), "
    "compared against the ~90-minute task budget. This is a true upper bound, not a soft "
    "estimate; actual runtime will likely be lower since SJM typically converges via tol_w "
    "before the cap.",
    "9. Evaluation report fields were scoped exactly to the task's 'Report' section list "
    "(balanced_accuracy, flip_rate, median_state_duration_bars, recall, true/pred_counts per "
    "year). The original script's additional non-causal in-sample batch-DP reference block was "
    "intentionally OMITTED here as not requested for this comparison.",
    "10. Reused the original script's VERIFIED cumret-sort assertion (ret_*freq non-increasing "
    "check before trusting the bull/chop/bear index mapping) unchanged for SJM fits -- "
    "re-verified applicable to SJM by reading sparse_jump.py (SJM.fit() calls "
    "jm.fit(..., sort_by=sort_by) each outer iteration, same base.py::sort_param_dict path).",
    "11. VERIFIED (not assumed) that sjm.predict_proba_online(X) correctly incorporates the "
    "learned per-feature feat_weights automatically -- confirmed by reading jump.py "
    "(JumpModel.fit stores self.feat_weights; predict_proba_online re-applies it via "
    "check_X_with_feat_weights before the distance computation). Raw (clipped+scaled, "
    "un-reweighted) 14-column X_eval matrices are passed directly into predict_proba_online().",
    "12. sjm.w (not sjm.feat_weights) is reported as the PRIMARY feature-weight vector "
    "(non-negative, sum-of-squares=1, directly interpretable against the max_feats/kappa^2 "
    "constraint); sjm.feat_weights (=sqrt(w)) is also reported. Nonzero/zero pattern is "
    "identical between the two.",
    "13. jump_penalty=50.0 is passed literally (matching the established lambda=50 plain-JM "
    "convention). SparseJumpModel.init_jm() automatically rescales this internally to "
    "50.0/sqrt(14)~=13.36 for its internal JumpModel (confirmed by reading sparse_jump.py) -- "
    "a package behavior, not a manual adjustment made here; the EFFECTIVE penalty therefore "
    "differs numerically from the 9-feature plain-JM run despite both starting from the "
    "literal '50' the task specified.",
    "K fixed at 3, matching the plain-JM convention. cont=False (discrete SJM only) -- "
    "continuous/soft SJM out of scope for this pass.",
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ---- ground truth: PER-FILE (cold-started each year), same convention as the original
    # build script's own judgment call 6. ----
    raw_frames: dict[str, pd.DataFrame] = {}
    gt_by_year: dict[str, pd.Series] = {}
    for year, path in CSV_PATHS.items():
        df = _read(path)
        raw_frames[year] = df
        gt = _labels(df, LABEL_MODE)
        gt_by_year[year] = pd.Series(gt, index=pd.DatetimeIndex(df["timestamp"], name="timestamp"))
        print(f"[load] {year}: rows={len(df)} range=[{df['timestamp'].iloc[0]} .. {df['timestamp'].iloc[-1]}]", flush=True)

    # ---- concatenated frame: source for both the CONTINUOUS EWM features and the
    # directly-selected (per-file cold-started) hurst columns -- see judgment calls 1/3/4. ----
    full = pd.concat(raw_frames.values(), ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    dup = int(full["timestamp"].duplicated().sum())
    if dup:
        raise RuntimeError(f"unexpected duplicate timestamps across year files: {dup}")
    full = full.set_index(pd.DatetimeIndex(full["timestamp"]))
    close = pd.to_numeric(full["close"], errors="coerce")
    assert close.notna().all(), "NaN close price in concatenated series"

    year_of_idx = full.index.year
    year_masks = {y: (year_of_idx == int(y)) for y in CSV_PATHS}

    # ---- 9 existing features: verbatim reuse of build_features() (import, not reimplement) ----
    feats9, log_ret = build_features(close)
    print(f"[features] existing 9-feature family (imported build_features): {list(feats9.columns)}", flush=True)

    # ---- 3 NEW realized-vol features ----
    feats_vol3 = build_realized_vol_features(close, log_ret)
    print(f"[features] NEW realized-vol 3-feature family: {list(feats_vol3.columns)}", flush=True)
    causality_check = _verify_vol_feature_causality(close, log_ret, feats_vol3, CAUSALITY_SPOT_IDX)

    # ---- 2 hurst features: precomputed columns already present in the per-year CSVs, pulled
    # directly off `full` (same direct-column-selection technique used for `close`). ----
    hurst_df = full[["hurst_48", "hurst_288"]].apply(pd.to_numeric, errors="coerce")
    hurst_nan_report: dict[str, Any] = {}
    for year, mask in year_masks.items():
        arr = hurst_df.loc[mask].to_numpy(dtype=np.float64)
        n_nan = int(np.isnan(arr).any(axis=1).sum())
        n_inf = int(np.isinf(arr).any(axis=1).sum())
        hurst_nan_report[year] = {"rows": int(mask.sum()), "n_nan_rows": n_nan, "n_inf_rows": n_inf}
        print(f"[hurst-check] {year}: rows={int(mask.sum())} n_nan_rows={n_nan} n_inf_rows={n_inf}", flush=True)
    assert not hurst_df.isna().any().any(), "unexpected NaN in hurst_48/hurst_288 -- floor/fill policy needed, see judgment calls"
    assert np.isfinite(hurst_df.to_numpy(dtype=np.float64)).all(), "unexpected inf in hurst_48/hurst_288"
    print("[hurst-check] VERIFIED: no NaN/inf found in hurst_48/hurst_288 in any year -- no floor/fill policy needed.", flush=True)

    feats = pd.concat([feats9, feats_vol3, hurst_df], axis=1)
    assert list(feats.columns) == list(feats9.columns) + list(feats_vol3.columns) + ["hurst_48", "hurst_288"]
    assert feats.shape[1] == 14, f"expected 14 candidate features, got {feats.shape[1]}: {list(feats.columns)}"
    print(f"[features] FULL 14-feature candidate pool: {list(feats.columns)}", flush=True)

    # sanity: continuous-feature year slice must align 1:1 (by timestamp) with the
    # independently-read per-file ground truth for that year.
    for year, mask in year_masks.items():
        idx_cont = feats.loc[mask].index
        idx_file = gt_by_year[year].index
        if not idx_cont.equals(idx_file):
            missing = idx_cont.difference(idx_file)
            extra = idx_file.difference(idx_cont)
            raise RuntimeError(f"{year}: continuous-vs-per-file index mismatch; missing={len(missing)} extra={len(extra)}")

    train_mask = year_masks["2024"]
    X_train_raw = feats.loc[train_mask]
    ret_train = log_ret.loc[train_mask]

    # ---- preprocessing: DataClipperStd(mul=3.0) -> StandardScalerPD(), fit TRAIN(2024) only,
    # transform elsewhere, over the full 14-column matrix. ----
    clipper = DataClipperStd(mul=CLIP_MUL)
    scaler = StandardScalerPD()
    X_train = scaler.fit_transform(clipper.fit_transform(X_train_raw))
    X_by_year = {year: scaler.transform(clipper.transform(feats.loc[mask])) for year, mask in year_masks.items()}
    print(f"[preprocess] clip(mul={CLIP_MUL}) + StandardScalerPD fit on TRAIN=2024 ({len(X_train)} rows), 14 cols", flush=True)

    report: dict[str, Any] = {
        "model_id": "eth_jm_sjm_feature_diversification_20260820",
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "package": "jumpmodels==0.1.1 (pip, author-maintained reference implementation), jumpmodels.sparse_jump.SparseJumpModel",
        "paper": "Nystrup, Kolm & Lindstrom (2021), 'Feature Selection in Jump Models', Expert Systems with Applications",
        "classes": CLASSES3,
        "label_mode": LABEL_MODE,
        "label_config": LABEL_CONFIGS[LABEL_MODE],
        "candidate_feature_cols": list(feats.columns),
        "n_components": N_COMPONENTS,
        "jump_penalty": JUMP_PENALTY,
        "random_state": SEED,
        "n_init_jm": N_INIT_JM,
        "max_iter_jm": MAX_ITER_JM,
        "tol_jm": TOL_JM,
        "max_iter_outer": MAX_ITER_OUTER,
        "tol_w": TOL_W,
        "clip_mul": CLIP_MUL,
        "max_feats_grid": MAX_FEATS_GRID,
        "train_window": "2024 (full year) ONLY -- deliberate cost tradeoff, see judgment call 5",
        "eval_years": list(CSV_PATHS),
        "rows_per_year": {y: int(m.sum()) for y, m in year_masks.items()},
        "sample_type_by_year": {"2024": "in_sample_train", "2025": "out_of_sample", "2026": "out_of_sample"},
        "vol_feature_causality_check": causality_check,
        "hurst_nan_inf_check": hurst_nan_report,
        "baseline_9feature_reference_lambda50": {
            "note": "quoted from the established eth_jm_paperfaithful_build_20260820.py lambda=50 run, NOT re-derived here",
            "2024": {"balanced_accuracy": 0.4376, "flip_rate": 0.0028, "median_state_duration_bars": 180.5},
            "2025": {"balanced_accuracy": 0.4360, "flip_rate": 0.0035, "median_state_duration_bars": 163.5},
            "2026": {
                "balanced_accuracy": 0.4422,
                "flip_rate": 0.0028,
                "median_state_duration_bars": 148.0,
                "chop_recall": 0.700,
                "bull_recall": 0.272,
                "bear_recall": 0.354,
            },
        },
        "judgment_calls": JUDGMENT_CALLS,
        "calibration": {},
        "by_max_feats": [],
    }

    # ------------------------------------------------------------------------------
    # Phase 1: timing calibration -- REQUIRED before committing to the full grid.
    # ------------------------------------------------------------------------------
    print(
        f"[calibration] fitting SJM max_feats={CALIB_MAX_FEATS} n_init_jm={N_INIT_JM} "
        f"max_iter(outer, TEMP-CAPPED)={CALIB_MAX_ITER_OUTER} on the real 14-feature pool ...",
        flush=True,
    )
    calib_sjm = SparseJumpModel(
        n_components=N_COMPONENTS,
        max_feats=CALIB_MAX_FEATS,
        jump_penalty=JUMP_PENALTY,
        cont=False,
        random_state=SEED,
        max_iter=CALIB_MAX_ITER_OUTER,
        tol_w=TOL_W,
        max_iter_jm=MAX_ITER_JM,
        tol_jm=TOL_JM,
        n_init_jm=N_INIT_JM,
        verbose=1,
    )
    calib_elapsed, calib_n_iter = _fit_with_iter_count(calib_sjm, X_train, ret_train)
    per_iter_s = calib_elapsed / calib_n_iter if calib_n_iter > 0 else calib_elapsed
    converged_before_cap = calib_n_iter < CALIB_MAX_ITER_OUTER
    worst_case_full_grid_s = per_iter_s * MAX_ITER_OUTER * len(MAX_FEATS_GRID)
    go = worst_case_full_grid_s <= BUDGET_SECONDS

    calibration_report = {
        "max_feats": CALIB_MAX_FEATS,
        "n_init_jm": N_INIT_JM,
        "max_iter_outer_cap_TEMPORARY": CALIB_MAX_ITER_OUTER,
        "elapsed_seconds": calib_elapsed,
        "actual_outer_iterations_run": calib_n_iter,
        "converged_before_cap": converged_before_cap,
        "measured_seconds_per_outer_iteration": per_iter_s,
        "full_grid_worst_case_seconds": worst_case_full_grid_s,
        "full_grid_worst_case_minutes": worst_case_full_grid_s / 60.0,
        "budget_seconds": BUDGET_SECONDS,
        "budget_minutes": BUDGET_SECONDS / 60.0,
        "decision": "GO" if go else "NO-GO",
        "decision_rationale": (
            f"worst-case bound = measured_seconds_per_outer_iteration({per_iter_s:.1f}s) * "
            f"max_iter_outer cap({MAX_ITER_OUTER}) * len(MAX_FEATS_GRID)({len(MAX_FEATS_GRID)}) = "
            f"{worst_case_full_grid_s:.0f}s ({worst_case_full_grid_s / 60:.1f} min); this is a HARD "
            f"upper bound (max_iter_outer is a strict cap in SJM's own while-loop condition). Actual "
            f"runtime will likely be lower if SJM converges via tol_w before the cap (the calibration "
            f"probe itself {'DID' if converged_before_cap else 'did NOT'} converge before its "
            f"{CALIB_MAX_ITER_OUTER}-iteration temporary cap)."
        ),
    }
    report["calibration"] = calibration_report
    print(f"[calibration] RESULT:\n{json.dumps(calibration_report, indent=2, default=_json_default)}", flush=True)

    if not go:
        report["full_grid_run"] = False
        report["total_seconds"] = time.time() - t_start
        REPORT_PATH.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
        print(
            f"[STOP] NO-GO decision -- full grid worst-case ({worst_case_full_grid_s / 60:.1f} min) "
            f"exceeds the {BUDGET_SECONDS / 60:.0f}-minute budget. Report written with "
            f"calibration-only results: {REPORT_PATH}",
            flush=True,
        )
        return

    print(
        f"[calibration] GO decision -- proceeding to full {len(MAX_FEATS_GRID)}-point max_feats grid "
        f"(worst-case {worst_case_full_grid_s / 60:.1f} min <= {BUDGET_SECONDS / 60:.0f}-min budget).",
        flush=True,
    )

    # ------------------------------------------------------------------------------
    # Phase 2: full max_feats grid.
    # ------------------------------------------------------------------------------
    for max_feats in MAX_FEATS_GRID:
        sjm = SparseJumpModel(
            n_components=N_COMPONENTS,
            max_feats=max_feats,
            jump_penalty=JUMP_PENALTY,
            cont=False,
            random_state=SEED,
            max_iter=MAX_ITER_OUTER,
            tol_w=TOL_W,
            max_iter_jm=MAX_ITER_JM,
            tol_jm=TOL_JM,
            n_init_jm=N_INIT_JM,
            verbose=1,
        )
        fit_secs, n_iter_outer = _fit_with_iter_count(sjm, X_train, ret_train)

        ret_ = np.asarray(sjm.ret_, dtype=np.float64)
        vol_ = np.asarray(sjm.vol_, dtype=np.float64)
        # See judgment call 10: same VERIFIED cumret*freq non-increasing check as the original
        # plain-JM script, re-applicable to SJM since it delegates to the same sort code path.
        proba_train_arr = sjm.proba_.to_numpy() if hasattr(sjm.proba_, "to_numpy") else np.asarray(sjm.proba_)
        freq_ = proba_train_arr.sum(axis=0)
        cumret_proxy = ret_ * freq_
        if not np.all(np.diff(cumret_proxy) <= 1e-6):
            raise RuntimeError(
                f"max_feats={max_feats}: post-fit state order is not descending by (ret_*freq) as "
                f"expected from sort_by='cumret' -- ret_={ret_} freq_={freq_} cumret_proxy={cumret_proxy}. "
                "Refusing to silently guess a bull/bear/chop mapping."
            )
        idx_to_class = {0: "bull", 1: "chop", 2: "bear"}

        w_series = sjm.w if isinstance(sjm.w, pd.Series) else pd.Series(np.asarray(sjm.w), index=feats.columns)
        feat_weights_series = (
            sjm.feat_weights
            if isinstance(sjm.feat_weights, pd.Series)
            else pd.Series(np.asarray(sjm.feat_weights), index=feats.columns)
        )
        selected = w_series[w_series > 0.0].sort_values(ascending=False)
        zeroed = w_series[w_series <= 0.0]

        print(
            f"[fit max_feats={max_feats}] {fit_secs:.1f}s outer_iters={n_iter_outer} "
            f"val_={sjm.jm_ins.val_:.2f} ret_={np.round(ret_, 6).tolist()} vol_={np.round(vol_, 6).tolist()} "
            f"freq_={freq_.tolist()} cumret_proxy={np.round(cumret_proxy, 6).tolist()} -> idx_to_class={idx_to_class}",
            flush=True,
        )
        print(f"    [w] selected(nonzero)={selected.round(4).to_dict()}", flush=True)
        print(f"    [w] zeroed_out={list(zeroed.index)}", flush=True)

        mf_entry: dict[str, Any] = {
            "max_feats": max_feats,
            "fit_seconds": fit_secs,
            "actual_outer_iterations_run": n_iter_outer,
            "converged_before_outer_cap": n_iter_outer < MAX_ITER_OUTER,
            "objective_val": float(sjm.jm_ins.val_),
            "state_ret_sorted": ret_.tolist(),
            "state_vol_sorted": vol_.tolist(),
            "state_to_class": {str(k): v for k, v in idx_to_class.items()},
            "transmat_sorted": np.asarray(sjm.jm_ins.transmat_).tolist(),
            "feature_weights_w": w_series.round(6).to_dict(),
            "feature_weights_sqrt_w": feat_weights_series.round(6).to_dict(),
            "n_selected_nonzero": int((w_series > 0.0).sum()),
            "selected_features_sorted_by_weight": list(selected.index),
            "zeroed_out_features": list(zeroed.index),
            "years": {},
        }

        for year, mask in year_masks.items():
            X_eval = X_by_year[year]
            t1 = time.time()
            proba = sjm.predict_proba_online(X_eval)  # see judgment call 11
            proba_arr = proba.to_numpy() if hasattr(proba, "to_numpy") else np.asarray(proba)
            transform_secs = time.time() - t1
            state_hard = proba_arr.argmax(axis=1)
            mapped_class = np.array([idx_to_class[int(s)] for s in state_hard])
            pred_idx = np.array([CLASS_TO_IDX[c] for c in mapped_class])

            y_true = gt_by_year[year].to_numpy()
            assert len(y_true) == len(pred_idx), f"{year}: length mismatch {len(y_true)} vs {len(pred_idx)}"

            proba_onehot = np.zeros((len(pred_idx), len(CLASSES3)), dtype=np.float64)
            proba_onehot[np.arange(len(pred_idx)), pred_idx] = 1.0
            ev = _eval(y_true, proba_onehot)
            max_state_share = max(ev["pred_counts"].values()) / ev["rows"]
            sample_type = "in_sample_train" if year == "2024" else "out_of_sample"

            mf_entry["years"][year] = {
                "sample_type": sample_type,
                "causal": "predict_proba_online (online/causal, argmax)",
                "transform_seconds": transform_secs,
                "rows": ev["rows"],
                "balanced_accuracy": ev["balanced_accuracy"],
                "accuracy": ev["accuracy"],
                "flip_rate": ev["flip_rate"],
                "median_state_duration_bars": ev["median_state_duration_bars"],
                "mean_state_duration_bars": ev["mean_state_duration_bars"],
                "recall": ev["recall"],
                "true_counts": ev["true_counts"],
                "pred_counts": ev["pred_counts"],
                "confusion_matrix": ev["confusion_matrix"],
                "max_state_share": max_state_share,
                "log_loss_onehot_diagnostic": ev["log_loss"],
            }
            print(
                f"    [{year} sample={sample_type}] online: bal_acc={ev['balanced_accuracy']:.4f} "
                f"flip_rate={ev['flip_rate']:.4f} median_dur={ev['median_state_duration_bars']:.1f} "
                f"chop_recall={ev['recall'].get('chop')} max_state_share={max_state_share:.4f} "
                f"({transform_secs:.2f}s)",
                flush=True,
            )

        report["by_max_feats"].append(mf_entry)
        # partial write after each grid point -- monitorable mid-run, resilient to interruption
        report["full_grid_run"] = "in_progress"
        report["total_seconds_so_far"] = time.time() - t_start
        REPORT_PATH.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")

    report["full_grid_run"] = True
    report["total_seconds"] = time.time() - t_start
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    print(f"[done] report={REPORT_PATH} total_seconds={report['total_seconds']:.1f}", flush=True)


if __name__ == "__main__":
    main()
