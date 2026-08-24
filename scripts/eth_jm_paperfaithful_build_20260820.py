#!/usr/bin/env python3
"""ETH 3-state regime classifier rebuilt paper-faithfully with the `jumpmodels` reference
package (2026-08-20).

Provenance
----------
True origin: Bemporad, Breschi, Piga, Boyd (2018), "Fitting jump models," Automatica 96:11-21.
Finance specialization: Nystrup, Lindstrom & Madsen (2020, ESWA) onward.
Feature family (EWM return / EWM downside deviation / EWM Sortino ratio): Shu, Yu & Kolm
(2024), Table 2 -- daily bars, halflives of 10/20/60 TRADING DAYS, 3000-day rolling window,
K=2 states. We adapt (not reproduce) this feature family to 5-minute ETH bars -- see the
"JUDGMENT CALLS" block below and the run report for exactly what is and is not paper-derived.

This script uses the REAL, author-maintained `jumpmodels` PyPI package (v0.1.1) directly:
    from jumpmodels.jump import JumpModel
    from jumpmodels.preprocess import StandardScalerPD, DataClipperStd
No DP / coordinate-descent / scaler logic is hand-rolled here.

Run with:
    /home/kbj20/anaconda3/envs/quant_ai/bin/python3 scripts/eth_jm_paperfaithful_build_20260820.py

Output: tmp/eth_jm_paperfaithful_20260820/report.json (+ progress on stdout).

JUDGMENT CALLS (see report.json["judgment_calls"] and the chat report for full detail):
  1. Feature halflives (72/288/864 bars) are an ADAPTATION of the paper's 10/20/60-TRADING-DAY
     halflives to 5-min bars, reused from scripts/test_statistical_jump_model_regimes_20260808.py
     for continuity with prior work in this repo -- NOT validated by the paper for intraday data.
  2. lambda is a small, explicitly-labeled grid sweep {50,150,300,600,1200}, not one of the
     5 named formal selection procedures in the literature (GIC/AIC/BIC, permutation Gap,
     ARI-CV, downstream-Sharpe CV, classification-accuracy).
  3. Preprocessing pipeline is DataClipperStd(mul=3.0) -> StandardScalerPD(), fit on TRAIN(2024)
     only and applied to all years. StandardScalerPD was explicitly requested; DataClipperStd
     was ADDED on top after confirming it is the package author's own canonical pairing in
     examples/nasdaq/example.py ("We first clip the data within three standard deviations for
     all features and then perform standardization before feeding the data into the JMs.").
     This goes beyond the letter of the task's Stage-2 instruction (which named only
     StandardScalerPD) -- flagged explicitly, not silently added.
  4. ret_ser passed to .fit() is the CONTEMPORANEOUS (same-bar) causal log return, not a
     forward return -- used only for post-hoc state labeling (ret_/vol_/sort_by="cumret"),
     not for the DP clustering objective itself.
  5. Model input features (EWM return/downside-dev/Sortino) are computed on the
     CHRONOLOGICALLY CONCATENATED 2024+2025+2026 close series (continuous EWM state across
     year boundaries -- verified contiguous 5-min spacing at both boundaries), NOT
     cold-started at each year file's start. This avoids an artificial multi-day cold-start
     degradation exactly at the start of the VAL/OOS evaluation windows. This is a deliberate
     departure from a naive "read each CSV independently" approach.
  6. Ground-truth rule-based labels (_labels/_current_labels3_thresholded, LABEL_MODE=
     "balancedish_adx16_slope15_bb012") are, BY CONTRAST, computed PER-FILE (cold-started
     each year), replicating exactly how experiment_regime3_current_hmm_wide24_20260529.py's
     own main()/_transform() computes them (`_labels(_read(src), label_mode)` per transform
     file) -- for maximal comparability with this session's other reference numbers. This is
     an intentional asymmetry from (5): JM model input is continuous, ground truth is per-file.
  7. VERIFIED MECHANISM (not a judgment call, documented here for reproducibility): the
     package's sort_by="cumret" criterion is `ret_ * freq` (mean return PER STATE WEIGHTED BY
     that state's occupancy count), not `ret_` alone -- confirmed by reading
     jumpmodels/base.py::sort_param_dict and by an empirical smoke test where a 2-bar
     micro-state had the single highest ret_ but, once weighted by its tiny freq, the lowest
     cumret contribution. We map post-fit index 0->bull, 1->chop, 2->bear (matching the
     package author's own worked example for K=2: "the state with higher cumulative returns
     is denoted s_t=0 (bull)") and ASSERT at runtime that ret_*freq is non-increasing in that
     order before trusting the mapping; the script aborts rather than silently mislabeling if
     the assertion ever fails.
  8. VERIFIED BUG FOUND & FIXED (not a judgment call): the naive Sortino ratio
     ewm_ret/(ewm_dd+1e-12) blows up to O(1e8-1e9) on real 2024 data for exactly the first 3
     bars of the whole series (ewm_dd is exactly 0.0 there -- no negative return has occurred
     yet in any EWM window -- then jumps to its normal ~1e-3 magnitude from bar 3 onward, a
     sharp break not a gradual decay). Left unguarded, those 3 pathological rows dominate
     DataClipperStd/StandardScalerPD's TRAIN mean/std (fit over all of 2024), silently making
     all three Sortino features near-zero/uninformative for the other ~105,377 real rows. Fixed
     via SORTINO_DD_FLOOR=1e-8: ewm_sortino := 0.0 when ewm_dd <= floor. Caught by inspecting
     real feature statistics before the full run, not assumed.
  K fixed at 3 (not searched). No periodic refitting (paper refits every 6 months on a rolling
  window; we fit once on 2024 and transform forward, matching this repo's existing
  regime-classifier convention). Sparse JM (feature selection) and CJM (soft/continuous
  probability) are explicitly out of scope for this pass.
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
# functions. mamba_ssm itself is never called by anything in this script.
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

from jumpmodels.jump import JumpModel  # noqa: E402
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
OUT_DIR = ROOT / "tmp/eth_jm_paperfaithful_20260820"
REPORT_PATH = OUT_DIR / "report.json"

LABEL_MODE = "balancedish_adx16_slope15_bb012"
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES3)}  # bull=0, bear=1, chop=2

# EWM feature halflives in 5-min bars: 72=6h, 288=1d, 864=3d. Adapted from
# scripts/test_statistical_jump_model_regimes_20260808.py::HALflIVES for continuity with
# prior work in this repo. NOT validated by the paper itself for intraday bars -- see module
# docstring judgment call (1).
HALFLIVES_BARS = [72, 288, 864]

N_COMPONENTS = 3
LAMBDA_GRID = [50.0, 150.0, 300.0, 600.0, 1200.0]
SEED = 7529
N_INIT = 10
MAX_ITER = 1000
TOL = 1e-8
CLIP_MUL = 3.0

# Sortino-ratio denominator floor: guards against div-by-~0 in the FIRST 3 BARS of the whole
# concatenated series only (2024-01-01 00:00/00:05/00:10, where no negative return has yet
# occurred in ANY EWM window so ewm_dd is exactly 0.0). Empirically confirmed (real 2024 data)
# this is a sharp, isolated warmup artifact, not a gradual decay: ewm_dd is exactly 0.0 for
# bars 0-2 and jumps to its normal ~1e-3 magnitude from bar 3 onward at all 3 halflives, so a
# tiny eps=1e-12 denominator would blow ewm_sortino up to O(1e8-1e9) for those 3 rows, which in
# turn wrecks DataClipperStd/StandardScalerPD (both fit via mean/std over ALL of TRAIN=2024),
# silently making the Sortino features near-zero/uninformative for the other ~105,377 real
# rows. Below the floor, ewm_sortino is defined as 0.0 (no downside experienced yet -> ratio
# undefined, treated as neutral) instead of an unbounded value.
SORTINO_DD_FLOOR = 1e-8

# ----------------------------------------------------------------------------------
# Feature construction
# ----------------------------------------------------------------------------------


def build_features(close: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Causal EWM return / EWM downside-deviation / EWM Sortino-ratio feature family
    (Shu, Yu & Kolm 2024 Table 2 feature family; halflives adapted to 5-min bars -- see
    judgment call 1). Returns (features_df [9 cols], contemporaneous causal log-return
    series), both indexed identically to `close`.

    Formulas (all causal -- pandas .ewm(halflife=hl).mean(), default adjust=True, depends
    only on data up to and including row t):
      log_ret_t        = log(close_t) - log(close_{t-1})   (first bar := 0)
      ewm_ret_hl_t      = EWM_mean(log_ret, halflife=hl)_t
      ewm_dd_hl_t       = sqrt( EWM_mean( min(log_ret, 0)^2, halflife=hl )_t )
      ewm_sortino_hl_t  = ewm_ret_hl_t / ewm_dd_hl_t   if ewm_dd_hl_t > SORTINO_DD_FLOOR
                        = 0.0                          otherwise (see SORTINO_DD_FLOOR comment)
    """
    log_close = np.log(close.astype(np.float64))
    log_ret = log_close.diff().fillna(0.0)
    cols: dict[str, pd.Series] = {}
    n_floor_hits = {}
    for hl in HALFLIVES_BARS:
        ewm_ret = log_ret.ewm(halflife=hl).mean()
        downside_sq = log_ret.clip(upper=0.0) ** 2
        ewm_dd = np.sqrt(downside_sq.ewm(halflife=hl).mean())
        below_floor = ewm_dd.to_numpy() <= SORTINO_DD_FLOOR
        n_floor_hits[hl] = int(below_floor.sum())
        ewm_sortino = pd.Series(
            np.where(below_floor, 0.0, ewm_ret.to_numpy() / np.where(below_floor, 1.0, ewm_dd.to_numpy())),
            index=close.index,
        )
        cols[f"ewm_ret_hl{hl}"] = ewm_ret
        cols[f"ewm_dd_hl{hl}"] = ewm_dd
        cols[f"ewm_sortino_hl{hl}"] = ewm_sortino
    print(f"[features] Sortino floor (<= {SORTINO_DD_FLOOR}) hit counts by halflife: {n_floor_hits}", flush=True)
    feats = pd.DataFrame(cols, index=close.index)
    assert not feats.isna().any().any(), "unexpected NaN in causal EWM features"
    assert np.isfinite(feats.to_numpy()).all(), "unexpected inf in causal EWM features"
    return feats, log_ret


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ---- ground truth: PER-FILE (cold-started each year), replicating exactly how
    # experiment_regime3_current_hmm_wide24_20260529.py's main()/_transform() computes it
    # (`_labels(_read(src), label_mode)` per transform file) -- see judgment call 6. ----
    raw_frames: dict[str, pd.DataFrame] = {}
    gt_by_year: dict[str, pd.Series] = {}
    for year, path in CSV_PATHS.items():
        df = _read(path)
        raw_frames[year] = df
        gt = _labels(df, LABEL_MODE)
        gt_by_year[year] = pd.Series(gt, index=pd.DatetimeIndex(df["timestamp"], name="timestamp"))
        print(f"[load] {year}: rows={len(df)} range=[{df['timestamp'].iloc[0]} .. {df['timestamp'].iloc[-1]}]", flush=True)

    # ---- model input features: CONTINUOUS across year boundaries -- see judgment call 5 ----
    full = pd.concat(raw_frames.values(), ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    dup = int(full["timestamp"].duplicated().sum())
    if dup:
        raise RuntimeError(f"unexpected duplicate timestamps across year files: {dup}")
    full = full.set_index(pd.DatetimeIndex(full["timestamp"]))
    close = pd.to_numeric(full["close"], errors="coerce")
    assert close.notna().all(), "NaN close price in concatenated series"

    feats, ret_ser = build_features(close)
    print(f"[features] built {feats.shape[1]} features over {feats.shape[0]} continuous bars: {list(feats.columns)}", flush=True)

    year_of_idx = feats.index.year
    year_masks = {y: (year_of_idx == int(y)) for y in CSV_PATHS}
    train_mask = year_masks["2024"]

    X_train_raw = feats.loc[train_mask]
    ret_train = ret_ser.loc[train_mask]

    # sanity: continuous-feature year slice must align 1:1 (by timestamp) with the
    # independently-read per-file ground truth for that year.
    for year, mask in year_masks.items():
        idx_cont = feats.loc[mask].index
        idx_file = gt_by_year[year].index
        if not idx_cont.equals(idx_file):
            missing = idx_cont.difference(idx_file)
            extra = idx_file.difference(idx_cont)
            raise RuntimeError(f"{year}: continuous-vs-per-file index mismatch; missing={len(missing)} extra={len(extra)}")

    # ---- preprocessing: fit DataClipperStd(mul=3.)+StandardScalerPD() on TRAIN(2024) only,
    # transform elsewhere. See judgment call 3 for why DataClipperStd was added. ----
    clipper = DataClipperStd(mul=CLIP_MUL)
    scaler = StandardScalerPD()
    X_train = scaler.fit_transform(clipper.fit_transform(X_train_raw))
    X_by_year = {year: scaler.transform(clipper.transform(feats.loc[mask])) for year, mask in year_masks.items()}

    print(f"[preprocess] clip(mul={CLIP_MUL}) + StandardScalerPD fit on TRAIN=2024 ({len(X_train)} rows)", flush=True)

    report: dict[str, Any] = {
        "model_id": "eth_jm_paperfaithful_20260820",
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "package": "jumpmodels==0.1.1 (pip, author-maintained reference implementation)",
        "classes": CLASSES3,
        "label_mode": LABEL_MODE,
        "label_config": LABEL_CONFIGS[LABEL_MODE],
        "feature_halflives_bars": HALFLIVES_BARS,
        "feature_cols": list(feats.columns),
        "n_components": N_COMPONENTS,
        "lambda_grid": LAMBDA_GRID,
        "random_state": SEED,
        "n_init": N_INIT,
        "max_iter": MAX_ITER,
        "tol": TOL,
        "clip_mul": CLIP_MUL,
        "train_window": "2024 (full year)",
        "eval_years": list(CSV_PATHS),
        "rows_per_year": {y: int(m.sum()) for y, m in year_masks.items()},
        "judgment_calls": [
            "1. Feature halflives (72/288/864 bars = 6h/1d/3d) adapt the paper's 10/20/60-TRADING-DAY "
            "halflives to 5-min bars; reused from test_statistical_jump_model_regimes_20260808.py for "
            "continuity with prior repo work. NOT validated by the paper itself for intraday data.",
            "2. lambda is a small explicit grid sweep {50,150,300,600,1200}, not one of the 5 named "
            "formal selection procedures (GIC/AIC/BIC, permutation Gap, ARI-CV, downstream-Sharpe CV, "
            "classification-accuracy).",
            "3. Preprocessing = DataClipperStd(mul=3.0) then StandardScalerPD(), fit on TRAIN(2024) only. "
            "StandardScalerPD was explicitly requested by the task; DataClipperStd was ADDED on top, "
            "sourced from the package author's own examples/nasdaq/example.py canonical pairing -- "
            "goes beyond the literal Stage-2 instruction, flagged explicitly.",
            "4. ret_ser passed to .fit() is the contemporaneous (same-bar) causal log return, not a "
            "forward return -- affects only post-hoc state labeling (ret_/vol_/cumret sort), not the "
            "DP clustering objective.",
            "5. JM input features are computed on the chronologically CONCATENATED 2024+2025+2026 "
            "close series (continuous EWM state across year boundaries), not cold-started per file.",
            "6. Ground-truth rule-based labels are, by contrast, computed PER-FILE (cold-started each "
            "year), replicating experiment_regime3_current_hmm_wide24_20260529.py's own convention "
            "exactly, for maximal comparability with this session's other reference numbers.",
            "7. VERIFIED MECHANISM (not a judgment call): sort_by='cumret' sorts by ret_*freq "
            "(occupancy-weighted), not ret_ alone. Post-fit index 0=bull,1=chop,2=bear, verified by "
            "runtime assertion that ret_*freq is non-increasing before trusting the mapping.",
            "8. VERIFIED BUG FOUND & FIXED (not a judgment call): naive Sortino ratio "
            "ewm_ret/(ewm_dd+1e-12) blows up to O(1e8-1e9) for exactly bars 0-2 of the whole series "
            "(ewm_dd exactly 0.0 there, sharp break not gradual decay) on real 2024 data, which would "
            "silently wreck DataClipperStd/StandardScalerPD's TRAIN-fit mean/std and make all 3 "
            "Sortino features near-zero/uninformative for the other ~105,377 rows. Fixed via "
            f"SORTINO_DD_FLOOR={SORTINO_DD_FLOOR}: ewm_sortino:=0.0 when ewm_dd<=floor.",
            "9. K fixed at 3, not searched. No periodic refitting (fit once on 2024, transform forward). "
            "Sparse JM (feature selection) and CJM (continuous/soft probability) not implemented -- "
            "out of scope for this pass.",
        ],
        "by_lambda": [],
    }

    for lam in LAMBDA_GRID:
        t0 = time.time()
        jm = JumpModel(
            n_components=N_COMPONENTS,
            jump_penalty=lam,
            cont=False,
            random_state=SEED,
            max_iter=MAX_ITER,
            tol=TOL,
            n_init=N_INIT,
            verbose=0,
        )
        jm.fit(X_train, ret_ser=ret_train, sort_by="cumret")
        fit_secs = time.time() - t0

        ret_ = np.asarray(jm.ret_, dtype=np.float64)
        vol_ = np.asarray(jm.vol_, dtype=np.float64)
        # IMPORTANT: the package's own "cumret" sort criterion (base.py::sort_param_dict,
        # sort_by="cumret") is `ret_ * freq` (per-state mean return WEIGHTED BY per-state
        # occupancy count), NOT `ret_` alone -- confirmed both by reading base.py and by an
        # empirical smoke test on real data where a 2-bar micro-state had the single highest
        # ret_ (1.9e-3) but the lowest cumret contribution once weighted by its tiny freq.
        # A naive argsort(ret_) would have mislabeled that 2-bar sliver as "bull". We instead
        # recompute the package's own criterion from jm.proba_ (freq = column sums of the
        # final, already-sorted in-sample assignment matrix) and ASSERT it is non-increasing
        # in the post-fit index order (0=highest cumret .. K-1=lowest), matching the package
        # author's own documented convention (examples/nasdaq/example.py: "The state with
        # higher cumulative returns is denoted as s_t=0 (bull market)"). If this assertion
        # ever fails, we abort rather than silently mislabel bull/bear/chop.
        proba_train_arr = jm.proba_.to_numpy() if hasattr(jm.proba_, "to_numpy") else np.asarray(jm.proba_)
        freq_ = proba_train_arr.sum(axis=0)
        cumret_proxy = ret_ * freq_
        if not np.all(np.diff(cumret_proxy) <= 1e-6):
            raise RuntimeError(
                f"lambda={lam}: post-fit state order is not descending by (ret_*freq) as expected "
                f"from sort_by='cumret' -- ret_={ret_} freq_={freq_} cumret_proxy={cumret_proxy}. "
                "Refusing to silently guess a bull/bear/chop mapping."
            )
        idx_to_class = {0: "bull", 1: "chop", 2: "bear"}
        print(
            f"[fit lambda={lam}] {fit_secs:.1f}s val_={jm.val_:.2f} "
            f"ret_={np.round(ret_, 6).tolist()} vol_={np.round(vol_, 6).tolist()} freq_={freq_.tolist()} "
            f"cumret_proxy={np.round(cumret_proxy, 6).tolist()} -> idx_to_class={idx_to_class}",
            flush=True,
        )

        lam_entry: dict[str, Any] = {
            "lambda": lam,
            "fit_seconds": fit_secs,
            "objective_val": float(jm.val_),
            "state_ret_sorted": ret_.tolist(),
            "state_vol_sorted": vol_.tolist(),
            "state_to_class": {str(k): v for k, v in idx_to_class.items()},
            "transmat_sorted": np.asarray(jm.transmat_).tolist(),
            "years": {},
        }

        for year, mask in year_masks.items():
            X_eval = X_by_year[year]
            t1 = time.time()
            proba = jm.predict_proba_online(X_eval)
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

            lam_entry["years"][year] = {
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
                f"    [{year}] online: bal_acc={ev['balanced_accuracy']:.4f} flip_rate={ev['flip_rate']:.4f} "
                f"median_dur={ev['median_state_duration_bars']:.1f} chop_recall={ev['recall'].get('chop')} "
                f"max_state_share={max_state_share:.4f} ({transform_secs:.2f}s)",
                flush=True,
            )

        # extra, explicitly-non-causal in-sample batch (full-DP) reference for 2024 only --
        # NOT to be compared to the online rows as if equivalent; see module docstring.
        labels_batch = jm.labels_
        labels_batch_arr = labels_batch.to_numpy() if hasattr(labels_batch, "to_numpy") else np.asarray(labels_batch)
        mapped_batch = np.array([idx_to_class[int(s)] for s in labels_batch_arr])
        pred_batch_idx = np.array([CLASS_TO_IDX[c] for c in mapped_batch])
        y_true_2024 = gt_by_year["2024"].to_numpy()
        proba_batch_onehot = np.zeros((len(pred_batch_idx), len(CLASSES3)), dtype=np.float64)
        proba_batch_onehot[np.arange(len(pred_batch_idx)), pred_batch_idx] = 1.0
        ev_batch = _eval(y_true_2024, proba_batch_onehot)
        lam_entry["insample_2024_batch_noncausal_reference_only"] = {
            "warning": "NOT causal (full-window backward-traceback DP). In-sample reference only. Do not compare to the online rows above as if equivalent, do not use for promotion.",
            "balanced_accuracy": ev_batch["balanced_accuracy"],
            "flip_rate": ev_batch["flip_rate"],
            "median_state_duration_bars": ev_batch["median_state_duration_bars"],
            "recall": ev_batch["recall"],
            "max_state_share": max(ev_batch["pred_counts"].values()) / ev_batch["rows"],
        }

        report["by_lambda"].append(lam_entry)

    report["total_seconds"] = time.time() - t_start
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    print(f"[done] report={REPORT_PATH} total_seconds={report['total_seconds']:.1f}", flush=True)


if __name__ == "__main__":
    main()
