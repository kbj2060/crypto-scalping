#!/usr/bin/env python3
"""RESEARCH ONLY -- walk-forward RETRAINING robustness fold for
`research_eth_omega461_exit_head_liveatr_relabel_20260813.py` (the h48qual/zig075 exit_head "live
ATR barrier relabel" recipe currently shadow-deployed via
`live_eth_exithead_asymmetric_shadow_20260813.py`, see
docs/experiments/eth_omega461_live_exit_head_liveatr_relabel_20260813.md).

Question this script exists to answer: that recipe's original run only trained ONCE, on the fixed
pre-2025-10-01 TRAIN split, and only confirmed the "relabel beats the original exit_head" pattern on
one VAL window. Does the pattern reproduce if the recipe is retrained from scratch on a DIFFERENT
training window (same label construction, same candidate-subsampling method/seed, same ATR barrier
params, same architecture/hyperparameters -- only the training window boundary changes), evaluated
each time on its own held-out confirm window? This is the same style of robustness question the
2026-08-13 JM N=5-seed experiment asked along the SEED axis (same recipe, does it reproduce across
random seeds); this script asks it along the TIME axis instead (same recipe, does it reproduce
across training windows).

The base script (`research_eth_omega461_exit_head_liveatr_relabel_20260813.py`, imported below as
`base`, NEVER edited) hardcodes its training window inside its own docstring/call to
`train_eval_omega4_3head_parent72_loose_entry_quality_20260620._prepare_frames` (imported there as
`omega4`): `train_raw = train_all[train_all["timestamp"] < parent.SPLIT_TS]`, i.e. always
"< 2025-10-01". Every OTHER piece of the recipe that function call touches (label source, candidate
subsampling, ATR barrier simulation, retrain-exit-head-only pattern) lives in module-level functions
with no window assumption baked in, so this script reuses ALL of them UNCHANGED via import
(`base._fast_timescale_checkpoint`, `base._build_exit_dataset_entry_label_live_atr_barrier`,
`base._retrain_component_exit_head_liveatr`, `base.LIVE_ATR_CFG`,
`base.TIMESCALE_GATE_MIN_MEDIAN_BARS`, `base.BASELINE_AVG_HOLD_BARS`) -- the ONLY new code here is
`_prepare_frames_walkforward`, a fork of `omega4._prepare_frames` that takes the train window as an
explicit `[--train-start, --train-end)` argument instead of the hardcoded `parent.SPLIT_TS` cutoff.
`quality_mode="same_as_direction"`/`disable_tp_sl=False` stay hardcoded (matching the base script's
own fixed call into `omega4._prepare_frames`) -- not re-exposed as flags, since which quality mode or
whether TP/SL is disabled is not the walk-forward question. `_prepare_frames_walkforward` only builds
and returns what the downstream pipeline actually reads (`train_df`, `s_train_label`) -- confirmed by
reading every call site of `frames[...]` in the base script's `main()` -- not the full return dict
`omega4._prepare_frames` produces (`val_raw`/`oos_raw`/`train_fixed`/`label_quality_summary`/
`quality_target_diag`/`label_contract` are never read downstream of this recipe, so they are not
reconstructed).

Training windows that stay within 2025 (folds A/B/C) are sliced directly out of
`omega._load_omega_frames()`'s `train_all` (the full-2025 frame `omega4._prepare_frames` itself
slices from) and aligned against `omega.TABM_2025`, exactly like the base script's own hardcoded
slice. A window that reaches into 2026 (fold D) additionally slices `eval_df` (the full-2026 frame)
for its 2026 portion and aligns that against `omega.TABM_2026` -- the SAME 2026 source
`omega4._prepare_frames` itself uses for OOS scoring elsewhere in this lineage, not a new/ad-hoc
choice -- then concatenates the two aligned pieces. Both pieces already carry `zigzag_action` (set
identically on `train_all`/`eval_df` before either slice) and share `feature_cols` (computed once, as
the intersection of numeric columns present in BOTH years, exactly as `omega4._prepare_frames` does),
so the concat is schema-safe for every column this recipe's exit-dataset builder actually reads
(`timestamp`, `zigzag_action`, `open`/`high`/`low`/`close`, `hard.ROUTE_COLS`).

Per-fold checkpoint-first gate (unchanged from the base script, reused not reimplemented): the
live-ATR barrier's bars-to-resolution distribution is computed BEFORE any dataset build / retrain; if
the median (long AND short) falls under `base.TIMESCALE_GATE_MIN_MEDIAN_BARS` (30), this script aborts
(exit code 1) with `gate_pass=false` in the fold's own report.json rather than proceeding -- same rule
the base script applies to its own single window, just re-checked per fold since a different (in
particular, shorter -- fold B is only 6 months) training window could in principle produce a
different-shaped candidate population.

fresh_forward_bar_by_bar=true (the ATR-barrier simulation and exit-dataset build are the base
script's own unmodified single forward-simulation functions; this script adds no new bar-by-bar
simulation logic of its own, only the parameterized frame-prep + orchestration). trade_ledgers_used_
as_input=false. saved_parent_exit_timestamps_used=false. future_rows_used_for_entry=false
(candidates' own forward barrier simulation only ever walks forward from that candidate's own entry
bar, exactly as in the base script). Confirm-window evaluation (comparing this fold's new exit_head
against the frozen original) is NOT done here -- that is
`scripts/eval_eth_omega461_exit_head_liveatr_relabel_walkforward_20260814.py`'s job, reusing
`eth_omega461_multiwindow_confirmation_gate_20260814.load_all_windows` for the 6 pre-registered
windows so each fold's confirm window is loaded via the same already-verified mechanism every other
Odyssey2/3 candidate uses, not a new one-off loader.

Does NOT touch trading_bot_modules/omega4_6_1_live.py, trading_bot.py, runtime_config.py, .env. Does
NOT modify `research_eth_omega461_exit_head_liveatr_relabel_20260813.py` or any other imported module
-- only reads/calls into them. Does NOT overwrite any live checkpoint or prior fold's output
(`--fold-name` isolates each fold's own `tmp/causal_regen_20260516/<MODEL_ID>_<fold-name>/` directory).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as omega4  # noqa: E402
import research_eth_omega461_exit_head_liveatr_relabel_20260813 as base  # noqa: E402

MODEL_ID = "eth_omega461_exit_head_liveatr_relabel_walkforward_fold_20260814"
DIRECTION_LABEL_DIR = base.DIRECTION_LABEL_DIR
YEAR_BOUNDARY = pd.Timestamp("2026-01-01")


def _prepare_frames_walkforward(train_start: str, train_end: str) -> dict[str, Any]:
    """See module docstring. `train_end` is an EXCLUSIVE upper bound (half-open interval), matching
    pandas convention -- e.g. train_end="2025-07-01" includes all of 2025-06-30 but not 2025-07-01
    itself."""
    omega.BASE_TEMPLATE["max_hold"] = 0
    omega.BASE_TEMPLATE["cooldown"] = 0
    train_all, eval_df, _overlay_report = omega._load_omega_frames()
    feature_cols = omega._numeric_feature_cols(train_all, eval_df)
    label_2025 = omega4._read_labels(DIRECTION_LABEL_DIR, 2025, require_diagnostics=False)
    label_2026 = omega4._read_labels(DIRECTION_LABEL_DIR, 2026, require_diagnostics=False)
    train_all, train_labels = omega._align(train_all, label_2025, "omega4 train labels")
    eval_df, eval_labels = omega._align(eval_df, label_2026, "omega4 oos labels")
    train_all = train_all.copy()
    eval_df = eval_df.copy()
    train_all["zigzag_action"] = pd.to_numeric(train_labels["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    eval_df["zigzag_action"] = pd.to_numeric(eval_labels["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)

    train_start_ts = pd.Timestamp(train_start)
    train_end_ts = pd.Timestamp(train_end)
    if train_start_ts >= train_end_ts:
        raise RuntimeError(f"train_start ({train_start_ts}) must be before train_end ({train_end_ts})")

    parts: list[pd.DataFrame] = []
    span_meta: dict[str, Any] = {}
    if train_start_ts < YEAR_BOUNDARY:
        end_2025 = min(train_end_ts, YEAR_BOUNDARY)
        raw_2025 = train_all[(train_all["timestamp"] >= train_start_ts) & (train_all["timestamp"] < end_2025)].reset_index(drop=True)
        if len(raw_2025) == 0:
            raise RuntimeError("walk-forward 2025 slice is empty")
        tabm_2025 = omega._read(omega.TABM_2025)
        aligned_2025, _src = omega._align(raw_2025, tabm_2025, "train_2025")
        parts.append(aligned_2025)
        span_meta["2025_rows_before_align"] = int(len(raw_2025))
        span_meta["2025_rows_after_align"] = int(len(aligned_2025))
    if train_end_ts > YEAR_BOUNDARY:
        start_2026 = max(train_start_ts, YEAR_BOUNDARY)
        raw_2026 = eval_df[(eval_df["timestamp"] >= start_2026) & (eval_df["timestamp"] < train_end_ts)].reset_index(drop=True)
        if len(raw_2026) == 0:
            raise RuntimeError("walk-forward 2026 slice is empty")
        tabm_2026 = omega._read(omega.TABM_2026)
        aligned_2026, _src = omega._align(raw_2026, tabm_2026, "train_2026")
        parts.append(aligned_2026)
        span_meta["2026_rows_before_align"] = int(len(raw_2026))
        span_meta["2026_rows_after_align"] = int(len(aligned_2026))
    if not parts:
        raise RuntimeError(f"walk-forward train window [{train_start}, {train_end}) produced zero rows")
    train_df = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
    s_train_label = parent._base_input(train_df, feature_cols)
    return {
        "train_df": train_df,
        "s_train_label": s_train_label,
        "feature_cols": feature_cols,
        "rows": int(len(train_df)),
        "span_meta": span_meta,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-start", required=True, help="inclusive, e.g. 2025-01-01")
    ap.add_argument("--train-end", required=True, help="EXCLUSIVE upper bound, e.g. 2025-07-01 means through 2025-06-30")
    ap.add_argument("--fold-name", required=True, help="e.g. foldB -- isolates tmp/causal_regen_20260516/<MODEL_ID>_<fold-name>/")
    ap.add_argument("--stage", choices=["checkpoint_only", "full"], default="full")
    ap.add_argument("--max-candidates", type=int, default=2000, help="base script's own argparse default, unchanged")
    ap.add_argument("--max-horizon-bars", type=int, default=6000)
    ap.add_argument("--max-rows", type=int, default=0, help="0 = no extra cap beyond --max-candidates")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=260813, help="base script's own default -- held fixed across folds; only the train window varies")
    args = ap.parse_args()

    base._seed_everything(int(args.seed))
    device = torch.device("cpu")
    out_dir = ROOT / "tmp/causal_regen_20260516" / f"{MODEL_ID}_{str(args.fold_name).strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"stage=prepare_frames_walkforward fold={args.fold_name} train_start={args.train_start} train_end={args.train_end}", flush=True)
    t0 = time.time()
    frames = _prepare_frames_walkforward(str(args.train_start), str(args.train_end))
    fee, slip = omega._load_fee_slip()
    print(f"  train_df rows={frames['rows']} span_meta={frames['span_meta']} elapsed={time.time() - t0:.1f}s", flush=True)

    print("stage=timescale_checkpoint", flush=True)
    t0 = time.time()
    tc = base._fast_timescale_checkpoint(frames["train_df"], atr_cfg=base.LIVE_ATR_CFG, max_horizon_bars=int(args.max_horizon_bars))
    checkpoint = {
        "stage": "0_live_atr_timescale_pretraining_gate",
        "fold_name": str(args.fold_name),
        "train_start": str(args.train_start),
        "train_end": str(args.train_end),
        "train_rows": frames["rows"],
        "span_meta": frames["span_meta"],
        "baseline_avg_hold_bars": base.BASELINE_AVG_HOLD_BARS,
        "new_recipe_bars_stats": {"long": tc["long_bars_stats"], "short": tc["short_bars_stats"]},
        "reason_counts": tc["reason_counts"],
        "used_candidates_full_population": tc["used_candidates"],
        "truncated_at_horizon": tc["truncated_at_horizon"],
        "truncated_rate": tc["truncated_rate"],
        "max_horizon_bars": tc["max_horizon_bars"],
        "atr_cfg": tc["atr_cfg"],
        "elapsed_sec": time.time() - t0,
    }
    long_median = tc["long_bars_stats"].get("median", 0.0)
    short_median = tc["short_bars_stats"].get("median", 0.0)
    gate_pass = bool(long_median >= base.TIMESCALE_GATE_MIN_MEDIAN_BARS and short_median >= base.TIMESCALE_GATE_MIN_MEDIAN_BARS)
    checkpoint["gate_pass"] = gate_pass
    checkpoint["gate_rule"] = f"median bars-to-resolution (both long and short) >= {base.TIMESCALE_GATE_MIN_MEDIAN_BARS}"
    (out_dir / "stage0_timescale_checkpoint.json").write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2, default=base._json_default) + "\n", encoding="utf-8"
    )
    print(json.dumps(checkpoint, ensure_ascii=False, indent=2, default=base._json_default), flush=True)

    if not gate_pass:
        print(f"stage=ABORT fold={args.fold_name} gate_pass=False -- new barrier still resolves too fast for this fold's window, not proceeding to training", flush=True)
        (out_dir / "report.json").write_text(
            json.dumps(
                {"model_id": f"{MODEL_ID}_{args.fold_name}", "fold_name": str(args.fold_name), "gate_pass": False, "aborted": True, "checkpoint": checkpoint},
                ensure_ascii=False, indent=2, default=base._json_default,
            ) + "\n", encoding="utf-8",
        )
        return 1
    if str(args.stage) == "checkpoint_only":
        print("stage=done (checkpoint_only)", flush=True)
        return 0

    rng = np.random.default_rng(int(args.seed))
    valid_idx = np.asarray(tc["valid_candidate_idx"], dtype=np.int64)
    n_sample = min(int(args.max_candidates), len(valid_idx)) if int(args.max_candidates) > 0 else len(valid_idx)
    candidate_idx = np.sort(rng.choice(valid_idx, size=n_sample, replace=False))
    print(f"stage=build_live_atr_barrier_exit_dataset candidates_sampled={len(candidate_idx)}/{len(valid_idx)}", flush=True)
    t0 = time.time()
    x_exit_raw, y_exit, frame_exit, exit_diag = base._build_exit_dataset_entry_label_live_atr_barrier(
        frames["train_df"], frames["s_train_label"],
        candidate_idx=candidate_idx, fee=fee, slip=slip, cost_mult=float(args.cost_mult),
        atr_cfg=base.LIVE_ATR_CFG, max_horizon_bars=int(args.max_horizon_bars), max_rows=int(args.max_rows),
    )
    build_elapsed = time.time() - t0
    print(f"  rows={exit_diag['rows']} used_candidates={exit_diag['used_candidates']} positive_rate={exit_diag['positive_rate']:.4f} elapsed={build_elapsed:.1f}s", flush=True)
    exit_diag["build_elapsed_sec"] = build_elapsed

    results: dict[str, Any] = {"checkpoint": checkpoint, "dataset": exit_diag, "components": {}}
    for component in ("h48qual", "zig075"):
        print(f"stage=retrain_exit_head component={component}", flush=True)
        t0 = time.time()
        retrain_info = base._retrain_component_exit_head_liveatr(
            component, x_exit_raw, y_exit, frame_exit,
            seed=int(args.seed), epochs=int(args.epochs), device=device, out_dir=out_dir,
        )
        print(f"  {component} retrain elapsed={time.time() - t0:.1f}s -> {retrain_info['new_bundle']}", flush=True)
        results["components"][component] = {"retrain": retrain_info}

    report = {
        "model_id": f"{MODEL_ID}_{args.fold_name}",
        "base_recipe_script": "scripts/research_eth_omega461_exit_head_liveatr_relabel_20260813.py",
        "fold_name": str(args.fold_name),
        "train_start": str(args.train_start),
        "train_end": str(args.train_end),
        "train_rows": frames["rows"],
        "design": (
            "Walk-forward RETRAINING robustness fold: identical recipe to the base script (label "
            "construction = live-ATR-adaptive-barrier every-zigzag_action-bar entry-label, candidate "
            "subsampling method+seed, ATR barrier params, architecture, hyperparameters all "
            "unchanged, encoder/direction_head/quality_head always frozen to the original live "
            "bundle) -- only the training window boundary differs from the base script's fixed "
            "pre-2025-10-01 TRAIN split."
        ),
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "max_candidates_arg": int(args.max_candidates),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "cost_mult": float(args.cost_mult),
        **results,
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=base._json_default) + "\n", encoding="utf-8")
    print(f"report={out_dir / 'report.json'}", flush=True)
    print("stage=done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
