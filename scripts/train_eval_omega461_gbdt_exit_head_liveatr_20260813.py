#!/usr/bin/env python3
"""RESEARCH ONLY -- Odyssey2 priority #4: replace h48qual's exit_head (currently TabM, live-ATR
relabel recipe -- scripts/research_eth_omega461_exit_head_liveatr_relabel_20260813.py, see
docs/experiments/eth_omega461_live_exit_head_liveatr_relabel_20260813.md -- the current confirmed
Odyssey2 baseline) with a GBDT classifier (LightGBM if importable, else sklearn
HistGradientBoostingClassifier) trained on the IDENTICAL dataset/label, to see whether the model
class itself matters once the label recipe is held fixed. zig075 is not touched by this script.

Dataset: rebuilt, not cached -- the original TabM run never persisted x_exit_raw/y_exit/frame_exit
to disk, only the trained bundle + report.json diagnostics. This script reproduces that exact
dataset by importing and calling research_eth_omega461_exit_head_liveatr_relabel_20260813's own
_fast_timescale_checkpoint / _build_exit_dataset_entry_label_live_atr_barrier functions UNCHANGED,
with the same seed (260813) and candidate count (1500) as the "full1500" TabM run. Candidate
sampling (np.random.default_rng(seed).choice over the deterministic valid-candidate population) is
reproduced line-for-line from that script's main(), so the sampled candidate_idx array is bit-
identical. This script then asserts the rebuilt row count / positive count / used-candidate count
match tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_20260813_full1500/
report.json's "dataset" block before training, so a silent dataset drift cannot masquerade as a
"GBDT vs TabM" difference later.

Per regime expert (hard.EXPERT_NAMES = bull/bear/chop), fits one GBDT classifier on the SAME
x_exit/y_exit rows as the other two experts, weighted by
compute_sample_weight(class_weight="balanced", y=y_exit) * that expert's soft Regime3 route
probability -- an exact mirror of how
train_eval_omega4_1_exit_head_price_move_sltp_retrain_20260622._fit_exit_head_only weights TabM's
own per-expert exit-head-only retrain (read, not modified). This isolates the ablation to "model
class only": same rows, same soft per-expert weighting scheme, same 115-dim (102 base + 13 pos_*)
feature contract via train_eval_omega1_2_tabm_3head_20260603._exit_input_from_position_rows, same
label. Three SEPARATE models are trained (one per regime), not a single GBDT with regime as a
feature -- matching the live bull/bear/chop expert-routed architecture structurally.

Output is a plain pickle bundle (NOT a torch bundle -- GBDT models are not nn.Module). See
scripts/research_eth_omega461_gbdt_exit_head_val_20260813.py for the duck-typed runtime wrapper
that lets these models be dropped into
train_eval_omega4_2_risk_sidecar_20260622._predict_exit_prob_one unchanged.

fresh_forward_bar_by_bar=true (dataset build is the same causal forward barrier simulation the TabM
run used, unmodified). trade_ledgers_used_as_input=false. saved_parent_exit_timestamps_used=false.
future_rows_used_for_entry=false. Training uses only the pre-2025-10-01 TRAIN split (identical
frames to the TabM run). Does NOT touch trading_bot_modules/omega4_6_1_live.py, trading_bot.py,
runtime_config.py, .env. Does NOT touch zig075 in any way.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as omega4  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import research_eth_omega461_exit_head_h48cons_relabel_20260813 as h48cons  # noqa: E402
import research_eth_omega461_exit_head_liveatr_relabel_20260813 as liveatr  # noqa: E402

try:
    import lightgbm as lgb

    GBDT_LIBRARY = "lightgbm"
except ImportError:  # pragma: no cover -- documented fallback, per task instructions
    lgb = None
    GBDT_LIBRARY = "sklearn_histgradientboosting"
    from sklearn.ensemble import HistGradientBoostingClassifier

MODEL_ID = "eth_omega461_gbdt_exit_head_liveatr_20260813"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
REFERENCE_REPORT = ROOT / "tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_20260813_full1500/report.json"
SEED = 260813
MAX_CANDIDATES_DEFAULT = 1500
MAX_HORIZON_BARS = 6000
COST_MULT = 3.0  # matches research_eth_omega461_exit_head_liveatr_relabel_20260813.py's --cost-mult default, used for the full1500 run
GBDT_PARAMS = {"n_estimators": 400, "num_leaves": 31, "learning_rate": 0.05}  # matches this repo's
# established LightGBM convention for regime-routed ETH classifiers, e.g.
# train_eval_eth_h48qual_final_boss_v2_regime_routed_20260813.py's lgb.LGBMClassifier call.


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _build_dataset(max_candidates: int) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Line-for-line reproduction of research_eth_omega461_exit_head_liveatr_relabel_20260813.main()'s
    dataset-build steps (prepare_frames -> timescale checkpoint -> seeded candidate subsample ->
    build), calling that module's own functions unchanged."""
    print("stage=prepare_frames", flush=True)
    t0 = time.time()
    frames = omega4._prepare_frames(
        disable_tp_sl=False, direction_label_dir=liveatr.DIRECTION_LABEL_DIR,
        quality_mode="same_as_direction", quality_label_dir=None,
        quality_min_edge=0.0, quality_max_mae=0.0, quality_min_mfe_mae=0.0, quality_max_hold_bars=0,
    )
    fee, slip = omega._load_fee_slip()
    print(f"  train_df rows={len(frames['train_df'])} elapsed={time.time() - t0:.1f}s", flush=True)

    print("stage=timescale_checkpoint", flush=True)
    tc = liveatr._fast_timescale_checkpoint(frames["train_df"], atr_cfg=liveatr.LIVE_ATR_CFG, max_horizon_bars=MAX_HORIZON_BARS)

    rng = np.random.default_rng(SEED)
    valid_idx = np.asarray(tc["valid_candidate_idx"], dtype=np.int64)
    n_sample = min(int(max_candidates), len(valid_idx))
    candidate_idx = np.sort(rng.choice(valid_idx, size=n_sample, replace=False))
    print(f"stage=build_live_atr_barrier_exit_dataset candidates_sampled={len(candidate_idx)}/{len(valid_idx)}", flush=True)
    t0 = time.time()
    x_exit_raw, y_exit, frame_exit, exit_diag = liveatr._build_exit_dataset_entry_label_live_atr_barrier(
        frames["train_df"], frames["s_train_label"],
        candidate_idx=candidate_idx, fee=fee, slip=slip, cost_mult=COST_MULT,
        atr_cfg=liveatr.LIVE_ATR_CFG, max_horizon_bars=MAX_HORIZON_BARS, max_rows=0,
    )
    exit_diag["build_elapsed_sec"] = time.time() - t0
    print(
        f"  rows={exit_diag['rows']} used_candidates={exit_diag['used_candidates']} "
        f"positive_rate={exit_diag['positive_rate']:.4f} elapsed={exit_diag['build_elapsed_sec']:.1f}s",
        flush=True,
    )
    return x_exit_raw, y_exit, frame_exit, exit_diag


def _fit_expert(x: pd.DataFrame, y: np.ndarray, route_w: np.ndarray, *, seed: int) -> tuple[Any, dict[str, Any]]:
    """Same 85/15 chronological-within-dataset-order split and the same
    compute_sample_weight(balanced) * route_prob weighting as
    train_eval_omega4_1_exit_head_price_move_sltp_retrain_20260622._fit_exit_head_only, so the only
    thing that differs from the TabM per-expert exit-head-only retrain is the model class."""
    weights = compute_sample_weight(class_weight="balanced", y=y).astype(np.float64) * route_w.astype(np.float64)
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        raise RuntimeError("invalid GBDT sample weights")
    n = len(y)
    split = max(int(n * 0.85), min(n - 1, 256))
    train_idx, val_idx = np.arange(split), np.arange(split, n)

    if GBDT_LIBRARY == "lightgbm":
        model = lgb.LGBMClassifier(objective="binary", random_state=int(seed), verbosity=-1, n_jobs=-1, **GBDT_PARAMS)
    else:
        model = HistGradientBoostingClassifier(
            max_iter=GBDT_PARAMS["n_estimators"], max_leaf_nodes=GBDT_PARAMS["num_leaves"],
            learning_rate=GBDT_PARAMS["learning_rate"], random_state=int(seed),
        )
    model.fit(x.iloc[train_idx], y[train_idx], sample_weight=weights[train_idx])

    proba_val = model.predict_proba(x.iloc[val_idx])[:, 1]
    diag = {
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "val_auc": float(roc_auc_score(y[val_idx], proba_val)) if len(np.unique(y[val_idx])) > 1 else None,
        "val_logloss": float(log_loss(y[val_idx], proba_val, labels=[0, 1])),
        "val_positive_rate": float(np.mean(y[val_idx])),
        "train_positive_rate": float(np.mean(y[train_idx])),
        "route_weight_sum": float(route_w.sum()),
    }
    return model, diag


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES_DEFAULT)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--skip-reference-check", action="store_true")
    args = ap.parse_args()

    print(f"stage=start gbdt_library={GBDT_LIBRARY}", flush=True)
    out_dir = OUT_DIR if not str(args.out_suffix).strip() else OUT_DIR.parent / f"{MODEL_ID}_{str(args.out_suffix).strip()}"
    (out_dir / "h48qual").mkdir(parents=True, exist_ok=True)

    x_exit_raw, y_exit, frame_exit, exit_diag = _build_dataset(int(args.max_candidates))

    reference_check: dict[str, Any] | None = None
    if int(args.max_candidates) == MAX_CANDIDATES_DEFAULT and REFERENCE_REPORT.exists() and not args.skip_reference_check:
        ref = json.loads(REFERENCE_REPORT.read_text(encoding="utf-8"))["dataset"]
        reference_check = {
            "rows_match": int(exit_diag["rows"]) == int(ref["rows"]),
            "positive_count_match": int(exit_diag["positive_count"]) == int(ref["positive_count"]),
            "used_candidates_match": int(exit_diag["used_candidates"]) == int(ref["used_candidates"]),
            "rebuilt_rows": int(exit_diag["rows"]), "reference_rows": int(ref["rows"]),
            "rebuilt_positive_count": int(exit_diag["positive_count"]), "reference_positive_count": int(ref["positive_count"]),
        }
        print(f"stage=dataset_reference_check {reference_check}", flush=True)
        if not (reference_check["rows_match"] and reference_check["positive_count_match"]):
            print(
                "WARNING: rebuilt dataset does NOT match the original full1500 TabM run's report.json -- "
                "the GBDT-vs-TabM comparison would not be apples-to-apples. Not aborting (recorded "
                "prominently in report.json instead, per instructions not to force a positive result).",
                flush=True,
            )

    # base_cols is a frozen 102-column feature contract, identical across the original/TabM-liveATR
    # h48qual bundles (bit-identical encoder/direction_head/quality_head, independently verified in
    # docs/experiments/eth_omega461_exit_head_asymmetric_shadow_20260813.md) -- read it from the
    # SAME source _retrain_component_exit_head_liveatr used for the TabM run, not re-derived.
    baseline_bundle_path = h48cons.sweep.COMPONENTS["h48qual"]["bundle"]
    base_cols = list(torch.load(baseline_bundle_path, map_location="cpu", weights_only=False)["base_cols"])
    x_exit = parent._exit_input_from_position_rows(x_exit_raw, base_cols)
    route_probs = parent._route_probs(frame_exit)  # (n, 3), columns ordered bull/bear/chop per hard.ROUTE_COLS

    models: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        print(f"stage=fit_expert expert={expert}", flush=True)
        t0 = time.time()
        model, diag = _fit_expert(x_exit, y_exit, route_probs[:, idx], seed=SEED + idx)
        diag["fit_elapsed_sec"] = time.time() - t0
        models[expert] = model
        diagnostics[expert] = diag
        print(f"  {expert}: {diag}", flush=True)

    bundle_path = out_dir / "h48qual" / "gbdt_exit_bundle.pkl"
    with open(bundle_path, "wb") as f:
        pickle.dump(
            {
                "models": models, "base_cols": base_cols, "pos_cols": list(parent.POS_COLS),
                "model_id": MODEL_ID, "gbdt_library": GBDT_LIBRARY, "gbdt_params": GBDT_PARAMS,
                "baseline_bundle_base_cols_source": str(baseline_bundle_path),
            },
            f,
        )
    print(f"bundle={bundle_path}", flush=True)

    report = {
        "model_id": MODEL_ID,
        "gbdt_library": GBDT_LIBRARY,
        "gbdt_params": GBDT_PARAMS,
        "design": (
            "Same live-ATR-barrier candidate/label recipe and dataset as the TabM baseline "
            "(research_eth_omega461_exit_head_liveatr_relabel_20260813.py, seed=260813, "
            "max_candidates=1500), same per-expert soft route-probability sample weighting as "
            "_fit_exit_head_only. Only the model class (TabM neural net -> GBDT) differs."
        ),
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "dataset": exit_diag,
        "dataset_reference_check": reference_check,
        "expert_diagnostics": diagnostics,
        "bundle_path": str(bundle_path),
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(f"report={out_dir / 'report.json'}", flush=True)
    print("stage=done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
