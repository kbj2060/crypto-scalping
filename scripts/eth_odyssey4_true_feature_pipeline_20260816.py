#!/usr/bin/env python3
"""RESEARCH INFRASTRUCTURE -- corrected data-loading pipeline for the Odyssey4 (TabM) research
line, on the TRUE live 102-base-feature contract instead of the 185/172-column auto-detected
proxy (`_numeric_feature_cols`) that earlier sessions fell back to.

Background (verified directly, see docs/experiments/eth_odyssey4_true_feature_pipeline_recovery_
20260816.md for the full writeup): a parallel session concluded the retraining pipeline was
broken and fell back to a proxy feature set, citing a missing LSTM/chronos AI-context CSV. That
diagnosis was too broad. The actual gap was narrower:

  1. `train_eval_omega1_2_tabm_3head_20260603._prepare_frames()` gets its base features from
     `train_eval_omega1_2_tabm_diffusion_risk_20260603._load_omega_frames()`, which loads fine --
     every source CSV exists. The genuinely broken piece is that `_prepare_frames()` separately
     calls `train_omega1_regime3_expert_direction_head_volpca_20260602._build_frame(year)` ONLY to
     extract the `zigzag_action` label column, and that call chain (volpca.ctx._build_frame ->
     tsfm_chronos._build_frame(include_core=True) -> base._exact_join(..., DIR3_VSNLSTM, ...))
     depends on a dead vsnlstm CSV (data/ensemble/supervised/omega1_dir3_vsnlstm_full_20260531/
     training_features_*.csv, missing -- only the model checkpoint survived). The label itself has
     a standalone, dependency-free source:
     `train_omega1_direction_head_direction_only_20260602._add_labels(year)`, which reads
     tmp/causal_regen_20260516/zigzag_action_labels_20260531/zigzag_action_labels_<year>.csv
     directly. This module uses that path (imported below as `label_base`), same as the existing
     `_prepare_frames_light()` pattern in
     research_eth_candidate_faithful_tabm_batchensemble_cheap_gate_20260816.py.

  2. `_numeric_feature_cols(train, eval_df)` auto-detects every numeric column common to both
     frames (172 as of 2026-08-16) -- MORE than the true 102, because these shared CSVs have
     accumulated extra research columns over time. This module reindexes to the EXACT ordered 102
     `base_cols` recorded in the live deployed bundle instead (see `true_base_cols()`).

  3. Of the true 102, all are present in `eval_df` (2026) but 7 are missing from `train` (2025):
     fibonacci_level, funding_roc_12, funding_roc_48, funding_z_score, short_squeeze_risk,
     hurst_288, regime_persistence. All 7 are recoverable directly from columns ALREADY present in
     `train` (last_funding_rate, oi_change_rate, high, low, close, mtf_trend_1h, mtf_trend_4h,
     hma_slope, chop_index, breakout_strength) -- see `compute_missing_train_columns()`, which
     replicates the exact live formulas from features/engineering.py and
     features/high_order_state.py (cited per-function). Fidelity verified by recomputing the SAME
     7 columns on eval_df (which already has real live-pipeline values) and diffing: 5/7 match to
     float precision or near-exact; the remaining funding_roc_12/48/funding_z_score mismatches
     (11, 47, 287 rows out of 16897) are fully explained by a cold-start edge effect at eval_df's
     own 2026-01-01 boundary (no pre-boundary history within the per-year CSV), not a formula bug
     -- see the experiment doc for the row-by-row trace. The same edge effect applies to train's
     own first ~288 rows (2025-01-01, <=0.3% of the 105064-row year); documented as a known, minor
     limitation rather than silently ignored.

Provides `prepare_frames_true()`, a drop-in replacement for
`train_eval_omega1_2_tabm_3head_20260603._prepare_frames()` / the cheap_gate script's
`_prepare_frames_light()` -- same returned dict shape, but `feature_cols` is the true ordered
102-column live contract instead of an auto-detected proxy set.

Does NOT modify any existing script, the live deployed bundle, or trading_bot_modules/ -- read-only
research infrastructure. Does NOT re-run the A1/C1/C2/C3 experiment suite (out of scope here).
"""
from __future__ import annotations

import sys
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

import train_eval_omega1_2_tabm_3head_20260603 as base3head  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega1_2_tabm_exit_head_20260603 as exit_head  # noqa: E402
import train_omega1_direction_head_direction_only_20260602 as label_base  # noqa: E402

TRUE_BUNDLE_PATH = (
    ROOT
    / "tmp/causal_regen_20260516"
    / "omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_20260630"
    / "true_3head_tabm_bundle.pt"
)

SPLIT_TS = base3head.SPLIT_TS

# Confirmed 2026-08-16: exactly these 7 of the true 102 base_cols are missing from `train`
# (2025) but already present in `eval_df` (2026). If this drifts (e.g. the shared CSVs get
# regenerated), `prepare_frames_true()` raises rather than silently reindexing with NaN/zeros.
MISSING_TRAIN_COLS = [
    "fibonacci_level",
    "funding_roc_12",
    "funding_roc_48",
    "funding_z_score",
    "short_squeeze_risk",
    "hurst_288",
    "regime_persistence",
]

_REQUIRED_RAW_INPUTS = [
    "last_funding_rate", "oi_change_rate", "high", "low", "close",
    "mtf_trend_1h", "mtf_trend_4h", "hma_slope", "chop_index", "breakout_strength",
]


def true_base_cols() -> list[str]:
    """Ground-truth ordered 102 base feature names used by the LIVE deployed Odyssey4 bundle."""
    bundle = torch.load(TRUE_BUNDLE_PATH, map_location="cpu", weights_only=False)
    cols = list(bundle["base_cols"])
    if len(cols) != 102:
        raise RuntimeError(f"expected 102 base_cols in {TRUE_BUNDLE_PATH}, got {len(cols)}")
    return cols


# ---------------------------------------------------------------------------------------------
# Replicated formulas for the 7 columns missing from `train`, verbatim from
# features/engineering.py and features/high_order_state.py (cited per function).
# ---------------------------------------------------------------------------------------------

def _funding_roc(funding_rate: pd.Series, window: int) -> pd.Series:
    """FundingRateMomentum._calculate_roc, features/engineering.py:1045-1049."""
    shifted = funding_rate.shift(window)
    roc = (funding_rate - shifted) / (shifted.abs().clip(lower=1e-4) + 1e-8)
    return roc.clip(-10, 10).fillna(0)


def _funding_zscore(funding_rate: pd.Series, window: int) -> pd.Series:
    """FundingRateMomentum._calculate_zscore, features/engineering.py:1051-1055."""
    mean = funding_rate.rolling(window, min_periods=1).mean()
    std = funding_rate.rolling(window, min_periods=1).std()
    return ((funding_rate - mean) / (std + 1e-8)).fillna(0)


def _short_squeeze_risk(funding_rate: pd.Series, funding_roc_12: pd.Series, oi_change_rate: pd.Series) -> pd.Series:
    """FundingRateMomentum._short_squeeze_score + ._funding_extreme(sign=-1.0), fixed
    (non-adaptive) mode -- ETH's live default (adaptive_squeeze=False), features/engineering.py:
    1057-1088."""
    funding_extreme = np.clip(-1.0 * funding_rate / 0.0002, 0, 1)
    funding_plunge = np.clip(-funding_roc_12 / 3, 0, 1)
    oi_buildup = np.clip(oi_change_rate * 10, 0, 1)
    return 0.5 * funding_extreme + 0.3 * funding_plunge + 0.2 * oi_buildup


def _fibonacci_level(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """QuantSignalFeatures._fibonacci, features/engineering.py:1000-1009."""
    window = 288
    swing_high = high.rolling(window).max()
    swing_low = low.rolling(window).min()
    swing_range = swing_high - swing_low
    position = (close.to_numpy() - swing_low.to_numpy()) / (swing_range.to_numpy() + 1e-8)
    return pd.Series(position, index=close.index).clip(0, 1).fillna(0.5)


def _hurst(close: pd.Series, window: int) -> pd.Series:
    """HurstExponentFeatures._rolling_hurst_fast, features/engineering.py:1119-1134."""
    returns = close.pct_change().fillna(0)

    def rs_hurst(x: np.ndarray) -> float:
        if len(x) < 10:
            return 0.5
        mean_r = x.mean()
        deviate = np.cumsum(x - mean_r)
        r = deviate.max() - deviate.min()
        s = x.std()
        if s < 1e-10:
            return 0.5
        return float(np.log(r / s + 1e-10) / np.log(len(x)))

    return returns.rolling(window, min_periods=window // 2).apply(rs_hurst, raw=True).fillna(0.5)


def _regime_persistence(
    mtf_1h: pd.Series, mtf_4h: pd.Series, hma_slope: pd.Series, chop_index: pd.Series, breakout_strength: pd.Series
) -> pd.Series:
    """add_high_order_state_features, features/high_order_state.py:64-73 (regime_persistence
    block only -- the other 4 high-order-state outputs are not among the missing 7)."""
    mtf_1h = pd.to_numeric(mtf_1h, errors="coerce").fillna(0.0)
    mtf_4h = pd.to_numeric(mtf_4h, errors="coerce").fillna(0.0)
    hma_slope = pd.to_numeric(hma_slope, errors="coerce").fillna(0.0)
    chop_index = pd.to_numeric(chop_index, errors="coerce").fillna(0.0)
    breakout_strength = pd.to_numeric(breakout_strength, errors="coerce").fillna(0.0)

    trend_dir = np.sign((mtf_1h + mtf_4h + hma_slope).fillna(0.0))
    break_dir = np.sign(breakout_strength.fillna(0.0))
    chop_flag = (chop_index > 61.8).astype(float)
    regime_code = pd.Series(
        np.where(chop_flag > 0.5, 0.0, np.where(break_dir != 0.0, break_dir, trend_dir)),
        index=mtf_1h.index,
    )
    regime_change = regime_code.ne(regime_code.shift(1)).cumsum()
    streak = regime_code.groupby(regime_change).cumcount() + 1
    persistence_core = np.log1p(streak.astype(float)) / np.log(49.0)
    persistence_strength = 0.55 * (mtf_1h.abs() + mtf_4h.abs()) + 0.45 * breakout_strength.abs()
    raw = np.sign(regime_code) * persistence_core * (1.0 + persistence_strength)
    out = np.tanh(raw.astype(float) / 1.8)
    return pd.Series(out, index=mtf_1h.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def compute_missing_train_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 7 base columns in MISSING_TRAIN_COLS, matching the live feature engine's exact
    formulas. All required raw inputs are already present in `df` -- purely causal rolling-window
    / shift ops (no negative shifts, no future rows referenced), so no lookahead is introduced.
    Returns a frame indexed identically to `df`, columns in MISSING_TRAIN_COLS order."""
    missing_inputs = [c for c in _REQUIRED_RAW_INPUTS if c not in df.columns]
    if missing_inputs:
        raise RuntimeError(f"compute_missing_train_columns: required input columns missing: {missing_inputs}")

    out = pd.DataFrame(index=df.index)
    funding = pd.to_numeric(df["last_funding_rate"], errors="coerce")
    out["fibonacci_level"] = _fibonacci_level(df["high"], df["low"], df["close"])
    out["funding_roc_12"] = _funding_roc(funding, 12)
    out["funding_roc_48"] = _funding_roc(funding, 48)
    out["funding_z_score"] = _funding_zscore(funding, 288)
    out["short_squeeze_risk"] = _short_squeeze_risk(
        funding, out["funding_roc_12"], pd.to_numeric(df["oi_change_rate"], errors="coerce")
    )
    out["hurst_288"] = _hurst(df["close"], 288)
    out["regime_persistence"] = _regime_persistence(
        df["mtf_trend_1h"], df["mtf_trend_4h"], df["hma_slope"], df["chop_index"], df["breakout_strength"]
    )
    if list(out.columns) != MISSING_TRAIN_COLS:
        raise RuntimeError("compute_missing_train_columns: column order drift vs MISSING_TRAIN_COLS")
    return out


def _assert_clean_102(frame: pd.DataFrame, base_cols: list[str], name: str) -> None:
    missing = [c for c in base_cols if c not in frame.columns]
    if missing:
        raise RuntimeError(f"{name}: missing true base_cols after merge: {missing}")
    block = frame.reindex(columns=base_cols).apply(pd.to_numeric, errors="coerce")
    arr = block.to_numpy(dtype=np.float64)
    if not np.isfinite(arr).all():
        bad_col_mask = ~np.isfinite(arr).all(axis=0)
        bad_cols = [c for c, bad in zip(block.columns, bad_col_mask) if bad]
        raise RuntimeError(f"{name}: NaN/inf present in true 102-col base matrix, cols={bad_cols[:20]}")


def prepare_frames_true(*, disable_tp_sl: bool = False) -> dict[str, Any]:
    """Drop-in replacement for base3head._prepare_frames() / the cheap_gate script's
    _prepare_frames_light(): same returned dict shape (train_raw, val_raw, oos_raw, train_df,
    train_fixed, s_train_label, feature_cols, overlay_report), but on the TRUE ordered 102-column
    live base-feature contract."""
    omega.BASE_TEMPLATE["max_hold"] = 0
    omega.BASE_TEMPLATE["cooldown"] = 0
    train_all, eval_df, overlay_report = omega._load_omega_frames()
    n_train_before = len(train_all)
    n_eval_before = len(eval_df)

    base_cols = true_base_cols()
    missing_from_train = sorted(c for c in base_cols if c not in train_all.columns)
    if missing_from_train != sorted(MISSING_TRAIN_COLS):
        raise RuntimeError(
            "train base_cols gap changed since the 2026-08-16 audit -- "
            f"expected exactly {sorted(MISSING_TRAIN_COLS)} missing, got {missing_from_train}. "
            "Re-derive compute_missing_train_columns before trusting this pipeline."
        )
    missing_from_eval = [c for c in base_cols if c not in eval_df.columns]
    if missing_from_eval:
        raise RuntimeError(f"eval_df missing true base_cols (unexpected, was fully present as of 2026-08-16): {missing_from_eval}")

    recovered = compute_missing_train_columns(train_all)
    if not recovered.index.equals(train_all.index):
        raise RuntimeError("compute_missing_train_columns: index mismatch vs train_all")
    train_all = pd.concat([train_all, recovered], axis=1)

    label_2025 = label_base._add_labels(2025)
    label_2026 = label_base._add_labels(2026)
    train_all, train_labels = omega._align(train_all, label_2025, "true-pipeline train labels")
    eval_df, eval_labels = omega._align(eval_df, label_2026, "true-pipeline oos labels")
    train_all = train_all.copy()
    eval_df = eval_df.copy()
    train_all["zigzag_action"] = pd.to_numeric(train_labels["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    eval_df["zigzag_action"] = pd.to_numeric(eval_labels["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)

    _assert_clean_102(train_all, base_cols, "train")
    _assert_clean_102(eval_df, base_cols, "eval_df")

    train_raw = train_all[train_all["timestamp"] < SPLIT_TS].reset_index(drop=True)
    val_raw = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)

    tabm_2025 = omega._read(omega.TABM_2025)
    train_df, train_src = omega._align(train_raw, tabm_2025, "true-pipeline train")
    train_fixed = omega._to_fixed_decisions(train_src, oof=True)
    if disable_tp_sl:
        train_fixed = exit_head._disable_tp_sl(train_fixed)
    s_train_label = base3head._base_input(train_df, base_cols)

    return {
        "train_raw": train_raw,
        "val_raw": val_raw,
        "oos_raw": eval_df.reset_index(drop=True),
        "train_df": train_df,
        "train_fixed": train_fixed,
        "s_train_label": s_train_label,
        "feature_cols": base_cols,
        "overlay_report": overlay_report,
        "recovered_train_cols": list(MISSING_TRAIN_COLS),
        "row_counts": {
            "train_all_before_label_align": n_train_before,
            "eval_df_before_label_align": n_eval_before,
            "train_raw": len(train_raw),
            "val_raw": len(val_raw),
            "oos_raw": len(eval_df),
        },
    }


def _fidelity_check() -> None:
    """Recompute the 7 columns on eval_df (2026), which already has real live-pipeline values for
    them, and diff -- validates formula fidelity independent of the train-side recovery."""
    train_all, eval_df, _ = omega._load_omega_frames()
    recomputed = compute_missing_train_columns(eval_df)
    print("=== fidelity check: recompute-on-eval_df (2026) vs real stored eval_df values ===")
    for col in MISSING_TRAIN_COLS:
        real = pd.to_numeric(eval_df[col], errors="coerce").to_numpy(dtype=np.float64)
        mine = recomputed[col].to_numpy(dtype=np.float64)
        diff = np.abs(real - mine)
        n_mismatch = int((diff > 1e-6).sum())
        print(
            f"  {col:20s} max_abs_diff={np.nanmax(diff):.6g} mean_abs_diff={np.nanmean(diff):.6g} "
            f"n_mismatch(>1e-6)={n_mismatch}/{len(diff)}"
        )
        if n_mismatch:
            last_mismatch_idx = int(np.flatnonzero(diff > 1e-6).max())
            print(f"    (last mismatch at row {last_mismatch_idx} -- consistent with a leading cold-start edge effect)")

    print()
    print("=== train (2025, recovered) vs eval_df (2026, real) distribution sanity check ===")
    recovered_train = compute_missing_train_columns(train_all)
    for col in MISSING_TRAIN_COLS:
        a = recovered_train[col].to_numpy(dtype=np.float64)
        b = pd.to_numeric(eval_df[col], errors="coerce").to_numpy(dtype=np.float64)
        print(f"  {col:20s} train2025 mean={np.nanmean(a):.6g} std={np.nanstd(a):.6g} min={np.nanmin(a):.6g} max={np.nanmax(a):.6g}")
        print(f"  {'':20s} eval2026  mean={np.nanmean(b):.6g} std={np.nanstd(b):.6g} min={np.nanmin(b):.6g} max={np.nanmax(b):.6g}")
        n_nan = int(np.isnan(a).sum())
        n_inf = int(np.isinf(a).sum())
        if n_nan or n_inf:
            print(f"  {'':20s} !! n_nan={n_nan} n_inf={n_inf}")


def main() -> int:
    _fidelity_check()
    print()
    print("=== prepare_frames_true() ===")
    frames = prepare_frames_true()
    print(f"  feature_cols: {len(frames['feature_cols'])} (must be 102): match={len(frames['feature_cols']) == 102}")
    print(f"  row_counts: {frames['row_counts']}")
    print(f"  recovered_train_cols: {frames['recovered_train_cols']}")
    for name in ("train_raw", "val_raw", "oos_raw"):
        fr = frames[name]
        block = fr.reindex(columns=frames["feature_cols"]).to_numpy(dtype=np.float64)
        print(f"  {name}: shape={fr.shape} finite_102col={bool(np.isfinite(block).all())}")
        y = fr["zigzag_action"].to_numpy()
        uniq = sorted(pd.unique(y).tolist())
        print(f"  {name}: zigzag_action unique={uniq} n_null={int(pd.isna(fr['zigzag_action']).sum())}")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
