#!/usr/bin/env python3
"""Odyssey2 #20 -- evidence-signal reversal features -> risk-sizing GBM context feature (user
decision: after Candidate C, the hard exit-veto rule, was rejected on VAL
[docs/experiments/eth_omega461_evidence_veto_exit_overlay_20260814.md], and the exit_head-retrain
feature-injection pre-gate came back weak/VAL-contradicted [docs/experiments/
eth_omega461_evidence_signal_exit_head_feature_rank_correlation_20260814.md], user chose to close
that axis and switch to the ORIGINAL design research doc's Candidate D: inject the evidence signals
into the sizing sidecar via `train_eval_omega4_2_risk_sidecar_20260622.py`'s `--risk-context-
feature-dir` extension point -- the same mechanism Odyssey2 #2 (ensemble-epistemic) and #3
(autoencoder-latent) already used for this exact injection point).

Structure copied from build_eth_autoencoder_latent_context_features_20260813.py (COMPONENT_CONFIG,
omega4._prepare_frames() call for row alignment, panel join, trend_ctx_* output convention) -- only
the feature computation differs: instead of fitting a new autoencoder, this reuses the ALREADY
cross-window-validated evidence-signal formulas verbatim from the evidence-study lineage
(scripts/analyze_eth_creative_reversal_evidence_signals_20260814.py,
scripts/backtest_eth_slowk_williamsr_persistence_confluence_20260814.py,
scripts/analyze_eth_deep_evidence_signal_sweep_round2_20260814.py,
scripts/analyze_eth_broad_evidence_signal_sweep_20260814.py) -- no new thresholds, no new training.

=== Feature set (6 columns, fixed BEFORE any retrain result is seen) ===
Deliberately RAW/continuous where the original evidence-study used a boolean threshold, and RAW
booleans (as 0/1) only for the one signal (liquidity_sweep) that is genuinely event-shaped rather
than a magnitude -- per the user's stated intent ("let the model learn it") rather than
hand-combining signals into a single trigger (the mistake candidate C's veto made: an AND-of-3
boolean throws away magnitude information a GBM could use on its own):
  trend_ctx_taker_delta_z      -- net aggressive buy/sell volume z-score (order flow), signed:
                                   very negative = sell climax (bottom evidence), very positive =
                                   buy climax (top evidence). From add_creative_indicators, raw.
  trend_ctx_p_fast             -- adaptive Fast-%K rolling-864 percentile (0=bottom decile,
                                   1=top decile). From compute_indicators, raw.
  trend_ctx_p_slow             -- adaptive Slow-%K rolling-864 percentile, same source.
  trend_ctx_ret3_z             -- 3-bar (15min) return z-score (short_term_return_z, this session's
                                   scorecard #4 stable signal, pure price, no order-flow needed).
  trend_ctx_liquidity_sweep_low  -- boolean (0/1): wick pokes below prior 48-bar swing low, closes
                                   back inside (ICT stop-hunt). Most SYMMETRIC strong signal in the
                                   scorecard (near-equal bottom/top lift) -- kept as its own two
                                   columns rather than combined into one signed feature since the
                                   sweep-low/sweep-high triggers are structurally distinct events,
                                   not two ends of one continuous quantity.
  trend_ctx_liquidity_sweep_high -- symmetric top-side sweep.
Deliberately excluded: orthogonal_combo, taker_sell/buy_climax as SEPARATE boolean columns --
these are just AND-combinations/thresholded versions of taker_delta_z/p_fast/p_slow already
included above; adding them too would (a) not give the GBM new information, only redundant
near-collinear columns, and (b) risk the exact "combining redundant same-family indicators
doesn't help" failure mode this session's own scorecard already documented for %R+SlowK.
volume_wick_climax was left out (needs vol_z/wick_ratio, a 7th+8th column, for the weakest-marginal
addition among the top-5 stable signals) to keep the context block compact -- this project's TCN
hpsearch already found the widest feature set performed worst in a low-SNR setting
(docs/model_contracts/odyssey_eth_h48qual_corrected_tabm_20260811_contract.md, raw_wide 0/15).

fresh_forward_bar_by_bar=true (every formula below is rolling/shift-only, verified by direct
inspection of the reused source functions before this script's use, no negative shift anywhere).
trade_ledgers_used_as_input=false. saved_parent_exit_timestamps_used=false.
future_rows_used_for_entry=false.

Does NOT touch trading_bot.py / trading_bot_modules/omega4_6_1_live.py / runtime_config.py / .env.
Does NOT modify train_eval_omega4_2_risk_sidecar_20260622.py, analyze_eth_creative_reversal_
evidence_signals_20260814.py, backtest_eth_slowk_williamsr_persistence_confluence_20260814.py,
analyze_eth_broad_evidence_signal_sweep_20260814.py -- all imported/read only. No GPU needed (pure
pandas feature computation, no model fit).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_2_risk_sidecar_20260622 as sidecar_script  # noqa: E402
from analyze_eth_creative_reversal_evidence_signals_20260814 import add_creative_indicators  # noqa: E402
from backtest_eth_slowk_williamsr_persistence_confluence_20260814 import compute_indicators  # noqa: E402

omega, omega4 = sidecar_script.omega, sidecar_script.omega4

PANEL_PATH = ROOT / "data/splits/year_oos/eth_features_2024_2026_analysis.csv"

# Copied verbatim from build_eth_autoencoder_latent_context_features_20260813.py (proven working
# row-alignment recipe for both components) -- not re-derived.
COMPONENT_CONFIG = {
    "h48qual": {
        "train_csv": ROOT / "tmp/causal_regen_20260516/alpha5_a5dir_2024_train_2025_score_20260529_cleanfunding_stable48/02_fixed_regime4_state24_sticky090_tp18_sl10_preprocess_2024_to_2025/trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv",
        "eval_csv": ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2026_alpha6_current_tail111_exact.csv",
        "direction_label_dir": ROOT / "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531",
        "quality_mode": "same_as_direction",
    },
    "zig075": {
        "train_csv": ROOT / "tmp/causal_regen_20260516/alpha5_a5dir_2024_train_2025_score_20260529_cleanfunding_stable48/02_fixed_regime4_state24_sticky090_tp18_sl10_preprocess_2024_to_2025/trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv",
        "eval_csv": ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2026_alpha6_current_tail111_exact.csv",
        "direction_label_dir": ROOT / "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531",
        "quality_mode": "same_as_direction",
    },
}


def log(msg: str) -> None:
    print(f"[evidence_ctx] {msg}", flush=True)


def _compute_evidence_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Computes the 6 context columns on the FULL panel (never split-by-split), matching this
    session's own established discipline (Odyssey2 #11's dual_momentum score, this session's
    Candidate C's _evidence_veto_score) for avoiding artificial NaN truncation at a window's own
    start. Reuses compute_indicators (p_fast/p_slow) and add_creative_indicators (delta_z)
    unmodified; liquidity_sweep and ret3_z are 2-3 line expressions copied verbatim from
    analyze_eth_broad_evidence_signal_sweep_20260814.py add_broad_indicators / analyze_eth_deep_
    evidence_signal_sweep_round2_20260814.py (neither exposes those two lines as a standalone
    importable function, so the exact expression is reproduced here rather than imported)."""
    df = panel[["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"]].copy()
    ind = compute_indicators(df)
    ind = add_creative_indicators(ind)

    close, low, high = df["close"], df["low"], df["high"]
    ret3 = close / close.shift(3) - 1.0
    ret3_mean, ret3_std = ret3.rolling(288, min_periods=288).mean(), ret3.rolling(288, min_periods=288).std()
    ret3_z = (ret3 - ret3_mean) / ret3_std.replace(0.0, np.nan)

    swing_low_prior = low.rolling(48, min_periods=48).min().shift(1)
    swing_high_prior = high.rolling(48, min_periods=48).max().shift(1)
    sweep_low = (low < swing_low_prior) & (close > swing_low_prior)
    sweep_high = (high > swing_high_prior) & (close < swing_high_prior)

    out = pd.DataFrame({"timestamp": df["timestamp"].to_numpy()})
    out["trend_ctx_taker_delta_z"] = ind["delta_z"].fillna(0.0).to_numpy()
    out["trend_ctx_p_fast"] = ind["p_fast"].fillna(0.5).to_numpy()
    out["trend_ctx_p_slow"] = ind["p_slow"].fillna(0.5).to_numpy()
    out["trend_ctx_ret3_z"] = ret3_z.fillna(0.0).to_numpy()
    out["trend_ctx_liquidity_sweep_low"] = sweep_low.fillna(False).astype(np.float32).to_numpy()
    out["trend_ctx_liquidity_sweep_high"] = sweep_high.fillna(False).astype(np.float32).to_numpy()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--component", choices=list(COMPONENT_CONFIG.keys()), required=True)
    args = ap.parse_args()
    cfg = COMPONENT_CONFIG[args.component]

    out_dir = ROOT / f"tmp/causal_regen_20260516/eth_{args.component}_evidence_signal_context_20260814"
    out_dir.mkdir(parents=True, exist_ok=True)

    omega.TRAIN_CSV = Path(cfg["train_csv"])
    omega.EVAL_CSV = Path(cfg["eval_csv"])
    log("stage=prepare_frames (sidecar's own frame construction, for row alignment)")
    frames = omega4._prepare_frames(
        disable_tp_sl=False, direction_label_dir=Path(cfg["direction_label_dir"]), quality_mode=str(cfg["quality_mode"]),
        quality_label_dir=None, quality_min_edge=0.0, quality_max_mae=0.0, quality_min_mfe_mae=0.0, quality_max_hold_bars=0,
    )
    train_frame, val_frame, oos_frame = frames["train_raw"], frames["val_raw"], frames["oos_raw"]
    log(f"  train={len(train_frame)} val={len(val_frame)} oos={len(oos_frame)}")

    log(f"stage=load_panel ({PANEL_PATH.name}, committed, 2024-06..2026-08 coverage)")
    panel = pd.read_csv(PANEL_PATH, low_memory=False)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)

    log("stage=compute_evidence_signal_panel (full-panel, causal, no per-window truncation)")
    evidence = _compute_evidence_panel(panel)

    def join_evidence(frame: pd.DataFrame, split: str) -> pd.DataFrame:
        merged = frame[["timestamp"]].merge(evidence, on="timestamp", how="left", validate="one_to_one")
        ctx_cols = [c for c in evidence.columns if c != "timestamp"]
        missing = merged[ctx_cols].isna().any(axis=1).sum()
        if missing:
            raise RuntimeError(f"{split}: {missing} rows have no matching evidence-panel timestamp -- coverage gap")
        return merged

    for split, frame in [("train", train_frame), ("validation", val_frame), ("oos", oos_frame)]:
        ctx = join_evidence(frame, split)
        out_path = out_dir / f"{split}_context_features.csv"
        ctx.to_csv(out_path, index=False)
        log(f"  wrote {out_path} rows={len(ctx)}")

    log(f"DONE component={args.component} out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
