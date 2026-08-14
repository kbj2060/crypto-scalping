#!/usr/bin/env python3
"""Out-of-window robustness check: does the evidence-signal ranking built on VAL 2025-09-01..
2025-12-31 + OOS 2026-01-01..2026-02-17 (docs/experiments/eth_confluence_oscillator_bottom_top_
evidence_20260814.md + 3 follow-up rounds, same day) hold up on a DIFFERENT window, 2026-03-01..
2026-07-20 (requested through July; data.year_oos/training_features_2026_rebuilt.csv, this
sub-project's own extended-2026 source, ends 2026-07-20 -- the closest available to "through
July")?

Reuses the exact same signal formulas as the 4 prior rounds (copied verbatim from
analyze_eth_confluence_oscillator_bottom_top_evidence_20260814.py, analyze_eth_creative_reversal_
evidence_signals_20260814.py, analyze_eth_broad_evidence_signal_sweep_20260814.py, and
analyze_eth_deep_evidence_signal_sweep_round2_20260814.py) and the same event_study/excess_move/
race_outcomes/load_zigzag_pivots harness (imported unmodified) -- only the raw data source and
window change, so any ranking difference is attributable to the regime, not a methodology drift.

Two signals from the prior rounds are DROPPED here and disclosed, not silently skipped:
  - A5 btc_lead_sell/buy_climax needed BTC's own taker_buy_base at 5m resolution; no extended-
    2026 BTC order-flow source was found on disk for Mar-Jul 2026 (only close_btc/volume_btc are
    available in the rebuilt panel). Re-adding it would need pulling fresh BTC order-flow data.
  - The session-of-day breakdown (round 2's category C) is not repeated here -- out of scope for
    a ranking-stability check specifically.
Everything else (16 reversal signals + 5 continuation signals + funding-rate flip) is reproduced.

Prior-window numbers are NOT recomputed -- they are read back from the CSVs each round already
wrote to tmp/, to guarantee the comparison uses the exact numbers already reported to the user
rather than a second, possibly-drifted implementation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from analyze_eth_broad_evidence_signal_sweep_20260814 import add_broad_indicators, race_outcomes  # noqa: E402
from analyze_eth_confluence_oscillator_bottom_top_evidence_20260814 import (  # noqa: E402
    K_HORIZONS,
    event_study,
    excess_move,
    load_zigzag_pivots,
)
from analyze_eth_creative_reversal_evidence_signals_20260814 import add_creative_indicators  # noqa: E402
from analyze_eth_deep_evidence_signal_sweep_round2_20260814 import add_short_term_and_patterns  # noqa: E402
from backtest_eth_slowk_williamsr_persistence_confluence_20260814 import compute_indicators  # noqa: E402

REBUILT_2026 = ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"
NEW_START, NEW_END = pd.Timestamp("2026-03-01"), pd.Timestamp("2026-07-20")

PRIOR_TABLES = [
    ROOT / "tmp/eth_confluence_oscillator_bottom_top_evidence_20260814/evidence_table.csv",
    ROOT / "tmp/eth_creative_reversal_evidence_signals_20260814/evidence_table.csv",
    ROOT / "tmp/eth_broad_evidence_signal_sweep_20260814/reversal_evidence_table.csv",
    ROOT / "tmp/eth_deep_evidence_signal_sweep_round2_20260814/reversal_evidence_table.csv",
]
PRIOR_CONTINUATION_TABLES = [
    ROOT / "tmp/eth_broad_evidence_signal_sweep_20260814/continuation_evidence_table.csv",
    ROOT / "tmp/eth_deep_evidence_signal_sweep_round2_20260814/funding_flip_continuation_table.csv",
]

# Canonical short label -> (original CSV's exact "signal" string, side value in that CSV)
REVERSAL_LABELS = {
    "oscillator_adaptive_both": ("adaptive_both (p_fast<=.10 AND p_slow<=.10)", "adaptive_both (p_fast>=.90 AND p_slow>=.90)"),
    "orthogonal_combo": ("orthogonal_combo (adaptive_OS AND taker_sell_climax)", "orthogonal_combo (adaptive_OB AND taker_buy_climax)"),
    "volume_wick_climax": ("volume_wick_climax_low (vol_z>=2, lower_wick>=.5)", "volume_wick_climax_high (vol_z>=2, upper_wick>=.5)"),
    "taker_climax": ("taker_sell_climax (delta_z<=-2)", "taker_buy_climax (delta_z>=2)"),
    "bollinger_pctb_extreme": ("A2_bollinger_pctb_extreme_low", "A2_bollinger_pctb_extreme_high"),
    "liquidity_sweep": ("A3_liquidity_sweep_low", "A3_liquidity_sweep_high"),
    "cvd_divergence": ("cvd_divergence (price 4h-LL, CVD rising)", "cvd_divergence (price 4h-HH, CVD falling)"),
    "hurst_gated_oscillator": ("A4_hurst_gated_oscillator_low", "A4_hurst_gated_oscillator_high"),
    "vwap_extreme": ("vwap_extreme_low (dev_z<=-2)", "vwap_extreme_high (dev_z>=2)"),
    "funding_extreme": ("A6_funding_extreme_short_crowded", "A6_funding_extreme_long_crowded"),
    "ethbtc_ratio_extreme": ("A7_ethbtc_ratio_extreme_low", "A7_ethbtc_ratio_extreme_high"),
    "short_term_return_z": ("A8_short_term_return_z_low", "A8_short_term_return_z_high"),
    "mfi_divergence": ("A1_mfi_divergence", "A1_mfi_divergence"),
    "momentum_divergence": ("momentum_divergence (price 4h-LL, %R momentum UP)", "momentum_divergence (price 4h-HH, %R momentum DOWN)"),
    "candlestick_hammer_pure": ("A9_hammer_pure", "A9_shooting_star_pure"),
    "round_number_approach": ("A10_round_number_approach_down", "A10_round_number_approach_up"),
}
CONTINUATION_LABELS = {
    "squeeze_breakout": ("B1_squeeze_breakout_up", "B1_squeeze_breakout_down"),
    "adx_di_cross": ("B2_adx_di_cross_up", "B2_adx_di_cross_down"),
    "donchian_breakout": ("B3_donchian_breakout_up", "B3_donchian_breakout_down"),
    "btc_momentum_spillover": ("B4_btc_momentum_spillover_up", "B4_btc_momentum_spillover_down"),
    "donchian_trend_filtered": ("B5_donchian_trend_filtered_up", "B5_donchian_trend_filtered_down"),
    "funding_flip": ("B6_funding_flip_bullish", "B6_funding_flip_bearish"),
}


def load_prior_original_window(side_target: str) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in PRIOR_TABLES]
    all_rev = pd.concat(frames, ignore_index=True)
    all_rev = all_rev[(all_rev["horizon"] == "K12_1h")]
    rows = []
    for short_name, (bottom_label, top_label) in REVERSAL_LABELS.items():
        label = bottom_label if side_target == "bottom" else top_label
        side_val = "bottom" if side_target == "bottom" else "top"
        match = all_rev[(all_rev["signal"] == label) & (all_rev["side"] == side_val)]
        if match.empty:
            continue
        rows.append({"signal": short_name, "lift": float(match.iloc[0]["lift"]), "precision": float(match.iloc[0]["precision"])})
    return pd.DataFrame(rows)


def load_prior_original_continuation() -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in PRIOR_CONTINUATION_TABLES]
    all_cont = pd.concat(frames, ignore_index=True)
    all_cont = all_cont[all_cont["horizon"] == "K12_1h"]
    rows = []
    for short_name, (up_label, down_label) in CONTINUATION_LABELS.items():
        for direction, label in (("up", up_label), ("down", down_label)):
            match = all_cont[all_cont["signal"] == label]
            if match.empty:
                continue
            rows.append({"signal": short_name, "direction": direction, "lift": float(match.iloc[0]["lift"])})
    return pd.DataFrame(rows)


def load_new_window() -> pd.DataFrame:
    raw = pd.read_csv(
        REBUILT_2026,
        usecols=["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base", "taker_buy_quote",
                 "last_funding_rate", "close_btc"],
        parse_dates=["timestamp"],
    )
    raw = raw.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    frame = compute_indicators(raw).reset_index(drop=True)
    frame["price_roc_48"] = frame["close"] / frame["close"].shift(48) - 1.0
    frame = add_creative_indicators(frame)
    frame = add_broad_indicators(frame)
    frame = add_short_term_and_patterns(frame)

    frame["funding_rate"] = raw["last_funding_rate"]
    frame["funding_pctile"] = frame["funding_rate"].rolling(8640, min_periods=2880).rank(pct=True)  # ~30d of 5m bars

    ratio = frame["close"] / raw["close_btc"]
    ratio_mean, ratio_std = ratio.rolling(864, min_periods=864).mean(), ratio.rolling(864, min_periods=864).std()
    frame["ethbtc_z"] = (ratio - ratio_mean) / ratio_std.replace(0.0, np.nan)

    btc_ret_12 = raw["close_btc"] / raw["close_btc"].shift(12) - 1.0
    hi_q = btc_ret_12.rolling(864, min_periods=864).quantile(0.90)
    lo_q = btc_ret_12.rolling(864, min_periods=864).quantile(0.10)
    frame["btc_spillover_up"] = btc_ret_12 >= hi_q
    frame["btc_spillover_down"] = btc_ret_12 <= lo_q
    return frame


def build_reversal_signals(frame: pd.DataFrame, side: str) -> dict:
    if side == "bottom":
        return {
            "oscillator_adaptive_both": (frame["p_fast"] <= 0.10) & (frame["p_slow"] <= 0.10),
            "orthogonal_combo": (frame["p_fast"] <= 0.10) & (frame["p_slow"] <= 0.10) & (frame["delta_z"] <= -2.0),
            "volume_wick_climax": (frame["vol_z"] >= 2.0) & (frame["lower_wick_ratio"] >= 0.5),
            "taker_climax": frame["delta_z"] <= -2.0,
            "bollinger_pctb_extreme": frame["bb_pctb"] <= 0.05,
            "liquidity_sweep": frame["sweep_low"],
            "cvd_divergence": (frame["price_roc_48"] <= -0.01) & (frame["cvd_roll_roc_48"] >= 0),
            "hurst_gated_oscillator": (frame["p_fast"] <= 0.10) & (frame["p_slow"] <= 0.10) & frame["mean_reverting_regime"],
            "vwap_extreme": frame["vwap_dev_z"] <= -2.0,
            "funding_extreme": frame["funding_pctile"] <= 0.10,
            "ethbtc_ratio_extreme": frame["ethbtc_z"] <= -2.0,
            "short_term_return_z": frame["ret3_z"] <= -2.5,
            "mfi_divergence": (frame["price_roc_48"] <= -0.01) & (frame["mfi_roc_48"] >= 5),
            "momentum_divergence": (frame["price_roc_48"] <= -0.01) & (frame["fast_k_roc_48"] >= 5),
            "candlestick_hammer_pure": frame["hammer"],
            "round_number_approach": frame["near_round"] & (frame["price_roc_48"] <= -0.01),
        }
    return {
        "oscillator_adaptive_both": (frame["p_fast"] >= 0.90) & (frame["p_slow"] >= 0.90),
        "orthogonal_combo": (frame["p_fast"] >= 0.90) & (frame["p_slow"] >= 0.90) & (frame["delta_z"] >= 2.0),
        "volume_wick_climax": (frame["vol_z"] >= 2.0) & (frame["upper_wick_ratio"] >= 0.5),
        "taker_climax": frame["delta_z"] >= 2.0,
        "bollinger_pctb_extreme": frame["bb_pctb"] >= 0.95,
        "liquidity_sweep": frame["sweep_high"],
        "cvd_divergence": (frame["price_roc_48"] >= 0.01) & (frame["cvd_roll_roc_48"] <= 0),
        "hurst_gated_oscillator": (frame["p_fast"] >= 0.90) & (frame["p_slow"] >= 0.90) & frame["mean_reverting_regime"],
        "vwap_extreme": frame["vwap_dev_z"] >= 2.0,
        "funding_extreme": frame["funding_pctile"] >= 0.90,
        "ethbtc_ratio_extreme": frame["ethbtc_z"] >= 2.0,
        "short_term_return_z": frame["ret3_z"] >= 2.5,
        "mfi_divergence": (frame["price_roc_48"] >= 0.01) & (frame["mfi_roc_48"] <= -5),
        "momentum_divergence": (frame["price_roc_48"] >= 0.01) & (frame["fast_k_roc_48"] <= -5),
        "candlestick_hammer_pure": frame["shooting_star"],
        "round_number_approach": frame["near_round"] & (frame["price_roc_48"] >= 0.01),
    }


def build_continuation_signals(frame: pd.DataFrame, direction: str) -> dict:
    if direction == "up":
        return {
            "squeeze_breakout": frame["squeeze_on_prev"] & (frame["close"] > frame["kc_upper"]),
            "adx_di_cross": (frame["adx14"] >= 25) & (frame["adx14"].shift(1) < 25) & (frame["pdi"] > frame["ndi"]),
            "donchian_breakout": frame["close"] > frame["donch_high"],
            "btc_momentum_spillover": frame["btc_spillover_up"],
            "donchian_trend_filtered": (frame["close"] > frame["donch_high"]) & frame["trending_regime"],
            "funding_flip": (frame["funding_rate"] > 0) & (frame["funding_rate"].shift(1) <= 0),
        }
    return {
        "squeeze_breakout": frame["squeeze_on_prev"] & (frame["close"] < frame["kc_lower"]),
        "adx_di_cross": (frame["adx14"] >= 25) & (frame["adx14"].shift(1) < 25) & (frame["ndi"] > frame["pdi"]),
        "donchian_breakout": frame["close"] < frame["donch_low"],
        "btc_momentum_spillover": frame["btc_spillover_down"],
        "donchian_trend_filtered": (frame["close"] < frame["donch_low"]) & frame["trending_regime"],
        "funding_flip": (frame["funding_rate"] < 0) & (frame["funding_rate"].shift(1) >= 0),
    }


def main() -> None:
    frame = load_new_window()
    pivots = load_zigzag_pivots()
    ts = frame["timestamp"]
    window_mask = ((ts >= NEW_START) & (ts <= NEW_END)).to_numpy()
    print(f"New window: {NEW_START.date()}..{NEW_END.date()}, {int(window_mask.sum())} bars, "
          f"{int(pivots['timestamp'].between(NEW_START, NEW_END).sum())} zigzag pivots in-window")

    close = frame["close"].to_numpy()
    new_rev_rows = []
    for side in ("bottom", "top"):
        all_pos = np.flatnonzero(window_mask)
        pivot_pos = frame.index[frame["timestamp"].isin(pivots.loc[pivots["pivot_type"] == side, "timestamp"])].to_numpy()
        for name, mask in build_reversal_signals(frame, side).items():
            trigger_pos = np.flatnonzero(mask.fillna(False).to_numpy() & window_mask)
            stats = event_study(trigger_pos, pivot_pos, all_pos, K_HORIZONS["K12_1h"])
            new_rev_rows.append({"side": side, "signal": name, **stats})
    new_rev = pd.DataFrame(new_rev_rows)

    high, low, atr_pct = frame["high"].to_numpy(), frame["low"].to_numpy(), frame["atr_pct"].to_numpy()
    new_cont_rows = []
    baseline_cache: dict[tuple[int, int], np.ndarray] = {}
    all_pos = np.flatnonzero(window_mask)
    for direction, d in (("up", 1), ("down", -1)):
        for name, mask in build_continuation_signals(frame, direction).items():
            trigger_pos = np.flatnonzero(mask.fillna(False).to_numpy() & window_mask)
            key = d
            if key not in baseline_cache:
                baseline_cache[key] = race_outcomes(all_pos, d, close, high, low, atr_pct, K_HORIZONS["K12_1h"])
            base_out = baseline_cache[key]
            base_decided = base_out != 0
            base_rate = float((base_out[base_decided] == 1).mean()) if base_decided.any() else float("nan")
            out = race_outcomes(trigger_pos, d, close, high, low, atr_pct, K_HORIZONS["K12_1h"])
            decided = out != 0
            hit_rate = float((out[decided] == 1).mean()) if decided.any() else float("nan")
            lift = hit_rate / base_rate if base_rate and np.isfinite(base_rate) and base_rate > 0 else float("nan")
            new_cont_rows.append({"signal": name, "direction": direction, "n_triggers": int(len(trigger_pos)),
                                   "precision": hit_rate, "baseline_rate": base_rate, "lift": lift})
    new_cont = pd.DataFrame(new_cont_rows)

    pd.set_option("display.width", 170)

    print("\n\n=== REVERSAL: original-window vs new-window (1h, bottom side) ===")
    prior_bottom = load_prior_original_window("bottom")
    cmp_bottom = prior_bottom.merge(new_rev[new_rev["side"] == "bottom"][["signal", "lift", "precision"]],
                                     on="signal", suffixes=("_orig", "_new"))
    cmp_bottom["orig_rank"] = cmp_bottom["lift_orig"].rank(ascending=False).astype(int)
    cmp_bottom["new_rank"] = cmp_bottom["lift_new"].rank(ascending=False).astype(int)
    cmp_bottom["rank_change"] = cmp_bottom["orig_rank"] - cmp_bottom["new_rank"]
    cmp_bottom = cmp_bottom.sort_values("orig_rank")
    print(cmp_bottom.to_string(index=False))
    spearman_bottom = cmp_bottom["lift_orig"].corr(cmp_bottom["lift_new"], method="spearman")
    print(f"\nSpearman rank correlation (bottom, original vs new window): {spearman_bottom:.3f}")

    print("\n\n=== REVERSAL: original-window vs new-window (1h, top side) ===")
    prior_top = load_prior_original_window("top")
    cmp_top = prior_top.merge(new_rev[new_rev["side"] == "top"][["signal", "lift", "precision"]],
                               on="signal", suffixes=("_orig", "_new"))
    cmp_top["orig_rank"] = cmp_top["lift_orig"].rank(ascending=False).astype(int)
    cmp_top["new_rank"] = cmp_top["lift_new"].rank(ascending=False).astype(int)
    cmp_top["rank_change"] = cmp_top["orig_rank"] - cmp_top["new_rank"]
    cmp_top = cmp_top.sort_values("orig_rank")
    print(cmp_top.to_string(index=False))
    spearman_top = cmp_top["lift_orig"].corr(cmp_top["lift_new"], method="spearman")
    print(f"\nSpearman rank correlation (top, original vs new window): {spearman_top:.3f}")

    print("\n\n=== CONTINUATION: original-window vs new-window (1h) ===")
    prior_cont = load_prior_original_continuation()
    cmp_cont = prior_cont.merge(new_cont[["signal", "direction", "lift"]], on=["signal", "direction"], suffixes=("_orig", "_new"))
    print(cmp_cont.to_string(index=False))

    out_dir = ROOT / "tmp" / "eth_evidence_signal_ranking_stability_mar_jul_2026_20260814"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmp_bottom.to_csv(out_dir / "ranking_comparison_bottom.csv", index=False)
    cmp_top.to_csv(out_dir / "ranking_comparison_top.csv", index=False)
    cmp_cont.to_csv(out_dir / "ranking_comparison_continuation.csv", index=False)
    print(f"\nWrote comparison tables to {out_dir}")


if __name__ == "__main__":
    main()
