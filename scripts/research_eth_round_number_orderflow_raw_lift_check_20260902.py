#!/usr/bin/env python3
"""Raw-rule lift pre-check for the ROUND-NUMBER ORDER-FLOW asymmetry as a Homer evidence-signal
candidate -- docs/homer/external_literature_signal_candidates_20260902.md's C-1. Last of the four
candidates from the 2026-09-02 external-literature survey; run to close out the klines-derived
axis (README ss5.12). Retrospective diagnostic, not Fresh-Forward gated.

WHAT IS NEW VS THE 2026-08-14 MEASUREMENT

  analyze_eth_deep_evidence_signal_sweep_round2_20260814.py already tested "A10 round-number
  approach" -- price within 15bp of a $50 level AND trending into it -- and got bottom 1.79x /
  top 1.44x. Not rejected, just not selected into the top-8. Its `near_round`/step=50/0.0015
  definition is reused VERBATIM here as the price-only control arm.

  The FRL (2026) paper "Buy-sell imbalances on and around round numbers in cryptocurrencies"
  (18 coins, high-frequency) does not claim a price-proximity effect. Its finding is about ORDER
  FLOW: abnormally high BUY pressure just BELOW a round number and abnormally high SELL pressure
  just ABOVE it (left-digit / threshold-trigger / cluster-undercutting effects). This repo has
  taker_buy_base, so that is directly measurable and has never been tested here.

TWO STAGES

  STAGE 1 (mechanism replication) -- does the asymmetry even exist in ETH 5m?
    Mean signed taker imbalance (2*taker_buy/volume - 1) bucketed by SIGNED distance to the nearest
    round level. The paper predicts imbalance > 0 just below (negative distance) and < 0 just above.
    Run on the true $50/$100 grids AND on OFFSET PLACEBO grids (same step, shifted half a step, so
    the "levels" are $x25/$x75 -- exactly as arbitrary geometrically, but not psychologically
    salient). If the true grid does not separate from its placebo, there is no round-number effect
    to trade and stage 2 is moot.

  STAGE 2 (trigger lift) -- the 2x2 grid, both pivot sides, standard event_study harness:
    below_buy   : just below a level, taker imbalance in the top quartile   (predicted pressure)
    below_sell  : just below a level, imbalance in the bottom quartile      (pressure fails)
    above_sell  : just above a level, imbalance in the bottom quartile      (predicted pressure)
    above_buy   : just above a level, imbalance in the top quartile         (pressure fails)
  plus controls: near_round_2026_08_14 (price-only, verbatim), and every arm re-run on the OFFSET
  PLACEBO grid. A real round-number signal must beat its own placebo -- that is the decisive test,
  not the raw lift.

Window: VAL 2025-09-01..2025-12-31 + OOS 2026-01-01..2026-02-17, identical to the sibling scripts.
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

from analyze_eth_confluence_oscillator_bottom_top_evidence_20260814 import (  # noqa: E402
    K_HORIZONS,
    OOS_END,
    event_study,
    load_zigzag_pivots,
)
from backtest_eth_slowk_williamsr_persistence_confluence_20260814 import (  # noqa: E402
    OOS_START,
    VAL_END,
    VAL_START,
)
from research_eth_lee_mykland_jump_raw_lift_check_20260902 import (  # noqa: E402
    OVERLAP_TOL_BARS,
    ZSCORE_WINDOW,
    overlap_stats,
    wilson_ci,
)
from research_eth_vpin_toxicity_jump_magnitude_raw_lift_check_20260902 import (  # noqa: E402
    load_frame_with_taker,
)

STEPS = (50.0, 100.0)
NEAR_PCT = 0.0015          # 15bp -- verbatim from the 2026-08-14 near_round definition
ROC_WINDOW = 48            # price_roc_48, verbatim from the same script
ROC_THRESHOLD = 0.01       # +-1%, verbatim
IMB_WINDOW = 288           # quartile reference window for the taker imbalance (1 day of 5m bars)
STRZ_THRESHOLD = 2.5
SWEEP_LOOKBACK = 48
DIST_BUCKETS = 10          # signed-distance buckets for stage 1


def signed_distance(close: pd.Series, step: float, offset: float) -> pd.Series:
    """Signed distance to the nearest grid level, in units of half-step (-1..+1).

    offset=0 -> the true round grid ($3200, $3250, ...). offset=0.5 -> the placebo grid shifted
    half a step ($3225, $3275, ...), geometrically identical but not psychologically salient.
    Negative = price sits BELOW the nearest level, positive = above."""
    shifted = close / step - offset
    nearest = shifted.round()
    return (shifted - nearest) * 2.0


def main() -> None:
    raw = load_frame_with_taker()
    pivots = load_zigzag_pivots()
    high, low, close = raw["high"], raw["low"], raw["close"]
    volume, taker_buy, ts = raw["volume"], raw["taker_buy_base"], raw["timestamp"]

    window_mask = (((ts >= VAL_START) & (ts <= VAL_END)) | ((ts >= OOS_START) & (ts <= OOS_END))).to_numpy()
    all_pos = np.flatnonzero(window_mask)
    print(f"Window: VAL {VAL_START.date()}..{VAL_END.date()} + OOS {OOS_START.date()}..{OOS_END.date()}, "
          f"{int(window_mask.sum())} bars, {len(pivots)} zigzag pivots")
    print(f"ETH close range in-window: {close[window_mask].min():.0f}..{close[window_mask].max():.0f} "
          f"(median {close[window_mask].median():.0f})")

    imb = (2.0 * taker_buy / volume.replace(0.0, np.nan) - 1.0)
    imb_hi = imb.rolling(IMB_WINDOW, min_periods=IMB_WINDOW).quantile(0.75)
    imb_lo = imb.rolling(IMB_WINDOW, min_periods=IMB_WINDOW).quantile(0.25)
    buy_pressure, sell_pressure = (imb >= imb_hi).fillna(False), (imb <= imb_lo).fillna(False)

    # ---------------- STAGE 1: mechanism replication ----------------
    print("\n=== STAGE 1: mean signed taker imbalance by signed distance to level ===")
    print("(paper predicts POSITIVE just below (dist<0) and NEGATIVE just above (dist>0))")
    stage1 = []
    for step in STEPS:
        for offset, gname in ((0.0, "TRUE"), (0.5, "PLACEBO")):
            d = signed_distance(close, step, offset)
            bucket = pd.cut(d[window_mask], bins=np.linspace(-1, 1, DIST_BUCKETS + 1))
            grp = imb[window_mask].groupby(bucket, observed=False).mean()
            below = float(imb[window_mask][(d[window_mask] < 0) & (d[window_mask] >= -0.3)].mean())
            above = float(imb[window_mask][(d[window_mask] > 0) & (d[window_mask] <= 0.3)].mean())
            stage1.append({"step": step, "grid": gname, "imb_just_below": round(below, 5),
                           "imb_just_above": round(above, 5), "asymmetry": round(below - above, 5)})
            print(f"  step=${step:.0f} {gname:8s}  by-bucket mean imbalance: "
                  + " ".join(f"{v:+.4f}" for v in grp.to_numpy()))
    s1 = pd.DataFrame(stage1)
    print("\n  just-below vs just-above (|dist| <= 0.3 half-steps):")
    print(s1.to_string(index=False))

    # ---------------- STAGE 2: trigger lift ----------------
    price_roc = close / close.shift(ROC_WINDOW) - 1.0
    arms: list[tuple[str, str, pd.Series]] = []
    for step in STEPS:
        for offset, gname in ((0.0, "TRUE"), (0.5, "PLACEBO")):
            d = signed_distance(close, step, offset)
            near = (d.abs() * step / 2.0 / close <= NEAR_PCT)
            below, above = near & (d < 0), near & (d > 0)
            tag = f"s{int(step)}_{gname}"
            arms += [
                (f"{tag}_below_buy", "top", below & buy_pressure),
                (f"{tag}_below_buy", "bottom", below & buy_pressure),
                (f"{tag}_below_sell", "bottom", below & sell_pressure),
                (f"{tag}_below_sell", "top", below & sell_pressure),
                (f"{tag}_above_sell", "bottom", above & sell_pressure),
                (f"{tag}_above_sell", "top", above & sell_pressure),
                (f"{tag}_above_buy", "top", above & buy_pressure),
                (f"{tag}_above_buy", "bottom", above & buy_pressure),
                # price-only control: the 2026-08-14 A10 rule, verbatim
                (f"{tag}_A10_priceonly", "bottom", near & (price_roc <= -ROC_THRESHOLD)),
                (f"{tag}_A10_priceonly", "top", near & (price_roc >= ROC_THRESHOLD)),
            ]

    ret3 = close / close.shift(3) - 1.0
    ret3_z = (ret3 - ret3.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_WINDOW).mean()) / \
        ret3.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_WINDOW).std().replace(0.0, np.nan)
    swing_low_prior = low.rolling(SWEEP_LOOKBACK, min_periods=SWEEP_LOOKBACK).min().shift(1)
    swing_high_prior = high.rolling(SWEEP_LOOKBACK, min_periods=SWEEP_LOOKBACK).max().shift(1)
    arms += [("short_term_return_z_ref", "bottom", ret3_z <= -STRZ_THRESHOLD),
             ("short_term_return_z_ref", "top", ret3_z >= STRZ_THRESHOLD),
             ("liquidity_sweep_ref", "bottom", (low < swing_low_prior) & (close > swing_low_prior)),
             ("liquidity_sweep_ref", "top", (high > swing_high_prior) & (close < swing_high_prior))]

    rows, fires = [], {}
    for name, side, trig in arms:
        side_pivots = pivots.loc[pivots["pivot_type"] == side]
        pivot_pos = raw.index[raw["timestamp"].isin(side_pivots["timestamp"])].to_numpy()
        trigger_pos = np.flatnonzero(trig.fillna(False).to_numpy() & window_mask)
        fires[(name, side)] = trigger_pos
        for k_name, K in K_HORIZONS.items():
            st = event_study(trigger_pos, pivot_pos, all_pos, K)
            n, prec = st["n_triggers"], st["precision"]
            hits = round(prec * n) if n and np.isfinite(prec) else 0
            lo, hi = wilson_ci(hits, n) if n else (float("nan"), float("nan"))
            rows.append({"signal": name, "side": side, "horizon": k_name, "n_triggers": n,
                         "precision": prec, "ci_lo": lo, "ci_hi": hi,
                         "baseline_rate": st["baseline_rate"], "lift": st["lift"]})

    df = pd.DataFrame(rows)
    out_dir = ROOT / "tmp" / "eth_round_number_orderflow_raw_lift_check_20260902"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "scorecard.csv", index=False)
    s1.to_csv(out_dir / "stage1_mechanism.csv", index=False)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_rows", 400)
    sub = df[df["horizon"] == "K12_1h"].copy()
    sub["lift_x"] = sub["lift"].round(2)
    print("\n=== STAGE 2: 1h lift, TRUE vs PLACEBO grid (the decisive comparison) ===")
    piv = sub.pivot_table(index=["signal"], columns="side", values=["n_triggers", "lift_x"])
    print(piv.to_string())

    print(f"\n=== overlap vs deployed signals (+-{OVERLAP_TOL_BARS} bars), best TRUE-grid arms ===")
    ov_rows = []
    for name in sorted({n for n, _ in fires} - {"short_term_return_z_ref", "liquidity_sweep_ref"}):
        if "TRUE" not in name:
            continue
        for ref in ("short_term_return_z_ref", "liquidity_sweep_ref"):
            for side in ("bottom", "top"):
                st = overlap_stats(fires[(name, side)], fires[(ref, side)], OVERLAP_TOL_BARS)
                ov_rows.append({"signal": name, "vs": ref, "side": side,
                                "n_sig": len(fires[(name, side)]),
                                "sig_near_ref_pct": round(st["frac_a_near_b"] * 100, 1)})
    ov = pd.DataFrame(ov_rows)
    ov.to_csv(out_dir / "overlap.csv", index=False)
    print(ov.pivot_table(index="signal", columns=["vs", "side"], values="sig_near_ref_pct").to_string())
    print(f"\nWrote {out_dir}/{{scorecard,stage1_mechanism,overlap}}.csv")


if __name__ == "__main__":
    main()
