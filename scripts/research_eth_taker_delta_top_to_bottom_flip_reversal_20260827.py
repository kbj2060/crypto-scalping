#!/usr/bin/env python3
"""User's discretionary trading hypothesis (2026-08-27): taker_delta_z_climax's TOP leg firing
(delta_z>=2, aggressive-buying climax) followed in short succession by its own BOTTOM leg firing
(delta_z<=-2, aggressive-selling climax) marks a bottom that HOLDS -- price does not decline much
further from there, and if it does dip, it is brief before bouncing back -- unlike an ordinary/
isolated bottom-climax fire with no recent top climax before it.

Tests this as a conditional event study: bucket every historical bottom_taker_delta_z_climax fire
(scripts/live_evidence_signal_dashboard_20260823.py::compute_signals(), exact live formula) by how
many bars earlier the most recent top_taker_delta_z_climax fire was, then compare forward
price-path outcomes (downside MAE, net return, recovery-to-entry speed) across buckets. Primary
comparison: "flip" (top fired <=12 bars/1h earlier) vs "clean" baseline (no top fire in the
preceding 288 bars/1 day).

Data/formula reused verbatim (not re-derived): build_evidence_frame() (research_eth_evidence_
signal_regime_chop_conditional_20260827.py) -> data/eth_5m_1year.csv (2023-12-31~2026-02-17) run
through the exact live compute_signals(). Purely descriptive/diagnostic event study (no cost-gate,
no promotion claim) -- a discretionary-judgment sanity check, not an automation candidate eval.
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

from research_eth_evidence_signal_regime_chop_conditional_20260827 import build_evidence_frame  # noqa: E402

HORIZONS = [6, 12, 24, 48, 96]  # 30m/1h/2h/4h/8h bars (5m bars)
DIP_THRESH = -0.003  # -0.3%: threshold to count as "did dip further at all"
NO_RECENT_TOP_BARS = 96  # 8h: "clean" baseline requires no top fire in this lookback (see note
                         # below -- top fires so often (~every 44 bars on average) that a 1-day
                         # cutoff left only 81 "clean" events; 8h+ merges enough of the tail
                         # buckets for a stable baseline while still meaning "no top anywhere near")
FLIP_GAP_BARS = 6  # primary "flip" cutoff: top fired within 30min before the bottom fire (closest
                   # literal match to the user's "갑자기" (suddenly)); 12-bar (1h) also reported
RECLAIM_SEARCH_BARS = 96
RNG_SEED = 20260827

OUT_DIR = ROOT / "tmp" / "eth_taker_delta_top_to_bottom_flip_reversal_20260827"


def bars_since_last_top(top_arr: np.ndarray, positions: np.ndarray) -> np.ndarray:
    top_pos = np.flatnonzero(top_arr)
    if len(top_pos) == 0:
        return np.full(len(positions), -1)
    insert_idx = np.searchsorted(top_pos, positions, side="left")
    has_prior = insert_idx > 0
    prior_top = np.where(has_prior, top_pos[np.clip(insert_idx - 1, 0, len(top_pos) - 1)], -1)
    return np.where(has_prior, positions - prior_top, -1)


def bucket_label(gap: int) -> str:
    if gap < 0:
        return "no_prior_top_ever"
    if gap <= 6:
        return "1-6 (<=30m)"
    if gap <= 12:
        return "7-12 (<=1h)"
    if gap <= 24:
        return "13-24 (<=2h)"
    if gap <= 48:
        return "25-48 (<=4h)"
    if gap <= 96:
        return "49-96 (<=8h)"
    if gap <= 288:
        return "97-288 (<=1d)"
    return ">288 (>1d)"


def forward_outcomes(close: np.ndarray, high: np.ndarray, low: np.ndarray, positions: np.ndarray) -> pd.DataFrame:
    n = len(close)
    rows = []
    for i in positions:
        p0 = close[i]
        row = {"pos": i}
        for h in HORIZONS:
            j = i + h
            if j >= n:
                row[f"mae_{h}"] = np.nan
                row[f"mfe_{h}"] = np.nan
                row[f"ret_{h}"] = np.nan
                continue
            fwd_low = low[i + 1:j + 1].min()
            fwd_high = high[i + 1:j + 1].max()
            row[f"mae_{h}"] = (fwd_low - p0) / p0
            row[f"mfe_{h}"] = (fwd_high - p0) / p0
            row[f"ret_{h}"] = (close[j] - p0) / p0
        reclaim = np.nan
        limit = min(i + RECLAIM_SEARCH_BARS, n - 1)
        for k in range(i + 1, limit + 1):
            if close[k] >= p0:
                reclaim = k - i
                break
        row["bars_to_reclaim"] = reclaim
        rows.append(row)
    out = pd.DataFrame(rows)
    out["did_dip_48"] = np.where(out["mae_48"].isna(), np.nan, (out["mae_48"] <= DIP_THRESH).astype(float))
    return out


def permutation_test(a: np.ndarray, b: np.ndarray, n_perm: int = 5000, seed: int = RNG_SEED) -> float:
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    obs = a.mean() - b.mean()
    pooled = np.concatenate([a, b])
    na = len(a)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_perm)
    for t in range(n_perm):
        rng.shuffle(pooled)
        diffs[t] = pooled[:na].mean() - pooled[na:].mean()
    return float((np.abs(diffs) >= abs(obs)).mean())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = build_evidence_frame()
    close, high, low = frame["close"].to_numpy(), frame["high"].to_numpy(), frame["low"].to_numpy()
    top_arr = frame["top_taker_delta_z_climax"].fillna(False).to_numpy()
    bottom_arr = frame["bottom_taker_delta_z_climax"].fillna(False).to_numpy()
    ts = frame["timestamp"]

    bottom_pos = np.flatnonzero(bottom_arr)
    top_pos = np.flatnonzero(top_arr)
    print(f"Data: {ts.iloc[0]} ~ {ts.iloc[-1]}, {len(frame)} bars")
    print(f"top_taker_delta_z_climax fires: {len(top_pos)}, bottom_taker_delta_z_climax fires: {len(bottom_pos)}")

    gaps = bars_since_last_top(top_arr, bottom_pos)
    buckets = np.array([bucket_label(g) for g in gaps])

    outcomes = forward_outcomes(close, high, low, bottom_pos)
    outcomes["bucket"] = buckets
    outcomes["gap"] = gaps
    outcomes["timestamp"] = ts.iloc[bottom_pos].to_numpy()

    order = ["1-6 (<=30m)", "7-12 (<=1h)", "13-24 (<=2h)", "25-48 (<=4h)", "49-96 (<=8h)",
             "97-288 (<=1d)", ">288 (>1d)", "no_prior_top_ever"]
    print("\n=== bucketed by bars-since-last-top-climax-fire ===")
    summary_rows = []
    for b in order:
        sub = outcomes[outcomes["bucket"] == b]
        if len(sub) == 0:
            continue
        row = {"bucket": b, "n": len(sub)}
        for h in HORIZONS:
            row[f"mean_ret_{h}"] = sub[f"ret_{h}"].mean()
            row[f"winrate_{h}"] = (sub[f"ret_{h}"] > 0).mean()
        row["mean_mae_48"] = sub["mae_48"].mean()
        row["median_mae_48"] = sub["mae_48"].median()
        row["dip_rate_48(<=-0.3%)"] = sub["did_dip_48"].mean()
        row["mean_bars_to_reclaim"] = sub["bars_to_reclaim"].mean()
        row["reclaim_rate_within_96"] = sub["bars_to_reclaim"].notna().mean()
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(summary.round(4).to_string(index=False))
    print(f"\nALL bottom_taker_delta_z_climax fires (n={len(outcomes)}, any gap, for context):"
          f" mean_ret_48={outcomes['ret_48'].mean():+.4%}  winrate_48={(outcomes['ret_48'] > 0).mean():.1%}"
          f"  mean_mae_48={outcomes['mae_48'].mean():+.4%}  dip_rate_48={outcomes['did_dip_48'].mean():.1%}"
          f"  mean_bars_to_reclaim={outcomes['bars_to_reclaim'].mean():.1f}"
          f"  reclaim_within_{RECLAIM_SEARCH_BARS}_rate={outcomes['bars_to_reclaim'].notna().mean():.1%}")

    flip_mask = (outcomes["gap"] >= 0) & (outcomes["gap"] <= FLIP_GAP_BARS)
    clean_mask = (outcomes["gap"] < 0) | (outcomes["gap"] > NO_RECENT_TOP_BARS)
    flip = outcomes[flip_mask]
    clean = outcomes[clean_mask]
    print(f"\n=== primary comparison: flip (top fired <={FLIP_GAP_BARS} bars earlier), n={len(flip)}"
          f"  vs  clean baseline (no top in preceding {NO_RECENT_TOP_BARS} bars), n={len(clean)} ===")
    for h in HORIZONS:
        fm, cm = flip[f"ret_{h}"].mean(), clean[f"ret_{h}"].mean()
        fw, cw = (flip[f"ret_{h}"] > 0).mean(), (clean[f"ret_{h}"] > 0).mean()
        p = permutation_test(flip[f"ret_{h}"].to_numpy(), clean[f"ret_{h}"].to_numpy())
        print(f"  ret@{h:>3}bar: flip mean={fm:+.4%} win={fw:.1%}   |  clean mean={cm:+.4%} win={cw:.1%}   |  perm p={p:.4f}")
    fmae, cmae = flip["mae_48"].mean(), clean["mae_48"].mean()
    pmae = permutation_test(flip["mae_48"].to_numpy(), clean["mae_48"].to_numpy())
    print(f"  MAE@48bar(downside): flip mean={fmae:+.4%}   |  clean mean={cmae:+.4%}   |  perm p={pmae:.4f}")
    print(f"  dip-rate(<=-0.3% within 4h): flip={flip['did_dip_48'].mean():.1%}   |  clean={clean['did_dip_48'].mean():.1%}")
    print(f"  mean bars-to-reclaim-entry: flip={flip['bars_to_reclaim'].mean():.1f}   |  clean={clean['bars_to_reclaim'].mean():.1f}"
          f"   (reclaim-within-{RECLAIM_SEARCH_BARS} rate: flip={flip['bars_to_reclaim'].notna().mean():.1%},"
          f" clean={clean['bars_to_reclaim'].notna().mean():.1%})")
    print("  -- tail-risk check: is the mean-return gap driven by fewer/smaller BAD outcomes, not a"
          " uniformly better outcome? --")
    for thresh in (-0.02, -0.03, -0.05):
        ft = (flip["mae_48"] <= thresh).mean()
        ct = (clean["mae_48"] <= thresh).mean()
        print(f"  P(mae_48 <= {thresh:+.0%}): flip={ft:.1%}   |  clean={ct:.1%}")
    for thresh in (-0.01, -0.02):
        ft = (flip["ret_48"] <= thresh).mean()
        ct = (clean["ret_48"] <= thresh).mean()
        print(f"  P(ret_48  <= {thresh:+.0%}): flip={ft:.1%}   |  clean={ct:.1%}")

    flip12 = outcomes[(outcomes["gap"] >= 0) & (outcomes["gap"] <= 12)]
    print(f"\n=== robustness: same comparison with a looser 1h (12-bar) flip cutoff, n={len(flip12)} ===")
    for h in HORIZONS:
        fm, cm = flip12[f"ret_{h}"].mean(), clean[f"ret_{h}"].mean()
        p = permutation_test(flip12[f"ret_{h}"].to_numpy(), clean[f"ret_{h}"].to_numpy())
        print(f"  ret@{h:>3}bar: flip(1h) mean={fm:+.4%}   |  clean mean={cm:+.4%}   |  perm p={p:.4f}")

    outcomes.to_csv(OUT_DIR / "events.csv", index=False)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    print(f"\nsaved: {OUT_DIR}/events.csv, {OUT_DIR}/summary.csv")


if __name__ == "__main__":
    main()
