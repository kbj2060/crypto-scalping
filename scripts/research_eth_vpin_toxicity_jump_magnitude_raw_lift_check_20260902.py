#!/usr/bin/env python3
"""Raw-rule lift pre-check for VPIN (order-flow toxicity) as a Homer evidence-signal candidate,
against an UNSIGNED JUMP-MAGNITUDE target -- docs/homer/external_literature_signal_candidates_
20260902.md's A-2, with the label framing the user chose on 2026-09-02 ("무방향 점프 크기 라벨").
Retrospective evidence-gathering diagnostic, not a live-tradeable signal claim, not Fresh-Forward
gated (same standing as the sibling pre-checks).

WHY A DIFFERENT TARGET FROM A-1/A-3

  The zigzag-pivot event_study harness used by every prior pre-check asks "does this trigger
  precede a top/bottom pivot" -- inherently DIRECTIONAL. The literature claim for VPIN is not
  directional: "Bitcoin wild moves: Evidence from order flow toxicity and price jumps" (Research in
  International Business and Finance, 2026) finds VPIN predicts future price JUMPS (magnitude),
  with positive serial correlation in both VPIN and jump size. Testing it against a pivot target
  would test something the paper does not claim. So the target here is the unsigned analogue,
  built to the repo's own intrabar barrier convention:

      event(i) = max( max_j high_j / close_i - 1 , 1 - min_j low_j / close_i ) >= K * atr_pct_i
                 over j in (i, i+H]

  lift = P(event | trigger) / P(event | all in-window bars), exactly parallel to event_study's
  precision/baseline/lift, with the pivot set replaced by the magnitude set.

  ATR normalisation (K * atr_pct_i, ATR measured causally through bar i) is deliberate: it makes
  the target volatility-neutral BY CONSTRUCTION, which is the single most important control here
  (volatility clustering would otherwise hand any high-vol trigger a free lift). README ss5.9's
  low-volatility caveat is reported as the median bp of the K*ATR threshold.

ARMS

  1. vpin_volclock -- PROPER VPIN (Easley/Lopez de Prado/O'Hara): equal-VOLUME buckets, not a time
     clock. Bucket size V = (pre-VAL mean daily volume)/50, VPIN = mean over the last n=50 buckets
     of |2*Vbuy_k - V|/V. Buy volume at a bucket edge is allocated proportionally by interpolating
     cumulative taker_buy_base against cumulative volume. Each bucket's value is stamped on the bar
     at which the bucket COMPLETES and then forward-filled -- causal by construction. V is estimated
     on pre-VAL_START bars only (same discipline as the Lee-Mykland periodicity fit).
  2. vpin_approx_48 -- the repo's existing time-clock approximation, _vpin_approx from
     scripts/eth_dc_financial_ml_feature_construction_20260820.py, reused verbatim. Included because
     that is the version already sitting in the 154-feature set (which tested at chance as a
     FEATURE, p=0.460) -- the whole premise of this pre-check is that trigger != feature (the
     DeMarker precedent: dem's standalone AUC 0.51 vs its trigger HOLDOUT 0.7464).

  CONTROLS, all pre-registered before looking at results (the 2026-08-25 taker-variance-compression
  rejection died on an unchecked collinearity; A-3's control inverted the expected conclusion):
  3. atr_pct_z      -- volatility-clustering null. THE null to beat.
  4. hl_range       -- the range null that beat Corwin-Schultz outright on 2026-09-02.
  5. abs_taker_dz   -- |signed taker delta z|, i.e. the deployed taker_delta_z_climax with its sign
                       thrown away. If VPIN is only this, it is not a new signal.
  6. volume         -- VPIN's own denominator.

  Trigger rule for all six: rolling-864 percentile >= 0.99 (this repo's percentile_window=864
  convention), with 0.95 reported as sensitivity. Unsigned target -> unsigned trigger, no side split.

Window: VAL 2025-09-01..2025-12-31 + OOS 2026-01-01..2026-02-17, identical to the sibling scripts.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backtest_eth_slowk_williamsr_persistence_confluence_20260814 import (  # noqa: E402
    OOS_START,
    VAL_END,
    VAL_START,
)
from analyze_eth_confluence_oscillator_bottom_top_evidence_20260814 import OOS_END  # noqa: E402
from research_eth_lee_mykland_jump_raw_lift_check_20260902 import (  # noqa: E402
    OVERLAP_TOL_BARS,
    ZSCORE_WINDOW,
    overlap_stats,
    wilson_ci,
)

DATA_PATH = ROOT / "data" / "eth_5m_1year.csv"
PCT_WINDOW = 864
PCT_PRIMARY, PCT_SECONDARY = 0.99, 0.95
ATR_WINDOW = 288
VPIN_BUCKETS_PER_DAY = 50    # Easley/LdP/O'Hara: bucket volume = daily volume / 50
VPIN_N_BUCKETS = 50          # ... and VPIN averaged over the last 50 buckets (~1 day)
VPIN_APPROX_WINDOW = 48      # repo's VPIN_WINDOW, verbatim
HORIZONS = {"H12_1h": 12, "H48_4h": 48, "H96_8h": 96}
K_GRID = (1.0, 1.5, 2.0, 2.5, 3.0)
K_PRIMARY = 2.0
TAKER_Z_THRESHOLD = 2.0      # deployed taker_delta_z_climax cutoff, verbatim


def load_frame_with_taker() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, usecols=["timestamp", "open", "high", "low", "close", "volume",
                                         "taker_buy_base"], parse_dates=["timestamp"])
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def load_vpin_approx_fn():
    path = ROOT / "scripts" / "eth_dc_financial_ml_feature_construction_20260820.py"
    spec = importlib.util.spec_from_file_location("finml_features_vpin_20260902", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._vpin_approx


def vpin_volume_clock(volume: np.ndarray, taker_buy: np.ndarray, bucket_volume: float,
                      n_buckets: int = VPIN_N_BUCKETS) -> np.ndarray:
    """Equal-volume-bucket VPIN, stamped on the bar where each bucket completes then ffilled.

    Buy volume at a bucket edge is obtained by interpolating cumulative taker_buy against cumulative
    volume, i.e. a bar straddling an edge has its buy volume split in proportion -- the standard
    bulk-volume allocation. No bar after the completing bar contributes, so the series is causal."""
    cum_vol = np.cumsum(volume)
    cum_buy = np.cumsum(taker_buy)
    n_edges = int(cum_vol[-1] // bucket_volume)
    if n_edges < n_buckets + 1:
        raise RuntimeError("not enough volume for the requested bucket count")
    edges = np.arange(1, n_edges + 1) * bucket_volume
    buy_at_edge = np.interp(edges, cum_vol, cum_buy)
    v_buy = np.diff(np.concatenate([[0.0], buy_at_edge]))
    imbalance = np.abs(2.0 * v_buy - bucket_volume) / bucket_volume
    vpin_bucket = pd.Series(imbalance).rolling(n_buckets, min_periods=n_buckets).mean().to_numpy()
    complete_bar = np.searchsorted(cum_vol, edges, side="left")   # bar at which each bucket closes
    out = np.full(len(volume), np.nan)
    valid = np.isfinite(vpin_bucket) & (complete_bar < len(volume))
    # later buckets overwrite earlier ones landing on the same bar -- keep the most recent
    out[complete_bar[valid]] = vpin_bucket[valid]
    return pd.Series(out).ffill().to_numpy()


def magnitude_event(high: np.ndarray, low: np.ndarray, close: np.ndarray, atr_pct: np.ndarray,
                    horizon: int, k: float) -> np.ndarray:
    """max(|forward excursion|)/close_i >= k*atr_pct_i over the next `horizon` bars, intrabar."""
    n = len(close)
    fwd_max = pd.Series(high).shift(-1).rolling(horizon, min_periods=1).max().shift(-(horizon - 1)).to_numpy()
    fwd_min = pd.Series(low).shift(-1).rolling(horizon, min_periods=1).min().shift(-(horizon - 1)).to_numpy()
    up = fwd_max / close - 1.0
    dn = 1.0 - fwd_min / close
    excursion = np.maximum(up, dn)
    ok = np.isfinite(excursion) & np.isfinite(atr_pct)
    ok[n - horizon:] = False                      # incomplete forward window
    return np.where(ok, excursion >= k * atr_pct, False)


def main() -> None:
    vpin_approx_fn = load_vpin_approx_fn()
    raw = load_frame_with_taker()
    high, low, close = raw["high"], raw["low"], raw["close"]
    volume, taker_buy, ts = raw["volume"], raw["taker_buy_base"], raw["timestamp"]

    window_mask = (((ts >= VAL_START) & (ts <= VAL_END)) | ((ts >= OOS_START) & (ts <= OOS_END))).to_numpy()
    fit_mask = (ts < VAL_START).to_numpy()
    all_pos = np.flatnonzero(window_mask)
    print(f"Window: VAL {VAL_START.date()}..{VAL_END.date()} + OOS {OOS_START.date()}..{OOS_END.date()}, "
          f"{int(window_mask.sum())} bars")

    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr_pct = (tr.rolling(ATR_WINDOW, min_periods=ATR_WINDOW).mean() / close)

    bucket_volume = float(volume[fit_mask].mean()) * 288.0 / VPIN_BUCKETS_PER_DAY
    print(f"VPIN volume clock: bucket = {bucket_volume:,.0f} base units "
          f"(pre-VAL mean daily volume / {VPIN_BUCKETS_PER_DAY}), n_buckets = {VPIN_N_BUCKETS}")
    vpin_vc = pd.Series(vpin_volume_clock(volume.to_numpy(), taker_buy.to_numpy(), bucket_volume),
                        index=close.index)
    vpin_ap = pd.Series(vpin_approx_fn(taker_buy.to_numpy(), volume.to_numpy(), VPIN_APPROX_WINDOW),
                        index=close.index)

    delta = 2.0 * taker_buy - volume
    delta_z = (delta - delta.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_WINDOW).mean()) / \
        delta.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_WINDOW).std().replace(0.0, np.nan)
    hl = np.log(high / low)

    estimators = {
        "vpin_volclock": vpin_vc,
        "vpin_approx48": vpin_ap,
        "atr_pct_CONTROL": atr_pct,
        "hl_range_CONTROL": hl,
        "abs_taker_dz_CONTROL": delta_z.abs(),
        "volume_CONTROL": volume,
    }

    diag = pd.DataFrame({k: v for k, v in estimators.items()})[window_mask].dropna()
    print("\n=== confound diagnostic: Spearman on the eval window ===")
    print(diag.corr(method="spearman").round(3).to_string())

    print("\n=== target calibration: P(|excursion| >= K*ATR) on all in-window bars ===")
    cal = []
    for hn, H in HORIZONS.items():
        for k in K_GRID:
            ev = magnitude_event(high.to_numpy(), low.to_numpy(), close.to_numpy(),
                                 atr_pct.to_numpy(), H, k)
            cal.append({"horizon": hn, "K": k, "baseline_pct": round(float(ev[window_mask].mean()) * 100, 1),
                        "thr_median_bp": round(float((k * atr_pct[window_mask]).median()) * 1e4, 1)})
    cal_df = pd.DataFrame(cal)
    print(cal_df.pivot(index="K", columns="horizon", values="baseline_pct").to_string())
    print("K*ATR threshold, median bp (README ss5.9 low-volatility check):")
    print(cal_df.pivot(index="K", columns="horizon", values="thr_median_bp").to_string())

    rows, fires = [], {}
    for nm, series in estimators.items():
        pct = series.rolling(PCT_WINDOW, min_periods=PCT_WINDOW).rank(pct=True)
        for cut, tag in ((PCT_PRIMARY, "p99"), (PCT_SECONDARY, "p95")):
            trig = (pct >= cut).fillna(False).to_numpy() & window_mask
            trigger_pos = np.flatnonzero(trig)
            fires[(nm, tag)] = trigger_pos
            for hn, H in HORIZONS.items():
                for k in K_GRID:
                    ev = magnitude_event(high.to_numpy(), low.to_numpy(), close.to_numpy(),
                                         atr_pct.to_numpy(), H, k)
                    base = float(ev[window_mask].mean())
                    sel = ev[trigger_pos]
                    n, prec = len(sel), float(sel.mean()) if len(sel) else float("nan")
                    lo, hi = wilson_ci(int(sel.sum()), n) if n else (float("nan"), float("nan"))
                    rows.append({"signal": f"{nm}_{tag}", "horizon": hn, "K": k, "n_triggers": n,
                                 "precision": prec, "ci_lo": lo, "ci_hi": hi,
                                 "baseline_rate": base,
                                 "lift": prec / base if base > 0 else float("nan")})

    df = pd.DataFrame(rows)
    out_dir = ROOT / "tmp" / "eth_vpin_toxicity_jump_magnitude_raw_lift_check_20260902"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "scorecard.csv", index=False)
    diag.corr(method="spearman").to_csv(out_dir / "confound_spearman.csv")
    cal_df.to_csv(out_dir / "target_calibration.csv", index=False)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_rows", 400)
    for hn in HORIZONS:
        sub = df[(df["horizon"] == hn) & (df["signal"].str.endswith("_p99"))].copy()
        print(f"\n=== horizon {hn}, p99 trigger -- lift by K ===")
        print(sub.pivot(index="signal", columns="K", values="lift").round(2).to_string())
    print(f"\n=== detail at K={K_PRIMARY} (primary), p99 trigger ===")
    d = df[(df["K"] == K_PRIMARY) & (df["signal"].str.endswith("_p99"))].copy()
    d["prec_pct"] = (d["precision"] * 100).round(1)
    d["ci_lo_pct"] = (d["ci_lo"] * 100).round(1)
    d["ci_hi_pct"] = (d["ci_hi"] * 100).round(1)
    d["base_pct"] = (d["baseline_rate"] * 100).round(1)
    d["lift_x"] = d["lift"].round(2)
    print(d[["signal", "horizon", "n_triggers", "prec_pct", "ci_lo_pct", "ci_hi_pct",
             "base_pct", "lift_x"]].to_string(index=False))

    print(f"\n=== overlap vs deployed taker_delta_z_climax (+-{OVERLAP_TOL_BARS} bars) ===")
    taker_fire = np.flatnonzero((delta_z.abs() >= TAKER_Z_THRESHOLD).fillna(False).to_numpy() & window_mask)
    ov_rows = []
    for nm in estimators:
        st = overlap_stats(fires[(nm, "p99")], taker_fire, OVERLAP_TOL_BARS)
        ov_rows.append({"signal": f"{nm}_p99", "n_sig": len(fires[(nm, "p99")]),
                        "n_taker_climax": len(taker_fire),
                        "jaccard_exact_pct": round(st["jaccard_exact_bar"] * 100, 1),
                        "sig_near_taker_pct": round(st["frac_a_near_b"] * 100, 1),
                        "taker_near_sig_pct": round(st["frac_b_near_a"] * 100, 1)})
    ov = pd.DataFrame(ov_rows)
    ov.to_csv(out_dir / "overlap.csv", index=False)
    print(ov.to_string(index=False))
    print(f"\nWrote {out_dir}/{{scorecard,overlap,confound_spearman,target_calibration}}.csv")


if __name__ == "__main__":
    main()
