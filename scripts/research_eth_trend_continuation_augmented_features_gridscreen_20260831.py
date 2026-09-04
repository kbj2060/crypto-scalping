#!/usr/bin/env python3
"""Re-run the trend-continuation EXTEND/REVERT H x GAP grid screen
(research_eth_trend_continuation_horizon_gap_gridscreen_20260831.py, which used Tier0's plain
23 features and came back flat 0.4872-0.5327 across all 24 cells) with 7 additional features
the user asked for, to check whether Tier0's "no directional information" wall is a property of
those specific 23 features or a more fundamental property of this event population:

4 engineered (klines-only, reusing existing repo implementations verbatim, not reinvented):
  - hawkes_mag / hawkes_signed: self-exciting jump intensity, reusing hawkes_intensity() and the
    exact jump-detection construction from research_hawkes_jump_clustering_skip_filter_eth_20260809.py
    (JUMP_QUANTILE=0.90, beta=0.10 -- a middle value from that script's own BETAS grid).
  - cvd_accel_48: reuses analyze_eth_creative_reversal_evidence_signals_20260814.py's exact
    cvd_roll_roc_48 construction (delta=2*taker_buy-volume, 288-bar rolling sum, 48-bar diff),
    then takes ONE more .diff() -- the discrete 2nd derivative, mirroring this repo's only other
    tried acceleration feature (test_creative_5m_features_20260705.py's accel_vwap =
    vwap_dist.diff().diff()).
  - hurst_48: features/engineering.py's HurstExponentFeatures class, used verbatim.
  - poc_migration_12: core/cvp.py's add_cvp_features(cvp_poc_dist), a 200-bar rolling K-means
    volume-profile POC distance (normalized by price range) -- poc_migration_12 is its 12-bar
    (1h) diff, the direct operationalization of "is the POC actively migrating".

3 from newly-joined DuckDB-sourced OI/positioning data (data/TOTAL_ETHUSDT_metrics_2024_2026.csv,
5m-aligned, verified 2024-01..present coverage per reference_clean_data_locations memory --
microstructure_1m/whale/L2 duckdb data was explicitly NOT used: it only exists from 2026-05-03,
zero overlap with this axis's TRAIN/VAL/OOS, and was already exhausted 4x as entry alpha):
  - oi_roc_48: 48-bar rate of change of sum_open_interest (stationarized, not raw level).
  - toptrader_ls_z / taker_ls_z: rolling z-scores of sum_toptrader_long_short_ratio and
    sum_taker_long_short_vol_ratio.
  Join: metrics.create_time is bucket-END labeled (per reference_clean_data_locations memory:
  "+5min end-label corrected"), klines.timestamp is bucket-START (Binance kline convention) --
  so metrics row at T+5min is joined to the klines bar at T (both become known at the same
  real-world instant, T+5min = that bar's own close). Getting this backwards would either leak
  the NEXT bar's OI reading into this bar (lookahead) or use a stale reading one full bar late.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from live_evidence_signal_dashboard_20260823 import SIGNAL_ORDER, compute_signals  # noqa: E402
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import (  # noqa: E402
    FEATURE_COLUMNS,
    build_indicator_frame,
)
from core.cvp import add_cvp_features  # noqa: E402
from features.engineering import HurstExponentFeatures  # noqa: E402

ETH_PATH = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
BTC_PATH = ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv"
METRICS_PATH = ROOT / "data/TOTAL_ETHUSDT_metrics_2024_2026.csv"
START = pd.Timestamp("2024-01-01")
VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")

HORIZON_GRID = [8, 12, 16, 20, 24, 30, 36, 48]
GAP_GRID = [3, 6, 12]
GBM_SEED = 20260831
JUMP_QUANTILE = 0.90
HAWKES_BETA = 0.10

NEW_COLS = [
    "hawkes_mag", "hawkes_signed", "cvd_accel_48", "hurst_48", "poc_migration_12",
    "oi_roc_48", "toptrader_ls_z", "taker_ls_z",
]


def log(msg: str) -> None:
    print(f"[trend_continuation_augmented] {msg}", flush=True)


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df[df["timestamp"] >= START - pd.Timedelta(days=10)].reset_index(drop=True)


def forward_extremes(close, high, low, h):
    fh = pd.Series(high).shift(-1).rolling(h, min_periods=h).max().shift(-(h - 1)).to_numpy()
    fl = pd.Series(low).shift(-1).rolling(h, min_periods=h).min().shift(-(h - 1)).to_numpy()
    return (fh - close) / close, (close - fl) / close


def hawkes_intensity(jump_indicator: np.ndarray, beta: float) -> np.ndarray:
    decay = np.exp(-beta)
    lam = np.empty_like(jump_indicator, dtype=np.float64)
    running = 0.0
    for i in range(len(jump_indicator)):
        running = running * decay + jump_indicator[i]
        lam[i] = running
    return lam


def add_new_features(eth: pd.DataFrame) -> pd.DataFrame:
    close = eth["close"].to_numpy()
    volume = eth["volume"].to_numpy()
    taker_buy = eth["taker_buy_base"].to_numpy()
    out = pd.DataFrame(index=eth.index)

    # --- Hawkes (reused verbatim from research_hawkes_jump_clustering_skip_filter_eth_20260809) ---
    t0 = time.time()
    log_close = np.log(close)
    bar_ret = log_close - np.roll(log_close, 1)
    bar_ret[0] = 0.0
    threshold = np.quantile(np.abs(bar_ret), JUMP_QUANTILE)
    is_jump = (np.abs(bar_ret) > threshold).astype(np.float64)
    jump_sign = np.sign(bar_ret) * is_jump
    out["hawkes_mag"] = hawkes_intensity(is_jump, HAWKES_BETA)
    out["hawkes_signed"] = hawkes_intensity(jump_sign, HAWKES_BETA)
    log(f"hawkes done in {time.time()-t0:.1f}s (jump_rate={is_jump.mean():.3f})")

    # --- CVD acceleration (reused verbatim from analyze_eth_creative_reversal_evidence_signals_20260814) ---
    t0 = time.time()
    delta = 2.0 * taker_buy - volume
    cvd_roll = pd.Series(delta).rolling(288, min_periods=288).sum()
    cvd_roll_roc_48 = cvd_roll - cvd_roll.shift(48)
    out["cvd_accel_48"] = cvd_roll_roc_48.diff()
    log(f"cvd_accel done in {time.time()-t0:.1f}s")

    # --- Hurst (reused verbatim from features/engineering.py::HurstExponentFeatures) ---
    t0 = time.time()
    hf = HurstExponentFeatures(eth[["open", "high", "low", "close", "volume"]].copy())
    hf_df = hf.add_all_features()
    out["hurst_48"] = hf_df["hurst_48"].to_numpy()
    log(f"hurst done in {time.time()-t0:.1f}s")

    # --- POC migration (reused verbatim from core/cvp.py::add_cvp_features) ---
    t0 = time.time()
    cvp_in = eth[["open", "high", "low", "close", "volume"]].copy()
    cvp_out = add_cvp_features(cvp_in, output_cols=["cvp_poc_dist"])
    out["poc_migration_12"] = cvp_out["cvp_poc_dist"].diff(12)
    log(f"poc_migration done in {time.time()-t0:.1f}s")

    return out


def add_metrics_features(ts: pd.Series) -> pd.DataFrame:
    m = pd.read_csv(METRICS_PATH, parse_dates=["create_time"])
    m = m.sort_values("create_time").reset_index(drop=True)
    # bucket-END label (per reference_clean_data_locations) -> shift back 5min to align with
    # klines' bucket-START timestamp (both then refer to the same real-world instant, the bar's close)
    m["klines_ts"] = m["create_time"] - pd.Timedelta(minutes=5)
    m = m.drop_duplicates("klines_ts", keep="last")

    oi = m["sum_open_interest"]
    oi_roc_48 = oi - oi.shift(48)
    tt = m["sum_toptrader_long_short_ratio"]
    tt_z = (tt - tt.rolling(288, min_periods=288).mean()) / tt.rolling(288, min_periods=288).std().replace(0.0, np.nan)
    tk = m["sum_taker_long_short_vol_ratio"]
    tk_z = (tk - tk.rolling(288, min_periods=288).mean()) / tk.rolling(288, min_periods=288).std().replace(0.0, np.nan)

    merged = pd.DataFrame({
        "klines_ts": m["klines_ts"], "oi_roc_48": oi_roc_48,
        "toptrader_ls_z": tt_z, "taker_ls_z": tk_z,
    })
    out = pd.DataFrame({"timestamp": ts}).merge(
        merged, left_on="timestamp", right_on="klines_ts", how="left"
    ).drop(columns=["klines_ts", "timestamp"])
    log(f"metrics join coverage: {out['oi_roc_48'].notna().mean()*100:.1f}% non-null "
        f"(NaN outside metrics' own {m['klines_ts'].min()}..{m['klines_ts'].max()} range is expected)")
    return out


def main() -> int:
    eth, btc = load(ETH_PATH), load(BTC_PATH)
    sig = compute_signals(eth, btc, None)
    sig = sig[sig["timestamp"] >= START].reset_index(drop=True)
    feats = build_indicator_frame(eth)
    feats = feats[feats["timestamp"] >= START].reset_index(drop=True)
    assert len(feats) == len(sig) and (feats["timestamp"].to_numpy() == sig["timestamp"].to_numpy()).all()

    eth_full = eth.reset_index(drop=True)  # need pre-START rows so 288/864-bar rolling windows warm up
    new_feats_full = add_new_features(eth_full)
    new_feats_full["timestamp"] = eth_full["timestamp"]
    new_feats = new_feats_full[new_feats_full["timestamp"] >= START].reset_index(drop=True)
    assert len(new_feats) == len(feats) and (new_feats["timestamp"].to_numpy() == feats["timestamp"].to_numpy()).all()

    metrics_feats = add_metrics_features(feats["timestamp"])
    assert len(metrics_feats) == len(feats)

    feats = pd.concat([feats.reset_index(drop=True), new_feats[NEW_COLS[:5]].reset_index(drop=True),
                        metrics_feats.reset_index(drop=True)], axis=1)

    ts = sig["timestamp"]
    close = sig["close"].to_numpy(); high = sig["high"].to_numpy(); low = sig["low"].to_numpy()
    atr_pct = feats["atr_pct"].to_numpy()
    tall = ts.to_numpy()

    names = [n for n, _ in SIGNAL_ORDER]
    bot = np.zeros(len(sig), bool); top = np.zeros(len(sig), bool)
    for n in names:
        if f"bottom_{n}" in sig: bot |= sig[f"bottom_{n}"].to_numpy()
        if f"top_{n}" in sig:    top |= sig[f"top_{n}"].to_numpy()

    feat_cols = [c for c in FEATURE_COLUMNS if c != "is_bottom"] + NEW_COLS
    log(f"feature count: Tier0 22 + is_bottom + {len(NEW_COLS)} new = {len(feat_cols)+1}")
    results = []
    for h in HORIZON_GRID:
        up, dn = forward_extremes(close, high, low, h)
        for gap in GAP_GRID:
            rows = []
            for side, m in (("bottom", bot), ("top", top)):
                last = -10**9
                for i in np.flatnonzero(m):
                    if i - last < gap:
                        continue
                    last = i
                    rows.append((i, side == "bottom"))
            ev = pd.DataFrame(rows, columns=["i", "is_bottom"]).sort_values("i").reset_index(drop=True)
            iu = ev["i"].to_numpy(); isb = ev["is_bottom"].to_numpy()
            cont = np.where(isb, dn[iu], up[iu]); rev = np.where(isb, up[iu], dn[iu])
            ok = ~np.isnan(cont) & ~np.isnan(rev) & (atr_pct[iu] > 0)
            if ok.sum() < 200:
                log(f"H={h:>3d} GAP={gap:>2d}: skipped (n={int(ok.sum())} too small)")
                continue
            k50 = float(np.median(cont[ok] / atr_pct[iu][ok]))
            y_ext = ok & (cont >= k50 * atr_pct[iu]); y_rev = ok & (rev >= k50 * atr_pct[iu])
            pure = y_ext ^ y_rev

            X = feats.iloc[iu][feat_cols].copy()
            X["is_bottom"] = isb.astype(int)
            y = y_ext.astype(int)
            t = tall[iu]
            tr = pure & (t < VAL_START)
            va = pure & (t >= VAL_START) & (t < OOS_START)
            oo = pure & (t >= OOS_START) & (t < HOLDOUT_START)
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2 or len(np.unique(y[oo])) < 2:
                log(f"H={h:>3d} GAP={gap:>2d}: skipped (degenerate class split)")
                continue
            clf = HistGradientBoostingClassifier(random_state=GBM_SEED)
            clf.fit(X[tr], y[tr])
            val_auc = roc_auc_score(y[va], clf.predict_proba(X[va])[:, 1])
            oos_auc = roc_auc_score(y[oo], clf.predict_proba(X[oo])[:, 1])
            row = {
                "horizon": h, "gap": gap, "k50": round(k50, 3),
                "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_oos": int(oo.sum()),
                "val_auc": round(float(val_auc), 4), "oos_auc": round(float(oos_auc), 4),
                "val_oos_gap": round(abs(val_auc - oos_auc), 4),
                "min_val_oos": round(min(val_auc, oos_auc), 4),
            }
            results.append(row)
            log(f"H={h:>3d} GAP={gap:>2d}: K={k50:.2f} n_tr={row['n_train']:>5d} n_va={row['n_val']:>4d} "
                f"n_oo={row['n_oos']:>4d}  VAL={val_auc:.4f} OOS={oos_auc:.4f} min={row['min_val_oos']:.4f}")

    df = pd.DataFrame(results)
    out_dir = ROOT / "tmp/eth_trend_continuation_20260831"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "horizon_gap_gridscreen_augmented.csv", index=False)

    pd.set_option("display.width", 200)
    df_sorted = df.sort_values("min_val_oos", ascending=False)
    log("\n=== TOP 10 by min(VAL,OOS) AUC (augmented, 31 features) ===")
    print(df_sorted.head(10).to_string(index=False))
    log("\n=== BOTTOM 10 by min(VAL,OOS) AUC ===")
    print(df_sorted.tail(10).to_string(index=False))

    # side-by-side vs the Tier0-only baseline grid
    base_path = out_dir / "horizon_gap_gridscreen.csv"
    if base_path.exists():
        base = pd.read_csv(base_path)
        cmp = df.merge(base, on=["horizon", "gap"], suffixes=("_aug", "_base"))
        cmp["delta_min_val_oos"] = cmp["min_val_oos_aug"] - cmp["min_val_oos_base"]
        log("\n=== augmented vs Tier0-only baseline, min(VAL,OOS) AUC delta per cell ===")
        print(cmp[["horizon", "gap", "min_val_oos_base", "min_val_oos_aug", "delta_min_val_oos"]]
              .sort_values("delta_min_val_oos", ascending=False).to_string(index=False))
        log(f"\nmean delta = {cmp['delta_min_val_oos'].mean():+.4f}, "
            f"cells improved = {(cmp['delta_min_val_oos']>0).sum()}/{len(cmp)}")

    log(f"\nfull grid ({len(df)} cells) saved -> {out_dir / 'horizon_gap_gridscreen_augmented.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
