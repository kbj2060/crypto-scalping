#!/usr/bin/env python3
"""HORIZON x CLUSTER_GAP grid screen for the 2 Homer candidates confirmed to proceed: DeMarker
extreme and kalman_deviation_meanrev. Follows research_eth_liquidity_sweep_topdown_metalabel_
gridscreen_20260830.py's exact methodology verbatim: fast GBM proxy (not TabPFN), K held FIXED
during this screen (per-signal calibrated value from phase1, HORIZON/GAP optimization comes first,
K gets its own dedicated sweep afterward -- docs/homer/README.md 5.5), selection by min(VAL,OOS)
AUC (not a single split's max -- the volume_wick_climax lesson).

Reuses build_indicator_frame/FEATURE_COLUMNS verbatim (research_eth_taker_delta_climax_metalabel_
tabpfn_20260829.py, the shared Tier0 bank every signal in this lineage uses) plus each signal's own
trigger value as a 24th feature (same pattern taker/short_term_return_z used for keeping their own
delta_z/ret3_z as a feature, and liquidity_sweep used adding rsi as a 23rd).

⚠️Every grid cell is scored under BOTH the plain touch-only hit definition AND the 2026-08-31
exclude-middle v2 design (peak-band=0.2 + touch-then-reverse excluded) side by side -- deliberately
NOT assuming v2 wins. This project has a direct, cautionary precedent for touching a touch-label
with "did it hold" information: taker_delta_z_climax's v5 (hit = touched AND end>0, i.e.
RECLASSIFY the reversal cases to 0 instead of excluding them) was tried and made VAL/OOS/HOLDOUT
AUC WORSE (0.622/0.608/0.650 -> 0.562/0.561/0.606) -- the theory for why EXCLUDE should behave
differently from RECLASSIFY (removing an ambiguous, high-variance example vs. injecting a
contradictory label for feature vectors indistinguishable from genuine hits at fire time) has not
been empirically tested for these 2 signals before this screen.
"""
from __future__ import annotations

import sys
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

from research_eth_candidate_pool_raw_lift_check_20260831 import (  # noqa: E402
    kalman_level_and_velocity,
    rolling_zscore,
)
from research_eth_demarker_evidence_signal_lift_check_20260831 import compute_demarker  # noqa: E402
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import (  # noqa: E402
    FEATURE_COLUMNS,
    build_indicator_frame,
)

KLINES_PATH = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
START = pd.Timestamp("2024-01-01")
VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")

HORIZON_GRID = [8, 12, 16, 20, 24, 30, 36, 48]  # matches liquidity_sweep_topdown's own grid
GAP_GRID = [3, 6, 12]
PEAK_BAND = 0.2
GBM_SEED = 20260831


def log(msg: str) -> None:
    print(f"[kalman_demarker_gridscreen] {msg}", flush=True)


def load_klines() -> pd.DataFrame:
    df = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    assert df["timestamp"].diff().dropna().eq(pd.Timedelta(minutes=5)).all(), "gap/dup in klines"
    return df


def cluster_dedup(idx: np.ndarray, extremeness: np.ndarray, most_negative: bool, gap: int) -> np.ndarray:
    order = np.argsort(idx)
    idx_sorted, ex_sorted = idx[order], extremeness[order]
    cluster_id = np.zeros(len(idx_sorted), dtype=int)
    cid = 0
    for i in range(1, len(idx_sorted)):
        if idx_sorted[i] - idx_sorted[i - 1] > gap:
            cid += 1
        cluster_id[i] = cid
    df = pd.DataFrame({"idx": idx_sorted, "cluster": cluster_id, "ex": ex_sorted})
    keep = df.loc[df.groupby("cluster")["ex"].idxmin()] if most_negative else df.loc[df.groupby("cluster")["ex"].idxmax()]
    return np.sort(keep["idx"].to_numpy())


def build_fires(klines: pd.DataFrame, ind: pd.DataFrame, trigger_top: pd.Series, trigger_bottom: pd.Series,
                 extremeness: np.ndarray, feature_cols: list[str], horizon: int, gap: int, K: float) -> pd.DataFrame:
    high, low, close = klines["high"].to_numpy(), klines["low"].to_numpy(), klines["close"].to_numpy()
    atr_pct = ind["atr_pct"].to_numpy()
    ts = klines["timestamp"].to_numpy()
    n = len(klines)
    rows = []
    for side, trig in (("bottom", trigger_bottom), ("top", trigger_top)):
        idx = np.flatnonzero(trig.fillna(False).to_numpy())
        idx = idx[(idx < n - horizon) & (ts[idx] >= np.datetime64(START))]
        idx = cluster_dedup(idx, extremeness[idx], most_negative=(side == "bottom"), gap=gap)
        entry = close[idx]
        a = atr_pct[idx]
        end_close = close[idx + horizon]
        if side == "bottom":
            fut_ext = np.array([high[i + 1:i + horizon + 1].max() for i in idx])
            move_pct = (fut_ext - entry) / entry
            end_ret_pct = (end_close - entry) / entry
        else:
            fut_ext = np.array([low[i + 1:i + horizon + 1].min() for i in idx])
            move_pct = (entry - fut_ext) / entry
            end_ret_pct = (entry - end_close) / entry
        peak = move_pct / a
        end = end_ret_pct / a
        near_miss = np.abs(peak - K) < PEAK_BAND
        reversal = (peak >= K) & (end < 0)
        feat_rows = ind.iloc[idx]
        out = pd.DataFrame({
            "pos": idx, "timestamp": feat_rows["timestamp"].to_numpy(), "side": side,
            "hit_plain": (peak >= K).astype(float), "exclude_v2": near_miss | reversal,
            "is_bottom": 1 if side == "bottom" else 0,
        })
        for col_name in feature_cols:
            if col_name == "is_bottom":
                continue
            out[col_name] = feat_rows[col_name].to_numpy()
        rows.append(out)
    return pd.concat(rows, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def screen_signal(name: str, klines: pd.DataFrame, ind: pd.DataFrame, trigger_top: pd.Series,
                   trigger_bottom: pd.Series, extremeness: np.ndarray, K: float, feature_cols: list[str]) -> pd.DataFrame:
    results = []
    for horizon in HORIZON_GRID:
        for gap in GAP_GRID:
            fires = build_fires(klines, ind, trigger_top, trigger_bottom, extremeness, feature_cols, horizon, gap, K)
            fires = fires.dropna(subset=feature_cols + ["hit_plain"]).reset_index(drop=True)

            for variant in ("plain", "v2_exclude"):
                pool = fires if variant == "plain" else fires[~fires["exclude_v2"]].reset_index(drop=True)
                y = pool["hit_plain"].to_numpy().astype(int)
                ts = pool["timestamp"]
                train_mask = (ts < VAL_START).to_numpy()
                val_mask = ((ts >= VAL_START) & (ts < OOS_START)).to_numpy()
                oos_mask = ((ts >= OOS_START) & (ts < HOLDOUT_START)).to_numpy()
                if len(np.unique(y[train_mask])) < 2 or len(np.unique(y[val_mask])) < 2 or len(np.unique(y[oos_mask])) < 2:
                    log(f"  {name} H={horizon} GAP={gap} {variant}: skipped (degenerate class split)")
                    continue
                clf = HistGradientBoostingClassifier(random_state=GBM_SEED)
                clf.fit(pool.loc[train_mask, feature_cols], y[train_mask])
                val_auc = roc_auc_score(y[val_mask], clf.predict_proba(pool.loc[val_mask, feature_cols])[:, 1])
                oos_auc = roc_auc_score(y[oos_mask], clf.predict_proba(pool.loc[oos_mask, feature_cols])[:, 1])
                row = {
                    "signal": name, "horizon": horizon, "gap": gap, "variant": variant,
                    "n_train": int(train_mask.sum()), "n_val": int(val_mask.sum()), "n_oos": int(oos_mask.sum()),
                    "hit_rate": round(float(y.mean()), 4),
                    "val_auc": round(float(val_auc), 4), "oos_auc": round(float(oos_auc), 4),
                    "val_oos_gap": round(abs(val_auc - oos_auc), 4), "min_val_oos": round(min(val_auc, oos_auc), 4),
                }
                results.append(row)
                log(f"  {name} H={horizon:>3d} GAP={gap:>2d} {variant:>10s}: n={len(pool):>5d} "
                    f"hit_rate={row['hit_rate']:.3f} VAL={val_auc:.4f} OOS={oos_auc:.4f} min={row['min_val_oos']:.4f}")
    return pd.DataFrame(results)


def main() -> int:
    log("loading klines + building shared Tier0 indicator frame...")
    klines = load_klines()
    ind = build_indicator_frame(klines)
    log(f"{len(klines)} bars ready")

    dem = compute_demarker(klines["high"], klines["low"])
    ind_dem = ind.copy()
    ind_dem["dem"] = dem.to_numpy()
    dem_cols = FEATURE_COLUMNS + ["dem"]
    r_dem = screen_signal("demarker_extreme", klines, ind_dem, dem >= 0.90, dem <= 0.10,
                          dem.fillna(0.5).to_numpy(), K=1.9, feature_cols=dem_cols)

    levels, _ = kalman_level_and_velocity(klines["close"].to_numpy())
    kalman_dev = pd.Series((klines["close"].to_numpy() - levels) / levels, index=klines.index)
    kalman_dev_z = rolling_zscore(kalman_dev)
    ind_kal = ind.copy()
    ind_kal["kalman_dev_z"] = kalman_dev_z.to_numpy()
    kal_cols = FEATURE_COLUMNS + ["kalman_dev_z"]
    r_kal = screen_signal("kalman_deviation_meanrev", klines, ind_kal, kalman_dev_z >= 2.0, kalman_dev_z <= -2.0,
                          kalman_dev_z.fillna(0.0).to_numpy(), K=1.65, feature_cols=kal_cols)

    all_results = pd.concat([r_dem, r_kal], ignore_index=True)
    out_dir = ROOT / "tmp/eth_kalman_demarker_gridscreen_20260831"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(out_dir / "gridscreen_results.csv", index=False)

    pd.set_option("display.width", 200)
    cols = ["horizon", "gap", "val_auc", "oos_auc", "min_val_oos", "val_oos_gap", "hit_rate", "n_train", "n_val", "n_oos"]
    for name in all_results["signal"].unique():
        for variant in ("plain", "v2_exclude"):
            sub = all_results[(all_results["signal"] == name) & (all_results["variant"] == variant)]
            sub = sub.sort_values("min_val_oos", ascending=False)
            log(f"\n=== {name} / {variant}: TOP 5 by min(VAL,OOS) AUC ===")
            print(sub.head(5)[cols].to_string(index=False))

    log(f"\nfull grid saved -> {out_dir / 'gridscreen_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
