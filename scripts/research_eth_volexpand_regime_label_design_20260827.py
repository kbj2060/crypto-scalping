#!/usr/bin/env python3
"""Phase 0+1 of the VolExpand plan (/home/kbj20/.claude/plans/pure-hugging-book.md, approved
2026-08-27): cheap go/no-go check on realized_vol_ratio as an early-warning signal for GBM2
chop->trend transitions, then (if Phase 0 doesn't kill it) a label-design grid (cutoff x debounce-K)
on the offline TRAIN split. No model training here -- see scripts/train_eth_regime_volexpand_
20260827.py (Phase 2) for that.

Motivation (see plan file): GBM2's efficiency-ratio label doesn't fire on a retracement-heavy
"grinding" decline even when net displacement is real -- concretely, ETH 2528->2417 over 17h on
2026-08-25/26 was mostly classified chop. Hypothesis: a magnitude-based volatility signal
(realized_vol_ratio = rv_short(12bar)/rv_long(288bar), already computed by features.engineering.
FeatureEngineer, no new formula) isn't reset by contrary bars the way efficiency ratio is, so it
could stabilize (need less debounce) and confirm earlier than GBM2 without any forward-looking
forecasting -- purely a same-time state classifier, like GBM2 itself.

Phase 0: live-fetch ~20 days (klines depth is fine; OI/ratio endpoints cap ~500 rows/~1.7d
regardless of requested range -- same fix as scripts in this session's scratchpad history: fetch
deep klines anyway, leave OI-derived columns NaN pre-coverage, rely on the model's own median-fill
fallback), zoom on the 08-25/26 episode. This window is OUTSIDE the offline TRAIN/VAL/OOS split
(those CSVs end 2026-08-19 23:55) -- treated as an illustrative case study only, never used to fit
any threshold/K below.

Phase 1: on the offline TRAIN split (2024-01-01~2026-06-30, the same 3 CSVs GBM2 trained on),
realized_vol_ratio already exists as a precomputed column (confirmed by direct inspection --
FeatureEngineer's output is baked into these CSVs, no need to recompute). Grid over (percentile
cutoff x debounce K) purely on label construction (no model fit, cheap): report flip_rate AND
class-share per cell side by side -- GBM2's own history showed flip_rate alone can hide a lock-up
(K=48 had the best flip_rate but silently collapsed trend_share 0.45->0.12) -- plus a flip_rate(K)
comparison against GBM2's own raw/confirmed numbers (0.1877 raw / 0.0128 at K=12) to directly test
this script's stated hypothesis.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from live_regime_wide24_signal_20260826 import SYMBOL, BTC_SYMBOL, _fetch_klines, _fetch_data_api, _fetch_funding  # noqa: E402
from retrain_clean_regime_hmm_raw_state12_20260517 import _with_raw_state12  # noqa: E402
from train_eth_regime_gbm2_trend_chop_20260827 import _apply_hysteresis, _debounce, K_BARS_LABEL  # noqa: E402
from features.engineering import FeatureEngineer  # noqa: E402
from features.elite import RegimeEngine  # noqa: E402

GBM2_MODEL_PATH = ROOT / "tmp/eth_regime_gbm2_trend_chop_20260827/model.joblib"
GBM3_MODEL_PATH = ROOT / "tmp/eth_regime_gbm3_independent_20260826/model.joblib"
CLASSES2 = ["chop", "trend"]
OUT_DIR = ROOT / "tmp" / "eth_volexpand_regime_label_design_20260827"

TRAIN_CSVS = [
    ROOT / "data/splits/year_oos/training_features_2024.csv",
    ROOT / "data/splits/year_oos/training_features_2025.csv",
    ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
]
TRAIN_START = pd.Timestamp("2024-01-01T00:00:00")
TRAIN_END = pd.Timestamp("2026-06-30T23:55:00")  # matches GBM2's TRAIN (incl its internal VAL)


# =============================================================================================
# Phase 0 -- live-fetch chart, concrete 2026-08-25/26 episode
# =============================================================================================

def phase0_live_chart() -> None:
    FETCH_DAYS = 20
    now = pd.Timestamp.now("UTC").tz_localize(None)
    end_ms = int(now.timestamp() * 1000)
    start_ms = int((now - pd.Timedelta(days=FETCH_DAYS)).timestamp() * 1000)

    print(f"[phase0] Fetching live data, now(UTC)={now}, FETCH_DAYS={FETCH_DAYS}...")
    eth_kline = _fetch_klines(SYMBOL, start_ms, end_ms)
    btc_kline = _fetch_klines(BTC_SYMBOL, start_ms, end_ms)
    oi = _fetch_data_api("/futures/data/openInterestHist", SYMBOL, start_ms, end_ms, {"sumOpenInterestValue": "sum_open_interest_value"})
    top_ratio = _fetch_data_api("/futures/data/topLongShortPositionRatio", SYMBOL, start_ms, end_ms, {"longShortRatio": "sum_toptrader_long_short_ratio"})
    acct_ratio = _fetch_data_api("/futures/data/globalLongShortAccountRatio", SYMBOL, start_ms, end_ms, {"longShortRatio": "count_long_short_ratio"})
    funding = _fetch_funding(SYMBOL, start_ms, end_ms)

    raw = eth_kline.copy()
    for extra in (oi, top_ratio, acct_ratio):
        raw = pd.merge_asof(raw.sort_values("timestamp"), extra.sort_values("timestamp"), on="timestamp", direction="backward")
    raw = pd.merge_asof(raw.sort_values("timestamp"), funding, on="timestamp", direction="backward")
    btc = btc_kline.rename(columns={"close": "close_btc", "volume": "volume_btc", "quote_volume": "quote_volume_btc"})
    raw = raw.merge(btc[["timestamp", "close_btc", "volume_btc", "quote_volume_btc"]], on="timestamp", how="left")
    raw = raw.dropna(subset=["close_btc"]).reset_index(drop=True)
    print(f"[phase0] raw rows: {len(raw)}, range {raw['timestamp'].min()} ~ {raw['timestamp'].max()}")

    eth_raw_cols = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume",
                     "trades", "taker_buy_base", "taker_buy_quote",
                     "sum_open_interest_value", "sum_toptrader_long_short_ratio",
                     "count_long_short_ratio", "last_funding_rate"]
    btc_raw_cols = ["timestamp", "close_btc", "volume_btc", "quote_volume_btc"]
    eth_df = raw[eth_raw_cols].copy()
    btc_df = raw[btc_raw_cols].copy()

    feats = FeatureEngineer().process(eth_df, btc_df)
    feats = _with_raw_state12(feats)

    # RegimeEngine ground-truth (rule-based, no model) -- for reference alongside the trained model
    labeled = RegimeEngine().compute(feats.copy())
    is_trend_raw = ((labeled["regime_bull"] + labeled["regime_bear"]) > 0).to_numpy().astype(int)
    is_trend_confirmed_gt = _debounce(is_trend_raw, K_BARS_LABEL)

    # GBM2 trained model (what the live dashboard actually serves)
    payload = joblib.load(GBM2_MODEL_PATH)
    cols = payload["feature_cols"]
    med = pd.Series(payload["feature_medians"])
    for c in cols:
        if c not in feats.columns:
            feats[c] = med.get(c, 0.0)
    x = feats[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    proba = payload["model"].predict_proba(x)
    trend_prob = proba[:, CLASSES2.index("trend")]
    hcfg = payload["hysteresis_config"]
    model_confirmed = _apply_hysteresis(trend_prob, hcfg["k_bars"], hcfg["band"])

    # GBM3 trained model (2026-08-27: re-instated as the actual live dashboard signal, replacing
    # GBM2 there per user decision -- so THIS is what's really on screen now, no debounce/
    # hysteresis of its own (that lack is exactly why GBM2 was built), raw argmax != chop)
    payload3 = joblib.load(GBM3_MODEL_PATH)
    cols3 = payload3["feature_cols"]
    med3 = pd.Series(payload3["feature_medians"])
    feats3 = feats.copy()
    for c in cols3:
        if c not in feats3.columns:
            feats3[c] = med3.get(c, 0.0)
    x3 = feats3[cols3].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med3).fillna(0.0)
    proba3 = payload3["model"].predict_proba(x3)
    classes3 = list(payload3["classes"])
    is_trend_gbm3 = (np.argmax(proba3, axis=1) != classes3.index("chop")).astype(int)

    out = pd.DataFrame({
        "timestamp": feats["timestamp"].reset_index(drop=True),
        "close": feats["close"].reset_index(drop=True),
        "realized_vol_ratio": feats["realized_vol_ratio"].reset_index(drop=True),
        "gbm2_model_confirmed": model_confirmed,
        "gbm2_gt_confirmed": is_trend_confirmed_gt[: len(feats)],
        "gbm3_is_trend": is_trend_gbm3,
    }).dropna().reset_index(drop=True)

    DISPLAY_START = pd.Timestamp("2026-08-24 00:00:00")
    disp = out[out["timestamp"] >= DISPLAY_START].reset_index(drop=True)
    disp["timestamp_kst"] = disp["timestamp"] + pd.Timedelta(hours=9)
    print(f"[phase0] display rows: {len(disp)}, KST range {disp['timestamp_kst'].iloc[0]} ~ {disp['timestamp_kst'].iloc[-1]}")

    def shade(ax, ts, codes, color_map):
        if len(codes) == 0:
            return
        start_i, cur = 0, codes[0]
        for i in range(1, len(codes)):
            if codes[i] != cur:
                ax.axvspan(ts[start_i], ts[i], color=color_map[cur], alpha=0.4, lw=0)
                start_i, cur = i, codes[i]
        ax.axvspan(ts[start_i], ts[len(codes) - 1], color=color_map[cur], alpha=0.4, lw=0)

    plt.rcParams.update({"font.size": 22, "axes.titlesize": 25, "axes.labelsize": 24,
                          "xtick.labelsize": 19, "ytick.labelsize": 19})
    fig, axes = plt.subplots(3, 1, figsize=(32, 28), sharex=True)
    ts_arr = disp["timestamp_kst"].to_numpy()
    kst_start, kst_end = disp["timestamp_kst"].iloc[0], disp["timestamp_kst"].iloc[-1]
    color_map = {0: "#8b91a6", 1: "#3b6fd6"}

    ax = axes[0]
    shade(ax, ts_arr, disp["gbm2_model_confirmed"].to_numpy(), color_map)
    ax.plot(disp["timestamp_kst"], disp["close"], color="#111", linewidth=2.4)
    ax.set_title(f"ETH close, shaded=GBM2 model confirmed_state(blue=trend/gray=chop), {kst_start:%m-%d %H:%M}~{kst_end:%m-%d %H:%M} KST\n"
                 f"(motivating episode: 08-25 13h~08-26 06h, 2528->2417, -4.4%)")
    ax.set_ylabel("close (GBM2)")
    ax.grid(alpha=0.25)

    ax3 = axes[1]
    shade(ax3, ts_arr, disp["gbm3_is_trend"].to_numpy(), color_map)
    ax3.plot(disp["timestamp_kst"], disp["close"], color="#111", linewidth=2.4)
    ax3.set_title("ETH close, shaded=GBM3 raw argmax!=chop (blue=trend/gray=chop) -- GBM3 is what's actually live on the dashboard "
                  "now (2026-08-27 revert), no debounce of its own")
    ax3.set_ylabel("close (GBM3)")
    ax3.grid(alpha=0.25)

    ax2 = axes[2]
    ax2.plot(disp["timestamp_kst"], disp["realized_vol_ratio"], color="#c1440e", linewidth=2.4)
    ax2.axhline(1.0, color="#555", linestyle=":", linewidth=1.5)
    ax2.set_title("realized_vol_ratio = rv_short(12bar)/rv_long(288bar) -- does it visibly rise before/during the episode above?")
    ax2.set_ylabel("ratio")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %Hh KST"))
    ax2.tick_params(axis="x", rotation=0)
    ax2.grid(alpha=0.25)

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "phase0_live_episode_chart.png"
    plt.savefig(out_path, dpi=145)
    print(f"[phase0] saved: {out_path}")

    disp.to_csv(OUT_DIR / "phase0_live_episode_data.csv", index=False)
    print(f"[phase0] saved: {OUT_DIR / 'phase0_live_episode_data.csv'}")


# =============================================================================================
# Phase 1 -- offline TRAIN grid: cutoff x debounce-K, flip_rate + class-share, no model fit
# =============================================================================================

def flip_rate(codes: np.ndarray) -> float:
    return float(np.mean(codes[1:] != codes[:-1])) if len(codes) > 1 else 0.0


def phase1_train_grid() -> None:
    frames = [pd.read_csv(p, parse_dates=["timestamp"]) for p in TRAIN_CSVS]
    raw = pd.concat(frames, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    raw = raw[(raw["timestamp"] >= TRAIN_START) & (raw["timestamp"] <= TRAIN_END)].reset_index(drop=True)
    print(f"[phase1] TRAIN rows: {len(raw)}, {raw['timestamp'].min()} ~ {raw['timestamp'].max()}")
    assert "realized_vol_ratio" in raw.columns, "realized_vol_ratio missing from offline TRAIN csv -- unexpected"
    ratio = raw["realized_vol_ratio"].to_numpy()
    valid = np.isfinite(ratio)
    print(f"[phase1] realized_vol_ratio valid: {valid.sum()}/{len(ratio)} ({valid.mean():.1%})")

    # GBM2 reference numbers (from its own report, for the flip_rate(K) comparison this script's
    # hypothesis rests on -- computed independently here too, not hardcoded, in case of drift)
    labeled = RegimeEngine().compute(raw.copy())
    gbm2_is_trend_raw = ((labeled["regime_bull"] + labeled["regime_bear"]) > 0).to_numpy().astype(int)
    gbm2_raw_flip = flip_rate(gbm2_is_trend_raw)
    gbm2_confirmed_flip = flip_rate(_debounce(gbm2_is_trend_raw, K_BARS_LABEL))
    print(f"[phase1] GBM2 reference: raw flip_rate={gbm2_raw_flip:.4f}, K={K_BARS_LABEL} confirmed flip_rate={gbm2_confirmed_flip:.4f}")

    # GBM3 reference (2026-08-27: re-instated as the actual live dashboard signal) -- its own
    # trained-model predictions on this same TRAIN split, not just the RegimeEngine ground truth,
    # since GBM3 has no debounce/hysteresis of its own (that gap is exactly why GBM2 exists) so its
    # raw model flip_rate IS effectively its live flip_rate.
    feats_train = _with_raw_state12(raw.copy())
    payload3 = joblib.load(GBM3_MODEL_PATH)
    cols3 = payload3["feature_cols"]
    med3 = pd.Series(payload3["feature_medians"])
    for c in cols3:
        if c not in feats_train.columns:
            feats_train[c] = med3.get(c, 0.0)
    x3 = feats_train[cols3].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med3).fillna(0.0)
    proba3 = payload3["model"].predict_proba(x3)
    classes3 = list(payload3["classes"])
    gbm3_is_trend = (np.argmax(proba3, axis=1) != classes3.index("chop")).astype(int)
    gbm3_flip = flip_rate(gbm3_is_trend)
    print(f"[phase1] GBM3 reference (live model, no debounce): flip_rate={gbm3_flip:.4f}, trend_share={gbm3_is_trend.mean():.4f}")

    CUTOFF_PCTS = [0.20, 0.25, 0.30]  # top-N% of ratio treated as "expanding" raw state
    K_GRID = [6, 12, 24, 48]

    rows = []
    for pct in CUTOFF_PCTS:
        thresh = float(np.nanquantile(ratio, 1.0 - pct))
        raw_expand = np.where(valid, (ratio >= thresh).astype(int), 0)
        for K in K_GRID:
            confirmed = _debounce(raw_expand, K)
            rows.append({
                "cutoff_top_pct": pct, "threshold": thresh, "K": K,
                "raw_flip_rate": flip_rate(raw_expand), "raw_expand_share": float(raw_expand.mean()),
                "confirmed_flip_rate": flip_rate(confirmed), "confirmed_expand_share": float(confirmed.mean()),
            })
    grid = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n[phase1] cutoff x K grid (label construction only, no model fit):")
    print(grid.round(4).to_string(index=False))

    print(f"\n[phase1] hypothesis check -- does any (cutoff,K) reach confirmed_flip_rate comparable to "
          f"or below GBM2's K={K_BARS_LABEL} confirmed flip_rate ({gbm2_confirmed_flip:.4f}) at a SMALLER K "
          f"(i.e. stabilizes faster)?")
    smaller_k_comparable = grid[(grid["K"] < K_BARS_LABEL) & (grid["confirmed_flip_rate"] <= gbm2_confirmed_flip * 1.5)]
    if len(smaller_k_comparable):
        print(smaller_k_comparable.round(4).to_string(index=False))
    else:
        print("  none -- no (cutoff,K<12) cell got within 1.5x of GBM2's confirmed flip_rate")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid.to_csv(OUT_DIR / "phase1_cutoff_k_grid.csv", index=False)
    print(f"\n[phase1] saved: {OUT_DIR / 'phase1_cutoff_k_grid.csv'}")


if __name__ == "__main__":
    phase0_live_chart()
    phase1_train_grid()
