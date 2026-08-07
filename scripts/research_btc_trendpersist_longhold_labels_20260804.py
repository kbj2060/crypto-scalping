"""
BTC label-scheme redesign around the two closed-session structural findings
(see memory project-btc-session-20260802-03-arc-summary): BTC has ~half ETH's
zigzag/reversal wave frequency and ~30% lower ATR, so a fixed short-horizon
TP/SL floor copied from ETH's calibration takes 1.5-2x longer to breach and
forces long unplanned holds. Instead of fighting that (previous recalibration
attempt made OOS worse, see project-btc-tpsl-calibration-real-gap-but-fix-fails),
this designs FOR it: rarer CUSUM events (higher mult), a much longer intended
holding horizon, and wider TP/SL scaled off BTC's own ATR -- explicit
long-hold trend-following instead of scalp-style quick-exit.

Trend-scan features (mtf1h_ts_t_value/ts_opt_L) are EXCLUDED from feat_cols --
already re-tested post-lookahead-fix and closed (no edge), see
project-trendscan-lookahead-bug-found-fixed-20260804 /
project-btc-cusum-trendscan-architecture-closed-20260804. hurst_48/hurst_288/
ou_halflife/regime_persistence (genuinely causal, pre-existing pipeline
features) are used instead as trend-persistence context.

Stage 1 (this script): pure label statistics across a horizon x TP/SL-mult x
CUSUM-mult grid, split by train/VAL/OOS and by trending vs non-trending regime
(hurst_48), to check whether the redesigned label scheme has favorable
risk/reward BEFORE spending a training cycle on a classifier. Diagnostic only,
not Fresh-Forward validated.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_omega1_2_triple_barrier_labels_btc_20260708 import _atr_price_move, _reason_and_return  # noqa: E402
from compare_btc_label_schemes_20260803 import cusum_events  # noqa: E402

FRAME_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
OUT_CSV = ROOT / "tmp/btc_trendpersist_longhold_label_stats_20260804.csv"

FEE_COST = 0.0007
COST_CONSERVATIVE = (0.0005 + 0.0002) * 2.0 * 3.0  # 0.42%, same buffer used in prior sessions

VAL_START, OOS_START, OOS_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2026-01-01"), pd.Timestamp("2026-04-01")

# Grid: horizon in 5m bars, TP/SL ATR-multiples, CUSUM mult
HORIZONS = [288, 576, 864]          # 1 day / 2 days / 3 days
TP_SL_MULTS = [(1.2, 0.8), (2.0, 1.2), (2.5, 1.5)]
CUSUM_MULTS = [2.0, 3.0, 4.0]
MIN_TP, MIN_SL = 0.006, 0.004


def build_tb_at_events(frame: pd.DataFrame, events: np.ndarray, horizon: int,
                        tp_mult: float, sl_mult: float) -> pd.DataFrame:
    n = len(frame)
    open_px = frame["open"].to_numpy(dtype=np.float64)
    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)
    close = frame["close"].to_numpy(dtype=np.float64)
    atr = _atr_price_move(frame)
    ts = frame["timestamp"]
    hurst = frame["hurst_48"].to_numpy(dtype=np.float64) if "hurst_48" in frame.columns else np.full(n, 0.5)

    rows = []
    for i in events:
        entry_i = i + 1
        end_i = entry_i + horizon
        if end_i + 1 > n:
            continue
        entry = float(open_px[entry_i])
        vol = float(atr[i])
        tp_move = max(MIN_TP, tp_mult * vol)
        sl_move = max(MIN_SL, sl_mult * vol)
        fh, fl, fc = high[entry_i:end_i + 1], low[entry_i:end_i + 1], close[entry_i:end_i + 1]
        long_ret, long_reason, _, _, long_bars = _reason_and_return(
            side=1, entry=entry, future_high=fh, future_low=fl, future_close=fc,
            tp_move=tp_move, sl_move=sl_move)
        short_ret, short_reason, _, _, short_bars = _reason_and_return(
            side=-1, entry=entry, future_high=fh, future_low=fl, future_close=fc,
            tp_move=tp_move, sl_move=sl_move)
        long_q = long_ret - FEE_COST - 0.003 * int(long_reason == "sl")
        short_q = short_ret - FEE_COST - 0.003 * int(short_reason == "sl")
        if long_q > 0 and long_q >= short_q:
            action, ret, bars = 1, long_ret, long_bars
        elif short_q > 0:
            action, ret, bars = -1, short_ret, short_bars
        else:
            action, ret, bars = 0, 0.0, 0
        rows.append({"i": i, "timestamp": ts.iloc[i], "action": action, "ret": ret,
                      "bars": bars, "hurst_48": float(hurst[i])})
    return pd.DataFrame(rows)


def main():
    frame = pd.read_parquet(FRAME_PATH).sort_values("timestamp").reset_index(drop=True)
    atr = _atr_price_move(frame)

    results = []
    for cusum_mult in CUSUM_MULTS:
        events = cusum_events(frame, atr, mult=cusum_mult)
        events = events[events < len(frame) - max(HORIZONS) - 2]
        for horizon in HORIZONS:
            for tp_mult, sl_mult in TP_SL_MULTS:
                labels = build_tb_at_events(frame, events, horizon, tp_mult, sl_mult)
                if labels.empty:
                    continue
                for split_name, lo, hi in [
                    ("train", pd.Timestamp("2000-01-01"), VAL_START),
                    ("VAL", VAL_START, OOS_START),
                    ("OOS", OOS_START, OOS_END),
                ]:
                    split = labels[(labels["timestamp"] >= lo) & (labels["timestamp"] < hi)]
                    for regime_name, mask in [
                        ("all", np.ones(len(split), dtype=bool)),
                        ("trending", (split["hurst_48"] > 0.51).to_numpy()),
                        ("nontrending", (split["hurst_48"] <= 0.51).to_numpy()),
                    ]:
                        sub = split[mask]
                        taken = sub[sub["action"] != 0]
                        if len(taken) == 0:
                            continue
                        net = taken["ret"].to_numpy() - COST_CONSERVATIVE
                        win = (net > 0).sum()
                        results.append({
                            "cusum_mult": cusum_mult, "horizon_bars": horizon, "horizon_days": horizon * 5 / 1440,
                            "tp_mult": tp_mult, "sl_mult": sl_mult, "split": split_name, "regime": regime_name,
                            "n_events": len(split), "n_trades": len(taken),
                            "win_pct": 100 * win / len(taken), "mean_net_pct": 100 * net.mean(),
                            "sum_net_pct": 100 * net.sum(), "mean_hold_bars": taken["bars"].mean(),
                        })

    out = pd.DataFrame(results)
    out.to_csv(OUT_CSV, index=False)
    print(f"wrote {len(out)} rows -> {OUT_CSV}")

    # Print the most informative slice: "all" regime, OOS, ranked by mean_net_pct
    oos_all = out[(out["split"] == "OOS") & (out["regime"] == "all")].sort_values("mean_net_pct", ascending=False)
    print("\n=== Top 15 OOS (all regime) by mean_net_pct ===")
    print(oos_all.head(15).to_string(index=False))

    print("\n=== Same configs, VAL (all regime) for comparison ===")
    key_cols = ["cusum_mult", "horizon_bars", "tp_mult", "sl_mult"]
    val_all = out[(out["split"] == "VAL") & (out["regime"] == "all")]
    merged = oos_all.head(15)[key_cols].merge(val_all, on=key_cols, how="left", suffixes=("", "_val"))
    print(merged[key_cols + ["n_trades", "win_pct", "mean_net_pct", "sum_net_pct"]].to_string(index=False))

    print("\n=== Trending vs non-trending split (OOS), top VAL+OOS-both-positive configs ===")
    trend_oos = out[(out["split"] == "OOS") & (out["regime"] == "trending")]
    nontrend_oos = out[(out["split"] == "OOS") & (out["regime"] == "nontrending")]
    cmp_df = trend_oos.merge(nontrend_oos, on=key_cols, suffixes=("_trend", "_nontrend"))
    print(cmp_df[key_cols + ["n_trades_trend", "mean_net_pct_trend", "n_trades_nontrend", "mean_net_pct_nontrend"]].to_string(index=False))


if __name__ == "__main__":
    main()
