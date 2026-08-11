"""Gate G4b -- how much effective sample size does pooling the 60-asset panel actually buy?

G3 established that effective sample size is a property of the data and the label span, not of the
sampling scheme, leaving three levers: shorter spans, longer history, or more assets. This gate
measures the third on the existing 60-symbol USDT-perp panel (data/panel/features/*.parquet, 5m,
2024-01-01..2026-08-04), using the identical triple-barrier construction BTC uses (12-bar cumret
dispersion over a 288-bar causal lookback, TP_MULT=2.5, SL_MULT=1.2, horizon=288).

Three things are measured, because the naive answer (60x) is certainly wrong:

1. **Per-asset temporal effective_n**, same average-uniqueness computation as G2/G3, so the numbers
   are directly comparable to BTC's 4,058 (stride-4) / 4,224 (stride-1).
2. **Per-asset oracle ceiling** (protocol step 3): trading the TRUE label through the same
   simulator. Pooling an asset whose label carries no economic ceiling adds noise, not data --
   BTC's own ceiling is 100% win rate / 44.3x OOS equity, and any asset far below that is a
   candidate for exclusion rather than inclusion.
3. **Cross-sectional redundancy.** Crypto perps are heavily co-moving, so 60 assets do not carry 60
   independent label streams. Measured two ways: pairwise agreement rate of the hard labels at
   matched timestamps, and the effective rank (participation ratio of the eigenvalue spectrum) of
   the 5m return correlation matrix. The pooled estimate is reported as
   `sum(per-asset effective_n) * effective_rank / n_assets`, explicitly an estimate, alongside the
   naive sum so the size of the correction is visible.

Per-asset labels and spans are written to data/panel/tripbarrier/ so a pooled training run does not
have to recompute them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numba
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.causal_futures_backtest import simulate_single_position  # noqa: E402

PANEL_DIR = ROOT / "data/panel/features"
LABEL_OUT_DIR = ROOT / "data/panel/tripbarrier"
OUT_DIR = ROOT / "tmp/btc_gate_g4b_panel_pooling_20260807"

CUMRET_BARS, VOL_LOOKBACK, TP_MULT, SL_MULT, HORIZON_BARS = 12, 288, 2.5, 1.2, 288
MARGIN_FRACTION, LEVERAGE, ROUNDTRIP_COST_RATE = 0.30, 3.0, 0.0010
WINDOW = 48
VAL_START, VAL_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-12-31")
OOS_START, OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31")
TRAIN_STRIDE = 4


@numba.njit(cache=True)
def _tb_label_and_span(open_, high, low, tp_move, sl_move, horizon):
    """Hard 3-class triple-barrier label (0=CASH/1=LONG/2=SHORT) plus the bar count until BOTH
    sides' races resolve -- identical semantics to
    scripts/build_btc_5m_tripbarrier_tradeoutcome_labels_20260806.py and
    scripts/build_btc_5m_tripbarrier_label_span_20260807.py."""
    n = len(open_)
    label = np.zeros(n, dtype=np.int8)
    span = np.full(n, horizon, dtype=np.int32)
    valid = np.zeros(n, dtype=np.bool_)
    for i in range(n - 1):
        entry_i = i + 1
        if not np.isfinite(tp_move[i]) or not np.isfinite(sl_move[i]):
            continue
        entry = open_[entry_i]
        if not np.isfinite(entry) or entry <= 0.0:
            continue
        valid[i] = True
        tp_l, sl_l = entry * (1.0 + tp_move[i]), entry * (1.0 - sl_move[i])
        tp_s, sl_s = entry * (1.0 - tp_move[i]), entry * (1.0 + sl_move[i])
        long_done, short_done = False, False
        long_sign, short_sign = 0, 0
        long_t, short_t = horizon, horizon
        final_i = entry_i + horizon - 1
        if final_i >= n:
            final_i = n - 1
        for j in range(entry_i, final_i + 1):
            t = j - entry_i + 1
            if not long_done:
                if low[j] <= sl_l:
                    long_done, long_sign, long_t = True, -1, t
                elif high[j] >= tp_l:
                    long_done, long_sign, long_t = True, 1, t
            if not short_done:
                if high[j] >= sl_s:
                    short_done, short_sign, short_t = True, -1, t
                elif low[j] <= tp_s:
                    short_done, short_sign, short_t = True, 1, t
            if long_done and short_done:
                break
        long_tp = long_done and long_sign == 1
        short_tp = short_done and short_sign == 1
        if long_tp and not short_tp:
            label[i] = 1
        elif short_tp and not long_tp:
            label[i] = 2
        span[i] = long_t if long_t > short_t else short_t
    return label, span, valid


def _uniqueness(sample_idx, label_end, n_bars):
    if len(sample_idx) == 0:
        return np.zeros(0, dtype=np.float64)
    starts = sample_idx + 1
    ends = np.minimum(label_end[sample_idx], n_bars - 1)
    keep = ends >= starts
    starts, ends = starts[keep], ends[keep]
    conc = np.zeros(n_bars + 1, dtype=np.float64)
    np.add.at(conc, starts, 1.0)
    np.add.at(conc, ends + 1, -1.0)
    conc = np.cumsum(conc)[:n_bars]
    inv_c = np.where(conc > 0, 1.0 / np.maximum(conc, 1e-12), 0.0)
    prefix = np.concatenate([[0.0], np.cumsum(inv_c)])
    return (prefix[ends + 1] - prefix[starts]) / np.maximum(ends - starts + 1, 1)


def _fresh_entry_mask(side_state):
    fresh = np.zeros(len(side_state), dtype=bool)
    fresh[0] = side_state[0] != 0
    fresh[1:] = (side_state[1:] != 0) & (side_state[1:] != side_state[:-1])
    return fresh


def _oracle(row_idx, side_full, tp_all, sl_all, frame):
    side = side_full[row_idx]
    fresh = _fresh_entry_mask(side)
    idx, sd = row_idx[fresh], side[fresh]
    tp, sl = tp_all[idx], sl_all[idx]
    ok = np.isfinite(tp) & np.isfinite(sl)
    idx, sd, tp, sl = idx[ok], sd[ok], tp[ok], sl[ok]
    if len(idx) == 0:
        return None
    r = simulate_single_position(
        timestamps=frame["timestamp"], open_px=frame["open"].to_numpy(dtype=np.float64),
        high=frame["high"].to_numpy(dtype=np.float64), low=frame["low"].to_numpy(dtype=np.float64),
        close=frame["close"].to_numpy(dtype=np.float64), decision_indices=idx,
        scores=sd.astype(np.float64), tp_moves=tp, sl_moves=sl, upper_threshold=0.0,
        lower_threshold=0.0, horizon_bars=HORIZON_BARS, margin_fraction=MARGIN_FRACTION,
        leverage=LEVERAGE, roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
    )
    if len(r.ledger) == 0:
        return None
    rets = r.ledger["trade_return"].to_numpy(dtype=np.float64)
    return {"n_trades": int(len(rets)), "win_rate": float((rets > 0).mean()),
            "final_equity": float(r.equity[-1])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="process only the first N symbols (smoke test)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(PANEL_DIR.glob("*.parquet"))
    if args.limit:
        files = files[: args.limit]
    print(f"{len(files)} symbols found")

    rows, action_by_symbol, ret_by_symbol = [], {}, {}
    for k, f in enumerate(files, 1):
        sym = f.stem
        fr = pd.read_parquet(f, columns=["timestamp", "open", "high", "low", "close"])
        fr = fr.sort_values("timestamp").reset_index(drop=True)
        ts = fr["timestamp"].to_numpy()
        close = fr["close"].to_numpy(dtype=np.float64)
        n = len(fr)

        log_ret = np.diff(np.log(np.clip(close, 1e-9, None)), prepend=np.log(max(close[0], 1e-9)))
        cumret = pd.Series(log_ret).rolling(CUMRET_BARS).sum().to_numpy()
        vol = pd.Series(cumret).rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).std().to_numpy()
        tp_all, sl_all = TP_MULT * vol, SL_MULT * vol

        label, span, valid = _tb_label_and_span(
            fr["open"].to_numpy(dtype=np.float64), fr["high"].to_numpy(dtype=np.float64),
            fr["low"].to_numpy(dtype=np.float64), tp_all, sl_all, HORIZON_BARS,
        )
        pd.DataFrame({"timestamp": fr["timestamp"], "trade_outcome_action": label,
                      "label_span_bars": span, "label_valid": valid}).to_parquet(
            LABEL_OUT_DIR / f"{sym}.parquet", index=False)

        label_end = np.arange(n, dtype=np.int64) + span
        val_start_i = int(np.searchsorted(ts, np.datetime64(VAL_START)))
        train_bars = np.arange(WINDOW - 1, val_start_i, dtype=np.int64)
        train_bars = train_bars[valid[train_bars]]
        u1 = _uniqueness(train_bars, label_end, n)
        u4 = _uniqueness(train_bars[::TRAIN_STRIDE], label_end, n)

        side_full = np.where(label == 1, 1, np.where(label == 2, -1, 0))
        splits = {
            "val": np.flatnonzero((ts >= np.datetime64(VAL_START)) & (ts <= np.datetime64(VAL_END))),
            "oos": np.flatnonzero((ts >= np.datetime64(OOS_START)) & (ts <= np.datetime64(OOS_END))),
        }
        orc = {s: _oracle(ix, side_full, tp_all, sl_all, fr) for s, ix in splits.items()}

        row = {
            "symbol": sym, "n_bars": int(n), "n_train_bars": int(len(train_bars)),
            "nan_close_share": float(np.isnan(close).mean()),
            "mean_span": float(span[train_bars].mean()) if len(train_bars) else float("nan"),
            "median_span": float(np.median(span[train_bars])) if len(train_bars) else float("nan"),
            "effective_n_stride1": float(u1.sum()), "effective_n_stride4": float(u4.sum()),
            "cash_share": float((label[train_bars] == 0).mean()) if len(train_bars) else float("nan"),
            "oracle_oos_win_rate": (orc["oos"] or {}).get("win_rate"),
            "oracle_oos_equity": (orc["oos"] or {}).get("final_equity"),
            "oracle_oos_trades": (orc["oos"] or {}).get("n_trades"),
            "oracle_val_win_rate": (orc["val"] or {}).get("win_rate"),
            "oracle_val_equity": (orc["val"] or {}).get("final_equity"),
        }
        rows.append(row)

        # panel files differ by a few bars in length/coverage, so cross-sectional work is done on
        # the timestamp INTERSECTION rather than by assuming a shared grid
        idx = pd.DatetimeIndex(ts)
        action_by_symbol[sym] = pd.Series(label, index=idx)
        ret_by_symbol[sym] = pd.Series(log_ret, index=idx)
        print(f"[{k}/{len(files)}] {sym}: eff_n(s4)={row['effective_n_stride4']:.0f} "
              f"span_med={row['median_span']:.0f} oracle_oos_wr={row['oracle_oos_win_rate']} "
              f"eq={row['oracle_oos_equity']}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "g4b_per_asset.csv", index=False)

    # ---- cross-sectional redundancy ----
    syms = sorted(action_by_symbol)
    A_df = pd.concat([action_by_symbol[s].rename(s) for s in syms], axis=1, join="inner")
    R_df = pd.concat([ret_by_symbol[s].rename(s) for s in syms], axis=1, join="inner")
    print(f"\ncommon timestamp intersection: {len(A_df)} bars across {len(syms)} symbols")
    A = A_df.to_numpy().T                                      # (S, n) hard labels
    R = R_df.to_numpy(dtype=np.float64).T                      # (S, n) 5m log returns
    # agreement must be conditioned PAIRWISE. Requiring all S assets to be simultaneously non-CASH
    # selects a vanishing, extreme subset once S is large (and inflates agreement toward 1), so each
    # pair is evaluated only on the bars where that pair is jointly active.
    agree = np.full((len(syms), len(syms)), np.nan)
    act = A != 0
    for i in range(len(syms)):
        for j in range(len(syms)):
            both = act[i] & act[j]
            agree[i, j] = float((A[i, both] == A[j, both]).mean()) if both.any() else np.nan
    off = agree[~np.eye(len(syms), dtype=bool)]
    off = off[np.isfinite(off)]

    finite = np.isfinite(R).all(axis=0)
    C = np.corrcoef(R[:, finite])
    eig = np.linalg.eigvalsh(C)
    eig = np.clip(eig, 0, None)
    effective_rank = float(eig.sum() ** 2 / (eig ** 2).sum())  # participation ratio

    naive_sum_s1 = float(df["effective_n_stride1"].sum())
    naive_sum_s4 = float(df["effective_n_stride4"].sum())
    corrected_s1 = naive_sum_s1 * effective_rank / len(syms)
    corrected_s4 = naive_sum_s4 * effective_rank / len(syms)

    btc = df[df["symbol"] == "BTCUSDT"].iloc[0] if (df["symbol"] == "BTCUSDT").any() else None

    print("\n=== per-asset effective_n (stride 4) ===")
    d = df.sort_values("effective_n_stride4", ascending=False)
    print(d[["symbol", "median_span", "effective_n_stride4", "oracle_oos_win_rate",
             "oracle_oos_equity", "oracle_oos_trades"]].head(12).to_string(index=False))
    print("  ...")
    print(d[["symbol", "median_span", "effective_n_stride4", "oracle_oos_win_rate",
             "oracle_oos_equity", "oracle_oos_trades"]].tail(5).to_string(index=False))

    print("\n=== oracle ceiling across the panel ===")
    print(f"  OOS win rate: min {df['oracle_oos_win_rate'].min():.3f} "
          f"median {df['oracle_oos_win_rate'].median():.3f} max {df['oracle_oos_win_rate'].max():.3f}")
    print(f"  OOS equity:   min {df['oracle_oos_equity'].min():.1f}x "
          f"median {df['oracle_oos_equity'].median():.1f}x max {df['oracle_oos_equity'].max():.1f}x")
    print(f"  assets with OOS oracle win rate < 0.95: {int((df['oracle_oos_win_rate'] < 0.95).sum())}")

    print("\n=== cross-sectional redundancy ===")
    print(f"  symbols on the common timestamp grid: {len(syms)}/{len(files)}")
    print(f"  pairwise hard-label agreement (off-diagonal): mean {off.mean():.3f} "
          f"p10 {np.percentile(off,10):.3f} p90 {np.percentile(off,90):.3f}")
    print(f"  5m return correlation: mean off-diag {C[~np.eye(len(syms),dtype=bool)].mean():.3f}")
    print(f"  effective rank (participation ratio of eigenspectrum): {effective_rank:.2f} of {len(syms)}")

    print("\n=== pooled effective sample size ===")
    if btc is not None:
        print(f"  BTC alone (stride4): {btc['effective_n_stride4']:.0f}   (stride1: {btc['effective_n_stride1']:.0f})")
    print(f"  naive sum over {len(df)} assets (stride4): {naive_sum_s4:.0f}   (stride1: {naive_sum_s1:.0f})")
    print(f"  redundancy-corrected estimate (stride4): {corrected_s4:.0f}   (stride1: {corrected_s1:.0f})")
    if btc is not None:
        print(f"  => multiplier over BTC alone: naive {naive_sum_s4/btc['effective_n_stride4']:.1f}x, "
              f"corrected {corrected_s4/btc['effective_n_stride4']:.1f}x")

    payload = {
        "config": {"tp_mult": TP_MULT, "sl_mult": SL_MULT, "horizon_bars": HORIZON_BARS,
                   "cumret_bars": CUMRET_BARS, "vol_lookback": VOL_LOOKBACK,
                   "train_stride": TRAIN_STRIDE, "window": WINDOW},
        "n_symbols": len(df), "n_symbols_on_common_grid": len(syms),
        "per_asset": rows,
        "cross_section": {
            "pairwise_label_agreement_mean": float(off.mean()),
            "pairwise_label_agreement_p10": float(np.percentile(off, 10)),
            "pairwise_label_agreement_p90": float(np.percentile(off, 90)),
            "return_corr_mean_offdiag": float(C[~np.eye(len(syms), dtype=bool)].mean()),
            "effective_rank": effective_rank,
        },
        "pooled": {
            "btc_effective_n_stride4": float(btc["effective_n_stride4"]) if btc is not None else None,
            "naive_sum_stride4": naive_sum_s4, "naive_sum_stride1": naive_sum_s1,
            "corrected_stride4": corrected_s4, "corrected_stride1": corrected_s1,
        },
    }
    (OUT_DIR / "g4b_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(OUT_DIR / "label_agreement_matrix.npy", agree)
    print(f"\nwrote {OUT_DIR} and per-asset labels to {LABEL_OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
