"""Gate G3 -- does sampling decisions at causal zigzag wave boundaries raise the effective sample
size above the 4,058 measured in gate G2?

The G2 finding was that the triple-barrier label's overlapping windows (span median 51 / mean 77 /
p90 189 bars) leave ~4,058 independent observations behind a nominal 43,798 training rows. The
proposed fix was to stop deciding on every bar and instead decide only where the market structure
says something changed -- at confirmed zigzag pivots.

Two controls make this an actual test rather than a demonstration that sparser sampling is sparser:

1. **Uniform-stride controls at matched sample counts.** Any sparsification raises per-sample
   uniqueness; the question is whether zigzag's event TIMING beats a dumb uniform grid holding the
   sample count fixed. If it does not, the "event sampler" is just `train_stride` with extra steps.
2. **The analytic ceiling.** If every label spanned exactly L bars and samples were spread evenly,
   the sum of uniqueness collapses to n_bars / L regardless of how many samples are drawn -- i.e.
   effective sample size is capped by the DATA, not by the sampling scheme, and sparsifying only
   stops double-counting rather than creating information. The measured curve is printed against
   that reference so the plateau (if any) is visible.

Events come from the causal pivot tracker in
scripts/build_btc_5m_zigzag_state_causal_features_20260806.py -- a decision fires on the bar where
the tracker's trend state FLIPS, which is the bar a reversal is confirmed in real time. The pivot
itself sits in the past at that moment; using the pivot bar would be lookahead.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numba
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from build_btc_5m_zigzag_state_causal_features_20260806 import _causal_zigzag_state  # noqa: E402

PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
SPAN_PATH = ROOT / "data/splits/year_oos/btc_5m_tripbarrier_label_span_20260807.parquet"
LABEL_PATH = ROOT / "data/splits/year_oos/btc_5m_tripbarrier_tradeoutcome_labels_flatsmooth_20260806.parquet"
OUT_DIR = ROOT / "tmp/btc_gate_g3_zigzag_event_sampler_20260807"

CUMRET_BARS, VOL_LOOKBACK = 12, 288
MIN_REVERSAL_PCT = 0.009
WINDOW = 48  # feature window, matches the deepfeat config -- rows before this can't be sampled
VAL_START = pd.Timestamp("2025-09-01")
STRIDES = (1, 4, 16, 32, 51, 96, 128, 256)
VOL_MULTIPLIERS = (0.5, 1.0, 2.0)
HOLD_EVERY = (0, 48, 96)  # 0 = flip bars only; else also sample every M bars inside a wave


@numba.njit(cache=True)
def _events_from_trend(trend_state, hold_every):
    """Bars where the causal tracker's trend flips (a reversal confirms), optionally plus a sample
    every `hold_every` bars while the same trend persists."""
    n = len(trend_state)
    out = np.zeros(n, dtype=np.bool_)
    last_event = -1
    prev = trend_state[0]
    for i in range(1, n):
        cur = trend_state[i]
        if cur != prev:
            out[i] = True
            last_event = i
        elif hold_every > 0 and last_event >= 0 and i - last_event >= hold_every:
            out[i] = True
            last_event = i
        prev = cur
    return out


def _uniqueness(sample_idx, label_end, n_bars):
    """Average uniqueness per sample: mean of 1/concurrency over the bars each label window covers,
    with concurrency counted over this sample set only."""
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
    span_len = np.maximum(ends - starts + 1, 1)
    return (prefix[ends + 1] - prefix[starts]) / span_len


def _describe(name, sample_idx, label_end, n_bars, actions, spans):
    u = _uniqueness(sample_idx, label_end, n_bars)
    if len(sample_idx) == 0:
        return {"sampler": name, "n_samples": 0}
    gaps = np.diff(sample_idx)
    act = actions[sample_idx]
    return {
        "sampler": name,
        "n_samples": int(len(sample_idx)),
        "mean_uniqueness": float(u.mean()),
        "effective_n": float(u.sum()),
        "median_gap_bars": float(np.median(gaps)) if len(gaps) else float("nan"),
        "mean_span_at_samples": float(spans[sample_idx].mean()),
        "cash_share": float((act == 0).mean()),
        "long_share": float((act == 1).mean()),
        "short_share": float((act == 2).mean()),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(PANEL_PATH, columns=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)
    spans_df = pd.read_parquet(SPAN_PATH).sort_values("timestamp").reset_index(drop=True)
    lab = pd.read_parquet(LABEL_PATH, columns=["timestamp", "trade_outcome_action"]).sort_values("timestamp").reset_index(drop=True)
    ts = panel["timestamp"].to_numpy()
    if not ((spans_df["timestamp"].to_numpy() == ts).all() and (lab["timestamp"].to_numpy() == ts).all()):
        raise RuntimeError("timestamp misalignment between panel / spans / labels")

    close = panel["close"].to_numpy(dtype=np.float64)
    spans = spans_df["label_span_bars"].to_numpy(dtype=np.int64)
    actions = lab["trade_outcome_action"].to_numpy()
    n_bars = len(ts)
    label_end = np.arange(n_bars, dtype=np.int64) + spans

    val_start_i = int(np.searchsorted(ts, np.datetime64(VAL_START)))
    train_bars = np.arange(WINDOW - 1, val_start_i, dtype=np.int64)
    n_train_bars = len(train_bars)

    log_ret = np.diff(np.log(np.clip(close, 1e-9, None)), prepend=np.log(close[0]))
    cumret = pd.Series(log_ret).rolling(CUMRET_BARS).sum().to_numpy()
    vol = pd.Series(cumret).rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).std().to_numpy()

    rows = []
    for s in STRIDES:
        rows.append(_describe(f"uniform_stride{s}", train_bars[::s], label_end, n_bars, actions, spans))

    event_masks = {}
    for vm in VOL_MULTIPLIERS:
        thr = np.where(np.isfinite(vol), np.maximum(MIN_REVERSAL_PCT, vm * vol), MIN_REVERSAL_PCT)
        trend_state, _, _, _ = _causal_zigzag_state(close, thr)
        for hold in HOLD_EVERY:
            mask = _events_from_trend(trend_state, hold)
            idx = np.flatnonzero(mask)
            idx = idx[(idx >= WINDOW - 1) & (idx < val_start_i)]
            name = f"zz_event_vm{vm}" + (f"_hold{hold}" if hold else "")
            event_masks[name] = idx
            rows.append(_describe(name, idx, label_end, n_bars, actions, spans))

    # matched-count uniform controls: same n_samples as each event sampler, evenly spaced
    for name, idx in event_masks.items():
        if len(idx) < 2:
            continue
        step = max(1, n_train_bars // len(idx))
        ctrl = train_bars[::step][: len(idx)]
        rows.append(_describe(f"CTRL_uniform_matched_{name}", ctrl, label_end, n_bars, actions, spans))

    df = pd.DataFrame(rows)
    train_span_mean = float(spans[train_bars].mean())
    analytic_cap = n_train_bars / train_span_mean

    hdr = (f"{'sampler':<34}{'n':>8}{'mean_uniq':>11}{'effective_n':>13}{'gap':>7}"
           f"{'span@s':>8}{'CASH%':>7}{'LONG%':>7}{'SHORT%':>7}")
    print(f"train bars={n_train_bars}  mean label span={train_span_mean:.1f}")
    print(f"analytic effective-n reference (n_bars / mean_span) = {analytic_cap:.0f}")
    print(f"G2 baseline (uniform_stride4) effective_n was 4,058 -- gate asks for >= 3x = 12,174\n")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if not r.get("n_samples"):
            continue
        print(f"{r['sampler']:<34}{r['n_samples']:>8}{r['mean_uniqueness']:>11.4f}{r['effective_n']:>13.0f}"
              f"{r['median_gap_bars']:>7.0f}{r['mean_span_at_samples']:>8.0f}{r['cash_share']*100:>7.1f}"
              f"{r['long_share']*100:>7.1f}{r['short_share']*100:>7.1f}")

    payload = {
        "train_bars": int(n_train_bars), "train_mean_label_span": train_span_mean,
        "analytic_effective_n_reference": analytic_cap,
        "g2_baseline_effective_n": 4058.0, "gate_threshold_3x": 3 * 4058.0,
        "config": {"min_reversal_pct": MIN_REVERSAL_PCT, "vol_multipliers": list(VOL_MULTIPLIERS),
                   "hold_every": list(HOLD_EVERY), "window": WINDOW},
        "samplers": rows,
    }
    (OUT_DIR / "g3_sampler_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    df.to_csv(OUT_DIR / "g3_samplers.csv", index=False)
    print(f"\nwrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
