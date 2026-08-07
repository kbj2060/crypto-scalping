"""Per-bar label SPAN for the causal triple-barrier trade-outcome label, needed by gate G2's
average-uniqueness sample weighting.

The 3-class label at bar i is only fully determined once BOTH races it describes have resolved --
LONG's own TP/SL race and SHORT's own TP/SL race -- because the label is a function of which of the
two (if either) reached TP first. So the label's event window is
``[i+1, i+max(long_t, short_t)]``, with an unresolved side counted as the full horizon.

Why this matters: uniqueness weighting is meaningless if every label is assumed to occupy the full
288-bar horizon, since then concurrency is near-constant and every sample gets the same weight. The
actual spans are highly variable (a bar whose barriers resolve in 3 bars overlaps far fewer
neighbours than one that runs the full day), and that variation is exactly what the weighting is
supposed to exploit.

Barrier construction (12-bar cumret dispersion over a 288-bar causal lookback, TP_MULT=2.5,
SL_MULT=1.2, horizon=288) is copied verbatim from
scripts/build_btc_5m_tripbarrier_tradeoutcome_labels_20260806.py so the spans describe the exact
label the model trains on.
"""
from __future__ import annotations

import json
from pathlib import Path

import numba
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
LABEL_PATH = ROOT / "data/splits/year_oos/btc_5m_tripbarrier_tradeoutcome_labels_flatsmooth_20260806.parquet"
OUT_PATH = ROOT / "data/splits/year_oos/btc_5m_tripbarrier_label_span_20260807.parquet"

CUMRET_BARS, VOL_LOOKBACK, TP_MULT, SL_MULT, HORIZON_BARS = 12, 288, 2.5, 1.2, 288


@numba.njit(cache=True)
def _label_span(open_, high, low, tp_move, sl_move, horizon):
    n = len(open_)
    span = np.full(n, horizon, dtype=np.int32)
    for i in range(n - 1):
        entry_i = i + 1
        if not np.isfinite(tp_move[i]) or not np.isfinite(sl_move[i]):
            continue
        entry = open_[entry_i]
        tp_l, sl_l = entry * (1.0 + tp_move[i]), entry * (1.0 - sl_move[i])
        tp_s, sl_s = entry * (1.0 - tp_move[i]), entry * (1.0 + sl_move[i])
        long_done, short_done = False, False
        long_t, short_t = horizon, horizon
        final_i = entry_i + horizon - 1
        if final_i >= n:
            final_i = n - 1
        for j in range(entry_i, final_i + 1):
            t = j - entry_i + 1
            if not long_done:
                if low[j] <= sl_l or high[j] >= tp_l:
                    long_done, long_t = True, t
            if not short_done:
                if high[j] >= sl_s or low[j] <= tp_s:
                    short_done, short_t = True, t
            if long_done and short_done:
                break
        span[i] = long_t if long_t > short_t else short_t
    return span


def main() -> int:
    panel = pd.read_parquet(PANEL_PATH, columns=["timestamp", "open", "high", "low", "close"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    labels = pd.read_parquet(LABEL_PATH, columns=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if not (panel["timestamp"].to_numpy() == labels["timestamp"].to_numpy()).all():
        raise RuntimeError("panel/label timestamp misalignment")

    close = panel["close"].to_numpy(dtype=np.float64)
    log_ret_1bar = np.diff(np.log(np.clip(close, 1e-9, None)), prepend=np.log(close[0]))
    cumret = pd.Series(log_ret_1bar).rolling(CUMRET_BARS).sum().to_numpy()
    vol = pd.Series(cumret).rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).std().to_numpy()

    span = _label_span(
        panel["open"].to_numpy(dtype=np.float64), panel["high"].to_numpy(dtype=np.float64),
        panel["low"].to_numpy(dtype=np.float64), TP_MULT * vol, SL_MULT * vol, HORIZON_BARS,
    )

    out = pd.DataFrame({"timestamp": panel["timestamp"], "label_span_bars": span})
    out.to_parquet(OUT_PATH, index=False)
    print(json.dumps({
        "rows": int(len(span)),
        "horizon_bars": HORIZON_BARS,
        "span_median": float(np.median(span)),
        "span_mean": float(span.mean()),
        "span_p10": float(np.percentile(span, 10)),
        "span_p90": float(np.percentile(span, 90)),
        "share_at_full_horizon": float((span >= HORIZON_BARS).mean()),
    }, indent=2))
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
