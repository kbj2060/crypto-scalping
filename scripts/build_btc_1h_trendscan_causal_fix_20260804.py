"""
FIX for confirmed lookahead bug in scripts/build_1h_trendscan_dataset_btc_full_20260801.py's
_trend_scan_numpy: sliding_window_view(values, L)[r] = values[r:r+L], but the original code
assigned this result to out_t[r] -- i.e. row r's trend value used bars r..r+L-1 (up to L-1 bars
INTO THE FUTURE relative to row r), confirmed empirically (exact match to forward-window
recomputation, mismatch to backward/causal window) on 2026-08-04.

This script recomputes ts_action/ts_t_value/ts_opt_L CAUSALLY: window[r] = values[r-L+1:r+1]
(ending AT r, using only bars up to and including r), assigned to out_t[r]. Rebuilds the full
1h feature parquet + re-merges into the 5m execution frame.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_1h_trendscan_dataset_btc_full_20260801 import resample_1h, compute_features  # noqa: E402

SOURCES = [ROOT / f"data/splits/year_oos/btc_features_{y}.csv" for y in (2024, 2025, 2026)]
TS_WINDOWS = [3, 6, 12, 24, 36, 48]
TS_THRESHOLD = 2.5
OUT_PATH = ROOT / "data/splits/year_oos/btc_1h_trendscan_causal_2024_2026.parquet"


def _trend_scan_causal(values: np.ndarray, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CAUSAL version: out_t[r] uses only values[r-L+1 : r+1] (ending at r), for each candidate L,
    keeping whichever L gives the largest |t| (same tie-break as the original: first/smallest
    window wins on an exact tie, windows iterated ascending)."""
    n = len(values)
    out_t = np.zeros(n, dtype=np.float64)
    out_l = np.full(n, -1, dtype=np.int32)
    out_beta = np.zeros(n, dtype=np.float64)
    finite = np.isfinite(values)
    for L in sorted(int(w) for w in windows if int(w) > 2):
        n_valid = n - L + 1
        if n_valid <= 0:
            continue
        win = np.lib.stride_tricks.sliding_window_view(values, L)[:n_valid]  # win[j] = values[j:j+L]
        ok = np.lib.stride_tricks.sliding_window_view(finite, L)[:n_valid].all(axis=1)
        mean_x = (L - 1) / 2.0
        var_x_sum = L * (L * L - 1.0) / 12.0
        k_centered = np.arange(L, dtype=np.float64) - mean_x
        mean_y = win.mean(axis=1)
        cov_xy = win @ k_centered
        beta = cov_xy / var_x_sum
        alpha = mean_y - beta * mean_x
        pred = alpha[:, None] + beta[:, None] * np.arange(L, dtype=np.float64)[None, :]
        rss = np.square(win - pred).sum(axis=1)
        se_beta = np.sqrt(np.maximum(rss, 0.0) / (L - 2.0)) / np.sqrt(var_x_sum)
        t_val = np.where((rss > 1e-12) & (se_beta > 1e-12), beta / np.where(se_beta > 1e-12, se_beta, 1.0), 0.0)
        t_val = np.where(ok, t_val, 0.0)
        # CAUSAL FIX: win[j] = values[j:j+L] ends at index j+L-1 -- assign to out index j+L-1, not j.
        dest = np.arange(L - 1, L - 1 + n_valid)
        improve = np.abs(t_val) > np.abs(out_t[dest])
        out_t[dest] = np.where(improve, t_val, out_t[dest])
        out_l[dest] = np.where(improve, L, out_l[dest])
        out_beta[dest] = np.where(improve, beta, out_beta[dest])
    return out_t, out_l, out_beta


def main():
    src = pd.concat([pd.read_csv(p, usecols=["timestamp", "open", "high", "low", "close", "volume",
                                              "quote_volume", "taker_buy_base", "last_funding_rate",
                                              "sum_open_interest_value", "sum_toptrader_long_short_ratio"])
                      for p in SOURCES], ignore_index=True)
    src["timestamp"] = pd.to_datetime(src["timestamp"])
    src = src.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    r_full = resample_1h(src)
    feats_full = compute_features(r_full)

    logc = np.log(np.maximum(feats_full["close"].to_numpy(dtype=np.float64), 1e-12))
    win = np.array(sorted(TS_WINDOWS), dtype=np.int32)
    t_vals, opt_l, betas = _trend_scan_causal(logc, win)
    labels = np.zeros(len(feats_full), dtype=np.int64)
    labels[(np.abs(t_vals) >= TS_THRESHOLD) & (betas > 0)] = 1
    labels[(np.abs(t_vals) >= TS_THRESHOLD) & (betas < 0)] = 2
    feats_full["ts_action"] = labels
    feats_full["ts_t_value"] = t_vals.astype(np.float32)
    feats_full["ts_opt_L"] = opt_l.astype(np.int16)

    # empirical causality self-check before writing anything
    close_np = feats_full["close"].to_numpy(dtype=np.float64)
    logc2 = np.log(np.maximum(close_np, 1e-12))
    for r in [1000, 5000, 8000]:
        L = int(feats_full["ts_opt_L"].iloc[r])
        if L <= 0:
            continue
        bwd = logc2[max(0, r - L + 1):r + 1]
        fwd = logc2[r:r + L]

        def ols_t(w):
            n = len(w)
            mean_x = (n - 1) / 2.0
            var_x_sum = n * (n * n - 1.0) / 12.0
            k = np.arange(n, dtype=np.float64) - mean_x
            cov_xy = w @ k
            beta = cov_xy / var_x_sum
            alpha = w.mean() - beta * mean_x
            pred = alpha + beta * np.arange(n, dtype=np.float64)
            rss = np.sum((w - pred) ** 2)
            se = np.sqrt(max(rss, 0.0) / (n - 2.0)) / np.sqrt(var_x_sum)
            return beta / se if se > 1e-12 else 0.0

        stored = float(feats_full["ts_t_value"].iloc[r])
        t_bwd = ols_t(bwd) if len(bwd) == L else None
        t_fwd = ols_t(fwd) if len(fwd) == L else None
        match_bwd = t_bwd is not None and abs(t_bwd - stored) < 1e-4
        match_fwd = t_fwd is not None and abs(t_fwd - stored) < 1e-4
        print(f"self-check row={r} L={L} stored={stored:.4f} bwd={t_bwd} fwd={t_fwd} "
              f"match_backward(causal)={match_bwd} match_forward(leaky)={match_fwd}")
        if match_fwd and not match_bwd:
            raise SystemExit("CAUSALITY FIX FAILED -- still matches forward/leaky window")

    feats_full.to_parquet(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH} shape={feats_full.shape}")


if __name__ == "__main__":
    main()
