"""Stage 2 (Rho1 panel design) step 1: cross-sectional rank labels.

For every 5m timestamp, compute each panel symbol's forward H-bar log return, then the
CAUSAL-AT-LABEL-TIME cross-sectional percentile rank of that return among symbols that are
currently listed (a symbol not yet listed at time t contributes no rank competition for t --
excluding it, not treating it as rank 0, avoids biasing early ranks upward before the panel is
full size).

Design doc Layer 2(B): rather than predicting BTC's own direction, predict where BTC's forward
return will land in the cross-sectional distribution ("is BTC top/bottom-k right now"), which
literature (TSFM-in-finance papers cited in the design doc) and this project's own experience
suggest is a more stable target than raw direction.

Output: long-format parquet (timestamp, symbol, fwd_ret, rank_pct, n_ranked) for all 60 symbols,
used to train the ranking head (train_rho1_ranking_head_20260804.py) pooled the same way Stage 1
pooled the quantile head.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_PATH = ROOT / "data/splits/panel_universe_symbols_20260804.json"
FEATURES_DIR = ROOT / "data/panel/features"
OUT_PATH = ROOT / "data/panel/rank_labels_20260804.parquet"

HORIZON_H = 48  # matches train_rho1_panel_backbone_20260804.py's quantile-head horizon


def main() -> int:
    universe = json.loads(UNIVERSE_PATH.read_text())
    symbols = [row["symbol"] for row in universe["symbols"]]

    t0 = time.time()
    opens = {}
    closes = {}
    for sym in symbols:
        df = pd.read_parquet(FEATURES_DIR / f"{sym}.parquet", columns=["timestamp", "open", "close"])
        aligned = df.set_index("timestamp")
        opens[sym] = aligned["open"]
        closes[sym] = aligned["close"]
    open_df = pd.DataFrame(opens).sort_index()
    close_df = pd.DataFrame(closes).sort_index()
    print(f"loaded+aligned {len(symbols)} symbols in {time.time()-t0:.1f}s, shape={close_df.shape}", flush=True)

    fwd_ret = np.log(close_df.shift(-HORIZON_H) / open_df.shift(-1))

    n_ranked = fwd_ret.notna().sum(axis=1)
    rank_pct = fwd_ret.rank(axis=1, pct=True, na_option="keep")

    long_rows = []
    for sym in symbols:
        sub = pd.DataFrame({
            "timestamp": close_df.index,
            "symbol": sym,
            "fwd_ret": fwd_ret[sym].to_numpy(),
            "rank_pct": rank_pct[sym].to_numpy(),
            "n_ranked": n_ranked.to_numpy(),
        })
        long_rows.append(sub)
    out = pd.concat(long_rows, ignore_index=True)
    out = out.dropna(subset=["fwd_ret", "rank_pct"])
    # Require a reasonably full panel at label time so early ranks (when few symbols were listed)
    # don't dominate -- with 60 symbols, require at least 45 ranked to keep a bar's label.
    out = out[out["n_ranked"] >= 45]

    out.to_parquet(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH}, {len(out)} rows", flush=True)
    print(out.groupby("symbol").size().describe())
    print("\nBTCUSDT rank_pct describe:")
    print(out[out["symbol"] == "BTCUSDT"]["rank_pct"].describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
