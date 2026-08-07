"""BTC 5m: zero-shot Chronos (Amazon time-series foundation model) forecast as an additional
LightGBM feature for Layer B -- per literature review recommendation #3 (foundation models as an
auxiliary feature source, same spirit as the DVOL/on-chain axis additions earlier this session,
not as a replacement predictor). Reuses this project's own established Chronos usage pattern
(scripts/test_chronos_multiseries_standalone_20260530.py::_chronos_quantiles), retargeted to
log_close on the 5m BTC panel with a stride that keeps compute tractable over the full 2024-2026
history (computed every STRIDE bars, forward-filled between -- causal, since only past context is
used and the forecast is available immediately after computation).

Model: amazon/chronos-bolt-tiny (cached locally, fast bolt architecture, offline).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
OUT_PATH = ROOT / "data/splits/year_oos/btc_5m_chronos_feature_20260806.parquet"

MODEL_ID = "amazon/chronos-2"
CONTEXT_LENGTH = 256
PREDICTION_LENGTH = 6
STRIDE = 12  # hourly cadence
BATCH_SIZE = 256


def main() -> int:
    from chronos import BaseChronosPipeline

    panel = pd.read_parquet(PANEL_PATH, columns=["timestamp", "close"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    log_close = np.log(panel["close"].to_numpy(dtype=np.float32))
    n = len(log_close)

    pipe = BaseChronosPipeline.from_pretrained(
        MODEL_ID,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    idx = np.arange(CONTEXT_LENGTH, n, STRIDE, dtype=np.int64)
    if idx.size == 0 or idx[-1] != n - 1:
        idx = np.append(idx, n - 1)

    cols = ["chronos_q10_delta", "chronos_q50_delta", "chronos_q90_delta", "chronos_width", "chronos_mean_delta"]
    out = pd.DataFrame(np.nan, index=np.arange(n), columns=cols, dtype="float32")
    qlevels = [0.1, 0.5, 0.9]

    with torch.no_grad():
        for start in range(0, len(idx), BATCH_SIZE):
            batch_idx = idx[start:start + BATCH_SIZE]
            windows = [torch.as_tensor(log_close[i - CONTEXT_LENGTH:i], dtype=torch.float32) for i in batch_idx]
            quantiles, mean = pipe.predict_quantiles(windows, prediction_length=PREDICTION_LENGTH, quantile_levels=qlevels)
            # chronos-2 returns a list of per-series tensors, each (1, prediction_length, n_quantiles)
            q = np.stack([t[0, -1, :].float().cpu().numpy() for t in quantiles], axis=0)  # (batch, n_quantiles)
            m = np.stack([t[0, -1].float().cpu().numpy() for t in mean], axis=0)  # (batch,)
            cur = log_close[batch_idx]
            vals = np.column_stack([q[:, 0] - cur, q[:, 1] - cur, q[:, 2] - cur, q[:, 2] - q[:, 0], m - cur])
            out.loc[batch_idx, cols] = vals.astype("float32")
            if start % (BATCH_SIZE * 10) == 0:
                print(f"chronos progress: {start}/{len(idx)}")

    out[cols] = out[cols].ffill().fillna(0.0)
    result = pd.concat([panel[["timestamp"]].reset_index(drop=True), out.reset_index(drop=True)], axis=1)
    result.to_parquet(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH}, shape={result.shape}")
    print(result[cols].describe().T.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
