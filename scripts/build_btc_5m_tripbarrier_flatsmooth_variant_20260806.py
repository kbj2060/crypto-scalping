"""Flat epsilon-smoothed one-hot soft-target variant of the triple-barrier label
(build_btc_5m_tripbarrier_tradeoutcome_labels_20260806.py), which itself now writes a race-based
soft label (see that script's docstring). This flatsmooth variant is the one that actually
performed best across the accuracy/backtest tuning done in this session (val 40.2%/OOS 36.0%
accuracy, the checkpoint everything downstream -- cash_weight sweep, GBDT comparison, multi-task
ensemble -- builds on). Kept as a separate file/script rather than overwriting the race-soft label,
since both are valid artifacts referenced by different downstream scripts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "data/splits/year_oos/btc_5m_tripbarrier_tradeoutcome_labels_20260806.parquet"
OUT_PATH = ROOT / "data/splits/year_oos/btc_5m_tripbarrier_tradeoutcome_labels_flatsmooth_20260806.parquet"

LABEL_SMOOTH_EPS = 0.05


def main() -> int:
    lab = pd.read_parquet(SRC_PATH, columns=["timestamp", "trade_outcome_action"])
    n = len(lab)
    soft = np.full((n, 3), LABEL_SMOOTH_EPS / 2.0, dtype=np.float32)
    soft[np.arange(n), lab["trade_outcome_action"].to_numpy()] = 1.0 - LABEL_SMOOTH_EPS

    out = lab.copy()
    out["trade_outcome_soft_cash"] = soft[:, 0]
    out["trade_outcome_soft_long"] = soft[:, 1]
    out["trade_outcome_soft_short"] = soft[:, 2]
    out.to_parquet(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH}, shape={out.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
