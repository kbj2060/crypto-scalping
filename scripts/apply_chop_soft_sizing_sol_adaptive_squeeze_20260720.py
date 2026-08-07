"""Apply ETH's chop-soft-sizing shadow rule (trading_bot_modules/omega4_6_1_chop_shadow.py:
shadow_notional = real_notional * max(0, 1 - chop_prob)) to the adaptive-squeeze SOL candidate's
scale-map ledger (1.0x multiplier, no extra leverage). Since cost is proportional to notional in
this project's convention (same as the ETH shadow's own docstring), shadow_trade_return =
trade_return * max(0, 1 - chop_prob) exactly -- a ledger-level rescale, not a re-replay, matching
how the ETH shadow itself only rescales the realized move, never changes exit timing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import apply_final_scale_map_sol_20260707 as sm  # noqa: E402

sm.omega.TRAIN_CSV = ROOT / "data/splits/year_oos_adaptive_squeeze_sol_20260720/sol_features_2025.csv"
sm.omega.EVAL_CSV = ROOT / "data/splits/year_oos_adaptive_squeeze_sol_20260720/sol_features_2026.csv"

LEDGER_DIR = ROOT / "tmp/causal_regen_20260516/sol_final_scale_map_adaptive_squeeze_20260720"
DIRECTION_LABEL_DIR = ROOT / "tmp/causal_regen_20260516/sol_zigzag_action_labels_20260707"
CHOP_COL = "regime3_current_sensitive_wide24_chop_prob"


def _compound(returns: pd.Series) -> dict:
    cash, peak, mdd = 1.0, 1.0, 0.0
    for r in returns.to_numpy(dtype=np.float64):
        cash *= 1.0 + float(r)
        peak = max(peak, cash)
        mdd = min(mdd, (cash - peak) / peak)
    return {"pnl": (cash - 1.0) * 100.0, "mdd": mdd * 100.0, "trades": int(len(returns))}


def main() -> int:
    frames = sm.omega4._prepare_frames(
        disable_tp_sl=False, direction_label_dir=DIRECTION_LABEL_DIR,
        quality_mode="same_as_direction", quality_label_dir=None,
        quality_min_edge=0.0, quality_max_mae=0.0, quality_min_mfe_mae=0.0, quality_max_hold_bars=0,
    )
    for split, frame_key in (("validation", "val_raw"), ("oos", "oos_raw")):
        ledger = pd.read_csv(LEDGER_DIR / f"{split}_ledger.csv", parse_dates=["entry_timestamp"])
        feat = frames[frame_key][["timestamp", CHOP_COL]].rename(columns={"timestamp": "entry_timestamp"})
        merged = ledger.merge(feat, on="entry_timestamp", how="left", validate="one_to_one")
        if merged[CHOP_COL].isna().any():
            raise RuntimeError(f"{split}: {int(merged[CHOP_COL].isna().sum())} trades failed to match a chop_prob")
        shadow_mult = np.maximum(0.0, 1.0 - merged[CHOP_COL].to_numpy(dtype=np.float64))
        merged["shadow_trade_return"] = merged["trade_return"].to_numpy(dtype=np.float64) * shadow_mult
        real_m = _compound(merged["trade_return"])
        shadow_m = _compound(merged["shadow_trade_return"])
        print(f"=== {split} ===", flush=True)
        print(f"  real   (no chop sizing): {real_m}", flush=True)
        print(f"  shadow (chop soft-sized): {shadow_m}", flush=True)
        print(f"  mean shadow_mult: {shadow_mult.mean():.4f}, min: {shadow_mult.min():.4f}, max: {shadow_mult.max():.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
