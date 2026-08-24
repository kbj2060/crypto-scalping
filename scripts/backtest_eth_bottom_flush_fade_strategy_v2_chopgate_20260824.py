#!/usr/bin/env python3
"""v2 = v1 frozen + chop-regime gate (wide24 HMM argmax). Pre-registered in the v1 doc
section 4.5. No recalibration, no threshold tuning -- pure gate isolation test with a
significance term in the verdict this time.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backtest_eth_bottom_flush_fade_strategy_v1_20260824 import (  # noqa: E402
    CAL, DEV, FRESH, COSTS_RT, H_HOLD, L_LIMIT, SL_FLOOR, load_frames, run_engine, report,
)

REGIME_DIR = ROOT / "tmp/ilias_labellogic_recheck_20260821"
REGIME_FILES = ["train_2024_2026H1_regime3_current_states24_sticky090.csv",
                "oos_20260701_20260819_regime3_current_states24_sticky090.csv"]
P = "regime3_current_sensitive_wide24_"


def main() -> None:
    f = load_frames()

    reg = pd.concat([pd.read_csv(REGIME_DIR / fn, parse_dates=["timestamp"]) for fn in REGIME_FILES],
                    ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    probs = reg[[f"{P}bull_prob", f"{P}bear_prob", f"{P}chop_prob"]].to_numpy()
    reg["regime3"] = np.array(["bull", "bear", "chop"])[probs.argmax(axis=1)]
    f = f.merge(reg[["timestamp", "regime3"]], on="timestamp", how="left")
    cov = f["regime3"].notna().mean()
    print(f"regime coverage on strategy frame: {cov * 100:.1f}% | "
          f"chop share: {(f['regime3'] == 'chop').mean() * 100:.1f}%")

    # reproduce v1 calibration exactly (ungated, deterministic), then freeze
    ts = f["timestamp"]
    cal_mask = (ts >= CAL[0]) & (ts <= CAL[1])
    trig_pos = np.flatnonzero((f["trigger"] & cal_mask).to_numpy())
    lows, closes, highs = f["low"].to_numpy(), f["close"].to_numpy(), f["high"].to_numpy()
    mae24 = np.array([(closes[t] - lows[t + 1:t + 1 + L_LIMIT].min()) / closes[t]
                      for t in trig_pos if t + 1 + L_LIMIT < len(f)])
    delta = float(np.median(mae24))
    probe = run_engine(f, *CAL, delta, 9.99, 9.99)
    mfe, mae = [], []
    for _, t in probe.iterrows():
        j0, j1 = int(t["fill_i"]) + 1, int(t["fill_i"]) + H_HOLD
        if j1 >= len(f):
            continue
        mfe.append(highs[j0:j1 + 1].max() / t["entry"] - 1.0)
        mae.append(1.0 - lows[j0:j1 + 1].min() / t["entry"])
    tp, sl = float(np.median(mfe)), max(float(np.percentile(mae, 75)), SL_FLOOR)
    print(f"v1 params (frozen): delta={delta * 100:.3f}% tp={tp * 100:.3f}% sl={sl * 100:.3f}%\n")

    # v2 gate: triggers valid only in chop
    f["trigger"] = (f["trigger"] & (f["regime3"] == "chop")).fillna(False)

    results = {}
    for name, (s, e) in (("CAL(in-sample)", CAL), ("DEV(consumed)", DEV), ("FRESH", FRESH)):
        tr = run_engine(f, s, e, delta, tp, sl)
        report(f"v2 {name}", f, s, e, tr)
        results[name] = tr

    # pre-registered verdict (with significance)
    cal_net = (results["CAL(in-sample)"]["gross"] - COSTS_RT["taker10bp"]).sum()
    out = pd.concat([results["DEV(consumed)"], results["FRESH"]], ignore_index=True)
    out_net = out["gross"] - COSTS_RT["taker10bp"]
    t_stat = out_net.mean() / (out_net.std(ddof=1) / np.sqrt(len(out_net))) if len(out_net) > 2 else float("nan")
    out["month"] = pd.to_datetime(out["ts"]).dt.to_period("M")
    monthly = out.groupby("month").apply(lambda g: (g["gross"] - COSTS_RT["taker10bp"]).sum(), include_groups=False)
    max_month_share = (monthly.max() / out_net.sum() * 100) if out_net.sum() > 0 else float("nan")
    c1 = cal_net > 0
    c2 = out_net.sum() > 0 and t_stat >= 1.5
    print(f"\nmonthly net@10bp (DEV+FRESH):\n{monthly.to_string()}")
    print(f"\nVERDICT v2: [1] CAL net@10bp={cal_net * 100:+.2f}% (>0): {c1} | "
          f"[2] out-of-CAL n={len(out)} net={out_net.sum() * 100:+.2f}% t={t_stat:.2f} (>0 & t>=1.5): {c2} | "
          f"[3] max-month share={max_month_share:.0f}%"
          f" -> {'PASS' if c1 and c2 else 'FAIL'}")


if __name__ == "__main__":
    main()
