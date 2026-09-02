#!/usr/bin/env python3
"""Can the evidence signals hold LONGER for more PnL? -- user 2026-09-02: "좀 길게 먹으면 훨씬 더
큰 pnl을 먹을거 같은데".

WHAT THE EXISTING EVIDENCE ALREADY SAYS (and why this is still worth running)
  * Looser trail is ALREADY answered: the grid held 0.1/0.2/0.3/0.5 and all five signals picked
    0.1, the TIGHTEST. Letting winners run by loosening the trail loses.
  * Bigger fixed TP is ALREADY answered: 2026-08-30 found it degenerates -- the TP becomes
    unreachable and the time-exit does the work instead.
  * ⭐HORIZON HAS NEVER BEEN VARIED. Each signal's exit horizon was pinned to its metalabel's own
    HORIZON and never swept. short_term_return_z exits 55 of 413 OOS trades (13%) on `timeout`, so
    the cap demonstrably binds. That is the untested axis behind the user's intuition.
  * ⚠️GRID-BOUNDARY HITS (README ss5.6 "never trust a grid boundary"): SL's ceiling 4.0 was chosen
    by orthogonal_combo and smt_divergence, ARM's ceiling 2.0 by liquidity_sweep and smt, and
    Trail's floor 0.1 by ALL FIVE. Three of the three axes are pinned at an edge, which by this
    repo's own rule means the grid was too narrow to locate the optimum.

DESIGN. Expand every boundary that was hit AND add the horizon axis, on the top-2 signals:
    horizon x{1, 2, 4, 8} of each signal's own H   (str_z 12 -> 12/24/48/96, sweep 30 -> 30/60/120/240)
    SL    3.0 4.0 5.0 6.0      (was capped at 4.0)
    ARM   1.0 2.0 3.0          (was capped at 2.0)
    Trail 0.05 0.1 0.2 0.5     (0.05 added below the old floor)
Selection on VAL ONLY, evaluation on OOS -- same discipline as the selection-bias study, so a
"longer horizon wins" claim cannot be an artifact of peeking at OOS. Direction-flip control on the
chosen config. HOLDOUT excluded.
"""
from __future__ import annotations

import json
import sys
import warnings

warnings.filterwarnings("ignore")
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import pandas as pd

from core.causal_futures_backtest import purged_decision_mask, simulate_single_position  # noqa: E402
from research_eth_regime_gated_costgate_ensemble_20260902 import (  # noqa: E402
    HOLDOUT_START, KLINES_PATH, LEVERAGE, MARGIN_FRACTION, OOS_START, ROUNDTRIP_COST_RATE,
    SIGNALS, VAL_START,
)

TARGETS = ["short_term_return_z", "liquidity_sweep"]
H_MULTS = [1, 2, 4, 8]
SL_GRID = [3.0, 4.0, 5.0, 6.0]
ARM_GRID = [1.0, 2.0, 3.0]
TRAIL_GRID = [0.05, 0.1, 0.2, 0.5]
OUT_DIR = ROOT / "tmp/eth_exit_horizon_boundary_20260902"


def log(m: str) -> None:
    print(f"[exit_horizon] {m}", flush=True)


def sim(ts, o, h, l, c, dec, sc, atr, horizon, sl, arm, tr):
    res = simulate_single_position(
        timestamps=ts, open_px=o, high=h, low=l, close=c, decision_indices=dec, scores=sc,
        tp_moves=np.full(len(dec), 999.0), sl_moves=sl * atr, upper_threshold=1.0,
        lower_threshold=-1.0, horizon_bars=horizon, margin_fraction=MARGIN_FRACTION,
        leverage=LEVERAGE, roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
        arm_moves=arm * atr, trail_moves=tr * atr)
    L = res.ledger
    if L.empty:
        return None
    r = L["trade_return"].to_numpy()
    w, ls = r[r > 0].sum(), -r[r < 0].sum()
    return {"n": int(len(r)), "mean_bp": float(r.mean() * 1e4), "total_bp": float(r.sum() * 1e4),
            "pf": float(w / ls) if ls > 0 else float("inf"),
            "profit_wr": float((r > 0).mean()),
            "bars_med": float(L["bars_held"].median()),
            "timeout_share": float((L["reason"] == "timeout").mean())}


def main() -> int:
    kl = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    ts = kl["timestamp"]
    o, h, l, c = (kl[k].to_numpy() for k in ("open", "high", "low", "close"))
    out = {}
    for name in TARGETS:
        cfg = SIGNALS[name]
        H0 = cfg["horizon"]
        f = pd.read_csv(ROOT / cfg["fires"], parse_dates=["timestamp"])
        f = f.loc[f["timestamp"] < HOLDOUT_START].sort_values("pos").reset_index(drop=True)
        dec = f["pos"].to_numpy(np.int64)
        sc = np.where(f["side"].to_numpy() == "bottom", 1.0, -1.0)
        atr = f["atr_pct"].to_numpy(float)
        log(f"\n=== {name} (base H={H0}, {len(f)} fires) ===")
        rows = []
        for hm in H_MULTS:
            H = H0 * hm
            ev = purged_decision_mask(ts, start=VAL_START, end=OOS_START, horizon_bars=H)
            eo = purged_decision_mask(ts, start=OOS_START, end=HOLDOUT_START, horizon_bars=H)
            vs, os_ = set(np.flatnonzero(ev).tolist()), set(np.flatnonzero(eo).tolist())
            vm = np.array([d in vs for d in dec]); om = np.array([d in os_ for d in dec])
            for sl in SL_GRID:
                for arm in ARM_GRID:
                    for tr in TRAIL_GRID:
                        v = sim(ts, o, h, l, c, dec[vm], sc[vm], atr[vm], H, sl, arm, tr)
                        if v is None:
                            continue
                        oo = sim(ts, o, h, l, c, dec[om], sc[om], atr[om], H, sl, arm, tr)
                        fv = sim(ts, o, h, l, c, dec[vm], -sc[vm], atr[vm], H, sl, arm, tr)
                        rows.append({"h_mult": hm, "H": H, "sl": sl, "arm": arm, "trail": tr,
                                     "val_mean": v["mean_bp"], "val_total": v["total_bp"],
                                     "val_pf": v["pf"], "val_n": v["n"],
                                     "val_flip_mean": fv["mean_bp"] if fv else np.nan,
                                     "oos_mean": oo["mean_bp"] if oo else np.nan,
                                     "oos_total": oo["total_bp"] if oo else np.nan,
                                     "oos_pf": oo["pf"] if oo else np.nan,
                                     "oos_n": oo["n"] if oo else 0,
                                     "oos_bars_med": oo["bars_med"] if oo else np.nan,
                                     "oos_timeout": oo["timeout_share"] if oo else np.nan})
        df = pd.DataFrame(rows)
        # VAL-only selection: positive, beats its own flip, flip negative -> max VAL mean bp
        elig = df[(df.val_mean > 0) & (df.val_mean > df.val_flip_mean) & (df.val_flip_mean < 0)]
        best_per_h = {}
        for hm in H_MULTS:
            sub = elig[elig.h_mult == hm]
            if sub.empty:
                log(f"  x{hm} (H={H0*hm:3d}): VAL 통과 조합 없음"); continue
            b = sub.loc[sub.val_mean.idxmax()]
            best_per_h[hm] = b
            log(f"  x{hm} (H={H0*hm:3d}): VAL선택 SL{b.sl}/ARM{b.arm}/Tr{b.trail} "
                f"VAL {b.val_mean:+6.2f}bp -> OOS {b.oos_mean:+6.2f}bp PF {b.oos_pf:.2f} "
                f"n={int(b.oos_n)} 총 {b.oos_total:+8.1f} | 보유중앙 {b.oos_bars_med:.0f}봉 "
                f"timeout {b.oos_timeout:.1%}")
        gb = elig.loc[elig.val_mean.idxmax()] if not elig.empty else None
        if gb is not None:
            log(f"  ⭐전체 VAL 최적: h_mult=x{int(gb.h_mult)} (H={int(gb.H)}) "
                f"SL{gb.sl}/ARM{gb.arm}/Tr{gb.trail} -> OOS {gb.oos_mean:+.2f}bp 총 {gb.oos_total:+.1f}")
        out[name] = {"base_H": H0,
                     "per_h": {str(k): {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv)
                                        for kk, vv in v.items()} for k, v in best_per_h.items()},
                     "global_best_h_mult": int(gb.h_mult) if gb is not None else None}
        df.to_csv(OUT_DIR / f"{name}_grid.csv", index=False) if OUT_DIR.exists() else None
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_DIR / f"{name}_grid.csv", index=False)
    (OUT_DIR / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    log(f"\nWrote {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
