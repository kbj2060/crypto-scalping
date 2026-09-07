#!/usr/bin/env python3
"""Two mandatory follow-ups after the exit-structure comparison found the incumbent cell was NOT
tight enough (ARM 0.25/Trail 0.05 beat ARM 0.5/Trail 0.1 by ~3x):

  1) BOUNDARY EXTENSION (docs/homer/README.md 5.6 -- "그리드 경계값 자체를 의심할 것"): ARM and
     Trail both sat at the tight edge of every grid run so far, and min(VAL,OOS) was still rising
     going tighter. Extend until an interior peak appears (or until the values stop being
     physically meaningful).
  2) OPTIMISTIC/PESSIMISTIC INTRABAR CROSS-CHECK (memory feedback_intrabar_ordering_optimistic_
     pessimistic_bracket_20260830 -- "트레일링폭 좁을수록 봉내순서 이중검증 필수, 직관과 반대"):
     the standard engine is PESSIMISTIC (stop checked before the favorable update). At a 0.05xATR
     trail the within-bar ordering assumption dominates, so resolve_optimistic() (reused verbatim
     from the fib_extension/liquidity_sweep crosscheck scripts) is run on the identical trades.
     A large divergence means the result is an artifact of the bar-resolution convention, not an
     edge -- this is the check that decides whether these tight cells are real.

Also reports the trail width in bp of price so the physical plausibility is visible.

VAL 2025-09..12 + OOS 2026-01..03. HOLDOUT (2026-04..08) NOT touched.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import pandas as pd

from backtest_eth_fib_extension_exhaustion_trailing_optimistic_crosscheck_20260831 import (  # noqa: E402
    resolve_optimistic,
)
from core.causal_futures_backtest import purged_decision_mask, simulate_single_position  # noqa: E402
from live_evidence_signal_dashboard_20260823 import SIGNAL_ORDER, compute_signals  # noqa: E402
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import build_indicator_frame  # noqa: E402

START = pd.Timestamp("2024-01-01")
VAL_START, OOS_START, HOLDOUT_START = (pd.Timestamp(x) for x in ("2025-09-01", "2026-01-01", "2026-04-01"))
GAP, H, SL = 12, 24, 3.5
MARGIN, LEV, COST = 0.30, 3.0, 0.001
ARM_GRID = [0.05, 0.10, 0.15, 0.25, 0.35, 0.50, 0.75]
TRAIL_GRID = [0.02, 0.03, 0.05, 0.08, 0.10, 0.15]
CROSSCHECK = [(0.25, 0.05), (0.25, 0.10), (0.50, 0.05), (0.50, 0.10), (0.10, 0.03), (0.15, 0.05)]


def load(name: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / f"binance_data/klines/{name}/{name}-5m-api.csv", parse_dates=["timestamp"])
    return df.loc[df["timestamp"] >= START - pd.Timedelta(days=10)].reset_index(drop=True)


def main() -> int:
    eth, btc = load("ETHUSDT"), load("BTCUSDT")
    sig = compute_signals(eth, btc, None)
    sig = sig.loc[sig["timestamp"] >= START].reset_index(drop=True)
    kl = eth.loc[eth["timestamp"] >= START].reset_index(drop=True)
    ind = build_indicator_frame(eth)
    ind = ind.loc[ind["timestamp"] >= START].reset_index(drop=True)

    bot = np.zeros(len(sig), bool); top = np.zeros(len(sig), bool)
    for n, _ in SIGNAL_ORDER:
        bot |= sig[f"bottom_{n}"].to_numpy(); top |= sig[f"top_{n}"].to_numpy()
    rows = []
    for side, m in (("bottom", bot), ("top", top)):
        last = -10**9
        for i in np.flatnonzero(m):
            if i - last < GAP:
                continue
            last = i; rows.append((i, side))
    ev = pd.DataFrame(rows, columns=["pos", "side"]).sort_values("pos").reset_index(drop=True)

    ts = kl["timestamp"]
    o, hi, lo, c = (kl[x].to_numpy() for x in ("open", "high", "low", "close"))
    atr_pct = ind["atr_pct"].to_numpy()
    dec = ev["pos"].to_numpy(np.int64)
    scores = np.where(ev["side"].to_numpy() == "bottom", -1.0, 1.0)
    masks = {}
    for w, (s, e) in (("val", (VAL_START, OOS_START)), ("oos", (OOS_START, HOLDOUT_START))):
        el = set(np.flatnonzero(purged_decision_mask(ts, start=s, end=e, horizon_bars=H)).tolist())
        masks[w] = np.array([d in el for d in dec])

    def pess(arm, trail, w):
        m = masks[w]; a = atr_pct[dec][m]
        r = simulate_single_position(
            timestamps=ts, open_px=o, high=hi, low=lo, close=c, decision_indices=dec[m],
            scores=scores[m], tp_moves=np.full(int(m.sum()), 999.0), sl_moves=SL * a,
            upper_threshold=1.0, lower_threshold=-1.0, horizon_bars=H, margin_fraction=MARGIN,
            leverage=LEV, roundtrip_cost_rate=COST, arm_moves=arm * a, trail_moves=trail * a)
        return r.ledger

    med_atr = float(np.nanmedian(atr_pct[dec]))
    print(f"median ATR at fire bars = {med_atr*100:.3f}%  -> 1.0xATR = {med_atr*1e4:.1f}bp of price\n")
    print("=== [1] boundary extension: min(VAL,OOS) bp, SL=3.5 fixed, H=24 ===")
    print(f"{'ARM\\Trail':<10}" + "".join(f"{t:>9}" for t in TRAIL_GRID))
    best = None
    grid = {}
    for arm in ARM_GRID:
        line = f"{arm:<10}"
        for trail in TRAIL_GRID:
            v = pess(arm, trail, "val")["trade_return"].mean() * 1e4
            oo = pess(arm, trail, "oos")["trade_return"].mean() * 1e4
            mb = min(v, oo)
            grid[(arm, trail)] = (v, oo)
            line += f"{mb:>9.2f}"
            if best is None or mb > best[0]:
                best = (mb, arm, trail)
        print(line)
    print(f"\nbest interior/edge cell: ARM={best[1]} Trail={best[2]}  min(VAL,OOS)={best[0]:+.2f}bp "
          f"(VAL {grid[(best[1],best[2])][0]:+.2f} / OOS {grid[(best[1],best[2])][1]:+.2f})")
    print(f"physical size: ARM={best[1]*med_atr*1e4:.1f}bp, Trail={best[2]*med_atr*1e4:.1f}bp of price "
          f"(round-trip cost is {COST*1e4:.0f}bp)")

    print("\n=== [2] optimistic vs pessimistic intrabar ordering (identical trades) ===")
    print(f"{'ARM':>6}{'Trail':>7}{'trail bp':>10}{'split':>6}{'pess bp':>10}{'opt bp':>10}"
          f"{'diverge':>10}{'pess win':>10}{'opt win':>9}")
    for arm, trail in CROSSCHECK:
        for w in ("val", "oos"):
            m = masks[w]; a = atr_pct[dec][m]
            L = pess(arm, trail, w)
            pos_of = pd.Series(np.arange(len(kl)), index=ts.to_numpy())
            fi = L["decision_timestamp"].map(pos_of).to_numpy()
            ei = fi + 1
            side_sign = L["score"].to_numpy()
            a_tr = atr_pct[fi]
            notional = MARGIN * LEV
            opt_moves = []
            for k in range(len(L)):
                s0, s1 = ei[k], min(ei[k] + H - 1, len(kl) - 1)
                pm, _, _ = resolve_optimistic(int(np.sign(side_sign[k])), o[ei[k]], hi[s0:s1 + 1],
                                              lo[s0:s1 + 1], c[s0:s1 + 1], SL * a_tr[k],
                                              arm * a_tr[k], trail * a_tr[k])
                opt_moves.append(pm)
            opt_moves = np.array(opt_moves)
            opt_bp = float((opt_moves * notional - COST * notional).mean() * 1e4)
            pess_bp = float(L["trade_return"].mean() * 1e4)
            print(f"{arm:>6}{trail:>7}{trail*med_atr*1e4:>10.1f}{w:>6}{pess_bp:>10.2f}{opt_bp:>10.2f}"
                  f"{opt_bp - pess_bp:>+10.2f}{(L['price_move']>0).mean():>10.1%}"
                  f"{(opt_moves>0).mean():>9.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
