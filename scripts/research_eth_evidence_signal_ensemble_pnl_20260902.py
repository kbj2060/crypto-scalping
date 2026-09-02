#!/usr/bin/env python3
"""PART B -- how should the evidence signals be ENSEMBLED, judged on PnL? 2026-09-02.

THE CONSTRAINT THAT DEFINES THE PROBLEM. trading_bot.py has no concurrent-position structure for a
symbol -- one slot. So "run all 7 signals" is not a portfolio, it is 7 overlapping backtests whose
trade counts do not add up to anything tradeable. This repo already hit that: the V-rebound
economics work found its earlier evaluation had allowed dozens of overlapping positions, inflating n,
and had to be redone as a SEQUENTIAL portfolio. Every arm here is therefore a 1-slot sequential
portfolio: walk bars forward, and only when flat may a signal take the slot.

PER-FIRE OUTCOMES WITHOUT RE-IMPLEMENTING THE EXIT. simulate_single_position() is deliberately
non-overlapping, so one call returns outcomes for only a non-overlapping subset of the fires. But a
trade's `trade_return` depends solely on (entry bar, sl/arm/trail, horizon, prices) -- it is
cash-independent (`trade_return = price_move*notional - account_cost`; cash is only compounded
afterwards). So calling it repeatedly on the fires it skipped, until none remain, yields an
INDEPENDENT outcome for every fire while reusing the validated exit logic verbatim. That table of
per-fire outcomes is what the portfolio rules then select from.

ARMS (all 1-slot sequential, all with a direction-flip control, all VAL/OOS split)
  single:<name>   -- each signal alone (the baseline each ensemble must beat)
  union_first     -- any signal may take a free slot; ties broken by the priority order
  pf_priority     -- same, but ties broken by each signal's own VAL profit-factor rank
  confluence2     -- only enter when >=2 signals fire the SAME side within CONFLUENCE_BARS
  chop_union      -- union, but only while the deployed S12_K3 regime model predicts chop
                     (this is the PnL version of the Phase-2/3b lift result)

⚠️ HOLDOUT excluded; all signals already spent theirs. VAL/OOS research/dev score only.
⚠️ fib_extension_exhaustion excluded (economics claim withdrawn 2026-09-01, 23.0pp cost-erosion gap).
"""
from __future__ import annotations

import json
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

from core.causal_futures_backtest import purged_decision_mask, simulate_single_position  # noqa: E402
from research_eth_regime_gated_costgate_ensemble_20260902 import (  # noqa: E402
    HOLDOUT_START, KLINES_PATH, LEVERAGE, MARGIN_FRACTION, OOS_START, OUT_DIR as PART_A_DIR,
    ROUNDTRIP_COST_RATE, SIGNALS, VAL_START, build_regime_pred,
)

CONFLUENCE_BARS = 6          # 30min -- two signals firing within this window count as agreeing
OUT_DIR = ROOT / "tmp/eth_evidence_signal_ensemble_pnl_20260902"


def log(m: str) -> None:
    print(f"[ensemble_pnl] {m}", flush=True)


def per_fire_outcomes(ts, o, h, l, c, dec, sc, atr, horizon, sl, arm, trail) -> pd.DataFrame:
    """One independent outcome per fire, via repeated non-overlapping passes (see module docstring)."""
    remaining = np.arange(len(dec))
    rows = []
    guard = 0
    while len(remaining) and guard < 200:
        guard += 1
        tp = np.full(len(remaining), 999.0)
        res = simulate_single_position(
            timestamps=ts, open_px=o, high=h, low=l, close=c,
            decision_indices=dec[remaining], scores=sc[remaining], tp_moves=tp,
            sl_moves=(sl * atr[remaining]), upper_threshold=1.0, lower_threshold=-1.0,
            horizon_bars=horizon, margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE,
            roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
            arm_moves=(arm * atr[remaining]), trail_moves=(trail * atr[remaining]))
        led = res.ledger
        if led.empty:
            break
        done_ts = set(led["decision_timestamp"])
        idx_ts = pd.DatetimeIndex(ts)
        for r in led.itertuples():
            rows.append({"decision_ts": r.decision_timestamp, "entry_ts": r.entry_timestamp,
                         "exit_ts": r.exit_timestamp, "side": r.side,
                         "trade_return": r.trade_return, "bars_held": r.bars_held,
                         "exit_pos": int(idx_ts.get_loc(r.exit_timestamp))})
        keep = [i for i in remaining if ts.iloc[dec[i]] not in done_ts]
        if len(keep) == len(remaining):
            break
        remaining = np.array(keep, dtype=int)
    out = pd.DataFrame(rows).drop_duplicates("decision_ts").sort_values("decision_ts").reset_index(drop=True)
    return out


def sequential_portfolio(cands: pd.DataFrame, priority: dict[str, int]) -> pd.DataFrame:
    """1-slot walk-forward: take the highest-priority candidate whenever the slot is free."""
    if cands.empty:
        return cands
    c = cands.sort_values(["decision_pos", "prio"]).reset_index(drop=True)
    taken, occupied_until = [], -1
    for r in c.itertuples():
        if r.decision_pos <= occupied_until:
            continue
        taken.append(r.Index)
        occupied_until = r.exit_pos
    return c.loc[taken].reset_index(drop=True)


def summarize(led: pd.DataFrame, tag: str) -> dict:
    if led.empty:
        return {"arm": tag, "n": 0}
    r = led["trade_return"].to_numpy()
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    eq = np.cumprod(1.0 + r)
    dd = float((eq / np.maximum.accumulate(eq) - 1.0).min()) if len(eq) else 0.0
    return {"arm": tag, "n": int(len(r)), "total_bp": round(float(r.sum() * 1e4), 1),
            "mean_bp": round(float(r.mean() * 1e4), 2), "median_bp": round(float(np.median(r) * 1e4), 2),
            "profit_wr": round(float((r > 0).mean()), 4),
            "pf": round(float(wins / losses), 3) if losses > 0 else float("inf"),
            "max_dd": round(dd, 4)}


def main() -> int:
    part_a = json.loads((PART_A_DIR / "part_a_regime_gate.json").read_text())
    klines = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    ts = klines["timestamp"]
    o, h, l, c = (klines[k].to_numpy() for k in ("open", "high", "low", "close"))
    reg = build_regime_pred()
    chop_ts = set(reg.loc[reg["regime"] == 2, "timestamp"])
    idx_ts = pd.DatetimeIndex(ts)

    # ---- per-signal per-fire outcome tables at each signal's own best genuine config ----
    tables, cfgs = {}, {}
    for name, cfg in SIGNALS.items():
        best = (part_a.get(name, {}).get("all") or {}).get("best")
        if not best:
            log(f"{name}: PART A에 진짜 조합 없음 -- 앙상블에서 제외"); continue
        cfgs[name] = best
        fires = pd.read_csv(ROOT / cfg["fires"], parse_dates=["timestamp"])
        fires = fires.loc[fires["timestamp"] < HOLDOUT_START].sort_values("pos").reset_index(drop=True)
        dec = fires["pos"].to_numpy(np.int64)
        sc = np.where(fires["side"].to_numpy() == "bottom", 1.0, -1.0)
        atr = fires["atr_pct"].to_numpy(float)
        for sgn, lbl in ((1.0, "real"), (-1.0, "flip")):
            t = per_fire_outcomes(ts, o, h, l, c, dec, sc * sgn, atr, cfg["horizon"],
                                  best["sl"], best["arm"], best["trail"])
            t["signal"] = name
            t["decision_pos"] = [int(idx_ts.get_loc(x)) for x in t["decision_ts"]]
            t["in_chop"] = t["decision_ts"].isin(chop_ts)
            tables[(name, lbl)] = t
        log(f"{name}: fires={len(fires)} outcomes={len(tables[(name,'real')])} "
            f"(SL{best['sl']}/ARM{best['arm']}/Tr{best['trail']}) chop={float(tables[(name,'real')]['in_chop'].mean()):.3f}")

    names = list(cfgs)
    if not names:
        log("사용 가능한 신호 없음 -- 중단"); return 1
    prio = {n: i for i, n in enumerate(sorted(names, key=lambda n: -cfgs[n]["val_bp"]))}
    windows = {"VAL": (VAL_START, OOS_START), "OOS": (OOS_START, HOLDOUT_START)}

    rows = []
    for lbl in ("real", "flip"):
        allc = pd.concat([tables[(n, lbl)] for n in names], ignore_index=True)
        allc["prio"] = allc["signal"].map(prio)
        for wname, (lo, hi) in windows.items():
            wmask = (allc["decision_ts"] >= lo) & (allc["decision_ts"] < hi)
            w = allc.loc[wmask].copy()
            arms = {}
            for n in names:
                arms[f"single:{n}"] = w[w["signal"] == n]
            arms["union_first"] = w.assign(prio=0)
            arms["pf_priority"] = w
            arms["chop_union"] = w[w["in_chop"]]
            # confluence: >=2 signals, same side, within CONFLUENCE_BARS
            conf_keep = []
            wp = w.sort_values("decision_pos").reset_index(drop=True)
            pos, side = wp["decision_pos"].to_numpy(), wp["side"].to_numpy()
            sig = wp["signal"].to_numpy()
            for i in range(len(wp)):
                near = (np.abs(pos - pos[i]) <= CONFLUENCE_BARS) & (side == side[i]) & (sig != sig[i])
                if near.any():
                    conf_keep.append(i)
            arms["confluence2"] = wp.loc[conf_keep]
            for aname, cand in arms.items():
                led = sequential_portfolio(cand, prio)
                s = summarize(led, aname)
                s.update({"window": wname, "kind": lbl})
                rows.append(s)

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "ensemble_pnl.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)

    real = df[df["kind"] == "real"].pivot_table(index="arm", columns="window",
              values=["n", "total_bp", "mean_bp", "pf", "profit_wr", "max_dd"])
    log("\n=== 1-slot 순차 포트폴리오 (real) ===")
    print(real.round(3).to_string())

    log("\n=== 방향뒤집기 대조 (real total_bp - flip total_bp) ===")
    piv = df.pivot_table(index="arm", columns=["window", "kind"], values="total_bp")
    for arm in piv.index:
        vv = piv.loc[arm, ("VAL", "real")] - piv.loc[arm, ("VAL", "flip")]
        oo = piv.loc[arm, ("OOS", "real")] - piv.loc[arm, ("OOS", "flip")]
        ok = "✅" if (vv > 0 and oo > 0) else "  "
        print(f"  {ok} {arm:28s} VAL {piv.loc[arm,('VAL','real')]:+9.1f} (flip {piv.loc[arm,('VAL','flip')]:+9.1f}, gap {vv:+9.1f}) | "
              f"OOS {piv.loc[arm,('OOS','real')]:+9.1f} (flip {piv.loc[arm,('OOS','flip')]:+9.1f}, gap {oo:+9.1f})")
    log(f"\nWrote {OUT_DIR/'ensemble_pnl.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
