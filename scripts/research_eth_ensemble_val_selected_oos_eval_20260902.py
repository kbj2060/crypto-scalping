#!/usr/bin/env python3
"""Config-selection-bias test for the 2026-09-02 ensemble result. User: "그렇게 해줘".

THE THREAT BEING TESTED. In the original study each signal's (SL, ARM, Trail) was picked as
argmax min(val_bp, oos_bp) over 96 combos -- i.e. OOS participated in the selection, so the
reported OOS numbers are not out-of-sample with respect to that choice. Seeds cannot detect this
(the backtest is deterministic); only re-selecting can.

METHOD. Select each signal's config on VAL ALONE -- VAL positive, VAL beats its own direction-flip,
flip negative on VAL -- maximising VAL mean bp. Then evaluate that frozen config on OOS. OOS is now
genuinely untouched by the selection. The ensemble/top-k arms are rebuilt on those frozen configs
and the arm ORDERING is re-checked on OOS:

  Q1  does top2 still beat top1 on TOTAL bp?          (original: +42%)
  Q2  does top1 still beat top2 on RISK-ADJUSTED?     (original: total/max_dd ~30% better)
  Q3  is confluence2 still rejected?

A config-stability diagnostic is reported alongside: if the VAL-only pick equals the VAL+OOS pick,
the bias was nil for that signal and its numbers stand as originally reported.

⚠️ VAL is consumed by the selection here, so only the OOS column is an honest read.
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

from core.causal_futures_backtest import purged_decision_mask  # noqa: E402
from research_eth_regime_gated_costgate_ensemble_20260902 import (  # noqa: E402
    HOLDOUT_START, KLINES_PATH, MIN_WINDOW_N, OOS_START, OUT_DIR as PA_DIR, SIGNALS, VAL_START,
    run_grid,
)
from research_eth_evidence_signal_ensemble_pnl_20260902 import (  # noqa: E402
    CONFLUENCE_BARS, per_fire_outcomes, sequential_portfolio, summarize,
)

OUT_DIR = ROOT / "tmp/eth_ensemble_val_selected_oos_eval_20260902"


def log(m: str) -> None:
    print(f"[val_sel_oos] {m}", flush=True)


def pick_val_only(real: pd.DataFrame, flip: pd.DataFrame) -> dict | None:
    """Best config using VAL evidence ONLY (OOS never consulted)."""
    fmap = {(r.sl, r.arm, r.trail): r for r in flip.itertuples()}
    cands = []
    for r in real.itertuples():
        f = fmap[(r.sl, r.arm, r.trail)]
        if r.val_n > 0 and r.val_avg_bp > 0 and (r.val_avg_bp - f.val_avg_bp) > 0 and f.val_avg_bp < 0:
            cands.append({"sl": r.sl, "arm": r.arm, "trail": r.trail, "val_bp": r.val_avg_bp,
                          "oos_bp_unseen": r.oos_avg_bp})
    return max(cands, key=lambda x: x["val_bp"]) if cands else None


def main() -> int:
    part_a = json.loads((PA_DIR / "part_a_regime_gate.json").read_text())
    klines = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    ts = klines["timestamp"]
    o, h, l, c = (klines[k].to_numpy() for k in ("open", "high", "low", "close"))
    idx = pd.DatetimeIndex(ts)

    chosen, stability = {}, []
    for name, cfg in SIGNALS.items():
        fires = pd.read_csv(ROOT / cfg["fires"], parse_dates=["timestamp"])
        fires = fires.loc[fires["timestamp"] < HOLDOUT_START].sort_values("pos").reset_index(drop=True)
        dec = fires["pos"].to_numpy(np.int64)
        sc = np.where(fires["side"].to_numpy() == "bottom", 1.0, -1.0)
        atr = fires["atr_pct"].to_numpy(float)
        horizon = cfg["horizon"]
        ev = purged_decision_mask(ts, start=VAL_START, end=OOS_START, horizon_bars=horizon)
        eo = purged_decision_mask(ts, start=OOS_START, end=HOLDOUT_START, horizon_bars=horizon)
        vset, oset = set(np.flatnonzero(ev).tolist()), set(np.flatnonzero(eo).tolist())
        vm = np.array([d in vset for d in dec]); om = np.array([d in oset for d in dec])
        if vm.sum() < MIN_WINDOW_N or om.sum() < MIN_WINDOW_N:
            log(f"{name}: 표본부족 -- 스킵"); continue
        real = run_grid(ts, o, h, l, c, dec, sc, atr, horizon, vm, om)
        flip = run_grid(ts, o, h, l, c, dec, -sc, atr, horizon, vm, om)
        pick = pick_val_only(real, flip)
        if pick is None:
            log(f"{name}: VAL 기준 통과 조합 없음 -- 제외"); continue
        chosen[name] = pick
        orig = part_a[name]["all"]["best"]
        same = (pick["sl"], pick["arm"], pick["trail"]) == (orig["sl"], orig["arm"], orig["trail"])
        stability.append({"signal": name, "val_only": f"SL{pick['sl']}/ARM{pick['arm']}/Tr{pick['trail']}",
                          "val_plus_oos": f"SL{orig['sl']}/ARM{orig['arm']}/Tr{orig['trail']}",
                          "identical": same, "val_bp": pick["val_bp"],
                          "oos_bp_unseen": pick["oos_bp_unseen"], "oos_bp_original": orig["oos_bp"]})
        log(f"{name:24s} VAL선택 SL{pick['sl']}/ARM{pick['arm']}/Tr{pick['trail']} "
            f"(VAL {pick['val_bp']:+6.2f}) -> OOS {pick['oos_bp_unseen']:+6.2f}  | "
            f"원본(VAL+OOS선택) SL{orig['sl']}/ARM{orig['arm']}/Tr{orig['trail']} OOS {orig['oos_bp']:+6.2f}"
            + ("  [동일]" if same else "  [다름]"))

    log("\n=== 설정 안정성 ===")
    st = pd.DataFrame(stability)
    print(st.to_string(index=False))
    log(f"  동일한 신호 {int(st['identical'].sum())}/{len(st)}")

    # ---- rebuild per-fire outcomes at the VAL-selected configs ----
    tabs = {}
    for name, cfg in SIGNALS.items():
        if name not in chosen:
            continue
        b = chosen[name]
        fires = pd.read_csv(ROOT / cfg["fires"], parse_dates=["timestamp"])
        fires = fires.loc[fires["timestamp"] < HOLDOUT_START].sort_values("pos").reset_index(drop=True)
        dec = fires["pos"].to_numpy(np.int64)
        sc = np.where(fires["side"].to_numpy() == "bottom", 1.0, -1.0)
        atr = fires["atr_pct"].to_numpy(float)
        for sgn, lb in ((1.0, "real"), (-1.0, "flip")):
            t = per_fire_outcomes(ts, o, h, l, c, dec, sc * sgn, atr, cfg["horizon"],
                                  b["sl"], b["arm"], b["trail"])
            t["signal"] = name
            t["decision_pos"] = [int(idx.get_loc(x)) for x in t["decision_ts"]]
            tabs[(name, lb)] = t

    # quality order fixed by VAL bp only (selection-consistent)
    order = sorted(chosen, key=lambda n: -chosen[n]["val_bp"])
    prio = {n: i for i, n in enumerate(order)}
    log(f"\n품질 순서 (VAL bp 기준): {order}")

    rows = []
    for lb in ("real", "flip"):
        allc = pd.concat([tabs[(n, lb)] for n in order], ignore_index=True)
        allc["prio"] = allc["signal"].map(prio)
        for wn, (lo, hi) in (("VAL", (VAL_START, OOS_START)), ("OOS", (OOS_START, HOLDOUT_START))):
            w = allc[(allc.decision_ts >= lo) & (allc.decision_ts < hi)].copy()
            arms = {f"top{k}_union": w[w["signal"].isin(order[:k])] for k in range(1, len(order) + 1)}
            wp = w.sort_values("decision_pos").reset_index(drop=True)
            pos, side, sig = wp["decision_pos"].to_numpy(), wp["side"].to_numpy(), wp["signal"].to_numpy()
            keep = [i for i in range(len(wp))
                    if ((np.abs(pos - pos[i]) <= CONFLUENCE_BARS) & (side == side[i]) & (sig != sig[i])).any()]
            arms["confluence2"] = wp.loc[keep]
            for an, cand in arms.items():
                s = summarize(sequential_portfolio(cand, prio), an)
                s.update({"window": wn, "kind": lb}); rows.append(s)

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "val_selected_oos_eval.csv", index=False)
    st.to_csv(OUT_DIR / "config_stability.csv", index=False)
    pd.set_option("display.width", 220)

    log("\n=== ⭐OOS (설정이 본 적 없는 창) -- 이것만이 정직한 수치 ===")
    oos = df[(df.kind == "real") & (df.window == "OOS")].set_index("arm")
    print(oos[["n", "total_bp", "mean_bp", "pf", "profit_wr", "max_dd"]].round(3).to_string())
    oos = oos.assign(ret_per_dd=(oos["total_bp"] / oos["max_dd"].abs()).round(0))
    print("\n  총bp / |최대낙폭| (위험조정):")
    for a in oos.index:
        print(f"    {a:14s} {oos.loc[a,'ret_per_dd']:>10.0f}")

    log("\n=== 방향뒤집기 대조 (OOS) ===")
    p = df[df.window == "OOS"].pivot_table(index="arm", columns="kind", values="total_bp")
    for a in p.index:
        gap = p.loc[a, "real"] - p.loc[a, "flip"]
        print(f"  {'✅' if gap > 0 and p.loc[a,'real'] > 0 else '❌'} {a:14s} "
              f"real {p.loc[a,'real']:+9.1f}  flip {p.loc[a,'flip']:+9.1f}  gap {gap:+9.1f}")
    log(f"\nWrote {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
