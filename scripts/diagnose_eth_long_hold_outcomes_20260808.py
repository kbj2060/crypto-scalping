"""DIAGNOSTIC ONLY -- how do long-held ETH positions resolve, and would flattening early help?

Triggered 2026-08-08 by the live ETH short (entry 1918.58, opened 2026-07-16, 6687 bars = 23.2d).

Per CLAUDE.md this is a saved-ledger / historical-reproduction diagnostic. It is NOT promotion
evidence, NOT a model-selection input, and it spends NO OOS look: the ledgers replayed here are the
N=1 arm produced today by research_eth_multislot_capacity_transfer_20260808.py, whose numbers are
already published (VAL +36.82, OOS +77.11).

Question asked precisely: CONDITIONAL on a position still being open after D days, what happens if
you flatten it right then versus letting the model's own TP/SL/exit-head finish? Survivorship is
handled by conditioning -- only trades that actually reached day D enter each row.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402
import replay_omega4_6_1_greedy_val_20260706 as gval  # noqa: E402

LED_DIR = ROOT / "tmp/eth_multislot_capacity_20260808"
OUT = ROOT / "tmp/eth_long_hold_diagnostic_20260808"
BARS_PER_DAY = 288
THRESH_DAYS = [3, 5, 10, 15, 20, 23, 30]
LIVE_HOLD_BARS = 6687


def load_windows():
    val = gval.load_val_frame()
    oos = retest.load_frame_current("2026-01-01", "2026-06-30")
    # the ledgers were produced on the component-aligned frames; realign identically
    out = {}
    for tag, frame, pred_paths in (
        ("val", val, {k: Path(v) for k, v in gval.VAL_PRED.items()}),
        ("oos", oos, {n: ROOT / "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706" / n
                      / f"oos_predictions_{c['q_tag']}.csv" for n, c in retest.COMPONENTS.items()}),
    ):
        common = frame["timestamp"]
        for n in retest.COMPONENTS:
            p = pd.read_csv(pred_paths[n], usecols=["timestamp"])
            p["timestamp"] = pd.to_datetime(p["timestamp"])
            common = common[common.isin(p["timestamp"])]
        out[tag] = frame.loc[frame["timestamp"].isin(common)].reset_index(drop=True)
    return out


def trade_paths(frame: pd.DataFrame, led: pd.DataFrame, fee_eff: float, slip_eff: float):
    """Reconstruct each trade's per-bar account return, using the replay's own price convention."""
    o = pd.to_numeric(frame["open"], errors="raise").to_numpy(float)
    c = pd.to_numeric(frame["close"], errors="raise").to_numpy(float)
    recs = []
    for _, r in led.iterrows():
        side, ei, xi = int(r["side"]), int(r["entry_i"]), int(r["exit_i"])
        n = float(r["notional"])
        entry_px = o[ei] * (1 + slip_eff if side > 0 else 1 - slip_eff)
        j = np.arange(ei, xi + 1)
        move = ((c[j] * (1 - slip_eff) - entry_px) / entry_px if side > 0
                else (entry_px - c[j] * (1 + slip_eff)) / entry_px)
        ret = (1.0 - fee_eff * n) * (1.0 + move * n - fee_eff * n) - 1.0
        recs.append({"side": side, "comp": r["source_component"], "reason": r["reason"],
                     "hold_bars": xi - ei, "final_return": float(r["trade_return"]),
                     "path": ret, "entry_ts": r["entry_timestamp"]})
    return recs


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fee, slip = omega._load_fee_slip()
    fee_eff, slip_eff = fee * retest.COST_MULT, slip * retest.COST_MULT
    frames = load_windows()

    allrecs = []
    for tag in ("val", "oos"):
        led = pd.read_csv(LED_DIR / f"ledger_{tag}_n1.csv")
        recs = trade_paths(frames[tag], led, fee_eff, slip_eff)
        for r in recs:
            r["window"] = tag
        allrecs += recs
        print(f"{tag}: {len(recs)} trades", flush=True)

    hb = np.array([r["hold_bars"] for r in allrecs])
    fr = np.array([r["final_return"] for r in allrecs])
    print("\n--- hold-time distribution (both windows, N=%d) ---" % len(allrecs))
    for q in (50, 75, 90, 100):
        print(f"  p{q} hold = {np.percentile(hb, q)/BARS_PER_DAY:.1f} d")
    print(f"  live position is at {LIVE_HOLD_BARS/BARS_PER_DAY:.1f} d "
          f"= percentile {float((hb < LIVE_HOLD_BARS).mean())*100:.1f} of this history")

    rows = []
    for d in THRESH_DAYS:
        b = int(d * BARS_PER_DAY)
        surv = [r for r in allrecs if r["hold_bars"] >= b]
        if not surv:
            rows.append({"day": d, "n_reaching": 0})
            continue
        at_d = np.array([r["path"][b] for r in surv])
        fin = np.array([r["final_return"] for r in surv])
        delta = fin - at_d          # >0 means HOLDING beat flattening at day d
        rows.append({
            "day": d, "n_reaching": len(surv),
            "flatten_at_d_mean_pct": round(float(at_d.mean() * 100), 3),
            "hold_to_model_exit_mean_pct": round(float(fin.mean() * 100), 3),
            "delta_hold_minus_flatten_mean_pct": round(float(delta.mean() * 100), 3),
            "delta_median_pct": round(float(np.median(delta) * 100), 3),
            "share_where_holding_won": round(float((delta > 0).mean()), 3),
            "final_positive_share": round(float((fin > 0).mean()), 3),
            "shorts": int(sum(1 for r in surv if r["side"] < 0)),
            "longs": int(sum(1 for r in surv if r["side"] > 0)),
        })
    tbl = pd.DataFrame(rows)
    print("\n--- conditional on still being open at day D ---")
    print(tbl.to_string(index=False))

    # the live position's own shape: short, zig075
    shorts = [r for r in allrecs if r["side"] < 0]
    z = [r for r in allrecs if r["comp"] == "zig075"]
    print(f"\n--- shape match: SHORT n={len(shorts)} mean_final={np.mean([r['final_return'] for r in shorts])*100:+.2f}% "
          f"| zig075 n={len(z)} mean_final={np.mean([r['final_return'] for r in z])*100:+.2f}%")
    long_shorts = [r for r in shorts if r["hold_bars"] >= 20 * BARS_PER_DAY]
    print(f"--- SHORT held >=20d: n={len(long_shorts)}", end="")
    if long_shorts:
        print(f" final={[round(r['final_return']*100,2) for r in long_shorts]} "
              f"reasons={[r['reason'] for r in long_shorts]}")
    else:
        print(" -- NO historical precedent in these two windows")

    res = {"diagnostic_only": True,
           "not_promotion_evidence": "saved-ledger diagnostic per CLAUDE.md; no OOS look spent",
           "ledgers": "tmp/eth_multislot_capacity_20260808/ledger_{val,oos}_n1.csv (N=1 = incumbent)",
           "n_trades": len(allrecs),
           "live_hold_days": round(LIVE_HOLD_BARS / BARS_PER_DAY, 2),
           "live_hold_percentile": round(float((hb < LIVE_HOLD_BARS).mean()) * 100, 1),
           "conditional_table": rows,
           "short_held_20d_plus_n": len(long_shorts)}
    (OUT / "result.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    tbl.to_csv(OUT / "conditional_flatten_vs_hold.csv", index=False)
    print(f"\nwrote {OUT / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
