"""Does h48qual actually contribute to the ETH Omega4.6.1 router? (2026-08-08)

Observation that opened this: on current data the live router fires 37 trades, of which zig075
supplies 29 and h48qual only 8 -- yet h48qual holds PRIORITY, so on every bar it is active it
BLOCKS zig075. The right question is therefore not "how much did h48qual's 8 trades earn" but
"does the account end up better off than if zig075 had those bars".

Three arms, all run through the SAME imported `greedy_replay` with the same frame, costs and
window -- the only difference is which components are offered to it. Because the function's
priority loop skips names absent from the dict, a single-component arm needs no code change:
  router        {h48qual, zig075}   the live configuration
  zig075_only   {zig075}            the counterfactual where h48qual never blocks anything
  h48qual_only  {h48qual}           h48qual standing alone, for reference

Equity compounds inside the replay (`cash`), so removing a component correctly changes both which
trades exist AND the notional of everything after -- a ledger-level subtraction could not do this,
which is exactly why the comparison is run as three replays rather than by deleting rows.

Reported: PnL / MDD / Calmar / trades / win rate per arm, per-quarter PnL, an attribution of the
router's h48qual-sourced trades, an overlap analysis of which zig075 entries h48qual displaced,
and a PAIRED calendar-block bootstrap on router-minus-zig075_only so the 37-trade sample size is
not silently treated as decisive.

Fresh-forward: same as the parent re-run -- bar-by-bar causal, no ledger used as input.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402
from replay_omega4_6_1_greedy_router_20260706 import greedy_replay, prepare_component  # noqa: E402

FROZEN_DIR = ROOT / "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706"
OUT_DIR = ROOT / "tmp/eth_h48qual_contribution_20260808"
B_BOOT, BLOCK_DAYS, SEED = 2000, 21, 903174


def metrics(returns: np.ndarray) -> dict:
    if len(returns) == 0:
        return {"pnl": 0.0, "mdd": 0.0, "calmar": None, "trades": 0, "wr": None}
    eq = np.cumprod(1.0 + returns)
    mdd = float((eq / np.maximum.accumulate(eq) - 1.0).min() * 100)
    pnl = float((eq[-1] - 1.0) * 100)
    return {"pnl": round(pnl, 2), "mdd": round(mdd, 2),
            "calmar": round(pnl / abs(mdd), 2) if mdd < -1e-9 else None,
            "trades": int(len(returns)), "wr": round(float((returns > 0).mean()), 3)}


def paired_block_bootstrap(ledgers: dict[str, pd.DataFrame], rng) -> tuple[dict, int]:
    all_ts = pd.concat([d["entry_ts"] for d in ledgers.values()])
    edges = pd.date_range(all_ts.min(), all_ts.max() + pd.Timedelta(days=BLOCK_DAYS), freq=f"{BLOCK_DAYS}D")
    blocks = list(zip(edges[:-1], edges[1:]))
    per = {k: [d.loc[(d["entry_ts"] >= a) & (d["entry_ts"] < z), "trade_return"].to_numpy(float)
               for a, z in blocks] for k, d in ledgers.items()}
    out = {k: np.empty(B_BOOT) for k in ledgers}
    for i in range(B_BOOT):
        pick = rng.integers(len(blocks), size=len(blocks))
        for k in ledgers:
            r = np.concatenate([per[k][j] for j in pick]) if len(blocks) else np.array([])
            out[k][i] = (np.cumprod(1.0 + r)[-1] - 1.0) * 100 if len(r) else 0.0
    return out, len(blocks)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default="2026-06-30")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = retest.DEVICE
    rng = np.random.default_rng(SEED)

    frame = retest.load_frame_current(args.start, args.end)
    fee, slip = omega._load_fee_slip()
    print(json.dumps({"rows": int(len(frame))}), flush=True)

    comps = {}
    for name, cfg in retest.COMPONENTS.items():
        pred_csv = FROZEN_DIR / name / f"oos_predictions_{cfg['q_tag']}.csv"
        full = pd.read_csv(pred_csv)
        full["timestamp"] = pd.to_datetime(full["timestamp"])
        full = full.loc[full["timestamp"].isin(frame["timestamp"]).to_numpy()].reset_index(drop=True)
        if len(full) != len(frame):
            raise RuntimeError(f"{name}: aligned prediction rows {len(full)} != frame {len(frame)}")
        tmp = OUT_DIR / f"_aligned_{name}.csv"
        full.to_csv(tmp, index=False)
        comps[name] = prepare_component(frame, tmp, cfg, device)
        print(f"prepared {name}", flush=True)

    arms = {"router": {"h48qual": comps["h48qual"], "zig075": comps["zig075"]},
            "zig075_only": {"zig075": comps["zig075"]},
            "h48qual_only": {"h48qual": comps["h48qual"]}}
    results, ledgers = {}, {}
    for arm, sel in arms.items():
        _, led = greedy_replay(frame, sel, fee=fee, slip=slip, cost_mult=retest.COST_MULT, device=device)
        led["entry_ts"] = pd.to_datetime(led["entry_timestamp"])
        led.to_csv(OUT_DIR / f"ledger_{arm}.csv", index=False)
        ledgers[arm] = led
        r = led["trade_return"].to_numpy(float)
        m = metrics(r)
        q = led.copy()
        q["q"] = q["entry_ts"].dt.to_period("Q").astype(str)
        m["by_quarter"] = {k: round(float((np.cumprod(1 + g["trade_return"].to_numpy(float))[-1] - 1) * 100), 2)
                           for k, g in q.groupby("q")}
        m["source_mix"] = led["source_component"].value_counts().to_dict()
        results[arm] = m
        print(json.dumps({arm: {k: m[k] for k in ('pnl', 'mdd', 'calmar', 'trades', 'wr')}}), flush=True)

    # what did the router's h48qual-sourced trades themselves earn, and what did they displace?
    r_led = ledgers["router"]
    h_rows = r_led.loc[r_led["source_component"] == "h48qual"]
    z_only = ledgers["zig075_only"]
    displaced = []
    for _, hr in h_rows.iterrows():
        overlap = z_only.loc[(z_only["entry_ts"] >= hr["entry_ts"])
                             & (z_only["entry_ts"] <= pd.to_datetime(hr["exit_timestamp"]))]
        displaced.append({"h48qual_entry": str(hr["entry_ts"]),
                          "h48qual_return_pct": round(float(hr["trade_return"]) * 100, 3),
                          "zig075_trades_blocked": int(len(overlap)),
                          "zig075_blocked_return_pct": round(float(
                              (np.cumprod(1 + overlap["trade_return"].to_numpy(float))[-1] - 1) * 100), 3)
                          if len(overlap) else 0.0})
    attribution = {
        "h48qual_trades_in_router": int(len(h_rows)),
        "h48qual_own_return_sum_pct": round(float(h_rows["trade_return"].sum() * 100), 2),
        "h48qual_own_win_rate": round(float((h_rows["trade_return"] > 0).mean()), 3) if len(h_rows) else None,
        "per_trade_displacement": displaced,
        "total_zig075_trades_blocked": int(sum(d["zig075_trades_blocked"] for d in displaced)),
    }

    boot, n_blocks = paired_block_bootstrap({k: ledgers[k] for k in ("router", "zig075_only")}, rng)
    diff = boot["router"] - boot["zig075_only"]
    boot_stats = {"n_blocks": n_blocks, "B": B_BOOT,
                  "p_router_better": round(float((diff > 0).mean()), 3),
                  "median_diff_pct": round(float(np.median(diff)), 2),
                  "diff_ci90": [round(float(np.percentile(diff, 5)), 2),
                                round(float(np.percentile(diff, 95)), 2)]}

    verdict = ("h48qual ADDS" if results["router"]["pnl"] > results["zig075_only"]["pnl"]
               else "h48qual SUBTRACTS")
    out = {"window": [args.start, args.end], "cost_mult": retest.COST_MULT,
           "arms": results, "attribution": attribution, "paired_bootstrap": boot_stats,
           "headline": {"router_pnl": results["router"]["pnl"],
                        "zig075_only_pnl": results["zig075_only"]["pnl"],
                        "delta_pct": round(results["router"]["pnl"] - results["zig075_only"]["pnl"], 2),
                        "verdict_on_point_estimate": verdict},
           "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
           "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
           "note": "three independent bar-by-bar replays; equity compounds inside each, so this is "
                   "a true counterfactual and not a ledger-row subtraction"}
    (OUT_DIR / "result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({"headline": out["headline"], "attribution": {k: attribution[k] for k in
                      ("h48qual_trades_in_router", "h48qual_own_return_sum_pct",
                       "h48qual_own_win_rate", "total_zig075_trades_blocked")},
                      "paired_bootstrap": boot_stats}, indent=2, ensure_ascii=False), flush=True)
    print(f"wrote {OUT_DIR / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
