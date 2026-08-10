"""ETH Omega4.6.1 greedy router: honest re-run on CURRENT data + fresh-forward declaration (2026-08-08).

Why this exists.  A fresh-forward audit of the live stack found that ETH's headline number
(+145.34% / -10.13% / 24 trades, `omega4_6_1_extended_oos_20260706/greedy_router_result.json`)
carries NONE of the four flags CLAUDE.md requires.  Reading the replay itself
(scripts/replay_omega4_6_1_greedy_router_20260706.py::greedy_replay) settles the methodology
question: it IS a genuine bar-by-bar causal replay -- one shared position slot, `for i in
range(n-2)` walking bars forward, priority tried only on bars where the account is flat, entry at
`open[i+1]`, exits decided each bar by that component's own TP/SL + exit head, and no saved ledger
read as input.  The audit's earlier "ledger/reconcile" hit was a FALSE POSITIVE off the result
file's own `note`, which says the opposite (that it is NOT the ledger-reconciliation method).

So the methodology is sound and the missing piece is (a) the declaration and (b) reproducibility:
the 07-06 number does not reproduce on current data, because Binance retroactively revises the
open-interest / long-short-ratio metrics this feature set is built on
(diagnosed 2026-07-30, project-omega461-baseline-drift-bisection-20260730).

This script therefore re-runs the SAME replay function, unmodified and imported (not copied), on
today's data, and writes the result to its OWN directory with the four flags recorded.  The 07-06
artifacts are NOT touched -- they stay as the historical record.

Both gate variants are reported, because ETH's live config runs with the duration gate OFF
(`FINAL_GOVERNOR_OMEGA4_6_1_ETH_DURATION_GATE_OFF=True` in .env), so the no-gate row is the one
that corresponds to what is actually wired live.
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
from replay_omega4_6_1_greedy_router_20260706 import (  # noqa: E402
    DURATION_THRESHOLD, greedy_replay, prepare_component,
)

FROZEN_DIR = ROOT / "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706"
OUT_DIR = ROOT / "tmp/eth_greedy_router_freshforward_20260808"
FROZEN_HEADLINE = {"no_gate": {"pnl": 138.19338965711995, "mdd": -14.154462813803049, "trades": 32},
                   "with_gate": {"pnl": 145.3353677513158, "mdd": -10.134492720083554, "trades": 24}}


def curve_metrics(returns: np.ndarray) -> dict:
    curve = np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    peak = np.maximum.accumulate(curve)
    dd = curve / np.maximum(peak, 1e-12) - 1.0
    return {"pnl": round(float((curve[-1] - 1.0) * 100.0), 4),
            "mdd": round(float(dd.min() * 100.0), 4),
            "trades": int(len(returns)),
            "wr": round(float((returns > 0).mean()), 4) if len(returns) else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default="2026-06-30")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = retest.DEVICE

    print("stage=load_frame_current", flush=True)
    frame = retest.load_frame_current(args.start, args.end)
    fee, slip = omega._load_fee_slip()
    print(json.dumps({"rows": int(len(frame)),
                      "range": [str(frame['timestamp'].iloc[0]), str(frame['timestamp'].iloc[-1])]}), flush=True)

    components = {}
    for name, cfg in retest.COMPONENTS.items():
        pred_csv = FROZEN_DIR / name / f"oos_predictions_{cfg['q_tag']}.csv"
        pred = pd.read_csv(pred_csv, usecols=["timestamp"])
        pred["timestamp"] = pd.to_datetime(pred["timestamp"])
        # prepare_component asserts exact timestamp equality, so align the frozen per-bar parent
        # predictions to the requested window rather than silently trusting the row counts
        keep = pred["timestamp"].isin(frame["timestamp"])
        if int(keep.sum()) != len(frame):
            raise RuntimeError(f"{name}: prediction rows covering the window = {int(keep.sum())} != frame rows {len(frame)}")
        full = pd.read_csv(pred_csv)
        full["timestamp"] = pd.to_datetime(full["timestamp"])
        full = full.loc[keep.to_numpy()].reset_index(drop=True)
        tmp = OUT_DIR / f"_aligned_{name}_{cfg['q_tag']}.csv"
        full.to_csv(tmp, index=False)
        components[name] = prepare_component(frame, tmp, cfg, device)
        print(f"{name}: prepared nonzero_side={(components[name]['dec']['side'] != 0).mean():.3f}", flush=True)

    print("stage=greedy_replay (bar-by-bar, single account)", flush=True)
    _, ledger = greedy_replay(frame, components, fee=fee, slip=slip, cost_mult=retest.COST_MULT, device=device)
    returns = ledger["trade_return"].to_numpy(dtype=float)
    no_gate = curve_metrics(returns)

    market = frame[["timestamp", "ou_halflife"]].rename(columns={"timestamp": "entry_timestamp_dt"})
    led = ledger.copy()
    led["entry_timestamp_dt"] = pd.to_datetime(led["entry_timestamp"])
    led = led.merge(market, on="entry_timestamp_dt", how="left")
    hit = led["ou_halflife"] <= DURATION_THRESHOLD
    gated = curve_metrics(led.loc[~hit, "trade_return"].to_numpy(dtype=float))
    gated["skipped"] = int(hit.sum())

    led.to_csv(OUT_DIR / "greedy_router_ledger_current_data.csv", index=False)

    out = {
        "method": "eth_omega4_6_1_greedy_single_account_bar_by_bar_replay_on_current_data",
        "replay_function": "scripts/replay_omega4_6_1_greedy_router_20260706.py::greedy_replay (imported unmodified)",
        "window": [args.start, args.end],
        "cost_mult": retest.COST_MULT,
        "duration_threshold": DURATION_THRESHOLD,
        "live_gate_note": "ETH live runs FINAL_GOVERNOR_OMEGA4_6_1_ETH_DURATION_GATE_OFF=True, so "
                          "'no_gate' is the row that corresponds to the live wiring",
        "no_gate": no_gate,
        "with_gate": gated,
        "source_component_counts": ledger["source_component"].value_counts().to_dict(),
        "frozen_20260706_headline_for_reference": FROZEN_HEADLINE,
        "reproduction_delta": {
            "no_gate_pnl": round(no_gate["pnl"] - FROZEN_HEADLINE["no_gate"]["pnl"], 2),
            "with_gate_pnl": round(gated["pnl"] - FROZEN_HEADLINE["with_gate"]["pnl"], 2),
            "cause": "Binance retroactively revises open-interest / long-short-ratio / whale-ratio "
                     "history; diagnosed 2026-07-30. The frozen 07-06 numbers are not recoverable.",
        },
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "fresh_forward_evidence": [
            "single shared position slot; entries only attempted on bars where the account is flat",
            "forward bar loop `for i in range(0, n-2)`; entry filled at open[i+1] (next bar)",
            "exits decided each bar from that component's own TP/SL barrier and exit head, using "
            "only bar-i state (hold, move, mfe, mae, giveback)",
            "parent direction/quality come from precomputed PER-BAR prediction CSVs at the exact "
            "quality tag, which CLAUDE.md's artifact-integrity rule requires rather than forbids",
            "no saved trade ledger, candidate-event ledger or parent exit timestamp is read as input",
        ],
        "caveat_duration_gate_is_a_ledger_post_filter": (
            "the duration gate is applied by zeroing gated trades in the resulting ledger, the same "
            "metric convention used across this lineage's reports; it does not feed the replay"),
    }
    (OUT_DIR / "result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({k: out[k] for k in ("no_gate", "with_gate", "source_component_counts",
                                          "reproduction_delta")}, indent=2, ensure_ascii=False), flush=True)
    print(f"wrote {OUT_DIR / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
