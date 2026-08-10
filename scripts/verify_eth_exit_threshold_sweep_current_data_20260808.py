"""Re-run the 2026-07-06 Candidate-2 exit-threshold sweep on CURRENT data.

WHY: Candidate 2 (docs/model_contracts/omega4_6_1_upgrade_investigation_20260706.md) closed the
"activate the exit head" line on a sweep whose OOS levels are the frozen-artifact numbers that no
longer reproduce (+145.34 gated / +138.19 no-gate then, vs +82.53 / +77.11 now -- Binance
retroactively revises the OI / long-short-ratio / whale-ratio history the features are built on,
which changes the model's DECISIONS, not just the reported level).

Closing a line on an ordering that was never re-checked under the data revision is an assumption,
not evidence. This script checks it.

STATUS OF THIS RUN: reproduction / verification of an ALREADY-CLOSED line, not a selection.
- If 0.95 stays best, the line stays closed and we now have current-data evidence for it.
- If the ordering inverts, that REOPENS the line -- and reopening requires its own pre-registered
  contract before any promotion claim. Nothing here may be used to promote a threshold change.

Harness: multislot_replay at n_slots=1 from research_eth_multislot_capacity_transfer_20260808,
which passed a hard regression gate against the incumbent greedy_replay (+77.11 / -15.48 / 37tr).
The threshold is overridden by writing comps[name]["exit_threshold"] -- the replay reads it from
there, so no code change is needed.
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
from replay_omega4_6_1_greedy_router_20260706 import prepare_component  # noqa: E402
from research_eth_multislot_capacity_transfer_20260808 import (  # noqa: E402
    gated, metrics, multislot_replay,
)

OOS_PRED_DIR = ROOT / "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706"
OUT = ROOT / "tmp/eth_exit_threshold_recheck_20260808"
THRESHOLDS = [0.95, 0.90, 0.80, 0.70, 0.60, 0.50]
FROZEN_20260706 = {  # the table being re-checked (OOS column, gated basis)
    0.95: (145.34, -10.13), 0.90: (89.22, -14.90), 0.80: (59.62, -16.23),
    0.70: (21.66, -25.02), 0.60: (-18.78, -39.67), 0.50: (-26.93, -37.54),
}


def build_comps(frame: pd.DataFrame, pred_paths: dict, device, tag: str):
    preds = {}
    for name in retest.COMPONENTS:
        p = pd.read_csv(pred_paths[name])
        p = p.rename(columns={c: c.replace("_expertdq_oof_", "_expertdq_") for c in p.columns})
        p["timestamp"] = pd.to_datetime(p["timestamp"])
        preds[name] = p
    common = frame["timestamp"]
    for p in preds.values():
        common = common[common.isin(p["timestamp"])]
    frame = frame.loc[frame["timestamp"].isin(common)].reset_index(drop=True)
    comps = {}
    for name, cfg in retest.COMPONENTS.items():
        full = preds[name].loc[preds[name]["timestamp"].isin(frame["timestamp"])].reset_index(drop=True)
        if len(full) != len(frame):
            raise RuntimeError(f"{tag}/{name}: aligned {len(full)} != frame {len(frame)}")
        tmp = OUT / f"_aligned_{tag}_{name}.csv"
        full.to_csv(tmp, index=False)
        comps[name] = prepare_component(frame, tmp, cfg, device)
    return frame, comps


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    device = retest.DEVICE
    fee, slip = omega._load_fee_slip()

    windows = {
        "val": (gval.load_val_frame(), {k: Path(v) for k, v in gval.VAL_PRED.items()}),
        "oos": (retest.load_frame_current("2026-01-01", "2026-06-30"),
                {n: OOS_PRED_DIR / n / f"oos_predictions_{c['q_tag']}.csv"
                 for n, c in retest.COMPONENTS.items()}),
    }

    rows = []
    for tag, (frame0, preds) in windows.items():
        frame, comps = build_comps(frame0, preds, device, tag)
        print(f"[{tag}] aligned rows={len(frame)}", flush=True)
        for th in THRESHOLDS:
            for c in comps.values():
                c["exit_threshold"] = float(th)
            led = multislot_replay(frame, comps, n_slots=1, fee=fee, slip=slip,
                                   cost_mult=retest.COST_MULT, device=device)
            led["entry_ts"] = pd.to_datetime(led["entry_timestamp"])
            ng = metrics(led["trade_return"].to_numpy(float))
            g = metrics(gated(led, frame))
            reasons = led["reason"].value_counts().to_dict()
            rows.append({"window": tag, "exit_th": th,
                         "no_gate_pnl": ng["pnl"], "no_gate_mdd": ng["mdd"], "no_gate_trades": ng["trades"],
                         "gated_pnl": g["pnl"], "gated_mdd": g["mdd"],
                         "exit_head_fires": int(reasons.get("exit_head", 0)),
                         "take_profit": int(reasons.get("take_profit", 0)),
                         "stop_loss": int(reasons.get("stop_loss", 0))})
            print(json.dumps({f"{tag}|th={th}": {"no_gate": ng, "gated_pnl": g["pnl"],
                                                 "exit_head": int(reasons.get("exit_head", 0))}}), flush=True)
            led.to_csv(OUT / f"ledger_{tag}_th{int(th*100)}.csv", index=False)

    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUT / "sweep.csv", index=False)
    print("\n=== current-data sweep ===")
    print(tbl.to_string(index=False))

    oos = tbl[tbl["window"] == "oos"].set_index("exit_th")
    best_ng = float(oos["no_gate_pnl"].idxmax())
    best_g = float(oos["gated_pnl"].idxmax())
    monotone_ng = bool(all(oos.loc[a, "no_gate_pnl"] >= oos.loc[b, "no_gate_pnl"]
                           for a, b in zip(THRESHOLDS[:-1], THRESHOLDS[1:])))
    print("\n=== verdict ===")
    print(f"  OOS best threshold: no_gate={best_ng}  gated={best_g}")
    print(f"  OOS no_gate monotone decreasing as threshold falls: {monotone_ng}")
    comp = []
    for th in THRESHOLDS:
        old = FROZEN_20260706[th]
        new = (float(oos.loc[th, "gated_pnl"]), float(oos.loc[th, "gated_mdd"]))
        comp.append({"exit_th": th, "frozen_20260706_gated": old, "current_gated": new})
    res = {
        "status": "verification of an already-closed line; NOT a selection; no promotion may rest on this",
        "why": "the Candidate-2 sweep's OOS levels are frozen-artifact numbers (+145.34) that no longer reproduce (+82.53)",
        "sweep": rows,
        "oos_best_threshold_no_gate": best_ng, "oos_best_threshold_gated": best_g,
        "oos_no_gate_monotone_in_threshold": monotone_ng,
        "frozen_vs_current": comp,
        "line_stays_closed": bool(best_ng == 0.95 and best_g == 0.95),
    }
    (OUT / "result.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
