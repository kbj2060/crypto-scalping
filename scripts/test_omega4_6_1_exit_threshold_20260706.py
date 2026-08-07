"""Exit-head threshold sweep. The exit head is architecturally present but EMPIRICALLY inert at
its frozen 0.95 threshold (every observed trade exited via TP/SL, never the exit head). Test
whether a lower threshold (letting the learned exit head actually cut trades early) improves
things. VAL = selection, OOS = one-shot confirm. Same discipline as the h48qual test.
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

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import replay_omega4_6_1_greedy_val_20260706 as valmod  # noqa: E402
from test_omega4_6_1_drop_h48qual_20260706 import _metrics  # noqa: E402

THRESHOLDS = [0.95, 0.90, 0.80, 0.70, 0.60, 0.50]


def build(window: str):
    device = retest.DEVICE
    if window == "VAL":
        frame = valmod.load_val_frame()
        comps = {}
        for cname, cfg in retest.COMPONENTS.items():
            pred = pd.read_csv(valmod.VAL_PRED[cname])
            pred = pred.rename(columns={c: c.replace("_expertdq_oof_", "_expertdq_") for c in pred.columns})
            pred["timestamp"] = pd.to_datetime(pred["timestamp"])
            pred = pred[(pred["timestamp"] >= valmod.START) & (pred["timestamp"] <= valmod.END)].reset_index(drop=True)
            common = frame["timestamp"].isin(pred["timestamp"])
            frame = frame[common].reset_index(drop=True)
            pred = pred[pred["timestamp"].isin(frame["timestamp"])].reset_index(drop=True)
            tmp = ROOT / f"tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/_val_{cname}_aligned.csv"
            pred.to_csv(tmp, index=False)
            comps[cname] = greedy.prepare_component(frame, tmp, cfg, device)
        return frame, comps
    else:
        frame = retest.load_frame_current("2026-01-01", "2026-06-30")
        comps = {}
        for cname, cfg in retest.COMPONENTS.items():
            pred_csv = retest.EXT_PRED_DIR / cname / f"oos_predictions_{cfg['q_tag']}.csv"
            comps[cname] = greedy.prepare_component(frame, pred_csv, cfg, device)
        return frame, comps


def run(window: str, frame, comps) -> None:
    fee, slip = omega._load_fee_slip()
    print(f"\n################ {window} ################")
    for th in THRESHOLDS:
        for c in comps.values():
            c["exit_threshold"] = th
        _, lg = greedy.greedy_replay(frame, comps, fee=fee, slip=slip, cost_mult=retest.COST_MULT, device=retest.DEVICE)
        wg = _metrics(lg, frame, apply_gate=True)
        n_exit = int((lg["reason"] == "exit_head").sum()) if not lg.empty else 0
        print(f"  exit_th={th:.2f}  gate: pnl={wg['pnl']:+7.2f}% mdd={wg['mdd']:+6.2f}% n={wg['trades']:2d} "
              f"wr={wg['wr']:.3f}  exit_head_fires={n_exit}")


def main() -> int:
    for window in ("VAL", "OOS"):
        frame, comps = build(window)
        run(window, frame, comps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
