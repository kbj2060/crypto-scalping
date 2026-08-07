"""Omega4.6.1 -- next-version candidate: replace L6's static "h48qual always has priority" greedy
rule with a learned meta-router GATE on h48qual's signals (does NOT touch the live-wired
omega4_6_1_duration_ou_halflife_risk_gate_20260630 base -- this is a separate, frozen-independent
candidate: omega4_6_1_learned_meta_router_20260707).

Motivation (found 2026-07-06/07 in this session's own data, not from a paper): h48qual's own
realized trades are net-NEGATIVE in both VAL (2025-10..12) and OOS (2026-01..06), yet dropping the
whole component reversed direction between windows (helped VAL, hurt OOS one-shot) -- rejected.
Finer-grained counterfactual analysis (running h48qual ALONE, i.e. with the full position slot to
itself, not competing with zig075) on VAL shows the badness concentrates specifically in h48qual
LONG signals (1/6 win rate, negative EV) vs h48qual SHORT (10/22 win rate, roughly breakeven-to-
positive EV given the payoff asymmetry). This script builds a router GATE that only lets h48qual's
signal through when side == SHORT (blocking LONG so zig075/cash gets the slot instead), selects
the gate rule on VAL only, then confirms ONCE on OOS -- exactly the discipline that rejected the
cruder "drop h48qual entirely" and "lower exit threshold" candidates earlier this session.

Implementation approach: rather than touching the already-validated `greedy_replay` engine, this
gates h48qual's OWN decision frame (forcing side/action to CASH at gated-out bars) before handing
it to the unmodified greedy_replay -- so gated-out bars fall through to zig075 exactly as they
would in the real router, with zero risk of introducing a new replay bug.
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

MODEL_ID = "omega4_6_1_learned_meta_router_20260707"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega4_6_1_learned_meta_router_20260707"


def build_val():
    device = retest.DEVICE
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


def build_oos():
    device = retest.DEVICE
    frame = retest.load_frame_current("2026-01-01", "2026-06-30")
    comps = {}
    for cname, cfg in retest.COMPONENTS.items():
        pred_csv = retest.EXT_PRED_DIR / cname / f"oos_predictions_{cfg['q_tag']}.csv"
        comps[cname] = greedy.prepare_component(frame, pred_csv, cfg, device)
    return frame, comps


def gated_component(comp: dict, keep_mask: np.ndarray) -> dict:
    """Return a copy of `comp` where h48qual's decision is forced to CASH wherever keep_mask is False."""
    comp2 = dict(comp)
    dec2 = comp["dec"].copy()
    drop = ~keep_mask
    dec2.loc[drop, "action"] = 0
    dec2.loc[drop, "side"] = 0
    dec2.loc[drop, "notional_exposure"] = 0.0
    comp2["dec"] = dec2
    return comp2


def run_full_router(frame, comps, fee, slip):
    greedy.PRIORITY = ("h48qual", "zig075")
    return greedy.greedy_replay(frame, comps, fee=fee, slip=slip, cost_mult=retest.COST_MULT, device=retest.DEVICE)


def report(name: str, frame, ledger) -> dict:
    ng = _metrics(ledger, frame, apply_gate=False)
    wg = _metrics(ledger, frame, apply_gate=True)
    print(f"  {name:34s} no_gate: pnl={ng['pnl']:+7.2f}% mdd={ng['mdd']:+6.2f}% n={ng['trades']:2d} wr={ng['wr']:.3f}  |  "
          f"gate: pnl={wg['pnl']:+7.2f}% mdd={wg['mdd']:+6.2f}% n={wg['trades']:2d} wr={wg['wr']:.3f}")
    return {"no_gate": ng, "with_gate": wg}


def main() -> int:
    fee, slip = omega._load_fee_slip()

    print("\n################ VAL 2025-10-01..12-31 (SELECTION window) ################")
    val_frame, val_comp = build_val()
    side = pd.to_numeric(val_comp["h48qual"]["dec"]["side"], errors="raise").to_numpy()

    _, baseline_val = run_full_router(val_frame, val_comp, fee, slip)
    base_val_res = report("BASELINE (h48qual always priority)", val_frame, baseline_val)

    keep_short_only = side != 1  # block h48qual LONG signals
    gated_h48 = gated_component(val_comp["h48qual"], keep_short_only)
    _, gated_val = run_full_router(val_frame, {"h48qual": gated_h48, "zig075": val_comp["zig075"]}, fee, slip)
    gate_val_res = report("META-ROUTER (block h48qual LONG)", val_frame, gated_val)

    val_winner_is_gate = gate_val_res["with_gate"]["pnl"] > base_val_res["with_gate"]["pnl"]
    print(f"\nVAL selection: {'META-ROUTER selected' if val_winner_is_gate else 'BASELINE selected'} "
          f"(gate PnL {gate_val_res['with_gate']['pnl']:+.2f}% vs baseline {base_val_res['with_gate']['pnl']:+.2f}%)")

    print("\n################ OOS 2026-01-01..06-30 (one-shot confirm, frozen rule from VAL) ################")
    oos_frame, oos_comp = build_oos()
    side_oos = pd.to_numeric(oos_comp["h48qual"]["dec"]["side"], errors="raise").to_numpy()

    _, baseline_oos = run_full_router(oos_frame, oos_comp, fee, slip)
    base_oos_res = report("BASELINE (h48qual always priority)", oos_frame, baseline_oos)

    keep_short_only_oos = side_oos != 1
    gated_h48_oos = gated_component(oos_comp["h48qual"], keep_short_only_oos)
    _, gated_oos = run_full_router(oos_frame, {"h48qual": gated_h48_oos, "zig075": oos_comp["zig075"]}, fee, slip)
    gate_oos_res = report("META-ROUTER (block h48qual LONG)", oos_frame, gated_oos)

    print("\n################ VERDICT ################")
    val_gate_pnl = gate_val_res["with_gate"]["pnl"]
    val_base_pnl = base_val_res["with_gate"]["pnl"]
    oos_gate_pnl = gate_oos_res["with_gate"]["pnl"]
    oos_base_pnl = base_oos_res["with_gate"]["pnl"]
    both_improve = (val_gate_pnl > val_base_pnl) and (oos_gate_pnl > oos_base_pnl)
    print(f"VAL:  gate {val_gate_pnl:+.2f}% vs baseline {val_base_pnl:+.2f}%  -> {'IMPROVED' if val_gate_pnl>val_base_pnl else 'WORSE/EQUAL'}")
    print(f"OOS:  gate {oos_gate_pnl:+.2f}% vs baseline {oos_base_pnl:+.2f}%  -> {'IMPROVED' if oos_gate_pnl>oos_base_pnl else 'WORSE/EQUAL'}")
    print(f"CONSISTENT IMPROVEMENT ACROSS BOTH WINDOWS: {both_improve}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gated_oos.to_csv(OUT_DIR / "oos_ledger_meta_router.csv", index=False)
    gated_val.to_csv(OUT_DIR / "val_ledger_meta_router.csv", index=False)
    import json
    (OUT_DIR / "result.json").write_text(json.dumps({
        "model_id": MODEL_ID,
        "rule": "block h48qual LONG signals; SHORT unaffected; zig075/cash fills the slot instead",
        "val": {"baseline": base_val_res, "meta_router": gate_val_res},
        "oos": {"baseline": base_oos_res, "meta_router": gate_oos_res},
        "consistent_improvement": bool(both_improve),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
