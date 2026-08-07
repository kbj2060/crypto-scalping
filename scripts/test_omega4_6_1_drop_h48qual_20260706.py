"""Disciplined upgrade test: does dropping the h48qual component (zig075-only) improve Omega4.6.1?

Hypothesis (formed from structural reasoning + the component-x-side breakdown showing h48qual is
net-NEGATIVE in BOTH the VAL 2025-10..12 and OOS 2026-01..06 windows, while also holding PRIORITY
in the greedy router and thus preempting zig075's slot): removing h48qual should help on BOTH
windows if the effect is genuine.

Process (Fresh-Forward-aware): VAL is the selection window; OOS is scored ONCE as a pre-registered
comparison. All numbers use the same frozen artifacts / greedy_replay / caps / gate as the live
model. Stored-ledger based -> DIAGNOSTIC research score, not a live-promotion claim.
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


def _metrics(ledger: pd.DataFrame, frame: pd.DataFrame, apply_gate: bool) -> dict:
    if ledger.empty:
        return {"pnl": 0.0, "mdd": 0.0, "trades": 0, "wr": 0.0}
    lg = ledger.copy()
    if apply_gate:
        market = frame[["timestamp", "ou_halflife"]].copy()
        lg["ets"] = pd.to_datetime(lg["entry_timestamp"])
        lg = lg.merge(market.rename(columns={"timestamp": "ets"}), on="ets", how="left")
        hit = lg["ou_halflife"] <= greedy.DURATION_THRESHOLD
        ret = np.where(hit, 0.0, lg["trade_return"].to_numpy())
        active = ~hit
    else:
        ret = lg["trade_return"].to_numpy()
        active = np.ones(len(lg), dtype=bool)
    curve = np.concatenate([[1.0], np.cumprod(1.0 + ret)])
    peak = np.maximum.accumulate(curve)
    dd = curve / np.maximum(peak, 1e-12) - 1.0
    return {"pnl": float((curve[-1] - 1) * 100), "mdd": float(dd.min() * 100),
            "trades": int(active.sum()), "wr": float((ret[active] > 0).mean()) if active.any() else 0.0}


def run_window(name: str, frame: pd.DataFrame, components: dict) -> None:
    fee, slip = omega._load_fee_slip()
    # full model
    orig_priority = greedy.PRIORITY
    greedy.PRIORITY = ("h48qual", "zig075")
    _, full = greedy.greedy_replay(frame, components, fee=fee, slip=slip, cost_mult=retest.COST_MULT, device=retest.DEVICE)
    # zig075-only
    greedy.PRIORITY = ("zig075",)
    _, zonly = greedy.greedy_replay(frame, {"zig075": components["zig075"]}, fee=fee, slip=slip, cost_mult=retest.COST_MULT, device=retest.DEVICE)
    greedy.PRIORITY = orig_priority

    print(f"\n################ {name} ################")
    for lbl, lg in [("FULL (h48qual+zig075)", full), ("zig075-ONLY", zonly)]:
        ng = _metrics(lg, frame, apply_gate=False)
        wg = _metrics(lg, frame, apply_gate=True)
        print(f"  {lbl:24s} no_gate: pnl={ng['pnl']:+7.2f}% mdd={ng['mdd']:+6.2f}% n={ng['trades']:2d} wr={ng['wr']:.3f}  |  "
              f"gate: pnl={wg['pnl']:+7.2f}% mdd={wg['mdd']:+6.2f}% n={wg['trades']:2d} wr={wg['wr']:.3f}")


def main() -> int:
    device = retest.DEVICE

    # ---- VAL 2025-10..12 ----
    val_frame = valmod.load_val_frame()
    val_components = {}
    for cname, cfg in retest.COMPONENTS.items():
        pred = pd.read_csv(valmod.VAL_PRED[cname])
        pred = pred.rename(columns={c: c.replace("_expertdq_oof_", "_expertdq_") for c in pred.columns})
        pred["timestamp"] = pd.to_datetime(pred["timestamp"])
        pred = pred[(pred["timestamp"] >= valmod.START) & (pred["timestamp"] <= valmod.END)].reset_index(drop=True)
        common = val_frame["timestamp"].isin(pred["timestamp"])
        val_frame = val_frame[common].reset_index(drop=True)
        pred = pred[pred["timestamp"].isin(val_frame["timestamp"])].reset_index(drop=True)
        tmp = ROOT / f"tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/_val_{cname}_aligned.csv"
        pred.to_csv(tmp, index=False)
        val_components[cname] = greedy.prepare_component(val_frame, tmp, cfg, device)
    run_window("VAL 2025-10-01..12-31 (SELECTION window)", val_frame, val_components)

    # ---- OOS 2026-01..06 (one-shot) ----
    oos_frame = retest.load_frame_current("2026-01-01", "2026-06-30")
    oos_components = {}
    for cname, cfg in retest.COMPONENTS.items():
        pred_csv = retest.EXT_PRED_DIR / cname / f"oos_predictions_{cfg['q_tag']}.csv"
        oos_components[cname] = greedy.prepare_component(oos_frame, pred_csv, cfg, device)
    run_window("OOS 2026-01-01..06-30 (one-shot confirm)", oos_frame, oos_components)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
