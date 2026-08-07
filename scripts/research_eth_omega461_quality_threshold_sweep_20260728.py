#!/usr/bin/env python3
"""RESEARCH ONLY -- entry-frequency sweep, axis 2 (follow-on to the rejected TP/SL floor axis in
research_eth_omega461_tpsl_floor_sweep_20260728.py / research_eth_omega461_tpsl_floor_portfolio_
check_20260728.py). User's actual complaint is "too few live trades"; empirical ETH 5m-bar return
distribution (checked interactively 2026-07-28: only 3.7% of 1-day windows and 10.9% of 2-day
windows see a >=7.5% move) shows the bottleneck is not TP/SL width but ENTRY SIGNAL SPARSITY --
h48qual's nonzero_side rate is ~0.5%, zig075's ~5.7% (see the frozen greedy-router run this
session). Both come from `final_action = where(dir_action!=0 & quality_for_action>=quality_
threshold, dir_action, 0)` -- the frozen pred CSVs already contain the raw `quality_for_action`
score AND the baked-in `final_action` at the ORIGINAL live threshold (h48qual 0.50, zig075 0.75).
This script re-derives final_action at LOWER thresholds directly from the already-saved
quality_for_action/dir_action columns -- no retraining, no re-scoring, just a cheaper replay-time
gate change (same "backtest lever, not training lever" class as the TP/SL floor axis).

VAL-first funnel: full threshold grid on VAL; OOS touched only for configs beating the live
baseline on BOTH pnl and mdd (component-standalone, matching the TP/SL sweep's own methodology --
any winner MUST separately pass the portfolio-level greedy-router check before being trusted,
per this session's TP/SL-axis finding that standalone wins do not reliably transfer to the shared-
position-slot router).

fresh_forward_bar_by_bar=true. trade_ledgers_used_as_input=false.
saved_parent_exit_timestamps_used=false. future_rows_used_for_entry=false.

Does NOT touch trading_bot_modules/omega4_6_1_live.py, trading_bot.py, runtime_config.py, .env.
Research artifact only -- no promotion-gate claim.
"""
from __future__ import annotations

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

import research_eth_omega461_exit_sweep_20260721 as base  # noqa: E402

OUT_DIR = ROOT / "tmp/research_20260728/quality_threshold_sweep"
LIVE_THRESHOLD = {"h48qual": 0.50, "zig075": 0.75}
THRESHOLD_GRID = {
    "h48qual": [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
    "zig075": [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
}


def _rethresholded_pred_csv(cname: str, orig_csv: Path, *, oof: bool, threshold: float) -> Path:
    prefix = "omega1_regime3_expertdq_oof_" if oof else "omega1_regime3_expertdq_"
    df = pd.read_csv(orig_csv)
    dir_action = pd.to_numeric(df[f"{prefix}dir_action"], errors="raise").to_numpy(dtype=np.int64)
    qual = pd.to_numeric(df[f"{prefix}quality_for_action"], errors="raise").to_numpy(dtype=np.float64)
    df[f"{prefix}final_action"] = np.where((dir_action != 0) & (qual >= threshold), dir_action, 0).astype(np.int64)
    df[f"{prefix}quality_threshold"] = float(threshold)
    tag = "oof" if oof else "oos"
    out = OUT_DIR / f"_rethreshold_{cname}_{tag}_{threshold:.2f}.csv"
    df.to_csv(out, index=False)
    return out


def replay(prepped: dict) -> dict:
    m, _ledger = base.replay_exit_variant(
        prepped["frame"], prepped["x"], prepped["dec"], prepped["loaded"],
        risk_margin_fraction=prepped["margin"], risk_leverage=prepped["leverage"],
        exit_threshold=base.BASELINE_EXIT_THRESHOLD, fee=prepped["fee"], slip=prepped["slip"],
        cost_mult=base.COST_MULT, notional_scaled_sltp=prepped["notional_scaled_sltp"], device=base.DEVICE,
    )
    return m


def sanity_check() -> pd.DataFrame:
    val_frame = base.load_frame(base.VAL_START, base.VAL_END, base_csv=base.BASE_2025, wide24_csv=base.WIDE24_2025)
    rows = []
    for cname, cfg in base.COMPONENTS.items():
        orig = base.EXT_PRED_DIR / cname / f"validation_predictions_{cfg['q_tag']}.csv"
        pred = _rethresholded_pred_csv(cname, orig, oof=True, threshold=LIVE_THRESHOLD[cname])
        p = base.prep_component(cname, cfg, val_frame, pred, oof=True)
        m = replay(p)
        rows.append({"component": cname, "threshold": LIVE_THRESHOLD[cname], **m, "exit_reasons": json.dumps(m["exit_reasons"])})
    return pd.DataFrame(rows)


def val_grid() -> pd.DataFrame:
    val_frame = base.load_frame(base.VAL_START, base.VAL_END, base_csv=base.BASE_2025, wide24_csv=base.WIDE24_2025)
    rows = []
    for cname, cfg in base.COMPONENTS.items():
        orig = base.EXT_PRED_DIR / cname / f"validation_predictions_{cfg['q_tag']}.csv"
        for thr in THRESHOLD_GRID[cname]:
            pred = _rethresholded_pred_csv(cname, orig, oof=True, threshold=thr)
            p = base.prep_component(cname, cfg, val_frame, pred, oof=True)
            m = replay(p)
            rows.append({"component": cname, "threshold": thr, **m, "exit_reasons": json.dumps(m["exit_reasons"])})
        print(f"stage=val_grid_done component={cname}", flush=True)
    return pd.DataFrame(rows)


def oos_for(cname: str, threshold: float) -> dict:
    oos_frame = base.load_frame(base.OOS_START, base.OOS_END, base_csv=base.BASE_2026, wide24_csv=base.WIDE24_2026)
    cfg = base.COMPONENTS[cname]
    orig = base.EXT_PRED_DIR / cname / f"oos_predictions_{cfg['q_tag']}.csv"
    pred = _rethresholded_pred_csv(cname, orig, oof=False, threshold=threshold)
    p = base.prep_component(cname, cfg, oos_frame, pred, oof=False)
    return replay(p)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("stage=sanity_check", flush=True)
    san = sanity_check()
    print(san[["component", "threshold", "pnl", "mdd", "trades", "wr"]].to_string(index=False), flush=True)
    san.to_csv(OUT_DIR / "sanity_check_baseline_reproduce.csv", index=False)
    ref_baseline = {"h48qual": (5.454527, -11.619634), "zig075": (40.311321, -13.065727)}
    for _, row in san.iterrows():
        exp_pnl, exp_mdd = ref_baseline[row["component"]]
        if abs(row["pnl"] - exp_pnl) > 0.01 or abs(row["mdd"] - exp_mdd) > 0.01:
            print(f"WARNING sanity check drift for {row['component']}: got pnl={row['pnl']} mdd={row['mdd']}, "
                  f"expected pnl={exp_pnl} mdd={exp_mdd} (data may have drifted since 2026-07-21, see "
                  f"project-eth-omega461-tpsl-floor-portfolio-check-20260728 memory)", flush=True)
    print("stage=sanity_check_done", flush=True)

    print("stage=val_grid", flush=True)
    val = val_grid()
    val.to_csv(OUT_DIR / "val_grid.csv", index=False)
    print(val[["component", "threshold", "pnl", "mdd", "trades", "wr"]].to_string(index=False), flush=True)

    winners = []
    for cname in base.COMPONENTS:
        live_row = san[san["component"] == cname].iloc[0]
        exp_pnl, exp_mdd = float(live_row["pnl"]), float(live_row["mdd"])
        sub = val[val["component"] == cname]
        for _, row in sub.iterrows():
            if row["threshold"] == LIVE_THRESHOLD[cname]:
                continue
            if row["pnl"] > exp_pnl and row["mdd"] >= exp_mdd - 1e-9:
                winners.append((cname, float(row["threshold"]), float(row["pnl"]), float(row["mdd"]), int(row["trades"])))
    print(f"stage=val_winners n={len(winners)}", flush=True)
    for w in winners:
        print(f"  {w}", flush=True)
    pd.DataFrame(winners, columns=["component", "threshold", "val_pnl", "val_mdd", "val_trades"]).to_csv(
        OUT_DIR / "val_winners.csv", index=False)

    if not winners:
        print("stage=done no VAL winners -- OOS not touched", flush=True)
        return 0

    print("stage=oos_confirm", flush=True)
    oos_rows = []
    ref_oos = {"h48qual": (9.494171, -6.537830), "zig075": (17.893271, -11.006707)}
    for cname, thr, _vp, _vm, _vt in winners:
        m = oos_for(cname, thr)
        exp_pnl, exp_mdd = ref_oos[cname]
        confirmed = bool(m["pnl"] > exp_pnl and m["mdd"] >= exp_mdd - 1e-9)
        oos_rows.append({"component": cname, "threshold": thr, **m, "exit_reasons": json.dumps(m["exit_reasons"]),
                          "live_oos_pnl": exp_pnl, "live_oos_mdd": exp_mdd, "confirmed": confirmed})
        print(f"  {cname} threshold={thr}: oos_pnl={m['pnl']:.3f} oos_mdd={m['mdd']:.3f} "
              f"trades={m['trades']} confirmed={confirmed}", flush=True)
    oos_df = pd.DataFrame(oos_rows)
    oos_df.to_csv(OUT_DIR / "oos_confirm.csv", index=False)
    print("stage=done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
