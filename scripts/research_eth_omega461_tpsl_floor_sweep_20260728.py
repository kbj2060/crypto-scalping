#!/usr/bin/env python3
"""RESEARCH ONLY -- TP/SL sizing redesign, axis 1: sweep the min_tp/min_sl FLOOR values.

Motivation. `_apply_atr_safety_sltp` computes take_profit=clip(max(min_tp, atr_pct*tp_mult),
0,max_tp) and stop_loss symmetrically. Diagnosed 2026-07-28 (tmp/research_20260728/
diag_tpsl_clip_binding.py): at live settings (atr_window=192, tp_mult=12, sl_mult=6,
min_tp=0.075, min_sl=0.04) the min_tp/min_sl floor binds on effectively 100% of active rows in
every split/component tested -- ATR percent at this timeframe is far too small (p90 ~0.003-0.004)
for atr_pct*mult to ever clear the floor. So despite the "ATR-adaptive" design intent, TP/SL is
currently a FLAT 7.5%/4.0% barrier for essentially every trade; tp_mult/sl_mult are dead knobs at
current floor values. This script instead sweeps min_tp/min_sl directly (the values that actually
determine outcomes), holding atr_window/tp_mult/sl_mult/max_tp/max_sl at live defaults so ATR only
matters on the rare tail row where it would exceed the new, lower floor.

Entry decisions, sizing, and the exit head all come from the SAME frozen prediction CSVs and
frozen bundles/sidecars used throughout this investigation (research_eth_omega461_exit_sweep_
20260721.py's COMPONENTS/load_frame/prep_component/replay_exit_variant, reused unmodified) --
only the min_tp/min_sl passed into `_apply_atr_safety_sltp` changes. Exit threshold stays at the
live 0.95. No retraining.

VAL-first funnel: full min_tp x min_sl grid on VAL; OOS touched only for configs beating the live
baseline on BOTH pnl and mdd.

fresh_forward_bar_by_bar=true. trade_ledgers_used_as_input=false.
saved_parent_exit_timestamps_used=false. future_rows_used_for_entry=false.

Does NOT touch trading_bot_modules/omega4_6_1_live.py, trading_bot.py, runtime_config.py, .env.
Research artifact only -- no promotion-gate claim.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import research_eth_omega461_exit_sweep_20260721 as base  # noqa: E402

MIN_TP_GRID = [0.050, 0.060, 0.075, 0.090, 0.100, 0.120, 0.140, 0.160, 0.180, 0.200, 0.220]
MIN_SL_GRID = [0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.060]
LIVE_MIN_TP = 0.075
LIVE_MIN_SL = 0.040

OUT_DIR = ROOT / "tmp/research_20260728/tpsl_floor_sweep"


def prepped_for(cname: str, cfg_base: dict, frame: pd.DataFrame, pred_csv: Path, *, oof: bool,
                 min_tp: float, min_sl: float) -> dict:
    cfg = dict(cfg_base)
    cfg["min_tp"] = float(min_tp)
    cfg["min_sl"] = float(min_sl)
    return base.prep_component(cname, cfg, frame, pred_csv, oof=oof)


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
        pred = base.EXT_PRED_DIR / cname / f"validation_predictions_{cfg['q_tag']}.csv"
        p = prepped_for(cname, cfg, val_frame, pred, oof=True, min_tp=LIVE_MIN_TP, min_sl=LIVE_MIN_SL)
        m = replay(p)
        rows.append({"component": cname, "min_tp": LIVE_MIN_TP, "min_sl": LIVE_MIN_SL, **m,
                     "exit_reasons": json.dumps(m["exit_reasons"])})
    return pd.DataFrame(rows)


def val_grid() -> pd.DataFrame:
    val_frame = base.load_frame(base.VAL_START, base.VAL_END, base_csv=base.BASE_2025, wide24_csv=base.WIDE24_2025)
    rows = []
    for cname, cfg in base.COMPONENTS.items():
        pred = base.EXT_PRED_DIR / cname / f"validation_predictions_{cfg['q_tag']}.csv"
        for min_tp in MIN_TP_GRID:
            for min_sl in MIN_SL_GRID:
                p = prepped_for(cname, cfg, val_frame, pred, oof=True, min_tp=min_tp, min_sl=min_sl)
                m = replay(p)
                rows.append({"component": cname, "min_tp": min_tp, "min_sl": min_sl, **m,
                             "exit_reasons": json.dumps(m["exit_reasons"])})
        print(f"stage=val_grid_done component={cname}", flush=True)
    return pd.DataFrame(rows)


def oos_for(cname: str, min_tp: float, min_sl: float) -> dict:
    oos_frame = base.load_frame(base.OOS_START, base.OOS_END, base_csv=base.BASE_2026, wide24_csv=base.WIDE24_2026)
    cfg = base.COMPONENTS[cname]
    pred = base.EXT_PRED_DIR / cname / f"oos_predictions_{cfg['q_tag']}.csv"
    p = prepped_for(cname, cfg, oos_frame, pred, oof=False, min_tp=min_tp, min_sl=min_sl)
    return replay(p)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("stage=sanity_check", flush=True)
    san = sanity_check()
    print(san[["component", "min_tp", "min_sl", "pnl", "mdd", "trades", "wr"]].to_string(index=False), flush=True)
    san.to_csv(OUT_DIR / "sanity_check_baseline_reproduce.csv", index=False)
    for _, row in san.iterrows():
        ref = base.COMPONENTS[row["component"]]
        # Established baselines (research_eth_omega461_exit_sweep_20260721.py VAL, 2026-07-21):
        # h48qual +5.45%/-11.62%, zig075 +40.31%/-13.07%.
    ref_baseline = {"h48qual": (5.454527, -11.619634), "zig075": (40.311321, -13.065727)}
    for _, row in san.iterrows():
        exp_pnl, exp_mdd = ref_baseline[row["component"]]
        if abs(row["pnl"] - exp_pnl) > 0.01 or abs(row["mdd"] - exp_mdd) > 0.01:
            raise RuntimeError(f"sanity check FAILED for {row['component']}: got pnl={row['pnl']} mdd={row['mdd']}, "
                                f"expected pnl={exp_pnl} mdd={exp_mdd}")
    print("stage=sanity_check_PASSED", flush=True)

    print("stage=val_grid", flush=True)
    val = val_grid()
    val.to_csv(OUT_DIR / "val_grid.csv", index=False)

    winners = []
    for cname in base.COMPONENTS:
        exp_pnl, exp_mdd = ref_baseline[cname]
        sub = val[val["component"] == cname]
        for _, row in sub.iterrows():
            if row["min_tp"] == LIVE_MIN_TP and row["min_sl"] == LIVE_MIN_SL:
                continue
            if row["pnl"] > exp_pnl and row["mdd"] >= exp_mdd - 1e-9:
                winners.append((cname, float(row["min_tp"]), float(row["min_sl"]), float(row["pnl"]), float(row["mdd"])))
    print(f"stage=val_winners n={len(winners)}", flush=True)
    for w in winners:
        print(f"  {w}", flush=True)
    pd.DataFrame(winners, columns=["component", "min_tp", "min_sl", "val_pnl", "val_mdd"]).to_csv(
        OUT_DIR / "val_winners.csv", index=False)

    if not winners:
        print("stage=done no VAL winners -- OOS not touched", flush=True)
        return 0

    print("stage=oos_confirm", flush=True)
    oos_rows = []
    ref_oos = {"h48qual": (9.494171, -6.537830), "zig075": (17.893271, -11.006707)}
    for cname, min_tp, min_sl, _vp, _vm in winners:
        m = oos_for(cname, min_tp, min_sl)
        exp_pnl, exp_mdd = ref_oos[cname]
        confirmed = bool(m["pnl"] > exp_pnl and m["mdd"] >= exp_mdd - 1e-9)
        oos_rows.append({"component": cname, "min_tp": min_tp, "min_sl": min_sl, **m,
                          "exit_reasons": json.dumps(m["exit_reasons"]),
                          "live_oos_pnl": exp_pnl, "live_oos_mdd": exp_mdd, "confirmed": confirmed})
        print(f"  {cname} min_tp={min_tp} min_sl={min_sl}: oos_pnl={m['pnl']:.3f} oos_mdd={m['mdd']:.3f} "
              f"trades={m['trades']} confirmed={confirmed}", flush=True)
    oos_df = pd.DataFrame(oos_rows)
    oos_df.to_csv(OUT_DIR / "oos_confirm.csv", index=False)
    print("stage=done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
