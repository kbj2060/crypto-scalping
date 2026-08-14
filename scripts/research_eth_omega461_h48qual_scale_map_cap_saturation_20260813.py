#!/usr/bin/env python3
"""RESEARCH ONLY -- quantify SCALE_MAP/LEVERAGE_CAP/NOTIONAL_CAP saturation for h48qual's SHORT
book, and ablate SCALE_MAP["h48qual_S"] to see if giving the risk sidecar's leverage prediction
room below LEVERAGE_CAP changes VAL PnL/MDD.

Trigger: tonight's raw sizing-bias audit (docs/experiments/eth_val_only_sizing_bias_quantification_
20260813.md, tmp/research_20260813/omega461_val_sizing_bias_quantification/sizing_bias_raw.csv)
found h48qual's raw (pre-SCALE_MAP) sidecar leverage averages ~2.0x for both VAL and OOS shorts.
Omega461LiveAdapter.decide_entry (trading_bot_modules/omega4_6_1_live.py:337-340) multiplies raw
leverage by SCALE_MAP["h48qual_S"]=2.499 BEFORE the LEVERAGE_CAP=5.0 clip -- 2.0 * 2.499 = 4.998,
landing almost exactly on the cap. Since h48qual trades 71-79% short, this raised the concern that
the sizing head's per-trade differentiation is being clipped away for most of the short book before
it reaches the executed trade. That prior sizing-bias script did NOT apply SCALE_MAP or either cap
at all (confirmed by reading research_eth_omega461_exit_sweep_20260721.prep_component/
replay_exit_variant -- neither touches SCALE_MAP, LEVERAGE_CAP, or NOTIONAL_CAP; the live-realistic
transform only exists in trading_bot_modules/omega4_6_1_live.py itself), so this script adds it.

EXACT LIVE TRANSFORM (verified by direct read of omega4_6_1_live.py:330-340, 2026-08-13):
    scale = SCALE_MAP[f"{alias}_{'L' if side>0 else 'S'}"]
    leverage = min(raw_leverage * scale, LEVERAGE_CAP)          # stage 1: leverage-cap clip
    notional = min(margin_fraction * leverage, NOTIONAL_CAP)     # stage 2: notional-cap clip
    leverage = notional / margin_fraction                        # final leverage re-derived
Both margin_fraction and raw_leverage come from the sidecar's selected_mapping sigmoid (identical
math to train_eval_omega4_2_risk_sidecar_20260622._risk_margins/_risk_leverage, which is what
prep_component already calls). Two DIFFERENT caps can bind: LEVERAGE_CAP (stage 1) and NOTIONAL_CAP
(stage 2). Only stage-2 (notional) saturation is truly terminal -- once notional==NOTIONAL_CAP, ALL
per-trade signal (both margin_fraction AND leverage) is erased from the executed trade regardless of
which cap nominally triggered it. Stage-1-only saturation still leaves some differentiation via
margin_fraction (notional = margin_fraction * LEVERAGE_CAP still varies with margin_fraction).

METHOD: reuses research_eth_omega461_exit_sweep_20260721.prep_component() for the RAW per-bar
margin_fraction/leverage arrays (identical sidecar math to the live component), applies the exact
live SCALE_MAP+cap transform above locally, then feeds the CORRECT capped leverage into
replay_exit_variant() -- a full causal bar-by-bar walk, NOT a frozen-ledger reprice. A reprice
(like the sizing-bias script used) would be invalid here: pos_notional/pos_leverage/pos_exposure
are inputs to the exit head (train_eval_omega1_2_tabm_3head_20260603 POS_COLS), so changing
SCALE_MAP changes what the exit head sees mid-trade, which can shift exit timing and therefore
downstream entry timing too. Only h48qual is touched; zig075's SCALE_MAP entries and h48qual_L are
never modified. Matches this harness's existing simplification (shared with every other script that
used prep_component/replay_exit_variant tonight) of replaying each component in isolation, without
the live adapter's cross-component PRIORITY routing or DURATION_THRESHOLD gate -- not fixed here,
inherited and unchanged.

fresh_forward_bar_by_bar=true. trade_ledgers_used_as_input=false. saved_parent_exit_timestamps_used=
false. future_rows_used_for_entry=false. No seeds/retraining -- frozen live artifacts only, SCALE_MAP
is a live-adapter constant per CLAUDE.md Futures Risk Sizing Contract, not a trained parameter.

VAL=2025-10-01..2025-12-31, OOS=2026-01-01..2026-03-31 (same as research_eth_omega461_exit_sweep_
20260721 / the sizing-bias script; one month short of the CLAUDE.md canonical VAL start because no
frozen OOF prediction exists before 2025-10-01, same documented deviation as every sibling tonight).

OOS DISCIPLINE (caller-specified): stage="val" (default) explores candidates on VAL ONLY. stage=
"oos_confirm" runs the CURRENT live baseline (2.499) and exactly ONE caller-chosen winning candidate
on OOS, in a single execution -- never a candidate sweep on OOS. The baseline is included in that one
read purely as fixed, already-deployed context (not a competing candidate being chosen via OOS), so
this remains one look, not multiple-comparison snooping.

Does NOT touch trading_bot_modules/omega4_6_1_live.py, trading_bot.py, runtime_config.py, .env.
Does NOT retrain anything.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import research_eth_omega461_val_sizing_bias_quantification_20260813 as bias  # noqa: E402
from trading_bot_modules.omega4_6_1_live import LEVERAGE_CAP, NOTIONAL_CAP, SCALE_MAP  # noqa: E402

OUT_DIR = ROOT / "tmp/research_20260813/omega461_h48qual_scale_map_cap_saturation"

H48QUAL_L_SCALE = float(SCALE_MAP["h48qual_L"])  # 0.38 -- fixed throughout, never ablated
BASELINE_S_SCALE = float(SCALE_MAP["h48qual_S"])  # 2.499 -- currently live

# Chosen AFTER inspecting this script's own diagnostic quantiles (raw short leverage median/mean
# ~2.0x, see docstring): targets landing at roughly 55/70/85% of LEVERAGE_CAP at the MEDIAN raw
# leverage, so the median trade sits meaningfully below the ceiling (room for the sidecar's
# per-trade score to matter) while above-median trades can still approach/reach it.
CANDIDATE_S_SCALES = [1.35, 1.70, 2.05]

CAP_EPS = 1e-9


def apply_scale_and_caps(raw_leverage: np.ndarray, margin_fraction: np.ndarray, side: np.ndarray,
                          *, scale_long: float, scale_short: float) -> dict[str, np.ndarray]:
    """Reproduces Omega461LiveAdapter.decide_entry (omega4_6_1_live.py:337-340) exactly."""
    side = np.asarray(side, dtype=np.int64)
    scale = np.where(side > 0, float(scale_long), np.where(side < 0, float(scale_short), 1.0))
    pre_lev_cap = raw_leverage * scale
    capped_lev = np.minimum(pre_lev_cap, LEVERAGE_CAP)
    pre_notional_cap = margin_fraction * capped_lev
    final_notional = np.minimum(pre_notional_cap, NOTIONAL_CAP)
    final_leverage = final_notional / np.maximum(margin_fraction, 1e-12)
    return {
        "pre_lev_cap": pre_lev_cap, "capped_lev": capped_lev,
        "pre_notional_cap": pre_notional_cap, "final_notional": final_notional, "final_leverage": final_leverage,
    }


def replay_h48qual(frame: pd.DataFrame, pred_csv: Path, *, oof: bool, scale_short: float) -> dict[str, Any]:
    prepped = sweep.prep_component("h48qual", sweep.COMPONENTS["h48qual"], frame, pred_csv, oof=oof)
    dec = prepped["dec"]
    side_all = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
    raw_leverage = np.asarray(prepped["leverage"], dtype=np.float64)
    margin = np.asarray(prepped["margin"], dtype=np.float64)
    scaled = apply_scale_and_caps(raw_leverage, margin, side_all, scale_long=H48QUAL_L_SCALE, scale_short=scale_short)

    m, ledger = sweep.replay_exit_variant(
        prepped["frame"], prepped["x"], prepped["dec"], prepped["loaded"],
        risk_margin_fraction=margin, risk_leverage=scaled["final_leverage"],
        exit_threshold=sweep.BASELINE_EXIT_THRESHOLD, fee=prepped["fee"], slip=prepped["slip"],
        cost_mult=sweep.COST_MULT, notional_scaled_sltp=prepped["notional_scaled_sltp"], device=sweep.DEVICE,
    )
    if not ledger.empty:
        idx = ledger["entry_signal_i"].to_numpy(dtype=np.int64)
        ledger = ledger.copy()
        ledger["pre_lev_cap"] = scaled["pre_lev_cap"][idx]
        ledger["pre_notional_cap"] = scaled["pre_notional_cap"][idx]
        ledger["lev_cap_bound"] = ledger["pre_lev_cap"] >= (LEVERAGE_CAP - CAP_EPS)
        ledger["notional_cap_bound"] = ledger["pre_notional_cap"] >= (NOTIONAL_CAP - CAP_EPS)

    return {"metrics": m, "ledger": ledger, "scaled": scaled, "side_all": side_all, "raw_leverage": raw_leverage, "margin": margin}


def quantiles(x: np.ndarray) -> dict[str, float]:
    qs = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]
    return {f"q{int(q * 100):02d}": float(np.quantile(x, q)) for q in qs} | {"mean": float(np.mean(x))}


def report_saturation(label: str, res: dict[str, Any]) -> dict[str, Any]:
    ledger = res["ledger"]
    short = ledger[ledger["side"] < 0] if not ledger.empty else ledger
    n_short = int(len(short))
    n_long = int(len(ledger) - n_short) if not ledger.empty else 0
    if n_short > 0:
        frac_lev_cap = float(short["lev_cap_bound"].mean())
        frac_notional_cap = float(short["notional_cap_bound"].mean())
        avg_final_leverage = float(short["leverage"].mean())
        avg_final_notional = float(short["notional"].mean())
    else:
        frac_lev_cap = frac_notional_cap = avg_final_leverage = avg_final_notional = float("nan")

    side_all, raw_leverage, margin = res["side_all"], res["raw_leverage"], res["margin"]
    raw_short_leverage = raw_leverage[side_all < 0]
    raw_short_margin = margin[side_all < 0]

    out = {
        "label": label, "n_short_trades": n_short, "n_long_trades": n_long,
        "frac_short_leverage_cap_bound": frac_lev_cap, "frac_short_notional_cap_bound": frac_notional_cap,
        "avg_final_short_leverage": avg_final_leverage, "avg_final_short_notional": avg_final_notional,
        "raw_short_leverage_quantiles": quantiles(raw_short_leverage) if len(raw_short_leverage) else {},
        "raw_short_margin_quantiles": quantiles(raw_short_margin) if len(raw_short_margin) else {},
        "n_short_decision_bars": int((side_all < 0).sum()),
    }
    print(f"\n=== saturation report: {label} ===", flush=True)
    print(f"  realized short trades: {n_short} (long: {n_long})", flush=True)
    print(f"  fraction leverage-cap-bound (stage1, pre_lev_cap>=LEVERAGE_CAP): {frac_lev_cap:.3f}", flush=True)
    print(f"  fraction notional-cap-bound (stage2, TERMINAL saturation):      {frac_notional_cap:.3f}", flush=True)
    print(f"  avg final delivered leverage/notional for shorts: {avg_final_leverage:.3f}x / {avg_final_notional:.3f}", flush=True)
    print(f"  raw (pre-SCALE_MAP) short leverage quantiles: {out['raw_short_leverage_quantiles']}", flush=True)
    print(f"  raw short margin_fraction quantiles: {out['raw_short_margin_quantiles']}", flush=True)
    return out


def run_val_stage() -> None:
    val_frame = sweep.load_frame(sweep.VAL_START, sweep.VAL_END, base_csv=sweep.BASE_2025, wide24_csv=sweep.WIDE24_2025)
    val_frame = bias._drop_route_nan_gaps(val_frame, label="VAL")
    print(f"VAL frame rows={len(val_frame)} range=[{val_frame['timestamp'].min()}, {val_frame['timestamp'].max()}]", flush=True)
    pred_csv = sweep.EXT_PRED_DIR / "h48qual" / f"validation_predictions_{sweep.COMPONENTS['h48qual']['q_tag']}.csv"

    scales = [BASELINE_S_SCALE, *CANDIDATE_S_SCALES]
    rows = []
    sat_rows = []
    for s in scales:
        label = f"h48qual_S={s}"
        print(f"\nstage=replay_val scale_short={s}", flush=True)
        res = replay_h48qual(val_frame, pred_csv, oof=True, scale_short=s)
        sat_rows.append(report_saturation(label, res))
        m = res["metrics"]
        ledger = res["ledger"]
        short = ledger[ledger["side"] < 0] if not ledger.empty else ledger
        rows.append({
            "scale_short": s, "is_baseline": bool(s == BASELINE_S_SCALE),
            "pnl": m["pnl"], "mdd": m["mdd"], "trades": m["trades"], "wr": m["wr"],
            "long_entries": m["long_entries"], "short_entries": m["short_entries"],
            "avg_notional": m["avg_notional"], "avg_leverage": m["avg_leverage"],
            "short_trades_win_rate": float(short["win"].mean()) if len(short) else float("nan"),
            "short_avg_trade_return_pct": float(short["trade_return"].mean() * 100.0) if len(short) else float("nan"),
            "exit_reasons": json.dumps(m["exit_reasons"]),
        })

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "val_scale_ablation.csv", index=False)
    with open(OUT_DIR / "val_saturation_report.json", "w") as f:
        json.dump(sat_rows, f, indent=2, default=str)

    pd.set_option("display.width", 220)
    print("\n=== VAL ablation summary (h48qual only, h48qual_L fixed at 0.38) ===", flush=True)
    print(df[["scale_short", "is_baseline", "pnl", "mdd", "trades", "short_entries", "wr",
              "short_trades_win_rate", "short_avg_trade_return_pct", "avg_notional", "avg_leverage"]].to_string(index=False), flush=True)
    print("\nstage=val_done", flush=True)


def run_oos_confirm_stage(winning_scale: float) -> None:
    """Single OOS read: CURRENT baseline (2.499, fixed/already-deployed) + the ONE winning
    candidate, in one execution. Not a sweep -- no other candidate is ever evaluated on OOS."""
    oos_frame = sweep.load_frame(sweep.OOS_START, sweep.OOS_END, base_csv=sweep.BASE_2026, wide24_csv=sweep.WIDE24_2026)
    oos_frame = bias._drop_route_nan_gaps(oos_frame, label="OOS")
    print(f"OOS frame rows={len(oos_frame)} range=[{oos_frame['timestamp'].min()}, {oos_frame['timestamp'].max()}]", flush=True)
    pred_csv = sweep.EXT_PRED_DIR / "h48qual" / f"oos_predictions_{sweep.COMPONENTS['h48qual']['q_tag']}.csv"

    scales = [BASELINE_S_SCALE, float(winning_scale)]
    rows = []
    sat_rows = []
    for s in scales:
        label = f"h48qual_S={s}"
        print(f"\nstage=replay_oos_confirm scale_short={s}", flush=True)
        res = replay_h48qual(oos_frame, pred_csv, oof=False, scale_short=s)
        sat_rows.append(report_saturation(label, res))
        m = res["metrics"]
        ledger = res["ledger"]
        short = ledger[ledger["side"] < 0] if not ledger.empty else ledger
        rows.append({
            "scale_short": s, "is_baseline": bool(s == BASELINE_S_SCALE),
            "pnl": m["pnl"], "mdd": m["mdd"], "trades": m["trades"], "wr": m["wr"],
            "long_entries": m["long_entries"], "short_entries": m["short_entries"],
            "avg_notional": m["avg_notional"], "avg_leverage": m["avg_leverage"],
            "short_trades_win_rate": float(short["win"].mean()) if len(short) else float("nan"),
            "short_avg_trade_return_pct": float(short["trade_return"].mean() * 100.0) if len(short) else float("nan"),
            "exit_reasons": json.dumps(m["exit_reasons"]),
        })

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "oos_confirm.csv", index=False)
    with open(OUT_DIR / "oos_confirm_saturation_report.json", "w") as f:
        json.dump(sat_rows, f, indent=2, default=str)

    pd.set_option("display.width", 220)
    print("\n=== OOS confirmation (ONE read, baseline vs single winning candidate) ===", flush=True)
    print(df[["scale_short", "is_baseline", "pnl", "mdd", "trades", "short_entries", "wr",
              "short_trades_win_rate", "short_avg_trade_return_pct", "avg_notional", "avg_leverage"]].to_string(index=False), flush=True)
    print("\nstage=oos_confirm_done", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["val", "oos_confirm"], default="val")
    ap.add_argument("--winning-scale", type=float, default=None, help="required for --stage oos_confirm")
    args = ap.parse_args()

    if args.stage == "val":
        run_val_stage()
    else:
        if args.winning_scale is None:
            raise SystemExit("--winning-scale is required for --stage oos_confirm")
        run_oos_confirm_stage(args.winning_scale)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
