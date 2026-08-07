#!/usr/bin/env python3
"""RESEARCH ONLY -- fresh-forward VAL-first->OOS funnel for the DP-reversal scale-out classifier
trained by train_eth_omega461_dp_reversal_scaleout_20260721.py (round 10 of the exit-logic
research thread).

Reuses research_eth_omega461_reversal_risk_scaleout_eval_20260721.py's (round 3's) causal replay
loop UNMODIFIED (imported, not copy-pasted) -- the trigger mechanism (reversal classifier
probability, positioned before the exit-head check, partial-notional reduction, fires at most
once/trade) is identical; only the LABEL the classifier was trained on differs (DP-based reversal
vs. round 3's giveback-based reversal), and the classifier's feature schema is identical, so the
same replay_reversal()/run_one()/BASELINES/beats_baseline() apply verbatim. Supports multiple
MAX_AGE horizons (one model dir per max_age, see train script) per task step 4.

Does NOT touch trading_bot_modules/omega4_6_1_live.py, trading_bot.py, runtime_config.py, or
.env.

Baselines (reused unmodified from ideas2 / exit_sweep, NOT recomputed):
  VAL  h48qual: pnl +5.45%  mdd -11.62%
  VAL  zig075:  pnl +40.31% mdd -13.07%
  OOS  h48qual: pnl +9.49%  mdd -6.54%
  OOS  zig075:  pnl +17.89% mdd -11.01%

Fresh-forward discipline: causal bar-by-bar single forward pass; no saved ledger used as input.
"""

from __future__ import annotations

import argparse
import json
import pickle
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
import research_eth_omega461_reversal_risk_scaleout_eval_20260721 as r3  # noqa: E402

OUT_DIR = ROOT / "tmp/research_20260721"
BASELINES = r3.BASELINES
beats_baseline = r3.beats_baseline
replay_reversal = r3.replay_reversal
run_one = r3.run_one
prep_with_proxy = r3.prep_with_proxy
MODEL_DIR = ROOT / "tmp/causal_regen_20260516"


def load_dp_reversal_model(name: str, max_age: int) -> dict[str, Any]:
    with open(MODEL_DIR / f"eth_omega461_dp_reversal_scaleout_20260721_maxage{max_age}_{name}" / "model.pkl", "rb") as f:
        return pickle.load(f)


def run_for_max_age(max_age: int, val_prepped: dict, oos_prepped: dict) -> bool:
    """Runs sanity checks + VAL grid + (if any VAL winners) OOS confirm for one max_age. Returns
    True iff at least one config cleared OOS (i.e. a real winner was found for this horizon)."""
    print(f"===== max_age={max_age} =====", flush=True)
    reversal_models = {name: load_dp_reversal_model(name, max_age) for name in sweep.COMPONENTS}

    # ---------------- Mandatory sanity checks ----------------
    print(f"stage=sanity_checks max_age={max_age}", flush=True)
    sanity_rows = []
    all_sane = True
    for split_name, prepped, base_key in (("VAL", val_prepped, "VAL"), ("OOS", oos_prepped, "OOS")):
        for name, p in prepped.items():
            b = BASELINES[(name, base_key)]
            m_noop, _ = run_one(p)  # reversal_model=None -> fully disabled no-op path
            m_thr101, _ = run_one(
                p, reversal_model=reversal_models[name], proxy_quality=p["proxy_quality"],
                proxy_dir_long=p["proxy_dir_long"], proxy_dir_short=p["proxy_dir_short"],
                reversal_activate_frac=0.7, reversal_prob_thr=1.01, reversal_close_frac=0.5,
            )
            tol = 0.01
            ok_noop = abs(m_noop["pnl"] - b["pnl"]) < tol and abs(m_noop["mdd"] - b["mdd"]) < tol
            ok_thr101_vs_baseline = abs(m_thr101["pnl"] - b["pnl"]) < tol and abs(m_thr101["mdd"] - b["mdd"]) < tol
            ok_thr101_vs_noop = abs(m_thr101["pnl"] - m_noop["pnl"]) < 1e-9 and abs(m_thr101["mdd"] - m_noop["mdd"]) < 1e-9
            ok_thr101 = ok_thr101_vs_baseline and ok_thr101_vs_noop
            all_sane = all_sane and ok_noop and ok_thr101
            sanity_rows.append({
                "max_age": max_age, "split": split_name, "component": name, "baseline_pnl": b["pnl"], "baseline_mdd": b["mdd"],
                "noop_pnl": m_noop["pnl"], "noop_mdd": m_noop["mdd"], "noop_ok": ok_noop,
                "thr101_pnl": m_thr101["pnl"], "thr101_mdd": m_thr101["mdd"], "thr101_ok": ok_thr101,
                "thr101_vs_noop_exact": ok_thr101_vs_noop,
            })
            print(f"  sanity max_age={max_age} split={split_name} component={name} noop_ok={ok_noop} thr101_ok={ok_thr101} "
                  f"thr101_vs_noop_exact={ok_thr101_vs_noop} "
                  f"(noop pnl={m_noop['pnl']:.4f}/mdd={m_noop['mdd']:.4f} thr101 pnl={m_thr101['pnl']:.4f}/mdd={m_thr101['mdd']:.4f} "
                  f"vs baseline pnl={b['pnl']:.4f}/mdd={b['mdd']:.4f})", flush=True)
    sanity_path = OUT_DIR / f"dp_reversal_scaleout_maxage{max_age}_sanity_checks.csv"
    pd.DataFrame(sanity_rows).to_csv(sanity_path, index=False)
    if not all_sane:
        print(f"stage=STOP max_age={max_age} sanity checks FAILED -- see {sanity_path.name}, not proceeding to grid", flush=True)
        return False
    print(f"stage=sanity_checks PASSED max_age={max_age}", flush=True)

    # ---------------- VAL grid ----------------
    print(f"stage=val_grid max_age={max_age}", flush=True)
    val_rows = []
    winners = []
    for name, p in val_prepped.items():
        for act in (0.6, 0.8):
            for thr in (0.5, 0.65, 0.8):
                m, _ = run_one(
                    p, reversal_model=reversal_models[name], proxy_quality=p["proxy_quality"],
                    proxy_dir_long=p["proxy_dir_long"], proxy_dir_short=p["proxy_dir_short"],
                    reversal_activate_frac=act, reversal_prob_thr=thr, reversal_close_frac=0.5,
                )
                row = {"max_age": max_age, "component": name, "activate_frac": act, "prob_thr": thr, "close_frac": 0.5, **m}
                val_rows.append(row)
                cleared = beats_baseline(name, "VAL", m["pnl"], m["mdd"])
                if cleared:
                    winners.append({"component": name, "activate_frac": act, "prob_thr": thr, "close_frac": 0.5})
    val_df = pd.DataFrame(val_rows)
    val_df["exit_reasons"] = val_df["exit_reasons"].apply(json.dumps)
    val_df.to_csv(OUT_DIR / f"dp_reversal_scaleout_maxage{max_age}_VAL.csv", index=False)
    cols = ["component", "activate_frac", "prob_thr", "close_frac", "pnl", "mdd", "trades", "wr", "avg_hold_bars"]
    print(val_df[cols].to_string(index=False), flush=True)
    print(f"\nmax_age={max_age} VAL winners (beat baseline on BOTH pnl and mdd): {len(winners)}", flush=True)
    for w in winners:
        print(f"  {w}", flush=True)

    if not winners:
        print(f"stage=done max_age={max_age} no_val_winners -> no OOS confirmation run", flush=True)
        return False

    # ---------------- OOS confirmation ----------------
    print(f"stage=oos_confirm max_age={max_age}", flush=True)
    oos_rows = []
    any_oos_winner = False
    for w in winners:
        p = oos_prepped[w["component"]]
        m_cand, _ = run_one(
            p, reversal_model=reversal_models[w["component"]], proxy_quality=p["proxy_quality"],
            proxy_dir_long=p["proxy_dir_long"], proxy_dir_short=p["proxy_dir_short"],
            reversal_activate_frac=w["activate_frac"], reversal_prob_thr=w["prob_thr"], reversal_close_frac=w["close_frac"],
        )
        b = BASELINES[(w["component"], "OOS")]
        cleared = beats_baseline(w["component"], "OOS", m_cand["pnl"], m_cand["mdd"])
        any_oos_winner = any_oos_winner or cleared
        row = {"max_age": max_age, **w, "oos_pnl": m_cand["pnl"], "oos_mdd": m_cand["mdd"], "oos_trades": m_cand["trades"],
               "oos_wr": m_cand["wr"], "oos_baseline_pnl": b["pnl"], "oos_baseline_mdd": b["mdd"], "cleared_oos": cleared}
        oos_rows.append(row)
        print(f"  max_age={max_age} {w} -> OOS pnl={m_cand['pnl']:.2f}% mdd={m_cand['mdd']:.2f}% trades={m_cand['trades']} "
              f"(baseline pnl={b['pnl']:.2f}% mdd={b['mdd']:.2f}%) cleared={cleared}", flush=True)
    pd.DataFrame(oos_rows).to_csv(OUT_DIR / f"dp_reversal_scaleout_maxage{max_age}_OOS_confirm.csv", index=False)
    print(f"stage=done max_age={max_age} any_oos_winner={any_oos_winner}", flush=True)
    return any_oos_winner


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-ages", type=int, nargs="+", default=[96])
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    val_frame = sweep.load_frame(sweep.VAL_START, sweep.VAL_END, base_csv=sweep.BASE_2025, wide24_csv=sweep.WIDE24_2025)
    print(f"VAL frame rows={len(val_frame)} range=[{val_frame['timestamp'].min()}, {val_frame['timestamp'].max()}]", flush=True)
    val_prepped = {name: prep_with_proxy(name, cfg, val_frame, sweep.EXT_PRED_DIR, oof=True) for name, cfg in sweep.COMPONENTS.items()}

    oos_frame = sweep.load_frame(sweep.OOS_START, sweep.OOS_END, base_csv=sweep.BASE_2026, wide24_csv=sweep.WIDE24_2026)
    print(f"OOS frame rows={len(oos_frame)} range=[{oos_frame['timestamp'].min()}, {oos_frame['timestamp'].max()}]", flush=True)
    oos_prepped = {name: prep_with_proxy(name, cfg, oos_frame, sweep.EXT_PRED_DIR, oof=False) for name, cfg in sweep.COMPONENTS.items()}

    any_winner = False
    for max_age in args.max_ages:
        won = run_for_max_age(max_age, val_prepped, oos_prepped)
        any_winner = any_winner or won

    print(f"stage=ALL_DONE any_oos_winner_any_horizon={any_winner}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
