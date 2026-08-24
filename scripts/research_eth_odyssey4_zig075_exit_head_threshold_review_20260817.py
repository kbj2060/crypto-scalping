#!/usr/bin/env python3
"""RESEARCH ONLY -- diagnostic, not a promotion candidate. User request: zig075 currently uses
liveATR exit_head at all in live/shadow (its component cfg has no bundle_override -- pure TP/SL,
no time limit, matching the very long real hold observed live, see docs/experiments/
eth_odyssey4_exit_head_tpsl_feature_barrier_mismatch_20260817.md's zig075 table). We already have
a feature-barrier-bug-fixed liveATR retrain for zig075
(tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_20260813_full1500_featurefix/
zig075/true_3head_tabm_bundle.pt) which at the DEFAULT exit_threshold=0.95 raises engagement from
0% to 77.0% but regresses VAL PnL from +40.31% to +6.35% -- a real but unexplained cost. Before
retraining again, this script asks two cheap (no-retrain) questions:

  (1) Trade-level: when this retrained exit_head DOES fire, is it cutting winners short (exiting
      well below MFE, i.e. leaving profit on the table it could have kept holding for) or
      correctly avoiding reversals (exiting near MFE, right before giveback)? Uses the ledger's
      own mfe_price_move/raw_exit_price_move columns.
  (2) Threshold sweep: is exit_threshold=0.95 (the same fixed constant used for h48qual) actually
      well-calibrated for zig075, or does a different threshold trade off engagement vs PnL more
      favorably? sweep.replay_exit_variant takes exit_threshold as a direct parameter -- no
      retraining needed, this is a pure inference-time sweep.

Neither the h48qual/zig075 bundles nor any live/shadow file are touched -- reads pre-existing
artifacts only. fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_odyssey4_zig075_exit_head_threshold_review_20260817"
WINDOW_KEYS = ("val", "oos_q1", "oos_q2", "2025q1", "2025q2", "2025q3")
ZIG075_FEATUREFIX_BUNDLE = ROOT / "tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_20260813_full1500_featurefix/zig075/true_3head_tabm_bundle.pt"
THRESHOLDS = [0.99, 0.97, 0.95, 0.90, 0.85, 0.80, 0.70]


def log(msg: str) -> None:
    print(msg, flush=True)


def _zig075_cfg() -> dict:
    cfg = dict(sweep.COMPONENTS["zig075"])
    cfg["bundle"] = ZIG075_FEATUREFIX_BUNDLE
    return cfg


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    windows = dict(gate.load_all_windows())
    cfg = _zig075_cfg()

    log("=== stage=1 trade-level giveback diagnostic (VAL, exit_threshold=0.95) ===")
    w = windows["val"]
    split = gate.WINDOW_DEFS["val"]["split"]
    aligned_frame, aligned_paths = gate.align_frame_and_predictions(w["frame"], {"zig075": cfg["q_tag"]}, split, OUT_DIR)
    prepped = sweep.prep_component("zig075", cfg, aligned_frame, aligned_paths["zig075"], oof=bool(w["oof"]))
    m95, ledger95 = sweep.replay_exit_variant(
        prepped["frame"], prepped["x"], prepped["dec"], prepped["loaded"],
        risk_margin_fraction=prepped["margin"], risk_leverage=prepped["leverage"],
        exit_threshold=0.95, fee=prepped["fee"], slip=prepped["slip"], cost_mult=sweep.COST_MULT,
        notional_scaled_sltp=prepped["notional_scaled_sltp"], device=sweep.DEVICE,
    )
    log(f"  pnl={m95['pnl']:+.2f}% mdd={m95['mdd']:+.2f}% trades={m95['trades']} reasons={m95['exit_reasons']}")
    by_reason = ledger95.groupby("reason").agg(
        n=("trade_return", "count"), avg_return=("trade_return", "mean"),
        avg_mfe=("mfe_price_move", "mean"), avg_raw_exit=("raw_exit_price_move", "mean"),
        win_rate=("win", "mean"),
    ).round(4)
    log(f"  by reason:\n{by_reason.to_string()}")
    eh = ledger95[ledger95["reason"] == "exit_head"].copy()
    if len(eh):
        eh["giveback_of_mfe"] = (eh["mfe_price_move"] - eh["raw_exit_price_move"]) / eh["mfe_price_move"].clip(lower=1e-8)
        eh_had_profit_left = eh[eh["mfe_price_move"] > 0.0]
        log(f"  exit_head trades with mfe>0: {len(eh_had_profit_left)}/{len(eh)}, "
            f"avg giveback fraction of MFE (when mfe>0): {eh_had_profit_left['giveback_of_mfe'].clip(0, 1).mean():.3f}")
        log(f"  exit_head trades ending in loss (trade_return<0): {(eh['trade_return'] < 0).sum()}/{len(eh)}")

    log("\n=== stage=2 exit_threshold sweep (all 6 windows) ===")
    rows = []
    for window_key in WINDOW_KEYS:
        w = windows[window_key]
        split = gate.WINDOW_DEFS[window_key]["split"]
        aligned_frame, aligned_paths = gate.align_frame_and_predictions(w["frame"], {"zig075": cfg["q_tag"]}, split, OUT_DIR)
        prepped = sweep.prep_component("zig075", cfg, aligned_frame, aligned_paths["zig075"], oof=bool(w["oof"]))
        for thr in THRESHOLDS:
            m, ledger = sweep.replay_exit_variant(
                prepped["frame"], prepped["x"], prepped["dec"], prepped["loaded"],
                risk_margin_fraction=prepped["margin"], risk_leverage=prepped["leverage"],
                exit_threshold=thr, fee=prepped["fee"], slip=prepped["slip"], cost_mult=sweep.COST_MULT,
                notional_scaled_sltp=prepped["notional_scaled_sltp"], device=sweep.DEVICE,
            )
            eh_n = m["exit_reasons"].get("exit_head", 0)
            eh_share = round(eh_n / m["trades"], 4) if m["trades"] else None
            rows.append({"window": window_key, "exit_threshold": thr, "pnl": round(m["pnl"], 2), "mdd": round(m["mdd"], 2),
                         "trades": m["trades"], "exit_head_n": eh_n, "exit_head_share": eh_share,
                         "reasons": m["exit_reasons"]})
            log(f"  {window_key} thr={thr}: pnl={m['pnl']:+7.2f}% mdd={m['mdd']:+7.2f}% trades={m['trades']:3d} "
                f"exit_head={eh_n}({eh_share}) reasons={m['exit_reasons']}")

    log("\n=== SUMMARY (exit_threshold sweep) ===")
    df = pd.DataFrame(rows)
    print(df.drop(columns=["reasons"]).to_string(index=False))
    df.to_csv(OUT_DIR / "zig075_threshold_sweep.csv", index=False)
    log(f"\nwrote {OUT_DIR / 'zig075_threshold_sweep.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
