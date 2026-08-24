#!/usr/bin/env python3
"""RESEARCH ONLY -- diagnostic. Answers: at the CURRENT (undisturbed) TP/SL floor, how often does
h48qual's exit_head actually fire, and by how much does it shorten holding time compared to
trades that get closed by take_profit/stop_loss instead? Uses the currently-deployed
NEW_H48QUAL_BUNDLE, component-only replay (sweep.prep_component + sweep.replay_exit_variant, same
methodology as research_eth_omega461_exit_head_h48cons_relabel_20260813._evaluate_val), across
all 6 standard windows. No floor/model changes -- pure measurement."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import research_eth_omega461_exit_head_portfolio_asymmetric_20260813 as portfolio  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_odyssey4_exit_head_hold_time_reduction_20260817"
WINDOW_KEYS = ("val", "oos_q1", "oos_q2", "2025q1", "2025q2", "2025q3")


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    windows = dict(gate.load_all_windows())
    cfg = portfolio._component_cfg("h48qual", bundle_override=portfolio.NEW_H48QUAL_BUNDLE)

    rows = []
    for window_key in WINDOW_KEYS:
        w = windows[window_key]
        split = gate.WINDOW_DEFS[window_key]["split"]
        aligned_frame, aligned_paths = gate.align_frame_and_predictions(w["frame"], {"h48qual": cfg["q_tag"]}, split, OUT_DIR)
        prepped = sweep.prep_component("h48qual", cfg, aligned_frame, aligned_paths["h48qual"], oof=bool(w["oof"]))
        m, ledger = sweep.replay_exit_variant(
            prepped["frame"], prepped["x"], prepped["dec"], prepped["loaded"],
            risk_margin_fraction=prepped["margin"], risk_leverage=prepped["leverage"],
            exit_threshold=sweep.BASELINE_EXIT_THRESHOLD, fee=prepped["fee"], slip=prepped["slip"],
            cost_mult=sweep.COST_MULT, notional_scaled_sltp=prepped["notional_scaled_sltp"], device=sweep.DEVICE,
        )
        ledger = ledger.copy()
        ledger["hold_bars"] = ledger["exit_i"].astype(int) - ledger["entry_i"].astype(int)
        n = len(ledger)
        overall_avg_hold = float(ledger["hold_bars"].mean()) if n else float("nan")
        by_reason = ledger.groupby("reason")["hold_bars"].agg(["count", "mean"]).round(1)
        exit_head_n = int((ledger["reason"] == "exit_head").sum())
        exit_head_share = round(exit_head_n / n, 4) if n else None
        non_exit_head_avg_hold = float(ledger.loc[ledger["reason"] != "exit_head", "hold_bars"].mean()) if (ledger["reason"] != "exit_head").any() else float("nan")
        exit_head_avg_hold = float(ledger.loc[ledger["reason"] == "exit_head", "hold_bars"].mean()) if exit_head_n else float("nan")

        log(f"=== {window_key}: n_trades={n} exit_head={exit_head_n}({exit_head_share:.1%}) "
            f"overall_avg_hold={overall_avg_hold:.1f}bar  exit_head_avg_hold={exit_head_avg_hold:.1f}bar  "
            f"non_exit_head_avg_hold={non_exit_head_avg_hold:.1f}bar ===")
        log(f"  by reason:\n{by_reason.to_string()}")

        rows.append({"window": window_key, "n_trades": n, "exit_head_n": exit_head_n, "exit_head_share": exit_head_share,
                      "overall_avg_hold_bars": round(overall_avg_hold, 1),
                      "exit_head_avg_hold_bars": round(exit_head_avg_hold, 1) if exit_head_n else None,
                      "non_exit_head_avg_hold_bars": round(non_exit_head_avg_hold, 1)})

    log("\n=== SUMMARY ===")
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(OUT_DIR / "hold_time_summary.csv", index=False)
    log(f"\nwrote {OUT_DIR / 'hold_time_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
