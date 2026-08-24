#!/usr/bin/env python3
"""RESEARCH ONLY -- Ilias #1: build the counterfactual TP/SL-barrier label dataset for h48qual's
real (unmodified) deployed direction/quality gate, TRAIN split only.

Resolves docs/model_contracts/ilias_eth_human_direction_risk_management_contract_20260817.md Open
Issue (b) per docs/experiments/ilias_eth_exit_head_passivity_root_cause_20260817.md's recommendation:
relabel exit_head's target as "does this trade eventually hit stop_loss vs take_profit" (not the
original entry_label_terminal_giveback/liveATR-relabel targets, both of which the diagnosis confirmed
are direction-quality-blind by construction). Uses
research_ilias_eth_adaptive_exit_signal_common_20260817.simulate_private_barrier_trades (see that
module's docstring for the exact causal-label-vs-causal-decision distinction and pitfall-4
circular-logic resolution).

TRAIN split scope note (checked, not assumed): the contract's Dataset Split table cites the nominal
Odyssey4 TRAIN range as 2024-01-01..2025-09-30 (183,936 rows, the full TabM training set). Only the
2025 portion of that range has a materialized OOF prediction CSV in this repo
(tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/h48qual/train_predictions_q050.csv,
verified by direct read: 78,509 rows, 2025-01-01 04:55 .. 2025-09-30 23:55) -- there is no 2024
h48qual OOF prediction file to reuse. This script therefore uses 2025-01-01..2025-09-30 as its label-
construction window: a proper SUBSET of the nominal TRAIN range, never touching VAL (2025-10-01..)
or OOS (2026-01-01..), so the Fresh-Forward/causal-split rule is respected even though it does not
cover the full nominal TRAIN span. Loaded as ONE continuous frame (not the gate module's per-quarter
2025q1/q2/q3 windows) specifically to avoid an artificial trade-truncation-at-quarter-boundary
artifact that would inflate the "truncated, dropped" diagnostic for no real reason.

fresh_forward_bar_by_bar=N/A (offline label construction, not a live decision -- see common module
docstring). trade_ledgers_used_as_input=false. saved_parent_exit_timestamps_used=false.
future_rows_used_for_entry=false (future bars are consumed only to resolve an already-open position's
own terminal barrier, never to decide entry timing).

Does NOT touch trading_bot.py / trading_bot_modules/* / runtime_config.py / .env. Does NOT modify any
imported module. No retraining, no GPU (DEVICE=cpu), conda env quant_ai.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import research_eth_omega461_exit_head_portfolio_asymmetric_20260813 as portfolio  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
import research_ilias_eth_adaptive_exit_signal_common_20260817 as common  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/ilias_eth_adaptive_exit_signal_baseline_20260817"
DEVICE = portfolio.DEVICE
TRAIN_START, TRAIN_END = "2025-01-01", "2025-09-30"


def log(msg: str) -> None:
    common.log("ilias_labels", msg)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = DEVICE
    fee, slip = omega._load_fee_slip()

    log("=== stage=load_train_window (continuous 2025-01-01..2025-09-30) ===")
    frame = sweep.load_frame(TRAIN_START, TRAIN_END, base_csv=sweep.BASE_2025, wide24_csv=sweep.WIDE24_2025)
    frame, n_dropped = gate._drop_route_nan(frame)
    aligned_frame, aligned_paths = gate.align_frame_and_predictions(
        frame, {common.COMPONENT: sweep.COMPONENTS[common.COMPONENT]["q_tag"]}, split="train", out_dir=OUT_DIR,
    )
    log(f"  rows={len(aligned_frame)} range=[{aligned_frame['timestamp'].min()}, {aligned_frame['timestamp'].max()}] route_nan_dropped={n_dropped}")

    log("=== stage=prepare_h48qual_real_model (unmodified direction, deployed asymmetric_tabm_liveatr bundle) ===")
    cfg = gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR[common.COMPONENT]
    comp = portfolio._prepare_component_val(aligned_frame, aligned_paths[common.COMPONENT], cfg, device)

    log("=== stage=simulate_private_barrier_trades (counterfactual, exit_head-free) ===")
    feat_df, diag = common.simulate_private_barrier_trades(aligned_frame, comp, fee=fee, slip=slip, cost_mult=sweep.COST_MULT)
    log(f"  {diag}")

    out_csv = OUT_DIR / "train_labels_h48qual_2025q1q3.csv"
    feat_df.to_csv(out_csv, index=False)
    log(f"wrote {out_csv} rows={len(feat_df)}")

    report = {
        "design": __doc__,
        "fresh_forward_bar_by_bar": "n_a_offline_label_construction",
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "component": common.COMPONENT,
        "label_window": [TRAIN_START, TRAIN_END],
        "feature_columns": common.FEATURE_COLUMNS,
        "diag": diag,
        "out_csv": str(out_csv),
    }
    (OUT_DIR / "labels_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o)),
        encoding="utf-8",
    )
    log(f"report={OUT_DIR / 'labels_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
