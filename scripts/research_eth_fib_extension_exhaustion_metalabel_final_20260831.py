#!/usr/bin/env python3
"""FINAL model + required validations for fib_extension_exhaustion (Homer signal #8, last
remaining). Label design locked in via research_eth_fib_extension_exhaustion_metalabel_tabpfn_
20260831.py (27-combo grid, HOLDOUT untouched; also fixed a K-calibration bimodality bug -- the
joint MFE/MAE hit_rate(K) curve has an interior peak and crosses 0.5 on both the rising AND
falling branch, and the original unfixed search picked whichever was numerically closest without
a tie-break, landing on a degenerate low-K solution for H=6/GAP=12 with OOS AUC=0.3885; fixed by
always taking the declining (post-peak, larger-K) branch) + research_eth_fib_extension_exhaustion_
gap_extend_confirm_20260831.py (GAP=12 was the grid's largest tested value and won 6/9 horizons;
checked GAP in {18,24} at H={16,20,24} -- GAP=18 edges out GAP=12 at H=20 (min(VAL,OOS) 0.6062 vs
0.6044, a noise-level ~0.3% relative gain) while GAP=24 is worse (0.6001), confirming GAP=18 is a
genuine (if shallow) interior peak, no further extension needed).

FINAL: HORIZON=20(100min)/GAP=18/K=2.35/K_loss=4.70 (K_LOSS_MULT=2.0).

**2026-08-31 user catch, unique to this signal**: the hit label is NOT plain touch-based MFE like
every other signal in this project -- a real example was found where MFE touched the profit target
(+2.54xATR) but the SAME horizon window later saw a -6.58xATR adverse crash, which a real position
might not have safely avoided (slippage/latency in a violent move). Redefined hit as a whole-window,
order-blind joint condition: MFE>=K AND MAE<K_LOSS_MULT*K, both measured over the full
[i+1,i+HORIZON] window regardless of order. K_LOSS_MULT=2.0 chosen after sweeping {1.0,1.5,2.0,3.0}
on a placeholder config (1.0x/symmetric cannot reach 50/50 balance at all, caps at 43.8%; 2.0x
reaches ~50% while disqualifying the most extreme 2-3% of former hits) -- confirmed via a
regenerated 20-example visual-verification chart showing the flipped whipsaw-into-crash cases
clearly separated from the clean remaining HIT population; user-approved.

Ambiguous-middle concentration check (same methodology that found orthogonal_combo's problem,
adapted here to only look at the "plain miss" (MFE<K) subset of NO_HIT, since MAE-flipped cases are
unambiguous by construction) re-run at this exact winning combo: clear-miss fraction = 23.8%,
squarely in the healthy 18-32% band -- no further redesign needed.

K is calibrated TRAIN-only throughout (proactive, per the cross-signal K-calibration audit).

Runs on the GPU server (quant_ai env, CUDA required for TabPFN). This is the ONLY script in this
signal's pipeline that touches HOLDOUT -- single touch, after the full label design was already
locked in by the screening scripts above.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd

from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import (
    FEATURE_COLUMNS, compute_permutation_importance, run_tabpfn_panel,
)
from research_eth_fib_extension_exhaustion_metalabel_tabpfn_20260831 import (
    HOLDOUT_START, OOS_START, VAL_START, K_LOSS_MULT, split_train_val_oos,
)

FIRES_CSV = ROOT / "data/labels/eth_5m_fib_extension_exhaustion_metalabel_20260831/eth_5m_fib_extension_exhaustion_metalabel_FINAL_features.csv"
REPORT_DIR = ROOT / "tmp/eth_fib_extension_exhaustion_metalabel_tabpfn_20260831"
HORIZON, GAP, K = 20, 18, 2.35


def log(msg: str) -> None:
    print(f"[fib_ext_final] {msg}", flush=True)


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fires = pd.read_csv(FIRES_CSV, parse_dates=["timestamp"])
    log(f"loaded {len(fires)} fires (H={HORIZON}/GAP={GAP}/K={K}/K_loss={K_LOSS_MULT*K}) from {FIRES_CSV}")

    train, val, oos = split_train_val_oos(fires)
    holdout = fires.loc[fires["timestamp"] >= HOLDOUT_START].reset_index(drop=True)
    log(f"TRAIN(<{VAL_START.date()}) n={len(train)}, VAL n={len(val)}, OOS n={len(oos)}, "
        f"HOLDOUT(>={HOLDOUT_START.date()}) n={len(holdout)}")
    log(f"hit_rate: TRAIN={train['hit'].mean():.4f} VAL={val['hit'].mean():.4f} "
        f"OOS={oos['hit'].mean():.4f} HOLDOUT={holdout['hit'].mean():.4f}")

    log("=== VAL evaluation (TRAIN-fit, 4 seeds) ===")
    val_result = run_tabpfn_panel(train, val, FEATURE_COLUMNS, "VAL")
    log(f"VAL -> AUC {val_result['auc_mean']:.4f}+/-{val_result['auc_std']:.4f}")

    log("=== OOS evaluation (TRAIN-fit, 4 seeds) ===")
    oos_result = run_tabpfn_panel(train, oos, FEATURE_COLUMNS, "OOS")
    log(f"OOS -> AUC {oos_result['auc_mean']:.4f}+/-{oos_result['auc_std']:.4f}")

    log("=== RESERVED HOLDOUT evaluation (SINGLE TOUCH, TRAIN-fit, 4 seeds) ===")
    holdout_result = run_tabpfn_panel(train, holdout, FEATURE_COLUMNS, "HOLDOUT") if len(holdout) >= 30 else {"note": "too few holdout fires"}
    if "auc_mean" in holdout_result:
        log(f"HOLDOUT -> AUC {holdout_result['auc_mean']:.4f}+/-{holdout_result['auc_std']:.4f}")

    log("=== permutation feature importance (VAL, single seed, 5 repeats) ===")
    perm_importance = compute_permutation_importance(train, val, FEATURE_COLUMNS)
    for row in perm_importance["importances"][:10]:
        log(f"  {row['feature']:<22s} importance={row['importance_mean']:+.5f}")

    report = {
        "signal": "fib_extension_exhaustion", "adopted_version": "v1",
        "horizon": HORIZON, "gap": GAP, "k": K, "k_loss_mult": K_LOSS_MULT, "k_loss": K_LOSS_MULT * K,
        "n_train": len(train), "n_val": len(val), "n_oos": len(oos), "n_holdout": len(holdout),
        "feature_columns": FEATURE_COLUMNS,
        "val": val_result, "oos": oos_result, "reserved_holdout": holdout_result,
        "permutation_importance_val": perm_importance,
    }
    out_path = REPORT_DIR / "final_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    log(f"final report saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
