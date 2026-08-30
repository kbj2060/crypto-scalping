#!/usr/bin/env python3
"""FINAL model + required validations for smt_divergence (Homer signal #7). Label design locked in
via research_eth_smt_divergence_metalabel_tabpfn_20260831.py (27-combo grid, HOLDOUT untouched) +
research_eth_smt_divergence_metalabel_horizon_extend_20260831.py (grid extended to H=60-96 after
finding min(VAL,OOS) still climbing at the original H=48 boundary -- a K_GRID ceiling bug (capped
at 3.50) was found and fixed mid-extension, since H=60+ needs K>3.50 for a true 50/50 split) +
research_eth_smt_divergence_h72_gap_confirm_20260831.py (confirmed GAP=12 still beats 3/6 at the
new winning horizon, extending its 9/9 dominance to 10/10).

FINAL: HORIZON=72(6h)/GAP=12/K=4.20 touch-based MFE, plain single-K (NOT exclude-middle) -- the
ambiguous-middle concentration check (same methodology that found orthogonal_combo's problem) was
re-run at this exact winning combo: NO_HIT clear-miss fraction = 24.8%, squarely in the healthy
18-32% band from taker/short_term_return_z/volume_wick_climax/dalton, not the 9.2% orthogonal_combo
outlier -- exclude-middle is NOT warranted here (user explicitly asked whether the 50/50 target was
sacrificing data quality; checked directly rather than assumed).

K is calibrated TRAIN-only throughout (proactive fix per the cross-signal K-calibration audit that
found taker's K went stale after an unrelated clustering-parameter change).

Runs on the GPU server (quant_ai env, CUDA required for TabPFN). This is the ONLY script in this
signal's pipeline that touches HOLDOUT -- single touch, after the full label design (HORIZON/GAP/K/
exclude-middle-or-not) was already locked in by the screening scripts above.
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
from research_eth_smt_divergence_metalabel_tabpfn_20260831 import (
    HOLDOUT_START, OOS_START, VAL_START, split_train_val_oos,
)

FIRES_CSV = ROOT / "data/labels/eth_5m_smt_divergence_metalabel_20260831/eth_5m_smt_divergence_metalabel_features.csv"
REPORT_DIR = ROOT / "tmp/eth_smt_divergence_metalabel_tabpfn_20260831"
HORIZON, GAP, K = 72, 12, 4.20


def log(msg: str) -> None:
    print(f"[smt_divergence_final] {msg}", flush=True)


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fires = pd.read_csv(FIRES_CSV, parse_dates=["timestamp"])
    log(f"loaded {len(fires)} fires (H={HORIZON}/GAP={GAP}/K={K}) from {FIRES_CSV}")

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
        "signal": "smt_divergence", "adopted_version": "v1",
        "horizon": HORIZON, "gap": GAP, "k": K,
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
