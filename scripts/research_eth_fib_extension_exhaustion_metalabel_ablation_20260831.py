#!/usr/bin/env python3
"""Group ablation for fib_extension_exhaustion final (H=20/GAP=18/K=2.35). Checks the same 3 groups
as every other signal in this project: volatility-regime features, session-timing features, and
this signal's own-definition variable exposed as a Tier0 feature (is_bottom -- the leg-direction/
fib-extension-zone condition itself isn't a model feature, same caveat pattern as dalton/
smt_divergence)."""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import FEATURE_COLUMNS, run_tabpfn_panel
from research_eth_fib_extension_exhaustion_metalabel_tabpfn_20260831 import HOLDOUT_START, VAL_START, split_train_val_oos

FIRES_CSV = ROOT / "data/labels/eth_5m_fib_extension_exhaustion_metalabel_20260831/eth_5m_fib_extension_exhaustion_metalabel_FINAL_features.csv"
REPORT_DIR = ROOT / "tmp/eth_fib_extension_exhaustion_metalabel_tabpfn_20260831"

VOL_REGIME_FEATURES = ["atr_pct", "atr_percentile_864", "realized_vol_ratio", "bb_width_pctile"]
SESSION_TIMING_FEATURES = ["nyse_open_flag", "hour_utc", "weekday"]
OWN_SIGNAL_FEATURES = ["is_bottom"]

ABLATIONS = {
    "full_23_features": FEATURE_COLUMNS,
    "ablated_no_vol_regime": [f for f in FEATURE_COLUMNS if f not in VOL_REGIME_FEATURES],
    "ablated_no_session_timing": [f for f in FEATURE_COLUMNS if f not in SESSION_TIMING_FEATURES],
    "ablated_no_own_signal_var": [f for f in FEATURE_COLUMNS if f not in OWN_SIGNAL_FEATURES],
}


def log(msg): print(f"[fib_ext_ablation] {msg}", flush=True)


def main() -> int:
    fires = pd.read_csv(FIRES_CSV, parse_dates=["timestamp"])
    train, val, oos = split_train_val_oos(fires)
    holdout = fires.loc[fires["timestamp"] >= HOLDOUT_START].reset_index(drop=True)
    log(f"TRAIN={len(train)} VAL={len(val)} OOS={len(oos)} HOLDOUT={len(holdout)}")

    results = {}
    for label, feats in ABLATIONS.items():
        log(f"=== {label} ({len(feats)} features) ===")
        val_r = run_tabpfn_panel(train, val, feats, f"{label}/VAL")
        oos_r = run_tabpfn_panel(train, oos, feats, f"{label}/OOS")
        holdout_r = run_tabpfn_panel(train, holdout, feats, f"{label}/HOLDOUT")
        results[label] = {"feature_columns": feats, "val": val_r, "oos": oos_r, "holdout": holdout_r}
        log(f"  {label}: VAL={val_r['auc_mean']:.4f} OOS={oos_r['auc_mean']:.4f} HOLDOUT={holdout_r['auc_mean']:.4f}")

    out_path = REPORT_DIR / "ablation_report.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    log(f"saved -> {out_path}")

    log("\n=== SUMMARY ===")
    for label, r in results.items():
        log(f"  {label}: VAL={r['val']['auc_mean']:.4f} OOS={r['oos']['auc_mean']:.4f} HOLDOUT={r['holdout']['auc_mean']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
