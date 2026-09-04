#!/usr/bin/env python3
"""Permutation feature importance (VAL, single seed, AUC-scored, 5 repeats) for the 2 Homer
candidates at their final confirmed (HORIZON, GAP, K). Reuses compute_permutation_importance
verbatim from research_eth_taker_delta_climax_metalabel_tabpfn_20260829.py (fits ONE TabPFN model
on train, computes baseline AUC on eval_df, then shuffles each eval-set feature column n_repeats
times and re-scores with the SAME fitted model -- no per-feature refit). That function hardcodes
the label column name "hit" -- fires here are built with "hit_plain" (this lineage's own naming),
renamed to "hit" only for this call, no semantic change.

Runs on the GPU server (quant_ai env, CUDA required for TabPFN) via handoff.sh.
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

from research_eth_candidate_pool_raw_lift_check_20260831 import (  # noqa: E402
    kalman_level_and_velocity,
    rolling_zscore,
)
from research_eth_demarker_evidence_signal_lift_check_20260831 import compute_demarker  # noqa: E402
from research_eth_kalman_demarker_gridscreen_20260831 import (  # noqa: E402
    OOS_START,
    VAL_START,
    build_fires,
    load_klines,
)
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import (  # noqa: E402
    FEATURE_COLUMNS,
    build_indicator_frame,
    compute_permutation_importance,
)

REPORT_DIR = ROOT / "tmp/eth_kalman_demarker_permutation_importance_20260831"

SIGNAL_CONFIG = {
    "demarker_extreme": {"horizon": 8, "gap": 12, "K": 0.70},
    "kalman_deviation_meanrev": {"horizon": 12, "gap": 12, "K": 2.5},
}


def log(msg: str) -> None:
    print(f"[kalman_demarker_permutation_importance] {msg}", flush=True)


def run_signal(name: str, klines: pd.DataFrame, ind: pd.DataFrame, trigger_top: pd.Series,
               trigger_bottom, extremeness, feature_cols: list[str]) -> dict:
    cfg = SIGNAL_CONFIG[name]
    fires = build_fires(klines, ind, trigger_top, trigger_bottom, extremeness, feature_cols,
                        cfg["horizon"], cfg["gap"], cfg["K"])
    fires = fires.dropna(subset=feature_cols + ["hit_plain"]).reset_index(drop=True)
    fires = fires.rename(columns={"hit_plain": "hit"})
    ts = fires["timestamp"]
    train = fires.loc[ts < VAL_START].reset_index(drop=True)
    val = fires.loc[(ts >= VAL_START) & (ts < OOS_START)].reset_index(drop=True)
    log(f"\n=== {name} (H={cfg['horizon']}, GAP={cfg['gap']}, K={cfg['K']}) ===")
    log(f"n_train={len(train)}  n_val={len(val)}")

    result = compute_permutation_importance(train, val, feature_cols)
    log(f"baseline VAL AUC (seed {result['seed']}): {result['baseline_auc']:.4f}")
    for row in result["importances"]:
        log(f"  {row['feature']:<22s} importance={row['importance_mean']:+.5f} (std={row['importance_std']:.5f})")
    return {"signal": name, **result}


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    log("loading klines + Tier0 indicator frame...")
    klines = load_klines()
    ind = build_indicator_frame(klines)

    dem = compute_demarker(klines["high"], klines["low"])
    ind_dem = ind.copy()
    ind_dem["dem"] = dem.to_numpy()
    r_dem = run_signal("demarker_extreme", klines, ind_dem, dem >= 0.90, dem <= 0.10,
                       dem.fillna(0.5).to_numpy(), FEATURE_COLUMNS + ["dem"])

    levels, _ = kalman_level_and_velocity(klines["close"].to_numpy())
    kalman_dev = pd.Series((klines["close"].to_numpy() - levels) / levels, index=klines.index)
    kalman_dev_z = rolling_zscore(kalman_dev)
    ind_kal = ind.copy()
    ind_kal["kalman_dev_z"] = kalman_dev_z.to_numpy()
    r_kal = run_signal("kalman_deviation_meanrev", klines, ind_kal, kalman_dev_z >= 2.0, kalman_dev_z <= -2.0,
                       kalman_dev_z.fillna(0.0).to_numpy(), FEATURE_COLUMNS + ["kalman_dev_z"])

    out_path = REPORT_DIR / "permutation_importance_report.json"
    out_path.write_text(json.dumps({"results": [r_dem, r_kal]}, indent=2, default=str))
    log(f"\nreport saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
