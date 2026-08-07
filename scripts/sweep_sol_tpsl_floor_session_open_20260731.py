"""Sweep TP/SL floor width for the session-open SOL candidate's risk-sidecar
chain, to find where VAL/OOS PnL peaks. Motivated by 2026-07-30/31 finding
that going from narrow TP/SL (parent-only default 0.026/0.014) to the live
floor (0.075/0.040) collapsed VAL PnL from +11.34% to +0.59% (before dynamic
sizing) -- confirms atr_pct_p50/p90 (~0.28-0.40%) is always well under the
floor, so min-tp/min-sl set the realized barrier for essentially all trades
(tp_p50==tp_p90==floor in every prior run). Same recipe as
train_eval_omega4_2_risk_sidecar_sol_20260707.py's session-open run (hgb,
parent_outputs, side-split-model, dynamic-leverage, log_risk objective,
validation_only scope, mdd floor 8.0), varying only --min-tp/--min-sl.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/train_eval_omega4_2_risk_sidecar_sol_20260707.py"
PRECOMP_DIR = ROOT / "tmp/causal_regen_20260516/sol_omega4_3head_parent72_loose_entry_quality_20260707_session_open_20260730"

# (min_tp, min_sl) pairs -- keep roughly the live ~1.875:1 tp:sl ratio.
WIDTHS = [
    (0.030, 0.016),
    (0.045, 0.024),
    (0.060, 0.032),
    (0.075, 0.040),  # current live floor, re-run for exact same-methodology reference
    (0.090, 0.048),
    (0.110, 0.060),
]


def main() -> int:
    for min_tp, min_sl in WIDTHS:
        suffix = f"sltp_sweep_tp{int(min_tp*1000):03d}_sl{int(min_sl*1000):03d}_20260731"
        cmd = [
            sys.executable, str(SCRIPT),
            "--precomputed-prediction-dir", str(PRECOMP_DIR),
            "--precomputed-prediction-tag", "q070",
            "--quality-threshold", "0.70",
            "--min-tp", str(min_tp),
            "--min-sl", str(min_sl),
            "--model-kind", "hgb",
            "--risk-feature-mode", "parent_outputs",
            "--side-split-model",
            "--dynamic-leverage",
            "--selection-objective", "log_risk",
            "--selection-scope", "validation_only",
            "--max-validation-mdd-abs", "8.0",
            "--out-suffix", suffix,
            "--device", "cuda",
        ]
        print(f"\n=== min_tp={min_tp} min_sl={min_sl} out_suffix={suffix} ===", flush=True)
        result = subprocess.run(cmd, cwd=str(ROOT))
        print(f"=== exit code: {result.returncode} ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
