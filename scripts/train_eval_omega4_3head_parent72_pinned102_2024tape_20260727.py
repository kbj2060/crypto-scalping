"""RESEARCH ONLY -- pinned102 retrain, but sourcing TRAIN from the recovered 2024+2025 tape
instead of today's 2025-only omega.TRAIN_CSV, so it can reproduce the live checkpoint's exact
183,936-row training window (2024-01-01..2025-09-30) instead of the 78,509-row 2025-only subset
train_eval_omega4_3head_parent72_pinned102_20260727.py was limited to.

Provenance of the 2024+2025 tape (verified 2026-07-27, see
tmp/research_20260727/find_2024_train_source/): both live label-contract dirs' "*_2025.csv"
files are misleadingly named -- they actually span 2024-01-01..2025-12-31 already. The one
missing link was the base candidate CSV; found at
tmp/causal_regen_20260516/omega_clean_regime_only_24_25_inputs_20260629/
trade_candidates_2024_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv (built 2026-06-29, same
day/dir as the live zig075 parent and the matching 2024+2025 regime overlays). Row-count identity
with the live checkpoint is exact: split at SPLIT_TS=2025-10-01 gives train=183,936, val=26,496,
both matching live's label_quality_summary with delta 0. The "regime4" in the filename is a
naming artifact of the source batch, not a description of its surviving columns -- its own
current_only_feature_contract_report.json (same dir) shows regime4_pred_*/clean_regime4_*
columns were explicitly DROPPED (93 of them) when this 102-column contract was built; none
remain in the tape used here.

Layers on top of train_eval_omega4_3head_parent72_pinned102_20260727.py's base_cols pin (returns
the live 102-column contract verbatim, fail-fast if any column is absent). Two further additions:
  1. omega.TRAIN_CSV and the three REGIME3_*_2025 overlay paths point at the 2024_2025 versions
     instead of the 2025-only ones (EVAL_CSV / 2026 overlays untouched -- 2026 side unaffected).
  2. A full replacement of omega._load_omega_frames (not just the 7-column repair the base pin
     script does, which is a no-op here since the 2024_2025 tape already carries all 7) that
     pre-drops 118 known-bad train timestamps before running any overlay. Diagnosed 2026-07-27
     (tmp/research_20260727/check_cmamba_overlay.py): the CryptoMamba regime sidecar overlay has
     two internal 59-row NaN clusters, exactly at 2024-01-01 00:00-04:50 and 2025-01-01
     00:00-04:50 -- a rolling-window warm-up artifact from generating the sidecar in separate
     per-calendar-year batches (same class as this project's other "extension boundary" warm-up
     artifacts, e.g. ou_halflife/garch_vol_z drift on every data extension). 118/210,432 = 0.056%
     of rows. The current-regime and stability-risk overlays have ZERO such gaps (checked
     separately). _overlay_required's own edge-drop logic already handles a SINGLE contiguous
     head/tail gap safely; it correctly refuses two disjoint interior clusters rather than
     silently guessing, which is why this script drops them explicitly and transparently instead
     of relaxing that check.

Does NOT touch trading_bot_modules/, trading_bot.py, runtime_config.py, .env, or any live
checkpoint. Research artifact only.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_pinned102_20260727 as pinned  # noqa: E402

omega = pinned.omega
parent_script = pinned.parent_script

TAPE_DIR = ROOT / "tmp/causal_regen_20260516/omega_clean_regime_only_24_25_inputs_20260629"
omega.TRAIN_CSV = TAPE_DIR / "trade_candidates_2024_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv"
omega.REGIME3_CURRENT_2025 = TAPE_DIR / "training_features_2024_2025_regime3_current_sensitive_hmm_wide24.csv"
omega.REGIME3_CMAMBA_2025 = TAPE_DIR / "training_features_2024_2025_regime3_cryptomamba_h6_sidecar_20260601.csv"
omega.REGIME3_RISK_2025 = TAPE_DIR / "training_features_2024_2025_regime3_stability_risk_h6.csv"

for _p in (omega.TRAIN_CSV, omega.REGIME3_CURRENT_2025, omega.REGIME3_CMAMBA_2025, omega.REGIME3_RISK_2025):
    if not _p.exists():
        raise FileNotFoundError(_p)


def _load_omega_frames_2024tape() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train = omega._read(omega.TRAIN_CSV)
    eval_df = omega._read(omega.EVAL_CSV)

    cmamba_src = omega._read(omega.REGIME3_CMAMBA_2025)
    bad_mask = cmamba_src[omega.REGIME3_CMAMBA_COLS].isna().any(axis=1)
    bad_ts = set(cmamba_src.loc[bad_mask, "timestamp"])
    if bad_ts:
        before = len(train)
        train = train.loc[~train["timestamp"].isin(bad_ts)].reset_index(drop=True)
        s = sorted(bad_ts)
        print(f"[pinned102_2024tape] dropped {before - len(train)} train rows with NaN CryptoMamba "
              f"warm-up values (year-boundary re-init artifact), ts range [{s[0]} .. {s[-1]}]", flush=True)

    train, train_current = omega._overlay_required(train, omega.REGIME3_CURRENT_2025, omega.REGIME3_CURRENT_COLS, tag="train_regime3_current")
    eval_df, eval_current = omega._overlay_required(eval_df, omega.REGIME3_CURRENT_2026, omega.REGIME3_CURRENT_COLS, tag="eval_regime3_current")
    train, train_cmamba = omega._overlay_required(train, omega.REGIME3_CMAMBA_2025, omega.REGIME3_CMAMBA_COLS, tag="train_regime3_cmamba")
    eval_df, eval_cmamba = omega._overlay_required(eval_df, omega.REGIME3_CMAMBA_2026, omega.REGIME3_CMAMBA_COLS, tag="eval_regime3_cmamba")
    train, train_risk = omega._overlay_required(train, omega.REGIME3_RISK_2025, omega.REGIME3_RISK_COLS, tag="train_regime3_risk")
    eval_df, eval_risk = omega._overlay_required(eval_df, omega.REGIME3_RISK_2026, omega.REGIME3_RISK_COLS, tag="eval_regime3_risk")

    train = pinned._repair_train_columns(train)  # no-op here: the 2024_2025 tape already has all 7
    return train, eval_df, {
        "train_current": train_current, "eval_current": eval_current,
        "train_cmamba": train_cmamba, "eval_cmamba": eval_cmamba,
        "train_risk": train_risk, "eval_risk": eval_risk,
        "dropped_cmamba_warmup_rows": len(bad_ts),
    }


def main() -> int:
    argv = list(sys.argv[1:])
    if "--pin-component" not in argv:
        raise SystemExit("--pin-component {h48qual,zig075} is required")
    i = argv.index("--pin-component")
    component = argv[i + 1]
    if component not in pinned.LIVE_BUNDLES:
        raise SystemExit(f"--pin-component must be one of {sorted(pinned.LIVE_BUNDLES)}")
    del argv[i : i + 2]
    sys.argv = [sys.argv[0], *argv]

    pinned._install_pin(component)  # sets base_cols pin + the (here no-op) 7-column repair path
    omega._load_omega_frames = _load_omega_frames_2024tape  # installed AFTER, so it wins
    return parent_script.main()


if __name__ == "__main__":
    raise SystemExit(main())
