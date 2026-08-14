"""RESEARCH ONLY -- retrain the Omega4.6.1 parent with base_cols PINNED to the live checkpoint's
exact 102-column input contract.

Why. The 2026-07-21 exit-head retrain bundles were trained on 172 base features (77 extra
m7_/ai_/patchtst*/tide_/dlinear*/regime3_cmamba_* columns) instead of the live checkpoints' 102,
because the trainer derives base_cols from whatever numeric columns the candidate CSVs happen to
carry at run time (omega._numeric_feature_cols). Two consequences, both measured 2026-07-27:
  - trading_bot_modules/omega4_6_1_live.py:113-114 rejects those prefixes outright, so such a
    bundle can never be loaded live;
  - the m7/NF columns only exist for part of 2025, which collapsed the training frame from the
    live 183,936 rows to 78,509.
So no comparison against the live checkpoint using those bundles is clean. This wrapper removes
both defects so that "retrain with a different exit label" can finally be tested against live on
equal footing.

Two monkeypatches, both narrowly scoped to this process:
  1. omega._load_omega_frames -- re-attaches the 7 live features that today's 2025 candidate CSV
     no longer carries (fibonacci_level, funding_roc_12/48, funding_z_score, short_squeeze_risk,
     hurst_288, regime_persistence), sourced by timestamp from data/splits/year_oos/
     training_features_2025.csv (verified 2026-07-27: 100% timestamp coverage, 100% non-null).
     The 2026 eval CSV still carries all 7, so only the train side needs repair.
  2. omega._numeric_feature_cols -- returns the live bundle's base_cols verbatim, in live order,
     after asserting every one is present in both frames (fail-fast; no silent substitution).

Everything else -- architecture, labels, hyperparameters, epochs, split (SPLIT_TS=2025-10-01) --
is the unmodified 20260620 trainer. Does NOT touch trading_bot_modules/, trading_bot.py,
runtime_config.py, .env, or any live checkpoint. Research artifact only.

Usage (live-equivalent control for h48qual):
  python scripts/train_eval_omega4_3head_parent72_pinned102_20260727.py \
      --pin-component h48qual --out-suffix pinned102_20260727_h48qual_control \
      --epochs 2 --max-train-rows 0 --max-exit-samples 30000 \
      --quality-thresholds 0.50 --exit-label-mode entry_label_terminal_giveback \
      --direction-label-dir tmp/.../zigzag_action_labels_20260531 \
      --quality-mode quality_label_action --quality-label-dir tmp/.../sltp_h48_conservative_padded_to_zigzag_timestamps
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as parent_script  # noqa: E402

omega = parent_script.omega

LIVE_BUNDLES = {
    "h48qual": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_20260630/true_3head_tabm_bundle.pt",
    "zig075": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_20260629/true_3head_tabm_bundle.pt",
}
# Present in the live contract but dropped from today's 2025 candidate CSV.
REPAIR_COLS = ["fibonacci_level", "funding_roc_12", "funding_roc_48", "funding_z_score",
               "short_squeeze_risk", "hurst_288", "regime_persistence"]
REPAIR_SOURCE = ROOT / "data/splits/year_oos/training_features_2025.csv"

_orig_load_frames = omega._load_omega_frames


def _repair_train_columns(frame: pd.DataFrame) -> pd.DataFrame:
    need = [c for c in REPAIR_COLS if c not in frame.columns]
    if not need:
        return frame
    src = pd.read_csv(REPAIR_SOURCE, usecols=["timestamp", *need], low_memory=False)
    src["timestamp"] = pd.to_datetime(src["timestamp"])
    src = src.dropna(subset=["timestamp"]).drop_duplicates("timestamp", keep="last")
    out = frame.copy()
    ts = pd.to_datetime(out["timestamp"])
    joined = src.set_index("timestamp").reindex(ts)
    for c in need:
        vals = pd.to_numeric(joined[c], errors="coerce").to_numpy()
        if pd.isna(vals).any():
            raise RuntimeError(f"pinned102 repair: {c} has {int(pd.isna(vals).sum())} missing values after join")
        out[c] = vals
    print(f"[pinned102] repaired {len(need)} train columns from {REPAIR_SOURCE.name}: {need}", flush=True)
    return out


def _patched_load_frames():
    train_all, eval_df, overlay_report = _orig_load_frames()
    return _repair_train_columns(train_all), eval_df, overlay_report


def _install_pin(component: str) -> None:
    import torch

    bundle = torch.load(LIVE_BUNDLES[component], map_location="cpu", weights_only=False)
    live_cols = list(bundle["base_cols"])
    print(f"[pinned102] pinning base_cols to live {component} contract: {len(live_cols)} columns", flush=True)

    def _patched_numeric_feature_cols(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> list[str]:
        missing_train = [c for c in live_cols if c not in train_df.columns]
        missing_eval = [c for c in live_cols if c not in eval_df.columns]
        if missing_train or missing_eval:
            raise RuntimeError(
                f"pinned102: live base_cols unavailable (train missing {missing_train}, eval missing {missing_eval})")
        return list(live_cols)

    omega._load_omega_frames = _patched_load_frames
    omega._numeric_feature_cols = _patched_numeric_feature_cols


def main() -> int:
    argv = list(sys.argv[1:])
    if "--pin-component" not in argv:
        raise SystemExit("--pin-component {h48qual,zig075} is required")
    i = argv.index("--pin-component")
    component = argv[i + 1]
    if component not in LIVE_BUNDLES:
        raise SystemExit(f"--pin-component must be one of {sorted(LIVE_BUNDLES)}")
    del argv[i : i + 2]
    sys.argv = [sys.argv[0], *argv]

    _install_pin(component)
    return parent_script.main()


if __name__ == "__main__":
    raise SystemExit(main())
