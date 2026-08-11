#!/usr/bin/env python3
"""Regenerate a TRUE full-window (2024-01 to 2025-09-30) train_predictions CSV
for the live h48qual bundle, using pure inference on the already-saved model.

Why this exists: the bundle directory's existing train_predictions_q050.csv
was silently regenerated later (2026-06-30 17:59, ~16h after the 01:59 training
run) by export_omega4_parent_predictions_from_bundle_20260630.py using an
override TRAIN_CSV that only spans 2025-01 to 2025-09 (78,509 rows), missing
all of 2024 (which the model actually trained on -- confirmed: report.json's
label_quality_summary.train.rows=183,936 exactly matches this script's
pre-split row count once the correct 2024-2025 sources are used, see below).

The CURRENT default TRAIN_CSV/REGIME3_*_2025 globals in
train_eval_omega1_2_tabm_diffusion_risk_20260603.py have since been changed
(TRAIN_CSV: git blame shows 2026-08-07) or deleted (REGIME3_*_2025/2026 files
are missing on disk entirely as of 2026-08-12) -- neither reflects what was
used to train this bundle on 2026-06-30. This script instead points at
tmp/causal_regen_20260516/omega_clean_regime_only_24_25_inputs_20260629/,
dated one day before training, whose files reproduce report.json's exact
183,936-row training count (verified independently before writing this
script). The 2026 (OOS) side is left on a .bak_pre_extend_20260704 snapshot
purely so eval_df construction doesn't crash -- OOS predictions from this run
are NOT used for anything; only the train split output is trusted.

Output: writes ONLY to a new directory (does not touch the live bundle dir).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as omega4  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402

SRC_DIR = ROOT / "tmp/causal_regen_20260516/omega_clean_regime_only_24_25_inputs_20260629"
BAK_2026 = {
    "current": ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/training_features_2026_rebuilt_regime3_current_sensitive_hmm_wide24.csv.bak_pre_extend_20260704",
    "cmamba": ROOT / "data/ensemble/supervised/regime3_cryptomamba_h6_sidecar_20260601/training_features_2026_rebuilt_regime3_cryptomamba_h6_sidecar_20260601.csv.bak_pre_extend_20260704",
    "risk": ROOT / "data/ensemble/supervised/regime3_stability_risk_h6_20260530/training_features_2026_rebuilt_regime3_stability_risk_h6.csv.bak_pre_extend_20260704",
}

BUNDLE_DIR = ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_20260630"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_20260630_fullwindow_predictions_recheck_20260812"


def _ffill_bfill_copy(src: Path, dst: Path) -> Path:
    """Fill internal NaN gaps in a cmamba/risk overlay source so the strict
    _overlay_required edge-contiguity check passes. Confirmed safe: the live
    bundle's 102 base_cols include zero cmamba/risk-prefixed columns -- these
    overlays are merged in but never read by this model, so filled values
    cannot affect predictions."""
    df = pd.read_csv(src, low_memory=False)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.ffill().bfill()
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)
    return dst


def main() -> int:
    for p in [
        SRC_DIR / "trade_candidates_2024_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv",
        SRC_DIR / "training_features_2024_2025_regime3_current_sensitive_hmm_wide24.csv",
        SRC_DIR / "training_features_2024_2025_regime3_cryptomamba_h6_sidecar_20260601.csv",
        SRC_DIR / "training_features_2024_2025_regime3_stability_risk_h6.csv",
        *BAK_2026.values(),
    ]:
        if not p.exists():
            raise FileNotFoundError(p)

    scratch = ROOT / "tmp/causal_regen_20260516/_scratch_fullwindow_recheck_20260812"
    omega4.omega.TRAIN_CSV = SRC_DIR / "trade_candidates_2024_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv"
    omega4.omega.REGIME3_CURRENT_2025 = SRC_DIR / "training_features_2024_2025_regime3_current_sensitive_hmm_wide24.csv"
    omega4.omega.REGIME3_CMAMBA_2025 = _ffill_bfill_copy(
        SRC_DIR / "training_features_2024_2025_regime3_cryptomamba_h6_sidecar_20260601.csv", scratch / "cmamba_2025_filled.csv"
    )
    omega4.omega.REGIME3_RISK_2025 = _ffill_bfill_copy(
        SRC_DIR / "training_features_2024_2025_regime3_stability_risk_h6.csv", scratch / "risk_2025_filled.csv"
    )
    omega4.omega.REGIME3_CURRENT_2026 = BAK_2026["current"]
    omega4.omega.REGIME3_CMAMBA_2026 = _ffill_bfill_copy(BAK_2026["cmamba"], scratch / "cmamba_2026_filled.csv")
    omega4.omega.REGIME3_RISK_2026 = _ffill_bfill_copy(BAK_2026["risk"], scratch / "risk_2026_filled.csv")
    # EVAL_CSV left at its current default (already exists, already verified to
    # reproduce canonical OOS row counts) -- not touched, not trusted either way.

    parent_report_path = BUNDLE_DIR / "report.json"
    import json

    def _rehome(raw: str) -> Path:
        # report.json paths were recorded on a different machine (/home/llewyn/...);
        # rebase anything under crypto-scalping/ onto this repo's ROOT instead.
        marker = "crypto-scalping/"
        idx = raw.find(marker)
        if idx == -1:
            raise RuntimeError(f"cannot rehome path, no {marker!r} marker: {raw}")
        return ROOT / raw[idx + len(marker):]

    parent_report = json.loads(parent_report_path.read_text(encoding="utf-8"))
    label_contract = parent_report["label_contract"]
    quality_rule = parent_report["quality_target_rule"]
    risk_template = parent_report["risk_template"]

    frames = omega4._prepare_frames(
        disable_tp_sl=bool(risk_template.get("tp_sl_disabled", False)),
        direction_label_dir=_rehome(label_contract["direction_label_dir"]),
        quality_mode=str(label_contract["quality_mode"]),
        quality_label_dir=_rehome(label_contract["quality_label_dir"]) if label_contract.get("quality_label_dir") else None,
        quality_min_edge=float(quality_rule.get("net_return_after_cost_min", 0.0010)),
        quality_max_mae=float(quality_rule.get("mae_max", 0.0100)),
        quality_min_mfe_mae=float(quality_rule.get("mfe_mae_min", 1.20)),
        quality_max_hold_bars=int(quality_rule.get("max_hold_bars", 288)),
    )

    train_raw = frames["train_raw"]
    print(f"train_raw rows: {len(train_raw)} (report.json claims 183936)", flush=True)
    print(f"train_raw span: {train_raw['timestamp'].min()} .. {train_raw['timestamp'].max()}", flush=True)
    if len(train_raw) != 183936:
        raise RuntimeError(f"row count mismatch vs report.json: got {len(train_raw)}, expected 183936 -- ABORTING, do not trust output")

    device = parent._device("cpu")
    bundle = torch.load(BUNDLE_DIR / "true_3head_tabm_bundle.pt", map_location=device, weights_only=False)
    models = dict(bundle["models"])
    base_cols = list(bundle["base_cols"])
    missing_cols = sorted(set(base_cols) - set(train_raw.columns))
    if missing_cols:
        raise RuntimeError(f"prepared frame missing model columns: {missing_cols[:20]}")

    x = parent._base_input(train_raw, base_cols)
    preds = {expert: parent._predict_payload(models[expert], x, device=device) for expert in hard.EXPERT_NAMES}
    route = hard._route_id(train_raw)
    direction = parent._routed(preds, route, "direction", 3)
    quality = parent._routed(preds, route, "quality", 3)
    pred = parent._prediction_output(train_raw, direction, quality, threshold=0.50, prefix="omega1_regime3_expertdq_oof")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "train_predictions_q050.csv"
    pred.to_csv(out_path, index=False)
    print(f"wrote {out_path} rows={len(pred)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
