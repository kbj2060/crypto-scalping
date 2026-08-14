#!/usr/bin/env python3
"""Odyssey2 priority #2 (docs/model_contracts/odyssey2_eth_live_injection_contract_20260813.md).
Extracts per-bar ensemble-disagreement (Depeweg et al. 2018 epistemic MI, k=8 TabM members) for a
LIVE component's OWN deployed bundle and writes `{split}_context_features.csv` files consumable by
train_eval_omega4_2_risk_sidecar_20260622.py's `--risk-context-feature-dir` (already-built
extension point, previously unused: risk_context_feature_dir=null in every existing sidecar
report.json). This is the architecture design's own explicitly-reserved use for ensemble
disagreement ("L4 리스크사이징 sidecar 피처 후보로만 사용") -- Odyssey(1) tested this signal's rank
correlation with realized returns for GATING (rejected, no reliable correlation) but never actually
tried it as a SIZING feature, which is what it was designed for.

Pure inference on the already-trained live bundle -- reuses predict_members/route_combine/
mi_decomposition verbatim from diagnose_eth_h48qual_ensemble_disagreement_20260811.py (no
retraining of the parent TabM; the ONLY thing retrained downstream is the risk-sidecar GBM that
consumes this new feature). CPU-only, no GPU needed for a few forward passes over frozen weights.

Frame construction matches train_eval_omega4_2_risk_sidecar_20260622.py's OWN _prepare_frames call
exactly (same TRAIN_CSV/EVAL_CSV/direction_label_dir/quality_mode reconstructed from that
component's actual deployed sidecar report.json) -- guarantees row-for-row alignment with whatever
sidecar retrain consumes these context features next.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_2_risk_sidecar_20260622 as sidecar_script  # noqa: E402
import diagnose_eth_h48qual_ensemble_disagreement_20260811 as ens  # noqa: E402

omega = sidecar_script.omega
omega4 = sidecar_script.omega4
parent = sidecar_script.parent
hard = sidecar_script.hard

# Exact args reconstructed from each component's deployed sidecar report.json ("risk_model"/
# "contract" fields) -- verified against the actual on-disk report.json before use.
COMPONENT_CONFIG = {
    "h48qual": {
        "bundle": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_20260630/true_3head_tabm_bundle.pt",
        "train_csv": ROOT / "tmp/causal_regen_20260516/alpha5_a5dir_2024_train_2025_score_20260529_cleanfunding_stable48/02_fixed_regime4_state24_sticky090_tp18_sl10_preprocess_2024_to_2025/trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv",
        "eval_csv": ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2026_alpha6_current_tail111_exact.csv",
        "direction_label_dir": ROOT / "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531",
        "quality_mode": "same_as_direction",
    },
    "zig075": {
        "bundle": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_20260629/true_3head_tabm_bundle.pt",
        "train_csv": ROOT / "tmp/causal_regen_20260516/alpha5_a5dir_2024_train_2025_score_20260529_cleanfunding_stable48/02_fixed_regime4_state24_sticky090_tp18_sl10_preprocess_2024_to_2025/trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv",
        "eval_csv": ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2026_alpha6_current_tail111_exact.csv",
        "direction_label_dir": ROOT / "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531",
        "quality_mode": "same_as_direction",
    },
}


def per_bar_epistemic(frame: pd.DataFrame, models: dict, base_cols: list[str]) -> pd.DataFrame:
    dir_per_expert, qual_per_expert = {}, {}
    for expert in hard.EXPERT_NAMES:
        d, q = ens.predict_members(models[expert], frame, base_cols)
        dir_per_expert[expert] = d
        qual_per_expert[expert] = q
    route = hard._route_id(frame)
    dir_probs = ens.route_combine(dir_per_expert, route)
    qual_probs = ens.route_combine(qual_per_expert, route)
    _, _, epi_dir = ens.mi_decomposition(dir_probs)
    _, _, epi_qual = ens.mi_decomposition(qual_probs)
    return pd.DataFrame({
        "timestamp": frame["timestamp"].to_numpy(),
        "trend_ctx_epistemic_direction": epi_dir,
        "trend_ctx_epistemic_quality": epi_qual,
    })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--component", choices=list(COMPONENT_CONFIG.keys()), required=True)
    args = ap.parse_args()
    cfg = COMPONENT_CONFIG[args.component]

    out_dir = ROOT / f"tmp/causal_regen_20260516/eth_{args.component}_ensemble_epistemic_context_20260813"
    out_dir.mkdir(parents=True, exist_ok=True)

    omega.TRAIN_CSV = Path(cfg["train_csv"])
    omega.EVAL_CSV = Path(cfg["eval_csv"])
    print("stage=prepare_frames", flush=True)
    frames = omega4._prepare_frames(
        disable_tp_sl=False, direction_label_dir=Path(cfg["direction_label_dir"]), quality_mode=str(cfg["quality_mode"]),
        quality_label_dir=None, quality_min_edge=0.0, quality_max_mae=0.0, quality_min_mfe_mae=0.0, quality_max_hold_bars=0,
    )

    bundle = torch.load(cfg["bundle"], map_location="cpu", weights_only=False)
    models, base_cols = bundle["models"], list(bundle["base_cols"])

    for split, key in [("train", "train_raw"), ("validation", "val_raw"), ("oos", "oos_raw")]:
        frame = frames[key]
        print(f"stage=predict split={split} rows={len(frame)}", flush=True)
        ctx = per_bar_epistemic(frame, models, base_cols)
        assert len(ctx) == len(frame), f"{split}: row count mismatch {len(ctx)} vs {len(frame)}"
        out_path = out_dir / f"{split}_context_features.csv"
        ctx.to_csv(out_path, index=False)
        print(f"  wrote {out_path} epistemic_direction[mean={ctx['trend_ctx_epistemic_direction'].mean():.4f}, "
              f"std={ctx['trend_ctx_epistemic_direction'].std():.4f}] "
              f"epistemic_quality[mean={ctx['trend_ctx_epistemic_quality'].mean():.4f}]", flush=True)

    print(f"DONE component={args.component} out_dir={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
