#!/usr/bin/env python3
"""Re-score of Omega4.6.1's frozen parent models (h48qual + zig075 3-head TabM bundles) on the
extended 2026-01-01..06-30 OOS window, WITHOUT retraining.

Context (2026-07-06): the user asked to retest omega4_6_1_duration_ou_halflife_risk_gate_20260630
(good returns, low OOS trade count) on the OOS window now that data extends past Feb 2026 to
06-30. Investigation found: (1) the parent models need ZERO m7/NF columns (only 102 base +
'regime3_current_sensitive_wide24_*' overlay), so the earlier m7-unrecoverable concern does not
apply to this model; (2) but the model was originally scored on a legacy feature file
(tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/
trade_candidates_2026_alpha6_current_tail111_exact.csv, Jan-Feb only) whose feature values differ
from the current canonical training_features_2026_rebuilt.csv for a handful of columns
(ou_halflife corr=-0.03, kel corr=0.62, evt_excess_z corr=0.79, btc_corr_60 corr=0.85,
dual_momentum corr=0.93 on the Jan-Feb overlap) -- most likely because features/elite.py's
formulas changed since the legacy file was built and git history for that file is too sparse to
recover the exact old version.

Per user direction: regenerate with CURRENT code and document the limitation. To avoid splicing
two inconsistent feature vintages at the Jan/Feb-March boundary, the ENTIRE Jan-Jun 2026 OOS
window is recomputed uniformly from training_features_2026_rebuilt.csv + regime3 overlay (not
just the new March-Jun tail), so the comparison is at least internally consistent across the
window, with the explicit caveat that ou_halflife/kel/evt_excess_z/btc_corr_60/dual_momentum are
NOT bit-identical to the original frozen scoring for the Jan-Feb portion either.
"""

from __future__ import annotations

import json
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

import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402

BASE_2026 = ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"
WIDE24 = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/training_features_2026_rebuilt_regime3_current_sensitive_hmm_wide24.csv"

COMPONENTS = {
    "h48qual": {
        "bundle": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_20260630/true_3head_tabm_bundle.pt",
        "orig_dir": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_20260630",
        "q_tag": "q050",
        "threshold": 0.50,
    },
    "zig075": {
        "bundle": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_20260629/true_3head_tabm_bundle.pt",
        "orig_dir": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_20260629",
        "q_tag": "q075",
        "threshold": 0.75,
    },
}
OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706"


def build_frame() -> pd.DataFrame:
    frame = pd.read_csv(BASE_2026, low_memory=False)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    overlay = pd.read_csv(WIDE24, low_memory=False)
    overlay["timestamp"] = pd.to_datetime(overlay["timestamp"])
    cols = [c for c in overlay.columns if c != "timestamp"]
    merged = frame.merge(overlay[["timestamp", *cols]], on="timestamp", how="left", validate="one_to_one")
    if merged[cols].isna().any().any():
        raise RuntimeError("regime3 overlay has gaps after merge")
    return merged


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = parent._device("cpu")
    frame = build_frame()
    print(f"extended OOS frame: {len(frame)} rows, {frame['timestamp'].min()}..{frame['timestamp'].max()}", flush=True)
    route = hard._route_id(frame)

    for name, cfg in COMPONENTS.items():
        bundle = torch.load(cfg["bundle"], map_location="cpu", weights_only=False)
        base_cols = bundle["base_cols"]
        models = bundle["models"]
        x = parent._base_input(frame, base_cols)
        preds = {expert: parent._predict_payload(models[expert], x, device=device) for expert in hard.EXPERT_NAMES}
        direction = parent._routed(preds, route, "direction", 3)
        quality = parent._routed(preds, route, "quality", 3)
        oos_src_oof = parent._prediction_output(frame, direction, quality, threshold=float(cfg["threshold"]), prefix="omega1_regime3_expertdq_oof")
        oos_src = oos_src_oof.rename(columns={c: c.replace("omega1_regime3_expertdq_oof_", "omega1_regime3_expertdq_") for c in oos_src_oof.columns})

        comp_out = OUT_DIR / name
        comp_out.mkdir(parents=True, exist_ok=True)
        oos_path = comp_out / f"oos_predictions_{cfg['q_tag']}.csv"
        oos_src.to_csv(oos_path, index=False)
        # train/validation predictions are unaffected by extending the 2026 OOS window; copy the
        # originals verbatim so the precomputed_prediction_dir contract (train/validation/oos
        # CSVs with the exact tag) is satisfied without recomputing 2024-2025 data.
        for split in ("train", "validation"):
            src = cfg["orig_dir"] / f"{split}_predictions_{cfg['q_tag']}.csv"
            dst = comp_out / f"{split}_predictions_{cfg['q_tag']}.csv"
            dst.write_bytes(src.read_bytes())
        final_action = oos_src[f"omega1_regime3_expertdq_final_action"] if f"omega1_regime3_expertdq_final_action" in oos_src.columns else oos_src.filter(like="final_action").iloc[:, 0]
        nonzero = float((final_action != 0).mean())
        print(f"{name}: wrote {len(oos_src)} rows -> {oos_path}, nonzero_action_rate={nonzero:.3f}", flush=True)

    (OUT_DIR / "build_report.json").write_text(json.dumps({
        "rows": int(len(frame)),
        "range": [str(frame["timestamp"].min()), str(frame["timestamp"].max())],
        "components": {k: {"q_tag": v["q_tag"], "threshold": v["threshold"]} for k, v in COMPONENTS.items()},
        "known_feature_drift_vs_original_alpha6_lineage": {
            "ou_halflife": "corr=-0.03 on Jan-Feb overlap -- essentially uncorrelated, HIGH RISK (duration gate depends on this feature)",
            "kel": "corr=0.62",
            "evt_excess_z": "corr=0.79",
            "btc_corr_60": "corr=0.85",
            "dual_momentum": "corr=0.93",
            "note": "remaining ~90/96 base columns have corr>=0.99 on the Jan-Feb overlap; drift is isolated to these 5 columns, likely a features/elite.py formula change since 2026-05-29 with sparse git history preventing recovery of the exact old version",
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
