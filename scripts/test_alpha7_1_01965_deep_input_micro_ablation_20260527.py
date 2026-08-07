#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import loop_alpha3_1_alpha6_alpha7_combo_search_until_0800_20260527 as combo  # noqa: E402
from scripts import train_eval_hf_v13_deep_alpha_candidate_expansion_v27 as v27  # noqa: E402
from scripts.precision_retest_01965_alpha7_combo_20260527 import CANDIDATE, _cfg_from_results  # noqa: E402
from scripts.test_alpha7_1_01965_deep_input_feature_sweep_20260527 import (  # noqa: E402
    MODEL_ID as PARENT_SWEEP_ID,
    RAW_M7_PRICE_FEATURES,
    STICKY_PREFIX,
    DeepVariant,
    _assert_deep_contract,
    _eval_rows,
    _load_augmented_frames,
    _train_deep_variant,
    _unique,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


MODEL_ID = "alpha7_1_01965_deep_input_micro_ablation_20260527"
OUT_DIR = ROOT / f"tmp/causal_regen_20260516/{MODEL_ID}"
SUMMARY_OUT = OUT_DIR / "summary.json"
GRID_OUT = OUT_DIR / "grid.csv"
FEATURE_CONTRACT_OUT = OUT_DIR / "feature_contracts.json"

EXISTING_TEACHER = [
    "teacher_side_margin",
    "teacher_side_disagreement",
    "teacher_uncertainty",
    "teacher_tail_warning",
]
TEACHER_EDGES = [
    "teacher_long_edge",
    "teacher_short_edge",
    "teacher_quantile_skew",
]
STICKY_RISK = [
    f"{STICKY_PREFIX}confidence",
    f"{STICKY_PREFIX}entropy",
    f"{STICKY_PREFIX}instability_prob",
    f"{STICKY_PREFIX}risk_off_prob",
    f"{STICKY_PREFIX}transition_risk",
    f"{STICKY_PREFIX}whipsaw_prob",
]
STICKY_DIRECTION = [
    f"{STICKY_PREFIX}bear_prob",
    f"{STICKY_PREFIX}bull_prob",
    f"{STICKY_PREFIX}chop_prob",
    f"{STICKY_PREFIX}directional_bias",
    f"{STICKY_PREFIX}margin",
    f"{STICKY_PREFIX}trend_bias",
    f"{STICKY_PREFIX}trend_prob",
]


def _available(frame: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in frame.columns]


def _micro_variants(frame: pd.DataFrame, baseline_seq_cols: list[str]) -> list[DeepVariant]:
    sanitized = [c for c in baseline_seq_cols if c not in RAW_M7_PRICE_FEATURES]
    teacher_edges = _available(frame, TEACHER_EDGES)
    sticky_risk = _available(frame, STICKY_RISK)
    sticky_direction = _available(frame, STICKY_DIRECTION)
    if len(sticky_risk) != len(STICKY_RISK):
        raise RuntimeError(f"missing sticky risk features: {sorted(set(STICKY_RISK) - set(sticky_risk))}")
    if len(sticky_direction) != len(STICKY_DIRECTION):
        raise RuntimeError(f"missing sticky direction features: {sorted(set(STICKY_DIRECTION) - set(sticky_direction))}")
    return [
        DeepVariant("baseline_01965", None, epochs=0),
        DeepVariant("micro_sanitized_plus_sticky_risk", _unique([*sanitized, *EXISTING_TEACHER, *sticky_risk]), epochs=80),
        DeepVariant("micro_plus_teacher_edges_sticky_risk", _unique([*sanitized, *EXISTING_TEACHER, *teacher_edges, *sticky_risk]), epochs=80),
        DeepVariant("micro_plus_sticky_direction", _unique([*sanitized, *EXISTING_TEACHER, *sticky_direction]), epochs=80),
        DeepVariant(
            "micro_plus_teacher_edges_direction",
            _unique([*sanitized, *EXISTING_TEACHER, *teacher_edges, *sticky_direction]),
            epochs=80,
        ),
    ]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _cfg_from_results()
    if cfg.get("source") != "alpha7_combo_primary_fallback":
        raise RuntimeError(f"01965 source contract changed: {cfg.get('source')}")

    stack = combo._load_stack()
    train_df, val_df, eval_df = _load_augmented_frames()
    sources = combo._decision_sources(val_df, eval_df, stack["parent"])
    dec_val, dec_eval = sources[str(cfg["source"])]
    baseline_seq_cols = list(stack["deep_payload"]["seq_cols"])
    variants = _micro_variants(train_df, baseline_seq_cols)

    rows: list[dict[str, Any]] = []
    contracts: dict[str, Any] = {
        "model_id": MODEL_ID,
        "base_candidate": CANDIDATE,
        "parent_sweep_reference": PARENT_SWEEP_ID,
        "policy": {
            "parent_fallback_inputs_fixed": True,
            "teacher_features_retained": True,
            "micro_ablation_only": True,
            "raw_m7_price_features_removed_from_new_variants": sorted(RAW_M7_PRICE_FEATURES),
            "required_regime_prefix": STICKY_PREFIX,
            "selection_uses_2026": False,
        },
        "variants": {},
    }

    for i, variant in enumerate(variants):
        if variant.seq_cols is None:
            val_q = v27._predict_all(stack["deep_model"], val_df, baseline_seq_cols, stack["deep_payload"]["norm"])
            eval_q = v27._predict_all(stack["deep_model"], eval_df, baseline_seq_cols, stack["deep_payload"]["norm"])
            contracts["variants"][variant.name] = {
                "type": "baseline_deep_model",
                "seq_count": int(len(baseline_seq_cols)),
                "seq_cols": baseline_seq_cols,
            }
        else:
            _assert_deep_contract(train_df, variant.seq_cols, name=f"{variant.name}:train")
            _assert_deep_contract(val_df, variant.seq_cols, name=f"{variant.name}:val")
            _assert_deep_contract(eval_df, variant.seq_cols, name=f"{variant.name}:eval")
            trained = _train_deep_variant(
                train_df,
                variant.seq_cols,
                epochs=int(variant.epochs),
                seed=8627 + i * 53,
                out_dir=OUT_DIR / variant.name,
            )
            val_q = v27._predict_all(trained["model"], val_df, trained["seq_cols"], trained["norm"])
            eval_q = v27._predict_all(trained["model"], eval_df, trained["seq_cols"], trained["norm"])
            contracts["variants"][variant.name] = {
                "type": "deep_input_micro_retrain",
                "epochs": int(variant.epochs),
                "train_samples": int(trained["train_samples"]),
                "seq_count": int(len(variant.seq_cols)),
                "teacher_count": int(sum(c.startswith("teacher_") for c in variant.seq_cols)),
                "sticky_v2_count": int(sum(c.startswith(STICKY_PREFIX) for c in variant.seq_cols)),
                "seq_cols": variant.seq_cols,
                "artifact": str(OUT_DIR / variant.name / "deep_model.pt"),
            }
        rows.extend(
            _eval_rows(
                variant=variant.name,
                val_df=val_df,
                eval_df=eval_df,
                val_q=val_q,
                eval_q=eval_q,
                dec_val=dec_val,
                dec_eval=dec_eval,
                stack=stack,
                cfg=cfg,
            )
        )

    grid = pd.DataFrame(rows)
    grid.to_csv(GRID_OUT, index=False)
    FEATURE_CONTRACT_OUT.write_text(json.dumps(contracts, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    cost3 = grid[grid["cost"].eq(3)].copy()
    baseline_oos = cost3[(cost3["variant"].eq("baseline_01965")) & (cost3["period"].eq("oos"))]
    if baseline_oos.empty or abs(float(baseline_oos.iloc[0]["pnl"]) - 274.53249150592416) > 1e-6:
        raise RuntimeError("baseline_01965 precision value was not reproduced; micro ablation is invalid")
    summary = {
        "model_id": MODEL_ID,
        "base_candidate": CANDIDATE,
        "selection_uses_2026": False,
        "feature_contracts": str(FEATURE_CONTRACT_OUT),
        "grid": str(GRID_OUT),
        "cost3": cost3.sort_values(["period", "pnl"], ascending=[True, False]).to_dict(orient="records"),
    }
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(SUMMARY_OUT), "grid": str(GRID_OUT), "feature_contracts": str(FEATURE_CONTRACT_OUT)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
