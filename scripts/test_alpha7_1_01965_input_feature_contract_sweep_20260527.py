#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import loop_alpha3_1_alpha6_alpha7_combo_search_until_0800_20260527 as combo  # noqa: E402
from scripts import train_eval_hf_v13_deep_alpha_candidate_expansion_v27 as v27  # noqa: E402
from scripts.analyze_alpha7_tp_sl_action_score_20260526 import (  # noqa: E402
    FB_EVAL_CSV,
    FB_TRAIN_CSV,
    FALLBACK_PARENT,
    FALLBACK_SUMMARY,
    PRIMARY_EVAL_CSV,
    PRIMARY_PARENT,
    PRIMARY_TRAIN_CSV,
    _combine_primary_fallback,
    _load_best_scale_runtime,
    _predict_scaled,
    _train_parent,
)
from scripts.precision_retest_01965_alpha7_combo_20260527 import CANDIDATE, _cfg_from_results, _eval  # noqa: E402
from scripts.rebuild_alpha7_v2_only_live_20260526 import _rename_clean4_v2  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default, _read  # noqa: E402


MODEL_ID = "alpha7_1_01965_input_feature_contract_sweep_20260527"
OUT_DIR = ROOT / f"tmp/causal_regen_20260516/{MODEL_ID}"
SUMMARY_OUT = OUT_DIR / "summary.json"
GRID_OUT = OUT_DIR / "grid.csv"
FEATURE_CONTRACT_OUT = OUT_DIR / "feature_contracts.json"

DERIVABLE = {
    "side_hint",
    "mom_21d",
    "abs_mom_21d",
    "mom_3d",
    "abs_mom_3d",
    "mom_1d",
    "abs_mom_1d",
}
STICKY_PREFIX = "clean_regime4_state24_sticky090_v2_"
LEGACY_REGIME_PREFIX = "clean_regime4_2024_unsup_v1_"
RAW_M7_PRICE_FEATURES = {
    "m7_entry_long_price",
    "m7_entry_short_price",
    "m7_tp_price",
    "m7_sl_price",
}
TEACHER_FEATURES = [
    "teacher_long_edge",
    "teacher_short_edge",
    "teacher_side_margin",
    "teacher_side_disagreement",
    "teacher_quantile_skew",
    "teacher_uncertainty",
    "teacher_tail_warning",
]


@dataclass(frozen=True)
class FeatureVariant:
    name: str
    primary_cols: list[str] | None
    fallback_cols: list[str] | None = None
    train_primary: bool = True
    train_fallback: bool = False


def _unique(cols: list[str]) -> list[str]:
    return list(dict.fromkeys(str(c) for c in cols))


def _cols_with_prefix(frame: pd.DataFrame, prefix: str) -> list[str]:
    return sorted(str(c) for c in frame.columns if str(c).startswith(prefix))


def _merge_sticky_v2(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    sticky = ["timestamp", *_cols_with_prefix(right, STICKY_PREFIX)]
    if len(sticky) == 1:
        raise RuntimeError("sticky_v2 regime columns missing from source frame")
    out = left.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="raise").dt.tz_convert(None)
    add = right[sticky].copy()
    add["timestamp"] = pd.to_datetime(add["timestamp"], utc=True, errors="raise").dt.tz_convert(None)
    add = add.drop_duplicates("timestamp", keep="last")
    overlap = [c for c in add.columns if c != "timestamp" and c in out.columns]
    if overlap:
        raise RuntimeError(f"sticky_v2 merge would overwrite existing columns: {overlap[:10]}")
    out = out.merge(add, on="timestamp", how="left", validate="one_to_one")
    bad = [c for c in sticky if c != "timestamp" and (c not in out.columns or out[c].isna().any())]
    if bad:
        raise RuntimeError(f"sticky_v2 exact timestamp merge failed: {bad[:10]}")
    return out.reset_index(drop=True)


def _assert_feature_contract(frame: pd.DataFrame, cols: list[str], *, name: str) -> None:
    legacy = [c for c in cols if str(c).startswith(LEGACY_REGIME_PREFIX)]
    if legacy:
        raise RuntimeError(f"{name}: legacy regime columns are forbidden: {legacy[:10]}")
    raw_m7 = [c for c in cols if c in RAW_M7_PRICE_FEATURES]
    if raw_m7:
        raise RuntimeError(f"{name}: raw M7 price-level features are forbidden: {raw_m7}")
    missing = [c for c in cols if c not in frame.columns and c not in DERIVABLE]
    if missing:
        raise RuntimeError(f"{name}: missing feature columns: {missing[:30]}")
    sticky_count = sum(c.startswith(STICKY_PREFIX) for c in cols)
    if sticky_count == 0:
        raise RuntimeError(f"{name}: sticky_v2 regime columns are required")
    teacher_count = sum(c.startswith("teacher_") for c in cols)
    if teacher_count == 0:
        raise RuntimeError(f"{name}: teacher_* features must be retained for this sweep")


def _load_training_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary_train = _rename_clean4_v2(_read(PRIMARY_TRAIN_CSV))
    primary_eval = _rename_clean4_v2(_read(PRIMARY_EVAL_CSV))
    fallback_train = _rename_clean4_v2(_read(FB_TRAIN_CSV))
    fallback_eval = _rename_clean4_v2(_read(FB_EVAL_CSV))
    fallback_train = _merge_sticky_v2(fallback_train, primary_train)
    fallback_eval = _merge_sticky_v2(fallback_eval, primary_eval)
    return primary_train, primary_eval, fallback_train, fallback_eval


def _build_feature_variants(primary_train: pd.DataFrame, primary_parent: dict[str, Any], fallback_parent: dict[str, Any]) -> list[FeatureVariant]:
    primary_base = list(primary_parent["feature_cols"])
    fallback_base = list(fallback_parent["feature_cols"])
    sticky_cols = _cols_with_prefix(primary_train, STICKY_PREFIX)
    regime_pred_cols = _cols_with_prefix(primary_train, "regime4_pred_")

    base_market = [
        c
        for c in primary_base
        if not c.startswith(("m7_", "ai_", "teacher_", "regime4_pred_", STICKY_PREFIX, "clean_regime4_"))
    ]
    model_architect_core = [
        "m7_gate_block",
        "m7_tail_risk",
        "m7_expected_ret",
        "m7_composite_score",
        "m7_confidence",
        "m7_qwidth",
        "m7_quant_up",
        "m7_quant_dn",
        "m7_q50",
        "m7_q90",
        "m7_quality_pred",
        "m7_hold_pred",
        "m7_action",
        "m7_trend_xgb_up",
        "m7_trend_xgb_dn",
        "m7_mtl_up",
        "m7_mtl_dn",
        "ai_dir_edge",
        "ai_dir_p_up",
        "ai_dir_p_down",
        "ai_dir_p_flat",
        "ai_dir_entropy",
        "ai_adverse_risk",
        "ai_reward_risk",
        "ai_vol_regime_pct",
        "ai_flow_pressure",
        "ai_flow_exhaustion",
        "ai_flow_flip_prob",
        "ai_flow_slope",
        "patchtst_median",
        "patchtst_regime_sim",
        "tide_vol_zscore",
        "dlinear_smf_ema",
        "dlinear_smf_slope",
        *TEACHER_FEATURES,
    ]
    architect_primary = _unique([*base_market, *model_architect_core, *sticky_cols, *regime_pred_cols])

    stable_context = [
        c
        for c in fallback_base
        if not c.startswith(("m7_", "ai_", "teacher_", "regime4_pred_", STICKY_PREFIX, "clean_regime4_"))
    ]
    stable_context += [
        "m7_gate_block",
        "m7_tail_risk",
        "m7_confidence",
        "m7_qwidth",
        "m7_expected_ret",
        "m7_composite_score",
        "m7_quality_pred",
        "m7_hold_pred",
        "ai_adverse_risk",
        "ai_reward_risk",
        "ai_vol_regime_pct",
        "ai_flow_pressure",
        "ai_flow_exhaustion",
        "ai_flow_flip_prob",
        "ai_flow_slope",
        "patchtst_median",
        "patchtst_regime_sim",
        "tide_vol_zscore",
        *TEACHER_FEATURES,
        *sticky_cols,
    ]
    fallback_stable = _unique(stable_context)

    return [
        FeatureVariant("baseline_01965", None, None, train_primary=False, train_fallback=False),
        FeatureVariant("primary_teacher_sticky_v2", _unique([*primary_base, *TEACHER_FEATURES])),
        FeatureVariant("primary_architect_core", architect_primary),
        FeatureVariant("full_layer_role_contract", architect_primary, fallback_stable, train_fallback=True),
    ]


def _fit_or_load_parent(
    *,
    variant_dir: Path,
    role: str,
    train_all: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    role_dir = variant_dir / role
    role_dir.mkdir(parents=True, exist_ok=True)
    return _train_parent(train_all=train_all, eval_df=eval_df, feature_cols=feature_cols, seed=int(seed), out_dir=role_dir)


def _eval_costs(
    *,
    variant: str,
    val_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    val_q: np.ndarray,
    eval_q: np.ndarray,
    dec_val: pd.DataFrame,
    dec_eval: pd.DataFrame,
    stack: dict[str, Any],
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, df, q, dec in (("val", val_df, val_q, dec_val), ("oos", eval_df, eval_q, dec_eval)):
        for cost in (1, 2, 3):
            row = _eval(df=df, q=q, dec=dec, stack=stack, cfg=cfg, period=split, cost_mult=cost, record=False)
            row["variant"] = variant
            rows.append(row)
    return rows


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _cfg_from_results()
    if cfg.get("source") != "alpha7_combo_primary_fallback":
        raise RuntimeError(f"01965 source contract changed: {cfg.get('source')}")

    primary_parent = joblib.load(PRIMARY_PARENT)
    fallback_parent = joblib.load(FALLBACK_PARENT)
    fallback_rt = _load_best_scale_runtime(FALLBACK_SUMMARY)
    primary_train, primary_eval, fallback_train, fallback_eval = _load_training_frames()
    variants = _build_feature_variants(primary_train, primary_parent, fallback_parent)

    for variant in variants:
        if variant.primary_cols is not None:
            _assert_feature_contract(primary_train, variant.primary_cols, name=f"{variant.name}:primary_train")
            _assert_feature_contract(primary_eval, variant.primary_cols, name=f"{variant.name}:primary_eval")
        if variant.fallback_cols is not None:
            _assert_feature_contract(fallback_train, variant.fallback_cols, name=f"{variant.name}:fallback_train")
            _assert_feature_contract(fallback_eval, variant.fallback_cols, name=f"{variant.name}:fallback_eval")

    stack = combo._load_stack()
    val_df, eval_df = combo._load_frames()
    sources = combo._decision_sources(val_df, eval_df, stack["parent"])
    val_q = v27._predict_all(stack["deep_model"], val_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    eval_q = v27._predict_all(stack["deep_model"], eval_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])

    rows: list[dict[str, Any]] = []
    contracts: dict[str, Any] = {
        "model_id": MODEL_ID,
        "base_candidate": CANDIDATE,
        "policy": {
            "teacher_features_retained": True,
            "required_regime_prefix": STICKY_PREFIX,
            "forbidden_regime_prefix": LEGACY_REGIME_PREFIX,
            "forbidden_raw_m7_price_features": sorted(RAW_M7_PRICE_FEATURES),
            "selection_uses_2026": False,
        },
        "variants": {},
    }

    for i, variant in enumerate(variants):
        variant_dir = OUT_DIR / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        if variant.name == "baseline_01965":
            dec_val = sources[str(cfg["source"])][0]
            dec_eval = sources[str(cfg["source"])][1]
            contracts["variants"][variant.name] = {
                "type": "baseline",
                "primary_feature_count": int(len(primary_parent["feature_cols"])),
                "fallback_feature_count": int(len(fallback_parent["feature_cols"])),
            }
        else:
            assert variant.primary_cols is not None
            new_primary, primary_rt, primary_summary = _fit_or_load_parent(
                variant_dir=variant_dir,
                role="primary",
                train_all=primary_train,
                eval_df=primary_eval,
                feature_cols=variant.primary_cols,
                seed=5527 + i * 37,
            )
            if variant.train_fallback:
                assert variant.fallback_cols is not None
                new_fallback, new_fallback_rt, fallback_summary = _fit_or_load_parent(
                    variant_dir=variant_dir,
                    role="fallback",
                    train_all=fallback_train,
                    eval_df=fallback_eval,
                    feature_cols=variant.fallback_cols,
                    seed=4327 + i * 37,
                )
            else:
                new_fallback = fallback_parent
                new_fallback_rt = fallback_rt
                fallback_summary = {"type": "baseline_fallback_unchanged"}

            primary_val = _predict_scaled(new_primary, val_df, primary_rt)
            primary_oos = _predict_scaled(new_primary, eval_df, primary_rt)
            fallback_val = _predict_scaled(new_fallback, val_df, new_fallback_rt)
            fallback_oos = _predict_scaled(new_fallback, eval_df, new_fallback_rt)
            dec_val = _combine_primary_fallback(primary_val, fallback_val)
            dec_eval = _combine_primary_fallback(primary_oos, fallback_oos)
            contracts["variants"][variant.name] = {
                "type": "feature_contract_retrain",
                "primary_feature_count": int(len(variant.primary_cols)),
                "primary_sticky_v2_count": int(sum(c.startswith(STICKY_PREFIX) for c in variant.primary_cols)),
                "primary_teacher_count": int(sum(c.startswith("teacher_") for c in variant.primary_cols)),
                "primary_feature_cols": variant.primary_cols,
                "primary_summary": primary_summary,
                "fallback_retrained": bool(variant.train_fallback),
                "fallback_feature_count": int(len(variant.fallback_cols or fallback_parent["feature_cols"])),
                "fallback_sticky_v2_count": int(sum(c.startswith(STICKY_PREFIX) for c in (variant.fallback_cols or []))),
                "fallback_teacher_count": int(sum(c.startswith("teacher_") for c in (variant.fallback_cols or []))),
                "fallback_feature_cols": variant.fallback_cols,
                "fallback_summary": fallback_summary,
            }

        rows.extend(
            _eval_costs(
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
        raise RuntimeError("baseline_01965 precision value was not reproduced; feature sweep is invalid")

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
