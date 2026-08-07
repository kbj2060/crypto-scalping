#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import loop_alpha3_1_alpha6_alpha7_combo_search_until_0800_20260527 as combo  # noqa: E402
from scripts import train_eval_hf_v13_deep_alpha_candidate_expansion_v27 as v27  # noqa: E402
from scripts import train_eval_hf_v13_deep_jackpot_sequence_verifier_v23 as v23  # noqa: E402
from scripts.analyze_alpha7_tp_sl_action_score_20260526 import PRIMARY_EVAL_CSV, PRIMARY_TRAIN_CSV  # noqa: E402
from scripts.precision_retest_01965_alpha7_combo_20260527 import CANDIDATE, _cfg_from_results, _eval  # noqa: E402
from scripts.rebuild_alpha7_v2_only_live_20260526 import _rename_clean4_v2  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default, _read  # noqa: E402


MODEL_ID = "alpha7_1_01965_deep_input_feature_sweep_20260527"
OUT_DIR = ROOT / f"tmp/causal_regen_20260516/{MODEL_ID}"
SUMMARY_OUT = OUT_DIR / "summary.json"
GRID_OUT = OUT_DIR / "grid.csv"
FEATURE_CONTRACT_OUT = OUT_DIR / "feature_contracts.json"

STICKY_PREFIX = "clean_regime4_state24_sticky090_v2_"
LEGACY_REGIME_PREFIX = "clean_regime4_2024_unsup_v1_"
LEGACY_V4_PREFIX = "clean_regime_2024_unsup_v4_"
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
class DeepVariant:
    name: str
    seq_cols: list[str] | None
    epochs: int = 100


def _unique(cols: list[str]) -> list[str]:
    return list(dict.fromkeys(str(c) for c in cols))


def _cols_with_prefix(frame: pd.DataFrame, prefix: str) -> list[str]:
    return sorted(str(c) for c in frame.columns if str(c).startswith(prefix))


def _numeric_available(frame: pd.DataFrame, cols: list[str]) -> list[str]:
    out: list[str] = []
    for c in cols:
        if c not in frame.columns:
            continue
        if pd.api.types.is_numeric_dtype(frame[c]):
            out.append(c)
    return out


def _load_augmented_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_all = combo._merge_state24(_read(combo.v31.DEFAULT_TRAIN), combo.alpha3_full.SIDE_CLEAN4_2025)
    eval_df = combo._merge_state24(_read(combo.v31.DEFAULT_EVAL), combo.alpha3_full.SIDE_CLEAN4_2026)
    a7_train = _rename_clean4_v2(_read(PRIMARY_TRAIN_CSV))
    a7_eval = _rename_clean4_v2(_read(PRIMARY_EVAL_CSV))
    train_all = combo._augment_with_alpha7_features(train_all, a7_train)
    eval_df = combo._augment_with_alpha7_features(eval_df, a7_eval)
    train_all["timestamp"] = pd.to_datetime(train_all["timestamp"], errors="raise")
    eval_df["timestamp"] = pd.to_datetime(eval_df["timestamp"], errors="raise")
    train = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    return train, val, eval_df.reset_index(drop=True)


def _assert_deep_contract(frame: pd.DataFrame, cols: list[str], *, name: str) -> None:
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise RuntimeError(f"{name}: missing deep sequence columns: {missing[:30]}")
    forbidden = [
        c
        for c in cols
        if c in RAW_M7_PRICE_FEATURES
        or c.startswith(LEGACY_REGIME_PREFIX)
        or c.startswith(LEGACY_V4_PREFIX)
        or any(tok in c.lower() for tok in v23.FORBIDDEN)
        or any(tok in c.lower() for tok in ("target", "label", "future", "cash_after"))
    ]
    if forbidden:
        raise RuntimeError(f"{name}: forbidden deep sequence columns selected: {forbidden[:30]}")
    if not any(c.startswith("teacher_") for c in cols):
        raise RuntimeError(f"{name}: teacher_* features must be retained")
    if not any(c.startswith(STICKY_PREFIX) for c in cols):
        raise RuntimeError(f"{name}: sticky_v2 regime columns are required")


def _build_variants(frame: pd.DataFrame, baseline_seq_cols: list[str]) -> list[DeepVariant]:
    sticky = _cols_with_prefix(frame, STICKY_PREFIX)
    base_seq = [
        c
        for c in baseline_seq_cols
        if c in frame.columns and not c.startswith(LEGACY_V4_PREFIX) and c not in RAW_M7_PRICE_FEATURES
    ]
    teacher_sticky = _unique([*base_seq, *TEACHER_FEATURES, *sticky])[:80]

    architect_context = [
        "log_return",
        "mtf_trend_1h",
        "mtf_trend_4h",
        "bb_width_z",
        "garch_vol_z",
        "amihud_illiquidity_z",
        "net_taker_ratio",
        "taker_acceleration",
        "trade_intensity",
        "oi_change_rate",
        "last_funding_rate",
        "long_squeeze_risk",
        "funding_price_divergence",
        "volatility_z",
        "rsi",
        "macd_hist",
        "realized_vol_ratio",
        "chop_index",
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
        *sticky,
    ]
    architect_context = _numeric_available(frame, _unique(architect_context))[:80]

    return [
        DeepVariant("baseline_01965", None, epochs=0),
        DeepVariant("deep_teacher_sticky_v2", teacher_sticky),
        DeepVariant("deep_architect_context", architect_context),
    ]


def _train_deep_variant(train: pd.DataFrame, seq_cols: list[str], *, epochs: int, seed: int, out_dir: Path) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train_ds = v27._build_train_set(train, seq_cols, fee=0.0004, slip=0.00015, stride=3)
    norm = v27._normalizer(train_ds["seq"])
    model = v27._train_model(train_ds, norm, epochs=int(epochs))
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_id": MODEL_ID,
            "variant": out_dir.name,
            "state_dict": model.state_dict(),
            "seq_cols": seq_cols,
            "norm": norm,
            "epochs": int(epochs),
            "seed": int(seed),
        },
        out_dir / "deep_model.pt",
    )
    return {"model": model, "seq_cols": seq_cols, "norm": norm, "train_samples": int(len(train_ds["target"]))}


def _eval_rows(
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

    stack = combo._load_stack()
    train_df, val_df, eval_df = _load_augmented_frames()
    sources = combo._decision_sources(val_df, eval_df, stack["parent"])
    dec_val, dec_eval = sources[str(cfg["source"])]

    baseline_seq_cols = list(stack["deep_payload"]["seq_cols"])
    variants = _build_variants(train_df, baseline_seq_cols)
    rows: list[dict[str, Any]] = []
    contracts: dict[str, Any] = {
        "model_id": MODEL_ID,
        "base_candidate": CANDIDATE,
        "policy": {
            "parent_fallback_inputs_fixed": True,
            "teacher_features_retained_in_new_deep_variants": True,
            "required_regime_prefix": STICKY_PREFIX,
            "forbidden_legacy_regime_prefixes": [LEGACY_REGIME_PREFIX, LEGACY_V4_PREFIX],
            "forbidden_raw_m7_price_features": sorted(RAW_M7_PRICE_FEATURES),
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
                seed=7527 + i * 97,
                out_dir=OUT_DIR / variant.name,
            )
            val_q = v27._predict_all(trained["model"], val_df, trained["seq_cols"], trained["norm"])
            eval_q = v27._predict_all(trained["model"], eval_df, trained["seq_cols"], trained["norm"])
            contracts["variants"][variant.name] = {
                "type": "deep_input_retrain",
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
        raise RuntimeError("baseline_01965 precision value was not reproduced; deep input sweep is invalid")
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
