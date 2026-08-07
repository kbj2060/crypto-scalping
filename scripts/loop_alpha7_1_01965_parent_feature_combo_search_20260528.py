#!/usr/bin/env python3
from __future__ import annotations

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
    FALLBACK_PARENT,
    FALLBACK_SUMMARY,
    PRIMARY_PARENT,
    _combine_primary_fallback,
    _load_best_scale_runtime,
    _predict_scaled,
    _train_parent,
)
from scripts.precision_retest_01965_alpha7_combo_20260527 import CANDIDATE, _cfg_from_results, _eval  # noqa: E402
from scripts.test_alpha7_1_01965_deep_input_feature_sweep_20260527 import (  # noqa: E402
    RAW_M7_PRICE_FEATURES,
    STICKY_PREFIX,
    TEACHER_FEATURES,
    _assert_deep_contract,
    _load_augmented_frames,
    _unique,
)
from scripts.test_alpha7_1_01965_deep_input_micro_ablation_20260527 import (  # noqa: E402
    STICKY_DIRECTION,
    STICKY_RISK,
    TEACHER_EDGES,
)
from scripts.test_alpha7_1_01965_input_feature_contract_sweep_20260527 import (  # noqa: E402
    _assert_feature_contract,
    _load_training_frames,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


MODEL_ID = "alpha7_1_01965_parent_feature_combo_search_20260528"
OUT_DIR = ROOT / f"tmp/causal_regen_20260516/{MODEL_ID}"
SUMMARY_OUT = OUT_DIR / "summary.json"
GRID_OUT = OUT_DIR / "grid.csv"
FEATURE_CONTRACT_OUT = OUT_DIR / "feature_contracts.json"
PROGRESS_JSONL = OUT_DIR / "progress.jsonl"


@dataclass(frozen=True)
class ParentSpec:
    name: str
    primary_cols: list[str] | None
    fallback_cols: list[str] | None
    train_fallback: bool
    seed: int
    notes: str


def _available(frame: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in frame.columns]


def _sanitize(cols: list[str]) -> list[str]:
    return [
        c
        for c in cols
        if c not in RAW_M7_PRICE_FEATURES
        and not str(c).startswith("clean_regime4_2024_unsup_v1_")
        and not str(c).startswith("clean_regime_2024_unsup_v4_")
    ]


def _corr_dedupe(frame: pd.DataFrame, cols: list[str], *, threshold: float) -> list[str]:
    cols = [c for c in _unique(cols) if c in frame.columns or c in {"side_hint", "mom_21d", "abs_mom_21d", "mom_3d", "abs_mom_3d", "mom_1d", "abs_mom_1d"}]
    numeric = [c for c in cols if c in frame.columns and pd.api.types.is_numeric_dtype(frame[c])]
    derivable = [c for c in cols if c not in frame.columns]
    if not numeric:
        return derivable
    x = frame[numeric].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    corr = x.corr().abs().fillna(0.0)
    kept: list[str] = []
    for c in numeric:
        if not kept or float(corr.loc[c, kept].max()) < threshold:
            kept.append(c)
    return [*kept, *derivable]


def _build_specs(
    primary_train: pd.DataFrame,
    fallback_train: pd.DataFrame,
    primary_parent: dict[str, Any],
    fallback_parent: dict[str, Any],
) -> list[ParentSpec]:
    primary_base = _sanitize(list(primary_parent["feature_cols"]))
    fallback_base = _sanitize(list(fallback_parent["feature_cols"]))
    sticky_risk_p = _available(primary_train, STICKY_RISK)
    sticky_dir_p = _available(primary_train, STICKY_DIRECTION)
    sticky_risk_f = _available(fallback_train, STICKY_RISK)
    sticky_dir_f = _available(fallback_train, STICKY_DIRECTION)
    teacher_edges_p = _available(primary_train, TEACHER_EDGES)
    teacher_edges_f = _available(fallback_train, TEACHER_EDGES)

    primary_risk = _unique([*primary_base, *TEACHER_FEATURES, *teacher_edges_p, *sticky_risk_p])
    primary_dir = _unique([*primary_base, *TEACHER_FEATURES, *teacher_edges_p, *sticky_dir_p])
    primary_all = _unique([*primary_base, *TEACHER_FEATURES, *teacher_edges_p, *sticky_risk_p, *sticky_dir_p])
    fallback_risk = _unique([*fallback_base, *TEACHER_FEATURES, *teacher_edges_f, *sticky_risk_f])
    fallback_dir = _unique([*fallback_base, *TEACHER_FEATURES, *teacher_edges_f, *sticky_dir_f])
    fallback_all = _unique([*fallback_base, *TEACHER_FEATURES, *teacher_edges_f, *sticky_risk_f, *sticky_dir_f])

    specs = [
        ParentSpec("baseline_01965", None, None, False, 0, "unchanged primary/fallback champion"),
        ParentSpec("primary_risk_edges", primary_risk, None, False, 2101, "primary retrain only; sticky risk + teacher edges"),
        ParentSpec("primary_direction_edges", primary_dir, None, False, 2138, "primary retrain only; sticky direction + teacher edges"),
        ParentSpec("primary_all_small", primary_all, None, False, 2175, "primary retrain only; sticky risk+direction + teacher edges"),
        ParentSpec(
            "primary_risk_edges_dedup95",
            _corr_dedupe(primary_train, primary_risk, threshold=0.95),
            None,
            False,
            2212,
            "primary retrain only; sticky risk + teacher edges; corr dedupe 0.95",
        ),
        ParentSpec(
            "primary_direction_edges_dedup95",
            _corr_dedupe(primary_train, primary_dir, threshold=0.95),
            None,
            False,
            2249,
            "primary retrain only; sticky direction + teacher edges; corr dedupe 0.95",
        ),
        ParentSpec("full_risk_edges", primary_risk, fallback_risk, True, 2286, "primary+fallback retrain; sticky risk + teacher edges"),
        ParentSpec("full_direction_edges", primary_dir, fallback_dir, True, 2323, "primary+fallback retrain; sticky direction + teacher edges"),
        ParentSpec("full_all_small", primary_all, fallback_all, True, 2360, "primary+fallback retrain; sticky risk+direction + teacher edges"),
    ]
    return specs


def _score_val(row: pd.Series) -> float:
    pnl = float(row["pnl"])
    mdd = abs(float(row["mdd"]))
    wr = float(row["wr"])
    trades = int(row["trades"])
    if trades < 60:
        return -1e9 + pnl
    return pnl - 1.5 * mdd + 35.0 * wr - 0.02 * trades


def _run_spec(
    *,
    spec: ParentSpec,
    primary_train: pd.DataFrame,
    primary_eval: pd.DataFrame,
    fallback_train: pd.DataFrame,
    fallback_eval: pd.DataFrame,
    primary_parent: dict[str, Any],
    fallback_parent: dict[str, Any],
    fallback_rt: Any,
    val_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    val_q: np.ndarray,
    eval_q: np.ndarray,
    stack: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if spec.primary_cols is None:
        sources = combo._decision_sources(val_df, eval_df, stack["parent"])
        dec_val, dec_eval = sources[str(cfg["source"])]
        contract = {
            "type": "baseline_parent_models",
            "primary_feature_count": int(len(primary_parent["feature_cols"])),
            "fallback_feature_count": int(len(fallback_parent["feature_cols"])),
        }
    else:
        _assert_feature_contract(primary_train, spec.primary_cols, name=f"{spec.name}:primary_train")
        _assert_feature_contract(primary_eval, spec.primary_cols, name=f"{spec.name}:primary_eval")
        variant_dir = OUT_DIR / spec.name
        (variant_dir / "primary").mkdir(parents=True, exist_ok=True)
        new_primary, primary_rt, primary_summary = _train_parent(
            train_all=primary_train,
            eval_df=primary_eval,
            feature_cols=spec.primary_cols,
            seed=spec.seed,
            out_dir=variant_dir / "primary",
        )
        if spec.train_fallback:
            assert spec.fallback_cols is not None
            _assert_feature_contract(fallback_train, spec.fallback_cols, name=f"{spec.name}:fallback_train")
            _assert_feature_contract(fallback_eval, spec.fallback_cols, name=f"{spec.name}:fallback_eval")
            (variant_dir / "fallback").mkdir(parents=True, exist_ok=True)
            new_fallback, new_fallback_rt, fallback_summary = _train_parent(
                train_all=fallback_train,
                eval_df=fallback_eval,
                feature_cols=spec.fallback_cols,
                seed=spec.seed + 503,
                out_dir=variant_dir / "fallback",
            )
        else:
            new_fallback = fallback_parent
            new_fallback_rt = fallback_rt
            fallback_summary = {"type": "baseline_fallback_unchanged"}

        p_val = _predict_scaled(new_primary, val_df, primary_rt)
        p_eval = _predict_scaled(new_primary, eval_df, primary_rt)
        f_val = _predict_scaled(new_fallback, val_df, new_fallback_rt)
        f_eval = _predict_scaled(new_fallback, eval_df, new_fallback_rt)
        dec_val = _combine_primary_fallback(p_val, f_val)
        dec_eval = _combine_primary_fallback(p_eval, f_eval)
        contract = {
            "type": "parent_feature_combo_retrain",
            "notes": spec.notes,
            "primary_feature_count": int(len(spec.primary_cols)),
            "primary_teacher_count": int(sum(c.startswith("teacher_") for c in spec.primary_cols)),
            "primary_sticky_v2_count": int(sum(c.startswith(STICKY_PREFIX) for c in spec.primary_cols)),
            "primary_feature_cols": spec.primary_cols,
            "primary_summary": primary_summary,
            "fallback_retrained": bool(spec.train_fallback),
            "fallback_feature_count": int(len(spec.fallback_cols or fallback_parent["feature_cols"])),
            "fallback_teacher_count": int(sum(c.startswith("teacher_") for c in (spec.fallback_cols or []))),
            "fallback_sticky_v2_count": int(sum(c.startswith(STICKY_PREFIX) for c in (spec.fallback_cols or []))),
            "fallback_feature_cols": spec.fallback_cols,
            "fallback_summary": fallback_summary,
        }

    rows = _eval_rows(
        variant=spec.name,
        val_df=val_df,
        eval_df=eval_df,
        val_q=val_q,
        eval_q=eval_q,
        dec_val=dec_val,
        dec_eval=dec_eval,
        stack=stack,
        cfg=cfg,
    )
    return rows, contract


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
    PROGRESS_JSONL.write_text("", encoding="utf-8")
    cfg = _cfg_from_results()
    if cfg.get("source") != "alpha7_combo_primary_fallback":
        raise RuntimeError(f"01965 source contract changed: {cfg.get('source')}")

    stack = combo._load_stack()
    val_df, eval_df = combo._load_frames()
    train_df, _, _ = _load_augmented_frames()
    primary_train, primary_eval, fallback_train, fallback_eval = _load_training_frames()
    primary_parent = joblib.load(PRIMARY_PARENT)
    fallback_parent = joblib.load(FALLBACK_PARENT)
    fallback_rt = _load_best_scale_runtime(FALLBACK_SUMMARY)
    specs = _build_specs(primary_train, fallback_train, primary_parent, fallback_parent)
    val_q = v27._predict_all(stack["deep_model"], val_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    eval_q = v27._predict_all(stack["deep_model"], eval_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])

    rows: list[dict[str, Any]] = []
    contracts: dict[str, Any] = {
        "model_id": MODEL_ID,
        "base_candidate": CANDIDATE,
        "policy": {
            "deep_input_fixed": True,
            "teacher_features_retained": True,
            "required_regime_prefix": STICKY_PREFIX,
            "raw_m7_price_features_removed_from_new_variants": sorted(RAW_M7_PRICE_FEATURES),
            "selection_uses_2026": False,
        },
        "variants": {},
    }
    for spec in specs:
        print(json.dumps({"event": "start", "variant": spec.name, "fallback": spec.train_fallback}, ensure_ascii=False), flush=True)
        spec_rows, contract = _run_spec(
            spec=spec,
            primary_train=primary_train,
            primary_eval=primary_eval,
            fallback_train=fallback_train,
            fallback_eval=fallback_eval,
            primary_parent=primary_parent,
            fallback_parent=fallback_parent,
            fallback_rt=fallback_rt,
            val_df=val_df,
            eval_df=eval_df,
            val_q=val_q,
            eval_q=eval_q,
            stack=stack,
            cfg=cfg,
        )
        rows.extend(spec_rows)
        contracts["variants"][spec.name] = contract
        grid_tmp = pd.DataFrame(rows)
        row = grid_tmp[(grid_tmp["variant"].eq(spec.name)) & (grid_tmp["period"].eq("val")) & (grid_tmp["cost"].eq(3))].iloc[0].to_dict()
        row["val_score"] = _score_val(pd.Series(row))
        with PROGRESS_JSONL.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"event": "done", **row}, ensure_ascii=False, default=_json_default) + "\n")
        print(json.dumps({"event": "done", "variant": spec.name, "val_cost3_pnl": row["pnl"], "val_cost3_mdd": row["mdd"], "val_score": row["val_score"]}, ensure_ascii=False), flush=True)

    grid = pd.DataFrame(rows)
    grid.to_csv(GRID_OUT, index=False)
    FEATURE_CONTRACT_OUT.write_text(json.dumps(contracts, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    cost3 = grid[grid["cost"].eq(3)].copy()
    baseline_oos = cost3[(cost3["variant"].eq("baseline_01965")) & (cost3["period"].eq("oos"))]
    if baseline_oos.empty or abs(float(baseline_oos.iloc[0]["pnl"]) - 274.53249150592416) > 1e-6:
        raise RuntimeError("baseline_01965 precision value was not reproduced; parent combo search is invalid")
    val_rank = cost3[cost3["period"].eq("val")].copy()
    val_rank["val_score"] = val_rank.apply(_score_val, axis=1)
    summary = {
        "model_id": MODEL_ID,
        "base_candidate": CANDIDATE,
        "selection_uses_2026": False,
        "feature_contracts": str(FEATURE_CONTRACT_OUT),
        "grid": str(GRID_OUT),
        "progress": str(PROGRESS_JSONL),
        "cost3_val_rank": val_rank.sort_values("val_score", ascending=False).to_dict(orient="records"),
        "cost3_oos_rank": cost3[cost3["period"].eq("oos")].sort_values("pnl", ascending=False).to_dict(orient="records"),
    }
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(SUMMARY_OUT), "grid": str(GRID_OUT), "feature_contracts": str(FEATURE_CONTRACT_OUT), "progress": str(PROGRESS_JSONL)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
