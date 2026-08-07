#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import loop_alpha3_1_alpha6_alpha7_combo_search_until_0800_20260527 as combo  # noqa: E402
from scripts import train_eval_hf_v13_deep_alpha_candidate_expansion_v27 as v27  # noqa: E402
from scripts.precision_retest_01965_alpha7_combo_20260527 import CANDIDATE, _cfg_from_results  # noqa: E402
from scripts.test_alpha7_1_01965_deep_input_feature_sweep_20260527 import (  # noqa: E402
    RAW_M7_PRICE_FEATURES,
    STICKY_PREFIX,
    TEACHER_FEATURES,
    DeepVariant,
    _assert_deep_contract,
    _eval_rows,
    _load_augmented_frames,
    _numeric_available,
    _train_deep_variant,
    _unique,
)
from scripts.test_alpha7_1_01965_deep_input_micro_ablation_20260527 import (  # noqa: E402
    EXISTING_TEACHER,
    STICKY_DIRECTION,
    STICKY_RISK,
    TEACHER_EDGES,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


MODEL_ID = "alpha7_1_01965_deep_feature_combo_search_20260528"
OUT_DIR = ROOT / f"tmp/causal_regen_20260516/{MODEL_ID}"
SUMMARY_OUT = OUT_DIR / "summary.json"
GRID_OUT = OUT_DIR / "grid.csv"
FEATURE_CONTRACT_OUT = OUT_DIR / "feature_contracts.json"
PROGRESS_JSONL = OUT_DIR / "progress.jsonl"
SCREEN_EPOCHS = 50
FINAL_EPOCHS = 120
TOP_K_FINAL = 3


@dataclass(frozen=True)
class ComboSpec:
    name: str
    cols: list[str] | None
    stage: str
    epochs: int
    seed: int
    notes: str


def _available(frame: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in frame.columns]


def _safe_base_cols(baseline_seq_cols: list[str]) -> list[str]:
    return [c for c in baseline_seq_cols if c not in RAW_M7_PRICE_FEATURES]


def _corr_dedupe(frame: pd.DataFrame, cols: list[str], *, threshold: float) -> list[str]:
    cols = [c for c in _unique(cols) if c in frame.columns]
    if not cols:
        return []
    x = frame[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    corr = x.corr().abs().fillna(0.0)
    kept: list[str] = []
    for c in cols:
        if c not in corr.columns:
            continue
        if not kept:
            kept.append(c)
            continue
        if float(corr.loc[c, kept].max()) < threshold:
            kept.append(c)
    return kept


def _score_val(row: pd.Series) -> float:
    pnl = float(row["pnl"])
    mdd = abs(float(row["mdd"]))
    wr = float(row["wr"])
    trades = int(row["trades"])
    if trades < 60:
        return -1e9 + pnl
    return pnl - 1.5 * mdd + 35.0 * wr - 0.02 * trades


def _core_cols(frame: pd.DataFrame) -> list[str]:
    cols = [
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
    ]
    return _numeric_available(frame, cols)


def _build_screen_specs(frame: pd.DataFrame, baseline_seq_cols: list[str]) -> list[ComboSpec]:
    safe_base = _safe_base_cols(baseline_seq_cols)
    teacher_edges = _available(frame, TEACHER_EDGES)
    sticky_risk = _available(frame, STICKY_RISK)
    sticky_direction = _available(frame, STICKY_DIRECTION)
    sticky_all = [c for c in frame.columns if str(c).startswith(STICKY_PREFIX)]
    core = _core_cols(frame)

    candidates: list[tuple[str, list[str], str]] = [
        ("safe_base_plus_risk_edges", [*safe_base, *teacher_edges, *sticky_risk], "remove raw M7 prices; add teacher edge and sticky risk"),
        ("safe_base_plus_direction_edges", [*safe_base, *teacher_edges, *sticky_direction], "remove raw M7 prices; add teacher edge and sticky direction"),
        ("safe_base_plus_all_sticky_edges", [*safe_base, *teacher_edges, *sticky_all], "remove raw M7 prices; add all sticky_v2 and teacher edge"),
        ("core_plus_risk_edges", [*core, *TEACHER_FEATURES, *sticky_risk], "model-architect core plus risk state"),
        ("core_plus_direction_edges", [*core, *TEACHER_FEATURES, *sticky_direction], "model-architect core plus direction state"),
        ("core_plus_all_sticky_edges", [*core, *TEACHER_FEATURES, *sticky_all], "model-architect core plus all sticky_v2"),
        ("compact_teacher_ai_risk", [*EXISTING_TEACHER, *teacher_edges, *sticky_risk, *[c for c in core if c.startswith('ai_') or c in {'log_return', 'volatility_z', 'garch_vol_z', 'net_taker_ratio', 'oi_change_rate'}]], "compact AI/teacher/risk only"),
    ]
    specs: list[ComboSpec] = []
    seed = 9101
    for name, cols, notes in candidates:
        cols = _unique([c for c in cols if c in frame.columns])
        specs.append(ComboSpec(name=name, cols=cols[:80], stage="screen", epochs=SCREEN_EPOCHS, seed=seed, notes=notes))
        seed += 37
        dedup985 = _corr_dedupe(frame, cols, threshold=0.985)
        specs.append(ComboSpec(name=f"{name}_dedup985", cols=dedup985[:80], stage="screen", epochs=SCREEN_EPOCHS, seed=seed, notes=f"{notes}; corr dedupe 0.985"))
        seed += 37
        dedup95 = _corr_dedupe(frame, cols, threshold=0.95)
        specs.append(ComboSpec(name=f"{name}_dedup95", cols=dedup95[:80], stage="screen", epochs=SCREEN_EPOCHS, seed=seed, notes=f"{notes}; corr dedupe 0.95"))
        seed += 37
    return specs


def _run_spec(
    *,
    spec: ComboSpec,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    dec_val: pd.DataFrame,
    dec_eval: pd.DataFrame,
    stack: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if spec.cols is None:
        baseline_seq_cols = list(stack["deep_payload"]["seq_cols"])
        val_q = v27._predict_all(stack["deep_model"], val_df, baseline_seq_cols, stack["deep_payload"]["norm"])
        eval_q = v27._predict_all(stack["deep_model"], eval_df, baseline_seq_cols, stack["deep_payload"]["norm"])
        contract = {"type": "baseline_deep_model", "seq_count": int(len(baseline_seq_cols)), "seq_cols": baseline_seq_cols}
    else:
        _assert_deep_contract(train_df, spec.cols, name=f"{spec.name}:train")
        _assert_deep_contract(val_df, spec.cols, name=f"{spec.name}:val")
        _assert_deep_contract(eval_df, spec.cols, name=f"{spec.name}:oos")
        trained = _train_deep_variant(
            train_df,
            spec.cols,
            epochs=spec.epochs,
            seed=spec.seed,
            out_dir=OUT_DIR / spec.stage / spec.name,
        )
        val_q = v27._predict_all(trained["model"], val_df, trained["seq_cols"], trained["norm"])
        eval_q = v27._predict_all(trained["model"], eval_df, trained["seq_cols"], trained["norm"])
        contract = {
            "type": "deep_feature_combo_retrain",
            "stage": spec.stage,
            "epochs": int(spec.epochs),
            "seed": int(spec.seed),
            "notes": spec.notes,
            "train_samples": int(trained["train_samples"]),
            "seq_count": int(len(spec.cols)),
            "teacher_count": int(sum(c.startswith("teacher_") for c in spec.cols)),
            "sticky_v2_count": int(sum(c.startswith(STICKY_PREFIX) for c in spec.cols)),
            "seq_cols": spec.cols,
            "artifact": str(OUT_DIR / spec.stage / spec.name / "deep_model.pt"),
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


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_JSONL.write_text("", encoding="utf-8")
    cfg = _cfg_from_results()
    if cfg.get("source") != "alpha7_combo_primary_fallback":
        raise RuntimeError(f"01965 source contract changed: {cfg.get('source')}")

    stack = combo._load_stack()
    train_df, val_df, eval_df = _load_augmented_frames()
    sources = combo._decision_sources(val_df, eval_df, stack["parent"])
    dec_val, dec_eval = sources[str(cfg["source"])]
    baseline_seq_cols = list(stack["deep_payload"]["seq_cols"])

    specs = [ComboSpec("baseline_01965", None, "baseline", 0, 0, "unchanged champion deep scout")]
    specs.extend(_build_screen_specs(train_df, baseline_seq_cols))

    all_rows: list[dict[str, Any]] = []
    contracts: dict[str, Any] = {
        "model_id": MODEL_ID,
        "base_candidate": CANDIDATE,
        "policy": {
            "parent_fallback_inputs_fixed": True,
            "selection_uses_2026": False,
            "screen_epochs": SCREEN_EPOCHS,
            "final_epochs": FINAL_EPOCHS,
            "top_k_final": TOP_K_FINAL,
            "raw_m7_price_features_removed_from_new_variants": sorted(RAW_M7_PRICE_FEATURES),
            "required_regime_prefix": STICKY_PREFIX,
        },
        "variants": {},
    }

    for spec in specs:
        print(json.dumps({"event": "start", "variant": spec.name, "stage": spec.stage, "epochs": spec.epochs, "seq_count": 0 if spec.cols is None else len(spec.cols)}, ensure_ascii=False), flush=True)
        rows, contract = _run_spec(
            spec=spec,
            train_df=train_df,
            val_df=val_df,
            eval_df=eval_df,
            dec_val=dec_val,
            dec_eval=dec_eval,
            stack=stack,
            cfg=cfg,
        )
        all_rows.extend(rows)
        contracts["variants"][spec.name] = contract
        grid_tmp = pd.DataFrame(all_rows)
        val_cost3 = grid_tmp[(grid_tmp["period"].eq("val")) & (grid_tmp["cost"].eq(3))].copy()
        row = val_cost3[val_cost3["variant"].eq(spec.name)].iloc[0].to_dict()
        row["val_score"] = _score_val(pd.Series(row))
        with PROGRESS_JSONL.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"event": "done", **row}, ensure_ascii=False, default=_json_default) + "\n")
        print(json.dumps({"event": "done", "variant": spec.name, "val_cost3_pnl": row["pnl"], "val_cost3_mdd": row["mdd"], "val_score": row["val_score"]}, ensure_ascii=False), flush=True)

    grid = pd.DataFrame(all_rows)
    screen_val = grid[(grid["period"].eq("val")) & (grid["cost"].eq(3)) & (~grid["variant"].eq("baseline_01965"))].copy()
    screen_val["val_score"] = screen_val.apply(_score_val, axis=1)
    finalists = screen_val.sort_values("val_score", ascending=False).head(TOP_K_FINAL)["variant"].tolist()

    final_rows: list[dict[str, Any]] = []
    for rank, name in enumerate(finalists):
        old = next(s for s in specs if s.name == name)
        assert old.cols is not None
        spec = ComboSpec(
            name=f"{name}_final{FINAL_EPOCHS}",
            cols=old.cols,
            stage="final",
            epochs=FINAL_EPOCHS,
            seed=12001 + rank * 101,
            notes=f"final retest from screen={name}; {old.notes}",
        )
        print(json.dumps({"event": "start", "variant": spec.name, "stage": spec.stage, "epochs": spec.epochs, "seq_count": len(spec.cols)}, ensure_ascii=False), flush=True)
        rows, contract = _run_spec(
            spec=spec,
            train_df=train_df,
            val_df=val_df,
            eval_df=eval_df,
            dec_val=dec_val,
            dec_eval=dec_eval,
            stack=stack,
            cfg=cfg,
        )
        final_rows.extend(rows)
        contracts["variants"][spec.name] = contract
        val_row = pd.DataFrame(rows)
        val_row = val_row[(val_row["period"].eq("val")) & (val_row["cost"].eq(3))].iloc[0].to_dict()
        val_row["val_score"] = _score_val(pd.Series(val_row))
        with PROGRESS_JSONL.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"event": "done", **val_row}, ensure_ascii=False, default=_json_default) + "\n")
        print(json.dumps({"event": "done", "variant": spec.name, "val_cost3_pnl": val_row["pnl"], "val_cost3_mdd": val_row["mdd"], "val_score": val_row["val_score"]}, ensure_ascii=False), flush=True)

    if final_rows:
        grid = pd.concat([grid, pd.DataFrame(final_rows)], ignore_index=True)
    grid.to_csv(GRID_OUT, index=False)
    FEATURE_CONTRACT_OUT.write_text(json.dumps(contracts, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    cost3 = grid[grid["cost"].eq(3)].copy()
    baseline_oos = cost3[(cost3["variant"].eq("baseline_01965")) & (cost3["period"].eq("oos"))]
    if baseline_oos.empty or abs(float(baseline_oos.iloc[0]["pnl"]) - 274.53249150592416) > 1e-6:
        raise RuntimeError("baseline_01965 precision value was not reproduced; combo search is invalid")
    val_rank = cost3[cost3["period"].eq("val")].copy()
    val_rank["val_score"] = val_rank.apply(_score_val, axis=1)
    summary = {
        "model_id": MODEL_ID,
        "base_candidate": CANDIDATE,
        "selection_uses_2026": False,
        "feature_contracts": str(FEATURE_CONTRACT_OUT),
        "grid": str(GRID_OUT),
        "progress": str(PROGRESS_JSONL),
        "screen_finalists": finalists,
        "cost3_val_rank": val_rank.sort_values("val_score", ascending=False).to_dict(orient="records"),
        "cost3_oos_rank": cost3[cost3["period"].eq("oos")].sort_values("pnl", ascending=False).to_dict(orient="records"),
    }
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(SUMMARY_OUT), "grid": str(GRID_OUT), "feature_contracts": str(FEATURE_CONTRACT_OUT), "progress": str(PROGRESS_JSONL)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
