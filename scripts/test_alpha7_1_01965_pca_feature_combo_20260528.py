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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
    _load_augmented_frames,
    _train_deep_variant,
    _unique,
)
from scripts.test_alpha7_1_01965_deep_input_micro_ablation_20260527 import (  # noqa: E402
    EXISTING_TEACHER,
    STICKY_DIRECTION,
    STICKY_RISK,
)
from scripts.test_alpha7_1_01965_input_feature_contract_sweep_20260527 import _load_training_frames  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


MODEL_ID = "alpha7_1_01965_pca_feature_combo_20260528"
OUT_DIR = ROOT / f"tmp/causal_regen_20260516/{MODEL_ID}"
SUMMARY_OUT = OUT_DIR / "summary.json"
GRID_OUT = OUT_DIR / "grid.csv"
FEATURE_CONTRACT_OUT = OUT_DIR / "feature_contracts.json"
PROGRESS_JSONL = OUT_DIR / "progress.jsonl"
PCA_PREFIX = "alpha7pca_"
DERIVABLE_FEATURES = {
    "side_hint",
    "mom_21d",
    "abs_mom_21d",
    "mom_3d",
    "abs_mom_3d",
    "mom_1d",
    "abs_mom_1d",
}


@dataclass(frozen=True)
class PcaSpec:
    name: str
    mode: str
    cols: list[str] | None
    epochs: int
    seed: int
    notes: str


def _available(frame: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in frame.columns and pd.api.types.is_numeric_dtype(frame[c])]


def _cols(frame: pd.DataFrame, prefix: str) -> list[str]:
    return sorted(c for c in frame.columns if str(c).startswith(prefix) and pd.api.types.is_numeric_dtype(frame[c]))


def _safe_baseline(cols: list[str]) -> list[str]:
    return [c for c in cols if c not in RAW_M7_PRICE_FEATURES]


def _family_sources(frame: pd.DataFrame) -> dict[str, list[str]]:
    m7 = [c for c in _cols(frame, "m7_") if c not in RAW_M7_PRICE_FEATURES and not c.startswith(("m7_target_",))]
    ai = _cols(frame, "ai_")
    teacher = _available(frame, TEACHER_FEATURES)
    sticky = _cols(frame, STICKY_PREFIX)
    return {"m7": m7, "ai": ai, "teacher": teacher, "sticky": sticky}


def _fit_pca(train_fit: pd.DataFrame, families: dict[str, list[str]], dims: dict[str, int]) -> dict[str, Any]:
    fitted: dict[str, Any] = {}
    for name, cols in families.items():
        n = min(int(dims.get(name, 0)), len(cols))
        if n <= 0:
            continue
        x = train_fit[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        scaler = StandardScaler()
        z = scaler.fit_transform(x)
        pca = PCA(n_components=n, random_state=0)
        pca.fit(z)
        fitted[name] = {"cols": cols, "scaler": scaler, "pca": pca, "n": n}
    return fitted


def _apply_pca(frame: pd.DataFrame, fitted: dict[str, Any], *, tag: str) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    out = frame.copy()
    pca_cols: list[str] = []
    audit: dict[str, Any] = {}
    for fam, payload in fitted.items():
        cols = list(payload["cols"])
        missing = [c for c in cols if c not in out.columns]
        if missing:
            raise RuntimeError(f"PCA source columns missing for {tag}:{fam}: {missing[:20]}")
        x = out[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        z = payload["scaler"].transform(x)
        v = payload["pca"].transform(z)
        names = [f"{PCA_PREFIX}{tag}_{fam}_{i:02d}" for i in range(v.shape[1])]
        for i, c in enumerate(names):
            out[c] = v[:, i].astype(np.float32)
        pca_cols.extend(names)
        audit[fam] = {
            "source_count": int(len(cols)),
            "n_components": int(v.shape[1]),
            "explained_variance_ratio_sum": float(np.sum(payload["pca"].explained_variance_ratio_)),
            "source_cols": cols,
            "pca_cols": names,
        }
    return out, pca_cols, audit


def _assert_pca_cols(frame: pd.DataFrame, cols: list[str], *, name: str) -> None:
    missing = [c for c in cols if c not in frame.columns and c not in DERIVABLE_FEATURES]
    if missing:
        raise RuntimeError(f"{name}: missing PCA feature columns: {missing[:30]}")
    forbidden = [
        c
        for c in cols
        if c in RAW_M7_PRICE_FEATURES
        or str(c).startswith("clean_regime4_2024_unsup_v1_")
        or str(c).startswith("clean_regime_2024_unsup_v4_")
        or any(tok in str(c).lower() for tok in ("target", "label", "future", "cash_after"))
    ]
    if forbidden:
        raise RuntimeError(f"{name}: forbidden columns selected: {forbidden[:30]}")


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


def _score_val(row: pd.Series) -> float:
    if int(row["trades"]) < 60:
        return -1e9 + float(row["pnl"])
    return float(row["pnl"]) - 1.5 * abs(float(row["mdd"])) + 35.0 * float(row["wr"]) - 0.02 * int(row["trades"])


def _deep_specs(train_df: pd.DataFrame, baseline_cols: list[str], pca_cols: list[str]) -> list[PcaSpec]:
    safe = _safe_baseline(baseline_cols)
    market = [
        c
        for c in safe
        if c in train_df.columns
        and not c.startswith(("m7_", "ai_", "teacher_", STICKY_PREFIX))
    ]
    risk = _available(train_df, STICKY_RISK)
    direction = _available(train_df, STICKY_DIRECTION)
    teacher = _available(train_df, EXISTING_TEACHER)
    return [
        PcaSpec("baseline_01965", "deep", None, 0, 0, "unchanged champion"),
        PcaSpec("deep_safe_plus_family_pca", "deep", _unique([*safe, *risk, *pca_cols])[:80], 70, 3101, "safe baseline + sticky risk + family PCA"),
        PcaSpec("deep_market_teacher_risk_pca", "deep", _unique([*market, *teacher, *risk, *pca_cols])[:80], 70, 3138, "market + teacher + sticky risk + family PCA"),
        PcaSpec("deep_market_teacher_direction_pca", "deep", _unique([*market, *teacher, *direction, *pca_cols])[:80], 70, 3175, "market + teacher + sticky direction + family PCA"),
    ]


def _parent_specs(primary_train: pd.DataFrame, primary_parent: dict[str, Any], pca_cols: list[str]) -> list[PcaSpec]:
    base = _safe_baseline(list(primary_parent["feature_cols"]))
    risk = _available(primary_train, STICKY_RISK)
    teacher = _available(primary_train, TEACHER_FEATURES)
    return [
        PcaSpec("parent_primary_safe_pca", "parent", _unique([*base, *teacher, *risk, *pca_cols]), 0, 4101, "primary only; safe base + teacher/risk + family PCA"),
    ]


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
    primary_train, primary_eval, fallback_train, fallback_eval = _load_training_frames()
    primary_parent = joblib.load(PRIMARY_PARENT)
    fallback_parent = joblib.load(FALLBACK_PARENT)
    fallback_rt = _load_best_scale_runtime(FALLBACK_SUMMARY)

    train_fit = train_df.reset_index(drop=True)
    deep_fitted = _fit_pca(train_fit, _family_sources(train_fit), {"m7": 8, "ai": 5, "teacher": 3, "sticky": 5})
    train_df, deep_pca_cols, deep_pca_audit = _apply_pca(train_df, deep_fitted, tag="deep")
    val_df, _, _ = _apply_pca(val_df, deep_fitted, tag="deep")
    eval_df, _, _ = _apply_pca(eval_df, deep_fitted, tag="deep")

    primary_fit = primary_train[primary_train["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    parent_fitted = _fit_pca(primary_fit, _family_sources(primary_fit), {"m7": 8, "ai": 5, "teacher": 3, "sticky": 5})
    primary_train, parent_pca_cols, parent_pca_audit = _apply_pca(primary_train, parent_fitted, tag="parent")
    primary_eval, _, _ = _apply_pca(primary_eval, parent_fitted, tag="parent")
    val_df, _, _ = _apply_pca(val_df, parent_fitted, tag="parent")
    eval_df, _, _ = _apply_pca(eval_df, parent_fitted, tag="parent")

    rows: list[dict[str, Any]] = []
    contracts: dict[str, Any] = {
        "model_id": MODEL_ID,
        "base_candidate": CANDIDATE,
        "policy": {
            "selection_uses_2026": False,
            "pca_fit_policy": "fit only on 2025 train segment before 2025-10-01; transform val/oos",
            "family_pca_only": True,
            "raw_m7_price_features_removed_from_new_variants": sorted(RAW_M7_PRICE_FEATURES),
        },
        "pca_audit": {"deep": deep_pca_audit, "parent": parent_pca_audit},
        "variants": {},
    }

    for spec in _deep_specs(train_df, list(stack["deep_payload"]["seq_cols"]), deep_pca_cols):
        print(json.dumps({"event": "start", "variant": spec.name, "mode": spec.mode, "epochs": spec.epochs}, ensure_ascii=False), flush=True)
        if spec.cols is None:
            val_q = v27._predict_all(stack["deep_model"], val_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
            eval_q = v27._predict_all(stack["deep_model"], eval_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
            contracts["variants"][spec.name] = {"type": "baseline_deep_model", "seq_count": len(stack["deep_payload"]["seq_cols"])}
        else:
            _assert_pca_cols(train_df, spec.cols, name=f"{spec.name}:train")
            _assert_pca_cols(val_df, spec.cols, name=f"{spec.name}:val")
            _assert_pca_cols(eval_df, spec.cols, name=f"{spec.name}:oos")
            trained = _train_deep_variant(train_df, spec.cols, epochs=spec.epochs, seed=spec.seed, out_dir=OUT_DIR / spec.name)
            val_q = v27._predict_all(trained["model"], val_df, trained["seq_cols"], trained["norm"])
            eval_q = v27._predict_all(trained["model"], eval_df, trained["seq_cols"], trained["norm"])
            contracts["variants"][spec.name] = {
                "type": "deep_pca_retrain",
                "seq_count": len(spec.cols),
                "epochs": spec.epochs,
                "seed": spec.seed,
                "notes": spec.notes,
                "cols": spec.cols,
                "artifact": str(OUT_DIR / spec.name / "deep_model.pt"),
            }
        spec_rows = _eval_rows(
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
        rows.extend(spec_rows)
        val_row = pd.DataFrame(spec_rows).query("period == 'val' and cost == 3").iloc[0].to_dict()
        val_row["val_score"] = _score_val(pd.Series(val_row))
        with PROGRESS_JSONL.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"event": "done", **val_row}, ensure_ascii=False, default=_json_default) + "\n")
        print(json.dumps({"event": "done", "variant": spec.name, "val_cost3_pnl": val_row["pnl"], "val_cost3_mdd": val_row["mdd"], "val_score": val_row["val_score"]}, ensure_ascii=False), flush=True)

    baseline_val_q = v27._predict_all(stack["deep_model"], val_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    baseline_eval_q = v27._predict_all(stack["deep_model"], eval_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    for spec in _parent_specs(primary_train, primary_parent, parent_pca_cols):
        print(json.dumps({"event": "start", "variant": spec.name, "mode": spec.mode}, ensure_ascii=False), flush=True)
        assert spec.cols is not None
        _assert_pca_cols(primary_train, spec.cols, name=f"{spec.name}:primary_train")
        _assert_pca_cols(primary_eval, spec.cols, name=f"{spec.name}:primary_eval")
        variant_dir = OUT_DIR / spec.name / "primary"
        variant_dir.mkdir(parents=True, exist_ok=True)
        parent, rt, summary = _train_parent(
            train_all=primary_train,
            eval_df=primary_eval,
            feature_cols=spec.cols,
            seed=spec.seed,
            out_dir=variant_dir,
        )
        p_val = _predict_scaled(parent, val_df, rt)
        p_eval = _predict_scaled(parent, eval_df, rt)
        f_val = _predict_scaled(fallback_parent, val_df, fallback_rt)
        f_eval = _predict_scaled(fallback_parent, eval_df, fallback_rt)
        spec_rows = _eval_rows(
            variant=spec.name,
            val_df=val_df,
            eval_df=eval_df,
            val_q=baseline_val_q,
            eval_q=baseline_eval_q,
            dec_val=_combine_primary_fallback(p_val, f_val),
            dec_eval=_combine_primary_fallback(p_eval, f_eval),
            stack=stack,
            cfg=cfg,
        )
        rows.extend(spec_rows)
        contracts["variants"][spec.name] = {
            "type": "parent_primary_pca_retrain",
            "feature_count": len(spec.cols),
            "seed": spec.seed,
            "notes": spec.notes,
            "cols": spec.cols,
            "primary_summary": summary,
            "fallback_retrained": False,
        }
        val_row = pd.DataFrame(spec_rows).query("period == 'val' and cost == 3").iloc[0].to_dict()
        val_row["val_score"] = _score_val(pd.Series(val_row))
        with PROGRESS_JSONL.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"event": "done", **val_row}, ensure_ascii=False, default=_json_default) + "\n")
        print(json.dumps({"event": "done", "variant": spec.name, "val_cost3_pnl": val_row["pnl"], "val_cost3_mdd": val_row["mdd"], "val_score": val_row["val_score"]}, ensure_ascii=False), flush=True)

    grid = pd.DataFrame(rows)
    grid.to_csv(GRID_OUT, index=False)
    FEATURE_CONTRACT_OUT.write_text(json.dumps(contracts, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    cost3 = grid[grid["cost"].eq(3)].copy()
    baseline_oos = cost3[(cost3["variant"].eq("baseline_01965")) & (cost3["period"].eq("oos"))]
    if baseline_oos.empty or abs(float(baseline_oos.iloc[0]["pnl"]) - 274.53249150592416) > 1e-6:
        raise RuntimeError("baseline_01965 precision value was not reproduced; PCA combo search is invalid")
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
