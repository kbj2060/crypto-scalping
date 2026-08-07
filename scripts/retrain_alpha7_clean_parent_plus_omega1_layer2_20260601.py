#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_alpha7_tp_sl_action_score_20260526 import (  # noqa: E402
    SPLIT_TS,
    TP_COL,
    _combine_primary_fallback,
    _combo_metrics,
    _json_default,
    _predict_scaled,
    _read,
)
from scripts.retrain_alpha7_1_01965_tp_sl_decontam_20260528 import (  # noqa: E402
    _assert_feature_cols,
    _contract_report,
    _load_or_train,
)


MODEL_ID = "alpha7_clean_parent_plus_omega1_layer2_20260601"
BASE_CLEAN_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_submodel_01965_cleanfunding_v1_20260529"
TRAIN_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2025_alpha6_current_tail111_exact.csv"
EVAL_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2026_alpha6_current_tail111_exact.csv"
LAYER2_FEATURES_JSON = ROOT / "tmp/causal_regen_20260516/omega1_layer12_action_model_family_compare_20260531_fast/selected_features.json"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_clean_parent_plus_omega1_layer2_20260601"

OVERLAY_SOURCES_2025 = [
    ROOT / "data/splits/year_oos/training_features_2025.csv",
    ROOT / "tmp/causal_regen_20260516/omega1_hgb_teacher_m7zigzag_cleanrisk_20260531/trade_candidates_2025_alpha6_current_tail111_exact.csv",
    ROOT / "data/splits/year_oos/rl_training_2025_m7_zigzag_direction.csv",
    ROOT / "data/ensemble/supervised/omega1_dir3_patch_full_20260531/training_features_2025_omega1_dir3_patch_full_20260531.csv",
    ROOT / "data/ensemble/supervised/omega1_dir3_vsnlstm_full_20260531/training_features_2025_omega1_dir3_vsnlstm_full_20260531.csv",
]
OVERLAY_SOURCES_2026 = [
    ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
    ROOT / "tmp/causal_regen_20260516/omega1_hgb_teacher_m7zigzag_cleanrisk_20260531/trade_candidates_2026_alpha6_current_tail111_exact.csv",
    ROOT / "data/splits/year_oos/rl_training_2026_m7_zigzag_direction.csv",
    ROOT / "data/ensemble/supervised/omega1_dir3_patch_full_20260531/training_features_2026_rebuilt_omega1_dir3_patch_full_20260531.csv",
    ROOT / "data/ensemble/supervised/omega1_dir3_vsnlstm_full_20260531/training_features_2026_rebuilt_omega1_dir3_vsnlstm_full_20260531.csv",
]


def _load_feature_list(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
        raise ValueError(f"{path} must contain a JSON list of feature names")
    return list(dict.fromkeys(raw))


def _read_overlay(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    if "timestamp" not in df.columns:
        raise ValueError(f"{path} missing timestamp")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _overlay_exact(base: pd.DataFrame, sources: list[Path], wanted: list[str], *, tag: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = base.copy()
    report: dict[str, Any] = {"tag": tag, "sources": [], "requested": len(wanted), "added": [], "dropped_edge_rows": []}
    remaining = [c for c in wanted if c not in out.columns]
    for source in sources:
        if not remaining:
            break
        src = _read_overlay(source)
        have = [c for c in remaining if c in src.columns]
        if not have:
            report["sources"].append({"path": str(source), "added": []})
            continue
        missing_ts = out.loc[~out["timestamp"].isin(set(src["timestamp"])), "timestamp"]
        if len(missing_ts) > 0:
            head_ts = out["timestamp"].head(len(missing_ts)).reset_index(drop=True)
            tail_ts = out["timestamp"].tail(len(missing_ts)).reset_index(drop=True)
            missing_norm = missing_ts.reset_index(drop=True)
            if missing_norm.equals(head_ts):
                edge = "head"
            elif missing_norm.equals(tail_ts):
                edge = "tail"
            else:
                raise RuntimeError(f"{tag}: source {source} is missing non-edge timestamps: {missing_ts.head(20).tolist()}")
            report["dropped_edge_rows"].append({"path": str(source), "edge": edge, "rows": int(len(missing_ts)), "first": str(missing_ts.iloc[0]), "last": str(missing_ts.iloc[-1])})
            out = out.loc[out["timestamp"].isin(set(src["timestamp"]))].reset_index(drop=True)
        before = len(out)
        payload = src[["timestamp", *have]].copy()
        out = out.merge(payload, on="timestamp", how="left", validate="one_to_one")
        if len(out) != before:
            raise RuntimeError(f"{tag}: row count changed after overlay {source}: {before} -> {len(out)}")
        missing = out[have].isna().any(axis=1)
        if bool(missing.any()):
            raise RuntimeError(f"{tag}: source {source} produced missing values for {int(missing.sum())} rows")
        report["sources"].append({"path": str(source), "added": have})
        report["added"].extend(have)
        remaining = [c for c in remaining if c not in have]
    report["unavailable"] = remaining
    return out, report


def _active_count(dec: pd.DataFrame) -> int:
    action = pd.to_numeric(dec["action"], errors="coerce").fillna(0).astype(int)
    side = pd.to_numeric(dec["side"], errors="coerce").fillna(0).astype(int)
    return int(((action != 0) & (side != 0)).sum())


def _predict_combo(primary: dict[str, Any], fallback: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    primary_dec = _predict_scaled(primary, df, None)
    fallback_dec = _predict_scaled(fallback, df, None)
    combo_dec = _combine_primary_fallback(primary_dec, fallback_dec)
    return {
        "primary_active": _active_count(primary_dec),
        "fallback_active": _active_count(fallback_dec),
        "combo_active": _active_count(combo_dec),
        "combo_costs": _combo_metrics(df, combo_dec),
    }


def _row(name: str, val: dict[str, Any], oos: dict[str, Any], primary_cols: list[str], fallback_cols: list[str]) -> dict[str, Any]:
    vc3 = val["combo_costs"]["cost3"]
    oc3 = oos["combo_costs"]["cost3"]
    return {
        "variant": name,
        "primary_feature_count": len(primary_cols),
        "fallback_feature_count": len(fallback_cols),
        "val_cost3_pnl": float(vc3["pnl"]),
        "val_cost3_mdd": float(vc3["mdd"]),
        "val_cost3_trades": int(vc3["trades"]),
        "val_cost3_wr": float(vc3["wr"]),
        "oos_cost3_pnl": float(oc3["pnl"]),
        "oos_cost3_mdd": float(oc3["mdd"]),
        "oos_cost3_trades": int(oc3["trades"]),
        "oos_cost3_wr": float(oc3["wr"]),
        "oos_combo_active": int(oos["combo_active"]),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_all_base = _read(TRAIN_CSV)
    eval_base = _read(EVAL_CSV)
    val_base = train_all_base[train_all_base["timestamp"] >= SPLIT_TS].reset_index(drop=True)

    layer2_all = _load_feature_list(LAYER2_FEATURES_JSON)
    train_all, train_overlay = _overlay_exact(train_all_base, OVERLAY_SOURCES_2025, layer2_all, tag="train_2025")
    eval_df, eval_overlay = _overlay_exact(eval_base, OVERLAY_SOURCES_2026, layer2_all, tag="eval_2026")
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)

    primary_base = joblib.load(BASE_CLEAN_DIR / "primary_no_tp/parent.pkl")
    fallback_base = joblib.load(BASE_CLEAN_DIR / "fallback_v2_tp/parent.pkl")
    primary_base_cols = list(primary_base["feature_cols"])
    fallback_base_cols = list(fallback_base["feature_cols"])
    layer2_available = [c for c in layer2_all if c in train_all.columns and c in eval_df.columns]
    layer2_core_prefixes = ("ai_", "tide_", "chronos_", "regime3_", "m7_zigzag_", "dir3_")
    layer2_core = [c for c in layer2_available if c.startswith(layer2_core_prefixes)]
    layer2_primary_cols = list(dict.fromkeys([*primary_base_cols, *layer2_available]))
    layer2_fallback_cols = list(dict.fromkeys([*fallback_base_cols, *layer2_available]))
    core_primary_cols = list(dict.fromkeys([*primary_base_cols, *layer2_core]))
    core_fallback_cols = list(dict.fromkeys([*fallback_base_cols, *layer2_core]))

    for name, cols in {
        "primary_base": primary_base_cols,
        "fallback_base": fallback_base_cols,
        "primary_layer2": layer2_primary_cols,
        "fallback_layer2": layer2_fallback_cols,
        "primary_layer2_core": core_primary_cols,
        "fallback_layer2_core": core_fallback_cols,
    }.items():
        _assert_feature_cols(train_all, cols, name=f"{name}_train")
        _assert_feature_cols(eval_df, cols, name=f"{name}_eval")

    val_base_aligned = train_all_base[train_all_base["timestamp"].isin(set(val_df["timestamp"]))].reset_index(drop=True)
    if list(val_base_aligned["timestamp"]) != list(val_df["timestamp"]):
        raise RuntimeError("baseline validation timestamps do not match layer2 validation timestamps")
    baseline_val = _predict_combo(primary_base, fallback_base, val_base_aligned)
    baseline_oos = _predict_combo(primary_base, fallback_base, eval_base)

    primary_core, _, primary_core_summary = _load_or_train(
        train_all=train_all,
        eval_df=eval_df,
        feature_cols=core_primary_cols,
        seed=6060111,
        out_dir=OUT_DIR / "primary_no_tp_plus_layer2_core",
    )
    fallback_core, _, fallback_core_summary = _load_or_train(
        train_all=train_all,
        eval_df=eval_df,
        feature_cols=core_fallback_cols,
        seed=6060112,
        out_dir=OUT_DIR / "fallback_v2_tp_plus_layer2_core",
    )
    core_val = _predict_combo(primary_core, fallback_core, val_df)
    core_oos = _predict_combo(primary_core, fallback_core, eval_df)

    primary_layer2, _, primary_summary = _load_or_train(
        train_all=train_all,
        eval_df=eval_df,
        feature_cols=layer2_primary_cols,
        seed=6060101,
        out_dir=OUT_DIR / "primary_no_tp_plus_layer2",
    )
    fallback_layer2, _, fallback_summary = _load_or_train(
        train_all=train_all,
        eval_df=eval_df,
        feature_cols=layer2_fallback_cols,
        seed=6060102,
        out_dir=OUT_DIR / "fallback_v2_tp_plus_layer2",
    )
    layer2_val = _predict_combo(primary_layer2, fallback_layer2, val_df)
    layer2_oos = _predict_combo(primary_layer2, fallback_layer2, eval_df)

    rows = [
        _row("clean_parent_baseline_primary_no_tp_fallback_v2", baseline_val, baseline_oos, primary_base_cols, fallback_base_cols),
        _row("clean_parent_plus_omega1_layer2_core", core_val, core_oos, core_primary_cols, core_fallback_cols),
        _row("clean_parent_plus_omega1_layer2", layer2_val, layer2_oos, layer2_primary_cols, layer2_fallback_cols),
    ]
    ranking = pd.DataFrame(rows).sort_values(["val_cost3_pnl", "oos_cost3_pnl"], ascending=[False, False])
    ranking_path = OUT_DIR / "ranking.csv"
    ranking.to_csv(ranking_path, index=False)

    report = {
        "model_id": MODEL_ID,
        "scope": "Retrain clean Alpha7/01965 parent architecture with Omega1 Layer2 features added to primary_no_tp and fallback_v2_tp contracts.",
        "selection_policy": "Report baseline and layer2 candidate; compare validation first, 2026 OOS fixed after training.",
        "inputs": {
            "train_csv": str(TRAIN_CSV),
            "eval_csv": str(EVAL_CSV),
            "base_clean_dir": str(BASE_CLEAN_DIR),
            "layer2_features_json": str(LAYER2_FEATURES_JSON),
        },
        "overlay_reports": {
            "train": train_overlay,
            "eval": eval_overlay,
            "layer2_requested": len(layer2_all),
            "layer2_available": len(layer2_available),
            "layer2_core": len(layer2_core),
            "layer2_unavailable": [c for c in layer2_all if c not in layer2_available],
        },
        "feature_contracts": {
            "primary_base": _contract_report(primary_base_cols),
            "fallback_base": _contract_report(fallback_base_cols),
            "primary_layer2": _contract_report(layer2_primary_cols),
            "fallback_layer2": _contract_report(layer2_fallback_cols),
            "primary_layer2_core": _contract_report(core_primary_cols),
            "fallback_layer2_core": _contract_report(core_fallback_cols),
        },
        "baseline": {"validation": baseline_val, "oos": baseline_oos},
        "layer2_core": {
            "primary_summary": primary_core_summary,
            "fallback_summary": fallback_core_summary,
            "validation": core_val,
            "oos": core_oos,
        },
        "layer2": {
            "primary_summary": primary_summary,
            "fallback_summary": fallback_summary,
            "validation": layer2_val,
            "oos": layer2_oos,
        },
        "ranking": rows,
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking_csv": str(ranking_path),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "ranking": rows}, ensure_ascii=False, default=_json_default, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
