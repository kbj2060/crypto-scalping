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
    _combine_primary_fallback,
    _combo_metrics,
    _json_default,
    _predict_scaled,
    _read,
)
from scripts.retrain_alpha7_1_01965_tp_sl_decontam_20260528 import (  # noqa: E402
    _assert_feature_cols,
    _load_or_train,
)


MODEL_ID = "alpha7_regime3_expert_moe_20260601"
BASE_CLEAN_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_submodel_01965_cleanfunding_v1_20260529"
TRAIN_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2025_alpha6_current_tail111_exact.csv"
EVAL_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2026_alpha6_current_tail111_exact.csv"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_regime3_expert_moe_20260601"

ROUTERS = {
    "regime3_cmamba_h6_future": {
        "train": ROOT / "data/ensemble/supervised/regime3_cryptomamba_pred_h6_nocurrent_20260531/training_features_2025_regime3_cryptomamba_pred_h6_nocurrent_20260531.csv",
        "eval": ROOT / "data/ensemble/supervised/regime3_cryptomamba_pred_h6_nocurrent_20260531/training_features_2026_rebuilt_regime3_cryptomamba_pred_h6_nocurrent_20260531.csv",
        "cols": [
            "regime3_cmamba_h6_future_bull_prob",
            "regime3_cmamba_h6_future_bear_prob",
            "regime3_cmamba_h6_future_chop_prob",
        ],
        "extra_cols": [
            "regime3_cmamba_h6_confidence",
            "regime3_cmamba_h6_transition_prob",
            "regime3_cmamba_h6_stability_score",
        ],
    },
    "regime3_current_context": {
        "train": ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/training_features_2025_regime3_current_sensitive_hmm_wide24.csv",
        "eval": ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/training_features_2026_rebuilt_regime3_current_sensitive_hmm_wide24.csv",
        "cols": [
            "regime3_current_sensitive_wide24_bull_prob",
            "regime3_current_sensitive_wide24_bear_prob",
            "regime3_current_sensitive_wide24_chop_prob",
        ],
        "extra_cols": [
            "regime3_current_sensitive_wide24_confidence",
            "regime3_current_sensitive_wide24_entropy",
            "regime3_current_sensitive_wide24_margin",
        ],
    },
}
EXPERT_NAMES = ["bull", "bear", "chop"]


def _read_overlay(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    if "timestamp" not in df.columns:
        raise ValueError(f"{path} missing timestamp")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _edge_name(mask: pd.Series) -> str | None:
    idx = np.flatnonzero(mask.to_numpy())
    if len(idx) == 0:
        return None
    if np.array_equal(idx, np.arange(len(idx))):
        return "head"
    if np.array_equal(idx, np.arange(len(mask) - len(idx), len(mask))):
        return "tail"
    return None


def _overlay_router(base: pd.DataFrame, source: Path, cols: list[str], *, tag: str, prefix: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    src = _read_overlay(source)
    missing = [c for c in cols if c not in src.columns]
    if missing:
        raise RuntimeError(f"{tag}: missing required {prefix} columns: {missing}")
    out = base.copy()
    missing_ts = out.loc[~out["timestamp"].isin(set(src["timestamp"])), "timestamp"]
    dropped: list[dict[str, Any]] = []
    if len(missing_ts) > 0:
        head_ts = out["timestamp"].head(len(missing_ts)).reset_index(drop=True)
        tail_ts = out["timestamp"].tail(len(missing_ts)).reset_index(drop=True)
        missing_norm = missing_ts.reset_index(drop=True)
        if missing_norm.equals(head_ts):
            edge = "head"
        elif missing_norm.equals(tail_ts):
            edge = "tail"
        else:
            raise RuntimeError(f"{tag}: {source} missing non-edge timestamps: {missing_ts.head(20).tolist()}")
        dropped.append({"edge": edge, "rows": int(len(missing_ts)), "first": str(missing_ts.iloc[0]), "last": str(missing_ts.iloc[-1]), "path": str(source)})
        out = out.loc[out["timestamp"].isin(set(src["timestamp"]))].reset_index(drop=True)
    before = len(out)
    out = out.merge(src[["timestamp", *cols]], on="timestamp", how="left", validate="one_to_one")
    if len(out) != before:
        raise RuntimeError(f"{tag}: row count changed after {prefix} overlay")
    nan_mask = out[cols].isna().any(axis=1)
    edge = _edge_name(nan_mask)
    if edge is None and bool(nan_mask.any()):
        raise RuntimeError(f"{tag}: {prefix} non-edge NaN rows: {out.loc[nan_mask, 'timestamp'].head(20).tolist()}")
    if edge is not None:
        bad = out.loc[nan_mask, "timestamp"]
        dropped.append({"edge": edge, "rows": int(len(bad)), "first": str(bad.iloc[0]), "last": str(bad.iloc[-1]), "path": str(source), "reason": "router_edge_nan"})
        out = out.loc[~nan_mask].reset_index(drop=True)
    return out, {"path": str(source), "cols": cols, "dropped_edge_rows": dropped}


def _load_router_frames(router_name: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    spec = ROUTERS[router_name]
    cols = [*spec["cols"], *spec["extra_cols"]]
    train = _read(TRAIN_CSV)
    eval_df = _read(EVAL_CSV)
    train, train_report = _overlay_router(train, spec["train"], cols, tag=f"{router_name}_train", prefix=router_name)
    eval_df, eval_report = _overlay_router(eval_df, spec["eval"], cols, tag=f"{router_name}_eval", prefix=router_name)
    return train, eval_df, {"train": train_report, "eval": eval_report}


def _route_id(df: pd.DataFrame, router_name: str) -> np.ndarray:
    cols = ROUTERS[router_name]["cols"]
    values = df[cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{router_name}: non-finite router values")
    return np.argmax(values, axis=1).astype(np.int64)


def _route_conf(df: pd.DataFrame, router_name: str) -> np.ndarray:
    cols = ROUTERS[router_name]["cols"]
    values = df[cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    return np.max(values, axis=1).astype(np.float64)


def _active(dec: pd.DataFrame) -> np.ndarray:
    action = pd.to_numeric(dec["action"], errors="raise").to_numpy(dtype=np.int64)
    side = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
    return (action != 0) & (side != 0)


def _predict_combo(primary: dict[str, Any], fallback: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    return _combine_primary_fallback(_predict_scaled(primary, df, None), _predict_scaled(fallback, df, None)).reset_index(drop=True)


def _blend_by_route(expert_decisions: dict[str, pd.DataFrame], route: np.ndarray, *, min_conf: float, conf: np.ndarray, fallback_dec: pd.DataFrame) -> pd.DataFrame:
    out = fallback_dec.copy().reset_index(drop=True)
    decision_cols = list(out.columns)
    selected = route.copy()
    selected[conf < float(min_conf)] = 3
    for idx, name in enumerate(EXPERT_NAMES):
        mask = selected == idx
        out.loc[mask, decision_cols] = expert_decisions[name].loc[mask, decision_cols].to_numpy()
    out["router_expert"] = np.where(selected == 0, "bull", np.where(selected == 1, "bear", np.where(selected == 2, "chop", "fallback")))
    out["router_confidence"] = conf
    return out


def _side_constrained(dec: pd.DataFrame, *, expert: str) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    if expert == "bull":
        block = _active(out) & (pd.to_numeric(out["side"], errors="raise").to_numpy(dtype=np.int64) < 0)
    elif expert == "bear":
        block = _active(out) & (pd.to_numeric(out["side"], errors="raise").to_numpy(dtype=np.int64) > 0)
    else:
        block = np.zeros(len(out), dtype=bool)
    out.loc[block, ["action", "side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = 0
    out.loc[block, "leverage"] = 1.0
    return out


def _score(costs: dict[str, Any]) -> float:
    c3 = costs["cost3"]
    if int(c3["trades"]) < 20:
        return -1e9
    return float(c3["pnl"] / max(abs(float(c3["mdd"])), 1e-9))


def _flatten(prefix: str, costs: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ["cost1", "cost2", "cost3"]:
        c = costs[name]
        out[f"{prefix}_{name}_pnl"] = float(c["pnl"])
        out[f"{prefix}_{name}_mdd"] = float(c["mdd"])
        out[f"{prefix}_{name}_trades"] = int(c["trades"])
        out[f"{prefix}_{name}_wr"] = float(c["wr"])
    return out


def _train_experts(router_name: str, train_all: pd.DataFrame, eval_df: pd.DataFrame, primary_cols: list[str], fallback_cols: list[str]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    route = _route_id(train_all, router_name)
    out: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    for idx, expert in enumerate(EXPERT_NAMES):
        mask = route == idx
        expert_train = train_all.loc[mask].reset_index(drop=True)
        if int(mask.sum()) < 2000:
            raise RuntimeError(f"{router_name}/{expert}: too few rows for expert training: {int(mask.sum())}")
        primary, _, primary_summary = _load_or_train(
            train_all=expert_train,
            eval_df=eval_df,
            feature_cols=primary_cols,
            seed=6060200 + idx * 10,
            out_dir=OUT_DIR / router_name / expert / "primary_no_tp",
        )
        fallback, _, fallback_summary = _load_or_train(
            train_all=expert_train,
            eval_df=eval_df,
            feature_cols=fallback_cols,
            seed=6060201 + idx * 10,
            out_dir=OUT_DIR / router_name / expert / "fallback_v2_tp",
        )
        out[expert] = {"primary": primary, "fallback": fallback}
        summaries[expert] = {
            "rows": int(mask.sum()),
            "primary": primary_summary,
            "fallback": fallback_summary,
        }
    return out, summaries


def _evaluate_router(
    router_name: str,
    val_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    experts: dict[str, dict[str, Any]],
    baseline_val_dec: pd.DataFrame,
    baseline_oos_dec: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    val_route = _route_id(val_df, router_name)
    oos_route = _route_id(eval_df, router_name)
    val_conf = _route_conf(val_df, router_name)
    oos_conf = _route_conf(eval_df, router_name)
    val_expert_dec: dict[str, pd.DataFrame] = {}
    oos_expert_dec: dict[str, pd.DataFrame] = {}
    val_expert_dec_constrained: dict[str, pd.DataFrame] = {}
    oos_expert_dec_constrained: dict[str, pd.DataFrame] = {}
    for expert, models in experts.items():
        val_dec = _predict_combo(models["primary"], models["fallback"], val_df)
        oos_dec = _predict_combo(models["primary"], models["fallback"], eval_df)
        val_expert_dec[expert] = val_dec
        oos_expert_dec[expert] = oos_dec
        val_expert_dec_constrained[expert] = _side_constrained(val_dec, expert=expert)
        oos_expert_dec_constrained[expert] = _side_constrained(oos_dec, expert=expert)

    rows: list[dict[str, Any]] = []
    selected_payload: dict[str, Any] = {}
    for constrained, val_map, oos_map in [
        (False, val_expert_dec, oos_expert_dec),
        (True, val_expert_dec_constrained, oos_expert_dec_constrained),
    ]:
        for min_conf in [0.0, 0.40, 0.50, 0.60, 0.70, 0.80]:
            val_dec = _blend_by_route(val_map, val_route, min_conf=min_conf, conf=val_conf, fallback_dec=baseline_val_dec)
            oos_dec = _blend_by_route(oos_map, oos_route, min_conf=min_conf, conf=oos_conf, fallback_dec=baseline_oos_dec)
            val_costs = _combo_metrics(val_df, val_dec)
            oos_costs = _combo_metrics(eval_df, oos_dec)
            row = {
                "router": router_name,
                "constrained": bool(constrained),
                "min_conf": float(min_conf),
                "score": float(_score(val_costs)),
                "validation_policy_counts": {str(k): int(v) for k, v in val_dec["router_expert"].value_counts().to_dict().items()},
                "oos_policy_counts": {str(k): int(v) for k, v in oos_dec["router_expert"].value_counts().to_dict().items()},
                "validation": val_costs,
                "oos": oos_costs,
            }
            rows.append(row)
            key = f"{router_name}_constrained{int(constrained)}_conf{min_conf:.2f}"
            selected_payload[key] = {"validation_decisions": val_dec, "oos_decisions": oos_dec}
    rows.sort(key=lambda r: float(r["score"]), reverse=True)
    return rows, selected_payload


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    primary_base = joblib.load(BASE_CLEAN_DIR / "primary_no_tp/parent.pkl")
    fallback_base = joblib.load(BASE_CLEAN_DIR / "fallback_v2_tp/parent.pkl")
    primary_cols = list(primary_base["feature_cols"])
    fallback_cols = list(fallback_base["feature_cols"])

    all_reports: dict[str, Any] = {}
    rows_for_csv: list[dict[str, Any]] = []
    for router_name in ["regime3_cmamba_h6_future", "regime3_current_context"]:
        train_all, eval_df, overlay = _load_router_frames(router_name)
        for name, cols in {"primary": primary_cols, "fallback": fallback_cols}.items():
            _assert_feature_cols(train_all, cols, name=f"{router_name}_{name}_train")
            _assert_feature_cols(eval_df, cols, name=f"{router_name}_{name}_eval")
        train_df = train_all[train_all["timestamp"] < SPLIT_TS].reset_index(drop=True)
        val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
        baseline_val_dec = _predict_combo(primary_base, fallback_base, val_df)
        baseline_oos_dec = _predict_combo(primary_base, fallback_base, eval_df)
        baseline_val = _combo_metrics(val_df, baseline_val_dec)
        baseline_oos = _combo_metrics(eval_df, baseline_oos_dec)
        experts, expert_summaries = _train_experts(router_name, train_all, eval_df, primary_cols, fallback_cols)
        grid, payload = _evaluate_router(router_name, val_df, eval_df, experts, baseline_val_dec, baseline_oos_dec)
        baseline_row = {
            "router": router_name,
            "constrained": None,
            "min_conf": None,
            "score": float(_score(baseline_val)),
            "validation_policy_counts": {"baseline": int(_active(baseline_val_dec).sum()), "cash": int((~_active(baseline_val_dec)).sum())},
            "oos_policy_counts": {"baseline": int(_active(baseline_oos_dec).sum()), "cash": int((~_active(baseline_oos_dec)).sum())},
            "validation": baseline_val,
            "oos": baseline_oos,
        }
        full_grid = [baseline_row, *grid]
        full_grid.sort(key=lambda r: float(r["score"]), reverse=True)
        selected = full_grid[0]
        if selected is baseline_row:
            selected_val_dec = baseline_val_dec.copy()
            selected_oos_dec = baseline_oos_dec.copy()
            selected_val_dec["router_expert"] = np.where(_active(selected_val_dec), "baseline", "cash")
            selected_oos_dec["router_expert"] = np.where(_active(selected_oos_dec), "baseline", "cash")
        else:
            key = f"{router_name}_constrained{int(bool(selected['constrained']))}_conf{float(selected['min_conf']):.2f}"
            selected_val_dec = payload[key]["validation_decisions"]
            selected_oos_dec = payload[key]["oos_decisions"]
        router_dir = OUT_DIR / router_name
        router_dir.mkdir(parents=True, exist_ok=True)
        selected_val_dec.to_csv(router_dir / "validation_decisions.csv", index=False)
        selected_oos_dec.to_csv(router_dir / "oos_2026_decisions.csv", index=False)
        for row in full_grid:
            rows_for_csv.append({
                "router": row["router"],
                "constrained": row["constrained"],
                "min_conf": row["min_conf"],
                "score": row["score"],
                **_flatten("val", row["validation"]),
                **_flatten("oos", row["oos"]),
                "validation_policy_counts": json.dumps(row["validation_policy_counts"], ensure_ascii=False),
                "oos_policy_counts": json.dumps(row["oos_policy_counts"], ensure_ascii=False),
            })
        all_reports[router_name] = {
            "overlay": overlay,
            "expert_summaries": expert_summaries,
            "baseline": {"validation": baseline_val, "oos": baseline_oos},
            "selected": {
                "router": selected["router"],
                "constrained": selected["constrained"],
                "min_conf": selected["min_conf"],
                "validation": selected["validation"],
                "oos": selected["oos"],
                "validation_policy_counts": {str(k): int(v) for k, v in selected_val_dec["router_expert"].value_counts().to_dict().items()},
                "oos_policy_counts": {str(k): int(v) for k, v in selected_oos_dec["router_expert"].value_counts().to_dict().items()},
            },
            "top_grid": full_grid[:8],
            "artifacts": {
                "validation_decisions": str(router_dir / "validation_decisions.csv"),
                "oos_decisions": str(router_dir / "oos_2026_decisions.csv"),
            },
        }
    pd.DataFrame(rows_for_csv).sort_values(["router", "score"], ascending=[True, False]).to_csv(OUT_DIR / "ranking.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "design": "Train separate bull/bear/chop Alpha7-style expert pairs. Route with Regime3 CryptoMamba h6 future-context sidecar and, separately, Regime3 current sensitive wide24 context.",
        "expert_feature_contract": {
            "primary_feature_count": len(primary_cols),
            "fallback_feature_count": len(fallback_cols),
            "primary_feature_cols": primary_cols,
            "fallback_feature_cols": fallback_cols,
        },
        "routers": all_reports,
        "artifacts": {"report": str(OUT_DIR / "report.json"), "ranking": str(OUT_DIR / "ranking.csv")},
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "ranking": str(OUT_DIR / "ranking.csv"), "selected": {k: v["selected"] for k, v in all_reports.items()}}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
