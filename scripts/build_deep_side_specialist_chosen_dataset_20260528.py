#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import precision_retest_01965_alpha7_combo_20260527 as precision  # noqa: E402
from scripts import runtime_retest_alpha7_1_01965_decontam_20260528 as decontam  # noqa: E402
from scripts import train_eval_deep_side_specialist_nn_veto_20260528 as nnv  # noqa: E402
from scripts import train_eval_hf_v13_deep_alpha_candidate_expansion_v27 as v27  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _json_default  # noqa: E402


MODEL_ID = "deep_side_specialist_chosen_dataset_20260528"
OUT_DIR = ROOT / f"tmp/causal_regen_20260516/{MODEL_ID}"
AUDIT_OUT = OUT_DIR / "audit.json"

POS_STRICT = 0.008
NEG_STRICT = -0.004
POS_SOFT = 0.004
NEG_SOFT = -0.002


SIDE_ALIGNED_BASE = [
    "net_taker_ratio",
    "taker_acceleration",
    "ofi_acceleration",
    "ai_flow_pressure",
    "clean_regime4_state24_sticky090_v2_directional_bias",
    "clean_regime4_state24_sticky090_v2_trend_bias",
    "regime4_pred_directional_bias",
]


def _safe_num(row: pd.Series, col: str) -> float:
    try:
        val = float(row.get(col, 0.0))
    except Exception:
        return 0.0
    return val if np.isfinite(val) else 0.0


def _side_aligned_features(row: pd.Series, side: int) -> dict[str, float]:
    sign = 1.0 if int(side) > 0 else -1.0
    bull = _safe_num(row, "clean_regime4_state24_sticky090_v2_bull_prob")
    bear = _safe_num(row, "clean_regime4_state24_sticky090_v2_bear_prob")
    pred_bull = _safe_num(row, "regime4_pred_bull_prob")
    pred_bear = _safe_num(row, "regime4_pred_bear_prob")
    out = {
        "side_sign": sign,
        "side_state24_trend_alignment": (bull - bear) * sign,
        "side_pred_trend_alignment": (pred_bull - pred_bear) * sign,
        "side_state24_risk_off": _safe_num(row, "clean_regime4_state24_sticky090_v2_risk_off_prob"),
        "side_state24_instability": _safe_num(row, "clean_regime4_state24_sticky090_v2_instability_prob"),
        "side_pred_instability": _safe_num(row, "regime4_pred_instability_prob"),
        "side_pred_whipsaw": _safe_num(row, "regime4_pred_whipsaw_prob"),
    }
    for col in SIDE_ALIGNED_BASE:
        out[f"side_{col}"] = _safe_num(row, col) * sign
    return out


def _selected_side_dataset(
    df: pd.DataFrame,
    q: np.ndarray,
    *,
    feature_cols: list[str],
    edge_th: float,
    margin_th: float,
    hold: int,
    fee: float,
    slip: float,
    split: str,
) -> pd.DataFrame:
    x = nnv._feature_frame(df, q, feature_cols)
    close = _close(df)
    rows: list[dict[str, Any]] = []
    for i in range(60, len(df) - max(hold, 2) - 1):
        ql = float(q[i, 0])
        qs = float(q[i, 1])
        edge = float(max(ql, qs))
        margin = float(abs(ql - qs))
        if edge < edge_th or margin < margin_th:
            continue
        side = 1 if ql > qs else -1
        q_side = ql if side > 0 else qs
        q_opp = qs if side > 0 else ql
        denom = max(abs(ql) + abs(qs), 1e-12)
        ret = nnv._path_return(close, i, side, hold, fee, slip)
        row = df.iloc[i]
        rec = {col: float(x.iloc[i][col]) for col in feature_cols}
        rec.update(_side_aligned_features(row, side))
        rec.update(
            {
                "split": split,
                "idx": int(i),
                "timestamp": str(row.get("timestamp", "")),
                "side": "LONG" if side > 0 else "SHORT",
                "side_int": int(side),
                "q_side": float(q_side),
                "q_opp": float(q_opp),
                "q_side_advantage": float(q_side - q_opp),
                "q_side_share": float(q_side / denom),
                "path_return": float(ret),
                "label_binary": int(ret > 0.0),
                "label_soft": int(ret >= POS_SOFT) if ret >= POS_SOFT or ret <= NEG_SOFT else -1,
                "label_strict": int(ret >= POS_STRICT) if ret >= POS_STRICT or ret <= NEG_STRICT else -1,
                "sample_weight": float(1.0 + min(abs(ret) * 35.0, 5.0)),
            }
        )
        rows.append(rec)
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(f"{split} selected-side dataset is empty")
    return out


def _summary(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {
        "rows": int(len(df)),
        "side_counts": df["side"].value_counts().sort_index().to_dict(),
        "path_return": df.groupby("side")["path_return"].describe(percentiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]).to_dict(),
    }
    for label in ("label_binary", "label_soft", "label_strict"):
        sub = df[df[label].ge(0)]
        out[label] = {
            "rows": int(len(sub)),
            "keep_rate": float(len(sub) / max(len(df), 1)),
            "by_side": sub.groupby("side")[label].agg(["count", "mean"]).reset_index().to_dict(orient="records") if len(sub) else [],
        }
    return out


def _write_split(df: pd.DataFrame, prefix: str) -> dict[str, str]:
    paths: dict[str, str] = {}
    all_path = OUT_DIR / f"{prefix}_chosen_all.csv"
    strict_path = OUT_DIR / f"{prefix}_chosen_strict.csv"
    soft_path = OUT_DIR / f"{prefix}_chosen_soft.csv"
    df.to_csv(all_path, index=False)
    df[df["label_strict"].ge(0)].to_csv(strict_path, index=False)
    df[df["label_soft"].ge(0)].to_csv(soft_path, index=False)
    paths["all"] = str(all_path)
    paths["strict"] = str(strict_path)
    paths["soft"] = str(soft_path)
    for side in ("LONG", "SHORT"):
        side_strict = OUT_DIR / f"{prefix}_{side.lower()}_strict.csv"
        df[df["side"].eq(side) & df["label_strict"].ge(0)].to_csv(side_strict, index=False)
        paths[f"{side.lower()}_strict"] = str(side_strict)
    return paths


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    decontam._assert_clean_frame(decontam.TRAIN_CSV, name="train")
    decontam._assert_clean_frame(decontam.EVAL_CSV, name="eval")
    decontam._assert_clean_parent(decontam.CANDIDATE_DIR / "primary_parent.pkl", name="primary")
    decontam._assert_clean_parent(decontam.CANDIDATE_DIR / "fallback_alpha43_no_legacy_parent.pkl", name="fallback")
    decontam._patch_runtime_sources()

    cfg = precision._cfg_from_results()
    stack = precision._load_stack()
    train_df, eval_df = precision._load_frames()
    overlay = precision._overlay(stack["overlay"], cfg)
    feature_cols = nnv._feature_cols(train_df)
    train_q = v27._predict_all(stack["deep_model"], train_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    eval_q = v27._predict_all(stack["deep_model"], eval_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    kwargs = {
        "feature_cols": feature_cols,
        "edge_th": float(overlay.edge_th),
        "margin_th": float(overlay.margin_th),
        "hold": int(overlay.base_hold),
        "fee": float(stack["fee"]) * 3.0,
        "slip": float(stack["slip"]) * 3.0,
    }
    train = _selected_side_dataset(train_df, train_q, split="train_2025", **kwargs)
    eval_ = _selected_side_dataset(eval_df, eval_q, split="eval_2026", **kwargs)
    paths = {
        "train": _write_split(train, "train_2025"),
        "eval": _write_split(eval_, "eval_2026"),
    }
    audit = {
        "model_id": MODEL_ID,
        "purpose": "V31 deep_alpha chosen-side LONG/SHORT specialist datasets. No counterfactual side duplication.",
        "positive_strict_threshold": POS_STRICT,
        "negative_strict_threshold": NEG_STRICT,
        "positive_soft_threshold": POS_SOFT,
        "negative_soft_threshold": NEG_SOFT,
        "feature_cols": feature_cols,
        "side_aligned_cols": list(_side_aligned_features(train_df.iloc[0], 1).keys()),
        "paths": paths,
        "train": _summary(train),
        "eval": _summary(eval_),
    }
    AUDIT_OUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"audit": str(AUDIT_OUT), "paths": paths}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
