#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import precision_retest_01965_alpha7_combo_20260527 as precision  # noqa: E402
from scripts import runtime_retest_alpha7_1_01965_decontam_20260528 as decontam  # noqa: E402
from scripts import sweep_decontam_deep_alpha_controls_20260528 as sweep  # noqa: E402
from scripts import train_eval_hf_v13_deep_alpha_candidate_expansion_v27 as v27  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


MODEL_ID = "deep_side_specialist_meta_veto_20260528"
OUT_DIR = ROOT / f"tmp/causal_regen_20260516/{MODEL_ID}"
SUMMARY_OUT = OUT_DIR / "summary.json"
GRID_OUT = OUT_DIR / "grid.csv"
MODEL_OUT = OUT_DIR / "side_specialist_meta_veto.joblib"
VAL_TRAIN_OUT = OUT_DIR / "train_deep_entries.csv"
OOS_LEDGER_OUT = OUT_DIR / "best_oos_cost3_ledger.csv"

Q_FEATURES = ["q_side", "q_opp", "q_edge", "q_margin", "q_side_minus_opp", "q_side_share"]
BASE_FEATURES = [
    "tp_sl_action_score",
    "net_taker_ratio",
    "taker_acceleration",
    "ofi_acceleration",
    "ai_flow_pressure",
    "atr_14",
    "atr",
    "volume",
    "quote_volume",
    "return_1",
    "return_3",
    "return_6",
    "volatility_20",
    "cvp_regime",
    "regime_trending",
]
PREFIXES = (
    "clean_regime4_state24_sticky090_v2_",
    "regime4_pred_",
)


def _feature_cols(df: pd.DataFrame) -> list[str]:
    cols = list(Q_FEATURES)
    cols.extend([c for c in BASE_FEATURES if c in df.columns])
    cols.extend([c for c in df.columns if str(c).startswith(PREFIXES)])
    seen: set[str] = set()
    out: list[str] = []
    for col in cols:
        if col not in seen:
            out.append(col)
            seen.add(col)
    return out


def _row_features(row: pd.Series, *, side: int, ql: float, qs: float, feature_cols: list[str]) -> dict[str, float]:
    q_side = float(ql if side > 0 else qs)
    q_opp = float(qs if side > 0 else ql)
    base = {
        "q_side": q_side,
        "q_opp": q_opp,
        "q_edge": float(max(ql, qs)),
        "q_margin": float(abs(ql - qs)),
        "q_side_minus_opp": float(q_side - q_opp),
        "q_side_share": float(q_side / max(abs(ql) + abs(qs), 1e-12)),
    }
    for col in feature_cols:
        if col in base:
            continue
        base[col] = float(pd.to_numeric(pd.Series([row.get(col, 0.0)]), errors="coerce").fillna(0.0).iloc[0])
    return {col: float(base.get(col, 0.0)) for col in feature_cols}


def _training_frame(records: list[dict[str, Any]], df: pd.DataFrame, q: np.ndarray, feature_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in records:
        if str(rec.get("owner", "")) != "deep_alpha":
            continue
        i = int(rec["entry_signal_idx"])
        if i < 0 or i >= len(df):
            continue
        side = 1 if str(rec.get("side", "")).upper() == "LONG" else -1
        ql, qs = float(q[i, 0]), float(q[i, 1])
        feat = _row_features(df.iloc[i], side=side, ql=ql, qs=qs, feature_cols=feature_cols)
        ret = float(rec.get("trade_return", 0.0) or 0.0)
        feat.update(
            {
                "side": "LONG" if side > 0 else "SHORT",
                "label": int(ret > 0.0),
                "trade_return": float(ret),
                "exit_reason": str(rec.get("exit_reason", "")),
                "entry_time": str(rec.get("entry_time", "")),
            }
        )
        rows.append(feat)
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("no deep_alpha training records were generated")
    return out


def _fit_models(train: pd.DataFrame, feature_cols: list[str]) -> dict[str, dict[str, Any]]:
    specs = {
        "logreg_balanced": lambda: Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced")),
            ]
        ),
    }
    fitted: dict[str, dict[str, Any]] = {}
    for model_name, factory in specs.items():
        fitted[model_name] = {}
        for side_name in ("LONG", "SHORT"):
            sub = train[train["side"].eq(side_name)].reset_index(drop=True)
            y = sub["label"].astype(int).to_numpy()
            if len(sub) < 20 or len(set(y.tolist())) < 2:
                raise RuntimeError(f"insufficient {side_name} training labels for {model_name}: rows={len(sub)} classes={sorted(set(y.tolist()))}")
            x = sub[feature_cols]
            w = 1.0 + np.minimum(np.abs(sub["trade_return"].to_numpy(dtype=float)) * 20.0, 4.0)
            model = factory()
            if model_name.startswith("hgb"):
                model.fit(x, y, model__sample_weight=w)
            else:
                model.fit(x, y, model__sample_weight=w)
            fitted[model_name][side_name] = model
    return fitted


def _side_prob_arrays(models: dict[str, Any], feature_cols: list[str], *, model_name: str, df: pd.DataFrame, q: np.ndarray) -> dict[str, np.ndarray]:
    idx = range(len(df))
    long_x = pd.DataFrame(
        [_row_features(df.iloc[i], side=1, ql=float(q[i, 0]), qs=float(q[i, 1]), feature_cols=feature_cols) for i in idx]
    )
    short_x = pd.DataFrame(
        [_row_features(df.iloc[i], side=-1, ql=float(q[i, 0]), qs=float(q[i, 1]), feature_cols=feature_cols) for i in idx]
    )
    return {
        "LONG": np.asarray(models[model_name]["LONG"].predict_proba(long_x)[:, 1], dtype=float),
        "SHORT": np.asarray(models[model_name]["SHORT"].predict_proba(short_x)[:, 1], dtype=float),
    }


def _make_array_gate(probs: dict[str, np.ndarray], *, model_name: str, long_thr: float, short_thr: float):
    def gate(i: int, side: int, ql: float, qs: float, row: pd.Series) -> tuple[bool, str]:
        side_name = "LONG" if side > 0 else "SHORT"
        prob = float(probs[side_name][int(i)])
        thr = float(long_thr if side > 0 else short_thr)
        if prob < thr:
            return False, f"{model_name.lower()}_{side_name.lower()}_veto"
        return True, ""

    return gate


def _select_side_thresholds(models: dict[str, Any], train: pd.DataFrame, feature_cols: list[str], *, model_name: str) -> dict[str, float]:
    selected: dict[str, float] = {}
    for side_name in ("LONG", "SHORT"):
        sub = train[train["side"].eq(side_name)].reset_index(drop=True)
        prob = np.asarray(models[model_name][side_name].predict_proba(sub[feature_cols])[:, 1], dtype=float)
        ret = sub["trade_return"].to_numpy(dtype=float)
        candidates = np.unique(np.quantile(prob, [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]))
        best_thr = float(candidates[0])
        best_score = -1e18
        for thr in candidates:
            keep = prob >= float(thr)
            keep_rate = float(keep.mean())
            if keep_rate < 0.45:
                continue
            kept_ret = float(ret[keep].sum())
            blocked_loss = float((-ret[~keep][ret[~keep] < 0.0]).sum())
            missed_win = float((ret[~keep][ret[~keep] > 0.0]).sum())
            score = kept_ret + 0.35 * blocked_loss - 0.65 * missed_win
            if score > best_score:
                best_score = score
                best_thr = float(thr)
        selected[side_name] = float(best_thr)
    return selected


def _row(name: str, res: dict[str, Any], *, model_name: str, long_thr: float, short_thr: float, split: str) -> dict[str, Any]:
    return {
        "name": name,
        "model_name": model_name,
        "split": split,
        "long_thr": float(long_thr),
        "short_thr": float(short_thr),
        "pnl": float(res["pnl"]),
        "mdd": float(res["mdd"]),
        "wr": float(res["wr"]),
        "trades": int(res["trades"]),
        "deep_entries": int(res.get("deep_entries", 0)),
        "long_entries": int(res.get("long_entries", 0)),
        "short_entries": int(res.get("short_entries", 0)),
        "sl_ratio": float(sweep._sl_ratio(res)),
        "score": float(sweep._score(res)),
        "actions": json.dumps(res.get("actions", {}), ensure_ascii=False, sort_keys=True),
        "exits": json.dumps(res.get("exits", {}), ensure_ascii=False, sort_keys=True),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    decontam._assert_clean_frame(decontam.TRAIN_CSV, name="train")
    decontam._assert_clean_frame(decontam.EVAL_CSV, name="eval")
    decontam._assert_clean_parent(decontam.CANDIDATE_DIR / "primary_parent.pkl", name="primary")
    decontam._assert_clean_parent(decontam.CANDIDATE_DIR / "fallback_alpha43_no_legacy_parent.pkl", name="fallback")
    decontam._patch_runtime_sources()

    cfg = precision._cfg_from_results()
    stack = precision._load_stack()
    val_df, eval_df = precision._load_frames()
    sources = precision._decision_sources(val_df, eval_df, stack["parent"])
    val_q = v27._predict_all(stack["deep_model"], val_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    eval_q = v27._predict_all(stack["deep_model"], eval_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    val_dec = sources[str(cfg["source"])][0]
    eval_dec = sources[str(cfg["source"])][1]

    base_variant = sweep.Variant("deep_stop_cd18", deep_stop_cooldown_extra=18)
    val_base = sweep._backtest_variant(df=val_df, q=val_q, dec=val_dec, stack=stack, cfg=cfg, variant=base_variant, cost_mult=3, record=True)
    feature_cols = _feature_cols(val_df)
    train = _training_frame(list(val_base.get("trade_records", [])), val_df, val_q, feature_cols)
    train.to_csv(VAL_TRAIN_OUT, index=False)
    models = _fit_models(train, feature_cols)

    rows: list[dict[str, Any]] = []
    baseline_oos = sweep._backtest_variant(df=eval_df, q=eval_q, dec=eval_dec, stack=stack, cfg=cfg, variant=base_variant, cost_mult=3, record=True)
    rows.append(_row("deep_stop_cd18", baseline_oos, model_name="none", long_thr=0.0, short_thr=0.0, split="oos"))

    model_name = "logreg_balanced"
    thresholds = _select_side_thresholds(models, train, feature_cols, model_name=model_name)
    long_thr = float(thresholds["LONG"])
    short_thr = float(thresholds["SHORT"])
    variant = sweep.Variant(
        f"deep_stop_cd18_meta_{model_name}_train_selected",
        deep_stop_cooldown_extra=18,
    )
    val_probs = _side_prob_arrays(models, feature_cols, model_name=model_name, df=val_df, q=val_q)
    oos_probs = _side_prob_arrays(models, feature_cols, model_name=model_name, df=eval_df, q=eval_q)
    val_res = sweep._backtest_variant(
        df=val_df,
        q=val_q,
        dec=val_dec,
        stack=stack,
        cfg=cfg,
        variant=variant,
        cost_mult=3,
        record=False,
        deep_gate=_make_array_gate(val_probs, model_name=model_name, long_thr=long_thr, short_thr=short_thr),
    )
    oos_res = sweep._backtest_variant(
        df=eval_df,
        q=eval_q,
        dec=eval_dec,
        stack=stack,
        cfg=cfg,
        variant=variant,
        cost_mult=3,
        record=True,
        deep_gate=_make_array_gate(oos_probs, model_name=model_name, long_thr=long_thr, short_thr=short_thr),
    )
    rows.append(_row(variant.name, val_res, model_name=model_name, long_thr=long_thr, short_thr=short_thr, split="val"))
    rows.append(_row(variant.name, oos_res, model_name=model_name, long_thr=long_thr, short_thr=short_thr, split="oos"))
    best: tuple[float, str, dict[str, Any], list[dict[str, Any]]] = (
        sweep._score(oos_res),
        variant.name,
        _row(variant.name, oos_res, model_name=model_name, long_thr=long_thr, short_thr=short_thr, split="oos"),
        list(oos_res.get("trade_records", [])),
    )

    grid = pd.DataFrame(rows)
    grid.to_csv(GRID_OUT, index=False)
    pd.DataFrame(best[3]).to_csv(OOS_LEDGER_OUT, index=False)
    payload = {
        "model_id": MODEL_ID,
        "feature_cols": feature_cols,
        "models": models,
        "train_rows": int(len(train)),
        "train_by_side": train.groupby("side")["label"].agg(["count", "mean"]).reset_index().to_dict(orient="records"),
        "best_name": best[1],
        "best_oos": best[2],
        "selected_thresholds": thresholds,
    }
    joblib.dump(payload, MODEL_OUT)
    summary = {
        "model_id": MODEL_ID,
        "train_rows": int(len(train)),
        "train_by_side": payload["train_by_side"],
        "feature_cols": feature_cols,
        "model_artifact": str(MODEL_OUT),
        "train_deep_entries": str(VAL_TRAIN_OUT),
        "grid": str(GRID_OUT),
        "best_name": best[1],
        "best_oos": best[2],
        "selected_thresholds": thresholds,
        "best_oos_ledger": str(OOS_LEDGER_OUT),
        "baseline_oos": _row("deep_stop_cd18", baseline_oos, model_name="none", long_thr=0.0, short_thr=0.0, split="oos"),
    }
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(SUMMARY_OUT), "grid": str(GRID_OUT), "best": best[1]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
