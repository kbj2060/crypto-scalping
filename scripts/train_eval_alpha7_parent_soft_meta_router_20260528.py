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

from scripts import precision_retest_alpha7_parent_soft_regime_veto_20260528 as parent_soft  # noqa: E402
from scripts import runtime_retest_alpha7_1_01965_decontam_20260528 as decontam  # noqa: E402
from scripts import sweep_decontam_deep_alpha_controls_20260528 as sweep  # noqa: E402
from scripts import train_eval_hf_v13_deep_alpha_candidate_expansion_v27 as v27  # noqa: E402
from scripts import walk_forward_alpha7_parent_soft_regime_veto_20260528 as walk  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


MODEL_ID = "alpha7_parent_soft_meta_router_20260528"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
SUMMARY_OUT = OUT_DIR / "summary.json"
GRID_OUT = OUT_DIR / "grid.csv"
DAILY_OUT = OUT_DIR / "daily_training_frame.csv"
PRED_OUT = OUT_DIR / "router_predictions.csv"

ROUTER_TRAIN_END = pd.Timestamp("2025-09-30 23:59:59")
ROUTER_VAL_START = pd.Timestamp("2025-10-01 00:00:00")
THRESHOLDS = (0.35, 0.45, 0.50, 0.55, 0.65)


def _to_day(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["timestamp"], errors="coerce").dt.to_period("D").astype(str)


def _score_row(variant: str, split: str, period: str, res: dict[str, Any]) -> dict[str, Any]:
    return parent_soft._row(variant, split, period, res)


def _eval(
    *,
    variant_name: str,
    split: str,
    period: str,
    df: pd.DataFrame,
    q: np.ndarray,
    dec: pd.DataFrame,
    stack: dict[str, Any],
    cfg: dict[str, Any],
    variant: sweep.Variant,
) -> dict[str, Any]:
    res = sweep._backtest_variant(
        df=df.reset_index(drop=True),
        q=q,
        dec=dec.reset_index(drop=True),
        stack=stack,
        cfg=cfg,
        variant=variant,
        cost_mult=3,
        record=False,
        deep_gate=None,
    )
    return _score_row(variant_name, split, period, res)


def _q_action_frame(q_day: np.ndarray) -> np.ndarray:
    arr = np.asarray(q_day)
    if arr.ndim == 3:
        arr = arr.mean(axis=1)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr.astype(np.float64, copy=False)


def _raw_day_features(df_day: pd.DataFrame, q_day: np.ndarray, dec_day: pd.DataFrame, soft_dec_day: pd.DataFrame) -> dict[str, float]:
    close = pd.to_numeric(df_day["close"], errors="coerce").ffill().bfill()
    ret = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    high = pd.to_numeric(df_day.get("high", close), errors="coerce").fillna(close)
    low = pd.to_numeric(df_day.get("low", close), errors="coerce").fillna(close)
    volume = pd.to_numeric(df_day.get("volume", 0.0), errors="coerce").fillna(0.0)

    action = pd.to_numeric(dec_day["action"], errors="coerce").fillna(0).astype(int)
    side = pd.to_numeric(dec_day["side"], errors="coerce").fillna(0).astype(int)
    soft_action = pd.to_numeric(soft_dec_day["action"], errors="coerce").fillna(0).astype(int)
    active = (action != 0) & (side != 0)
    blocked = active & (soft_action == 0)
    conf = pd.to_numeric(dec_day.get("confidence", 0.0), errors="coerce").fillna(0.0)
    quality = pd.to_numeric(dec_day.get("quality_score", 0.0), errors="coerce").fillna(0.0)
    notional = pd.to_numeric(dec_day.get("notional_exposure", 0.0), errors="coerce").fillna(0.0)

    counter = pd.Series(
        [parent_soft._counter_regime_prob(row, int(s)) for (_, row), s in zip(df_day.iterrows(), side)],
        index=df_day.index,
    )
    q2 = _q_action_frame(q_day)
    if q2.shape[1] >= 2:
        q_sorted = np.sort(q2, axis=1)
        q_margin = q_sorted[:, -1] - q_sorted[:, -2]
        q_best = q_sorted[:, -1]
    else:
        q_margin = np.zeros(len(q2), dtype=np.float64)
        q_best = q2[:, 0] if len(q2) else np.zeros(0, dtype=np.float64)

    out: dict[str, float] = {
        "bars": float(len(df_day)),
        "ret_sum": float(ret.sum()),
        "ret_vol": float(ret.std(ddof=0)),
        "close_ret": float(close.iloc[-1] / max(float(close.iloc[0]), 1e-12) - 1.0) if len(close) else 0.0,
        "hl_range_mean": float(((high - low) / close.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0).mean()),
        "volume_z": float((volume.iloc[-1] - volume.mean()) / (volume.std(ddof=0) + 1e-12)) if len(volume) else 0.0,
        "active_rows": float(active.sum()),
        "long_active_rows": float((active & (side > 0)).sum()),
        "short_active_rows": float((active & (side < 0)).sum()),
        "active_long_ratio": float((active & (side > 0)).sum() / max(active.sum(), 1)),
        "blocked_rows": float(blocked.sum()),
        "blocked_active_ratio": float(blocked.sum() / max(active.sum(), 1)),
        "blocked_bear_long_rows": float((blocked & (side > 0)).sum()),
        "blocked_bull_short_rows": float((blocked & (side < 0)).sum()),
        "confidence_mean": float(conf[active].mean()) if bool(active.any()) else 0.0,
        "confidence_low_ratio": float((conf[active] < parent_soft.PARENT_MIN_MODEL_CONF).mean()) if bool(active.any()) else 0.0,
        "quality_mean": float(quality[active].mean()) if bool(active.any()) else 0.0,
        "quality_low_ratio": float((quality[active] < parent_soft.PARENT_MIN_QUALITY).mean()) if bool(active.any()) else 0.0,
        "notional_mean": float(notional[active].mean()) if bool(active.any()) else 0.0,
        "counter_prob_mean": float(counter[active].mean()) if bool(active.any()) else 0.0,
        "counter_prob_max": float(counter[active].max()) if bool(active.any()) else 0.0,
        "deep_q_best_mean": float(np.nanmean(q_best)) if q_best.size else 0.0,
        "deep_q_margin_mean": float(np.nanmean(q_margin)) if q_margin.size else 0.0,
    }
    regime_cols = [
        "clean_regime4_state24_sticky090_v2_bull_prob",
        "clean_regime4_state24_sticky090_v2_bear_prob",
        "clean_regime4_state24_sticky090_v2_chop_prob",
        "clean_regime4_state24_sticky090_v2_whipsaw_prob",
        "clean_regime4_state24_sticky090_v2_trend_prob",
        "clean_regime4_state24_sticky090_v2_risk_off_prob",
        "clean_regime4_state24_sticky090_v2_transition_risk",
        "clean_regime4_state24_sticky090_v2_confidence",
        "clean_regime4_state24_sticky090_v2_entropy",
    ]
    for col in regime_cols:
        if col in df_day.columns:
            out[col] = float(pd.to_numeric(df_day[col], errors="coerce").fillna(0.0).mean())
    return out


def _daily_frame(
    *,
    split: str,
    df: pd.DataFrame,
    q: np.ndarray,
    dec: pd.DataFrame,
    soft_dec: pd.DataFrame,
    stack: dict[str, Any],
    cfg: dict[str, Any],
    variant: sweep.Variant,
) -> pd.DataFrame:
    days = _to_day(df)
    rows: list[dict[str, Any]] = []
    for day in sorted(days.dropna().unique()):
        mask = days.eq(day).to_numpy(dtype=bool)
        if int(mask.sum()) < 200:
            continue
        base = _eval(
            variant_name="baseline",
            split=split,
            period=day,
            df=df.loc[mask],
            q=q[mask],
            dec=dec.loc[mask],
            stack=stack,
            cfg=cfg,
            variant=variant,
        )
        soft = _eval(
            variant_name="parent_soft",
            split=split,
            period=day,
            df=df.loc[mask],
            q=q[mask],
            dec=soft_dec.loc[mask],
            stack=stack,
            cfg=cfg,
            variant=variant,
        )
        feat = _raw_day_features(df.loc[mask], q[mask], dec.loc[mask], soft_dec.loc[mask])
        rows.append(
            {
                "split": split,
                "day": day,
                "day_ts": pd.Timestamp(day),
                "baseline_pnl": base["pnl"],
                "baseline_mdd": base["mdd"],
                "baseline_wr": base["wr"],
                "baseline_trades": base["trades"],
                "baseline_score": base["score"],
                "parent_soft_pnl": soft["pnl"],
                "parent_soft_mdd": soft["mdd"],
                "parent_soft_wr": soft["wr"],
                "parent_soft_trades": soft["trades"],
                "parent_soft_score": soft["score"],
                "score_delta": float(soft["score"] - base["score"]),
                "pnl_delta": float(soft["pnl"] - base["pnl"]),
                "use_soft_label": int(float(soft["score"]) > float(base["score"])),
                **feat,
            }
        )
    return pd.DataFrame(rows)


def _attach_previous_day_features(daily: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    raw_cols = [
        c
        for c in daily.columns
        if c
        not in {
            "split",
            "day",
            "day_ts",
            "baseline_pnl",
            "baseline_mdd",
            "baseline_wr",
            "baseline_trades",
            "baseline_score",
            "parent_soft_pnl",
            "parent_soft_mdd",
            "parent_soft_wr",
            "parent_soft_trades",
            "parent_soft_score",
            "score_delta",
            "pnl_delta",
            "use_soft_label",
        }
    ]
    out = daily.sort_values("day_ts").reset_index(drop=True).copy()
    for col in raw_cols:
        out[f"prev_{col}"] = pd.to_numeric(out[col], errors="coerce").shift(1)
    out["has_prev_day"] = out["prev_bars"].notna().astype(float)
    feature_cols = [f"prev_{c}" for c in raw_cols] + ["has_prev_day"]
    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out, feature_cols


def _fit_router_models(train: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    x = train[feature_cols].to_numpy(dtype=np.float64)
    y = train["use_soft_label"].to_numpy(dtype=np.int64)
    if len(np.unique(y)) < 2:
        return {}
    weight = np.clip(np.abs(train["score_delta"].to_numpy(dtype=np.float64)), 0.25, 25.0)
    models: dict[str, Any] = {
        "logistic": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
            ]
        ),
        "hgb": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=160,
                        learning_rate=0.035,
                        max_leaf_nodes=7,
                        min_samples_leaf=12,
                        l2_regularization=2.0,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }
    fitted: dict[str, Any] = {}
    for name, model in models.items():
        model.fit(x, y, model__sample_weight=weight)
        fitted[name] = model
    return fitted


def _predict_router(model: Any, frame: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    x = frame[feature_cols].to_numpy(dtype=np.float64)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    raise RuntimeError("router model does not expose predict_proba")


def _combine_decisions(df: pd.DataFrame, base_dec: pd.DataFrame, soft_dec: pd.DataFrame, route_by_day: dict[str, bool]) -> pd.DataFrame:
    out = base_dec.copy().reset_index(drop=True)
    soft = soft_dec.reset_index(drop=True)
    days = _to_day(df.reset_index(drop=True))
    for day, use_soft in route_by_day.items():
        if not use_soft:
            continue
        mask = days.eq(str(day)).to_numpy(dtype=bool)
        if int(mask.sum()) == 0:
            continue
        out.loc[mask, out.columns] = soft.loc[mask, out.columns].to_numpy()
    return out


def _route_map(frame: pd.DataFrame, use_soft: np.ndarray) -> dict[str, bool]:
    return {str(day): bool(flag) for day, flag in zip(frame["day"], use_soft)}


def _eval_policy(
    *,
    name: str,
    split: str,
    period: str,
    df: pd.DataFrame,
    q: np.ndarray,
    base_dec: pd.DataFrame,
    soft_dec: pd.DataFrame,
    daily: pd.DataFrame,
    use_soft: np.ndarray,
    stack: dict[str, Any],
    cfg: dict[str, Any],
    variant: sweep.Variant,
) -> dict[str, Any]:
    routed = _combine_decisions(df, base_dec, soft_dec, _route_map(daily, use_soft))
    return _eval(
        variant_name=name,
        split=split,
        period=period,
        df=df,
        q=q,
        dec=routed,
        stack=stack,
        cfg=cfg,
        variant=variant,
    )


def _period_slice(df: pd.DataFrame, dec: pd.DataFrame, soft_dec: pd.DataFrame, q: np.ndarray, start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    mask = ts.ge(start).to_numpy(dtype=bool)
    return (
        df.loc[mask].reset_index(drop=True),
        dec.loc[mask].reset_index(drop=True),
        soft_dec.loc[mask].reset_index(drop=True),
        q[mask],
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    decontam._assert_clean_frame(decontam.TRAIN_CSV, name="train")
    decontam._assert_clean_frame(decontam.EVAL_CSV, name="eval")
    decontam._assert_clean_parent(decontam.CANDIDATE_DIR / "primary_parent.pkl", name="primary")
    decontam._assert_clean_parent(decontam.CANDIDATE_DIR / "fallback_alpha43_no_legacy_parent.pkl", name="fallback")
    decontam._patch_runtime_sources()

    cfg = walk.precision._cfg_from_results()
    stack = walk.precision._load_stack()
    train_df = walk._load_full_train_frame()
    _, eval_df = walk.precision._load_frames()
    sources = walk.precision._decision_sources(train_df, eval_df, stack["parent"])
    train_dec = sources[str(cfg["source"])][0].reset_index(drop=True)
    eval_dec = sources[str(cfg["source"])][1].reset_index(drop=True)
    variant = sweep.Variant("deep_stop_cd18", deep_stop_cooldown_extra=18)

    train_q = v27._predict_all(stack["deep_model"], train_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    eval_q = v27._predict_all(stack["deep_model"], eval_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    train_soft_dec, _, train_block_summary = parent_soft._parent_soft_veto(train_df, train_dec, split="train2025")
    eval_soft_dec, _, eval_block_summary = parent_soft._parent_soft_veto(eval_df, eval_dec, split="oos2026")

    train_daily = _daily_frame(
        split="train2025",
        df=train_df,
        q=train_q,
        dec=train_dec,
        soft_dec=train_soft_dec,
        stack=stack,
        cfg=cfg,
        variant=variant,
    )
    eval_daily = _daily_frame(
        split="oos2026",
        df=eval_df,
        q=eval_q,
        dec=eval_dec,
        soft_dec=eval_soft_dec,
        stack=stack,
        cfg=cfg,
        variant=variant,
    )
    daily, feature_cols = _attach_previous_day_features(pd.concat([train_daily, eval_daily], ignore_index=True))
    daily.to_csv(DAILY_OUT, index=False)

    train_router = daily[(daily["split"].eq("train2025")) & (pd.to_datetime(daily["day_ts"]) <= ROUTER_TRAIN_END)].copy()
    val_router = daily[(daily["split"].eq("train2025")) & (pd.to_datetime(daily["day_ts"]) >= ROUTER_VAL_START)].copy()
    oos_router = daily[daily["split"].eq("oos2026")].copy()
    models = _fit_router_models(train_router, feature_cols)

    val_df_q4, val_dec_q4, val_soft_q4, val_q_q4 = _period_slice(train_df, train_dec, train_soft_dec, train_q, ROUTER_VAL_START)
    val_daily_q4 = val_router.copy().reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    rows.append(
        _eval(
            variant_name="baseline",
            split="val",
            period="2025Q4",
            df=val_df_q4,
            q=val_q_q4,
            dec=val_dec_q4,
            stack=stack,
            cfg=cfg,
            variant=variant,
        )
    )
    rows.append(
        _eval(
            variant_name="parent_soft_static",
            split="val",
            period="2025Q4",
            df=val_df_q4,
            q=val_q_q4,
            dec=val_soft_q4,
            stack=stack,
            cfg=cfg,
            variant=variant,
        )
    )
    rows.append(
        _eval(
            variant_name="baseline",
            split="oos",
            period="2026_full",
            df=eval_df,
            q=eval_q,
            dec=eval_dec,
            stack=stack,
            cfg=cfg,
            variant=variant,
        )
    )
    rows.append(
        _eval(
            variant_name="parent_soft_static",
            split="oos",
            period="2026_full",
            df=eval_df,
            q=eval_q,
            dec=eval_soft_dec,
            stack=stack,
            cfg=cfg,
            variant=variant,
        )
    )

    predictions: list[pd.DataFrame] = []
    val_policy_rows: list[dict[str, Any]] = []
    oos_policy_rows: list[dict[str, Any]] = []
    for model_name, model in models.items():
        val_prob = _predict_router(model, val_router, feature_cols)
        oos_prob = _predict_router(model, oos_router, feature_cols)
        predictions.append(pd.DataFrame({"split": "val", "day": val_router["day"].to_numpy(), "model": model_name, "prob_soft": val_prob}))
        predictions.append(pd.DataFrame({"split": "oos", "day": oos_router["day"].to_numpy(), "model": model_name, "prob_soft": oos_prob}))
        for threshold in THRESHOLDS:
            policy_name = f"meta_{model_name}_t{threshold:.2f}"
            val_use = val_prob >= threshold
            oos_use = oos_prob >= threshold
            val_row = _eval_policy(
                name=policy_name,
                split="val",
                period="2025Q4",
                df=val_df_q4,
                q=val_q_q4,
                base_dec=val_dec_q4,
                soft_dec=val_soft_q4,
                daily=val_daily_q4,
                use_soft=val_use,
                stack=stack,
                cfg=cfg,
                variant=variant,
            )
            oos_row = _eval_policy(
                name=policy_name,
                split="oos",
                period="2026_full",
                df=eval_df,
                q=eval_q,
                base_dec=eval_dec,
                soft_dec=eval_soft_dec,
                daily=oos_router,
                use_soft=oos_use,
                stack=stack,
                cfg=cfg,
                variant=variant,
            )
            val_row["soft_days"] = int(val_use.sum())
            val_row["total_days"] = int(len(val_use))
            oos_row["soft_days"] = int(oos_use.sum())
            oos_row["total_days"] = int(len(oos_use))
            val_policy_rows.append(val_row)
            oos_policy_rows.append(oos_row)
            rows.extend([val_row, oos_row])

    for threshold in (0.02, 0.05, 0.10):
        policy_name = f"heur_prev_block_ratio_t{threshold:.2f}"
        val_use = val_router["prev_blocked_active_ratio"].to_numpy(dtype=np.float64) >= threshold
        oos_use = oos_router["prev_blocked_active_ratio"].to_numpy(dtype=np.float64) >= threshold
        val_row = _eval_policy(
            name=policy_name,
            split="val",
            period="2025Q4",
            df=val_df_q4,
            q=val_q_q4,
            base_dec=val_dec_q4,
            soft_dec=val_soft_q4,
            daily=val_daily_q4,
            use_soft=val_use,
            stack=stack,
            cfg=cfg,
            variant=variant,
        )
        oos_row = _eval_policy(
            name=policy_name,
            split="oos",
            period="2026_full",
            df=eval_df,
            q=eval_q,
            base_dec=eval_dec,
            soft_dec=eval_soft_dec,
            daily=oos_router,
            use_soft=oos_use,
            stack=stack,
            cfg=cfg,
            variant=variant,
        )
        val_row["soft_days"] = int(val_use.sum())
        val_row["total_days"] = int(len(val_use))
        oos_row["soft_days"] = int(oos_use.sum())
        oos_row["total_days"] = int(len(oos_use))
        val_policy_rows.append(val_row)
        oos_policy_rows.append(oos_row)
        rows.extend([val_row, oos_row])

    val_oracle_use = val_router["use_soft_label"].to_numpy(dtype=bool)
    oos_oracle_use = oos_router["use_soft_label"].to_numpy(dtype=bool)
    rows.append(
        _eval_policy(
            name="oracle_daily_non_deployable",
            split="val",
            period="2025Q4",
            df=val_df_q4,
            q=val_q_q4,
            base_dec=val_dec_q4,
            soft_dec=val_soft_q4,
            daily=val_daily_q4,
            use_soft=val_oracle_use,
            stack=stack,
            cfg=cfg,
            variant=variant,
        )
    )
    rows.append(
        _eval_policy(
            name="oracle_daily_non_deployable",
            split="oos",
            period="2026_full",
            df=eval_df,
            q=eval_q,
            base_dec=eval_dec,
            soft_dec=eval_soft_dec,
            daily=oos_router,
            use_soft=oos_oracle_use,
            stack=stack,
            cfg=cfg,
            variant=variant,
        )
    )

    grid = pd.DataFrame(rows)
    grid.to_csv(GRID_OUT, index=False)
    if predictions:
        pd.concat(predictions, ignore_index=True).to_csv(PRED_OUT, index=False)
    else:
        pd.DataFrame(columns=["split", "day", "model", "prob_soft"]).to_csv(PRED_OUT, index=False)

    val_candidates = pd.DataFrame(val_policy_rows)
    if len(val_candidates):
        best_val = val_candidates.sort_values(["score", "pnl"], ascending=False).iloc[0].to_dict()
        best_oos = pd.DataFrame(oos_policy_rows)
        best_oos = best_oos[best_oos["variant"].eq(best_val["variant"])].iloc[0].to_dict()
    else:
        best_val = {}
        best_oos = {}

    summary = {
        "model_id": MODEL_ID,
        "scope": "Research-only path-aware router for choosing baseline vs parent-soft by day. Training uses previous-day features; 2026 labels are never used for fitting or threshold selection.",
        "train_window": {"fit_end": str(ROUTER_TRAIN_END), "validation_start": str(ROUTER_VAL_START)},
        "artifacts": {"grid": str(GRID_OUT), "daily_training_frame": str(DAILY_OUT), "router_predictions": str(PRED_OUT)},
        "block_summary": {"train2025": train_block_summary, "oos2026": eval_block_summary},
        "router_training_rows": int(len(train_router)),
        "router_validation_rows": int(len(val_router)),
        "router_oos_rows": int(len(oos_router)),
        "router_positive_label_rate": {
            "train_fit": float(train_router["use_soft_label"].mean()) if len(train_router) else 0.0,
            "validation": float(val_router["use_soft_label"].mean()) if len(val_router) else 0.0,
            "oos_diagnostic_only": float(oos_router["use_soft_label"].mean()) if len(oos_router) else 0.0,
        },
        "best_by_validation": {"validation": best_val, "oos_same_policy": best_oos},
        "decision": "Do not promote unless validation beats baseline and OOS improvement is not only oracle/path artifact.",
    }
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(SUMMARY_OUT), "grid": str(GRID_OUT), "daily_training_frame": str(DAILY_OUT)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
