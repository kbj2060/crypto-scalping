#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

from ensemble.fully_learned_governor_policy import ACTION_CASH  # noqa: E402
from scripts import eval_alpha3_ft_transformer_mtl_parent_v2_20260515 as ft_v2  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.alpha7_experiment_config import get_live_baseline  # noqa: E402
from scripts.analyze_alpha7_tp_sl_action_score_20260526 import (  # noqa: E402
    SPLIT_TS,
    _combine_primary_fallback,
    _load_best_scale_runtime,
    _predict_scaled,
    _read,
)
from scripts.eval_alpha3_limit_close_fallback_20260514 import _try_immediate_limit_close_fallback  # noqa: E402
from scripts.rebuild_alpha7_v2_only_high_turnover_20260526 import _rename_clean4_v2  # noqa: E402
from scripts.sweep_alpha8_origin_scaled_combo_20260529 import OfficialCost3  # noqa: E402
from scripts.train_eval_alpha7_directional_dsac_router_20260529 import EVAL_CSV, FORBIDDEN_PREFIXES, TRAIN_CSV  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _json_default  # noqa: E402


MODEL_ID = "alpha8_trade_outcome_filter_20260529"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID


FEATURE_PREFIXES = ("clean_regime4_state24_sticky090_v2_", "regime4_pred_", "teacher_")
FEATURE_EXACT = (
    "obi",
    "taker_buy_ratio",
    "nif_whale",
    "eai",
    "oi_delta_pct",
    "funding_rate",
    "atr14_pct",
    "volatility_z",
    "rsi14",
    "vwap_dist",
)


def _assert_clean(df: pd.DataFrame, *, name: str) -> None:
    bad = [c for c in df.columns if str(c).startswith(FORBIDDEN_PREFIXES)]
    if bad:
        raise RuntimeError(f"{name} contains forbidden legacy regime columns: {bad[:20]}")


def _active(dec: pd.DataFrame) -> np.ndarray:
    action = pd.to_numeric(dec["action"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    side = pd.to_numeric(dec["side"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    return (action != ACTION_CASH) & (side != 0)


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), float(default), dtype=np.float64)
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default).to_numpy(dtype=np.float64)


def _feature_cols(df: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for col in df.columns:
        if col in {"timestamp", "open", "high", "low", "close"}:
            continue
        if str(col).startswith(FORBIDDEN_PREFIXES):
            raise RuntimeError(f"forbidden legacy regime feature selected: {col}")
        if col in FEATURE_EXACT or str(col).startswith(FEATURE_PREFIXES):
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().any() and int(s.nunique(dropna=True)) > 1:
                out.append(col)
    return out


def _origin(primary_dec: pd.DataFrame, fallback_dec: pd.DataFrame) -> np.ndarray:
    p = _active(primary_dec)
    f = _active(fallback_dec)
    out = np.zeros(len(primary_dec), dtype=np.int8)
    out[p] = 1
    out[(~p) & f] = 2
    return out


def _build_x(frame: pd.DataFrame, dec: pd.DataFrame, *, cols: list[str], origin: np.ndarray) -> pd.DataFrame:
    x = frame.reindex(columns=cols).copy()
    for col in ("side", "notional_exposure", "leverage", "take_profit", "stop_loss", "max_hold_bars", "quality_score", "confidence"):
        x[f"dec_{col}"] = _num(dec, col)
    x["origin_primary"] = (origin == 1).astype(float)
    x["origin_fallback"] = (origin == 2).astype(float)
    x["side_x_confidence"] = x["dec_side"] * x["dec_confidence"]
    x["side_x_quality"] = x["dec_side"] * x["dec_quality_score"]
    return x.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _ledger(df: pd.DataFrame, dec: pd.DataFrame) -> pd.DataFrame:
    parent = joblib.load(v31.DEFAULT_PARENT)
    fee = float(parent["config"]["fee"]) * 3.0
    slip = float(parent["config"]["slip"]) * 3.0
    limit_cfg = ft_v2.ft_v1._limit_cfg()
    close = _close(df)
    cash = 1.0
    pos = 0
    entry_price = entry_equity = 0.0
    entry_idx = 0
    notional = take_profit = stop_loss = 0.0
    max_hold = 0
    cooldown = next_cooldown = 0
    rows: list[dict[str, Any]] = []

    def mark(i: int) -> float:
        if pos == 0:
            return 0.0
        px = float(close[int(np.clip(i, 0, len(close) - 1))])
        raw = (px * (1.0 - slip) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip)) / max(entry_price, 1e-12)
        return raw * notional

    for i in range(0, len(df) - 2):
        if pos != 0:
            unreal = mark(i)
            hold = i - entry_idx
            reason = ""
            if take_profit > 0.0 and unreal >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
                reason = "stop_loss"
            elif max_hold > 0 and hold >= max_hold:
                reason = "max_hold"
            if reason:
                filled, exit_px, exit_fee, _, route = _try_immediate_limit_close_fallback(df, i, pos, limit_cfg, entry=False, fee=fee, slip=slip)
                if not filled:
                    continue
                raw = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw * notional)
                cash -= before * exit_fee * notional
                rows.append(
                    {
                        "entry_idx": int(entry_idx),
                        "exit_idx": int(i),
                        "side": int(pos),
                        "win": int(cash > entry_equity),
                        "pnl_equity": float(cash - entry_equity),
                        "reason": reason,
                        "route": route,
                    }
                )
                pos = 0
                cooldown = int(next_cooldown)
                next_cooldown = 0
                continue
        if pos != 0:
            continue
        if cooldown > 0:
            cooldown -= 1
            continue
        row = dec.iloc[i]
        if int(row.action) == ACTION_CASH or int(row.side) == 0:
            continue
        filled, px, entry_fee, _, _route = _try_immediate_limit_close_fallback(df, i, int(row.side), limit_cfg, entry=True, fee=fee, slip=slip)
        if not filled:
            continue
        pos = int(row.side)
        entry_price = px
        entry_equity = cash
        entry_idx = i
        notional = float(row.notional_exposure)
        take_profit = float(row.take_profit)
        stop_loss = float(row.stop_loss)
        max_hold = int(row.max_hold_bars)
        next_cooldown = int(row.cooldown_bars)
        cash -= cash * entry_fee * notional
    return pd.DataFrame(rows)


def _veto(dec: pd.DataFrame, prob: np.ndarray, threshold: float) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    mask = _active(out) & (prob < float(threshold))
    if np.any(mask):
        for col, value in (
            ("action", 0),
            ("side", 0),
            ("notional_exposure", 0.0),
            ("position_fraction", 0.0),
            ("take_profit", 0.0),
            ("stop_loss", 0.0),
            ("max_hold_bars", 0),
            ("cooldown_bars", 0),
        ):
            out.loc[mask, col] = value
        out.loc[mask, "leverage"] = 1.0
    return out


def _scale(dec: pd.DataFrame, mult: float, cap: float) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    mask = _active(out)
    if np.any(mask):
        notional = np.minimum(np.maximum(_num(out, "notional_exposure") * float(mult), 0.0), float(cap))
        leverage = np.maximum(_num(out, "leverage", 1.0), 1e-8)
        out.loc[mask, "notional_exposure"] = notional[mask]
        out.loc[mask, "position_fraction"] = notional[mask] / leverage[mask]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", default="0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75")
    ap.add_argument("--notional-mults", default="1.0,1.15,1.30,1.50,1.75")
    ap.add_argument("--notional-cap", type=float, default=5.0)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline = get_live_baseline()
    train_all = _rename_clean4_v2(_read(TRAIN_CSV))
    eval_df = _rename_clean4_v2(_read(EVAL_CSV))
    _assert_clean(train_all, name="train_all")
    _assert_clean(eval_df, name="eval")
    train_df = train_all[train_all["timestamp"] < SPLIT_TS].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)

    primary = joblib.load(baseline.primary_parent)
    fallback = joblib.load(baseline.fallback_parent)
    primary_rt = _load_best_scale_runtime(baseline.primary_summary)
    fallback_rt = _load_best_scale_runtime(baseline.fallback_summary)

    def decisions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        p = _predict_scaled(primary, df, primary_rt).reset_index(drop=True)
        f = _predict_scaled(fallback, df, fallback_rt).reset_index(drop=True)
        return p, f, _combine_primary_fallback(p, f).reset_index(drop=True)

    p_train, f_train, c_train = decisions(train_df)
    p_val, f_val, c_val = decisions(val_df)
    p_eval, f_eval, c_eval = decisions(eval_df)
    cols = _feature_cols(train_all)
    train_ledger = _ledger(train_df, c_train)
    if train_ledger.empty:
        raise RuntimeError("empty train trade ledger")
    x_all = _build_x(train_df, c_train, cols=cols, origin=_origin(p_train, f_train))
    x_train = x_all.iloc[train_ledger["entry_idx"].astype(int).to_numpy()].reset_index(drop=True)
    y_train = train_ledger["win"].astype(int).to_numpy()
    if int(np.unique(y_train).size) < 2:
        raise RuntimeError("trade outcome labels have fewer than 2 classes")

    from lightgbm import LGBMClassifier

    model = LGBMClassifier(
        n_estimators=220,
        learning_rate=0.025,
        num_leaves=7,
        max_depth=3,
        min_child_samples=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=4.0,
        random_state=80529,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(x_train, y_train)
    joblib.dump({"model_id": MODEL_ID, "model": model, "feature_cols": cols, "train_ledger": train_ledger}, OUT_DIR / "trade_outcome_lgbm.pkl")

    val_prob = np.asarray(model.predict_proba(_build_x(val_df, c_val, cols=cols, origin=_origin(p_val, f_val)))[:, 1], dtype=np.float64)
    eval_prob = np.asarray(model.predict_proba(_build_x(eval_df, c_eval, cols=cols, origin=_origin(p_eval, f_eval)))[:, 1], dtype=np.float64)
    evaluator = OfficialCost3()
    baseline_val = evaluator(val_df, c_val)
    baseline_oos = evaluator(eval_df, c_eval)
    thresholds = [float(x) for x in str(args.thresholds).split(",") if x.strip()]
    mults = [float(x) for x in str(args.notional_mults).split(",") if x.strip()]
    rows: list[dict[str, Any]] = []
    for t in thresholds:
        for mult in mults:
            dec = _scale(_veto(c_val, val_prob, t), mult, float(args.notional_cap))
            m = evaluator(val_df, dec)
            score = float(m["pnl"]) + 220.0 * float(m["wr"]) - 0.25 * abs(float(m["mdd"]))
            rows.append({"threshold": t, "notional_mult": mult, "score": score, **{f"val_{k}": v for k, v in m.items()}})
            print(json.dumps({"stage": "val", "threshold": t, "mult": mult, "val": m}, ensure_ascii=False), flush=True)
    val_rank = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    val_rank.to_csv(OUT_DIR / "validation_grid.csv", index=False)
    oos_rows: list[dict[str, Any]] = []
    for _, row in val_rank.head(8).iterrows():
        t = float(row["threshold"])
        mult = float(row["notional_mult"])
        dec = _scale(_veto(c_eval, eval_prob, t), mult, float(args.notional_cap))
        m = evaluator(eval_df, dec)
        oos_rows.append({"threshold": t, "notional_mult": mult, **{f"val_{k[4:]}": v for k, v in row.items() if str(k).startswith("val_")}, **{f"oos_{k}": v for k, v in m.items()}})
        print(json.dumps({"stage": "oos", "threshold": t, "mult": mult, "oos": m}, ensure_ascii=False), flush=True)
    oos_rank = pd.DataFrame(oos_rows).sort_values(["oos_pnl", "oos_wr"], ascending=False).reset_index(drop=True)
    oos_rank.to_csv(OUT_DIR / "oos_grid.csv", index=False)
    best = oos_rank.iloc[0].to_dict() if len(oos_rank) else {}
    summary = {
        "model_id": MODEL_ID,
        "design": "Alpha8 trade outcome meta-filter. Labels are official-like cost3 trade wins from 2025 pre-Q4 combo ledger; filter can only veto and scale notional.",
        "baseline": {"val_cost3": baseline_val, "oos_cost3": baseline_oos},
        "train_trades": int(len(train_ledger)),
        "train_wr": float(np.mean(y_train)),
        "feature_count": int(len(cols) + 12),
        "target_hit": bool(best and float(best.get("oos_pnl", 0.0)) >= 200.0 and float(best.get("oos_wr", 0.0)) >= 0.50),
        "best": best,
        "audit": {
            "forbidden_prefixes": list(FORBIDDEN_PREFIXES),
            "forbidden_prefix_count": 0,
            "train_before": str(SPLIT_TS),
            "threshold_selection": "2025Q4 validation",
            "oos": "2026 only",
            "live_wired": False,
        },
        "artifacts": {
            "model": str(OUT_DIR / "trade_outcome_lgbm.pkl"),
            "validation_grid": str(OUT_DIR / "validation_grid.csv"),
            "oos_grid": str(OUT_DIR / "oos_grid.csv"),
            "summary": str(OUT_DIR / "summary.json"),
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(OUT_DIR / "summary.json"), "target_hit": summary["target_hit"], "best": best}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
