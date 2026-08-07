#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import (  # noqa: E402
    FullyLearnedGovernorConfig,
    build_training_set,
    predict_policy_frame,
    train_policy,
)
from scripts import precision_retest_01965_alpha7_combo_20260527 as precision  # noqa: E402
from scripts import runtime_retest_alpha7_1_01965_decontam_20260528 as decontam  # noqa: E402
from scripts import sweep_decontam_deep_alpha_controls_20260528 as sweep  # noqa: E402
from scripts import train_eval_hf_v13_deep_alpha_candidate_expansion_v27 as v27  # noqa: E402
from scripts.analyze_alpha7_tp_sl_action_score_20260526 import (  # noqa: E402
    SPLIT_TS,
    TP_COL,
    _close,
    _combine_primary_fallback,
    _read,
)
from scripts.retrain_alpha7_1_01965_tp_sl_decontam_20260528 import (  # noqa: E402
    EVAL_CSV,
    FALLBACK_PARENT,
    PRIMARY_PARENT,
    TRAIN_CSV,
    _assert_clean_frame,
    _assert_feature_cols,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


MODEL_ID = "alpha7_daytrade_parent_topk_retrain_20260528"
OUT_DIR = ROOT / f"tmp/causal_regen_20260516/{MODEL_ID}"
GRID_OUT = OUT_DIR / "grid.csv"
SUMMARY_OUT = OUT_DIR / "summary.json"
DAYTRADE_LABEL_TOP_K_PER_DAY = 3


DAYTRADE_LABEL_CFG = FullyLearnedGovernorConfig(
    notional_buckets=(0.35, 0.55, 0.80, 1.10, 1.50, 2.10, 3.00),
    leverage_buckets=(1.5, 2.0, 3.0, 4.0, 5.0),
    take_profit_buckets=(0.018, 0.030, 0.050, 0.090, 0.180, 0.450, 1.000),
    stop_loss_buckets=(0.012, 0.018, 0.024, 0.035, 0.055, 0.080),
    max_hold_buckets=(24, 48, 72, 96, 144, 288, 576, 864),
    cooldown_buckets=(12, 24, 48, 72, 96),
    max_train_horizon_bars=864,
    fee=0.0005,
    slip=0.0002,
    cash_score=0.0015,
    adverse_penalty=0.95,
    size_penalty=0.020,
    hold_penalty=0.0010,
    turnover_bonus=0.0,
    max_margin_fraction=1.0,
)


def _train_or_load(
    *,
    name: str,
    train_all: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
    stride_bars: int,
) -> dict[str, Any]:
    model_dir = OUT_DIR / name
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "parent.pkl"
    summary_path = model_dir / "summary.json"
    if model_path.exists() and summary_path.exists():
        return {"bundle": joblib.load(model_path), "summary": json.loads(summary_path.read_text(encoding="utf-8"))}

    train_df = train_all[train_all["timestamp"] < SPLIT_TS].reset_index(drop=True)
    x_train, y_train, meta = build_training_set(
        train_df,
        cfg=DAYTRADE_LABEL_CFG,
        stride_bars=int(stride_bars),
        batch_size=384,
        feature_cols=feature_cols,
    )
    y_train, topk_meta = _apply_daily_topk_labels(
        y_train,
        train_df=train_df,
        stride_bars=int(stride_bars),
        top_k=int(DAYTRADE_LABEL_TOP_K_PER_DAY),
    )
    bundle = train_policy(x_train, y_train, cfg=DAYTRADE_LABEL_CFG, random_state=int(seed), feature_cols=feature_cols)
    joblib.dump(bundle, model_path)
    summary = {
        "name": name,
        "model_path": str(model_path),
        "feature_count": int(len(feature_cols)),
        "contains_tp_sl_action_score": bool(TP_COL in feature_cols),
        "label_cfg": asdict(DAYTRADE_LABEL_CFG),
        "train_meta": meta,
        "topk_meta": topk_meta,
        "label_distribution": bundle.get("label_distribution", {}),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return {"bundle": bundle, "summary": summary}


def _apply_daily_topk_labels(
    y: dict[str, np.ndarray],
    *,
    train_df: pd.DataFrame,
    stride_bars: int,
    top_k: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    out = {k: np.array(v, copy=True) for k, v in y.items()}
    h = int(DAYTRADE_LABEL_CFG.max_train_horizon_bars)
    valid = np.arange(0, max(0, len(train_df) - h - 1), max(1, int(stride_bars)), dtype=np.int64)
    if len(valid) != len(out["action"]):
        raise RuntimeError(f"top-k label index mismatch: valid={len(valid)} labels={len(out['action'])}")
    ts = pd.to_datetime(train_df.iloc[valid]["timestamp"], errors="coerce").reset_index(drop=True)
    action = pd.Series(out["action"])
    quality = pd.Series(out["quality"])
    trade_mask = action.ne(0)
    keep = pd.Series(False, index=action.index)
    for _, idx in action[trade_mask].groupby(ts[trade_mask].dt.date).groups.items():
        chosen = quality.loc[list(idx)].sort_values(ascending=False).head(int(top_k)).index
        keep.loc[chosen] = True
    drop = trade_mask & keep.ne(True)
    for key in ("action", "notional", "leverage", "take_profit", "stop_loss", "max_hold", "cooldown"):
        out[key][drop.to_numpy()] = 0
    out["quality"][drop.to_numpy()] = float(DAYTRADE_LABEL_CFG.cash_score)
    return out, {
        "top_k_per_day": int(top_k),
        "labels_before": {str(k): int(v) for k, v in action.value_counts().sort_index().to_dict().items()},
        "labels_after": {str(k): int(v) for k, v in pd.Series(out["action"]).value_counts().sort_index().to_dict().items()},
        "kept_trade_labels": int(keep.sum()),
        "dropped_trade_labels": int(drop.sum()),
    }


def _active_count(dec: pd.DataFrame) -> int:
    action = pd.to_numeric(dec["action"], errors="coerce").fillna(0).astype(int)
    side = pd.to_numeric(dec["side"], errors="coerce").fillna(0).astype(int)
    return int(((action != 0) & (side != 0)).sum())


def _base_runtime_cfg() -> dict[str, Any]:
    cfg = precision._cfg_from_results()
    cfg.update(
        {
            "name": "alpha7_daytrade_parent_retrain",
            "entry_quality_min": -999.0,
            "entry_conf_min": 0.0,
            "parent_notional_mult": 1.0,
            "parent_notional_cap": 2.0,
            "parent_tp_mult": 1.0,
            "parent_sl_mult": 1.0,
            "parent_hold_mult": 1.0,
            "parent_hold_cap": 864,
            "alpha6_bucketize_hold": False,
            "hard_sl_mult": 2.4,
            "soft_sl_mult": 1.05,
            "early_bars": 72,
            "early_sl_mult": 1.4,
            "soft_min_hold": 24,
            "soft_persist_bars": 2,
            "regime_bad_th": 0.55,
            "flow_bad_th": 0.03,
            "giveback_trigger": 0.96,
            "giveback_min_mfe": 0.014,
            "giveback_min_hold": 24,
            "same_side_entry_gap": 48,
            "cooldown_after_hard_stop": 12,
            "cooldown_after_soft_stop": 12,
            "cooldown_after_giveback": 24,
            "deep_notional_mult": 1.0,
            "deep_tp_mult": 1.0,
            "deep_sl_mult": 1.0,
            "deep_hold_mult": 1.5,
            "deep_trail_activation": 0.018,
        }
    )
    return cfg


def _eval_decisions(
    *,
    name: str,
    val_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    val_dec: pd.DataFrame,
    eval_dec: pd.DataFrame,
    stack: dict[str, Any],
    val_q: np.ndarray,
    eval_q: np.ndarray,
    cfg: dict[str, Any],
    deep_side: str,
) -> dict[str, Any]:
    variant = sweep.Variant(name=name, deep_side=deep_side, deep_stop_cooldown_extra=18)
    out: dict[str, Any] = {}
    for split, df, q, dec in (("val", val_df, val_q, val_dec), ("oos", eval_df, eval_q, eval_dec)):
        res = sweep._backtest_variant(
            df=df,
            q=q,
            dec=dec,
            stack=stack,
            cfg=cfg,
            variant=variant,
            cost_mult=3,
            record=split == "oos",
        )
        records = list(res.pop("trade_records", [])) if split == "oos" else []
        row = {
            "pnl": float(res["pnl"]),
            "mdd": float(res["mdd"]),
            "wr": float(res["wr"]),
            "trades": int(res["trades"]),
            "trades_per_day": float(res["trades_per_day"]),
            "deep_entries": int(res.get("deep_entries", 0)),
            "long_entries": int(res.get("long_entries", 0)),
            "short_entries": int(res.get("short_entries", 0)),
            "sl_ratio": float(sweep._sl_ratio(res)),
            "score": float(sweep._score(res)),
            "exits": res.get("exits", {}),
        }
        out[split] = row
        if split == "oos":
            ledger_path = OUT_DIR / f"{name}_oos_cost3_ledger.csv"
            pd.DataFrame(records).to_csv(ledger_path, index=False)
            out["oos_ledger"] = str(ledger_path)
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_all = _read(TRAIN_CSV)
    eval_df = _read(EVAL_CSV)
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
    _assert_clean_frame(train_all, name="train")
    _assert_clean_frame(eval_df, name="eval")

    live_primary = joblib.load(PRIMARY_PARENT)
    live_fallback = joblib.load(FALLBACK_PARENT)
    primary_cols = list(live_primary["feature_cols"])
    fallback_cols = list(live_fallback["feature_cols"])
    for name, cols in {"primary": primary_cols, "fallback": fallback_cols}.items():
        _assert_feature_cols(train_all, cols, name=name)
        _assert_feature_cols(eval_df, cols, name=name)

    primary = _train_or_load(name="primary_daytrade", train_all=train_all, feature_cols=primary_cols, seed=5288001, stride_bars=12)
    fallback = _train_or_load(name="fallback_daytrade", train_all=train_all, feature_cols=fallback_cols, seed=5288002, stride_bars=12)

    p_val = predict_policy_frame(primary["bundle"], val_df, close=_close(val_df), strict=False)
    p_oos = predict_policy_frame(primary["bundle"], eval_df, close=_close(eval_df), strict=False)
    f_val = predict_policy_frame(fallback["bundle"], val_df, close=_close(val_df), strict=False)
    f_oos = predict_policy_frame(fallback["bundle"], eval_df, close=_close(eval_df), strict=False)
    combo_val = _combine_primary_fallback(p_val, f_val)
    combo_oos = _combine_primary_fallback(p_oos, f_oos)

    stack = precision._load_stack()
    val_q = v27._predict_all(stack["deep_model"], val_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    eval_q = v27._predict_all(stack["deep_model"], eval_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    cfg = _base_runtime_cfg()

    variants: dict[str, tuple[pd.DataFrame, pd.DataFrame, str]] = {
        "daytrade_primary_deep_off": (p_val, p_oos, "none"),
        "daytrade_combo_deep_off": (combo_val, combo_oos, "none"),
        "daytrade_primary_deep_cd18": (p_val, p_oos, "both"),
        "daytrade_combo_deep_cd18": (combo_val, combo_oos, "both"),
    }

    rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    for name, (vdec, odec, deep_side) in variants.items():
        report = _eval_decisions(
            name=name,
            val_df=val_df,
            eval_df=eval_df,
            val_dec=vdec,
            eval_dec=odec,
            stack=stack,
            val_q=val_q,
            eval_q=eval_q,
            cfg=cfg,
            deep_side=deep_side,
        )
        reports[name] = report
        rows.append(
            {
                "name": name,
                "deep_side": deep_side,
                "val_pnl": report["val"]["pnl"],
                "val_mdd": report["val"]["mdd"],
                "val_wr": report["val"]["wr"],
                "val_trades": report["val"]["trades"],
                "val_trades_per_day": report["val"]["trades_per_day"],
                "val_sl_ratio": report["val"]["sl_ratio"],
                "oos_pnl": report["oos"]["pnl"],
                "oos_mdd": report["oos"]["mdd"],
                "oos_wr": report["oos"]["wr"],
                "oos_trades": report["oos"]["trades"],
                "oos_trades_per_day": report["oos"]["trades_per_day"],
                "oos_deep_entries": report["oos"]["deep_entries"],
                "oos_long_entries": report["oos"]["long_entries"],
                "oos_short_entries": report["oos"]["short_entries"],
                "oos_sl_ratio": report["oos"]["sl_ratio"],
                "oos_score": report["oos"]["score"],
                "oos_ledger": report.get("oos_ledger", ""),
            }
        )

    grid = pd.DataFrame(rows).sort_values(["oos_pnl", "val_pnl"], ascending=[False, False])
    grid.to_csv(GRID_OUT, index=False)
    summary = {
        "model_id": MODEL_ID,
        "scope": "Retrain Alpha7 primary/fallback with daytrade label config. Feature/data contract unchanged.",
        "label_cfg": asdict(DAYTRADE_LABEL_CFG),
        "runtime_cfg": cfg,
        "artifacts": {
            "primary": primary["summary"],
            "fallback": fallback["summary"],
        },
        "active_counts": {
            "primary_val": _active_count(p_val),
            "primary_oos": _active_count(p_oos),
            "fallback_val": _active_count(f_val),
            "fallback_oos": _active_count(f_oos),
            "combo_val": _active_count(combo_val),
            "combo_oos": _active_count(combo_oos),
        },
        "grid": str(GRID_OUT),
        "ranking": rows,
        "best_by_oos_pnl": str(grid.iloc[0]["name"]) if not grid.empty else "",
        "reports": reports,
    }
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(SUMMARY_OUT), "grid": str(GRID_OUT), "best": summary["best_by_oos_pnl"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
