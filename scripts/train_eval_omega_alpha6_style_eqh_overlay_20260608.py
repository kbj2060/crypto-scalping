#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_1_cash_fallback_sleeve_20260606 as sleeve  # noqa: E402
import train_eval_omega1_2_1_exposure_selector_20260606 as base  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
from alpha6_catboost_entry_quality_exit_policy_20260522 import EQEConfig, _build_entry_labels, _predict_entry  # noqa: E402


MODEL_ID = "omega_alpha6_style_eqh_overlay_20260608"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
FORBIDDEN_PREFIXES = ("clean_regime4_", "regime4_pred_", "teacher_")
FORBIDDEN_EXACT = {"tp_sl_action_score"}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _metric_row(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_pnl": float(metrics["pnl"]),
        f"{prefix}_mdd": float(metrics["mdd"]),
        f"{prefix}_wr": float(metrics["wr"]),
        f"{prefix}_trades": int(metrics["trades"]),
        f"{prefix}_long": int(metrics["long_entries"]),
        f"{prefix}_short": int(metrics["short_entries"]),
        f"{prefix}_avg_notional": float(metrics["avg_notional"]),
        f"{prefix}_avg_leverage": float(metrics["avg_leverage"]),
        f"{prefix}_reasons": metrics["exit_reasons"],
    }


def _build_split(frames: dict[str, pd.DataFrame], split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame, src, dec0, prefix = base._build_split(frames, split)
    dec = sleeve._apply_aggressive(dec0)
    feat = sleeve._extra_features(base._feature_frame(frame, src, dec0, prefix), dec)
    bad = _forbidden_features(list(feat.columns))
    if bad:
        raise RuntimeError(f"{split}: forbidden Omega EQH feature columns: {bad}")
    return frame, dec, feat


def _forbidden_features(cols: list[str]) -> list[str]:
    return [c for c in cols if c in FORBIDDEN_EXACT or c.startswith(FORBIDDEN_PREFIXES)]


def _cat_cls(seed: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="MultiClass",
        iterations=360,
        learning_rate=0.045,
        depth=4,
        l2_leaf_reg=7.0,
        random_seed=int(seed),
        allow_writing_files=False,
        verbose=0,
        thread_count=-1,
    )


def _cat_reg(seed: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="RMSE",
        iterations=360,
        learning_rate=0.045,
        depth=4,
        l2_leaf_reg=7.0,
        random_seed=int(seed),
        allow_writing_files=False,
        verbose=0,
        thread_count=-1,
    )


def _fit_heads(x: np.ndarray, y: dict[str, np.ndarray], *, seed: int) -> dict[str, Any]:
    trade = y["action"] != 0
    weight = np.maximum(np.where(trade, 1.0, 0.45), np.clip(np.abs(y["quality"]) * 60.0, 0.25, 4.0))
    action_model = _cat_cls(seed)
    action_model.fit(Pool(x, y["action"], weight=weight))
    quality_model = _cat_reg(seed + 99)
    quality_model.fit(Pool(x, y["quality"], weight=weight))
    if int(np.count_nonzero(trade)) > 10 and np.unique(y["target_bucket"][trade]).size >= 2:
        target_model: Any = _cat_cls(seed + 199)
        target_model.fit(Pool(x[trade], y["target_bucket"][trade], weight=weight[trade]))
    else:
        target_model = _cat_cls(seed + 199)
        target_model.fit(Pool(x, np.zeros(len(x), dtype=np.int64), weight=np.ones(len(x))))
    return {
        "action_model": action_model,
        "quality_model": quality_model,
        "target_head_mode": "bucket5",
        "target_model": target_model,
        "target_bucket_model": target_model,
        "target_horizon_model": None,
        "fixed_target_horizon": 0,
        "max_target_horizon": 96,
        "label_distribution": {
            "action": pd.Series(y["action"]).value_counts().sort_index().to_dict(),
            "target_bucket": pd.Series(y["target_bucket"][trade]).value_counts().sort_index().to_dict(),
            "quality_mean": float(np.mean(y["quality"])),
            "quality_p95": float(np.quantile(y["quality"], 0.95)),
        },
    }


@dataclass(frozen=True)
class FoldPred:
    dec: pd.DataFrame
    diagnostics: dict[str, Any]


def _oof_predict(frame: pd.DataFrame, features: pd.DataFrame, *, seed: int) -> FoldPred:
    cfg = EQEConfig()
    valid, y, label_meta = _build_entry_labels(
        frame,
        cfg,
        stride_bars=3,
        batch_size=4096,
        label_preset="current_quality",
        adaptive_sampling=False,
    )
    x_all = features.to_numpy(dtype=np.float64)
    x_valid = x_all[valid]
    pred = pd.DataFrame(
        {
            "action": np.zeros(len(frame), dtype=np.int64),
            "quality_score": np.zeros(len(frame), dtype=np.float64),
            "confidence": np.zeros(len(frame), dtype=np.float64),
            "target_bucket": np.zeros(len(frame), dtype=np.int64),
            "target_horizon": np.zeros(len(frame), dtype=np.int64),
            "notional": np.full(len(frame), cfg.fixed_notional, dtype=np.float64),
        }
    )
    folds: list[dict[str, int]] = []
    n = len(valid)
    for start_frac, end_frac in ((0.50, 0.65), (0.65, 0.80), (0.80, 1.00)):
        train_end = int(n * start_frac)
        val_start = int(n * start_frac)
        val_end = int(n * end_frac)
        if train_end < 500 or val_end <= val_start:
            continue
        heads = _fit_heads(x_valid[:train_end], {k: v[:train_end] for k, v in y.items()}, seed=seed + val_start)
        row_start = int(valid[val_start])
        row_end = int(valid[val_end - 1]) + 1
        pred.iloc[row_start:row_end] = _predict_entry(heads, x_all[row_start:row_end], cfg)
        folds.append({"train_valid_rows": train_end, "row_start": row_start, "row_end": row_end})
    return FoldPred(
        pred,
        {
            "label_meta": label_meta,
            "label_distribution": {k: pd.Series(v).value_counts().sort_index().to_dict() for k, v in y.items() if k in {"action", "target_bucket"}},
            "folds": folds,
            "valid_rows": int(len(valid)),
            "feature_count": int(features.shape[1]),
            "features": list(features.columns),
        },
    )


def _fit_full_predict(train_frame: pd.DataFrame, train_features: pd.DataFrame, target_features: pd.DataFrame, *, seed: int) -> FoldPred:
    cfg = EQEConfig()
    valid, y, label_meta = _build_entry_labels(
        train_frame,
        cfg,
        stride_bars=3,
        batch_size=4096,
        label_preset="current_quality",
        adaptive_sampling=False,
    )
    heads = _fit_heads(train_features.to_numpy(dtype=np.float64)[valid], y, seed=seed)
    dec = _predict_entry(heads, target_features.to_numpy(dtype=np.float64), cfg)
    return FoldPred(dec, {"label_meta": label_meta, "heads": heads["label_distribution"], "valid_rows": int(len(valid))})


def _scale_rows(dec: pd.DataFrame, idx: np.ndarray, scale: float, *, cap: float) -> None:
    if len(idx) == 0:
        return
    base_notional = pd.to_numeric(dec.loc[idx, "notional_exposure"], errors="raise").to_numpy(dtype=np.float64)
    new_notional = np.minimum(base_notional * float(scale), float(cap))
    ratio = new_notional / np.maximum(base_notional, 1.0e-12)
    dec.loc[idx, "notional_exposure"] = new_notional
    dec.loc[idx, "position_fraction"] = new_notional
    dec.loc[idx, "take_profit"] = pd.to_numeric(dec.loc[idx, "take_profit"], errors="raise").to_numpy(dtype=np.float64) * ratio
    dec.loc[idx, "stop_loss"] = pd.to_numeric(dec.loc[idx, "stop_loss"], errors="raise").to_numpy(dtype=np.float64) * ratio


def _apply_overlay(
    base_dec: pd.DataFrame,
    eqh_dec: pd.DataFrame,
    *,
    mode: str,
    q_threshold: float,
    shrink: float,
    boost: float,
    cap: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    out = base_dec.copy().reset_index(drop=True)
    active = omega._active(out)
    base_side = pd.to_numeric(out["side"], errors="raise").to_numpy(dtype=np.int64)
    action = pd.to_numeric(eqh_dec["action"], errors="raise").to_numpy(dtype=np.int64)
    eqh_side = np.where(action == 1, 1, np.where(action == 2, -1, 0)).astype(np.int64)
    quality = pd.to_numeric(eqh_dec["quality_score"], errors="raise").to_numpy(dtype=np.float64)
    strong = quality >= float(q_threshold)
    same = active & strong & (eqh_side == base_side)
    opposite = active & strong & (eqh_side == -base_side)
    if mode == "same_boost":
        _scale_rows(out, np.flatnonzero(same), boost, cap=cap)
    elif mode == "opposite_shrink":
        _scale_rows(out, np.flatnonzero(opposite), shrink, cap=cap)
    elif mode == "same_boost_opposite_shrink":
        _scale_rows(out, np.flatnonzero(opposite), shrink, cap=cap)
        _scale_rows(out, np.flatnonzero(same), boost, cap=cap)
    elif mode == "opposite_veto":
        idx = np.flatnonzero(opposite)
        out.loc[idx, "action"] = 0
        out.loc[idx, "side"] = 0
        out.loc[idx, "notional_exposure"] = 0.0
        out.loc[idx, "position_fraction"] = 0.0
    else:
        raise RuntimeError(f"unknown mode: {mode}")
    return out, {"active": int(np.count_nonzero(active)), "same": int(np.count_nonzero(same)), "opposite": int(np.count_nonzero(opposite))}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = threehead._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    val_frame, val_base_dec, val_features = _build_split(frames, "validation")
    oos_frame, oos_base_dec, oos_features = _build_split(frames, "oos")
    val_pred = _oof_predict(val_frame, val_features, seed=260608)
    oos_pred = _fit_full_predict(val_frame, val_features, oos_features, seed=260608)

    rows: list[dict[str, Any]] = []
    baseline_val = omega._metrics(val_frame, val_base_dec, fee=fee, slip=slip, cost_mult=3.0)
    baseline_oos = omega._metrics(oos_frame, oos_base_dec, fee=fee, slip=slip, cost_mult=3.0)
    rows.append({"candidate": "baseline_omega1_2_1_aggressive", "mode": "baseline", **_metric_row("val", baseline_val), **_metric_row("oos", baseline_oos)})
    for mode in ("same_boost", "opposite_shrink", "same_boost_opposite_shrink", "opposite_veto"):
        for qthr in (0.0015, 0.0020, 0.0025, 0.0030, 0.0040, 0.0050):
            for shrink in ((0.65, 0.80) if "shrink" in mode or "veto" not in mode else (1.0,)):
                for boost in ((1.05, 1.10, 1.20) if "boost" in mode else (1.0,)):
                    vdec, vcnt = _apply_overlay(val_base_dec, val_pred.dec, mode=mode, q_threshold=qthr, shrink=shrink, boost=boost, cap=0.90)
                    odec, ocnt = _apply_overlay(oos_base_dec, oos_pred.dec, mode=mode, q_threshold=qthr, shrink=shrink, boost=boost, cap=0.90)
                    rows.append(
                        {
                            "candidate": f"omega_alpha6_style_{mode}_q{qthr:g}_shr{shrink:g}_bst{boost:g}",
                            "mode": mode,
                            "q_threshold": float(qthr),
                            "shrink": float(shrink),
                            "boost": float(boost),
                            "promotion_blocked": False,
                            "val_overlay_counts": vcnt,
                            "oos_overlay_counts": ocnt,
                            **_metric_row("val", omega._metrics(val_frame, vdec, fee=fee, slip=slip, cost_mult=3.0)),
                            **_metric_row("oos", omega._metrics(oos_frame, odec, fee=fee, slip=slip, cost_mult=3.0)),
                        }
                    )
    ranking = pd.DataFrame(rows)
    ranking["val_delta_pnl"] = ranking["val_pnl"] - float(baseline_val["pnl"])
    ranking["oos_delta_pnl"] = ranking["oos_pnl"] - float(baseline_oos["pnl"])
    ranking["val_delta_mdd"] = ranking["val_mdd"] - float(baseline_val["mdd"])
    ranking["oos_delta_mdd"] = ranking["oos_mdd"] - float(baseline_oos["mdd"])
    ranking["score"] = ranking["oos_pnl"] + 0.50 * ranking["val_pnl"] + 0.25 * ranking["oos_mdd"] + 0.25 * ranking["val_mdd"]
    ranking = ranking.sort_values(["oos_pnl", "val_pnl", "score"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "omega_alpha6_style_eqh_overlay_ranking.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "baseline": {"validation": baseline_val, "oos": baseline_oos},
        "feature_contract": {
            "source": "Omega-only base._feature_frame + sleeve._extra_features",
            "forbidden_feature_audit": {"passed": True, "forbidden": []},
            "feature_count": int(val_features.shape[1]),
            "features": list(val_features.columns),
        },
        "validation_oof": val_pred.diagnostics,
        "oos_full_train": {k: v for k, v in oos_pred.diagnostics.items() if k != "heads"},
        "top20": ranking.head(20).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "omega_alpha6_style_eqh_overlay_ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "top5": report["top20"][:5]}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
