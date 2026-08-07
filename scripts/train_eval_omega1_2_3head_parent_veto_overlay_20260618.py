#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega1_2_true3head_overlays_20260604 as overlay  # noqa: E402
import train_eval_omega1_2_1_cash_fallback_sleeve_20260606 as sleeve  # noqa: E402
import train_eval_omega1_2_1_exposure_selector_20260606 as base_features  # noqa: E402
import train_eval_omega1_2_1_full_retrain_cash_alpha43_20260608 as full_parent  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_3head_parent_veto_overlay_20260618"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
PREFIX_VAL = "omega1_regime3_expertdq_oof_"
PREFIX_OOS = "omega1_regime3_expertdq_"
VETO_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)


@dataclass(frozen=True)
class VetoCfg:
    name: str
    positive_mode: str
    min_net: float
    block_stop_loss: bool


VETO_CFGS = (
    VetoCfg("net_positive", "net", 0.0, False),
    VetoCfg("net_positive_no_stop", "net", 0.0, True),
    VetoCfg("net_001_no_stop", "net", 0.001, True),
)


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _align(frame: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    out = frame[["timestamp"]].merge(pred, on="timestamp", how="left", validate="one_to_one")
    if out.isna().any().any():
        bad = out.loc[out.isna().any(axis=1), "timestamp"].head(10).tolist()
        raise RuntimeError(f"prediction alignment produced NaN: {bad}")
    return out


def _apply_overlay_risk(dec: pd.DataFrame) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active = omega._active(out)
    for expert, scale in overlay.SCALE_MAP.items():
        key = "chop_expert" if expert == "chop" else expert
        mask = active & out["router_expert"].astype(str).eq(key)
        ratio = float(scale) / float(overlay.BASE_SCALES[key])
        out.loc[mask, "notional_exposure"] = pd.to_numeric(out.loc[mask, "notional_exposure"], errors="raise") * ratio
        out.loc[mask, "position_fraction"] = pd.to_numeric(out.loc[mask, "position_fraction"], errors="raise") * ratio
    active = omega._active(out)
    out.loc[active, "take_profit"] = float(overlay.TP)
    out.loc[active, "stop_loss"] = float(overlay.SL)
    out.loc[active, "max_hold_bars"] = 0
    out.loc[active, "cooldown_bars"] = 0
    return out


def _direction_decision(src: pd.DataFrame, prefix: str, *, oof: bool) -> pd.DataFrame:
    work = src.copy()
    work[f"{prefix}final_action"] = pd.to_numeric(work[f"{prefix}dir_action"], errors="raise").to_numpy(dtype=np.int64)
    return _apply_overlay_risk(omega._to_fixed_decisions(work, oof=oof))


def _current_decision(src: pd.DataFrame, prefix: str, *, oof: bool) -> pd.DataFrame:
    return overlay._build_dec(src, prefix, oof=oof)


def _feature_frame(frame: pd.DataFrame, src: pd.DataFrame, dec: pd.DataFrame, prefix: str) -> pd.DataFrame:
    features = sleeve._extra_features(base_features._feature_frame(frame, src, dec, prefix), dec)
    bad = [
        c
        for c in features.columns
        if c == "tp_sl_action_score" or c.startswith("clean_regime4_") or c.startswith("regime4_pred_") or c.startswith("teacher_")
    ]
    if bad:
        raise RuntimeError(f"forbidden veto feature columns: {bad[:20]}")
    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _load_split(frames: dict[str, pd.DataFrame], split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str, bool]:
    if split == "validation":
        frame = frames["val_raw"].reset_index(drop=True)
        pred = pd.read_csv(full_parent.PARENT_DIR / "validation_predictions_2025_true3head.csv", parse_dates=["timestamp"])
        prefix = PREFIX_VAL
        oof = True
    elif split == "oos":
        frame = frames["oos_raw"].reset_index(drop=True)
        pred = pd.read_csv(full_parent.PARENT_DIR / "oos_predictions_2026_true3head.csv", parse_dates=["timestamp"])
        prefix = PREFIX_OOS
        oof = False
    else:
        raise RuntimeError(f"unknown split: {split}")
    src = _align(frame, pred)
    current_dec = sleeve._apply_aggressive(_current_decision(src, prefix, oof=oof))
    direction_dec = sleeve._apply_aggressive(_direction_decision(src, prefix, oof=oof))
    features = _feature_frame(frame, src, direction_dec, prefix)
    return frame, src, current_dec, direction_dec, features, prefix, oof


def _label_candidates(frame: pd.DataFrame, dec: pd.DataFrame, *, fee: float, slip: float, cost_mult: float) -> pd.DataFrame:
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    rows: list[dict[str, Any]] = []
    active_idx = np.flatnonzero(omega._active(dec))
    for k, i in enumerate(active_idx):
        if k % 1000 == 0:
            print(json.dumps({"stage": "veto_labels", "row": int(k), "total": int(len(active_idx))}, ensure_ascii=True), flush=True)
        if i >= len(frame) - 3:
            continue
        score, meta = omega._simulate_trade(frame, arrays, int(i), dec.iloc[int(i)], fee=fee, slip=slip, cost_mult=cost_mult)
        if int(meta.get("active", 0)) != 1:
            continue
        rows.append(
            {
                "i": int(i),
                "score": float(score),
                "net": float(meta.get("net", 0.0)),
                "win": int(meta.get("win", 0)),
                "exit_reason": str(meta.get("exit_reason", "")),
                "mfe": float(meta.get("mfe", 0.0)),
                "mae": float(meta.get("mae", 0.0)),
            }
        )
    if not rows:
        raise RuntimeError("no veto labels generated")
    return pd.DataFrame(rows)


def _target(labels: pd.DataFrame, cfg: VetoCfg) -> np.ndarray:
    y = (labels["net"].to_numpy(dtype=np.float64) > float(cfg.min_net)).astype(np.int64)
    if cfg.block_stop_loss:
        stop = labels["exit_reason"].astype(str).eq("stop_loss").to_numpy()
        y[stop] = 0
    if len(np.unique(y)) < 2:
        raise RuntimeError(f"veto target is single-class for {cfg.name}")
    return y


def _chron_folds(idx: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    n = len(idx)
    for train_frac, end_frac in ((0.35, 0.50), (0.50, 0.65), (0.65, 0.80), (0.80, 1.00)):
        train_end = int(n * train_frac)
        val_end = int(n * end_frac)
        if train_end >= 100 and val_end > train_end:
            folds.append((idx[:train_end], idx[train_end:val_end]))
    return folds


def _fit_predict_veto(
    x_val: pd.DataFrame,
    x_oos: pd.DataFrame,
    labels: pd.DataFrame,
    y: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], HistGradientBoostingClassifier]:
    idx = labels["i"].to_numpy(dtype=np.int64)
    val_pass = np.zeros(len(x_val), dtype=np.float64)
    folds_meta: list[dict[str, Any]] = []
    for fold_id, (tr_pos, va_pos) in enumerate(_chron_folds(np.arange(len(idx)))):
        tr_idx = idx[tr_pos]
        va_idx = idx[va_pos]
        model = HistGradientBoostingClassifier(
            max_iter=160,
            learning_rate=0.035,
            max_leaf_nodes=9,
            l2_regularization=2.0,
            random_state=int(seed + fold_id * 17),
        )
        model.fit(x_val.iloc[tr_idx].to_numpy(dtype=np.float64), y[tr_pos])
        classes = list(model.classes_)
        if 1 not in classes:
            raise RuntimeError("veto OOF fold is missing positive class")
        pos_col = classes.index(1)
        val_pass[va_idx] = model.predict_proba(x_val.iloc[va_idx].to_numpy(dtype=np.float64))[:, pos_col]
        folds_meta.append({"fold": int(fold_id), "train_rows": int(len(tr_idx)), "val_rows": int(len(va_idx)), "positive_rate": float(y[tr_pos].mean())})
    final = HistGradientBoostingClassifier(
        max_iter=160,
        learning_rate=0.035,
        max_leaf_nodes=9,
        l2_regularization=2.0,
        random_state=int(seed + 999),
    )
    final.fit(x_val.iloc[idx].to_numpy(dtype=np.float64), y)
    classes = list(final.classes_)
    if 1 not in classes:
        raise RuntimeError("final veto model is missing positive class")
    oos_pass = final.predict_proba(x_oos.to_numpy(dtype=np.float64))[:, classes.index(1)]
    diag = {"label_rows": int(len(idx)), "positive_rate": float(y.mean()), "folds": folds_meta}
    return val_pass, oos_pass.astype(np.float64), diag, final


def _apply_veto(dec: pd.DataFrame, pass_prob: np.ndarray, threshold: float) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active = omega._active(out)
    keep = active & (np.asarray(pass_prob, dtype=np.float64) >= float(threshold))
    drop = active & ~keep
    out.loc[drop, "action"] = omega.ACTION_CASH
    out.loc[drop, "side"] = 0
    out.loc[drop, "notional_exposure"] = 0.0
    out.loc[drop, "position_fraction"] = 0.0
    out.loc[drop, "take_profit"] = 0.0
    out.loc[drop, "stop_loss"] = 0.0
    out.loc[drop, "max_hold_bars"] = 0
    out.loc[drop, "cooldown_bars"] = 0
    out["veto_pass_prob"] = np.asarray(pass_prob, dtype=np.float64)
    out["veto_threshold"] = float(threshold)
    return out


def _metric_row(candidate: str, cfg: VetoCfg | None, threshold: float | None, val_m: dict[str, Any], oos_m: dict[str, Any], base_val: dict[str, Any], base_oos: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "candidate": candidate,
        "veto_cfg": None if cfg is None else cfg.name,
        "veto_threshold": None if threshold is None else float(threshold),
    }
    if cfg is not None:
        row.update({f"cfg_{k}": v for k, v in asdict(cfg).items()})
    row.update(sleeve._metric_row("val", {**val_m, "primary_entries": val_m["long_entries"] + val_m["short_entries"], "fallback_entries": 0, "primary_takeovers": 0}))
    row.update(sleeve._metric_row("oos", {**oos_m, "primary_entries": oos_m["long_entries"] + oos_m["short_entries"], "fallback_entries": 0, "primary_takeovers": 0}))
    row["val_delta_vs_current"] = float(row["val_pnl"] - float(base_val["pnl"]))
    row["oos_delta_vs_current"] = float(row["oos_pnl"] - float(base_oos["pnl"]))
    row["val_stop_loss"] = int(row["val_reasons"].get("stop_loss", 0)) if isinstance(row["val_reasons"], dict) else 0
    row["val_take_profit"] = int(row["val_reasons"].get("take_profit", 0)) if isinstance(row["val_reasons"], dict) else 0
    row["selection_score_val_only"] = (
        row["val_delta_vs_current"]
        + 10.0 * float(row["val_wr"])
        + 0.25 * float(row["val_mdd"])
        - 0.75 * float(row["val_stop_loss"])
        - 0.05 * max(0.0, float(row["val_trades"]) - 80.0)
    )
    return row


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not full_parent.PARENT_DIR.exists():
        raise RuntimeError(f"missing parent artifact: {full_parent.PARENT_DIR}")
    print(json.dumps({"stage": "build_payloads", "model_id": MODEL_ID}, ensure_ascii=True), flush=True)
    frames = threehead._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    val_frame, val_src, current_val_dec, raw_val_dec, val_x, _prefix_val, _oof_val = _load_split(frames, "validation")
    oos_frame, oos_src, current_oos_dec, raw_oos_dec, oos_x, _prefix_oos, _oof_oos = _load_split(frames, "oos")
    if list(val_x.columns) != list(oos_x.columns):
        raise RuntimeError("veto feature column mismatch")

    current_val_m = omega._metrics(val_frame, current_val_dec, fee=fee, slip=slip, cost_mult=3.0)
    current_oos_m = omega._metrics(oos_frame, current_oos_dec, fee=fee, slip=slip, cost_mult=3.0)
    raw_val_m = omega._metrics(val_frame, raw_val_dec, fee=fee, slip=slip, cost_mult=3.0)
    raw_oos_m = omega._metrics(oos_frame, raw_oos_dec, fee=fee, slip=slip, cost_mult=3.0)

    print(json.dumps({"stage": "label_veto_candidates"}, ensure_ascii=True), flush=True)
    labels = _label_candidates(val_frame, raw_val_dec, fee=fee, slip=slip, cost_mult=3.0)
    rows: list[dict[str, Any]] = [
        _metric_row("current_quality_gate_parent", None, None, current_val_m, current_oos_m, current_val_m, current_oos_m),
        _metric_row("raw_direction_no_quality_gate", None, None, raw_val_m, raw_oos_m, current_val_m, current_oos_m),
    ]
    diagnostics: dict[str, Any] = {
        "labels": {
            "rows": int(len(labels)),
            "net_positive_rate": float((labels["net"].to_numpy(dtype=np.float64) > 0.0).mean()),
            "exit_reason_counts": labels["exit_reason"].value_counts().sort_index().to_dict(),
        },
        "feature_count": int(val_x.shape[1]),
        "features": list(val_x.columns),
    }
    models: dict[str, Any] = {}
    for cfg_id, cfg in enumerate(VETO_CFGS):
        print(json.dumps({"stage": "fit_veto", "cfg": cfg.name}, ensure_ascii=True), flush=True)
        y = _target(labels, cfg)
        val_pass, oos_pass, diag, model = _fit_predict_veto(val_x, oos_x, labels, y, seed=618300 + cfg_id * 100)
        diagnostics[cfg.name] = diag
        models[cfg.name] = model
        for threshold in VETO_THRESHOLDS:
            val_dec = _apply_veto(raw_val_dec, val_pass, threshold)
            oos_dec = _apply_veto(raw_oos_dec, oos_pass, threshold)
            val_m = omega._metrics(val_frame, val_dec, fee=fee, slip=slip, cost_mult=3.0)
            oos_m = omega._metrics(oos_frame, oos_dec, fee=fee, slip=slip, cost_mult=3.0)
            rows.append(_metric_row(f"{cfg.name}_thr{threshold:.2f}".replace(".", "p"), cfg, threshold, val_m, oos_m, current_val_m, current_oos_m))

    ranking = pd.DataFrame(rows).sort_values(["selection_score_val_only", "val_delta_vs_current", "val_pnl"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "veto_overlay_ranking.csv", index=False)
    selected = ranking[ranking["veto_cfg"].notna()].iloc[0].to_dict()
    best_oos = ranking[ranking["veto_cfg"].notna()].sort_values(["oos_pnl", "oos_delta_vs_current"], ascending=False).iloc[0].to_dict()
    joblib.dump(
        {
            "model_id": MODEL_ID,
            "models": models,
            "feature_cols": list(val_x.columns),
            "veto_cfgs": [asdict(c) for c in VETO_CFGS],
            "diagnostics": diagnostics,
        },
        OUT_DIR / "veto_models.joblib",
    )
    report = {
        "model_id": MODEL_ID,
        "status": "redteam_pass_veto_overlay_eval",
        "method": "Keep existing 3-head parent predictions. Remove the hard quality gate only for candidate generation, train a separate HGB veto model on validation candidate trade outcomes, then select veto threshold on validation only.",
        "parent_dir": str(full_parent.PARENT_DIR),
        "current_quality_gate_parent": {"validation": current_val_m, "oos": current_oos_m},
        "raw_direction_no_quality_gate": {"validation": raw_val_m, "oos": raw_oos_m},
        "selected_by_validation": selected,
        "best_by_oos_diagnostic": best_oos,
        "top20": ranking.head(20).to_dict(orient="records"),
        "diagnostics": diagnostics,
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "veto_overlay_ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
            "models": str(OUT_DIR / "veto_models.joblib"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "selected": selected, "best_oos": best_oos}, indent=2, ensure_ascii=True, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
