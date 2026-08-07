#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega1_2_true3head_overlays_20260604 as overlay  # noqa: E402
import train_eval_omega1_2_1_exposure_selector_20260606 as exposure  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as th  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_1_learned_risk_selector_20260606"
BASE_DIR = exposure.BASE_DIR
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

AGGR_VAL = {
    "pnl": 100.54272942091158,
    "mdd": -10.677652697162888,
    "wr": 0.6363636363636364,
    "trades": 33,
}
AGGR_OOS = {
    "pnl": 72.76004148106665,
    "mdd": -8.108170708968387,
    "wr": 0.7222222222222222,
    "trades": 18,
}


@dataclass(frozen=True)
class RiskAction:
    name: str
    notional_scale: float
    cap: float
    tp_mult: float
    sl_mult: float


RISK_ACTIONS = [
    RiskAction("parent_base", 1.00, 0.90, 1.00, 1.00),
    RiskAction("balanced_135_cap055", 1.35, 0.55, 1.35, 1.35),
    RiskAction("scale150_cap070", 1.50, 0.70, 1.50, 1.50),
    RiskAction("scale175_cap090", 1.75, 0.90, 1.75, 1.75),
    RiskAction("aggressive_200_cap090", 2.00, 0.90, 2.00, 2.00),
    RiskAction("aggr_tight_sl", 2.00, 0.90, 2.00, 1.50),
    RiskAction("aggr_wide_sl", 2.00, 0.90, 2.00, 2.50),
    RiskAction("aggr_runner_tp", 2.00, 0.90, 2.50, 2.00),
    RiskAction("aggr_runner_wide", 2.00, 0.90, 2.50, 2.50),
    RiskAction("cap070_runner", 2.00, 0.70, 2.20, 2.00),
    RiskAction("cap070_tight", 2.00, 0.70, 2.00, 1.50),
    RiskAction("stable_tp_boost", 1.35, 0.55, 1.75, 1.35),
    RiskAction("stable_sl_wide", 1.35, 0.55, 1.35, 1.75),
    RiskAction("mid_runner", 1.75, 0.90, 2.20, 1.75),
    RiskAction("mid_defensive", 1.75, 0.90, 1.75, 1.35),
    RiskAction("low_risk", 1.00, 0.55, 1.20, 1.00),
]
AGGRESSIVE_ACTION_ID = [a.name for a in RISK_ACTIONS].index("aggressive_200_cap090")


def _json_default(obj: Any) -> Any:
    return exposure._json_default(obj)


def _apply_risk_action(dec: pd.DataFrame, idx: np.ndarray, action_ids: np.ndarray) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    for row_i, action_id in zip(idx, action_ids):
        action = RISK_ACTIONS[int(action_id)]
        base_notional = float(out.loc[int(row_i), "notional_exposure"])
        new_notional = min(base_notional * float(action.notional_scale), float(action.cap))
        out.loc[int(row_i), "notional_exposure"] = new_notional
        out.loc[int(row_i), "position_fraction"] = new_notional
        out.loc[int(row_i), "take_profit"] = float(out.loc[int(row_i), "take_profit"]) * float(action.tp_mult)
        out.loc[int(row_i), "stop_loss"] = float(out.loc[int(row_i), "stop_loss"]) * float(action.sl_mult)
    return out


def _static_action_metrics(frame: pd.DataFrame, dec: pd.DataFrame, active_idx: np.ndarray, *, fee: float, slip: float, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for aid, action in enumerate(RISK_ACTIONS):
        action_ids = np.full(len(active_idx), int(aid), dtype=np.int64)
        cur_dec = _apply_risk_action(dec, active_idx, action_ids)
        metrics = omega._metrics(frame, cur_dec, fee=fee, slip=slip, cost_mult=3.0)
        rows.append(
            {
                "split": split,
                "action_id": int(aid),
                "action_name": action.name,
                "notional_scale": action.notional_scale,
                "cap": action.cap,
                "tp_mult": action.tp_mult,
                "sl_mult": action.sl_mult,
                **exposure._metric_row(split, metrics),
            }
        )
    return rows


def _prospective_exit_features(frame: pd.DataFrame, src: pd.DataFrame, dec: pd.DataFrame, prefix: str, *, device: torch.device) -> pd.DataFrame:
    bundle = torch.load(BASE_DIR / "true_3head_tabm_bundle.pt", map_location="cpu", weights_only=False)
    loaded = th._load_payloads(bundle["models"], device=device)
    x = th._base_input(frame, bundle["base_cols"]).reset_index(drop=True)
    x["pos_side"] = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.float32)
    x["pos_hold_bars"] = 0.0
    x["pos_unrealized"] = 0.0
    x["pos_mfe"] = 0.0
    x["pos_mae"] = 0.0
    x["pos_giveback"] = 0.0
    x["pos_dist_to_tp"] = pd.to_numeric(dec["take_profit"], errors="raise").to_numpy(dtype=np.float32)
    x["pos_dist_to_sl"] = pd.to_numeric(dec["stop_loss"], errors="raise").abs().to_numpy(dtype=np.float32)
    x["pos_notional"] = pd.to_numeric(dec["notional_exposure"], errors="raise").to_numpy(dtype=np.float32)
    x["pos_leverage"] = pd.to_numeric(dec["leverage"], errors="raise").to_numpy(dtype=np.float32)
    x["pos_exposure"] = x["pos_notional"] * x["pos_leverage"]
    x["pos_tp"] = pd.to_numeric(dec["take_profit"], errors="raise").to_numpy(dtype=np.float32)
    x["pos_sl"] = pd.to_numeric(dec["stop_loss"], errors="raise").abs().to_numpy(dtype=np.float32)

    expert = src[f"{prefix}router_expert"].astype(str).replace({"chop_expert": "chop"}).to_numpy()
    probs = np.zeros(len(x), dtype=np.float64)
    for name, (model, scaler) in loaded.items():
        mask = expert == name
        if bool(np.any(mask)):
            probs[mask] = th._predict_loaded_exit(model, scaler, x.loc[mask].reset_index(drop=True), device=device)[:, 1]
    out = pd.DataFrame(index=frame.index)
    out["exit_head_entry_risk"] = probs
    out["exit_head_x_quality"] = probs * pd.to_numeric(src[f"{prefix}quality_for_action"], errors="raise").to_numpy(dtype=np.float64)
    out["exit_head_x_dir_edge"] = probs * pd.to_numeric(src[f"{prefix}dir_side_edge"], errors="raise").abs().to_numpy(dtype=np.float64)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _feature_frame_with_exit(frame: pd.DataFrame, src: pd.DataFrame, dec: pd.DataFrame, prefix: str, *, device: torch.device) -> pd.DataFrame:
    out = exposure._feature_frame(frame, src, dec, prefix)
    exit_feat = _prospective_exit_features(frame, src, dec, prefix, device=device)
    out = pd.concat([out.reset_index(drop=True), exit_feat.reset_index(drop=True)], axis=1)
    bad = [
        c
        for c in out.columns
        if str(c).startswith("clean_regime4_")
        or str(c).startswith("regime4_pred_")
        or str(c).startswith("teacher_")
        or str(c) == "tp_sl_action_score"
    ]
    if bad:
        raise RuntimeError(f"forbidden learned-risk feature columns: {bad}")
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _build_labels(frame: pd.DataFrame, dec: pd.DataFrame, active_idx: np.ndarray, *, fee: float, slip: float) -> tuple[np.ndarray, dict[str, Any]]:
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    y = np.full(len(active_idx), AGGRESSIVE_ACTION_ID, dtype=np.int64)
    reason_counts: dict[str, int] = {}
    best_scores: list[float] = []
    aggr_scores: list[float] = []
    for k, idx in enumerate(active_idx):
        scores: list[float] = []
        for aid, action in enumerate(RISK_ACTIONS):
            row = dec.iloc[int(idx)].copy()
            base_notional = float(row["notional_exposure"])
            new_notional = min(base_notional * float(action.notional_scale), float(action.cap))
            row.loc["notional_exposure"] = new_notional
            row.loc["position_fraction"] = new_notional
            row.loc["take_profit"] = float(row["take_profit"]) * float(action.tp_mult)
            row.loc["stop_loss"] = float(row["stop_loss"]) * float(action.sl_mult)
            score, meta = omega._simulate_trade(frame, arrays, int(idx), row, fee=fee, slip=slip, cost_mult=3.0)
            # Use the Cost3-aware single-trade score returned by the official simulator,
            # with a small tie-break against avoidable risk expansion.
            risk_tiebreak = 0.0015 * max(0.0, new_notional - 0.81) + 0.0010 * max(0.0, float(action.sl_mult) - 2.0)
            scores.append(float(score) - risk_tiebreak)
            reason = str(meta.get("exit_reason", "inactive"))
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        arr = np.asarray(scores, dtype=np.float64)
        y[k] = int(np.argmax(arr))
        best_scores.append(float(np.max(arr)))
        aggr_scores.append(float(arr[AGGRESSIVE_ACTION_ID]))
    return y, {
        "rows": int(len(active_idx)),
        "label_counts": {RISK_ACTIONS[i].name: int(v) for i, v in enumerate(np.bincount(y, minlength=len(RISK_ACTIONS)))},
        "sim_exit_reasons": reason_counts,
        "best_score_mean": float(np.mean(best_scores)) if best_scores else 0.0,
        "aggressive_score_mean": float(np.mean(aggr_scores)) if aggr_scores else 0.0,
        "edge_vs_aggressive_mean": float(np.mean(np.asarray(best_scores) - np.asarray(aggr_scores))) if best_scores else 0.0,
    }


def _make_model(kind: str, seed: int) -> Any:
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=140,
            learning_rate=0.035,
            max_leaf_nodes=7,
            min_samples_leaf=35,
            l2_regularization=1.5,
            random_state=int(seed),
        )
    if kind == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=260,
            max_depth=5,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=int(seed),
            n_jobs=-1,
        )
    raise RuntimeError(f"unknown model kind: {kind}")


def _predict_with_conf(model: Any, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not hasattr(model, "predict_proba"):
        pred = np.asarray(model.predict(x), dtype=np.int64)
        return pred, np.ones(len(pred), dtype=np.float64), np.ones(len(pred), dtype=np.float64)
    probs = np.asarray(model.predict_proba(x), dtype=np.float64)
    classes = np.asarray(model.classes_, dtype=np.int64)
    order = np.argsort(probs, axis=1)[:, ::-1]
    pred = classes[order[:, 0]]
    conf = probs[np.arange(len(probs)), order[:, 0]]
    margin = conf - probs[np.arange(len(probs)), order[:, 1]] if probs.shape[1] > 1 else conf
    return pred.astype(np.int64), conf.astype(np.float64), margin.astype(np.float64)


def _fit_oof_actions(kind: str, x: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    n = len(y)
    pred = np.full(n, AGGRESSIVE_ACTION_ID, dtype=np.int64)
    conf = np.zeros(n, dtype=np.float64)
    margin = np.zeros(n, dtype=np.float64)
    covered = np.zeros(n, dtype=bool)
    folds: list[dict[str, Any]] = []
    for start_frac, end_frac in ((0.50, 0.65), (0.65, 0.80), (0.80, 1.00)):
        train_end = int(n * start_frac)
        val_start = train_end
        val_end = int(n * end_frac)
        if train_end < 50 or val_end <= val_start:
            continue
        model = _make_model(kind, seed + val_start)
        model.fit(x[:train_end], y[:train_end])
        pred_fold, conf_fold, margin_fold = _predict_with_conf(model, x[val_start:val_end])
        pred[val_start:val_end] = pred_fold
        conf[val_start:val_end] = conf_fold
        margin[val_start:val_end] = margin_fold
        covered[val_start:val_end] = True
        folds.append(
            {
                "train_end": int(train_end),
                "val_start": int(val_start),
                "val_end": int(val_end),
                "train_acc": float(np.mean(np.asarray(model.predict(x[:train_end]), dtype=np.int64) == y[:train_end])),
            }
        )
    return pred, conf, margin, {"folds": folds, "oof_rows": int(covered.sum()), "oof_coverage": float(covered.mean())}


def _full_predict(kind: str, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    model = _make_model(kind, seed)
    model.fit(x_train, y_train)
    pred, conf, margin = _predict_with_conf(model, x_test)
    train_pred = np.asarray(model.predict(x_train), dtype=np.int64)
    return pred, conf, margin, {"train_acc": float(np.mean(train_pred == y_train))}


def _action_counts(action_ids: np.ndarray) -> dict[str, int]:
    counts = np.bincount(np.asarray(action_ids, dtype=np.int64), minlength=len(RISK_ACTIONS))
    return {RISK_ACTIONS[i].name: int(v) for i, v in enumerate(counts) if int(v) > 0}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = th._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()

    val_frame, val_src, val_dec, val_prefix = exposure._build_split(frames, "validation")
    oos_frame, oos_src, oos_dec, oos_prefix = exposure._build_split(frames, "oos")
    val_active = np.flatnonzero(omega._active(val_dec))
    oos_active = np.flatnonzero(omega._active(oos_dec))

    val_x_all = _feature_frame_with_exit(val_frame, val_src, val_dec, val_prefix, device=device)
    oos_x_all = _feature_frame_with_exit(oos_frame, oos_src, oos_dec, oos_prefix, device=device)
    if list(val_x_all.columns) != list(oos_x_all.columns):
        raise RuntimeError("learned risk feature contract mismatch between validation and OOS")

    y_val, label_diag = _build_labels(val_frame, val_dec, val_active, fee=fee, slip=slip)
    x_val_active = val_x_all.iloc[val_active].to_numpy(dtype=np.float64)
    x_oos_active = oos_x_all.iloc[oos_active].to_numpy(dtype=np.float64)

    static_rows = []
    static_rows.extend(_static_action_metrics(val_frame, val_dec, val_active, fee=fee, slip=slip, split="val"))
    static_rows.extend(_static_action_metrics(oos_frame, oos_dec, oos_active, fee=fee, slip=slip, split="oos"))
    static_df = pd.DataFrame(static_rows)
    static_df.to_csv(OUT_DIR / "static_risk_action_grid.csv", index=False)

    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "device": str(device),
        "feature_count": int(val_x_all.shape[1]),
        "features": list(val_x_all.columns),
        "val_active_rows": int(len(val_active)),
        "oos_active_rows": int(len(oos_active)),
        "risk_actions": [a.__dict__ for a in RISK_ACTIONS],
        "label_diagnostics": label_diag,
    }
    for kind in ("hgb", "extratrees"):
        val_pred, val_conf, val_margin, oof_diag = _fit_oof_actions(kind, x_val_active, y_val, seed=260606)
        oos_pred, oos_conf, oos_margin, full_diag = _full_predict(kind, x_val_active, y_val, x_oos_active, seed=260606)
        variants: list[tuple[str, np.ndarray, np.ndarray]] = [(kind, val_pred, oos_pred)]
        for min_prob in (0.55, 0.65, 0.75, 0.85, 0.95):
            for min_margin in (0.00, 0.10, 0.20):
                val_cons = np.full_like(val_pred, AGGRESSIVE_ACTION_ID)
                oos_cons = np.full_like(oos_pred, AGGRESSIVE_ACTION_ID)
                val_mask = (val_pred != AGGRESSIVE_ACTION_ID) & (val_conf >= float(min_prob)) & (val_margin >= float(min_margin))
                oos_mask = (oos_pred != AGGRESSIVE_ACTION_ID) & (oos_conf >= float(min_prob)) & (oos_margin >= float(min_margin))
                val_cons[val_mask] = val_pred[val_mask]
                oos_cons[oos_mask] = oos_pred[oos_mask]
                variants.append((f"{kind}_aggressive_default_p{min_prob:.2f}_m{min_margin:.2f}", val_cons, oos_cons))
        for variant_name, val_actions, oos_actions in variants:
            val_sel_dec = _apply_risk_action(val_dec, val_active, val_actions)
            oos_sel_dec = _apply_risk_action(oos_dec, oos_active, oos_actions)
            val_m = omega._metrics(val_frame, val_sel_dec, fee=fee, slip=slip, cost_mult=3.0)
            oos_m = omega._metrics(oos_frame, oos_sel_dec, fee=fee, slip=slip, cost_mult=3.0)
            row = {
                "model_kind": variant_name,
                "val_action_counts": _action_counts(val_actions),
                "oos_action_counts": _action_counts(oos_actions),
                "val_delta_vs_aggressive_pnl": float(val_m["pnl"]) - AGGR_VAL["pnl"],
                "val_delta_vs_aggressive_mdd": float(val_m["mdd"]) - AGGR_VAL["mdd"],
                "oos_delta_vs_aggressive_pnl": float(oos_m["pnl"]) - AGGR_OOS["pnl"],
                "oos_delta_vs_aggressive_mdd": float(oos_m["mdd"]) - AGGR_OOS["mdd"],
                "oof": oof_diag,
                "full_fit": full_diag,
            }
            row.update(exposure._metric_row("val", val_m))
            row.update(exposure._metric_row("oos", oos_m))
            rows.append(row)

    ranking = pd.DataFrame(rows)
    ranking["score"] = ranking["oos_pnl"] + 0.50 * ranking["val_pnl"] + 0.25 * ranking["oos_mdd"] + 0.25 * ranking["val_mdd"]
    ranking = ranking.sort_values(["oos_pnl", "val_pnl"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "learned_risk_selector_ranking.csv", index=False)

    report = {
        "model_id": MODEL_ID,
        "baseline": {
            "name": "omega1_2_1_aggressive_compensated_scale200_cap090",
            "validation": AGGR_VAL,
            "oos": AGGR_OOS,
        },
        "method": "Risk template replacement experiment. The 3-head TabM parent and final action are frozen; the learned selector chooses a curated TP/SL/notional bucket action per active signal. Validation uses expanding OOF selector predictions; OOS uses selector refit on all validation active rows.",
        "accounting": "Official Cost3 maker-limit replay, fee/slippage multiplier 3.0, no legacy aliases or feature compatibility fallback.",
        "diagnostics": diagnostics,
        "ranking": ranking.to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "learned_risk_selector_ranking.csv"),
            "static_risk_action_grid": str(OUT_DIR / "static_risk_action_grid.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "ranking": ranking.to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
