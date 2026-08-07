#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega1_2_true3head_overlays_20260604 as overlay  # noqa: E402
import train_eval_omega1_2_1_cash_fallback_sleeve_20260606 as sleeve  # noqa: E402
import train_eval_omega1_2_1_exposure_selector_20260606 as base_features  # noqa: E402
import train_eval_omega1_2_1_full_retrain_cash_alpha43_20260608 as full_parent  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402


MODEL_ID = "omega3_feature_quantile_barrier_loop9_20260619"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
TRAIN_PRED_CACHE = ROOT / "tmp/causal_regen_20260516/omega3_entry_veto_exit_hazard_loop6_20260619/train_predictions_2025_true3head.csv"
PARENT_BUNDLE = full_parent.PARENT_DIR / "true_3head_tabm_bundle.pt"
PREFIX_TRAIN_VAL = "omega1_regime3_expertdq_oof_"
PREFIX_OOS = "omega1_regime3_expertdq_"
CURRENT = {
    "validation": {"pnl": 100.542729421, "mdd": -10.677653, "trades": 33, "wr": 0.636364},
    "oos": {"pnl": 72.760041481, "mdd": -8.108171, "trades": 18, "wr": 0.722222},
}


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


def _predict_train(frame: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    if TRAIN_PRED_CACHE.exists():
        return pd.read_csv(TRAIN_PRED_CACHE, parse_dates=["timestamp"])
    bundle = torch.load(PARENT_BUNDLE, map_location=device, weights_only=False)
    x = threehead._base_input(frame, list(bundle["base_cols"]))
    preds = {expert: threehead._predict_payload(bundle["models"][expert], x, device=device) for expert in hard.EXPERT_NAMES}
    route = hard._route_id(frame)
    direction = threehead._routed(preds, route, "direction", 3)
    quality = threehead._routed(preds, route, "quality", 3)
    src = threehead._prediction_output(frame, direction, quality, threshold=0.0, prefix=PREFIX_TRAIN_VAL.rstrip("_"))
    TRAIN_PRED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    src.to_csv(TRAIN_PRED_CACHE, index=False)
    return src


def _split(frames: dict[str, pd.DataFrame], name: str, device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if name == "train":
        frame = frames["train_raw"].reset_index(drop=True)
        pred = _predict_train(frame, device)
        prefix = PREFIX_TRAIN_VAL
        oof = True
    elif name == "validation":
        frame = frames["val_raw"].reset_index(drop=True)
        pred = pd.read_csv(full_parent.PARENT_DIR / "validation_predictions_2025_true3head.csv", parse_dates=["timestamp"])
        prefix = PREFIX_TRAIN_VAL
        oof = True
    elif name == "oos":
        frame = frames["oos_raw"].reset_index(drop=True)
        pred = pd.read_csv(full_parent.PARENT_DIR / "oos_predictions_2026_true3head.csv", parse_dates=["timestamp"])
        prefix = PREFIX_OOS
        oof = False
    else:
        raise RuntimeError(f"unknown split: {name}")
    src = frame[["timestamp"]].merge(pred, on="timestamp", how="left", validate="one_to_one")
    if src.isna().any().any():
        bad = src.loc[src.isna().any(axis=1), "timestamp"].head(10).tolist()
        raise RuntimeError(f"{name} prediction alignment produced NaN: {bad}")
    dec0 = overlay._build_dec(src, prefix, oof=oof)
    x = sleeve._extra_features(base_features._feature_frame(frame, src, dec0, prefix), dec0)
    bad_cols = [
        c
        for c in x.columns
        if c == "tp_sl_action_score" or c.startswith("clean_regime4_") or c.startswith("regime4_pred_") or c.startswith("teacher_")
    ]
    if bad_cols:
        raise RuntimeError(f"{name}: forbidden features: {bad_cols[:20]}")
    return frame, src, dec0, x.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _path_labels(frame: pd.DataFrame, dec0: pd.DataFrame, *, max_rows: int, seed: int, horizon: int) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "close")}
    active = np.flatnonzero(omega._active(dec0))
    if len(active) > max_rows:
        active = np.sort(np.random.default_rng(seed).choice(active, size=max_rows, replace=False))
    rows: list[dict[str, Any]] = []
    for k, i in enumerate(active):
        if k % 1000 == 0:
            print(json.dumps({"stage": "quantile_path_labels", "seen": int(k), "total": int(len(active))}), flush=True)
        side = int(dec0.iloc[int(i)]["side"])
        entry_i = min(int(i) + 1, len(frame) - 1)
        end = min(entry_i + int(horizon), len(frame) - 2)
        entry = float(arrays["open"][entry_i])
        if side == 0 or entry <= 0.0 or end <= entry_i:
            continue
        px = arrays["close"][entry_i : end + 1]
        raw = (px - entry) / entry if side > 0 else (entry - px) / entry
        mfe = float(np.max(raw))
        mae = float(abs(np.min(raw)))
        end_ret = float(raw[-1])
        rows.append(
            {
                "tp_mfe": float(np.clip(mfe, 0.004, 0.14)),
                "sl_mae": float(np.clip(mae, 0.004, 0.10)),
                "end_ret": end_ret,
                "good_path": int((mfe > mae * 0.85) and (mfe > 0.006)),
            }
        )
    labels = pd.DataFrame(rows)
    idx = active[: len(labels)]
    diag = {
        "rows": int(len(labels)),
        "tp_median": float(labels["tp_mfe"].median()) if len(labels) else 0.0,
        "sl_median": float(labels["sl_mae"].median()) if len(labels) else 0.0,
        "good_rate": float(labels["good_path"].mean()) if len(labels) else 0.0,
    }
    return idx.astype(np.int64), labels, diag


def _fit_quantile(x: pd.DataFrame, idx: np.ndarray, y: np.ndarray, *, quantile: float, seed: int) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=float(quantile),
        max_iter=180,
        learning_rate=0.04,
        max_leaf_nodes=15,
        min_samples_leaf=35,
        l2_regularization=0.5,
        random_state=seed,
    )
    model.fit(x.iloc[idx].to_numpy(dtype=np.float64), np.asarray(y, dtype=np.float64))
    return model


def _apply_risk(
    dec0: pd.DataFrame,
    x: pd.DataFrame,
    tp_model: HistGradientBoostingRegressor,
    sl_model: HistGradientBoostingRegressor,
    *,
    tp_scale: float,
    sl_scale: float,
    account_sl_target: float,
    notional_cap: float,
    notional_floor: float,
    quality_power: float,
    quality_min: float,
) -> pd.DataFrame:
    out = dec0.copy().reset_index(drop=True)
    q_all = pd.to_numeric(out["quality_score"], errors="raise").to_numpy(dtype=np.float64)
    low_q = omega._active(out) & (q_all < float(quality_min))
    if bool(low_q.any()):
        out.loc[low_q, ["action", "side", "notional_exposure", "position_fraction", "take_profit", "stop_loss"]] = [omega.ACTION_CASH, 0, 0.0, 0.0, 0.0, 0.0]
    active = np.flatnonzero(omega._active(out))
    if len(active) == 0:
        return out
    xx = x.to_numpy(dtype=np.float64)
    tp_pred = np.clip(tp_model.predict(xx), 0.004, 0.14) * float(tp_scale)
    sl_pred = np.clip(sl_model.predict(xx), 0.004, 0.10) * float(sl_scale)
    q = q_all
    quality_mult = np.clip(0.55 + np.power(np.clip(q, 0.0, 1.0), float(quality_power)), 0.35, 1.55)
    sl_move = np.maximum(sl_pred[active], 1e-6)
    n = (float(account_sl_target) / sl_move) * quality_mult[active]
    n = np.clip(n, float(notional_floor), float(notional_cap))
    out.loc[active, "notional_exposure"] = n
    out.loc[active, "position_fraction"] = n / 2.0
    out.loc[active, "leverage"] = 2.0
    out.loc[active, "take_profit"] = np.maximum(tp_pred[active], 0.004) * n
    out.loc[active, "stop_loss"] = np.maximum(sl_move, 0.004) * n
    out.loc[active, "max_hold_bars"] = 0
    out.loc[active, "cooldown_bars"] = 0
    return out


def _row(name: str, vm: dict[str, Any], om: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    row = {"candidate": name, **cfg}
    for prefix, m in (("val", vm), ("oos", om)):
        row[f"{prefix}_pnl"] = float(m["pnl"])
        row[f"{prefix}_mdd"] = float(m["mdd"])
        row[f"{prefix}_wr"] = float(m["wr"])
        row[f"{prefix}_trades"] = int(m["trades"])
        row[f"{prefix}_avg_notional"] = float(m.get("avg_notional", 0.0))
        row[f"{prefix}_reasons"] = dict(m.get("exit_reasons", {}))
    row["val_delta_vs_current"] = float(row["val_pnl"] - CURRENT["validation"]["pnl"])
    row["oos_delta_vs_current"] = float(row["oos_pnl"] - CURRENT["oos"]["pnl"])
    row["validation_only_score"] = float(row["val_pnl"] + row["val_mdd"] - 0.05 * max(0, row["val_trades"] - CURRENT["validation"]["trades"]))
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-label-rows", type=int, default=7000)
    ap.add_argument("--horizon", type=int, default=384)
    ap.add_argument("--seed", type=int, default=260629)
    ap.add_argument("--fast-grid", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = threehead._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    train_frame, _train_src, train_dec0, x_train = _split(frames, "train", device)
    val_frame, _val_src, val_dec0, x_val = _split(frames, "validation", device)
    oos_frame, _oos_src, oos_dec0, x_oos = _split(frames, "oos", device)

    idx, labels, label_diag = _path_labels(train_frame, train_dec0, max_rows=int(args.max_label_rows), seed=int(args.seed), horizon=int(args.horizon))
    if len(labels) < 100:
        raise RuntimeError("too few path labels")
    tp_models = {
        q: _fit_quantile(x_train, idx, labels["tp_mfe"].to_numpy(dtype=np.float64), quantile=q, seed=int(args.seed + int(q * 1000)))
        for q in ((0.45, 0.60) if bool(args.fast_grid) else (0.35, 0.45, 0.60, 0.75))
    }
    sl_models = {
        q: _fit_quantile(x_train, idx, labels["sl_mae"].to_numpy(dtype=np.float64), quantile=q, seed=int(args.seed + 100 + int(q * 1000)))
        for q in ((0.55, 0.70) if bool(args.fast_grid) else (0.45, 0.55, 0.70, 0.85))
    }
    rows: list[dict[str, Any]] = []
    tp_scale_grid = (1.0,) if bool(args.fast_grid) else (0.8, 1.0, 1.2)
    sl_scale_grid = (1.0,) if bool(args.fast_grid) else (0.8, 1.0, 1.2)
    acct_sl_grid = (0.014, 0.022) if bool(args.fast_grid) else (0.010, 0.014, 0.018, 0.022, 0.028)
    cap_grid = (1.2, 1.8) if bool(args.fast_grid) else (0.9, 1.2, 1.8, 2.2)
    quality_grid = (0.75,) if bool(args.fast_grid) else (0.5, 0.75, 1.0)
    quality_min_grid = (0.0, 0.62, 0.68, 0.74) if bool(args.fast_grid) else (0.0, 0.60, 0.64, 0.68, 0.72, 0.76)
    for tq, tm in tp_models.items():
        for sq, sm in sl_models.items():
            for tp_scale in tp_scale_grid:
                for sl_scale in sl_scale_grid:
                    for account_sl in acct_sl_grid:
                        for cap in cap_grid:
                            for qp in quality_grid:
                                for qmin in quality_min_grid:
                                    cfg = {
                                        "tp_quantile": float(tq),
                                        "sl_quantile": float(sq),
                                        "tp_scale": float(tp_scale),
                                        "sl_scale": float(sl_scale),
                                        "account_sl_target": float(account_sl),
                                        "notional_cap": float(cap),
                                        "quality_power": float(qp),
                                        "quality_min": float(qmin),
                                    }
                                    vd = _apply_risk(val_dec0, x_val, tm, sm, tp_scale=tp_scale, sl_scale=sl_scale, account_sl_target=account_sl, notional_cap=cap, notional_floor=0.2, quality_power=qp, quality_min=qmin)
                                    od = _apply_risk(oos_dec0, x_oos, tm, sm, tp_scale=tp_scale, sl_scale=sl_scale, account_sl_target=account_sl, notional_cap=cap, notional_floor=0.2, quality_power=qp, quality_min=qmin)
                                    vm = omega._metrics(val_frame, vd, fee=fee, slip=slip, cost_mult=3.0)
                                    om = omega._metrics(oos_frame, od, fee=fee, slip=slip, cost_mult=3.0)
                                    name = f"tq{tq:g}_sq{sq:g}_tps{tp_scale:g}_sls{sl_scale:g}_asl{account_sl:g}_cap{cap:g}_qp{qp:g}_qmin{qmin:g}"
                                    rows.append(_row(name, vm, om, cfg))
    ranking = pd.DataFrame(rows).sort_values(["validation_only_score", "val_pnl"], ascending=[False, False]).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "ranking.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "design": "Feature-conditioned quantile risk layer. Train split forward MFE/MAE labels train tp_price_move and sl_price_move quantile regressors; no BASE_TP/BASE_SL constants are used. Account stop-risk target converts learned sl_price_move into notional.",
        "comparison_baseline": CURRENT,
        "label_diag": label_diag,
        "selected_by_validation": ranking.iloc[0].to_dict(),
        "best_oos_diagnostic": ranking.sort_values(["oos_pnl", "val_pnl"], ascending=[False, False]).iloc[0].to_dict(),
        "top": ranking.head(40).to_dict(orient="records"),
        "artifacts": {"out_dir": str(OUT_DIR), "ranking": str(OUT_DIR / "ranking.csv")},
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default))
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "selected": report["selected_by_validation"], "best_oos": report["best_oos_diagnostic"]}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
