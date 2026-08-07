#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_3head_parent_veto_overlay_20260618 as veto  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega1_2_1_cash_fallback_sleeve_20260606 as sleeve  # noqa: E402


MODEL_ID = "omega1_2_multitarget_supervised_gate_20260618"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _chron_folds(n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for train_frac, end_frac in ((0.35, 0.50), (0.50, 0.65), (0.65, 0.80), (0.80, 1.00)):
        train_end = int(n * train_frac)
        val_end = int(n * end_frac)
        if train_end >= 100 and val_end > train_end:
            folds.append((np.arange(train_end), np.arange(train_end, val_end)))
    return folds


def _clf(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=180, learning_rate=0.035, max_leaf_nodes=9, l2_regularization=2.0, random_state=int(seed))


def _reg(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=180, learning_rate=0.035, max_leaf_nodes=9, l2_regularization=2.0, random_state=int(seed))


def _predict_positive(model: HistGradientBoostingClassifier, x: np.ndarray) -> np.ndarray:
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(x), dtype=np.float64)
    return model.predict_proba(x)[:, classes.index(1)].astype(np.float64)


def _fit_multitarget(
    x_val: pd.DataFrame,
    x_oos: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    cal_q: float,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    labels = labels.sort_values("i").reset_index(drop=True)
    idx = labels["i"].to_numpy(dtype=np.int64)
    y_tp = labels["exit_reason"].astype(str).eq("take_profit").to_numpy(dtype=np.int64)
    y_sl = labels["exit_reason"].astype(str).eq("stop_loss").to_numpy(dtype=np.int64)
    y_net = labels["net"].to_numpy(dtype=np.float64)
    y_mae = labels["mae"].to_numpy(dtype=np.float64)

    val = {
        "p_tp": np.zeros(len(x_val), dtype=np.float64),
        "p_sl": np.ones(len(x_val), dtype=np.float64),
        "net_lb": np.full(len(x_val), -1.0, dtype=np.float64),
        "mae_lb": np.full(len(x_val), -1.0, dtype=np.float64),
    }
    folds_meta: list[dict[str, Any]] = []
    for fold_id, (tr_pos, va_pos) in enumerate(_chron_folds(len(idx))):
        tr_idx = idx[tr_pos]
        va_idx = idx[va_pos]
        x_tr = x_val.iloc[tr_idx].to_numpy(dtype=np.float64)
        x_va = x_val.iloc[va_idx].to_numpy(dtype=np.float64)
        tp_model = _clf(seed + fold_id * 100 + 1).fit(x_tr, y_tp[tr_pos])
        sl_model = _clf(seed + fold_id * 100 + 2).fit(x_tr, y_sl[tr_pos])
        net_model = _reg(seed + fold_id * 100 + 3).fit(x_tr, y_net[tr_pos])
        mae_model = _reg(seed + fold_id * 100 + 4).fit(x_tr, y_mae[tr_pos])
        net_train_pred = net_model.predict(x_tr).astype(np.float64)
        mae_train_pred = mae_model.predict(x_tr).astype(np.float64)
        net_offset = float(np.quantile(np.abs(y_net[tr_pos] - net_train_pred), cal_q))
        mae_offset = float(np.quantile(np.abs(y_mae[tr_pos] - mae_train_pred), cal_q))
        val["p_tp"][va_idx] = _predict_positive(tp_model, x_va)
        val["p_sl"][va_idx] = _predict_positive(sl_model, x_va)
        val["net_lb"][va_idx] = net_model.predict(x_va).astype(np.float64) - net_offset
        val["mae_lb"][va_idx] = mae_model.predict(x_va).astype(np.float64) - mae_offset
        folds_meta.append(
            {
                "fold": int(fold_id),
                "train_rows": int(len(tr_idx)),
                "val_rows": int(len(va_idx)),
                "tp_rate": float(y_tp[tr_pos].mean()),
                "sl_rate": float(y_sl[tr_pos].mean()),
                "net_abs_resid_q": net_offset,
                "mae_abs_resid_q": mae_offset,
            }
        )

    x_train = x_val.iloc[idx].to_numpy(dtype=np.float64)
    x_eval = x_oos.to_numpy(dtype=np.float64)
    final_tp = _clf(seed + 9001).fit(x_train, y_tp)
    final_sl = _clf(seed + 9002).fit(x_train, y_sl)
    final_net = _reg(seed + 9003).fit(x_train, y_net)
    final_mae = _reg(seed + 9004).fit(x_train, y_mae)
    net_offset = float(np.quantile(np.abs(y_net - final_net.predict(x_train)), cal_q))
    mae_offset = float(np.quantile(np.abs(y_mae - final_mae.predict(x_train)), cal_q))
    oos = {
        "p_tp": _predict_positive(final_tp, x_eval),
        "p_sl": _predict_positive(final_sl, x_eval),
        "net_lb": final_net.predict(x_eval).astype(np.float64) - net_offset,
        "mae_lb": final_mae.predict(x_eval).astype(np.float64) - mae_offset,
    }
    diag = {
        "label_rows": int(len(idx)),
        "tp_rate": float(y_tp.mean()),
        "sl_rate": float(y_sl.mean()),
        "net_mean": float(y_net.mean()),
        "mae_mean": float(y_mae.mean()),
        "cal_q": float(cal_q),
        "folds": folds_meta,
        "final_net_abs_resid_q": net_offset,
        "final_mae_abs_resid_q": mae_offset,
        "exit_reason_counts": labels["exit_reason"].value_counts().sort_index().to_dict(),
    }
    models = {"tp": final_tp, "sl": final_sl, "net": final_net, "mae": final_mae}
    return val, oos, diag, models


def _apply_gate(
    dec: pd.DataFrame,
    preds: dict[str, np.ndarray],
    *,
    net_min: float,
    edge_min: float,
    psl_max: float,
    mae_min: float,
) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active = omega._active(out)
    edge = np.asarray(preds["p_tp"], dtype=np.float64) - np.asarray(preds["p_sl"], dtype=np.float64)
    keep = (
        active
        & (np.asarray(preds["net_lb"], dtype=np.float64) > float(net_min))
        & (edge >= float(edge_min))
        & (np.asarray(preds["p_sl"], dtype=np.float64) <= float(psl_max))
        & (np.asarray(preds["mae_lb"], dtype=np.float64) >= float(mae_min))
    )
    drop = active & ~keep
    out.loc[drop, "action"] = omega.ACTION_CASH
    out.loc[drop, "side"] = 0
    out.loc[drop, "notional_exposure"] = 0.0
    out.loc[drop, "position_fraction"] = 0.0
    out.loc[drop, "take_profit"] = 0.0
    out.loc[drop, "stop_loss"] = 0.0
    out.loc[drop, "max_hold_bars"] = 0
    out.loc[drop, "cooldown_bars"] = 0
    out["mt_p_tp"] = np.asarray(preds["p_tp"], dtype=np.float64)
    out["mt_p_sl"] = np.asarray(preds["p_sl"], dtype=np.float64)
    out["mt_net_lb"] = np.asarray(preds["net_lb"], dtype=np.float64)
    out["mt_mae_lb"] = np.asarray(preds["mae_lb"], dtype=np.float64)
    return out


def _metric_row(
    candidate: str,
    params: dict[str, Any],
    val_m: dict[str, Any],
    oos_m: dict[str, Any],
    base_val: dict[str, Any],
    base_oos: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {"candidate": candidate, **params}
    row.update(sleeve._metric_row("val", {**val_m, "primary_entries": val_m["long_entries"] + val_m["short_entries"], "fallback_entries": 0, "primary_takeovers": 0}))
    row.update(sleeve._metric_row("oos", {**oos_m, "primary_entries": oos_m["long_entries"] + oos_m["short_entries"], "fallback_entries": 0, "primary_takeovers": 0}))
    row["val_delta_vs_current"] = float(row["val_pnl"] - float(base_val["pnl"]))
    row["oos_delta_vs_current"] = float(row["oos_pnl"] - float(base_oos["pnl"]))
    val_reasons = row["val_reasons"] if isinstance(row["val_reasons"], dict) else {}
    row["val_stop_loss"] = int(val_reasons.get("stop_loss", 0))
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
    print(json.dumps({"stage": "build_payloads", "model_id": MODEL_ID}, ensure_ascii=True), flush=True)
    frames = threehead._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    val_frame, _val_src, current_val_dec, raw_val_dec, val_x, *_ = veto._load_split(frames, "validation")
    oos_frame, _oos_src, current_oos_dec, raw_oos_dec, oos_x, *_ = veto._load_split(frames, "oos")
    if list(val_x.columns) != list(oos_x.columns):
        raise RuntimeError("feature column mismatch")

    current_val_m = omega._metrics(val_frame, current_val_dec, fee=fee, slip=slip, cost_mult=3.0)
    current_oos_m = omega._metrics(oos_frame, current_oos_dec, fee=fee, slip=slip, cost_mult=3.0)
    raw_val_m = omega._metrics(val_frame, raw_val_dec, fee=fee, slip=slip, cost_mult=3.0)
    raw_oos_m = omega._metrics(oos_frame, raw_oos_dec, fee=fee, slip=slip, cost_mult=3.0)

    print(json.dumps({"stage": "label_candidates"}, ensure_ascii=True), flush=True)
    labels = veto._label_candidates(val_frame, raw_val_dec, fee=fee, slip=slip, cost_mult=3.0)
    val_preds, oos_preds, diag, models = _fit_multitarget(val_x, oos_x, labels, cal_q=0.80, seed=618500)

    rows: list[dict[str, Any]] = [
        _metric_row("current_quality_gate_parent", {"family": "baseline"}, current_val_m, current_oos_m, current_val_m, current_oos_m),
        _metric_row("raw_direction_no_quality_gate", {"family": "baseline"}, raw_val_m, raw_oos_m, current_val_m, current_oos_m),
    ]
    grid = itertools.product(
        (-0.004, -0.002, 0.0, 0.001, 0.002),
        (-0.25, -0.10, 0.0, 0.10, 0.20),
        (0.45, 0.55, 0.65, 0.75),
        (-0.035, -0.025, -0.015),
    )
    for net_min, edge_min, psl_max, mae_min in grid:
        params = {"family": "multitarget_supervised_gate", "net_min": net_min, "edge_min": edge_min, "psl_max": psl_max, "mae_min": mae_min}
        val_dec = _apply_gate(raw_val_dec, val_preds, net_min=net_min, edge_min=edge_min, psl_max=psl_max, mae_min=mae_min)
        oos_dec = _apply_gate(raw_oos_dec, oos_preds, net_min=net_min, edge_min=edge_min, psl_max=psl_max, mae_min=mae_min)
        val_m = omega._metrics(val_frame, val_dec, fee=fee, slip=slip, cost_mult=3.0)
        oos_m = omega._metrics(oos_frame, oos_dec, fee=fee, slip=slip, cost_mult=3.0)
        name = f"mt_net{net_min:.3f}_edge{edge_min:.2f}_psl{psl_max:.2f}_mae{mae_min:.3f}"
        rows.append(_metric_row(name.replace(".", "p").replace("-", "m"), params, val_m, oos_m, current_val_m, current_oos_m))

    ranking = pd.DataFrame(rows).sort_values(["selection_score_val_only", "val_delta_vs_current", "val_pnl"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "multitarget_gate_ranking.csv", index=False)
    candidates = ranking[ranking["family"].eq("multitarget_supervised_gate")].copy()
    selected = candidates.iloc[0].to_dict()
    best_oos = candidates.sort_values(["oos_pnl", "oos_delta_vs_current"], ascending=False).iloc[0].to_dict()
    joblib.dump(
        {
            "model_id": MODEL_ID,
            "models": models,
            "feature_cols": list(val_x.columns),
            "diagnostics": diag,
        },
        OUT_DIR / "multitarget_gate_models.joblib",
    )
    report = {
        "model_id": MODEL_ID,
        "status": "redteam_pass_multitarget_supervised_gate_eval",
        "method": "Use existing 3-head parent raw direction candidates. Train separate supervised P(TP), P(SL), net lower-bound, and MAE lower-bound models on validation trade simulations; select decision rule by validation only.",
        "current_quality_gate_parent": {"validation": current_val_m, "oos": current_oos_m},
        "raw_direction_no_quality_gate": {"validation": raw_val_m, "oos": raw_oos_m},
        "selected_by_validation": selected,
        "best_by_oos_diagnostic": best_oos,
        "top30": ranking.head(30).to_dict(orient="records"),
        "diagnostics": {"feature_count": int(val_x.shape[1]), "features": list(val_x.columns), "labels": diag},
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "multitarget_gate_ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
            "models": str(OUT_DIR / "multitarget_gate_models.joblib"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "selected": selected, "best_oos": best_oos}, indent=2, ensure_ascii=True, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
