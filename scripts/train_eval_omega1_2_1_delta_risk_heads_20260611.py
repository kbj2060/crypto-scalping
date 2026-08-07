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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega1_2_1_independent_risk_heads_20260611 as indep  # noqa: E402
import train_eval_omega1_2_1_tabm_7head_risk_20260611 as seven  # noqa: E402


MODEL_ID = "omega1_2_1_delta_risk_heads_20260611"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

TP_MULT = np.asarray([0.85, 1.00, 1.15], dtype=np.float64)
SL_MULT = np.asarray([0.90, 1.00, 1.15], dtype=np.float64)
MARGIN_MULT = np.asarray([0.75, 1.00, 1.10], dtype=np.float64)
LEVERAGE_MULT = np.asarray([0.75, 1.00], dtype=np.float64)
MAX_HOLD_BUCKETS = np.asarray([0, 96], dtype=np.int64)


class ConstantClassifier:
    def __init__(self, value: int) -> None:
        self.value = int(value)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.full(len(x), self.value, dtype=np.int64)


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _base_risk_for_row(row: pd.Series) -> tuple[float, float, float, float]:
    expert = str(row.get("router_expert", ""))
    scale = float(indep.OVERLAY_SCALES.get(expert, 0.90))
    base_margin0 = float(indep.BASE_NOTIONAL) * scale
    base_margin = min(base_margin0 * float(indep.COMPENSATED_SCALE), float(indep.MARGIN_CAP))
    ratio = base_margin / max(base_margin0, 1e-12)
    leverage = float(indep.BASE_LEVERAGE)
    tp = float(indep.BASE_TP) * ratio * leverage
    sl = float(indep.BASE_SL) * ratio * leverage
    return tp, sl, base_margin, leverage


def _risk_values(row: pd.Series, ids: tuple[int, int, int, int, int]) -> tuple[float, float, float, float, int]:
    tp0, sl0, margin0, lev0 = _base_risk_for_row(row)
    tp_i, sl_i, margin_i, lev_i, hold_i = ids
    margin = min(float(margin0) * float(MARGIN_MULT[margin_i]), float(indep.MARGIN_CAP))
    lev = min(float(lev0) * float(LEVERAGE_MULT[lev_i]), 3.0)
    return (
        float(tp0) * float(TP_MULT[tp_i]),
        float(sl0) * float(SL_MULT[sl_i]),
        float(margin),
        float(lev),
        int(MAX_HOLD_BUCKETS[hold_i]),
    )


def _delta_risk_labels(
    frame: pd.DataFrame,
    candidate_dec: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    max_rows: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    n = len(frame)
    labels = {
        "tp": np.ones(n, dtype=np.int64),
        "sl": np.ones(n, dtype=np.int64),
        "margin": np.ones(n, dtype=np.int64),
        "leverage": np.ones(n, dtype=np.int64),
        "max_hold": np.zeros(n, dtype=np.int64),
        "risk_weight": np.zeros(n, dtype=np.float32),
        "risk_score": np.zeros(n, dtype=np.float32),
    }
    active_idx = np.flatnonzero(pd.to_numeric(candidate_dec["side"], errors="raise").to_numpy(dtype=np.int64) != 0)
    active_idx = active_idx[active_idx < n - 3]
    if int(max_rows) > 0 and len(active_idx) > int(max_rows):
        pick = np.linspace(0, len(active_idx) - 1, int(max_rows)).round().astype(np.int64)
        active_idx = active_idx[pick]
    reason_counts: dict[str, int] = {}
    for row_num, i in enumerate(active_idx):
        row = candidate_dec.iloc[int(i)]
        side = int(row.get("side", 0) or 0)
        best_score = -1e18
        best = (1, 1, 1, 1, 0)
        best_meta: dict[str, Any] = {}
        for tp_i in range(len(TP_MULT)):
            for sl_i in range(len(SL_MULT)):
                for margin_i in range(len(MARGIN_MULT)):
                    for lev_i in range(len(LEVERAGE_MULT)):
                        for hold_i in range(len(MAX_HOLD_BUCKETS)):
                            tp, sl, margin, lev, hold = _risk_values(row, (tp_i, sl_i, margin_i, lev_i, hold_i))
                            score, meta = seven._simulate_one_risk(
                                arrays,
                                int(i),
                                side,
                                tp=tp,
                                sl=sl,
                                margin=margin,
                                leverage=lev,
                                max_hold=hold,
                                fee_eff=fee_eff,
                                slip_eff=slip_eff,
                            )
                            # Extra conservative pressure prevents "max all" hindsight collapse.
                            exposure = margin * lev
                            score -= 0.020 * max(0.0, exposure - 1.65)
                            if score > best_score:
                                best_score = score
                                best = (tp_i, sl_i, margin_i, lev_i, hold_i)
                                best_meta = meta
        labels["tp"][int(i)], labels["sl"][int(i)], labels["margin"][int(i)], labels["leverage"][int(i)], labels["max_hold"][int(i)] = best
        labels["risk_weight"][int(i)] = float(np.clip(1.0 + max(best_score, -0.05) * 8.0, 0.25, 3.0))
        labels["risk_score"][int(i)] = float(best_score)
        reason = str(best_meta.get("exit_reason", "unknown"))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if (row_num + 1) % 1000 == 0:
            print(json.dumps({"delta_risk_label_progress": int(row_num + 1), "total": int(len(active_idx))}), flush=True)
    diag = {
        "active_labeled_rows": int(len(active_idx)),
        "label_exit_reasons": reason_counts,
        "tp_counts": {str(i): int(v) for i, v in enumerate(np.bincount(labels["tp"][active_idx], minlength=len(TP_MULT)))},
        "sl_counts": {str(i): int(v) for i, v in enumerate(np.bincount(labels["sl"][active_idx], minlength=len(SL_MULT)))},
        "margin_counts": {str(i): int(v) for i, v in enumerate(np.bincount(labels["margin"][active_idx], minlength=len(MARGIN_MULT)))},
        "leverage_counts": {str(i): int(v) for i, v in enumerate(np.bincount(labels["leverage"][active_idx], minlength=len(LEVERAGE_MULT)))},
        "max_hold_counts": {str(i): int(v) for i, v in enumerate(np.bincount(labels["max_hold"][active_idx], minlength=len(MAX_HOLD_BUCKETS)))},
    }
    return labels, diag


def _delta_risk_soft_dataset(
    frame: pd.DataFrame,
    candidate_dec: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    max_rows: int,
    top_k: int,
    temp: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    n = len(frame)
    active_idx = np.flatnonzero(pd.to_numeric(candidate_dec["side"], errors="raise").to_numpy(dtype=np.int64) != 0)
    active_idx = active_idx[active_idx < n - 3]
    if int(max_rows) > 0 and len(active_idx) > int(max_rows):
        pick = np.linspace(0, len(active_idx) - 1, int(max_rows)).round().astype(np.int64)
        active_idx = active_idx[pick]
    rows: list[int] = []
    y = {name: [] for name in ("tp", "sl", "margin", "leverage", "max_hold")}
    weights: list[float] = []
    reason_counts: dict[str, int] = {}
    top_k = max(1, int(top_k))
    temp = max(float(temp), 1e-6)
    for row_num, i in enumerate(active_idx):
        row = candidate_dec.iloc[int(i)]
        side = int(row.get("side", 0) or 0)
        scored: list[tuple[float, tuple[int, int, int, int, int], dict[str, Any]]] = []
        for tp_i in range(len(TP_MULT)):
            for sl_i in range(len(SL_MULT)):
                for margin_i in range(len(MARGIN_MULT)):
                    for lev_i in range(len(LEVERAGE_MULT)):
                        for hold_i in range(len(MAX_HOLD_BUCKETS)):
                            tp, sl, margin, lev, hold = _risk_values(row, (tp_i, sl_i, margin_i, lev_i, hold_i))
                            score, meta = seven._simulate_one_risk(
                                arrays,
                                int(i),
                                side,
                                tp=tp,
                                sl=sl,
                                margin=margin,
                                leverage=lev,
                                max_hold=hold,
                                fee_eff=fee_eff,
                                slip_eff=slip_eff,
                            )
                            exposure = margin * lev
                            score -= 0.020 * max(0.0, exposure - 1.65)
                            scored.append((float(score), (tp_i, sl_i, margin_i, lev_i, hold_i), meta))
        scored.sort(key=lambda z: z[0], reverse=True)
        chosen = scored[:top_k]
        scores = np.asarray([s for s, _ids, _meta in chosen], dtype=np.float64)
        probs = np.exp((scores - scores.max()) / temp)
        probs = probs / np.clip(probs.sum(), 1e-12, None)
        for p, (score, ids, meta) in zip(probs, chosen, strict=True):
            rows.append(int(i))
            y["tp"].append(int(ids[0]))
            y["sl"].append(int(ids[1]))
            y["margin"].append(int(ids[2]))
            y["leverage"].append(int(ids[3]))
            y["max_hold"].append(int(ids[4]))
            weights.append(float(np.clip(p * top_k * (1.0 + max(score, -0.05) * 8.0), 0.05, 4.0)))
        reason = str(chosen[0][2].get("exit_reason", "unknown"))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if (row_num + 1) % 1000 == 0:
            print(json.dumps({"delta_soft_label_progress": int(row_num + 1), "total": int(len(active_idx))}), flush=True)
    out = {
        "rows": np.asarray(rows, dtype=np.int64),
        "weight": np.asarray(weights, dtype=np.float32),
        **{name: np.asarray(vals, dtype=np.int64) for name, vals in y.items()},
    }
    diag = {
        "active_labeled_rows": int(len(active_idx)),
        "expanded_rows": int(len(rows)),
        "top_k": int(top_k),
        "temperature": float(temp),
        "label_exit_reasons": reason_counts,
        "tp_counts": {str(i): int(v) for i, v in enumerate(np.bincount(out["tp"], minlength=len(TP_MULT)))},
        "sl_counts": {str(i): int(v) for i, v in enumerate(np.bincount(out["sl"], minlength=len(SL_MULT)))},
        "margin_counts": {str(i): int(v) for i, v in enumerate(np.bincount(out["margin"], minlength=len(MARGIN_MULT)))},
        "leverage_counts": {str(i): int(v) for i, v in enumerate(np.bincount(out["leverage"], minlength=len(LEVERAGE_MULT)))},
        "max_hold_counts": {str(i): int(v) for i, v in enumerate(np.bincount(out["max_hold"], minlength=len(MAX_HOLD_BUCKETS)))},
    }
    return out, diag


def _fit_hgb(x: pd.DataFrame, y: np.ndarray, w: np.ndarray, *, seed: int) -> HistGradientBoostingClassifier:
    y = np.asarray(y, dtype=np.int64)
    weights = compute_sample_weight(class_weight="balanced", y=y).astype(np.float64) * np.asarray(w, dtype=np.float64)
    model = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.035,
        max_leaf_nodes=7,
        min_samples_leaf=55,
        l2_regularization=4.0,
        random_state=int(seed),
    )
    model.fit(x, y, sample_weight=weights)
    return model


def _apply_delta_models(dec: pd.DataFrame, x: pd.DataFrame, models: dict[str, Any]) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active = pd.to_numeric(out["action"], errors="raise").to_numpy(dtype=np.int64) != 0
    if not bool(active.any()):
        return out
    xa = x.loc[active].reset_index(drop=True)
    pred = {
        "tp": models["tp"].predict(xa).astype(np.int64),
        "sl": models["sl"].predict(xa).astype(np.int64),
        "margin": models["margin"].predict(xa).astype(np.int64),
        "leverage": models["leverage"].predict(xa).astype(np.int64),
        "max_hold": models["max_hold"].predict(xa).astype(np.int64),
    }
    idx = np.flatnonzero(active)
    for out_pos, frame_idx in enumerate(idx):
        tp, sl, margin, lev, hold = _risk_values(
            out.iloc[int(frame_idx)],
            (
                int(pred["tp"][out_pos]),
                int(pred["sl"][out_pos]),
                int(pred["margin"][out_pos]),
                int(pred["leverage"][out_pos]),
                int(pred["max_hold"][out_pos]),
            ),
        )
        out.loc[int(frame_idx), "take_profit"] = tp
        out.loc[int(frame_idx), "stop_loss"] = sl
        out.loc[int(frame_idx), "position_fraction"] = margin
        out.loc[int(frame_idx), "leverage"] = lev
        out.loc[int(frame_idx), "notional_exposure"] = margin * lev
        out.loc[int(frame_idx), "max_hold_bars"] = hold
        out.loc[int(frame_idx), "cooldown_bars"] = 0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-threshold", type=float, default=0.80)
    ap.add_argument("--candidate-threshold", type=float, default=0.50)
    ap.add_argument("--risk-label-max-rows", type=int, default=0)
    ap.add_argument("--force-zero-max-hold", action="store_true")
    ap.add_argument("--soft-top-k", type=int, default=1)
    ap.add_argument("--soft-temp", type=float, default=0.025)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=260611)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    global MAX_HOLD_BUCKETS
    if bool(args.force_zero_max_hold):
        MAX_HOLD_BUCKETS = np.asarray([0], dtype=np.int64)

    device = indep._device(str(args.device))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = seven._prepare_frames()
    fee, slip = omega._load_fee_slip()
    bundle = indep._load_parent_bundle(device)
    base_cols = list(bundle["base_cols"])
    train = frames["train_raw"]
    val = frames["val_raw"]
    oos = frames["oos_raw"]
    prefix = "omega1_regime3_expertdq"

    train_live_parent = indep._predict_parent(train, bundle, threshold=float(args.live_threshold), device=device, prefix=prefix)
    train_cand_parent = indep._predict_parent(train, bundle, threshold=float(args.candidate_threshold), device=device, prefix=prefix)
    val_parent = indep._predict_parent(val, bundle, threshold=float(args.live_threshold), device=device, prefix=prefix)
    oos_parent = indep._predict_parent(oos, bundle, threshold=float(args.live_threshold), device=device, prefix=prefix)

    train_cand_dec = indep._parent_to_decisions(train_cand_parent, prefix=prefix)
    val_dec_base = indep._parent_to_decisions(val_parent, prefix=prefix)
    oos_dec_base = indep._parent_to_decisions(oos_parent, prefix=prefix)
    x_train = indep._risk_feature_frame(train, base_cols, train_cand_parent, prefix=prefix)
    if int(args.soft_top_k) > 1:
        risk_labels, risk_diag = _delta_risk_soft_dataset(
            train,
            train_cand_dec,
            fee=fee,
            slip=slip,
            cost_mult=float(args.cost_mult),
            max_rows=int(args.risk_label_max_rows),
            top_k=int(args.soft_top_k),
            temp=float(args.soft_temp),
        )
        x_fit = x_train.iloc[np.asarray(risk_labels["rows"], dtype=np.int64)].reset_index(drop=True)
        w_fit = np.asarray(risk_labels["weight"], dtype=np.float32)
    else:
        risk_labels, risk_diag = _delta_risk_labels(
            train,
            train_cand_dec,
            fee=fee,
            slip=slip,
            cost_mult=float(args.cost_mult),
            max_rows=int(args.risk_label_max_rows),
        )
        labeled = np.asarray(risk_labels["risk_weight"], dtype=np.float32) > 0.0
        x_fit = x_train.loc[labeled].reset_index(drop=True)
        w_fit = np.asarray(risk_labels["risk_weight"], dtype=np.float32)[labeled]
        risk_labels = {k: np.asarray(v)[labeled] if k in {"tp", "sl", "margin", "leverage", "max_hold"} else v for k, v in risk_labels.items()}
    models = {
        "tp": _fit_hgb(x_fit, np.asarray(risk_labels["tp"], dtype=np.int64), w_fit, seed=int(args.seed) + 1),
        "sl": _fit_hgb(x_fit, np.asarray(risk_labels["sl"], dtype=np.int64), w_fit, seed=int(args.seed) + 2),
        "margin": _fit_hgb(x_fit, np.asarray(risk_labels["margin"], dtype=np.int64), w_fit, seed=int(args.seed) + 3),
        "leverage": _fit_hgb(x_fit, np.asarray(risk_labels["leverage"], dtype=np.int64), w_fit, seed=int(args.seed) + 4),
        "max_hold": (
            ConstantClassifier(0)
            if bool(args.force_zero_max_hold)
            else _fit_hgb(x_fit, np.asarray(risk_labels["max_hold"], dtype=np.int64), w_fit, seed=int(args.seed) + 5)
        ),
    }
    x_val = indep._risk_feature_frame(val, base_cols, val_parent, prefix=prefix)
    x_oos = indep._risk_feature_frame(oos, base_cols, oos_parent, prefix=prefix)
    val_dec_delta = _apply_delta_models(val_dec_base, x_val, models)
    oos_dec_delta = _apply_delta_models(oos_dec_base, x_oos, models)

    base_val = omega._metrics(val, val_dec_base, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
    base_oos = omega._metrics(oos, oos_dec_base, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
    delta_val = omega._metrics(val, val_dec_delta, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
    delta_oos = omega._metrics(oos, oos_dec_delta, fee=fee, slip=slip, cost_mult=float(args.cost_mult))

    train_live_parent.to_csv(OUT_DIR / "train_live_parent_predictions.csv", index=False)
    train_cand_parent.to_csv(OUT_DIR / "train_candidate_parent_predictions.csv", index=False)
    val_dec_base.to_csv(OUT_DIR / "validation_base_decisions.csv", index=False)
    oos_dec_base.to_csv(OUT_DIR / "oos_base_decisions.csv", index=False)
    val_dec_delta.to_csv(OUT_DIR / "validation_delta_risk_decisions.csv", index=False)
    oos_dec_delta.to_csv(OUT_DIR / "oos_delta_risk_decisions.csv", index=False)
    joblib.dump({"models": models, "feature_cols": list(x_train.columns)}, OUT_DIR / "delta_risk_heads.joblib")
    ranking = pd.DataFrame(
        [
            {"variant": "fixed_template", "split": "validation", **base_val},
            {"variant": "fixed_template", "split": "oos", **base_oos},
            {"variant": "delta_risk_heads", "split": "validation", **delta_val},
            {"variant": "delta_risk_heads", "split": "oos", **delta_oos},
        ]
    )
    ranking.to_csv(OUT_DIR / "ranking.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "design": "Frozen D/Q parent at live threshold; risk heads trained on lower-threshold candidate pool. Risk actions are baseline-relative delta buckets, not absolute free buckets.",
        "thresholds": {"live": float(args.live_threshold), "candidate": float(args.candidate_threshold)},
        "force_zero_max_hold": bool(args.force_zero_max_hold),
        "soft_top_k": int(args.soft_top_k),
        "soft_temp": float(args.soft_temp),
        "delta_buckets": {
            "tp_mult": TP_MULT.tolist(),
            "sl_mult": SL_MULT.tolist(),
            "margin_mult": MARGIN_MULT.tolist(),
            "leverage_mult": LEVERAGE_MULT.tolist(),
            "max_hold": MAX_HOLD_BUCKETS.tolist(),
        },
        "risk_label_diag": risk_diag,
        "results": {
            "fixed_template": {"validation": base_val, "oos": base_oos},
            "delta_risk_heads": {"validation": delta_val, "oos": delta_oos},
        },
        "bucket_summary": {
            "validation_base": indep._bucket_summary(val_dec_base),
            "oos_base": indep._bucket_summary(oos_dec_base),
            "validation_delta": indep._bucket_summary(val_dec_delta),
            "oos_delta": indep._bucket_summary(oos_dec_delta),
        },
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "model": str(OUT_DIR / "delta_risk_heads.joblib"),
            "ranking": str(OUT_DIR / "ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "results": report["results"], "bucket_summary": report["bucket_summary"], "risk_label_diag": risk_diag}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
