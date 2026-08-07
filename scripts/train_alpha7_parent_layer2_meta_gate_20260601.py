#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_alpha7_tp_sl_action_score_20260526 import (  # noqa: E402
    SPLIT_TS,
    _combine_primary_fallback,
    _combo_metrics,
    _json_default,
    _predict_scaled,
    _read,
)
from scripts.retrain_alpha7_clean_parent_plus_omega1_layer2_20260601 import (  # noqa: E402
    LAYER2_FEATURES_JSON,
    OVERLAY_SOURCES_2025,
    OVERLAY_SOURCES_2026,
    _load_feature_list,
    _overlay_exact,
)


MODEL_ID = "alpha7_parent_layer2_meta_gate_20260601"
BASE_CLEAN_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_submodel_01965_cleanfunding_v1_20260529"
TRAIN_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2025_alpha6_current_tail111_exact.csv"
EVAL_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2026_alpha6_current_tail111_exact.csv"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_parent_layer2_meta_gate_20260601"

PARENT_DECISION_COLS = [
    "action",
    "side",
    "notional_exposure",
    "leverage",
    "position_fraction",
    "take_profit",
    "stop_loss",
    "max_hold_bars",
    "cooldown_bars",
    "quality_score",
    "confidence",
]


def _active(dec: pd.DataFrame) -> np.ndarray:
    action = pd.to_numeric(dec["action"], errors="raise").to_numpy(dtype=np.int64)
    side = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
    return (action != 0) & (side != 0)


def _counterfactual_net_pnl(frame: pd.DataFrame, dec: pd.DataFrame, *, fee: float, slip: float) -> np.ndarray:
    open_px = pd.to_numeric(frame["open"], errors="raise").to_numpy(dtype=np.float64)
    high = pd.to_numeric(frame["high"], errors="raise").to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="raise").to_numpy(dtype=np.float64)
    close = pd.to_numeric(frame["close"], errors="raise").to_numpy(dtype=np.float64)
    side = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
    notional = pd.to_numeric(dec["notional_exposure"], errors="raise").to_numpy(dtype=np.float64)
    tp = pd.to_numeric(dec["take_profit"], errors="raise").to_numpy(dtype=np.float64)
    sl = pd.to_numeric(dec["stop_loss"], errors="raise").to_numpy(dtype=np.float64)
    hold = pd.to_numeric(dec["max_hold_bars"], errors="raise").fillna(0).to_numpy(dtype=np.int64)
    out = np.full(len(frame), np.nan, dtype=np.float64)
    active = _active(dec)
    for i in np.flatnonzero(active):
        fill_i = i + 1
        if fill_i >= len(frame):
            continue
        s = int(side[i])
        n = max(float(notional[i]), 0.0)
        if s == 0 or n <= 0.0:
            continue
        entry = open_px[fill_i] * (1.0 + float(slip) if s > 0 else 1.0 - float(slip))
        max_h = int(max(1, hold[i]))
        exit_px = close[min(fill_i + max_h, len(close) - 1)]
        reason = "max_hold"
        for j in range(fill_i, min(fill_i + max_h + 1, len(frame))):
            if s > 0:
                tp_px = entry * (1.0 + max(float(tp[i]), 0.0))
                sl_px = entry * (1.0 - max(float(sl[i]), 0.0))
                hit_tp = high[j] >= tp_px
                hit_sl = low[j] <= sl_px
                if hit_sl or hit_tp:
                    if hit_sl:
                        exit_px = sl_px * (1.0 - float(slip))
                        reason = "sl"
                    else:
                        exit_px = tp_px * (1.0 - float(slip))
                        reason = "tp"
                    break
            else:
                tp_px = entry * (1.0 - max(float(tp[i]), 0.0))
                sl_px = entry * (1.0 + max(float(sl[i]), 0.0))
                hit_tp = low[j] <= tp_px
                hit_sl = high[j] >= sl_px
                if hit_sl or hit_tp:
                    if hit_sl:
                        exit_px = sl_px * (1.0 + float(slip))
                        reason = "sl"
                    else:
                        exit_px = tp_px * (1.0 + float(slip))
                        reason = "tp"
                    break
        if reason == "max_hold":
            exit_px = exit_px * (1.0 - float(slip) if s > 0 else 1.0 + float(slip))
        raw = (exit_px - entry) / max(entry, 1e-12) if s > 0 else (entry - exit_px) / max(entry, 1e-12)
        out[i] = raw * n - 2.0 * float(fee) * n
    return out


def _meta_frame(frame: pd.DataFrame, dec: pd.DataFrame, layer2_cols: list[str]) -> pd.DataFrame:
    x = pd.DataFrame(index=frame.index)
    for col in PARENT_DECISION_COLS:
        x[f"parent_{col}"] = pd.to_numeric(dec[col], errors="raise")
    for col in layer2_cols:
        x[col] = pd.to_numeric(frame[col], errors="raise")
    if x.isna().any().any():
        bad = x.columns[x.isna().any()].tolist()
        raise RuntimeError(f"meta features contain NaN: {bad[:20]}")
    return x


def _fit_calibrated_hgb(x: pd.DataFrame, y: np.ndarray, *, seed: int) -> tuple[HistGradientBoostingClassifier, IsotonicRegression, dict[str, Any]]:
    n = len(x)
    cut = int(n * 0.80)
    if cut <= 100 or n - cut <= 50:
        raise RuntimeError(f"not enough active rows for meta fit: {n}")
    clf = HistGradientBoostingClassifier(
        max_iter=350,
        max_leaf_nodes=31,
        learning_rate=0.035,
        l2_regularization=0.15,
        min_samples_leaf=30,
        class_weight="balanced",
        random_state=int(seed),
    )
    clf.fit(x.iloc[:cut], y[:cut])
    raw = clf.predict_proba(x.iloc[cut:])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw, y[cut:])
    cal = iso.predict(raw)
    metrics = {
        "rows": int(n),
        "train_rows": int(cut),
        "cal_rows": int(n - cut),
        "positive_rate": float(np.mean(y)),
        "raw_auc": float(roc_auc_score(y[cut:], raw)) if len(np.unique(y[cut:])) > 1 else None,
        "cal_auc": float(roc_auc_score(y[cut:], cal)) if len(np.unique(y[cut:])) > 1 else None,
        "raw_ap": float(average_precision_score(y[cut:], raw)) if len(np.unique(y[cut:])) > 1 else None,
        "cal_ap": float(average_precision_score(y[cut:], cal)) if len(np.unique(y[cut:])) > 1 else None,
    }
    return clf, iso, metrics


def _predict_meta(clf: HistGradientBoostingClassifier, iso: IsotonicRegression, x: pd.DataFrame) -> np.ndarray:
    raw = clf.predict_proba(x)[:, 1]
    return np.asarray(iso.predict(raw), dtype=np.float64)


def _apply_gate(dec: pd.DataFrame, p: np.ndarray, *, threshold: float, resize: bool) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active = _active(out)
    veto = active & (p < float(threshold))
    out.loc[veto, ["action", "side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = 0
    out.loc[veto, "leverage"] = 1.0
    if resize:
        mid = active & ~veto & (p < min(float(threshold) + 0.10, 0.98))
        out.loc[mid, "notional_exposure"] = pd.to_numeric(out.loc[mid, "notional_exposure"], errors="raise") * 0.50
        out.loc[mid, "position_fraction"] = pd.to_numeric(out.loc[mid, "position_fraction"], errors="raise") * 0.50
    out["meta_success_prob"] = p
    return out


def _score(costs: dict[str, Any]) -> float:
    c3 = costs["cost3"]
    if int(c3["trades"]) < 20:
        return -1e9
    return float(c3["pnl"] / max(abs(float(c3["mdd"])), 1e-9))


def _eval_grid(frame: pd.DataFrame, dec: pd.DataFrame, p: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        for resize in [False, True]:
            gated = _apply_gate(dec, p, threshold=threshold, resize=resize)
            costs = _combo_metrics(frame, gated)
            rows.append({
                "threshold": float(threshold),
                "resize": bool(resize),
                "score": float(_score(costs)),
                "costs": costs,
                "active": int(_active(gated).sum()),
            })
    return rows


def _load_frames() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_base = _read(TRAIN_CSV)
    eval_base = _read(EVAL_CSV)
    layer2 = _load_feature_list(LAYER2_FEATURES_JSON)
    train, train_overlay = _overlay_exact(train_base, OVERLAY_SOURCES_2025, layer2, tag="train_2025")
    eval_df, eval_overlay = _overlay_exact(eval_base, OVERLAY_SOURCES_2026, layer2, tag="eval_2026")
    layer2_cols = [c for c in layer2 if c in train.columns and c in eval_df.columns]
    core_prefixes = ("ai_", "tide_", "chronos_", "regime3_", "m7_zigzag_", "dir3_")
    core = [c for c in layer2_cols if c.startswith(core_prefixes)]
    report = {"train_overlay": train_overlay, "eval_overlay": eval_overlay, "layer2_cols": layer2_cols, "core_cols": core}
    return train, eval_df, report


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_all, eval_df, overlay_report = _load_frames()
    train_df = train_all[train_all["timestamp"] < SPLIT_TS].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)

    primary = joblib.load(BASE_CLEAN_DIR / "primary_no_tp/parent.pkl")
    fallback = joblib.load(BASE_CLEAN_DIR / "fallback_v2_tp/parent.pkl")
    train_dec = _combine_primary_fallback(_predict_scaled(primary, train_df, None), _predict_scaled(fallback, train_df, None))
    val_dec = _combine_primary_fallback(_predict_scaled(primary, val_df, None), _predict_scaled(fallback, val_df, None))
    oos_dec = _combine_primary_fallback(_predict_scaled(primary, eval_df, None), _predict_scaled(fallback, eval_df, None))

    fee = float(primary["config"]["fee"]) * 3.0
    slip = float(primary["config"]["slip"]) * 3.0
    pnl = _counterfactual_net_pnl(train_df, train_dec, fee=fee, slip=slip)
    active = _active(train_dec) & np.isfinite(pnl)
    y = (pnl[active] > 0.0).astype(np.int64)
    layer2_core = overlay_report["core_cols"]
    x_train_all = _meta_frame(train_df, train_dec, layer2_core).loc[active].reset_index(drop=True)
    clf, iso, fit_metrics = _fit_calibrated_hgb(x_train_all, y, seed=6060121)
    joblib.dump({"model_id": MODEL_ID, "model": clf, "calibrator": iso, "feature_cols": list(x_train_all.columns), "layer2_cols": layer2_core}, OUT_DIR / "meta_gate_hgb.joblib")

    x_val = _meta_frame(val_df, val_dec, layer2_core)
    x_oos = _meta_frame(eval_df, oos_dec, layer2_core)
    p_val = _predict_meta(clf, iso, x_val)
    p_oos = _predict_meta(clf, iso, x_oos)

    baseline_val = _combo_metrics(val_df, val_dec)
    baseline_oos = _combo_metrics(eval_df, oos_dec)
    grid = _eval_grid(val_df, val_dec, p_val)
    grid.sort(key=lambda r: float(r["score"]), reverse=True)
    selected = grid[0]
    selected_oos_dec = _apply_gate(oos_dec, p_oos, threshold=float(selected["threshold"]), resize=bool(selected["resize"]))
    selected_oos = _combo_metrics(eval_df, selected_oos_dec)
    selected_val_dec = _apply_gate(val_dec, p_val, threshold=float(selected["threshold"]), resize=bool(selected["resize"]))
    selected_val = _combo_metrics(val_df, selected_val_dec)

    grid_rows = []
    for r in grid:
        c3 = r["costs"]["cost3"]
        c1 = r["costs"]["cost1"]
        c2 = r["costs"]["cost2"]
        grid_rows.append({
            "threshold": r["threshold"],
            "resize": r["resize"],
            "score": r["score"],
            "active": r["active"],
            "val_cost1_pnl": c1["pnl"],
            "val_cost2_pnl": c2["pnl"],
            "val_cost3_pnl": c3["pnl"],
            "val_cost3_mdd": c3["mdd"],
            "val_cost3_trades": c3["trades"],
            "val_cost3_wr": c3["wr"],
        })
    pd.DataFrame(grid_rows).to_csv(OUT_DIR / "grid.csv", index=False)

    report = {
        "model_id": MODEL_ID,
        "design": "Frozen clean Alpha7 parent/fallback decisions with Layer2 meta gate and optional notional resize. Parent artifacts are not retrained.",
        "mutable_surface": "meta_gate_only",
        "label": "Cost3 counterfactual net-positive label on parent-active 2025 Jan-Sep rows only.",
        "fit_metrics": fit_metrics,
        "overlay_report": {
            "layer2_core_count": len(layer2_core),
            "dropped_train_edges": overlay_report["train_overlay"]["dropped_edge_rows"],
            "dropped_eval_edges": overlay_report["eval_overlay"]["dropped_edge_rows"],
        },
        "baseline": {"validation": baseline_val, "oos": baseline_oos},
        "selected": {"threshold": selected["threshold"], "resize": selected["resize"], "validation": selected_val, "oos": selected_oos},
        "top_grid": grid[:10],
        "artifacts": {"model": str(OUT_DIR / "meta_gate_hgb.joblib"), "grid": str(OUT_DIR / "grid.csv"), "report": str(OUT_DIR / "report.json")},
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "selected": report["selected"]}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
