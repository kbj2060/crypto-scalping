#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_1_cash_fallback_label_family_20260606 as label_family  # noqa: E402
import train_eval_omega1_2_1_cash_fallback_sleeve_20260606 as sleeve  # noqa: E402
import train_eval_omega1_2_1_full_retrain_cash_alpha43_20260608 as full  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega2_architect_priority_experiments_20260609"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
SEEDS = (260000, 260001, 260002, 260003, 260004, 260005, 260006, 260007, 260008, 260009, 260608, 260780)
THRESHOLDS = (0.50, 0.55, 0.60, 0.65)
RISK = sleeve.FallbackRisk("tp026_sl014_n0.30_h192", 0.026, 0.014, 0.30, 2.0, 192)
BASELINE_VAL_PNL = 100.542729
BASELINE_OOS_PNL = 72.760041


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


def _counts(arr: np.ndarray) -> dict[str, int]:
    return {str(k): int(v) for k, v in pd.Series(arr).value_counts().sort_index().items()}


def _model(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.035,
        max_leaf_nodes=7,
        l2_regularization=2.0,
        random_state=int(seed),
    )


def _classes_to_proba(model: Any, proba: np.ndarray) -> np.ndarray:
    out = np.zeros((len(proba), 3), dtype=np.float64)
    classes = np.asarray(model.classes_, dtype=np.int64)
    for j, cls in enumerate(classes):
        if 0 <= int(cls) <= 2:
            out[:, int(cls)] = proba[:, j]
    return out


def _fit_predict_full_proba(x_train: pd.DataFrame, y: np.ndarray, train_mask: np.ndarray, x_eval: pd.DataFrame, seed: int, weight: np.ndarray | None = None) -> np.ndarray:
    idx = np.flatnonzero(train_mask)
    if len(np.unique(y[idx])) < 2:
        return np.zeros((len(x_eval), 3), dtype=np.float64)
    m = _model(seed)
    kwargs = {}
    if weight is not None:
        kwargs["sample_weight"] = np.asarray(weight[idx], dtype=np.float64)
    m.fit(x_train.iloc[idx].to_numpy(dtype=np.float64), y[idx], **kwargs)
    return _classes_to_proba(m, m.predict_proba(x_eval.to_numpy(dtype=np.float64)))


def _oof_proba(x: pd.DataFrame, y: np.ndarray, train_mask: np.ndarray, seed: int, weight: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    idx = np.flatnonzero(train_mask)
    proba = np.zeros((len(x), 3), dtype=np.float64)
    folds = []
    n = len(idx)
    for fold_id, (train_frac, end_frac) in enumerate(((0.35, 0.50), (0.50, 0.65), (0.65, 0.80), (0.80, 1.00))):
        train_end = int(n * train_frac)
        val_end = int(n * end_frac)
        if train_end < 100 or val_end <= train_end:
            continue
        train_idx = idx[:train_end]
        val_idx = idx[train_end:val_end]
        if len(np.unique(y[train_idx])) < 2:
            folds.append({"fold": int(fold_id), "skipped": "single_class"})
            continue
        m = _model(seed + train_end)
        kwargs = {}
        if weight is not None:
            kwargs["sample_weight"] = np.asarray(weight[train_idx], dtype=np.float64)
        m.fit(x.iloc[train_idx].to_numpy(dtype=np.float64), y[train_idx], **kwargs)
        proba[val_idx] = _classes_to_proba(m, m.predict_proba(x.iloc[val_idx].to_numpy(dtype=np.float64)))
        folds.append({"fold": int(fold_id), "train_rows": int(len(train_idx)), "val_rows": int(len(val_idx)), "classes": np.asarray(m.classes_, dtype=int).tolist()})
    return proba, {"folds": folds, "oof_rows": int(np.count_nonzero(proba.max(axis=1) > 0.0))}


def _action_conf_from_proba(proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    action = np.argmax(proba, axis=1).astype(np.int64)
    conf = proba[np.arange(len(proba)), action].astype(np.float64)
    return action, conf


def _metric(prefix: str, m: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_pnl": float(m["pnl"]),
        f"{prefix}_mdd": float(m["mdd"]),
        f"{prefix}_wr": float(m["wr"]),
        f"{prefix}_trades": int(m["trades"]),
        f"{prefix}_fallback_entries": int(m.get("fallback_entries", 0)),
        f"{prefix}_primary_takeovers": int(m.get("primary_takeovers", 0)),
        f"{prefix}_reasons": m["exit_reasons"],
    }


def _evaluate_proba(
    *,
    label: str,
    variant: str,
    seed: int | str,
    threshold: float,
    val_proba: np.ndarray,
    oos_proba: np.ndarray,
    val_frame: pd.DataFrame,
    val_dec: pd.DataFrame,
    oos_frame: pd.DataFrame,
    oos_dec: pd.DataFrame,
    fee: float,
    slip: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    va, vc = _action_conf_from_proba(val_proba)
    oa, oc = _action_conf_from_proba(oos_proba)
    val_m = sleeve._metrics_with_fallback(val_frame, val_dec, RISK, va, vc, float(threshold), fee=fee, slip=slip, cost_mult=3.0)
    oos_m = sleeve._metrics_with_fallback(oos_frame, oos_dec, RISK, oa, oc, float(threshold), fee=fee, slip=slip, cost_mult=3.0)
    row = {
        "label": label,
        "variant": variant,
        "seed": seed,
        "threshold": float(threshold),
        **_metric("val", val_m),
        **_metric("oos", oos_m),
    }
    if extra:
        row.update(extra)
    return row


def _triple_with_takeover(frame: pd.DataFrame, dec: pd.DataFrame, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    close = pd.to_numeric(frame["close"], errors="raise").to_numpy(dtype=np.float64)
    high = pd.to_numeric(frame["high"], errors="raise").to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="raise").to_numpy(dtype=np.float64)
    atrp = np.maximum(label_family._atr_pct(frame), 0.0035)
    active = omega._active(dec)
    y = np.zeros(len(frame), dtype=np.int64)
    valid = np.zeros(len(frame), dtype=bool)
    weight = np.ones(len(frame), dtype=np.float64)
    takeover_before_outcome = 0
    censored = 0
    tie_count = 0
    max_hold = 24
    for i in range(0, len(frame) - max_hold - 2):
        base_px = float(close[i])
        if base_px <= 0.0 or not np.isfinite(base_px):
            continue
        barrier = float(atrp[i])
        up = base_px * (1.0 + barrier)
        dn = base_px * (1.0 - barrier)
        valid[i] = True
        hit_i = i + max_hold + 1
        hit_y = 0
        for j in range(i + 1, min(len(frame), i + max_hold + 1)):
            hit_up = bool(high[j] >= up)
            hit_dn = bool(low[j] <= dn)
            if hit_up and hit_dn:
                tie_count += 1
                hit_i = j
                hit_y = sleeve.ACTION_LONG if close[j] >= base_px else sleeve.ACTION_SHORT
                break
            if hit_up:
                hit_i = j
                hit_y = sleeve.ACTION_LONG
                break
            if hit_dn:
                hit_i = j
                hit_y = sleeve.ACTION_SHORT
                break
        takeover_idx = np.flatnonzero(active[i + 1 : min(len(frame), i + max_hold + 1)])
        takeover_i = int(i + 1 + takeover_idx[0]) if len(takeover_idx) else 10**12
        if takeover_i < hit_i:
            takeover_before_outcome += 1
            if mode == "takeover_censored":
                valid[i] = False
                censored += 1
                continue
            if mode == "takeover_as_cash":
                y[i] = 0
                continue
            if mode == "takeover_downweight_0p25":
                weight[i] = 0.25
        y[i] = hit_y
    return y, valid, weight, {
        "mode": mode,
        "atr_mult": 1.0,
        "max_hold": max_hold,
        "min_barrier": 0.0035,
        "tie_count": int(tie_count),
        "takeover_before_outcome": int(takeover_before_outcome),
        "censored": int(censored),
        "label_counts": _counts(y[valid]),
    }


def _summarize_group(rows: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    out = []
    for key, g in rows.groupby(keys, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        item = dict(zip(keys, key))
        item.update(
            {
                "runs": int(len(g)),
                "val_pnl_mean": float(g["val_pnl"].mean()),
                "val_pnl_median": float(g["val_pnl"].median()),
                "val_pnl_min": float(g["val_pnl"].min()),
                "oos_pnl_mean": float(g["oos_pnl"].mean()),
                "oos_pnl_median": float(g["oos_pnl"].median()),
                "oos_pnl_min": float(g["oos_pnl"].min()),
                "oos_pnl_max": float(g["oos_pnl"].max()),
                "oos_mdd_worst": float(g["oos_mdd"].min()),
                "oos_wr_mean": float(g["oos_wr"].mean()),
                "oos_trades_mean": float(g["oos_trades"].mean()),
                "beat_both_rate": float(((g["val_pnl"] > BASELINE_VAL_PNL) & (g["oos_pnl"] > BASELINE_OOS_PNL)).mean()),
                "oos_above_100_rate": float((g["oos_pnl"] >= 100.0).mean()),
            }
        )
        out.append(item)
    return pd.DataFrame(out).sort_values(
        ["beat_both_rate", "oos_pnl_median", "oos_pnl_mean"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _ensemble_proba(
    x_train: pd.DataFrame,
    y: np.ndarray,
    train_mask: np.ndarray,
    x_eval: pd.DataFrame,
    seeds: tuple[int, ...],
    weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    probs = []
    preds = []
    for seed in seeds:
        p = _fit_predict_full_proba(x_train, y, train_mask, x_eval, seed, weight=weight)
        probs.append(p)
        preds.append(np.argmax(p, axis=1))
    stack = np.stack(probs, axis=0)
    pred_stack = np.stack(preds, axis=0)
    return stack.mean(axis=0), pred_stack


def _ensemble_oof_proba(x: pd.DataFrame, y: np.ndarray, train_mask: np.ndarray, seeds: tuple[int, ...], weight: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    probs = []
    preds = []
    for seed in seeds:
        p, _diag = _oof_proba(x, y, train_mask, seed, weight=weight)
        probs.append(p)
        preds.append(np.argmax(p, axis=1))
    stack = np.stack(probs, axis=0)
    pred_stack = np.stack(preds, axis=0)
    return stack.mean(axis=0), pred_stack


def _agreement_filter(proba: np.ndarray, pred_stack: np.ndarray, min_agree: int) -> np.ndarray:
    pred = np.argmax(proba, axis=1)
    agree = (pred_stack == pred[None, :]).sum(axis=0)
    out = proba.copy()
    out[agree < int(min_agree)] = 0.0
    out[agree < int(min_agree), 0] = 1.0
    return out


def _purged_threshold_stability(
    x: pd.DataFrame,
    y: np.ndarray,
    train_mask: np.ndarray,
    val_frame: pd.DataFrame,
    val_dec: pd.DataFrame,
    fee: float,
    slip: float,
) -> list[dict[str, Any]]:
    idx = np.flatnonzero(train_mask)
    blocks = np.array_split(idx, 6)
    rows: list[dict[str, Any]] = []
    embargo = 192
    for fold, test_idx in enumerate(blocks):
        if len(test_idx) < 100:
            continue
        lo, hi = int(test_idx[0]), int(test_idx[-1])
        train_idx = idx[(idx < lo - embargo) | (idx > hi + embargo)]
        if len(train_idx) < 500 or len(np.unique(y[train_idx])) < 2:
            continue
        m = _model(260900 + fold)
        m.fit(x.iloc[train_idx].to_numpy(dtype=np.float64), y[train_idx])
        fold_proba = np.zeros((len(x), 3), dtype=np.float64)
        fold_proba[test_idx] = _classes_to_proba(m, m.predict_proba(x.iloc[test_idx].to_numpy(dtype=np.float64)))
        action, conf = _action_conf_from_proba(fold_proba)
        for thr in np.arange(0.50, 0.71, 0.05):
            mtr = sleeve._metrics_with_fallback(val_frame, val_dec, RISK, action, conf, float(thr), fee=fee, slip=slip, cost_mult=3.0)
            rows.append(
                {
                    "fold": int(fold),
                    "threshold": float(round(thr, 2)),
                    "train_rows": int(len(train_idx)),
                    "test_rows": int(len(test_idx)),
                    "val_pnl": float(mtr["pnl"]),
                    "val_mdd": float(mtr["mdd"]),
                    "val_wr": float(mtr["wr"]),
                    "val_trades": int(mtr["trades"]),
                    "val_fallback_entries": int(mtr.get("fallback_entries", 0)),
                    "val_primary_takeovers": int(mtr.get("primary_takeovers", 0)),
                }
            )
    return rows


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = threehead._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    val_frame, val_dec, val_features = full._build_split(frames, "validation")
    oos_frame, oos_dec, oos_features = full._build_split(frames, "oos")
    val_cash = ~omega._active(val_dec)

    detail_rows: list[dict[str, Any]] = []
    ensemble_rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {"label_variants": []}

    for mode in ("original", "takeover_censored", "takeover_as_cash", "takeover_downweight_0p25"):
        if mode == "original":
            y, valid, diag = label_family._triple_barrier_labels(val_frame, atr_mult=1.0, max_hold=24, min_barrier=0.0035)
            weight = None
        else:
            y, valid, weight, diag = _triple_with_takeover(val_frame, val_dec, mode)
        train_mask = val_cash & valid
        diagnostics["label_variants"].append({"mode": mode, "train_rows": int(np.count_nonzero(train_mask)), "diag": diag})
        for seed in SEEDS:
            val_p, _ = _oof_proba(val_features, y, train_mask, int(seed), weight=weight)
            oos_p = _fit_predict_full_proba(val_features, y, train_mask, oos_features, int(seed), weight=weight)
            for thr in THRESHOLDS:
                detail_rows.append(
                    _evaluate_proba(
                        label=mode,
                        variant="single_seed",
                        seed=int(seed),
                        threshold=float(thr),
                        val_proba=val_p,
                        oos_proba=oos_p,
                        val_frame=val_frame,
                        val_dec=val_dec,
                        oos_frame=oos_frame,
                        oos_dec=oos_dec,
                        fee=fee,
                        slip=slip,
                    )
                )
        print(json.dumps({"stage": "takeover_label", "mode": mode, "done": True}), flush=True)

        if mode == "original":
            val_mean, val_pred_stack = _ensemble_oof_proba(val_features, y, train_mask, SEEDS)
            oos_mean, oos_pred_stack = _ensemble_proba(val_features, y, train_mask, oos_features, SEEDS)
            for min_agree in (0, 7, 8, 9):
                vp = _agreement_filter(val_mean, val_pred_stack, min_agree) if min_agree else val_mean
                op = _agreement_filter(oos_mean, oos_pred_stack, min_agree) if min_agree else oos_mean
                for thr in (0.55, 0.60):
                    ensemble_rows.append(
                        _evaluate_proba(
                            label=mode,
                            variant="seed_ensemble_agreement",
                            seed=f"12seed_agree{min_agree}",
                            threshold=float(thr),
                            val_proba=vp,
                            oos_proba=op,
                            val_frame=val_frame,
                            val_dec=val_dec,
                            oos_frame=oos_frame,
                            oos_dec=oos_dec,
                            fee=fee,
                            slip=slip,
                            extra={"min_agree": int(min_agree), "ensemble_size": int(len(SEEDS))},
                        )
                    )
            purged_rows = _purged_threshold_stability(val_features, y, train_mask, val_frame, val_dec, fee, slip)

    detail = pd.DataFrame(detail_rows)
    ensemble = pd.DataFrame(ensemble_rows)
    purged = pd.DataFrame(purged_rows)
    label_summary = _summarize_group(detail, ["label", "threshold"])
    ensemble_summary = _summarize_group(ensemble, ["seed", "threshold"]) if len(ensemble) else pd.DataFrame()
    if len(purged):
        purged_summary = (
            purged.groupby("threshold")
            .agg(
                folds=("fold", "nunique"),
                pnl_mean=("val_pnl", "mean"),
                pnl_median=("val_pnl", "median"),
                pnl_min=("val_pnl", "min"),
                trades_mean=("val_trades", "mean"),
                fallback_entries_mean=("val_fallback_entries", "mean"),
            )
            .reset_index()
            .sort_values(["pnl_median", "pnl_mean"], ascending=False)
        )
    else:
        purged_summary = pd.DataFrame()

    detail.to_csv(OUT_DIR / "takeover_label_multiseed_detail.csv", index=False)
    label_summary.to_csv(OUT_DIR / "takeover_label_multiseed_summary.csv", index=False)
    ensemble.to_csv(OUT_DIR / "seed_ensemble_agreement_detail.csv", index=False)
    ensemble_summary.to_csv(OUT_DIR / "seed_ensemble_agreement_summary.csv", index=False)
    purged.to_csv(OUT_DIR / "purged_threshold_detail.csv", index=False)
    purged_summary.to_csv(OUT_DIR / "purged_threshold_summary.csv", index=False)

    report = {
        "model_id": MODEL_ID,
        "diagnostics": diagnostics,
        "top_takeover_label_summary": label_summary.head(20).to_dict(orient="records"),
        "top_seed_ensemble_summary": ensemble_summary.head(20).to_dict(orient="records") if len(ensemble_summary) else [],
        "purged_threshold_summary": purged_summary.to_dict(orient="records") if len(purged_summary) else [],
        "top_single_runs": detail.sort_values(["oos_pnl", "val_pnl"], ascending=[False, False]).head(20).to_dict(orient="records"),
        "artifacts": {
            "takeover_detail": str(OUT_DIR / "takeover_label_multiseed_detail.csv"),
            "takeover_summary": str(OUT_DIR / "takeover_label_multiseed_summary.csv"),
            "ensemble_detail": str(OUT_DIR / "seed_ensemble_agreement_detail.csv"),
            "ensemble_summary": str(OUT_DIR / "seed_ensemble_agreement_summary.csv"),
            "purged_detail": str(OUT_DIR / "purged_threshold_detail.csv"),
            "purged_summary": str(OUT_DIR / "purged_threshold_summary.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "top_takeover": report["top_takeover_label_summary"][:8], "top_ensemble": report["top_seed_ensemble_summary"][:8], "purged": report["purged_threshold_summary"]}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
