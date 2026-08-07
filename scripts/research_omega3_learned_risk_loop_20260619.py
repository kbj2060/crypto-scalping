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
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402


MODEL_ID = "omega3_learned_risk_skip_profile_loop2_20260619"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
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


def _load_bundle(device: torch.device) -> dict[str, Any]:
    bundle = torch.load(PARENT_BUNDLE, map_location=device, weights_only=False)
    if sorted(bundle["models"].keys()) != sorted(hard.EXPERT_NAMES):
        raise RuntimeError(f"unexpected parent experts: {bundle['models'].keys()}")
    return bundle


def _predict_frame(bundle: dict[str, Any], frame: pd.DataFrame, prefix: str, device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = threehead._base_input(frame, list(bundle["base_cols"]))
    preds = {expert: threehead._predict_payload(bundle["models"][expert], x, device=device) for expert in hard.EXPERT_NAMES}
    route = hard._route_id(frame)
    direction = threehead._routed(preds, route, "direction", 3)
    quality = threehead._routed(preds, route, "quality", 3)
    src = threehead._prediction_output(frame, direction, quality, threshold=0.0, prefix=prefix.rstrip("_"))
    return x, src


def _read_parent_split(frames: dict[str, pd.DataFrame], split: str, bundle: dict[str, Any], device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame, str, bool]:
    if split == "train":
        frame = frames["train_raw"].reset_index(drop=True)
        _x, src = _predict_frame(bundle, frame, PREFIX_TRAIN_VAL, device)
        return frame, src, PREFIX_TRAIN_VAL, True
    if split == "validation":
        frame = frames["val_raw"].reset_index(drop=True)
        pred = pd.read_csv(full_parent.PARENT_DIR / "validation_predictions_2025_true3head.csv", parse_dates=["timestamp"])
        src = frame[["timestamp"]].merge(pred, on="timestamp", how="left", validate="one_to_one")
        if src.isna().any().any():
            raise RuntimeError("validation parent prediction alignment produced NaN")
        return frame, src, PREFIX_TRAIN_VAL, True
    if split == "oos":
        frame = frames["oos_raw"].reset_index(drop=True)
        pred = pd.read_csv(full_parent.PARENT_DIR / "oos_predictions_2026_true3head.csv", parse_dates=["timestamp"])
        src = frame[["timestamp"]].merge(pred, on="timestamp", how="left", validate="one_to_one")
        if src.isna().any().any():
            raise RuntimeError("oos parent prediction alignment produced NaN")
        return frame, src, PREFIX_OOS, False
    raise RuntimeError(f"unknown split: {split}")


def _feature_frame(frame: pd.DataFrame, src: pd.DataFrame, dec0: pd.DataFrame, prefix: str) -> pd.DataFrame:
    x = sleeve._extra_features(base_features._feature_frame(frame, src, dec0, prefix), dec0)
    bad = [
        c
        for c in x.columns
        if c == "tp_sl_action_score" or c.startswith("clean_regime4_") or c.startswith("regime4_pred_") or c.startswith("teacher_")
    ]
    if bad:
        raise RuntimeError(f"forbidden feature columns: {bad[:20]}")
    return x.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _profiles(max_notional: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"name": "SKIP", "tp": 0.0, "sl": 0.0, "notional": 0.0}]
    notionals = tuple(n for n in (0.35, 0.60, 0.90, 1.20, 1.80) if n <= float(max_notional) + 1e-12)
    for tp in (0.018, 0.026, 0.04, 0.052, 0.07, 0.095):
        for sl in (0.010, 0.014, 0.022, 0.028, 0.04, 0.055):
            rr = tp / max(sl, 1e-12)
            if rr < 1.15 or rr > 7.5:
                continue
            for notional in notionals:
                out.append({"name": f"tp{tp:g}_sl{sl:g}_n{notional:g}", "tp": tp, "sl": sl, "notional": notional})
    return out


def _apply_profile(dec: pd.DataFrame, profile_idx: np.ndarray, profiles: list[dict[str, Any]], *, confidence: np.ndarray | None = None, min_conf: float = 0.0) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active = omega._active(out)
    prof = np.asarray(profile_idx, dtype=np.int64)
    conf = np.ones(len(out), dtype=np.float64) if confidence is None else np.asarray(confidence, dtype=np.float64)
    for i in np.flatnonzero(active):
        p = profiles[int(prof[int(i)])]
        if p["name"] == "SKIP" or conf[int(i)] < float(min_conf):
            out.loc[int(i), ["action", "side", "notional_exposure", "position_fraction", "take_profit", "stop_loss"]] = [omega.ACTION_CASH, 0, 0.0, 0.0, 0.0, 0.0]
            continue
        notional = float(p["notional"])
        out.loc[int(i), "notional_exposure"] = notional
        out.loc[int(i), "position_fraction"] = notional / 2.0
        out.loc[int(i), "leverage"] = 2.0
        out.loc[int(i), "take_profit"] = float(p["tp"]) * notional
        out.loc[int(i), "stop_loss"] = float(p["sl"]) * notional
        out.loc[int(i), "max_hold_bars"] = 0
        out.loc[int(i), "cooldown_bars"] = 0
    return out


def _label_profiles(
    frame: pd.DataFrame,
    dec0: pd.DataFrame,
    profiles: list[dict[str, Any]],
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    max_rows: int,
    seed: int,
    utility: str,
    mae_penalty: float,
    high_notional_penalty: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    active_idx = np.flatnonzero(omega._active(dec0))
    if len(active_idx) > max_rows:
        active_idx = np.sort(rng.choice(active_idx, size=max_rows, replace=False))
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    y = np.zeros(len(active_idx), dtype=np.int64)
    reason_counts: dict[str, int] = {}
    profile_counts: dict[str, int] = {}
    net_sum = 0.0
    for k, i in enumerate(active_idx):
        if k % 1000 == 0:
            print(json.dumps({"stage": "label_skip_profiles", "seen": int(k), "total": int(len(active_idx))}), flush=True)
        best_j = 0
        best_u = 0.0
        best_meta: dict[str, Any] = {"net": 0.0, "exit_reason": "skip"}
        for j, p in enumerate(profiles[1:], start=1):
            row = dec0.iloc[int(i)].copy()
            n = float(p["notional"])
            row.loc["notional_exposure"] = n
            row.loc["position_fraction"] = n / 2.0
            row.loc["leverage"] = 2.0
            row.loc["take_profit"] = float(p["tp"]) * n
            row.loc["stop_loss"] = float(p["sl"]) * n
            row.loc["max_hold_bars"] = 0
            row.loc["cooldown_bars"] = 0
            _score, meta = omega._simulate_trade(frame, arrays, int(i), row, fee=fee, slip=slip, cost_mult=cost_mult)
            net = float(meta.get("net", 0.0))
            mae = abs(float(meta.get("mae", 0.0)))
            mfe = float(meta.get("mfe", 0.0))
            if utility == "downside":
                u = net - float(mae_penalty) * mae - float(high_notional_penalty) * max(0.0, n - 0.9)
            elif utility == "asym":
                u = net + 0.15 * max(0.0, mfe) - float(mae_penalty) * mae - float(high_notional_penalty) * max(0.0, n - 0.9)
            else:
                u = net
            if u > best_u:
                best_u = float(u)
                best_j = int(j)
                best_meta = meta
        y[k] = best_j
        name = str(profiles[best_j]["name"])
        profile_counts[name] = profile_counts.get(name, 0) + 1
        reason = str(best_meta.get("exit_reason", "skip"))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        net_sum += float(best_meta.get("net", 0.0))
    diag = {
        "rows": int(len(active_idx)),
        "skip_rate": float(np.mean(y == 0)) if len(y) else 0.0,
        "reason_counts": reason_counts,
        "profile_counts_top": dict(sorted(profile_counts.items(), key=lambda kv: kv[1], reverse=True)[:12]),
        "selected_net_mean": float(net_sum / max(len(y), 1)),
        "utility": utility,
        "mae_penalty": float(mae_penalty),
        "high_notional_penalty": float(high_notional_penalty),
    }
    return active_idx.astype(np.int64), y, diag


def _fit_model(x_train: pd.DataFrame, train_idx: np.ndarray, y: np.ndarray, seed: int) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        max_iter=220,
        learning_rate=0.035,
        max_leaf_nodes=15,
        l2_regularization=0.25,
        random_state=seed,
    )
    model.fit(x_train.iloc[train_idx].to_numpy(dtype=np.float64), y)
    return model


def _predict(model: HistGradientBoostingClassifier, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    proba = model.predict_proba(x.to_numpy(dtype=np.float64))
    classes = np.asarray(model.classes_, dtype=np.int64)
    best = np.argmax(proba, axis=1)
    return classes[best].astype(np.int64), proba[np.arange(len(x)), best].astype(np.float64)


def _row(candidate: str, val_m: dict[str, Any], oos_m: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    def pack(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
        return {
            f"{prefix}_pnl": float(metrics["pnl"]),
            f"{prefix}_mdd": float(metrics["mdd"]),
            f"{prefix}_wr": float(metrics["wr"]),
            f"{prefix}_trades": int(metrics["trades"]),
            f"{prefix}_avg_notional": float(metrics.get("avg_notional", 0.0)),
            f"{prefix}_reasons": dict(metrics.get("reasons", {})),
        }

    row = {"candidate": candidate, **extra, **pack("val", val_m), **pack("oos", oos_m)}
    row["val_delta_vs_current"] = float(row["val_pnl"] - CURRENT["validation"]["pnl"])
    row["oos_delta_vs_current"] = float(row["oos_pnl"] - CURRENT["oos"]["pnl"])
    row["validation_only_score"] = float(row["val_pnl"] + row["val_mdd"] - 0.05 * max(0, row["val_trades"] - CURRENT["validation"]["trades"]))
    return row


def main() -> int:
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-label-rows", type=int, default=9000)
    ap.add_argument("--seed", type=int, default=260619)
    ap.add_argument("--max-notional", type=float, default=1.8)
    ap.add_argument("--mae-penalty", type=float, default=0.8)
    ap.add_argument("--high-notional-penalty", type=float, default=0.006)
    args = ap.parse_args()
    run_id = (
        f"{MODEL_ID}_ncap{str(args.max_notional).replace('.', 'p')}"
        f"_mae{str(args.mae_penalty).replace('.', 'p')}"
        f"_npen{str(args.high_notional_penalty).replace('.', 'p')}"
    )
    OUT_DIR = ROOT / "tmp/causal_regen_20260516" / run_id
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = threehead._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    bundle = _load_bundle(device)

    train_frame, train_src, train_prefix, train_oof = _read_parent_split(frames, "train", bundle, device)
    val_frame, val_src, val_prefix, val_oof = _read_parent_split(frames, "validation", bundle, device)
    oos_frame, oos_src, oos_prefix, oos_oof = _read_parent_split(frames, "oos", bundle, device)

    train_dec0 = overlay._build_dec(train_src, train_prefix, oof=train_oof)
    val_dec0 = overlay._build_dec(val_src, val_prefix, oof=val_oof)
    oos_dec0 = overlay._build_dec(oos_src, oos_prefix, oof=oos_oof)
    current_val_dec = sleeve._apply_aggressive(val_dec0)
    current_oos_dec = sleeve._apply_aggressive(oos_dec0)
    current_val = omega._metrics(val_frame, current_val_dec, fee=fee, slip=slip, cost_mult=3.0)
    current_oos = omega._metrics(oos_frame, current_oos_dec, fee=fee, slip=slip, cost_mult=3.0)

    x_train = _feature_frame(train_frame, train_src, train_dec0, train_prefix)
    x_val = _feature_frame(val_frame, val_src, val_dec0, val_prefix)
    x_oos = _feature_frame(oos_frame, oos_src, oos_dec0, oos_prefix)
    profiles = _profiles(float(args.max_notional))

    rows: list[dict[str, Any]] = []
    experiments: dict[str, Any] = {}
    for util in ("downside", "asym"):
        idx, y, label_diag = _label_profiles(
            train_frame,
            train_dec0,
            profiles,
            fee=fee,
            slip=slip,
            cost_mult=3.0,
            max_rows=int(args.max_label_rows),
            seed=int(args.seed),
            utility=util,
            mae_penalty=float(args.mae_penalty),
            high_notional_penalty=float(args.high_notional_penalty),
        )
        if len(np.unique(y)) < 2:
            experiments[util] = {"label_diag": label_diag, "error": "single-class label"}
            continue
        model = _fit_model(x_train, idx, y, int(args.seed) + (17 if util == "asym" else 0))
        val_pred, val_conf = _predict(model, x_val)
        oos_pred, oos_conf = _predict(model, x_oos)
        experiments[util] = {
            "label_diag": label_diag,
            "classes": [int(x) for x in model.classes_],
            "val_pred_counts": pd.Series(val_pred).value_counts().head(12).to_dict(),
            "oos_pred_counts": pd.Series(oos_pred).value_counts().head(12).to_dict(),
        }
        for min_conf in (0.0, 0.35, 0.45, 0.55, 0.65):
            vd = _apply_profile(val_dec0, val_pred, profiles, confidence=val_conf, min_conf=min_conf)
            od = _apply_profile(oos_dec0, oos_pred, profiles, confidence=oos_conf, min_conf=min_conf)
            vm = omega._metrics(val_frame, vd, fee=fee, slip=slip, cost_mult=3.0)
            om = omega._metrics(oos_frame, od, fee=fee, slip=slip, cost_mult=3.0)
            rows.append(_row(f"learned_skip_profile_{util}_conf{min_conf:g}", vm, om, {"utility": util, "min_conf": min_conf}))

    ranking = pd.DataFrame(rows).sort_values(["validation_only_score", "val_pnl"], ascending=[False, False]).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "ranking.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "run_id": run_id,
        "design": "Parent entry fixed. A learned risk classifier selects SKIP or a tp_price_move/sl_price_move/notional profile. Live hardcoded price-barrier template is removed from candidate decisions; TP/SL account thresholds are price_move * notional.",
        "parent_dir": str(full_parent.PARENT_DIR),
        "current_recomputed": {"validation": current_val, "oos": current_oos},
        "comparison_baseline": CURRENT,
        "profiles": profiles,
        "feature_contract": {"feature_count": int(x_train.shape[1]), "forbidden_feature_audit": {"passed": True}},
        "experiments": experiments,
        "selected_by_validation": ranking.iloc[0].to_dict() if len(ranking) else None,
        "best_by_oos_diagnostic": ranking.sort_values(["oos_pnl", "val_pnl"], ascending=[False, False]).iloc[0].to_dict() if len(ranking) else None,
        "ranking": ranking.to_dict(orient="records"),
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default))
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "selected": report["selected_by_validation"], "best_oos": report["best_by_oos_diagnostic"]}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
