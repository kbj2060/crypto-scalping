#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_mamba_sac_3head_feature_coordinator_20260604 as feat_coord  # noqa: E402
import train_eval_omega1_2_mamba_sac_lifecycle_controller_20260604 as lifecycle  # noqa: E402
import train_eval_omega1_2_supervised_risk_selector_20260604 as sup_risk  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_bucket_risk_selector_position_20260605"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
BASELINE_LIFECYCLE_DIR = ROOT / "data/ensemble/supervised/omega1_2_exit_feature_lifecycle_baseline_20260604"

TP_BUCKETS = np.asarray([0.018, 0.022, 0.026, 0.030, 0.034], dtype=np.float32)
SL_BUCKETS = np.asarray([0.008, 0.010, 0.012, 0.014, 0.018], dtype=np.float32)
NOTIONAL_BUCKETS = np.asarray([0.25, 0.3375, 0.405, 0.45, 0.55], dtype=np.float32)
LEVERAGE_BUCKETS = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
BASE_IDS = (2, 3, 3, 1)


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cuda" if (name == "cuda" or (name == "auto" and torch.cuda.is_available())) else "cpu")


def _fit_norm(x: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    arr = x.to_numpy(dtype=np.float32)
    med = np.nanmedian(arr, axis=0).astype(np.float32)
    q25 = np.nanpercentile(arr, 25, axis=0).astype(np.float32)
    q75 = np.nanpercentile(arr, 75, axis=0).astype(np.float32)
    scale = q75 - q25
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    out = (arr - med) / scale
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite bucket selector training matrix")
    return np.tanh(out / 3.0).astype(np.float32), {"columns": list(x.columns), "median": med, "scale": scale}


def _apply_norm(x: pd.DataFrame, norm: dict[str, Any]) -> np.ndarray:
    if list(x.columns) != list(norm["columns"]):
        raise RuntimeError("bucket selector feature column contract mismatch")
    arr = x.to_numpy(dtype=np.float32)
    out = (arr - norm["median"]) / norm["scale"]
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite bucket selector inference matrix")
    return np.tanh(out / 3.0).astype(np.float32)


def _prefix(oof: bool) -> str:
    return "omega1_regime3_expertdq_oof_" if oof else "omega1_regime3_expertdq_"


def _final_action(src: pd.DataFrame, *, oof: bool) -> np.ndarray:
    prefix = _prefix(oof)
    action = pd.to_numeric(src[f"{prefix}final_action"], errors="raise").to_numpy(dtype=np.int64)
    if not set(np.unique(action)).issubset({omega.ACTION_CASH, omega.ACTION_LONG, omega.ACTION_SHORT}):
        raise RuntimeError(f"unexpected final_action values: {sorted(np.unique(action).tolist())}")
    return action


def _side_from_action(action: np.ndarray) -> np.ndarray:
    return np.where(action == omega.ACTION_LONG, 1, np.where(action == omega.ACTION_SHORT, -1, 0)).astype(np.int64)


def _bucket_to_risk(ids: tuple[int, int, int, int] | np.ndarray) -> dict[str, float]:
    tp_i, sl_i, n_i, lev_i = [int(x) for x in ids]
    return {
        "tp": float(TP_BUCKETS[tp_i]),
        "sl": float(SL_BUCKETS[sl_i]),
        "notional": float(NOTIONAL_BUCKETS[n_i]),
        "leverage": float(LEVERAGE_BUCKETS[lev_i]),
    }


def _single_dec_row(action: int, side: int, ids: tuple[int, int, int, int] | np.ndarray) -> pd.Series:
    r = _bucket_to_risk(ids)
    return pd.Series(
        {
            "action": int(action),
            "side": int(side),
            "quality_score": 1.0,
            "confidence": 1.0,
            "notional_exposure": r["notional"],
            "position_fraction": r["notional"],
            "leverage": r["leverage"],
            "max_hold_bars": 0,
            "cooldown_bars": 0,
            "take_profit": r["tp"],
            "stop_loss": r["sl"],
        }
    )


def _position_feature_frame(n: int) -> pd.DataFrame:
    out = pd.DataFrame(index=np.arange(int(n)))
    for col in lifecycle.POS_COLS:
        out[col] = 0.0
    return out.astype(np.float32)


def _risk_features_with_position(base_x: pd.DataFrame, src: pd.DataFrame, *, oof: bool) -> pd.DataFrame:
    x = sup_risk._risk_features(base_x, src, oof=oof).reset_index(drop=True)
    pos = _position_feature_frame(len(x)).add_prefix("selector_")
    out = pd.concat([x, pos], axis=1)
    bad = [c for c in out.columns if "clean_regime4" in c or "regime4_pred" in c or "tp_sl_action_score" in c or str(c).startswith("teacher_")]
    if bad:
        raise RuntimeError(f"forbidden bucket selector features passed audit: {bad[:20]}")
    if out.columns.duplicated().any():
        dup = out.columns[out.columns.duplicated()].tolist()
        raise RuntimeError(f"duplicate bucket selector features: {dup[:20]}")
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)


def _candidate_ids(rng: np.random.Generator, n: int) -> np.ndarray:
    ids = [BASE_IDS]
    anchors = [
        (2, 3, 3, 1),
        (1, 2, 2, 1),
        (3, 3, 3, 1),
        (4, 4, 2, 1),
        (0, 1, 1, 0),
        (2, 2, 4, 2),
    ]
    ids.extend(anchors)
    while len(ids) < int(n):
        ids.append(
            (
                int(rng.integers(0, len(TP_BUCKETS))),
                int(rng.integers(0, len(SL_BUCKETS))),
                int(rng.integers(0, len(NOTIONAL_BUCKETS))),
                int(rng.integers(0, len(LEVERAGE_BUCKETS))),
            )
        )
    return np.asarray(ids[: int(n)], dtype=np.int64)


def _build_labels(
    frame: pd.DataFrame,
    src: pd.DataFrame,
    x_risk: pd.DataFrame,
    *,
    oof: bool,
    max_rows: int,
    candidates_per_row: int,
    seed: int,
    fee: float,
    slip: float,
    cost_mult: float,
    min_score: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    action = _final_action(src, oof=oof)
    side = _side_from_action(action)
    active_idx = np.flatnonzero(action != omega.ACTION_CASH)
    if int(max_rows) > 0 and len(active_idx) > int(max_rows):
        active_idx = np.sort(rng.choice(active_idx, size=int(max_rows), replace=False))
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    y: list[tuple[int, int, int, int]] = []
    weights: list[float] = []
    chosen_scores: list[float] = []
    reasons: dict[str, int] = {}
    for idx in active_idx:
        candidates = _candidate_ids(rng, int(candidates_per_row))
        scores = []
        meta_rows = []
        for ids in candidates:
            score, meta = omega._simulate_trade(
                frame,
                arrays,
                int(idx),
                _single_dec_row(int(action[int(idx)]), int(side[int(idx)]), ids),
                fee=fee,
                slip=slip,
                cost_mult=float(cost_mult),
            )
            scores.append(float(score))
            meta_rows.append(meta)
        best_i = int(np.argmax(scores))
        best_ids = tuple(int(x) for x in candidates[best_i])
        best_score = float(scores[best_i])
        if best_score < float(min_score):
            best_ids = BASE_IDS
        best_meta = meta_rows[best_i]
        reasons[str(best_meta.get("exit_reason", "unknown"))] = reasons.get(str(best_meta.get("exit_reason", "unknown")), 0) + 1
        y.append(best_ids)
        chosen_scores.append(best_score)
        scale = max(float(np.std(scores)), 1e-4)
        weights.append(float(np.exp(np.clip((best_score - float(np.median(scores))) / scale, -4.0, 4.0))))
    x_sel = x_risk.iloc[active_idx].reset_index(drop=True)
    if len(y) < 200:
        raise RuntimeError(f"not enough bucket selector rows: {len(y)}")
    y_np = np.asarray(y, dtype=np.int64)
    return (
        x_sel,
        y_np,
        np.asarray(weights, dtype=np.float32),
        active_idx.astype(np.int64),
        {
            "rows": int(len(y)),
            "tp_counts": {str(i): int(v) for i, v in enumerate(np.bincount(y_np[:, 0], minlength=len(TP_BUCKETS)))},
            "sl_counts": {str(i): int(v) for i, v in enumerate(np.bincount(y_np[:, 1], minlength=len(SL_BUCKETS)))},
            "notional_counts": {str(i): int(v) for i, v in enumerate(np.bincount(y_np[:, 2], minlength=len(NOTIONAL_BUCKETS)))},
            "leverage_counts": {str(i): int(v) for i, v in enumerate(np.bincount(y_np[:, 3], minlength=len(LEVERAGE_BUCKETS)))},
            "best_exit_reasons": reasons,
            "score_mean": float(np.mean(chosen_scores)),
            "score_p10": float(np.percentile(chosen_scores, 10)),
            "score_p90": float(np.percentile(chosen_scores, 90)),
        },
    )


def _make_model(kind: str, seed: int) -> Any:
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.04,
            max_leaf_nodes=7,
            l2_regularization=1.0,
            min_samples_leaf=50,
            random_state=int(seed),
        )
    if kind == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=240,
            max_depth=8,
            min_samples_leaf=35,
            random_state=int(seed),
            n_jobs=-1,
        )
    raise RuntimeError(f"unknown selector kind: {kind}")


def _train_heads(x: np.ndarray, y: np.ndarray, w: np.ndarray, *, kind: str, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    names = ["tp", "sl", "notional", "leverage"]
    models: dict[str, Any] = {}
    diag: dict[str, Any] = {}
    for j, name in enumerate(names):
        model = _make_model(kind, int(seed) + j)
        model.fit(x, y[:, j], sample_weight=w)
        pred = np.asarray(model.predict(x), dtype=np.int64).reshape(-1)
        models[name] = model
        diag[f"{name}_train_acc"] = float(np.mean(pred == y[:, j]))
    return models, {"kind": kind, **diag}


def _predict_ids(models: dict[str, Any], x: pd.DataFrame, norm: dict[str, Any]) -> np.ndarray:
    xn = _apply_norm(x, norm)
    return np.column_stack(
        [
            np.asarray(models["tp"].predict(xn), dtype=np.int64).reshape(-1),
            np.asarray(models["sl"].predict(xn), dtype=np.int64).reshape(-1),
            np.asarray(models["notional"].predict(xn), dtype=np.int64).reshape(-1),
            np.asarray(models["leverage"].predict(xn), dtype=np.int64).reshape(-1),
        ]
    )


def _decision_from_bucket_ids(src: pd.DataFrame, ids: np.ndarray, *, oof: bool) -> pd.DataFrame:
    action = _final_action(src, oof=oof)
    side = _side_from_action(action)
    active = action != omega.ACTION_CASH
    prefix = _prefix(oof)
    dec = pd.DataFrame(
        {
            "timestamp": src["timestamp"].to_numpy(),
            "action": action,
            "side": side,
            "notional_exposure": 0.0,
            "position_fraction": 0.0,
            "leverage": 1.0,
            "max_hold_bars": 0,
            "cooldown_bars": 0,
            "take_profit": 0.0,
            "stop_loss": 0.0,
            "quality_score": pd.to_numeric(src[f"{prefix}quality_for_action"], errors="raise").to_numpy(dtype=np.float64),
            "confidence": pd.to_numeric(src[f"{prefix}dir_confidence"], errors="raise").to_numpy(dtype=np.float64),
        }
    )
    for i in np.flatnonzero(active):
        r = _bucket_to_risk(ids[int(i)])
        dec.loc[int(i), "notional_exposure"] = r["notional"]
        dec.loc[int(i), "position_fraction"] = r["notional"]
        dec.loc[int(i), "leverage"] = r["leverage"]
        dec.loc[int(i), "take_profit"] = r["tp"]
        dec.loc[int(i), "stop_loss"] = r["sl"]
    return dec


def _prepare_frames(
    *,
    threehead_dir: Path,
    quality_threshold: float,
    device: torch.device,
    selector_kind: str,
    selector_rows: int,
    candidates_per_row: int,
    min_score: float,
    seed: int,
    cost_mult: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    base_frames = feat_coord._prepare_frames(threehead_dir, quality_threshold=float(quality_threshold), device=device)
    fee, slip = omega._load_fee_slip()
    bundle = feat_coord._load_3head_payloads(threehead_dir)
    train_x, train_src = feat_coord._predict_3head_frame(base_frames["train_df"], bundle, quality_threshold=float(quality_threshold), device=device, oof=True)
    val_x, val_src = feat_coord._predict_3head_frame(base_frames["val_df"], bundle, quality_threshold=float(quality_threshold), device=device, oof=True)
    oos_x, oos_src = feat_coord._predict_3head_frame(base_frames["oos_df"], bundle, quality_threshold=float(quality_threshold), device=device, oof=False)
    train_x_risk = _risk_features_with_position(train_x, train_src, oof=True)
    x_sel, y, w, _active_idx, label_diag = _build_labels(
        base_frames["train_df"],
        train_src,
        train_x_risk,
        oof=True,
        max_rows=int(selector_rows),
        candidates_per_row=int(candidates_per_row),
        seed=int(seed),
        fee=fee,
        slip=slip,
        cost_mult=float(cost_mult),
        min_score=float(min_score),
    )
    x_np, norm = _fit_norm(x_sel)
    models, train_diag = _train_heads(x_np, y, w, kind=str(selector_kind), seed=int(seed))

    def predict_ids(base_x: pd.DataFrame, src: pd.DataFrame, *, oof: bool) -> np.ndarray:
        action = _final_action(src, oof=oof)
        ids = np.repeat(np.asarray(BASE_IDS, dtype=np.int64)[None, :], len(action), axis=0)
        active = action != omega.ACTION_CASH
        if bool(active.any()):
            x_r = _risk_features_with_position(base_x, src, oof=oof)
            ids[active] = _predict_ids(models, x_r.loc[active].reset_index(drop=True), norm)
        return ids

    train_dec = _decision_from_bucket_ids(train_src, predict_ids(train_x, train_src, oof=True), oof=True)
    val_dec = _decision_from_bucket_ids(val_src, predict_ids(val_x, val_src, oof=True), oof=True)
    oos_dec = _decision_from_bucket_ids(oos_src, predict_ids(oos_x, oos_src, oof=False), oof=False)
    feature_cols = omega._numeric_feature_cols(pd.concat([base_frames["train_df"], base_frames["val_df"]], axis=0, ignore_index=True), base_frames["oos_df"])
    s_train = omega._build_state_frame(base_frames["train_df"], train_src, train_dec, oof=True, feature_cols=feature_cols)
    s_val = omega._build_state_frame(base_frames["val_df"], val_src, val_dec, oof=True, feature_cols=feature_cols)
    s_oos = omega._build_state_frame(base_frames["oos_df"], oos_src, oos_dec, oof=False, feature_cols=feature_cols)
    for state, src, prefix in (
        (s_train, train_src, "omega1_regime3_expertdq_oof"),
        (s_val, val_src, "omega1_regime3_expertdq_oof"),
        (s_oos, oos_src, "omega1_regime3_expertdq"),
    ):
        state["threehead_exit_p_hold_feature_only"] = pd.to_numeric(src[f"{prefix}_exit_p_hold_feature_only"], errors="raise").to_numpy(dtype=np.float64)
        state["threehead_exit_p_exit_feature_only"] = pd.to_numeric(src[f"{prefix}_exit_p_exit_feature_only"], errors="raise").to_numpy(dtype=np.float64)
        state["threehead_exit_edge_feature_only"] = pd.to_numeric(src[f"{prefix}_exit_edge_feature_only"], errors="raise").to_numpy(dtype=np.float64)
    out = dict(base_frames)
    out.update({"train_dec": train_dec, "val_dec": val_dec, "oos_dec": oos_dec, "s_train": s_train, "s_val": s_val, "s_oos": s_oos})
    return out, models, norm, {"label_diag": label_diag, "train_diag": train_diag}


def _risk_summary(dec: pd.DataFrame) -> dict[str, float]:
    active = pd.to_numeric(dec["action"], errors="raise").to_numpy(dtype=np.int64) != omega.ACTION_CASH
    out: dict[str, float] = {"active": int(active.sum())}
    if bool(active.any()):
        for col in ("take_profit", "stop_loss", "leverage", "notional_exposure"):
            vals = pd.to_numeric(dec.loc[active, col], errors="raise").to_numpy(dtype=np.float64)
            out[f"{col}_mean"] = float(np.mean(vals))
            out[f"{col}_p10"] = float(np.percentile(vals, 10))
            out[f"{col}_p90"] = float(np.percentile(vals, 90))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threehead-dir", type=Path, default=feat_coord.DEFAULT_3HEAD_DIR)
    ap.add_argument("--baseline-lifecycle-dir", type=Path, default=BASELINE_LIFECYCLE_DIR)
    ap.add_argument("--use-frozen-lifecycle", action="store_true")
    ap.add_argument("--quality-threshold", type=float, default=0.75)
    ap.add_argument("--selector-kind", choices=["hgb", "extratrees"], default="hgb")
    ap.add_argument("--selector-rows", type=int, default=2500)
    ap.add_argument("--candidates-per-row", type=int, default=48)
    ap.add_argument("--min-score", type=float, default=0.001)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--max-train-entries", type=int, default=600)
    ap.add_argument("--samples-per-entry", type=int, default=6)
    ap.add_argument("--train-max-sim-bars", type=int, default=96)
    ap.add_argument("--min-action-edge", type=float, default=0.002)
    ap.add_argument("--disable-resize", action="store_true")
    ap.add_argument("--disable-reverse", action="store_true")
    ap.add_argument("--class-balance-actor", action="store_true")
    ap.add_argument("--select-mode", choices=["actor_q", "q_only"], default="actor_q")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=260650)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()
    _seed_everything(int(args.seed))
    device = _device(str(args.device))
    out_dir = OUT_DIR if not str(args.out_suffix).strip() else OUT_DIR.parent / f"{MODEL_ID}_{str(args.out_suffix).strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames, models, selector_norm, selector_info = _prepare_frames(
        threehead_dir=Path(args.threehead_dir),
        quality_threshold=float(args.quality_threshold),
        device=device,
        selector_kind=str(args.selector_kind),
        selector_rows=int(args.selector_rows),
        candidates_per_row=int(args.candidates_per_row),
        min_score=float(args.min_score),
        seed=int(args.seed),
        cost_mult=float(args.cost_mult),
    )
    fee, slip = omega._load_fee_slip()
    if bool(args.use_frozen_lifecycle):
        ckpt = torch.load(Path(args.baseline_lifecycle_dir) / "lifecycle_controller.pt", map_location="cpu", weights_only=False)
        state_cols = list(ckpt["state_columns"])
        bad = [c for c in state_cols if "clean_regime4" in c or "regime4_pred" in c or "tp_sl_action_score" in c or str(c).startswith("teacher_")]
        if bad:
            raise RuntimeError(f"forbidden frozen lifecycle state columns passed audit: {bad[:20]}")
        model = lifecycle.MambaDiscreteActorCritic(len(state_cols), len(lifecycle.ACTION_NAMES))
        model.load_state_dict(ckpt["model_state_dict"])
        norm = ckpt["normalizer"]
        data_diag = {"mode": "frozen_lifecycle_no_retrain"}
        train_diag = {"mode": "frozen_lifecycle_no_retrain"}
        common = dict(seq_len=int(ckpt["seq_len"]), fee=fee, slip=slip, cost_mult=float(args.cost_mult), device=device, disable_resize=True, disable_reverse=True, select_mode=str(args.select_mode), force_parent_entry=False, force_entry_mult=1.0)
    else:
        state_cols = [c for c in lifecycle._base_state(frames["s_train"]).columns if c != "timestamp"]
        bad = [c for c in state_cols if "clean_regime4" in c or "regime4_pred" in c or "tp_sl_action_score" in c or str(c).startswith("teacher_")]
        if bad:
            raise RuntimeError(f"forbidden lifecycle state columns passed audit: {bad[:20]}")
        norm = lifecycle._fit_norm(lifecycle._base_state(frames["s_train"])[state_cols])
        data, data_diag = lifecycle._build_dataset(
            frames,
            seq_len=int(args.seq_len),
            max_entries=int(args.max_train_entries),
            samples_per_entry=int(args.samples_per_entry),
            seed=int(args.seed),
            fee=fee,
            slip=slip,
            cost_mult=float(args.cost_mult),
            max_sim_bars=int(args.train_max_sim_bars),
            min_action_edge=float(args.min_action_edge),
            disable_resize=bool(args.disable_resize),
            disable_reverse=bool(args.disable_reverse),
            position_only_training=False,
            norm=norm,
        )
        print(json.dumps({"stage": "bucket_position_lifecycle_train_start", "device": str(device), "seq_shape": list(data.seq.shape), "data_diag": data_diag}, ensure_ascii=False), flush=True)
        model, train_diag = lifecycle._train(data, device=device, steps=int(args.steps), batch_size=int(args.batch_size), lr=float(args.lr), class_balance_actor=bool(args.class_balance_actor))
        common = dict(seq_len=int(args.seq_len), fee=fee, slip=slip, cost_mult=float(args.cost_mult), device=device, disable_resize=bool(args.disable_resize), disable_reverse=bool(args.disable_reverse), select_mode=str(args.select_mode), force_parent_entry=False, force_entry_mult=1.0)
    val = lifecycle._replay(frames, "val", model, norm, **common)
    oos = lifecycle._replay(frames, "oos", model, norm, **common)
    with (out_dir / "bucket_risk_selector.pkl").open("wb") as f:
        pickle.dump(
            {
                "models": models,
                "normalizer": selector_norm,
                "state_columns": list(selector_norm["columns"]),
                "tp_buckets": TP_BUCKETS,
                "sl_buckets": SL_BUCKETS,
                "notional_buckets": NOTIONAL_BUCKETS,
                "leverage_buckets": LEVERAGE_BUCKETS,
                "base_ids": BASE_IDS,
            },
            f,
        )
    model_artifact = ""
    if not bool(args.use_frozen_lifecycle):
        torch.save({"model_state_dict": model.state_dict(), "normalizer": norm, "seq_len": int(args.seq_len), "state_columns": state_cols, "action_names": lifecycle.ACTION_NAMES}, out_dir / "lifecycle_controller.pt")
        model_artifact = str(out_dir / "lifecycle_controller.pt")
    report = {
        "model_id": MODEL_ID,
        "design": "Adapter Bucket Selector with explicit selector_lc_pos_* position-state input columns. Risk is selected as factorized TP/SL/notional/leverage bucket ids, not named templates.",
        "frozen_lifecycle": bool(args.use_frozen_lifecycle),
        "accounting_note": "Official lifecycle PnL uses notional_exposure as effective account exposure. leverage is stored and penalized in selector labels but is not an additional PnL multiplier in lifecycle replay.",
        "quality_threshold": float(args.quality_threshold),
        "selector": {
            "kind": str(args.selector_kind),
            "tp_buckets": TP_BUCKETS.tolist(),
            "sl_buckets": SL_BUCKETS.tolist(),
            "notional_buckets": NOTIONAL_BUCKETS.tolist(),
            "leverage_buckets": LEVERAGE_BUCKETS.tolist(),
            **selector_info,
        },
        "risk_summary": {split: _risk_summary(frames[f"{split}_dec"]) for split in ("train", "val", "oos")},
        "state_columns": state_cols,
        "training": {"data_diag": data_diag, "train_diag": train_diag, "min_action_edge": float(args.min_action_edge), "steps": int(args.steps), "class_balance_actor": bool(args.class_balance_actor)},
        "results": {"validation": val, "oos": oos},
        "artifacts": {"out_dir": str(out_dir), "report": str(out_dir / "report.json"), "selector": str(out_dir / "bucket_risk_selector.pkl"), "model": model_artifact},
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out_dir / "report.json"), "results": report["results"]}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
