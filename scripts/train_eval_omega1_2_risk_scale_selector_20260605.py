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


MODEL_ID = "omega1_2_risk_scale_selector_20260605"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
BASE_RISK = {"tp": 0.026, "sl": 0.014, "leverage": 2.0, "notional": 0.45}
SCALE_PROFILES = [
    {"name": "defensive_tight", "tp": 0.85, "sl": 0.75, "leverage": 0.85, "notional": 0.70},
    {"name": "defensive_wide", "tp": 0.90, "sl": 1.10, "leverage": 0.85, "notional": 0.75},
    {"name": "base", "tp": 1.00, "sl": 1.00, "leverage": 1.00, "notional": 1.00},
    {"name": "base_plus", "tp": 1.05, "sl": 0.95, "leverage": 1.00, "notional": 1.10},
    {"name": "runner", "tp": 1.25, "sl": 1.00, "leverage": 1.00, "notional": 1.00},
    {"name": "conviction", "tp": 1.15, "sl": 1.05, "leverage": 1.10, "notional": 1.25},
    {"name": "scalp", "tp": 0.70, "sl": 0.70, "leverage": 0.90, "notional": 0.90},
]


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
        raise RuntimeError("non-finite scale selector matrix")
    return np.tanh(out / 3.0).astype(np.float32), {"columns": list(x.columns), "median": med, "scale": scale}


def _apply_norm(x: pd.DataFrame, norm: dict[str, Any]) -> np.ndarray:
    if list(x.columns) != list(norm["columns"]):
        raise RuntimeError("risk scale selector feature column contract mismatch")
    arr = x.to_numpy(dtype=np.float32)
    out = (arr - norm["median"]) / norm["scale"]
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite scale selector inference matrix")
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


def _profile_to_risk(profile_id: int) -> dict[str, float]:
    p = SCALE_PROFILES[int(profile_id)]
    return {
        "tp": float(np.clip(BASE_RISK["tp"] * p["tp"], 0.010, 0.050)),
        "sl": float(np.clip(BASE_RISK["sl"] * p["sl"], 0.006, 0.030)),
        "leverage": float(np.clip(BASE_RISK["leverage"] * p["leverage"], 1.0, 4.0)),
        "notional": float(np.clip(BASE_RISK["notional"] * p["notional"], 0.15, 0.90)),
    }


def _single_dec_row(action: int, side: int, profile_id: int) -> pd.Series:
    r = _profile_to_risk(int(profile_id))
    return pd.Series(
        {
            "action": int(action),
            "side": int(side),
            "quality_score": 1.0,
            "confidence": 1.0,
            "notional_exposure": r["notional"],
            "position_fraction": r["notional"],
            "leverage": r["leverage"],
            "max_hold_bars": 72,
            "cooldown_bars": 0,
            "take_profit": r["tp"],
            "stop_loss": r["sl"],
        }
    )


def _decision_from_profiles(src: pd.DataFrame, *, oof: bool, profile: np.ndarray) -> pd.DataFrame:
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
    for k in range(len(SCALE_PROFILES)):
        risk = _profile_to_risk(k)
        mask = active & (profile == int(k))
        dec.loc[mask, "notional_exposure"] = risk["notional"]
        dec.loc[mask, "position_fraction"] = risk["notional"]
        dec.loc[mask, "leverage"] = risk["leverage"]
        dec.loc[mask, "take_profit"] = risk["tp"]
        dec.loc[mask, "stop_loss"] = risk["sl"]
    return dec


def _build_labels(
    frame: pd.DataFrame,
    src: pd.DataFrame,
    x_risk: pd.DataFrame,
    *,
    oof: bool,
    max_rows: int,
    fee: float,
    slip: float,
    cost_mult: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    action = _final_action(src, oof=oof)
    side = _side_from_action(action)
    active_idx = np.flatnonzero(action != omega.ACTION_CASH)
    if int(max_rows) > 0 and len(active_idx) > int(max_rows):
        keep = np.linspace(0, len(active_idx) - 1, int(max_rows)).round().astype(np.int64)
        active_idx = active_idx[keep]
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    y: list[int] = []
    weights: list[float] = []
    reasons: dict[str, int] = {}
    for idx in active_idx:
        scores = []
        profile_reasons = []
        for k in range(len(SCALE_PROFILES)):
            score, meta = omega._simulate_trade(
                frame,
                arrays,
                int(idx),
                _single_dec_row(int(action[int(idx)]), int(side[int(idx)]), k),
                fee=fee,
                slip=slip,
                cost_mult=float(cost_mult),
            )
            scores.append(float(score))
            profile_reasons.append(str(meta.get("exit_reason", "unknown")))
        best = int(np.argmax(scores))
        y.append(best)
        reasons[profile_reasons[best]] = reasons.get(profile_reasons[best], 0) + 1
        scale = max(float(np.std(scores)), 1e-4)
        weights.append(float(np.exp(np.clip((float(scores[best]) - float(np.median(scores))) / scale, -4.0, 4.0))))
    x_sel = x_risk.iloc[active_idx].reset_index(drop=True)
    if len(y) < 200:
        raise RuntimeError(f"not enough risk scale rows: {len(y)}")
    return np.asarray(y, dtype=np.int64), np.asarray(weights, dtype=np.float32), {"rows": int(len(y)), "label_counts": {str(i): int(v) for i, v in enumerate(np.bincount(np.asarray(y), minlength=len(SCALE_PROFILES)))}, "best_exit_reasons": reasons, "x_sel": x_sel}


def _train_selector(x: np.ndarray, y: np.ndarray, w: np.ndarray, *, kind: str, seed: int) -> tuple[Any, dict[str, Any]]:
    if kind == "hgb":
        model: Any = HistGradientBoostingClassifier(
            max_iter=160,
            learning_rate=0.04,
            max_leaf_nodes=7,
            l2_regularization=1.0,
            min_samples_leaf=50,
            random_state=int(seed),
        )
    elif kind == "extratrees":
        model = ExtraTreesClassifier(
            n_estimators=220,
            max_depth=7,
            min_samples_leaf=35,
            random_state=int(seed),
            n_jobs=-1,
        )
    else:
        raise RuntimeError(f"unknown selector kind: {kind}")
    model.fit(x, y, sample_weight=w)
    pred = model.predict(x)
    return model, {"kind": kind, "train_acc": float(np.mean(pred == y))}


def _prepare_frames(
    *,
    threehead_dir: Path,
    quality_threshold: float,
    device: torch.device,
    selector_kind: str,
    selector_rows: int,
    seed: int,
    cost_mult: float,
) -> tuple[dict[str, Any], Any, dict[str, Any], dict[str, Any]]:
    base_frames = feat_coord._prepare_frames(threehead_dir, quality_threshold=float(quality_threshold), device=device)
    fee, slip = omega._load_fee_slip()
    bundle = feat_coord._load_3head_payloads(threehead_dir)
    train_x, train_src = feat_coord._predict_3head_frame(base_frames["train_df"], bundle, quality_threshold=float(quality_threshold), device=device, oof=True)
    val_x, val_src = feat_coord._predict_3head_frame(base_frames["val_df"], bundle, quality_threshold=float(quality_threshold), device=device, oof=True)
    oos_x, oos_src = feat_coord._predict_3head_frame(base_frames["oos_df"], bundle, quality_threshold=float(quality_threshold), device=device, oof=False)
    train_x_risk = sup_risk._risk_features(train_x, train_src, oof=True)
    y, w, label_diag = _build_labels(
        base_frames["train_df"],
        train_src,
        train_x_risk,
        oof=True,
        max_rows=int(selector_rows),
        fee=fee,
        slip=slip,
        cost_mult=float(cost_mult),
    )
    x_train = label_diag.pop("x_sel")
    x_np, norm = _fit_norm(x_train)
    selector, train_diag = _train_selector(x_np, y, w, kind=str(selector_kind), seed=int(seed))
    def predict_profile(base_x: pd.DataFrame, src: pd.DataFrame, *, oof: bool) -> np.ndarray:
        action = _final_action(src, oof=oof)
        profile = np.full(len(action), 2, dtype=np.int64)
        active = action != omega.ACTION_CASH
        if bool(active.any()):
            x_r = sup_risk._risk_features(base_x, src, oof=oof)
            profile[active] = selector.predict(_apply_norm(x_r.loc[active].reset_index(drop=True), norm)).astype(np.int64)
        return profile
    train_dec = _decision_from_profiles(train_src, oof=True, profile=predict_profile(train_x, train_src, oof=True))
    val_dec = _decision_from_profiles(val_src, oof=True, profile=predict_profile(val_x, val_src, oof=True))
    oos_dec = _decision_from_profiles(oos_src, oof=False, profile=predict_profile(oos_x, oos_src, oof=False))
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
    return out, selector, norm, {"label_diag": label_diag, "train_diag": train_diag}


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
    ap.add_argument("--quality-threshold", type=float, default=0.75)
    ap.add_argument("--selector-kind", choices=["hgb", "extratrees"], default="extratrees")
    ap.add_argument("--selector-rows", type=int, default=6000)
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
    ap.add_argument("--seed", type=int, default=260635)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()
    _seed_everything(int(args.seed))
    device = _device(str(args.device))
    out_dir = OUT_DIR if not str(args.out_suffix).strip() else OUT_DIR.parent / f"{MODEL_ID}_{str(args.out_suffix).strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames, selector, selector_norm, selector_info = _prepare_frames(
        threehead_dir=Path(args.threehead_dir),
        quality_threshold=float(args.quality_threshold),
        device=device,
        selector_kind=str(args.selector_kind),
        selector_rows=int(args.selector_rows),
        seed=int(args.seed),
        cost_mult=float(args.cost_mult),
    )
    fee, slip = omega._load_fee_slip()
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
    print(json.dumps({"stage": "risk_scale_lifecycle_train_start", "device": str(device), "seq_shape": list(data.seq.shape), "data_diag": data_diag}, ensure_ascii=False), flush=True)
    model, train_diag = lifecycle._train(data, device=device, steps=int(args.steps), batch_size=int(args.batch_size), lr=float(args.lr), class_balance_actor=bool(args.class_balance_actor))
    common = dict(seq_len=int(args.seq_len), fee=fee, slip=slip, cost_mult=float(args.cost_mult), device=device, disable_resize=bool(args.disable_resize), disable_reverse=bool(args.disable_reverse), select_mode=str(args.select_mode), force_parent_entry=False, force_entry_mult=1.0)
    val = lifecycle._replay(frames, "val", model, norm, **common)
    oos = lifecycle._replay(frames, "oos", model, norm, **common)
    with (out_dir / "risk_scale_selector.pkl").open("wb") as f:
        pickle.dump({"model": selector, "normalizer": selector_norm, "state_columns": list(selector_norm["columns"]), "scale_profiles": SCALE_PROFILES, "base_risk": BASE_RISK}, f)
    torch.save({"model_state_dict": model.state_dict(), "normalizer": norm, "seq_len": int(args.seq_len), "state_columns": state_cols, "action_names": lifecycle.ACTION_NAMES}, out_dir / "lifecycle_controller.pt")
    report = {
        "model_id": MODEL_ID,
        "design": "Exit Head feature-only + Mamba lifecycle baseline with learned risk scale selector over baseline-centered TP/SL/leverage/notional profiles.",
        "quality_threshold": float(args.quality_threshold),
        "selector": {"kind": str(args.selector_kind), "profiles": SCALE_PROFILES, **selector_info},
        "risk_summary": {split: _risk_summary(frames[f"{split}_dec"]) for split in ("train", "val", "oos")},
        "state_columns": state_cols,
        "training": {"data_diag": data_diag, "train_diag": train_diag, "min_action_edge": float(args.min_action_edge), "steps": int(args.steps), "class_balance_actor": bool(args.class_balance_actor)},
        "results": {"validation": val, "oos": oos},
        "artifacts": {"out_dir": str(out_dir), "report": str(out_dir / "report.json"), "selector": str(out_dir / "risk_scale_selector.pkl"), "model": str(out_dir / "lifecycle_controller.pt")},
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out_dir / "report.json"), "results": report["results"]}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
