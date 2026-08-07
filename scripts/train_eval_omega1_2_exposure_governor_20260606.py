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
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_post_lifecycle_bucket_adapter_20260605 as base  # noqa: E402


MODEL_ID = "omega1_2_exposure_governor_20260606"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
DEFAULT_BASE_ADAPTER = (
    ROOT
    / "tmp/causal_regen_20260516"
    / "omega1_2_post_lifecycle_bucket_adapter_20260605_hgb_base_nogate_traink3_replayk2_s260693"
    / "post_bucket_adapter.pkl"
)


def _seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))


def _load_adapter(path: Path) -> dict[str, Any]:
    if not Path(path).exists():
        raise RuntimeError(f"baseline adapter missing: {path}")
    with Path(path).open("rb") as f:
        artifact = pickle.load(f)
    if "models" not in artifact or "normalizer" not in artifact:
        raise RuntimeError("baseline adapter artifact contract mismatch")
    return artifact


def _scaled_risk(ids: np.ndarray, scale: float, cap: float, *, compensate_sltp: bool) -> dict[str, float]:
    r = base._risk_from_ids(ids)
    old_notional = max(float(r["notional"]), 1e-8)
    notional = float(np.clip(old_notional * float(scale), 0.0, float(cap)))
    tp = float(r["tp"])
    sl = abs(float(r["sl"]))
    if bool(compensate_sltp):
        mult = notional / old_notional
        tp *= mult
        sl *= mult
    return {
        "tp": tp,
        "sl": sl,
        "notional": notional,
        "leverage": float(r["leverage"]),
        "margin": float(r["margin_notional"]),
    }


def _enter_scaled(
    cash: float,
    arrays: dict[str, np.ndarray],
    dec: pd.DataFrame,
    i: int,
    risk: dict[str, float],
    *,
    fee_eff: float,
    slip_eff: float,
) -> tuple[float, base.Position, str]:
    row = dec.iloc[int(i)]
    side = int(row.get("side", 0) or 0)
    if side == 0 or int(row.get("action", 0) or 0) == base.omega.ACTION_CASH:
        return cash, base.Position(), "no_signal"
    filled, entry_px, entry_fee, _route = base.omega._try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
    if not filled:
        return cash, base.Position(), "entry_miss"
    notional = float(risk["notional"])
    cash -= cash * float(entry_fee) * notional
    return cash, base.Position(side=side, entry_price=float(entry_px), entry_i=min(int(i) + 1, len(arrays["close"]) - 1), notional=notional, take_profit=float(risk["tp"]), stop_loss=abs(float(risk["sl"]))), "entry"


def _simulate_scaled(
    arrays: dict[str, np.ndarray],
    dec: pd.DataFrame,
    i: int,
    ids: np.ndarray,
    scale: float,
    *,
    cap: float,
    compensate_sltp: bool,
    fee: float,
    slip: float,
    cost_mult: float,
    max_bars: int,
) -> tuple[float, str]:
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    risk = _scaled_risk(ids, scale, cap, compensate_sltp=compensate_sltp)
    cash, pos, reason = _enter_scaled(1.0, arrays, dec, int(i), risk, fee_eff=fee_eff, slip_eff=slip_eff)
    if pos.side != 0:
        cash, reason = base._continue_to_end(cash, arrays, pos, max(int(i) + 1, pos.entry_i), fee_eff=fee_eff, slip_eff=slip_eff, max_bars=max_bars)
    return float(cash - 1.0), reason


def _make_model(kind: str, seed: int) -> Any:
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_iter=160, learning_rate=0.04, max_leaf_nodes=5, l2_regularization=1.5, min_samples_leaf=35, random_state=int(seed))
    if kind == "extratrees":
        return ExtraTreesClassifier(n_estimators=220, max_depth=7, min_samples_leaf=18, random_state=int(seed), n_jobs=-1)
    raise RuntimeError(f"unknown governor kind: {kind}")


def _build_exposure_labels(
    x_train: pd.DataFrame,
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    entry_idx: np.ndarray,
    baseline_artifact: dict[str, Any],
    *,
    scales: np.ndarray,
    cap: float,
    compensate_sltp: bool,
    min_edge: float,
    fee: float,
    slip: float,
    cost_mult: float,
    max_bars: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    arrays = base._arrays(frame)
    y: list[int] = []
    w: list[float] = []
    reasons: dict[str, int] = {}
    selected_scores: list[float] = []
    baseline_scores: list[float] = []
    for row_i, i in enumerate(entry_idx):
        feat = x_train.iloc[[int(row_i)]]
        ids_arr, _meta = base._predict_hgb_ids(baseline_artifact["models"], feat, baseline_artifact["normalizer"])
        ids = ids_arr[0]
        scores = []
        local_reasons = []
        for scale in scales:
            score, reason = _simulate_scaled(arrays, dec, int(i), ids, float(scale), cap=cap, compensate_sltp=compensate_sltp, fee=fee, slip=slip, cost_mult=cost_mult, max_bars=max_bars)
            scores.append(float(score))
            local_reasons.append(str(reason))
            reasons[reason] = reasons.get(reason, 0) + 1
        scores_np = np.asarray(scores, dtype=np.float64)
        base_score = float(scores_np[0])
        best_idx = int(np.argmax(scores_np))
        if float(scores_np[best_idx]) <= base_score + float(min_edge):
            best_idx = 0
        y.append(best_idx)
        selected_scores.append(float(scores_np[best_idx]))
        baseline_scores.append(base_score)
        spread = max(float(np.std(scores_np)), 1e-4)
        w.append(float(np.exp(np.clip((scores_np[best_idx] - np.median(scores_np)) / spread, -3.0, 3.0))))
    return np.asarray(y, dtype=np.int64), np.asarray(w, dtype=np.float32), np.asarray(baseline_scores, dtype=np.float32), {
        "rows": int(len(y)),
        "scale_counts": {str(i): int(v) for i, v in enumerate(np.bincount(np.asarray(y, dtype=np.int64), minlength=len(scales)))},
        "exit_reasons": reasons,
        "selected_score_mean": float(np.mean(selected_scores)) if selected_scores else 0.0,
        "baseline_score_mean": float(np.mean(baseline_scores)) if baseline_scores else 0.0,
        "edge_mean": float(np.mean(np.asarray(selected_scores) - np.asarray(baseline_scores))) if selected_scores else 0.0,
    }


def _predict_scale_id(model: Any, feat: pd.DataFrame, norm: dict[str, Any], *, min_prob: float, min_margin: float) -> tuple[int, dict[str, float]]:
    x = base._apply_norm(feat, norm)
    pred = int(np.asarray(model.predict(x), dtype=np.int64).reshape(-1)[0])
    meta = {"prob": 1.0, "margin": 1.0}
    if hasattr(model, "predict_proba"):
        probs = np.asarray(model.predict_proba(x), dtype=np.float64).reshape(-1)
        classes = np.asarray(model.classes_, dtype=np.int64)
        order = np.argsort(probs)[::-1]
        prob = float(probs[order[0]]) if len(order) else 0.0
        margin = float(probs[order[0]] - probs[order[1]]) if len(order) > 1 else prob
        pred = int(classes[order[0]]) if len(order) else pred
        meta = {"prob": prob, "margin": margin}
        if pred > 0 and (prob < float(min_prob) or margin < float(min_margin)):
            pred = 0
    return pred, meta


def _replay_governor(
    frames: dict[str, Any],
    split: str,
    lifecycle_model: base.lifecycle.MambaDiscreteActorCritic,
    lifecycle_ckpt: dict[str, Any],
    baseline_artifact: dict[str, Any],
    governor: Any,
    governor_norm: dict[str, Any],
    *,
    scales: np.ndarray,
    cap: float,
    compensate_sltp: bool,
    min_prob: float,
    min_margin: float,
    fee: float,
    slip: float,
    cost_mult: float,
    device: Any,
    select_mode: str,
    replay_enter_topk: int,
) -> dict[str, Any]:
    frame = frames[f"{split}_df"]
    state = base.lifecycle._base_state(frames[f"s_{split}"])
    dec = frames[f"{split}_dec"]
    arrays = base._arrays(frame)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    base_norm = base.lifecycle._apply_norm(state, lifecycle_ckpt["normalizer"])
    base_seq = base.lifecycle._rolling_sequences(base_norm, int(lifecycle_ckpt["seq_len"]))
    active = base.omega._active(dec)
    cash = peak = 1.0
    mdd = 0.0
    pos = base.Position()
    lifecycle_pos = base.lifecycle.Position()
    trades = wins = long_entries = short_entries = 0
    reasons: dict[str, int] = {}
    scale_counts: dict[str, int] = {}
    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            vals = base._position_values(arrays, pos, i, slip_eff=slip_eff)
            pos.mfe = max(pos.mfe, vals["lc_pos_unrealized"])
            pos.mae = min(pos.mae, vals["lc_pos_unrealized"])
            eq = cash * (1.0 + vals["lc_pos_unrealized"])
            if pos.stop_loss > 0.0 and vals["lc_pos_unrealized"] <= -pos.stop_loss:
                before = cash
                cash, pos, _ = base._realize_fraction(cash, arrays, pos, i, 1.0, fee_eff=fee_eff, slip_eff=slip_eff)
                lifecycle_pos = base.lifecycle.Position()
                trades += 1
                wins += int(cash > before)
                reasons["stop_loss"] = reasons.get("stop_loss", 0) + 1
                continue
            if pos.take_profit > 0.0 and vals["lc_pos_unrealized"] >= pos.take_profit:
                before = cash
                cash, pos, _ = base._realize_fraction(cash, arrays, pos, i, 1.0, fee_eff=fee_eff, slip_eff=slip_eff)
                lifecycle_pos = base.lifecycle.Position()
                trades += 1
                wins += int(cash > before)
                reasons["take_profit"] = reasons.get("take_profit", 0) + 1
                continue
        else:
            eq = cash
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if pos.side == 0 and not bool(active[i]):
            continue
        lc_row = base.lifecycle._state_row(state, arrays, lifecycle_pos, i, slip_eff=slip_eff)
        allowed = base.lifecycle._allowed_actions(arrays, dec, lifecycle_pos, i, slip_eff=slip_eff, disable_resize=True, disable_reverse=True)
        scores = base._lifecycle_scores(lifecycle_model, lifecycle_ckpt, base_seq, lc_row, allowed, i, device=device, select_mode=select_mode)
        lc_action = int(np.argmax(scores))
        if lifecycle_pos.side == 0 and lc_action not in (base.lifecycle.ENTER_BASE, base.lifecycle.ENTER_AGGRESSIVE):
            top_actions = np.argsort(scores)[::-1][: max(int(replay_enter_topk), 1)]
            enter_scores = [(base.lifecycle.ENTER_BASE, float(scores[base.lifecycle.ENTER_BASE])), (base.lifecycle.ENTER_AGGRESSIVE, float(scores[base.lifecycle.ENTER_AGGRESSIVE]))]
            enter_scores = [(a, s) for a, s in enter_scores if np.isfinite(s) and s > -1e8 and int(a) in set(int(x) for x in top_actions)]
            if not enter_scores:
                reasons["skip"] = reasons.get("skip", 0) + 1
                continue
            lc_action = int(max(enter_scores, key=lambda x: x[1])[0])
            reasons["topk_enter_candidate"] = reasons.get("topk_enter_candidate", 0) + 1
        if lifecycle_pos.side == 0:
            feat = base._adapter_feature_row(lc_row, lc_action)
            ids_arr, _meta = base._predict_hgb_ids(baseline_artifact["models"], feat, baseline_artifact["normalizer"])
            scale_id, scale_meta = _predict_scale_id(governor, feat, governor_norm, min_prob=min_prob, min_margin=min_margin)
            scale = float(scales[int(scale_id)])
            if int(scale_id) == 0 and scale_meta["prob"] < float(min_prob):
                reasons["scale_conf_fallback"] = reasons.get("scale_conf_fallback", 0) + 1
            risk = _scaled_risk(ids_arr[0], scale, cap, compensate_sltp=compensate_sltp)
            before = cash
            cash, pos, reason = _enter_scaled(cash, arrays, dec, i, risk, fee_eff=fee_eff, slip_eff=slip_eff)
            lifecycle_pos = base._to_lifecycle_position(pos)
            reasons[reason] = reasons.get(reason, 0) + 1
            if reason == "entry":
                long_entries += int(pos.side > 0)
                short_entries += int(pos.side < 0)
                scale_counts[str(scale)] = scale_counts.get(str(scale), 0) + 1
            continue
        if lc_action == base.lifecycle.FULL_EXIT:
            before = cash
            cash, pos, _ = base._realize_fraction(cash, arrays, pos, i, 1.0, fee_eff=fee_eff, slip_eff=slip_eff)
            lifecycle_pos = base.lifecycle.Position()
            trades += 1
            wins += int(cash > before)
            reasons["full_exit"] = reasons.get("full_exit", 0) + 1
        elif lc_action == base.lifecycle.REDUCE50:
            cash, pos, _ = base._realize_fraction(cash, arrays, pos, i, 0.5, fee_eff=fee_eff, slip_eff=slip_eff)
            lifecycle_pos = base._to_lifecycle_position(pos)
            reasons["reduce50"] = reasons.get("reduce50", 0) + 1
        else:
            reasons["hold"] = reasons.get("hold", 0) + 1
    if pos.side != 0:
        before = cash
        cash, pos, _ = base._realize_fraction(cash, arrays, pos, len(frame) - 1, 1.0, fee_eff=fee_eff, slip_eff=slip_eff)
        trades += 1
        wins += int(cash > before)
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / trades) if trades else 0.0,
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "reasons": reasons,
        "scale_counts": scale_counts,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threehead-dir", type=Path, default=base.feat_coord.DEFAULT_3HEAD_DIR)
    ap.add_argument("--baseline-lifecycle-dir", type=Path, default=base.BASELINE_LIFECYCLE_DIR)
    ap.add_argument("--baseline-adapter-path", type=Path, default=DEFAULT_BASE_ADAPTER)
    ap.add_argument("--governor-kind", choices=["hgb", "extratrees"], default="hgb")
    ap.add_argument("--quality-threshold", type=float, default=0.75)
    ap.add_argument("--max-label-rows", type=int, default=0)
    ap.add_argument("--scales", default="1.0,1.15,1.30,1.55,1.85,2.20")
    ap.add_argument("--notional-cap", type=float, default=1.0)
    ap.add_argument("--compensate-sltp-by-notional", action="store_true")
    ap.add_argument("--min-edge", type=float, default=0.001)
    ap.add_argument("--min-prob", type=float, default=0.0)
    ap.add_argument("--min-margin", type=float, default=0.0)
    ap.add_argument("--train-max-sim-bars", type=int, default=96)
    ap.add_argument("--enter-topk", type=int, default=3)
    ap.add_argument("--replay-enter-topk", type=int, default=2)
    ap.add_argument("--select-mode", choices=["actor_q", "q_only"], default="actor_q")
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=261000)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    _seed(int(args.seed))
    device = base._device(str(args.device))
    scales = np.asarray([float(x) for x in str(args.scales).split(",") if str(x).strip()], dtype=np.float32)
    if len(scales) < 1 or abs(float(scales[0]) - 1.0) > 1e-8:
        raise RuntimeError("scales must start with 1.0 baseline scale")
    out_dir = OUT_DIR if not str(args.out_suffix).strip() else OUT_DIR.parent / f"{MODEL_ID}_{str(args.out_suffix).strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_artifact = _load_adapter(Path(args.baseline_adapter_path))
    frames = base._base_frames(Path(args.threehead_dir), float(args.quality_threshold), device)
    lifecycle_model, lifecycle_ckpt = base._load_baseline_lifecycle(Path(args.baseline_lifecycle_dir))
    bad = [c for c in lifecycle_ckpt["state_columns"] if "clean_regime4" in c or "regime4_pred" in c or "tp_sl_action_score" in c or str(c).startswith("teacher_")]
    if bad:
        raise RuntimeError(f"forbidden lifecycle state columns passed audit: {bad[:20]}")
    fee, slip = base.omega._load_fee_slip()
    x_train, entry_idx, _lc_actions, collect_diag = base._collect_train_entries(
        frames,
        lifecycle_model,
        lifecycle_ckpt,
        device=device,
        select_mode=str(args.select_mode),
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        max_rows=int(args.max_label_rows),
        enter_topk=int(args.enter_topk),
    )
    y, w, _base_scores, label_diag = _build_exposure_labels(
        x_train,
        frames["train_df"],
        frames["train_dec"],
        entry_idx,
        baseline_artifact,
        scales=scales,
        cap=float(args.notional_cap),
        compensate_sltp=bool(args.compensate_sltp_by_notional),
        min_edge=float(args.min_edge),
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        max_bars=int(args.train_max_sim_bars),
    )
    x_arr, governor_norm = base._fit_norm(x_train)
    governor = _make_model(str(args.governor_kind), int(args.seed))
    governor.fit(x_arr, y, sample_weight=w)
    train_pred = np.asarray(governor.predict(x_arr), dtype=np.int64).reshape(-1)
    train_diag = {
        "train_acc": float(np.mean(train_pred == y)),
        "pred_counts": {str(i): int(v) for i, v in enumerate(np.bincount(train_pred, minlength=len(scales)))},
    }
    with (out_dir / "exposure_governor.pkl").open("wb") as f:
        pickle.dump({"model": governor, "normalizer": governor_norm, "scales": scales, "args": vars(args)}, f)
    val = _replay_governor(
        frames,
        "val",
        lifecycle_model,
        lifecycle_ckpt,
        baseline_artifact,
        governor,
        governor_norm,
        scales=scales,
        cap=float(args.notional_cap),
        compensate_sltp=bool(args.compensate_sltp_by_notional),
        min_prob=float(args.min_prob),
        min_margin=float(args.min_margin),
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        device=device,
        select_mode=str(args.select_mode),
        replay_enter_topk=int(args.replay_enter_topk),
    )
    oos = _replay_governor(
        frames,
        "oos",
        lifecycle_model,
        lifecycle_ckpt,
        baseline_artifact,
        governor,
        governor_norm,
        scales=scales,
        cap=float(args.notional_cap),
        compensate_sltp=bool(args.compensate_sltp_by_notional),
        min_prob=float(args.min_prob),
        min_margin=float(args.min_margin),
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        device=device,
        select_mode=str(args.select_mode),
        replay_enter_topk=int(args.replay_enter_topk),
    )
    forbidden = [c for c in governor_norm["columns"] if "clean_regime4" in c or "regime4_pred" in c or "tp_sl_action_score" in c or str(c).startswith("teacher_")]
    report = {
        "model_id": MODEL_ID,
        "design": "Exposure-only governor over explicit base_nogate_topk2 adapter. Baseline TP/SL/lifecycle/side are preserved; governor can only scale notional with optional SLTP distance compensation.",
        "baseline_adapter_path": str(args.baseline_adapter_path),
        "governor_kind": str(args.governor_kind),
        "scales": scales.tolist(),
        "notional_cap": float(args.notional_cap),
        "compensate_sltp_by_notional": bool(args.compensate_sltp_by_notional),
        "min_edge": float(args.min_edge),
        "min_prob": float(args.min_prob),
        "min_margin": float(args.min_margin),
        "feature_audit": {"columns": int(len(governor_norm["columns"])), "forbidden_count": int(len(forbidden)), "forbidden": forbidden[:20]},
        "training": {"collect_diag": collect_diag, "label_diag": label_diag, "train_diag": train_diag},
        "results": {"validation": val, "oos": oos},
        "artifacts": {"out_dir": str(out_dir), "model": str(out_dir / "exposure_governor.pkl"), "report": str(out_dir / "report.json")},
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=base._json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out_dir / "report.json"), "results": report["results"]}, ensure_ascii=False, default=base._json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
