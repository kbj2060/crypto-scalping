#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, predict_policy_frame, prepare_features  # noqa: E402
from scripts import eval_alpha1_l2_execution_replay_20260513 as l2  # noqa: E402
from scripts import eval_alpha1_teacher_constrained_deep_parent_20260513 as teacher  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _feature_cols, _read  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_deep_entry_parent_lite_v38 import DeepEntryParentLite, _apply_norm, _normalizer  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402


MODEL_ID = "alpha2_stabilized_teacher_l2_20260514"
OUT_DIR = ROOT / "data/ensemble/supervised/alpha2_stabilized_teacher_l2_20260514"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha2_stabilized_teacher_l2_20260514_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/alpha2_stabilized_teacher_l2_20260514_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/alpha2_stabilized_teacher_l2_20260514_grid.csv"


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1.get("trades", 0)) < 20:
        return -1e9 + float(c1.get("pnl", 0.0))
    return float(c1["pnl"] + 0.45 * c2["pnl"] + 0.30 * c3["pnl"] - 0.35 * abs(c1["mdd"]))


def _bucket_labels(dec: pd.DataFrame, buckets: tuple[float, ...]) -> np.ndarray:
    vals = pd.to_numeric(dec["notional_exposure"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    b = np.asarray(buckets, dtype=np.float64)
    return np.argmin(np.abs(vals[:, None] - b[None, :]), axis=1).astype(np.int64)


def _seq_tensor_fast(features: pd.DataFrame, cols: list[str]) -> np.ndarray:
    arr = features.loc[:, cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    seq_len = 72
    pad = np.zeros((seq_len - 1, arr.shape[1]), dtype=np.float32)
    padded = np.vstack([pad, arr])
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=seq_len, axis=0)
    if windows.shape[1] == arr.shape[1]:
        windows = windows.transpose(0, 2, 1)
    return np.ascontiguousarray(windows)


def _loss_parts(
    model: DeepEntryParentLite,
    xb: torch.Tensor,
    ab: torch.Tensor,
    qb: torch.Tensor,
    nb: torch.Tensor,
    *,
    ce_action: nn.Module,
    ce_size: nn.Module,
    huber: nn.Module,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits, qhat, nlogits = model(xb)
    active = ab != ACTION_CASH
    action_loss = ce_action(logits, ab)
    quality_loss = huber(qhat, qb)
    size_loss = ce_size(nlogits[active], nb[active]) if torch.any(active) else torch.tensor(0.0, device=xb.device)
    loss = action_loss + 0.7 * quality_loss + 0.15 * size_loss
    with torch.no_grad():
        acc = float((torch.argmax(logits, dim=1) == ab).float().mean().detach().cpu())
    return loss, {
        "loss": float(loss.detach().cpu()),
        "action_loss": float(action_loss.detach().cpu()),
        "quality_loss": float(quality_loss.detach().cpu()),
        "size_loss": float(size_loss.detach().cpu()),
        "action_acc": acc,
    }


def _eval_teacher_loss(
    model: DeepEntryParentLite,
    loader: DataLoader,
    *,
    ce_action: nn.Module,
    ce_size: nn.Module,
    huber: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {"loss": 0.0, "action_loss": 0.0, "quality_loss": 0.0, "size_loss": 0.0, "action_acc": 0.0}
    n = 0
    with torch.no_grad():
        for xb, ab, qb, nb in loader:
            xb, ab, qb, nb = xb.to(device), ab.to(device), qb.to(device), nb.to(device)
            _, parts = _loss_parts(model, xb, ab, qb, nb, ce_action=ce_action, ce_size=ce_size, huber=huber)
            bs = int(xb.shape[0])
            n += bs
            for k in sums:
                sums[k] += parts[k] * bs
    return {k: v / max(n, 1) for k, v in sums.items()}


def _train_long_teacher(
    seq: np.ndarray,
    action: np.ndarray,
    quality: np.ndarray,
    notional: np.ndarray,
    *,
    val_seq: np.ndarray | None = None,
    val_action: np.ndarray | None = None,
    val_quality: np.ndarray | None = None,
    val_notional: np.ndarray | None = None,
    n_buckets: int,
    max_epochs: int = 120,
    inner_val_frac: float = 0.15,
    patience: int = 14,
    lr_patience: int = 4,
    lr_factor: float = 0.5,
    min_lr: float = 2e-5,
) -> tuple[DeepEntryParentLite, dict[str, Any]]:
    torch.manual_seed(20260514)
    np.random.seed(20260514)
    n = int(len(seq))
    if val_seq is None:
        split = max(512, min(n - 512, int(n * (1.0 - float(inner_val_frac)))))
        norm = _normalizer(seq[:split])
        x = _apply_norm(seq, norm)
        x_train, x_val = x[:split], x[split:]
        a_train, a_val = action[:split], action[split:]
        q_train, q_val = quality[:split], quality[split:]
        n_train, n_val = notional[:split], notional[split:]
        val_scope = "last_train_15pct"
    else:
        assert val_action is not None and val_quality is not None and val_notional is not None
        norm = _normalizer(seq)
        x_train = _apply_norm(seq, norm)
        x_val = _apply_norm(val_seq, norm)
        a_train, a_val = action, val_action
        q_train, q_val = quality, val_quality
        n_train, n_val = notional, val_notional
        val_scope = "2025Q4_selection_teacher_labels"

    device = _device()
    model = DeepEntryParentLite(x_train.shape[-1], notional_classes=int(n_buckets)).to(device)
    counts = np.bincount(a_train, minlength=3).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights[0] *= 0.25
    weights = weights / max(float(weights.mean()), 1e-6)
    ce_action = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device), label_smoothing=0.03)
    ce_size = nn.CrossEntropyLoss()
    huber = nn.SmoothL1Loss()
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train),
            torch.from_numpy(a_train.astype(np.int64)),
            torch.from_numpy(q_train.astype(np.float32)),
            torch.from_numpy(n_train.astype(np.int64)),
        ),
            batch_size=512,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_val),
            torch.from_numpy(a_val.astype(np.int64)),
            torch.from_numpy(q_val.astype(np.float32)),
            torch.from_numpy(n_val.astype(np.int64)),
        ),
        batch_size=1024,
        shuffle=False,
        drop_last=False,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=float(lr_factor),
        patience=max(1, int(lr_patience)),
        min_lr=float(min_lr),
        threshold=1e-3,
        threshold_mode="rel",
    )
    best_loss = float("inf")
    best_epoch = 0
    bad_count = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    print(
        f"[{MODEL_ID}] long-train cfg epochs={max_epochs} patience={patience} "
        f"lr_patience={lr_patience} lr_factor={lr_factor} min_lr={min_lr:.1e} "
        f"train={len(x_train)} val={len(x_val)} val_scope={val_scope} device={device}",
        flush=True,
    )
    for ep in range(1, int(max_epochs) + 1):
        model.train()
        train_sum = 0.0
        batches = 0
        for xb, ab, qb, nb in train_loader:
            xb, ab, qb, nb = xb.to(device), ab.to(device), qb.to(device), nb.to(device)
            loss, _ = _loss_parts(model, xb, ab, qb, nb, ce_action=ce_action, ce_size=ce_size, huber=huber)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_sum += float(loss.detach().cpu())
            batches += 1
        val = _eval_teacher_loss(model, val_loader, ce_action=ce_action, ce_size=ce_size, huber=huber, device=device)
        prev_lr = float(opt.param_groups[0]["lr"])
        scheduler.step(float(val["loss"]))
        new_lr = float(opt.param_groups[0]["lr"])
        improved = float(val["loss"]) < best_loss * (1.0 - 1e-4)
        if improved:
            best_loss = float(val["loss"])
            best_epoch = int(ep)
            bad_count = 0
            best_state = copy.deepcopy(model.cpu().state_dict())
            model.to(device)
        else:
            bad_count += 1
        rec = {
            "epoch": int(ep),
            "train_loss": train_sum / max(batches, 1),
            "val_loss": float(val["loss"]),
            "val_action_acc": float(val["action_acc"]),
            "lr": float(new_lr),
            "bad_count": int(bad_count),
            "best_epoch": int(best_epoch),
        }
        history.append(rec)
        if ep == 1 or ep % 10 == 0 or improved or new_lr < prev_lr:
            lr_msg = f" lr_drop={prev_lr:.2e}->{new_lr:.2e}" if new_lr < prev_lr else ""
            print(
                f"[{MODEL_ID}] epoch={ep:03d} train={rec['train_loss']:.5f} "
                f"val={rec['val_loss']:.5f} acc={rec['val_action_acc']:.4f} "
                f"best_ep={best_epoch} bad={bad_count} lr={new_lr:.2e}{lr_msg}",
                flush=True,
            )
        if int(patience) > 0 and bad_count >= int(patience):
            print(f"[{MODEL_ID}] early_stop epoch={ep} best_epoch={best_epoch} best_val={best_loss:.5f}", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu().eval(), {
        "norm": norm,
        "label_counts": {str(i): int(v) for i, v in enumerate(np.bincount(action, minlength=3))},
        "train_inner_rows": int(len(x_train)),
        "val_inner_rows": int(len(x_val)),
        "validation_scope": val_scope,
        "max_epochs": int(max_epochs),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_loss),
        "history_tail": history[-25:],
        "lr_scheduler": {
            "type": "ReduceLROnPlateau",
            "mode": "min",
            "factor": float(lr_factor),
            "patience": int(lr_patience),
            "min_lr": float(min_lr),
        },
        "early_stop_patience": int(patience),
    }


def _metrics(
    df: pd.DataFrame,
    parent: dict[str, Any],
    jackpot_model: dict[str, Any],
    add_cfg: CostRunnerConfig,
    q: np.ndarray,
    decisions: pd.DataFrame,
    variant: Any,
    *,
    fee: float,
    slip: float,
) -> dict[str, Any]:
    return {
        f"cost{mult}": l2._run_with_l2_proxy(
            df,
            parent,
            jackpot_model,
            add_cfg,
            q,
            decisions,
            variant,
            fee,
            slip,
            cost_mult=float(mult),
        )
        for mult in (1, 2, 3)
    }


def main() -> int:
    print(f"[{MODEL_ID}] loading stack", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    parent = joblib.load(v31.DEFAULT_PARENT)
    jackpot_payload = joblib.load(v31.DEFAULT_JACKPOT)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = v31._load_v27(v31.DEFAULT_V27)
    base = dict(parent["config"])
    fee = float(base["fee"])
    slip = float(base["slip"])
    buckets = tuple(base.get("notional_buckets", (0.23, 0.368, 0.575, 0.8625, 1.2075, 1.6675, 2.3, 3.105, 4.14)))

    train_all = _read(v31.DEFAULT_TRAIN)
    eval_df = _read(v31.DEFAULT_EVAL)
    train = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    feature_cols = _feature_cols(train_all, eval_df)
    parent_audit = _audit_contract(train_all, eval_df, list(parent.get("feature_cols") or []))
    l2_stats = l2._live_l2_stats()

    print(f"[{MODEL_ID}] parent decisions", flush=True)
    train_dec = predict_policy_frame(parent, train, close=_close(train))
    val_dec = predict_policy_frame(parent, val, close=_close(val))
    eval_dec = predict_policy_frame(parent, eval_df, close=_close(eval_df))

    print(f"[{MODEL_ID}] building sequence tensors", flush=True)
    train_features = prepare_features(train, side_hint=0, close=_close(train), feature_cols=feature_cols)
    val_features_for_training = prepare_features(val, side_hint=0, close=_close(val), feature_cols=feature_cols)
    train_seq = _seq_tensor_fast(train_features, feature_cols)
    val_seq_for_training = _seq_tensor_fast(val_features_for_training, feature_cols)
    y_action = train_dec["action"].astype(int).to_numpy(dtype=np.int64)
    y_quality = pd.to_numeric(train_dec["quality_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    y_notional = _bucket_labels(train_dec, buckets)
    y_val_action = val_dec["action"].astype(int).to_numpy(dtype=np.int64)
    y_val_quality = pd.to_numeric(val_dec["quality_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    y_val_notional = _bucket_labels(val_dec, buckets)

    model, train_meta = _train_long_teacher(
        train_seq,
        y_action,
        y_quality,
        y_notional,
        val_seq=val_seq_for_training,
        val_action=y_val_action,
        val_quality=y_val_quality,
        val_notional=y_val_notional,
        n_buckets=len(buckets),
        max_epochs=120,
        inner_val_frac=0.15,
        patience=14,
        lr_patience=4,
        lr_factor=0.5,
        min_lr=2e-5,
    )

    print(f"[{MODEL_ID}] predicting teacher and V27", flush=True)
    val_features = prepare_features(val, side_hint=0, close=_close(val), feature_cols=feature_cols)
    eval_features = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=feature_cols)
    val_pred = teacher._predict_deep(model, val_features, feature_cols, train_meta["norm"])
    eval_pred = teacher._predict_deep(model, eval_features, feature_cols, train_meta["norm"])
    val_q = v31._predict_all(v27_model, val, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])

    print(f"[{MODEL_ID}] selecting runtime and L2 variant on 2025Q4", flush=True)
    replay_variants = [v for v in l2._variants() if v.name != "alpha1_taker_baseline"]
    rows: list[dict[str, Any]] = []
    selected_runtime: teacher.Runtime | None = None
    selected_variant: Any | None = None
    best_score = -1e18
    for runtime in teacher._grid():
        dec = teacher._constrained_decisions(val_dec, val_pred, buckets, runtime)
        for variant in replay_variants:
            vm = _metrics(val, parent, jackpot_model, add_cfg, val_q, dec, variant, fee=fee, slip=slip)
            score = _score(vm["cost1"], vm["cost2"], vm["cost3"])
            rows.append(
                {
                    "runtime": runtime.name,
                    "variant": variant.name,
                    "selection_score": score,
                    "val_cost1_pnl": vm["cost1"]["pnl"],
                    "val_cost1_mdd": vm["cost1"]["mdd"],
                    "val_cost1_trades": vm["cost1"]["trades"],
                    "val_cost2_pnl": vm["cost2"]["pnl"],
                    "val_cost3_pnl": vm["cost3"]["pnl"],
                    "runtime_config": asdict(runtime),
                    "variant_config": asdict(variant),
                }
            )
            if score > best_score:
                best_score = score
                selected_runtime = runtime
                selected_variant = variant
                print(
                    f"[{MODEL_ID}] new best runtime={runtime.name} variant={variant.name} "
                    f"score={score:.2f} c1={vm['cost1']['pnl']:.2f} c2={vm['cost2']['pnl']:.2f} c3={vm['cost3']['pnl']:.2f}",
                    flush=True,
                )
    assert selected_runtime is not None and selected_variant is not None
    pd.DataFrame(rows).sort_values("selection_score", ascending=False).to_csv(GRID_OUT, index=False)

    print(f"[{MODEL_ID}] evaluating fixed 2026 OOS", flush=True)
    teacher_eval_dec = teacher._constrained_decisions(eval_dec, eval_pred, buckets, selected_runtime)
    baseline_variant = l2._variants()[0]
    experiments: list[dict[str, Any]] = []
    for name, decisions, variant in (
        ("alpha1_taker_baseline", eval_dec, baseline_variant),
        ("alpha1_l2_replay", eval_dec, selected_variant),
        (f"longtrain_teacher_l2::{selected_runtime.name}::{selected_variant.name}", teacher_eval_dec, selected_variant),
    ):
        metrics = _metrics(eval_df, parent, jackpot_model, add_cfg, eval_q, decisions, variant, fee=fee, slip=slip)
        experiments.append(
            {
                "name": name,
                "runtime": asdict(selected_runtime) if name.startswith("longtrain_teacher_l2") else None,
                "variant": asdict(variant),
                "metrics": metrics,
                "score": _score(metrics["cost1"], metrics["cost2"], metrics["cost3"]),
            }
        )
        print(
            f"[{MODEL_ID}] {name} cost1={metrics['cost1']['pnl']:.2f} mdd={metrics['cost1']['mdd']:.2f} "
            f"cost2={metrics['cost2']['pnl']:.2f} cost3={metrics['cost3']['pnl']:.2f}",
            flush=True,
        )

    model_path = OUT_DIR / "teacher_deep_parent_l2_longtrain.pt"
    torch.save(
        {
            "model_id": MODEL_ID,
            "state_dict": model.state_dict(),
            "feature_cols": feature_cols,
            "train_meta": train_meta,
            "selected_runtime": asdict(selected_runtime),
            "selected_variant": asdict(selected_variant),
            "buckets": buckets,
        },
        model_path,
    )
    best = max(experiments, key=lambda x: x["score"])
    alpha2_reference = {
        "cost1": {"pnl": 699.1379839727641, "mdd": -29.72199717591575},
        "cost2": {"pnl": 463.5399506168309},
        "cost3": {"pnl": 420.8033028238044},
    }
    blocking = list(parent_audit.get("blocking", []))
    warnings = list(parent_audit.get("warnings", []))
    if not l2_stats.get("usable_for_replay", False):
        warnings.append("historical_l2_snapshots_insufficient_conservative_ohlc_replay_only")
    warnings.append("real_live_l2_fill_model_requires_forward_shadow_collection")
    if not best["name"].startswith("longtrain_teacher_l2"):
        warnings.append("selected_best_is_not_longtrain_teacher_l2")
    else:
        if best["metrics"]["cost1"]["pnl"] <= alpha2_reference["cost1"]["pnl"]:
            warnings.append("longtrain_teacher_did_not_beat_alpha2_cost1")
        if best["metrics"]["cost2"]["pnl"] <= alpha2_reference["cost2"]["pnl"]:
            warnings.append("longtrain_teacher_did_not_beat_alpha2_cost2")
        if best["metrics"]["cost3"]["pnl"] <= alpha2_reference["cost3"]["pnl"]:
            warnings.append("longtrain_teacher_did_not_beat_alpha2_cost3")
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": "shadow_collect_l2" if not blocking else "fail",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "train_window": "2025-01-01..2025-09-30",
        "inner_validation_window": "2025-10-01..2025-12-31 teacher labels for scheduler/early stop",
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS only after selection",
        "training_techniques": {
            "max_epochs": 120,
            "early_stop_patience": 14,
            "lr_scheduler": "ReduceLROnPlateau(mode=min,factor=0.5,patience=4,min_lr=2e-5)",
            "learning_rate": 0.0002,
            "weight_decay": 0.0005,
            "action_label_smoothing": 0.03,
            "quality_loss_weight": 0.7,
            "size_loss_weight": 0.15,
            "gradient_clip_norm": 1.0,
            "best_checkpoint": "inner validation loss",
        },
        "train_meta": train_meta,
        "live_l2_stats": l2_stats,
        "selected_runtime": asdict(selected_runtime),
        "selected_variant": asdict(selected_variant),
        "parent_audit": parent_audit,
        "red_team_note": "Longer teacher training is still a shadow candidate because L2 replay requires live forward fill validation.",
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Alpha2 stabilized teacher experiment. The HGB parent, V27 scout, V21.2 jackpot runner, V31 exit, and L2 replay layer are fixed. The teacher verifier is retrained with DSAC-style early stop, ReduceLROnPlateau, best checkpoint restoration, gradient clipping, lower LR, stronger weight decay, label smoothing, and lower auxiliary loss weights.",
        "experiments": experiments,
        "audit": audit,
        "artifacts": {
            "model": str(model_path),
            "report": str(REPORT_OUT),
            "audit": str(AUDIT_OUT),
            "grid": str(GRID_OUT),
        },
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "best": best["name"], "verdict": audit["verdict"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
