#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
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

from ensemble.fully_learned_governor_policy import ACTION_CASH, ACTION_LONG, ACTION_SHORT, predict_policy_frame, prepare_features  # noqa: E402
from scripts import eval_alpha1_rl_exit_and_sizing_20260513 as alpha1  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _feature_cols, _read  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_deep_entry_parent_lite_v38 import DeepEntryParentLite, SEQ_LEN, _apply_norm, _normalizer, _seq_tensor  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402


MODEL_ID = "alpha1_teacher_constrained_deep_parent_20260513"
OUT_DIR = ROOT / "data/ensemble/supervised/alpha1_teacher_constrained_deep_parent_20260513"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha1_teacher_constrained_deep_parent_20260513_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/alpha1_teacher_constrained_deep_parent_20260513_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/alpha1_teacher_constrained_deep_parent_20260513_grid.csv"


@dataclass(frozen=True)
class Runtime:
    name: str
    confidence: float
    skip_on_cash: bool
    allow_flip: bool
    use_learned_size: bool
    notional_scale: float
    max_notional: float


def _grid() -> list[Runtime]:
    rows: list[Runtime] = []
    for conf in (0.38, 0.44, 0.50, 0.56):
        rows.append(Runtime(f"cash_preserve_noflip_c{conf:.2f}", conf, True, False, False, 1.0, 2.75))
        rows.append(Runtime(f"cash_preserve_flip_c{conf:.2f}", conf, True, True, False, 1.0, 2.75))
        rows.append(Runtime(f"cash_preserve_size_c{conf:.2f}", conf, True, False, True, 1.0, 2.75))
    return rows


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _bucket_labels(dec: pd.DataFrame, buckets: tuple[float, ...]) -> np.ndarray:
    vals = pd.to_numeric(dec["notional_exposure"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    b = np.asarray(buckets, dtype=np.float64)
    return np.argmin(np.abs(vals[:, None] - b[None, :]), axis=1).astype(np.int64)


def _train_teacher_model(seq: np.ndarray, action: np.ndarray, quality: np.ndarray, notional: np.ndarray, *, n_buckets: int, epochs: int = 35) -> tuple[DeepEntryParentLite, dict[str, Any]]:
    torch.manual_seed(20260513)
    norm = _normalizer(seq)
    x = _apply_norm(seq, norm)
    device = _device()
    model = DeepEntryParentLite(x.shape[-1], notional_classes=int(n_buckets)).to(device)
    counts = np.bincount(action, minlength=3).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights[0] *= 0.25
    weights = weights / max(float(weights.mean()), 1e-6)
    ce_action = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
    ce_size = nn.CrossEntropyLoss()
    huber = nn.SmoothL1Loss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(action.astype(np.int64)), torch.from_numpy(quality.astype(np.float32)), torch.from_numpy(notional.astype(np.int64))),
        batch_size=256,
        shuffle=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    model.train()
    for ep in range(int(epochs)):
        loss_sum = 0.0
        for xb, ab, qb, nb in loader:
            xb, ab, qb, nb = xb.to(device), ab.to(device), qb.to(device), nb.to(device)
            logits, qhat, nlogits = model(xb)
            active = ab != ACTION_CASH
            loss = ce_action(logits, ab) + 1.2 * huber(qhat, qb)
            if torch.any(active):
                loss = loss + 0.25 * ce_size(nlogits[active], nb[active])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.detach().cpu())
        if ep in {0, int(epochs) - 1} or (ep + 1) % 10 == 0:
            print(f"[{MODEL_ID}] epoch={ep+1} loss={loss_sum/max(len(loader),1):.5f}", flush=True)
    return model.cpu().eval(), {"norm": norm, "label_counts": {str(i): int(v) for i, v in enumerate(counts)}, "epochs": int(epochs)}


def _predict_deep(model: DeepEntryParentLite, features: pd.DataFrame, cols: list[str], norm: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    indices = np.arange(len(features), dtype=np.int64)
    seq = _seq_tensor(features, indices, cols)
    x = _apply_norm(seq, norm)
    device = _device()
    model = model.to(device).eval()
    probs: list[np.ndarray] = []
    qvals: list[np.ndarray] = []
    nprobs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            logits, qhat, nlogits = model(torch.from_numpy(x[start : start + 4096]).to(device))
            probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
            qvals.append(qhat.detach().cpu().numpy())
            nprobs.append(torch.softmax(nlogits, dim=1).detach().cpu().numpy())
    return {"action_proba": np.vstack(probs), "quality": np.concatenate(qvals), "notional_proba": np.vstack(nprobs)}


def _constrained_decisions(teacher: pd.DataFrame, pred: dict[str, np.ndarray], buckets: tuple[float, ...], rt: Runtime) -> pd.DataFrame:
    out = teacher.copy()
    p = np.asarray(pred["action_proba"], dtype=np.float64)
    nprob = np.asarray(pred["notional_proba"], dtype=np.float64)
    pred_action = np.argmax(p, axis=1).astype(np.int64)
    conf = np.max(p, axis=1)
    teacher_active = (out["action"].astype(int).to_numpy() != ACTION_CASH) & (out["side"].astype(int).to_numpy() != 0)
    active = teacher_active.copy()
    if rt.skip_on_cash:
        active &= conf >= float(rt.confidence)
    side = out["side"].astype(int).to_numpy()
    if rt.allow_flip:
        side = np.where(pred_action == ACTION_LONG, 1, np.where(pred_action == ACTION_SHORT, -1, 0))
        active &= pred_action != ACTION_CASH
    else:
        active &= pred_action != ACTION_CASH
    out.loc[~active, ["action", "side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = 0
    out.loc[~active, "leverage"] = 1.0
    out.loc[active, "side"] = side[active].astype(int)
    out.loc[active, "action"] = np.where(side[active] > 0, ACTION_LONG, ACTION_SHORT).astype(int)
    if rt.use_learned_size:
        bucket_vals = np.asarray(buckets, dtype=np.float64)
        learned_n = np.sum(nprob * bucket_vals[None, :], axis=1) * float(rt.notional_scale)
        learned_n = np.minimum(learned_n, float(rt.max_notional))
        out.loc[active, "notional_exposure"] = learned_n[active]
        lev = pd.to_numeric(out["leverage"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
        out.loc[active, "position_fraction"] = learned_n[active] / np.maximum(lev[active], 1e-12)
    out.loc[:, "quality_score"] = pred["quality"]
    out.loc[:, "confidence"] = conf
    return out


def _predict_v27_fast(model: torch.nn.Module, df: pd.DataFrame, seq_cols: list[str], norm: dict[str, np.ndarray]) -> np.ndarray:
    device = _device()
    model = model.to(device).eval()
    arr = df.loc[:, seq_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    pad = np.zeros((v31.SEQ_LEN - 1, arr.shape[1]), dtype=np.float32)
    padded = np.vstack([pad, arr])
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=v31.SEQ_LEN, axis=0)
    if windows.shape[1] == arr.shape[1]:
        windows = windows.transpose(0, 2, 1)
    outs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(df), 4096):
            seqs = np.ascontiguousarray(windows[start : start + 4096])
            xx = ((seqs - norm["mean"][None, None, :]) / norm["std"][None, None, :]).astype(np.float32)
            outs.append(model(torch.from_numpy(xx).to(device)).detach().cpu().numpy())
    return np.vstack(outs).astype(np.float32)


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1.get("trades", 0)) < 20:
        return -1e9 + float(c1.get("pnl", 0.0))
    return float(c1["pnl"] + 0.45 * c2["pnl"] + 0.25 * c3["pnl"] - 0.25 * abs(c1["mdd"]))


def main() -> int:
    print(f"[{MODEL_ID}] loading artifacts", flush=True)
    parent = joblib.load(v31.DEFAULT_PARENT)
    jackpot_payload = joblib.load(v31.DEFAULT_JACKPOT)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = v31._load_v27(v31.DEFAULT_V27)
    base = dict(parent["config"])
    buckets = tuple(base.get("notional_buckets", (0.23, 0.368, 0.575, 0.8625, 1.2075, 1.6675, 2.3, 3.105, 4.14)))
    train_all = _read(v31.DEFAULT_TRAIN)
    eval_df = _read(v31.DEFAULT_EVAL)
    train = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    feature_cols = _feature_cols(train_all, eval_df)
    parent_audit = _audit_contract(train_all, eval_df, list(parent.get("feature_cols") or []))
    print(f"[{MODEL_ID}] teacher parent decisions", flush=True)
    train_dec = predict_policy_frame(parent, train, close=_close(train))
    val_dec = predict_policy_frame(parent, val, close=_close(val))
    eval_dec = predict_policy_frame(parent, eval_df, close=_close(eval_df))
    train_features = prepare_features(train, side_hint=0, close=_close(train), feature_cols=feature_cols)
    valid = np.arange(0, len(train), dtype=np.int64)
    print(f"[{MODEL_ID}] sequence tensor", flush=True)
    train_seq = _seq_tensor(train_features, valid, feature_cols)
    y_action = train_dec["action"].astype(int).to_numpy(dtype=np.int64)
    y_quality = pd.to_numeric(train_dec["quality_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    y_notional = _bucket_labels(train_dec, buckets)
    model, meta = _train_teacher_model(train_seq, y_action, y_quality, y_notional, n_buckets=len(buckets), epochs=35)
    val_features = prepare_features(val, side_hint=0, close=_close(val), feature_cols=feature_cols)
    eval_features = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=feature_cols)
    val_pred = _predict_deep(model, val_features, feature_cols, meta["norm"])
    eval_pred = _predict_deep(model, eval_features, feature_cols, meta["norm"])
    val_q = _predict_v27_fast(v27_model, val, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = _predict_v27_fast(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])

    grid_rows: list[dict[str, Any]] = []
    selected: Runtime | None = None
    best_score = -1e18
    for rt in _grid():
        dec = _constrained_decisions(val_dec, val_pred, buckets, rt)
        v1 = alpha1.backtest_alpha1(val, parent, jackpot_model, add_cfg, val_q, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=1.0, decisions=dec)
        v2 = alpha1.backtest_alpha1(val, parent, jackpot_model, add_cfg, val_q, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=2.0, decisions=dec)
        v3 = alpha1.backtest_alpha1(val, parent, jackpot_model, add_cfg, val_q, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=3.0, decisions=dec)
        score = _score(v1, v2, v3)
        grid_rows.append({**asdict(rt), "score": score, "val_pnl": v1["pnl"], "val_mdd": v1["mdd"], "val_trades": v1["trades"], "val_deep_entries": v1["deep_entries"], "val_c2_pnl": v2["pnl"], "val_c3_pnl": v3["pnl"]})
        if score > best_score:
            best_score = score
            selected = rt
    assert selected is not None
    experiments = []
    for name, dec in (
        ("alpha1", eval_dec),
        (f"teacher_constrained::{selected.name}", _constrained_decisions(eval_dec, eval_pred, buckets, selected)),
    ):
        metrics = {
            f"cost{mult}": alpha1.backtest_alpha1(eval_df, parent, jackpot_model, add_cfg, eval_q, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=float(mult), decisions=dec)
            for mult in (1, 2, 3)
        }
        experiments.append({"name": name, "metrics": metrics, "score": _score(metrics["cost1"], metrics["cost2"], metrics["cost3"])})
        print(f"[{MODEL_ID}] {name} cost1={metrics['cost1']['pnl']:.2f} mdd={metrics['cost1']['mdd']:.2f} cost2={metrics['cost2']['pnl']:.2f} cost3={metrics['cost3']['pnl']:.2f}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUT_DIR / "teacher_constrained_deep_parent.pt"
    torch.save({"model_id": MODEL_ID, "state_dict": model.state_dict(), "feature_cols": feature_cols, "meta": meta, "selected_config": asdict(selected), "buckets": buckets}, model_path)
    pd.DataFrame(grid_rows).sort_values("score", ascending=False).to_csv(GRID_OUT, index=False)
    best = max(experiments, key=lambda e: e["score"])
    blocking = list(parent_audit.get("blocking", []))
    warnings = list(parent_audit.get("warnings", []))
    if best["name"] != "alpha1" and best["metrics"]["cost1"]["pnl"] <= alpha1.ALPHA1_BASELINE["cost1"]["pnl"]:
        warnings.append("teacher_constrained_did_not_beat_alpha1_cost1")
    if best["metrics"]["cost2"]["pnl"] <= 0.0:
        warnings.append("best_cost2_not_survived")
    if best["metrics"]["cost3"]["pnl"] <= 0.0:
        warnings.append("best_cost3_not_survived")
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": "promote" if not blocking and best["name"] != "alpha1" and best["metrics"]["cost1"]["pnl"] > alpha1.ALPHA1_BASELINE["cost1"]["pnl"] else "iterate",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS only after selection",
        "cash_preserving": True,
        "new_parent_entries_allowed_in_teacher_cash": False,
        "v27_deep_scout_preserved": True,
        "selected_config": asdict(selected),
        "parent_audit": parent_audit,
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Teacher-constrained deep parent. A sequence Transformer imitates alpha1 parent active trajectories, but original parent CASH bars remain CASH so V27 deep scout is preserved. It can skip/resize/optionally flip only inside teacher-active parent candidates.",
        "selected_config": asdict(selected),
        "experiments": experiments,
        "audit": audit,
        "artifacts": {"model": str(model_path), "report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "grid": str(GRID_OUT)},
    }
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "best": best}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
