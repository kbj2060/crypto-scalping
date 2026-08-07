#!/usr/bin/env python3
from __future__ import annotations

import copy
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

from ensemble.fully_learned_governor_policy import ACTION_CASH, predict_policy_frame, prepare_features  # noqa: E402
from scripts import eval_alpha1_l2_execution_replay_20260513 as l2  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _feature_cols, _read  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_deep_entry_parent_lite_v38 import SEQ_LEN, _apply_norm, _normalizer  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402


MODEL_ID = "alpha2_outcome_teacher_l2_20260514"
OUT_DIR = ROOT / "data/ensemble/supervised/alpha2_outcome_teacher_l2_20260514"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha2_outcome_teacher_l2_20260514_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/alpha2_outcome_teacher_l2_20260514_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/alpha2_outcome_teacher_l2_20260514_grid.csv"


@dataclass(frozen=True)
class OutcomeRuntime:
    name: str
    keep_threshold: float
    parent_notional_scale: float
    max_notional: float = 2.75


class OutcomeTeacherNet(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 96) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=hidden * 3,
            dropout=0.15,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.pos = nn.Parameter(torch.zeros(1, SEQ_LEN, hidden))
        self.attn = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1))
        self.recency_bias = nn.Parameter(torch.linspace(-0.20, 0.35, SEQ_LEN).view(1, SEQ_LEN, 1))
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(0.10), nn.Linear(hidden // 2, 1))

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        h = self.proj(seq) + self.pos[:, -seq.shape[1] :, :]
        h = self.encoder(h)
        w = torch.softmax(self.attn(h) + self.recency_bias[:, -h.shape[1] :, :], dim=1)
        ctx = torch.sum(h * w, dim=1)
        return self.head(ctx).squeeze(-1)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1.get("trades", 0)) < 20:
        return -1e9 + float(c1.get("pnl", 0.0))
    return float(c1["pnl"] + 0.45 * c2["pnl"] + 0.30 * c3["pnl"] - 0.35 * abs(c1["mdd"]))


def _seq_tensor_fast(features: pd.DataFrame, cols: list[str]) -> np.ndarray:
    arr = features.loc[:, cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    pad = np.zeros((SEQ_LEN - 1, arr.shape[1]), dtype=np.float32)
    padded = np.vstack([pad, arr])
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=SEQ_LEN, axis=0)
    if windows.shape[1] == arr.shape[1]:
        windows = windows.transpose(0, 2, 1)
    return np.ascontiguousarray(windows)


def _runtime_grid() -> list[OutcomeRuntime]:
    rows: list[OutcomeRuntime] = []
    for th in (0.42, 0.46, 0.50, 0.54, 0.58, 0.62):
        for scale in (0.90, 1.00, 1.10):
            rows.append(OutcomeRuntime(f"outcome_keep{th:.2f}_scale{scale:.2f}", float(th), float(scale)))
    return rows


def _make_trade_labels(
    df: pd.DataFrame,
    trade_records: list[dict[str, Any]],
    *,
    min_good_pct: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    ts = pd.to_datetime(df["timestamp"]).astype("datetime64[ns]")
    index_by_ts = {str(t): i for i, t in enumerate(ts.astype(str))}
    rows: list[dict[str, Any]] = []
    indices: list[int] = []
    labels: list[float] = []
    for rec in trade_records:
        if str(rec.get("owner", "")) != "v21_2":
            continue
        ts_key = str(pd.Timestamp(rec.get("entry_signal_timestamp")).to_datetime64())
        idx = index_by_ts.get(ts_key)
        if idx is None:
            ts_key = str(pd.Timestamp(rec.get("entry_signal_timestamp")))
            idx = index_by_ts.get(ts_key)
        if idx is None:
            continue
        pnl = float(rec.get("realized_net_pct", 0.0))
        label = 1.0 if pnl > float(min_good_pct) else 0.0
        rows.append({**rec, "row_index": int(idx), "label_good": label, "realized_net_pct": pnl})
        indices.append(int(idx))
        labels.append(label)
    if not indices:
        raise RuntimeError("no parent-owned trade records found for outcome teacher labels")
    return np.asarray(indices, dtype=np.int64), np.asarray(labels, dtype=np.float32), pd.DataFrame(rows)


def _train_outcome_teacher(seq: np.ndarray, y: np.ndarray, *, max_epochs: int = 160) -> tuple[OutcomeTeacherNet, dict[str, Any]]:
    torch.manual_seed(20260514)
    np.random.seed(20260514)
    n = len(seq)
    if n < 40:
        raise RuntimeError(f"too few outcome labels: {n}")
    split = max(16, min(n - 16, int(n * 0.80)))
    norm = _normalizer(seq[:split])
    x = _apply_norm(seq, norm)
    x_train, x_val = x[:split], x[split:]
    y_train, y_val = y[:split], y[split:]
    device = _device()
    model = OutcomeTeacherNet(x.shape[-1]).to(device)
    pos = float(np.sum(y_train > 0.5))
    neg = float(len(y_train) - pos)
    pos_weight = torch.tensor([max(neg / max(pos, 1.0), 0.25)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)), batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)), batch_size=512, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=7e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5, min_lr=2e-5, threshold=1e-3, threshold_mode="rel")
    best_loss = float("inf")
    best_epoch = 0
    bad = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    print(
        f"[{MODEL_ID}] train outcome teacher rows={n} train={len(x_train)} val={len(x_val)} "
        f"pos_rate={float(np.mean(y)):.3f} device={device}",
        flush=True,
    )
    for ep in range(1, int(max_epochs) + 1):
        model.train()
        train_sum = 0.0
        batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_sum += float(loss.detach().cpu())
            batches += 1
        model.eval()
        val_sum = 0.0
        val_n = 0
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                bs = int(xb.shape[0])
                val_sum += float(loss.detach().cpu()) * bs
                val_n += bs
                preds.append(torch.sigmoid(logits).detach().cpu().numpy())
        val_loss = val_sum / max(val_n, 1)
        pred = np.concatenate(preds) if preds else np.zeros(0)
        acc = float(np.mean((pred >= 0.5) == (y_val >= 0.5))) if len(pred) else 0.0
        prev_lr = float(opt.param_groups[0]["lr"])
        scheduler.step(val_loss)
        lr = float(opt.param_groups[0]["lr"])
        improved = val_loss < best_loss * (1.0 - 1e-4)
        if improved:
            best_loss = val_loss
            best_epoch = int(ep)
            bad = 0
            best_state = copy.deepcopy(model.cpu().state_dict())
            model.to(device)
        else:
            bad += 1
        rec = {
            "epoch": int(ep),
            "train_loss": train_sum / max(batches, 1),
            "val_loss": float(val_loss),
            "val_acc": float(acc),
            "lr": float(lr),
            "bad_count": int(bad),
            "best_epoch": int(best_epoch),
        }
        history.append(rec)
        if ep == 1 or ep % 10 == 0 or improved or lr < prev_lr:
            drop = f" lr_drop={prev_lr:.2e}->{lr:.2e}" if lr < prev_lr else ""
            print(
                f"[{MODEL_ID}] epoch={ep:03d} train={rec['train_loss']:.5f} "
                f"val={val_loss:.5f} acc={acc:.3f} best_ep={best_epoch} bad={bad} lr={lr:.2e}{drop}",
                flush=True,
            )
        if bad >= 18:
            print(f"[{MODEL_ID}] early_stop epoch={ep} best_epoch={best_epoch} best_val={best_loss:.5f}", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu().eval(), {
        "norm": norm,
        "rows": int(n),
        "train_rows": int(len(x_train)),
        "val_rows": int(len(x_val)),
        "positive_rate": float(np.mean(y)),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_loss),
        "history_tail": history[-25:],
        "training_techniques": {
            "max_epochs": int(max_epochs),
            "early_stop_patience": 18,
            "lr_scheduler": "ReduceLROnPlateau(mode=min,factor=0.5,patience=5,min_lr=2e-5)",
            "learning_rate": 0.0002,
            "weight_decay": 0.0007,
            "gradient_clip_norm": 1.0,
            "best_checkpoint": "chronological validation BCE",
            "target": "parent-owned trade realized_net_pct > 0 under conservative L2 replay",
        },
    }


def _predict_good(model: OutcomeTeacherNet, features: pd.DataFrame, cols: list[str], norm: dict[str, np.ndarray]) -> np.ndarray:
    seq = _seq_tensor_fast(features, cols)
    x = _apply_norm(seq, norm)
    device = _device()
    model = model.to(device).eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            logits = model(torch.from_numpy(x[start : start + 4096]).to(device))
            out.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def _predict_v27_fast(model: torch.nn.Module, df: pd.DataFrame, seq_cols: list[str], norm: dict[str, np.ndarray]) -> np.ndarray:
    device = _device()
    model = model.to(device).eval()
    arr = df.loc[:, seq_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    pad = np.zeros((SEQ_LEN - 1, arr.shape[1]), dtype=np.float32)
    padded = np.vstack([pad, arr])
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=SEQ_LEN, axis=0)
    if windows.shape[1] == arr.shape[1]:
        windows = windows.transpose(0, 2, 1)
    outs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(df), 4096):
            seqs = np.ascontiguousarray(windows[start : start + 4096])
            xx = ((seqs - norm["mean"][None, None, :]) / norm["std"][None, None, :]).astype(np.float32)
            outs.append(model(torch.from_numpy(xx).to(device)).detach().cpu().numpy())
    return np.vstack(outs).astype(np.float32)


def _apply_outcome_gate(decisions: pd.DataFrame, p_good: np.ndarray, rt: OutcomeRuntime) -> pd.DataFrame:
    out = decisions.copy()
    active = (out["action"].astype(int).to_numpy() != ACTION_CASH) & (out["side"].astype(int).to_numpy() != 0)
    keep = active & (np.asarray(p_good) >= float(rt.keep_threshold))
    out.loc[active & ~keep, ["action", "side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = 0
    out.loc[active & ~keep, "leverage"] = 1.0
    notional = pd.to_numeric(out["notional_exposure"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    leverage = pd.to_numeric(out["leverage"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
    scaled = np.minimum(notional * float(rt.parent_notional_scale), float(rt.max_notional))
    out.loc[keep, "notional_exposure"] = scaled[keep]
    out.loc[keep, "position_fraction"] = scaled[keep] / np.maximum(leverage[keep], 1e-12)
    out.loc[:, "outcome_teacher_p_good"] = np.asarray(p_good)
    return out


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
    print(f"[{MODEL_ID}] loading Alpha2 stack", flush=True)
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
    train_all = _read(v31.DEFAULT_TRAIN)
    eval_df = _read(v31.DEFAULT_EVAL)
    train = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    feature_cols = _feature_cols(train_all, eval_df)
    parent_audit = _audit_contract(train_all, eval_df, list(parent.get("feature_cols") or []))
    l2_stats = l2._live_l2_stats()

    variant = next(v for v in l2._variants() if v.name == "alpha1_l2_conservative_fee20")
    print(f"[{MODEL_ID}] base decisions and V27", flush=True)
    train_dec = predict_policy_frame(parent, train, close=_close(train))
    val_dec = predict_policy_frame(parent, val, close=_close(val))
    eval_dec = predict_policy_frame(parent, eval_df, close=_close(eval_df))
    train_q = _predict_v27_fast(v27_model, train, v27_payload["seq_cols"], v27_payload["norm"])
    val_q = _predict_v27_fast(v27_model, val, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = _predict_v27_fast(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])

    print(f"[{MODEL_ID}] collecting cost-adjusted outcome labels from train replay", flush=True)
    train_replay = l2._run_with_l2_proxy(train, parent, jackpot_model, add_cfg, train_q, train_dec, variant, fee, slip, cost_mult=3.0, record=True)
    label_idx, labels, label_frame = _make_trade_labels(train, train_replay.get("trade_records", []), min_good_pct=0.0)
    label_frame.to_csv(OUT_DIR / "outcome_teacher_trade_labels.csv", index=False)
    train_features = prepare_features(train, side_hint=0, close=_close(train), feature_cols=feature_cols)
    all_seq = _seq_tensor_fast(train_features, feature_cols)
    model, meta = _train_outcome_teacher(all_seq[label_idx], labels, max_epochs=160)

    print(f"[{MODEL_ID}] predicting outcome probabilities", flush=True)
    val_features = prepare_features(val, side_hint=0, close=_close(val), feature_cols=feature_cols)
    eval_features = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=feature_cols)
    val_p = _predict_good(model, val_features, feature_cols, meta["norm"])
    eval_p = _predict_good(model, eval_features, feature_cols, meta["norm"])

    print(f"[{MODEL_ID}] selecting runtime on 2025Q4", flush=True)
    rows: list[dict[str, Any]] = []
    selected: OutcomeRuntime | None = None
    best_score = -1e18
    for rt in _runtime_grid():
        dec = _apply_outcome_gate(val_dec, val_p, rt)
        vm = _metrics(val, parent, jackpot_model, add_cfg, val_q, dec, variant, fee=fee, slip=slip)
        score = _score(vm["cost1"], vm["cost2"], vm["cost3"])
        rows.append(
            {
                **asdict(rt),
                "variant": variant.name,
                "selection_score": score,
                "val_cost1_pnl": vm["cost1"]["pnl"],
                "val_cost1_mdd": vm["cost1"]["mdd"],
                "val_cost1_trades": vm["cost1"]["trades"],
                "val_cost2_pnl": vm["cost2"]["pnl"],
                "val_cost3_pnl": vm["cost3"]["pnl"],
            }
        )
        if score > best_score:
            best_score = score
            selected = rt
            print(
                f"[{MODEL_ID}] new best {rt.name} score={score:.2f} "
                f"c1={vm['cost1']['pnl']:.2f} c2={vm['cost2']['pnl']:.2f} c3={vm['cost3']['pnl']:.2f}",
                flush=True,
            )
    assert selected is not None
    pd.DataFrame(rows).sort_values("selection_score", ascending=False).to_csv(GRID_OUT, index=False)

    print(f"[{MODEL_ID}] fixed 2026 OOS", flush=True)
    selected_dec = _apply_outcome_gate(eval_dec, eval_p, selected)
    experiments: list[dict[str, Any]] = []
    for name, decisions in (
        ("alpha1_l2_replay", eval_dec),
        (f"outcome_teacher_l2::{selected.name}", selected_dec),
    ):
        metrics = _metrics(eval_df, parent, jackpot_model, add_cfg, eval_q, decisions, variant, fee=fee, slip=slip)
        experiments.append({"name": name, "runtime": asdict(selected) if name.startswith("outcome") else None, "variant": asdict(variant), "metrics": metrics, "score": _score(metrics["cost1"], metrics["cost2"], metrics["cost3"])})
        print(
            f"[{MODEL_ID}] {name} cost1={metrics['cost1']['pnl']:.2f} mdd={metrics['cost1']['mdd']:.2f} "
            f"cost2={metrics['cost2']['pnl']:.2f} cost3={metrics['cost3']['pnl']:.2f}",
            flush=True,
        )

    model_path = OUT_DIR / "outcome_teacher_l2.pt"
    torch.save(
        {
            "model_id": MODEL_ID,
            "state_dict": model.state_dict(),
            "feature_cols": feature_cols,
            "train_meta": meta,
            "selected_runtime": asdict(selected),
            "selected_variant": asdict(variant),
        },
        model_path,
    )
    best = max(experiments, key=lambda x: x["score"])
    reference = next(e for e in experiments if e["name"] == "alpha1_l2_replay")
    warnings = list(parent_audit.get("warnings", []))
    if not l2_stats.get("usable_for_replay", False):
        warnings.append("historical_l2_snapshots_insufficient_conservative_ohlc_replay_only")
    warnings.append("real_live_l2_fill_model_requires_forward_shadow_collection")
    if best["name"] != reference["name"] and best["metrics"]["cost1"]["pnl"] <= 699.1379839727641:
        warnings.append("outcome_teacher_did_not_beat_alpha2_cost1")
    if best["name"] == reference["name"]:
        warnings.append("selected_best_is_alpha1_l2_replay_not_outcome_teacher")
    audit = {
        "status": "pass" if not parent_audit.get("blocking") else "fail",
        "verdict": "shadow_collect_l2" if not parent_audit.get("blocking") else "fail",
        "blocking": list(parent_audit.get("blocking", [])),
        "warnings": warnings,
        "selection_uses_2026": False,
        "train_window": "2025-01-01..2025-09-30",
        "label_source": "cost3 conservative L2 replay parent-owned trade_records on train only",
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS only after selection",
        "train_meta": meta,
        "label_rows": int(len(labels)),
        "label_positive_rate": float(np.mean(labels)),
        "live_l2_stats": l2_stats,
        "selected_runtime": asdict(selected),
        "selected_variant": asdict(variant),
        "parent_audit": parent_audit,
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Outcome teacher for Alpha2: instead of imitating HGB parent actions, train a sequence model to predict parent-owned trade survival from realized net trade outcome under cost3 conservative L2 replay. Runtime gates only parent trades; parent CASH still leaves V27 scout active.",
        "experiments": experiments,
        "audit": audit,
        "artifacts": {
            "model": str(model_path),
            "label_csv": str(OUT_DIR / "outcome_teacher_trade_labels.csv"),
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
