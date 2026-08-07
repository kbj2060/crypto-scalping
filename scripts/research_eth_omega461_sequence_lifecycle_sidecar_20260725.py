#!/usr/bin/env python3
"""Research-only sequence lifecycle sidecar for ETH Omega4.6.1.

Trains a small TCN-style PyTorch model on in-position state sequences from the existing
research-only stopping dataset. No live code, runtime config, or production artifact is changed.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, mean_squared_error, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "eth_omega461_sequence_lifecycle_sidecar_20260725"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
SOURCE_DATASET = (
    ROOT
    / "tmp/causal_regen_20260516"
    / "eth_omega461_censored_stopping_value_20260724"
    / "train_live_router_stopping_dataset.csv.gz"
)
SEED = 260725
SEQ_LEN = 48
HORIZON = 96
NON_FEATURE_COLUMNS = {
    "episode_id",
    "entry_timestamp",
    "state_timestamp",
    "source_component",
    "baseline_cause",
    "bars_to_baseline_exit",
    "q_exit",
    "q_hold",
    "advantage",
    "sample_weight",
    "risk_label_h12",
    "risk_label_h48",
    "risk_label_h96",
    "risk_label_h384",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


class SequenceTCN(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(feature_dim, 64, kernel_size=3, padding=2, dilation=2),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Conv1d(64, 64, kernel_size=3, padding=4, dilation=4),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Conv1d(64, 64, kernel_size=3, padding=8, dilation=8),
            nn.SiLU(),
        )
        self.norm = nn.LayerNorm(64)
        self.advantage_head = nn.Linear(64, 1)
        self.risk_head = nn.Linear(64, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: batch, seq, feature
        z = self.net(x.transpose(1, 2))[:, :, -x.shape[1] :]
        pooled = self.norm(z[:, :, -1])
        return self.advantage_head(pooled).squeeze(-1), self.risk_head(pooled)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))


def _load_dataset() -> pd.DataFrame:
    if not SOURCE_DATASET.exists():
        raise RuntimeError(f"missing source dataset: {SOURCE_DATASET}")
    data = pd.read_csv(SOURCE_DATASET)
    data["entry_timestamp"] = pd.to_datetime(data["entry_timestamp"], errors="raise")
    data["state_timestamp"] = pd.to_datetime(data["state_timestamp"], errors="raise")
    return data.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _feature_columns(data: pd.DataFrame) -> list[str]:
    return [
        col
        for col in data.columns
        if col not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(data[col])
    ]


def _episode_split(data: pd.DataFrame) -> dict[str, set[int]]:
    episode_time = data.groupby("episode_id", sort=False)["entry_timestamp"].first().sort_values()
    episodes = episode_time.index.to_numpy()
    n = len(episodes)
    if n < 30:
        raise RuntimeError(f"too few independent episodes for sequence diagnostic: {n}")
    train_cut = max(1, int(n * 0.70))
    cal_cut = max(train_cut + 1, int(n * 0.85))
    return {
        "train": set(int(x) for x in episodes[:train_cut]),
        "calibration": set(int(x) for x in episodes[train_cut:cal_cut]),
        "validation": set(int(x) for x in episodes[cal_cut:]),
    }


def _build_sequences(data: pd.DataFrame, feature_cols: list[str], scaler: StandardScaler | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    ordered = data.sort_values(["episode_id", "state_timestamp"]).reset_index(drop=True)
    raw = ordered[feature_cols].to_numpy(dtype=np.float64)
    if scaler is None:
        scaler = StandardScaler().fit(raw)
    x_scaled = scaler.transform(raw).astype(np.float32)
    seqs: list[np.ndarray] = []
    y_adv: list[float] = []
    y_risk: list[int] = []
    weights: list[float] = []
    for _, idx in ordered.groupby("episode_id", sort=False).indices.items():
        idx_arr = np.asarray(idx, dtype=np.int64)
        episode_x = x_scaled[idx_arr]
        for local_i, global_i in enumerate(idx_arr):
            start = max(0, local_i - SEQ_LEN + 1)
            chunk = episode_x[start : local_i + 1]
            padded = np.zeros((SEQ_LEN, len(feature_cols)), dtype=np.float32)
            padded[-len(chunk) :] = chunk
            seqs.append(padded)
            y_adv.append(float(ordered.at[int(global_i), "advantage"]))
            y_risk.append(int(ordered.at[int(global_i), f"risk_label_h{HORIZON}"]))
            weights.append(float(ordered.at[int(global_i), "sample_weight"]))
    return (
        np.stack(seqs).astype(np.float32),
        np.asarray(y_adv, dtype=np.float32),
        np.asarray(y_risk, dtype=np.int64),
        np.asarray(weights, dtype=np.float32),
        scaler,
    )


def _class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y.astype(np.int64), minlength=3).astype(np.float64)
    weights = len(y) / np.maximum(3.0 * counts, 1.0)
    return torch.tensor(weights, dtype=torch.float32)


def _train(
    x_train: np.ndarray,
    y_adv_train: np.ndarray,
    y_risk_train: np.ndarray,
    w_train: np.ndarray,
    x_cal: np.ndarray,
    y_adv_cal: np.ndarray,
    y_risk_cal: np.ndarray,
) -> tuple[SequenceTCN, dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SequenceTCN(x_train.shape[2]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=8.0e-4, weight_decay=1.0e-4)
    class_weights = _class_weights(y_risk_train).to(device)
    ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_adv_train, dtype=torch.float32),
        torch.tensor(y_risk_train, dtype=torch.long),
        torch.tensor(w_train, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=512, shuffle=True, drop_last=False)
    cal_x = torch.tensor(x_cal, dtype=torch.float32, device=device)
    cal_adv = torch.tensor(y_adv_cal, dtype=torch.float32, device=device)
    cal_risk = torch.tensor(y_risk_cal, dtype=torch.long, device=device)
    best_state = None
    best_loss = float("inf")
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, 31):
        model.train()
        total = 0.0
        seen = 0
        for xb, advb, riskb, wb in loader:
            xb = xb.to(device)
            advb = advb.to(device)
            riskb = riskb.to(device)
            wb = wb.to(device)
            pred_adv, logits = model(xb)
            reg_loss = ((pred_adv - advb) ** 2 * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            cls_raw = nn.functional.cross_entropy(logits, riskb, weight=class_weights, reduction="none")
            cls_loss = (cls_raw * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            loss = reg_loss + 0.40 * cls_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.detach().cpu()) * len(xb)
            seen += len(xb)
        model.eval()
        with torch.no_grad():
            cal_adv_pred, cal_logits = model(cal_x)
            cal_reg = nn.functional.mse_loss(cal_adv_pred, cal_adv)
            cal_cls = nn.functional.cross_entropy(cal_logits, cal_risk, weight=class_weights)
            cal_loss = float((cal_reg + 0.40 * cal_cls).detach().cpu())
        history.append({"epoch": epoch, "train_loss": total / max(seen, 1), "cal_loss": cal_loss})
        if cal_loss < best_loss - 1.0e-5:
            best_loss = cal_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 5:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"device": str(device), "best_cal_loss": best_loss, "epochs": len(history), "history_tail": history[-5:]}


def _predict(model: SequenceTCN, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    model.eval()
    adv_chunks = []
    prob_chunks = []
    with torch.no_grad():
        for start in range(0, len(x), 2048):
            xb = torch.tensor(x[start : start + 2048], dtype=torch.float32, device=device)
            adv, logits = model(xb)
            adv_chunks.append(adv.detach().cpu().numpy())
            prob_chunks.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
    return np.concatenate(adv_chunks), np.concatenate(prob_chunks)


def _metrics(name: str, y_adv: np.ndarray, y_risk: np.ndarray, pred_adv: np.ndarray, pred_prob: np.ndarray, episodes: int) -> dict[str, Any]:
    p_sl = pred_prob[:, 2]
    out: dict[str, Any] = {
        "split": name,
        "rows": int(len(y_adv)),
        "episodes": int(episodes),
        "advantage_rmse": float(mean_squared_error(y_adv, pred_adv) ** 0.5),
        "advantage_corr": float(np.corrcoef(y_adv, pred_adv)[0, 1]) if len(y_adv) > 2 else 0.0,
        "sl_brier_h96": float(brier_score_loss((y_risk == 2).astype(np.int64), p_sl)),
        "sl_positive_rate_h96": float(np.mean(y_risk == 2)),
        "pred_sl_mean_h96": float(np.mean(p_sl)),
        "risk_class_counts_h96": np.bincount(y_risk, minlength=3).tolist(),
    }
    if len(np.unique((y_risk == 2).astype(np.int64))) == 2:
        out["sl_auc_h96"] = float(roc_auc_score((y_risk == 2).astype(np.int64), p_sl))
    else:
        out["sl_auc_h96"] = None
    return out


def main() -> int:
    _set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = _load_dataset()
    feature_cols = _feature_columns(data)
    split_episodes = _episode_split(data)
    split_data = {name: data[data["episode_id"].isin(eps)].copy() for name, eps in split_episodes.items()}
    x_train, adv_train, risk_train, w_train, scaler = _build_sequences(split_data["train"], feature_cols)
    x_cal, adv_cal, risk_cal, _, _ = _build_sequences(split_data["calibration"], feature_cols, scaler)
    x_val, adv_val, risk_val, _, _ = _build_sequences(split_data["validation"], feature_cols, scaler)
    model, train_diag = _train(x_train, adv_train, risk_train, w_train, x_cal, adv_cal, risk_cal)
    metrics = {}
    for name, x, adv, risk in (
        ("train", x_train, adv_train, risk_train),
        ("calibration", x_cal, adv_cal, risk_cal),
        ("validation", x_val, adv_val, risk_val),
    ):
        pred_adv, pred_prob = _predict(model, x)
        metrics[name] = _metrics(name, adv, risk, pred_adv, pred_prob, len(split_episodes[name]))
    model_path = OUT_DIR / "model.pt"
    torch.save(
        {
            "model_id": MODEL_ID,
            "state_dict": model.cpu().state_dict(),
            "feature_columns": feature_cols,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "seq_len": SEQ_LEN,
            "horizon": HORIZON,
            "status": "research_only_not_live_promoted",
        },
        model_path,
    )
    report = {
        "model_id": MODEL_ID,
        "status": "research_only_not_live_promoted",
        "deployment_verdict": "do_not_apply_to_live",
        "source_dataset": SOURCE_DATASET,
        "model_path": model_path,
        "task": {
            "kind": "sequence_tcn_multitask_lifecycle_sidecar",
            "seq_len": SEQ_LEN,
            "targets": ["advantage_regression", f"risk_label_h{HORIZON}_classification"],
            "intended_future_use": "shadow-only sequence risk score; not an exit owner",
        },
        "dataset": {
            "rows": int(len(data)),
            "episodes": int(data["episode_id"].nunique()),
            "feature_count": int(len(feature_cols)),
            "baseline_cause_counts": data["baseline_cause"].value_counts().to_dict(),
            f"risk_label_h{HORIZON}_counts": data[f"risk_label_h{HORIZON}"].value_counts().sort_index().to_dict(),
        },
        "split_protocol": {
            "split_by": "entry_timestamp_ordered_episode_id",
            "trade_ledgers_used_as_input": False,
            "saved_parent_exit_timestamps_used": False,
            "live_modules_changed": False,
            "evaluation_scope": "label learnability only; no PnL promotion claim",
        },
        "training": train_diag,
        "metrics": metrics,
        "promotion_blockers": [
            "Only 85 independent live-router positions in source dataset.",
            "Internal temporal validation only; no untouched forward interval.",
            "No fresh-forward PnL replay for this model.",
            "Model is not wired to trading_bot.py and must remain research-only.",
        ],
    }
    (OUT_DIR / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8"
    )
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "metrics": metrics["validation"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
