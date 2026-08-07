#!/usr/bin/env python3
"""Research-only tabular deep lifecycle sidecar for ETH Omega4.6.1.

This script does not touch live modules, runtime config, environment settings, or promoted
artifacts. It trains a small PyTorch MLP on the previously built live-router stopping-state
dataset from the censored stopping-value research branch. The goal is label/feature learnability
diagnostics for a future shadow-only sidecar, not live promotion.
"""

from __future__ import annotations

import json
import random
import sys
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
MODEL_ID = "eth_omega461_tabular_deep_lifecycle_sidecar_20260725"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
SOURCE_DATASET = (
    ROOT
    / "tmp/causal_regen_20260516"
    / "eth_omega461_censored_stopping_value_20260724"
    / "train_live_router_stopping_dataset.csv.gz"
)
SEED = 260725
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


class LifecycleMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(0.08),
        )
        self.advantage_head = nn.Linear(64, 1)
        self.risk_head = nn.Linear(64, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.backbone(x)
        return self.advantage_head(z).squeeze(-1), self.risk_head(z)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))


def _load_dataset() -> pd.DataFrame:
    if not SOURCE_DATASET.exists():
        raise RuntimeError(f"missing source dataset: {SOURCE_DATASET}")
    data = pd.read_csv(SOURCE_DATASET)
    if data.empty:
        raise RuntimeError("empty source dataset")
    data["entry_timestamp"] = pd.to_datetime(data["entry_timestamp"], errors="raise")
    data = data.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return data


def _feature_columns(data: pd.DataFrame) -> list[str]:
    cols = [col for col in data.columns if col not in NON_FEATURE_COLUMNS]
    numeric_cols = []
    for col in cols:
        if pd.api.types.is_numeric_dtype(data[col]):
            numeric_cols.append(col)
    return numeric_cols


def _episode_split(data: pd.DataFrame) -> dict[str, np.ndarray]:
    episode_time = data.groupby("episode_id", sort=False)["entry_timestamp"].first().sort_values()
    episodes = episode_time.index.to_numpy()
    n = len(episodes)
    if n < 30:
        raise RuntimeError(f"too few independent episodes for deep sidecar diagnostic: {n}")
    train_cut = max(1, int(n * 0.70))
    cal_cut = max(train_cut + 1, int(n * 0.85))
    split_episodes = {
        "train": set(episodes[:train_cut]),
        "calibration": set(episodes[train_cut:cal_cut]),
        "validation": set(episodes[cal_cut:]),
    }
    ep = data["episode_id"].to_numpy()
    return {
        name: np.asarray([item in group for item in ep], dtype=bool)
        for name, group in split_episodes.items()
    }


def _class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y.astype(np.int64), minlength=3).astype(np.float64)
    weights = len(y) / np.maximum(3.0 * counts, 1.0)
    return torch.tensor(weights, dtype=torch.float32)


def _train_model(
    x_train: np.ndarray,
    y_adv_train: np.ndarray,
    y_risk_train: np.ndarray,
    w_train: np.ndarray,
    x_cal: np.ndarray,
    y_adv_cal: np.ndarray,
    y_risk_cal: np.ndarray,
) -> tuple[LifecycleMLP, dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LifecycleMLP(x_train.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    class_weights = _class_weights(y_risk_train).to(device)
    ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_adv_train, dtype=torch.float32),
        torch.tensor(y_risk_train, dtype=torch.long),
        torch.tensor(w_train, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=1024, shuffle=True, drop_last=False)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    patience = 5
    stale = 0
    cal_x_t = torch.tensor(x_cal, dtype=torch.float32, device=device)
    cal_adv_t = torch.tensor(y_adv_cal, dtype=torch.float32, device=device)
    cal_risk_t = torch.tensor(y_risk_cal, dtype=torch.long, device=device)
    history: list[dict[str, float]] = []
    for epoch in range(1, 41):
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
            cls_loss_raw = nn.functional.cross_entropy(logits, riskb, weight=class_weights, reduction="none")
            cls_loss = (cls_loss_raw * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            loss = reg_loss + 0.40 * cls_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.detach().cpu()) * len(xb)
            seen += len(xb)
        model.eval()
        with torch.no_grad():
            cal_adv_pred, cal_logits = model(cal_x_t)
            cal_reg = nn.functional.mse_loss(cal_adv_pred, cal_adv_t)
            cal_cls = nn.functional.cross_entropy(cal_logits, cal_risk_t, weight=class_weights)
            cal_loss = float((cal_reg + 0.40 * cal_cls).detach().cpu())
        history.append({"epoch": epoch, "train_loss": total / max(seen, 1), "cal_loss": cal_loss})
        if cal_loss < best_loss - 1.0e-5:
            best_loss = cal_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"device": str(device), "best_cal_loss": best_loss, "epochs": len(history), "history_tail": history[-5:]}


def _predict(model: LifecycleMLP, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    model.eval()
    adv_chunks = []
    prob_chunks = []
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            xb = torch.tensor(x[start : start + 4096], dtype=torch.float32, device=device)
            adv, logits = model(xb)
            adv_chunks.append(adv.detach().cpu().numpy())
            prob_chunks.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
    return np.concatenate(adv_chunks), np.concatenate(prob_chunks)


def _metrics(name: str, data: pd.DataFrame, mask: np.ndarray, pred_adv: np.ndarray, pred_prob: np.ndarray) -> dict[str, Any]:
    y_adv = data.loc[mask, "advantage"].to_numpy(dtype=np.float64)
    y_risk = data.loc[mask, f"risk_label_h{HORIZON}"].to_numpy(dtype=np.int64)
    p_sl = pred_prob[:, 2]
    out: dict[str, Any] = {
        "split": name,
        "rows": int(mask.sum()),
        "episodes": int(data.loc[mask, "episode_id"].nunique()),
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
    masks = _episode_split(data)
    scaler = StandardScaler()
    x_all_raw = data[feature_cols].to_numpy(dtype=np.float64)
    x_all = np.zeros_like(x_all_raw, dtype=np.float32)
    x_all[masks["train"]] = scaler.fit_transform(x_all_raw[masks["train"]]).astype(np.float32)
    for split in ("calibration", "validation"):
        x_all[masks[split]] = scaler.transform(x_all_raw[masks[split]]).astype(np.float32)
    y_adv = data["advantage"].to_numpy(dtype=np.float32)
    y_risk = data[f"risk_label_h{HORIZON}"].to_numpy(dtype=np.int64)
    weights = data["sample_weight"].to_numpy(dtype=np.float32)

    model, train_diag = _train_model(
        x_all[masks["train"]],
        y_adv[masks["train"]],
        y_risk[masks["train"]],
        weights[masks["train"]],
        x_all[masks["calibration"]],
        y_adv[masks["calibration"]],
        y_risk[masks["calibration"]],
    )

    metrics = {}
    for split in ("train", "calibration", "validation"):
        pred_adv, pred_prob = _predict(model, x_all[masks[split]])
        metrics[split] = _metrics(split, data, masks[split], pred_adv, pred_prob)

    model_path = OUT_DIR / "model.pt"
    torch.save(
        {
            "model_id": MODEL_ID,
            "state_dict": model.cpu().state_dict(),
            "feature_columns": feature_cols,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
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
            "kind": "tabular_deep_multitask_lifecycle_sidecar",
            "targets": ["advantage_regression", f"risk_label_h{HORIZON}_classification"],
            "intended_future_use": "shadow-only risk/lifecycle score; not an exit owner",
        },
        "dataset": {
            "rows": int(len(data)),
            "episodes": int(data["episode_id"].nunique()),
            "feature_count": int(len(feature_cols)),
            "baseline_cause_counts": data["baseline_cause"].value_counts().to_dict(),
            "source_component_counts": data["source_component"].value_counts().to_dict(),
            "positive_advantage_rows": int((data["advantage"] > 0.0).sum()),
            f"risk_label_h{HORIZON}_counts": data[f"risk_label_h{HORIZON}"].value_counts().sort_index().to_dict(),
        },
        "split_protocol": {
            "split_by": "entry_timestamp_ordered_episode_id",
            "train_fraction": 0.70,
            "calibration_fraction": 0.15,
            "validation_fraction": 0.15,
            "trade_ledgers_used_as_input": False,
            "saved_parent_exit_timestamps_used": False,
            "live_modules_changed": False,
            "evaluation_scope": "label learnability only; no PnL promotion claim",
        },
        "training": train_diag,
        "metrics": metrics,
        "promotion_blockers": [
            "Only 85 independent live-router positions in source dataset.",
            "Validation is an internal temporal split of an already researched training dataset.",
            "No fresh-forward PnL replay or untouched forward interval is used here.",
            "Model is not wired to trading_bot.py and must remain research-only until separate shadow collection.",
        ],
    }
    (OUT_DIR / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8"
    )
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "metrics": metrics["validation"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
