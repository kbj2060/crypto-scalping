#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, log_loss
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_regime_pred_moe_20260517 import (  # noqa: E402
    CLEAN_PREFIX,
    CLASSES,
    CLASS_TO_ID,
    DEFAULT_PREDICT_2025,
    DEFAULT_TRAIN_2024,
    _future_path_frame,
    _json_default,
    _label_thresholds,
    _labels,
    _predicted_path_diagnostics,
)


MODEL_ID = "regime_pred_moe_tft_20260517"
PRED_PREFIX = "regime_pred_"
DEFAULT_CLEAN_2024 = ROOT / "data/ensemble/supervised/clean_regime_hmm_v6_20260517/training_features_2024_clean_regime_hmm_v6.csv"
DEFAULT_CLEAN_2025 = ROOT / "data/ensemble/supervised/clean_regime_hmm_v6_20260517/training_features_2025_clean_regime_hmm_v6.csv"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/regime_pred_moe_tft_20260517"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/regime_pred_moe_tft_20260517_report.json"
NON_FEATURES = {
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "m7_entry_long_price",
    "m7_entry_short_price",
    "m7_tp_price",
    "m7_sl_price",
}
FORBIDDEN_EXACT = {
    "regime_bull",
    "regime_bear",
    "regime_chop",
    "regime_whipsaw",
    "regime_normal",
    "regime_trending",
    "regime_break",
    "cvp_regime",
    f"{CLEAN_PREFIX}risk_off_prob",
    f"{CLEAN_PREFIX}transition_risk",
    f"{CLEAN_PREFIX}state_code",
    f"{CLEAN_PREFIX}cluster",
}
FORBIDDEN_FRAGMENTS = (
    "future",
    "target",
    "label",
    "realized",
    "trade_pnl",
    "cash_after",
    "legacy",
    "hdb",
    "hmm_",
)


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"{path} missing timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return df.reset_index(drop=True)


def _merge_clean(base: pd.DataFrame, clean_path: Path | None) -> pd.DataFrame:
    if clean_path is None or not clean_path.exists():
        return base.copy()
    clean = _read(clean_path)
    keep = ["timestamp"] + [
        c
        for c in clean.columns
        if c.startswith(CLEAN_PREFIX)
        and c not in FORBIDDEN_EXACT
        and not c.endswith("risk_off_prob")
        and not c.endswith("transition_risk")
        and not c.endswith("state_code")
        and "cluster" not in c
    ]
    return base.merge(clean[keep], on="timestamp", how="left").sort_values("timestamp").reset_index(drop=True)


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _is_feature(col: str) -> bool:
    lower = col.lower()
    if col in NON_FEATURES or col in FORBIDDEN_EXACT:
        return False
    if lower.startswith("_"):
        return False
    if lower.startswith(CLEAN_PREFIX):
        return True
    if lower.startswith(PRED_PREFIX):
        return False
    if "regime" in lower:
        return False
    if any(fragment in lower for fragment in FORBIDDEN_FRAGMENTS):
        return False
    return True


def _feature_cols(train: pd.DataFrame, pred: pd.DataFrame, max_features: int) -> list[str]:
    common = set(train.columns) & set(pred.columns)
    clean_cols = sorted([c for c in common if c.startswith(CLEAN_PREFIX) and _is_feature(c)])
    raw_cols: list[str] = []
    for col in sorted(common):
        if col in clean_cols or not _is_feature(col):
            continue
        try:
            if pd.to_numeric(train[col], errors="coerce").notna().any() or pd.to_numeric(pred[col], errors="coerce").notna().any():
                raw_cols.append(str(col))
        except Exception:
            continue
    priority = [
        "log_return",
        "volatility_z",
        "rsi",
        "macd_hist",
        "bb_width_z",
        "hma_slope",
        "wick_ratio",
        "garman_klass_vol",
        "realized_vol_ratio",
        "mtf_trend_1h",
        "mtf_trend_4h",
        "rogers_satchell_vol",
        "parkinson_vol",
        "amihud_illiquidity_z",
        "btc_corr_60",
        "eth_btc_ratio_change",
        "fvg_dist",
        "chop_index",
        "cvp_poc_dist",
        "cvp_cluster_position",
        "cvp_volume_imbalance",
        "turtle_signal",
        "dual_momentum",
        "mean_reversion_z",
        "breakout_strength",
        "volume_profile_signal",
        "funding_roc_288",
        "long_squeeze_risk",
        "funding_price_divergence",
        "ofi_acceleration",
        "kalman_velocity",
        "realized_skewness",
        "ofti",
        "kel",
        "mta_funding",
        "svps",
        "pred_mdjd",
        "conf_mdjd",
        "volume",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "sum_open_interest_value",
        "sum_toptrader_long_short_ratio",
        "count_long_short_ratio",
        "last_funding_rate",
        "whale_retail_ratio",
        "smart_money_flow",
        "squeeze_power",
        "oi_change_rate",
        "net_taker_ratio",
        "taker_acceleration",
        "trade_intensity",
        "big_trade_ratio",
        "hour_sin",
        "hour_cos",
        "minute_sin",
        "minute_cos",
        "session_europe",
        "session_us",
        "is_hour_open",
    ]
    ordered_raw: list[str] = []
    for col in priority + raw_cols:
        if col in raw_cols and col not in ordered_raw:
            ordered_raw.append(col)
    room = max(0, int(max_features) - len(clean_cols))
    return clean_cols + ordered_raw[:room]


def _matrix(frame: pd.DataFrame, cols: list[str], medians: pd.Series | None = None) -> pd.DataFrame:
    out = pd.DataFrame({c: _num(frame, c) for c in cols}, index=frame.index)
    if medians is not None:
        out = out.fillna(medians).fillna(0.0)
    return out


def _known_future_covariates(ts: pd.Series, horizon: int) -> np.ndarray:
    target_ts = pd.to_datetime(ts) + pd.to_timedelta(int(horizon) * 5, unit="m")
    minute_of_day = target_ts.dt.hour.to_numpy(dtype=float) * 60.0 + target_ts.dt.minute.to_numpy(dtype=float)
    phase = 2.0 * np.pi * minute_of_day / (24.0 * 60.0)
    hour = target_ts.dt.hour.to_numpy(dtype=float)
    dow = target_ts.dt.dayofweek.to_numpy(dtype=float)
    return np.column_stack(
        [
            np.sin(phase),
            np.cos(phase),
            np.sin(2.0 * np.pi * hour / 24.0),
            np.cos(2.0 * np.pi * hour / 24.0),
            np.sin(2.0 * np.pi * dow / 7.0),
            np.cos(2.0 * np.pi * dow / 7.0),
        ]
    ).astype(np.float32)


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, known: np.ndarray, indices: np.ndarray, y: np.ndarray | None, seq_len: int) -> None:
        self.x = x.astype(np.float32, copy=False)
        self.known = known.astype(np.float32, copy=False)
        self.indices = indices.astype(np.int64, copy=False)
        self.y = None if y is None else y.astype(np.int64, copy=False)
        self.seq_len = int(seq_len)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
        end = int(self.indices[idx])
        start = end - self.seq_len + 1
        if start < 0:
            pad = np.repeat(self.x[[0]], -start, axis=0)
            seq = np.concatenate([pad, self.x[0 : end + 1]], axis=0)
        else:
            seq = self.x[start : end + 1]
        k = self.known[end]
        if self.y is None:
            return torch.from_numpy(seq), torch.from_numpy(k)
        return torch.from_numpy(seq), torch.from_numpy(k), torch.tensor(int(self.y[end]), dtype=torch.long)


class GatedResidual(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.gate = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc2(self.dropout(F.elu(self.fc1(x))))
        g = torch.sigmoid(self.gate(x))
        return self.norm(x + g * h)


class TemporalFusionRegimeNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_known: int,
        n_classes: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        seq_len: int,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.feature_gate = nn.Sequential(nn.Linear(n_features, n_features), nn.Sigmoid())
        self.input_proj = nn.Linear(n_features, d_model)
        self.known_proj = nn.Linear(n_known, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.context = GatedResidual(d_model * 5, d_model * 4, dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model * 5, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, n_classes),
        )

    @staticmethod
    def _tail_mean(x: torch.Tensor, length: int) -> torch.Tensor:
        return x[:, -min(length, x.shape[1]) :, :].mean(dim=1)

    def forward(self, x: torch.Tensor, known: torch.Tensor) -> torch.Tensor:
        x = x * self.feature_gate(x)
        h = self.input_proj(x) + self.pos[:, : x.shape[1], :]
        h = self.encoder(h)
        known_h = self.known_proj(known)
        pooled = torch.cat(
            [
                h[:, -1, :],
                self._tail_mean(h, 12),
                self._tail_mean(h, 36),
                self._tail_mean(h, 72),
                known_h,
            ],
            dim=1,
        )
        return self.head(self.context(pooled))


@dataclass
class FitResult:
    model: TemporalFusionRegimeNet
    history: list[dict[str, float]]


def _class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y, minlength=len(CLASSES)).astype(np.float64)
    total = float(counts.sum())
    weights = total / np.clip(len(CLASSES) * counts, 1.0, None)
    weights = np.clip(weights, 0.35, 4.0)
    return torch.tensor(weights, dtype=torch.float32)


def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            seq, known = batch[0].to(device), batch[1].to(device)
            logits = model(seq, known)
            rows.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
    out = np.vstack(rows) if rows else np.zeros((0, len(CLASSES)), dtype=np.float64)
    out = out.astype(np.float64)
    out /= np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
    return out


def _eval_report(y_true: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    pred = np.argmax(proba, axis=1)
    return {
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "log_loss": float(log_loss(y_true, proba, labels=list(range(len(CLASSES))))),
        "true_counts": {CLASSES[i]: int((y_true == i).sum()) for i in range(len(CLASSES))},
        "pred_counts": {CLASSES[i]: int((pred == i).sum()) for i in range(len(CLASSES))},
        "confusion_matrix": confusion_matrix(y_true, pred, labels=list(range(len(CLASSES)))).tolist(),
    }


def _fit_model(
    x: np.ndarray,
    known: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray | None,
    *,
    seq_len: int,
    epochs: int,
    batch_size: int,
    lr: float,
    d_model: int,
    n_heads: int,
    n_layers: int,
    dropout: float,
    seed: int,
    device: torch.device,
) -> FitResult:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model = TemporalFusionRegimeNet(
        n_features=x.shape[1],
        n_known=known.shape[1],
        n_classes=len(CLASSES),
        d_model=int(d_model),
        n_heads=int(n_heads),
        n_layers=int(n_layers),
        dropout=float(dropout),
        seq_len=int(seq_len),
    ).to(device)
    train_ds = SequenceDataset(x, known, train_idx, y, seq_len)
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, num_workers=0, drop_last=False)
    val_loader = None
    if val_idx is not None and len(val_idx):
        val_loader = DataLoader(SequenceDataset(x, known, val_idx, y, seq_len), batch_size=int(batch_size) * 2, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss(weight=_class_weights(y[train_idx]).to(device), label_smoothing=0.03)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    patience = 2
    stale = 0
    for epoch in range(1, int(epochs) + 1):
        model.train()
        losses: list[float] = []
        for seq, known_batch, target in train_loader:
            seq = seq.to(device)
            known_batch = known_batch.to(device)
            target = target.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(seq, known_batch), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        row = {"epoch": float(epoch), "train_loss": float(np.mean(losses) if losses else 0.0)}
        if val_loader is not None:
            val_proba = _predict(model, val_loader, device)
            val_y = y[val_idx]
            row["val_log_loss"] = float(log_loss(val_y, val_proba, labels=list(range(len(CLASSES)))))
            row["val_balanced_accuracy"] = float(balanced_accuracy_score(val_y, np.argmax(val_proba, axis=1)))
            if row["val_log_loss"] < best_val:
                best_val = row["val_log_loss"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
        history.append(row)
        print(f"[{MODEL_ID}] epoch={epoch} train_loss={row['train_loss']:.5f} val_log_loss={row.get('val_log_loss', float('nan')):.5f}", flush=True)
        if val_loader is not None and stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return FitResult(model=model, history=history)


def _output_frame(ts: pd.Series, proba: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame({"timestamp": ts.reset_index(drop=True)})
    for i, name in enumerate(CLASSES):
        out[f"{PRED_PREFIX}{name}_prob"] = proba[:, i]
    sorted_prob = np.sort(proba, axis=1)
    out[f"{PRED_PREFIX}trend_prob"] = out[f"{PRED_PREFIX}bull_prob"] + out[f"{PRED_PREFIX}bear_prob"]
    out[f"{PRED_PREFIX}micro_prob"] = out[f"{PRED_PREFIX}chop_prob"] + out[f"{PRED_PREFIX}whipsaw_prob"] + out[f"{PRED_PREFIX}normal_prob"]
    out[f"{PRED_PREFIX}directional_bias"] = out[f"{PRED_PREFIX}bull_prob"] - out[f"{PRED_PREFIX}bear_prob"]
    out[f"{PRED_PREFIX}range_prob"] = out[f"{PRED_PREFIX}chop_prob"] + out[f"{PRED_PREFIX}normal_prob"]
    out[f"{PRED_PREFIX}instability_prob"] = out[f"{PRED_PREFIX}whipsaw_prob"]
    out[f"{PRED_PREFIX}confidence"] = sorted_prob[:, -1]
    out[f"{PRED_PREFIX}entropy"] = -np.sum(proba * np.log(np.clip(proba, 1e-12, None)), axis=1) / math.log(len(CLASSES))
    out[f"{PRED_PREFIX}margin"] = sorted_prob[:, -1] - sorted_prob[:, -2]
    return out


def _prepare_arrays(
    frame: pd.DataFrame,
    cols: list[str],
    scaler: StandardScaler | None = None,
    medians: pd.Series | None = None,
    fit_rows: np.ndarray | None = None,
) -> tuple[np.ndarray, StandardScaler, pd.Series]:
    raw = _matrix(frame, cols)
    if medians is None:
        fit_raw = raw if fit_rows is None else raw.iloc[np.asarray(fit_rows, dtype=np.int64)]
        medians = fit_raw.median(numeric_only=True).fillna(0.0)
    filled = raw.fillna(medians).fillna(0.0)
    if scaler is None:
        scaler = StandardScaler()
        if fit_rows is None:
            scaler.fit(filled)
        else:
            scaler.fit(filled.iloc[np.asarray(fit_rows, dtype=np.int64)])
        x = scaler.transform(filled).astype(np.float32)
    else:
        x = scaler.transform(filled).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, scaler, medians


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TFT-style 5-class future regime predictor features for MoE routing.")
    parser.add_argument("--train-2024", type=Path, default=DEFAULT_TRAIN_2024)
    parser.add_argument("--predict-2025", type=Path, default=DEFAULT_PREDICT_2025)
    parser.add_argument("--clean-2024", type=Path, default=DEFAULT_CLEAN_2024)
    parser.add_argument("--clean-2025", type=Path, default=DEFAULT_CLEAN_2025)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--horizon", type=int, default=36)
    parser.add_argument("--seq-len", type=int, default=72)
    parser.add_argument("--val-start", default="2024-10-01")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--train-stride", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--max-features", type=int, default=96)
    parser.add_argument("--seed", type=int, default=2517)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_raw = _merge_clean(_read(args.train_2024), args.clean_2024)
    pred_raw = _merge_clean(_read(args.predict_2025), args.clean_2025)
    cols = _feature_cols(train_raw, pred_raw, int(args.max_features))
    if len(cols) < 12:
        raise ValueError(f"not enough TFT feature columns: {len(cols)}")

    val_start = pd.Timestamp(args.val_start)
    raw_ts = pd.to_datetime(train_raw["timestamp"])
    raw_train_mask = raw_ts < val_start
    threshold_path = _future_path_frame(train_raw.loc[raw_train_mask].copy(), int(args.horizon))
    train_only_thresholds = _label_thresholds(threshold_path)
    label_frame, label_meta = _labels(train_raw, int(args.horizon), thresholds=train_only_thresholds)
    train_labeled = train_raw.loc[label_frame.index].copy().join(label_frame[["_label_name", "_label_id"]])
    y = train_labeled["_label_id"].astype(int).to_numpy()
    ts = pd.to_datetime(train_labeled["timestamp"])
    first_val_idx = int(np.searchsorted(ts.to_numpy(dtype="datetime64[ns]"), np.datetime64(val_start)))
    embargo = int(args.horizon)
    train_end = max(int(args.seq_len) - 1, first_val_idx - embargo)
    train_idx = np.arange(int(args.seq_len) - 1, train_end, max(1, int(args.train_stride)), dtype=np.int64)
    val_idx = np.arange(max(int(args.seq_len) - 1, first_val_idx), len(train_labeled), dtype=np.int64)
    if len(train_idx) < 1000 or len(val_idx) < 1000:
        split = int(len(train_labeled) * 0.80)
        train_idx = np.arange(int(args.seq_len) - 1, split, max(1, int(args.train_stride)), dtype=np.int64)
        val_idx = np.arange(split, len(train_labeled), dtype=np.int64)
    x, val_scaler, val_medians = _prepare_arrays(train_labeled, cols, fit_rows=train_idx)
    known = _known_future_covariates(train_labeled["timestamp"], int(args.horizon))

    val_fit = _fit_model(
        x,
        known,
        y,
        train_idx,
        val_idx,
        seq_len=int(args.seq_len),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        d_model=int(args.d_model),
        n_heads=int(args.heads),
        n_layers=int(args.layers),
        dropout=float(args.dropout),
        seed=int(args.seed),
        device=device,
    )
    val_loader = DataLoader(SequenceDataset(x, known, val_idx, y, int(args.seq_len)), batch_size=int(args.batch_size) * 2, shuffle=False, num_workers=0)
    val_proba = _predict(val_fit.model, val_loader, device)

    full_label_frame, full_label_meta = _labels(train_raw, int(args.horizon))
    full_labeled = train_raw.loc[full_label_frame.index].copy().join(full_label_frame[["_label_name", "_label_id"]])
    y_full = full_labeled["_label_id"].astype(int).to_numpy()
    best_epoch = int(min(val_fit.history, key=lambda row: row.get("val_log_loss", float("inf")))["epoch"])
    x_full, scaler, medians = _prepare_arrays(full_labeled, cols)
    known_full = _known_future_covariates(full_labeled["timestamp"], int(args.horizon))
    full_idx = np.arange(int(args.seq_len) - 1, len(full_labeled), max(1, int(args.train_stride)), dtype=np.int64)
    final_fit = _fit_model(
        x_full,
        known_full,
        y_full,
        full_idx,
        None,
        seq_len=int(args.seq_len),
        epochs=max(1, best_epoch),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        d_model=int(args.d_model),
        n_heads=int(args.heads),
        n_layers=int(args.layers),
        dropout=float(args.dropout),
        seed=int(args.seed) + 101,
        device=device,
    )

    pred_x, _, _ = _prepare_arrays(pred_raw, cols, scaler=scaler, medians=medians)
    pred_known = _known_future_covariates(pred_raw["timestamp"], int(args.horizon))
    pred_idx = np.arange(len(pred_raw), dtype=np.int64)
    pred_loader = DataLoader(SequenceDataset(pred_x, pred_known, pred_idx, None, int(args.seq_len)), batch_size=int(args.batch_size) * 2, shuffle=False, num_workers=0)
    pred_proba = _predict(final_fit.model, pred_loader, device)
    pred_output = _output_frame(pred_raw["timestamp"], pred_proba)

    full_pred_loader = DataLoader(
        SequenceDataset(x_full, known_full, np.arange(len(full_labeled), dtype=np.int64), None, int(args.seq_len)),
        batch_size=int(args.batch_size) * 2,
        shuffle=False,
        num_workers=0,
    )
    full_proba = _predict(final_fit.model, full_pred_loader, device)
    train_output = _output_frame(full_labeled["timestamp"], full_proba)

    pred_sidecar = args.out_dir / f"{args.predict_2025.stem}_regime_pred_tft_moe.csv"
    train_sidecar = args.out_dir / f"{args.train_2024.stem}_regime_pred_tft_moe.csv"
    model_path = args.out_dir / "regime_pred_tft_moe_2024.pt"
    pred_output.to_csv(pred_sidecar, index=False)
    train_output.to_csv(train_sidecar, index=False)
    torch.save(
        {
            "model_id": MODEL_ID,
            "classes": CLASSES,
            "feature_cols": cols,
            "feature_medians": medians.to_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "state_dict": {k: v.detach().cpu() for k, v in final_fit.model.state_dict().items()},
            "horizon": int(args.horizon),
            "seq_len": int(args.seq_len),
            "d_model": int(args.d_model),
            "heads": int(args.heads),
            "layers": int(args.layers),
            "dropout": float(args.dropout),
        },
        model_path,
    )

    report = {
        "model_id": MODEL_ID,
        "model_path": str(model_path),
        "train_source": str(args.train_2024),
        "predict_source": str(args.predict_2025),
        "clean_2024": str(args.clean_2024),
        "clean_2025": str(args.clean_2025),
        "device": str(device),
        "horizon_bars": int(args.horizon),
        "seq_len": int(args.seq_len),
        "classes": CLASSES,
        "validation_label_meta": {
            **label_meta,
            "threshold_policy": "thresholds_fit_on_pre_validation_2024_rows_only",
            "threshold_fit_rows": int(raw_train_mask.sum()),
            "embargo_bars": int(embargo),
        },
        "final_label_meta": {
            **full_label_meta,
            "threshold_policy": "thresholds_fit_on_all_2024_rows_for_final_2024_only_model",
        },
        "feature_count": int(len(cols)),
        "feature_cols": cols,
        "clean_features_used": [c for c in cols if c.startswith(CLEAN_PREFIX)],
        "validation_training_history": val_fit.history,
        "selected_final_epochs": int(best_epoch),
        "final_training_history": final_fit.history,
        "validation": _eval_report(y[val_idx], val_proba),
        "train_sidecar": str(train_sidecar),
        "predict_sidecar": str(pred_sidecar),
        "predict_probability_sum_min": float(pred_proba.sum(axis=1).min()),
        "predict_probability_sum_max": float(pred_proba.sum(axis=1).max()),
        "predict_counts": {CLASSES[i]: int((np.argmax(pred_proba, axis=1) == i).sum()) for i in range(len(CLASSES))},
        "predict_confidence_mean": float(pred_output[f"{PRED_PREFIX}confidence"].mean()),
        "predict_entropy_mean": float(pred_output[f"{PRED_PREFIX}entropy"].mean()),
        "predict_path_diagnostics": _predicted_path_diagnostics(pred_raw, pred_output, pred_proba, int(args.horizon)),
        "notes": [
            "Future regime predictor uses a TFT-style PyTorch sequence model with variable gating, multi-scale attention pooling, and known future time covariates.",
            "Feature columns are named regime_pred_* instead of regime_future_* because the certified feature audit blocks the token 'future' in model inputs.",
            "risk_off and transition are not classes or outputs.",
            "The clean_regime HMM sidecar is consumed as current-state context for MoE routing.",
            "Sidecar intentionally writes soft probability and routing-shape columns only; no hard argmax label columns.",
        ],
    }
    args.report.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    print(f"[{MODEL_ID}] model={model_path}", flush=True)
    print(f"[{MODEL_ID}] train_sidecar={train_sidecar}", flush=True)
    print(f"[{MODEL_ID}] predict_sidecar={pred_sidecar}", flush=True)
    print(f"[{MODEL_ID}] report={args.report}", flush=True)


if __name__ == "__main__":
    main()
