#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
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

from scripts.build_regime_pred_moe_20260517 import DEFAULT_PREDICT_2025, DEFAULT_TRAIN_2024, _json_default  # noqa: E402


MODEL_ID = "regime4_pred_tft_clean_target_20260517"
CLASSES4 = ["bull", "bear", "chop", "whipsaw"]
PRED_PREFIX = "regime4_pred_"
CLEAN4_PREFIX = "clean_regime4_2024_unsup_v1_"
DEFAULT_CLEAN_2024 = ROOT / "data/ensemble/supervised/clean_regime4_raw_state12_v1_20260517/training_features_2024_clean_regime4_raw_state12_v1.csv"
DEFAULT_CLEAN_2025 = ROOT / "data/ensemble/supervised/clean_regime4_raw_state12_v1_20260517/training_features_2025_clean_regime4_raw_state12_v1.csv"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/regime4_pred_tft_clean_target_20260517"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/regime4_pred_tft_clean_target_20260517_report.json"


RAW_PRIORITY = [
    "log_return", "volatility_z", "rsi", "macd_hist", "bb_width_z", "hma_slope", "wick_ratio",
    "garman_klass_vol", "mtf_trend_1h", "mtf_trend_4h", "rogers_satchell_vol", "parkinson_vol",
    "amihud_illiquidity_z", "btc_corr_60", "eth_btc_ratio_change", "fvg_dist", "chop_index",
    "cvp_poc_dist", "cvp_cluster_position", "cvp_volume_imbalance", "turtle_signal",
    "dual_momentum", "mean_reversion_z", "breakout_strength", "volume_profile_signal",
    "funding_roc_288", "long_squeeze_risk", "funding_price_divergence", "ofi_acceleration",
    "kalman_velocity", "ofti", "kel", "mta_funding", "svps", "pred_mdjd", "conf_mdjd",
    "volume", "quote_volume", "trades", "taker_buy_base", "sum_open_interest_value",
    "sum_toptrader_long_short_ratio", "count_long_short_ratio", "last_funding_rate",
    "whale_retail_ratio", "smart_money_flow", "squeeze_power", "oi_change_rate",
    "net_taker_ratio", "taker_acceleration", "trade_intensity", "big_trade_ratio",
    "hour_sin", "hour_cos", "minute_sin", "minute_cos", "session_europe", "session_us",
    "is_hour_open", "bb_width", "close_btc", "quote_volume_btc", "taker_buy_quote", "volume_btc",
]
NON_FEATURES = {"timestamp", "open", "high", "low", "close", "m7_entry_long_price", "m7_entry_short_price", "m7_tp_price", "m7_sl_price"}
FORBIDDEN_FRAGMENTS = ("future", "target", "label", "realized", "trade_pnl", "cash_after", "legacy", "hdb", "hmm_")


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _merge_clean4(base: pd.DataFrame, clean_path: Path) -> pd.DataFrame:
    clean = _read(clean_path)
    keep = ["timestamp"] + [c for c in clean.columns if c.startswith(CLEAN4_PREFIX)]
    return base.merge(clean[keep], on="timestamp", how="left").sort_values("timestamp").reset_index(drop=True)


def _is_raw_feature(col: str) -> bool:
    lower = col.lower()
    if col in NON_FEATURES or lower.startswith("_") or lower.startswith(PRED_PREFIX):
        return False
    if lower.startswith(CLEAN4_PREFIX):
        return True
    if "regime" in lower:
        return False
    if any(x in lower for x in FORBIDDEN_FRAGMENTS):
        return False
    return True


def _feature_cols(train: pd.DataFrame, pred: pd.DataFrame, max_features: int) -> list[str]:
    common = set(train.columns) & set(pred.columns)
    clean_cols = sorted(c for c in common if c.startswith(CLEAN4_PREFIX))
    raw = []
    for col in RAW_PRIORITY + sorted(common):
        if col in raw or col in clean_cols or col not in common or not _is_raw_feature(col):
            continue
        if pd.to_numeric(train[col], errors="coerce").notna().any() or pd.to_numeric(pred[col], errors="coerce").notna().any():
            raw.append(col)
    return clean_cols + raw[: max(0, int(max_features) - len(clean_cols))]


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _matrix(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: _num(frame, c) for c in cols}, index=frame.index)


def _prepare_arrays(frame: pd.DataFrame, cols: list[str], scaler: StandardScaler | None = None, medians: pd.Series | None = None, fit_rows: np.ndarray | None = None):
    raw = _matrix(frame, cols)
    if medians is None:
        fit_raw = raw if fit_rows is None else raw.iloc[np.asarray(fit_rows, dtype=np.int64)]
        medians = fit_raw.median(numeric_only=True).fillna(0.0)
    filled = raw.fillna(medians).fillna(0.0)
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(filled if fit_rows is None else filled.iloc[np.asarray(fit_rows, dtype=np.int64)])
    x = scaler.transform(filled).astype(np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), scaler, medians


def _known_cov(ts: pd.Series, horizon: int) -> np.ndarray:
    target_ts = pd.to_datetime(ts) + pd.to_timedelta(int(horizon) * 5, unit="m")
    minute = target_ts.dt.hour.to_numpy(float) * 60.0 + target_ts.dt.minute.to_numpy(float)
    phase = 2.0 * np.pi * minute / 1440.0
    hour = target_ts.dt.hour.to_numpy(float)
    dow = target_ts.dt.dayofweek.to_numpy(float)
    return np.column_stack([np.sin(phase), np.cos(phase), np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24), np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7)]).astype(np.float32)


class SeqDS(Dataset):
    def __init__(self, x: np.ndarray, known: np.ndarray, idx: np.ndarray, y: np.ndarray | None, seq_len: int) -> None:
        self.x, self.known, self.idx, self.y, self.seq_len = x, known, idx.astype(np.int64), y, int(seq_len)

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        end = int(self.idx[i]); start = end - self.seq_len + 1
        if start < 0:
            seq = np.concatenate([np.repeat(self.x[[0]], -start, axis=0), self.x[: end + 1]], axis=0)
        else:
            seq = self.x[start : end + 1]
        if self.y is None:
            return torch.from_numpy(seq), torch.from_numpy(self.known[end])
        return torch.from_numpy(seq), torch.from_numpy(self.known[end]), torch.tensor(int(self.y[end]), dtype=torch.long)


class GRN(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.fc1, self.fc2, self.gate, self.norm, self.drop = nn.Linear(dim, hidden), nn.Linear(hidden, dim), nn.Linear(dim, dim), nn.LayerNorm(dim), nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc2(self.drop(F.elu(self.fc1(x))))
        return self.norm(x + torch.sigmoid(self.gate(x)) * h)


class TFTLite4(nn.Module):
    def __init__(self, n_features: int, n_known: int, d_model: int, heads: int, layers: int, dropout: float, seq_len: int) -> None:
        super().__init__()
        self.feature_gate = nn.Sequential(nn.Linear(n_features, n_features), nn.Sigmoid())
        self.input_proj = nn.Linear(n_features, d_model)
        self.known_proj = nn.Linear(n_known, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=d_model*4, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.context = GRN(d_model * 5, d_model * 4, dropout)
        self.head = nn.Sequential(nn.Linear(d_model * 5, d_model * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 2, len(CLASSES4)))

    @staticmethod
    def tail_mean(x: torch.Tensor, n: int) -> torch.Tensor:
        return x[:, -min(n, x.shape[1]):, :].mean(dim=1)

    def forward(self, x: torch.Tensor, known: torch.Tensor) -> torch.Tensor:
        x = x * self.feature_gate(x)
        h = self.encoder(self.input_proj(x) + self.pos[:, :x.shape[1], :])
        pooled = torch.cat([h[:, -1], self.tail_mean(h, 12), self.tail_mean(h, 36), self.tail_mean(h, 72), self.known_proj(known)], dim=1)
        return self.head(self.context(pooled))


def _clean_future_labels(frame: pd.DataFrame, horizon: int):
    prob_cols = [f"{CLEAN4_PREFIX}{c}_prob" for c in CLASSES4]
    probs = frame[prob_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    probs /= np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
    n = max(0, len(frame) - int(horizon))
    y = np.argmax(probs[int(horizon):int(horizon)+n], axis=1).astype(int)
    labels = pd.DataFrame({"_label_id": y, "_label_name": [CLASSES4[i] for i in y]}, index=frame.index[:n])
    counts = labels["_label_name"].value_counts().reindex(CLASSES4, fill_value=0)
    return labels, {"horizon": int(horizon), "label_source": "argmax_clean_regime4_at_t_plus_horizon", "label_counts": {k: int(v) for k, v in counts.items()}}


def _class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y, minlength=len(CLASSES4)).astype(float)
    w = counts.sum() / np.clip(len(CLASSES4) * counts, 1.0, None)
    return torch.tensor(np.clip(w, 0.35, 4.0), dtype=torch.float32)


def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval(); rows = []
    with torch.no_grad():
        for batch in loader:
            rows.append(torch.softmax(model(batch[0].to(device), batch[1].to(device)), dim=1).cpu().numpy())
    out = np.vstack(rows).astype(float)
    out /= np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
    return out


def _eval(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    pred = np.argmax(p, axis=1)
    return {
        "rows": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, p, labels=list(range(len(CLASSES4))))),
        "true_counts": {CLASSES4[i]: int((y == i).sum()) for i in range(len(CLASSES4))},
        "pred_counts": {CLASSES4[i]: int((pred == i).sum()) for i in range(len(CLASSES4))},
        "confusion_matrix": confusion_matrix(y, pred, labels=list(range(len(CLASSES4)))).tolist(),
    }


def _fit(x, known, y, train_idx, val_idx, args, seed, device):
    torch.manual_seed(seed); np.random.seed(seed)
    model = TFTLite4(x.shape[1], known.shape[1], args.d_model, args.heads, args.layers, args.dropout, args.seq_len).to(device)
    train_loader = DataLoader(SeqDS(x, known, train_idx, y, args.seq_len), batch_size=args.batch_size, shuffle=True)
    val_loader = None if val_idx is None else DataLoader(SeqDS(x, known, val_idx, y, args.seq_len), batch_size=args.batch_size*2, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=_class_weights(y[train_idx]).to(device), label_smoothing=0.03)
    history=[]; best=None; best_ll=float("inf"); stale=0
    for epoch in range(1, args.epochs+1):
        model.train(); losses=[]
        for seq,k,t in train_loader:
            opt.zero_grad(set_to_none=True)
            loss = crit(model(seq.to(device), k.to(device)), t.to(device))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
            losses.append(float(loss.detach().cpu()))
        row={"epoch": float(epoch), "train_loss": float(np.mean(losses))}
        if val_loader is not None:
            p=_predict(model, val_loader, device); yy=y[val_idx]
            row["val_log_loss"]=float(log_loss(yy, p, labels=list(range(len(CLASSES4)))))
            row["val_balanced_accuracy"]=float(balanced_accuracy_score(yy, np.argmax(p, axis=1)))
            if row["val_log_loss"] < best_ll:
                best_ll=row["val_log_loss"]; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
            else:
                stale += 1
        history.append(row)
        print(f"[{MODEL_ID}] epoch={epoch} train_loss={row['train_loss']:.5f} val_log_loss={row.get('val_log_loss', float('nan')):.5f}", flush=True)
        if val_loader is not None and stale >= 2:
            break
    if best is not None:
        model.load_state_dict(best)
    return model, history


def _output(ts: pd.Series, p: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame({"timestamp": ts.reset_index(drop=True)})
    for i,c in enumerate(CLASSES4): out[f"{PRED_PREFIX}{c}_prob"] = p[:, i]
    sp=np.sort(p, axis=1)
    out[f"{PRED_PREFIX}trend_prob"] = out[f"{PRED_PREFIX}bull_prob"] + out[f"{PRED_PREFIX}bear_prob"]
    out[f"{PRED_PREFIX}micro_prob"] = out[f"{PRED_PREFIX}chop_prob"] + out[f"{PRED_PREFIX}whipsaw_prob"]
    out[f"{PRED_PREFIX}directional_bias"] = out[f"{PRED_PREFIX}bull_prob"] - out[f"{PRED_PREFIX}bear_prob"]
    out[f"{PRED_PREFIX}range_prob"] = out[f"{PRED_PREFIX}chop_prob"]
    out[f"{PRED_PREFIX}instability_prob"] = out[f"{PRED_PREFIX}whipsaw_prob"]
    out[f"{PRED_PREFIX}confidence"] = sp[:, -1]
    out[f"{PRED_PREFIX}entropy"] = -np.sum(p*np.log(np.clip(p,1e-12,None)), axis=1)/math.log(len(CLASSES4))
    out[f"{PRED_PREFIX}margin"] = sp[:, -1] - sp[:, -2]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--seed", type=int, default=4417)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True); args.report.parent.mkdir(parents=True, exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_raw=_merge_clean4(_read(args.train_2024), args.clean_2024); pred_raw=_merge_clean4(_read(args.predict_2025), args.clean_2025)
    cols=_feature_cols(train_raw, pred_raw, args.max_features)
    labels, label_meta=_clean_future_labels(train_raw, args.horizon)
    labeled=train_raw.loc[labels.index].copy().join(labels)
    y=labeled["_label_id"].astype(int).to_numpy()
    ts=pd.to_datetime(labeled["timestamp"]); first_val=int(np.searchsorted(ts.to_numpy(dtype="datetime64[ns]"), np.datetime64(pd.Timestamp(args.val_start))))
    train_end=max(args.seq_len-1, first_val-args.horizon)
    train_idx=np.arange(args.seq_len-1, train_end, max(1,args.train_stride), dtype=np.int64)
    val_idx=np.arange(max(args.seq_len-1, first_val), len(labeled), dtype=np.int64)
    x,_,_= _prepare_arrays(labeled, cols, fit_rows=train_idx); known=_known_cov(labeled["timestamp"], args.horizon)
    val_model, val_hist = _fit(x, known, y, train_idx, val_idx, args, args.seed, device)
    val_p = _predict(val_model, DataLoader(SeqDS(x, known, val_idx, y, args.seq_len), batch_size=args.batch_size*2, shuffle=False), device)
    val_report=_eval(y[val_idx], val_p)
    best_epoch=int(min(val_hist, key=lambda r: r.get("val_log_loss", float("inf")))["epoch"])
    full_labels, full_meta=_clean_future_labels(train_raw, args.horizon)
    full=train_raw.loc[full_labels.index].copy().join(full_labels); y_full=full["_label_id"].astype(int).to_numpy()
    x_full, scaler, medians=_prepare_arrays(full, cols); known_full=_known_cov(full["timestamp"], args.horizon)
    full_idx=np.arange(args.seq_len-1, len(full), max(1,args.train_stride), dtype=np.int64)
    final_epochs=args.epochs; args.epochs=max(1,best_epoch)
    final_model, final_hist=_fit(x_full, known_full, y_full, full_idx, None, args, args.seed+101, device)
    args.epochs=final_epochs
    pred_x,_,_=_prepare_arrays(pred_raw, cols, scaler=scaler, medians=medians); pred_known=_known_cov(pred_raw["timestamp"], args.horizon)
    pred_p=_predict(final_model, DataLoader(SeqDS(pred_x, pred_known, np.arange(len(pred_raw)), None, args.seq_len), batch_size=args.batch_size*2, shuffle=False), device)
    full_p=_predict(final_model, DataLoader(SeqDS(x_full, known_full, np.arange(len(full)), None, args.seq_len), batch_size=args.batch_size*2, shuffle=False), device)
    pred_sidecar=args.out_dir / f"{args.predict_2025.stem}_regime4_pred_tft_clean_target.csv"
    train_sidecar=args.out_dir / f"{args.train_2024.stem}_regime4_pred_tft_clean_target.csv"
    model_path=args.out_dir / "regime4_pred_tft_clean_target_2024.pt"
    _output(pred_raw["timestamp"], pred_p).to_csv(pred_sidecar, index=False)
    _output(full["timestamp"], full_p).to_csv(train_sidecar, index=False)
    torch.save({"model_id": MODEL_ID, "classes": CLASSES4, "feature_cols": cols, "feature_medians": medians.to_dict(), "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_, "state_dict": {k:v.detach().cpu() for k,v in final_model.state_dict().items()}}, model_path)
    report={
        "model_id": MODEL_ID, "model_path": str(model_path), "classes": CLASSES4, "feature_count": len(cols), "feature_cols": cols,
        "clean_2024": str(args.clean_2024), "clean_2025": str(args.clean_2025), "horizon_bars": args.horizon, "seq_len": args.seq_len,
        "validation_label_meta": {**label_meta, "split_policy": "Q4_2024_validation_with_horizon_embargo", "embargo_bars": args.horizon},
        "validation_training_history": val_hist, "validation": val_report, "selected_final_epochs": int(best_epoch), "final_label_meta": full_meta, "final_training_history": final_hist,
        "train_sidecar": str(train_sidecar), "predict_sidecar": str(pred_sidecar),
        "predict_probability_sum_min": float(pred_p.sum(axis=1).min()), "predict_probability_sum_max": float(pred_p.sum(axis=1).max()),
        "predict_counts": {CLASSES4[i]: int((np.argmax(pred_p, axis=1)==i).sum()) for i in range(len(CLASSES4))},
        "predict_confidence_mean": float(_output(pred_raw["timestamp"], pred_p)[f"{PRED_PREFIX}confidence"].mean()),
        "predict_entropy_mean": float(_output(pred_raw["timestamp"], pred_p)[f"{PRED_PREFIX}entropy"].mean()),
    }
    args.report.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    print(f"[{MODEL_ID}] model={model_path}", flush=True); print(f"[{MODEL_ID}] predict_sidecar={pred_sidecar}", flush=True); print(f"[{MODEL_ID}] report={args.report}", flush=True)


if __name__ == "__main__":
    main()
