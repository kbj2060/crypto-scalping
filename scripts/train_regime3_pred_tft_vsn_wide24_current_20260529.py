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

from scripts.retrain_clean_regime_hmm_20260517 import _json_default  # noqa: E402
from scripts.train_regime3_hmm_mamba_20260529 import RAW_PRIORITY, _read  # noqa: E402


MODEL_ID = "regime3_pred_tft_vsn_wide24_current_cleanfunding_20260529"
CLASSES3 = ["bull", "bear", "chop"]
CURRENT_PREFIX = "regime3_current_wide24_"
PRED_PREFIX = "regime3_pred_tft_h12_"
CURRENT_SIDECAR_STEM = "regime3_current_hmm_wide24"
OUTPUT_STEM = "regime3_pred_tft_vsn_wide24_current"
DEFAULT_TRAIN_2024 = ROOT / "tmp/causal_regen_20260516/funding_clean_splits_20260528/training_features_2024.csv"
DEFAULT_TRANSFORMS = (
    ROOT / "tmp/causal_regen_20260516/funding_clean_splits_20260528/training_features_2024.csv",
    ROOT / "tmp/causal_regen_20260516/funding_clean_splits_20260528/training_features_2025.csv",
    ROOT / "tmp/causal_regen_20260516/funding_clean_splits_20260528/training_features_2026_rebuilt.csv",
)
DEFAULT_CURRENT_DIR = ROOT / "data/ensemble/supervised/regime3_current_hmm_wide24_experiment_20260529"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/regime3_pred_tft_vsn_wide24_current_cleanfunding_20260529"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/regime3_pred_tft_vsn_wide24_current_cleanfunding_20260529_report.json"
FORBIDDEN_PREFIXES = ("clean_regime_2024_unsup_v4_", "clean_regime4_2024_unsup_v1_", "clean_regime4_state24_sticky090_v2_", "regime4_pred_")
FORBIDDEN_FRAGMENTS = ("future", "target", "label", "realized", "trade_pnl", "cash_after", "legacy", "hdb", "hmm_")
NON_FEATURES = {"timestamp", "open", "high", "low", "close"}
DOCS_REGIME_PRED_FEATURES = [
    "compression_score",
    "atr_pct_rank_288",
    "bb_width_pct_rank_288",
    "btc_volume_impulse_z",
    "vwap_dist_96",
    "vwap_dist_24",
    "cvd_288",
    "cvd_12",
    "eth_btc_ret_spread_12",
    "eth_btc_ret_spread_48",
    "btc_lead_eth_follow_gap_3",
    "btc_impulse_x_eth_beta",
    "anchored_vwap_session_dist",
    "range_contraction_breakout_dir",
    "distance_to_day_high_low_pct",
    "price_cvd_divergence",
    "funding_oi_divergence",
    "funding_price_divergence",
    "cvp_volume_imbalance",
    "crowded_short_squeeze_risk",
    "crowding_pressure",
    "long_squeeze_risk",
    "mean_reversion_z",
    "dual_momentum",
    "mtf_trend_1h",
    "mtf_trend_4h",
    "rsi",
    "macd_hist",
    "log_return",
    "oi_up_price_up",
    "cvp_regime",
]
DOCS_REGIME_PRED_ALL_EXTRA = [
    "last_funding_rate",
    "funding_pressure",
    "funding_roc_288",
    "volume_btc",
    "quote_volume_btc",
    "taker_buy_base",
    "taker_buy_quote",
    "volume",
    "quote_volume",
]
ROLLING_BASE_COLS = [
    "last_funding_rate",
    "funding_pressure",
    "funding_roc_288",
    "volume_btc",
    "quote_volume_btc",
    "taker_buy_base",
    "taker_buy_quote",
    "volume",
    "quote_volume",
]


def _current_path(current_dir: Path, source: Path) -> Path:
    return current_dir / f"{source.stem}_{CURRENT_SIDECAR_STEM}.csv"


def _merge_current(frame: pd.DataFrame, current_path: Path) -> pd.DataFrame:
    current = _read(current_path)
    required = [f"{CURRENT_PREFIX}{c}_prob" for c in CLASSES3]
    missing = [c for c in required if c not in current.columns]
    if missing:
        raise ValueError(f"{current_path} missing required current columns: {missing}")
    keep = ["timestamp"] + [c for c in current.columns if c.startswith(CURRENT_PREFIX)]
    out = frame.merge(current[keep], on="timestamp", how="left", validate="one_to_one")
    null_cols = [c for c in keep if c != "timestamp" and out[c].isna().any()]
    if null_cols:
        raise ValueError(f"current merge produced nulls: {null_cols[:10]}")
    out[f"{CURRENT_PREFIX}directional_bias"] = out[f"{CURRENT_PREFIX}bull_prob"] - out[f"{CURRENT_PREFIX}bear_prob"]
    out[f"{CURRENT_PREFIX}trend_prob"] = out[f"{CURRENT_PREFIX}bull_prob"] + out[f"{CURRENT_PREFIX}bear_prob"]
    out[f"{CURRENT_PREFIX}range_prob"] = out[f"{CURRENT_PREFIX}chop_prob"]
    return out


def _rolling_rank(series: pd.Series, window: int) -> pd.Series:
    def rank_last(values: np.ndarray) -> float:
        valid = values[np.isfinite(values)]
        if len(valid) <= 1:
            return 0.5
        return float((valid <= valid[-1]).mean())

    return series.rolling(window, min_periods=max(20, window // 10)).apply(rank_last, raw=True)


def _add_rolling_stable_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ROLLING_BASE_COLS:
        if col not in out.columns:
            continue
        s = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if col in {"volume_btc", "quote_volume_btc", "taker_buy_base", "taker_buy_quote", "volume", "quote_volume"}:
            signed_log = np.log1p(s.clip(lower=0.0))
        else:
            signed_log = np.sign(s) * np.log1p(s.abs())
        med = signed_log.rolling(288, min_periods=48).median()
        q25 = signed_log.rolling(288, min_periods=48).quantile(0.25)
        q75 = signed_log.rolling(288, min_periods=48).quantile(0.75)
        iqr = (q75 - q25).replace(0.0, np.nan)
        out[f"{col}_roll_log_iqr_288"] = ((signed_log - med) / iqr).clip(-8.0, 8.0)
        out[f"{col}_roll_pct_288"] = _rolling_rank(signed_log, 288)
        out[f"{col}_roll_delta_log_12"] = (signed_log - signed_log.shift(12)).clip(-8.0, 8.0)
    return out


def _is_feature(col: str) -> bool:
    lower = col.lower()
    if col in NON_FEATURES or lower.startswith("_") or col.startswith(PRED_PREFIX):
        return False
    if col.startswith(CURRENT_PREFIX):
        return True
    if any(col.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
        return False
    if "regime" in lower:
        return False
    if any(x in lower for x in FORBIDDEN_FRAGMENTS):
        return False
    return True


def _feature_cols(frames: list[pd.DataFrame], max_features: int, feature_pack: str = "default", include_current_features: bool = True) -> list[str]:
    common = set(frames[0].columns)
    for frame in frames[1:]:
        common &= set(frame.columns)
    current_cols = sorted(c for c in common if c.startswith(CURRENT_PREFIX)) if include_current_features else []
    if feature_pack in {"docs_regime_pred", "docs_regime_pred_all", "docs_regime_pred_rolled"}:
        requested = list(DOCS_REGIME_PRED_FEATURES)
        if feature_pack == "docs_regime_pred_all":
            requested += DOCS_REGIME_PRED_ALL_EXTRA
        elif feature_pack == "docs_regime_pred_rolled":
            for col in ROLLING_BASE_COLS:
                requested += [
                    f"{col}_roll_log_iqr_288",
                    f"{col}_roll_pct_288",
                    f"{col}_roll_delta_log_12",
                ]
        raw_cols = [c for c in requested if c in common and _is_feature(c)]
        cols = current_cols + raw_cols
        if not raw_cols:
            raise ValueError(f"feature_pack={feature_pack} produced no raw features")
        return cols[:max_features]
    if feature_pack != "default":
        raise ValueError(f"unknown feature_pack={feature_pack}")
    raw_cols: list[str] = []
    for col in RAW_PRIORITY + sorted(common):
        if col in raw_cols or col in current_cols or col not in common or not _is_feature(col):
            continue
        if pd.to_numeric(frames[0][col], errors="coerce").notna().any():
            raw_cols.append(col)
        if len(current_cols) + len(raw_cols) >= max_features:
            break
    cols = current_cols + raw_cols
    bad = [
        c
        for c in cols
        if any(c.startswith(prefix) for prefix in FORBIDDEN_PREFIXES)
        or ("regime" in c.lower() and not c.startswith(CURRENT_PREFIX))
    ]
    if bad:
        raise ValueError(f"forbidden features in TFT/VSN input: {bad[:10]}")
    return cols


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _matrix(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: _num(frame, c) for c in cols}, index=frame.index)


def _prepare(frame: pd.DataFrame, cols: list[str], scaler: StandardScaler | None = None, medians: pd.Series | None = None, fit_rows: np.ndarray | None = None):
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
        self.x = x
        self.known = known
        self.idx = idx.astype(np.int64)
        self.y = y
        self.seq_len = int(seq_len)

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        end = int(self.idx[i])
        start = end - self.seq_len + 1
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
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.gate = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc2(self.drop(F.elu(self.fc1(x))))
        return self.norm(x + torch.sigmoid(self.gate(x)) * h)


class TFTLite3(nn.Module):
    def __init__(self, n_features: int, n_known: int, d_model: int, heads: int, layers: int, dropout: float, seq_len: int) -> None:
        super().__init__()
        self.feature_gate = nn.Sequential(nn.Linear(n_features, n_features), nn.Sigmoid())
        self.input_proj = nn.Linear(n_features, d_model)
        self.known_proj = nn.Linear(n_known, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.context = GRN(d_model * 5, d_model * 4, dropout)
        self.head = nn.Sequential(nn.Linear(d_model * 5, d_model * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 2, len(CLASSES3)))

    @staticmethod
    def tail_mean(x: torch.Tensor, n: int) -> torch.Tensor:
        return x[:, -min(n, x.shape[1]):, :].mean(dim=1)

    def forward(self, x: torch.Tensor, known: torch.Tensor) -> torch.Tensor:
        x = x * self.feature_gate(x)
        h = self.encoder(self.input_proj(x) + self.pos[:, : x.shape[1], :])
        pooled = torch.cat([h[:, -1], self.tail_mean(h, 12), self.tail_mean(h, 36), self.tail_mean(h, 72), self.known_proj(known)], dim=1)
        return self.head(self.context(pooled))


def _future_current_labels(frame: pd.DataFrame, horizon: int):
    prob_cols = [f"{CURRENT_PREFIX}{c}_prob" for c in CLASSES3]
    probs = frame[prob_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    probs /= np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
    n = max(0, len(frame) - int(horizon))
    y = np.argmax(probs[int(horizon) : int(horizon) + n], axis=1).astype(int)
    labels = pd.DataFrame({"_label_id": y, "_label_name": [CLASSES3[i] for i in y]}, index=frame.index[:n])
    counts = labels["_label_name"].value_counts().reindex(CLASSES3, fill_value=0)
    return labels, {"horizon": int(horizon), "label_source": f"argmax_{CURRENT_PREFIX.rstrip('_')}_at_t_plus_horizon", "label_counts": {k: int(v) for k, v in counts.items()}}


def _class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y, minlength=len(CLASSES3)).astype(float)
    w = counts.sum() / np.clip(len(CLASSES3) * counts, 1.0, None)
    return torch.tensor(np.clip(w, 0.35, 4.0), dtype=torch.float32)


def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            rows.append(torch.softmax(model(batch[0].to(device), batch[1].to(device)), dim=1).cpu().numpy())
    out = np.vstack(rows).astype(float)
    out /= np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
    return out


def _eval(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    p = np.asarray(p, dtype=float)
    p /= np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
    pred = np.argmax(p, axis=1)
    cm = confusion_matrix(y, pred, labels=list(range(len(CLASSES3))))
    recall = {CLASSES3[i]: (None if cm[i].sum() == 0 else float(cm[i, i] / cm[i].sum())) for i in range(len(CLASSES3))}
    return {
        "rows": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, p, labels=list(range(len(CLASSES3))))),
        "recall": recall,
        "true_counts": {CLASSES3[i]: int((y == i).sum()) for i in range(len(CLASSES3))},
        "pred_counts": {CLASSES3[i]: int((pred == i).sum()) for i in range(len(CLASSES3))},
        "confusion_matrix": cm.tolist(),
    }


def _fit(x, known, y, train_idx, val_idx, args, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TFTLite3(x.shape[1], known.shape[1], args.d_model, args.heads, args.layers, args.dropout, args.seq_len).to(device)
    train_loader = DataLoader(SeqDS(x, known, train_idx, y, args.seq_len), batch_size=args.batch_size, shuffle=True)
    val_loader = None if val_idx is None else DataLoader(SeqDS(x, known, val_idx, y, args.seq_len), batch_size=args.batch_size * 2, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=_class_weights(y[train_idx]).to(device), label_smoothing=0.03)
    history = []
    best = None
    best_ll = float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for seq, k, t in train_loader:
            opt.zero_grad(set_to_none=True)
            loss = crit(model(seq.to(device), k.to(device)), t.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        row = {"epoch": int(epoch), "train_loss": float(np.mean(losses))}
        if val_loader is not None:
            p = _predict(model, val_loader, device)
            yy = y[val_idx]
            row["val_log_loss"] = float(log_loss(yy, p, labels=list(range(len(CLASSES3)))))
            row["val_balanced_accuracy"] = float(balanced_accuracy_score(yy, np.argmax(p, axis=1)))
            if row["val_log_loss"] < best_ll:
                best_ll = row["val_log_loss"]
                best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
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
    for i, c in enumerate(CLASSES3):
        out[f"{PRED_PREFIX}{c}_prob"] = p[:, i]
    sp = np.sort(p, axis=1)
    out[f"{PRED_PREFIX}confidence"] = sp[:, -1]
    out[f"{PRED_PREFIX}entropy"] = -np.sum(p * np.log(np.clip(p, 1e-12, None)), axis=1) / math.log(len(CLASSES3))
    out[f"{PRED_PREFIX}margin"] = sp[:, -1] - sp[:, -2]
    out[f"{PRED_PREFIX}directional_bias"] = out[f"{PRED_PREFIX}bull_prob"] - out[f"{PRED_PREFIX}bear_prob"]
    out[f"{PRED_PREFIX}trend_prob"] = out[f"{PRED_PREFIX}bull_prob"] + out[f"{PRED_PREFIX}bear_prob"]
    return out


def _importance(model: nn.Module, x: np.ndarray, known: np.ndarray, idx: np.ndarray, cols: list[str], args, device) -> pd.DataFrame:
    loader = DataLoader(SeqDS(x, known, idx, None, args.seq_len), batch_size=args.batch_size * 2, shuffle=False)
    model.eval()
    acc = np.zeros(len(cols), dtype=np.float64)
    n = 0
    with torch.no_grad():
        for seq, _known in loader:
            seq = seq.to(device)
            score = (model.feature_gate(seq) * seq.abs()).mean(dim=(0, 1)).cpu().numpy()
            acc += score
            n += 1
    raw = acc / max(n, 1)
    norm = raw / max(float(raw.sum()), 1e-12)
    return pd.DataFrame({"feature": cols, "vsn_importance": norm, "raw_gate_abs_score": raw}).sort_values("vsn_importance", ascending=False).reset_index(drop=True)


def _select(imp: pd.DataFrame, threshold: float, min_selected: int, max_selected: int) -> tuple[list[str], str]:
    cols = imp[imp["vsn_importance"] >= threshold]["feature"].tolist()
    policy = f"drop_importance_below_{threshold:g}"
    if len(cols) > max_selected:
        cols = imp.head(max_selected)["feature"].tolist()
        policy += f"_cap_top_{max_selected}"
    elif len(cols) < min_selected:
        cols = imp.head(min_selected)["feature"].tolist()
        policy += f"_floor_top_{min_selected}"
    return cols, policy


def _split(frame: pd.DataFrame, cols: list[str], args) -> dict[str, Any]:
    labels, meta = _future_current_labels(frame, args.horizon)
    labeled = frame.loc[labels.index].copy().join(labels)
    y = labeled["_label_id"].astype(int).to_numpy()
    ts = pd.to_datetime(labeled["timestamp"])
    first_val = int(np.searchsorted(ts.to_numpy(dtype="datetime64[ns]"), np.datetime64(pd.Timestamp(args.val_start))))
    train_end = max(args.seq_len - 1, first_val - args.horizon)
    train_idx = np.arange(args.seq_len - 1, train_end, max(1, args.train_stride), dtype=np.int64)
    val_idx = np.arange(max(args.seq_len - 1, first_val), len(labeled), dtype=np.int64)
    x, scaler, medians = _prepare(labeled, cols, fit_rows=train_idx)
    known = _known_cov(labeled["timestamp"], args.horizon)
    return {"labeled": labeled, "meta": meta, "y": y, "x": x, "known": known, "train_idx": train_idx, "val_idx": val_idx, "scaler": scaler, "medians": medians}


def _fit_eval(split: dict[str, Any], args, seed: int, device) -> dict[str, Any]:
    model, hist = _fit(split["x"], split["known"], split["y"], split["train_idx"], split["val_idx"], args, seed, device)
    p = _predict(model, DataLoader(SeqDS(split["x"], split["known"], split["val_idx"], split["y"], args.seq_len), batch_size=args.batch_size * 2, shuffle=False), device)
    return {"model": model, "history": hist, "validation": _eval(split["y"][split["val_idx"]], p), "best_epoch": int(min(hist, key=lambda r: r.get("val_log_loss", float("inf")))["epoch"])}


def main() -> None:
    global MODEL_ID, CURRENT_PREFIX, PRED_PREFIX, CURRENT_SIDECAR_STEM, OUTPUT_STEM
    p = argparse.ArgumentParser(description="Train Regime3 h12 TFT/VSN selected predictor with raw + wide24 current inputs.")
    p.add_argument("--train-2024", type=Path, default=DEFAULT_TRAIN_2024)
    p.add_argument("--transform", type=Path, action="append", default=None)
    p.add_argument("--current-dir", type=Path, default=DEFAULT_CURRENT_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--seq-len", type=int, default=72)
    p.add_argument("--val-start", default="2024-10-01")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=768)
    p.add_argument("--train-stride", type=int, default=2)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.12)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--max-features", type=int, default=112)
    p.add_argument("--feature-pack", choices=["default", "docs_regime_pred", "docs_regime_pred_all", "docs_regime_pred_rolled"], default="default")
    p.add_argument("--importance-threshold", type=float, default=0.01)
    p.add_argument("--min-selected", type=int, default=30)
    p.add_argument("--max-selected", type=int, default=74)
    p.add_argument("--seed", type=int, default=9529)
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--current-prefix", default=CURRENT_PREFIX)
    p.add_argument("--pred-prefix", default=PRED_PREFIX)
    p.add_argument("--current-sidecar-stem", default=CURRENT_SIDECAR_STEM)
    p.add_argument("--output-stem", default=OUTPUT_STEM)
    p.add_argument("--exclude-current-features", action="store_true")
    args = p.parse_args()
    MODEL_ID = args.model_id
    CURRENT_PREFIX = args.current_prefix
    PRED_PREFIX = args.pred_prefix
    CURRENT_SIDECAR_STEM = args.current_sidecar_stem
    OUTPUT_STEM = args.output_stem

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sources = list(args.transform or DEFAULT_TRANSFORMS)
    raw_frames = [_add_rolling_stable_features(_read(path)) for path in sources]
    frames = [_merge_current(frame, _current_path(args.current_dir, path)) for frame, path in zip(raw_frames, sources)]
    train = _merge_current(_add_rolling_stable_features(_read(args.train_2024)), _current_path(args.current_dir, args.train_2024))

    include_current_features = not bool(args.exclude_current_features)
    all_cols = _feature_cols([train] + frames, args.max_features, args.feature_pack, include_current_features=include_current_features)
    all_split = _split(train, all_cols, args)
    all_res = _fit_eval(all_split, args, args.seed, device)
    imp = _importance(all_res["model"], all_split["x"], all_split["known"], all_split["val_idx"], all_cols, args, device)
    imp_path = args.out_dir / "regime3_pred_tft_vsn_importance.csv"
    imp.to_csv(imp_path, index=False)
    selected_cols, policy = _select(imp, args.importance_threshold, args.min_selected, args.max_selected)
    selected_split = _split(train, selected_cols, args)
    sel_res = _fit_eval(selected_split, args, args.seed + 11, device)

    labels, full_meta = _future_current_labels(train, args.horizon)
    full = train.loc[labels.index].copy().join(labels)
    y_full = full["_label_id"].astype(int).to_numpy()
    best_epoch = max(1, int(sel_res["best_epoch"]))
    old_epochs = args.epochs
    args.epochs = best_epoch
    x_full, scaler, medians = _prepare(full, selected_cols)
    known_full = _known_cov(full["timestamp"], args.horizon)
    full_idx = np.arange(args.seq_len - 1, len(full), max(1, args.train_stride), dtype=np.int64)
    final_model, final_hist = _fit(x_full, known_full, y_full, full_idx, None, args, args.seed + 101, device)
    args.epochs = old_epochs

    full_p = _predict(final_model, DataLoader(SeqDS(x_full, known_full, np.arange(len(full)), None, args.seq_len), batch_size=args.batch_size * 2, shuffle=False), device)
    train_sidecar = args.out_dir / f"{args.train_2024.stem}_{OUTPUT_STEM}.csv"
    _output(full["timestamp"], full_p).to_csv(train_sidecar, index=False)
    model_path = args.out_dir / f"{OUTPUT_STEM}_2024.pt"
    torch.save(
        {
            "model_id": MODEL_ID,
            "classes": CLASSES3,
            "horizon": int(args.horizon),
            "current_prefix": CURRENT_PREFIX,
            "current_sidecar_stem": CURRENT_SIDECAR_STEM,
            "pred_prefix": PRED_PREFIX,
            "output_stem": OUTPUT_STEM,
            "include_current_features": include_current_features,
            "feature_cols": selected_cols,
            "feature_medians": medians.to_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "state_dict": {k: v.detach().cpu() for k, v in final_model.state_dict().items()},
        },
        model_path,
    )

    report: dict[str, Any] = {
        "model_id": MODEL_ID,
        "model_path": str(model_path),
        "classes": CLASSES3,
        "horizon_bars": int(args.horizon),
        "seq_len": int(args.seq_len),
        "current_prefix": CURRENT_PREFIX,
        "current_sidecar_stem": CURRENT_SIDECAR_STEM,
        "pred_prefix": PRED_PREFIX,
        "output_stem": OUTPUT_STEM,
        "include_current_features": include_current_features,
        "all_feature_count": len(all_cols),
        "all_feature_cols": all_cols,
        "selected_feature_count": len(selected_cols),
        "selected_feature_cols": selected_cols,
        "current_feature_count": int(sum(c.startswith(CURRENT_PREFIX) for c in selected_cols)),
        "selection_policy": policy,
        "feature_pack": args.feature_pack,
        "importance_path": str(imp_path),
        "top_20_vsn_importance": imp.head(20).to_dict(orient="records"),
        "validation_label_meta": {**all_split["meta"], "split_policy": "Q4_2024_validation_with_horizon_embargo", "embargo_bars": args.horizon},
        "all_features": {"training_history": all_res["history"], "best_epoch": all_res["best_epoch"], "validation": all_res["validation"]},
        "selected_features": {"training_history": sel_res["history"], "best_epoch": sel_res["best_epoch"], "validation": sel_res["validation"]},
        "final_label_meta": full_meta,
        "selected_final_epochs": best_epoch,
        "final_training_history": final_hist,
        "train_sidecar": str(train_sidecar),
        "outputs": {},
        "leakage_audit": {
            "forbidden_prefixes": list(FORBIDDEN_PREFIXES),
            "forbidden_regime_features": [c for c in selected_cols if ("regime" in c.lower() and not c.startswith(CURRENT_PREFIX)) or any(c.startswith(p) for p in FORBIDDEN_PREFIXES)],
            "uses_2026_for_selection": False,
            "future_labels_use_future_current_argmax": True,
            "future_labels_are_targets_not_features": True,
        },
    }

    for path, frame in zip(sources, frames):
        x_pred, _, _ = _prepare(frame, selected_cols, scaler=scaler, medians=medians)
        known_pred = _known_cov(frame["timestamp"], args.horizon)
        pred_p = _predict(final_model, DataLoader(SeqDS(x_pred, known_pred, np.arange(len(frame)), None, args.seq_len), batch_size=args.batch_size * 2, shuffle=False), device)
        sidecar = args.out_dir / f"{path.stem}_{OUTPUT_STEM}.csv"
        _output(frame["timestamp"], pred_p).to_csv(sidecar, index=False)
        labels_eval, _ = _future_current_labels(frame, args.horizon)
        y_eval = labels_eval["_label_id"].astype(int).to_numpy()
        report["outputs"][path.name] = {
            "source": str(path),
            "sidecar": str(sidecar),
            "rows": int(len(frame)),
            "range": [str(frame["timestamp"].iloc[0]), str(frame["timestamp"].iloc[-1])],
            "accuracy": _eval(y_eval, pred_p[: len(y_eval)]),
        }
        print(f"[{MODEL_ID}] wrote {sidecar}", flush=True)

    args.report.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    print(f"[{MODEL_ID}] all_features={len(all_cols)} selected={len(selected_cols)} policy={policy}", flush=True)
    print(f"[{MODEL_ID}] report={args.report}", flush=True)


if __name__ == "__main__":
    main()
