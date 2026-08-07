#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, log_loss, roc_auc_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.retrain_clean_regime_hmm_20260517 import GaussianStateModel, _json_default  # noqa: E402
from scripts.retrain_clean_regime_hmm_raw_state12_20260517 import STATE12_COLS, _fit_obs12, _with_raw_state12  # noqa: E402


MODEL_ID = "regime3_hmm_mamba_risk_cleanfunding_20260529"
CURRENT_PREFIX = "regime3_current_"
PRED_PREFIX = "regime3_pred_"
CLASSES3 = ["bull", "bear", "chop"]
HORIZONS = [12, 24, 48]
RISK_COLS = ["whipsaw_risk", "instability_prob", "transition_risk", "false_breakout_risk"]
FORBIDDEN_PREFIXES = ("clean_regime_2024_unsup_v4_", "clean_regime4_2024_unsup_v1_")
FORBIDDEN_FEATURE_FRAGMENTS = (
    "future",
    "target",
    "label",
    "realized",
    "trade_pnl",
    "cash_after",
    "legacy",
    "hdb",
    "hmm_",
    "regime3_",
    "regime4_",
    "clean_regime",
)
NON_FEATURES = {"timestamp", "open", "high", "low", "close"}
DEFAULT_TRAIN_2024 = ROOT / "tmp/causal_regen_20260516/funding_clean_splits_20260528/training_features_2024.csv"
DEFAULT_TRANSFORMS = (
    ROOT / "tmp/causal_regen_20260516/funding_clean_splits_20260528/training_features_2024.csv",
    ROOT / "tmp/causal_regen_20260516/funding_clean_splits_20260528/training_features_2025.csv",
    ROOT / "tmp/causal_regen_20260516/funding_clean_splits_20260528/training_features_2026_rebuilt.csv",
)
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/regime3_hmm_mamba_risk_cleanfunding_20260529"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/regime3_hmm_mamba_risk_cleanfunding_20260529_report.json"
RAW_PRIORITY = [
    "log_return", "volatility_z", "rsi", "macd_hist", "bb_width_z", "hma_slope", "wick_ratio",
    "garman_klass_vol", "realized_vol_ratio", "mtf_trend_1h", "mtf_trend_4h", "rogers_satchell_vol",
    "parkinson_vol", "amihud_illiquidity_z", "btc_corr_60", "eth_btc_ratio_change", "fvg_dist",
    "chop_index", "cvp_poc_dist", "cvp_cluster_position", "cvp_volume_imbalance", "turtle_signal",
    "dual_momentum", "mean_reversion_z", "breakout_strength", "volume_profile_signal",
    "ofi_acceleration", "kalman_velocity", "ofti", "kel", "svps", "volume", "quote_volume",
    "trades", "taker_buy_base", "taker_buy_quote", "sum_open_interest_value", "sum_toptrader_long_short_ratio",
    "count_long_short_ratio", "last_funding_rate", "whale_retail_ratio", "whale_conviction",
    "smart_money_flow", "squeeze_power", "oi_change_rate", "net_taker_ratio", "taker_acceleration",
    "trade_intensity", "big_trade_ratio", "hour_sin", "hour_cos", "minute_sin", "minute_cos",
    "session_europe", "session_us", "is_hour_open", "bb_width", "close_btc", "volume_btc",
    "quote_volume_btc",
]


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"{path} missing timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    up = high.diff()
    down = -low.diff()
    pdm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    ndm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    pdi = 100.0 * pdm.ewm(span=period, adjust=False).mean() / (atr + 1e-12)
    ndi = 100.0 * ndm.ewm(span=period, adjust=False).mean() / (atr + 1e-12)
    dx = 100.0 * (pdi - ndi).abs() / (pdi + ndi + 1e-12)
    return dx.ewm(span=period, adjust=False).mean()


def _current_labels3(frame: pd.DataFrame) -> np.ndarray:
    close = _num(frame, "close")
    high = _num(frame, "high")
    low = _num(frame, "low")
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema_slope = (ema21 - ema21.shift(5)) / (close * 5.0 + 1e-12)
    adx = _num(frame, "adx_14", np.nan)
    if adx.isna().all():
        adx = _adx(high, low, close)
    bb_width = _num(frame, "bb_width", np.nan)
    if bb_width.isna().all():
        sma20 = close.rolling(20, min_periods=5).mean()
        bb_width = 2.0 * close.rolling(20, min_periods=5).std() / (sma20 + 1e-12)
    labels = np.full(len(frame), 2, dtype=np.int64)
    trending = adx.fillna(0.0).to_numpy() >= 22.0
    slope = ema_slope.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    labels[trending & (slope > 0.00025)] = 0
    labels[trending & (slope < -0.00025)] = 1
    labels[(adx.fillna(0.0).to_numpy() < 18.0) | (bb_width.fillna(0.0).to_numpy() < 0.018)] = 2
    return labels


def _monotonic_score(path: np.ndarray) -> float:
    diffs = np.diff(path)
    return float(abs(diffs.sum()) / (np.abs(diffs).sum() + 1e-12))


def _future_labels_and_risk(frame: pd.DataFrame, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    close = _num(frame, "close").to_numpy(dtype=np.float64)
    high = _num(frame, "high").to_numpy(dtype=np.float64)
    low = _num(frame, "low").to_numpy(dtype=np.float64)
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - np.roll(close, 1)),
        np.abs(low - np.roll(close, 1)),
    ])
    tr[0] = high[0] - low[0]
    atr_pct = (
        pd.Series(tr / np.clip(close, 1e-12, None))
        .ewm(span=14, adjust=False)
        .mean()
        .rolling(96, min_periods=10)
        .median()
        .ffill()
        .fillna(0.004)
        .to_numpy()
    )
    n = max(0, len(frame) - int(horizon))
    y = np.full(n, 2, dtype=np.int64)
    risk = np.zeros((n, len(RISK_COLS)), dtype=np.float32)
    base_min = {12: 0.0025, 24: 0.0035, 48: 0.0050}.get(int(horizon), 0.0035)
    for i in range(n):
        path = close[i + 1 : i + 1 + int(horizon)]
        base = close[i]
        total_ret = (path[-1] - base) / max(base, 1e-12)
        max_range = (np.nanmax(path) - np.nanmin(path)) / max(base, 1e-12)
        mono = _monotonic_score(path)
        sign_flip = float(np.mean(np.diff(np.sign(np.diff(path))) != 0)) if len(path) > 2 else 0.0
        threshold = max(base_min, 0.65 * float(atr_pct[i]) * math.sqrt(int(horizon) / 12.0))
        if total_ret >= threshold and mono >= 0.35:
            y[i] = 0
        elif total_ret <= -threshold and mono >= 0.35:
            y[i] = 1
        else:
            y[i] = 2
        up_exc = (np.nanmax(path) - base) / max(base, 1e-12)
        dn_exc = (base - np.nanmin(path)) / max(base, 1e-12)
        whipsaw = float((max_range > 1.8 * threshold) and (abs(total_ret) < 0.65 * threshold) and (mono < 0.42 or sign_flip > 0.45))
        false_breakout = float((up_exc > threshold and dn_exc > 0.65 * threshold) or (dn_exc > threshold and up_exc > 0.65 * threshold))
        instability = float(max_range / max(2.5 * threshold, 1e-12))
        transition = float((mono < 0.35) or (sign_flip > 0.55))
        risk[i] = [whipsaw, min(1.0, instability), transition, false_breakout]
    return y, risk


def _prob_frame(ts: pd.Series, prefix: str, probs: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame({"timestamp": ts.reset_index(drop=True)})
    for i, name in enumerate(CLASSES3):
        out[f"{prefix}{name}_prob"] = probs[:, i]
    sp = np.sort(probs, axis=1)
    out[f"{prefix}confidence"] = sp[:, -1]
    out[f"{prefix}entropy"] = -np.sum(probs * np.log(np.clip(probs, 1e-12, None)), axis=1) / math.log(len(CLASSES3))
    out[f"{prefix}margin"] = sp[:, -1] - sp[:, -2]
    out[f"{prefix}directional_bias"] = out[f"{prefix}bull_prob"] - out[f"{prefix}bear_prob"]
    out[f"{prefix}trend_prob"] = out[f"{prefix}bull_prob"] + out[f"{prefix}bear_prob"]
    return out


def _eval_class(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    p = np.asarray(p, dtype=np.float64)
    p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
    pred = np.argmax(p, axis=1)
    return {
        "rows": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, p, labels=[0, 1, 2])),
        "true_counts": {CLASSES3[i]: int((y == i).sum()) for i in range(3)},
        "pred_counts": {CLASSES3[i]: int((pred == i).sum()) for i in range(3)},
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
    }


def _state_class_matrix(state_prob: np.ndarray, y: np.ndarray, smoothing: float = 0.02) -> np.ndarray:
    mat = np.full((state_prob.shape[1], 3), float(smoothing), dtype=np.float64)
    for cls in range(3):
        mat[:, cls] += state_prob[y == cls].sum(axis=0) / max(int((y == cls).sum()), 1)
    mat /= np.clip(mat.sum(axis=1, keepdims=True), 1e-300, None)
    return mat


def _hmm_fit(train: pd.DataFrame, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    ts = pd.to_datetime(train["timestamp"])
    train_mask = ts < pd.Timestamp(args.val_start)
    train_part = _with_raw_state12(train.loc[train_mask].copy())
    val_part = _with_raw_state12(train.loc[~train_mask].copy())
    train_obs, val_obs, _, _ = _fit_obs12(train_part, val_part)
    val_model = GaussianStateModel(args.hmm_states, args.hmm_iter, args.seed, sticky=args.hmm_sticky).fit(train_obs)
    y_train = _current_labels3(train_part)
    y_val = _current_labels3(val_part)
    state_class_val = _state_class_matrix(val_model.filter_proba(train_obs), y_train)
    val_probs = val_model.filter_proba(val_obs) @ state_class_val
    val_probs /= np.clip(val_probs.sum(axis=1, keepdims=True), 1e-300, None)
    full = _with_raw_state12(train.copy())
    full_obs, _, scaler, medians = _fit_obs12(full, full.iloc[:1].copy())
    model = GaussianStateModel(args.hmm_states, args.hmm_iter, args.seed + 101, sticky=args.hmm_sticky).fit(full_obs)
    y_full = _current_labels3(full)
    state_class = _state_class_matrix(model.filter_proba(full_obs), y_full)
    payload = {
        "model_id": f"{MODEL_ID}_current_hmm",
        "prefix": CURRENT_PREFIX,
        "classes": CLASSES3,
        "feature_cols": STATE12_COLS,
        "feature_medians": medians.to_dict(),
        "scaler": scaler,
        "model": model,
        "state_class_matrix": state_class,
        "state_count": int(args.hmm_states),
        "sticky": float(args.hmm_sticky),
    }
    return payload, {
        "validation": _eval_class(y_val, val_probs),
        "label_source": "causal_adx_ema_slope_bb_width_current_rule",
        "fit_rows": int(len(full)),
        "fit_range": [str(full["timestamp"].iloc[0]), str(full["timestamp"].iloc[-1])],
    }


def _hmm_transform(payload: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    work = _with_raw_state12(frame.copy())
    med = pd.Series(payload["feature_medians"])
    raw = work[STATE12_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    obs = payload["scaler"].transform(raw)
    probs = payload["model"].filter_proba(obs) @ payload["state_class_matrix"]
    probs /= np.clip(probs.sum(axis=1, keepdims=True), 1e-300, None)
    return _prob_frame(work["timestamp"], CURRENT_PREFIX, probs)


def _is_feature(col: str) -> bool:
    lower = col.lower()
    if col in NON_FEATURES or lower.startswith("_"):
        return False
    if col.startswith(FORBIDDEN_PREFIXES):
        return False
    if "regime" in lower or any(x in lower for x in FORBIDDEN_FEATURE_FRAGMENTS):
        return False
    return True


def _feature_cols(frames: list[pd.DataFrame], max_features: int) -> list[str]:
    common = set(frames[0].columns)
    for frame in frames[1:]:
        common &= set(frame.columns)
    cols: list[str] = []
    for col in RAW_PRIORITY + sorted(common):
        if col in cols or col not in common or not _is_feature(col):
            continue
        if pd.to_numeric(frames[0][col], errors="coerce").notna().any():
            cols.append(col)
        if len(cols) >= max_features:
            break
    bad = [c for c in cols if c.startswith(FORBIDDEN_PREFIXES) or "regime" in c.lower()]
    if bad:
        raise ValueError(f"forbidden regime features in Regime3 Mamba input: {bad[:10]}")
    return cols


def _matrix(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: _num(frame, c) for c in cols}, index=frame.index)


def _prepare(frame: pd.DataFrame, cols: list[str], fit_idx: np.ndarray | None = None, scaler: StandardScaler | None = None, medians: pd.Series | None = None):
    raw = _matrix(frame, cols)
    if medians is None:
        medians = (raw if fit_idx is None else raw.iloc[fit_idx]).median(numeric_only=True).fillna(0.0)
    filled = raw.fillna(medians).fillna(0.0)
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(filled if fit_idx is None else filled.iloc[fit_idx])
    return np.nan_to_num(scaler.transform(filled).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0), scaler, medians


class SeqDS(Dataset):
    def __init__(self, x: np.ndarray, idx: np.ndarray, y: dict[int, np.ndarray] | None, risk: np.ndarray | None, seq_len: int) -> None:
        self.x = x
        self.idx = idx.astype(np.int64)
        self.y = y
        self.risk = risk
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
        if self.y is None or self.risk is None:
            return torch.from_numpy(seq)
        return (
            torch.from_numpy(seq),
            *(torch.tensor(int(self.y[h][end]), dtype=torch.long) for h in HORIZONS),
            torch.from_numpy(self.risk[end]),
        )


class SharedMambaRegime(nn.Module):
    def __init__(self, n_features: int, d_model: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.input = nn.Sequential(nn.Linear(n_features, d_model), nn.LayerNorm(d_model), nn.SiLU())
        self.blocks = nn.ModuleList([Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({f"h{h}": nn.Linear(d_model, 3) for h in HORIZONS})
        self.risk = nn.Linear(d_model, len(RISK_COLS))

    def forward(self, x: torch.Tensor) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        h = self.input(x)
        for block, norm in zip(self.blocks, self.norms):
            h = norm(h + self.drop(block(h)))
        z = h[:, -1]
        return {horizon: self.heads[f"h{horizon}"](z) for horizon in HORIZONS}, self.risk(z)


def _labels_for_frame(frame: pd.DataFrame, max_h: int) -> tuple[dict[int, np.ndarray], np.ndarray, int]:
    n = len(frame) - max_h
    ys: dict[int, np.ndarray] = {}
    risks = []
    for h in HORIZONS:
        y, r = _future_labels_and_risk(frame, h)
        ys[h] = y[:n]
        risks.append(r[:n])
    risk = np.mean(np.stack(risks, axis=0), axis=0).astype(np.float32)
    return ys, risk, n


def _class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y, minlength=3).astype(float)
    w = counts.sum() / np.clip(3.0 * counts, 1.0, None)
    return torch.tensor(np.clip(w, 0.35, 4.0), dtype=torch.float32)


def _predict_mamba(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[int, np.ndarray], np.ndarray]:
    model.eval()
    cls_rows = {h: [] for h in HORIZONS}
    risk_rows = []
    with torch.no_grad():
        for seq in loader:
            if isinstance(seq, (list, tuple)):
                seq = seq[0]
            logits, risk = model(seq.to(device))
            for h in HORIZONS:
                cls_rows[h].append(torch.softmax(logits[h], dim=1).cpu().numpy())
            risk_rows.append(torch.sigmoid(risk).cpu().numpy())
    return {h: np.vstack(cls_rows[h]).astype(float) for h in HORIZONS}, np.vstack(risk_rows).astype(float)


def _fit_mamba(x: np.ndarray, y: dict[int, np.ndarray], risk: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray | None, args: argparse.Namespace, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = SharedMambaRegime(x.shape[1], args.d_model, args.mamba_layers, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    weights = {h: _class_weights(y[h][train_idx]).to(device) for h in HORIZONS}
    train_loader = DataLoader(SeqDS(x, train_idx, y, risk, args.seq_len), batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = None if val_idx is None else DataLoader(SeqDS(x, val_idx, None, None, args.seq_len), batch_size=args.batch_size * 2, shuffle=False)
    best = None
    best_score = float("inf")
    history = []
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            seq = batch[0].to(device)
            targets = {h: batch[i + 1].to(device) for i, h in enumerate(HORIZONS)}
            risk_t = batch[-1].to(device)
            opt.zero_grad(set_to_none=True)
            logits, risk_logits = model(seq)
            loss = sum(F.cross_entropy(logits[h], targets[h], weight=weights[h], label_smoothing=0.03) for h in HORIZONS)
            loss = loss + 0.35 * F.binary_cross_entropy_with_logits(risk_logits, risk_t)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        row: dict[str, Any] = {"epoch": int(epoch), "train_loss": float(np.mean(losses))}
        if val_idx is not None and val_loader is not None:
            probs, risk_p = _predict_mamba(model, val_loader, device)
            val_losses = []
            for h in HORIZONS:
                yy = y[h][val_idx]
                probs[h] = probs[h] / np.clip(probs[h].sum(axis=1, keepdims=True), 1e-12, None)
                row[f"val_h{h}_accuracy"] = float(accuracy_score(yy, np.argmax(probs[h], axis=1)))
                row[f"val_h{h}_balanced_accuracy"] = float(balanced_accuracy_score(yy, np.argmax(probs[h], axis=1)))
                row[f"val_h{h}_log_loss"] = float(log_loss(yy, probs[h], labels=[0, 1, 2]))
                val_losses.append(row[f"val_h{h}_log_loss"])
            row["val_mean_log_loss"] = float(np.mean(val_losses))
            row["val_risk_bce_proxy"] = float(np.mean(-(risk[val_idx] * np.log(np.clip(risk_p, 1e-6, 1.0)) + (1.0 - risk[val_idx]) * np.log(np.clip(1.0 - risk_p, 1e-6, 1.0)))))
            score = row["val_mean_log_loss"] + 0.15 * row["val_risk_bce_proxy"]
            if score < best_score:
                best_score = score
                best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
        history.append(row)
        print(f"[{MODEL_ID}] epoch={epoch} train_loss={row['train_loss']:.5f} val_mean_log_loss={row.get('val_mean_log_loss', float('nan')):.5f}", flush=True)
        if val_idx is not None and stale >= args.patience:
            break
    if best is not None:
        model.load_state_dict(best)
    return model, history, device


def _risk_eval(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for i, name in enumerate(RISK_COLS):
        target = y_true[:, i] >= 0.5
        pred = y_pred[:, i]
        out[name] = {
            "mean_target": float(y_true[:, i].mean()),
            "mean_pred": float(pred.mean()),
            "auc": None if target.min() == target.max() else float(roc_auc_score(target.astype(int), pred)),
        }
    return out


def _mamba_outputs(frame: pd.DataFrame, x: np.ndarray, model: nn.Module, device: torch.device, args: argparse.Namespace) -> pd.DataFrame:
    idx = np.arange(len(frame), dtype=np.int64)
    probs, risk = _predict_mamba(model, DataLoader(SeqDS(x, idx, None, None, args.seq_len), batch_size=args.batch_size * 2, shuffle=False), device)
    out = pd.DataFrame({"timestamp": frame["timestamp"].reset_index(drop=True)})
    for h in HORIZONS:
        pf = _prob_frame(frame["timestamp"], f"{PRED_PREFIX}h{h}_", probs[h])
        out = out.merge(pf, on="timestamp", how="left")
    for i, name in enumerate(RISK_COLS):
        out[name] = risk[:, i]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Train clean Regime3 current HMM and shared Mamba future/risk predictor.")
    p.add_argument("--train-2024", type=Path, default=DEFAULT_TRAIN_2024)
    p.add_argument("--transform", type=Path, action="append", default=None)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--val-start", default="2024-10-01")
    p.add_argument("--hmm-states", type=int, default=12)
    p.add_argument("--hmm-iter", type=int, default=22)
    p.add_argument("--hmm-sticky", type=float, default=0.93)
    p.add_argument("--seq-len", type=int, default=72)
    p.add_argument("--train-stride", type=int, default=2)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=640)
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--mamba-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-features", type=int, default=96)
    p.add_argument("--seed", type=int, default=529)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    transforms = list(args.transform or DEFAULT_TRANSFORMS)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    frames = [_read(path) for path in transforms]
    train = _read(args.train_2024)
    hmm_payload, hmm_report = _hmm_fit(train, args)
    hmm_model_path = args.out_dir / "regime3_current_hmm_2024.joblib"
    joblib.dump(hmm_payload, hmm_model_path)

    cols = _feature_cols([train] + frames, args.max_features)
    y_all, risk_all, n_valid = _labels_for_frame(train, max(HORIZONS))
    labeled = train.iloc[:n_valid].copy()
    ts = pd.to_datetime(labeled["timestamp"])
    first_val = int(np.searchsorted(ts.to_numpy(dtype="datetime64[ns]"), np.datetime64(pd.Timestamp(args.val_start))))
    train_end = max(args.seq_len - 1, first_val - max(HORIZONS))
    train_idx = np.arange(args.seq_len - 1, train_end, max(1, args.train_stride), dtype=np.int64)
    val_idx = np.arange(max(args.seq_len - 1, first_val), n_valid, dtype=np.int64)
    x, scaler, medians = _prepare(labeled, cols, fit_idx=train_idx)
    model, val_history, device = _fit_mamba(x, y_all, risk_all, train_idx, val_idx, args, args.seed)
    val_probs, val_risk = _predict_mamba(model, DataLoader(SeqDS(x, val_idx, None, None, args.seq_len), batch_size=args.batch_size * 2, shuffle=False), device)
    validation = {f"h{h}": _eval_class(y_all[h][val_idx], val_probs[h]) for h in HORIZONS}
    validation["risk"] = _risk_eval(risk_all[val_idx], val_risk)

    full_y, full_risk, full_n = _labels_for_frame(train, max(HORIZONS))
    full = train.iloc[:full_n].copy()
    x_full, scaler_full, medians_full = _prepare(full, cols)
    full_idx = np.arange(args.seq_len - 1, full_n, max(1, args.train_stride), dtype=np.int64)
    best_epoch = int(min(val_history, key=lambda r: r.get("val_mean_log_loss", float("inf")))["epoch"])
    saved_epochs = args.epochs
    args.epochs = max(1, best_epoch)
    final_model, final_history, final_device = _fit_mamba(x_full, full_y, full_risk, full_idx, None, args, args.seed + 101)
    args.epochs = saved_epochs
    mamba_path = args.out_dir / "regime3_pred_mamba_shared_2024.pt"
    torch.save(
        {
            "model_id": f"{MODEL_ID}_pred_mamba",
            "classes": CLASSES3,
            "horizons": HORIZONS,
            "risk_cols": RISK_COLS,
            "feature_cols": cols,
            "feature_medians": medians_full.to_dict(),
            "scaler_mean": scaler_full.mean_,
            "scaler_scale": scaler_full.scale_,
            "seq_len": int(args.seq_len),
            "d_model": int(args.d_model),
            "mamba_layers": int(args.mamba_layers),
            "state_dict": {k: v.detach().cpu() for k, v in final_model.state_dict().items()},
        },
        mamba_path,
    )

    report: dict[str, Any] = {
        "model_id": MODEL_ID,
        "fit_source": str(args.train_2024),
        "training_policy": "fit on 2024 only; select epochs on 2024Q4 validation; 2025/2026 are accuracy tests only",
        "current_hmm": {**hmm_report, "model_path": str(hmm_model_path), "prefix": CURRENT_PREFIX, "feature_cols": STATE12_COLS},
        "future_mamba": {
            "model_path": str(mamba_path),
            "prefix": PRED_PREFIX,
            "classes": CLASSES3,
            "horizons": HORIZONS,
            "risk_cols": RISK_COLS,
            "feature_cols": cols,
            "feature_count": len(cols),
            "validation_history": val_history,
            "selected_final_epochs": best_epoch,
            "final_training_history": final_history,
            "validation": validation,
        },
        "outputs": {},
        "leakage_audit": {
            "forbidden_prefixes": list(FORBIDDEN_PREFIXES),
            "forbidden_regime_features_in_mamba_input": [c for c in cols if c.startswith(FORBIDDEN_PREFIXES) or "regime" in c.lower()],
            "uses_2026_for_selection": False,
            "future_labels_use_future_path": True,
            "future_labels_are_targets_not_features": True,
        },
    }

    for path, frame in zip(transforms, frames):
        current = _hmm_transform(hmm_payload, frame)
        x_pred, _, _ = _prepare(frame, cols, scaler=scaler_full, medians=medians_full)
        pred = _mamba_outputs(frame, x_pred, final_model, final_device, args)
        sidecar = current.merge(pred, on="timestamp", how="inner")
        out_path = args.out_dir / f"{path.stem}_regime3_hmm_mamba_risk.csv"
        sidecar.to_csv(out_path, index=False)

        cur_label = _current_labels3(frame)
        cur_probs = current[[f"{CURRENT_PREFIX}{c}_prob" for c in CLASSES3]].to_numpy(float)
        source_report: dict[str, Any] = {
            "source": str(path),
            "sidecar": str(out_path),
            "rows": int(len(frame)),
            "range": [str(frame["timestamp"].iloc[0]), str(frame["timestamp"].iloc[-1])],
            "current_accuracy": _eval_class(cur_label, cur_probs),
        }
        y_eval, risk_eval, n_eval = _labels_for_frame(frame, max(HORIZONS))
        pred_eval = pred.iloc[:n_eval].copy()
        for h in HORIZONS:
            pp = pred_eval[[f"{PRED_PREFIX}h{h}_{c}_prob" for c in CLASSES3]].to_numpy(float)
            source_report[f"future_h{h}_accuracy"] = _eval_class(y_eval[h], pp)
        source_report["risk_accuracy"] = _risk_eval(risk_eval, pred_eval[RISK_COLS].to_numpy(float))
        report["outputs"][path.name] = source_report
        print(f"[{MODEL_ID}] wrote {out_path} rows={len(sidecar)}", flush=True)

    args.report.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    print(f"[{MODEL_ID}] hmm_model={hmm_model_path}", flush=True)
    print(f"[{MODEL_ID}] mamba_model={mamba_path}", flush=True)
    print(f"[{MODEL_ID}] report={args.report}", flush=True)


if __name__ == "__main__":
    main()
