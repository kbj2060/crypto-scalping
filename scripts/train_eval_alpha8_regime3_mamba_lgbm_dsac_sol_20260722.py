#!/usr/bin/env python3
"""Alpha8 (HMM/Regime -> Mamba -> LightGBM -> DSAC hybrid) ported to SOL.

Simplified vs the ETH Alpha8 chain (documented per CLAUDE.md instructions):
  - Layer 1 regime context uses SOL's existing Regime3 sidecar
    (`regime3_current_sensitive_wide24_{bull,bear,chop}_prob` + confidence/entropy/margin)
    instead of ETH's clean_regime4_state24_sticky090_v2_* (4-class) + regime4_pred_* (future
    predictor). SOL has no whipsaw class and no future-regime predictor surface, so those are
    simply omitted (contract lists regime4_pred_* as allowed-not-required).
  - There is no SOL equivalent of the Alpha6->Alpha7 ETH "trade candidate" pipeline (barrier
    labels + AI-forecast/M7-ensemble context + a trained governor-policy primary/fallback
    parent model). Reproducing that full chain for SOL was assessed as disproportionate (same
    conclusion reached for Alpha4.3). Instead:
      * Directional/flow context = SOL's own raw 5m feature frame (`data/splits/year_oos/
        sol_features_{2024,2025,2026}.csv`), which already contains all of the ETH Alpha7
        SOURCE_COLS except `tp_sl_action_score`, `ai_dir_edge`, `ai_flow_pressure`,
        `m7_expected_ret`, `m7_q50`, `m7_quality_pred` (SOL lacks the AI-forecast/M7-ensemble
        pipeline entirely) -- those six columns are dropped, not substituted.
      * Layer 3 direction label = a fixed 12-bar / 0.25% barrier label computed directly from
        SOL close prices (same definition Alpha8's own Mamba layer already uses for ETH), not
        the zigzag action labels (avoids depending on the other in-flight SOL zigzag-label
        extension work) and not the un-portable Alpha6 governor-policy barrier definition.
      * Layer 4 "primary"/"fallback" parents (ETH Alpha7 governor-policy models, asset- and
        regime4-column-specific, cannot run on SOL data at all) are replaced by two fixed
        rule-based decision templates keyed directly off this script's own LightGBM
        hold/long/short probabilities: a stricter "primary" template (higher confidence
        threshold, larger size) and a looser "fallback" template (lower threshold, smaller
        size). DSAC still learns to choose among skip/primary/fallback per bar, preserving the
        Alpha8 four-layer wiring and the discrete-SAC action space semantics.

Evaluation is a from-scratch causal bar-by-bar walk-forward simulator (own code, not reused
from any ETH ledger/backtest harness): for each active decision bar, entry is the next bar's
close, exit is the first TP/SL barrier hit (or max_hold timeout) walking strictly forward bar by
bar, and the position is busy (no new entries) until exit + cooldown. No stored trade ledger,
candidate-event replay, or saved parent exit timestamp is used as input anywhere in this script.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_ID = "alpha8_regime3_mamba_lgbm_dsac_sol_20260722"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha8_regime3_mamba_lgbm_dsac_sol_20260722"

SOL_FEATURE_CSVS = {
    2024: ROOT / "data/splits/year_oos/sol_features_2024.csv",
    2025: ROOT / "data/splits/year_oos/sol_features_2025.csv",
    2026: ROOT / "data/splits/year_oos/sol_features_2026.csv",
}
REGIME3_DIR = ROOT / "data/ensemble/supervised/sol_regime3_current_hmm_sensitive_wide24_20260707"
REGIME3_CSVS = {
    2024: REGIME3_DIR / "sol_features_2024_regime3_current_sensitive_hmm_wide24.csv",
    2025: REGIME3_DIR / "sol_features_2025_regime3_current_sensitive_hmm_wide24.csv",
    2026: REGIME3_DIR / "sol_features_2026_regime3_current_sensitive_hmm_wide24.csv",
}

REGIME_COLS = [
    "regime3_current_sensitive_wide24_bull_prob",
    "regime3_current_sensitive_wide24_bear_prob",
    "regime3_current_sensitive_wide24_chop_prob",
    "regime3_current_sensitive_wide24_confidence",
    "regime3_current_sensitive_wide24_entropy",
    "regime3_current_sensitive_wide24_margin",
]

# SOL equivalents of the ETH Alpha7 SOURCE_COLS. Dropped (no SOL equivalent, documented above):
# tp_sl_action_score, ai_dir_edge, ai_flow_pressure, m7_expected_ret, m7_q50, m7_quality_pred.
SOURCE_COLS = [
    "net_taker_ratio",
    "taker_acceleration",
    "ofi_acceleration",
    "smart_money_flow",
    "funding_price_divergence",
    "hurst_48",
    "mtf_trend_1h",
    "mtf_trend_4h",
    "breakout_strength",
    "rsi",
    *REGIME_COLS,
]

DIRECTIONAL_COLS = [
    "logret_1",
    "price_momentum_3b",
    "price_momentum_6b",
    "price_momentum_12b",
    "price_momentum_24b",
    "ema_cross_signal",
    "linear_slope_12b",
    "linear_slope_24b",
    "higher_high_12b",
    "lower_low_12b",
    "range_atr_proxy",
    "volume_momentum_12b",
]

SEQUENCE_COLS = [*DIRECTIONAL_COLS, *SOURCE_COLS]

DECISION_COLS = [
    "action",
    "side",
    "quality_score",
    "confidence",
    "notional_exposure",
    "leverage",
    "take_profit",
    "stop_loss",
    "max_hold_bars",
    "cooldown_bars",
]

ACTION_SKIP = 0
ACTION_PRIMARY = 1
ACTION_FALLBACK = 2
ACTION_DIM = 3

# Fresh-forward split contract (CLAUDE.md canonical + one additional never-touched window).
TRAIN_END = pd.Timestamp("2025-09-01")
VAL_START = pd.Timestamp("2025-09-01")
VAL_END = pd.Timestamp("2025-12-31 23:59:59")
OOS_START = pd.Timestamp("2026-01-01")
OOS_END = pd.Timestamp("2026-03-31 23:59:59")
FRESH_OOS_START = pd.Timestamp("2026-04-01")
FRESH_OOS_END = pd.Timestamp("2026-07-21 23:59:59")

PRIMARY_CONF_THRESH = 0.20
FALLBACK_CONF_THRESH = 0.06


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _safe_num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise RuntimeError(f"feature contract violation: missing column {col}")
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _load_frame() -> pd.DataFrame:
    feats = []
    for year, path in SOL_FEATURE_CSVS.items():
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        reg = pd.read_csv(REGIME3_CSVS[year])
        reg["timestamp"] = pd.to_datetime(reg["timestamp"])
        reg = reg[["timestamp", *REGIME_COLS]]
        merged = df.merge(reg, on="timestamp", how="inner", validate="one_to_one")
        if len(merged) != len(df):
            raise RuntimeError(f"regime3 merge dropped rows for {year}: {len(df)} -> {len(merged)}")
        feats.append(merged)
    out = pd.concat(feats, axis=0, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    missing = [c for c in SEQUENCE_COLS if c not in out.columns and c not in DIRECTIONAL_COLS]
    if missing:
        raise RuntimeError(f"sol frame missing required columns: {missing}")
    return out


def _linear_slope(close: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=np.float64)
    x = x - x.mean()
    denom = float(np.sum(x * x))

    def slope(values: np.ndarray) -> float:
        y = np.asarray(values, dtype=np.float64)
        if y.size != window:
            return 0.0
        y = y - y.mean()
        return float(np.sum(x * y) / max(denom, 1e-12))

    return close.rolling(window, min_periods=window).apply(slope, raw=True).fillna(0.0)


def _directional_features(df: pd.DataFrame) -> pd.DataFrame:
    close = _safe_num(df, "close").replace(0.0, np.nan).ffill().bfill().fillna(1.0)
    high = _safe_num(df, "high").fillna(close)
    low = _safe_num(df, "low").fillna(close)
    volume = _safe_num(df, "volume")

    ret = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    atr_proxy = ((high - low).abs() / close.abs().clip(lower=1e-12)).rolling(24, min_periods=4).mean().fillna(0.0)
    vol = ret.rolling(24, min_periods=4).std(ddof=0).fillna(ret.std(ddof=0) or 1e-6).abs().clip(lower=1e-6)

    ema9 = close.ewm(span=9, adjust=False, min_periods=1).mean()
    ema21 = close.ewm(span=21, adjust=False, min_periods=1).mean()
    rolling_high_prev = high.rolling(12, min_periods=2).max().shift(1)
    rolling_low_prev = low.rolling(12, min_periods=2).min().shift(1)

    out = pd.DataFrame(index=df.index)
    out["logret_1"] = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for n in (3, 6, 12, 24):
        out[f"price_momentum_{n}b"] = (close / close.shift(n) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["ema_cross_signal"] = ((ema9 - ema21) / close.abs().clip(lower=1e-12) / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["linear_slope_12b"] = (_linear_slope(close, 12) / close.abs().clip(lower=1e-12) / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["linear_slope_24b"] = (_linear_slope(close, 24) / close.abs().clip(lower=1e-12) / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["higher_high_12b"] = (high > rolling_high_prev).fillna(False).astype(float)
    out["lower_low_12b"] = (low < rolling_low_prev).fillna(False).astype(float)
    out["range_atr_proxy"] = atr_proxy
    out["volume_momentum_12b"] = (volume / volume.rolling(12, min_periods=2).mean().replace(0.0, np.nan) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _context_frame(df: pd.DataFrame) -> pd.DataFrame:
    parts = [_directional_features(df), pd.DataFrame({c: _safe_num(df, c) for c in SOURCE_COLS}, index=df.index)]
    out = pd.concat(parts, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _fit_robust_norm(df: pd.DataFrame, cols: list[str]) -> dict[str, Any]:
    arr = df[cols].to_numpy(dtype=np.float64)
    med = np.nanmedian(arr, axis=0)
    q25 = np.nanpercentile(arr, 25, axis=0)
    q75 = np.nanpercentile(arr, 75, axis=0)
    scale = q75 - q25
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    return {"columns": cols, "median": med.tolist(), "scale": scale.tolist()}


def _apply_robust_norm(df: pd.DataFrame, norm: dict[str, Any]) -> np.ndarray:
    cols = list(norm["columns"])
    arr = df[cols].to_numpy(dtype=np.float64)
    med = np.asarray(norm["median"], dtype=np.float64)
    scale = np.asarray(norm["scale"], dtype=np.float64)
    z = (arr - med) / scale
    return np.tanh(np.nan_to_num(z, nan=0.0, posinf=8.0, neginf=-8.0) / 3.0).astype(np.float32)


def _rolling_sequences(arr: np.ndarray, seq_len: int) -> np.ndarray:
    pad = np.repeat(arr[:1], int(seq_len) - 1, axis=0)
    padded = np.concatenate([pad, arr], axis=0)
    out = np.lib.stride_tricks.sliding_window_view(padded, int(seq_len), axis=0)
    return np.swapaxes(out, 1, 2).copy().astype(np.float32)


def _direction_labels(df: pd.DataFrame, *, horizon: int, barrier: float) -> np.ndarray:
    close = _safe_num(df, "close").replace(0.0, np.nan).ffill().bfill().fillna(1.0)
    fwd = (close.shift(-int(horizon)) / close - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = np.zeros(len(df), dtype=np.int64)
    y[fwd.to_numpy(dtype=np.float64) > float(barrier)] = 1
    y[fwd.to_numpy(dtype=np.float64) < -float(barrier)] = 2
    return y


class Alpha8MambaEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int, emb_dim: int, n_classes: int = 3):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_dim, d_model), nn.LayerNorm(d_model), nn.SiLU())
        self.mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.norm = nn.LayerNorm(d_model)
        self.emb = nn.Sequential(nn.Linear(d_model, emb_dim), nn.SiLU())
        self.head = nn.Linear(emb_dim, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)
        h = self.mamba(h)
        last = self.norm(h[:, -1, :])
        emb = self.emb(last)
        return self.head(emb), emb


@dataclass
class MambaArtifacts:
    model: Alpha8MambaEncoder
    train_diag: dict[str, Any]


def _train_mamba(seq: np.ndarray, labels: np.ndarray, *, device, epochs, batch_size, d_model, emb_dim) -> MambaArtifacts:
    model = Alpha8MambaEncoder(seq.shape[-1], d_model=d_model, emb_dim=emb_dim).to(device)
    x = torch.from_numpy(seq)
    y = torch.from_numpy(labels.astype(np.int64))
    dl = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True, drop_last=False)
    counts = np.bincount(labels.astype(np.int64), minlength=3).astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / max(weights.mean(), 1e-12)
    class_weight = torch.tensor(weights, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    last = {"loss": 0.0, "acc": 0.0}
    for epoch in range(1, int(epochs) + 1):
        model.train()
        losses, correct, total = [], 0, 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            logits, _ = model(xb)
            loss = F.cross_entropy(logits, yb, weight=class_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            losses.append(float(loss.item()))
            pred = torch.argmax(logits.detach(), dim=-1)
            correct += int((pred == yb).sum().item())
            total += int(yb.numel())
        last = {"epoch": int(epoch), "loss": float(np.mean(losses)), "acc": float(correct / max(total, 1))}
        print(json.dumps({"stage": "mamba_train", **last}), flush=True)
    return MambaArtifacts(model=model.cpu(), train_diag=last)


def _mamba_predict(model: Alpha8MambaEncoder, seq: np.ndarray, *, device, batch_size) -> tuple[np.ndarray, np.ndarray]:
    model = model.to(device)
    model.eval()
    probs, embs = [], []
    with torch.no_grad():
        for start in range(0, len(seq), int(batch_size)):
            xb = torch.from_numpy(seq[start : start + int(batch_size)]).to(device)
            logits, emb = model(xb)
            probs.append(F.softmax(logits, dim=-1).cpu().numpy().astype(np.float32))
            embs.append(emb.cpu().numpy().astype(np.float32))
    return np.concatenate(probs), np.concatenate(embs)


def _lightgbm_features(ctx: pd.DataFrame, mamba_probs: np.ndarray, mamba_emb: np.ndarray) -> pd.DataFrame:
    out = ctx[SEQUENCE_COLS].copy()
    for i, name in enumerate(["hold", "long", "short"]):
        out[f"mamba_p_{name}"] = mamba_probs[:, i]
        out[f"mamba_p_{name}_delta1"] = pd.Series(mamba_probs[:, i]).diff().fillna(0.0).to_numpy()
        out[f"mamba_p_{name}_mean3"] = pd.Series(mamba_probs[:, i]).rolling(3, min_periods=1).mean().to_numpy()
    for i in range(mamba_emb.shape[1]):
        out[f"mamba_emb_{i:02d}"] = mamba_emb[:, i]
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _fit_lgbm(x: pd.DataFrame, y: np.ndarray) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=3, n_estimators=360, learning_rate=0.035, num_leaves=31,
        max_depth=5, min_child_samples=80, subsample=0.85, colsample_bytree=0.85, reg_alpha=1.0,
        reg_lambda=2.0, class_weight="balanced", random_state=220722, n_jobs=-1, verbosity=-1,
    )
    model.fit(x, y)
    return model


def _alpha8_prob_frame(probs: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame({"alpha8_p_hold": probs[:, 0], "alpha8_p_long": probs[:, 1], "alpha8_p_short": probs[:, 2]})
    out["alpha8_dir_edge"] = out["alpha8_p_long"] - out["alpha8_p_short"]
    out["alpha8_confidence"] = np.maximum(out["alpha8_p_long"], out["alpha8_p_short"]) - out["alpha8_p_hold"]
    out["alpha8_direction_abs"] = np.abs(out["alpha8_dir_edge"])
    for col in ["alpha8_p_hold", "alpha8_p_long", "alpha8_p_short", "alpha8_dir_edge", "alpha8_confidence"]:
        out[f"{col}_delta1"] = out[col].diff().fillna(0.0)
        out[f"{col}_mean3"] = out[col].rolling(3, min_periods=1).mean()
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _make_template_decisions(probs: np.ndarray, *, conf_thresh: float, tp: float, sl: float,
                              notional: float, leverage: float, max_hold: int, cooldown: int) -> pd.DataFrame:
    p_hold, p_long, p_short = probs[:, 0], probs[:, 1], probs[:, 2]
    side = np.where(p_long >= p_short, 1, -1)
    conf = np.maximum(p_long, p_short) - p_hold
    active = conf > float(conf_thresh)
    n = len(probs)
    out = pd.DataFrame(
        {
            "action": active.astype(int),
            "side": np.where(active, side, 0).astype(int),
            "quality_score": conf,
            "confidence": conf,
            "notional_exposure": np.where(active, notional, 0.0),
            "leverage": np.full(n, leverage, dtype=np.float64),
            "take_profit": np.full(n, tp, dtype=np.float64),
            "stop_loss": np.full(n, sl, dtype=np.float64),
            "max_hold_bars": np.full(n, max_hold, dtype=np.int64),
            "cooldown_bars": np.full(n, cooldown, dtype=np.int64),
        }
    )
    return out


def _decision_num(dec: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(dec[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _state_frame(frame: pd.DataFrame, primary: pd.DataFrame, fallback: pd.DataFrame,
                  alpha8_probs: np.ndarray, mamba_probs: np.ndarray, mamba_emb: np.ndarray) -> pd.DataFrame:
    parts = [_directional_features(frame).reset_index(drop=True)]
    parts.append(pd.DataFrame({c: _safe_num(frame, c) for c in SOURCE_COLS}, index=frame.index).reset_index(drop=True))
    for prefix, dec in (("primary", primary), ("fallback", fallback)):
        d = pd.DataFrame(index=range(len(frame)))
        for col in DECISION_COLS:
            d[f"{prefix}_{col}"] = _decision_num(dec, col).to_numpy(dtype=np.float64)
        parts.append(d)
    pa = _decision_num(primary, "action").astype(int)
    ps = _decision_num(primary, "side").astype(int)
    fa = _decision_num(fallback, "action").astype(int)
    fs = _decision_num(fallback, "side").astype(int)
    pq = _decision_num(primary, "quality_score")
    fq = _decision_num(fallback, "quality_score")
    meta = pd.DataFrame(index=range(len(frame)))
    meta["primary_active"] = ((pa != 0) & (ps != 0)).astype(float).to_numpy()
    meta["fallback_active"] = ((fa != 0) & (fs != 0)).astype(float).to_numpy()
    meta["side_agree"] = ((ps == fs) & (ps != 0)).astype(float).to_numpy()
    meta["quality_diff_primary_fallback"] = (pq - fq).to_numpy()
    parts.append(meta)
    alpha = _alpha8_prob_frame(alpha8_probs).reset_index(drop=True)
    mprob = pd.DataFrame({"mamba_p_hold": mamba_probs[:, 0], "mamba_p_long": mamba_probs[:, 1], "mamba_p_short": mamba_probs[:, 2]})
    emb_cols = min(16, mamba_emb.shape[1])
    memb = pd.DataFrame({f"mamba_state_emb_{i:02d}": mamba_emb[:, i] for i in range(emb_cols)})
    parts.extend([alpha, mprob, memb])
    out = pd.concat(parts, axis=1)
    if out.columns.duplicated().any():
        dup = out.columns[out.columns.duplicated()].tolist()
        raise RuntimeError(f"duplicate SOL alpha8 state columns: {dup[:20]}")
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _fit_norm(x: pd.DataFrame) -> dict[str, Any]:
    arr = x.to_numpy(dtype=np.float64)
    med = np.nanmedian(arr, axis=0)
    q25 = np.nanpercentile(arr, 25, axis=0)
    q75 = np.nanpercentile(arr, 75, axis=0)
    scale = q75 - q25
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    return {"columns": list(x.columns), "median": med.tolist(), "scale": scale.tolist()}


def _apply_norm(x: pd.DataFrame, norm: dict[str, Any]) -> np.ndarray:
    cols = list(norm["columns"])
    arr = x[cols].to_numpy(dtype=np.float64)
    med = np.asarray(norm["median"], dtype=np.float64)
    scale = np.asarray(norm["scale"], dtype=np.float64)
    z = (arr - med) / scale
    return np.tanh(np.nan_to_num(z, nan=0.0, posinf=8.0, neginf=-8.0) / 3.0).astype(np.float32)


def _first_hit(path: np.ndarray, tp: float, sl: float, hold: int) -> int:
    m = min(int(max(1, hold)), len(path))
    if m <= 1:
        return 0
    p = path[:m]
    hit = np.flatnonzero((p >= float(tp)) | (p <= -abs(float(sl))))
    return int(hit[0]) if hit.size else int(m - 1)


def _candidate_reward(close: np.ndarray, i: int, dec_row: pd.Series, *, fee: float, slip: float) -> tuple[float, dict[str, Any]]:
    action = int(dec_row.get("action", 0) or 0)
    side = int(dec_row.get("side", 0) or 0)
    if action == 0 or side == 0:
        return 0.0, {"active": 0, "net": 0.0, "win": 0}
    notional = float(dec_row.get("notional_exposure", 0.0) or 0.0)
    tp = float(dec_row.get("take_profit", 0.0) or 0.0)
    sl = float(dec_row.get("stop_loss", 0.0) or 0.0)
    hold = int(dec_row.get("max_hold_bars", 0) or 0)
    if notional <= 0.0 or hold <= 0:
        return 0.0, {"active": 0, "net": 0.0, "win": 0}
    entry_i = min(int(i) + 1, len(close) - 1)
    end = min(len(close), entry_i + hold + 1)
    if end <= entry_i + 1:
        return 0.0, {"active": 0, "net": 0.0, "win": 0}
    entry = max(float(close[entry_i]), 1e-12)
    fut = close[entry_i + 1 : end]
    side_ret = ((fut / entry) - 1.0) * float(side)
    path = side_ret * notional
    exit_i = _first_hit(path, tp, sl, hold)
    gross = float(path[exit_i])
    net = gross - 2.0 * (fee + slip) * notional
    win = int(net > 0.0)
    reward = 140.0 * net + (0.35 if win else -0.18)
    reward += (45.0 * net) if net > 0.0 else (25.0 * net)
    return float(reward), {"active": 1, "net": float(net), "win": win}


@dataclass
class DatasetBundle:
    states: np.ndarray
    next_states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray


def _build_counterfactual_dataset(frame: pd.DataFrame, states: np.ndarray, primary: pd.DataFrame,
                                   fallback: pd.DataFrame, *, fee: float, slip: float) -> tuple[DatasetBundle, dict[str, Any]]:
    close = _safe_num(frame, "close").to_numpy(dtype=np.float64)
    s_list, sp_list, a_list, r_list, d_list = [], [], [], [], []
    reward_stats: dict[int, list[float]] = {ACTION_SKIP: [], ACTION_PRIMARY: [], ACTION_FALLBACK: []}
    win_stats: dict[int, list[int]] = {ACTION_SKIP: [], ACTION_PRIMARY: [], ACTION_FALLBACK: []}
    for i in range(len(frame) - 2):
        rewards = {
            ACTION_SKIP: (0.0, {"active": 0, "net": 0.0, "win": 0}),
            ACTION_PRIMARY: _candidate_reward(close, i, primary.iloc[i], fee=fee, slip=slip),
            ACTION_FALLBACK: _candidate_reward(close, i, fallback.iloc[i], fee=fee, slip=slip),
        }
        for action, (reward, meta) in rewards.items():
            s_list.append(states[i])
            sp_list.append(states[i + 1])
            a_list.append(int(action))
            r_list.append(float(reward))
            d_list.append(1.0 if i == len(frame) - 3 else 0.0)
            if int(meta["active"]) == 1:
                reward_stats[int(action)].append(float(meta["net"]))
                win_stats[int(action)].append(int(meta["win"]))
    rewards_np = np.asarray(r_list, dtype=np.float32)
    scale = float(np.nanstd(rewards_np))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    rewards_np = np.clip(rewards_np / scale, -8.0, 8.0).astype(np.float32)
    diagnostics = {
        "reward_scale": scale,
        "candidate_net_mean": {str(k): float(np.mean(v)) if v else 0.0 for k, v in reward_stats.items()},
        "candidate_win_rate": {str(k): float(np.mean(v)) if v else 0.0 for k, v in win_stats.items()},
        "candidate_active_count": {str(k): int(len(v)) for k, v in reward_stats.items()},
    }
    return (
        DatasetBundle(
            states=np.asarray(s_list, dtype=np.float32), next_states=np.asarray(sp_list, dtype=np.float32),
            actions=np.asarray(a_list, dtype=np.int64), rewards=rewards_np, dones=np.asarray(d_list, dtype=np.float32),
        ),
        diagnostics,
    )


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(0.05),
            nn.Linear(256, 192), nn.SiLU(), nn.Linear(192, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(0.05),
            nn.Linear(256, 192), nn.SiLU(), nn.Linear(192, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _train_dsac_offline(data: DatasetBundle, *, state_dim: int, action_dim: int, device, steps: int,
                         batch_size: int, gamma: float = 0.995, tau: float = 0.01, lr: float = 2.5e-4) -> dict[str, Any]:
    actor = Actor(state_dim, action_dim).to(device)
    q1, q2 = Critic(state_dim, action_dim).to(device), Critic(state_dim, action_dim).to(device)
    tq1, tq2 = Critic(state_dim, action_dim).to(device), Critic(state_dim, action_dim).to(device)
    tq1.load_state_dict(q1.state_dict())
    tq2.load_state_dict(q2.state_dict())
    log_alpha = torch.tensor(math.log(0.15), device=device, requires_grad=True)
    target_entropy = 0.75 * math.log(float(action_dim))
    opt_actor = torch.optim.AdamW(actor.parameters(), lr=lr, weight_decay=1e-5)
    opt_q1 = torch.optim.AdamW(q1.parameters(), lr=lr, weight_decay=1e-5)
    opt_q2 = torch.optim.AdamW(q2.parameters(), lr=lr, weight_decay=1e-5)
    opt_alpha = torch.optim.Adam([log_alpha], lr=lr)
    dl = DataLoader(
        TensorDataset(torch.from_numpy(data.states), torch.from_numpy(data.next_states),
                       torch.from_numpy(data.actions), torch.from_numpy(data.rewards), torch.from_numpy(data.dones)),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )
    it = iter(dl)
    last = {"q_loss": 0.0, "actor_loss": 0.0, "alpha": 0.0, "entropy": 0.0}
    for step in range(1, int(steps) + 1):
        try:
            s, sp, a, r, d = next(it)
        except StopIteration:
            it = iter(dl)
            s, sp, a, r, d = next(it)
        s, sp, a, r, d = s.to(device), sp.to(device), a.to(device), r.to(device), d.to(device)
        with torch.no_grad():
            next_logits = actor(sp)
            next_logp = F.log_softmax(next_logits, dim=-1)
            next_pi = next_logp.exp()
            alpha = log_alpha.exp()
            next_q = torch.min(tq1(sp), tq2(sp))
            v_next = (next_pi * (next_q - alpha * next_logp)).sum(dim=-1)
            y = r + (1.0 - d) * gamma * v_next
        qa1 = q1(s).gather(1, a.view(-1, 1)).squeeze(1)
        qa2 = q2(s).gather(1, a.view(-1, 1)).squeeze(1)
        q_loss = F.smooth_l1_loss(qa1, y) + F.smooth_l1_loss(qa2, y)
        opt_q1.zero_grad(set_to_none=True)
        opt_q2.zero_grad(set_to_none=True)
        q_loss.backward()
        nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), 5.0)
        opt_q1.step()
        opt_q2.step()
        logits = actor(s)
        logp = F.log_softmax(logits, dim=-1)
        pi = logp.exp()
        alpha = log_alpha.exp()
        q_min = torch.min(q1(s), q2(s))
        actor_loss = (pi * (alpha * logp - q_min)).sum(dim=-1).mean()
        opt_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
        opt_actor.step()
        entropy = -(pi * logp).sum(dim=-1).mean().detach()
        alpha_loss = -(log_alpha * (entropy - target_entropy)).mean()
        opt_alpha.zero_grad(set_to_none=True)
        alpha_loss.backward()
        opt_alpha.step()
        log_alpha.data.clamp_(math.log(1e-4), math.log(3.0))
        with torch.no_grad():
            for p, tp in zip(q1.parameters(), tq1.parameters()):
                tp.data.mul_(1.0 - tau).add_(tau * p.data)
            for p, tp in zip(q2.parameters(), tq2.parameters()):
                tp.data.mul_(1.0 - tau).add_(tau * p.data)
        if step % 250 == 0:
            last = {"q_loss": float(q_loss.item()), "actor_loss": float(actor_loss.item()),
                    "alpha": float(log_alpha.exp().item()), "entropy": float(entropy.item()), "step": int(step)}
        if step % 1000 == 0:
            print(json.dumps({"stage": "dsac_train_progress", **last}), flush=True)
    return {"actor": actor.cpu(), "train_diag": last}


def _policy_action(actor: nn.Module, states: np.ndarray, *, device) -> np.ndarray:
    actor = actor.to(device)
    actor.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(states), 8192):
            x = torch.from_numpy(states[start : start + 8192]).to(device)
            logits = actor(x)
            out.append(torch.argmax(logits, dim=-1).cpu().numpy().astype(np.int64))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def _compose_decisions(primary: pd.DataFrame, fallback: pd.DataFrame, actions: np.ndarray) -> pd.DataFrame:
    out = primary.copy().reset_index(drop=True)
    fallback = fallback.reset_index(drop=True)
    for i in range(len(out)):
        a = int(actions[i])
        if a == ACTION_SKIP:
            out.loc[i, ["action", "side", "notional_exposure"]] = [0, 0, 0.0]
        elif a == ACTION_FALLBACK:
            out.iloc[i] = fallback.iloc[i]
        elif a == ACTION_PRIMARY:
            continue
        else:
            raise RuntimeError(f"invalid DSAC action: {a}")
    return out


def _usage(actions: np.ndarray) -> dict[str, int]:
    return {
        "skip": int(np.sum(actions == ACTION_SKIP)),
        "primary": int(np.sum(actions == ACTION_PRIMARY)),
        "fallback": int(np.sum(actions == ACTION_FALLBACK)),
    }


def _simulate(frame: pd.DataFrame, dec: pd.DataFrame, *, fee: float, slip: float) -> dict[str, Any]:
    """Own causal bar-by-bar walk-forward PnL simulator (no stored ledger reuse).

    At each decision bar i with action!=skip, entry = close[i+1]; walks forward bar-by-bar
    checking TP/SL barrier hit (first_hit) up to max_hold_bars, exits at first hit or timeout,
    then stays flat (busy) until exit + cooldown_bars before considering a new entry.
    """
    close = _safe_num(frame, "close").to_numpy(dtype=np.float64)
    n = len(frame)
    equity = 0.0
    peak = 0.0
    mdd = 0.0
    trades: list[dict[str, Any]] = []
    busy_until = -1
    for i in range(n - 2):
        if i <= busy_until:
            continue
        row = dec.iloc[i]
        action = int(row.get("action", 0) or 0)
        side = int(row.get("side", 0) or 0)
        if action == 0 or side == 0:
            continue
        notional = float(row.get("notional_exposure", 0.0) or 0.0)
        tp = float(row.get("take_profit", 0.0) or 0.0)
        sl = float(row.get("stop_loss", 0.0) or 0.0)
        hold = int(row.get("max_hold_bars", 0) or 0)
        cooldown = int(row.get("cooldown_bars", 0) or 0)
        if notional <= 0.0 or hold <= 0:
            continue
        entry_i = min(i + 1, n - 1)
        end = min(n, entry_i + hold + 1)
        if end <= entry_i + 1:
            continue
        entry = max(float(close[entry_i]), 1e-12)
        fut = close[entry_i + 1 : end]
        side_ret = ((fut / entry) - 1.0) * side
        path = side_ret * notional
        exit_i = _first_hit(path, tp, sl, hold)
        gross = float(path[exit_i])
        net = gross - 2.0 * (fee + slip) * notional
        equity += net
        peak = max(peak, equity)
        mdd = min(mdd, equity - peak)
        trades.append({"entry_i": int(entry_i), "exit_i": int(entry_i + exit_i + 1), "net": net, "win": bool(net > 0.0)})
        busy_until = entry_i + exit_i + cooldown
    n_trades = len(trades)
    wins = sum(1 for t in trades if t["win"])
    return {
        "pnl_pct": float(equity * 100.0),
        "mdd_pct": float(mdd * 100.0),
        "trades": int(n_trades),
        "win_rate_pct": float(100.0 * wins / n_trades) if n_trades else 0.0,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
    }


def _pick(grid: pd.DataFrame, split: str, variant: str) -> dict[str, Any]:
    row = grid[(grid["split"].eq(split)) & (grid["variant"].eq(variant))]
    return {} if row.empty else row.iloc[0].to_dict()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mamba-epochs", type=int, default=3)
    p.add_argument("--mamba-batch-size", type=int, default=512)
    p.add_argument("--mamba-d-model", type=int, default=96)
    p.add_argument("--mamba-emb-dim", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--label-horizon", type=int, default=12)
    p.add_argument("--label-barrier", type=float, default=0.0025)
    p.add_argument("--dsac-steps", type=int, default=2500)
    p.add_argument("--dsac-batch-size", type=int, default=768)
    args = p.parse_args()

    _seed_everything(220722)
    if not torch.cuda.is_available():
        raise RuntimeError("Alpha8 Mamba requires CUDA; mamba_ssm kernels are not CPU-compatible in this environment.")
    device = torch.device("cuda")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    full = _load_frame()
    splits = {
        "train": full[full["timestamp"] < TRAIN_END].reset_index(drop=True),
        "val": full[(full["timestamp"] >= VAL_START) & (full["timestamp"] <= VAL_END)].reset_index(drop=True),
        "oos": full[(full["timestamp"] >= OOS_START) & (full["timestamp"] <= OOS_END)].reset_index(drop=True),
        "oos_fresh": full[(full["timestamp"] >= FRESH_OOS_START) & (full["timestamp"] <= FRESH_OOS_END)].reset_index(drop=True),
    }
    for name, df in splits.items():
        print(json.dumps({"stage": "split", "name": name, "rows": len(df),
                           "start": str(df["timestamp"].min()) if len(df) else None,
                           "end": str(df["timestamp"].max()) if len(df) else None}), flush=True)
        if len(df) < 50:
            raise RuntimeError(f"split {name} too small: {len(df)} rows")

    ctx = {k: _context_frame(v) for k, v in splits.items()}
    ctx_norm = _fit_robust_norm(ctx["train"], SEQUENCE_COLS)
    seq = {k: _rolling_sequences(_apply_robust_norm(v, ctx_norm), args.seq_len) for k, v in ctx.items()}
    labels = {k: _direction_labels(v, horizon=args.label_horizon, barrier=args.label_barrier) for k, v in splits.items()}

    print(json.dumps({"stage": "alpha8_sol_start", "device": str(device),
                       "train_rows": len(splits["train"]), "seq_shape": list(seq["train"].shape),
                       "label_counts_train": np.bincount(labels["train"], minlength=3).astype(int).tolist()}), flush=True)

    mamba_art = _train_mamba(seq["train"], labels["train"], device=device, epochs=args.mamba_epochs,
                              batch_size=args.mamba_batch_size, d_model=args.mamba_d_model, emb_dim=args.mamba_emb_dim)
    m_prob, m_emb = {}, {}
    for k in splits:
        m_prob[k], m_emb[k] = _mamba_predict(mamba_art.model, seq[k], device=device, batch_size=args.mamba_batch_size)

    x_lgb = {k: _lightgbm_features(ctx[k], m_prob[k], m_emb[k]) for k in splits}
    lgbm = _fit_lgbm(x_lgb["train"], labels["train"])
    a8_prob = {k: lgbm.predict_proba(x_lgb[k]) for k in splits}

    primary = {k: _make_template_decisions(a8_prob[k], conf_thresh=PRIMARY_CONF_THRESH, tp=0.012, sl=0.006,
                                            notional=2.0, leverage=3.0, max_hold=48, cooldown=6) for k in splits}
    fallback = {k: _make_template_decisions(a8_prob[k], conf_thresh=FALLBACK_CONF_THRESH, tp=0.008, sl=0.005,
                                             notional=1.0, leverage=2.0, max_hold=24, cooldown=4) for k in splits}

    state_df = {k: _state_frame(splits[k], primary[k], fallback[k], a8_prob[k], m_prob[k], m_emb[k]) for k in splits}
    state_norm = _fit_norm(state_df["train"])
    x_state = {k: _apply_norm(state_df[k], state_norm) for k in splits}

    fee, slip = 0.0005, 0.0002
    data, data_diag = _build_counterfactual_dataset(splits["train"], x_state["train"], primary["train"], fallback["train"], fee=fee, slip=slip)
    print(json.dumps({"stage": "dsac_start", "state_dim": int(x_state["train"].shape[1]),
                       "samples": int(len(data.states)), "dataset_diagnostics": data_diag}), flush=True)
    trained = _train_dsac_offline(data, state_dim=int(x_state["train"].shape[1]), action_dim=ACTION_DIM,
                                   device=device, steps=int(args.dsac_steps), batch_size=int(args.dsac_batch_size))
    actor = trained["actor"]

    act = {k: _policy_action(actor, x_state[k], device=device) for k in splits}
    alpha8_dec = {k: _compose_decisions(primary[k], fallback[k], act[k]) for k in splits}

    def baseline_decisions(k: str) -> pd.DataFrame:
        # Baseline = always-primary-template (no DSAC routing / no fallback), for comparison.
        return primary[k]

    rows: list[dict[str, Any]] = []
    for split in splits:
        for name, dec in [("baseline_primary_only", baseline_decisions(split)), ("alpha8_regime3_mamba_lgbm_dsac", alpha8_dec[split])]:
            vals = _simulate(splits[split], dec, fee=fee, slip=slip)
            rows.append({"split": split, "variant": name, **vals})
    grid = pd.DataFrame(rows)
    grid_path = OUT_DIR / "grid.csv"
    grid.to_csv(grid_path, index=False)

    torch.save(
        {
            "model_id": MODEL_ID,
            "mamba_state_dict": mamba_art.model.state_dict(),
            "dsac_actor_state_dict": actor.state_dict(),
            "state_dim": int(x_state["train"].shape[1]),
            "action_dim": ACTION_DIM,
            "state_columns": list(state_norm["columns"]),
            "state_normalizer": state_norm,
            "context_normalizer": ctx_norm,
            "sequence_cols": SEQUENCE_COLS,
            "label_horizon": int(args.label_horizon),
            "label_barrier": float(args.label_barrier),
        },
        OUT_DIR / "alpha8_sol_mamba_dsac.pt",
    )
    joblib.dump(lgbm, OUT_DIR / "alpha8_sol_directional_lgbm.pkl")
    (OUT_DIR / "state_columns.json").write_text(json.dumps(list(state_norm["columns"]), indent=2) + "\n")

    summary = {
        "model_id": MODEL_ID,
        "design": "Alpha8 ported to SOL: Regime3 (bull/bear/chop) -> CUDA Mamba sequence encoder -> "
                   "LightGBM directional alpha probabilities -> discrete SAC execution router over "
                   "skip/primary-template/fallback-template (rule-based templates replace the "
                   "ETH-only Alpha7 governor-policy primary/fallback parents, which have no SOL equivalent).",
        "live_wired": False,
        "simplifications_vs_eth_alpha6_alpha7_chain": [
            "Layer1 regime: SOL Regime3 (bull/bear/chop, no whipsaw, no future regime4_pred_* surface) "
            "replaces ETH clean_regime4_state24_sticky090_v2_*/regime4_pred_*.",
            "Directional/flow context: SOL raw 5m feature frame (data/splits/year_oos/sol_features_*.csv); "
            "tp_sl_action_score, ai_dir_edge, ai_flow_pressure, m7_expected_ret, m7_q50, m7_quality_pred "
            "dropped (SOL has no AI-forecast/M7-ensemble columns).",
            "Layer3 direction label: fixed 12-bar/0.25% close-price barrier (same definition ETH Alpha8 "
            "already used for its own Mamba supervision), not the ETH Alpha6 governor-policy barrier "
            "definition and not the separate SOL zigzag action-label pipeline.",
            "Layer4 primary/fallback: two fixed rule-based decision templates keyed on this script's own "
            "LightGBM p_hold/p_long/p_short confidence, NOT the ETH Alpha7 trained governor-policy "
            "primary/fallback parent models (asset- and regime4-column-specific, no SOL equivalent).",
            "Backtest/PnL: own from-scratch causal bar-by-bar walk-forward simulator (_simulate), not the "
            "ETH _combo_metrics/_metrics harness (which requires ETH governor-policy parent artifacts).",
        ],
        "train_rows": {k: int(len(splits[k])) for k in splits},
        "split_windows": {
            "train": f"< {TRAIN_END}",
            "val": f"{VAL_START} .. {VAL_END}",
            "oos": f"{OOS_START} .. {OOS_END}",
            "oos_fresh": f"{FRESH_OOS_START} .. {FRESH_OOS_END}",
        },
        "mamba": {"seq_len": int(args.seq_len), "d_model": int(args.mamba_d_model), "embedding_dim": int(args.mamba_emb_dim),
                  "epochs": int(args.mamba_epochs), "train_diag": mamba_art.train_diag},
        "lightgbm": {"feature_count": int(x_lgb["train"].shape[1]), "label_horizon": int(args.label_horizon),
                     "label_barrier": float(args.label_barrier),
                     "label_counts": {k: np.bincount(labels[k], minlength=3).astype(int).tolist() for k in splits}},
        "dsac": {"state_dim": int(x_state["train"].shape[1]), "action_dim": ACTION_DIM, "steps": int(args.dsac_steps),
                 "batch_size": int(args.dsac_batch_size), "dataset_diagnostics": data_diag,
                 "train_diag": trained["train_diag"], "action_usage": {k: _usage(act[k]) for k in splits}},
        "fee": fee, "slip": slip,
        "results": {f"{split}_{name.split('_')[0]}": _pick(grid, split, name)
                    for split in splits for name in ["baseline_primary_only", "alpha8_regime3_mamba_lgbm_dsac"]},
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "artifacts": {
            "grid": str(grid_path),
            "torch": str(OUT_DIR / "alpha8_sol_mamba_dsac.pt"),
            "lightgbm": str(OUT_DIR / "alpha8_sol_directional_lgbm.pkl"),
            "state_columns": str(OUT_DIR / "state_columns.json"),
        },
    }
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"summary": str(summary_path)}, ensure_ascii=False, indent=2))
    print(grid.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
