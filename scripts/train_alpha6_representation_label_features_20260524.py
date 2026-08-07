#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.alpha6_catboost_5head_policy_20260522 import (  # noqa: E402
    DEFAULT_FEATURE_CSV,
    DEFAULT_LABEL_DIR,
    DEFAULT_SPEC_DIR,
    _label_frame,
    _numeric_matrix,
    _read_feature_frame,
    _read_spec,
)
from scripts.alpha6_catboost_entry_quality_exit_policy_20260522 import CONTEXT_COLS  # noqa: E402


@dataclass(frozen=True)
class RepConfig:
    seq_len: int = 64
    horizon: int = 24
    hidden: int = 96
    emb_dim: int = 64
    batch_size: int = 192
    epochs: int = 3
    lr: float = 1e-3
    mc_paths: int = 24
    diffusion_steps: int = 24


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, x: np.ndarray, idx: np.ndarray, seq_len: int, future: np.ndarray | None = None, labels: np.ndarray | None = None) -> None:
        self.x = np.asarray(x, dtype=np.float32)
        self.idx = np.asarray(idx, dtype=np.int64)
        self.seq_len = int(seq_len)
        self.future = None if future is None else np.asarray(future, dtype=np.float32)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)

    def __len__(self) -> int:
        return int(len(self.idx))

    def __getitem__(self, j: int) -> tuple[torch.Tensor, ...]:
        i = int(self.idx[j])
        seq = self.x[i - self.seq_len + 1 : i + 1]
        out: list[torch.Tensor] = [torch.from_numpy(seq)]
        if self.future is not None:
            out.append(torch.from_numpy(self.future[i]))
        if self.labels is not None:
            out.append(torch.tensor(float(self.labels[i]), dtype=torch.float32))
        return tuple(out)


class ConvEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, emb_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden, 3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=4, dilation=4),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=8, dilation=8),
            nn.GELU(),
        )
        self.proj = nn.Linear(hidden, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x.transpose(1, 2))
        z = z[..., -x.shape[1] :]
        return F.normalize(self.proj(z[:, :, -1]), dim=-1)


class MambaRisk(nn.Module):
    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("mamba_ssm is required for the real Mamba representation preset") from exc
        self.inp = nn.Linear(in_dim, hidden)
        self.mamba = Mamba(d_model=hidden, d_state=16, d_conv=4, expand=2)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.mamba(self.inp(x))
        return self.head(h[:, -1]).squeeze(-1)


class TimeGradNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int, horizon: int) -> None:
        super().__init__()
        self.encoder = nn.GRU(in_dim, hidden, batch_first=True)
        self.t_emb = nn.Embedding(256, hidden)
        self.net = nn.Sequential(
            nn.Linear(hidden + horizon + hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, horizon),
        )

    def forward(self, x: torch.Tensor, noisy_future: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        _, h = self.encoder(x)
        cond = h[-1]
        te = self.t_emb(t)
        return self.net(torch.cat([cond, noisy_future, te], dim=-1))


class TimeLLMReprogrammer(nn.Module):
    def __init__(self, in_dim: int, hidden: int, model_name: str) -> None:
        super().__init__()
        from transformers import AutoModel
        self.llm = AutoModel.from_pretrained(model_name)
        for p in self.llm.parameters():
            p.requires_grad_(False)
        llm_dim = int(self.llm.config.hidden_size)
        self.patch = nn.Linear(in_dim, llm_dim)
        self.head = nn.Sequential(nn.Linear(llm_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.patch(x)
        out = self.llm(inputs_embeds=emb).last_hidden_state
        return self.head(out[:, -1]).squeeze(-1)


def _select_sequence_features(frame: pd.DataFrame, spec_features: list[str]) -> list[str]:
    preferred = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "atr14_pct",
        "realized_vol_ratio",
        "volatility_z",
        "jump_z",
        "obi",
        "cvp_volume_imbalance",
        "taker_buy_ratio",
        "nif_whale",
        "eai",
        "last_funding_rate",
        "funding_price_divergence",
        "ou_funding_z",
        "clean_regime4_state24_sticky090_v2_instability_prob",
        "clean_regime4_state24_sticky090_v2_whipsaw_prob",
        "clean_regime4_state24_sticky090_v2_trend_prob",
    ]
    cols = [c for c in preferred if c in frame.columns]
    for c in spec_features:
        if c in frame.columns and c not in cols:
            cols.append(c)
        if len(cols) >= 48:
            break
    return cols


def _valid_indices(n: int, seq_len: int, horizon: int, allowed: np.ndarray) -> np.ndarray:
    allowed_set = set(int(i) for i in allowed)
    idx = [i for i in allowed if i >= seq_len - 1 and i + horizon < n and all((i - k) in allowed_set for k in range(seq_len))]
    return np.asarray(idx, dtype=np.int64)


def _future_returns(close: np.ndarray, horizon: int) -> np.ndarray:
    fut = np.zeros((len(close), horizon), dtype=np.float32)
    for k in range(1, horizon + 1):
        src = np.minimum(np.arange(len(close)) + k, len(close) - 1)
        fut[:, k - 1] = (close[src] / np.maximum(close, 1e-12) - 1.0).astype(np.float32)
    return fut


def _adverse_label(future: np.ndarray, atr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    fut = future[idx]
    atr_now = np.maximum(atr[idx], 1e-6)
    long_bad = np.min(fut, axis=1) < -1.15 * atr_now
    short_bad = np.max(fut, axis=1) > 1.15 * atr_now
    return (long_bad | short_bad).astype(np.float32)


def _profit_mask(future: np.ndarray, idx: np.ndarray) -> np.ndarray:
    fut = future[idx]
    best = np.maximum(np.max(fut, axis=1), -np.min(fut, axis=1))
    worst = np.minimum(np.min(fut, axis=1), -np.max(fut, axis=1))
    return best > (np.abs(worst) + 0.0015)


def _loader(ds: SequenceDataset, batch_size: int, shuffle: bool = True) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(ds, batch_size=int(batch_size), shuffle=shuffle, drop_last=False, num_workers=0)


def _augment(x: torch.Tensor) -> torch.Tensor:
    noise = 0.02 * torch.randn_like(x)
    mask = (torch.rand_like(x[..., :1]) > 0.10).float()
    return x * mask + noise


def _contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, temp: float = 0.2) -> torch.Tensor:
    logits = z1 @ z2.T / temp
    labels = torch.arange(z1.shape[0], device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def _train_ts2vec(x: np.ndarray, idx: np.ndarray, cfg: RepConfig, device: torch.device) -> ConvEncoder:
    model = ConvEncoder(x.shape[1], cfg.hidden, cfg.emb_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    ds = SequenceDataset(x, idx, cfg.seq_len)
    for _ in range(cfg.epochs):
        for (seq,) in _loader(ds, cfg.batch_size):
            seq = seq.to(device)
            loss = _contrastive_loss(model(_augment(seq)), model(_augment(seq)))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def _train_cost(x: np.ndarray, idx: np.ndarray, cfg: RepConfig, device: torch.device) -> tuple[ConvEncoder, ConvEncoder]:
    trend_model = ConvEncoder(x.shape[1], cfg.hidden, cfg.emb_dim).to(device)
    resid_model = ConvEncoder(x.shape[1], cfg.hidden, cfg.emb_dim).to(device)
    opt = torch.optim.AdamW([*trend_model.parameters(), *resid_model.parameters()], lr=cfg.lr, weight_decay=1e-4)
    ds = SequenceDataset(x, idx, cfg.seq_len)
    kernel = torch.ones(1, 1, 9, device=device) / 9.0
    for _ in range(cfg.epochs):
        for (seq,) in _loader(ds, cfg.batch_size):
            seq = seq.to(device)
            flat = seq.transpose(1, 2).reshape(-1, 1, seq.shape[1])
            trend = F.conv1d(F.pad(flat, (4, 4), mode="replicate"), kernel).reshape(seq.shape[0], seq.shape[2], seq.shape[1]).transpose(1, 2)
            resid = seq - trend
            loss = _contrastive_loss(trend_model(_augment(trend)), trend_model(_augment(trend)))
            loss = loss + _contrastive_loss(resid_model(_augment(resid)), resid_model(_augment(resid)))
            loss = loss + 0.05 * torch.mean(torch.abs((trend_model(trend) * resid_model(resid)).sum(dim=1)))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return trend_model, resid_model


def _train_mamba(x: np.ndarray, idx: np.ndarray, labels: np.ndarray, cfg: RepConfig, device: torch.device) -> MambaRisk:
    model = MambaRisk(x.shape[1], cfg.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    ds = SequenceDataset(x, idx, cfg.seq_len, labels=labels)
    pos = max(float(labels[idx].mean()), 1e-3)
    pos_weight = torch.tensor((1.0 - pos) / pos, device=device)
    for _ in range(cfg.epochs):
        for seq, y in _loader(ds, cfg.batch_size):
            seq = seq.to(device)
            y = y.to(device)
            loss = F.binary_cross_entropy_with_logits(model(seq), y, pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def _train_timegrad(x: np.ndarray, idx: np.ndarray, future: np.ndarray, cfg: RepConfig, device: torch.device) -> TimeGradNet:
    model = TimeGradNet(x.shape[1], cfg.hidden, cfg.horizon).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    betas = torch.linspace(1e-4, 0.03, cfg.diffusion_steps, device=device)
    alphas = torch.cumprod(1.0 - betas, dim=0)
    ds = SequenceDataset(x, idx, cfg.seq_len, future=future)
    for _ in range(cfg.epochs):
        for seq, fut in _loader(ds, cfg.batch_size):
            seq = seq.to(device)
            fut = fut.to(device)
            t = torch.randint(0, cfg.diffusion_steps, (seq.shape[0],), device=device)
            noise = torch.randn_like(fut)
            a = alphas[t].sqrt().unsqueeze(1)
            b = (1.0 - alphas[t]).sqrt().unsqueeze(1)
            pred = model(seq, a * fut + b * noise, t)
            loss = F.mse_loss(pred, noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def _train_timellm(x: np.ndarray, idx: np.ndarray, labels: np.ndarray, cfg: RepConfig, device: torch.device, model_name: str) -> TimeLLMReprogrammer:
    model = TimeLLMReprogrammer(x.shape[1], cfg.hidden, model_name).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=1e-4)
    ds = SequenceDataset(x, idx, cfg.seq_len, labels=labels)
    pos = max(float(labels[idx].mean()), 1e-3)
    pos_weight = torch.tensor((1.0 - pos) / pos, device=device)
    for _ in range(max(1, cfg.epochs)):
        for seq, y in _loader(ds, cfg.batch_size):
            seq = seq.to(device)
            y = y.to(device)
            loss = F.binary_cross_entropy_with_logits(model(seq), y, pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def _embed(model: nn.Module, x: np.ndarray, idx: np.ndarray, cfg: RepConfig, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    for (seq,) in _loader(SequenceDataset(x, idx, cfg.seq_len), cfg.batch_size, shuffle=False):
        out.append(model(seq.to(device)).detach().cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, cfg.emb_dim), dtype=np.float32)


@torch.no_grad()
def _predict_sigmoid(model: nn.Module, x: np.ndarray, idx: np.ndarray, cfg: RepConfig, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    for (seq,) in _loader(SequenceDataset(x, idx, cfg.seq_len), cfg.batch_size, shuffle=False):
        out.append(torch.sigmoid(model(seq.to(device))).detach().cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros(0, dtype=np.float32)


@torch.no_grad()
def _timegrad_probs(model: TimeGradNet, x: np.ndarray, idx: np.ndarray, cfg: RepConfig, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    betas = torch.linspace(1e-4, 0.03, cfg.diffusion_steps, device=device)
    alphas = torch.cumprod(1.0 - betas, dim=0)
    long_probs: list[np.ndarray] = []
    short_probs: list[np.ndarray] = []
    uncerts: list[np.ndarray] = []
    for (seq,) in _loader(SequenceDataset(x, idx, cfg.seq_len), max(16, cfg.batch_size // 2), shuffle=False):
        seq = seq.to(device)
        paths = []
        for _ in range(cfg.mc_paths):
            y = torch.randn(seq.shape[0], cfg.horizon, device=device) * 0.003
            for t_int in reversed(range(cfg.diffusion_steps)):
                t = torch.full((seq.shape[0],), t_int, dtype=torch.long, device=device)
                eps = model(seq, y, t)
                a = alphas[t_int].sqrt()
                b = (1.0 - alphas[t_int]).sqrt()
                y = (y - b * eps) / torch.clamp(a, min=1e-4)
                if t_int > 0:
                    y = 0.98 * y + betas[t_int].sqrt() * torch.randn_like(y)
            paths.append(y.detach().cpu().numpy())
        arr = np.stack(paths, axis=1)
        long_win = (arr.max(axis=2) > 0.004) & (arr.min(axis=2) > -0.003)
        short_win = ((-arr).max(axis=2) > 0.004) & ((-arr).min(axis=2) > -0.003)
        long_probs.append(long_win.mean(axis=1))
        short_probs.append(short_win.mean(axis=1))
        uncerts.append(arr[:, :, -1].std(axis=1))
    return np.concatenate(long_probs), np.concatenate(short_probs), np.concatenate(uncerts)


def _ood_scores(emb_fit: np.ndarray, fit_profit: np.ndarray, emb_pred: np.ndarray) -> np.ndarray:
    good = emb_fit[fit_profit]
    if len(good) < 16:
        good = emb_fit
    k = int(max(1, min(8, len(good) // 32)))
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(good)
    d_fit = np.min(((emb_fit[:, None, :] - km.cluster_centers_[None, :, :]) ** 2).sum(axis=2), axis=1) ** 0.5
    d_pred = np.min(((emb_pred[:, None, :] - km.cluster_centers_[None, :, :]) ** 2).sum(axis=2), axis=1) ** 0.5
    med = float(np.median(d_fit))
    mad = float(np.median(np.abs(d_fit - med)))
    return (d_pred - med) / max(1.4826 * mad, 1e-9)


def _fill_output(out: pd.DataFrame, idx: np.ndarray, cols: dict[str, np.ndarray]) -> None:
    for col, vals in cols.items():
        out.loc[idx, col] = vals


def main() -> None:
    ap = argparse.ArgumentParser(description="Train real OOF representation models for Alpha6 label presets.")
    ap.add_argument("--variant", default="current_tail111")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "tmp/causal_regen_20260516/alpha6_representation_label_features_20260524")
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--emb-dim", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=192)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mc-paths", type=int, default=24)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--models", default="ts2vec,cost,mamba,timegrad,timellm")
    ap.add_argument("--timellm-model", default="sshleifer/tiny-gpt2")
    ap.add_argument("--skip-timellm-on-fail", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = RepConfig(
        seq_len=int(args.seq_len),
        horizon=int(args.horizon),
        hidden=int(args.hidden),
        emb_dim=int(args.emb_dim),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        lr=float(args.lr),
        mc_paths=int(args.mc_paths),
    )
    device = torch.device(args.device if torch.cuda.is_available() or str(args.device) == "cpu" else "cpu")
    spec = _read_spec(DEFAULT_SPEC_DIR, args.variant)
    feat, _, _ = _read_feature_frame(DEFAULT_FEATURE_CSV, list(spec["features"]), CONTEXT_COLS)
    frame = feat.merge(_label_frame(DEFAULT_LABEL_DIR), on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
    seq_cols = _select_sequence_features(frame, list(spec["features"]))
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler(quantile_range=(10, 90)))])
    split = frame["dataset_split"].astype(str).str.lower().to_numpy()
    train_pos = np.flatnonzero(split == "train")
    val_pos = np.flatnonzero(split != "train")
    pipe.fit(_numeric_matrix(frame.iloc[train_pos], seq_cols))
    x = pipe.transform(_numeric_matrix(frame, seq_cols)).astype(np.float32)
    close = pd.to_numeric(frame["close"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    atr = pd.to_numeric(frame.get("atr14_pct", 0.003), errors="coerce").fillna(0.003).to_numpy(dtype=np.float64)
    future = _future_returns(close, cfg.horizon)
    adverse = np.zeros(len(frame), dtype=np.float32)
    all_valid = _valid_indices(len(frame), cfg.seq_len, cfg.horizon, np.arange(len(frame), dtype=np.int64))
    adverse[all_valid] = _adverse_label(future, atr, all_valid)
    out = frame[["timestamp", "dataset_split"]].copy()
    for c in [
        "rep_ts2vec_ood_z",
        "rep_ts2vec_inlier_score",
        "rep_cost_trend_strength",
        "rep_cost_residual_strength",
        "rep_cost_beta_neutral_score",
        "rep_mamba_toxic_prob",
        "rep_timegrad_long_win_prob",
        "rep_timegrad_short_win_prob",
        "rep_timegrad_uncertainty",
        "rep_timellm_uncertainty",
    ]:
        out[c] = np.nan
    selected = {m.strip().lower() for m in str(args.models).split(",") if m.strip()}
    meta: dict[str, Any] = {"config": asdict(cfg), "device": str(device), "sequence_features": seq_cols, "folds": []}
    fold_parts = np.array_split(train_pos, int(args.folds))
    final_pred_pos = _valid_indices(len(frame), cfg.seq_len, cfg.horizon, val_pos)
    for fold_id, fold_pos_raw in enumerate(fold_parts, start=1):
        fold_pos = _valid_indices(len(frame), cfg.seq_len, cfg.horizon, fold_pos_raw)
        lo, hi = int(fold_pos_raw.min()), int(fold_pos_raw.max())
        purge_lo = max(0, lo - cfg.seq_len - cfg.horizon)
        purge_hi = min(len(frame) - 1, hi + cfg.horizon)
        fit_raw = train_pos[(train_pos < purge_lo) | (train_pos > purge_hi)]
        fit_pos = _valid_indices(len(frame), cfg.seq_len, cfg.horizon, fit_raw)
        fold_meta = {"fold": fold_id, "fit_rows": int(len(fit_pos)), "pred_rows": int(len(fold_pos))}
        print(f"[alpha6-rep] fold={fold_id}/{args.folds} fit={len(fit_pos)} pred={len(fold_pos)}", flush=True)
        if "ts2vec" in selected:
            ts = _train_ts2vec(x, fit_pos, cfg, device)
            emb_fit = _embed(ts, x, fit_pos, cfg, device)
            emb_pred = _embed(ts, x, fold_pos, cfg, device)
            z = _ood_scores(emb_fit, _profit_mask(future, fit_pos), emb_pred)
            _fill_output(out, fold_pos, {"rep_ts2vec_ood_z": z, "rep_ts2vec_inlier_score": 1.0 / (1.0 + np.maximum(z, 0.0))})
            joblib.dump(ts.cpu(), args.out_dir / f"ts2vec_fold{fold_id}.joblib")
        if "cost" in selected:
            trend, resid = _train_cost(x, fit_pos, cfg, device)
            te = _embed(trend, x, fold_pos, cfg, device)
            re = _embed(resid, x, fold_pos, cfg, device)
            trend_s = np.linalg.norm(te, axis=1)
            resid_s = np.linalg.norm(re, axis=1)
            _fill_output(
                out,
                fold_pos,
                {
                    "rep_cost_trend_strength": trend_s,
                    "rep_cost_residual_strength": resid_s,
                    "rep_cost_beta_neutral_score": resid_s / np.maximum(trend_s + resid_s, 1e-9),
                },
            )
            joblib.dump({"trend": trend.cpu(), "resid": resid.cpu()}, args.out_dir / f"cost_fold{fold_id}.joblib")
        if "mamba" in selected:
            mb = _train_mamba(x, fit_pos, adverse, cfg, device)
            _fill_output(out, fold_pos, {"rep_mamba_toxic_prob": _predict_sigmoid(mb, x, fold_pos, cfg, device)})
            joblib.dump(mb.cpu(), args.out_dir / f"mamba_fold{fold_id}.joblib")
        if "timegrad" in selected:
            tg = _train_timegrad(x, fit_pos, future.astype(np.float32), cfg, device)
            lp, sp, un = _timegrad_probs(tg, x, fold_pos, cfg, device)
            _fill_output(out, fold_pos, {"rep_timegrad_long_win_prob": lp, "rep_timegrad_short_win_prob": sp, "rep_timegrad_uncertainty": un})
            joblib.dump(tg.cpu(), args.out_dir / f"timegrad_fold{fold_id}.joblib")
        if "timellm" in selected:
            try:
                tl = _train_timellm(x, fit_pos, adverse, cfg, device, str(args.timellm_model))
                _fill_output(out, fold_pos, {"rep_timellm_uncertainty": _predict_sigmoid(tl, x, fold_pos, cfg, device)})
                joblib.dump(tl.cpu(), args.out_dir / f"timellm_fold{fold_id}.joblib")
            except Exception as exc:
                if not args.skip_timellm_on_fail:
                    raise
                fold_meta["timellm_error"] = repr(exc)
        meta["folds"].append(fold_meta)
    train_valid = _valid_indices(len(frame), cfg.seq_len, cfg.horizon, train_pos)
    if len(final_pred_pos):
        print(f"[alpha6-rep] final train={len(train_valid)} val={len(final_pred_pos)}", flush=True)
        if "ts2vec" in selected:
            ts = _train_ts2vec(x, train_valid, cfg, device)
            z = _ood_scores(_embed(ts, x, train_valid, cfg, device), _profit_mask(future, train_valid), _embed(ts, x, final_pred_pos, cfg, device))
            _fill_output(out, final_pred_pos, {"rep_ts2vec_ood_z": z, "rep_ts2vec_inlier_score": 1.0 / (1.0 + np.maximum(z, 0.0))})
            joblib.dump(ts.cpu(), args.out_dir / "ts2vec_final.joblib")
        if "cost" in selected:
            trend, resid = _train_cost(x, train_valid, cfg, device)
            te = _embed(trend, x, final_pred_pos, cfg, device)
            re = _embed(resid, x, final_pred_pos, cfg, device)
            trend_s = np.linalg.norm(te, axis=1)
            resid_s = np.linalg.norm(re, axis=1)
            _fill_output(out, final_pred_pos, {"rep_cost_trend_strength": trend_s, "rep_cost_residual_strength": resid_s, "rep_cost_beta_neutral_score": resid_s / np.maximum(trend_s + resid_s, 1e-9)})
            joblib.dump({"trend": trend.cpu(), "resid": resid.cpu()}, args.out_dir / "cost_final.joblib")
        if "mamba" in selected:
            mb = _train_mamba(x, train_valid, adverse, cfg, device)
            _fill_output(out, final_pred_pos, {"rep_mamba_toxic_prob": _predict_sigmoid(mb, x, final_pred_pos, cfg, device)})
            joblib.dump(mb.cpu(), args.out_dir / "mamba_final.joblib")
        if "timegrad" in selected:
            tg = _train_timegrad(x, train_valid, future.astype(np.float32), cfg, device)
            lp, sp, un = _timegrad_probs(tg, x, final_pred_pos, cfg, device)
            _fill_output(out, final_pred_pos, {"rep_timegrad_long_win_prob": lp, "rep_timegrad_short_win_prob": sp, "rep_timegrad_uncertainty": un})
            joblib.dump(tg.cpu(), args.out_dir / "timegrad_final.joblib")
        if "timellm" in selected:
            try:
                tl = _train_timellm(x, train_valid, adverse, cfg, device, str(args.timellm_model))
                _fill_output(out, final_pred_pos, {"rep_timellm_uncertainty": _predict_sigmoid(tl, x, final_pred_pos, cfg, device)})
                joblib.dump(tl.cpu(), args.out_dir / "timellm_final.joblib")
            except Exception as exc:
                if not args.skip_timellm_on_fail:
                    raise
                meta["timellm_final_error"] = repr(exc)
    out_path = args.out_dir / "alpha6_representation_oof_features.parquet"
    out.to_parquet(out_path, index=False)
    (args.out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str))
    print(json.dumps({"out": str(out_path), "meta": str(args.out_dir / "meta.json")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
