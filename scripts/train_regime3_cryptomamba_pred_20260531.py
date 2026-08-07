#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from mamba_ssm import Mamba
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.retrain_clean_regime_hmm_20260517 import _json_default  # noqa: E402
from scripts.train_regime3_pred_tft_vsn_wide24_current_20260529 import (  # noqa: E402
    CLASSES3,
    DOCS_REGIME_PRED_FEATURES,
    RAW_PRIORITY,
    ROLLING_BASE_COLS,
    _add_rolling_stable_features,
    _read,
)


MODEL_ID = "regime3_cryptomamba_pred_h6_nocurrent_20260531"
CURRENT_PREFIX = "regime3_current_sensitive_wide24_"
CURRENT_SIDECAR_STEM = "regime3_current_sensitive_hmm_wide24"
OUT_PREFIX = "regime3_cmamba_h6_"
DEFAULT_CURRENT_DIR = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
DEFAULT_TRAIN_2024 = ROOT / "tmp/causal_regen_20260516/funding_clean_splits_20260528/training_features_2024.csv"
DEFAULT_TRANSFORMS = (
    ROOT / "tmp/causal_regen_20260516/funding_clean_splits_20260528/training_features_2024.csv",
    ROOT / "tmp/causal_regen_20260516/funding_clean_splits_20260528/training_features_2025.csv",
    ROOT / "tmp/causal_regen_20260516/funding_clean_splits_20260528/training_features_2026_rebuilt.csv",
)
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/regime3_cryptomamba_pred_h6_nocurrent_20260531"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/regime3_cryptomamba_pred_h6_nocurrent_20260531_report.json"

FORBIDDEN_PREFIXES = (
    "teacher_",
    "m7_",
    "a5dir_",
    "clean_regime_",
    "clean_regime4_",
    "clean_regime4_state24_",
    "regime4_pred_",
    "regime3_pred_",
    "regime3_cmamba_",
)
FORBIDDEN_TOKENS = ("label", "target", "future", "realized", "pnl", "cash_after", "action_score", "zigzag", "wave3")
NON_FEATURES = {"timestamp", "open", "high", "low", "close"}


def _current_path(current_dir: Path, source: Path) -> Path:
    return current_dir / f"{source.stem}_{CURRENT_SIDECAR_STEM}.csv"


def _merge_current(frame: pd.DataFrame, current_path: Path) -> pd.DataFrame:
    current = _read(current_path)
    required = [f"{CURRENT_PREFIX}{name}_prob" for name in CLASSES3]
    missing = [col for col in required if col not in current.columns]
    if missing:
        raise ValueError(f"{current_path} missing required current columns: {missing}")
    keep = ["timestamp"] + [col for col in current.columns if col.startswith(CURRENT_PREFIX)]
    out = frame.merge(current[keep], on="timestamp", how="left", validate="one_to_one")
    null_cols = [col for col in keep if col != "timestamp" and out[col].isna().any()]
    if null_cols:
        raise ValueError(f"current merge produced nulls: {null_cols[:10]}")
    return out


def _current_ids(frame: pd.DataFrame) -> np.ndarray:
    cols = [f"{CURRENT_PREFIX}{name}_prob" for name in CLASSES3]
    probs = frame[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    probs /= np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
    return np.argmax(probs, axis=1).astype(np.int64)


def _labels(frame: pd.DataFrame, horizon: int) -> tuple[np.ndarray, np.ndarray, int]:
    cur = _current_ids(frame)
    n = max(0, len(cur) - int(horizon))
    now = cur[:n]
    future = cur[int(horizon) : int(horizon) + n]
    transition = (now != future).astype(np.int64)
    return future, transition, n


def _rolling_rank(series: pd.Series, window: int) -> pd.Series:
    def rank_last(values: np.ndarray) -> float:
        valid = values[np.isfinite(values)]
        if len(valid) <= 1:
            return 0.5
        return float((valid <= valid[-1]).mean())

    return series.rolling(window, min_periods=max(20, window // 10)).apply(rank_last, raw=True)


def _add_volume_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "taker_buy_base" in out.columns and "volume" in out.columns and "taker_sell_base" not in out.columns:
        out["taker_sell_base"] = pd.to_numeric(out["volume"], errors="coerce") - pd.to_numeric(out["taker_buy_base"], errors="coerce")
    if "taker_buy_base" in out.columns and "taker_sell_base" in out.columns:
        tb = pd.to_numeric(out["taker_buy_base"], errors="coerce")
        ts = pd.to_numeric(out["taker_sell_base"], errors="coerce")
        out["volume_delta"] = (tb - ts) / (tb + ts + 1e-8)
    for col in ROLLING_BASE_COLS:
        if col not in out.columns:
            continue
        s = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        signed_log = np.log1p(s.clip(lower=0.0)) if "volume" in col or "quote" in col or "taker" in col else np.sign(s) * np.log1p(s.abs())
        med = signed_log.rolling(288, min_periods=48).median()
        q25 = signed_log.rolling(288, min_periods=48).quantile(0.25)
        q75 = signed_log.rolling(288, min_periods=48).quantile(0.75)
        iqr = (q75 - q25).replace(0.0, np.nan)
        out[f"{col}_roll_log_iqr_288"] = ((signed_log - med) / iqr).clip(-8.0, 8.0)
        out[f"{col}_roll_pct_288"] = _rolling_rank(signed_log, 288)
        out[f"{col}_roll_delta_log_12"] = (signed_log - signed_log.shift(12)).clip(-8.0, 8.0)
    return out


def _feature_cols(frames: list[pd.DataFrame], max_features: int, feature_pack: str) -> list[str]:
    common = set(frames[0].columns)
    for frame in frames[1:]:
        common &= set(frame.columns)
    if feature_pack == "docs_rolled":
        requested = list(DOCS_REGIME_PRED_FEATURES)
        requested += ["volume", "quote_volume", "taker_buy_base", "taker_sell_base", "volume_delta", "trade_intensity", "big_trade_ratio"]
        for col in ROLLING_BASE_COLS:
            requested += [f"{col}_roll_log_iqr_288", f"{col}_roll_pct_288", f"{col}_roll_delta_log_12"]
    elif feature_pack == "raw_priority":
        requested = list(RAW_PRIORITY)
    elif feature_pack == "all_sanitized":
        priority = list(DOCS_REGIME_PRED_FEATURES) + list(RAW_PRIORITY)
        for col in ROLLING_BASE_COLS:
            priority += [f"{col}_roll_log_iqr_288", f"{col}_roll_pct_288", f"{col}_roll_delta_log_12"]
        requested = priority + sorted(common)
    else:
        raise ValueError(f"unknown feature_pack={feature_pack}")
    cols: list[str] = []
    for col in requested:
        lower = col.lower()
        if col not in common or col in cols or col in NON_FEATURES:
            continue
        if col.startswith(CURRENT_PREFIX) or any(col.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
            continue
        if "regime" in lower:
            continue
        if any(token in lower for token in FORBIDDEN_TOKENS):
            continue
        if pd.to_numeric(frames[0][col], errors="coerce").notna().any():
            cols.append(col)
        if len(cols) >= int(max_features):
            break
    if not cols:
        raise ValueError("CryptoMamba feature selection produced no columns")
    return cols


def _prepare(train: pd.DataFrame, frames: list[pd.DataFrame], cols: list[str], fit_idx: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], StandardScaler, pd.Series]:
    raw_train = train[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    fit = raw_train.iloc[np.asarray(fit_idx, dtype=np.int64)]
    med = fit.median(axis=0).fillna(0.0)
    filled = raw_train.fillna(med).fillna(0.0)
    scaler = StandardScaler()
    scaler.fit(filled.iloc[np.asarray(fit_idx, dtype=np.int64)])
    x_train = scaler.transform(filled).astype(np.float32)
    x_frames: list[np.ndarray] = []
    for frame in frames:
        raw = frame[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        x_frames.append(scaler.transform(raw.fillna(med).fillna(0.0)).astype(np.float32))
    return np.nan_to_num(x_train), [np.nan_to_num(x) for x in x_frames], scaler, med


class SeqDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray | None, idx: np.ndarray, seq_len: int) -> None:
        self.x = x
        self.y = y
        self.idx = idx.astype(np.int64)
        self.seq_len = int(seq_len)

    def __len__(self) -> int:
        return int(len(self.idx))

    def __getitem__(self, item: int):
        end = int(self.idx[item])
        start = end - self.seq_len + 1
        seq = self.x[start : end + 1]
        if self.y is None:
            return torch.from_numpy(seq)
        return torch.from_numpy(seq), torch.tensor(int(self.y[end]), dtype=torch.long)


class CMBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mamba(self.norm(x))


class CBlock(nn.Module):
    def __init__(self, d_model: int, n_cmblocks: int, seq_len_in: int, seq_len_out: int, d_state: int, d_conv: int, expand: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([CMBlock(d_model, d_state, d_conv, expand) for _ in range(int(n_cmblocks))])
        self.seq_proj = nn.Linear(int(seq_len_in), int(seq_len_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.seq_proj(x.permute(0, 2, 1)).permute(0, 2, 1)


class CryptoMambaRegimePred(nn.Module):
    def __init__(self, n_features: int, seq_len: int, d_model: int, n_cblocks: int, n_cmblocks: int, d_state: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Linear(int(n_features), int(d_model))
        seq_lens = [int(seq_len)]
        for _ in range(int(n_cblocks)):
            seq_lens.append(max(seq_lens[-1] * 3 // 4, 8))
        self.cblocks = nn.ModuleList(
            [
                CBlock(d_model, n_cmblocks, seq_lens[i], seq_lens[i + 1], d_state=d_state, d_conv=4, expand=2)
                for i in range(int(n_cblocks))
            ]
        )
        self.merge = nn.Sequential(nn.Dropout(float(dropout)), nn.Linear(int(d_model) * int(n_cblocks), int(d_model)), nn.GELU(), nn.LayerNorm(int(d_model)))
        self.head = nn.Sequential(nn.Dropout(float(dropout)), nn.Linear(int(d_model), 64), nn.GELU(), nn.Dropout(float(dropout)), nn.Linear(64, len(CLASSES3)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(x)
        outs: list[torch.Tensor] = []
        for block in self.cblocks:
            z = block(z)
            outs.append(z[:, -1, :])
        return self.head(self.merge(torch.cat(outs, dim=-1)))


def _class_weights(y: np.ndarray, idx: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y[idx], minlength=len(CLASSES3)).astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32, device=device)


@torch.no_grad()
def _predict(model: nn.Module, x: np.ndarray, idx: np.ndarray, seq_len: int, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    loader = DataLoader(SeqDataset(x, None, idx, seq_len), batch_size=int(batch_size), shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    outs: list[np.ndarray] = []
    for xb in loader:
        logits = model(xb.to(device, non_blocking=True))
        outs.append(torch.softmax(logits, dim=-1).detach().cpu().numpy())
    return np.concatenate(outs, axis=0)


def _eval(y: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    pred = np.argmax(proba, axis=1).astype(np.int64)
    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "label_counts": {CLASSES3[i]: int(v) for i, v in enumerate(np.bincount(y, minlength=len(CLASSES3)))},
        "pred_counts": {CLASSES3[i]: int(v) for i, v in enumerate(np.bincount(pred, minlength=len(CLASSES3)))},
        "confusion_matrix": confusion_matrix(y, pred, labels=list(range(len(CLASSES3)))).tolist(),
        "log_loss": float(log_loss(y, proba, labels=list(range(len(CLASSES3))))),
    }
    try:
        out["ovr_auc"] = float(roc_auc_score(y, proba, multi_class="ovr", labels=list(range(len(CLASSES3)))))
    except ValueError:
        out["ovr_auc"] = None
    return out


def _fit(x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, args: argparse.Namespace) -> tuple[nn.Module, list[dict[str, Any]], torch.device]:
    if not torch.cuda.is_available() and not args.cpu:
        raise RuntimeError("CUDA is unavailable; pass --cpu explicitly for a slow CPU run")
    device = torch.device("cpu" if args.cpu else "cuda")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    model = CryptoMambaRegimePred(x.shape[1], args.seq_len, args.d_model, args.cblocks, args.cmblocks, args.d_state, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loss_fn = nn.CrossEntropyLoss(weight=_class_weights(y, train_idx, device), label_smoothing=0.02)
    loader = DataLoader(SeqDataset(x, y, train_idx, args.seq_len), batch_size=int(args.batch_size), shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    best_state: dict[str, torch.Tensor] | None = None
    best = float("inf")
    bad = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses: list[float] = []
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_proba = _predict(model, x, val_idx, args.seq_len, args.batch_size * 2, device)
        ev = _eval(y[val_idx], val_proba)
        row = {"epoch": int(epoch), "loss": float(np.mean(losses)), "val": ev}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False, default=_json_default), flush=True)
        score = float(ev["log_loss"])
        if score < best - 1e-4:
            best = score
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= int(args.patience):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, device


def _output(frame: pd.DataFrame, idx: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame({"timestamp": frame["timestamp"].reset_index(drop=True)})
    for name in CLASSES3:
        out[f"{OUT_PREFIX}future_{name}_prob"] = np.nan
    out[f"{OUT_PREFIX}future_pred_id"] = np.nan
    out[f"{OUT_PREFIX}future_pred_name"] = ""
    out[f"{OUT_PREFIX}confidence"] = np.nan
    out[f"{OUT_PREFIX}transition_prob"] = np.nan
    out[f"{OUT_PREFIX}stability_score"] = np.nan
    current = _current_ids(frame)
    pred = np.argmax(proba, axis=1).astype(np.int64)
    conf = np.max(proba, axis=1)
    transition_prob = 1.0 - proba[np.arange(len(idx)), current[idx]]
    for i, name in enumerate(CLASSES3):
        out.loc[idx, f"{OUT_PREFIX}future_{name}_prob"] = proba[:, i]
    out.loc[idx, f"{OUT_PREFIX}future_pred_id"] = pred
    out.loc[idx, f"{OUT_PREFIX}future_pred_name"] = [CLASSES3[i] for i in pred]
    out.loc[idx, f"{OUT_PREFIX}confidence"] = conf
    out.loc[idx, f"{OUT_PREFIX}transition_prob"] = transition_prob
    out.loc[idx, f"{OUT_PREFIX}stability_score"] = 1.0 - transition_prob
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Train CryptoMamba-style Regime3 h6 future/transition prediction sidecar.")
    p.add_argument("--train-2024", type=Path, default=DEFAULT_TRAIN_2024)
    p.add_argument("--transform", type=Path, action="append", default=None)
    p.add_argument("--current-dir", type=Path, default=DEFAULT_CURRENT_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--seq-len", type=int, default=60)
    p.add_argument("--val-start", default="2024-10-01")
    p.add_argument("--train-stride", type=int, default=2)
    p.add_argument("--max-features", type=int, default=96)
    p.add_argument("--feature-pack", choices=("docs_rolled", "raw_priority", "all_sanitized"), default="docs_rolled")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--d-state", type=int, default=32)
    p.add_argument("--cblocks", type=int, default=4)
    p.add_argument("--cmblocks", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=20260531)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    sources = list(args.transform or DEFAULT_TRANSFORMS)
    frames = [_merge_current(_add_volume_features(_add_rolling_stable_features(_read(path))), _current_path(args.current_dir, path)) for path in sources]
    train = _merge_current(_add_volume_features(_add_rolling_stable_features(_read(args.train_2024))), _current_path(args.current_dir, args.train_2024))
    cols = _feature_cols([train] + frames, args.max_features, args.feature_pack)
    future, transition, n = _labels(train, args.horizon)
    labeled = train.iloc[:n].copy()
    ts = pd.to_datetime(labeled["timestamp"])
    val_start = pd.Timestamp(args.val_start)
    val_start_idx = int(np.searchsorted(ts.to_numpy(dtype="datetime64[ns]"), np.datetime64(val_start)))
    train_end = max(args.seq_len - 1, val_start_idx - int(args.horizon))
    train_idx = np.arange(args.seq_len - 1, train_end, max(1, int(args.train_stride)), dtype=np.int64)
    val_idx = np.arange(max(args.seq_len - 1, val_start_idx), n, dtype=np.int64)
    x_train, x_frames, scaler, medians = _prepare(labeled, frames, cols, train_idx)

    model, history, device = _fit(x_train, future, train_idx, val_idx, args)
    model_path = args.out_dir / f"{MODEL_ID}_2024.pt"
    torch.save(
        {
            "model_id": MODEL_ID,
            "classes": CLASSES3,
            "horizon": int(args.horizon),
            "output_prefix": OUT_PREFIX,
            "current_prefix": CURRENT_PREFIX,
            "feature_cols": cols,
            "feature_medians": medians.to_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "seq_len": int(args.seq_len),
            "d_model": int(args.d_model),
            "d_state": int(args.d_state),
            "cblocks": int(args.cblocks),
            "cmblocks": int(args.cmblocks),
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        },
        model_path,
    )

    report: dict[str, Any] = {
        "model_id": MODEL_ID,
        "purpose": "Regime transition/future-context sidecar only; not an action owner.",
        "model_path": str(model_path),
        "architecture": {
            "type": "CryptoMamba C-Block Merge",
            "seq_len": int(args.seq_len),
            "d_model": int(args.d_model),
            "d_state": int(args.d_state),
            "cblocks": int(args.cblocks),
            "cmblocks": int(args.cmblocks),
        },
        "seed": int(args.seed),
        "feature_count": int(len(cols)),
        "feature_pack": str(args.feature_pack),
        "feature_cols": cols,
        "current_probability_features_used_as_inputs": False,
        "current_sidecar_used_for_target_and_transition_eval": True,
        "history": history,
        "validation": {},
        "outputs": {},
        "leakage_audit": {
            "uses_2026_for_selection": False,
            "current_probability_inputs": [c for c in cols if c.startswith(CURRENT_PREFIX)],
            "forbidden_feature_hits": [
                c
                for c in cols
                if any(c.startswith(prefix) for prefix in FORBIDDEN_PREFIXES) or any(token in c.lower() for token in FORBIDDEN_TOKENS)
            ],
        },
    }
    val_proba = _predict(model, x_train, val_idx, args.seq_len, args.batch_size * 2, device)
    report["validation"] = {
        "future_h6": _eval(future[val_idx], val_proba),
        "transition_rate": float(np.mean(transition[val_idx])),
    }

    for path, frame, x in zip(sources, frames, x_frames):
        future_i, transition_i, n_i = _labels(frame, args.horizon)
        idx = np.arange(args.seq_len - 1, len(frame), dtype=np.int64)
        proba = _predict(model, x, idx, args.seq_len, args.batch_size * 2, device)
        out = _output(frame, idx, proba)
        out_path = args.out_dir / f"{path.stem}_{MODEL_ID}.csv"
        out.to_csv(out_path, index=False)
        eval_idx = idx[idx < n_i]
        eval_proba = proba[: len(eval_idx)]
        current = _current_ids(frame)
        transition_prob = 1.0 - eval_proba[np.arange(len(eval_idx)), current[eval_idx]]
        report["outputs"][path.name] = {
            "source": str(path),
            "sidecar": str(out_path),
            "rows": int(len(frame)),
            "valid_eval_rows": int(len(eval_idx)),
            "range": [str(frame["timestamp"].iloc[0]), str(frame["timestamp"].iloc[-1])],
            "future_h6": _eval(future_i[eval_idx], eval_proba),
            "transition": {
                "transition_rate": float(np.mean(transition_i[eval_idx])),
                "transition_prob_auc": float(roc_auc_score(transition_i[eval_idx], transition_prob)) if len(np.unique(transition_i[eval_idx])) > 1 else None,
                "transition_prob_mean": float(np.mean(transition_prob)),
            },
        }
        print(f"[{MODEL_ID}] wrote {out_path}", flush=True)

    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report), "model": str(model_path), "validation": report["validation"], "oos": report["outputs"].get("training_features_2026_rebuilt.csv")}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
