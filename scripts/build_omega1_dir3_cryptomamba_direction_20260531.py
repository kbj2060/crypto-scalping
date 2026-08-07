#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from mamba_ssm import Mamba
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "omega1_dir3_cryptomamba_direction_20260531"
DEFAULT_SPLIT_DIR = ROOT / "data/splits/year_oos"
DEFAULT_LABEL_DIR = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_20260531"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/omega1_dir3_cryptomamba_20260531"
DEFAULT_REPORT_DIR = ROOT / "tmp/causal_regen_20260516/omega1_dir3_cryptomamba_20260531"

SEQ_LEN = 60
CLASS_NAMES = ("fl", "up", "dn")
ROLLING_BASE_COLS = (
    "volume",
    "quote_volume",
    "taker_buy_base",
    "taker_buy_quote",
    "sum_open_interest",
    "sum_open_interest_value",
    "last_funding_rate",
)
PRIORITY_COLS = [
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
    "amihud_illiquidity_z",
    "btc_corr_60",
    "eth_btc_ratio_change",
    "chop_index",
    "cvp_poc_dist",
    "cvp_cluster_position",
    "cvp_volume_imbalance",
    "cvp_regime",
    "breakout_strength",
    "funding_roc_12",
    "funding_roc_48",
    "funding_roc_288",
    "funding_z_score",
    "long_squeeze_risk",
    "short_squeeze_risk",
    "funding_price_divergence",
    "regime_trending",
    "regime_persistence",
    "ofi_acceleration",
    "kalman_velocity",
    "ofti",
    "kel",
    "funding_abs",
    "funding_pressure",
    "cvd_12",
    "cvd_48",
    "cvd_slope_12",
    "cvd_slope_48",
    "btc_ret_1",
    "btc_ret_3",
    "btc_ret_6",
    "eth_btc_ret_spread_12",
    "compression_score",
    "range_contraction_breakout_dir",
    "vwap_dist_24",
    "funding_oi_divergence",
    "oi_up_price_down",
    "oi_up_price_up",
    "upper_wick_z",
    "lower_wick_z",
    "sweep_prev_high_reclaim",
    "sweep_prev_low_reclaim",
    "failed_breakout_up",
    "failed_breakout_down",
    "garch_vol_z",
    "jump_z",
    "evt_excess_z",
    "sig_volume_confirm",
    "sig_liquidity_trap",
    "sig_trend_health",
    "liquidity_vacuum",
    "crowding_pressure",
    "execution_quality",
]
FORBIDDEN_PREFIXES = (
    "teacher_",
    "a5dir_",
    "clean_regime_",
    "clean_regime4_",
    "regime4_pred_",
    "regime3_pred_",
    "regime3_cmamba_",
    "dir3_",
)
FORBIDDEN_TOKENS = ("label", "target", "future", "realized", "pnl", "cash_after", "action_score", "zigzag", "wave3")
NON_FEATURES = {"timestamp", "open", "high", "low", "close"}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _split_file(split_dir: Path, year: int) -> Path:
    return split_dir / ("training_features_2026_rebuilt.csv" if int(year) == 2026 else f"training_features_{int(year)}.csv")


def _read(split_dir: Path, year: int) -> pd.DataFrame:
    path = _split_file(split_dir, year)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    return frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _exact_join(left: pd.DataFrame, right: pd.DataFrame, cols: list[str], source: str) -> pd.DataFrame:
    before = len(left)
    out = left.merge(right[["timestamp", *cols]], on="timestamp", how="left", validate="one_to_one")
    if len(out) != before:
        raise RuntimeError(f"{source} changed row count: {before} -> {len(out)}")
    missing = {col: int(out[col].isna().sum()) for col in cols if int(out[col].isna().sum())}
    if missing:
        raise RuntimeError(f"{source} exact join missing values: {missing}")
    return out


def _add_label(frame: pd.DataFrame, label_dir: Path, year: int) -> pd.DataFrame:
    labels = pd.read_csv(label_dir / f"zigzag_action_labels_{int(year)}.csv", parse_dates=["timestamp"])
    return _exact_join(frame, labels, ["zigzag_action"], f"ZigZag labels {year}")


def _rolling_rank(series: pd.Series, window: int) -> pd.Series:
    def rank_last(values: np.ndarray) -> float:
        valid = values[np.isfinite(values)]
        if len(valid) <= 1:
            return 0.5
        return float((valid <= valid[-1]).mean())

    return series.rolling(window, min_periods=max(20, window // 10)).apply(rank_last, raw=True)


def _add_rolling_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "taker_buy_base" in out.columns and "volume" in out.columns and "taker_sell_base" not in out.columns:
        out["taker_sell_base"] = pd.to_numeric(out["volume"], errors="coerce") - pd.to_numeric(out["taker_buy_base"], errors="coerce")
    for col in ROLLING_BASE_COLS:
        if col not in out.columns:
            continue
        s = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        signed_log = np.log1p(s.clip(lower=0.0)) if "volume" in col or "quote" in col or "taker" in col or "interest" in col else np.sign(s) * np.log1p(s.abs())
        med = signed_log.rolling(288, min_periods=48).median()
        q25 = signed_log.rolling(288, min_periods=48).quantile(0.25)
        q75 = signed_log.rolling(288, min_periods=48).quantile(0.75)
        iqr = (q75 - q25).replace(0.0, np.nan)
        out[f"{col}_roll_log_iqr_288"] = ((signed_log - med) / iqr).clip(-8.0, 8.0)
        out[f"{col}_roll_pct_288"] = _rolling_rank(signed_log, 288)
        out[f"{col}_roll_delta_log_12"] = (signed_log - signed_log.shift(12)).clip(-8.0, 8.0)
    return out


def _is_forbidden(col: str) -> bool:
    lower = col.lower()
    return col in NON_FEATURES or col.startswith(FORBIDDEN_PREFIXES) or any(token in lower for token in FORBIDDEN_TOKENS)


def _feature_cols(frames: list[pd.DataFrame], max_features: int) -> list[str]:
    common = set(frames[0].columns)
    for frame in frames[1:]:
        common &= set(frame.columns)
    requested: list[str] = []
    requested.extend(PRIORITY_COLS)
    for col in ROLLING_BASE_COLS:
        requested.extend([f"{col}_roll_log_iqr_288", f"{col}_roll_pct_288", f"{col}_roll_delta_log_12"])
    requested.extend(sorted(common))
    cols: list[str] = []
    for col in requested:
        if col not in common or col in cols or _is_forbidden(col):
            continue
        if pd.to_numeric(frames[0][col], errors="coerce").notna().any():
            cols.append(col)
        if len(cols) >= int(max_features):
            break
    if not cols:
        raise RuntimeError("no usable CryptoMamba direction features selected")
    bad = [c for c in cols if _is_forbidden(c)]
    if bad:
        raise RuntimeError(f"forbidden features selected: {bad[:20]}")
    return cols


def _prepare(train: pd.DataFrame, frames: list[pd.DataFrame], cols: list[str], fit_idx: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], StandardScaler, pd.Series]:
    raw_train = train[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    fit = raw_train.iloc[np.asarray(fit_idx, dtype=np.int64)]
    med = fit.median(axis=0).fillna(0.0)
    train_filled = raw_train.fillna(med).fillna(0.0)
    scaler = StandardScaler()
    scaler.fit(train_filled.iloc[np.asarray(fit_idx, dtype=np.int64)])
    x_train = scaler.transform(train_filled).astype(np.float32)
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
    def __init__(self, d_model: int, n_cmblocks: int, seq_len_in: int, seq_len_out: int, d_state: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([CMBlock(d_model, d_state, d_conv=4, expand=2) for _ in range(int(n_cmblocks))])
        self.seq_proj = nn.Linear(int(seq_len_in), int(seq_len_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.seq_proj(x.permute(0, 2, 1)).permute(0, 2, 1)


class CryptoMambaDirection(nn.Module):
    def __init__(self, n_features: int, seq_len: int, d_model: int, n_cblocks: int, n_cmblocks: int, d_state: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Linear(int(n_features), int(d_model))
        seq_lens = [int(seq_len)]
        for _ in range(int(n_cblocks)):
            seq_lens.append(max(seq_lens[-1] * 3 // 4, 8))
        self.cblocks = nn.ModuleList(
            [CBlock(d_model, n_cmblocks, seq_lens[i], seq_lens[i + 1], d_state=d_state) for i in range(int(n_cblocks))]
        )
        self.merge = nn.Sequential(nn.Dropout(float(dropout)), nn.Linear(int(d_model) * int(n_cblocks), int(d_model)), nn.GELU(), nn.LayerNorm(int(d_model)))
        self.head = nn.Sequential(nn.Dropout(float(dropout)), nn.Linear(int(d_model), 64), nn.GELU(), nn.Dropout(float(dropout)), nn.Linear(64, len(CLASS_NAMES)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(x)
        outs: list[torch.Tensor] = []
        for block in self.cblocks:
            z = block(z)
            outs.append(z[:, -1, :])
        return self.head(self.merge(torch.cat(outs, dim=-1)))


def _class_weights(y: np.ndarray, idx: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y[idx], minlength=len(CLASS_NAMES)).astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32, device=device)


@torch.no_grad()
def _predict(model: nn.Module, x: np.ndarray, idx: np.ndarray, seq_len: int, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    loader = DataLoader(SeqDataset(x, None, idx, seq_len), batch_size=int(batch_size), shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    outs: list[np.ndarray] = []
    for xb in loader:
        outs.append(torch.softmax(model(xb.to(device, non_blocking=True)), dim=-1).detach().cpu().numpy())
    return np.concatenate(outs, axis=0)


def _eval(y: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    pred = np.argmax(proba, axis=1).astype(np.int64)
    trade = pred != 0
    return {
        "rows": int(len(y)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "ovr_auc": float(roc_auc_score(y, proba, multi_class="ovr", labels=list(range(len(CLASS_NAMES))))),
        "label_counts": {str(i): int(v) for i, v in enumerate(np.bincount(y, minlength=len(CLASS_NAMES)))},
        "pred_counts": {str(i): int(v) for i, v in enumerate(np.bincount(pred, minlength=len(CLASS_NAMES)))},
        "proxy_trades": int(trade.sum()),
        "proxy_long_trades": int((pred == 1).sum()),
        "proxy_short_trades": int((pred == 2).sum()),
        "proxy_trade_rate": float(trade.mean()),
        "proxy_wr": float((pred[trade] == y[trade]).mean()) if trade.any() else None,
    }


def _fit(x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, args: argparse.Namespace) -> tuple[nn.Module, list[dict[str, Any]], torch.device]:
    if not torch.cuda.is_available() and not args.cpu:
        raise RuntimeError("CUDA is unavailable; pass --cpu explicitly for a slow CPU run")
    device = torch.device("cpu" if args.cpu else "cuda")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    model = CryptoMambaDirection(x.shape[1], args.seq_len, args.d_model, args.cblocks, args.cmblocks, args.d_state, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loss_fn = nn.CrossEntropyLoss(weight=_class_weights(y, train_idx, device), label_smoothing=0.02)
    loader = DataLoader(SeqDataset(x, y, train_idx, args.seq_len), batch_size=int(args.batch_size), shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    best_state: dict[str, torch.Tensor] | None = None
    best = -1.0
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
        score = float(ev["balanced_accuracy"])
        if score > best + 1e-4:
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
    for col in [
        "dir3_cryptomamba_h6_fl_prob",
        "dir3_cryptomamba_h6_up_prob",
        "dir3_cryptomamba_h6_dn_prob",
        "dir3_cryptomamba_h6_confidence",
        "dir3_cryptomamba_h6_side_edge",
        "dir3_cryptomamba_h6_trade_prob",
    ]:
        out[col] = np.nan
    out.loc[idx, "dir3_cryptomamba_h6_fl_prob"] = proba[:, 0]
    out.loc[idx, "dir3_cryptomamba_h6_up_prob"] = proba[:, 1]
    out.loc[idx, "dir3_cryptomamba_h6_dn_prob"] = proba[:, 2]
    out.loc[idx, "dir3_cryptomamba_h6_confidence"] = proba.max(axis=1)
    out.loc[idx, "dir3_cryptomamba_h6_side_edge"] = proba[:, 1] - proba[:, 2]
    out.loc[idx, "dir3_cryptomamba_h6_trade_prob"] = proba[:, 1] + proba[:, 2]
    return out


def _write_features(out_dir: Path, year: int, out: pd.DataFrame) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"training_features_{year}_omega1_dir3_cryptomamba_20260531.csv" if int(year) != 2026 else "training_features_2026_rebuilt_omega1_dir3_cryptomamba_20260531.csv"
    path = out_dir / name
    out.to_csv(path, index=False)
    return path


def main() -> int:
    p = argparse.ArgumentParser(description="Train CryptoMamba C-Block direction sidecar on ZigZag action labels.")
    p.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    p.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    p.add_argument("--seq-len", type=int, default=SEQ_LEN)
    p.add_argument("--val-start", default="2024-10-01")
    p.add_argument("--train-stride", type=int, default=2)
    p.add_argument("--max-features", type=int, default=128)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=768)
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
    args.report_dir.mkdir(parents=True, exist_ok=True)
    frames = {year: _add_label(_add_rolling_features(_read(args.split_dir, year)), args.label_dir, year) for year in [2024, 2025, 2026]}
    cols = _feature_cols([frames[2024], frames[2025], frames[2026]], args.max_features)
    y = frames[2024]["zigzag_action"].astype(int).to_numpy()
    ts = pd.to_datetime(frames[2024]["timestamp"])
    val_start = pd.Timestamp(args.val_start)
    val_start_idx = int(np.searchsorted(ts.to_numpy(dtype="datetime64[ns]"), np.datetime64(val_start)))
    train_idx = np.arange(args.seq_len - 1, val_start_idx, max(1, int(args.train_stride)), dtype=np.int64)
    val_idx = np.arange(max(args.seq_len - 1, val_start_idx), len(frames[2024]), dtype=np.int64)
    x_train, x_frames, scaler, medians = _prepare(frames[2024], [frames[2025], frames[2026]], cols, train_idx)
    model, history, device = _fit(x_train, y, train_idx, val_idx, args)
    torch.save(
        {
            "model_id": MODEL_ID,
            "classes": CLASS_NAMES,
            "feature_cols": cols,
            "feature_medians": medians.to_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "seq_len": int(args.seq_len),
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        },
        args.out_dir / "dir3_cryptomamba_direction.pt",
    )

    outputs: dict[str, Any] = {}
    for year, x in zip([2025, 2026], x_frames):
        frame = frames[year]
        idx = np.arange(args.seq_len - 1, len(frame), dtype=np.int64)
        proba = _predict(model, x, idx, args.seq_len, args.batch_size * 2, device)
        out = _output(frame, idx, proba)
        path = _write_features(args.out_dir, year, out)
        outputs[str(year)] = {
            "path": str(path),
            "metrics": _eval(frame["zigzag_action"].astype(int).to_numpy()[idx], proba),
        }
    val_proba = _predict(model, x_train, val_idx, args.seq_len, args.batch_size * 2, device)
    report = {
        "model_id": MODEL_ID,
        "architecture": {
            "type": "CryptoMamba C-Block Merge",
            "seq_len": int(args.seq_len),
            "d_model": int(args.d_model),
            "d_state": int(args.d_state),
            "cblocks": int(args.cblocks),
            "cmblocks": int(args.cmblocks),
        },
        "train_year": 2024,
        "selection_year": 2025,
        "oos_year": 2026,
        "label_source": "zigzag_action",
        "feature_count": int(len(cols)),
        "feature_cols": cols,
        "history": history,
        "internal_validation": _eval(y[val_idx], val_proba),
        "outputs": outputs,
        "contract": {
            "forbidden_inputs": ["teacher_*", "a5dir_*", "Regime4", "regime3_pred_*", "regime3_cmamba_*", "label/target/future/PnL/action_score", "same-level dir3 outputs"],
            "forbidden_feature_hits": [col for col in cols if _is_forbidden(col)],
            "notes": ["Architecture is ported from Regime3 CryptoMamba, but target is ZigZag action direction.", "Uses current/past rows only; scaler/median fitted on 2024 train split only."],
        },
    }
    report_path = args.report_dir / "dir3_cryptomamba_audit.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "outputs": outputs, "internal_validation": report["internal_validation"]}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
