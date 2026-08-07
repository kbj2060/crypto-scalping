#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chronos import Chronos2Pipeline  # noqa: E402
from tsfm.model.kairos import AutoModel as KairosAutoModel  # noqa: E402

from ensemble.fully_learned_governor_policy import prepare_features  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import (  # noqa: E402
    _audit_contract,
    _close,
    _days,
    _feature_cols,
    _fill_price,
    _read,
)
from scripts.train_eval_hf_v13_deep_best_utility_parent_v39_1 import (  # noqa: E402
    MAX_LABEL_HOLD,
    MARGIN110_COST1,
    TARGET_SCALE,
    V31_COST1,
    _json_default,
    _parent_cfg,
)


MODEL_ID = "hf_v13_multitrack_foundation_parent_v40_20260512"
DEFAULT_TRAIN = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_multitrack_foundation_parent_v40_20260512"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_multitrack_foundation_parent_v40_20260512_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_multitrack_foundation_parent_v40_20260512_audit.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/hf_v13_multitrack_foundation_parent_v40_20260512_grid.csv"

CHRONOS_MODEL = "amazon/chronos-2"
KAIROS_MODEL = "mldi-lab/Kairos_23m"
MACRO_LEN = 1024
MICRO_LEN = 72

MACRO_COLS = ["open", "high", "low", "close", "volume"]
MICRO_PRIORITY = [
    "net_taker_ratio",
    "ofi_acceleration",
    "taker_acceleration",
    "trade_intensity",
    "amihud_illiquidity_z",
    "smart_money_flow",
    "big_trade_ratio",
    "whale_retail_ratio",
    "liquidity_vacuum",
    "execution_quality",
    "squeeze_power",
    "breakout_strength",
]


@dataclass(frozen=True)
class RuntimeConfig:
    name: str
    edge_th: float
    margin_th: float
    adverse_weight: float
    base_notional: float
    max_notional: float
    risk_budget: float
    base_tp: float
    base_sl: float
    max_hold: int
    cooldown: int
    tp_mult: float = 1.05
    sl_mult: float = 1.05
    trail: bool = True
    vol_throttle: bool = True


class QuantFastKANLayer(nn.Module):
    """
    RBF-based fast KAN layer for tabular state crossing.
    """

    def __init__(self, input_dim: int, output_dim: int, grid_size: int = 8, base_activation: type[nn.Module] = nn.SiLU) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.base_linear = nn.Linear(input_dim, output_dim)
        self.base_activation = base_activation()
        grid = torch.linspace(-3.0, 3.0, grid_size)
        self.register_buffer("grid", grid)
        self.spline_weight = nn.Parameter(torch.randn(output_dim, input_dim, grid_size) / (grid_size**0.5))
        self.denominator = (6.0 / max(grid_size - 1, 1)) ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(self.base_activation(x))
        diff = x.unsqueeze(-1) - self.grid
        rbf = torch.exp(-((diff**2) / self.denominator))
        spline_out = torch.einsum("bik,oik->bo", rbf, self.spline_weight)
        return base_out + spline_out


class KANStateEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 32) -> None:
        super().__init__()
        self.layer1 = QuantFastKANLayer(in_dim, 64, grid_size=8)
        self.layer2 = QuantFastKANLayer(64, out_dim, grid_size=8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(F.gelu(self.layer1(x)))


class MultiTrackParent(nn.Module):
    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.macro_proj = nn.Sequential(nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.10), nn.Linear(256, 128))
        self.micro_proj = nn.Sequential(nn.Linear(384, 96), nn.GELU(), nn.Dropout(0.10), nn.Linear(96, 32))
        self.state_enc = KANStateEncoder(state_dim, out_dim=32)
        self.head = nn.Sequential(
            nn.Linear(128 + 32 + 32, 192),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(192, 96),
            nn.GELU(),
            nn.Linear(96, 4),
        )

    def forward(self, macro: torch.Tensor, micro: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.macro_proj(macro), self.micro_proj(micro), self.state_enc(state)], dim=-1)
        return self.head(x)


def _runtime_grid() -> list[RuntimeConfig]:
    rows: list[RuntimeConfig] = []
    for edge in (0.006, 0.008, 0.010, 0.012, 0.016):
        rows.append(RuntimeConfig(f"v40_balanced_e{edge:.3f}", edge, 0.0025, 0.30, 0.90, 1.50, 0.016, 0.040, 0.018, 48, 12))
        rows.append(RuntimeConfig(f"v40_precision_e{edge:.3f}", edge, 0.0040, 0.55, 0.70, 1.20, 0.012, 0.036, 0.015, 48, 12))
        rows.append(RuntimeConfig(f"v40_convex_e{edge:.3f}", edge, 0.0030, 0.20, 1.00, 1.80, 0.018, 0.045, 0.020, 72, 12))
    return rows


def _state_cols(feature_cols: list[str], prepared: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in feature_cols:
        if c.startswith("clean_regime_2024_unsup_v4_") or c.startswith("m7_") or c.startswith("ai_"):
            cols.append(c)
    for c in (
        "pred_patchtst",
        "conf_patchtst",
        "patchtst_median",
        "patchtst_regime_sim",
        "tide_vol_raw",
        "tide_vol_zscore",
        "timesnet_cycle_sin",
        "timesnet_cycle_cos",
        "timesnet_cycle_delta",
        "dlinear_smf_ema",
        "dlinear_smf_slope",
        "mtf_trend_1h",
        "mtf_trend_4h",
        "rsi",
        "smart_money_flow",
        "net_taker_ratio",
    ):
        if c in prepared.columns and c not in cols:
            cols.append(c)
    return cols


def _safe_num_array(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df.loc[:, cols].astype(float).replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)


def _window_2d(arr: np.ndarray, idx: int, length: int) -> np.ndarray:
    start = max(0, idx - length + 1)
    cut = arr[start : idx + 1]
    out = np.full((length, arr.shape[1]), np.nan, dtype=np.float32)
    out[-len(cut) :] = cut
    return out


def _embedding_cache_path(
    cache_dir: Path,
    *,
    prefix: str,
    model_name: str,
    frame: pd.DataFrame,
    indices: np.ndarray,
    cols: list[str],
    window_len: int,
    extra_tag: str = "",
) -> Path:
    ts = pd.to_datetime(frame["timestamp"], errors="coerce") if "timestamp" in frame.columns else pd.Series(dtype="datetime64[ns]")
    idx = np.asarray(indices, dtype=np.int64)
    payload = {
        "prefix": prefix,
        "model": model_name,
        "rows": int(len(frame)),
        "ts_min": str(ts.iloc[0]) if len(ts) else "",
        "ts_max": str(ts.iloc[-1]) if len(ts) else "",
        "cols": list(cols),
        "window_len": int(window_len),
        "index_count": int(len(idx)),
        "index_min": int(idx.min()) if len(idx) else -1,
        "index_max": int(idx.max()) if len(idx) else -1,
        "extra_tag": extra_tag,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{prefix}_{digest}.npy"


def _build_best_targets(frame: pd.DataFrame, indices: np.ndarray, *, fee: float, slip: float) -> np.ndarray:
    open_px = pd.to_numeric(frame["open"], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().to_numpy(dtype=np.float64)
    horizons = np.arange(1, MAX_LABEL_HOLD + 1, dtype=np.int64)
    targets = np.zeros((len(indices), 4), dtype=np.float32)
    cost = 2.0 * float(fee + slip)
    for r, idx in enumerate(indices):
        entry_i = min(int(idx) + 1, len(open_px) - 1)
        exit_i = np.minimum(entry_i + horizons, len(open_px) - 1)
        long_entry = open_px[entry_i] * (1.0 + slip)
        short_entry = open_px[entry_i] * (1.0 - slip)
        long_exit = open_px[exit_i] * (1.0 - slip)
        short_exit = open_px[exit_i] * (1.0 + slip)
        long_path = long_exit / max(long_entry, 1e-12) - 1.0 - cost
        short_path = (short_entry - short_exit) / max(short_entry, 1e-12) - cost
        targets[r, 0] = float(np.max(long_path))
        targets[r, 1] = float(np.max(short_path))
        targets[r, 2] = max(0.0, -float(np.min(long_path)))
        targets[r, 3] = max(0.0, -float(np.min(short_path)))
    return targets


def _vol_anchor(row: pd.Series) -> float:
    vals = []
    for c, scale in (
        ("bb_width", 0.15),
        ("garman_klass_vol", 2.5),
        ("rogers_satchell_vol", 2.5),
        ("parkinson_vol", 2.5),
    ):
        try:
            x = abs(float(row.get(c, 0.0)))
        except Exception:
            x = 0.0
        vals.append(x * scale)
    try:
        volz = abs(float(row.get("volatility_z", 0.0)))
    except Exception:
        volz = 0.0
    try:
        rv = abs(float(row.get("realized_vol_ratio", 1.0)))
    except Exception:
        rv = 1.0
    base = max(0.0015, *vals)
    return float(np.clip(base * (1.0 + 0.08 * min(volz, 3.0) + 0.05 * max(rv - 1.0, 0.0)), 0.0015, 0.030))


def _scores(pred_row: np.ndarray, cfg: RuntimeConfig, *, fee: float, slip: float) -> tuple[float, float]:
    long_best, short_best, long_adv, short_adv = [float(x) for x in pred_row]
    cost_buffer = 2.0 * (fee + slip)
    return (
        long_best - cfg.adverse_weight * max(long_adv, 0.0) - cost_buffer,
        short_best - cfg.adverse_weight * max(short_adv, 0.0) - cost_buffer,
    )


def _entry_params(pred_row: np.ndarray, cfg: RuntimeConfig, side: int, row: pd.Series) -> tuple[float, float, float]:
    best = max(float(pred_row[0 if side > 0 else 1]), 0.0)
    adverse = max(float(pred_row[2 if side > 0 else 3]), 0.003)
    risk_cap = cfg.risk_budget / max(adverse, 0.004)
    edge_scale = float(np.clip(0.75 + best / 0.035, 0.65, 1.45))
    notional = min(cfg.max_notional, risk_cap, cfg.base_notional * edge_scale)
    if cfg.vol_throttle:
        va = _vol_anchor(row)
        if va > 0.020:
            notional *= 0.50
        elif va > 0.014:
            notional *= 0.70
    tp = float(np.clip(max(cfg.base_tp, best * notional * cfg.tp_mult), cfg.base_tp * 0.75, 0.090))
    sl = float(np.clip(max(cfg.base_sl, adverse * notional * cfg.sl_mult), cfg.base_sl * 0.65, 0.045))
    return float(max(notional, 0.0)), tp, sl


def backtest(df: pd.DataFrame, pred: np.ndarray, cfg: RuntimeConfig, *, fee: float, slip: float, record: bool = False) -> dict[str, Any]:
    close = _close(df)
    cash = peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = entry_equity = 0.0
    entry_idx = 0
    notional = tp = sl = 0.0
    cooldown = 0
    trades = wins = long_entries = short_entries = 0
    notional_sum = 0.0
    exits: dict[str, int] = {}
    records: list[dict[str, Any]] = []
    open_record: dict[str, Any] | None = None
    mfe = mae = 0.0

    def mark(i: int) -> tuple[float, float]:
        if pos == 0:
            return cash, 0.0
        px = float(close[int(np.clip(i, 0, len(close) - 1))])
        raw = (px * (1.0 - slip) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip)) / max(entry_price, 1e-12)
        unreal = raw * notional
        return cash * (1.0 + unreal), unreal

    for i in range(0, len(df) - 2):
        eq, unreal = mark(i)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if pos != 0:
            mfe = max(mfe, unreal)
            mae = min(mae, unreal)
            hold = i - entry_idx
            eff_sl = sl
            if cfg.trail and mfe > 0.0:
                gap = max(_vol_anchor(df.iloc[entry_idx]) * notional * 0.75, 0.004)
                if hold >= 18:
                    gap = max(gap * 0.35, gap - 0.025 * (hold - 18) * gap)
                eff_sl = min(eff_sl, max(0.001, mfe - gap))
            reason = ""
            if tp > 0.0 and unreal >= tp:
                reason = "take_profit"
            elif eff_sl > 0.0 and unreal <= -abs(eff_sl):
                reason = "stop_loss"
            elif hold >= cfg.max_hold:
                reason = "max_hold"
            if reason:
                fill_i = min(i + 1, len(df) - 1)
                exit_px = _fill_price(df, fill_i, pos, slip, entry=False)
                raw = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw * notional)
                cash -= before * fee * notional
                trades += 1
                wins += int(cash > entry_equity)
                exits[reason] = exits.get(reason, 0) + 1
                if record and open_record is not None:
                    out = dict(open_record)
                    out.update({"exit_signal_timestamp": str(df["timestamp"].iloc[i]), "exit_fill_timestamp": str(df["timestamp"].iloc[fill_i]), "exit_reason": reason, "realized_net_pct": float((cash / max(entry_equity, 1e-12) - 1.0) * 100.0), "mfe_pct": float(mfe * 100.0), "mae_pct": float(mae * 100.0), "cash_after": float(cash)})
                    records.append(out)
                pos = 0
                cooldown = int(cfg.cooldown)
                open_record = None
                continue
        if pos != 0:
            continue
        if cooldown > 0:
            cooldown -= 1
            continue
        long_score, short_score = _scores(pred[i], cfg, fee=fee, slip=slip)
        if max(long_score, short_score) < cfg.edge_th or abs(long_score - short_score) < cfg.margin_th:
            continue
        side = 1 if long_score > short_score else -1
        n, tp_new, sl_new = _entry_params(pred[i], cfg, side, df.iloc[i])
        if n <= 0.05:
            continue
        fill_i = min(i + 1, len(df) - 1)
        pos = side
        entry_price = _fill_price(df, fill_i, pos, slip, entry=True)
        entry_equity = cash
        entry_idx = i
        notional, tp, sl = float(n), float(tp_new), float(sl_new)
        cash -= cash * fee * notional
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        notional_sum += notional
        mfe = mae = 0.0
        if record:
            open_record = {"entry_signal_timestamp": str(df["timestamp"].iloc[i]), "entry_fill_timestamp": str(df["timestamp"].iloc[fill_i]), "side": "LONG" if pos > 0 else "SHORT", "entry_price": float(entry_price), "notional_exposure": float(notional), "long_score": float(long_score), "short_score": float(short_score), "take_profit": float(tp), "stop_loss": float(sl), "fee_entry_pct": float(fee * notional * 100.0)}
    if pos != 0:
        fill_i = len(df) - 1
        exit_px = _fill_price(df, fill_i, pos, slip, entry=False)
        raw = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * notional)
        cash -= before * fee * notional
        trades += 1
        wins += int(cash > entry_equity)
        exits["forced_end"] = exits.get("forced_end", 0) + 1
    entries = max(long_entries + short_entries, 1)
    out: dict[str, Any] = {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / max(trades, 1)),
        "trades_per_day": float(trades / _days(df)),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "avg_notional": float(notional_sum / entries),
        "avg_leverage": 1.0,
        "exits": exits,
    }
    if record:
        out["trade_records"] = records
    return out


def _selection_score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1["trades"]) < 20:
        return -1e9 + float(c1["pnl"])
    return float(c1["pnl"] + 0.35 * c2["pnl"] + 0.20 * c3["pnl"] - 0.45 * abs(c1["mdd"]) + 0.08 * min(c1["trades"], 160))


def _fit_norm(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=0).astype(np.float32)
    std = (np.nanstd(x, axis=0) + 1e-6).astype(np.float32)
    return mean, std


def _apply_norm(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean[None, :]) / std[None, :]).astype(np.float32)


def _extract_macro_embeddings(
    pipe: Chronos2Pipeline,
    df: pd.DataFrame,
    indices: np.ndarray,
    *,
    cache_path: Path,
    batch_size: int,
) -> np.ndarray:
    if cache_path.exists():
        return np.load(cache_path)
    raw = _safe_num_array(df, [c for c in MACRO_COLS if c in df.columns])
    chunks: list[np.ndarray] = []
    for start in range(0, len(indices), batch_size):
        take = indices[start : start + batch_size]
        samples = [_window_2d(raw, int(idx), MACRO_LEN).T for idx in take]
        embeds, _ = pipe.embed(samples, batch_size=batch_size, context_length=MACRO_LEN)
        chunks.append(np.stack([e[:, -1, :].mean(dim=0).cpu().numpy().astype(np.float32) for e in embeds]))
    out = np.vstack(chunks)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, out)
    return out


def _extract_micro_embeddings(
    model: Any,
    df: pd.DataFrame,
    indices: np.ndarray,
    micro_cols: list[str],
    *,
    cache_path: Path,
    batch_size: int,
) -> np.ndarray:
    if cache_path.exists():
        return np.load(cache_path)
    raw = _safe_num_array(df, micro_cols)
    outputs: list[np.ndarray] = []
    for start in range(0, len(indices), batch_size):
        take = indices[start : start + batch_size]
        batch = []
        for idx in take:
            win = _window_2d(raw, int(idx), MICRO_LEN)
            batch.append(win.T)
        stacked = np.concatenate(batch, axis=0).astype(np.float32)
        hidden, *_ = model.encode(torch.from_numpy(stacked).to(next(model.parameters()).device))
        pooled = hidden[:, -1, :].detach().cpu().numpy().astype(np.float32)
        pooled = pooled.reshape(len(take), len(micro_cols), -1).mean(axis=1)
        outputs.append(pooled)
    out = np.vstack(outputs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, out)
    return out


def _fit_model(macro: np.ndarray, micro: np.ndarray, state: np.ndarray, y: np.ndarray, *, epochs: int, seed: int) -> tuple[MultiTrackParent, dict[str, np.ndarray]]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    macro_mean, macro_std = _fit_norm(macro)
    micro_mean, micro_std = _fit_norm(micro)
    state_mean, state_std = _fit_norm(state)
    x_macro = _apply_norm(macro, macro_mean, macro_std)
    x_micro = _apply_norm(micro, micro_mean, micro_std)
    x_state = _apply_norm(state, state_mean, state_std)
    y_scaled = (y * TARGET_SCALE).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTrackParent(state.shape[1]).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_macro), torch.from_numpy(x_micro), torch.from_numpy(x_state), torch.from_numpy(y_scaled)),
        batch_size=256,
        shuffle=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    huber = nn.SmoothL1Loss()
    model.train()
    for _ in range(int(epochs)):
        for mb, kb, sb, yb in loader:
            mb, kb, sb, yb = mb.to(device), kb.to(device), sb.to(device), yb.to(device)
            pred = model(mb, kb, sb)
            loss = huber(pred, yb) + 0.04 * (F.relu(-pred[:, 2]).mean() + F.relu(-pred[:, 3]).mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    norm = {"macro_mean": macro_mean, "macro_std": macro_std, "micro_mean": micro_mean, "micro_std": micro_std, "state_mean": state_mean, "state_std": state_std}
    return model.cpu().eval(), norm


def _predict_all(model: MultiTrackParent, macro: np.ndarray, micro: np.ndarray, state: np.ndarray, norm: dict[str, np.ndarray]) -> np.ndarray:
    x_macro = _apply_norm(macro, norm["macro_mean"], norm["macro_std"])
    x_micro = _apply_norm(micro, norm["micro_mean"], norm["micro_std"])
    x_state = _apply_norm(state, norm["state_mean"], norm["state_std"])
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x_macro), 1024):
            pred = model(
                torch.from_numpy(x_macro[start : start + 1024]),
                torch.from_numpy(x_micro[start : start + 1024]),
                torch.from_numpy(x_state[start : start + 1024]),
            ).numpy()
            out.append(pred / TARGET_SCALE)
    return np.vstack(out).astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chronos-2 + Kairos_23m + KAN state encoder parent challenger.")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--grid-out", type=Path, default=DEFAULT_GRID)
    p.add_argument("--train-stride", type=int, default=24)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--embed-batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=2040)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print(f"[{MODEL_ID}] loading data", flush=True)
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    split_ts = pd.Timestamp("2025-10-01")
    train_df = train_all[train_all["timestamp"] < split_ts].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= split_ts].reset_index(drop=True)
    feature_cols = _feature_cols(train_all, eval_df)
    audit = _audit_contract(train_all, eval_df, feature_cols)
    cfg = _parent_cfg()

    print(f"[{MODEL_ID}] preparing features", flush=True)
    train_feat = prepare_features(train_df, side_hint=0, close=_close(train_df), feature_cols=feature_cols)
    val_feat = prepare_features(val_df, side_hint=0, close=_close(val_df), feature_cols=feature_cols)
    eval_feat = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=feature_cols)
    state_cols = _state_cols(feature_cols, train_feat)
    micro_cols = [c for c in MICRO_PRIORITY if c in train_feat.columns]
    if not micro_cols:
        raise RuntimeError("no microstructure columns available for Kairos track")

    train_idx = np.arange(max(MACRO_LEN, MICRO_LEN) - 1, max(MACRO_LEN, MICRO_LEN, len(train_df) - MAX_LABEL_HOLD - 2), max(1, int(args.train_stride)), dtype=np.int64)
    val_idx = np.arange(len(val_df), dtype=np.int64)
    eval_idx = np.arange(len(eval_df), dtype=np.int64)
    y = _build_best_targets(train_df, train_idx, fee=cfg.fee, slip=cfg.slip)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{MODEL_ID}] loading Chronos-2 and Kairos_23m on {device}", flush=True)
    chronos = Chronos2Pipeline.from_pretrained(CHRONOS_MODEL, device_map=device)
    kairos = KairosAutoModel.from_pretrained(KAIROS_MODEL, trust_remote_code=True).to(device).eval()

    emb_dir = args.out_dir / "embeddings"
    print(f"[{MODEL_ID}] extracting Chronos macro embeddings", flush=True)
    train_macro = _extract_macro_embeddings(
        chronos,
        train_df,
        train_idx,
        cache_path=_embedding_cache_path(
            emb_dir,
            prefix="train_macro",
            model_name=CHRONOS_MODEL,
            frame=train_df,
            indices=train_idx,
            cols=[c for c in MACRO_COLS if c in train_df.columns],
            window_len=MACRO_LEN,
            extra_tag=f"csv={args.train_csv.name}|stride={args.train_stride}|split=train",
        ),
        batch_size=args.embed_batch,
    )
    val_macro = _extract_macro_embeddings(
        chronos,
        val_df,
        val_idx,
        cache_path=_embedding_cache_path(
            emb_dir,
            prefix="val_macro",
            model_name=CHRONOS_MODEL,
            frame=val_df,
            indices=val_idx,
            cols=[c for c in MACRO_COLS if c in val_df.columns],
            window_len=MACRO_LEN,
            extra_tag=f"csv={args.train_csv.name}|split=val",
        ),
        batch_size=args.embed_batch,
    )
    eval_macro = _extract_macro_embeddings(
        chronos,
        eval_df,
        eval_idx,
        cache_path=_embedding_cache_path(
            emb_dir,
            prefix="eval_macro",
            model_name=CHRONOS_MODEL,
            frame=eval_df,
            indices=eval_idx,
            cols=[c for c in MACRO_COLS if c in eval_df.columns],
            window_len=MACRO_LEN,
            extra_tag=f"csv={args.eval_csv.name}|split=eval",
        ),
        batch_size=args.embed_batch,
    )
    print(f"[{MODEL_ID}] extracting Kairos micro embeddings", flush=True)
    train_micro = _extract_micro_embeddings(
        kairos,
        train_feat,
        train_idx,
        micro_cols,
        cache_path=_embedding_cache_path(
            emb_dir,
            prefix="train_micro",
            model_name=KAIROS_MODEL,
            frame=train_feat,
            indices=train_idx,
            cols=micro_cols,
            window_len=MICRO_LEN,
            extra_tag=f"csv={args.train_csv.name}|stride={args.train_stride}|split=train",
        ),
        batch_size=args.embed_batch,
    )
    val_micro = _extract_micro_embeddings(
        kairos,
        val_feat,
        val_idx,
        micro_cols,
        cache_path=_embedding_cache_path(
            emb_dir,
            prefix="val_micro",
            model_name=KAIROS_MODEL,
            frame=val_feat,
            indices=val_idx,
            cols=micro_cols,
            window_len=MICRO_LEN,
            extra_tag=f"csv={args.train_csv.name}|split=val",
        ),
        batch_size=args.embed_batch,
    )
    eval_micro = _extract_micro_embeddings(
        kairos,
        eval_feat,
        eval_idx,
        micro_cols,
        cache_path=_embedding_cache_path(
            emb_dir,
            prefix="eval_micro",
            model_name=KAIROS_MODEL,
            frame=eval_feat,
            indices=eval_idx,
            cols=micro_cols,
            window_len=MICRO_LEN,
            extra_tag=f"csv={args.eval_csv.name}|split=eval",
        ),
        batch_size=args.embed_batch,
    )

    train_state = train_feat.iloc[train_idx].reindex(columns=state_cols).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    val_state = val_feat.reindex(columns=state_cols).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    eval_state = eval_feat.reindex(columns=state_cols).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)

    print(f"[{MODEL_ID}] training head epochs={args.epochs}", flush=True)
    model, norm = _fit_model(train_macro, train_micro, train_state, y, epochs=int(args.epochs), seed=int(args.seed))
    print(f"[{MODEL_ID}] predicting validation/OOS", flush=True)
    val_pred = _predict_all(model, val_macro, val_micro, val_state, norm)
    eval_pred = _predict_all(model, eval_macro, eval_micro, eval_state, norm)

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for runtime in _runtime_grid():
        print(f"[{MODEL_ID}] validation {runtime.name}", flush=True)
        v1 = backtest(val_df, val_pred, runtime, fee=cfg.fee, slip=cfg.slip)
        v2 = backtest(val_df, val_pred, runtime, fee=cfg.fee * 2.0, slip=cfg.slip * 2.0)
        v3 = backtest(val_df, val_pred, runtime, fee=cfg.fee * 3.0, slip=cfg.slip * 3.0)
        row = {"runtime": asdict(runtime), "validation_cost1": v1, "validation_cost2": v2, "validation_cost3": v3, "selection_score": _selection_score(v1, v2, v3)}
        rows.append(row)
        if best is None or row["selection_score"] > best["selection_score"]:
            best = row
    if best is None:
        raise RuntimeError("no runtime candidates")
    selected = RuntimeConfig(**best["runtime"])
    print(f"[{MODEL_ID}] selected {selected.name}; running OOS", flush=True)
    metrics: dict[str, Any] = {}
    ledger_path = args.report_out.with_name(f"{args.report_out.stem}_cost1_ledger.csv")
    for mult in (1, 2, 3):
        r = backtest(eval_df, eval_pred, selected, fee=cfg.fee * mult, slip=cfg.slip * mult, record=(mult == 1))
        if mult == 1:
            ledger = pd.DataFrame(r.pop("trade_records", []))
            ledger_path.parent.mkdir(parents=True, exist_ok=True)
            ledger.to_csv(ledger_path, index=False)
        metrics[f"cost{mult}"] = r

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "multitrack_foundation_parent_v40.pt"
    torch.save(
        {
            "model_id": MODEL_ID,
            "state_dict": model.state_dict(),
            "feature_cols": feature_cols,
            "state_cols": state_cols,
            "micro_cols": micro_cols,
            "norm": norm,
            "selected_runtime": asdict(selected),
            "chronos_model": CHRONOS_MODEL,
            "kairos_model": KAIROS_MODEL,
        },
        model_path,
    )

    pd.DataFrame(
        [
            {
                **{f"rt_{k}": v for k, v in row["runtime"].items()},
                "selection_score": row["selection_score"],
                "val_pnl": row["validation_cost1"]["pnl"],
                "val_mdd": row["validation_cost1"]["mdd"],
                "val_trades": row["validation_cost1"]["trades"],
                "val_cost2_pnl": row["validation_cost2"]["pnl"],
                "val_cost3_pnl": row["validation_cost3"]["pnl"],
            }
            for row in rows
        ]
    ).to_csv(args.grid_out, index=False)

    blocking = list(audit.get("blocking", []))
    warnings = list(audit.get("warnings", []))
    if metrics["cost1"]["pnl"] <= 0.0:
        warnings.append("cost1_not_survived")
    if metrics["cost2"]["pnl"] <= 0.0:
        warnings.append("cost2_not_survived")
    if metrics["cost3"]["pnl"] <= 0.0:
        warnings.append("cost3_not_survived")
    final_audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": "promote" if not blocking and metrics["cost1"]["pnl"] > MARGIN110_COST1 else "iterate",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS only after selection",
        "feature_audit": audit,
        "baseline_margin110_cost1": MARGIN110_COST1,
        "baseline_v31_cost1": V31_COST1,
        "selected_runtime": asdict(selected),
        "metrics": metrics,
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Three-track parent challenger. Chronos-2 embeds 1024-bar OHLCV macro context, Kairos_23m encodes 72-bar microstructure sequences, and a lightweight KAN-style spline state encoder maps current-bar state and AI features. The fused representation predicts long/short best utility and adverse risk; abstention is decided by a threshold gate on 2025 Q4.",
        "split_policy": "train=2025 Jan-Sep, selection=2025 Q4, OOS=2026 fixed",
        "chronos_model": CHRONOS_MODEL,
        "kairos_model": KAIROS_MODEL,
        "macro_cols": MACRO_COLS,
        "micro_cols": micro_cols,
        "state_cols": state_cols,
        "train_rows": int(len(train_idx)),
        "metrics": metrics,
        "selection": best,
        "audit": final_audit,
        "artifacts": {"model": str(model_path), "report": str(args.report_out), "audit": str(args.audit_out), "grid": str(args.grid_out), "ledger": str(ledger_path)},
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(final_audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "model": str(model_path), "selected": selected.name, "metrics": metrics, "verdict": final_audit["verdict"]}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
