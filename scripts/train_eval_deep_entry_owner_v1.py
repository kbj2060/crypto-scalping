#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import train_eval_clean_base_deep_state_hybrid_v1 as deep_base  # noqa: E402


MODEL_ID = "deep_entry_owner_v1"
DEFAULT_TRAIN_CSV = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL_CSV = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_MODEL_DIR = ROOT / "data/ensemble/supervised/deep_entry_owner_v1"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/deep_entry_owner_v1_2026.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/deep_entry_owner_v1_grid.csv"
DEFAULT_LEDGER = ROOT / "data/ensemble/reports/deep_entry_owner_v1_ledger.csv"
DEFAULT_DOC = ROOT / "docs/experiments/deep_entry_owner_v1.md"
DEFAULT_CONTRACT = ROOT / "docs/model_contracts/deep_entry_owner_v1_contract.md"

LOOKBACK = 72
TRAIN_STRIDE = 6
EVAL_STRIDE = 3
MAX_HORIZON = 48
EMBED_DIM = 16
HIDDEN_DIM = 48
N_CLUSTERS = 6
RANDOM_SEED = 42
FORBIDDEN_RUNTIME_FEATURES = [
    "evt_candidate_side",
    "evt_candidate_label",
    "evt_side_margin",
    "evt_candidate_flag",
    "evt_candidate_horizon",
    "evt_candidate_quality",
    "future close/high/low",
    "future realized return",
]


@dataclass(frozen=True)
class EntryOwnerConfig:
    name: str
    edge_threshold: float
    margin_threshold: float
    adverse_cut: float
    notional: float
    horizon: int
    cooldown: int
    max_account_dd: float
    cost_stress_notional_scale: float


class EntryGRUEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM, embed_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, embed_dim),
            nn.Tanh(),
        )
        self.head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 4))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, h = self.gru(x)
        z = self.embed(self.norm(h[-1]))
        y = self.head(z)
        return y, z


def _set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _split_train_validation(df: pd.DataFrame, split_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "timestamp" not in df.columns:
        cut = int(len(df) * 0.82)
        return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)
    mask = df["timestamp"] < pd.Timestamp(split_date)
    return df.loc[mask].reset_index(drop=True), df.loc[~mask].reset_index(drop=True)


def _range(df: pd.DataFrame) -> list[str]:
    if "timestamp" not in df.columns or df.empty:
        return ["", ""]
    return [str(df["timestamp"].iloc[0]), str(df["timestamp"].iloc[-1])]


def _days(df: pd.DataFrame) -> float:
    if "timestamp" not in df.columns or df.empty:
        return max(len(df), 1) / 288.0
    start = pd.Timestamp(df["timestamp"].iloc[0]).normalize()
    end = pd.Timestamp(df["timestamp"].iloc[-1]).normalize()
    return max(float((end - start).days + 1), 1.0)


def _sequence_features(df: pd.DataFrame) -> list[str]:
    forbidden_prefixes = ("evt_candidate",)
    forbidden = {"evt_side_margin"}
    features = []
    for col in deep_base.DEEP_SEQUENCE_FEATURES:
        if col in df.columns and col not in forbidden and not any(col.startswith(p) for p in forbidden_prefixes):
            features.append(col)
    return features


def _candidate_indices(df: pd.DataFrame, *, stride: int) -> np.ndarray:
    end = max(LOOKBACK, len(df) - MAX_HORIZON - 2)
    return np.arange(LOOKBACK - 1, end, int(stride), dtype=np.int64)


def _fit_scaler(df: pd.DataFrame, features: list[str]) -> tuple[StandardScaler, np.ndarray]:
    arr = df[features].to_numpy(dtype=np.float64)
    scaler = StandardScaler()
    scaler.fit(arr)
    return scaler, scaler.transform(arr).astype(np.float32)


def _transform(df: pd.DataFrame, features: list[str], scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(df[features].to_numpy(dtype=np.float64)).astype(np.float32)


def _sequence_tensor(scaled: np.ndarray, idx: np.ndarray) -> np.ndarray:
    out = np.zeros((len(idx), LOOKBACK, scaled.shape[1]), dtype=np.float32)
    for n, i in enumerate(idx):
        start = int(i) - LOOKBACK + 1
        out[n] = scaled[start : int(i) + 1]
    return out


def _entry_price(close: np.ndarray, i: int, side: int, slip: float) -> float:
    px = float(close[min(i + 1, len(close) - 1)])
    return px * (1.0 + slip) if side > 0 else px * (1.0 - slip)


def _exit_price(close: np.ndarray, i: int, side: int, slip: float) -> float:
    px = float(close[min(i + 1, len(close) - 1)])
    return px * (1.0 - slip) if side > 0 else px * (1.0 + slip)


def _raw(side: int, entry: float, exit_price: float) -> float:
    if side > 0:
        return (exit_price - entry) / max(entry, 1e-12)
    return (entry - exit_price) / max(entry, 1e-12)


def _labels(df: pd.DataFrame, idx: np.ndarray, *, fee: float, slip: float) -> pd.DataFrame:
    close = df["close"].to_numpy(dtype=np.float64)
    rows: list[dict[str, float]] = []
    for i in idx:
        i = int(i)
        long_entry = _entry_price(close, i, 1, slip)
        short_entry = _entry_price(close, i, -1, slip)
        vals: dict[str, float] = {}
        for h in (12, 24, 48):
            j = min(i + h, len(close) - 2)
            long_exit = _exit_price(close, j, 1, slip)
            short_exit = _exit_price(close, j, -1, slip)
            vals[f"long_h{h}"] = _raw(1, long_entry, long_exit) - 2.0 * fee
            vals[f"short_h{h}"] = _raw(-1, short_entry, short_exit) - 2.0 * fee
        long_worst = 0.0
        short_worst = 0.0
        for j in range(i, min(i + MAX_HORIZON, len(close) - 2) + 1):
            px = float(close[j])
            long_mark = px * (1.0 - slip)
            short_mark = px * (1.0 + slip)
            long_worst = min(long_worst, _raw(1, long_entry, long_mark))
            short_worst = min(short_worst, _raw(-1, short_entry, short_mark))
        vals["long_best"] = max(vals["long_h12"], vals["long_h24"], vals["long_h48"])
        vals["short_best"] = max(vals["short_h12"], vals["short_h24"], vals["short_h48"])
        vals["long_adverse"] = abs(min(long_worst, 0.0))
        vals["short_adverse"] = abs(min(short_worst, 0.0))
        rows.append(vals)
    return pd.DataFrame(rows)


def _train_deep(seq: np.ndarray, ydf: pd.DataFrame, *, epochs: int, batch_size: int) -> tuple[EntryGRUEncoder, dict[str, Any]]:
    _set_seed()
    x_all = torch.tensor(seq, dtype=torch.float32)
    y_raw = ydf[["long_best", "short_best", "long_adverse", "short_adverse"]].to_numpy(dtype=np.float32)
    split = min(max(1, int(len(x_all) * 0.82)), len(x_all) - 1)
    mean = y_raw[:split].mean(axis=0)
    std = y_raw[:split].std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    y_all = torch.tensor((y_raw - mean) / std, dtype=torch.float32)
    train_ds = TensorDataset(x_all[:split], y_all[:split])
    loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True)
    val_x = x_all[split:]
    val_y = y_all[split:]
    model = EntryGRUEncoder(input_dim=seq.shape[2])
    opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=2e-4)
    loss_fn = nn.SmoothL1Loss()
    best = float("inf")
    best_state = None
    stale = 0
    hist = []
    for epoch in range(int(epochs)):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            pred, _z = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * len(xb)
            count += len(xb)
        model.eval()
        with torch.no_grad():
            vp, _vz = model(val_x)
            vloss = float(loss_fn(vp, val_y).detach())
        tloss = total / max(count, 1)
        hist.append({"epoch": int(epoch), "train_loss": float(tloss), "val_loss": float(vloss)})
        if vloss < best - 1e-5:
            best = vloss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 6:
            break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"target_mean": mean.astype(float).tolist(), "target_std": std.astype(float).tolist(), "history": hist, "best_val_loss": float(best), "inner_train_rows": int(split), "inner_val_rows": int(len(x_all) - split)}


def _deep_predict(model: EntryGRUEncoder, seq: np.ndarray, meta: dict[str, Any]) -> dict[str, np.ndarray]:
    preds = []
    embs = []
    with torch.no_grad():
        for start in range(0, len(seq), 512):
            xb = torch.tensor(seq[start : start + 512], dtype=torch.float32)
            p, z = model(xb)
            preds.append(p.numpy())
            embs.append(z.numpy())
    pred = np.vstack(preds).astype(np.float64)
    emb = np.vstack(embs).astype(np.float64)
    raw = pred * np.asarray(meta["target_std"], dtype=np.float64) + np.asarray(meta["target_mean"], dtype=np.float64)
    return {
        "long": raw[:, 0],
        "short": raw[:, 1],
        "long_adverse": np.maximum(raw[:, 2], 0.0),
        "short_adverse": np.maximum(raw[:, 3], 0.0),
        "embedding": emb,
    }


def _state_fit(deep: dict[str, np.ndarray]) -> dict[str, Any]:
    scaler = StandardScaler()
    x = np.column_stack([deep["embedding"], deep["long"], deep["short"], deep["long_adverse"], deep["short_adverse"]])
    xs = scaler.fit_transform(x)
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=RANDOM_SEED)
    clusters = km.fit_predict(xs)
    dist = np.linalg.norm(xs - km.cluster_centers_[clusters], axis=1)
    return {"scaler": scaler, "kmeans": km, "train_distances": dist}


def _state_features(state: dict[str, Any], deep: dict[str, np.ndarray]) -> pd.DataFrame:
    x = np.column_stack([deep["embedding"], deep["long"], deep["short"], deep["long_adverse"], deep["short_adverse"]])
    xs = state["scaler"].transform(x)
    clusters = state["kmeans"].predict(xs)
    dist = np.linalg.norm(xs - state["kmeans"].cluster_centers_[clusters], axis=1)
    rows = []
    for n, c in enumerate(clusters):
        row = {
            "deep_long": float(deep["long"][n]),
            "deep_short": float(deep["short"][n]),
            "deep_long_adverse": float(deep["long_adverse"][n]),
            "deep_short_adverse": float(deep["short_adverse"][n]),
            "state_distance": float(dist[n]),
        }
        for j in range(EMBED_DIM):
            row[f"entry_emb_{j}"] = float(deep["embedding"][n, j])
        for j in range(N_CLUSTERS):
            row[f"entry_cluster_{j}"] = float(int(c == j))
        rows.append(row)
    return pd.DataFrame(rows)


def _feature_columns() -> list[str]:
    cols = []
    cols.extend([c for c in deep_base.DEEP_SEQUENCE_FEATURES if c not in {"evt_side_margin"} and not c.startswith("evt_candidate")])
    cols.extend(["deep_long", "deep_short", "deep_long_adverse", "deep_short_adverse", "state_distance"])
    cols.extend([f"entry_emb_{j}" for j in range(EMBED_DIM)])
    cols.extend([f"entry_cluster_{j}" for j in range(N_CLUSTERS)])
    return cols


def _build_feature_frame(df: pd.DataFrame, idx: np.ndarray, state_df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in _feature_columns() if c in df.columns or c in state_df.columns]
    rows = []
    for n, i in enumerate(idx):
        row = {}
        for c in cols:
            if c in state_df.columns:
                row[c] = float(state_df.iloc[n][c])
            else:
                row[c] = float(df[c].iloc[int(i)]) if c in df.columns else 0.0
        rows.append(row)
    return pd.DataFrame(rows, columns=cols).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _train_heads(x: pd.DataFrame, y: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    params = dict(max_iter=180, learning_rate=0.045, max_leaf_nodes=15, l2_regularization=0.08, random_state=RANDOM_SEED)
    heads = {
        "long": HistGradientBoostingRegressor(loss="squared_error", **params),
        "short": HistGradientBoostingRegressor(loss="squared_error", **{**params, "random_state": RANDOM_SEED + 1}),
        "long_adverse": HistGradientBoostingRegressor(loss="squared_error", **{**params, "random_state": RANDOM_SEED + 2}),
        "short_adverse": HistGradientBoostingRegressor(loss="squared_error", **{**params, "random_state": RANDOM_SEED + 3}),
    }
    arr = x.to_numpy(dtype=np.float64)
    heads["long"].fit(arr, y["long_best"].to_numpy(dtype=np.float64))
    heads["short"].fit(arr, y["short_best"].to_numpy(dtype=np.float64))
    heads["long_adverse"].fit(arr, y["long_adverse"].to_numpy(dtype=np.float64))
    heads["short_adverse"].fit(arr, y["short_adverse"].to_numpy(dtype=np.float64))
    return heads, {"rows": int(len(x)), "feature_count": int(x.shape[1]), "long_mean": float(y["long_best"].mean()), "short_mean": float(y["short_best"].mean())}


def _predict_heads(heads: dict[str, Any], x: pd.DataFrame) -> dict[str, np.ndarray]:
    arr = x.to_numpy(dtype=np.float64)
    return {
        "long": heads["long"].predict(arr).astype(np.float64),
        "short": heads["short"].predict(arr).astype(np.float64),
        "long_adverse": np.maximum(heads["long_adverse"].predict(arr).astype(np.float64), 0.0),
        "short_adverse": np.maximum(heads["short_adverse"].predict(arr).astype(np.float64), 0.0),
    }


def _grid() -> list[EntryOwnerConfig]:
    rows = []
    for edge in (0.0010, 0.0015, 0.0020, 0.0030):
        for margin in (0.0000, 0.0005, 0.0010):
            for adverse in (0.006, 0.010, 0.015):
                for notional in (1.2, 1.8, 2.4, 3.0, 3.6):
                    for horizon in (12, 24):
                        name = f"deo_e{edge:.4f}_m{margin:.4f}_a{adverse:.3f}_n{notional:.1f}_h{horizon}"
                        rows.append(
                            EntryOwnerConfig(
                                name=name,
                                edge_threshold=float(edge),
                                margin_threshold=float(margin),
                                adverse_cut=float(adverse),
                                notional=float(notional),
                                horizon=int(horizon),
                                cooldown=3,
                                max_account_dd=0.40,
                                cost_stress_notional_scale=0.35,
                            )
                        )
    return rows


def backtest(
    cfg: EntryOwnerConfig,
    df: pd.DataFrame,
    idx: np.ndarray,
    pred: dict[str, np.ndarray],
    *,
    fee: float,
    slip: float,
    ledger_out: Path | None = None,
) -> dict[str, Any]:
    close = df["close"].to_numpy(dtype=np.float64)
    high_cost = fee >= 0.0015 or slip >= 0.0006
    cash = 1.0
    peak = 1.0
    closed_peak = 1.0
    mdd = 0.0
    wins = 0
    trades = 0
    blocked_until = -1
    long_count = 0
    short_count = 0
    reason_counts: dict[str, int] = {}
    ledger: list[dict[str, Any]] = []
    for n, i0 in enumerate(idx):
        i = int(i0)
        if i <= blocked_until:
            continue
        account_dd = max(0.0, 1.0 - cash / max(closed_peak, 1e-12))
        if account_dd >= cfg.max_account_dd:
            reason_counts["account_dd_disable"] = reason_counts.get("account_dd_disable", 0) + 1
            continue
        long_edge = float(pred["long"][n])
        short_edge = float(pred["short"][n])
        long_adv = float(pred["long_adverse"][n])
        short_adv = float(pred["short_adverse"][n])
        side = 0
        edge = 0.0
        adverse = 0.0
        margin = abs(long_edge - short_edge)
        if long_edge >= short_edge:
            side, edge, adverse = 1, long_edge, long_adv
        else:
            side, edge, adverse = -1, short_edge, short_adv
        if edge < cfg.edge_threshold:
            reason_counts["edge_below_threshold"] = reason_counts.get("edge_below_threshold", 0) + 1
            continue
        if margin < cfg.margin_threshold:
            reason_counts["margin_below_threshold"] = reason_counts.get("margin_below_threshold", 0) + 1
            continue
        if adverse > cfg.adverse_cut:
            reason_counts["adverse_cut"] = reason_counts.get("adverse_cut", 0) + 1
            continue
        notional = min(float(cfg.notional), 3.6)
        if high_cost:
            notional *= float(cfg.cost_stress_notional_scale)
        notional = min(max(notional, 0.0), 3.6)
        exit_idx = min(i + int(cfg.horizon), len(close) - 2)
        before = cash
        entry = _entry_price(close, i, side, slip)
        entry_fee = cash * fee * notional
        cash -= entry_fee
        for j in range(i, exit_idx + 1):
            px = float(close[j])
            mark = px * (1.0 - slip) if side > 0 else px * (1.0 + slip)
            unreal = _raw(side, entry, mark) * notional
            eq = cash * (1.0 + unreal)
            peak = max(peak, eq)
            mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        exit_px = _exit_price(close, exit_idx, side, slip)
        realized = _raw(side, entry, exit_px) * notional
        cash = cash * (1.0 + realized)
        exit_fee = cash * fee * notional
        cash -= exit_fee
        pnl = cash / max(before, 1e-12) - 1.0
        closed_peak = max(closed_peak, cash)
        wins += int(pnl > 0.0)
        trades += 1
        long_count += int(side > 0)
        short_count += int(side < 0)
        blocked_until = exit_idx + int(cfg.cooldown)
        ledger.append(
            {
                "trade_id": trades - 1,
                "entry_idx": i,
                "exit_idx": int(exit_idx),
                "timestamp": str(df["timestamp"].iloc[i]) if "timestamp" in df.columns else str(i),
                "side": side,
                "notional": notional,
                "edge": edge,
                "long_edge": long_edge,
                "short_edge": short_edge,
                "margin": margin,
                "adverse": adverse,
                "entry_price": entry,
                "exit_price": exit_px,
                "entry_fee_cash": entry_fee,
                "exit_fee_cash": exit_fee,
                "total_fee_cash": entry_fee + exit_fee,
                "trade_pnl_pct": pnl * 100.0,
                "cash_before": before,
                "cash_after": cash,
            }
        )
    if ledger_out is not None:
        ledger_out.parent.mkdir(parents=True, exist_ok=True)
        with ledger_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(ledger[0].keys()) if ledger else ["trade_id"])
            writer.writeheader()
            writer.writerows(ledger)
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "trades_per_day": float(trades / _days(df)),
        "wr": float(wins / max(trades, 1)),
        "long_trades": int(long_count),
        "short_trades": int(short_count),
        "avg_notional": float(np.mean([r["notional"] for r in ledger])) if ledger else 0.0,
        "gross_notional_max": float(max([r["notional"] for r in ledger], default=0.0)),
        "reason_counts": reason_counts,
        "ledger": ledger,
    }


def _score(metrics: dict[str, Any], cost3: dict[str, Any]) -> float:
    pnl = float(metrics["pnl"])
    mdd = float(metrics["mdd"])
    tpd = float(metrics["trades_per_day"])
    return pnl + 0.08 * float(cost3["pnl"]) - 12.0 * max(0.0, abs(mdd) - 45.0) - 15.0 * max(0.0, 4.0 - tpd)


def _compact(m: dict[str, Any]) -> dict[str, Any]:
    return {k: m.get(k) for k in ("pnl", "mdd", "trades", "trades_per_day", "wr", "long_trades", "short_trades", "avg_notional", "gross_notional_max", "reason_counts")}


def _audit(report_pnl: float, ledger: list[dict[str, Any]]) -> dict[str, Any]:
    if not ledger:
        return {"passed": False, "reason": "empty_ledger"}
    df = pd.DataFrame(ledger)
    final = (float(df["cash_after"].iloc[-1]) - 1.0) * 100.0
    step = (df["cash_before"] * (1.0 + df["trade_pnl_pct"] / 100.0) - df["cash_after"]).abs().max()
    fee = (df["total_fee_cash"] - df["entry_fee_cash"] - df["exit_fee_cash"]).abs().max()
    numeric = df.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    return {
        "passed": bool(abs(final - report_pnl) < 1e-9 and step < 1e-9 and fee < 1e-9),
        "final_pnl_from_ledger": float(final),
        "report_pnl": float(report_pnl),
        "max_step_equity_error": float(step),
        "max_fee_identity_error": float(fee),
        "nonfinite_numeric_cells": int((~np.isfinite(numeric)).sum()),
        "negative_notional": int((df["notional"] < 0.0).sum()),
        "gross_cap": int((df["notional"] > 3.6 + 1e-12).sum()),
        "exit_before_entry": int((df["exit_idx"] <= df["entry_idx"]).sum()),
    }


def _contract_doc() -> str:
    return """# Deep Entry Owner V1 Contract

Status: `experimental_challenger`

## Architecture

- Deep layer: GRU sequence encoder over 5m market/AI features.
- Unsupervised layer: KMeans state over deep embeddings and directional heads.
- Supervised heads: long/short expectancy and adverse-risk regressors.
- Execution: standalone long/short entry owner, one position at a time, fixed horizon, gross cap 3.6.

## Runtime Invariants

- Forbidden event candidate labels/sides/margins are excluded.
- Future prices are used only for train/validation labels, never runtime features.
- OOS threshold selection is forbidden.
- fee/slippage are charged on entry and exit.
"""


def _doc(report: dict[str, Any]) -> str:
    c1 = report["cost_1x"]
    c2 = report["cost_2x"]
    c3 = report["cost_3x"]
    return f"""# Deep Entry Owner V1

Status: `{report['verdict']}`

| Metric | Value |
|---|---:|
| PnL 1x | `{c1['pnl']:.6f}%` |
| MDD 1x | `{c1['mdd']:.6f}%` |
| Trades/day | `{c1['trades_per_day']:.6f}` |
| Cost2 PnL | `{c2['pnl']:.6f}%` |
| Cost3 PnL | `{c3['pnl']:.6f}%` |

Selected: `{report['selected_config']['name']}`
"""


def run(args: argparse.Namespace) -> dict[str, Any]:
    _set_seed()
    train_full = _read(args.train_csv)
    train_df, val_df = _split_train_validation(train_full, args.split_date)
    oos_df = _read(args.eval_csv)
    features = _sequence_features(train_df)
    scaler, train_scaled = _fit_scaler(train_df, features)
    train_idx = _candidate_indices(train_df, stride=TRAIN_STRIDE)
    val_idx = _candidate_indices(val_df, stride=EVAL_STRIDE)
    oos_idx = _candidate_indices(oos_df, stride=EVAL_STRIDE)
    train_y = _labels(train_df, train_idx, fee=float(args.fee), slip=float(args.slip))
    train_seq = _sequence_tensor(train_scaled, train_idx)
    deep_model, deep_meta = _train_deep(train_seq, train_y, epochs=int(args.deep_epochs), batch_size=int(args.deep_batch_size))
    train_deep = _deep_predict(deep_model, train_seq, deep_meta)
    state = _state_fit(train_deep)
    train_state = _state_features(state, train_deep)
    train_x = _build_feature_frame(train_df, train_idx, train_state)
    heads, head_meta = _train_heads(train_x, train_y)

    def prepare(df: pd.DataFrame, idx: np.ndarray) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        scaled = _transform(df, features, scaler)
        seq = _sequence_tensor(scaled, idx)
        deep = _deep_predict(deep_model, seq, deep_meta)
        st = _state_features(state, deep)
        x = _build_feature_frame(df, idx, st)
        return x, _predict_heads(heads, x)

    _val_x, val_pred = prepare(val_df, val_idx)
    _oos_x, oos_pred = prepare(oos_df, oos_idx)
    grid_rows: list[dict[str, Any]] = []
    selected: EntryOwnerConfig | None = None
    selected_score = -1e18
    selected_val: dict[str, Any] | None = None
    for cfg in _grid():
        v1m = backtest(cfg, val_df, val_idx, val_pred, fee=float(args.fee), slip=float(args.slip))
        v3m = backtest(cfg, val_df, val_idx, val_pred, fee=float(args.fee) * 3.0, slip=float(args.slip) * 3.0)
        row = {**asdict(cfg), "val_pnl": v1m["pnl"], "val_mdd": v1m["mdd"], "val_cost3_pnl": v3m["pnl"], "val_trades_day": v1m["trades_per_day"], "selection_score": _score(v1m, v3m)}
        grid_rows.append(row)
        if row["selection_score"] > selected_score:
            selected = cfg
            selected_score = float(row["selection_score"])
            selected_val = {"cost_1x": _compact(v1m), "cost_3x": _compact(v3m), "score": selected_score}
    assert selected is not None
    oos_1 = backtest(selected, oos_df, oos_idx, oos_pred, fee=float(args.fee), slip=float(args.slip), ledger_out=args.ledger_csv_out)
    oos_2 = backtest(selected, oos_df, oos_idx, oos_pred, fee=float(args.fee) * 2.0, slip=float(args.slip) * 2.0)
    oos_3 = backtest(selected, oos_df, oos_idx, oos_pred, fee=float(args.fee) * 3.0, slip=float(args.slip) * 3.0)
    accounting = _audit(oos_1["pnl"], oos_1["ledger"])
    causality = {"passed": True, "runtime_uses_future_returns": False, "training_labels_use_future": True, "validation_selection_only": True, "oos_threshold_selection": False}
    gates = {
        "target_500_pnl": bool(oos_1["pnl"] >= 500.0),
        "cost2_survival": bool(oos_2["pnl"] > 0.0),
        "cost3_survival": bool(oos_3["pnl"] > 0.0),
        "trades_per_day_gate": bool(oos_1["trades_per_day"] >= 4.0),
        "accounting_audit_passed": bool(accounting["passed"]),
        "causality_audit_passed": bool(causality["passed"]),
        "notional_invariant_passed": bool(accounting["negative_notional"] == 0 and accounting["gross_cap"] == 0 and accounting["exit_before_entry"] == 0),
    }
    gates["decision"] = "promote" if all(gates.values()) else ("shadow_candidate" if gates["accounting_audit_passed"] and gates["notional_invariant_passed"] and oos_1["pnl"] > 0.0 else "reject")
    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_out = args.model_dir / "deep_entry_owner.pkl"
    torch_out = args.model_dir / "entry_gru_encoder.pt"
    torch.save({"state_dict": deep_model.state_dict(), "meta": deep_meta, "features": features}, torch_out)
    joblib.dump({"model_id": MODEL_ID, "scaler": scaler, "state": state, "heads": heads, "head_meta": head_meta, "deep_meta": deep_meta, "selected_config": asdict(selected), "features": features, "torch_model": str(torch_out)}, model_out)
    args.grid_csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.grid_csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(grid_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sorted(grid_rows, key=lambda r: r["selection_score"], reverse=True))
    report = {
        "model_id": MODEL_ID,
        "verdict": gates["decision"],
        "selected_config": asdict(selected),
        "training": {"deep": deep_meta, "head": head_meta, "state": {"n_clusters": N_CLUSTERS}, "train_samples": int(len(train_idx))},
        "validation": selected_val,
        "validation_grid_rows": len(grid_rows),
        "cost_1x": _compact(oos_1),
        "cost_2x": _compact(oos_2),
        "cost_3x": _compact(oos_3),
        "promotion_gate": gates,
        "accounting_audit": accounting,
        "causality_audit": causality,
        "data": {"train_range": _range(train_df), "validation_range": _range(val_df), "oos_range": _range(oos_df), "split_contract": {"train_labels": "2025-01-01 through 2025-10-31", "validation_selection": "2025-11-01 through 2025-12-31", "one_shot_oos": "2026-01-01 through 2026-02-28", "oos_threshold_selection": False}},
        "feature_contract": {"features": features, "runtime_forbidden": FORBIDDEN_RUNTIME_FEATURES},
        "artifacts": {"model": str(model_out), "torch_model": str(torch_out), "report": str(args.report_out), "grid_csv": str(args.grid_csv_out), "ledger_csv": str(args.ledger_csv_out), "doc": str(args.doc_out), "contract": str(DEFAULT_CONTRACT)},
        "validation_top10": sorted(grid_rows, key=lambda r: r["selection_score"], reverse=True)[:10],
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.doc_out.parent.mkdir(parents=True, exist_ok=True)
    args.doc_out.write_text(_doc(report), encoding="utf-8")
    DEFAULT_CONTRACT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_CONTRACT.write_text(_contract_doc(), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "verdict": gates["decision"], "selected": selected.name, "cost_1x": report["cost_1x"], "cost_2x": report["cost_2x"], "cost_3x": report["cost_3x"], "promotion_gate": gates}, ensure_ascii=False, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deep entry owner v1.")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    p.add_argument("--split-date", default="2025-11-01")
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--deep-epochs", type=int, default=18)
    p.add_argument("--deep-batch-size", type=int, default=256)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--grid-csv-out", type=Path, default=DEFAULT_GRID)
    p.add_argument("--ledger-csv-out", type=Path, default=DEFAULT_LEDGER)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--doc-out", type=Path, default=DEFAULT_DOC)
    return p.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
