#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, ACTION_LONG, predict_policy_frame
from scripts import train_eval_hf_v13_deep_jackpot_sequence_verifier_v23 as v23
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _days, _fill_price, _read
from scripts.train_eval_hf_v13_convex_runner_pyramid_v18 import _feature_frame
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig, _predict_cost_runner


MODEL_ID = "hf_v13_deep_alpha_mamba_ssm_v41_20260512"
DEFAULT_PARENT = v23.DEFAULT_PARENT
DEFAULT_JACKPOT = v23.DEFAULT_JACKPOT
DEFAULT_TRAIN = v23.DEFAULT_TRAIN
DEFAULT_EVAL = v23.DEFAULT_EVAL
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_deep_alpha_mamba_ssm_v41_20260512"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_deep_alpha_mamba_ssm_v41_20260512_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_deep_alpha_mamba_ssm_v41_20260512_audit.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/hf_v13_deep_alpha_mamba_ssm_v41_20260512_grid.csv"
HORIZONS = (12, 24, 48)
V21_2_COST1 = 199.5442148936891
V21_2_COST2 = 113.24305052028865
V21_2_COST3 = 24.714228358176072


@dataclass(frozen=True)
class DeepAlphaConfig:
    name: str
    edge_th: float
    margin_th: float
    notional: float
    take_profit: float
    stop_loss: float
    max_hold: int
    cooldown: int


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class MambaStyleSSMBlock(nn.Module):
    """Self-contained gated diagonal SSM block for the V27 replacement experiment."""

    def __init__(self, hidden: int, kernel_size: int = 5, dropout: float = 0.10) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.norm = RMSNorm(hidden)
        self.in_proj = nn.Linear(hidden, hidden * 2)
        self.dwconv = nn.Conv1d(hidden, hidden, kernel_size, groups=hidden)
        self.dt_proj = nn.Linear(hidden, hidden)
        self.b_proj = nn.Linear(hidden, hidden)
        self.c_proj = nn.Linear(hidden, hidden)
        self.a_log = nn.Parameter(torch.zeros(hidden))
        self.d = nn.Parameter(torch.ones(hidden))
        self.out_proj = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        u, gate = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        u = self.dwconv(F.pad(u.transpose(1, 2), (self.kernel_size - 1, 0))).transpose(1, 2)
        u = F.silu(u)

        dt = F.softplus(self.dt_proj(u)) + 1e-4
        b = torch.tanh(self.b_proj(u))
        c = torch.tanh(self.c_proj(u))
        a = torch.exp(self.a_log).view(1, -1)
        state = torch.zeros(u.size(0), u.size(-1), device=u.device, dtype=u.dtype)
        ys: list[torch.Tensor] = []
        for t in range(u.size(1)):
            decay = torch.exp(-dt[:, t, :] * a)
            state = decay * state + (1.0 - decay) * b[:, t, :] * u[:, t, :]
            ys.append(c[:, t, :] * state + self.d * u[:, t, :])
        y = torch.stack(ys, dim=1) * torch.sigmoid(gate)
        return residual + self.dropout(self.out_proj(y))


class DeepAlphaMambaStyleSSM(nn.Module):
    def __init__(self, seq_dim: int, hidden: int = 96, blocks: int = 2) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(seq_dim, hidden), nn.GELU(), nn.Dropout(0.05))
        self.blocks = nn.ModuleList([MambaStyleSSMBlock(hidden) for _ in range(blocks)])
        self.norm = RMSNorm(hidden)
        self.head = nn.Sequential(nn.Linear(hidden * 2, 96), nn.GELU(), nn.Dropout(0.10), nn.Linear(96, 2))

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(seq)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        pooled = torch.cat([x[:, -1, :], x.mean(dim=1)], dim=-1)
        return self.head(pooled)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _grid() -> list[DeepAlphaConfig]:
    rows: list[DeepAlphaConfig] = []
    i = 0
    for edge in (0.004, 0.006, 0.008):
        for margin in (0.0015, 0.0030):
            for notional in (0.8, 1.2):
                rows.append(DeepAlphaConfig(f"v27_e{edge:.3f}_m{margin:.4f}_n{notional:.1f}", edge, margin, notional, 0.030, 0.018, 24, 6))
                i += 1
    rows.append(DeepAlphaConfig("v27_slow_high_edge", 0.010, 0.0040, 1.2, 0.045, 0.022, 48, 12))
    return rows


def _seq_cols(df: pd.DataFrame) -> list[str]:
    cols = v23._select_seq_cols(df)
    extra = [
        c
        for c in df.columns
        if c.startswith("m7_") or c.startswith("teacher_") or c.startswith("clean_regime_2024_unsup_v4_")
    ]
    out: list[str] = []
    for c in cols + extra:
        lc = c.lower()
        if c not in out and c in df.columns and not any(tok in lc for tok in v23.FORBIDDEN) and not any(tok in lc for tok in ("target", "label", "future", "cash_after")):
            out.append(c)
    return out[:80]


def _normalizer(seqs: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "mean": np.nanmean(seqs, axis=(0, 1)).astype(np.float32),
        "std": (np.nanstd(seqs, axis=(0, 1)) + 1e-6).astype(np.float32),
    }


def _apply_norm(seqs: np.ndarray, norm: dict[str, np.ndarray]) -> np.ndarray:
    return ((seqs - norm["mean"][None, None, :]) / norm["std"][None, None, :]).astype(np.float32)


def _build_train_set(df: pd.DataFrame, seq_cols: list[str], *, fee: float, slip: float, stride: int = 3) -> dict[str, np.ndarray]:
    close = _close(df)
    seqs: list[np.ndarray] = []
    targets: list[list[float]] = []
    for i in range(v23.SEQ_LEN, len(df) - max(HORIZONS) - 2, stride):
        entry_i = min(i + 1, len(df) - 1)
        long_rewards: list[float] = []
        short_rewards: list[float] = []
        long_entry = _fill_price(df, entry_i, 1, slip, entry=True)
        short_entry = _fill_price(df, entry_i, -1, slip, entry=True)
        for h in HORIZONS:
            exit_i = min(i + h, len(df) - 1)
            long_exit = _fill_price(df, exit_i, 1, slip, entry=False)
            short_exit = _fill_price(df, exit_i, -1, slip, entry=False)
            long_rewards.append((long_exit - long_entry) / max(long_entry, 1e-12) - fee * 2.0)
            short_rewards.append((short_entry - short_exit) / max(short_entry, 1e-12) - fee * 2.0)
        seqs.append(v23._seq_at(df, i, seq_cols))
        targets.append([float(max(long_rewards)), float(max(short_rewards))])
    if not seqs:
        raise RuntimeError("no deep alpha train sequences")
    return {"seq": np.stack(seqs).astype(np.float32), "target": np.asarray(targets, dtype=np.float32)}


def _train_model(ds: dict[str, np.ndarray], norm: dict[str, np.ndarray], *, epochs: int) -> DeepAlphaMambaStyleSSM:
    x = _apply_norm(ds["seq"], norm)
    y = ds["target"].astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepAlphaMambaStyleSSM(x.shape[-1]).to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=128, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        batches = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            print(json.dumps({"stage": "train", "epoch": epoch + 1, "epochs": epochs, "loss": total_loss / max(batches, 1)}, ensure_ascii=False), flush=True)
    return model.cpu().eval()


def _predict_all(model: DeepAlphaMambaStyleSSM, df: pd.DataFrame, seq_cols: list[str], norm: dict[str, np.ndarray]) -> np.ndarray:
    seqs = np.stack([v23._seq_at(df, i, seq_cols) for i in range(len(df))]).astype(np.float32)
    x = _apply_norm(seqs, norm)
    out: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), 512):
            pred = model(torch.from_numpy(x[start : start + 512])).numpy()
            out.append(pred)
    return np.vstack(out).astype(np.float32)


def backtest(
    df: pd.DataFrame,
    bundle: dict[str, Any],
    jackpot_model: dict[str, Any],
    add_cfg: CostRunnerConfig,
    deep_q: np.ndarray,
    cfg: DeepAlphaConfig,
    *,
    fee: float,
    slip: float,
    cost_mult: float = 1.0,
    decisions: pd.DataFrame | None = None,
    record: bool = False,
) -> dict[str, Any]:
    close = _close(df)
    if decisions is None:
        decisions = predict_policy_frame(bundle, df, close=close)
    fee_eff = fee * cost_mult
    slip_eff = slip * cost_mult
    cash = peak = 1.0
    mdd = 0.0
    pos = 0
    owner = ""
    entry_price = entry_equity = 0.0
    entry_idx = 0
    parent_notional = notional = 0.0
    take_profit = stop_loss = 0.0
    max_hold = 0
    cooldown = next_cooldown = deep_cooldown = 0
    add_done = False
    mfe = mae = 0.0
    trades = wins = long_entries = short_entries = deep_entries = 0
    notional_sum = leverage_sum = 0.0
    exits: dict[str, int] = {}
    actions: dict[str, int] = {}
    records: list[dict[str, Any]] = []
    open_record: dict[str, Any] | None = None

    def mark(i: int) -> tuple[float, float]:
        if pos == 0:
            return cash, 0.0
        px = float(close[int(np.clip(i, 0, len(close) - 1))])
        raw = (px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1e-12)
        unreal = raw * notional
        return cash * (1.0 + unreal), unreal

    for i in range(0, len(df) - 2):
        eq, unreal = mark(i)
        peak = max(peak, eq)
        dd_abs = max(0.0, 1.0 - eq / max(peak, 1e-12))
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if pos != 0:
            mfe = max(mfe, unreal)
            mae = min(mae, unreal)
            hold = i - entry_idx
            reason = ""
            if take_profit > 0.0 and unreal >= take_profit:
                reason = f"{owner}_take_profit"
            elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
                reason = f"{owner}_stop_loss"
            elif max_hold > 0 and hold >= max_hold:
                reason = f"{owner}_max_hold"
            if owner == "v21_2" and not reason and not add_done and add_cfg.full_add_frac > 0.0 and unreal >= add_cfg.min_unrealized and hold >= add_cfg.min_bars_since_entry and dd_abs <= add_cfg.dd_block:
                state = {"parent_notional": parent_notional, "notional": notional, "bars_since_entry": hold, "unrealized": unreal, "mfe": mfe, "mae": mae, "drawdown_abs": dd_abs, "take_profit": take_profit, "stop_loss": stop_loss, "max_hold": max_hold}
                x = _feature_frame(df, bundle, decisions, i, state)
                _, _, _, q90, p_jackpot, p_bad, p_cost3 = _predict_cost_runner(jackpot_model, x)
                if p_jackpot >= add_cfg.jackpot_p and q90 >= add_cfg.jackpot_q90 and p_bad <= add_cfg.bad_cap and p_cost3 >= 0.40:
                    fill_i = min(i + 1, len(df) - 1)
                    delta = max(0.0, min(parent_notional * add_cfg.full_add_frac, parent_notional * add_cfg.max_total_mult - notional))
                    add_px = _fill_price(df, fill_i, pos, slip_eff, entry=True)
                    new_notional = notional + delta
                    entry_price = (entry_price * notional + add_px * delta) / max(new_notional, 1e-12)
                    before = cash
                    cash -= before * fee_eff * delta
                    notional = new_notional
                    actions["v21_add_on"] = actions.get("v21_add_on", 0) + 1
                else:
                    actions["v21_reject"] = actions.get("v21_reject", 0) + 1
                add_done = True
            if reason:
                fill_i = min(i + 1, len(df) - 1)
                exit_px = _fill_price(df, fill_i, pos, slip_eff, entry=False)
                raw = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw * notional)
                cash -= before * fee_eff * notional
                trades += 1
                wins += int(cash > entry_equity)
                exits[reason] = exits.get(reason, 0) + 1
                if record and open_record is not None:
                    out = dict(open_record)
                    out.update({"exit_signal_timestamp": str(df["timestamp"].iloc[i]), "exit_fill_timestamp": str(df["timestamp"].iloc[fill_i]), "exit_reason": reason, "realized_net_pct": float((cash / max(entry_equity, 1e-12) - 1.0) * 100.0), "final_notional_exposure": float(notional), "mfe_pct": float(mfe * 100.0), "mae_pct": float(mae * 100.0), "fee_exit_pct": float(fee_eff * notional * 100.0), "cash_after": float(cash)})
                    records.append(out)
                pos = 0
                owner = ""
                cooldown = int(next_cooldown)
                next_cooldown = 0
                deep_cooldown = max(deep_cooldown, int(cfg.cooldown))
                add_done = False
                open_record = None
                continue
        if pos != 0:
            continue
        if cooldown > 0:
            cooldown -= 1
            continue
        if deep_cooldown > 0:
            deep_cooldown -= 1
        dec = decisions.iloc[i]
        if int(dec.action) != ACTION_CASH and int(dec.side) != 0:
            fill_i = min(i + 1, len(df) - 1)
            pos = int(dec.side)
            owner = "v21_2"
            entry_price = _fill_price(df, fill_i, pos, slip_eff, entry=True)
            entry_equity = cash
            entry_idx = i
            parent_notional = min(float(dec.notional_exposure), add_cfg.max_entry_notional)
            notional = parent_notional
            take_profit = float(dec.take_profit)
            stop_loss = float(dec.stop_loss)
            max_hold = int(dec.max_hold_bars)
            next_cooldown = int(dec.cooldown_bars)
            cash -= cash * fee_eff * notional
            long_entries += int(pos > 0)
            short_entries += int(pos < 0)
            notional_sum += notional
            leverage_sum += float(dec.leverage)
            mfe = mae = 0.0
            add_done = False
            actions["v21_entry"] = actions.get("v21_entry", 0) + 1
            if record:
                open_record = {"entry_signal_timestamp": str(df["timestamp"].iloc[i]), "entry_fill_timestamp": str(df["timestamp"].iloc[fill_i]), "owner": owner, "side": "LONG" if pos > 0 else "SHORT", "entry_price": float(entry_price), "notional_exposure": float(notional), "leverage": float(dec.leverage), "take_profit": float(take_profit), "stop_loss": float(stop_loss), "max_hold_bars": int(max_hold), "fee_entry_pct": float(fee_eff * notional * 100.0)}
            continue
        if deep_cooldown <= 0 and i >= v23.SEQ_LEN:
            ql, qs = float(deep_q[i, 0]), float(deep_q[i, 1])
            side = 1 if ql > qs else -1
            edge = max(ql, qs)
            margin = abs(ql - qs)
            if edge >= cfg.edge_th and margin >= cfg.margin_th:
                fill_i = min(i + 1, len(df) - 1)
                pos = side
                owner = "deep_alpha"
                entry_price = _fill_price(df, fill_i, pos, slip_eff, entry=True)
                entry_equity = cash
                entry_idx = i
                parent_notional = float(cfg.notional)
                notional = float(cfg.notional)
                take_profit = float(cfg.take_profit)
                stop_loss = float(cfg.stop_loss)
                max_hold = int(cfg.max_hold)
                next_cooldown = int(cfg.cooldown)
                cash -= cash * fee_eff * notional
                long_entries += int(pos > 0)
                short_entries += int(pos < 0)
                deep_entries += 1
                notional_sum += notional
                leverage_sum += max(notional, 1.0)
                mfe = mae = 0.0
                add_done = True
                actions["deep_entry"] = actions.get("deep_entry", 0) + 1
                if record:
                    open_record = {"entry_signal_timestamp": str(df["timestamp"].iloc[i]), "entry_fill_timestamp": str(df["timestamp"].iloc[fill_i]), "owner": owner, "side": "LONG" if pos > 0 else "SHORT", "entry_price": float(entry_price), "notional_exposure": float(notional), "deep_q_long": ql, "deep_q_short": qs, "take_profit": float(take_profit), "stop_loss": float(stop_loss), "max_hold_bars": int(max_hold), "fee_entry_pct": float(fee_eff * notional * 100.0)}
    if pos != 0:
        fill_i = len(df) - 1
        exit_px = _fill_price(df, fill_i, pos, slip_eff, entry=False)
        raw = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * notional)
        cash -= before * fee_eff * notional
        trades += 1
        wins += int(cash > entry_equity)
        exits["forced_end"] = exits.get("forced_end", 0) + 1
    n = max(long_entries + short_entries, 1)
    out = {"pnl": float((cash - 1.0) * 100.0), "mdd": float(mdd * 100.0), "trades": int(trades), "wr": float(wins / max(trades, 1)), "trades_per_day": float(trades / _days(df)), "deep_entries": int(deep_entries), "long_entries": int(long_entries), "short_entries": int(short_entries), "avg_notional": float(notional_sum / n), "avg_leverage": float(leverage_sum / n), "exits": exits, "runner_actions": actions}
    if record:
        out["trade_records"] = records
    return out


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1["trades"]) < 20:
        return -1e9 + float(c1["pnl"])
    return float(c1["pnl"] + 0.35 * c2["pnl"] + 0.15 * c3["pnl"] - 0.35 * abs(c1["mdd"]) + 0.25 * min(c1.get("deep_entries", 0), 80))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V41 deep alpha Mamba-style SSM sleeve.")
    p.add_argument("--parent-model", type=Path, default=DEFAULT_PARENT)
    p.add_argument("--jackpot-model", type=Path, default=DEFAULT_JACKPOT)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--grid-out", type=Path, default=DEFAULT_GRID)
    p.add_argument("--epochs", type=int, default=140)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bundle = joblib.load(args.parent_model)
    jackpot_payload = joblib.load(args.jackpot_model)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    base = dict(bundle["config"])
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    train = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    seq_cols = _seq_cols(train_all)
    forbidden_cols = [c for c in seq_cols if any(tok in c.lower() for tok in v23.FORBIDDEN)]
    feature_audit = _audit_contract(train_all, eval_df, list(bundle.get("feature_cols") or []))
    train_ds = _build_train_set(train, seq_cols, fee=float(base["fee"]), slip=float(base["slip"]), stride=3)
    norm = _normalizer(train_ds["seq"])
    model = _train_model(train_ds, norm, epochs=args.epochs)
    val_q = _predict_all(model, val, seq_cols, norm)
    eval_q = _predict_all(model, eval_df, seq_cols, norm)
    val_dec = predict_policy_frame(bundle, val, close=_close(val))
    eval_dec = predict_policy_frame(bundle, eval_df, close=_close(eval_df))
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for cfg in _grid():
        v1 = backtest(val, bundle, jackpot_model, add_cfg, val_q, cfg, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=1.0, decisions=val_dec)
        v2 = backtest(val, bundle, jackpot_model, add_cfg, val_q, cfg, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=2.0, decisions=val_dec)
        v3 = backtest(val, bundle, jackpot_model, add_cfg, val_q, cfg, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=3.0, decisions=val_dec)
        row = {"config": asdict(cfg), "validation_cost1": v1, "validation_cost2": v2, "validation_cost3": v3, "selection_score": _score(v1, v2, v3)}
        rows.append(row)
        if best is None or row["selection_score"] > best["selection_score"]:
            best = row
    selected = DeepAlphaConfig(**best["config"])
    metrics: dict[str, Any] = {}
    ledgers: dict[str, str] = {}
    for mult in (1, 2, 3):
        r = backtest(eval_df, bundle, jackpot_model, add_cfg, eval_q, selected, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=float(mult), decisions=eval_dec, record=(mult == 1))
        if mult == 1:
            ledger = pd.DataFrame(r.pop("trade_records", []))
            lp = args.report_out.with_name(args.report_out.stem + "_cost1_ledger.csv")
            lp.parent.mkdir(parents=True, exist_ok=True)
            ledger.to_csv(lp, index=False)
            ledgers["cost1"] = str(lp)
        metrics[f"cost{mult}"] = r
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "v41_deep_alpha_mamba_ssm.pt"
    torch.save({"model_id": MODEL_ID, "architecture": "mamba_style_gated_diagonal_ssm", "state_dict": model.state_dict(), "seq_cols": seq_cols, "norm": norm, "selected_config": asdict(selected), "add_config": asdict(add_cfg), "parent_model": str(args.parent_model), "jackpot_model": str(args.jackpot_model)}, model_path)
    manifest_path = args.out_dir / "feature_manifest.json"
    manifest_path.write_text(json.dumps({"seq_cols": seq_cols, "seq_len": v23.SEQ_LEN, "forbidden_cols": forbidden_cols}, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.grid_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{**{f"cfg_{k}": v for k, v in r["config"].items()}, "score": r["selection_score"], "val_pnl": r["validation_cost1"]["pnl"], "val_mdd": r["validation_cost1"]["mdd"], "val_trades": r["validation_cost1"]["trades"], "val_deep_entries": r["validation_cost1"].get("deep_entries", 0), "val_c2_pnl": r["validation_cost2"]["pnl"], "val_c3_pnl": r["validation_cost3"]["pnl"]} for r in rows]).to_csv(args.grid_out, index=False)
    blocking: list[str] = []
    warnings: list[str] = []
    if feature_audit["status"] != "pass":
        blocking.extend(feature_audit["blocking"])
    if forbidden_cols:
        blocking.append(f"forbidden_sequence_columns={forbidden_cols}")
    warnings.extend(feature_audit.get("warnings", []))
    if metrics["cost1"]["pnl"] <= V21_2_COST1:
        warnings.append("oos_cost1_did_not_beat_v21_2")
    if metrics["cost2"]["pnl"] <= 0.0:
        warnings.append("cost2_not_survived")
    if metrics["cost3"]["pnl"] <= 0.0:
        warnings.append("cost3_not_survived")
    verdict = "promote" if not blocking and metrics["cost1"]["pnl"] > V21_2_COST1 and metrics["cost2"]["pnl"] > 0.0 and metrics["cost3"]["pnl"] > 0.0 else "iterate"
    audit = {"status": "pass" if not blocking else "fail", "verdict": verdict, "blocking": blocking, "warnings": warnings, "selection_uses_2026": False, "selection_window": "2025-10-01..2025-12-31", "oos_window": "2026 fixed OOS only after selection", "training_epochs": int(args.epochs), "policy": "deep_alpha_mamba_ssm_v41", "ssm_backend": "self_contained_mamba2_style_no_mamba_ssm_package", "v21_2_preserved": True, "deep_sleeve_only_when_parent_cash": True, "forbidden_sequence_columns": forbidden_cols, "train_snapshot_count": int(len(train_ds["target"])), "target_mean": np.mean(train_ds["target"], axis=0), "feature_audit": feature_audit, "selected_config": asdict(selected), "metrics": metrics}
    report = {"model_id": MODEL_ID, "design": "Deep Alpha Mamba-Style SSM V41. The V27 TCN sequence utility model is replaced with a self-contained gated diagonal SSM block inspired by Mamba-2 selective state-space modeling. V21.2 entries and jackpot add-ons are preserved; the deep sleeve may open extra trades only when the parent is CASH. The official mamba_ssm package is not installed in this environment, so this run is an SSM architecture comparison rather than an official Mamba-2 kernel run.", "parent_model": str(args.parent_model), "jackpot_model": str(args.jackpot_model), "model": str(model_path), "feature_manifest": str(manifest_path), "training_epochs": int(args.epochs), "split_policy": "Train 2025 Jan-Sep; select thresholds/notional on 2025 Oct-Dec; evaluate fixed 2026 OOS only after selection.", "selected_config": asdict(selected), "selection_result": best, "metrics": metrics, "audit": audit, "artifacts": {"model": str(model_path), "manifest": str(manifest_path), "report": str(args.report_out), "audit": str(args.audit_out), "grid": str(args.grid_out), "ledgers": ledgers}}
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "model": str(model_path), "selected": asdict(selected), "metrics": metrics, "verdict": verdict}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
