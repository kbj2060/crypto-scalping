#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, predict_policy_frame
from scripts import train_eval_hf_v13_deep_jackpot_sequence_verifier_v23 as v23
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _days, _fill_price, _read
from scripts.train_eval_hf_v13_convex_runner_pyramid_v18 import _feature_frame
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig, _predict_cost_runner


MODEL_ID = "hf_v13_rl_entry_addon_supervisor_v26_20260511"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_rl_entry_addon_supervisor_v26_20260511"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_rl_entry_addon_supervisor_v26_20260511_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_rl_entry_addon_supervisor_v26_20260511_audit.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/hf_v13_rl_entry_addon_supervisor_v26_20260511_grid.csv"
ACTION_MULTS = np.asarray([0.0, 0.70, 1.00, 1.20], dtype=np.float32)


class EntryAddonQ(nn.Module):
    def __init__(self, seq_dim: int, ctx_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv1d(seq_dim, hidden, 3, padding=2, dilation=2),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Conv1d(hidden, hidden, 3, padding=4, dilation=4),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.ctx = nn.Sequential(nn.Linear(ctx_dim, 64), nn.GELU())
        self.head = nn.Sequential(nn.Linear(hidden + 64, 64), nn.GELU(), nn.Linear(64, len(ACTION_MULTS)))

    def forward(self, seq: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        h = self.seq(seq.transpose(1, 2)).squeeze(-1)
        return self.head(torch.cat([h, self.ctx(ctx)], dim=1))


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


def _grid() -> list[v23.VerifierConfig]:
    return [
        v23.VerifierConfig("v26_support_bias_000", 0.000, 0.000, 0.0, -0.006),
        v23.VerifierConfig("v26_support_bias_003", 0.003, 0.000, 0.0, -0.006),
        v23.VerifierConfig("v26_no_skip_margin_002", 0.000, 0.002, 0.0, -0.006),
        v23.VerifierConfig("v26_aggressive_margin_004", 0.000, 0.004, 0.0, -0.006),
    ]


def _raw(side: int, entry: float, px: float) -> float:
    return (px - entry) / max(entry, 1e-12) if side > 0 else (entry - px) / max(entry, 1e-12)


def _parent_exit(frame: pd.DataFrame, close: np.ndarray, *, side: int, entry_idx: int, entry_price: float, notional: float, take_profit: float, stop_loss: float, max_hold: int, slip: float) -> tuple[int, float]:
    exit_i = min(entry_idx + max(max_hold, 1) + 1, len(frame) - 1)
    for j in range(entry_idx, min(entry_idx + max_hold + 1, len(frame) - 1)):
        px = float(close[j])
        mark = px * (1.0 - slip) if side > 0 else px * (1.0 + slip)
        unreal = _raw(side, entry_price, mark) * notional
        if (take_profit > 0.0 and unreal >= take_profit) or (stop_loss > 0.0 and unreal <= -abs(stop_loss)) or (max_hold > 0 and j - entry_idx >= max_hold):
            exit_i = min(j + 1, len(frame) - 1)
            break
    exit_px = _fill_price(frame, exit_i, side, slip, entry=False)
    return int(exit_i), float(_raw(side, entry_price, exit_px))


def _collect_dataset(frame: pd.DataFrame, bundle: dict[str, Any], seq_cols: list[str], *, fee: float, slip: float, max_entry_notional: float) -> dict[str, Any]:
    decisions = predict_policy_frame(bundle, frame, close=_close(frame))
    close = _close(frame)
    seqs: list[np.ndarray] = []
    ctxs: list[list[float]] = []
    rewards: list[list[float]] = []
    for i in range(0, len(frame) - 2):
        dec = decisions.iloc[i]
        if int(dec.action) == ACTION_CASH or int(dec.side) == 0:
            continue
        side = int(dec.side)
        fill_i = min(i + 1, len(frame) - 1)
        entry_price = _fill_price(frame, fill_i, side, slip, entry=True)
        base_notional = min(float(dec.notional_exposure), max_entry_notional)
        take_profit = float(dec.take_profit)
        stop_loss = float(dec.stop_loss)
        max_hold = int(dec.max_hold_bars)
        _, raw_exit = _parent_exit(frame, close, side=side, entry_idx=i, entry_price=entry_price, notional=base_notional, take_profit=take_profit, stop_loss=stop_loss, max_hold=max_hold, slip=slip)
        row = frame.iloc[i]
        ctx = [
            float(side),
            float(base_notional),
            float(getattr(dec, "leverage", 1.0)),
            float(getattr(dec, "confidence", 0.0)),
            float(getattr(dec, "quality_score", 0.0)),
            float(take_profit),
            float(stop_loss),
            float(max_hold),
            float(row.get("clean_regime_2024_unsup_v4_transition_risk", 0.0)),
            float(row.get("clean_regime_2024_unsup_v4_risk_off_prob", 0.0)),
            float(row.get("clean_regime_2024_unsup_v4_entropy", 0.0)),
            float(row.get("teacher_uncertainty", 0.0)),
            float(row.get("ai_adverse_risk", 0.0)),
            float(row.get("ai_reward_risk", 0.0)),
        ]
        rs = []
        for mult in ACTION_MULTS:
            n = float(base_notional * float(mult))
            if n <= 1e-12:
                rs.append(0.0)
            else:
                # one entry fee and one exit fee, plus a drawdown-sensitive penalty
                pnl = raw_exit * n - fee * n * 2.0
                rs.append(float(pnl - 0.10 * max(0.0, -pnl)))
        seqs.append(v23._seq_at(frame, i, seq_cols))
        ctxs.append(ctx)
        rewards.append(rs)
    if not rewards:
        raise RuntimeError("no entry candidates for RL supervisor")
    return {"seq": np.stack(seqs).astype(np.float32), "ctx": np.asarray(ctxs, dtype=np.float32), "reward": np.asarray(rewards, dtype=np.float32)}


def _norm(seq: np.ndarray, ctx: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "seq_mean": np.nanmean(seq, axis=(0, 1)).astype(np.float32),
        "seq_std": (np.nanstd(seq, axis=(0, 1)) + 1e-6).astype(np.float32),
        "ctx_mean": np.nanmean(ctx, axis=0).astype(np.float32),
        "ctx_std": (np.nanstd(ctx, axis=0) + 1e-6).astype(np.float32),
    }


def _apply(seq: np.ndarray, ctx: np.ndarray, norm: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    return ((seq - norm["seq_mean"][None, None, :]) / norm["seq_std"][None, None, :]).astype(np.float32), ((ctx - norm["ctx_mean"][None, :]) / norm["ctx_std"][None, :]).astype(np.float32)


def _train(ds: dict[str, Any], norm: dict[str, np.ndarray], *, epochs: int) -> EntryAddonQ:
    seq, ctx = _apply(ds["seq"], ds["ctx"], norm)
    reward = ds["reward"].astype(np.float32)
    model = EntryAddonQ(seq.shape[-1], ctx.shape[-1]).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    device = next(model.parameters()).device
    loader = DataLoader(TensorDataset(torch.from_numpy(seq), torch.from_numpy(ctx), torch.from_numpy(reward)), batch_size=128, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    for _ in range(epochs):
        for xb, cb, rb in loader:
            xb, cb, rb = xb.to(device), cb.to(device), rb.to(device)
            q = model(xb, cb)
            # Conservative residual regularization: stay close to behavior action
            # unless counterfactual reward clearly supports a change.
            loss = loss_fn(q, rb) + 0.01 * (torch.logsumexp(q, dim=1).mean() - q[:, 2].mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model.cpu().eval()


def _predict(model: EntryAddonQ, seq: np.ndarray, ctx: np.ndarray, norm: dict[str, np.ndarray]) -> np.ndarray:
    s, c = _apply(seq[None].astype(np.float32), ctx[None].astype(np.float32), norm)
    with torch.no_grad():
        return model(torch.from_numpy(s), torch.from_numpy(c))[0].numpy()


def backtest(df: pd.DataFrame, bundle: dict[str, Any], jackpot_model: dict[str, Any], model: EntryAddonQ, norm: dict[str, np.ndarray], add_cfg: CostRunnerConfig, cfg: v23.VerifierConfig, seq_cols: list[str], *, fee: float, slip: float, cost_mult: float = 1.0, decisions: pd.DataFrame | None = None, record: bool = False) -> dict[str, Any]:
    close = _close(df)
    if decisions is None:
        decisions = predict_policy_frame(bundle, df, close=close)
    fee_eff = fee * cost_mult
    slip_eff = slip * cost_mult
    cash = peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = entry_equity = 0.0
    entry_idx = 0
    parent_notional = notional = 0.0
    take_profit = stop_loss = 0.0
    max_hold = 0
    cooldown = next_cooldown = 0
    add_done = False
    mfe = mae = 0.0
    trades = wins = long_entries = short_entries = 0
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
                reason = "learned_take_profit"
            elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
                reason = "learned_stop_loss"
            elif max_hold > 0 and hold >= max_hold:
                reason = "learned_max_hold"
            if not reason and not add_done and add_cfg.full_add_frac > 0.0 and unreal >= add_cfg.min_unrealized and hold >= add_cfg.min_bars_since_entry and dd_abs <= add_cfg.dd_block:
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
                    actions["add_on"] = actions.get("add_on", 0) + 1
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
                cooldown = int(next_cooldown)
                next_cooldown = 0
                add_done = False
                open_record = None
                continue
        if pos != 0:
            continue
        if cooldown > 0:
            cooldown -= 1
            continue
        dec = decisions.iloc[i]
        if int(dec.action) == ACTION_CASH or int(dec.side) == 0:
            continue
        side = int(dec.side)
        base_notional = min(float(dec.notional_exposure), add_cfg.max_entry_notional)
        row = df.iloc[i]
        ctx = np.asarray([
            float(side),
            float(base_notional),
            float(dec.leverage),
            float(getattr(dec, "confidence", 0.0)),
            float(getattr(dec, "quality_score", 0.0)),
            float(dec.take_profit),
            float(dec.stop_loss),
            float(dec.max_hold_bars),
            float(row.get("clean_regime_2024_unsup_v4_transition_risk", 0.0)),
            float(row.get("clean_regime_2024_unsup_v4_risk_off_prob", 0.0)),
            float(row.get("clean_regime_2024_unsup_v4_entropy", 0.0)),
            float(row.get("teacher_uncertainty", 0.0)),
            float(row.get("ai_adverse_risk", 0.0)),
            float(row.get("ai_reward_risk", 0.0)),
        ], dtype=np.float32)
        q = _predict(model, v23._seq_at(df, i, seq_cols), ctx, norm)
        q = q.copy()
        q[2] += cfg.fragile_th
        q[3] -= cfg.edge_th
        action_idx = int(np.argmax(q))
        mult = float(ACTION_MULTS[action_idx])
        actions[f"entry_{action_idx}"] = actions.get(f"entry_{action_idx}", 0) + 1
        if mult <= 1e-12:
            cooldown = int(dec.cooldown_bars)
            continue
        fill_i = min(i + 1, len(df) - 1)
        pos = side
        entry_price = _fill_price(df, fill_i, pos, slip_eff, entry=True)
        entry_equity = cash
        entry_idx = i
        parent_notional = base_notional
        notional = base_notional * mult
        take_profit = float(dec.take_profit)
        stop_loss = float(dec.stop_loss)
        max_hold = int(dec.max_hold_bars)
        next_cooldown = int(dec.cooldown_bars)
        leverage = float(dec.leverage)
        cash -= cash * fee_eff * notional
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        notional_sum += notional
        leverage_sum += leverage
        mfe = mae = 0.0
        add_done = False
        if record:
            open_record = {"entry_signal_timestamp": str(df["timestamp"].iloc[i]), "entry_fill_timestamp": str(df["timestamp"].iloc[fill_i]), "side": "LONG" if pos > 0 else "SHORT", "entry_price": float(entry_price), "parent_notional_exposure": float(dec.notional_exposure), "notional_exposure": float(notional), "rl_entry_action": int(action_idx), "rl_entry_mult": float(mult), "leverage": float(leverage), "position_fraction": float(notional / max(leverage, 1e-12)), "take_profit": float(take_profit), "stop_loss": float(stop_loss), "max_hold_bars": int(max_hold), "fee_entry_pct": float(fee_eff * notional * 100.0)}
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
    out = {"pnl": float((cash - 1.0) * 100.0), "mdd": float(mdd * 100.0), "trades": int(trades), "wr": float(wins / max(trades, 1)), "trades_per_day": float(trades / _days(df)), "long_entries": int(long_entries), "short_entries": int(short_entries), "avg_notional": float(notional_sum / n), "avg_leverage": float(leverage_sum / n), "exits": exits, "runner_actions": actions}
    if record:
        out["trade_records"] = records
    return out


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1["trades"]) < 20:
        return -1e9 + float(c1["pnl"])
    return float(c1["pnl"] + 0.35 * float(c2["pnl"]) + 0.15 * float(c3["pnl"]) - 0.30 * abs(float(c1["mdd"])))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RL supervisor with larger role: entry sizing plus V21.2 add-on.")
    p.add_argument("--parent-model", type=Path, default=v23.DEFAULT_PARENT)
    p.add_argument("--jackpot-model", type=Path, default=v23.DEFAULT_JACKPOT)
    p.add_argument("--train-csv", type=Path, default=v23.DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=v23.DEFAULT_EVAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--grid-out", type=Path, default=DEFAULT_GRID)
    p.add_argument("--epochs", type=int, default=220)
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
    seq_cols = v23._select_seq_cols(train_all)
    feature_audit = _audit_contract(train_all, eval_df, list(bundle.get("feature_cols") or []))
    forbidden_cols = [c for c in seq_cols if any(tok in c.lower() for tok in v23.FORBIDDEN)]
    train_ds = _collect_dataset(train, bundle, seq_cols, fee=float(base["fee"]), slip=float(base["slip"]), max_entry_notional=add_cfg.max_entry_notional)
    norm = _norm(train_ds["seq"], train_ds["ctx"])
    model = _train(train_ds, norm, epochs=args.epochs)
    val_dec = predict_policy_frame(bundle, val, close=_close(val))
    eval_dec = predict_policy_frame(bundle, eval_df, close=_close(eval_df))
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for cfg in _grid():
        v1 = backtest(val, bundle, jackpot_model, model, norm, add_cfg, cfg, seq_cols, fee=float(base["fee"]), slip=float(base["slip"]), decisions=val_dec, cost_mult=1.0)
        v2 = backtest(val, bundle, jackpot_model, model, norm, add_cfg, cfg, seq_cols, fee=float(base["fee"]), slip=float(base["slip"]), decisions=val_dec, cost_mult=2.0)
        v3 = backtest(val, bundle, jackpot_model, model, norm, add_cfg, cfg, seq_cols, fee=float(base["fee"]), slip=float(base["slip"]), decisions=val_dec, cost_mult=3.0)
        row = {"config": asdict(cfg), "validation_cost1": v1, "validation_cost2": v2, "validation_cost3": v3, "selection_score": _score(v1, v2, v3)}
        rows.append(row)
        if best is None or row["selection_score"] > best["selection_score"]:
            best = row
    selected = v23.VerifierConfig(**best["config"])
    metrics: dict[str, Any] = {}
    ledgers: dict[str, str] = {}
    for mult in (1, 2, 3):
        r = backtest(eval_df, bundle, jackpot_model, model, norm, add_cfg, selected, seq_cols, fee=float(base["fee"]), slip=float(base["slip"]), decisions=eval_dec, cost_mult=float(mult), record=(mult == 1))
        if mult == 1:
            ledger = pd.DataFrame(r.pop("trade_records", []))
            lp = args.report_out.with_name(args.report_out.stem + "_cost1_ledger.csv")
            lp.parent.mkdir(parents=True, exist_ok=True)
            ledger.to_csv(lp, index=False)
            ledgers["cost1"] = str(lp)
        metrics[f"cost{mult}"] = r
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "v26_rl_entry_addon_supervisor.pt"
    torch.save({"model_id": MODEL_ID, "state_dict": model.state_dict(), "seq_cols": seq_cols, "ctx_cols": ["side", "base_notional", "leverage", "confidence", "quality", "take_profit", "stop_loss", "max_hold", "transition_risk", "risk_off_prob", "entropy", "teacher_uncertainty", "ai_adverse_risk", "ai_reward_risk"], "norm": norm, "selected_config": asdict(selected), "add_config": asdict(add_cfg)}, model_path)
    args.grid_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{**{f"cfg_{k}": v for k, v in r["config"].items()}, "score": r["selection_score"], "val_pnl": r["validation_cost1"]["pnl"], "val_mdd": r["validation_cost1"]["mdd"], "val_trades": r["validation_cost1"]["trades"], "val_c2_pnl": r["validation_cost2"]["pnl"], "val_c3_pnl": r["validation_cost3"]["pnl"], "val_actions": json.dumps(r["validation_cost1"].get("runner_actions", {}), ensure_ascii=False)} for r in rows]).to_csv(args.grid_out, index=False)
    blocking: list[str] = []
    warnings: list[str] = []
    if feature_audit["status"] != "pass":
        blocking.extend(feature_audit["blocking"])
    if forbidden_cols:
        blocking.append(f"forbidden_sequence_columns={forbidden_cols}")
    warnings.extend(feature_audit.get("warnings", []))
    if metrics["cost1"]["pnl"] <= v23.V21_2_COST1:
        warnings.append("oos_cost1_did_not_beat_v21_2")
    if metrics["cost2"]["pnl"] <= 0.0:
        warnings.append("cost2_not_survived")
    if metrics["cost3"]["pnl"] <= 0.0:
        warnings.append("cost3_not_survived")
    verdict = "promote" if not blocking and metrics["cost1"]["pnl"] > v23.V21_2_COST1 and metrics["cost2"]["pnl"] > 0.0 and metrics["cost3"]["pnl"] > 0.0 else "iterate"
    audit = {"status": "pass" if not blocking else "fail", "verdict": verdict, "blocking": blocking, "warnings": warnings, "selection_uses_2026": False, "selection_window": "2025-10-01..2025-12-31", "oos_window": "2026 fixed OOS only after selection", "policy": "rl_entry_sizing_plus_v21_2_addon", "allowed_actions": {"entry": ["skip", "0.70x", "1.00x", "1.20x"], "addon": "V21.2 jackpot runner unchanged"}, "rl_does_not_choose_side_or_exit": True, "forbidden_sequence_columns": forbidden_cols, "train_snapshot_count": int(len(train_ds["reward"])), "reward_mean_by_action": np.mean(train_ds["reward"], axis=0), "feature_audit": feature_audit, "selected_config": asdict(selected), "metrics": metrics}
    report = {"model_id": MODEL_ID, "design": "RL role expanded from add-on sleeve only to entry exposure supervisor plus V21.2 jackpot add-on. Parent model still owns side and exit; RL can skip, downsize, keep, or upsize parent entry exposure.", "parent_model": str(args.parent_model), "jackpot_model": str(args.jackpot_model), "model": str(model_path), "split_policy": "Train 2025 Jan-Sep; select on 2025 Oct-Dec; evaluate fixed 2026 OOS only after selection.", "selected_config": asdict(selected), "selection_result": best, "metrics": metrics, "audit": audit, "artifacts": {"model": str(model_path), "report": str(args.report_out), "audit": str(args.audit_out), "grid": str(args.grid_out), "ledgers": ledgers}}
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "model": str(model_path), "selected": asdict(selected), "metrics": metrics, "verdict": verdict}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
