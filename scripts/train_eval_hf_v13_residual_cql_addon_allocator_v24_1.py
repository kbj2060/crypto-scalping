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
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _fill_price, _read
from scripts.train_eval_hf_v13_convex_runner_pyramid_v18 import _feature_frame
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig, _addon_utility, _predict_cost_runner


MODEL_ID = "hf_v13_residual_cql_addon_allocator_v24_1_20260511"
DEFAULT_OUT_DIR = v23.ROOT / "data/ensemble/supervised/hf_v13_residual_cql_addon_allocator_v24_1_20260511"
DEFAULT_REPORT = v23.ROOT / "data/ensemble/reports/hf_v13_residual_cql_addon_allocator_v24_1_20260511_summary.json"
DEFAULT_AUDIT = v23.ROOT / "data/ensemble/reports/hf_v13_residual_cql_addon_allocator_v24_1_20260511_audit.json"
DEFAULT_GRID = v23.ROOT / "data/ensemble/reports/hf_v13_residual_cql_addon_allocator_v24_1_20260511_grid.csv"


class ResidualQAllocator(nn.Module):
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
        self.ctx = nn.Sequential(nn.Linear(ctx_dim, 48), nn.GELU())
        self.head = nn.Sequential(nn.Linear(hidden + 48, 64), nn.GELU(), nn.Linear(64, 3))

    def forward(self, seq: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.seq(seq.transpose(1, 2)).squeeze(-1), self.ctx(ctx)], dim=1))


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
        v23.VerifierConfig("v24_1_residual_full_bias_000", 0.0, 0.00, 0.10, -0.006),
        v23.VerifierConfig("v24_1_residual_full_bias_002", 0.0, 0.002, 0.10, -0.006),
        v23.VerifierConfig("v24_1_residual_full_bias_004", 0.0, 0.004, 0.10, -0.006),
        v23.VerifierConfig("v24_1_residual_reject_margin_002", 0.002, 0.002, 0.10, -0.006),
    ]


def _collect_bandit(
    frame: pd.DataFrame,
    bundle: dict[str, Any],
    jackpot_model: dict[str, Any],
    add_cfg: CostRunnerConfig,
    seq_cols: list[str],
    *,
    fee: float,
    slip: float,
) -> dict[str, Any]:
    decisions = predict_policy_frame(bundle, frame, close=_close(frame))
    close = _close(frame)
    seqs: list[np.ndarray] = []
    ctxs: list[list[float]] = []
    rewards: list[list[float]] = []
    pos = 0
    entry_price = 0.0
    entry_idx = 0
    parent_notional = notional = 0.0
    take_profit = stop_loss = 0.0
    max_hold = 0
    cash = peak = 1.0
    mfe = mae = 0.0
    add_done = False
    for i in range(0, len(frame) - 2):
        if pos != 0:
            px = float(close[i])
            raw = (px * (1.0 - slip) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip)) / max(entry_price, 1e-12)
            unreal = raw * notional
            mfe = max(mfe, unreal)
            mae = min(mae, unreal)
            eq = cash * (1.0 + unreal)
            peak = max(peak, eq)
            hold = i - entry_idx
            state = {"parent_notional": parent_notional, "notional": notional, "bars_since_entry": hold, "unrealized": unreal, "mfe": mfe, "mae": mae, "drawdown_abs": max(0.0, 1.0 - eq / max(peak, 1e-12)), "take_profit": take_profit, "stop_loss": stop_loss, "max_hold": max_hold}
            reason = ""
            if take_profit > 0.0 and unreal >= take_profit:
                reason = "tp"
            elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
                reason = "sl"
            elif max_hold > 0 and hold >= max_hold:
                reason = "hold"
            if not reason and not add_done and add_cfg.full_add_frac > 0.0 and unreal >= add_cfg.min_unrealized and hold >= add_cfg.min_bars_since_entry and state["drawdown_abs"] <= add_cfg.dd_block:
                x = _feature_frame(frame, bundle, decisions, i, state)
                edge, p, q10, q90, p_jackpot, p_bad, p_cost3 = _predict_cost_runner(jackpot_model, x)
                is_jackpot = p_jackpot >= add_cfg.jackpot_p and q90 >= add_cfg.jackpot_q90 and p_bad <= add_cfg.bad_cap and p_cost3 >= 0.40
                if is_jackpot:
                    action_rewards = [0.0]
                    for frac in (0.10, 0.20):
                        vals = []
                        for mult, w in ((1.0, 1.0), (2.0, 0.45), (3.0, 0.25)):
                            u = _addon_utility(frame, close, pos=pos, entry_idx=entry_idx, snapshot_idx=i, entry_price=entry_price, current_notional=notional, parent_notional=parent_notional, take_profit=take_profit, stop_loss=stop_loss, max_hold=max_hold, add_frac=frac, fee=fee, slip=slip, cost_mult=mult)
                            vals.append((u, w))
                        # Residual safe RL reward: keep cost1/2 alpha, penalize cost3 tail.
                        r = sum(u * w for u, w in vals) - 1.4 * max(0.0, -vals[2][0])
                        action_rewards.append(float(r))
                    ctx = [float(pos), float(parent_notional), float(notional), float(hold), float(unreal), float(mfe), float(mae), float(mfe - unreal), float(unreal - mae), float(state["drawdown_abs"]), float(take_profit), float(stop_loss), float(max_hold), float(edge), float(p), float(q10), float(q90), float(p_jackpot), float(p_bad), float(p_cost3)]
                    seqs.append(v23._seq_at(frame, i, seq_cols))
                    ctxs.append(ctx)
                    rewards.append(action_rewards)
                add_done = True
            if reason:
                exit_i = min(i + 1, len(frame) - 1)
                exit_px = _fill_price(frame, exit_i, pos, slip, entry=False)
                raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * fee * notional
                pos = 0
                add_done = False
                continue
        if pos == 0:
            dec = decisions.iloc[i]
            if int(dec.action) == ACTION_CASH or int(dec.side) == 0:
                continue
            fill_i = min(i + 1, len(frame) - 1)
            pos = int(dec.side)
            entry_price = _fill_price(frame, fill_i, pos, slip, entry=True)
            entry_idx = i
            parent_notional = min(float(dec.notional_exposure), add_cfg.max_entry_notional)
            notional = parent_notional
            take_profit = float(dec.take_profit)
            stop_loss = float(dec.stop_loss)
            max_hold = int(dec.max_hold_bars)
            cash -= cash * fee * notional
            mfe = mae = 0.0
            add_done = False
    if not rewards:
        raise RuntimeError("no bandit add-on candidates")
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


def _train(ds: dict[str, Any], norm: dict[str, np.ndarray], *, epochs: int) -> ResidualQAllocator:
    seq, ctx = _apply(ds["seq"], ds["ctx"], norm)
    reward = ds["reward"].astype(np.float32)
    model = ResidualQAllocator(seq.shape[-1], ctx.shape[-1]).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    device = next(model.parameters()).device
    loader = DataLoader(TensorDataset(torch.from_numpy(seq), torch.from_numpy(ctx), torch.from_numpy(reward)), batch_size=64, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    for _ in range(epochs):
        for xb, cb, rb in loader:
            xb, cb, rb = xb.to(device), cb.to(device), rb.to(device)
            q = model(xb, cb)
            bellman = loss_fn(q, rb)
            # Conservative regularization: do not invent out-of-support action advantages.
            cql = torch.logsumexp(q, dim=1).mean() - q[:, 2].mean()
            loss = bellman + 0.015 * cql
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model.cpu().eval()


def _make_predict(model_ref: ResidualQAllocator):
    def predict(_: Any, seq: np.ndarray, ctx: np.ndarray, norm: dict[str, np.ndarray]) -> dict[str, float]:
        s, c = _apply(seq[None, ...].astype(np.float32), ctx[None, ...].astype(np.float32), norm)
        with torch.no_grad():
            q = model_ref(torch.from_numpy(s), torch.from_numpy(c))[0].numpy()
        return {"q_reject": float(q[0]), "q_reduce": float(q[1]), "q_full": float(q[2])}

    return predict


def _action(pred: dict[str, float], cfg: v23.VerifierConfig) -> tuple[str, float]:
    q = np.asarray([pred["q_reject"], pred["q_reduce"], pred["q_full"] + cfg.edge_th], dtype=np.float64)
    if q[0] > max(q[1], q[2]) + cfg.fragile_th:
        return "reject", 0.0
    if q[1] > q[2]:
        return "reduce", cfg.reduce_frac
    return "full", 0.20


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1["trades"]) < 20:
        return -1e9 + float(c1["pnl"])
    return float(float(c1["pnl"]) + 0.45 * float(c2["pnl"]) + 0.25 * float(c3["pnl"]) - 0.20 * abs(float(c1["mdd"])))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Residual conservative Q allocator for V21.2 add-on sizing.")
    p.add_argument("--parent-model", type=Path, default=v23.DEFAULT_PARENT)
    p.add_argument("--jackpot-model", type=Path, default=v23.DEFAULT_JACKPOT)
    p.add_argument("--train-csv", type=Path, default=v23.DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=v23.DEFAULT_EVAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--grid-out", type=Path, default=DEFAULT_GRID)
    p.add_argument("--epochs", type=int, default=240)
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
    train_ds = _collect_bandit(train, bundle, jackpot_model, add_cfg, seq_cols, fee=float(base["fee"]), slip=float(base["slip"]))
    norm = _norm(train_ds["seq"], train_ds["ctx"])
    allocator = _train(train_ds, norm, epochs=args.epochs)
    val_dec = predict_policy_frame(bundle, val, close=_close(val))
    eval_dec = predict_policy_frame(bundle, eval_df, close=_close(eval_df))
    old_predict = v23._predict_one
    old_action = v23._verifier_action
    v23._predict_one = _make_predict(allocator)
    v23._verifier_action = _action
    try:
        rows: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        for cfg in _grid():
            v1 = v23.backtest(val, bundle, jackpot_model, allocator, norm, add_cfg, cfg, seq_cols, fee=float(base["fee"]), slip=float(base["slip"]), decisions=val_dec, cost_mult=1.0)
            v2 = v23.backtest(val, bundle, jackpot_model, allocator, norm, add_cfg, cfg, seq_cols, fee=float(base["fee"]), slip=float(base["slip"]), decisions=val_dec, cost_mult=2.0)
            v3 = v23.backtest(val, bundle, jackpot_model, allocator, norm, add_cfg, cfg, seq_cols, fee=float(base["fee"]), slip=float(base["slip"]), decisions=val_dec, cost_mult=3.0)
            row = {"config": asdict(cfg), "validation_cost1": v1, "validation_cost2": v2, "validation_cost3": v3, "selection_score": _score(v1, v2, v3)}
            rows.append(row)
            if best is None or row["selection_score"] > best["selection_score"]:
                best = row
        selected = v23.VerifierConfig(**best["config"])
        metrics: dict[str, Any] = {}
        ledgers: dict[str, str] = {}
        for mult in (1, 2, 3):
            r = v23.backtest(eval_df, bundle, jackpot_model, allocator, norm, add_cfg, selected, seq_cols, fee=float(base["fee"]), slip=float(base["slip"]), decisions=eval_dec, cost_mult=float(mult), record=(mult == 1))
            if mult == 1:
                ledger = pd.DataFrame(r.pop("trade_records", []))
                lp = args.report_out.with_name(args.report_out.stem + "_cost1_ledger.csv")
                lp.parent.mkdir(parents=True, exist_ok=True)
                ledger.to_csv(lp, index=False)
                ledgers["cost1"] = str(lp)
            metrics[f"cost{mult}"] = r
    finally:
        v23._predict_one = old_predict
        v23._verifier_action = old_action
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "v24_1_residual_cql_addon_allocator.pt"
    torch.save({"model_id": MODEL_ID, "state_dict": allocator.state_dict(), "seq_cols": seq_cols, "ctx_cols": v23.CTX_COLS, "norm": norm, "selected_config": asdict(selected), "add_config": asdict(add_cfg)}, model_path)
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
    audit = {"status": "pass" if not blocking else "fail", "verdict": verdict, "blocking": blocking, "warnings": warnings, "selection_uses_2026": False, "selection_window": "2025-10-01..2025-12-31", "oos_window": "2026 fixed OOS only after selection", "policy": "residual_conservative_q_addon_allocator", "research_basis": ["safe offline RL / constrained policy selection", "conservative Q regularization", "residual RL on top of a strong behavior policy"], "forbidden_sequence_columns": forbidden_cols, "train_snapshot_count": int(len(train_ds["reward"])), "reward_mean_by_action": np.mean(train_ds["reward"], axis=0), "feature_audit": feature_audit, "selected_config": asdict(selected), "metrics": metrics}
    report = {"model_id": MODEL_ID, "design": "Residual conservative Q allocator. It does not own entry/exit; it learns only reject/0.10/0.20 add-on sizing for audited V21.2 jackpot candidates using counterfactual cost1/2/3 rewards.", "parent_model": str(args.parent_model), "jackpot_model": str(args.jackpot_model), "model": str(model_path), "split_policy": "Train on 2025 Jan-Sep; select residual/full-bias grid on 2025 Oct-Dec; evaluate fixed 2026 OOS after selection only.", "selected_config": asdict(selected), "selection_result": best, "metrics": metrics, "audit": audit, "artifacts": {"model": str(model_path), "report": str(args.report_out), "audit": str(args.audit_out), "grid": str(args.grid_out), "ledgers": ledgers}}
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "model": str(model_path), "selected": asdict(selected), "metrics": metrics, "verdict": verdict}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
