#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_1_exit_only_rl_editor_20260610 as ex  # noqa: E402
import train_eval_omega1_2_1_exit_q_market_features_20260612 as mkt  # noqa: E402


MODEL_ID = "omega1_2_1_exit_q_no_tighten_20260612"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

ACTION_MAP = (ex.HOLD, ex.REDUCE50, ex.FULL_EXIT)
ACTION_NAMES = ("hold", "partial50", "close")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _conservative_rewards(raw: np.ndarray, *, buffer: float, action_penalty: float, close_penalty: float) -> np.ndarray:
    out = raw[:, list(ACTION_MAP)].copy().astype(np.float32)
    hold = out[:, [0]]
    for idx in (1, 2):
        penalty = float(close_penalty if idx == 2 else action_penalty)
        weak = out[:, [idx]] <= hold + float(buffer)
        out[:, [idx]] = out[:, [idx]] - penalty
        out[:, [idx]] = np.where(weak, hold - float(buffer), out[:, [idx]])
    return out.astype(np.float32)


def _collect_dataset_no_tighten(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    state: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    stride: int,
    max_states: int,
    max_forward_bars: int,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    arrays = ex._arrays(frame)
    active = np.asarray(ex.omega._active(dec), dtype=bool)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    cash = 1.0
    pos = ex.Position()
    rows: list[pd.DataFrame] = []
    rewards: list[np.ndarray] = []
    entries = 0
    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            unreal = ex._unreal(arrays, pos, i, slip_eff)
            pos.mfe = max(pos.mfe, unreal)
            pos.mae = min(pos.mae, unreal)
            reason = ex._hit_reason(unreal, pos)
            if reason:
                cash, pos, _ = ex._close_fraction(cash, arrays, pos, i, 1.0, fee_eff, slip_eff)
                continue
            hold = int(i) - int(pos.entry_i)
            sample = (
                hold >= 1
                and (
                    hold % int(stride) == 0
                    or (pos.take_profit > 0.0 and unreal >= 0.45 * pos.take_profit)
                    or (pos.mfe > 0.0 and (pos.mfe - unreal) / max(abs(pos.mfe), 1e-8) > 0.35)
                    or (pos.stop_loss > 0.0 and unreal <= -0.55 * pos.stop_loss)
                )
            )
            if sample:
                rows.append(mkt._pos_features_market(state, pos, unreal, i))
                rewards.append(
                    np.asarray(
                        [
                            ex._simulate_first_action(cash, arrays, pos, i, action, fee_eff=fee_eff, slip_eff=slip_eff, max_forward_bars=max_forward_bars)
                            for action in range(len(ex.ACTION_NAMES))
                        ],
                        dtype=np.float32,
                    )
                )
                if len(rows) >= int(max_states):
                    break
            continue
        if bool(active[i]):
            cash, pos, entered = ex._enter(cash, arrays, dec, i, fee_eff, slip_eff)
            entries += int(entered)
    if not rows:
        raise RuntimeError("empty no-tighten exit dataset")
    x = pd.concat(rows, ignore_index=True)
    r = np.stack(rewards, axis=0).astype(np.float32)
    diag = {
        "states": int(len(x)),
        "entries_seen": int(entries),
        "raw_best_counts": {ex.ACTION_NAMES[i]: int(np.sum(np.argmax(r, axis=1) == i)) for i in range(len(ex.ACTION_NAMES))},
        "raw_mean_reward_by_action": {ex.ACTION_NAMES[i]: float(np.mean(r[:, i])) for i in range(len(ex.ACTION_NAMES))},
    }
    ex._reject_forbidden(list(x.columns), "exit_q_no_tighten_dataset")
    return x, r, diag


@torch.no_grad()
def _action(model: ex.QNet, x: pd.DataFrame, *, min_adv: float, partial_tp_min: float, close_giveback_min: float, close_sl_min: float) -> int:
    device = next(model.parameters()).device
    q = model(torch.from_numpy(ex._apply_norm(x, model.norm)).to(device))[0].detach().cpu().numpy().astype(np.float64)  # type: ignore[attr-defined]
    row = x.iloc[0]
    unreal = float(row.get("pos_unrealized", 0.0))
    tp_progress = float(row.get("pos_tp_progress", 0.0))
    giveback = float(row.get("pos_giveback", 0.0))
    sl_progress = float(row.get("pos_sl_progress", 0.0))
    allowed = np.zeros_like(q, dtype=bool)
    allowed[0] = True
    allowed[1] = bool(unreal > 0.004 and tp_progress >= float(partial_tp_min))
    allowed[2] = bool((unreal > 0.0 and giveback >= float(close_giveback_min) and tp_progress >= 0.35) or (unreal < 0.0 and sl_progress >= float(close_sl_min)))
    q[~allowed] = -1e9
    best = int(np.argmax(q))
    if best == 0:
        return ex.HOLD
    if float(q[best] - q[0]) < float(min_adv):
        return ex.HOLD
    return ACTION_MAP[best]


def _ledger_row(frame: pd.DataFrame, arrays: dict[str, np.ndarray], pos: ex.Position, exit_i: int, cash: float, net_pct: float, reason: str) -> dict[str, Any]:
    return {
        "side": "LONG" if pos.side > 0 else "SHORT",
        "entry_signal_i": int(pos.entry_signal_i),
        "entry_i": int(pos.entry_i),
        "exit_i": int(exit_i),
        "entry_time": str(frame["timestamp"].iloc[int(pos.entry_signal_i)]),
        "exit_time": str(frame["timestamp"].iloc[int(exit_i)]),
        "entry_price": float(pos.entry_price),
        "exit_price": float(arrays["close"][int(exit_i)]),
        "effective_exposure": float(pos.notional),
        "margin_notional": float(pos.margin_notional),
        "leverage": float(pos.leverage),
        "tp_equity_ret": float(pos.take_profit),
        "sl_equity_ret": float(pos.stop_loss),
        "net_trade_return_pct": float(net_pct),
        "mfe_pct": float(pos.mfe * 100.0),
        "mae_pct": float(pos.mae * 100.0),
        "exit_reason": str(reason),
        "cash_after": float(cash),
    }


def _simulate(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    state: pd.DataFrame,
    *,
    model: ex.QNet | None,
    min_adv: float,
    partial_tp_min: float,
    close_giveback_min: float,
    close_sl_min: float,
    fee: float,
    slip: float,
    cost_mult: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    arrays = ex._arrays(frame)
    active = np.asarray(ex.omega._active(dec), dtype=bool)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = ex.Position()
    trades: list[float] = []
    rows: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    action_counts = {k: 0 for k in ACTION_NAMES}
    long_entries = short_entries = 0
    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            unreal = ex._unreal(arrays, pos, i, slip_eff)
            pos.mfe = max(pos.mfe, unreal)
            pos.mae = min(pos.mae, unreal)
            eq = cash * (1.0 + unreal)
            peak = max(peak, eq)
            mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
            reason = ex._hit_reason(unreal, pos)
            if not reason and model is not None:
                act = _action(
                    model,
                    mkt._pos_features_market(state, pos, unreal, i),
                    min_adv=float(min_adv),
                    partial_tp_min=float(partial_tp_min),
                    close_giveback_min=float(close_giveback_min),
                    close_sl_min=float(close_sl_min),
                )
                before = ex.Position(**pos.__dict__)
                cash, pos, name = ex._apply_action(cash, arrays, pos, i, act, unreal, fee_eff, slip_eff)
                if act == ex.REDUCE50:
                    action_counts["partial50"] += 1
                elif act == ex.FULL_EXIT:
                    action_counts["close"] += 1
                else:
                    action_counts["hold"] += 1
                if before.side != 0 and pos.side == 0:
                    reason = "learned_close"
                    net_pct = float((cash / max(before.entry_equity, 1e-12) - 1.0) * 100.0)
                    trades.append(net_pct)
                    reasons[reason] = reasons.get(reason, 0) + 1
                    rows.append(_ledger_row(frame, arrays, before, i, cash, net_pct, reason))
                    continue
            if reason:
                close_pos = ex.Position(**pos.__dict__)
                cash, pos, _ = ex._close_fraction(cash, arrays, close_pos, i, 1.0, fee_eff, slip_eff)
                net_pct = float((cash / max(close_pos.entry_equity, 1e-12) - 1.0) * 100.0)
                trades.append(net_pct)
                reasons[reason] = reasons.get(reason, 0) + 1
                rows.append(_ledger_row(frame, arrays, close_pos, i, cash, net_pct, reason))
            continue
        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1e-12) - 1.0)
        if not bool(active[i]):
            continue
        side = int(dec.iloc[int(i)].get("side", 0) or 0)
        cash, pos, entered = ex._enter(cash, arrays, dec, i, fee_eff, slip_eff)
        if entered:
            long_entries += int(side > 0)
            short_entries += int(side < 0)
    if pos.side != 0:
        close_pos = ex.Position(**pos.__dict__)
        cash, pos, _ = ex._close_fraction(cash, arrays, close_pos, len(frame) - 1, 1.0, fee_eff, slip_eff)
        net_pct = float((cash / max(close_pos.entry_equity, 1e-12) - 1.0) * 100.0)
        trades.append(net_pct)
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
        rows.append(_ledger_row(frame, arrays, close_pos, len(frame) - 1, cash, net_pct, "forced_end"))
    arr = np.asarray(trades, dtype=np.float64)
    metrics = {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(len(trades)),
        "wr": float(np.mean(arr > 0.0)) if len(arr) else 0.0,
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "exit_reasons": reasons,
        "adapter_actions": action_counts,
    }
    return metrics, pd.DataFrame(rows)


def _row(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_pnl": float(metrics["pnl"]),
        f"{prefix}_mdd": float(metrics["mdd"]),
        f"{prefix}_wr": float(metrics["wr"]),
        f"{prefix}_trades": int(metrics["trades"]),
        f"{prefix}_long": int(metrics["long_entries"]),
        f"{prefix}_short": int(metrics["short_entries"]),
        f"{prefix}_reasons": metrics["exit_reasons"],
        f"{prefix}_adapter_actions": metrics.get("adapter_actions", {}),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=700)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--max-states", type=int, default=4200)
    ap.add_argument("--max-forward-bars", type=int, default=432)
    ap.add_argument("--buffer", type=float, default=0.008)
    ap.add_argument("--action-penalty", type=float, default=0.002)
    ap.add_argument("--close-penalty", type=float, default=0.004)
    ap.add_argument("--cql-weight", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=260612)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fee, slip = ex.omega._load_fee_slip()
    splits = ex._build_splits()
    built = {}
    for split, payload in splits.items():
        dec = ex._to_decisions(payload["src"], payload["prefix"], oof=payload["oof"], thresholds=ex.HIGH_THRESHOLDS)
        state = mkt._state_base_market(payload["frame"], payload["src"], dec, payload["prefix"])
        built[split] = {"frame": payload["frame"], "dec": dec, "state": state}
    x_train, raw_rewards, data_diag = _collect_dataset_no_tighten(
        built["validation"]["frame"],
        built["validation"]["dec"],
        built["validation"]["state"],
        fee=fee,
        slip=slip,
        cost_mult=3.0,
        stride=int(args.stride),
        max_states=int(args.max_states),
        max_forward_bars=int(args.max_forward_bars),
    )
    rewards = _conservative_rewards(raw_rewards, buffer=float(args.buffer), action_penalty=float(args.action_penalty), close_penalty=float(args.close_penalty))
    model, train_diag = ex._train_q(x_train, rewards, epochs=int(args.epochs), seed=int(args.seed), cql_weight=float(args.cql_weight))
    model_path = OUT_DIR / "exit_q_no_tighten.pt"
    torch.save({"state_dict": model.state_dict(), "norm": model.norm, "actions": ACTION_NAMES, "train_diag": train_diag, "config": vars(args)}, model_path)  # type: ignore[attr-defined]
    x_train.to_csv(OUT_DIR / "train_states.csv", index=False)
    pd.DataFrame(raw_rewards, columns=[f"raw_reward_{a}" for a in ex.ACTION_NAMES]).to_csv(OUT_DIR / "train_raw_rewards.csv", index=False)
    pd.DataFrame(rewards, columns=[f"reward_{a}" for a in ACTION_NAMES]).to_csv(OUT_DIR / "train_no_tighten_rewards.csv", index=False)
    rows = []
    val_base, val_base_ledger = _simulate(built["validation"]["frame"], built["validation"]["dec"], built["validation"]["state"], model=None, min_adv=999, partial_tp_min=999, close_giveback_min=999, close_sl_min=999, fee=fee, slip=slip, cost_mult=3.0)
    oos_base, oos_base_ledger = _simulate(built["oos"]["frame"], built["oos"]["dec"], built["oos"]["state"], model=None, min_adv=999, partial_tp_min=999, close_giveback_min=999, close_sl_min=999, fee=fee, slip=slip, cost_mult=3.0)
    val_base_ledger.to_csv(OUT_DIR / "validation_baseline_ledger.csv", index=False)
    oos_base_ledger.to_csv(OUT_DIR / "oos_baseline_ledger.csv", index=False)
    rows.append({"variant": "baseline", **_row("val", val_base), **_row("oos", oos_base)})
    configs = [
        (0.005, 0.55, 0.60, 0.80),
        (0.010, 0.60, 0.65, 0.85),
        (0.020, 0.65, 0.70, 0.90),
        (0.035, 0.70, 0.75, 0.95),
    ]
    for min_adv, partial_tp, close_gb, close_sl in configs:
        val_m, val_ledger = _simulate(built["validation"]["frame"], built["validation"]["dec"], built["validation"]["state"], model=model, min_adv=min_adv, partial_tp_min=partial_tp, close_giveback_min=close_gb, close_sl_min=close_sl, fee=fee, slip=slip, cost_mult=3.0)
        oos_m, oos_ledger = _simulate(built["oos"]["frame"], built["oos"]["dec"], built["oos"]["state"], model=model, min_adv=min_adv, partial_tp_min=partial_tp, close_giveback_min=close_gb, close_sl_min=close_sl, fee=fee, slip=slip, cost_mult=3.0)
        tag = f"adv{str(min_adv).replace('.', 'p')}_ptp{str(partial_tp).replace('.', 'p')}_cgb{str(close_gb).replace('.', 'p')}"
        val_ledger.to_csv(OUT_DIR / f"validation_{tag}_ledger.csv", index=False)
        oos_ledger.to_csv(OUT_DIR / f"oos_{tag}_ledger.csv", index=False)
        rows.append({"variant": tag, "min_adv": min_adv, "partial_tp_min": partial_tp, "close_giveback_min": close_gb, "close_sl_min": close_sl, **_row("val", val_m), **_row("oos", oos_m), "val_entry_audit": ex._entry_audit(val_base_ledger, val_ledger), "oos_entry_audit": ex._entry_audit(oos_base_ledger, oos_ledger)})
    ranking = pd.DataFrame(rows)
    base = ranking[ranking["variant"].eq("baseline")].iloc[0]
    ranking["delta_oos_pnl"] = ranking["oos_pnl"] - float(base["oos_pnl"])
    ranking["delta_val_pnl"] = ranking["val_pnl"] - float(base["val_pnl"])
    ranking["score"] = ranking["oos_pnl"] + 0.45 * ranking["val_pnl"] + 0.35 * ranking["oos_mdd"] + 0.25 * ranking["val_mdd"]
    ranking = ranking.sort_values(["oos_pnl", "val_pnl"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "exit_q_no_tighten_ranking.csv", index=False)
    promotable = ranking[(ranking["variant"] != "baseline") & (ranking["oos_pnl"] > float(base["oos_pnl"])) & (ranking["val_pnl"] > float(base["val_pnl"]) * 0.85) & (ranking["oos_mdd"] >= float(base["oos_mdd"]) * 1.20)].copy()
    promotable.to_csv(OUT_DIR / "exit_q_no_tighten_promotable.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "purpose": "Exit-Q without learned tighten_sl; actions are hold, partial50, close.",
        "baseline": base.to_dict(),
        "dataset": data_diag,
        "training": train_diag,
        "promotable_count": int(len(promotable)),
        "top": ranking.to_dict(orient="records"),
        "artifacts": {"out_dir": str(OUT_DIR), "model": str(model_path), "ranking": str(OUT_DIR / "exit_q_no_tighten_ranking.csv"), "report": str(OUT_DIR / "report.json")},
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "promotable_count": int(len(promotable)), "top": ranking.to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
