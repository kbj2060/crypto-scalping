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


MODEL_ID = "omega1_2_1_exit_q_market_conservative_20260612"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID


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


def _conservative_rewards(rewards: np.ndarray, *, train_buffer: float, mask_penalty: float, action_penalty: float) -> np.ndarray:
    out = rewards.copy().astype(np.float32)
    hold = out[:, [ex.HOLD]]
    for action in (ex.TIGHTEN_SL, ex.REDUCE50, ex.FULL_EXIT):
        raw = out[:, [action]]
        weak = raw <= hold + float(train_buffer)
        out[:, [action]] = raw - float(action_penalty)
        out[:, [action]] = np.where(weak, hold - float(mask_penalty), out[:, [action]])
    return out.astype(np.float32)


@torch.no_grad()
def _masked_q_action(
    model: ex.QNet,
    x: pd.DataFrame,
    *,
    min_adv: float,
    giveback_min: float,
    tp_progress_min: float,
    sl_progress_min: float,
) -> int:
    row = x.iloc[0]
    device = next(model.parameters()).device
    arr = torch.from_numpy(ex._apply_norm(x, model.norm)).to(device)  # type: ignore[attr-defined]
    q = model(arr)[0].detach().cpu().numpy().astype(np.float64)
    hold_q = float(q[ex.HOLD])
    unreal = float(row.get("pos_unrealized", 0.0))
    mfe = float(row.get("pos_mfe", 0.0))
    giveback = float(row.get("pos_giveback", 0.0))
    tp_progress = float(row.get("pos_tp_progress", 0.0))
    sl_progress = float(row.get("pos_sl_progress", 0.0))

    allowed = np.zeros_like(q, dtype=bool)
    allowed[ex.HOLD] = True

    # Tighten only when the trade has something to protect or is clearly near adverse risk.
    protect_profit = unreal > 0.001 and mfe > 0.004 and (giveback >= float(giveback_min) or tp_progress >= float(tp_progress_min))
    protect_loss = unreal < 0.0 and sl_progress >= float(sl_progress_min)
    allowed[ex.TIGHTEN_SL] = bool(protect_profit or protect_loss)

    # Partial is only for profitable deep-progress trades; no loss-side partials.
    allowed[ex.REDUCE50] = bool(unreal > 0.004 and tp_progress >= max(float(tp_progress_min), 0.55))

    # Full exit remains disabled in this experiment. It was too destructive in prior runs.
    allowed[ex.FULL_EXIT] = False
    q[~allowed] = -1e9
    best = int(np.argmax(q))
    if best == ex.HOLD:
        return ex.HOLD
    if float(q[best] - hold_q) < float(min_adv):
        return ex.HOLD
    return best


def _simulate_policy_masked(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    state: pd.DataFrame,
    *,
    model: ex.QNet | None,
    min_adv: float,
    fee: float,
    slip: float,
    cost_mult: float,
    giveback_min: float,
    tp_progress_min: float,
    sl_progress_min: float,
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
    action_counts = {name: 0 for name in ex.ACTION_NAMES}
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
                features = mkt._pos_features_market(state, pos, unreal, i)
                action = _masked_q_action(
                    model,
                    features,
                    min_adv=float(min_adv),
                    giveback_min=float(giveback_min),
                    tp_progress_min=float(tp_progress_min),
                    sl_progress_min=float(sl_progress_min),
                )
                before_pos = ex.Position(**pos.__dict__)
                cash, pos, action_name = ex._apply_action(cash, arrays, pos, i, action, unreal, fee_eff, slip_eff)
                action_counts[action_name] = action_counts.get(action_name, 0) + 1
                if before_pos.side != 0 and pos.side == 0:
                    reason = "masked_full_exit"
                    net_pct = float((cash / max(before_pos.entry_equity, 1e-12) - 1.0) * 100.0)
                    trades.append(net_pct)
                    reasons[reason] = reasons.get(reason, 0) + 1
                    rows.append(_ledger_row(frame, arrays, before_pos, i, cash, net_pct, reason))
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
        before_side = int(dec.iloc[int(i)].get("side", 0) or 0)
        cash, pos, entered = ex._enter(cash, arrays, dec, i, fee_eff, slip_eff)
        if entered:
            long_entries += int(before_side > 0)
            short_entries += int(before_side < 0)

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


def _ledger_row(frame: pd.DataFrame, arrays: dict[str, np.ndarray], pos: ex.Position, exit_i: int, cash: float, net_pct: float, reason: str) -> dict[str, Any]:
    return {
        "trade_id": 0,
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
    ap.add_argument("--epochs", type=int, default=900)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--max-states", type=int, default=4200)
    ap.add_argument("--max-forward-bars", type=int, default=432)
    ap.add_argument("--train-buffer", type=float, default=0.006)
    ap.add_argument("--mask-penalty", type=float, default=0.006)
    ap.add_argument("--action-penalty", type=float, default=0.0015)
    ap.add_argument("--cql-weight", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=260612)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ex._state_base = mkt._state_base_market  # type: ignore[assignment]
    ex._pos_features = mkt._pos_features_market  # type: ignore[assignment]

    fee, slip = ex.omega._load_fee_slip()
    splits = ex._build_splits()
    built: dict[str, dict[str, Any]] = {}
    for split, payload in splits.items():
        dec = ex._to_decisions(payload["src"], payload["prefix"], oof=payload["oof"], thresholds=ex.HIGH_THRESHOLDS)
        state = mkt._state_base_market(payload["frame"], payload["src"], dec, payload["prefix"])
        built[split] = {"frame": payload["frame"], "dec": dec, "state": state}

    x_train, raw_rewards, data_diag = ex._collect_dataset(
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
    rewards = _conservative_rewards(
        raw_rewards,
        train_buffer=float(args.train_buffer),
        mask_penalty=float(args.mask_penalty),
        action_penalty=float(args.action_penalty),
    )
    model, train_diag = ex._train_q(x_train, rewards, epochs=int(args.epochs), seed=int(args.seed), cql_weight=float(args.cql_weight))
    model_path = OUT_DIR / "high_exit_q_market_conservative.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "norm": model.norm,  # type: ignore[attr-defined]
            "actions": ex.ACTION_NAMES,
            "train_diag": train_diag,
            "reward_config": vars(args),
        },
        model_path,
    )
    x_train.to_csv(OUT_DIR / "high_train_states.csv", index=False)
    pd.DataFrame(raw_rewards, columns=[f"raw_reward_{a}" for a in ex.ACTION_NAMES]).to_csv(OUT_DIR / "high_train_raw_rewards.csv", index=False)
    pd.DataFrame(rewards, columns=[f"reward_{a}" for a in ex.ACTION_NAMES]).to_csv(OUT_DIR / "high_train_conservative_rewards.csv", index=False)

    rows: list[dict[str, Any]] = []
    val_base, val_base_ledger = _simulate_policy_masked(
        built["validation"]["frame"],
        built["validation"]["dec"],
        built["validation"]["state"],
        model=None,
        min_adv=999.0,
        fee=fee,
        slip=slip,
        cost_mult=3.0,
        giveback_min=999.0,
        tp_progress_min=999.0,
        sl_progress_min=999.0,
    )
    oos_base, oos_base_ledger = _simulate_policy_masked(
        built["oos"]["frame"],
        built["oos"]["dec"],
        built["oos"]["state"],
        model=None,
        min_adv=999.0,
        fee=fee,
        slip=slip,
        cost_mult=3.0,
        giveback_min=999.0,
        tp_progress_min=999.0,
        sl_progress_min=999.0,
    )
    val_base_ledger.to_csv(OUT_DIR / "validation_baseline_ledger.csv", index=False)
    oos_base_ledger.to_csv(OUT_DIR / "oos_baseline_ledger.csv", index=False)
    rows.append({"variant": "baseline_no_exit_q", "min_adv": None, **_row("val", val_base), **_row("oos", oos_base)})

    configs = [
        (0.005, 0.45, 0.55, 0.70),
        (0.010, 0.45, 0.60, 0.75),
        (0.020, 0.50, 0.65, 0.80),
        (0.035, 0.55, 0.70, 0.85),
    ]
    for min_adv, giveback_min, tp_progress_min, sl_progress_min in configs:
        val_m, val_ledger = _simulate_policy_masked(
            built["validation"]["frame"],
            built["validation"]["dec"],
            built["validation"]["state"],
            model=model,
            min_adv=float(min_adv),
            fee=fee,
            slip=slip,
            cost_mult=3.0,
            giveback_min=float(giveback_min),
            tp_progress_min=float(tp_progress_min),
            sl_progress_min=float(sl_progress_min),
        )
        oos_m, oos_ledger = _simulate_policy_masked(
            built["oos"]["frame"],
            built["oos"]["dec"],
            built["oos"]["state"],
            model=model,
            min_adv=float(min_adv),
            fee=fee,
            slip=slip,
            cost_mult=3.0,
            giveback_min=float(giveback_min),
            tp_progress_min=float(tp_progress_min),
            sl_progress_min=float(sl_progress_min),
        )
        tag = f"adv{str(min_adv).replace('.', 'p')}_gb{str(giveback_min).replace('.', 'p')}_tp{str(tp_progress_min).replace('.', 'p')}"
        val_ledger.to_csv(OUT_DIR / f"validation_{tag}_ledger.csv", index=False)
        oos_ledger.to_csv(OUT_DIR / f"oos_{tag}_ledger.csv", index=False)
        rows.append(
            {
                "variant": tag,
                "min_adv": float(min_adv),
                "giveback_min": float(giveback_min),
                "tp_progress_min": float(tp_progress_min),
                "sl_progress_min": float(sl_progress_min),
                **_row("val", val_m),
                **_row("oos", oos_m),
                "val_entry_audit": ex._entry_audit(val_base_ledger, val_ledger),
                "oos_entry_audit": ex._entry_audit(oos_base_ledger, oos_ledger),
            }
        )

    ranking = pd.DataFrame(rows)
    base = ranking[ranking["variant"].eq("baseline_no_exit_q")].iloc[0]
    ranking["delta_oos_pnl"] = ranking["oos_pnl"] - float(base["oos_pnl"])
    ranking["delta_val_pnl"] = ranking["val_pnl"] - float(base["val_pnl"])
    ranking["score"] = ranking["oos_pnl"] + 0.45 * ranking["val_pnl"] + 0.35 * ranking["oos_mdd"] + 0.25 * ranking["val_mdd"]
    ranking = ranking.sort_values(["oos_pnl", "val_pnl"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "exit_q_market_conservative_ranking.csv", index=False)
    promotable = ranking[
        (ranking["variant"] != "baseline_no_exit_q")
        & (ranking["oos_pnl"] > float(base["oos_pnl"]))
        & (ranking["val_pnl"] > float(base["val_pnl"]) * 0.85)
        & (ranking["oos_mdd"] >= float(base["oos_mdd"]) * 1.20)
    ].copy()
    promotable.to_csv(OUT_DIR / "exit_q_market_conservative_promotable.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "purpose": "Conservative exit-Q with market features. Exit actions require hold advantage and state masks.",
        "baseline": base.to_dict(),
        "dataset": data_diag,
        "training": train_diag,
        "reward_config": vars(args),
        "promotable_count": int(len(promotable)),
        "top": ranking.to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "model": str(model_path),
            "ranking": str(OUT_DIR / "exit_q_market_conservative_ranking.csv"),
            "promotable": str(OUT_DIR / "exit_q_market_conservative_promotable.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "promotable_count": int(len(promotable)), "top": ranking.to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
