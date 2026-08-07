#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, ACTION_LONG  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.analyze_alpha7_tp_sl_action_score_20260526 import SPLIT_TS, _json_default  # noqa: E402
from scripts.train_alpha7_regime3_current_moe_feature_variants_20260601 import _load_frames_with_risk  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _days, _fill_price  # noqa: E402


MODEL_ID = "alpha7_regime3_current_moe_trade_ledger_wr_20260601"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_regime3_current_moe_trade_ledger_wr_20260601"
ACTIVE_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_regime3_current_moe_active_mix_expert_scale_refine_20260601"


def _backtest_decisions(df: pd.DataFrame, dec: pd.DataFrame, *, fee: float, slip: float) -> dict[str, Any]:
    close = _close(df)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = 0.0
    entry_equity = 1.0
    entry_idx = 0
    notional = 0.0
    take_profit = 0.0
    stop_loss = 0.0
    max_hold = 0
    next_cooldown = 0
    cooldown_left = 0
    trades = 0
    wins = 0
    exits: dict[str, int] = {}
    records: list[dict[str, Any]] = []
    open_record: dict[str, Any] | None = None
    mfe = 0.0
    mae = 0.0

    def mark_equity(i: int) -> tuple[float, float]:
        if pos == 0:
            return cash, 0.0
        px = float(close[int(np.clip(i, 0, len(close) - 1))])
        raw = (px * (1.0 - slip) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip)) / max(entry_price, 1e-12)
        unreal = raw * notional
        return cash * (1.0 + unreal), unreal

    for i in range(0, len(df) - 2):
        eq, unreal = mark_equity(i)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if pos != 0:
            mfe = max(mfe, unreal)
            mae = min(mae, unreal)
            hold_bars = i - entry_idx
            reason = ""
            if take_profit > 0.0 and unreal >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
                reason = "stop_loss"
            elif max_hold > 0 and hold_bars >= max_hold:
                reason = "max_hold"
            if reason:
                fill_idx = min(i + 1, len(df) - 1)
                exit_price = _fill_price(df, fill_idx, pos, slip, entry=False)
                raw = (exit_price - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw * notional)
                cash -= before * fee * notional
                trades += 1
                wins += int(cash > entry_equity)
                exits[reason] = exits.get(reason, 0) + 1
                if open_record is not None:
                    rec = dict(open_record)
                    rec.update({
                        "exit_signal_timestamp": str(df["timestamp"].iloc[i]),
                        "exit_fill_timestamp": str(df["timestamp"].iloc[fill_idx]),
                        "exit_reason": reason,
                        "hold_bars_realized": int(hold_bars),
                        "realized_net_pct": float((cash / max(entry_equity, 1e-12) - 1.0) * 100.0),
                        "mfe_pct": float(mfe * 100.0),
                        "mae_pct": float(mae * 100.0),
                    })
                    records.append(rec)
                pos = 0
                notional = 0.0
                cooldown_left = int(next_cooldown)
                next_cooldown = 0
                open_record = None
                mfe = 0.0
                mae = 0.0
                continue
        if pos != 0:
            continue
        if cooldown_left > 0:
            cooldown_left -= 1
            continue
        row = dec.iloc[i]
        if int(row.action) == ACTION_CASH or int(row.side) == 0:
            continue
        fill_idx = min(i + 1, len(df) - 1)
        pos = int(row.side)
        entry_price = _fill_price(df, fill_idx, pos, slip, entry=True)
        entry_equity = cash
        entry_idx = i
        notional = float(row.notional_exposure)
        take_profit = float(row.take_profit)
        stop_loss = float(row.stop_loss)
        max_hold = int(row.max_hold_bars)
        next_cooldown = int(row.cooldown_bars)
        cash -= cash * fee * notional
        open_record = {
            "entry_signal_timestamp": str(df["timestamp"].iloc[i]),
            "entry_fill_timestamp": str(df["timestamp"].iloc[fill_idx]),
            "side": "LONG" if int(row.action) == ACTION_LONG else "SHORT",
            "router_expert": str(row.router_expert),
            "router_confidence": float(row.router_confidence),
            "quality_score": float(row.quality_score),
            "confidence": float(row.confidence),
            "notional_exposure": float(row.notional_exposure),
            "take_profit": float(row.take_profit),
            "stop_loss": float(row.stop_loss),
            "max_hold_bars": int(row.max_hold_bars),
        }

    if pos != 0:
        fill_idx = len(df) - 1
        exit_price = _fill_price(df, fill_idx, pos, slip, entry=False)
        raw = (exit_price - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * notional)
        cash -= before * fee * notional
        trades += 1
        wins += int(cash > entry_equity)
        exits["forced_end"] = exits.get("forced_end", 0) + 1
        if open_record is not None:
            rec = dict(open_record)
            rec.update({
                "exit_signal_timestamp": str(df["timestamp"].iloc[-1]),
                "exit_fill_timestamp": str(df["timestamp"].iloc[fill_idx]),
                "exit_reason": "forced_end",
                "hold_bars_realized": int(len(df) - 1 - entry_idx),
                "realized_net_pct": float((cash / max(entry_equity, 1e-12) - 1.0) * 100.0),
                "mfe_pct": float(mfe * 100.0),
                "mae_pct": float(mae * 100.0),
            })
            records.append(rec)

    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / max(trades, 1)),
        "trades_per_day": float(trades / _days(df)),
        "exits": exits,
        "trade_records": records,
    }


def _ledger_stats(ledger: pd.DataFrame) -> dict[str, Any]:
    if ledger.empty:
        return {}
    pnl = pd.to_numeric(ledger["realized_net_pct"], errors="raise")
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    out: dict[str, Any] = {
        "trades": int(len(ledger)),
        "wr": float((pnl > 0).mean()),
        "avg_trade_pct": float(pnl.mean()),
        "median_trade_pct": float(pnl.median()),
        "avg_win_pct": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_pct": float(losses.mean()) if len(losses) else 0.0,
        "payoff_ratio": float((wins.mean() if len(wins) else 0.0) / max(abs(losses.mean() if len(losses) else 0.0), 1e-12)),
        "profit_factor": float(gross_win / max(gross_loss, 1e-12)),
        "gross_win_pct": gross_win,
        "gross_loss_pct": gross_loss,
        "exit_counts": {str(k): int(v) for k, v in ledger["exit_reason"].value_counts().to_dict().items()},
    }
    by_expert: dict[str, Any] = {}
    for expert, g in ledger.groupby("router_expert"):
        gp = pd.to_numeric(g["realized_net_pct"], errors="raise")
        gw = gp[gp > 0]
        gl = gp[gp <= 0]
        by_expert[str(expert)] = {
            "trades": int(len(g)),
            "wr": float((gp > 0).mean()),
            "avg_trade_pct": float(gp.mean()),
            "avg_win_pct": float(gw.mean()) if len(gw) else 0.0,
            "avg_loss_pct": float(gl.mean()) if len(gl) else 0.0,
            "payoff_ratio": float((gw.mean() if len(gw) else 0.0) / max(abs(gl.mean() if len(gl) else 0.0), 1e-12)),
            "profit_factor": float(float(gw.sum()) / max(float(-gl.sum()), 1e-12)),
            "exit_counts": {str(k): int(v) for k, v in g["exit_reason"].value_counts().to_dict().items()},
        }
    out["by_expert"] = by_expert
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_all, eval_df, overlay = _load_frames_with_risk()
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
    val_dec = pd.read_csv(ACTIVE_DIR / "validation_decisions.csv").reset_index(drop=True)
    oos_dec = pd.read_csv(ACTIVE_DIR / "oos_2026_decisions.csv").reset_index(drop=True)
    fee = float(joblib.load(v31.DEFAULT_PARENT)["config"]["fee"]) * 3.0
    slip = float(joblib.load(v31.DEFAULT_PARENT)["config"]["slip"]) * 3.0
    if len(val_df) != len(val_dec) or len(eval_df) != len(oos_dec):
        raise RuntimeError(f"frame/decision mismatch: val {len(val_df)} {len(val_dec)} oos {len(eval_df)} {len(oos_dec)}")
    val_bt = _backtest_decisions(val_df, val_dec, fee=fee, slip=slip)
    oos_bt = _backtest_decisions(eval_df, oos_dec, fee=fee, slip=slip)
    val_ledger = pd.DataFrame(val_bt.pop("trade_records", []))
    oos_ledger = pd.DataFrame(oos_bt.pop("trade_records", []))
    val_ledger.to_csv(OUT_DIR / "validation_trade_ledger_cost3_approx.csv", index=False)
    oos_ledger.to_csv(OUT_DIR / "oos_2026_trade_ledger_cost3_approx.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "note": "Diagnostic direct open-fill ledger using 3x fee/slip. Official promotion metrics remain the existing _combo_metrics reports; this ledger is for WR/payoff/exits diagnosis.",
        "fee": fee,
        "slip": slip,
        "overlay": overlay,
        "validation_backtest": val_bt,
        "oos_backtest": oos_bt,
        "validation_ledger_stats": _ledger_stats(val_ledger),
        "oos_ledger_stats": _ledger_stats(oos_ledger),
        "artifacts": {
            "report": str(OUT_DIR / "report.json"),
            "validation_ledger": str(OUT_DIR / "validation_trade_ledger_cost3_approx.csv"),
            "oos_ledger": str(OUT_DIR / "oos_2026_trade_ledger_cost3_approx.csv"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "validation_ledger_stats": report["validation_ledger_stats"], "oos_ledger_stats": report["oos_ledger_stats"]}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
