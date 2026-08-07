#!/usr/bin/env python3
"""Fresh-forward offline backtest for the Omega6 synthesis adapter.

Reuses the L7 next-open-limit execution contract (omega._try_execution/_fill_price)
and the same canonical funding-clean dataset the L2 trainer loads
(train_eval_omega1_2_tabm_diffusion_risk_20260603.TRAIN_CSV/EVAL_CSV).

AGENTS.md Fresh-Forward Validation/OOS/Test Rule compliance:
- Bar-by-bar causal walk. Each decision only sees a trailing context window ending at
  the current bar (no future rows, no stored trade ledgers as decision input).
- Validation window: 2025-10-01 to 2025-12-31 (see VAL_START comment below for why this
  differs from AGENTS.md's 2025-09-01 default). OOS window (2026-01-01 to 2026-03-31) is
  reserved and only scored when --score-oos is passed, after the L2 config is frozen.
- Entry bar/side/sizing is determined once at cost_mult=1.0 (the model does not see
  cost_mult) and replayed identically at cost_mult 2x/3x. Exit timing is NOT fixed across
  tiers: matching the established _metrics_with_shared_exit convention in
  train_eval_omega1_2_tabm_diffusion_risk_20260603.py, TP/SL triggers are checked against a
  slippage-adjusted unrealized price move (`px * (1 - slip_eff)` for longs), so higher
  slip_eff can genuinely delay a TP/SL trigger into a later bar (or push it to time_stop).
  This is realistic (slippage affects when an exit would actually clear), not a bug -- but
  it does mean cost1/cost2/cost3 are not guaranteed to hold the exact same trade count.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
from trading_bot_modules.omega6_live import Omega6LiveAdapter  # noqa: E402

MODEL_ID = "omega6_synthesis_v1_20260703"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

# AGENTS.md's default fresh-forward validation boundary is 2025-09-01, but the L2 trainer's
# own SPLIT_TS (scripts/train_eval_omega6_tabm_3head_20260703.py) trains on rows strictly
# before 2025-10-01 -- scoring from 09-01 would silently include ~1 month of L2's own training
# rows as "validation". Per AGENTS.md ("날짜 경계가 바뀌면 리포트에 명시해야 한다"), the boundary
# is moved to 2025-10-01 here to guarantee zero overlap with L2 training data. See
# docs/model_contracts/omega6_synthesis_v1_20260703_contract.md Open Issues for the
# contamination finding and the 2025-09-01 vs 2025-10-01 comparison.
VAL_START = pd.Timestamp("2025-10-01")
VAL_END = pd.Timestamp("2025-12-31 23:59:59")
OOS_START = pd.Timestamp("2026-01-01")
OOS_END = pd.Timestamp("2026-03-31 23:59:59")

# >= max(L5 atr_window=192, L3 tcn lookback=24, L6 ret_4h lag=48) plus buffer.
CONTEXT_BARS = 260

DEFAULT_PRIMARY_BUNDLE = ROOT / "tmp/causal_regen_20260516/omega6_true_3head_tabm_20260703_primary/true_3head_tabm_bundle.pt"
DEFAULT_FALLBACK_BUNDLE = ROOT / "tmp/causal_regen_20260516/omega6_true_3head_tabm_20260703_fallback/true_3head_tabm_bundle.pt"
# L3/L4 defaults below point at artifacts refit specifically against Omega6's own L2 decision
# trace, train-split only (< SPLIT_TS 2025-10-01) -- see scripts/train_omega6_risk_sidecar_20260703.py
# and scripts/train_omega6_sequence_gate_20260703.py, fixing contamination-audit Findings 2/L3
# in docs/model_contracts/omega6_synthesis_v1_20260703_contract.md.
DEFAULT_TCN_GATE = ROOT / "tmp/causal_regen_20260516/omega6_sequence_gate_20260703/tcn_seq_gate_L24_omega6.pt"
DEFAULT_RISK_SIDECAR = ROOT / "tmp/causal_regen_20260516/omega6_risk_sidecar_20260703/risk_sidecar.pkl"


def _load_combined_frame() -> pd.DataFrame:
    # omega._load_omega_frames() overlays the Regime3 current/cmamba/risk columns
    # (incl. ROUTE_COLS) required by L2 routing -- a raw _read(TRAIN_CSV/EVAL_CSV) is missing them.
    train, eval_df, _overlay_report = omega._load_omega_frames()
    combined = pd.concat([train, eval_df], ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    return combined


def _window_bounds(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[int, int]:
    ts = frame["timestamp"]
    start_idx = int(ts.searchsorted(start, side="left"))
    end_idx = int(ts.searchsorted(end, side="right"))
    return start_idx, end_idx


def _run_pass(
    frame: pd.DataFrame,
    adapter: Omega6LiveAdapter | None,
    arrays: dict[str, np.ndarray],
    *,
    start_idx: int,
    end_idx: int,
    fee: float,
    slip: float,
    cost_mult: float,
    overrides: list[dict[str, Any]] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    trades: list[dict[str, Any]] = []
    pos = 0
    entry_price = 0.0
    entry_equity = 1.0
    hold_start = 0
    notional = 0.0
    leverage = 1.0
    take_profit = 0.0
    stop_loss = 0.0
    max_hold = 0
    wins = 0
    reasons: dict[str, int] = {}
    override_iter = iter(overrides) if overrides is not None else None
    pending = next(override_iter, None) if override_iter is not None else None
    i = start_idx
    while i < end_idx:
        if pos == 0:
            if overrides is not None:
                # Skip any override whose entry_i already passed (e.g. we were still holding a
                # prior position at that bar, since exit timing is cost_mult-dependent -- see
                # note below). Without this, a single missed override would permanently orphan
                # `pending` and silently drop every subsequent trade for the rest of the pass.
                while pending is not None and int(pending["entry_i"]) < i:
                    pending = next(override_iter, None)
                if pending is None or int(pending["entry_i"]) != i:
                    i += 1
                    continue
                side = int(pending["side"])
                notional = float(pending["notional"])
                leverage = float(pending["leverage"])
                take_profit = float(pending["take_profit"])
                stop_loss = float(pending["stop_loss"])
                max_hold = int(pending["max_hold_bars"])
            else:
                window = frame.iloc[max(0, i - CONTEXT_BARS + 1) : i + 1]
                dec = adapter.decide_latest(window)  # type: ignore[union-attr]
                if dec.side == 0:
                    i += 1
                    continue
                side = int(dec.side)
                notional = float(dec.notional_exposure)
                leverage = float(dec.leverage)
                take_profit = float(dec.take_profit)
                stop_loss = float(dec.stop_loss)
                max_hold = int(dec.max_hold_bars)
            filled, px, entry_fee, _route = omega._try_execution(arrays, i, side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
            if not filled:
                if overrides is not None:
                    pending = next(override_iter, None)
                i += 1
                continue
            pos = side
            entry_price = float(px)
            entry_equity = cash
            hold_start = i
            cash -= cash * entry_fee * notional
            i += 1
            continue
        px = float(arrays["close"][i])
        raw = (
            (px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1e-12)
            if pos > 0
            else (entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1e-12)
        )
        unreal = raw * notional
        eq = cash * (1.0 + unreal)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        hold_bars = i - hold_start
        reason = ""
        if take_profit > 0.0 and unreal >= take_profit:
            reason = "take_profit"
        elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
            reason = "stop_loss"
        elif hold_bars >= max_hold:
            reason = "time_stop"
        if reason:
            filled, exit_px, exit_fee, _route = omega._try_execution(arrays, i, pos, entry=False, fee_base=fee_eff, slip_base=slip_eff)
            if not filled:
                i += 1
                continue
            raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
            before = cash
            cash = cash * (1.0 + raw_exit * notional)
            cash -= before * exit_fee * notional
            wins += int(cash > entry_equity)
            reasons[reason] = reasons.get(reason, 0) + 1
            trades.append(
                {
                    "entry_i": int(hold_start),
                    "exit_i": int(i),
                    "side": int(pos),
                    "notional": float(notional),
                    "leverage": float(leverage),
                    "take_profit": float(take_profit),
                    "stop_loss": float(stop_loss),
                    "max_hold_bars": int(max_hold),
                    "exit_reason": reason,
                }
            )
            pos = 0
            if overrides is not None:
                pending = next(override_iter, None)
        i += 1
    if pos != 0:
        exit_px = omega._fill_price(arrays, end_idx - 1, pos, slip_eff, entry=False)
        raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw_exit * notional)
        cash -= before * fee_eff * notional
        wins += int(cash > entry_equity)
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
        trades.append(
            {
                "entry_i": int(hold_start),
                "exit_i": int(end_idx - 1),
                "side": int(pos),
                "notional": float(notional),
                "leverage": float(leverage),
                "take_profit": float(take_profit),
                "stop_loss": float(stop_loss),
                "max_hold_bars": int(max_hold),
                "exit_reason": "forced_end",
            }
        )
    avg_notional = float(np.mean([t["notional"] for t in trades])) if trades else 0.0
    avg_leverage = float(np.mean([t["leverage"] for t in trades])) if trades else 0.0
    metrics = {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(len(trades)),
        "wr": float(wins / len(trades)) if trades else 0.0,
        "avg_notional": avg_notional,
        "avg_leverage": avg_leverage,
        "exit_reasons": reasons,
    }
    return metrics, trades


def _walk_forward_monthly(frame: pd.DataFrame, trades: list[dict[str, Any]], fee: float, slip: float) -> dict[str, dict[str, Any]]:
    """Per-calendar-month breakdown of the cost1 trade list, each month scored as its own
    independent equity curve (starting fresh at 1.0) -- checks whether the aggregate edge is
    consistent across sub-periods rather than concentrated in one lucky month."""
    if not trades:
        return {}
    months: dict[str, list[dict[str, Any]]] = {}
    for t in trades:
        month_key = pd.Timestamp(frame.iloc[t["entry_i"]]["timestamp"]).strftime("%Y-%m")
        months.setdefault(month_key, []).append(t)
    out: dict[str, dict[str, Any]] = {}
    for month_key, month_trades in sorted(months.items()):
        cash = 1.0
        peak = 1.0
        mdd = 0.0
        wins = 0
        for t in month_trades:
            entry_i = int(t["entry_i"])
            exit_i = int(t["exit_i"])
            side = int(t["side"])
            notional = float(t["notional"])
            entry_price = float(frame.iloc[min(entry_i + 1, len(frame) - 1)]["open"])
            exit_price = float(frame.iloc[exit_i]["close"])
            raw = (exit_price - entry_price) / max(entry_price, 1e-12) if side > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
            net = raw - fee - fee
            before = cash
            cash = cash * (1.0 + net * notional)
            peak = max(peak, cash)
            mdd = min(mdd, cash / max(peak, 1e-12) - 1.0)
            wins += int(cash > before)
        out[month_key] = {
            "trades": len(month_trades),
            "pnl": float((cash - 1.0) * 100.0),
            "mdd": float(mdd * 100.0),
            "wr": float(wins / len(month_trades)),
        }
    return out


def _score_window(
    frame: pd.DataFrame,
    adapter: Omega6LiveAdapter,
    arrays: dict[str, np.ndarray],
    *,
    start_idx: int,
    end_idx: int,
    fee: float,
    slip: float,
) -> dict[str, Any]:
    ref_metrics, trades = _run_pass(frame, adapter, arrays, start_idx=start_idx, end_idx=end_idx, fee=fee, slip=slip, cost_mult=1.0, overrides=None)
    cost_stress = {"cost1": ref_metrics}
    for mult, tag in ((2.0, "cost2"), (3.0, "cost3")):
        m, _ = _run_pass(frame, None, arrays, start_idx=start_idx, end_idx=end_idx, fee=fee, slip=slip, cost_mult=mult, overrides=trades)
        cost_stress[tag] = m
    days = max((end_idx - start_idx) / (12.0 * 24.0), 1e-6)
    walk_forward_monthly = _walk_forward_monthly(frame, trades, fee, slip)
    return {
        **ref_metrics,
        "trades_per_day": float(ref_metrics["trades"] / days),
        "cost_stress": cost_stress,
        "walk_forward_monthly": walk_forward_monthly,
        "window_bars": int(end_idx - start_idx),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary-bundle", default=str(DEFAULT_PRIMARY_BUNDLE))
    ap.add_argument("--fallback-bundle", default=str(DEFAULT_FALLBACK_BUNDLE))
    ap.add_argument("--tcn-gate", default=str(DEFAULT_TCN_GATE))
    ap.add_argument("--risk-sidecar", default=str(DEFAULT_RISK_SIDECAR))
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    ap.add_argument("--score-oos", action="store_true", help="Also score the reserved OOS window (2026-01-01..03-31). Only use after L2 config is frozen.")
    ap.add_argument("--disable-l3-gate", action="store_true", help="Disable the L3 TCN gate even if a retrained artifact is available.")
    args = ap.parse_args()

    device = "cuda" if (args.device == "auto" and __import__("torch").cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)
    adapter = Omega6LiveAdapter(
        primary_bundle_path=args.primary_bundle,
        fallback_bundle_path=args.fallback_bundle,
        tcn_gate_path=args.tcn_gate,
        risk_sidecar_path=args.risk_sidecar,
        device=device,
        enable_l3_gate=not args.disable_l3_gate,
    )

    frame = _load_combined_frame()
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    fee, slip = omega._load_fee_slip()

    val_start_idx, val_end_idx = _window_bounds(frame, VAL_START, VAL_END)
    val_start_idx = max(val_start_idx, CONTEXT_BARS)
    report: dict[str, Any] = {
        "model_id": MODEL_ID,
        "adapter_module": "trading_bot_modules/omega6_live.py",
        "design_doc": "docs/model_contracts/omega6_synthesis_design_20260703.md",
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "context_bars": int(CONTEXT_BARS),
        "artifacts": {
            "primary_bundle": str(args.primary_bundle),
            "fallback_bundle": str(args.fallback_bundle),
            "tcn_gate": str(args.tcn_gate),
            "risk_sidecar": str(args.risk_sidecar),
        },
        "enable_l3_gate": bool(not args.disable_l3_gate),
        "windows": {
            "validation": {"start": str(VAL_START), "end": str(VAL_END)},
            "oos_reserved": {"start": str(OOS_START), "end": str(OOS_END)},
        },
    }

    if val_end_idx <= val_start_idx:
        raise RuntimeError(f"Omega6 backtest validation window is empty in dataset (start_idx={val_start_idx}, end_idx={val_end_idx})")
    report["validation"] = _score_window(frame, adapter, arrays, start_idx=val_start_idx, end_idx=val_end_idx, fee=fee, slip=slip)
    report["validation"]["actual_bar_range"] = {
        "start": str(frame.iloc[val_start_idx]["timestamp"]),
        "end": str(frame.iloc[val_end_idx - 1]["timestamp"]),
    }

    if args.score_oos:
        oos_start_idx, oos_end_idx = _window_bounds(frame, OOS_START, OOS_END)
        oos_start_idx = max(oos_start_idx, CONTEXT_BARS)
        if oos_end_idx > oos_start_idx:
            report["oos"] = _score_window(frame, adapter, arrays, start_idx=oos_start_idx, end_idx=oos_end_idx, fee=fee, slip=slip)
            report["oos"]["actual_bar_range"] = {
                "start": str(frame.iloc[oos_start_idx]["timestamp"]),
                "end": str(frame.iloc[oos_end_idx - 1]["timestamp"]),
            }
            report["oos"]["coverage_note"] = "Dataset OOS coverage may fall short of 2026-03-31; see actual_bar_range."
        else:
            report["oos"] = {"skipped": True, "reason": "no rows in requested OOS window"}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out_path), "validation": report["validation"]}, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
