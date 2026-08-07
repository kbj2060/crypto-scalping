#!/usr/bin/env python3
"""Candidate: detect regime reversal AGAINST an open position (using the already-live
regime3_current_sensitive_wide24_{bull,bear}_prob columns -- no new model at all, not even the
small HistGradientBoosting sidecar from replay_omega4_6_1_dynamic_risk_sltp_20260720.py) and shrink
the TP/SL barrier once, the first time it fires. Real live stack (h48qual+zig075, frozen bundles +
sidecars + SCALE_MAP + greedy router + duration gate) is unchanged; only the exit-barrier value is
touched, exactly like the trailing-exit and dynamic-risk-sidecar candidates before it.

Rationale (user's follow-up after the dynamic-risk-sidecar test came back essentially inert --
ratchet fired 1/22 VAL trades, 0/31 OOS trades): that sidecar's "how much room is realistically
left" opinion rarely disagreed with the ATR floor for these already quality-gated trades. This
candidate instead asks a much more direct, literal version of the user's original hypothesis --
"has the market's own regime call reversed since I entered?" -- using regime3's own bull/bear
probability gap (edge = bull_prob - bear_prob), which the live model already computes every bar for
its own expert routing. No training at all; only two free parameters (reversal threshold, shrink
fraction), grid-searched on VAL only per the Fresh-Forward Rule, then frozen and scored once on OOS.

Stored-ledger based -> DIAGNOSTIC research score, not a live-promotion claim. trading_bot.py /
omega4_6_1_live.py are NOT touched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as sidecar  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import replay_omega4_6_1_greedy_val_20260706 as valmod  # noqa: E402
from test_omega4_6_1_drop_h48qual_20260706 import _metrics  # noqa: E402
import replay_omega4_6_1_dynamic_risk_sltp_20260720 as dynrisk  # noqa: E402 (pandas string-dtype fix + load_components)

OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega4_6_1_regime_reversal_sltp_20260720"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_TP, MAX_TP, MIN_SL, MAX_SL = 0.075, 0.22, 0.040, 0.12
BULL_COL = "regime3_current_sensitive_wide24_bull_prob"
BEAR_COL = "regime3_current_sensitive_wide24_bear_prob"

THRESHOLD_GRID = [0.15, 0.25, 0.35, 0.50]
SHRINK_FRAC_GRID = [0.3, 0.5, 0.7]


@torch.no_grad()
def greedy_replay_regime_reversal(frame: pd.DataFrame, components: dict, *, fee: float, slip: float,
                                   cost_mult: float, device: torch.device,
                                   threshold: float | None, shrink_frac: float | None) -> tuple[dict, pd.DataFrame]:
    """Fork of greedy.greedy_replay: while a position is open, track edge = bull_prob - bear_prob at
    entry vs now. If it has moved against the position by more than `threshold`, shrink TP/SL once
    (multiply by shrink_frac, floor-clamped) -- never re-widens even if the regime call reverts."""
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    edge = pd.to_numeric(frame[BULL_COL], errors="raise").to_numpy(dtype=np.float64) - pd.to_numeric(frame[BEAR_COL], errors="raise").to_numpy(dtype=np.float64)
    n = len(frame)
    fee_eff, slip_eff = float(fee) * float(cost_mult), float(slip) * float(cost_mult)
    cash = peak = 1.0
    mdd = 0.0
    pos = 0
    active_comp = None
    entry_price = entry_equity = 1.0
    entry_i = entry_signal_i = 0
    notional = leverage_v = margin_fraction = 0.0
    take_profit = stop_loss = 0.0
    entry_edge = 0.0
    shrunk = False
    mfe = mae = 0.0
    reversal_events = 0
    rows: list[dict] = []
    reasons: dict[str, int] = {}

    enabled = threshold is not None and shrink_frac is not None

    for i in range(0, n - 2):
        if pos != 0:
            comp = components[active_comp]
            move = (arrays["close"][i] * (1 - slip_eff) - entry_price) / entry_price if pos > 0 else (entry_price - arrays["close"][i] * (1 + slip_eff)) / entry_price
            unreal = move * notional
            mfe, mae = max(mfe, move), min(mae, move)
            eq = cash * (1.0 + unreal)
            peak = max(peak, eq)
            mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)

            if enabled and not shrunk:
                moved_against = (entry_edge - edge[i]) if pos > 0 else (edge[i] - entry_edge)
                if moved_against > float(threshold):
                    take_profit = max(MIN_TP, take_profit * float(shrink_frac))
                    stop_loss = max(MIN_SL, stop_loss * float(shrink_frac))
                    shrunk = True
                    reversal_events += 1

            reason = ""
            if take_profit > 0.0 and move >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and move <= -abs(stop_loss):
                reason = "stop_loss"
            else:
                hold = max(i - entry_i, 0)
                giveback = (mfe - move) / max(abs(mfe), 1e-8) if mfe > 0.0 else 0.0
                expert = hard.EXPERT_NAMES[int(comp["route"][i])]
                prob = sidecar._predict_exit_prob_one(
                    comp["base_np"], comp["exit_runtime"], comp["pos_idx"], row_i=int(i), expert=expert,
                    pos_values=[float(pos), float(hold), float(move), float(mfe), float(mae),
                                float(np.clip(giveback, 0.0, 10.0)), float(take_profit - move),
                                float(move + abs(stop_loss)), float(notional), float(leverage_v),
                                float(notional * leverage_v), float(take_profit), float(stop_loss)],
                    device=device,
                )
                if prob >= comp["exit_threshold"]:
                    reason = "exit_head"
            if reason:
                exit_px = arrays["close"][i] * (1 - slip_eff if pos > 0 else 1 + slip_eff)
                raw_exit = (exit_px - entry_price) / entry_price if pos > 0 else (entry_price - exit_px) / entry_price
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * fee_eff * notional
                trade_return = cash / max(entry_equity, 1e-12) - 1.0
                reasons[reason] = reasons.get(reason, 0) + 1
                rows.append({"entry_signal_i": entry_signal_i, "entry_i": entry_i, "exit_i": i,
                             "entry_timestamp": str(frame["timestamp"].iloc[entry_signal_i]),
                             "exit_timestamp": str(frame["timestamp"].iloc[i]), "side": int(pos),
                             "source_component": active_comp, "reason": reason, "shrunk": bool(shrunk),
                             "win": int(cash > entry_equity), "trade_return": float(trade_return),
                             "notional": float(notional), "margin_fraction": float(margin_fraction),
                             "leverage": float(leverage_v)})
                pos, active_comp = 0, None
                continue
            continue

        for name in greedy.PRIORITY:
            comp = components[name]
            side = int(comp["dec"]["side"].iloc[i])
            if side == 0 or not bool(omega._active(comp["dec"]).iloc[i] if hasattr(omega._active(comp["dec"]), "iloc") else omega._active(comp["dec"])[i]):
                continue
            row_margin, row_leverage = float(comp["margin"][i]), float(comp["leverage"][i])
            if row_margin <= 0.0:
                continue
            scale = greedy.SCALE_MAP.get(f"{name}_{'L' if side > 0 else 'S'}", 1.0)
            row_leverage = min(row_leverage * scale, greedy.LEVERAGE_CAP)
            row_notional = min(row_margin * row_leverage, greedy.NOTIONAL_CAP)
            row_leverage = row_notional / max(row_margin, 1e-12)
            if row_notional <= 0.0:
                continue
            entry_px = arrays["open"][min(i + 1, n - 1)] * (1 + slip_eff if side > 0 else 1 - slip_eff)
            pos, active_comp = side, name
            entry_price, entry_equity = float(entry_px), cash
            entry_i, entry_signal_i = min(i + 1, n - 1), i
            margin_fraction, leverage_v, notional = row_margin, row_leverage, row_notional
            take_profit = float(comp["dec"]["take_profit"].iloc[i])
            stop_loss = float(comp["dec"]["stop_loss"].iloc[i])
            entry_edge = float(edge[entry_i])
            shrunk = False
            mfe = mae = 0.0
            break

    return {"reason_counts": reasons, "reversal_events": reversal_events}, pd.DataFrame(rows)


def score(frame, components, *, fee, slip, threshold, shrink_frac) -> tuple[dict, dict, pd.DataFrame]:
    diag, lg = greedy_replay_regime_reversal(frame, components, fee=fee, slip=slip, cost_mult=retest.COST_MULT,
                                              device=retest.DEVICE, threshold=threshold, shrink_frac=shrink_frac)
    return _metrics(lg, frame, apply_gate=True), diag, lg


def main() -> int:
    device = retest.DEVICE
    fee, slip = omega._load_fee_slip()

    val_frame_raw = valmod.load_val_frame()
    val_frame, val_components = dynrisk.load_components(val_frame_raw, device, val=True)

    baseline_val, _, _ = score(val_frame, val_components, fee=fee, slip=slip, threshold=None, shrink_frac=None)
    print(f"VAL baseline (static TP/SL): {baseline_val}", flush=True)

    grid_results = []
    for threshold in THRESHOLD_GRID:
        for shrink_frac in SHRINK_FRAC_GRID:
            m, diag, _ = score(val_frame, val_components, fee=fee, slip=slip, threshold=threshold, shrink_frac=shrink_frac)
            grid_results.append({"threshold": threshold, "shrink_frac": shrink_frac, "reversal_events": diag["reversal_events"], **m})
            print(f"  threshold={threshold:.2f} shrink_frac={shrink_frac:.2f} -> pnl={m['pnl']:+7.2f}% mdd={m['mdd']:+6.2f}% "
                  f"n={m['trades']:2d} wr={m['wr']:.3f} reversal_events={diag['reversal_events']}", flush=True)

    grid_results.sort(key=lambda r: r["pnl"], reverse=True)
    best = grid_results[0]
    print(f"\nBest VAL config: {best}", flush=True)
    adopt = bool(best["pnl"] > baseline_val["pnl"])
    print(f"Decision (VAL-only, pre-registered): {'ADOPT' if adopt else 'REJECT'} regime-reversal shrink "
          f"(best VAL pnl={best['pnl']:+.2f}% vs baseline {baseline_val['pnl']:+.2f}%)", flush=True)

    oos_frame_raw = retest.load_frame_current("2026-01-01", "2026-06-30")
    oos_frame, oos_components = dynrisk.load_components(oos_frame_raw, device, val=False)
    baseline_oos, _, _ = score(oos_frame, oos_components, fee=fee, slip=slip, threshold=None, shrink_frac=None)
    frozen_oos, frozen_diag, oos_ledger = score(oos_frame, oos_components, fee=fee, slip=slip, threshold=best["threshold"], shrink_frac=best["shrink_frac"])
    print(f"\nOOS baseline: {baseline_oos}", flush=True)
    print(f"OOS frozen regime-reversal config: {frozen_oos} reversal_events={frozen_diag['reversal_events']}", flush=True)

    result = {
        "model_id": "omega4_6_1_regime_reversal_sltp_20260720",
        "grid": grid_results,
        "best_val_config": {"threshold": best["threshold"], "shrink_frac": best["shrink_frac"]},
        "val": {"baseline": baseline_val, "best": best},
        "oos": {"baseline": baseline_oos, "frozen": frozen_oos, "reversal_events": frozen_diag["reversal_events"]},
        "adopt_decision_val_only": adopt,
    }
    (OUT_DIR / "result.json").write_text(json.dumps(result, indent=2, default=str))
    oos_ledger.to_csv(OUT_DIR / "oos_ledger_frozen.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'result.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
