"""Test CryptoMamba's future-regime prediction as a POST-HOC intra-trade early-exit gate (not a
model input) for ETH's live greedy router: while a position is open, if CryptoMamba's future
(+6bar) prediction flags a DIRECTIONAL reversal against the current position (long + predicts
bear, or short + predicts bull -- excluding the "into chop" calls, which this session's precision
breakdown showed are the easy/inflated case, not the genuinely valuable directional signal), exit
immediately at that bar's close instead of waiting for TP/SL/exit-head.

Mirrors the same "external gate, not a model feature" pattern that worked for ETH's chop
soft-sizing (docs/model_contracts/eth_leverage_chop_softsize_fresh_forward_20260720.md) instead of
the repeatedly-failed "feed it into the model" pattern (docs/model_contracts/sol_btc_regime_models_retrain_tuning_20260721.md).
Reuses greedy_replay's exact bar-by-bar loop (imported logic, TP/SL/exit-head unchanged) with one
added early-exit branch.
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

import replay_omega4_6_1_greedy_router_20260706 as router  # noqa: E402
import replay_omega4_6_1_greedy_val_20260706 as valmod  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402

CONFIRM_SWEEP = [3, 6, 12, 24, 48]  # bars of consecutive against-position signal required (5m bars: 15/30/60/120/240 min)

CMAMBA_DIR = ROOT / "data/ensemble/supervised/regime3_cryptomamba_pred_h6_nocurrent_20260531"
CMAMBA_2025 = CMAMBA_DIR / "training_features_2025_regime3_cryptomamba_pred_h6_nocurrent_20260531.csv"
CMAMBA_2026 = CMAMBA_DIR / "training_features_2026_rebuilt_regime3_cryptomamba_pred_h6_nocurrent_20260531.csv"


def _cmamba_dir_signal(frame: pd.DataFrame, cmamba_path: Path) -> np.ndarray:
    """+1 if CryptoMamba predicts bull, -1 if bear, 0 if chop (or missing) -- aligned to frame rows."""
    cm = pd.read_csv(cmamba_path, parse_dates=["timestamp"])
    cm = cm[["timestamp", "regime3_cmamba_h6_future_pred_id"]]
    merged = frame[["timestamp"]].merge(cm, on="timestamp", how="left", validate="one_to_one")
    pred_id = merged["regime3_cmamba_h6_future_pred_id"].to_numpy()
    # CLASSES3 = ["bull", "bear", "chop"] per train_regime3_cryptomamba_pred_20260531.py
    sig = np.zeros(len(merged), dtype=np.int64)
    sig[pred_id == 0] = 1   # bull
    sig[pred_id == 1] = -1  # bear
    return sig  # 0 (chop or NaN) means "no directional call"


def greedy_replay_with_reversal_gate(frame: pd.DataFrame, components: dict, cmamba_dir_sig: np.ndarray, *,
                                      fee: float, slip: float, cost_mult: float, device: torch.device,
                                      gate_enabled: bool, confirm_bars: int = 1) -> tuple[dict, pd.DataFrame]:
    """confirm_bars: the against-position directional signal must hold for this many CONSECUTIVE
    bars (reset to 0 the instant it stops agreeing) before the gate fires -- a persistence filter
    to stop reacting to single-bar noise in CryptoMamba's own bar-to-bar fluctuation."""
    hard, omega, sidecar = router.hard, router.omega, router.sidecar
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
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
    mfe = mae = 0.0
    rows: list[dict] = []
    reasons: dict[str, int] = {}
    confirm_streak = 0

    for i in range(0, n - 2):
        if pos != 0:
            comp = components[active_comp]
            move = (arrays["close"][i] * (1 - slip_eff) - entry_price) / entry_price if pos > 0 else (entry_price - arrays["close"][i] * (1 + slip_eff)) / entry_price
            unreal = move * notional
            mfe, mae = max(mfe, move), min(mae, move)
            eq = cash * (1.0 + unreal)
            peak = max(peak, eq)
            mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)

            against = (pos > 0 and cmamba_dir_sig[i] == -1) or (pos < 0 and cmamba_dir_sig[i] == 1)
            confirm_streak = confirm_streak + 1 if against else 0

            reason = ""
            if gate_enabled and confirm_streak >= int(confirm_bars):
                reason = "cmamba_reversal_gate"
            elif take_profit > 0.0 and move >= take_profit:
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
                             "source_component": active_comp, "reason": reason,
                             "win": int(cash > entry_equity), "trade_return": float(trade_return),
                             "notional": float(notional), "margin_fraction": float(margin_fraction),
                             "leverage": float(leverage_v)})
                pos, active_comp = 0, None
                continue
            continue

        for name in router.PRIORITY:
            comp = components[name]
            side = int(comp["dec"]["side"].iloc[i])
            if side == 0 or not bool(omega._active(comp["dec"]).iloc[i] if hasattr(omega._active(comp["dec"]), "iloc") else omega._active(comp["dec"])[i]):
                continue
            row_margin, row_leverage = float(comp["margin"][i]), float(comp["leverage"][i])
            if row_margin <= 0.0:
                continue
            scale = router.SCALE_MAP.get(f"{name}_{'L' if side > 0 else 'S'}", 1.0)
            row_leverage = min(row_leverage * scale, router.LEVERAGE_CAP)
            row_notional = min(row_margin * row_leverage, router.NOTIONAL_CAP)
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
            cash -= cash * fee_eff * notional
            mfe = mae = 0.0
            confirm_streak = 0
            break

    return {"reason_counts": reasons}, pd.DataFrame(rows)


def _compound(returns: np.ndarray) -> dict:
    curve = np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    peak = np.maximum.accumulate(curve)
    dd = curve / np.maximum(peak, 1e-12) - 1.0
    return {"pnl": float((curve[-1] - 1.0) * 100.0), "mdd": float(dd.min() * 100.0),
            "trades": int(len(returns)), "wr": float((returns > 0).mean()) if len(returns) else 0.0}


def run_oos():
    device = retest.DEVICE
    ext_frame = retest.load_frame_current("2026-01-01", "2026-06-30")
    fee, slip = router.omega._load_fee_slip()
    end_ts = ext_frame["timestamp"].iloc[-1]
    cmamba_sig = _cmamba_dir_signal(ext_frame, CMAMBA_2026)

    components = {}
    for name, cfg in retest.COMPONENTS.items():
        pred_csv = router.OUT_DIR / name / f"oos_predictions_{cfg['q_tag']}.csv"
        pred = pd.read_csv(pred_csv)
        pred["timestamp"] = pd.to_datetime(pred["timestamp"])
        truncated = pred[pred["timestamp"] <= end_ts].reset_index(drop=True)
        tmp_csv = Path(f"/tmp/{name}_oos_predictions_truncated_reversal.csv")
        truncated.to_csv(tmp_csv, index=False)
        components[name] = router.prepare_component(ext_frame, tmp_csv, cfg, device)

    configs = [("no_gate (baseline)", False, 1)] + [(f"confirm_bars={cb}", True, cb) for cb in CONFIRM_SWEEP]
    for label, gate, cb in configs:
        _, ledger = greedy_replay_with_reversal_gate(ext_frame, components, cmamba_sig, fee=fee, slip=slip,
                                                       cost_mult=retest.COST_MULT, device=device, gate_enabled=gate,
                                                       confirm_bars=cb)
        r = _compound(ledger["trade_return"].to_numpy()) if not ledger.empty else {"pnl":0,"mdd":0,"trades":0,"wr":0}
        print(f"OOS {label}: {r}", flush=True)
        if not ledger.empty:
            print("  reason counts:", ledger["reason"].value_counts().to_dict(), flush=True)


def run_val():
    device = retest.DEVICE
    frame = valmod.load_val_frame()
    fee, slip = router.omega._load_fee_slip()
    cmamba_sig_full = _cmamba_dir_signal(frame, CMAMBA_2025)

    components = {}
    for name, cfg in retest.COMPONENTS.items():
        pred = pd.read_csv(valmod.VAL_PRED[name])
        pred = pred.rename(columns={c: c.replace("_expertdq_oof_", "_expertdq_") for c in pred.columns})
        pred["timestamp"] = pd.to_datetime(pred["timestamp"])
        pred = pred[(pred["timestamp"] >= valmod.START) & (pred["timestamp"] <= valmod.END)].reset_index(drop=True)
        common = frame["timestamp"].isin(pred["timestamp"])
        frame_c = frame[common].reset_index(drop=True) if not common.all() else frame
        pred = pred[pred["timestamp"].isin(frame_c["timestamp"])].reset_index(drop=True)
        tmp_pred = Path(f"/tmp/_val_{name}_aligned_reversal.csv")
        pred.to_csv(tmp_pred, index=False)
        components[name] = router.prepare_component(frame_c, tmp_pred, cfg, device)
        frame = frame_c
    cmamba_sig = _cmamba_dir_signal(frame, CMAMBA_2025)

    configs = [("no_gate (baseline)", False, 1)] + [(f"confirm_bars={cb}", True, cb) for cb in CONFIRM_SWEEP]
    for label, gate, cb in configs:
        _, ledger = greedy_replay_with_reversal_gate(frame, components, cmamba_sig, fee=fee, slip=slip,
                                                       cost_mult=retest.COST_MULT, device=device, gate_enabled=gate,
                                                       confirm_bars=cb)
        r = _compound(ledger["trade_return"].to_numpy()) if not ledger.empty else {"pnl":0,"mdd":0,"trades":0,"wr":0}
        print(f"VAL {label}: {r}", flush=True)
        if not ledger.empty:
            print("  reason counts:", ledger["reason"].value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    print("=== VAL (2025-10-01..12-31) ===", flush=True)
    run_val()
    print("\n=== OOS (2026-01-01..06-30) ===", flush=True)
    run_oos()
