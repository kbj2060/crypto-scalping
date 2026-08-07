"""Test CryptoMamba's future-regime prediction as an ENTRY-TIME-ONLY filter (read once at entry,
never re-checked during the hold) -- unlike the failed intra-trade reversal-exit gate
(eval_eth_cmamba_reversal_exit_gate_20260721.py, whipsawed by bar-to-bar noise), this mirrors
exactly the pattern that worked for chop soft-sizing: read the signal ONCE when the trade opens.

Rule: at the bar the router wants to open a position, if CryptoMamba's current +6bar prediction
directionally DISAGREES with the entry side (long entry but CryptoMamba predicts bear, or short
entry but predicts bull), skip that entry entirely (stay flat, wait for the next signal). Chop
predictions or agreement never block entry.
"""
from __future__ import annotations

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

CMAMBA_DIR = ROOT / "data/ensemble/supervised/regime3_cryptomamba_pred_h6_nocurrent_20260531"
CMAMBA_2025 = CMAMBA_DIR / "training_features_2025_regime3_cryptomamba_pred_h6_nocurrent_20260531.csv"
CMAMBA_2026 = CMAMBA_DIR / "training_features_2026_rebuilt_regime3_cryptomamba_pred_h6_nocurrent_20260531.csv"


def _cmamba_dir_signal(frame: pd.DataFrame, cmamba_path: Path) -> np.ndarray:
    cm = pd.read_csv(cmamba_path, parse_dates=["timestamp"])
    cm = cm[["timestamp", "regime3_cmamba_h6_future_pred_id"]]
    merged = frame[["timestamp"]].merge(cm, on="timestamp", how="left", validate="one_to_one")
    pred_id = merged["regime3_cmamba_h6_future_pred_id"].to_numpy()
    sig = np.zeros(len(merged), dtype=np.int64)
    sig[pred_id == 0] = 1   # bull
    sig[pred_id == 1] = -1  # bear
    return sig


def greedy_replay_entry_filter(frame: pd.DataFrame, components: dict, cmamba_dir_sig: np.ndarray, *,
                                fee: float, slip: float, cost_mult: float, device: torch.device,
                                filter_enabled: bool) -> tuple[dict, pd.DataFrame]:
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
    skipped_by_filter = 0

    for i in range(0, n - 2):
        if pos != 0:
            comp = components[active_comp]
            move = (arrays["close"][i] * (1 - slip_eff) - entry_price) / entry_price if pos > 0 else (entry_price - arrays["close"][i] * (1 + slip_eff)) / entry_price
            unreal = move * notional
            mfe, mae = max(mfe, move), min(mae, move)
            eq = cash * (1.0 + unreal)
            peak = max(peak, eq)
            mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)

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
            if filter_enabled and ((side > 0 and cmamba_dir_sig[i] == -1) or (side < 0 and cmamba_dir_sig[i] == 1)):
                skipped_by_filter += 1
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
            break

    return {"reason_counts": reasons, "skipped_by_filter": skipped_by_filter}, pd.DataFrame(rows)


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
        tmp_csv = Path(f"/tmp/{name}_oos_predictions_truncated_entryfilter.csv")
        truncated.to_csv(tmp_csv, index=False)
        components[name] = router.prepare_component(ext_frame, tmp_csv, cfg, device)

    for label, on in [("no_filter (baseline)", False), ("cmamba_entry_filter", True)]:
        meta, ledger = greedy_replay_entry_filter(ext_frame, components, cmamba_sig, fee=fee, slip=slip,
                                                    cost_mult=retest.COST_MULT, device=device, filter_enabled=on)
        r = _compound(ledger["trade_return"].to_numpy()) if not ledger.empty else {"pnl":0,"mdd":0,"trades":0,"wr":0}
        print(f"OOS {label}: {r}  skipped_by_filter={meta['skipped_by_filter']}", flush=True)


def run_val():
    device = retest.DEVICE
    frame = valmod.load_val_frame()
    fee, slip = router.omega._load_fee_slip()

    components = {}
    for name, cfg in retest.COMPONENTS.items():
        pred = pd.read_csv(valmod.VAL_PRED[name])
        pred = pred.rename(columns={c: c.replace("_expertdq_oof_", "_expertdq_") for c in pred.columns})
        pred["timestamp"] = pd.to_datetime(pred["timestamp"])
        pred = pred[(pred["timestamp"] >= valmod.START) & (pred["timestamp"] <= valmod.END)].reset_index(drop=True)
        common = frame["timestamp"].isin(pred["timestamp"])
        frame_c = frame[common].reset_index(drop=True) if not common.all() else frame
        pred = pred[pred["timestamp"].isin(frame_c["timestamp"])].reset_index(drop=True)
        tmp_pred = Path(f"/tmp/_val_{name}_aligned_entryfilter.csv")
        pred.to_csv(tmp_pred, index=False)
        components[name] = router.prepare_component(frame_c, tmp_pred, cfg, device)
        frame = frame_c
    cmamba_sig = _cmamba_dir_signal(frame, CMAMBA_2025)

    for label, on in [("no_filter (baseline)", False), ("cmamba_entry_filter", True)]:
        meta, ledger = greedy_replay_entry_filter(frame, components, cmamba_sig, fee=fee, slip=slip,
                                                    cost_mult=retest.COST_MULT, device=device, filter_enabled=on)
        r = _compound(ledger["trade_return"].to_numpy()) if not ledger.empty else {"pnl":0,"mdd":0,"trades":0,"wr":0}
        print(f"VAL {label}: {r}  skipped_by_filter={meta['skipped_by_filter']}", flush=True)


if __name__ == "__main__":
    print("=== VAL (2025-10-01..12-31) ===", flush=True)
    run_val()
    print("\n=== OOS (2026-01-01..06-30) ===", flush=True)
    run_oos()
