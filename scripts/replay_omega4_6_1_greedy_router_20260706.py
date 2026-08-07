#!/usr/bin/env python3
"""Genuine GREEDY, single-account, interleaved bar-by-bar replay of Omega4.6.1 (h48qual > zig075
priority), matching what a REAL live system can actually do -- unlike every prior backtest in
this lineage, which simulated h48qual and zig075 as two INDEPENDENT full ledgers (each with its
own imaginary 100% capital) and reconciled overlaps post-hoc via
build_omega_plus_t12_livepass_candidate_20260630.py::priority_route(). That reconciliation cannot
be replicated by a real-time system (it requires knowing both components' full counterfactual
futures in advance). Runtime-native parity testing (see
trading_bot_modules/omega4_6_1_duration_gate_live_draft_20260706.py) found the greedy live
adapter's decisions differ from the offline priority-routed ledger on 8 of 33 combined-OOS trades
(24%) -- this script re-derives the PnL a real greedy live system would actually achieve, so it is
not silently overstated.

Single shared position slot: at each bar with no position open, try h48qual's decision first (if
active this bar, take it); otherwise try zig075. Once a position opens from a given component, its
OWN exit-head model + TP/SL barrier govern the exit (matching how each component's own frozen
exit-head/contract works) -- another component's competing signal is ignored while a position is
open (a real account only has one position budget here, consistent with the router's own
intent). Duration gate applied on top exactly as before (VAL-reselected 2026-07-06 threshold).
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as sidecar  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706"
SCALE_MAP = {"h48qual_L": 0.38, "h48qual_S": 2.499, "zig075_L": 2.446, "zig075_S": 2.478}
LEVERAGE_CAP, NOTIONAL_CAP, LIVE_RISK_SCALE = 5.0, 1.8, 1.0
PRIORITY = ("h48qual", "zig075")
DURATION_THRESHOLD = 0.005417


def prepare_component(frame: pd.DataFrame, pred_csv: Path, cfg: dict, device: torch.device):
    bundle = torch.load(cfg["bundle"], map_location="cpu", weights_only=False)
    base_cols, models = bundle["base_cols"], bundle["models"]
    pred = pd.read_csv(pred_csv)
    pred["timestamp"] = pd.to_datetime(pred["timestamp"])
    if not pred["timestamp"].equals(frame["timestamp"]):
        raise RuntimeError("timestamp mismatch")

    x = parent._base_input(frame, base_cols)
    dec_base = parent._to_decisions(pred, oof=False)
    dec, _ = atr_eval._apply_atr_safety_sltp(dec_base, frame, atr_window=cfg["atr_window"], tp_mult=cfg["tp_mult"],
                                              sl_mult=cfg["sl_mult"], min_tp=cfg["min_tp"], min_sl=cfg["min_sl"],
                                              max_tp=cfg["max_tp"], max_sl=cfg["max_sl"])
    atr = atr_eval._atr_pct(frame, cfg["atr_window"])
    loaded = parent._load_payloads(models, device=device)

    with open(cfg["sidecar_pkl"], "rb") as f:
        pkl = pickle.load(f)
    features = sidecar._risk_feature_frame(frame, pred, dec, base_cols, atr_pct=atr, feature_mode=pkl["risk_feature_mode"])
    x_all, _ = sidecar._feature_matrix(features, pkl["feature_columns"])
    side_all = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
    score = sidecar._predict_side_split_models(pkl["model"], x_all, side_all) if pkl["side_split_model"] else np.asarray(pkl["model"].predict(x_all))
    mapping = pkl["selected_mapping"]
    margin = sidecar._risk_margins(dec, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sidecar.MARGIN_CFG_KEYS})
    lev = sidecar._risk_leverage(dec, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sidecar.LEVERAGE_CFG_KEYS}) if pkl["dynamic_leverage"] else np.ones(len(dec))

    base_np, exit_runtime, pos_idx = sidecar._prepare_exit_runtime(x, loaded)
    route = hard._route_id(frame)
    return {
        "dec": dec, "atr": atr, "margin": margin, "leverage": lev, "base_np": base_np,
        "exit_runtime": exit_runtime, "pos_idx": pos_idx, "route": route, "exit_threshold": cfg["exit_threshold"],
    }


@torch.no_grad()
def greedy_replay(frame: pd.DataFrame, components: dict, *, fee: float, slip: float, cost_mult: float,
                  device: torch.device, trailing_activate_frac: float | None = None,
                  trailing_trail_frac: float | None = None) -> tuple[dict, pd.DataFrame]:
    """`trailing_*` added 2026-08-07: optional fixed-distance trailing stop carried over from the
    BTC gate-G1 result -- once favorable excursion reaches `activate * take_profit`, force an exit
    when profit falls `trail * |stop_loss|` below the peak. Checked after TP/SL and before the
    exit head, mirroring research_eth_omega461_exit_sweep_20260721.replay_exit_variant. Leaving
    both at None reproduces this script's original behaviour exactly."""
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
    armed = False
    trailing_enabled = trailing_activate_frac is not None and trailing_trail_frac is not None
    rows: list[dict] = []
    reasons: dict[str, int] = {}

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
            if not reason and trailing_enabled:
                if (not armed) and take_profit > 0.0 and mfe >= float(trailing_activate_frac) * take_profit:
                    armed = True
                if armed and mfe > 0.0 and move <= mfe - float(trailing_trail_frac) * abs(stop_loss):
                    reason = "trailing_stop"
            if not reason:
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

        # flat: try priority order
        for name in PRIORITY:
            if name not in components:
                continue
            comp = components[name]
            side = int(comp["dec"]["side"].iloc[i])
            if side == 0 or not bool(omega._active(comp["dec"]).iloc[i] if hasattr(omega._active(comp["dec"]), "iloc") else omega._active(comp["dec"])[i]):
                continue
            row_margin, row_leverage = float(comp["margin"][i]), float(comp["leverage"][i])
            if row_margin <= 0.0:
                continue
            scale = SCALE_MAP.get(f"{name}_{'L' if side > 0 else 'S'}", 1.0)
            row_leverage = min(row_leverage * scale, LEVERAGE_CAP)
            row_notional = min(row_margin * row_leverage, NOTIONAL_CAP)
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
            armed = False
            break

    return {"reason_counts": reasons}, pd.DataFrame(rows)


def main() -> int:
    device = retest.DEVICE
    ext_frame = retest.load_frame_current("2026-01-01", "2026-06-30")
    fee, slip = omega._load_fee_slip()

    components = {}
    for name, cfg in retest.COMPONENTS.items():
        pred_csv = OUT_DIR / name / f"oos_predictions_{cfg['q_tag']}.csv"
        components[name] = prepare_component(ext_frame, pred_csv, cfg, device)
        print(f"{name}: prepared, nonzero_side={(components[name]['dec']['side'] != 0).mean():.3f}", flush=True)

    _, ledger = greedy_replay(ext_frame, components, fee=fee, slip=slip, cost_mult=retest.COST_MULT, device=device)
    active = ledger.copy()
    returns = active["trade_return"].to_numpy()
    curve = np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    peak = np.maximum.accumulate(curve)
    dd = curve / np.maximum(peak, 1e-12) - 1.0
    no_gate = {"pnl": float((curve[-1] - 1.0) * 100.0), "mdd": float(dd.min() * 100.0), "trades": int(len(active)),
               "wr": float((returns > 0).mean()) if len(returns) else 0.0}
    print(f"\n=== Greedy live-realistic router (no duration gate) ===", flush=True)
    print(no_gate, flush=True)
    print("source_component counts:", active["source_component"].value_counts().to_dict(), flush=True)

    market = ext_frame[["timestamp", "ou_halflife"]]
    active["entry_timestamp_dt"] = pd.to_datetime(active["entry_timestamp"])
    active = active.merge(market.rename(columns={"timestamp": "entry_timestamp_dt"}), on="entry_timestamp_dt", how="left")
    hit = active["ou_halflife"] <= DURATION_THRESHOLD
    gated_returns = np.where(hit, 0.0, active["trade_return"])
    curve_g = np.concatenate([[1.0], np.cumprod(1.0 + gated_returns)])
    peak_g = np.maximum.accumulate(curve_g)
    dd_g = curve_g / np.maximum(peak_g, 1e-12) - 1.0
    n_active_after_gate = int((~hit).sum())
    with_gate = {"pnl": float((curve_g[-1] - 1.0) * 100.0), "mdd": float(dd_g.min() * 100.0),
                 "trades": n_active_after_gate, "wr": float((gated_returns[~hit] > 0).mean()) if n_active_after_gate else 0.0,
                 "skipped": int(hit.sum())}
    print(f"\n=== Greedy live-realistic router + duration gate (HONEST final number) ===", flush=True)
    print(with_gate, flush=True)

    ledger.to_csv(OUT_DIR / "greedy_router_ledger_extended.csv", index=False)
    (OUT_DIR / "greedy_router_result.json").write_text(json.dumps({
        "no_gate": no_gate, "with_gate": with_gate,
        "note": "genuine single-account greedy priority routing (h48qual>zig075), NOT the two-independent-ledgers-then-reconcile method used everywhere else in this lineage -- this is what a real live system can actually achieve",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
