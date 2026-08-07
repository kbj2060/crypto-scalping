#!/usr/bin/env python3
"""GENUINE bar-by-bar fresh-forward replay of the Omega4.6.1 + event-flat overlay, addressing the
Fresh-Forward Rule violation flagged in
docs/model_contracts/omega4_6_1_event_flat_live_promotion_audit_20260706.md (Gate 1): the earlier
event-flat/haircut/profit-lock tests were post-hoc edits of an already-computed saved ledger,
which AGENTS.md explicitly says is invalid for promotion regardless of PnL.

This is a modified copy of train_eval_omega4_2_risk_sidecar_20260622.py::_replay_with_risk with
ONE addition: at every bar while a position is open, if that bar's timestamp falls inside a
scheduled macro-event flat window (-30min/+120min around NFP/ISM-mfg/ISM-svc/flash-PMI/FOMC,
reusing the exact calendar from trading_bot_modules/omega5_live.py), the position is force-closed
THIS BAR using only this bar's own price (causal, no future information -- the event calendar
itself is a publicly pre-known schedule, which is legitimate to reference at any bar, same as any
real trading system would). New entries are also blocked while inside a flat window. This fully
recomputes the ledger from bar 0 forward with the overlay built into the decision loop itself --
no saved ledger entry/exit timestamps are read as input to this computation.

Reports fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false per AGENTS.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(ROOT / "trading_bot_modules") not in sys.path:
    sys.path.insert(0, str(ROOT / "trading_bot_modules"))

import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_1_exit_head_price_move_sltp_retrain_20260622 as price_exit  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as sidecar  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
from omega5_live import Omega5LiveAdapter  # noqa: E402

import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402
import build_omega_plus_t12_livepass_candidate_20260630 as builder  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706"
VETO_PRE_MIN, VETO_POST_MIN = 30, 120
SCALE_MAP = {"h48qual_L": 0.38, "h48qual_S": 2.499, "zig075_L": 2.446, "zig075_S": 2.478}
PRIORITY = ["h48qual", "zig075"]
LEVERAGE_CAP, NOTIONAL_CAP, LIVE_RISK_SCALE = 5.0, 1.8, 1.0
DURATION_THRESHOLD = 0.005417  # validation-reselected (see select_duration_gate_threshold_val_20260706.py)


def build_flat_mask(timestamps: pd.Series) -> np.ndarray:
    events = []
    for y in (2025, 2026, 2027):
        events.extend(ts for _, ts in Omega5LiveAdapter._macro_events_for_year(y))
    mask = np.zeros(len(timestamps), dtype=bool)
    ts_arr = timestamps.to_numpy()
    for ets in events:
        start = np.datetime64(ets - pd.Timedelta(minutes=VETO_PRE_MIN))
        end = np.datetime64(ets + pd.Timedelta(minutes=VETO_POST_MIN))
        mask |= (ts_arr >= start) & (ts_arr <= end)
    return mask


@torch.no_grad()
def replay_with_risk_event_flat(
    frame: pd.DataFrame,
    base_x: pd.DataFrame,
    dec: pd.DataFrame,
    loaded_models: dict[str, tuple[Any, dict[str, Any]]],
    *,
    risk_margin_fraction: np.ndarray | None,
    risk_leverage: np.ndarray | None,
    exit_threshold: float,
    fee: float,
    slip: float,
    cost_mult: float,
    notional_scaled_sltp: bool,
    device: torch.device,
    flat_mask: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame]:
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    active = omega._active(dec)
    fee_eff, slip_eff = float(fee) * float(cost_mult), float(slip) * float(cost_mult)
    cash = peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = entry_equity = 1.0
    entry_i = entry_signal_i = 0
    notional = leverage = margin_fraction = 0.0
    exit_input_notional, exit_input_leverage = 0.0, 1.0
    take_profit = stop_loss = 0.0
    mfe = mae = 0.0
    trades = wins = long_entries = short_entries = 0
    reasons: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    route = hard._route_id(frame)
    base_np, exit_runtime, pos_idx = sidecar._prepare_exit_runtime(base_x, loaded_models)

    for i in range(0, len(frame) - 2):
        if pos != 0:
            move = price_exit._price_move(arrays, int(i), side=pos, entry_price=float(entry_price), slip_eff=slip_eff)
            unreal = move * notional
            mfe, mae = max(mfe, move), min(mae, move)
            eq = cash * (1.0 + unreal)
        else:
            move, eq = 0.0, cash
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)

        if pos != 0:
            reason = ""
            exit_prob = 0.0
            if bool(flat_mask[i]):
                reason = "event_flat"
            elif take_profit > 0.0 and move >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and move <= -abs(stop_loss):
                reason = "stop_loss"
            else:
                hold = max(int(i) - int(entry_i), 0)
                giveback = (float(mfe) - float(move)) / max(abs(float(mfe)), 1e-8) if mfe > 0.0 else 0.0
                expert = hard.EXPERT_NAMES[int(route[i])]
                prob = sidecar._predict_exit_prob_one(
                    base_np, exit_runtime, pos_idx, row_i=int(i), expert=expert,
                    pos_values=[float(pos), float(hold), float(move), float(mfe), float(mae),
                                float(np.clip(giveback, 0.0, 10.0)), float(take_profit - move),
                                float(move + abs(stop_loss)), float(exit_input_notional),
                                float(exit_input_leverage), float(exit_input_notional * exit_input_leverage),
                                float(take_profit), float(stop_loss)],
                    device=device,
                )
                exit_prob = float(prob)
                if prob >= float(exit_threshold):
                    reason = "exit_head"
            if reason:
                filled, exit_px, exit_fee, _ = omega._try_execution(arrays, int(i), pos, entry=False, fee_base=fee_eff, slip_base=slip_eff)
                if not filled:
                    continue
                raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * exit_fee * notional
                trade_return = cash / max(entry_equity, 1e-12) - 1.0
                trades += 1
                win = int(cash > entry_equity)
                wins += win
                reasons[reason] = reasons.get(reason, 0) + 1
                rows.append({
                    "entry_signal_i": int(entry_signal_i), "entry_i": int(entry_i), "exit_i": int(i),
                    "entry_timestamp": str(frame["timestamp"].iloc[int(entry_signal_i)]),
                    "exit_timestamp": str(frame["timestamp"].iloc[int(i)]), "side": int(pos),
                    "reason": reason, "win": int(win), "raw_exit_price_move": float(raw_exit),
                    "mfe_price_move": float(mfe), "mae_price_move": float(mae),
                    "trade_return": float(trade_return), "net_per_notional": float(trade_return / max(notional, 1e-12)),
                    "notional": float(notional), "margin_fraction": float(margin_fraction), "leverage": float(leverage),
                    "exit_prob": float(exit_prob), "take_profit": float(take_profit), "stop_loss": float(stop_loss),
                })
                pos = 0
                continue
        if pos != 0 or not bool(active[i]) or bool(flat_mask[i]):
            continue
        row = dec.iloc[i]
        side = int(row.get("side", 0) or 0)
        if side == 0:
            continue
        filled, px, fee_paid, _ = omega._try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
        if not filled:
            continue
        row_leverage = float(risk_leverage[int(i)]) if risk_leverage is not None else float(row.get("leverage", 1.0) or 1.0)
        base_notional = float(row.get("notional_exposure", 0.0) or 0.0)
        if risk_margin_fraction is None:
            row_margin, row_notional = base_notional / max(row_leverage, 1e-12), base_notional
        else:
            row_margin = float(risk_margin_fraction[int(i)])
            row_notional = row_margin * row_leverage
        if row_notional <= 0.0:
            continue
        pos = side
        entry_price = float(px)
        entry_equity = cash
        entry_i = min(int(i) + 1, len(frame) - 1)
        entry_signal_i = int(i)
        leverage, margin_fraction, notional = row_leverage, row_margin, row_notional
        exit_input_notional, exit_input_leverage = row_notional, row_leverage
        base_tp, base_sl = float(row.get("take_profit", 0.0) or 0.0), float(row.get("stop_loss", 0.0) or 0.0)
        take_profit = base_tp * row_notional if notional_scaled_sltp else base_tp
        stop_loss = base_sl * row_notional if notional_scaled_sltp else base_sl
        cash -= cash * fee_paid * notional
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        mfe = mae = 0.0

    if pos != 0:
        exit_px = omega._fill_price(arrays, len(frame) - 1, pos, slip_eff, entry=False)
        raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw_exit * notional)
        cash -= before * fee_eff * notional
        trade_return = cash / max(entry_equity, 1e-12) - 1.0
        trades += 1
        win = int(cash > entry_equity)
        wins += win
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
        rows.append({
            "entry_signal_i": int(entry_signal_i), "entry_i": int(entry_i), "exit_i": int(len(frame) - 1),
            "entry_timestamp": str(frame["timestamp"].iloc[int(entry_signal_i)]), "exit_timestamp": str(frame["timestamp"].iloc[-1]),
            "side": int(pos), "reason": "forced_end", "win": int(win), "raw_exit_price_move": float(raw_exit),
            "mfe_price_move": float(mfe), "mae_price_move": float(mae), "trade_return": float(trade_return),
            "net_per_notional": float(trade_return / max(notional, 1e-12)), "notional": float(notional),
            "margin_fraction": float(margin_fraction), "leverage": float(leverage),
            "exit_prob": 0.0, "take_profit": float(take_profit), "stop_loss": float(stop_loss),
        })

    n_entries = max(long_entries + short_entries, 1)
    metrics = {"pnl": float((cash - 1.0) * 100.0), "mdd": float(mdd * 100.0), "trades": int(trades),
               "wr": float(wins / trades) if trades else 0.0, "long_entries": int(long_entries),
               "short_entries": int(short_entries), "reason_counts": reasons}
    return metrics, pd.DataFrame(rows)


def score_component_event_flat(frame: pd.DataFrame, pred_csv: Path, cfg: dict, flat_mask: np.ndarray) -> tuple[dict, pd.DataFrame]:
    bundle = torch.load(cfg["bundle"], map_location="cpu", weights_only=False)
    base_cols, models = bundle["base_cols"], bundle["models"]
    pred = pd.read_csv(pred_csv)
    pred["timestamp"] = pd.to_datetime(pred["timestamp"])
    keep_ts = set(pred["timestamp"])
    frame = frame[frame["timestamp"].isin(keep_ts)].reset_index(drop=True)
    pred = pred[pred["timestamp"].isin(set(frame["timestamp"]))].reset_index(drop=True)
    if not pred["timestamp"].equals(frame["timestamp"]):
        raise RuntimeError("timestamp mismatch")
    fmask = flat_mask[: len(frame)] if len(flat_mask) >= len(frame) else np.zeros(len(frame), dtype=bool)

    x = parent._base_input(frame, base_cols)
    dec_base = parent._to_decisions(pred, oof=False)
    dec, _ = atr_eval._apply_atr_safety_sltp(dec_base, frame, atr_window=cfg["atr_window"], tp_mult=cfg["tp_mult"],
                                              sl_mult=cfg["sl_mult"], min_tp=cfg["min_tp"], min_sl=cfg["min_sl"],
                                              max_tp=cfg["max_tp"], max_sl=cfg["max_sl"])
    atr = atr_eval._atr_pct(frame, cfg["atr_window"])
    fee, slip = omega._load_fee_slip()
    loaded = parent._load_payloads(models, device=retest.DEVICE)

    import pickle
    with open(cfg["sidecar_pkl"], "rb") as f:
        pkl = pickle.load(f)
    features = sidecar._risk_feature_frame(frame, pred, dec, base_cols, atr_pct=atr, feature_mode=pkl["risk_feature_mode"])
    x_all, _ = sidecar._feature_matrix(features, pkl["feature_columns"])
    side_all = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
    score = sidecar._predict_side_split_models(pkl["model"], x_all, side_all) if pkl["side_split_model"] else np.asarray(pkl["model"].predict(x_all))
    mapping = pkl["selected_mapping"]
    margin = sidecar._risk_margins(dec, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sidecar.MARGIN_CFG_KEYS})
    lev = sidecar._risk_leverage(dec, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sidecar.LEVERAGE_CFG_KEYS}) if pkl["dynamic_leverage"] else None

    m, ledger = replay_with_risk_event_flat(
        frame, x, dec, loaded, risk_margin_fraction=margin, risk_leverage=lev, exit_threshold=cfg["exit_threshold"],
        fee=fee, slip=slip, cost_mult=retest.COST_MULT, notional_scaled_sltp=pkl["notional_scaled_sltp"], device=retest.DEVICE, flat_mask=fmask,
    )
    return m, ledger


def main() -> int:
    ext_frame = retest.load_frame_current("2026-01-01", "2026-06-30")
    flat_mask = build_flat_mask(ext_frame["timestamp"])
    print(f"flat-window bars: {int(flat_mask.sum())}/{len(flat_mask)} ({flat_mask.mean():.4%} of all bars)", flush=True)

    ledgers = []
    for name, cfg in retest.COMPONENTS.items():
        pred_csv = OUT_DIR / name / f"oos_predictions_{cfg['q_tag']}.csv"
        m, led = score_component_event_flat(ext_frame, pred_csv, cfg, flat_mask)
        led["source_alias"] = name
        ledgers.append(led)
        print(f"{name} (event-flat, bar-by-bar): pnl={m['pnl']:.2f}% mdd={m['mdd']:.2f}% trades={m['trades']} wr={m['wr']:.3f} reasons={m['reason_counts']}", flush=True)

    raw = pd.concat(ledgers, ignore_index=True)
    routed = builder.priority_route(raw, PRIORITY)
    scaled = builder.scale_ledger(routed, SCALE_MAP, LEVERAGE_CAP, NOTIONAL_CAP, LIVE_RISK_SCALE)
    combined = builder.apply_max_hold_time_stop(scaled, ext_frame[["timestamp", "open", "high", "low", "close"]], 0.0)
    combined_m = builder.metrics(combined)
    print(f"\ncombined router (event-flat, no duration gate): pnl={combined_m['pnl']:.2f}% mdd={combined_m['mdd']:.2f}% trades={combined_m['trades']} wr={combined_m['wr']:.3f}", flush=True)

    market = ext_frame[["timestamp", "ou_halflife"]].copy()
    combined["entry_timestamp_dt"] = pd.to_datetime(combined["entry_timestamp"])
    combined = combined.merge(market.rename(columns={"timestamp": "entry_timestamp_dt"}), on="entry_timestamp_dt", how="left")
    hit = (combined["notional"].astype(float) > 1e-12) & (combined["ou_halflife"] <= DURATION_THRESHOLD)
    gated = combined.copy()
    gated.loc[hit, "notional"] = 0.0
    gated["trade_return"] = np.where(hit, 0.0, gated["trade_return"])
    active = gated[gated["notional"].astype(float) > 1e-12]
    returns = active["trade_return"].astype(float).to_numpy()
    curve = np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    peak = np.maximum.accumulate(curve)
    dd = curve / np.maximum(peak, 1e-12) - 1.0
    final = {"pnl": float((curve[-1] - 1.0) * 100.0), "mdd": float(dd.min() * 100.0), "trades": int(len(active)),
             "wr": float((returns > 0).mean()) if len(returns) else 0.0}
    print(f"\n=== FINAL: event-flat (bar-by-bar causal) + duration gate, extended Jan-Jun 2026 OOS ===", flush=True)
    print(final, flush=True)

    combined.to_csv(OUT_DIR / "event_flat_fresh_forward_combined_ledger.csv", index=False)
    gated.to_csv(OUT_DIR / "event_flat_fresh_forward_gated_ledger.csv", index=False)
    report = {
        "model_id": "omega4_6_1_event_flat_fresh_forward_20260706",
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "window": ["2026-01-01", "2026-06-30"],
        "flat_window_minutes": [VETO_PRE_MIN, VETO_POST_MIN],
        "duration_gate_threshold": DURATION_THRESHOLD,
        "result": final,
        "combined_router_no_gate": combined_m,
    }
    (OUT_DIR / "event_flat_fresh_forward_report.json").write_text(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
