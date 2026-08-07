#!/usr/bin/env python3
"""Candidate: bar-by-bar dynamic SLTP overlay on Omega4.6.1's REAL live stack (no retraining of
h48qual/zig075 -- both components' frozen 3-head TabM bundles + risk-sizing sidecars + SCALE_MAP +
greedy router + duration gate are used exactly as-is, matching trading_bot_modules/omega4_6_1_live.py
and the same harness as train_eval_omega4_6_1_trailing_exit_20260707.py's candidate 5).

Motivation (user's request, this session): earlier attempts added a 4th TabM head jointly trained
with direction/quality/exit, which requires retraining the whole parent -- expensive and, per the
user, unnecessary. This instead trains a small, SEPARATE, fast gradient-boosted sidecar (not a
neural head) that predicts "how much realistic room is there right now, for either side" from
current market features alone (no position state) -- then re-queries it every bar WHILE a position
from either live component is open, ratcheting TP/SL tighter (never wider) toward its current
opinion. This directly targets the user's hypothesis: a barrier frozen at entry-time market
conditions doesn't react when the regime turns unfavorable mid-trade.

Sidecar training labels: dense (every bar, both hypothetical sides) forward-horizon MFE/MAE, same
label design validated in train_eval_eth_tabm_4head_sltp_v3_dense_dynamic_20260720.py (the sparse,
zigzag-segment-scoped label collapsed to the ATR floor; the dense 1-week-horizon label did not).
Training-label only -- never used as a live inference input themselves, only the sidecar MODEL's
own output (computed from current market features) is used live/in-replay.

Fresh-Forward-aware: sidecar trained ONLY on 2025-01-01..09-30 (strictly before VAL start), scored
on VAL (2025-10-01..12-31) and OOS (2026-01-01..06-30) with the exact same frozen entry/exit/sizing
components as the established harness. Stored-ledger based -> DIAGNOSTIC research score, not a
live-promotion claim. trading_bot.py / omega4_6_1_live.py are NOT touched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

# pandas>=2.2's "future.infer_string" (default True on pandas 3.x) loads CSV string columns as
# pandas.StringDtype instead of numpy object -- this silently breaks
# train_eval_omega4_2_risk_sidecar_20260622.py::_risk_feature_frame's `dtype == object` check for
# one-hot-encoding router_expert, turning it into a single all-zero column instead of raising.
# Found 2026-07-20 while wiring this script; scoped to this process only (not a shared-file edit).
pd.set_option("future.infer_string", False)

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as sidecar  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import replay_omega4_6_1_greedy_val_20260706 as valmod  # noqa: E402
from test_omega4_6_1_drop_h48qual_20260706 import _metrics  # noqa: E402
import train_eval_eth_tabm_4head_sltp_v3_dense_dynamic_20260720 as v3  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega4_6_1_dynamic_risk_sltp_20260720"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_TP, MAX_TP, MIN_SL, MAX_SL = 0.075, 0.22, 0.040, 0.12
TRAIN_START, TRAIN_END = "2025-01-01", "2025-09-30 23:59:59"
LABEL_HORIZON_BARS = 2016
TP_CAPTURE_FRAC, SL_CAPTURE_FRAC = 0.70, 0.90


def load_train_frame(base_cols: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(valmod.BASE_2025, low_memory=False)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    overlay = pd.read_csv(valmod.WIDE24_2025, low_memory=False)
    overlay["timestamp"] = pd.to_datetime(overlay["timestamp"])
    cols = [c for c in overlay.columns if c != "timestamp"]
    frame = frame.merge(overlay[["timestamp", *cols]], on="timestamp", how="left", validate="one_to_one")
    frame = frame[(frame["timestamp"] >= TRAIN_START) & (frame["timestamp"] <= TRAIN_END)].reset_index(drop=True)
    missing = [c for c in base_cols if c not in frame.columns]
    if missing:
        raise RuntimeError(f"training frame missing base_cols: {missing[:10]}")
    return frame


def train_risk_sidecar(train_frame: pd.DataFrame, base_cols: list[str]) -> dict[str, HistGradientBoostingRegressor]:
    x = parent._base_input(train_frame, base_cols).drop(columns=parent.POS_COLS)
    dense = v3._dense_horizon_mfe_mae(train_frame, horizon_bars=LABEL_HORIZON_BARS)
    tp_long = np.clip(np.abs(dense["mfe_long"]) * TP_CAPTURE_FRAC, MIN_TP, MAX_TP)
    sl_long = np.clip(np.abs(dense["mae_long"]) * SL_CAPTURE_FRAC, MIN_SL, MAX_SL)
    tp_short = np.clip(np.abs(dense["mfe_short"]) * TP_CAPTURE_FRAC, MIN_TP, MAX_TP)
    sl_short = np.clip(np.abs(dense["mae_short"]) * SL_CAPTURE_FRAC, MIN_SL, MAX_SL)
    models = {}
    for name, y in (("tp_long", tp_long), ("sl_long", sl_long), ("tp_short", tp_short), ("sl_short", sl_short)):
        m = HistGradientBoostingRegressor(max_depth=6, max_iter=200, learning_rate=0.05, random_state=260720)
        m.fit(x, y)
        models[name] = m
    return models, list(x.columns)


def predict_risk_sidecar(models: dict[str, HistGradientBoostingRegressor], feature_cols: list[str], frame: pd.DataFrame, base_cols: list[str]) -> dict[str, np.ndarray]:
    x = parent._base_input(frame, base_cols).drop(columns=parent.POS_COLS)[feature_cols]
    return {
        "tp_long": np.clip(models["tp_long"].predict(x), MIN_TP, MAX_TP),
        "sl_long": np.clip(models["sl_long"].predict(x), MIN_SL, MAX_SL),
        "tp_short": np.clip(models["tp_short"].predict(x), MIN_TP, MAX_TP),
        "sl_short": np.clip(models["sl_short"].predict(x), MIN_SL, MAX_SL),
    }


@torch.no_grad()
def greedy_replay_dynamic_risk(frame: pd.DataFrame, components: dict, risk_preds: dict[str, np.ndarray], *,
                                fee: float, slip: float, cost_mult: float, device: torch.device,
                                enable_ratchet: bool) -> tuple[dict, pd.DataFrame]:
    """Fork of greedy.greedy_replay with the exit-barrier block augmented: while a position is open,
    the dynamic-risk-sidecar's CURRENT-bar opinion (for the position's own side) is used to ratchet
    take_profit/stop_loss tighter (never wider) every bar. Entry logic, priority routing, caps, fees
    are byte-identical to the original."""
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
    ratchet_events = 0
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

            if enable_ratchet:
                fresh_tp = float(risk_preds["tp_long"][i] if pos > 0 else risk_preds["tp_short"][i])
                fresh_sl = float(risk_preds["sl_long"][i] if pos > 0 else risk_preds["sl_short"][i])
                if fresh_tp < take_profit - 1e-9 or fresh_sl < stop_loss - 1e-9:
                    ratchet_events += 1
                take_profit = min(take_profit, fresh_tp)
                stop_loss = min(stop_loss, fresh_sl)

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
            cash -= cash * fee_eff * notional
            mfe = mae = 0.0
            break

    return {"reason_counts": reasons, "ratchet_events": ratchet_events}, pd.DataFrame(rows)


def load_components(frame: pd.DataFrame, device, *, val: bool) -> tuple[pd.DataFrame, dict]:
    components = {}
    for cname, cfg in retest.COMPONENTS.items():
        if val:
            pred = pd.read_csv(valmod.VAL_PRED[cname])
            pred = pred.rename(columns={c: c.replace("_expertdq_oof_", "_expertdq_") for c in pred.columns})
            pred["timestamp"] = pd.to_datetime(pred["timestamp"])
            pred = pred[(pred["timestamp"] >= valmod.START) & (pred["timestamp"] <= valmod.END)].reset_index(drop=True)
        else:
            pred_csv = retest.EXT_PRED_DIR / cname / f"oos_predictions_{cfg['q_tag']}.csv"
            pred = pd.read_csv(pred_csv)
            pred["timestamp"] = pd.to_datetime(pred["timestamp"])
        common = frame["timestamp"].isin(pred["timestamp"])
        frame = frame[common].reset_index(drop=True)
        pred = pred[pred["timestamp"].isin(frame["timestamp"])].reset_index(drop=True)
        tag = "val" if val else "oos"
        tmp = OUT_DIR / f"_{tag}_{cname}_aligned.csv"
        pred.to_csv(tmp, index=False)
        components[cname] = greedy.prepare_component(frame, tmp, cfg, device)
    return frame, components


def main() -> int:
    device = retest.DEVICE
    fee, slip = omega._load_fee_slip()
    base_cols = list(torch.load(retest.COMPONENTS["zig075"]["bundle"], map_location="cpu", weights_only=False)["base_cols"])
    base_cols_h48 = list(torch.load(retest.COMPONENTS["h48qual"]["bundle"], map_location="cpu", weights_only=False)["base_cols"])
    if base_cols != base_cols_h48:
        raise RuntimeError("h48qual/zig075 base_cols differ -- dynamic-risk sidecar assumption (shared feature contract) is invalid")

    print("stage=train_sidecar", flush=True)
    train_frame = load_train_frame(base_cols)
    sidecar_models, feature_cols = train_risk_sidecar(train_frame, base_cols)
    print(f"sidecar trained on {len(train_frame)} rows ({TRAIN_START}..{TRAIN_END})", flush=True)

    print("stage=load_val", flush=True)
    val_frame_raw = valmod.load_val_frame()
    val_frame, val_components = load_components(val_frame_raw, device, val=True)
    val_risk_preds = predict_risk_sidecar(sidecar_models, feature_cols, val_frame, base_cols)

    print("stage=score_val", flush=True)
    _, val_ledger_base = greedy_replay_dynamic_risk(val_frame, val_components, val_risk_preds, fee=fee, slip=slip, cost_mult=retest.COST_MULT, device=device, enable_ratchet=False)
    diag_dyn, val_ledger_dyn = greedy_replay_dynamic_risk(val_frame, val_components, val_risk_preds, fee=fee, slip=slip, cost_mult=retest.COST_MULT, device=device, enable_ratchet=True)
    val_base_m = _metrics(val_ledger_base, val_frame, apply_gate=True)
    val_dyn_m = _metrics(val_ledger_dyn, val_frame, apply_gate=True)
    print(f"VAL baseline (real live stack, static TP/SL): {val_base_m}", flush=True)
    print(f"VAL dynamic-risk-sidecar ratchet: {val_dyn_m} ratchet_events={diag_dyn['ratchet_events']}", flush=True)

    print("stage=load_oos", flush=True)
    oos_frame_raw = retest.load_frame_current("2026-01-01", "2026-06-30")
    oos_frame, oos_components = load_components(oos_frame_raw, device, val=False)
    oos_risk_preds = predict_risk_sidecar(sidecar_models, feature_cols, oos_frame, base_cols)

    print("stage=score_oos", flush=True)
    _, oos_ledger_base = greedy_replay_dynamic_risk(oos_frame, oos_components, oos_risk_preds, fee=fee, slip=slip, cost_mult=retest.COST_MULT, device=device, enable_ratchet=False)
    diag_oos, oos_ledger_dyn = greedy_replay_dynamic_risk(oos_frame, oos_components, oos_risk_preds, fee=fee, slip=slip, cost_mult=retest.COST_MULT, device=device, enable_ratchet=True)
    oos_base_m = _metrics(oos_ledger_base, oos_frame, apply_gate=True)
    oos_dyn_m = _metrics(oos_ledger_dyn, oos_frame, apply_gate=True)
    print(f"OOS baseline (real live stack, static TP/SL): {oos_base_m}", flush=True)
    print(f"OOS dynamic-risk-sidecar ratchet: {oos_dyn_m} ratchet_events={diag_oos['ratchet_events']}", flush=True)

    result = {
        "model_id": "omega4_6_1_dynamic_risk_sltp_20260720",
        "design": "Real live-stack (h48qual+zig075, frozen bundles+sidecars+SCALE_MAP+greedy router+duration gate, NOT retrained) with a bar-by-bar dynamic SLTP overlay: a small separately-trained HistGradientBoostingRegressor sidecar (not a neural head) predicts, from current market features alone, a realistic TP/SL for either side; while a position is open, its current-bar opinion ratchets the barrier tighter (never wider).",
        "caveats": [
            "OOS window here is 2026-01-01..06-30 (this harness's established 'extended' convention), not the project's canonical 2026-01-01..03-31 -- consistent with train_eval_omega4_6_1_trailing_exit_20260707.py's own convention, but flagging the deviation from the strict canonical contract.",
            "Sidecar trained on 2025-01-01..09-30 only (strictly before VAL start) to avoid any leakage into VAL/OOS scoring.",
            "trade_ledgers_used_as_input=false; sidecar labels are dense forward-horizon MFE/MAE training targets only, computed once offline -- the ratchet mechanism at replay time uses only the sidecar's own live prediction from that bar's market features, never a saved ledger or future information.",
            "cost_mult=1.0 here (matching retest.COST_MULT, i.e. real fee/slip, not the 3x cost-stress convention used in this session's earlier from-scratch-retrain experiments) -- numbers are comparable to the registry's documented live baseline, unlike the earlier single-component quick-reproduction numbers.",
            "h48qual and zig075 entries/exits/sizing are completely unchanged from the live-wired model; only the TP/SL barrier value is touched by this overlay.",
        ],
        "sidecar_config": {"label_horizon_bars": LABEL_HORIZON_BARS, "tp_capture_frac": TP_CAPTURE_FRAC, "sl_capture_frac": SL_CAPTURE_FRAC, "train_start": TRAIN_START, "train_end": TRAIN_END},
        "val": {"baseline": val_base_m, "dynamic_risk_sidecar": val_dyn_m, "ratchet_events": diag_dyn["ratchet_events"]},
        "oos": {"baseline": oos_base_m, "dynamic_risk_sidecar": oos_dyn_m, "ratchet_events": diag_oos["ratchet_events"]},
    }
    (OUT_DIR / "result.json").write_text(json.dumps(result, indent=2, default=str))
    val_ledger_dyn.to_csv(OUT_DIR / "val_ledger_dynamic.csv", index=False)
    oos_ledger_dyn.to_csv(OUT_DIR / "oos_ledger_dynamic.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'result.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
