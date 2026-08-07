"""Candidate 6: learned bar-level exit-timing model (regression-based / fitted return-to-go,
"offline RL"-style) for Omega4.6.1's zig075 component, in response to "isn't there an RL/DL
candidate?".

Why not full episode-level RL: VAL/OOS windows only contain 22-37 completed zig075 trades each --
far too few episodes for a policy-gradient/Q-learning agent to learn signal from noise (this is
the same sample-size ceiling that makes every trade-level statistic in this project's history
noisy). The exit_head already IS a trained deep-learning exit-timing model and was already tested
(Candidate 2, exit-threshold sweep) -- it failed at every non-frozen threshold on OOS.

What IS statistically viable: a NEW bar-level model, trained on the much larger population of
in-trade BARS (not trade-level outcomes) from the TRAIN window (2025-01-01..2025-09-30, strictly
before VAL's 2025-10-01 start -- no leakage into VAL/OOS). For every bar of every zig075 trade
(using the static TP/SL barrier, zig075-only ledger, no h48qual competition), record the bar-level
state (hold, move, mfe, mae, giveback, notional, leverage, take_profit, stop_loss, side) and a
Monte-Carlo return-to-go target: rtg = move_at_actual_exit - move_now (how much more/less favorable
the trade's move became between now and its natural TP/SL/exit_head conclusion). Fit a
HistGradientBoostingRegressor (same model family already used elsewhere in this project, e.g. the
L4 risk sidecar) to predict rtg from bar-level state. Exit rule: exit now if predicted rtg <=
threshold (threshold grid-searched on VAL). This is standard fitted-value / return-to-go
regression, the simplest form of offline RL.

Fresh-Forward-aware: model is fit ONLY on TRAIN (pre-VAL) bars. threshold is grid-searched on VAL
only. Frozen threshold is scored ONCE on OOS 2026-01-01..06-30. trading_bot.py /
trading_bot_modules/omega4_6_1_live.py are NOT touched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

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

OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega4_6_1_learned_exit_dl_20260707"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_START, TRAIN_END = "2025-01-01", "2025-09-30 23:59:59"
ZIG_BUNDLE_DIR = ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_20260629"
FEATURE_COLS = ["hold", "move", "mfe", "mae", "giveback", "notional", "leverage", "take_profit", "stop_loss", "side"]
THRESHOLD_GRID = [-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02]


def load_train_frame_and_component(device) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(valmod.BASE_2025, low_memory=False)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    overlay = pd.read_csv(valmod.WIDE24_2025, low_memory=False)
    overlay["timestamp"] = pd.to_datetime(overlay["timestamp"])
    cols = [c for c in overlay.columns if c != "timestamp"]
    frame = frame.merge(overlay[["timestamp", *cols]], on="timestamp", how="left", validate="one_to_one")
    frame = frame[(frame["timestamp"] >= TRAIN_START) & (frame["timestamp"] <= TRAIN_END)].reset_index(drop=True)

    pred = pd.read_csv(ZIG_BUNDLE_DIR / "train_predictions_q075.csv")
    pred = pred.rename(columns={c: c.replace("_expertdq_oof_", "_expertdq_") for c in pred.columns})
    pred["timestamp"] = pd.to_datetime(pred["timestamp"])
    pred = pred[(pred["timestamp"] >= TRAIN_START) & (pred["timestamp"] <= TRAIN_END)].reset_index(drop=True)
    common = frame["timestamp"].isin(pred["timestamp"])
    frame = frame[common].reset_index(drop=True)
    pred = pred[pred["timestamp"].isin(frame["timestamp"])].reset_index(drop=True)
    tmp = OUT_DIR / "_train_zig075_aligned.csv"
    pred.to_csv(tmp, index=False)
    cfg = retest.COMPONENTS["zig075"]
    comp = greedy.prepare_component(frame, tmp, cfg, device)
    return frame, comp


@torch.no_grad()
def replay_record_bars(frame: pd.DataFrame, comp: dict, *, fee: float, slip: float, cost_mult: float, device: torch.device) -> pd.DataFrame:
    """zig075-only static-barrier replay that ALSO records every open-trade bar's state, tagged
    with a trade_id, so a return-to-go target can be backfilled once each trade's exit is known."""
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    n = len(frame)
    fee_eff, slip_eff = float(fee) * float(cost_mult), float(slip) * float(cost_mult)
    pos = 0
    entry_price = 1.0
    entry_i = 0
    notional = leverage_v = 0.0
    take_profit = stop_loss = 0.0
    mfe = mae = 0.0
    trade_id = -1
    bar_rows: list[dict] = []
    trade_start_idx: list[int] = []

    for i in range(0, n - 2):
        if pos != 0:
            move = (arrays["close"][i] * (1 - slip_eff) - entry_price) / entry_price if pos > 0 else (entry_price - arrays["close"][i] * (1 + slip_eff)) / entry_price
            mfe, mae = max(mfe, move), min(mae, move)
            hold = max(i - entry_i, 0)
            giveback = (mfe - move) / max(abs(mfe), 1e-8) if mfe > 0.0 else 0.0
            bar_rows.append({"trade_id": trade_id, "hold": float(hold), "move": float(move), "mfe": float(mfe),
                              "mae": float(mae), "giveback": float(np.clip(giveback, 0.0, 10.0)),
                              "notional": float(notional), "leverage": float(leverage_v),
                              "take_profit": float(take_profit), "stop_loss": float(stop_loss), "side": float(pos)})

            reason = ""
            if take_profit > 0.0 and move >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and move <= -abs(stop_loss):
                reason = "stop_loss"
            else:
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
                pos = 0
            continue

        side = int(comp["dec"]["side"].iloc[i])
        if side == 0 or not bool(omega._active(comp["dec"]).iloc[i] if hasattr(omega._active(comp["dec"]), "iloc") else omega._active(comp["dec"])[i]):
            continue
        row_margin, row_leverage = float(comp["margin"][i]), float(comp["leverage"][i])
        if row_margin <= 0.0:
            continue
        scale = greedy.SCALE_MAP.get(f"zig075_{'L' if side > 0 else 'S'}", 1.0)
        row_leverage = min(row_leverage * scale, greedy.LEVERAGE_CAP)
        row_notional = min(row_margin * row_leverage, greedy.NOTIONAL_CAP)
        if row_notional <= 0.0:
            continue
        entry_px = arrays["open"][min(i + 1, n - 1)] * (1 + slip_eff if side > 0 else 1 - slip_eff)
        pos = side
        entry_price = float(entry_px)
        entry_i = min(i + 1, n - 1)
        leverage_v, notional = row_leverage, row_notional
        take_profit = float(comp["dec"]["take_profit"].iloc[i])
        stop_loss = float(comp["dec"]["stop_loss"].iloc[i])
        mfe = mae = 0.0
        trade_id += 1
        trade_start_idx.append(i)

    bars = pd.DataFrame(bar_rows)
    if bars.empty:
        return bars
    # backfill return-to-go target: move at the trade's LAST recorded bar (its actual exit move)
    exit_move = bars.groupby("trade_id")["move"].transform("last")
    bars["rtg"] = exit_move - bars["move"]
    return bars


@torch.no_grad()
def greedy_replay_learned_exit(frame: pd.DataFrame, comp: dict, model, threshold: float, *, fee: float, slip: float,
                                cost_mult: float, device: torch.device) -> pd.DataFrame:
    """zig075-only replay: TP hard cap unchanged; SL/exit_head replaced by the learned rtg model
    (exit when predicted rtg <= threshold)."""
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    n = len(frame)
    fee_eff, slip_eff = float(fee) * float(cost_mult), float(slip) * float(cost_mult)
    cash = 1.0
    pos = 0
    entry_price = entry_equity = 1.0
    entry_i = entry_signal_i = 0
    notional = leverage_v = margin_fraction = 0.0
    take_profit = stop_loss = 0.0
    mfe = mae = 0.0
    rows: list[dict] = []

    for i in range(0, n - 2):
        if pos != 0:
            move = (arrays["close"][i] * (1 - slip_eff) - entry_price) / entry_price if pos > 0 else (entry_price - arrays["close"][i] * (1 + slip_eff)) / entry_price
            mfe, mae = max(mfe, move), min(mae, move)
            hold = max(i - entry_i, 0)
            giveback = (mfe - move) / max(abs(mfe), 1e-8) if mfe > 0.0 else 0.0
            reason = ""
            if take_profit > 0.0 and move >= take_profit:
                reason = "take_profit"
            else:
                x = np.asarray([[hold, move, mfe, mae, np.clip(giveback, 0.0, 10.0), notional, leverage_v, take_profit, stop_loss, float(pos)]])
                pred_rtg = float(model.predict(x)[0])
                if pred_rtg <= threshold:
                    reason = "learned_exit"
            if reason:
                exit_px = arrays["close"][i] * (1 - slip_eff if pos > 0 else 1 + slip_eff)
                raw_exit = (exit_px - entry_price) / entry_price if pos > 0 else (entry_price - exit_px) / entry_price
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * fee_eff * notional
                trade_return = cash / max(entry_equity, 1e-12) - 1.0
                rows.append({"entry_signal_i": entry_signal_i, "entry_i": entry_i, "exit_i": i,
                             "entry_timestamp": str(frame["timestamp"].iloc[entry_signal_i]),
                             "exit_timestamp": str(frame["timestamp"].iloc[i]), "side": int(pos),
                             "source_component": "zig075", "reason": reason,
                             "win": int(cash > entry_equity), "trade_return": float(trade_return),
                             "notional": float(notional), "margin_fraction": float(margin_fraction),
                             "leverage": float(leverage_v)})
                pos = 0
                continue
            continue

        side = int(comp["dec"]["side"].iloc[i])
        if side == 0 or not bool(omega._active(comp["dec"]).iloc[i] if hasattr(omega._active(comp["dec"]), "iloc") else omega._active(comp["dec"])[i]):
            continue
        row_margin, row_leverage = float(comp["margin"][i]), float(comp["leverage"][i])
        if row_margin <= 0.0:
            continue
        scale = greedy.SCALE_MAP.get(f"zig075_{'L' if side > 0 else 'S'}", 1.0)
        row_leverage = min(row_leverage * scale, greedy.LEVERAGE_CAP)
        row_notional = min(row_margin * row_leverage, greedy.NOTIONAL_CAP)
        row_leverage = row_notional / max(row_margin, 1e-12)
        if row_notional <= 0.0:
            continue
        entry_px = arrays["open"][min(i + 1, n - 1)] * (1 + slip_eff if side > 0 else 1 - slip_eff)
        pos = side
        entry_price, entry_equity = float(entry_px), cash
        entry_i, entry_signal_i = min(i + 1, n - 1), i
        margin_fraction, leverage_v, notional = row_margin, row_leverage, row_notional
        take_profit = float(comp["dec"]["take_profit"].iloc[i])
        stop_loss = float(comp["dec"]["stop_loss"].iloc[i])
        cash -= cash * fee_eff * notional
        mfe = mae = 0.0

    return pd.DataFrame(rows)


def main() -> int:
    device = retest.DEVICE
    fee, slip = omega._load_fee_slip()

    print("Loading TRAIN (2025-01-01..09-30) frame + zig075 component...", flush=True)
    train_frame, train_comp = load_train_frame_and_component(device)
    bars = replay_record_bars(train_frame, train_comp, fee=fee, slip=slip, cost_mult=retest.COST_MULT, device=device)
    n_trades = int(bars["trade_id"].nunique()) if not bars.empty else 0
    print(f"TRAIN bars collected: {len(bars)} across {n_trades} trades", flush=True)
    if bars.empty or n_trades < 5:
        raise RuntimeError(f"insufficient TRAIN trades ({n_trades}) to fit a bar-level exit model")

    model = HistGradientBoostingRegressor(max_iter=200, max_depth=4, learning_rate=0.05, random_state=0)
    model.fit(bars[FEATURE_COLS].to_numpy(), bars["rtg"].to_numpy())
    train_r2 = model.score(bars[FEATURE_COLS].to_numpy(), bars["rtg"].to_numpy())
    print(f"Fitted rtg regressor, train R^2={train_r2:.4f}", flush=True)

    val_frame_raw = valmod.load_val_frame()
    val_frame, val_components = {}, {}
    val_frame_full, val_components_full = None, None
    # reuse existing loader for BOTH components so we can compute the FULL-router baseline too
    from train_eval_omega4_6_1_trailing_exit_20260707 import load_components  # noqa: E402
    val_frame_full, val_components_full = load_components(val_frame_raw, device, val=True)

    orig_priority = greedy.PRIORITY
    greedy.PRIORITY = ("h48qual", "zig075")
    _, full_val_ledger = greedy.greedy_replay(val_frame_full, val_components_full, fee=fee, slip=slip, cost_mult=retest.COST_MULT, device=device)
    greedy.PRIORITY = orig_priority
    baseline_val = _metrics(full_val_ledger, val_frame_full, apply_gate=True)
    print(f"VAL baseline (FULL router, static TP/SL): {baseline_val}", flush=True)

    zig_comp_val = val_components_full["zig075"]
    grid_results = []
    for th in THRESHOLD_GRID:
        lg = greedy_replay_learned_exit(val_frame_full, zig_comp_val, model, th, fee=fee, slip=slip, cost_mult=retest.COST_MULT, device=device)
        m = _metrics(lg, val_frame_full, apply_gate=True)
        grid_results.append({"threshold": th, **m})
        print(f"  learned_exit threshold={th:+.3f} (zig075-only) -> pnl={m['pnl']:+7.2f}% mdd={m['mdd']:+6.2f}% n={m['trades']:2d} wr={m['wr']:.3f}", flush=True)

    # zig075-only static baseline for a fair apples-to-apples comparison
    greedy.PRIORITY = ("zig075",)
    _, zonly_static_val = greedy.greedy_replay(val_frame_full, {"zig075": zig_comp_val}, fee=fee, slip=slip, cost_mult=retest.COST_MULT, device=device)
    greedy.PRIORITY = orig_priority
    zonly_static_val_m = _metrics(zonly_static_val, val_frame_full, apply_gate=True)
    print(f"VAL zig075-only STATIC baseline: {zonly_static_val_m}", flush=True)

    grid_results.sort(key=lambda r: r["pnl"], reverse=True)
    best = grid_results[0]
    print(f"\nBest VAL learned-exit threshold: {best}", flush=True)
    adopt = bool(best["pnl"] > zonly_static_val_m["pnl"])
    print(f"Decision (VAL-only, pre-registered, vs zig075-only static): {'ADOPT' if adopt else 'REJECT'}", flush=True)

    # ---- OOS one-shot confirm ----
    oos_frame_raw = retest.load_frame_current("2026-01-01", "2026-06-30")
    oos_frame, oos_components = load_components(oos_frame_raw, device, val=False)
    zig_comp_oos = oos_components["zig075"]

    greedy.PRIORITY = ("zig075",)
    _, zonly_static_oos = greedy.greedy_replay(oos_frame, {"zig075": zig_comp_oos}, fee=fee, slip=slip, cost_mult=retest.COST_MULT, device=device)
    greedy.PRIORITY = orig_priority
    zonly_static_oos_m = _metrics(zonly_static_oos, oos_frame, apply_gate=True)

    lg_oos = greedy_replay_learned_exit(oos_frame, zig_comp_oos, model, best["threshold"], fee=fee, slip=slip, cost_mult=retest.COST_MULT, device=device)
    frozen_oos_m = _metrics(lg_oos, oos_frame, apply_gate=True)
    print(f"\nOOS zig075-only STATIC baseline: {zonly_static_oos_m}", flush=True)
    print(f"OOS frozen learned-exit (threshold={best['threshold']}): {frozen_oos_m}", flush=True)

    result = {
        "model_id": "omega4_6_1_learned_exit_dl_20260707",
        "train_trades": n_trades, "train_bars": len(bars), "train_r2": train_r2,
        "grid": grid_results,
        "val": {"full_router_baseline": baseline_val, "zig075_only_static": zonly_static_val_m, "best_learned_exit": best},
        "oos": {"zig075_only_static": zonly_static_oos_m, "frozen_learned_exit": frozen_oos_m},
        "adopt_decision_val_only": adopt,
    }
    (OUT_DIR / "result.json").write_text(json.dumps(result, indent=2))
    print(f"\nWrote {OUT_DIR / 'result.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
