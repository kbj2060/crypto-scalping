"""Candidate 6 (combo check): does the learned bar-level zig075 exit model from
train_eval_omega4_6_1_learned_exit_dl_20260707.py beat the TRUE live baseline (h48qual+zig075,
both static TP/SL, greedy priority h48qual>zig075) when run in the SAME combined single-account
router -- not just zig075 run alone? The standalone script only compared zig075-only variants;
this is the actual promotion-relevant comparison since the live model always runs both components
together. VAL selects, OOS confirms once. trading_bot.py is not touched."""
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
from test_omega4_6_1_drop_h48qual_20260706 import _metrics  # noqa: E402
from train_eval_omega4_6_1_trailing_exit_20260707 import load_components  # noqa: E402
import train_eval_omega4_6_1_learned_exit_dl_20260707 as cand6  # noqa: E402

OUT_DIR = cand6.OUT_DIR


@torch.no_grad()
def greedy_replay_combo(frame: pd.DataFrame, components: dict, model, threshold: float, *, fee: float, slip: float,
                         cost_mult: float, device: torch.device) -> pd.DataFrame:
    """Full combined router (h48qual priority > zig075), byte-identical to greedy.greedy_replay
    EXCEPT: when the open position's source_component is zig075, its SL/exit_head decision is
    replaced by the learned rtg model (TP hard cap unchanged for both components)."""
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    n = len(frame)
    fee_eff, slip_eff = float(fee) * float(cost_mult), float(slip) * float(cost_mult)
    cash = 1.0
    pos = 0
    active_comp = None
    entry_price = entry_equity = 1.0
    entry_i = entry_signal_i = 0
    notional = leverage_v = margin_fraction = 0.0
    take_profit = stop_loss = 0.0
    mfe = mae = 0.0
    rows: list[dict] = []

    for i in range(0, n - 2):
        if pos != 0:
            comp = components[active_comp]
            move = (arrays["close"][i] * (1 - slip_eff) - entry_price) / entry_price if pos > 0 else (entry_price - arrays["close"][i] * (1 + slip_eff)) / entry_price
            mfe, mae = max(mfe, move), min(mae, move)
            hold = max(i - entry_i, 0)
            giveback = (mfe - move) / max(abs(mfe), 1e-8) if mfe > 0.0 else 0.0

            reason = ""
            if take_profit > 0.0 and move >= take_profit:
                reason = "take_profit"
            elif active_comp == "zig075":
                x = np.asarray([[hold, move, mfe, mae, np.clip(giveback, 0.0, 10.0), notional, leverage_v, take_profit, stop_loss, float(pos)]])
                pred_rtg = float(model.predict(x)[0])
                if pred_rtg <= threshold:
                    reason = "learned_exit"
            else:
                if stop_loss > 0.0 and move <= -abs(stop_loss):
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
                exit_px = arrays["close"][i] * (1 - slip_eff if pos > 0 else 1 + slip_eff)
                raw_exit = (exit_px - entry_price) / entry_price if pos > 0 else (entry_price - exit_px) / entry_price
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * fee_eff * notional
                trade_return = cash / max(entry_equity, 1e-12) - 1.0
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

    return pd.DataFrame(rows)


def main() -> int:
    device = retest.DEVICE
    fee, slip = omega._load_fee_slip()

    print("Refitting TRAIN rtg model (same as candidate 6)...", flush=True)
    train_frame, train_comp = cand6.load_train_frame_and_component(device)
    bars = cand6.replay_record_bars(train_frame, train_comp, fee=fee, slip=slip, cost_mult=retest.COST_MULT, device=device)
    from sklearn.ensemble import HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor(max_iter=200, max_depth=4, learning_rate=0.05, random_state=0)
    model.fit(bars[cand6.FEATURE_COLS].to_numpy(), bars["rtg"].to_numpy())

    val_frame_raw = cand6.valmod.load_val_frame() if hasattr(cand6, "valmod") else None
    import replay_omega4_6_1_greedy_val_20260706 as valmod
    val_frame, val_components = load_components(valmod.load_val_frame(), device, val=True)

    orig_priority = greedy.PRIORITY
    greedy.PRIORITY = ("h48qual", "zig075")
    _, base_val_ledger = greedy.greedy_replay(val_frame, val_components, fee=fee, slip=slip, cost_mult=retest.COST_MULT, device=device)
    baseline_val = _metrics(base_val_ledger, val_frame, apply_gate=True)

    THRESHOLD_GRID = cand6.THRESHOLD_GRID
    grid = []
    for th in THRESHOLD_GRID:
        lg = greedy_replay_combo(val_frame, val_components, model, th, fee=fee, slip=slip, cost_mult=retest.COST_MULT, device=device)
        m = _metrics(lg, val_frame, apply_gate=True)
        grid.append({"threshold": th, **m})
        print(f"  COMBO learned-exit(zig075) threshold={th:+.3f} -> pnl={m['pnl']:+7.2f}% mdd={m['mdd']:+6.2f}% n={m['trades']:2d} wr={m['wr']:.3f}", flush=True)
    greedy.PRIORITY = orig_priority

    print(f"\nVAL baseline (FULL router, both static): {baseline_val}", flush=True)
    grid.sort(key=lambda r: r["pnl"], reverse=True)
    best = grid[0]
    print(f"Best VAL combo config: {best}", flush=True)
    adopt = bool(best["pnl"] > baseline_val["pnl"] and best["mdd"] > baseline_val["mdd"])
    print(f"Decision (VAL-only vs TRUE baseline): {'ADOPT' if adopt else 'REJECT'}", flush=True)

    oos_frame_raw = retest.load_frame_current("2026-01-01", "2026-06-30")
    oos_frame, oos_components = load_components(oos_frame_raw, device, val=False)
    greedy.PRIORITY = ("h48qual", "zig075")
    _, base_oos_ledger = greedy.greedy_replay(oos_frame, oos_components, fee=fee, slip=slip, cost_mult=retest.COST_MULT, device=device)
    baseline_oos = _metrics(base_oos_ledger, oos_frame, apply_gate=True)
    lg_oos = greedy_replay_combo(oos_frame, oos_components, model, best["threshold"], fee=fee, slip=slip, cost_mult=retest.COST_MULT, device=device)
    frozen_oos = _metrics(lg_oos, oos_frame, apply_gate=True)
    greedy.PRIORITY = orig_priority

    print(f"\nOOS baseline (FULL router, both static): {baseline_oos}", flush=True)
    print(f"OOS frozen COMBO (threshold={best['threshold']}): {frozen_oos}", flush=True)

    result = {"model_id": "omega4_6_1_learned_exit_dl_combo_20260707", "grid": grid,
              "val": {"baseline": baseline_val, "best_combo": best},
              "oos": {"baseline": baseline_oos, "frozen_combo": frozen_oos},
              "adopt_decision_val_only": adopt}
    (OUT_DIR / "combo_result.json").write_text(json.dumps(result, indent=2))
    print(f"\nWrote {OUT_DIR / 'combo_result.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
