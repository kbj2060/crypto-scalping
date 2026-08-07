#!/usr/bin/env python3
"""RESEARCH ONLY -- fresh-forward VAL-first->OOS funnel for the continuous position-weight
regressor trained by train_eth_omega461_continuous_weight_20260721.py.

Forked from research_eth_omega461_exit_ideas2_20260721.py's replay_idea loop (imported, not
copy-pasted): reuses its exact TP/SL/exit-head order and fill/cost model. Adds ONE new mechanism:
at every bar the position is open, the regressor predicts a target weight w in [0,1] from ONLY
current/past causal features (pos_state + base_cols + regime + atr, matching the training
feature set), snapped to the nearest of the 5 training levels {0,0.25,0.5,0.75,1.0}. If the
snapped level differs from the currently-applied level, the position's notional is rebalanced to
level*notional_initial and an incremental transaction cost (fee_eff+slip_eff, the SAME realistic
per-side cost rate used everywhere else in this harness) is charged against cash on the changed
portion -- rebalancing is NOT free. TP/SL price-move thresholds are unchanged (Futures Risk
Sizing Contract: they are price-move levels, independent of notional).

Mandatory sanity check: forcing the regressor's output to a constant 1.0 (weight_mode="const1")
must reproduce the baseline bit-for-bit (no rebalances ever fire => notional never changes =>
identical trade sequence/PnL/MDD to the unmodulated baseline).

Baselines (reused unmodified from ideas2/sweep, NOT recomputed):
  VAL  h48qual: pnl +5.45%  mdd -11.62%
  VAL  zig075:  pnl +40.31% mdd -13.07%
  OOS  h48qual: pnl +9.49%  mdd -6.54%
  OOS  zig075:  pnl +17.89% mdd -11.01%

Fresh-forward discipline: causal bar-by-bar single forward pass; no saved ledger used as input.
"""

from __future__ import annotations

import json
import pickle
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

import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import research_eth_omega461_exit_ideas2_20260721 as ideas2  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as rs  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402
import train_eth_omega461_continuous_weight_20260721 as train_mod  # noqa: E402

OUT_DIR = ROOT / "tmp/research_20260721"
BASELINES = ideas2.BASELINES
beats_baseline = ideas2.beats_baseline
MODEL_DIR = ROOT / "tmp/causal_regen_20260516"
LEVELS = train_mod.LEVELS


def load_weight_model(name: str) -> dict[str, Any]:
    with open(MODEL_DIR / f"eth_omega461_continuous_weight_20260721_{name}" / "model.pkl", "rb") as f:
        return pickle.load(f)


def snap_to_level(w: float) -> float:
    idx = int(np.argmin(np.abs(LEVELS - w)))
    return float(LEVELS[idx])


@torch.no_grad()
def replay_weight_modulation(
    frame: pd.DataFrame,
    base_x: pd.DataFrame,
    dec: pd.DataFrame,
    loaded_models: dict[str, tuple],
    regime: dict[str, np.ndarray],
    atr_pct: np.ndarray,
    *,
    risk_margin_fraction: np.ndarray,
    risk_leverage: np.ndarray,
    fee: float,
    slip: float,
    cost_mult: float,
    notional_scaled_sltp: bool,
    device: torch.device,
    exit_threshold: float = sweep.BASELINE_EXIT_THRESHOLD,
    weight_mode: str | None = None,  # None | "const1" | "model"
    weight_model: dict[str, Any] | None = None,
    rebalance_cooldown_bars: int | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Causal bar-by-bar replay. Identical structure to ideas2.replay_idea's baseline path (TP/SL
    then exit-head), with one addition: an optional per-bar weight-modulation rebalance step
    evaluated right after mfe/mae update, before the TP/SL/exit-head checks. Reduces exactly to
    the baseline when weight_mode is None. fresh_forward_bar_by_bar=true; no saved ledger used as
    input.
    """
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    active = omega._active(dec)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    turnover_rate = fee_eff + slip_eff
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = 0.0
    entry_equity = 1.0
    entry_i = 0
    entry_signal_i = 0
    notional = 0.0
    notional_initial = 0.0
    cur_level = 1.0
    leverage = 1.0
    margin_fraction = 0.0
    take_profit = 0.0
    stop_loss = 0.0
    mfe = 0.0
    mae = 0.0
    trades = 0
    wins = 0
    long_entries = 0
    short_entries = 0
    notional_sum = 0.0
    leverage_sum = 0.0
    margin_sum = 0.0
    rebalances = 0
    rebalance_cost_total = 0.0
    last_rebalance_i = -10**9
    reasons: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    route = hard._route_id(frame)
    from train_eval_omega1_2_tabm_diffusion_risk_20260603 import _try_execution as omega_try_execution, _fill_price as omega_fill_price
    import train_eval_omega4_1_exit_head_price_move_sltp_retrain_20260622 as price_exit

    base_np, exit_runtime, pos_idx = rs._prepare_exit_runtime(base_x, loaded_models)
    base_cols = list(base_x.columns)
    reg_model = weight_model["model"] if weight_model is not None else None
    reg_cols = weight_model["feature_columns"] if weight_model is not None else None

    for i in range(0, len(frame) - 2):
        if pos != 0:
            move = price_exit._price_move(arrays, int(i), side=pos, entry_price=float(entry_price), slip_eff=slip_eff)
            mfe = max(mfe, move)
            mae = min(mae, move)
        else:
            move = 0.0

        if pos != 0:
            hold = max(int(i) - int(entry_i), 0)
            # --- weight-modulation rebalance step (new mechanism) ---
            if weight_mode is not None:
                if weight_mode == "const1":
                    target_level = 1.0
                else:  # "model"
                    giveback_now = (float(mfe) - float(move)) / max(abs(float(mfe)), 1.0e-8) if mfe > 0.0 else 0.0
                    feat = {base_cols[j]: float(base_np[i, j]) for j in range(len(base_cols))}
                    feat.update({
                        "pos_side": float(pos), "pos_hold_bars": float(hold), "pos_unrealized": float(move),
                        "pos_mfe": float(mfe), "pos_mae": float(mae), "pos_giveback": float(np.clip(giveback_now, 0.0, 10.0)),
                        "pos_dist_to_tp": float(take_profit - move), "pos_dist_to_sl": float(move + abs(stop_loss)),
                        "pos_notional": float(notional), "pos_leverage": float(leverage), "pos_exposure": float(notional * leverage),
                        "pos_tp": float(take_profit), "pos_sl": float(stop_loss),
                        "regime_chop_prob": float(regime["chop"][i]), "regime_bull_prob": float(regime["bull"][i]),
                        "regime_bear_prob": float(regime["bear"][i]), "atr_pct": float(atr_pct[i]),
                    })
                    x_row = pd.DataFrame([feat]).reindex(columns=reg_cols, fill_value=0.0).to_numpy(dtype=np.float64)
                    w_pred = float(reg_model.predict(x_row)[0])
                    target_level = snap_to_level(float(np.clip(w_pred, 0.0, 1.0)))
                cooldown_ok = rebalance_cooldown_bars is None or (int(i) - last_rebalance_i) >= int(rebalance_cooldown_bars)
                if target_level != cur_level and cooldown_ok:
                    new_notional = notional_initial * target_level
                    delta = abs(new_notional - notional)
                    cash -= cash * turnover_rate * delta
                    notional = new_notional
                    cur_level = target_level
                    rebalances += 1
                    last_rebalance_i = int(i)

            reason = ""
            exit_prob = 0.0
            if take_profit > 0.0 and move >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and move <= -abs(stop_loss):
                reason = "stop_loss"
            if not reason:
                giveback = (float(mfe) - float(move)) / max(abs(float(mfe)), 1.0e-8) if mfe > 0.0 else 0.0
                expert = hard.EXPERT_NAMES[int(route[i])]
                prob = rs._predict_exit_prob_one(
                    base_np, exit_runtime, pos_idx, row_i=int(i), expert=expert,
                    pos_values=[
                        float(pos), float(hold), float(move), float(mfe), float(mae),
                        float(np.clip(giveback, 0.0, 10.0)), float(take_profit - move), float(move + abs(stop_loss)),
                        float(notional), float(leverage), float(notional * leverage), float(take_profit), float(stop_loss),
                    ],
                    device=device,
                )
                exit_prob = float(prob)
                if prob >= float(exit_threshold):
                    reason = "exit_head"
            if reason:
                filled, exit_px, exit_fee, _route = omega_try_execution(arrays, int(i), pos, entry=False, fee_base=fee_eff, slip_base=slip_eff)
                if not filled:
                    continue
                raw_exit = (exit_px - entry_price) / max(entry_price, 1.0e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1.0e-12)
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * exit_fee * notional
                trade_return = cash / max(entry_equity, 1.0e-12) - 1.0
                trades += 1
                win = int(cash > entry_equity)
                wins += win
                reasons[reason] = reasons.get(reason, 0) + 1
                rows.append({
                    "entry_signal_i": int(entry_signal_i), "entry_i": int(entry_i), "exit_i": int(i),
                    "entry_timestamp": str(frame["timestamp"].iloc[int(entry_signal_i)]),
                    "exit_timestamp": str(frame["timestamp"].iloc[int(i)]), "side": int(pos), "reason": reason,
                    "win": int(win), "raw_exit_price_move": float(raw_exit), "mfe_price_move": float(mfe),
                    "mae_price_move": float(mae), "trade_return": float(trade_return),
                    "net_per_notional": float(trade_return / max(notional_initial, 1.0e-12)), "notional": float(notional),
                    "margin_fraction": float(margin_fraction), "leverage": float(leverage),
                    "exit_prob": float(exit_prob), "take_profit": float(take_profit), "stop_loss": float(stop_loss),
                    "final_weight_level": float(cur_level),
                })
                pos = 0
                continue
        eq = cash if pos == 0 else cash * (1.0 + move * notional)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1.0e-12) - 1.0)
        if pos != 0 or not bool(active[i]):
            continue
        row = dec.iloc[i]
        side = int(row.get("side", 0) or 0)
        if side == 0:
            continue
        filled, px, fee_paid, _route = omega_try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
        if not filled:
            continue
        row_leverage = float(risk_leverage[int(i)])
        row_margin = float(risk_margin_fraction[int(i)])
        row_notional = row_margin * row_leverage
        if row_notional <= 0.0:
            continue
        pos = side
        entry_price = float(px)
        entry_equity = cash
        entry_i = min(int(i) + 1, len(frame) - 1)
        entry_signal_i = int(i)
        leverage = row_leverage
        margin_fraction = row_margin
        notional = row_notional
        notional_initial = row_notional
        cur_level = 1.0
        base_tp = float(row.get("take_profit", 0.0) or 0.0)
        base_sl = float(row.get("stop_loss", 0.0) or 0.0)
        if bool(notional_scaled_sltp):
            take_profit = base_tp * row_notional
            stop_loss = base_sl * row_notional
        else:
            take_profit = base_tp
            stop_loss = base_sl
        cash -= cash * fee_paid * notional
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        notional_sum += notional
        leverage_sum += leverage
        margin_sum += margin_fraction
        mfe = 0.0
        mae = 0.0

    if pos != 0:
        exit_px = omega_fill_price(arrays, len(frame) - 1, pos, slip_eff, entry=False)
        raw_exit = (exit_px - entry_price) / max(entry_price, 1.0e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1.0e-12)
        before = cash
        cash = cash * (1.0 + raw_exit * notional)
        cash -= before * fee_eff * notional
        trade_return = cash / max(entry_equity, 1.0e-12) - 1.0
        trades += 1
        win = int(cash > entry_equity)
        wins += win
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
        rows.append({
            "entry_signal_i": int(entry_signal_i), "entry_i": int(entry_i), "exit_i": int(len(frame) - 1),
            "entry_timestamp": str(frame["timestamp"].iloc[int(entry_signal_i)]), "exit_timestamp": str(frame["timestamp"].iloc[-1]),
            "side": int(pos), "reason": "forced_end", "win": int(win), "raw_exit_price_move": float(raw_exit),
            "mfe_price_move": float(mfe), "mae_price_move": float(mae), "trade_return": float(trade_return),
            "net_per_notional": float(trade_return / max(notional_initial, 1.0e-12)), "notional": float(notional),
            "margin_fraction": float(margin_fraction), "leverage": float(leverage), "exit_prob": 0.0,
            "take_profit": float(take_profit), "stop_loss": float(stop_loss), "final_weight_level": float(cur_level),
        })

    n_entries = max(long_entries + short_entries, 1)
    ledger = pd.DataFrame(rows)
    hold_bars = (ledger["exit_i"] - ledger["entry_i"]).clip(lower=0) if len(ledger) else pd.Series(dtype=float)
    return (
        {
            "pnl": float((cash - 1.0) * 100.0), "mdd": float(mdd * 100.0), "trades": int(trades),
            "wr": float(wins / trades) if trades else 0.0, "trades_per_day": float(trades / rs._duration_days(frame)),
            "avg_notional": float(notional_sum / n_entries), "avg_leverage": float(leverage_sum / n_entries),
            "avg_hold_bars": float(hold_bars.mean()) if len(hold_bars) else 0.0,
            "max_trade_pnl": float(ledger["trade_return"].max() * 100.0) if len(ledger) else 0.0,
            "p95_trade_pnl": float(ledger["trade_return"].quantile(0.95) * 100.0) if len(ledger) else 0.0,
            "long_entries": int(long_entries), "short_entries": int(short_entries), "exit_reasons": reasons,
            "rebalances": int(rebalances), "rebalances_per_trade": float(rebalances / trades) if trades else 0.0,
        },
        ledger,
    )


def run_one(p: dict, **kwargs) -> tuple[dict[str, Any], pd.DataFrame]:
    return replay_weight_modulation(
        p["frame"], p["x"], p["dec"], p["loaded"], p["regime"], p["atr"],
        risk_margin_fraction=p["margin"], risk_leverage=p["leverage"],
        fee=p["fee"], slip=p["slip"], cost_mult=sweep.COST_MULT, notional_scaled_sltp=p["notional_scaled_sltp"],
        device=sweep.DEVICE, **kwargs,
    )


def prep_with_regime(name: str, cfg: dict, frame: pd.DataFrame, pred_dir: Path, *, oof: bool) -> dict[str, Any]:
    pred = pred_dir / name / (f"validation_predictions_{cfg['q_tag']}.csv" if oof else f"oos_predictions_{cfg['q_tag']}.csv")
    p = sweep.prep_component(name, cfg, frame, pred, oof=oof)
    p["regime"] = ideas2.get_regime_arrays(p["frame"])
    p["atr"] = atr_eval._atr_pct(p["frame"], cfg["atr_window"])
    return p


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    weight_models = {name: load_weight_model(name) for name in sweep.COMPONENTS}

    val_frame = sweep.load_frame(sweep.VAL_START, sweep.VAL_END, base_csv=sweep.BASE_2025, wide24_csv=sweep.WIDE24_2025)
    print(f"VAL frame rows={len(val_frame)} range=[{val_frame['timestamp'].min()}, {val_frame['timestamp'].max()}]", flush=True)
    val_prepped = {name: prep_with_regime(name, cfg, val_frame, sweep.EXT_PRED_DIR, oof=True) for name, cfg in sweep.COMPONENTS.items()}

    oos_frame = sweep.load_frame(sweep.OOS_START, sweep.OOS_END, base_csv=sweep.BASE_2026, wide24_csv=sweep.WIDE24_2026)
    print(f"OOS frame rows={len(oos_frame)} range=[{oos_frame['timestamp'].min()}, {oos_frame['timestamp'].max()}]", flush=True)
    oos_prepped = {name: prep_with_regime(name, cfg, oos_frame, sweep.EXT_PRED_DIR, oof=False) for name, cfg in sweep.COMPONENTS.items()}

    # ---------------- Mandatory sanity checks ----------------
    print("stage=sanity_checks", flush=True)
    sanity_rows = []
    all_sane = True
    for split_name, prepped, base_key in (("VAL", val_prepped, "VAL"), ("OOS", oos_prepped, "OOS")):
        for name, p in prepped.items():
            b = BASELINES[(name, base_key)]
            m_noop, _ = run_one(p)  # weight_mode=None -> fully disabled no-op path
            m_const1, _ = run_one(p, weight_mode="const1")  # forces weight=1.0 always
            tol = 0.01
            ok_noop = abs(m_noop["pnl"] - b["pnl"]) < tol and abs(m_noop["mdd"] - b["mdd"]) < tol
            ok_const1_vs_baseline = abs(m_const1["pnl"] - b["pnl"]) < tol and abs(m_const1["mdd"] - b["mdd"]) < tol
            ok_const1_vs_noop = abs(m_const1["pnl"] - m_noop["pnl"]) < 1e-9 and abs(m_const1["mdd"] - m_noop["mdd"]) < 1e-9
            ok = ok_noop and ok_const1_vs_baseline and ok_const1_vs_noop
            all_sane = all_sane and ok
            sanity_rows.append({
                "split": split_name, "component": name, "baseline_pnl": b["pnl"], "baseline_mdd": b["mdd"],
                "noop_pnl": m_noop["pnl"], "noop_mdd": m_noop["mdd"], "const1_pnl": m_const1["pnl"], "const1_mdd": m_const1["mdd"],
                "const1_rebalances": m_const1["rebalances"], "ok": ok,
            })
            print(f"  sanity split={split_name} component={name} ok={ok} "
                  f"(noop pnl={m_noop['pnl']:.4f}/mdd={m_noop['mdd']:.4f} const1 pnl={m_const1['pnl']:.4f}/mdd={m_const1['mdd']:.4f} "
                  f"rebalances={m_const1['rebalances']} vs baseline pnl={b['pnl']:.4f}/mdd={b['mdd']:.4f})", flush=True)
    pd.DataFrame(sanity_rows).to_csv(OUT_DIR / "continuous_weight_sanity_checks.csv", index=False)
    if not all_sane:
        print("stage=STOP sanity checks FAILED -- see continuous_weight_sanity_checks.csv, not proceeding", flush=True)
        return 1
    print("stage=sanity_checks PASSED", flush=True)

    # ---------------- VAL: model-driven weight modulation ----------------
    print("stage=val", flush=True)
    val_rows = []
    winners = []
    for name, p in val_prepped.items():
        m, _ = run_one(p, weight_mode="model", weight_model=weight_models[name])
        row = {"component": name, **m}
        val_rows.append(row)
        cleared = beats_baseline(name, "VAL", m["pnl"], m["mdd"])
        b = BASELINES[(name, "VAL")]
        print(f"  component={name} -> VAL pnl={m['pnl']:.2f}% mdd={m['mdd']:.2f}% trades={m['trades']} "
              f"rebalances={m['rebalances']} ({m['rebalances_per_trade']:.2f}/trade) "
              f"(baseline pnl={b['pnl']:.2f}% mdd={b['mdd']:.2f}%) beats_baseline={cleared}", flush=True)
        if cleared:
            winners.append({"component": name})
    val_df = pd.DataFrame(val_rows)
    val_df["exit_reasons"] = val_df["exit_reasons"].apply(json.dumps)
    val_df.to_csv(OUT_DIR / "continuous_weight_VAL.csv", index=False)

    print(f"\nVAL winners (beat baseline on BOTH pnl and mdd): {len(winners)}", flush=True)

    # ---------------- Diagnostic: does a rebalance cooldown fix the churn? ----------------
    # Not part of the funnel decision (churn/cooldown was not pre-registered before seeing the
    # model-driven result) -- purely to distinguish "excessive turnover from noisy predictions"
    # from "the regressor genuinely has no signal" for the report.
    print("stage=cooldown_diagnostic_val", flush=True)
    cooldown_rows = []
    for name, p in val_prepped.items():
        b = BASELINES[(name, "VAL")]
        for cd in (20, 50, 100, 300):
            m, _ = run_one(p, weight_mode="model", weight_model=weight_models[name], rebalance_cooldown_bars=cd)
            cleared = beats_baseline(name, "VAL", m["pnl"], m["mdd"])
            cooldown_rows.append({"component": name, "cooldown_bars": cd, "pnl": m["pnl"], "mdd": m["mdd"],
                                   "trades": m["trades"], "rebalances": m["rebalances"], "cleared": cleared})
            print(f"  component={name} cooldown={cd} -> pnl={m['pnl']:.2f}% mdd={m['mdd']:.2f}% "
                  f"rebalances={m['rebalances']} (baseline pnl={b['pnl']:.2f}% mdd={b['mdd']:.2f}%) cleared={cleared}", flush=True)
    pd.DataFrame(cooldown_rows).to_csv(OUT_DIR / "continuous_weight_cooldown_diagnostic_VAL.csv", index=False)

    if not winners:
        print("stage=done no_val_winners -> no OOS confirmation run", flush=True)
        return 0

    # ---------------- OOS confirmation ----------------
    print("stage=oos_confirm", flush=True)
    oos_rows = []
    for w in winners:
        p = oos_prepped[w["component"]]
        m_cand, _ = run_one(p, weight_mode="model", weight_model=weight_models[w["component"]])
        b = BASELINES[(w["component"], "OOS")]
        cleared = beats_baseline(w["component"], "OOS", m_cand["pnl"], m_cand["mdd"])
        row = {**w, "oos_pnl": m_cand["pnl"], "oos_mdd": m_cand["mdd"], "oos_trades": m_cand["trades"],
               "oos_wr": m_cand["wr"], "oos_baseline_pnl": b["pnl"], "oos_baseline_mdd": b["mdd"], "cleared_oos": cleared}
        oos_rows.append(row)
        print(f"  {w} -> OOS pnl={m_cand['pnl']:.2f}% mdd={m_cand['mdd']:.2f}% trades={m_cand['trades']} "
              f"(baseline pnl={b['pnl']:.2f}% mdd={b['mdd']:.2f}%) cleared={cleared}", flush=True)
    pd.DataFrame(oos_rows).to_csv(OUT_DIR / "continuous_weight_OOS_confirm.csv", index=False)
    print("stage=done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
