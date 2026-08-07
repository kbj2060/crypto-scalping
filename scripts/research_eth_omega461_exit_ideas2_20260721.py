#!/usr/bin/env python3
"""RESEARCH ONLY -- 4 new exit-logic ideas for the LIVE ETH Omega4.6.1 h48qual/zig075
components, fresh-forward VAL-first funnel to OOS (CLAUDE.md Fresh-Forward rule).

Does NOT touch trading_bot_modules/omega4_6_1_live.py, trading_bot.py, runtime_config.py, or
.env, and does NOT modify the two prior research scripts
(research_eth_omega461_exit_sweep_20260721.py, research_eth_omega461_exit_sweep_oos_confirm_20260721.py)
-- reuses their prep_component/COMPONENTS/frame-loading/replay conventions unmodified (imported,
not copy-pasted, except for the one new replay loop below which extends the same causal
bar-by-bar structure with 4 new optional exit mechanisms).

Baselines to beat (reused from the prior run, NOT recomputed):
  VAL  h48qual: pnl +5.45%  mdd -11.62% (29 trades, wr 0.41)
  VAL  zig075:  pnl +40.31% mdd -13.07% (29 trades, wr 0.48)
  OOS  h48qual: pnl +9.49%  mdd -6.54%  (14 trades, wr 0.50)
  OOS  zig075:  pnl +17.89% mdd -11.01% (25 trades, wr 0.44)

Rule (CLAUDE.md Fresh-Forward + this task's instructions): every idea's grid is run on VAL only
first. Only configs that beat baseline on BOTH pnl and mdd (mdd less negative = better) are
carried forward to OOS confirmation. Grids capped at ~9 combos per idea.

Idea 1 -- regime-conditioned exit veto: giveback trailing-stop (same activate/retain mechanism as
  the prior run's Experiment B) additionally gated by the regime3 columns
  (regime3_current_sensitive_wide24_{bull,bear,chop}_prob, the same "current-HMM sensitive
  wide24" regime model whose chop_prob column Sigma6's live "not_chop" entry veto uses --
  see scripts/research_f4b_sigma6_dated_ledger_20260719.py CONFIGS reg_mode="not_chop":
  `ok = chop[i] < reg_thr` with reg_thr in {0.42, 0.50}). Two veto definitions:
    hard: force-exit only fires if chop_prob[i] >= reg_thr (regime has turned choppy)
    soft: force-exit only fires if chop_prob[i] > directional_prob[i], where directional_prob is
          bull_prob for a long / bear_prob for a short -- i.e. "regime moved one notch away from
          trending-in-the-trade's-favor" (chop has overtaken the trade's own directional regime
          as the single most likely state), a strictly looser trigger than the hard veto.
  reg_thr in {0.42, 0.50} (Sigma6's own confirmed values) used only for the hard mode.

Idea 2 -- partial scale-out: on the same giveback trigger, close close_frac of the position
  (reusing "notional" as the mutable open-exposure tracker) and let the remainder ride to the
  original TP/SL/exit-head with no further trailing checks (partial fires at most once/trade).

Idea 3 -- ATR Chandelier trailing stop: once armed (mfe >= activate_frac*TP), the stop trails at
  peak_move - N*atr_pct[i] instead of retain_frac*peak_move, using the SAME atr_pct series
  (same atr_window) the component's own ATR-based TP/SL sizing uses.

Idea 4 -- time-decaying TP: TP_effective(hold_bars) = TP_initial * max(decay_floor,
  1 - decay_rate*max(0, hold_bars - grace_bars)), decay_floor fixed at 0.5. No SL/trailing change.
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

import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as rs  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402

OUT_DIR = ROOT / "tmp/research_20260721"
PFX = "regime3_current_sensitive_wide24_"
BASELINES = {
    ("h48qual", "VAL"): {"pnl": 5.45, "mdd": -11.62},
    ("zig075", "VAL"): {"pnl": 40.31, "mdd": -13.07},
    ("h48qual", "OOS"): {"pnl": 9.49, "mdd": -6.54},
    ("zig075", "OOS"): {"pnl": 17.89, "mdd": -11.01},
}


def beats_baseline(component: str, split: str, pnl: float, mdd: float) -> bool:
    b = BASELINES[(component, split)]
    return pnl > b["pnl"] and mdd > b["mdd"]  # mdd is negative; less negative == better


@torch.no_grad()
def replay_idea(
    frame: pd.DataFrame,
    base_x: pd.DataFrame,
    dec: pd.DataFrame,
    loaded_models: dict[str, tuple],
    *,
    risk_margin_fraction: np.ndarray,
    risk_leverage: np.ndarray,
    fee: float,
    slip: float,
    cost_mult: float,
    notional_scaled_sltp: bool,
    device: torch.device,
    exit_threshold: float = sweep.BASELINE_EXIT_THRESHOLD,
    trailing_activate_frac: float | None = None,
    trailing_retain_frac: float | None = None,
    regime_mode: str | None = None,       # None | "hard" | "soft"
    regime_thr: float = 0.50,
    chop_arr: np.ndarray | None = None,
    bull_arr: np.ndarray | None = None,
    bear_arr: np.ndarray | None = None,
    partial_close_frac: float | None = None,
    atr_n: float | None = None,
    atr_arr: np.ndarray | None = None,
    decay_grace_bars: int | None = None,
    decay_rate: float | None = None,
    decay_floor: float = 0.5,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Causal bar-by-bar replay. Extends sweep.replay_exit_variant's structure (same TP/SL/
    exit-head order, same fill/cost model) with 4 optional new mechanisms, each gated by which
    kwargs are non-None so a single loop covers ideas 1-4 (and reduces exactly to the prior
    baseline when all idea-specific kwargs are None). fresh_forward_bar_by_bar=true (single
    forward pass, i increasing, only row i + already-closed history used at bar i); no saved
    ledger used as input.
    """
    trailing_enabled = trailing_activate_frac is not None and (trailing_retain_frac is not None or atr_n is not None)
    time_decay_enabled = decay_grace_bars is not None and decay_rate is not None
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    active = omega._active(dec)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
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
    leverage = 1.0
    margin_fraction = 0.0
    take_profit = 0.0
    stop_loss = 0.0
    mfe = 0.0
    mae = 0.0
    armed = False
    partial_done = False
    trades = 0
    wins = 0
    long_entries = 0
    short_entries = 0
    notional_sum = 0.0
    leverage_sum = 0.0
    margin_sum = 0.0
    reasons: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    route = hard._route_id(frame)
    from train_eval_omega1_2_tabm_diffusion_risk_20260603 import _try_execution as omega_try_execution, _fill_price as omega_fill_price
    import train_eval_omega4_1_exit_head_price_move_sltp_retrain_20260622 as price_exit

    base_np, exit_runtime, pos_idx = rs._prepare_exit_runtime(base_x, loaded_models)

    for i in range(0, len(frame) - 2):
        if pos != 0:
            move = price_exit._price_move(arrays, int(i), side=pos, entry_price=float(entry_price), slip_eff=slip_eff)
            mfe = max(mfe, move)
            mae = min(mae, move)
        else:
            move = 0.0

        if pos != 0:
            reason = ""
            exit_prob = 0.0
            hold = max(int(i) - int(entry_i), 0)
            tp_eff = take_profit
            if time_decay_enabled and take_profit > 0.0:
                shrink = max(0.0, 1.0 - float(decay_rate) * max(0, hold - int(decay_grace_bars)))
                tp_eff = take_profit * max(float(decay_floor), shrink)
            if tp_eff > 0.0 and move >= tp_eff:
                reason = "take_profit"
            elif stop_loss > 0.0 and move <= -abs(stop_loss):
                reason = "stop_loss"
            elif trailing_enabled and (not armed) and mfe >= float(trailing_activate_frac) * take_profit and take_profit > 0.0:
                armed = True
            if not reason and trailing_enabled and armed and (not partial_done) and mfe > 0.0:
                if atr_n is not None:
                    trigger = move <= (mfe - float(atr_n) * float(atr_arr[i]))
                else:
                    trigger = move <= float(trailing_retain_frac) * mfe
                if trigger and regime_mode is not None:
                    if regime_mode == "hard":
                        trigger = bool(chop_arr[i] >= float(regime_thr))
                    elif regime_mode == "soft":
                        directional = bull_arr[i] if pos > 0 else bear_arr[i]
                        trigger = bool(chop_arr[i] > directional)
                if trigger:
                    reason = "partial_scaleout" if (partial_close_frac is not None) else "trailing_stop"
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
            if reason == "partial_scaleout":
                filled, exit_px, exit_fee, _route = omega_try_execution(arrays, int(i), pos, entry=False, fee_base=fee_eff, slip_base=slip_eff)
                if not filled:
                    continue
                raw_exit = (exit_px - entry_price) / max(entry_price, 1.0e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1.0e-12)
                part_notional = notional * float(partial_close_frac)
                before = cash
                cash = cash * (1.0 + raw_exit * part_notional)
                cash -= before * exit_fee * part_notional
                reasons["partial_scaleout"] = reasons.get("partial_scaleout", 0) + 1
                rows.append({
                    "entry_signal_i": int(entry_signal_i), "entry_i": int(entry_i), "exit_i": int(i),
                    "entry_timestamp": str(frame["timestamp"].iloc[int(entry_signal_i)]),
                    "exit_timestamp": str(frame["timestamp"].iloc[int(i)]), "side": int(pos), "reason": "partial_scaleout",
                    "win": int(raw_exit > 0), "raw_exit_price_move": float(raw_exit), "mfe_price_move": float(mfe),
                    "mae_price_move": float(mae), "trade_return": float("nan"),
                    "net_per_notional": float("nan"), "notional": float(part_notional),
                    "margin_fraction": float(margin_fraction), "leverage": float(leverage),
                    "exit_prob": 0.0, "take_profit": float(take_profit), "stop_loss": float(stop_loss),
                })
                notional = notional - part_notional
                partial_done = True
                continue
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
                })
                pos = 0
                armed = False
                partial_done = False
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
        armed = False
        partial_done = False

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
            "take_profit": float(take_profit), "stop_loss": float(stop_loss),
        })

    n_entries = max(long_entries + short_entries, 1)
    ledger = pd.DataFrame(rows)
    closed = ledger[ledger["reason"] != "partial_scaleout"] if len(ledger) else ledger
    hold_bars = (closed["exit_i"] - closed["entry_i"]).clip(lower=0) if len(closed) else pd.Series(dtype=float)
    return (
        {
            "pnl": float((cash - 1.0) * 100.0), "mdd": float(mdd * 100.0), "trades": int(trades),
            "wr": float(wins / trades) if trades else 0.0, "trades_per_day": float(trades / rs._duration_days(frame)),
            "avg_notional": float(notional_sum / n_entries), "avg_leverage": float(leverage_sum / n_entries),
            "avg_hold_bars": float(hold_bars.mean()) if len(hold_bars) else 0.0,
            "max_trade_pnl": float(closed["trade_return"].max() * 100.0) if len(closed) else 0.0,
            "p95_trade_pnl": float(closed["trade_return"].quantile(0.95) * 100.0) if len(closed) else 0.0,
            "long_entries": int(long_entries), "short_entries": int(short_entries), "exit_reasons": reasons,
        },
        ledger,
    )


def get_regime_arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "chop": pd.to_numeric(frame[f"{PFX}chop_prob"], errors="raise").to_numpy(dtype=np.float64),
        "bull": pd.to_numeric(frame[f"{PFX}bull_prob"], errors="raise").to_numpy(dtype=np.float64),
        "bear": pd.to_numeric(frame[f"{PFX}bear_prob"], errors="raise").to_numpy(dtype=np.float64),
    }


def prep_all(components: dict, frame: pd.DataFrame, pred_dir: Path, *, oof: bool) -> dict[str, dict[str, Any]]:
    out = {}
    for name, cfg in components.items():
        pred = pred_dir / name / (f"validation_predictions_{cfg['q_tag']}.csv" if oof else f"oos_predictions_{cfg['q_tag']}.csv")
        p = sweep.prep_component(name, cfg, frame, pred, oof=oof)
        p["regime"] = get_regime_arrays(p["frame"])
        p["atr"] = atr_eval._atr_pct(p["frame"], cfg["atr_window"])
        out[name] = p
    return out


def run_one(p: dict, **kwargs) -> tuple[dict[str, Any], pd.DataFrame]:
    return replay_idea(
        p["frame"], p["x"], p["dec"], p["loaded"], risk_margin_fraction=p["margin"], risk_leverage=p["leverage"],
        fee=p["fee"], slip=p["slip"], cost_mult=sweep.COST_MULT, notional_scaled_sltp=p["notional_scaled_sltp"],
        device=sweep.DEVICE, **kwargs,
    )


def summarize(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["exit_reasons"] = df["exit_reasons"].apply(json.dumps)
    return df


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    val_frame = sweep.load_frame(sweep.VAL_START, sweep.VAL_END, base_csv=sweep.BASE_2025, wide24_csv=sweep.WIDE24_2025)
    print(f"VAL frame rows={len(val_frame)} range=[{val_frame['timestamp'].min()}, {val_frame['timestamp'].max()}]", flush=True)
    val_prepped = prep_all(sweep.COMPONENTS, val_frame, sweep.EXT_PRED_DIR, oof=True)

    all_val_rows: list[dict[str, Any]] = []
    winners: list[dict[str, Any]] = []  # dicts with component + replay kwargs, for OOS confirm

    # ---------------- Idea 1: regime-conditioned exit veto ----------------
    print("stage=idea1_val", flush=True)
    for name, p in val_prepped.items():
        reg = p["regime"]
        for act in (0.6, 0.8):
            for ret in (0.3, 0.5):
                for mode, thr_list in (("hard", (0.42, 0.50)), ("soft", (None,))):
                    for thr in thr_list:
                        m, _ = run_one(
                            p, trailing_activate_frac=act, trailing_retain_frac=ret,
                            regime_mode=mode, regime_thr=(thr if thr is not None else 0.50),
                            chop_arr=reg["chop"], bull_arr=reg["bull"], bear_arr=reg["bear"],
                        )
                        row = {"idea": "1_regime_veto", "component": name, "activate": act, "retain": ret,
                               "regime_mode": mode, "regime_thr": thr, **m}
                        all_val_rows.append(row)
                        if beats_baseline(name, "VAL", m["pnl"], m["mdd"]):
                            winners.append({"idea": "1_regime_veto", "component": name, "label": f"act={act} ret={ret} mode={mode} thr={thr}",
                                             "kwargs": dict(trailing_activate_frac=act, trailing_retain_frac=ret, regime_mode=mode,
                                                            regime_thr=(thr if thr is not None else 0.50),
                                                            chop_arr=None, bull_arr=None, bear_arr=None)})

    # ---------------- Idea 2: partial scale-out ----------------
    print("stage=idea2_val", flush=True)
    for name, p in val_prepped.items():
        for act in (0.6, 0.8):
            for ret in (0.4, 0.6):
                m, _ = run_one(p, trailing_activate_frac=act, trailing_retain_frac=ret, partial_close_frac=0.5)
                row = {"idea": "2_partial_scaleout", "component": name, "activate": act, "retain": ret,
                       "close_frac": 0.5, **m}
                all_val_rows.append(row)
                if beats_baseline(name, "VAL", m["pnl"], m["mdd"]):
                    winners.append({"idea": "2_partial_scaleout", "component": name, "label": f"act={act} ret={ret} close_frac=0.5",
                                     "kwargs": dict(trailing_activate_frac=act, trailing_retain_frac=ret, partial_close_frac=0.5)})

    # ---------------- Idea 3: ATR Chandelier trailing stop ----------------
    print("stage=idea3_val", flush=True)
    for name, p in val_prepped.items():
        atr_arr = p["atr"]
        for n in (2.0, 3.0, 4.0):
            for act in (0.6, 0.8):
                m, _ = run_one(p, trailing_activate_frac=act, atr_n=n, atr_arr=atr_arr)
                row = {"idea": "3_atr_chandelier", "component": name, "activate": act, "atr_n": n, **m}
                all_val_rows.append(row)
                if beats_baseline(name, "VAL", m["pnl"], m["mdd"]):
                    winners.append({"idea": "3_atr_chandelier", "component": name, "label": f"act={act} atr_n={n}",
                                     "kwargs": dict(trailing_activate_frac=act, atr_n=n, atr_arr=atr_arr)})

    # ---------------- Idea 4: time-decaying TP ----------------
    print("stage=idea4_val", flush=True)
    for name, p in val_prepped.items():
        for grace in (200, 400):
            for rate in (0.0005, 0.001):
                m, _ = run_one(p, decay_grace_bars=grace, decay_rate=rate, decay_floor=0.5)
                row = {"idea": "4_time_decay_tp", "component": name, "grace_bars": grace, "decay_rate": rate, **m}
                all_val_rows.append(row)
                if beats_baseline(name, "VAL", m["pnl"], m["mdd"]):
                    winners.append({"idea": "4_time_decay_tp", "component": name, "label": f"grace={grace} rate={rate}",
                                     "kwargs": dict(decay_grace_bars=grace, decay_rate=rate, decay_floor=0.5)})

    val_df = summarize(all_val_rows)
    val_df.to_csv(OUT_DIR / "exit_ideas2_VAL.csv", index=False)
    cols = ["idea", "component", "activate", "retain", "regime_mode", "regime_thr", "close_frac", "atr_n",
            "grace_bars", "decay_rate", "pnl", "mdd", "trades", "wr", "avg_hold_bars"]
    print(val_df[[c for c in cols if c in val_df.columns]].to_string(index=False), flush=True)

    print(f"\nVAL winners (beat baseline on BOTH pnl and mdd): {len(winners)}", flush=True)
    for w in winners:
        print(f"  idea={w['idea']} component={w['component']} {w['label']}", flush=True)

    if not winners:
        print("stage=done no_winners", flush=True)
        return 0

    # ---------------- OOS confirmation for VAL winners ----------------
    print("stage=oos_confirm", flush=True)
    oos_frame = sweep.load_frame(sweep.OOS_START, sweep.OOS_END, base_csv=sweep.BASE_2026, wide24_csv=sweep.WIDE24_2026)
    oos_prepped = prep_all(sweep.COMPONENTS, oos_frame, sweep.EXT_PRED_DIR, oof=False)

    oos_rows = []
    for w in winners:
        p = oos_prepped[w["component"]]
        kwargs = dict(w["kwargs"])
        if "atr_arr" in kwargs:
            kwargs["atr_arr"] = p["atr"]
        if "chop_arr" in kwargs:
            reg = p["regime"]
            kwargs["chop_arr"] = reg["chop"]
            kwargs["bull_arr"] = reg["bull"]
            kwargs["bear_arr"] = reg["bear"]
        m_base, _ = run_one(p)  # baseline no-op replay (all idea kwargs None) for a same-window sanity check
        m_cand, _ = run_one(p, **kwargs)
        cleared = beats_baseline(w["component"], "OOS", m_cand["pnl"], m_cand["mdd"])
        row = {"idea": w["idea"], "component": w["component"], "label": w["label"],
               "oos_pnl": m_cand["pnl"], "oos_mdd": m_cand["mdd"], "oos_trades": m_cand["trades"], "oos_wr": m_cand["wr"],
               "oos_baseline_reproduced_pnl": m_base["pnl"], "oos_baseline_reproduced_mdd": m_base["mdd"],
               "oos_baseline_reference_pnl": BASELINES[(w["component"], "OOS")]["pnl"],
               "oos_baseline_reference_mdd": BASELINES[(w["component"], "OOS")]["mdd"],
               "cleared_oos": cleared}
        oos_rows.append(row)
        print(f"  idea={w['idea']} component={w['component']} {w['label']} -> "
              f"OOS pnl={m_cand['pnl']:.2f}% mdd={m_cand['mdd']:.2f}% trades={m_cand['trades']} "
              f"(baseline pnl={BASELINES[(w['component'], 'OOS')]['pnl']:.2f}% mdd={BASELINES[(w['component'], 'OOS')]['mdd']:.2f}%) "
              f"cleared={cleared}", flush=True)

    pd.DataFrame(oos_rows).to_csv(OUT_DIR / "exit_ideas2_OOS_confirm.csv", index=False)
    print("stage=done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
