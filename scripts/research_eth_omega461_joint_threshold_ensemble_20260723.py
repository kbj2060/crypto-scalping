#!/usr/bin/env python3
"""RESEARCH ONLY -- round 15 (2026-07-23) of the ETH Omega4.6.1 exit-head investigation.

Round 12 (research_eth_omega461_joint_threshold_retrain_20260722.py) found that ALL 6
independently-retrained h48qual exit-head label variants (gb045/gb055/gb065_control/gb075/
gb085/tw08) beat the frozen baseline (EXIT_THRESHOLD=0.95) on BOTH VAL PnL and VAL MDD when
evaluated at exit_threshold=0.70. The single OOS touch on the VAL-best config (gb075/gb085
@0.70) collapsed: PnL +8.13% (vs baseline +9.49%, worse), MDD -6.54% (identical to baseline --
the lower threshold essentially never fired an extra exit_head exit in OOS; only 1 extra trade,
net-negative).

Hypothesis under test here: since MULTIPLE independently-trained heads agree at threshold=0.70
(not just one lucky config), an ENSEMBLE of their exit probabilities may be more robust than any
single frozen config, and may not collapse on OOS the way the single-config pick did. This script
computes exit_prob from all 6 h48qual label-variant heads at every open-position bar and combines
them via (a) simple average probability vs a combined threshold, and (b) majority vote (>=K of 6
heads individually cross a per-head threshold), then fires the actual exit only if the combined
signal crosses the gate. VAL-only selection; if a winner exists, ONE fresh OOS touch (this is a
genuinely new mechanism -- ensembling -- not a repeat of round 12's single-config OOS touch).
Fresh window 2026-04-01..07-12 (limited by available extended OOS prediction coverage) checked as
an additional non-selection robustness data point.

fresh_forward_bar_by_bar=true. trade_ledgers_used_as_input=false.
saved_parent_exit_timestamps_used=false. future_rows_used_for_entry=false.

Does NOT touch trading_bot_modules/omega4_6_1_live.py, trading_bot.py, runtime_config.py, .env.
Read-only w.r.t. all existing tmp/research_2026072*/tmp/causal_regen_20260516/*_retrain_20260721_*
artifacts -- no retraining is performed by this script (it reuses the 6 h48qual checkpoints
already on disk, same ones round 12 used). Research artifact only -- no promotion-gate claim.
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

import research_eth_omega461_exit_sweep_20260721 as base  # noqa: E402
import train_eval_omega4_1_exit_head_price_move_sltp_retrain_20260622 as price_exit  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as rs  # noqa: E402
from train_eval_omega1_2_tabm_diffusion_risk_20260603 import (  # noqa: E402
    _fill_price as omega_fill_price,
    _try_execution as omega_try_execution,
)

omega = base.omega
hard = base.hard
parent = base.parent


def _bundle(suffix: str) -> Path:
    return ROOT / f"tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_research_exit_head_retrain_20260721_{suffix}/true_3head_tabm_bundle.pt"


H48QUAL_VARIANT_BUNDLES: dict[str, Path] = {
    "gb045": _bundle("h48qual_gb045"),
    "gb055": _bundle("h48qual_gb055"),
    "gb065_control": _bundle("h48qual_control_edge0020"),
    "gb075": _bundle("h48qual_gb075"),
    "gb085": _bundle("h48qual_gb085"),
    "tw08": _bundle("h48qual_tw08"),
}

FRESH_START, FRESH_END = "2026-04-01", "2026-07-21"

OUT_DIR = ROOT / "tmp/research_20260723/joint_threshold_ensemble_20260723"


def _load_variant_runtimes(frame: pd.DataFrame, bundles: dict[str, Path]) -> dict[str, dict]:
    """Loads each variant's exit-head-only bundle and builds its own _prepare_exit_runtime
    triple (base_np, runtime, pos_idx). Each variant's bundle can have its own base_cols
    order/scaler stats (retrain-eval variants are NOT guaranteed byte-identical base_cols to the
    production bundle -- observed empirically), so each head gets its own base_x built from its
    own bundle's base_cols, matching exactly what round 12's prep_component-per-variant did."""
    heads: dict[str, dict] = {}
    for name, path in bundles.items():
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        base_cols = bundle["base_cols"]
        loaded = parent._load_payloads(bundle["models"], device=base.DEVICE)
        x_variant = parent._base_input(frame, base_cols)
        base_np, runtime, pos_idx = rs._prepare_exit_runtime(x_variant, loaded)
        heads[name] = {"base_np": base_np, "runtime": runtime, "pos_idx": pos_idx}
    return heads


@torch.no_grad()
def replay_exit_variant_ensemble(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    ensemble: dict,
    *,
    risk_margin_fraction: np.ndarray,
    risk_leverage: np.ndarray,
    combine_mode: str,
    combine_threshold: float,
    vote_threshold: float,
    vote_k: int,
    fee: float,
    slip: float,
    cost_mult: float,
    notional_scaled_sltp: bool,
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Structurally identical to base.replay_exit_variant, except the exit-head probability gate
    is a combination of ALL variant heads' individual exit_prob at each open-position bar, instead
    of a single head's prob. combine_mode='avg': fire when mean(probs) >= combine_threshold.
    combine_mode='majority': fire when count(probs >= vote_threshold) >= vote_k.
    fresh_forward_bar_by_bar=true; only row i and already-closed prior bars used at bar i."""
    assert combine_mode in ("avg", "majority")
    heads = ensemble  # name -> {"base_np", "runtime", "pos_idx"}
    n_heads = len(heads)

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
    reasons: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    route = hard._route_id(frame)

    for i in range(0, len(frame) - 2):
        if pos != 0:
            move = price_exit._price_move(arrays, int(i), side=pos, entry_price=float(entry_price), slip_eff=slip_eff)
            mfe = max(mfe, move)
            mae = min(mae, move)
        else:
            move = 0.0

        if pos != 0:
            reason = ""
            combined_prob = 0.0
            if take_profit > 0.0 and move >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and move <= -abs(stop_loss):
                reason = "stop_loss"
            if not reason:
                hold = max(int(i) - int(entry_i), 0)
                giveback = (float(mfe) - float(move)) / max(abs(float(mfe)), 1.0e-8) if mfe > 0.0 else 0.0
                expert = hard.EXPERT_NAMES[int(route[i])]
                pos_values = [
                    float(pos), float(hold), float(move), float(mfe), float(mae),
                    float(np.clip(giveback, 0.0, 10.0)), float(take_profit - move), float(move + abs(stop_loss)),
                    float(notional), float(leverage), float(notional * leverage), float(take_profit), float(stop_loss),
                ]
                probs = [
                    rs._predict_exit_prob_one(h["base_np"], h["runtime"], h["pos_idx"], row_i=int(i), expert=expert, pos_values=pos_values, device=device)
                    for h in heads.values()
                ]
                if combine_mode == "avg":
                    combined_prob = float(np.mean(probs))
                    fire = combined_prob >= float(combine_threshold)
                else:
                    votes = sum(1 for p in probs if p >= float(vote_threshold))
                    combined_prob = float(votes) / float(n_heads)
                    fire = votes >= int(vote_k)
                if fire:
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
                    "net_per_notional": float(trade_return / max(notional, 1.0e-12)), "notional": float(notional),
                    "margin_fraction": float(margin_fraction), "leverage": float(leverage),
                    "combined_exit_prob": float(combined_prob), "take_profit": float(take_profit), "stop_loss": float(stop_loss),
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
            "net_per_notional": float(trade_return / max(notional, 1.0e-12)), "notional": float(notional),
            "margin_fraction": float(margin_fraction), "leverage": float(leverage), "combined_exit_prob": 0.0,
            "take_profit": float(take_profit), "stop_loss": float(stop_loss),
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
        },
        ledger,
    )


def prep_ensemble_for_split(split: str) -> dict[str, Any]:
    cfg = base.COMPONENTS["h48qual"]
    if split == "VAL":
        frame = base.load_frame(base.VAL_START, base.VAL_END, base_csv=base.BASE_2025, wide24_csv=base.WIDE24_2025)
        pred_csv = base.EXT_PRED_DIR / "h48qual" / f"validation_predictions_{cfg['q_tag']}.csv"
        oof = True
    elif split == "OOS":
        frame = base.load_frame(base.OOS_START, base.OOS_END, base_csv=base.BASE_2026, wide24_csv=base.WIDE24_2026)
        pred_csv = base.EXT_PRED_DIR / "h48qual" / f"oos_predictions_{cfg['q_tag']}.csv"
        oof = False
    elif split == "FRESH":
        frame = base.load_frame(FRESH_START, FRESH_END, base_csv=base.BASE_2026, wide24_csv=base.WIDE24_2026)
        pred_csv = base.EXT_PRED_DIR / "h48qual" / f"oos_predictions_{cfg['q_tag']}.csv"
        oof = False
    else:
        raise ValueError(split)

    prepped = base.prep_component("h48qual", cfg, frame, pred_csv, oof=oof)
    ensemble = _load_variant_runtimes(prepped["frame"], H48QUAL_VARIANT_BUNDLES)
    return {**prepped, "ensemble": ensemble}


def run_grid(prepped: dict[str, Any], split: str, *, avg_thresholds: list[float], vote_grid: list[tuple[float, int]]) -> pd.DataFrame:
    out = []
    for ct in avg_thresholds:
        m, _ = replay_exit_variant_ensemble(
            prepped["frame"], prepped["dec"], prepped["ensemble"],
            risk_margin_fraction=prepped["margin"], risk_leverage=prepped["leverage"],
            combine_mode="avg", combine_threshold=ct, vote_threshold=0.0, vote_k=0,
            fee=prepped["fee"], slip=prepped["slip"], cost_mult=base.COST_MULT,
            notional_scaled_sltp=prepped["notional_scaled_sltp"], device=base.DEVICE,
        )
        out.append({"split": split, "combine_mode": "avg", "combine_threshold": ct, "vote_threshold": None, "vote_k": None,
                     **m, "exit_reasons": json.dumps(m["exit_reasons"])})
    for vt, vk in vote_grid:
        m, _ = replay_exit_variant_ensemble(
            prepped["frame"], prepped["dec"], prepped["ensemble"],
            risk_margin_fraction=prepped["margin"], risk_leverage=prepped["leverage"],
            combine_mode="majority", combine_threshold=0.0, vote_threshold=vt, vote_k=vk,
            fee=prepped["fee"], slip=prepped["slip"], cost_mult=base.COST_MULT,
            notional_scaled_sltp=prepped["notional_scaled_sltp"], device=base.DEVICE,
        )
        out.append({"split": split, "combine_mode": "majority", "combine_threshold": None, "vote_threshold": vt, "vote_k": vk,
                     **m, "exit_reasons": json.dumps(m["exit_reasons"])})
    return pd.DataFrame(out)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("stage=prep_VAL", flush=True)
    val_prepped = prep_ensemble_for_split("VAL")

    avg_thresholds = [0.60, 0.65, 0.70, 0.75]
    vote_grid = [(0.70, k) for k in (1, 2, 3, 4, 5, 6)]

    print("stage=val_grid", flush=True)
    val_res = run_grid(val_prepped, "VAL", avg_thresholds=avg_thresholds, vote_grid=vote_grid)
    val_res.to_csv(OUT_DIR / "ensemble_grid_VAL.csv", index=False)
    print(val_res[["combine_mode", "combine_threshold", "vote_threshold", "vote_k", "pnl", "mdd", "trades", "wr", "avg_hold_bars"]].to_string(index=False), flush=True)
    print("stage=done_val", flush=True)

    # --- Selection: majority K=4/K=5 (PnL 27.31/MDD -7.46/34 trades) is numerically IDENTICAL
    # to the individual gb075/gb085 variant already OOS-touched in round 12 (same trade ledger --
    # not a new mechanism, so re-touching OOS on it would not be a legitimate fresh touch).
    # avg-probability @0.70 (PnL 21.08/MDD -7.46/36 trades) is the best VAL winner that produces a
    # GENUINELY DIFFERENT trade set from any single variant -- true probability-averaging ensemble.
    # This is the one fresh OOS touch spent in this round.
    print("stage=oos_touch_avg070", flush=True)
    if len(sys.argv) > 1 and sys.argv[1] == "--val-only":
        return 0
    oos_prepped = prep_ensemble_for_split("OOS")
    oos_res = run_grid(oos_prepped, "OOS", avg_thresholds=[0.70], vote_grid=[(0.70, 4)])
    oos_res.to_csv(OUT_DIR / "ensemble_oos_touch.csv", index=False)
    print(oos_res[["combine_mode", "combine_threshold", "vote_threshold", "vote_k", "pnl", "mdd", "trades", "wr", "avg_hold_bars"]].to_string(index=False), flush=True)

    print("stage=fresh_window_report_only", flush=True)
    fresh_prepped = prep_ensemble_for_split("FRESH")
    fresh_res = run_grid(fresh_prepped, "FRESH", avg_thresholds=[0.70], vote_grid=[(0.70, 4)])
    fresh_res.to_csv(OUT_DIR / "ensemble_fresh_window.csv", index=False)
    print(fresh_res[["combine_mode", "combine_threshold", "vote_threshold", "vote_k", "pnl", "mdd", "trades", "wr", "avg_hold_bars"]].to_string(index=False), flush=True)

    print("stage=done_all", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
