#!/usr/bin/env python3
"""RESEARCH ONLY -- fresh-forward VAL-first->OOS funnel for the reversal-risk scale-out
classifier trained by train_eth_omega461_reversal_risk_scaleout_20260721.py.

Forked from research_eth_omega461_exit_ideas2_20260721.py's replay_idea loop (imported, not
copy-pasted from anything else): reuses its exact TP/SL/exit-head order, fill/cost model, and
its "partial_scaleout" two-tranche accounting block verbatim (idea 2 in that file). This script
adds ONE new trigger -- the reversal-risk classifier probability -- positioned BEFORE the
exit_head probability check, gated by (reversal_activate_frac, reversal_prob_thr,
reversal_close_frac), firing at most once per trade via a partial_done-style flag (reused from
ideas2's own trailing/partial machinery). Does NOT touch trading_bot_modules/omega4_6_1_live.py,
trading_bot.py, runtime_config.py, or .env.

Baselines (reused unmodified from ideas2 / exit_sweep, NOT recomputed):
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
import train_eth_omega461_reversal_risk_scaleout_20260721 as train_mod  # noqa: E402

OUT_DIR = ROOT / "tmp/research_20260721"
BASELINES = ideas2.BASELINES
beats_baseline = ideas2.beats_baseline
MODEL_DIR = ROOT / "tmp/causal_regen_20260516"


def load_reversal_model(name: str) -> dict[str, Any]:
    with open(MODEL_DIR / f"eth_omega461_reversal_risk_scaleout_20260721_{name}" / "model.pkl", "rb") as f:
        return pickle.load(f)


def load_proxy_for_frame(pred_csv: Path, frame: pd.DataFrame) -> pd.DataFrame:
    keep_ts = set(frame["timestamp"])
    # OOF (train/validation) prediction CSVs use the "..._oof_..." column prefix;
    # OOS (held-out) prediction CSVs drop "_oof_" since they aren't out-of-fold. Same underlying
    # per-bar quality/direction-head semantics either way -- just a naming difference.
    header = pd.read_csv(pred_csv, nrows=0).columns
    if train_mod.PROXY_QUALITY_COL in header:
        qual_col, long_col, short_col = train_mod.PROXY_QUALITY_COL, train_mod.PROXY_DIR_LONG_COL, train_mod.PROXY_DIR_SHORT_COL
    else:
        qual_col = train_mod.PROXY_QUALITY_COL.replace("_oof_", "_")
        long_col = train_mod.PROXY_DIR_LONG_COL.replace("_oof_", "_")
        short_col = train_mod.PROXY_DIR_SHORT_COL.replace("_oof_", "_")
        if qual_col not in header:
            raise RuntimeError(f"proxy columns not found in {pred_csv} (tried oof and non-oof naming)")
    src = pd.read_csv(pred_csv, usecols=["timestamp", qual_col, long_col, short_col])
    src = src.rename(columns={
        qual_col: train_mod.PROXY_QUALITY_COL, long_col: train_mod.PROXY_DIR_LONG_COL, short_col: train_mod.PROXY_DIR_SHORT_COL,
    })
    src["timestamp"] = pd.to_datetime(src["timestamp"])
    src = src[src["timestamp"].isin(keep_ts)].reset_index(drop=True)
    if len(src) != len(frame) or not src["timestamp"].equals(frame["timestamp"]):
        raise RuntimeError(f"proxy/frame timestamp mismatch ({len(src)} vs {len(frame)})")
    return src


@torch.no_grad()
def replay_reversal(
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
    reversal_model: dict[str, Any] | None = None,
    proxy_quality: np.ndarray | None = None,
    proxy_dir_long: np.ndarray | None = None,
    proxy_dir_short: np.ndarray | None = None,
    reversal_activate_frac: float | None = None,
    reversal_prob_thr: float | None = None,
    reversal_close_frac: float | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Causal bar-by-bar replay. Identical structure to ideas2.replay_idea (TP/SL, then the new
    reversal-risk trigger, then the exit-head check, then the same partial-scaleout two-tranche
    accounting block reused verbatim). Reduces exactly to the baseline (no trailing/partial/
    regime kwargs) when reversal_model is None. fresh_forward_bar_by_bar=true; no saved ledger
    used as input.
    """
    reversal_enabled = (
        reversal_model is not None
        and reversal_activate_frac is not None
        and reversal_prob_thr is not None
        and reversal_close_frac is not None
    )
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
    base_cols = list(base_x.columns)
    rev_clf = reversal_model["model"] if reversal_model is not None else None
    rev_cols = reversal_model["feature_columns"] if reversal_model is not None else None

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
            if take_profit > 0.0 and move >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and move <= -abs(stop_loss):
                reason = "stop_loss"
            elif reversal_enabled and (not armed) and take_profit > 0.0 and mfe >= float(reversal_activate_frac) * take_profit:
                armed = True
            if not reason and reversal_enabled and armed and (not partial_done):
                giveback_now = (float(mfe) - float(move)) / max(abs(float(mfe)), 1.0e-8) if mfe > 0.0 else 0.0
                feat = {base_cols[j]: float(base_np[i, j]) for j in range(len(base_cols))}
                feat.update({
                    "pos_side": float(pos), "pos_hold_bars": float(hold), "pos_unrealized": float(move),
                    "pos_mfe": float(mfe), "pos_mae": float(mae), "pos_giveback": float(np.clip(giveback_now, 0.0, 10.0)),
                    "pos_dist_to_tp": float(take_profit - move), "pos_dist_to_sl": float(move + abs(stop_loss)),
                    "pos_notional": float(notional), "pos_leverage": float(leverage), "pos_exposure": float(notional * leverage),
                    "pos_tp": float(take_profit), "pos_sl": float(stop_loss),
                    "proxy_quality_for_action": float(proxy_quality[i]),
                    "proxy_dir_p_side": float(proxy_dir_long[i] if pos > 0 else proxy_dir_short[i]),
                })
                x_row = pd.DataFrame([feat]).reindex(columns=rev_cols, fill_value=0.0).to_numpy(dtype=np.float64)
                rev_prob = float(rev_clf.predict_proba(x_row)[0, 1])
                if rev_prob >= float(reversal_prob_thr):
                    reason = "reversal_scaleout"
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
            if reason == "reversal_scaleout":
                # Reused verbatim from ideas2.replay_idea's partial-scaleout two-tranche block.
                filled, exit_px, exit_fee, _route = omega_try_execution(arrays, int(i), pos, entry=False, fee_base=fee_eff, slip_base=slip_eff)
                if not filled:
                    continue
                raw_exit = (exit_px - entry_price) / max(entry_price, 1.0e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1.0e-12)
                part_notional = notional * float(reversal_close_frac)
                before = cash
                cash = cash * (1.0 + raw_exit * part_notional)
                cash -= before * exit_fee * part_notional
                reasons["reversal_scaleout"] = reasons.get("reversal_scaleout", 0) + 1
                rows.append({
                    "entry_signal_i": int(entry_signal_i), "entry_i": int(entry_i), "exit_i": int(i),
                    "entry_timestamp": str(frame["timestamp"].iloc[int(entry_signal_i)]),
                    "exit_timestamp": str(frame["timestamp"].iloc[int(i)]), "side": int(pos), "reason": "reversal_scaleout",
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
    closed = ledger[ledger["reason"] != "reversal_scaleout"] if len(ledger) else ledger
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


def run_one(p: dict, **kwargs) -> tuple[dict[str, Any], pd.DataFrame]:
    return replay_reversal(
        p["frame"], p["x"], p["dec"], p["loaded"], risk_margin_fraction=p["margin"], risk_leverage=p["leverage"],
        fee=p["fee"], slip=p["slip"], cost_mult=sweep.COST_MULT, notional_scaled_sltp=p["notional_scaled_sltp"],
        device=sweep.DEVICE, **kwargs,
    )


def prep_with_proxy(name: str, cfg: dict, frame: pd.DataFrame, pred_dir: Path, *, oof: bool) -> dict[str, Any]:
    pred = pred_dir / name / (f"validation_predictions_{cfg['q_tag']}.csv" if oof else f"oos_predictions_{cfg['q_tag']}.csv")
    p = sweep.prep_component(name, cfg, frame, pred, oof=oof)
    proxy = load_proxy_for_frame(pred, p["frame"])
    p["proxy_quality"] = proxy[train_mod.PROXY_QUALITY_COL].to_numpy(dtype=np.float64)
    p["proxy_dir_long"] = proxy[train_mod.PROXY_DIR_LONG_COL].to_numpy(dtype=np.float64)
    p["proxy_dir_short"] = proxy[train_mod.PROXY_DIR_SHORT_COL].to_numpy(dtype=np.float64)
    return p


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reversal_models = {name: load_reversal_model(name) for name in sweep.COMPONENTS}

    val_frame = sweep.load_frame(sweep.VAL_START, sweep.VAL_END, base_csv=sweep.BASE_2025, wide24_csv=sweep.WIDE24_2025)
    print(f"VAL frame rows={len(val_frame)} range=[{val_frame['timestamp'].min()}, {val_frame['timestamp'].max()}]", flush=True)
    val_prepped = {name: prep_with_proxy(name, cfg, val_frame, sweep.EXT_PRED_DIR, oof=True) for name, cfg in sweep.COMPONENTS.items()}

    oos_frame = sweep.load_frame(sweep.OOS_START, sweep.OOS_END, base_csv=sweep.BASE_2026, wide24_csv=sweep.WIDE24_2026)
    print(f"OOS frame rows={len(oos_frame)} range=[{oos_frame['timestamp'].min()}, {oos_frame['timestamp'].max()}]", flush=True)
    oos_prepped = {name: prep_with_proxy(name, cfg, oos_frame, sweep.EXT_PRED_DIR, oof=False) for name, cfg in sweep.COMPONENTS.items()}

    # ---------------- Mandatory sanity checks (Step 4) ----------------
    print("stage=sanity_checks", flush=True)
    sanity_rows = []
    all_sane = True
    for split_name, prepped, base_key in (("VAL", val_prepped, "VAL"), ("OOS", oos_prepped, "OOS")):
        for name, p in prepped.items():
            b = BASELINES[(name, base_key)]
            m_noop, _ = run_one(p)  # reversal_model=None -> fully disabled no-op path
            m_thr101, _ = run_one(
                p, reversal_model=reversal_models[name], proxy_quality=p["proxy_quality"],
                proxy_dir_long=p["proxy_dir_long"], proxy_dir_short=p["proxy_dir_short"],
                reversal_activate_frac=0.7, reversal_prob_thr=1.01, reversal_close_frac=0.5,
            )
            # BASELINES constants are recorded rounded to 2 decimal places; tolerance vs. them
            # matches that rounding precision (observed reproduction diffs were all <=0.0045).
            # noop vs thr101 must additionally be exactly (near bit-) identical to each other --
            # both take the literal same code path (reversal_enabled=True either way) and
            # thr101=1.01 can mathematically never fire since predict_proba in [0,1], so this
            # isolates a >=/> off-by-one bug from ordinary baseline-rounding noise.
            tol = 0.01
            ok_noop = abs(m_noop["pnl"] - b["pnl"]) < tol and abs(m_noop["mdd"] - b["mdd"]) < tol
            ok_thr101_vs_baseline = abs(m_thr101["pnl"] - b["pnl"]) < tol and abs(m_thr101["mdd"] - b["mdd"]) < tol
            ok_thr101_vs_noop = abs(m_thr101["pnl"] - m_noop["pnl"]) < 1e-9 and abs(m_thr101["mdd"] - m_noop["mdd"]) < 1e-9
            ok_thr101 = ok_thr101_vs_baseline and ok_thr101_vs_noop
            all_sane = all_sane and ok_noop and ok_thr101
            sanity_rows.append({
                "split": split_name, "component": name, "baseline_pnl": b["pnl"], "baseline_mdd": b["mdd"],
                "noop_pnl": m_noop["pnl"], "noop_mdd": m_noop["mdd"], "noop_ok": ok_noop,
                "thr101_pnl": m_thr101["pnl"], "thr101_mdd": m_thr101["mdd"], "thr101_ok": ok_thr101,
                "thr101_vs_noop_exact": ok_thr101_vs_noop,
            })
            print(f"  sanity split={split_name} component={name} noop_ok={ok_noop} thr101_ok={ok_thr101} "
                  f"thr101_vs_noop_exact={ok_thr101_vs_noop} "
                  f"(noop pnl={m_noop['pnl']:.4f}/mdd={m_noop['mdd']:.4f} thr101 pnl={m_thr101['pnl']:.4f}/mdd={m_thr101['mdd']:.4f} "
                  f"vs baseline pnl={b['pnl']:.4f}/mdd={b['mdd']:.4f})", flush=True)
    pd.DataFrame(sanity_rows).to_csv(OUT_DIR / "reversal_scaleout_sanity_checks.csv", index=False)
    if not all_sane:
        print("stage=STOP sanity checks FAILED -- see reversal_scaleout_sanity_checks.csv, not proceeding to grid", flush=True)
        return 1
    print("stage=sanity_checks PASSED", flush=True)

    # ---------------- Step 3: VAL grid ----------------
    print("stage=val_grid", flush=True)
    val_rows = []
    winners = []
    for name, p in val_prepped.items():
        for act in (0.6, 0.8):
            for thr in (0.5, 0.65, 0.8):
                m, _ = run_one(
                    p, reversal_model=reversal_models[name], proxy_quality=p["proxy_quality"],
                    proxy_dir_long=p["proxy_dir_long"], proxy_dir_short=p["proxy_dir_short"],
                    reversal_activate_frac=act, reversal_prob_thr=thr, reversal_close_frac=0.5,
                )
                row = {"component": name, "activate_frac": act, "prob_thr": thr, "close_frac": 0.5, **m}
                val_rows.append(row)
                cleared = beats_baseline(name, "VAL", m["pnl"], m["mdd"])
                if cleared:
                    winners.append({"component": name, "activate_frac": act, "prob_thr": thr, "close_frac": 0.5})
    val_df = pd.DataFrame(val_rows)
    val_df["exit_reasons"] = val_df["exit_reasons"].apply(json.dumps)
    val_df.to_csv(OUT_DIR / "reversal_scaleout_VAL.csv", index=False)
    cols = ["component", "activate_frac", "prob_thr", "close_frac", "pnl", "mdd", "trades", "wr", "avg_hold_bars"]
    print(val_df[cols].to_string(index=False), flush=True)
    print(f"\nVAL winners (beat baseline on BOTH pnl and mdd): {len(winners)}", flush=True)
    for w in winners:
        print(f"  {w}", flush=True)

    if not winners:
        print("stage=done no_val_winners -> no OOS confirmation run", flush=True)
        return 0

    # ---------------- Step 3: OOS confirmation ----------------
    print("stage=oos_confirm", flush=True)
    oos_rows = []
    for w in winners:
        p = oos_prepped[w["component"]]
        m_cand, _ = run_one(
            p, reversal_model=reversal_models[w["component"]], proxy_quality=p["proxy_quality"],
            proxy_dir_long=p["proxy_dir_long"], proxy_dir_short=p["proxy_dir_short"],
            reversal_activate_frac=w["activate_frac"], reversal_prob_thr=w["prob_thr"], reversal_close_frac=w["close_frac"],
        )
        b = BASELINES[(w["component"], "OOS")]
        cleared = beats_baseline(w["component"], "OOS", m_cand["pnl"], m_cand["mdd"])
        row = {**w, "oos_pnl": m_cand["pnl"], "oos_mdd": m_cand["mdd"], "oos_trades": m_cand["trades"],
               "oos_wr": m_cand["wr"], "oos_baseline_pnl": b["pnl"], "oos_baseline_mdd": b["mdd"], "cleared_oos": cleared}
        oos_rows.append(row)
        print(f"  {w} -> OOS pnl={m_cand['pnl']:.2f}% mdd={m_cand['mdd']:.2f}% trades={m_cand['trades']} "
              f"(baseline pnl={b['pnl']:.2f}% mdd={b['mdd']:.2f}%) cleared={cleared}", flush=True)
    pd.DataFrame(oos_rows).to_csv(OUT_DIR / "reversal_scaleout_OOS_confirm.csv", index=False)
    print("stage=done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
