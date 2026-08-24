#!/usr/bin/env python3
"""RESEARCH ONLY -- Phase 2 evaluation for eth_candidate_unified_single_component_redesign,
closing out Phase 2 with what exists (user decision 2026-08-18: stop chasing the full N=5 seed
retrain -- WSL2 VM instability killed 4 consecutive attempts tonight -- and conclude with the 3
exit-head seeds that ARE trained, clearly caveated as pre-fix/N=3 rather than a clean result).

Reuses eval_eth_candidate_unified_phase1_quality_ab_20260817.py's frame-loading and
direction/quality-inference harness UNCHANGED (same WINDOWS, same _extended_2026_frame/
_fresh_oos_preds OOS-truncation workaround). Direction/quality decisions come from the FROZEN
Phase-1 quality_B_samedir seed2559205075 parent at its CONFIRMED threshold=0.80 (Phase 1's
finding) -- this is bundle-identical to what each Phase 2 exit-head bundle's own encoder/
direction/quality heads compute, since Phase 2 only retrains exit_head and copies the rest of the
state_dict unchanged (verified by reading train_eth_candidate_unified_phase2_exit_head_
giveback_recal_20260817.py's _retrain_exit_head_only). Adds exit_head-aware replay (TP/SL first,
then exit_head prob>=0.95, matching the live EXIT_THRESHOLD) via the same sidecar._prepare_exit_
runtime/_predict_exit_prob_one machinery replay_omega4_6_1_greedy_router_20260706.greedy_replay
uses, single-component (no priority routing needed -- this candidate isn't paired with anything).

Fixed sizing (BASE_TEMPLATE notional=0.45, leverage=2.0) -- SAME convention as Phase 1, since this
candidate has no trained risk sidecar yet. NOT directly comparable to the 3 baselines' PnL% (G0/
h48qual-alone/zig075-alone all use REAL risk-sidecar sizing) -- flagged explicitly in the output,
not silently glossed over. The internal comparison this script IS built for -- same quality_B
system, exit_head vs no-exit_head, both fixed-sized -- is apples-to-apples and is the one that
answers "does the giveback_min=0.25 recalibration help."

KNOWN CAVEAT (not remedied here): all 3 available seeds predate the 2026-08-18 pos_unrealized/
pos_mfe/pos_mae scaling fix to research_eth_omega461_exit_head_liveatr_relabel_20260813.py's
_build_exit_dataset_entry_label_live_atr_barrier (see docs/experiments/eth_odyssey4_exit_head_
liveatr_barrier_and_label_reaudit_20260818.md) -- these exit heads were trained on the
pre-fix (0.45x-compressed) pos_unrealized/mfe/mae features. Also N=3, not the N>=5 this repo's
own seed-diversity gate requires (tabm_hp_low_signal_pattern memory) -- results below are
preliminary/directional, not a promotion-grade conclusion.

fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false, saved_parent_exit_timestamps_
used=false, future_rows_used_for_entry=false. No live/shadow files touched.
"""
from __future__ import annotations

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

import eval_eth_candidate_unified_phase1_quality_ab_20260817 as phase1  # noqa: E402
import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as base_sweep  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent3head  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as sidecar  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402

WIDE24_FOR = {phase1.BASE_2025: base_sweep.WIDE24_2025, phase1.BASE_2026: base_sweep.WIDE24_2026}
_route_cache: dict[Path, pd.DataFrame] = {}


def _route_cols_frame(base_csv: Path) -> pd.DataFrame:
    """timestamp + Regime3 route probability columns for the FULL year (cached), merged onto a
    price window at replay time -- phase1._load_price_frame deliberately drops all non-OHLC
    columns (Phase 1 never needed route), so this is a separate, minimal merge added for Phase 2's
    exit_head (which routes to bull/bear/chop_expert per bar, same as greedy_replay/live)."""
    if base_csv not in _route_cache:
        overlay = pd.read_csv(WIDE24_FOR[base_csv], usecols=["timestamp", *hard.ROUTE_COLS], low_memory=False)
        overlay["timestamp"] = pd.to_datetime(overlay["timestamp"])
        _route_cache[base_csv] = overlay.dropna(subset=hard.ROUTE_COLS).drop_duplicates("timestamp", keep="last")
    return _route_cache[base_csv]

PARENT_BUNDLE_DIR = ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_eth_candidate_unified_phase1_quality_B_samedir_seed2559205075"
PHASE2_SEEDS = (548794457, 3646016929, 2988156591)  # the only 3 that completed (pre-fix, N=3 not 5)
THRESHOLD = 0.80  # Phase 1's confirmed winner
EXIT_THRESHOLD = 0.95  # matches live EXIT_THRESHOLD
NOTIONAL, LEVERAGE = 0.45, 2.0
FEE, SLIP = 0.0005, 0.0002


def log(msg: str) -> None:
    print(msg, flush=True)


def _decisions_for_window(merged: pd.DataFrame) -> pd.DataFrame:
    direction_action = np.select(
        [merged["dir_p_long"] > merged[["dir_p_cash", "dir_p_short"]].max(axis=1),
         merged["dir_p_short"] > merged[["dir_p_cash", "dir_p_long"]].max(axis=1)],
        [1, 2], default=0,
    )
    quality_for_action = merged["quality_for_action"].to_numpy(dtype=np.float64)
    final_action = np.where((direction_action != 0) & (quality_for_action >= THRESHOLD), direction_action, 0)
    side = np.where(final_action == 1, 1, np.where(final_action == 2, -1, 0))
    atr_pct = atr_eval._atr_pct(merged, phase1.ATR_CFG["atr_window"])
    tp_move = np.clip(np.maximum(phase1.ATR_CFG["min_tp"], atr_pct * phase1.ATR_CFG["tp_mult"]), 0.0, phase1.ATR_CFG["max_tp"])
    sl_move = np.clip(np.maximum(phase1.ATR_CFG["min_sl"], atr_pct * phase1.ATR_CFG["sl_mult"]), 0.0, phase1.ATR_CFG["max_sl"])
    return pd.DataFrame({"side": side, "take_profit": np.where(side != 0, tp_move, 0.0), "stop_loss": np.where(side != 0, sl_move, 0.0)})


@torch.no_grad()
def _replay_with_exit_head(frame: pd.DataFrame, dec: pd.DataFrame, base_np: np.ndarray,
                            exit_runtime: dict, pos_idx: list[int], route: np.ndarray, device: torch.device) -> dict:
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    n = len(frame)
    cash = peak = 1.0
    mdd = 0.0
    pos = 0
    entry_i = 0
    entry_price = entry_equity = 1.0
    take_profit = stop_loss = 0.0
    mfe = mae = 0.0
    trades = wins = 0
    exit_head_fires = 0
    reason_counts: dict[str, int] = {}
    for i in range(0, n - 2):
        if pos != 0:
            move = (arrays["close"][i] * (1 - SLIP) - entry_price) / entry_price if pos > 0 else (entry_price - arrays["close"][i] * (1 + SLIP)) / entry_price
            mfe, mae = max(mfe, move), min(mae, move)
            eq = cash * (1.0 + move * NOTIONAL)
            peak = max(peak, eq)
            mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
            reason = ""
            if take_profit > 0.0 and move >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and move <= -abs(stop_loss):
                reason = "stop_loss"
            if not reason:
                hold = max(i - entry_i, 0)
                giveback = (mfe - move) / max(abs(mfe), 1e-8) if mfe > 0.0 else 0.0
                expert = hard.EXPERT_NAMES[int(route[i])]
                prob = sidecar._predict_exit_prob_one(
                    base_np, exit_runtime, pos_idx, row_i=int(i), expert=expert,
                    pos_values=[float(pos), float(hold), float(move), float(mfe), float(mae),
                                float(np.clip(giveback, 0.0, 10.0)), float(take_profit - move),
                                float(move + abs(stop_loss)), float(NOTIONAL), float(LEVERAGE),
                                float(NOTIONAL * LEVERAGE), float(take_profit), float(stop_loss)],
                    device=device,
                )
                if prob >= EXIT_THRESHOLD:
                    reason = "exit_head"
                    exit_head_fires += 1
            if reason:
                exit_px = arrays["close"][i] * (1 - SLIP if pos > 0 else 1 + SLIP)
                raw_exit = (exit_px - entry_price) / entry_price if pos > 0 else (entry_price - exit_px) / entry_price
                cash = cash * (1.0 + raw_exit * NOTIONAL)
                cash -= cash * FEE * NOTIONAL
                trades += 1
                wins += int(cash > entry_equity)
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                pos = 0
            continue
        side = int(dec["side"].iloc[i])
        if side == 0:
            continue
        entry_px = arrays["open"][min(i + 1, n - 1)] * (1 + SLIP if side > 0 else 1 - SLIP)
        pos = side
        entry_price, entry_equity = float(entry_px), cash
        entry_i = min(i + 1, n - 1)
        cash -= cash * FEE * NOTIONAL
        take_profit, stop_loss = float(dec["take_profit"].iloc[i]), float(dec["stop_loss"].iloc[i])
        mfe = mae = 0.0
    return {"pnl": round((cash - 1.0) * 100.0, 2), "mdd": round(mdd * 100.0, 2), "trades": trades,
            "wr": round(wins / trades, 4) if trades else 0.0,
            "exit_head_fire_rate": round(exit_head_fires / trades, 4) if trades else 0.0,
            "reason_counts": reason_counts}


def main() -> int:
    device = parent3head._device("cpu")
    parent_preds = phase1._load_bundle_preds(PARENT_BUNDLE_DIR, device)

    rows = []
    for seed in PHASE2_SEEDS:
        bundle_dir = ROOT / f"tmp/causal_regen_20260516/eth_candidate_unified_phase2_exit_head_giveback_recal_20260817_seed{seed}"
        bundle = torch.load(bundle_dir / "true_3head_tabm_bundle.pt", map_location="cpu", weights_only=False)
        base_cols, models = bundle["base_cols"], bundle["models"]
        loaded = parent3head._load_payloads(models, device=device)

        for window_key, (start, end, base_csv) in phase1.WINDOWS.items():
            price = phase1._load_price_frame(base_csv)
            price_w = price[(price["timestamp"] >= start) & (price["timestamp"] <= end)].reset_index(drop=True)
            src = phase1._pred_source_for(pd.Timestamp(start), parent_preds)
            merged = price_w.merge(src, on="timestamp", how="left").merge(_route_cols_frame(base_csv), on="timestamp", how="left")
            n_missing = int(merged["dir_p_long"].isna().sum())
            if n_missing:
                if n_missing > 350:
                    raise RuntimeError(f"seed{seed}/{window_key}: {n_missing} bars missing predictions")
                merged = merged.dropna(subset=["dir_p_long"]).reset_index(drop=True)
            n_missing_route = int(merged[hard.ROUTE_COLS[0]].isna().sum())
            if n_missing_route:
                if n_missing_route > 350:
                    raise RuntimeError(f"seed{seed}/{window_key}: {n_missing_route} bars missing route probs")
                merged = merged.dropna(subset=hard.ROUTE_COLS).reset_index(drop=True)

            dec = _decisions_for_window(merged)
            base_x = parent3head._base_input(merged, base_cols)
            base_np, exit_runtime, pos_idx = sidecar._prepare_exit_runtime(base_x, loaded)
            route = hard._route_id(merged)

            m = _replay_with_exit_head(merged, dec, base_np, exit_runtime, pos_idx, route, device)
            rows.append({"seed": seed, "window": window_key, **{k: v for k, v in m.items() if k != "reason_counts"},
                         "reason_counts": str(m["reason_counts"])})
            log(f"seed={seed} window={window_key:8s} pnl={m['pnl']:+7.2f}% mdd={m['mdd']:+7.2f}% trades={m['trades']:3d} "
                f"wr={m['wr']*100:5.1f}% exit_head_fire={m['exit_head_fire_rate']*100:5.1f}%")

    df = pd.DataFrame(rows)
    out_dir = ROOT / "tmp/causal_regen_20260516/eth_candidate_unified_phase2_eval_20260818"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "per_seed_detail.csv", index=False)

    log("\n=== per window: mean across N=3 seeds (WITH exit_head, giveback_min=0.25, pre-fix) ===")
    agg = df.groupby("window").agg(mean_pnl=("pnl", "mean"), std_pnl=("pnl", "std"), mean_mdd=("mdd", "mean"),
                                    mean_trades=("trades", "mean"), mean_exit_head_fire=("exit_head_fire_rate", "mean")).round(3)
    print(agg.to_string())
    agg.to_csv(out_dir / "aggregated_by_window.csv")

    log("\n=== comparison: quality_B (threshold=0.80) WITHOUT exit_head, Phase 1 seed2559205075 only (same fixed sizing) ===")
    phase1_detail_path = ROOT / "tmp/causal_regen_20260516/eth_candidate_unified_phase1_eval_20260817/per_seed_detail.csv"
    if phase1_detail_path.exists():
        p1 = pd.read_csv(phase1_detail_path)
        p1 = p1[(p1["variant"] == "quality_B_samedir") & (p1["seed"] == 2559205075) & (p1["threshold"] == THRESHOLD)]
        print(p1[["window", "pnl", "mdd", "trades", "wr"]].to_string(index=False))
    else:
        log(f"  (Phase 1 detail not found at {phase1_detail_path})")

    log(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
