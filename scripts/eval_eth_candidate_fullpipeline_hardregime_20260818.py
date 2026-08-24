#!/usr/bin/env python3
"""RESEARCH ONLY -- Phase 2 full-pipeline evaluation (design doc section 3-C / 5): unlike Phase 1's
exit_head-free backtest (eval_eth_candidate_unified_phase1_quality_ab_20260817.py), this INCLUDES
the exit head in the replay -- direction+quality come from the frozen Phase 1 Variant B parent
(quality_B_samedir seed=2559205075, threshold=0.80, the confirmed-best config), exit-head decisions
come from one of the two paired full-pipeline retrains (soft-weight control vs hard-regime-filter
treatment, both seed=2559205075, both trained 2026-08-18 -- see
eth_candidate_fullpipeline_{soft,hard}_seed2559205075 on the server).

Still fixed sizing (notional=0.45, leverage=2.0, no risk sidecar -- Phase 3's job), same as Phase 1
-- this isolates "does the exit head change the picture" from "does real risk sizing change the
picture". exit_threshold=0.95 matches the live EXIT_THRESHOLD constant
(trading_bot_modules/odyssey_live_adapter.py / omega4_6_1_live.py) for production comparability.

Per-bar exit-head inference pattern (_predict_exit_prob_one, pos_values 13-tuple order) copied from
train_eval_omega4_2_risk_sidecar_20260622.py::_replay_with_risk, simplified to fixed sizing (no
risk-sidecar margin/leverage prediction).

fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false, saved_parent_exit_timestamps_
used=false, future_rows_used_for_entry=false.
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

import eval_eth_candidate_unified_phase1_quality_ab_20260817 as phase1eval  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent3head  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as sidecar  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402

PARENT_BUNDLE_DIR = ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_eth_candidate_unified_phase1_quality_B_samedir_seed2559205075"
EXIT_BUNDLES = {
    "soft": ROOT / "tmp/causal_regen_20260516/eth_candidate_unified_phase2_exit_head_giveback_recal_20260817_fullpipeline_soft_seed2559205075/true_3head_tabm_bundle.pt",
    "hard": ROOT / "tmp/causal_regen_20260516/eth_candidate_unified_phase2_exit_head_giveback_recal_20260817_fullpipeline_hard_seed2559205075/true_3head_tabm_bundle.pt",
}
THRESHOLD = 0.80  # Phase 1 confirmed sweet spot for quality_B_samedir
EXIT_THRESHOLD = 0.95  # matches live EXIT_THRESHOLD (trading_bot_modules/odyssey_live_adapter.py)
NOTIONAL, LEVERAGE = phase1eval.NOTIONAL, phase1eval.LEVERAGE
WIDE24_2025 = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/training_features_2025_regime3_current_sensitive_hmm_wide24.csv"


def log(msg: str) -> None:
    print(msg, flush=True)


_full_2025_frame_cache: pd.DataFrame | None = None


def _full_2025_frame() -> pd.DataFrame:
    """BASE_2025 has no route columns natively (unlike BASE_2026, which _extended_2026_frame
    already patches) -- same fix, mirrored for 2025: merge in the wide24 regime3 overlay."""
    global _full_2025_frame_cache
    if _full_2025_frame_cache is None:
        frame = pd.read_csv(phase1eval.BASE_2025, low_memory=False)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        overlay = pd.read_csv(WIDE24_2025, low_memory=False)
        overlay["timestamp"] = pd.to_datetime(overlay["timestamp"])
        cols = [c for c in overlay.columns if c != "timestamp"]
        merged = frame.merge(overlay[["timestamp", *cols]], on="timestamp", how="left", validate="one_to_one")
        before = len(merged)
        merged = merged.dropna(subset=cols).reset_index(drop=True)
        dropped = before - len(merged)
        if dropped:
            log(f"  [_full_2025_frame] dropped {dropped} rows missing regime3 overlay")
        _full_2025_frame_cache = merged
    return _full_2025_frame_cache


def _full_frame_for(base_csv: Path) -> pd.DataFrame:
    return _full_2025_frame() if base_csv == phase1eval.BASE_2025 else phase1eval._extended_2026_frame()


@torch.no_grad()
def _greedy_replay_with_exit(frame: pd.DataFrame, dec: pd.DataFrame, exit_loaded, *, fee: float, slip: float, device: torch.device) -> dict:
    base_x = parent3head._base_input(frame, list(next(iter(exit_loaded.values()))[1]["columns"]))
    base_np, exit_runtime, pos_idx = sidecar._prepare_exit_runtime(base_x, exit_loaded)
    route = hard._route_id(frame)
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    n = len(frame)
    cash = peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = entry_equity = 1.0
    entry_i = 0
    take_profit = stop_loss = 0.0
    mfe = mae = 0.0
    trades = 0
    wins = 0
    reasons: dict[str, int] = {}
    for i in range(0, n - 2):
        if pos != 0:
            move = (arrays["close"][i] * (1 - slip) - entry_price) / entry_price if pos > 0 else (entry_price - arrays["close"][i] * (1 + slip)) / entry_price
            mfe = max(mfe, move)
            mae = min(mae, move)
            eq = cash * (1.0 + move * NOTIONAL)
            peak = max(peak, eq)
            mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
            reason = ""
            if take_profit > 0.0 and move >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and move <= -abs(stop_loss):
                reason = "stop_loss"
            else:
                hold = max(int(i) - int(entry_i), 0)
                giveback = (float(mfe) - float(move)) / max(abs(float(mfe)), 1.0e-8) if mfe > 0.0 else 0.0
                expert = hard.EXPERT_NAMES[int(route[i])]
                prob = sidecar._predict_exit_prob_one(
                    base_np, exit_runtime, pos_idx, row_i=int(i), expert=expert,
                    pos_values=[
                        float(pos), float(hold), float(move), float(mfe), float(mae),
                        float(np.clip(giveback, 0.0, 10.0)), float(take_profit - move), float(move + abs(stop_loss)),
                        float(NOTIONAL), float(LEVERAGE), float(NOTIONAL * LEVERAGE), float(take_profit), float(stop_loss),
                    ],
                    device=device,
                )
                if prob >= EXIT_THRESHOLD:
                    reason = "exit_head"
            if reason:
                exit_px = arrays["close"][i] * (1 - slip if pos > 0 else 1 + slip)
                raw_exit = (exit_px - entry_price) / entry_price if pos > 0 else (entry_price - exit_px) / entry_price
                cash = cash * (1.0 + raw_exit * NOTIONAL)
                cash -= cash * fee * NOTIONAL
                trades += 1
                wins += int(cash > entry_equity)
                reasons[reason] = reasons.get(reason, 0) + 1
                pos = 0
                mfe = mae = 0.0
            continue
        side = int(dec["side"].iloc[i])
        if side == 0:
            continue
        entry_px = arrays["open"][min(i + 1, n - 1)] * (1 + slip if side > 0 else 1 - slip)
        pos = side
        entry_price, entry_equity = float(entry_px), cash
        entry_i = i
        cash -= cash * fee * NOTIONAL
        take_profit = float(dec["take_profit"].iloc[i])
        stop_loss = float(dec["stop_loss"].iloc[i])
    return {"pnl": round((cash - 1.0) * 100.0, 2), "mdd": round(mdd * 100.0, 2), "trades": trades,
            "wr": round(wins / trades, 4) if trades else 0.0, "reasons": reasons}


def main() -> int:
    device = parent3head._device("cpu")
    log(f"=== loading parent (direction+quality) predictions: {PARENT_BUNDLE_DIR.name} ===")
    parent_preds = phase1eval._load_bundle_preds(PARENT_BUNDLE_DIR, device)

    exit_loaded = {}
    for tag, bpath in EXIT_BUNDLES.items():
        bundle = torch.load(bpath, map_location=device, weights_only=False)
        exit_loaded[tag] = parent3head._load_payloads(bundle["models"], device=device)
        log(f"  loaded exit bundle '{tag}': {bpath}")

    rows = []
    for window_key, (start, end, base_csv) in phase1eval.WINDOWS.items():
        full = _full_frame_for(base_csv)
        full_w = full[(full["timestamp"] >= start) & (full["timestamp"] <= end)].reset_index(drop=True)
        src = phase1eval._pred_source_for(pd.Timestamp(start), parent_preds)
        merged = full_w.merge(src, on="timestamp", how="left")
        n_missing = int(merged["dir_p_long"].isna().sum())
        if n_missing:
            if n_missing > 350:
                raise RuntimeError(f"{window_key}: {n_missing} bars missing predictions -- too many to be an isolated gap")
            merged = merged.dropna(subset=["dir_p_long"]).reset_index(drop=True)

        direction_action = np.select(
            [merged["dir_p_long"] > merged[["dir_p_cash", "dir_p_short"]].max(axis=1),
             merged["dir_p_short"] > merged[["dir_p_cash", "dir_p_long"]].max(axis=1)],
            [1, 2], default=0,
        )
        quality_for_action = merged["quality_for_action"].to_numpy(dtype=np.float64)
        atr_pct = phase1eval.atr_eval._atr_pct(merged, phase1eval.ATR_CFG["atr_window"])
        final_action = np.where((direction_action != 0) & (quality_for_action >= THRESHOLD), direction_action, 0)
        side = np.where(final_action == 1, 1, np.where(final_action == 2, -1, 0))
        tp_move = np.clip(np.maximum(phase1eval.ATR_CFG["min_tp"], atr_pct * phase1eval.ATR_CFG["tp_mult"]), 0.0, phase1eval.ATR_CFG["max_tp"])
        sl_move = np.clip(np.maximum(phase1eval.ATR_CFG["min_sl"], atr_pct * phase1eval.ATR_CFG["sl_mult"]), 0.0, phase1eval.ATR_CFG["max_sl"])
        dec = pd.DataFrame({"side": side, "take_profit": np.where(side != 0, tp_move, 0.0), "stop_loss": np.where(side != 0, sl_move, 0.0)})

        for tag in ("soft", "hard"):
            m = _greedy_replay_with_exit(merged, dec, exit_loaded[tag], fee=0.0005, slip=0.0002, device=device)
            log(f"  {window_key}/{tag}: pnl={m['pnl']} mdd={m['mdd']} trades={m['trades']} wr={m['wr']} reasons={m['reasons']}")
            rows.append({"window": window_key, "variant": tag, **m})

    df = pd.DataFrame(rows)
    out_dir = ROOT / "tmp/causal_regen_20260516/eth_candidate_fullpipeline_hardregime_eval_20260818"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "detail.csv", index=False)
    pivot = df.pivot(index="window", columns="variant", values="pnl")
    pivot["delta_hard_minus_soft"] = (pivot["hard"] - pivot["soft"]).round(2)
    log("\n=== PnL: soft vs hard (exit-head-inclusive, threshold=0.80, exit_threshold=0.95) ===")
    print(pivot.to_string())
    pivot.to_csv(out_dir / "pivot_pnl.csv")
    log(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
