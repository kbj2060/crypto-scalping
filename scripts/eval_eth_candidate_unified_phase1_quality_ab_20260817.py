#!/usr/bin/env python3
"""RESEARCH ONLY -- Phase 1 evaluation for eth_candidate_unified_single_component_redesign
(docs/experiments/eth_candidate_unified_single_component_redesign_20260817.md). Evaluates the
10 trained bundles (2 quality-label variants x 5 genuinely random seeds) across all 9 swept
quality thresholds (0.45-0.85) and all 6 standard windows (VAL/OOS-Q1/OOS-Q2/2025Q1/Q2/Q3).

Deliberately simple, exit_head-free backtest: fixed accounting (BASE_TEMPLATE notional=0.45,
leverage=2.0 -- these candidates have no trained risk sidecar yet, Phase 1 isolates
direction+quality signal quality only, not sizing), ATR-scaled TP/SL floor (same formula/
constants as the live floor: atr_window=192, tp_mult=12, sl_mult=6, min_tp=0.075, min_sl=0.040,
max_tp=0.22, max_sl=0.12), no exit_head, no time limit -- matches zig075's own current live
design (pure TP/SL) for direct comparability with the existing single-vs-dual diagnostic.

Reads the raw per-bar direction/quality probabilities directly from each bundle's own
{train,validation}_predictions_q050.csv (any qXXX file has identical raw proba columns -- only
final_action/quality_threshold differ per file) and re-derives final_action at each of the 9
thresholds locally, avoiding needing to read 9 separate files per split.

IMPORTANT: the trainer's own built-in oos_predictions_q050.csv is truncated at 2026-02-28 (its
eval_df source, TABM_2026, is a fixed 2026-06-02 snapshot that was never extended) -- confirmed
by direct inspection this session. OOS-Q2 (Apr-Jun) would be silently missing if read naively.
Fixed the same way build_omega4_6_1_extended_parent_predictions_20260706.py already solved this
for the live h48qual/zig075 bundles (validated precedent, not a new approach): rebuild the frame
from the FULL BASE_2026 (data/splits/year_oos/training_features_2026_rebuilt.csv, confirmed this
session to extend to 2026-07-20) + the regime3 wide24 overlay, then run raw inference
(parent._predict_payload per expert + hard._route_id routing) directly against each Phase 1
bundle -- bypassing the truncated eval_df entirely.

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

import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent3head  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402

WIDE24_2026 = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/training_features_2026_rebuilt_regime3_current_sensitive_hmm_wide24.csv"

BUNDLE_ROOT = ROOT / "tmp/causal_regen_20260516"
DIR_PREFIX = "omega4_3head_parent72_loose_entry_quality_20260620_eth_candidate_unified_phase1_"
VARIANTS = ("quality_A_barrier", "quality_B_samedir")
SEEDS = (2559205075, 1355646609, 2549217127, 1801478137, 2105606360)
THRESHOLDS = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

BASE_2025 = ROOT / "data/splits/year_oos/training_features_2025.csv"
BASE_2026 = ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"

WINDOWS = {
    "2025q1": ("2025-01-01", "2025-03-31 23:59:59", BASE_2025),
    "2025q2": ("2025-04-01", "2025-06-30 23:59:59", BASE_2025),
    "2025q3": ("2025-07-01", "2025-09-30 23:59:59", BASE_2025),
    "val": ("2025-10-01", "2025-12-31 23:59:59", BASE_2025),
    "oos_q1": ("2026-01-01", "2026-03-31 23:59:59", BASE_2026),
    "oos_q2": ("2026-04-01", "2026-06-30 23:59:59", BASE_2026),
}

ATR_CFG = {"atr_window": 192, "tp_mult": 12.0, "sl_mult": 6.0, "min_tp": 0.075, "min_sl": 0.040, "max_tp": 0.22, "max_sl": 0.12}
NOTIONAL, LEVERAGE = 0.45, 2.0


def log(msg: str) -> None:
    print(msg, flush=True)


_price_cache: dict[Path, pd.DataFrame] = {}


def _load_price_frame(base_csv: Path) -> pd.DataFrame:
    if base_csv not in _price_cache:
        f = pd.read_csv(base_csv, usecols=["timestamp", "open", "high", "low", "close"], low_memory=False)
        f["timestamp"] = pd.to_datetime(f["timestamp"])
        f = f.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        _price_cache[base_csv] = f
    return _price_cache[base_csv]


def _pred_source_for(ts: pd.Timestamp, preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if ts < pd.Timestamp("2025-10-01"):
        return preds["train"]
    if ts < pd.Timestamp("2026-01-01"):
        return preds["validation"]
    return preds["oos"]


_extended_2026_frame_cache: pd.DataFrame | None = None


def _extended_2026_frame() -> pd.DataFrame:
    """Full-range 2026 frame (through 2026-07-20+), same construction as the already-validated
    build_omega4_6_1_extended_parent_predictions_20260706.py precedent -- BASE_2026 + wide24
    regime overlay, causal (no future leakage: overlay is a per-bar feature merge, not a label)."""
    global _extended_2026_frame_cache
    if _extended_2026_frame_cache is None:
        frame = pd.read_csv(BASE_2026, low_memory=False)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        overlay = pd.read_csv(WIDE24_2026, low_memory=False)
        overlay["timestamp"] = pd.to_datetime(overlay["timestamp"])
        cols = [c for c in overlay.columns if c != "timestamp"]
        merged = frame.merge(overlay[["timestamp", *cols]], on="timestamp", how="left", validate="one_to_one")
        # overlay has a small isolated 8h gap (2026-02-28 16:00 -> 03-01 00:00, 96 bars) and does
        # not extend past 2026-06-30 (frame goes to 07-20) -- neither affects the 6 windows this
        # script evaluates (OOS-Q2 ends 06-30), so drop the unusable rows rather than hard-fail.
        before = len(merged)
        merged = merged.dropna(subset=cols).reset_index(drop=True)
        dropped = before - len(merged)
        if dropped:
            log(f"  [_extended_2026_frame] dropped {dropped} rows missing regime3 overlay (known gap/tail, see comment)")
        _extended_2026_frame_cache = merged
    return _extended_2026_frame_cache


def _fresh_oos_preds(bundle_dir: Path, device: torch.device) -> pd.DataFrame:
    frame = _extended_2026_frame()
    bundle = torch.load(bundle_dir / "true_3head_tabm_bundle.pt", map_location="cpu", weights_only=False)
    base_cols, models = bundle["base_cols"], bundle["models"]
    x = parent3head._base_input(frame, base_cols)
    route = hard._route_id(frame)
    preds = {expert: parent3head._predict_payload(models[expert], x, device=device) for expert in hard.EXPERT_NAMES}
    direction = parent3head._routed(preds, route, "direction", 3)  # (n,3): cash/long/short
    quality = parent3head._routed(preds, route, "quality", 3)
    direction_action = direction.argmax(axis=1)
    quality_for_action = quality[np.arange(len(quality)), direction_action]
    return pd.DataFrame({
        "timestamp": frame["timestamp"], "dir_p_cash": direction[:, 0], "dir_p_long": direction[:, 1],
        "dir_p_short": direction[:, 2], "quality_for_action": quality_for_action,
    })


def _load_bundle_preds(bundle_dir: Path, device: torch.device) -> dict[str, pd.DataFrame]:
    out = {}
    for split in ("train", "validation"):
        p = pd.read_csv(bundle_dir / f"{split}_predictions_q050.csv")
        p["timestamp"] = pd.to_datetime(p["timestamp"])
        # normalize the long prefix (omega1_regime3_expertdq_oof_*) down to short names
        rename = {}
        for c in p.columns:
            if c == "timestamp":
                continue
            for suffix in ("dir_p_cash", "dir_p_long", "dir_p_short", "quality_for_action"):
                if c.endswith(suffix):
                    rename[c] = suffix
        p = p.rename(columns=rename)
        out[split] = p[["timestamp", "dir_p_cash", "dir_p_long", "dir_p_short", "quality_for_action"]]
    out["oos"] = _fresh_oos_preds(bundle_dir, device)
    return out


def _greedy_replay_no_exit(frame: pd.DataFrame, dec: pd.DataFrame, *, fee: float, slip: float) -> dict:
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    n = len(frame)
    cash = peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = entry_equity = 1.0
    take_profit = stop_loss = 0.0
    trades = 0
    wins = 0
    for i in range(0, n - 2):
        if pos != 0:
            move = (arrays["close"][i] * (1 - slip) - entry_price) / entry_price if pos > 0 else (entry_price - arrays["close"][i] * (1 + slip)) / entry_price
            eq = cash * (1.0 + move * NOTIONAL)
            peak = max(peak, eq)
            mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
            reason = ""
            if take_profit > 0.0 and move >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and move <= -abs(stop_loss):
                reason = "stop_loss"
            if reason:
                exit_px = arrays["close"][i] * (1 - slip if pos > 0 else 1 + slip)
                raw_exit = (exit_px - entry_price) / entry_price if pos > 0 else (entry_price - exit_px) / entry_price
                cash = cash * (1.0 + raw_exit * NOTIONAL)
                cash -= cash * fee * NOTIONAL
                trades += 1
                wins += int(cash > entry_equity)
                pos = 0
            continue
        side = int(dec["side"].iloc[i])
        if side == 0:
            continue
        entry_px = arrays["open"][min(i + 1, n - 1)] * (1 + slip if side > 0 else 1 - slip)
        pos = side
        entry_price, entry_equity = float(entry_px), cash
        cash -= cash * fee * NOTIONAL
        take_profit = float(dec["take_profit"].iloc[i])
        stop_loss = float(dec["stop_loss"].iloc[i])
    return {"pnl": round((cash - 1.0) * 100.0, 2), "mdd": round(mdd * 100.0, 2), "trades": trades,
            "wr": round(wins / trades, 4) if trades else 0.0}


def evaluate_variant_seed(variant: str, seed: int, device: torch.device) -> list[dict]:
    bundle_dir = BUNDLE_ROOT / f"{DIR_PREFIX}{variant}_seed{seed}"
    preds = _load_bundle_preds(bundle_dir, device)
    rows = []
    for window_key, (start, end, base_csv) in WINDOWS.items():
        price = _load_price_frame(base_csv)
        price_w = price[(price["timestamp"] >= start) & (price["timestamp"] <= end)].reset_index(drop=True)
        src = _pred_source_for(pd.Timestamp(start), preds)
        merged = price_w.merge(src, on="timestamp", how="left")
        n_missing = int(merged["dir_p_long"].isna().sum())
        if n_missing:
            if n_missing > 350:  # known isolated gaps, all verified as edge/tail cutoffs not mid-window holes: ~96 bars (overlay 2026-02-28->03-01 8h hole), ~287 bars (overlay tail ends exactly 2026-06-30 00:00, so OOS-Q2's last day is missing), and <=30 bars elsewhere
                raise RuntimeError(f"{variant}/{seed}/{window_key}: {n_missing} bars missing predictions -- too many to be an isolated gap")
            merged = merged.dropna(subset=["dir_p_long"]).reset_index(drop=True)

        direction_action = np.select(
            [merged["dir_p_long"] > merged[["dir_p_cash", "dir_p_short"]].max(axis=1),
             merged["dir_p_short"] > merged[["dir_p_cash", "dir_p_long"]].max(axis=1)],
            [1, 2], default=0,
        )
        quality_for_action = merged["quality_for_action"].to_numpy(dtype=np.float64)
        atr_pct = atr_eval._atr_pct(merged, ATR_CFG["atr_window"])

        for thr in THRESHOLDS:
            final_action = np.where((direction_action != 0) & (quality_for_action >= thr), direction_action, 0)
            side = np.where(final_action == 1, 1, np.where(final_action == 2, -1, 0))
            tp_move = np.clip(np.maximum(ATR_CFG["min_tp"], atr_pct * ATR_CFG["tp_mult"]), 0.0, ATR_CFG["max_tp"])
            sl_move = np.clip(np.maximum(ATR_CFG["min_sl"], atr_pct * ATR_CFG["sl_mult"]), 0.0, ATR_CFG["max_sl"])
            dec = pd.DataFrame({"side": side, "take_profit": np.where(side != 0, tp_move, 0.0), "stop_loss": np.where(side != 0, sl_move, 0.0)})
            m = _greedy_replay_no_exit(merged, dec, fee=0.0005, slip=0.0002)
            rows.append({"variant": variant, "seed": seed, "window": window_key, "threshold": thr, **m})
    return rows


def main() -> int:
    device = parent3head._device("cpu")
    all_rows = []
    for variant in VARIANTS:
        for seed in SEEDS:
            log(f"=== evaluating {variant} seed={seed} ===")
            all_rows.extend(evaluate_variant_seed(variant, seed, device))

    df = pd.DataFrame(all_rows)
    out_dir = ROOT / "tmp/causal_regen_20260516/eth_candidate_unified_phase1_eval_20260817"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "per_seed_detail.csv", index=False)

    log("\n=== per (variant, window, threshold): mean PnL/MDD across 5 seeds ===")
    agg = df.groupby(["variant", "window", "threshold"]).agg(
        mean_pnl=("pnl", "mean"), std_pnl=("pnl", "std"), mean_mdd=("mdd", "mean"),
        mean_trades=("trades", "mean"),
    ).round(2).reset_index()
    print(agg.to_string(index=False))
    agg.to_csv(out_dir / "aggregated_by_threshold_window.csv", index=False)
    log(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
