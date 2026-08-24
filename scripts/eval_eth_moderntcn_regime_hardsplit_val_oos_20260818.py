#!/usr/bin/env python3
"""RESEARCH ONLY -- evaluates the ModernTCN regime-hard-split FINAL N=3-seed models (train_eval_eth_
moderntcn_direction_quality_regime_hardsplit_20260818.py --stage final) on REAL VAL/OOS periods,
using the SAME fixed-sizing/no-exit_head PnL backtest methodology as Phase 1's eval_eth_candidate_
unified_phase1_quality_ab_20260817.py (notional=0.45, leverage=2.0, ATR TP/SL).

2026-08-19: N=3 not N=5 -- user stopped the 5-seed run early (time cost) after seeds 839864/503468/
587472 completed; seed 954073's bull regime never finished (no report.json written) and seed 120968
never started. Per CLAUDE.md's Seed-Diversity Ensemble Promotion Gate (N>=5 genuinely-random seeds
required to claim OOS sign-consistency for a seed-averaged/bagged model), this run's results are NOT
valid promotion evidence -- reported here as a directional research readout only, with per-seed
numbers shown alongside the mean so the reader can see the actual spread, not just an average that
implies more confidence than 3 points support.

No TabM hard-split baseline comparison in this script: audited 2026-08-19 and found every existing
"regime_hard_split": true report.json in tmp/causal_regen_20260516/ is one of THIS ModernTCN run's own
seed bundles -- the hard-split TabM parent was never retrained on the same extended TRAIN_END=2026-02-
28/VAL=2026-03-01..04-30/OOS=2026-05-01..06-30 windows this script uses. The only existing hard-split
TabM numbers (tmp/causal_regen_20260516/eth_candidate_hardregime_pilot_eval_20260818/pivot_pnl.csv)
use the OLD pre-extension windows (2025q1-q3/oos_q1/oos_q2/val) which are now inside ModernTCN's TRAIN
range -- comparing against them would be an in-sample-vs-out-of-sample mismatch, not a fair baseline.
A same-window TabM retrain is a separate, not-yet-done follow-up if that comparison is wanted.

Why a separate eval, not the training script's own internal metric: the training script's
"selected_bacc" is measured on an EMBARGOED HOLD-OUT SLICE OF TRAIN, not real forward VAL/OOS -- not
comparable to anything, and per this repo's repeated finding, classification accuracy and PnL are
frequently decoupled anyway (every TCN experiment this session found this). This script recomputes
each regime's exact TRAIN-time standardization (mean/std), deterministically re-derived by reusing
load_data_samedir_with_regime/_valid_indices_regime/_split_with_embargo unmodified (none of those
depend on the training seed -- only fit/init do), then does real regime-routed inference on VAL/OOS
bars, once per seed bundle.

fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false, saved_parent_exit_timestamps_
used=false, future_rows_used_for_entry=false (each bar's window only reaches backward; VAL/OOS
labels are never used for anything but backtest bookkeeping).
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

import train_eval_eth_moderntcn_direction_quality_regime_hardsplit_20260818 as hs  # noqa: E402
import train_eval_eth_direction_quality_nhits_moderntcn_20260816 as base_nt  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402

THRESHOLDS = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]  # same 9-point grid as Phase 1's TabM sweep -- 0.80 was never confirmed optimal for ModernTCN specifically
ATR_CFG = {"atr_window": 192, "tp_mult": 12.0, "sl_mult": 6.0, "min_tp": 0.075, "min_sl": 0.040, "max_tp": 0.22, "max_sl": 0.12}
NOTIONAL, LEVERAGE = 0.45, 2.0

WINDOWS = {
    # 2025Q1-2026Q1 are all inside the extended TRAIN range (2025-01-01..2026-02-28) as of the
    # 2026-08-18 data-extension retrain -- no longer meaningful eval windows, dropped rather than
    # reported as if they were still held-out (see train script docstring point 4).
    "val": (hs.VAL_START, hs.VAL_END),
    "oos": (hs.OOS_START, hs.OOS_END),
}


def log(msg: str) -> None:
    print(f"[eval_moderntcn] {msg}", flush=True)


def load_models_and_scalers(bundle_dir: Path, data: dict, device: torch.device) -> dict[str, dict]:
    out = {}
    for regime_idx, expert in enumerate(hard.EXPERT_NAMES):
        train_mask = ((data["panel"]["timestamp"] >= hs.TRAIN_START) & (data["panel"]["timestamp"] <= hs.TRAIN_END)).to_numpy()
        train_idx_all = hs._valid_indices_regime(train_mask, base_nt.DEFAULT_WINDOW, data["y_dir_full"], data["y_qual_full"], data["route_id_full"], regime_idx)
        fit_idx, _es_idx = base_nt._split_with_embargo(train_idx_all, base_nt.DEFAULT_WINDOW)
        raw_std, _stats = base_nt._standardize_fit(data["raw"], fit_idx, base_nt.DEFAULT_WINDOW)

        ckpt_path = bundle_dir / "models" / f"{expert}_moderntcn_regime_hardsplit.pt"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        backbone = base_nt.build_backbone(ckpt["arch"], len(base_nt.SEQ_COLS), ckpt["window"], ckpt["arch_params"]).to(device)
        model = base_nt.TwoHeadClassifier(backbone, backbone.hidden_dim).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        out[expert] = {"model": model, "raw_std": raw_std, "window": ckpt["window"]}
        log(f"  loaded {expert}: fit_rows={len(fit_idx)} window={ckpt['window']}")
    return out


@torch.no_grad()
def _regime_routed_predictions(idx_arr: np.ndarray, route_id_full: np.ndarray, regime_models: dict[str, dict], device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Returns (direction_probs[n,3], quality_for_action[n]) aligned to idx_arr order."""
    n = len(idx_arr)
    dir_probs = np.zeros((n, 3), dtype=np.float64)
    quality_for_action = np.zeros(n, dtype=np.float64)
    route_here = route_id_full[idx_arr]
    for regime_idx, expert in enumerate(hard.EXPERT_NAMES):
        sel = np.flatnonzero(route_here == regime_idx)
        if len(sel) == 0:
            continue
        m = regime_models[expert]
        raw_std, window, model = m["raw_std"], m["window"], m["model"]
        batch = np.stack([raw_std[idx_arr[j] - window + 1: idx_arr[j] + 1].T for j in sel], axis=0)  # (B, C, WINDOW)
        xb = torch.from_numpy(batch.astype(np.float32)).to(device)
        out = model(xb)
        pdir = torch.softmax(out["direction"], dim=-1).mean(dim=1).cpu().numpy()  # (B,3)
        pqual = torch.softmax(out["quality"], dim=-1).mean(dim=1).cpu().numpy()  # (B,3)
        dir_probs[sel] = pdir
        action = pdir.argmax(axis=1)
        quality_for_action[sel] = pqual[np.arange(len(sel)), action]
    return dir_probs, quality_for_action


def _greedy_replay_no_exit(price: pd.DataFrame, side: np.ndarray, take_profit: np.ndarray, stop_loss: np.ndarray, *, fee: float, slip: float) -> dict:
    arrays = {c: pd.to_numeric(price[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    n = len(price)
    cash = peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = entry_equity = 1.0
    tp = sl = 0.0
    trades = wins = 0
    for i in range(0, n - 2):
        if pos != 0:
            move = (arrays["close"][i] * (1 - slip) - entry_price) / entry_price if pos > 0 else (entry_price - arrays["close"][i] * (1 + slip)) / entry_price
            eq = cash * (1.0 + move * NOTIONAL)
            peak = max(peak, eq)
            mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
            reason = ""
            if tp > 0.0 and move >= tp:
                reason = "take_profit"
            elif sl > 0.0 and move <= -abs(sl):
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
        s = int(side[i])
        if s == 0:
            continue
        entry_px = arrays["open"][min(i + 1, n - 1)] * (1 + slip if s > 0 else 1 - slip)
        pos = s
        entry_price, entry_equity = float(entry_px), cash
        cash -= cash * fee * NOTIONAL
        tp, sl = float(take_profit[i]), float(stop_loss[i])
    return {"pnl": round((cash - 1.0) * 100.0, 2), "mdd": round(mdd * 100.0, 2), "trades": trades,
            "wr": round(wins / trades, 4) if trades else 0.0}


SEED_BUNDLES = {
    839864: ROOT / "tmp/causal_regen_20260516/eth_moderntcn_direction_quality_regime_hardsplit_20260818_final_seed839864",
    503468: ROOT / "tmp/causal_regen_20260516/eth_moderntcn_direction_quality_regime_hardsplit_20260818_final_seed503468",
    587472: ROOT / "tmp/causal_regen_20260516/eth_moderntcn_direction_quality_regime_hardsplit_20260818_final_seed587472",
}


def main() -> int:
    device = base_nt._device("cpu")

    log("loading panel+labels+regime route...")
    data = hs.load_data_samedir_with_regime()
    window = base_nt.DEFAULT_WINDOW

    rows = []
    for seed, bundle_dir in SEED_BUNDLES.items():
        log(f"=== seed={seed} ===")
        log("  loading trained models + re-deriving TRAIN-time standardization...")
        regime_models = load_models_and_scalers(bundle_dir, data, device)

        for wname, (start, end) in WINDOWS.items():
            start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
            mask = (data["panel"]["timestamp"] >= start_ts) & (data["panel"]["timestamp"] <= end_ts)
            idx_arr = np.flatnonzero(mask.to_numpy())
            idx_arr = idx_arr[idx_arr >= window - 1]
            route_valid = data["route_id_full"][idx_arr] >= 0
            n_missing_route = int((~route_valid).sum())
            idx_arr = idx_arr[route_valid]
            if len(idx_arr) == 0:
                log(f"  {wname}: 0 valid rows (route missing={n_missing_route}), skipping")
                continue

            dir_probs, quality_for_action = _regime_routed_predictions(idx_arr, data["route_id_full"], regime_models, device)
            price = data["panel"].iloc[idx_arr][["open", "high", "low", "close"]].reset_index(drop=True)
            # atr_pct needs enough backward history within the sliced frame; compute on a wider slice to avoid edge truncation.
            wide_start = max(0, idx_arr[0] - ATR_CFG["atr_window"] - 5)
            wide = data["panel"].iloc[wide_start: idx_arr[-1] + 1].reset_index(drop=True)
            atr_pct_wide = atr_eval._atr_pct(wide, ATR_CFG["atr_window"])
            atr_pct = atr_pct_wide[idx_arr - wide_start]

            direction_action = dir_probs.argmax(axis=1)  # 0=cash,1=long,2=short
            tp_move = np.clip(np.maximum(ATR_CFG["min_tp"], atr_pct * ATR_CFG["tp_mult"]), 0.0, ATR_CFG["max_tp"])
            sl_move = np.clip(np.maximum(ATR_CFG["min_sl"], atr_pct * ATR_CFG["sl_mult"]), 0.0, ATR_CFG["max_sl"])

            for thr in THRESHOLDS:
                final_action = np.where((direction_action != 0) & (quality_for_action >= thr), direction_action, 0)
                side = np.where(final_action == 1, 1, np.where(final_action == 2, -1, 0))
                take_profit = np.where(side != 0, tp_move, 0.0)
                stop_loss = np.where(side != 0, sl_move, 0.0)
                m = _greedy_replay_no_exit(price, side, take_profit, stop_loss, fee=0.0005, slip=0.0002)
                log(f"  seed={seed}/{wname}/thr={thr}: rows={len(idx_arr)} route_missing={n_missing_route} pnl={m['pnl']} mdd={m['mdd']} trades={m['trades']} wr={m['wr']}")
                rows.append({"seed": seed, "window": wname, "threshold": thr, **m})

    df = pd.DataFrame(rows)
    out_dir = ROOT / "tmp/causal_regen_20260516/eth_moderntcn_regime_hardsplit_eval_20260818_final_3seed"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "pnl_by_seed_window_threshold_sweep.csv", index=False)

    agg = df.groupby(["window", "threshold"])["pnl"].agg(["mean", "std"]).reset_index()
    agg.to_csv(out_dir / "pnl_mean_std_across_3seeds.csv", index=False)
    log("\n=== ModernTCN regime-hard-split, N=3 seeds, 9-point threshold sweep, real VAL/OOS -- mean PnL ===")
    print(agg.pivot(index="threshold", columns="window", values="mean").to_string())
    log("\n=== ...same, std across seeds (N=3, NOT a promotion-grade seed-diversity gate per CLAUDE.md) ===")
    print(agg.pivot(index="threshold", columns="window", values="std").to_string())
    log(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
