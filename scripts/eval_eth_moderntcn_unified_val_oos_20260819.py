#!/usr/bin/env python3
"""RESEARCH ONLY -- evaluates the ModernTCN UNIFIED (non-split) single-seed pilot (train_eval_eth_
moderntcn_direction_quality_regime_hardsplit_20260818.py --split unified --seed 839864) on REAL VAL/
OOS periods, same PnL backtest methodology as eval_eth_moderntcn_regime_hardsplit_val_oos_20260818.py
(notional=0.45, leverage=2.0, ATR TP/SL) minus the regime routing -- one model, no route lookup.

2026-08-19: comparison axis requested by user against the N=3-seed hard-split result (which was
broadly negative, 8/9 threshold cells, see that eval's own output) to check whether hard-splitting by
regime is itself the problem or whether ModernTCN is weak here regardless of split. Single seed only
(839864, reused from the hard-split run for direct seed-level comparability) -- NOT a promotion-grade
N>=5 seed check (CLAUDE.md Seed-Diversity Gate), a one-point directional read only.

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

import train_eval_eth_moderntcn_direction_quality_regime_hardsplit_20260818 as hs  # noqa: E402
import train_eval_eth_direction_quality_nhits_moderntcn_20260816 as base_nt  # noqa: E402
import eval_eth_moderntcn_regime_hardsplit_val_oos_20260818 as hard_eval  # noqa: E402

BUNDLE_DIR = ROOT / "tmp/causal_regen_20260516/eth_moderntcn_direction_quality_regime_hardsplit_20260818_unified_seed839864"


def log(msg: str) -> None:
    print(f"[eval_moderntcn_unified] {msg}", flush=True)


def load_model_and_scaler(data: dict, device: torch.device) -> dict:
    window = base_nt.DEFAULT_WINDOW
    train_mask = ((data["panel"]["timestamp"] >= hs.TRAIN_START) & (data["panel"]["timestamp"] <= hs.TRAIN_END)).to_numpy()
    train_idx_all = base_nt._valid_indices(train_mask, window, data["y_dir_full"], data["y_qual_full"])
    fit_idx, _es_idx = base_nt._split_with_embargo(train_idx_all, window)
    raw_std, _stats = base_nt._standardize_fit(data["raw"], fit_idx, window)

    ckpt_path = BUNDLE_DIR / "models" / "unified_moderntcn_regime_hardsplit.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    backbone = base_nt.build_backbone(ckpt["arch"], len(base_nt.SEQ_COLS), ckpt["window"], ckpt["arch_params"]).to(device)
    model = base_nt.TwoHeadClassifier(backbone, backbone.hidden_dim).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    log(f"  loaded unified: fit_rows={len(fit_idx)} window={ckpt['window']}")
    return {"model": model, "raw_std": raw_std, "window": ckpt["window"]}


@torch.no_grad()
def _predict(idx_arr: np.ndarray, m: dict, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    raw_std, window, model = m["raw_std"], m["window"], m["model"]
    n = len(idx_arr)
    dir_probs = np.zeros((n, 3), dtype=np.float64)
    quality_for_action = np.zeros(n, dtype=np.float64)
    batch_size = 2048
    for start in range(0, n, batch_size):
        sel = np.arange(start, min(start + batch_size, n))
        batch = np.stack([raw_std[idx_arr[j] - window + 1: idx_arr[j] + 1].T for j in sel], axis=0)
        xb = torch.from_numpy(batch.astype(np.float32)).to(device)
        out = model(xb)
        pdir = torch.softmax(out["direction"], dim=-1).mean(dim=1).cpu().numpy()
        pqual = torch.softmax(out["quality"], dim=-1).mean(dim=1).cpu().numpy()
        dir_probs[sel] = pdir
        action = pdir.argmax(axis=1)
        quality_for_action[sel] = pqual[np.arange(len(sel)), action]
    return dir_probs, quality_for_action


def main() -> int:
    device = base_nt._device("cpu")

    log("loading panel+labels+regime route...")
    data = hs.load_data_samedir_with_regime()
    log("loading unified model + re-deriving TRAIN-time standardization...")
    m = load_model_and_scaler(data, device)
    window = m["window"]

    rows = []
    for wname, (start, end) in hard_eval.WINDOWS.items():
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        mask = (data["panel"]["timestamp"] >= start_ts) & (data["panel"]["timestamp"] <= end_ts)
        idx_arr = np.flatnonzero(mask.to_numpy())
        idx_arr = idx_arr[idx_arr >= window - 1]
        if len(idx_arr) == 0:
            log(f"  {wname}: 0 valid rows, skipping")
            continue

        dir_probs, quality_for_action = _predict(idx_arr, m, device)
        price = data["panel"].iloc[idx_arr][["open", "high", "low", "close"]].reset_index(drop=True)
        wide_start = max(0, idx_arr[0] - hard_eval.ATR_CFG["atr_window"] - 5)
        wide = data["panel"].iloc[wide_start: idx_arr[-1] + 1].reset_index(drop=True)
        atr_pct_wide = hard_eval.atr_eval._atr_pct(wide, hard_eval.ATR_CFG["atr_window"])
        atr_pct = atr_pct_wide[idx_arr - wide_start]

        direction_action = dir_probs.argmax(axis=1)
        tp_move = np.clip(np.maximum(hard_eval.ATR_CFG["min_tp"], atr_pct * hard_eval.ATR_CFG["tp_mult"]), 0.0, hard_eval.ATR_CFG["max_tp"])
        sl_move = np.clip(np.maximum(hard_eval.ATR_CFG["min_sl"], atr_pct * hard_eval.ATR_CFG["sl_mult"]), 0.0, hard_eval.ATR_CFG["max_sl"])

        for thr in hard_eval.THRESHOLDS:
            final_action = np.where((direction_action != 0) & (quality_for_action >= thr), direction_action, 0)
            side = np.where(final_action == 1, 1, np.where(final_action == 2, -1, 0))
            take_profit = np.where(side != 0, tp_move, 0.0)
            stop_loss = np.where(side != 0, sl_move, 0.0)
            r = hard_eval._greedy_replay_no_exit(price, side, take_profit, stop_loss, fee=0.0005, slip=0.0002)
            log(f"  {wname}/thr={thr}: rows={len(idx_arr)} pnl={r['pnl']} mdd={r['mdd']} trades={r['trades']} wr={r['wr']}")
            rows.append({"window": wname, "threshold": thr, **r})

    df = pd.DataFrame(rows)
    out_dir = ROOT / "tmp/causal_regen_20260516/eth_moderntcn_unified_eval_20260819"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "pnl_by_window_threshold_sweep.csv", index=False)
    log("\n=== ModernTCN unified (non-split), 1-seed, 9-point threshold sweep, real VAL/OOS ===")
    print(df.pivot(index="threshold", columns="window", values="pnl").to_string())
    log(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
