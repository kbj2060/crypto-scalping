#!/usr/bin/env python3
"""RESEARCH ONLY -- ETH conformal veto step 2: build the episode-start label dataset the conformal downside
regressors will train/calibrate on (docs/model_contracts/eth_candidate_conformal_downside_veto_
contract_20260816.md). User chose (A) proceed to the real regression model despite the cheap_gate's
ambiguous result (docs/experiments/eth_candidate_conformal_veto_cheap_gate_20260816.md).

For each component (h48qual, zig075) and each of 2025q1/2025q2/2025q3 (training pool) + val
(calibration), find every quality-gate-passing signal EPISODE (first bar of each contiguous
side!=0-and-active run -- NOT every realized/slot-winning trade, which the contract's feasibility
check found is 40x too sparse to train on). For each episode-start bar, run a standalone
counterfactual forward simulation (entry_idx/side/TP/SL/exit_head only -- no portfolio, no other
component, no slot competition) to get the two BTC-style regression targets:
  - full: net price-move return if this signal alone had been taken and held to its own exit
    (TP/SL/exit_head, same rule this component actually uses -- for h48qual, honors the Odyssey3
    regime-guard exit_head weight switch at each bar exactly like the real replay).
  - adverse: magnitude of the worst intra-trade drawdown (MAE) against the position.
Entry-time causal features = the component's own base_np row at the signal bar, MINUS the 13
POS_COLS (position-state placeholders, zero at entry -- excluded the same way BTC's editor module
excludes DYNAMIC_FEATURES).

OOS-Q1/OOS-Q2 are NEVER loaded by this script -- training-pool and calibration windows only.

fresh_forward_bar_by_bar=true (each simulation only uses bars from its own entry forward).
trade_ledgers_used_as_input=false (labels are freshly simulated per episode, not read from any
saved ledger). saved_parent_exit_timestamps_used=false. future_rows_used_for_entry=false (the
episode-start bar's OWN features are causal; the simulation walks forward from there).

Does NOT touch trading_bot.py / trading_bot_modules/omega4_6_1_live.py /
trading_bot_modules/runtime_config.py / .env. Does NOT modify any imported module. No GPU
(DEVICE=cpu).
"""
from __future__ import annotations

import json
import sys
import time
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

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as rs  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
import research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814 as guard  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_candidate_conformal_veto_episode_labels_20260816"
DEVICE = torch.device("cpu")

TRAIN_WINDOWS = ("2025q1", "2025q2", "2025q3")
CALIBRATION_WINDOW = "val"
GUARD_COMPONENT = "h48qual"
MAX_HORIZON_BARS = 2000  # ~7 days at 5m bars -- safety cap, capped episodes are flagged not silently dropped


def log(msg: str) -> None:
    print(f"[candidate_conformal_veto_episode_labels] {msg}", flush=True)


def _episode_starts(side: np.ndarray, active: np.ndarray) -> np.ndarray:
    sig = (side != 0) & active
    starts = sig.copy()
    starts[1:] &= ~sig[:-1]
    return starts


@torch.no_grad()
def _simulate_episode(
    name: str,
    comp: dict[str, Any],
    arrays: dict[str, np.ndarray],
    n: int,
    i: int,
    *,
    fee: float,
    slip: float,
    device: torch.device,
    guard_component: str,
    max_horizon: int,
) -> dict[str, Any]:
    side = int(comp["dec"]["side"].iloc[i])
    entry_i = min(i + 1, n - 1)
    entry_price = float(arrays["open"][entry_i]) * (1 + slip if side > 0 else 1 - slip)
    take_profit = float(comp["dec"]["take_profit"].iloc[i])
    stop_loss = float(comp["dec"]["stop_loss"].iloc[i])
    row_margin, row_leverage = float(comp["margin"][i]), float(comp["leverage"][i])
    scale = greedy.SCALE_MAP.get(f"{name}_{'L' if side > 0 else 'S'}", 1.0)
    row_leverage = min(row_leverage * scale, greedy.LEVERAGE_CAP)
    row_notional = min(row_margin * row_leverage, greedy.NOTIONAL_CAP)
    row_leverage = row_notional / max(row_margin, 1e-12)
    notional = row_notional
    mfe = mae = 0.0
    end = min(entry_i + int(max_horizon), n - 2)
    mask = comp.get("sustained_uptrend_mask")
    exit_j, exit_reason = end, "horizon_cap"
    for j in range(entry_i, end + 1):
        px = float(arrays["close"][j])
        mark = px * (1 - slip) if side > 0 else px * (1 + slip)
        move = (mark - entry_price) / entry_price if side > 0 else (entry_price - mark) / entry_price
        mfe, mae = max(mfe, move), min(mae, move)
        reason = ""
        if take_profit > 0.0 and move >= take_profit:
            reason = "take_profit"
        elif stop_loss > 0.0 and move <= -abs(stop_loss):
            reason = "stop_loss"
        if not reason:
            hold = max(j - entry_i, 0)
            giveback = (mfe - move) / max(abs(mfe), 1e-8) if mfe > 0.0 else 0.0
            giveback_clipped = float(np.clip(giveback, 0.0, 10.0))
            expert = hard.EXPERT_NAMES[int(comp["route"][j])]
            pos_values = [float(side), float(hold), float(move), float(mfe), float(mae),
                          giveback_clipped, float(take_profit - move),
                          float(move + abs(stop_loss)), float(notional), float(row_leverage),
                          float(notional * row_leverage), float(take_profit), float(stop_loss)]
            use_guard = name == guard_component and mask is not None and bool(mask[j])
            if use_guard:
                prob = rs._predict_exit_prob_one(comp["guard_base_np"], comp["guard_exit_runtime"], comp["guard_pos_idx"], row_i=j, expert=expert, pos_values=pos_values, device=device)
                active_threshold = float(comp.get("guard_exit_threshold", comp["exit_threshold"]))
            else:
                prob = rs._predict_exit_prob_one(comp["base_np"], comp["exit_runtime"], comp["pos_idx"], row_i=j, expert=expert, pos_values=pos_values, device=device)
                active_threshold = float(comp["exit_threshold"])
            if prob >= active_threshold:
                reason = "exit_head"
        if reason:
            exit_j, exit_reason = j, reason
            break
    exit_px = float(arrays["close"][exit_j]) * (1 - slip if side > 0 else 1 + slip)
    raw = (exit_px - entry_price) / entry_price if side > 0 else (entry_price - exit_px) / entry_price
    net = raw - 2.0 * float(fee)
    return {
        "full": float(net), "adverse": float(abs(min(mae, 0.0))), "favorable": float(max(mfe, 0.0)),
        "exit_reason": exit_reason, "hold_bars": int(exit_j - entry_i), "capped": bool(exit_reason == "horizon_cap"),
    }


def _feature_row(comp: dict[str, Any], i: int) -> np.ndarray:
    row = comp["base_np"][i].astype(np.float64).copy()
    static_idx = [k for k in range(row.shape[0]) if k not in set(comp["pos_idx"])]
    return row[static_idx]


def _build_dataset(wname: str, components: dict[str, Any], aligned_frame: pd.DataFrame, *, fee: float, slip: float, device: torch.device) -> dict[str, pd.DataFrame]:
    arrays = {c: pd.to_numeric(aligned_frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    n = len(aligned_frame)
    out: dict[str, pd.DataFrame] = {}
    for name in ("h48qual", "zig075"):
        comp = components[name]
        side = pd.to_numeric(comp["dec"]["side"], errors="raise").to_numpy()
        active = omega._active(comp["dec"])
        active = active.to_numpy() if hasattr(active, "to_numpy") else np.asarray(active)
        starts = _episode_starts(side, active)
        idxs = np.where(starts)[0]
        idxs = idxs[idxs < n - 2]
        t0 = time.time()
        rows = []
        for k, i in enumerate(idxs):
            sim = _simulate_episode(name, comp, arrays, n, int(i), fee=fee, slip=slip, device=device, guard_component=GUARD_COMPONENT, max_horizon=MAX_HORIZON_BARS)
            feat = _feature_row(comp, int(i))
            row = {"window": wname, "component": name, "entry_signal_i": int(i), "side": int(side[i]), **sim}
            for fi, fv in enumerate(feat):
                row[f"f{fi}"] = float(fv)
            rows.append(row)
            if (k + 1) % 200 == 0:
                log(f"  {wname} {name}: {k+1}/{len(idxs)} episodes simulated ({time.time()-t0:.1f}s elapsed)")
        df = pd.DataFrame(rows)
        log(f"  {wname} {name}: DONE {len(idxs)} episodes in {time.time()-t0:.1f}s")
        out[name] = df
    return out


def _adjacent_correlation(df: pd.DataFrame) -> dict[str, Any]:
    if len(df) < 3:
        return {"n": int(len(df)), "lag1_autocorr_full": None, "frac_within_12bar": None, "frac_within_48bar": None}
    d = df.sort_values("entry_signal_i").reset_index(drop=True)
    full = d["full"].to_numpy(dtype=np.float64)
    lag1 = float(np.corrcoef(full[:-1], full[1:])[0, 1]) if len(full) > 2 else None
    gaps = np.diff(d["entry_signal_i"].to_numpy(dtype=np.float64))
    return {
        "n": int(len(d)),
        "lag1_autocorr_full": lag1,
        "frac_within_12bar": float(np.mean(gaps <= 12)) if len(gaps) else None,
        "frac_within_48bar": float(np.mean(gaps <= 48)) if len(gaps) else None,
        "median_gap_bars": float(np.median(gaps)) if len(gaps) else None,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = DEVICE
    fee, slip = omega._load_fee_slip()
    score_by_base, robustness_thresholds, threshold = guard.build_detector()
    windows = gate.load_all_windows()

    report: dict[str, Any] = {
        "design": "ETH conformal veto episode-start counterfactual label generation, train pool 2025q1-q3 + val calibration only, OOS never loaded.",
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "max_horizon_bars": MAX_HORIZON_BARS,
        "datasets": {},
        "diagnostics": {},
    }

    for wname in (*TRAIN_WINDOWS, CALIBRATION_WINDOW):
        log(f"=== stage=build_dataset window={wname} ===")
        aligned_frame, components, prep_diag = guard.prepare_regime_aware_components(wname, windows, score_by_base, threshold, OUT_DIR, device)
        datasets = _build_dataset(wname, components, aligned_frame, fee=fee, slip=slip, device=device)
        for name, df in datasets.items():
            out_path = OUT_DIR / f"episode_labels_{wname}_{name}.parquet"
            df.to_parquet(out_path, index=False)
            diag = _adjacent_correlation(df)
            capped_frac = float(df["capped"].mean()) if len(df) else None
            full_stats = {
                "n": int(len(df)),
                "full_positive_rate": float((df["full"] > 0).mean()) if len(df) else None,
                "full_mean": float(df["full"].mean()) if len(df) else None,
                "full_std": float(df["full"].std()) if len(df) else None,
                "adverse_mean": float(df["adverse"].mean()) if len(df) else None,
                "capped_fraction": capped_frac,
                "exit_reason_counts": df["exit_reason"].value_counts().to_dict() if len(df) else {},
            }
            report["datasets"][f"{wname}_{name}"] = {"path": str(out_path), "stats": full_stats}
            report["diagnostics"][f"{wname}_{name}"] = diag
            log(f"  {wname} {name}: n={full_stats['n']} full_mean={full_stats['full_mean']} "
                f"pos_rate={full_stats['full_positive_rate']} capped={capped_frac} "
                f"lag1_autocorr={diag['lag1_autocorr_full']} frac_within_12bar={diag['frac_within_12bar']}")

    report["stage_reached"] = "done"
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=omega._json_default) + "\n", encoding="utf-8")
    log(f"report={OUT_DIR / 'report.json'}")
    log("stage=done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
