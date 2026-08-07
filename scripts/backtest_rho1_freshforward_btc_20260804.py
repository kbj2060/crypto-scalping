"""Corrected causal Rho1 validation and forward test.

Timeline:
  train       < 2025-09-01
  model val   2025-09-01 .. 2025-12-31 (checkpoint selection)
  calibration 2026-01-01 .. 2026-03-31 (entry/config selection)
  test        2026-04-01 .. 2026-07-31 (one frozen evaluation)

The test period was previously inspected for Stage-1 pinball loss, so this run is valid research
evidence but is not promotion-eligible as a pristine holdout.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from core.backtest_metrics import bar_level_performance  # noqa: E402
from core.causal_futures_backtest import (  # noqa: E402
    fit_tail_thresholds,
    purged_decision_mask,
    simulate_single_position,
)
import train_rho1_panel_backbone_20260804 as base  # noqa: E402
from eval_rho1_btc_oos_20260804 import predict_model  # noqa: E402

QUANTILE_CKPT = base.CKPT_DIR / "rho1_panel_backbone_best.pt"
RANK_CKPT = base.CKPT_DIR / "rho1_ranking_head_best.pt"
OUT_DIR = ROOT / "tmp/rho1_corrected_causal_20260804"

CAL_START, CAL_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-04-01")
TEST_START, TEST_END = pd.Timestamp("2026-04-01"), pd.Timestamp("2026-08-01")

TB_MIN_TP, TB_MIN_SL = 0.006, 0.004
MARGIN_FRACTION = 0.30
LEVERAGE = 3.0
ROUNDTRIP_COST_RATE = (0.0005 + 0.0002) * 2.0
TPSL_QUANTILE_CONFIGS = [("moderate", 0.75, 0.25), ("wide", 0.90, 0.10)]
TAIL_CONFIGS = [(0.80, 0.20), (0.90, 0.10), (0.95, 0.05)]


def _load_btc() -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_parquet(base.FEATURES_DIR / "BTCUSDT.parquet").sort_values("timestamp").reset_index(drop=True)
    features = df[base.FEATURE_COLS].to_numpy(dtype=np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return df, np.clip(features, -20.0, 20.0)


def _decision_indices(ts: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    mask = purged_decision_mask(ts, start=start, end=end, horizon_bars=base.HORIZON_H)
    idxs = np.arange(base.WINDOW_L - 1, len(ts) - base.HORIZON_H)
    return idxs[mask[idxs]]


def _score(
    features: np.ndarray, decision_indices: np.ndarray, *, rank_sym_id: int, quantile_sym_id: int
) -> tuple[np.ndarray, np.ndarray]:
    windows = np.stack(
        [features[i - base.WINDOW_L + 1 : i + 1] for i in decision_indices]
    )
    rank_raw, _ = predict_model(RANK_CKPT, windows, sym_id=rank_sym_id, n_quantiles=1)
    rank_score = 1.0 / (1.0 + np.exp(-rank_raw.squeeze(-1)))
    quantile_norm, _ = predict_model(QUANTILE_CKPT, windows, sym_id=quantile_sym_id)
    return rank_score, quantile_norm


def _moves(
    quantile_norm: np.ndarray,
    realized_vol: np.ndarray,
    decision_indices: np.ndarray,
    *,
    tp_quantile: float,
    sl_quantile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scale = realized_vol[decision_indices] * math.sqrt(base.HORIZON_H)
    raw = np.sort(quantile_norm * scale[:, None], axis=1)
    q_index = {q: i for i, q in enumerate(base.QUANTILES)}
    long_tp = np.maximum(TB_MIN_TP, raw[:, q_index[tp_quantile]])
    long_sl = np.maximum(TB_MIN_SL, -raw[:, q_index[sl_quantile]])
    short_tp = np.maximum(TB_MIN_TP, -raw[:, q_index[sl_quantile]])
    short_sl = np.maximum(TB_MIN_SL, raw[:, q_index[tp_quantile]])
    return long_tp, long_sl, short_tp, short_sl


def _evaluate(
    *,
    df: pd.DataFrame,
    decision_indices: np.ndarray,
    scores: np.ndarray,
    directional_moves: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    upper_threshold: float,
    lower_threshold: float,
) -> tuple[dict, pd.DataFrame]:
    long_tp, long_sl, short_tp, short_sl = directional_moves
    take_long = scores >= upper_threshold
    tp_moves = np.where(take_long, long_tp, short_tp)
    sl_moves = np.where(take_long, long_sl, short_sl)
    result = simulate_single_position(
        timestamps=df["timestamp"],
        open_px=df["open"].to_numpy(),
        high=df["high"].to_numpy(),
        low=df["low"].to_numpy(),
        close=df["close"].to_numpy(),
        decision_indices=decision_indices,
        scores=scores,
        tp_moves=tp_moves,
        sl_moves=sl_moves,
        upper_threshold=upper_threshold,
        lower_threshold=lower_threshold,
        horizon_bars=base.HORIZON_H,
        margin_fraction=MARGIN_FRACTION,
        leverage=LEVERAGE,
        roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
    )
    metrics = bar_level_performance(result.equity, result.ledger)
    metrics["mean_trade_return_pct"] = (
        float(result.ledger["trade_return"].mean() * 100.0) if len(result.ledger) else 0.0
    )
    metrics["skipped_while_open"] = result.skipped_while_open
    return metrics, result.ledger


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, features = _load_btc()
    cal_idxs = _decision_indices(df["timestamp"], CAL_START, CAL_END)
    test_idxs = _decision_indices(df["timestamp"], TEST_START, TEST_END)

    rank_ckpt = torch.load(RANK_CKPT, map_location="cpu", weights_only=False)
    quantile_ckpt = torch.load(QUANTILE_CKPT, map_location="cpu", weights_only=False)
    cal_scores, cal_quantiles = _score(
        features,
        cal_idxs,
        rank_sym_id=rank_ckpt["symbol_to_id"]["BTCUSDT"],
        quantile_sym_id=quantile_ckpt["symbol_to_id"]["BTCUSDT"],
    )
    test_scores, test_quantiles = _score(
        features,
        test_idxs,
        rank_sym_id=rank_ckpt["symbol_to_id"]["BTCUSDT"],
        quantile_sym_id=quantile_ckpt["symbol_to_id"]["BTCUSDT"],
    )
    realized_vol = df["realized_vol_288"].to_numpy(dtype=np.float64)

    calibration_rows = []
    candidates = []
    for tpsl_name, tp_q, sl_q in TPSL_QUANTILE_CONFIGS:
        cal_moves = _moves(
            cal_quantiles, realized_vol, cal_idxs, tp_quantile=tp_q, sl_quantile=sl_q
        )
        for upper_q, lower_q in TAIL_CONFIGS:
            thresholds = fit_tail_thresholds(
                cal_scores, upper_quantile=upper_q, lower_quantile=lower_q
            )
            metrics, ledger = _evaluate(
                df=df,
                decision_indices=cal_idxs,
                scores=cal_scores,
                directional_moves=cal_moves,
                upper_threshold=thresholds.upper,
                lower_threshold=thresholds.lower,
            )
            row = {
                "tpsl": tpsl_name,
                "tp_quantile": tp_q,
                "sl_quantile": sl_q,
                "upper_quantile": upper_q,
                "lower_quantile": lower_q,
                "upper_threshold": thresholds.upper,
                "lower_threshold": thresholds.lower,
                **metrics,
            }
            calibration_rows.append(row)
            candidates.append((metrics["pnl"], row, ledger))

    calibration = pd.DataFrame(calibration_rows).sort_values("pnl", ascending=False)
    calibration.to_csv(OUT_DIR / "calibration_candidates.csv", index=False)
    _, selected, selected_cal_ledger = max(candidates, key=lambda item: item[0])
    selected_cal_ledger.to_csv(OUT_DIR / "selected_calibration_ledger.csv", index=False)

    test_moves = _moves(
        test_quantiles,
        realized_vol,
        test_idxs,
        tp_quantile=selected["tp_quantile"],
        sl_quantile=selected["sl_quantile"],
    )
    test_metrics, test_ledger = _evaluate(
        df=df,
        decision_indices=test_idxs,
        scores=test_scores,
        directional_moves=test_moves,
        upper_threshold=selected["upper_threshold"],
        lower_threshold=selected["lower_threshold"],
    )
    test_ledger.to_csv(OUT_DIR / "test_ledger.csv", index=False)

    report = {
        "architecture": "rho1_pooled_shared_encoder_corrected",
        "selected_on": "calibration_2026-01-01_2026-03-31",
        "test_period": "2026-04-01_2026-07-31",
        "selected_config": selected,
        "test_metrics": test_metrics,
        "contracts": {
            "fresh_forward_bar_by_bar": True,
            "thresholds_fit_on_calibration_only": True,
            "trade_ledgers_used_as_input": False,
            "saved_parent_exit_timestamps_used": False,
            "future_rows_used_for_entry": False,
            "single_position": True,
            "bar_level_mark_to_market": True,
            "split_targets_purged": True,
            "margin_fraction": MARGIN_FRACTION,
            "leverage": LEVERAGE,
            "notional": MARGIN_FRACTION * LEVERAGE,
            "roundtrip_cost_rate": ROUNDTRIP_COST_RATE,
        },
        "promotion_eligible": False,
        "promotion_blocker": "test period was previously inspected for Stage-1 pinball loss",
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(calibration.to_string(index=False))
    print("\nSELECTED CALIBRATION CONFIG")
    print(json.dumps(selected, indent=2, default=str))
    print("\nFROZEN TEST")
    print(json.dumps(test_metrics, indent=2))
    print(f"\nwrote {OUT_DIR / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
