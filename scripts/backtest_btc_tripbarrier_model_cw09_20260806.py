"""Backtest for the transformer trained on the causal triple-barrier trade-outcome label
(scripts/build_btc_5m_tripbarrier_tradeoutcome_labels_20260806.py) -- the fundamental fix for
root causes 1 (retrospective label != forward prediction) and 3 (SL sized off single-bar noise)
from project-btc-deepfeat-acc-pnl-gap-diagnosis-20260806.

Unlike the zigzag-label backtests, entry logic is a direct argmax of the 3-class prediction (the
label IS "would a LONG/SHORT opened now hit TP before SL", so no threshold-quantile fitting is
needed -- CASH means skip, LONG/SHORT means take that side). TP/SL use the EXACT same corrected
vol basis the label was built with (12-bar cumulative-return dispersion, 288-bar lookback,
TP_MULT=2.5/SL_MULT=1.2), so there is no train/live mismatch between what the label promised and
what the backtest simulates.

Compares continuous entry (every bar argmax != CASH) vs fresh-entry gate (only on regime-change
bars), on VAL then OOS.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.causal_futures_backtest import simulate_single_position  # noqa: E402
from ensemble.deep_features.btc_deepfeat_dataset_20260806 import build_dataset  # noqa: E402
from ensemble.deep_features.btc_deepfeat_encoders_20260806 import build_model  # noqa: E402

CHECKPOINT = ROOT / "tmp/btc_deepfeat_tripbarrier_20260806/flatsmooth_cw_0.9/deepfeat_bundle.pt"
PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
LABEL_PATH = ROOT / "data/splits/year_oos/btc_5m_tripbarrier_tradeoutcome_labels_20260806.parquet"
OUT_DIR = ROOT / "tmp/btc_deepfeat_tripbarrier_backtest_cw09_20260806"

CUMRET_BARS = 12
VOL_LOOKBACK = 288
TP_MULT = 2.5
SL_MULT = 1.2
HORIZON_BARS = 288
MARGIN_FRACTION, LEVERAGE, ROUNDTRIP_COST_RATE = 0.30, 3.0, 0.0010


@torch.no_grad()
def _predict(model, ds, split, device, batch_size=1024):
    model.eval()
    row_idx = ds.end_idx[split]
    out = []
    for i in range(0, len(row_idx), batch_size):
        chunk = row_idx[i : i + batch_size]
        x = torch.from_numpy(ds.get_batch(chunk)).to(device)
        logits, _, _ = model(x)
        out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


def _fresh_entry_mask(side_state: np.ndarray) -> np.ndarray:
    fresh = np.zeros(len(side_state), dtype=bool)
    fresh[0] = side_state[0] != 0
    fresh[1:] = (side_state[1:] != 0) & (side_state[1:] != side_state[:-1])
    return fresh


def _run(row_idx, side_state, tp_moves, sl_moves, panel, fresh_only: bool):
    mask = _fresh_entry_mask(side_state) if fresh_only else (side_state != 0)
    idx = row_idx[mask]
    side = side_state[mask]
    tp = tp_moves[idx]
    sl = sl_moves[idx]
    finite = np.isfinite(tp) & np.isfinite(sl)
    idx, side, tp, sl = idx[finite], side[finite], tp[finite], sl[finite]
    scores = side.astype(np.float64)  # +1 long, -1 short
    return simulate_single_position(
        timestamps=panel["timestamp"], open_px=panel["open"].to_numpy(dtype=np.float64),
        high=panel["high"].to_numpy(dtype=np.float64), low=panel["low"].to_numpy(dtype=np.float64),
        close=panel["close"].to_numpy(dtype=np.float64), decision_indices=idx, scores=scores,
        tp_moves=tp, sl_moves=sl, upper_threshold=0.0, lower_threshold=0.0, horizon_bars=HORIZON_BARS,
        margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE, roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
    )


def _summarize(result, split, mode) -> dict:
    ledger = result.ledger
    n = len(ledger)
    if n == 0:
        return {"split": split, "mode": mode, "n_trades": 0}
    equity = result.equity
    running_max = np.maximum.accumulate(equity)
    mdd_pct = float(((equity - running_max) / running_max).min() * 100.0)
    return {
        "split": split, "mode": mode, "n_trades": n,
        "sum_ret_pct": float(ledger["trade_return"].sum() * 100.0),
        "mean_ret_pct": float(ledger["trade_return"].mean() * 100.0),
        "win_rate": float((ledger["trade_return"] > 0).mean()),
        "mdd_pct": mdd_pct, "final_equity": float(equity[-1]),
        "exit_reasons": ledger["reason"].value_counts().to_dict(),
        "long_trades": int((ledger["side"] == 1).sum()), "short_trades": int((ledger["side"] == -1).sum()),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    config = bundle["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = build_dataset(
        window=config["window"], label_path=LABEL_PATH, hard_col="trade_outcome_action",
        soft_cols=["trade_outcome_soft_cash", "trade_outcome_soft_long", "trade_outcome_soft_short"],
    )
    model = build_model(
        config["arch"], config["n_features"], config["category_sizes"], embed_dim=config["embed_dim"],
        d_model=config["d_model"], n_heads=config["n_heads"], n_layers=config["n_layers"],
        ffn_mult=config["ffn_mult"], dropout=config["dropout"], quality_head=config["quality_head"],
        head_type=config.get("head_type", "linear"),
    ).to(device)
    model.load_state_dict(bundle["model_state"])

    panel = pd.read_parquet(PANEL_PATH, columns=["timestamp", "open", "high", "low", "close"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    close = panel["close"].to_numpy(dtype=np.float64)
    log_ret_1bar = np.diff(np.log(np.clip(close, 1e-9, None)), prepend=np.log(close[0]))
    cumret = pd.Series(log_ret_1bar).rolling(CUMRET_BARS).sum().to_numpy()
    vol = pd.Series(cumret).rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).std().to_numpy()
    tp_moves_all, sl_moves_all = TP_MULT * vol, SL_MULT * vol

    results = []
    for split in ("val", "oos"):
        probs = _predict(model, ds, split, device)
        pred_hard = probs.argmax(axis=1)
        side_state = np.where(pred_hard == 1, 1, np.where(pred_hard == 2, -1, 0))
        row_idx = ds.end_idx[split]
        for fresh_only, mode in [(False, "continuous"), (True, "fresh_entry")]:
            r = _run(row_idx, side_state, tp_moves_all, sl_moves_all, panel, fresh_only)
            summary = _summarize(r, split, mode)
            results.append(summary)
            print(json.dumps(summary, default=str))

    (OUT_DIR / "backtest_summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"wrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
