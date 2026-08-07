"""First trading backtest on top of the tuned BTC deep-feature transformer
(docs/btc_deepfeat_cnn_transformer_zigzag_soft_label_20260806.md, final config:
label_sharpen=0.7, cash_weight=0.2, window=48, d_model=96, n_layers=3, dropout=0.25 --
checkpoint at tmp/btc_deepfeat_sharpen_sweep/cw_0.2/deepfeat_bundle.pt).

Reuses this repo's canonical futures backtest primitives (core/causal_futures_backtest.py):
`fit_tail_thresholds` (VAL-only quantile threshold fitting) + `simulate_single_position`
(chronological, non-overlapping, mark-to-market simulation with margin/leverage/notional applied
per the CLAUDE.md Futures Risk Sizing Contract).

Score = P(LONG) - P(SHORT) in [-1, 1] (CASH implicitly means the score sits between the fitted
thresholds -- no trade). TP/SL are volatility-adaptive: TP_MULT/SL_MULT x trailing 24h (288-bar)
realized volatility of 5m log returns, matching this session's established BTC 5m Layer B
convention (scripts/backtest_btc_5m_layerA_qualityweightedB_combined_20260806.py).

Fresh-Forward discipline: entry-threshold quantile pair is grid-searched on VAL only (with a
MIN_TRADES floor so thin samples can't win), the single VAL-winning config is then evaluated on
OOS exactly once -- no further tuning after seeing OOS.
fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false.
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
from core.causal_futures_backtest import fit_tail_thresholds, simulate_single_position  # noqa: E402
from ensemble.deep_features.btc_deepfeat_dataset_20260806 import build_dataset  # noqa: E402
from ensemble.deep_features.btc_deepfeat_encoders_20260806 import build_model  # noqa: E402

CHECKPOINT = ROOT / "tmp/btc_deepfeat_sharpen_sweep/cw_0.2/deepfeat_bundle.pt"
PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
OUT_DIR = ROOT / "tmp/btc_deepfeat_transformer_backtest_20260806"

TRAIL_VOL_BARS = 288  # 24h of 5m bars
TP_MULT = 2.5
SL_MULT = 1.2
HORIZON_BARS = 288  # 24h max hold
MARGIN_FRACTION = 0.30
LEVERAGE = 3.0
ROUNDTRIP_COST_RATE = 0.0010  # 10bps

QUANTILE_GRID = [(0.60, 0.40), (0.70, 0.30), (0.80, 0.20), (0.90, 0.10)]
MIN_TRADES = 15


@torch.no_grad()
def _predict_probs(model, ds, split: str, device: torch.device, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    row_idx = ds.end_idx[split]
    out = []
    for i in range(0, len(row_idx), batch_size):
        chunk = row_idx[i : i + batch_size]
        x = torch.from_numpy(ds.get_batch(chunk)).to(device)
        logits, _ = model(x)
        out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 3), dtype=np.float32)


def _trailing_vol(close: np.ndarray) -> np.ndarray:
    log_ret = np.diff(np.log(np.clip(close, 1e-9, None)), prepend=np.log(close[0]))
    vol = pd.Series(log_ret).rolling(TRAIL_VOL_BARS, min_periods=TRAIL_VOL_BARS).std().to_numpy()
    return vol


def _fresh_entry_mask(score: np.ndarray, upper: float, lower: float) -> np.ndarray:
    """Only fire a decision on the first bar of a new directional regime (score crosses INTO the
    entry zone), not on every bar the score happens to sit past threshold. Without this, a single
    zigzag wave generates a new decision on nearly every one of its bars -- most of them deep into
    an already-progressed move with poor remaining risk/reward, which is what produced the
    catastrophic (win rate ~30%, MDD ~-90%) first-pass backtest result. This mirrors the prior
    session's pivot-transition gate idea (state-change detection instead of level detection)."""
    side_state = np.where(score >= upper, 1, np.where(score <= lower, -1, 0))
    fresh = np.zeros(len(side_state), dtype=bool)
    fresh[0] = side_state[0] != 0
    fresh[1:] = (side_state[1:] != 0) & (side_state[1:] != side_state[:-1])
    return fresh


def _run(row_idx: np.ndarray, score: np.ndarray, vol: np.ndarray, panel: pd.DataFrame, upper: float, lower: float):
    fresh = _fresh_entry_mask(score, upper, lower)
    row_idx, score = row_idx[fresh], score[fresh]
    tp_moves = TP_MULT * vol[row_idx]
    sl_moves = SL_MULT * vol[row_idx]
    finite = np.isfinite(tp_moves) & np.isfinite(sl_moves) & np.isfinite(score)
    idx = row_idx[finite]
    sc = score[finite]
    tp = tp_moves[finite]
    sl = sl_moves[finite]
    return simulate_single_position(
        timestamps=panel["timestamp"],
        open_px=panel["open"].to_numpy(dtype=np.float64),
        high=panel["high"].to_numpy(dtype=np.float64),
        low=panel["low"].to_numpy(dtype=np.float64),
        close=panel["close"].to_numpy(dtype=np.float64),
        decision_indices=idx,
        scores=sc,
        tp_moves=tp,
        sl_moves=sl,
        upper_threshold=upper,
        lower_threshold=lower,
        horizon_bars=HORIZON_BARS,
        margin_fraction=MARGIN_FRACTION,
        leverage=LEVERAGE,
        roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
    )


def _summarize(result, split: str, upper: float, lower: float) -> dict:
    ledger = result.ledger
    n = len(ledger)
    if n == 0:
        return {"split": split, "upper": upper, "lower": lower, "n_trades": 0, "sum_ret_pct": 0.0}
    equity = result.equity
    running_max = np.maximum.accumulate(equity)
    mdd_pct = float(((equity - running_max) / running_max).min() * 100.0)
    return {
        "split": split,
        "upper": upper,
        "lower": lower,
        "n_trades": n,
        "sum_ret_pct": float(ledger["trade_return"].sum() * 100.0),
        "mean_ret_pct": float(ledger["trade_return"].mean() * 100.0),
        "win_rate": float((ledger["trade_return"] > 0).mean()),
        "mdd_pct": mdd_pct,
        "final_equity": float(equity[-1]),
        "exit_reasons": ledger["reason"].value_counts().to_dict(),
        "skipped_while_open": int(result.skipped_while_open),
        "long_trades": int((ledger["side"] == 1).sum()),
        "short_trades": int((ledger["side"] == -1).sum()),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    config = bundle["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = build_dataset(window=config["window"])
    model = build_model(
        config["arch"], config["n_features"], config["category_sizes"], embed_dim=config["embed_dim"],
        d_model=config["d_model"], n_heads=config["n_heads"], n_layers=config["n_layers"],
        ffn_mult=config["ffn_mult"], dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(bundle["model_state"])

    panel = pd.read_parquet(PANEL_PATH, columns=["timestamp", "open", "high", "low", "close"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    vol = _trailing_vol(panel["close"].to_numpy(dtype=np.float64))

    probs_val = _predict_probs(model, ds, "val", device)
    probs_oos = _predict_probs(model, ds, "oos", device)
    score_val = probs_val[:, 1] - probs_val[:, 2]  # P(LONG) - P(SHORT)
    score_oos = probs_oos[:, 1] - probs_oos[:, 2]
    row_val = ds.end_idx["val"]
    row_oos = ds.end_idx["oos"]

    grid_results = []
    for upper_q, lower_q in QUANTILE_GRID:
        th = fit_tail_thresholds(score_val, upper_quantile=upper_q, lower_quantile=lower_q)
        result_val = _run(row_val, score_val, vol, panel, th.upper, th.lower)
        summary = _summarize(result_val, "val", th.upper, th.lower)
        summary["upper_q"] = upper_q
        summary["lower_q"] = lower_q
        grid_results.append(summary)
        print(json.dumps(summary))

    eligible = [r for r in grid_results if r["n_trades"] >= MIN_TRADES]
    if not eligible:
        raise RuntimeError(f"no grid config reached MIN_TRADES={MIN_TRADES} on VAL")
    best = max(eligible, key=lambda r: r["sum_ret_pct"])
    print("BEST VAL CONFIG:", json.dumps(best))

    result_oos = _run(row_oos, score_oos, vol, panel, best["upper"], best["lower"])
    oos_summary = _summarize(result_oos, "oos", best["upper"], best["lower"])
    oos_summary["upper_q"] = best["upper_q"]
    oos_summary["lower_q"] = best["lower_q"]
    print("OOS CONFIRM (single run, VAL-selected thresholds):", json.dumps(oos_summary))

    out = {
        "checkpoint": str(CHECKPOINT),
        "config": config,
        "tp_mult": TP_MULT,
        "sl_mult": SL_MULT,
        "horizon_bars": HORIZON_BARS,
        "margin_fraction": MARGIN_FRACTION,
        "leverage": LEVERAGE,
        "roundtrip_cost_rate": ROUNDTRIP_COST_RATE,
        "grid_results_val": grid_results,
        "best_val_config": best,
        "oos_confirm": oos_summary,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
    }
    (OUT_DIR / "backtest_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    result_oos.ledger.to_csv(OUT_DIR / "oos_ledger.csv", index=False)
    print(f"wrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
