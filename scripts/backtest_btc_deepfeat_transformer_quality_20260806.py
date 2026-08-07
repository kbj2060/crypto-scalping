"""Quality-filtered version of backtest_btc_deepfeat_transformer_20260806.py.

That first backtest (fresh-entry direction gate only) was deeply negative on both VAL and OOS
(win rate ~30%, MDD ~-70 to -90%, right at/below the TP/SL ratio's breakeven win rate) --
bar-level direction classification accuracy (67%/64.8%) does not by itself imply the entry is a
good risk/reward point. User asked to try filtering by the model's own predicted trade quality
(log1p(zigzag_path_calmar), added as a second regression head -- see
`--quality-head`/`--quality-loss-weight` in scripts/train_btc_deepfeat_encoders_20260806.py,
retrained checkpoint at tmp/btc_deepfeat_sharpen_sweep/cw_0.2_quality/deepfeat_bundle.pt).

Same TP/SL, horizon, margin/leverage, and fresh-entry direction gate as the prior backtest
(direction quantile pair fixed at the prior winner, 0.90/0.10, refit on this checkpoint's own VAL
score distribution). New axis: grid a quality-percentile floor (computed from the VAL fresh-entry
candidate pool's own predicted-quality distribution) on top of the direction gate. VAL-selects the
best quality floor (with a MIN_TRADES floor), OOS confirmed exactly once.
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

CHECKPOINT = ROOT / "tmp/btc_deepfeat_sharpen_sweep/cw_0.2_quality/deepfeat_bundle.pt"
PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
OUT_DIR = ROOT / "tmp/btc_deepfeat_transformer_backtest_quality_20260806"

TRAIL_VOL_BARS = 288
TP_MULT = 2.5
SL_MULT = 1.2
HORIZON_BARS = 288
MARGIN_FRACTION = 0.30
LEVERAGE = 3.0
ROUNDTRIP_COST_RATE = 0.0010

DIRECTION_UPPER_Q = 0.90
DIRECTION_LOWER_Q = 0.10
QUALITY_PCTL_GRID = [0.0, 0.25, 0.50, 0.70, 0.85]
MIN_TRADES = 15


@torch.no_grad()
def _predict(model, ds, split: str, device: torch.device, batch_size: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    row_idx = ds.end_idx[split]
    probs_out, quality_out = [], []
    for i in range(0, len(row_idx), batch_size):
        chunk = row_idx[i : i + batch_size]
        x = torch.from_numpy(ds.get_batch(chunk)).to(device)
        logits, quality_pred, _ = model(x)
        probs_out.append(torch.softmax(logits, dim=-1).cpu().numpy())
        quality_out.append(quality_pred.cpu().numpy())
    probs = np.concatenate(probs_out, axis=0) if probs_out else np.zeros((0, 3), dtype=np.float32)
    quality = np.concatenate(quality_out, axis=0) if quality_out else np.zeros((0,), dtype=np.float32)
    return probs, quality


def _trailing_vol(close: np.ndarray) -> np.ndarray:
    log_ret = np.diff(np.log(np.clip(close, 1e-9, None)), prepend=np.log(close[0]))
    return pd.Series(log_ret).rolling(TRAIL_VOL_BARS, min_periods=TRAIL_VOL_BARS).std().to_numpy()


def _fresh_entry_mask(score: np.ndarray, upper: float, lower: float) -> np.ndarray:
    side_state = np.where(score >= upper, 1, np.where(score <= lower, -1, 0))
    fresh = np.zeros(len(side_state), dtype=bool)
    fresh[0] = side_state[0] != 0
    fresh[1:] = (side_state[1:] != 0) & (side_state[1:] != side_state[:-1])
    return fresh


def _run(row_idx, score, quality, vol, panel, upper, lower, quality_threshold):
    fresh = _fresh_entry_mask(score, upper, lower)
    row_idx, score, quality = row_idx[fresh], score[fresh], quality[fresh]
    keep = quality >= quality_threshold
    row_idx, score = row_idx[keep], score[keep]
    tp_moves = TP_MULT * vol[row_idx]
    sl_moves = SL_MULT * vol[row_idx]
    finite = np.isfinite(tp_moves) & np.isfinite(sl_moves) & np.isfinite(score)
    idx, sc, tp, sl = row_idx[finite], score[finite], tp_moves[finite], sl_moves[finite]
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


def _summarize(result, split: str, quality_pctl: float, quality_threshold: float) -> dict:
    ledger = result.ledger
    n = len(ledger)
    if n == 0:
        return {"split": split, "quality_pctl": quality_pctl, "n_trades": 0, "sum_ret_pct": 0.0}
    equity = result.equity
    running_max = np.maximum.accumulate(equity)
    mdd_pct = float(((equity - running_max) / running_max).min() * 100.0)
    return {
        "split": split,
        "quality_pctl": quality_pctl,
        "quality_threshold": quality_threshold,
        "n_trades": n,
        "sum_ret_pct": float(ledger["trade_return"].sum() * 100.0),
        "mean_ret_pct": float(ledger["trade_return"].mean() * 100.0),
        "win_rate": float((ledger["trade_return"] > 0).mean()),
        "mdd_pct": mdd_pct,
        "final_equity": float(equity[-1]),
        "exit_reasons": ledger["reason"].value_counts().to_dict(),
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
        ffn_mult=config["ffn_mult"], dropout=config["dropout"], quality_head=config["quality_head"],
    ).to(device)
    model.load_state_dict(bundle["model_state"])

    panel = pd.read_parquet(PANEL_PATH, columns=["timestamp", "open", "high", "low", "close"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    vol = _trailing_vol(panel["close"].to_numpy(dtype=np.float64))

    probs_val, quality_val = _predict(model, ds, "val", device)
    probs_oos, quality_oos = _predict(model, ds, "oos", device)
    score_val = probs_val[:, 1] - probs_val[:, 2]
    score_oos = probs_oos[:, 1] - probs_oos[:, 2]
    row_val, row_oos = ds.end_idx["val"], ds.end_idx["oos"]

    th = fit_tail_thresholds(score_val, upper_quantile=DIRECTION_UPPER_Q, lower_quantile=DIRECTION_LOWER_Q)
    print(f"direction thresholds (refit on this checkpoint's VAL scores): upper={th.upper:.4f} lower={th.lower:.4f}")

    fresh_val = _fresh_entry_mask(score_val, th.upper, th.lower)
    candidate_quality_val = quality_val[fresh_val]
    print(f"VAL fresh-entry candidate pool: n={len(candidate_quality_val)}, quality quantiles={np.quantile(candidate_quality_val, [0, .25, .5, .7, .85, 1]).round(3).tolist()}")

    grid_results = []
    for pctl in QUALITY_PCTL_GRID:
        q_thresh = float(np.quantile(candidate_quality_val, pctl)) if len(candidate_quality_val) else 0.0
        result_val = _run(row_val, score_val, quality_val, vol, panel, th.upper, th.lower, q_thresh)
        summary = _summarize(result_val, "val", pctl, q_thresh)
        grid_results.append(summary)
        print(json.dumps(summary))

    eligible = [r for r in grid_results if r["n_trades"] >= MIN_TRADES]
    if not eligible:
        raise RuntimeError(f"no grid config reached MIN_TRADES={MIN_TRADES} on VAL")
    best = max(eligible, key=lambda r: r["sum_ret_pct"])
    print("BEST VAL CONFIG:", json.dumps(best))

    result_oos = _run(row_oos, score_oos, quality_oos, vol, panel, th.upper, th.lower, best["quality_threshold"])
    oos_summary = _summarize(result_oos, "oos", best["quality_pctl"], best["quality_threshold"])
    print("OOS CONFIRM (single run, VAL-selected quality floor):", json.dumps(oos_summary))

    out = {
        "checkpoint": str(CHECKPOINT),
        "config": config,
        "direction_upper_q": DIRECTION_UPPER_Q,
        "direction_lower_q": DIRECTION_LOWER_Q,
        "direction_upper_threshold": th.upper,
        "direction_lower_threshold": th.lower,
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
