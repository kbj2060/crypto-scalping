"""Deep-dive: why did classification accuracy (67%/64.8%) improve across every tuning round while
the backtest PnL stayed negative (win rate ~30% vs ~32.4% breakeven) across all three entry-filter
attempts? Runs several diagnostics against the tuned quality-head checkpoint
(tmp/btc_deepfeat_sharpen_sweep/cw_0.2_quality/) and its OOS ledger.
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
LABEL_PATH = ROOT / "data/splits/year_oos/btc_5m_zigzag_labels_20260806.parquet"
OUT_DIR = ROOT / "tmp/btc_deepfeat_acc_pnl_gap_analysis_20260806"

DIRECTION_UPPER_Q, DIRECTION_LOWER_Q = 0.90, 0.10
TRAIL_VOL_BARS, TP_MULT, SL_MULT = 288, 2.5, 1.2
HORIZON_BARS = 288
MARGIN_FRACTION, LEVERAGE, ROUNDTRIP_COST_RATE = 0.30, 3.0, 0.0010
FORWARD_HORIZONS = [12, 48, 144, 288]  # 1h, 4h, 12h, 24h


@torch.no_grad()
def _predict(model, ds, split, device, batch_size=1024):
    model.eval()
    row_idx = ds.end_idx[split]
    probs_out, quality_out = [], []
    for i in range(0, len(row_idx), batch_size):
        chunk = row_idx[i : i + batch_size]
        x = torch.from_numpy(ds.get_batch(chunk)).to(device)
        logits, quality_pred, _ = model(x)
        probs_out.append(torch.softmax(logits, dim=-1).cpu().numpy())
        quality_out.append(quality_pred.cpu().numpy())
    return np.concatenate(probs_out, axis=0), np.concatenate(quality_out, axis=0)


def _fresh_entry_mask(score, upper, lower):
    side_state = np.where(score >= upper, 1, np.where(score <= lower, -1, 0))
    fresh = np.zeros(len(side_state), dtype=bool)
    fresh[0] = side_state[0] != 0
    fresh[1:] = (side_state[1:] != 0) & (side_state[1:] != side_state[:-1])
    return fresh, side_state


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    bundle = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    config = bundle["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = build_dataset(window=config["window"])
    model = build_model(
        config["arch"], config["n_features"], config["category_sizes"], embed_dim=config["embed_dim"],
        d_model=config["d_model"], n_heads=config["n_heads"], n_layers=config["n_layers"],
        ffn_mult=config["ffn_mult"], dropout=config["dropout"], quality_head=config["quality_head"],
        head_type=config.get("head_type", "linear"),
    ).to(device)
    model.load_state_dict(bundle["model_state"])

    probs_val, _ = _predict(model, ds, "val", device)
    probs_oos, quality_oos = _predict(model, ds, "oos", device)
    score_val = probs_val[:, 1] - probs_val[:, 2]
    score_oos = probs_oos[:, 1] - probs_oos[:, 2]
    row_val, row_oos = ds.end_idx["val"], ds.end_idx["oos"]

    th = fit_tail_thresholds(score_val, upper_quantile=DIRECTION_UPPER_Q, lower_quantile=DIRECTION_LOWER_Q)

    # --- Diagnostic 1: bar-level accuracy overall vs restricted to fresh-entry decision bars ---
    y_hard_oos = ds.y_hard_all[row_oos]
    pred_hard_oos = probs_oos.argmax(axis=1)
    overall_acc = float((pred_hard_oos == y_hard_oos).mean())

    fresh_mask, side_state_oos = _fresh_entry_mask(score_oos, th.upper, th.lower)
    entry_pred = pred_hard_oos[fresh_mask]
    entry_true = y_hard_oos[fresh_mask]
    entry_acc = float((entry_pred == entry_true).mean())
    # accuracy restricted to LONG/SHORT-only bars (excluding CASH), for apples-to-apples vs entry subset (entries never fire CASH)
    active_mask = y_hard_oos != 0
    active_acc = float((pred_hard_oos[active_mask] == y_hard_oos[active_mask]).mean())

    report["diag1_bar_level_vs_entry_level_accuracy"] = {
        "overall_oos_acc_all_bars": overall_acc,
        "oos_acc_active_bars_only_long_short": active_acc,
        "oos_acc_at_fresh_entry_decision_bars_only": entry_acc,
        "n_fresh_entry_bars": int(fresh_mask.sum()),
        "note": "if entry-bar accuracy << overall accuracy, the model is accurate on 'obviously trending, already-progressed' bars but not at the specific moment a fresh trade would be opened",
    }
    print(json.dumps(report["diag1_bar_level_vs_entry_level_accuracy"], indent=2))

    # --- Diagnostic 2: does the label's hard direction even predict simple forward realized return sign? ---
    panel = pd.read_parquet(PANEL_PATH, columns=["timestamp", "open", "high", "low", "close"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    close = panel["close"].to_numpy(dtype=np.float64)
    n = len(close)

    entry_rows = row_oos[fresh_mask]
    entry_side = side_state_oos[fresh_mask]  # 1=long,-1=short (matches model's predicted direction at decision)
    entry_label_hard = y_hard_oos[fresh_mask]  # 0/1/2 from the zigzag wave label at that bar

    horizon_hitrates = {}
    for h in FORWARD_HORIZONS:
        valid = entry_rows + h < n
        rows_h = entry_rows[valid]
        side_h = entry_side[valid]
        fwd_ret = close[rows_h + h] / close[rows_h] - 1.0
        fwd_sign = np.sign(fwd_ret)
        model_dir = np.where(side_h > 0, 1.0, -1.0)
        hit = (fwd_sign == model_dir).mean()
        label_dir = np.where(entry_label_hard[valid] == 1, 1.0, np.where(entry_label_hard[valid] == 2, -1.0, 0.0))
        label_hit = (fwd_sign == label_dir).mean()
        horizon_hitrates[f"h{h}bars"] = {
            "model_score_direction_vs_realized_close_sign_hitrate": float(hit),
            "wave_hard_label_direction_vs_realized_close_sign_hitrate": float(label_hit),
            "n": int(valid.sum()),
        }
    report["diag2_direction_vs_simple_forward_return_sign"] = horizon_hitrates
    print(json.dumps(horizon_hitrates, indent=2))

    # --- Diagnostic 3: score/quality vs realized trade_return correlation, using a ledger built
    # from THIS exact model/checkpoint (not the earlier separately-trained cw_0.2 checkpoint's
    # saved ledger, to keep the whole analysis internally consistent) ---
    vol = pd.Series(np.diff(np.log(np.clip(close, 1e-9, None)), prepend=np.log(close[0]))).rolling(TRAIL_VOL_BARS, min_periods=TRAIL_VOL_BARS).std().to_numpy()
    tp_moves = TP_MULT * vol[entry_rows]
    sl_moves = SL_MULT * vol[entry_rows]
    finite = np.isfinite(tp_moves) & np.isfinite(sl_moves)
    sim_idx, sim_score, sim_tp, sim_sl = entry_rows[finite], score_oos[fresh_mask][finite], tp_moves[finite], sl_moves[finite]
    result = simulate_single_position(
        timestamps=panel["timestamp"], open_px=panel["open"].to_numpy(dtype=np.float64),
        high=panel["high"].to_numpy(dtype=np.float64), low=panel["low"].to_numpy(dtype=np.float64),
        close=close, decision_indices=sim_idx, scores=sim_score, tp_moves=sim_tp, sl_moves=sim_sl,
        upper_threshold=th.upper, lower_threshold=th.lower, horizon_bars=HORIZON_BARS,
        margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE, roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
    )
    ledger = result.ledger
    ledger_rows = ts_to_row = None
    dt_to_row = pd.Series(np.arange(n), index=panel["timestamp"])
    ledger_rows = dt_to_row.reindex(ledger["decision_timestamp"]).to_numpy().astype(int)

    row_to_pos = {int(r): i for i, r in enumerate(row_oos)}
    q_pred = np.array([quality_oos[row_to_pos[r]] if r in row_to_pos else np.nan for r in ledger_rows])
    ledger["quality_pred"] = q_pred
    valid_q = np.isfinite(q_pred)
    report["ledger_rebuilt_summary"] = {
        "n_trades": int(len(ledger)), "win_rate": float((ledger["trade_return"] > 0).mean()),
        "sum_ret_pct": float(ledger["trade_return"].sum() * 100),
    }

    corr_score = float(np.corrcoef(ledger["score"], ledger["trade_return"])[0, 1])
    corr_abs_score = float(np.corrcoef(ledger["score"].abs(), ledger["trade_return"])[0, 1])
    corr_quality = float(np.corrcoef(ledger.loc[valid_q, "quality_pred"], ledger.loc[valid_q, "trade_return"])[0, 1]) if valid_q.sum() > 2 else None
    report["diag3_score_quality_vs_trade_return_correlation"] = {
        "corr_signed_score_vs_trade_return": corr_score,
        "corr_abs_score_confidence_vs_trade_return": corr_abs_score,
        "corr_quality_pred_vs_trade_return": corr_quality,
        "n_trades": int(len(ledger)),
    }
    print(json.dumps(report["diag3_score_quality_vs_trade_return_correlation"], indent=2))

    # --- Diagnostic 4: exit timing -- are SL hits early (noise stop-out) or late (real reversal)? ---
    by_reason = ledger.groupby("reason")["bars_held"].agg(["count", "mean", "median"]).to_dict(orient="index")
    early_sl_frac = float((ledger.loc[ledger["reason"] == "sl", "bars_held"] <= 3).mean()) if (ledger["reason"] == "sl").any() else None
    report["diag4_exit_timing"] = {
        "bars_held_by_reason": by_reason,
        "frac_sl_exits_within_3_bars_15min": early_sl_frac,
        "horizon_bars": 288,
        "note": "bars_held near 1 for SL exits => essentially immediate stop-out (entry noise/spread, not a real reversal after the model's signal had time to play out)",
    }
    print(json.dumps(report["diag4_exit_timing"], indent=2, default=str))

    # --- Diagnostic 5: is the SL distance simply too tight vs typical short-horizon noise, independent of direction? ---
    vol = pd.Series(np.diff(np.log(np.clip(close, 1e-9, None)), prepend=np.log(close[0]))).rolling(TRAIL_VOL_BARS, min_periods=TRAIL_VOL_BARS).std().to_numpy()
    sl_move_at_entry = SL_MULT * vol[ledger_rows]
    # realized |return| over the next 3 bars from entry, regardless of direction (pure noise magnitude check)
    valid_noise = ledger_rows + 3 < n
    noise_3bar = np.abs(close[ledger_rows[valid_noise] + 3] / close[ledger_rows[valid_noise]] - 1.0)
    report["diag5_sl_distance_vs_noise"] = {
        "mean_sl_move_pct": float(np.nanmean(sl_move_at_entry) * 100),
        "median_sl_move_pct": float(np.nanmedian(sl_move_at_entry) * 100),
        "mean_abs_3bar_realized_move_pct": float(np.mean(noise_3bar) * 100),
        "median_abs_3bar_realized_move_pct": float(np.median(noise_3bar) * 100),
        "note": "if the 3-bar realized move magnitude is comparable to (or bigger than) the SL distance, most trades are vulnerable to being stopped out by ordinary short-horizon noise before direction has time to resolve",
    }
    print(json.dumps(report["diag5_sl_distance_vs_noise"], indent=2))

    (OUT_DIR / "analysis_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"wrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
