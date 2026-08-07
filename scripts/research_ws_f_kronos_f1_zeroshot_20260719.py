"""WS-F F1: Kronos zero-shot sanity - 방향 정확도 vs 모멘텀 벤치마크,
불확실성(q10-q90 폭) vs 실현변동성 상관. 탐색 구간(<2025-08-31)만 사용, 다중 롤링윈도.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "data/research/kronos/repo")
from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402

OUT_DIR = Path("docs/test_designs_duckdb_live_20260719/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

KLINE_1M_CSV = "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
EXPLORATION_END = "2025-08-31"
LOOKBACK = 400
PRED_LEN = 5
N_WINDOWS = 25
SAMPLE_COUNT = 5  # for q10/q90 uncertainty estimate


def main():
    report = {"stage": "WS-F-F1", "generated_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat()}

    import torch
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

    df = pd.read_csv(KLINE_1M_CSV, usecols=["timestamp", "open", "high", "low", "close", "volume", "quote_volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[df["timestamp"] < EXPLORATION_END].reset_index(drop=True)
    df_1h = df.set_index("timestamp").resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last",
         "volume": "sum", "quote_volume": "sum"}
    ).dropna().reset_index().rename(columns={"quote_volume": "amount"})

    n = len(df_1h)
    # evenly spaced rolling window starts across the exploration set (excluding the very tail
    # to leave room for pred_len)
    max_start = n - LOOKBACK - PRED_LEN - 1
    starts = np.linspace(0, max_start, N_WINDOWS).astype(int)
    starts = sorted(set(starts.tolist()))
    report["n_windows_requested"] = N_WINDOWS
    report["n_windows_actual"] = len(starts)

    results = []
    for i, s in enumerate(starts):
        x_df = df_1h.loc[s:s + LOOKBACK - 1, ["open", "high", "low", "close", "volume", "amount"]]
        x_ts = df_1h.loc[s:s + LOOKBACK - 1, "timestamp"]
        y_ts = df_1h.loc[s + LOOKBACK: s + LOOKBACK + PRED_LEN - 1, "timestamp"]
        y_true = df_1h.loc[s + LOOKBACK: s + LOOKBACK + PRED_LEN - 1, "close"].values
        last_hist_close = x_df["close"].values[-1]
        # realized vol proxy: std of log returns over the lookback window
        realized_vol = float(np.std(np.diff(np.log(x_df["close"].values))))

        preds = []
        for _ in range(SAMPLE_COUNT):
            pred_df = predictor.predict(
                df=x_df, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=PRED_LEN,
                T=1.0, top_p=0.9, sample_count=1, verbose=False,
            )
            preds.append(pred_df["close"].values[-1])
        preds = np.array(preds)
        pred_median = float(np.median(preds))
        q10, q90 = float(np.percentile(preds, 10)), float(np.percentile(preds, 90))

        pred_dir = np.sign(pred_median - last_hist_close)
        true_dir = np.sign(y_true[-1] - last_hist_close)
        momentum_dir = np.sign(x_df["close"].values[-1] - x_df["close"].values[-6])  # 6h momentum

        results.append({
            "window_idx": i,
            "window_end_history": str(x_ts.iloc[-1]),
            "kronos_dir_correct": bool(pred_dir == true_dir) if true_dir != 0 else None,
            "momentum_dir_correct": bool(momentum_dir == true_dir) if true_dir != 0 else None,
            "q10_q90_width": q90 - q10,
            "realized_vol_lookback": realized_vol,
            "actual_fwd_abs_ret": float(abs(np.log(y_true[-1] / last_hist_close))),
        })
        print(f"window {i+1}/{len(starts)} done")

    df_res = pd.DataFrame(results)
    valid = df_res.dropna(subset=["kronos_dir_correct", "momentum_dir_correct"])
    kronos_acc = float(valid["kronos_dir_correct"].mean())
    momentum_acc = float(valid["momentum_dir_correct"].mean())
    coin_flip = 0.5

    unc_corr = float(df_res["q10_q90_width"].corr(df_res["actual_fwd_abs_ret"], method="spearman"))

    report["results_per_window"] = results
    report["summary"] = {
        "n_valid_windows": int(len(valid)),
        "kronos_direction_accuracy": kronos_acc,
        "momentum_benchmark_accuracy": momentum_acc,
        "coin_flip_baseline": coin_flip,
        "uncertainty_vs_realized_fwd_move_spearman": unc_corr,
    }

    kill = (kronos_acc <= momentum_acc) and (abs(unc_corr) < 0.2)
    report["F1_kill_gate_triggered"] = bool(kill)
    report["F1_verdict"] = (
        "KILL -- direction accuracy at/below momentum benchmark AND uncertainty proxy uncorrelated "
        "with realized move -- stop WS-F track per design doc kill gate"
        if kill else
        "PROCEED to F2 -- at least one signal (direction beats momentum, or uncertainty calibrated)"
    )

    out_json = OUT_DIR / "ws_f_f1_zeroshot_20260719.json"
    out_json.write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False))
    print("WROTE", out_json)
    print(json.dumps(report["summary"], indent=2, default=str))
    print(report["F1_verdict"])


if __name__ == "__main__":
    main()
