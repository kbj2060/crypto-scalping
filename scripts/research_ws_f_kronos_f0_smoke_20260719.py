"""WS-F F0: Kronos 준비 게이트 - 로드 스모크 테스트 + 사전학습 오염 조사 메모.

탐색 구간(~2025-08-31)만 사용 -- frozen holdout(>=2026-07-14)과 val/OOS 구간 접촉 금지.
별도 venv_kronos 사용 (numpy 버전 충돌 회피).
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


def main():
    report = {"stage": "WS-F-F0", "generated_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat()}

    print("Loading Kronos-small + tokenizer (downloads from HF hub on first run)...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
    report["device"] = device
    report["model"] = "NeoQuasar/Kronos-small"
    report["tokenizer"] = "NeoQuasar/Kronos-Tokenizer-base"

    print("Loading ETHUSDT 1h resample from 1m kline archive, exploration window only...")
    df = pd.read_csv(KLINE_1M_CSV, usecols=["timestamp", "open", "high", "low", "close", "volume", "quote_volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[df["timestamp"] < EXPLORATION_END].reset_index(drop=True)
    report["exploration_window_note"] = f"Using rows strictly before {EXPLORATION_END} only (pre-registered exploration split)"
    report["n_rows_1m_exploration"] = int(len(df))

    # resample to 1h to match Sigma6 timeframe
    df_1h = df.set_index("timestamp").resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last",
         "volume": "sum", "quote_volume": "sum"}
    ).dropna().reset_index()
    df_1h = df_1h.rename(columns={"quote_volume": "amount"})
    report["n_rows_1h_exploration"] = int(len(df_1h))

    lookback = 400
    pred_len = 20
    if len(df_1h) < lookback + pred_len + 10:
        report["error"] = "insufficient 1h rows in exploration window for smoke test"
        Path(OUT_DIR / "ws_f_f0_smoke_20260719.json").write_text(json.dumps(report, indent=2, default=str))
        print(json.dumps(report, indent=2, default=str))
        return

    start_idx = len(df_1h) - lookback - pred_len - 1
    x_df = df_1h.loc[start_idx:start_idx + lookback - 1, ["open", "high", "low", "close", "volume", "amount"]]
    x_ts = df_1h.loc[start_idx:start_idx + lookback - 1, "timestamp"]
    y_ts = df_1h.loc[start_idx + lookback: start_idx + lookback + pred_len - 1, "timestamp"]
    y_true = df_1h.loc[start_idx + lookback: start_idx + lookback + pred_len - 1, ["open", "high", "low", "close"]].reset_index(drop=True)

    print(f"Running inference: lookback={lookback}, pred_len={pred_len}...")
    pred_df = predictor.predict(
        df=x_df, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=pred_len,
        T=1.0, top_p=0.9, sample_count=3, verbose=False,
    )

    report["smoke_test"] = {
        "lookback": lookback,
        "pred_len": pred_len,
        "window_start": str(x_ts.iloc[0]),
        "window_end_history": str(x_ts.iloc[-1]),
        "pred_window": [str(y_ts.iloc[0]), str(y_ts.iloc[-1])],
        "pred_has_nan": bool(pred_df.isnull().values.any()),
        "pred_has_inf": bool(np.isinf(pred_df.select_dtypes(include=[np.number]).values).any()),
        "pred_close_head": pred_df["close"].head(5).tolist(),
        "true_close_head": y_true["close"].head(5).tolist(),
    }

    # simple direction-accuracy sanity vs momentum benchmark (single window, illustrative only --
    # real F1 test needs many windows, this is just a load/sanity smoke test)
    pred_dir = np.sign(pred_df["close"].values[-1] - x_df["close"].values[-1])
    true_dir = np.sign(y_true["close"].values[-1] - x_df["close"].values[-1])
    report["smoke_test"]["single_window_direction_match"] = bool(pred_dir == true_dir)

    report["F0_verdict"] = (
        "PASS -- model loads, runs, no NaN/Inf in single-window smoke test"
        if not report["smoke_test"]["pred_has_nan"] and not report["smoke_test"]["pred_has_inf"]
        else "FAIL -- NaN/Inf detected, do not proceed to F1"
    )

    out_json = OUT_DIR / "ws_f_f0_smoke_20260719.json"
    out_json.write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False))
    print("WROTE", out_json)
    print(json.dumps(report, indent=2, default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()
