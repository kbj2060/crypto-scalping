#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(__file__).resolve().parents[1]
MICRO_DB = ROOT / "data/live/microstructure.duckdb"
TAIL_DB = ROOT / "data/live/tail_risk.duckdb"
MODEL_OUT = ROOT / "data/ensemble/ckpt/meta_30m_duckdb_model.pkl"
METRIC_OUT = ROOT / "data/ensemble/metrics/meta_30m_duckdb_metrics.json"


@dataclass
class TrainConfig:
    symbol: str = "ETHUSDT"
    days: int = 7
    horizon_min: int = 30
    fee: float = 0.0005
    slip: float = 0.0002
    min_train_rows: int = 1500
    test_ratio: float = 0.2
    val_ratio: float = 0.2


def _fetch_binance_1m(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    url = "https://fapi.binance.com/fapi/v1/klines"
    out: list[list] = []
    cur = int(start_ms)
    while cur <= end_ms:
        params = {
            "symbol": symbol.upper(),
            "interval": "1m",
            "startTime": cur,
            "endTime": int(end_ms),
            "limit": 1500,
        }
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break
        out.extend(rows)
        last_open = int(rows[-1][0])
        nxt = last_open + 60_000
        if nxt <= cur:
            break
        cur = nxt
        time.sleep(0.02)

    if not out:
        raise RuntimeError("No kline rows from Binance")

    df = pd.DataFrame(out, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert("Asia/Seoul")
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[["open_time", "open", "high", "low", "close", "volume"]].dropna()
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    return df


def _load_duckdb_frames(cfg: TrainConfig) -> pd.DataFrame:
    con_m = duckdb.connect(str(MICRO_DB), read_only=True)
    con_t = duckdb.connect(str(TAIL_DB), read_only=True)
    q_m = f"""
    SELECT
      ts,
      obi, taker_buy_ratio, spoofing_score,
      nif_whale, nif_retail, eai, oi_delta_pct, funding_rate,
      signal_bias, shadow_toxicity_score, shadow_queue_collapse, shadow_absorption_score,
      shadow_queue_bias, shadow_regime_conf
    FROM microstructure_1m
    WHERE ts >= now() - INTERVAL '{int(cfg.days)} days'
    ORDER BY ts
    """
    q_t = f"""
    SELECT
      ts,
      long_usd_1m, short_usd_1m, mu_long, sigma_long, mu_short, sigma_short,
      shadow_aftershock_prob
    FROM tail_risk_1m
    WHERE ts >= now() - INTERVAL '{int(cfg.days)} days'
    ORDER BY ts
    """
    df_m = con_m.execute(q_m).df()
    df_t = con_t.execute(q_t).df()
    if df_m.empty:
        raise RuntimeError("microstructure rows are empty")
    if df_t.empty:
        df = df_m.copy()
    else:
        df = pd.merge_asof(
            df_m.sort_values("ts"),
            df_t.sort_values("ts"),
            on="ts",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=2),
        )
    if df.empty:
        raise RuntimeError("No merged micro/tail rows in selected period")
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("Asia/Seoul")
    return df


def _build_dataset(cfg: TrainConfig) -> tuple[pd.DataFrame, list[str]]:
    base = _load_duckdb_frames(cfg)
    ts_min = base["ts"].min() - pd.Timedelta(minutes=90)
    ts_max = base["ts"].max() + pd.Timedelta(minutes=cfg.horizon_min + 5)

    px = _fetch_binance_1m(
        symbol=cfg.symbol,
        start_ms=int(ts_min.tz_convert("UTC").timestamp() * 1000),
        end_ms=int(ts_max.tz_convert("UTC").timestamp() * 1000),
    )
    px = px.rename(columns={"open_time": "ts"})

    df = base.merge(px[["ts", "close"]], on="ts", how="left")
    df = df.dropna(subset=["close"]).copy()
    df = df.sort_values("ts").reset_index(drop=True)

    # label: 30분 후 수익률
    h = int(cfg.horizon_min)
    df["ret_fwd"] = df["close"].shift(-h) / df["close"] - 1.0

    # 안정적 학습을 위한 피처 엔지니어링
    df["liq_imbalance"] = (df["long_usd_1m"] - df["short_usd_1m"]) / (df["long_usd_1m"] + df["short_usd_1m"] + 1e-8)
    df["nif_x_obi"] = df["nif_whale"] * df["obi"]
    df["tox_x_collapse"] = df["shadow_toxicity_score"] * df["shadow_queue_collapse"]
    df["flow_abs"] = np.abs(df["nif_whale"])
    df["oi_abs"] = np.abs(df["oi_delta_pct"])
    df["eai_funding"] = df["eai"] * df["funding_rate"]

    feature_cols = [
        "obi", "taker_buy_ratio", "spoofing_score",
        "nif_whale", "nif_retail", "eai", "oi_delta_pct", "funding_rate",
        "signal_bias", "shadow_toxicity_score", "shadow_queue_collapse", "shadow_absorption_score",
        "shadow_queue_bias", "shadow_regime_conf", "shadow_aftershock_prob",
        "long_usd_1m", "short_usd_1m", "mu_long", "sigma_long", "mu_short", "sigma_short",
        "liq_imbalance", "nif_x_obi", "tox_x_collapse", "flow_abs", "oi_abs", "eai_funding",
    ]
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    df = df.dropna(subset=["ret_fwd"]).copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols + ["ret_fwd"]).reset_index(drop=True)
    return df, feature_cols


def _simulate_net_pnl(pred_ret: np.ndarray, real_ret: np.ndarray, entry_th: float, fee: float, slip: float) -> dict[str, float]:
    act = np.where(pred_ret > entry_th, 1, np.where(pred_ret < -entry_th, -1, 0))
    gross = act * real_ret
    trade_cost = (fee + slip) * (act != 0).astype(float)
    net = gross - trade_cost
    eq = np.cumprod(1.0 + net)
    wins = float((net > 0).mean()) if len(net) else 0.0
    return {
        "pnl_sum": float(np.sum(net)),
        "pnl_mean": float(np.mean(net)) if len(net) else 0.0,
        "win_rate": wins,
        "trades": int(np.sum(act != 0)),
        "final_equity": float(eq[-1]) if len(eq) else 1.0,
    }


def train(cfg: TrainConfig) -> dict:
    df, feat_cols = _build_dataset(cfg)
    if len(df) < cfg.min_train_rows:
        raise RuntimeError(f"not enough rows: {len(df)} < {cfg.min_train_rows}")

    n = len(df)
    n_test = max(200, int(n * cfg.test_ratio))
    n_val = max(200, int(n * cfg.val_ratio))
    n_train = n - n_val - n_test
    if n_train < 500:
        raise RuntimeError(f"not enough train rows after split: {n_train}")

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()

    X_train = train_df[feat_cols].to_numpy(dtype=np.float32)
    y_train = train_df["ret_fwd"].to_numpy(dtype=np.float32)
    X_val = val_df[feat_cols].to_numpy(dtype=np.float32)
    y_val = val_df["ret_fwd"].to_numpy(dtype=np.float32)
    X_test = test_df[feat_cols].to_numpy(dtype=np.float32)
    y_test = test_df["ret_fwd"].to_numpy(dtype=np.float32)

    # 큰 절대수익 구간에 더 큰 가중치
    w_train = np.clip(np.abs(y_train) / (np.quantile(np.abs(y_train), 0.75) + 1e-8), 0.5, 3.0)

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.03,
        max_iter=600,
        max_depth=6,
        min_samples_leaf=30,
        l2_regularization=1e-3,
        random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=w_train)

    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    # 검증 구간에서 임계치 탐색: 30분 수익 극대화
    grid = np.linspace(0.0003, 0.0040, 24)
    best = None
    for th in grid:
        m = _simulate_net_pnl(pred_val, y_val, float(th), fee=cfg.fee, slip=cfg.slip)
        score = m["final_equity"]
        if best is None or score > best["score"]:
            best = {"score": score, "entry_th": float(th), "metrics": m}
    assert best is not None

    test_metrics = _simulate_net_pnl(pred_test, y_test, best["entry_th"], fee=cfg.fee, slip=cfg.slip)

    payload = {
        "model": model,
        "features": feat_cols,
        "entry_th": float(best["entry_th"]),
        "horizon_min": int(cfg.horizon_min),
        "fee": float(cfg.fee),
        "slip": float(cfg.slip),
        "symbol": cfg.symbol.upper(),
        "trained_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
    }
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRIC_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, MODEL_OUT)

    metrics = {
        "rows_total": int(n),
        "rows_train": int(n_train),
        "rows_val": int(n_val),
        "rows_test": int(n_test),
        "period_start": str(df["ts"].min()),
        "period_end": str(df["ts"].max()),
        "best_entry_th": float(best["entry_th"]),
        "val": best["metrics"],
        "test": test_metrics,
        "model_out": str(MODEL_OUT),
    }
    METRIC_OUT.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Train 30m-return-maximizing model from DuckDB micro/tail data")
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--horizon-min", type=int, default=30)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--min-train-rows", type=int, default=1500)
    args = p.parse_args()

    cfg = TrainConfig(
        symbol=args.symbol,
        days=args.days,
        horizon_min=args.horizon_min,
        fee=args.fee,
        slip=args.slip,
        min_train_rows=args.min_train_rows,
    )
    out = train(cfg)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
