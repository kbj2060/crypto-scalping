#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from playbook_router import PlaybookRouter

MICRO_DB = ROOT / "data/live/microstructure.duckdb"
TAIL_DB = ROOT / "data/live/tail_risk.duckdb"


@dataclass
class BacktestResult:
    objective: float
    pnl_pct: float
    sharpe_1m: float
    mdd_pct: float
    win_rate_pct: float
    trades: int
    avg_pos: float
    fee_bps: float
    slip_bps: float
    leverage: float
    params: dict


def load_signals() -> pd.DataFrame:
    con_m = duckdb.connect(str(MICRO_DB))
    con_t = duckdb.connect(str(TAIL_DB))
    m = con_m.execute(
        """
        SELECT
          date_trunc('minute', ts) AS ts,
          obi, nif_whale, eai, funding_rate,
          shadow_toxicity_score, shadow_queue_collapse, shadow_absorption_score, shadow_queue_bias
        FROM microstructure_1m
        ORDER BY ts
        """
    ).df()
    t = con_t.execute(
        """
        SELECT
          date_trunc('minute', ts) AS ts,
          long_usd_1m, short_usd_1m, shadow_aftershock_prob, shadow_risk_bucket
        FROM tail_risk_1m
        ORDER BY ts
        """
    ).df()
    con_m.close()
    con_t.close()

    m["ts"] = pd.to_datetime(m["ts"], utc=True)
    t["ts"] = pd.to_datetime(t["ts"], utc=True)
    df = pd.merge(m, t, on="ts", how="inner")
    return df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def fetch_binance_1m(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    url = "https://fapi.binance.com/fapi/v1/klines"
    out = []
    cur = start_ms
    while cur <= end_ms:
        params = {
            "symbol": symbol,
            "interval": "1m",
            "startTime": cur,
            "endTime": end_ms,
            "limit": 1500,
        }
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break
        out.extend(rows)
        cur = int(rows[-1][0]) + 60_000
        if len(rows) < 1500:
            break
    if not out:
        raise RuntimeError("No Binance klines fetched.")
    px = pd.DataFrame(
        out,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_base", "taker_quote", "ignore",
        ],
    )
    px["ts"] = pd.to_datetime(px["open_time"].astype(np.int64), unit="ms", utc=True)
    px["close"] = pd.to_numeric(px["close"], errors="coerce")
    return px[["ts", "close"]].dropna().drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def build_feature_frame(symbol: str) -> pd.DataFrame:
    sig = load_signals()
    if len(sig) < 300:
        raise RuntimeError(f"Not enough signal rows: {len(sig)}")
    start_ms = int(sig["ts"].min().timestamp() * 1000) - 180_000
    end_ms = int(sig["ts"].max().timestamp() * 1000) + 180_000
    px = fetch_binance_1m(symbol, start_ms=start_ms, end_ms=end_ms)

    sig["ts"] = pd.to_datetime(sig["ts"], utc=True).astype("datetime64[ns, UTC]")
    px["ts"] = pd.to_datetime(px["ts"], utc=True).astype("datetime64[ns, UTC]")
    df = pd.merge_asof(sig.sort_values("ts"), px.sort_values("ts"), on="ts", direction="backward")
    df = df.dropna(subset=["close"]).reset_index(drop=True)

    # Price-derived MFT features
    df["price_change_30m"] = df["close"] / df["close"].shift(30) - 1.0
    roll30_max = df["close"].rolling(30, min_periods=10).max()
    roll30_min = df["close"].rolling(30, min_periods=10).min()
    df["price_volatility_30m"] = (roll30_max - roll30_min) / (df["close"] + 1e-12)
    prev55_max = df["close"].shift(5).rolling(55, min_periods=20).max()
    prev55_min = df["close"].shift(5).rolling(55, min_periods=20).min()
    df["price_breakout_60m"] = df["close"] > prev55_max
    df["price_breakdown_60m"] = df["close"] < prev55_min

    # Flow/structure rolling features
    df["nif_whale_sum_30m"] = df["nif_whale"].rolling(30, min_periods=10).sum()
    df["nif_whale_avg_30m"] = df["nif_whale"].rolling(30, min_periods=10).mean()
    df["nif_whale_std_30m"] = df["nif_whale"].rolling(30, min_periods=10).std()
    df["absorption_avg_30m"] = df["shadow_absorption_score"].rolling(30, min_periods=10).mean()
    df["bias_avg_30m"] = df["shadow_queue_bias"].rolling(30, min_periods=10).mean()
    df["toxicity_avg_30m"] = df["shadow_toxicity_score"].rolling(30, min_periods=10).mean()
    df["eai_delta_15m"] = df["eai"] - df["eai"].shift(15)

    # Tail-derived features expected by router
    mu_l = df["long_usd_1m"].rolling(30, min_periods=10).mean()
    sd_l = df["long_usd_1m"].rolling(30, min_periods=10).std().clip(lower=1e-6)
    mu_s = df["short_usd_1m"].rolling(30, min_periods=10).mean()
    sd_s = df["short_usd_1m"].rolling(30, min_periods=10).std().clip(lower=1e-6)
    df["z_long"] = (df["long_usd_1m"] - mu_l) / sd_l
    df["z_short"] = (df["short_usd_1m"] - mu_s) / sd_s
    ret1 = (df["close"] / df["close"].shift(1) - 1.0).abs().clip(lower=1e-4)
    dominant = np.where(df["z_long"] >= df["z_short"], df["long_usd_1m"], df["short_usd_1m"])
    df["lai"] = dominant / ret1

    df["ret_fwd_1m"] = df["close"].pct_change().shift(-1)
    keep = [
        "ts", "ret_fwd_1m", "obi", "nif_whale", "eai", "funding_rate",
        "shadow_toxicity_score", "shadow_queue_collapse", "shadow_absorption_score", "shadow_queue_bias",
        "shadow_aftershock_prob", "shadow_risk_bucket",
        "price_change_30m", "price_volatility_30m", "price_breakout_60m", "price_breakdown_60m",
        "nif_whale_sum_30m", "nif_whale_avg_30m", "nif_whale_std_30m",
        "absorption_avg_30m", "bias_avg_30m", "toxicity_avg_30m", "eai_delta_15m",
        "z_long", "z_short", "lai",
    ]
    out = df[keep].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if len(out) < 200:
        raise RuntimeError(f"Not enough rows after feature build: {len(out)}")
    return out


def simulate(df: pd.DataFrame, params: dict, fee_bps: float, slip_bps: float, leverage: float) -> BacktestResult:
    r = PlaybookRouter()
    n = len(df)
    pos = np.zeros(n, dtype=np.float64)
    hold = 0
    min_hold = int(params["min_hold"])

    for i, row in enumerate(df.itertuples(index=False)):
        ms = {
            "obi": float(row.obi),
            "nif_whale": float(row.nif_whale),
            "eai": float(row.eai),
            "funding_rate": float(row.funding_rate),
            "shadow_absorption_score": float(row.shadow_absorption_score),
            "shadow_queue_collapse": float(row.shadow_queue_collapse),
            "shadow_toxicity_score": float(row.shadow_toxicity_score),
            "shadow_queue_bias": int(row.shadow_queue_bias),
            "price_change_30m": float(row.price_change_30m),
            "price_volatility_30m": float(row.price_volatility_30m),
            "price_breakout_60m": bool(row.price_breakout_60m),
            "price_breakdown_60m": bool(row.price_breakdown_60m),
            "nif_whale_sum_30m": float(row.nif_whale_sum_30m),
            "nif_whale_avg_30m": float(row.nif_whale_avg_30m),
            "nif_whale_std_30m": float(row.nif_whale_std_30m),
            "absorption_avg_30m": float(row.absorption_avg_30m),
            "bias_avg_30m": float(row.bias_avg_30m),
            "toxicity_avg_30m": float(row.toxicity_avg_30m),
            "eai_delta_15m": float(row.eai_delta_15m),
        }
        tr = {
            "z_long": float(row.z_long),
            "z_short": float(row.z_short),
            "lai": float(row.lai),
            "shadow_aftershock_prob": float(row.shadow_aftershock_prob),
            "shadow_risk_bucket": str(row.shadow_risk_bucket),
        }

        # Base directional edge
        edge = (
            params["w_obi"] * float(row.obi)
            + params["w_nif"] * float(row.nif_whale)
            + params["w_abs"] * float(row.shadow_absorption_score)
            - params["w_tox"] * float(row.shadow_toxicity_score)
            - params["w_aft"] * float(row.shadow_aftershock_prob)
        )
        edge = float(np.tanh(edge))
        base_action = 1 if edge > params["edge_enter"] else (2 if edge < -params["edge_enter"] else 0)
        base_kelly = float(np.clip(abs(edge), 0.0, 1.0))

        out = r.evaluate_all(action=base_action, pos=None, kelly=base_kelly, ms=ms, tr=tr)
        hft = out["winner_hft"]
        mft = out["winner_mft"]

        # Meta-combo:
        # 1) PB9 veto
        # 2) MFT matched -> primary decision
        # 3) MFT no-match + high conviction HFT -> HFT assist
        # 4) otherwise base signal
        action = base_action
        kelly = base_kelly
        if hft.get("name") == "PB9_VACUUM_WHIPSAW" and bool(hft.get("matched", False)):
            action, kelly = 0, 0.0
        elif bool(mft.get("matched", False)):
            action = int(mft.get("action", 0))
            kelly = float(mft.get("kelly", base_kelly))
        elif bool(hft.get("matched", False)) and abs(edge) >= params["hft_assist_edge"]:
            action = int(hft.get("action", base_action))
            kelly = float(hft.get("kelly", base_kelly)) * params["hft_assist_mult"]

        # global risk brakes
        if float(row.shadow_toxicity_score) > params["tox_veto"] or float(row.shadow_aftershock_prob) > params["aft_veto"]:
            action = 0
            kelly = 0.0

        kelly = float(np.clip(kelly, 0.0, 1.0))
        desired = 0.0 if action == 0 else (1.0 if action == 1 else -1.0)
        desired *= kelly

        prev = pos[i - 1] if i > 0 else 0.0
        if abs(prev) > 1e-9 and np.sign(prev) != np.sign(desired) and hold < min_hold:
            pos[i] = prev
            hold += 1
        else:
            pos[i] = desired
            hold = 0 if abs(desired) < 1e-9 else hold + 1

    ret = df["ret_fwd_1m"].to_numpy(dtype=np.float64)
    fee = float(fee_bps) / 10_000.0
    slip = float(slip_bps) / 10_000.0
    lev = float(max(leverage, 0.0))
    turn = np.abs(np.diff(pos, prepend=0.0))
    pnl = lev * pos * ret - (fee + slip) * lev * turn
    eq = np.cumprod(1.0 + pnl)
    total = float(eq[-1] - 1.0)
    mu = float(np.mean(pnl))
    sd = float(np.std(pnl) + 1e-12)
    sharpe = mu / sd
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    mdd = float(-dd.min())
    wr = float((pnl > 0).mean())
    trades = int((turn > 1e-9).sum())
    obj = total - 0.55 * mdd + 0.04 * sharpe - 0.000015 * trades
    return BacktestResult(
        objective=float(obj),
        pnl_pct=float(total * 100.0),
        sharpe_1m=float(sharpe),
        mdd_pct=float(mdd * 100.0),
        win_rate_pct=float(wr * 100.0),
        trades=int(trades),
        avg_pos=float(np.mean(np.abs(pos))),
        fee_bps=float(fee_bps),
        slip_bps=float(slip_bps),
        leverage=float(leverage),
        params=params,
    )


def random_search(df: pd.DataFrame, trials: int, fee_bps: float, slip_bps: float, leverage: float, seed: int) -> BacktestResult:
    rng = np.random.default_rng(seed)
    best: BacktestResult | None = None
    for _ in range(trials):
        p = {
            "w_obi": float(rng.uniform(0.3, 1.8)),
            "w_nif": float(rng.uniform(0.4, 2.2)),
            "w_abs": float(rng.uniform(0.2, 1.6)),
            "w_tox": float(rng.uniform(0.6, 2.4)),
            "w_aft": float(rng.uniform(0.6, 2.4)),
            "edge_enter": float(rng.uniform(0.08, 0.35)),
            "hft_assist_edge": float(rng.uniform(0.20, 0.55)),
            "hft_assist_mult": float(rng.uniform(0.55, 1.15)),
            "tox_veto": float(rng.uniform(0.72, 0.92)),
            "aft_veto": float(rng.uniform(0.65, 0.90)),
            "min_hold": int(rng.integers(1, 9)),
        }
        r = simulate(df, p, fee_bps=fee_bps, slip_bps=slip_bps, leverage=leverage)
        if best is None or r.objective > best.objective:
            best = r
    assert best is not None
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="ETHUSDT")
    ap.add_argument("--fee-bps", type=float, default=2.0)
    ap.add_argument("--slip-bps", type=float, default=1.0)
    ap.add_argument("--leverage", type=float, default=1.0)
    ap.add_argument("--trials", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/ensemble/metrics/hft_mft_meta_combo_backtest.json")
    args = ap.parse_args()

    df = build_feature_frame(args.symbol)
    best = random_search(
        df,
        trials=args.trials,
        fee_bps=args.fee_bps,
        slip_bps=args.slip_bps,
        leverage=args.leverage,
        seed=args.seed,
    )
    best_run = simulate(df, best.params, fee_bps=args.fee_bps, slip_bps=args.slip_bps, leverage=args.leverage)

    result = {
        "symbol": args.symbol,
        "rows": int(len(df)),
        "ts_min": str(df["ts"].min()),
        "ts_max": str(df["ts"].max()),
        "best": asdict(best_run),
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== HFT+MFT Meta Combo Backtest ===")
    print(f"symbol={args.symbol} rows={len(df)} range=[{df['ts'].min()} ~ {df['ts'].max()}]")
    print(
        f"BEST pnl={best_run.pnl_pct:.3f}% sharpe={best_run.sharpe_1m:.4f} "
        f"mdd={best_run.mdd_pct:.3f}% win={best_run.win_rate_pct:.2f}% "
        f"trades={best_run.trades} avg_pos={best_run.avg_pos:.3f} lev={args.leverage}"
    )
    print("BEST params:", json.dumps(best_run.params, ensure_ascii=False))
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
