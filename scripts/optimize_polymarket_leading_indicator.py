#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from itertools import product
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


def _http_json(url: str, timeout: float = 8.0):
    req = Request(
        url=url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json,text/plain,*/*",
            "Connection": "close",
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def _parse_strike(text: str) -> float | None:
    if not text:
        return None
    m = re.search(r"\$?\s*([0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?)", str(text))
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except Exception:
            return None
    m2 = re.search(r"\$?\s*([0-9]+(?:\.[0-9]+)?)\s*([kK])\b", str(text))
    if m2:
        try:
            return float(m2.group(1)) * 1000.0
        except Exception:
            return None
    return None


def _fetch_polymarket_event(slug: str) -> dict:
    ev = _http_json(f"https://gamma-api.polymarket.com/events?{urlencode({'slug': slug})}")
    if isinstance(ev, list):
        return dict(ev[0] or {}) if ev else {}
    if isinstance(ev, dict):
        arr = ev.get("events", ev.get("data", []))
        if isinstance(arr, list) and arr:
            return dict(arr[0] or {})
        return dict(ev)
    return {}


def _fetch_market_history(token_id: str, start_ts: int, end_ts: int, fidelity: int = 1) -> pd.Series:
    q = urlencode({"market": token_id, "startTs": start_ts, "endTs": end_ts, "fidelity": int(max(1, fidelity))})
    raw = _http_json(f"https://clob.polymarket.com/prices-history?{q}")
    hist = list((raw or {}).get("history", []) or [])
    vals = []
    for x in hist:
        try:
            t = pd.Timestamp(int(x.get("t")), unit="s", tz="UTC")
            p = float(x.get("p"))
        except Exception:
            continue
        if not np.isfinite(p):
            continue
        vals.append((t, float(np.clip(p, 0.0, 1.0))))
    if not vals:
        return pd.Series(dtype=float)
    return pd.Series({t: p for t, p in vals}, dtype=float).sort_index()


def _fetch_binance_1m(start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> pd.DataFrame:
    start_ms = int(start_utc.floor("min").timestamp() * 1000)
    end_ms = int(end_utc.ceil("min").timestamp() * 1000)
    cursor = start_ms
    rows: list[list] = []
    while cursor <= end_ms:
        q = urlencode(
            {
                "symbol": "ETHUSDT",
                "interval": "1m",
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1500,
            }
        )
        raw = _http_json(f"https://fapi.binance.com/fapi/v1/klines?{q}")
        if not isinstance(raw, list) or not raw:
            break
        rows.extend(raw)
        nxt = int(raw[-1][0]) + 60_000
        if nxt <= cursor:
            break
        cursor = nxt
    if not rows:
        return pd.DataFrame(columns=["ts", "close"])
    df = pd.DataFrame(rows).iloc[:, [0, 4]]
    df.columns = ["open_time", "close"]
    df["ts"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True).astype("datetime64[ns, UTC]")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["ts", "close"]).drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return df[["ts", "close"]]


def _rolling_z(s: pd.Series, win: int) -> pd.Series:
    mu = s.rolling(win, min_periods=max(5, win // 4)).mean()
    sd = s.rolling(win, min_periods=max(5, win // 4)).std(ddof=0).replace(0.0, np.nan)
    z = (s - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _build_feature_frame(slug: str, tz: str, days: int, fidelity: int, z_window: int) -> pd.DataFrame:
    now_local = pd.Timestamp.now(tz=tz)
    start_local = (now_local - pd.Timedelta(days=max(1, days))).floor("min")
    end_local = now_local.ceil("min")
    start_ts = int(start_local.tz_convert("UTC").timestamp())
    end_ts = int(end_local.tz_convert("UTC").timestamp())

    event = _fetch_polymarket_event(slug)
    markets = list((event or {}).get("markets", []) or [])
    if not markets:
        return pd.DataFrame()

    series_by_market: list[tuple[float, pd.Series]] = []
    for m in markets:
        mm = dict(m or {})
        question = str(mm.get("question", mm.get("title", mm.get("name", ""))) or "")
        strike = _parse_strike(question)
        if strike is None:
            continue
        token_ids = mm.get("clobTokenIds", [])
        if isinstance(token_ids, str):
            try:
                token_ids = json.loads(token_ids)
            except Exception:
                token_ids = [x.strip().strip("\"").strip("'") for x in token_ids.split(",") if str(x).strip()]
        if not isinstance(token_ids, list) or not token_ids:
            continue
        token_id = str(token_ids[0]).strip()
        if not token_id:
            continue
        s = _fetch_market_history(token_id, start_ts=start_ts, end_ts=end_ts, fidelity=fidelity)
        if len(s) == 0:
            continue
        series_by_market.append((float(strike), s))

    if not series_by_market:
        return pd.DataFrame()

    idx = pd.Index(sorted(set().union(*[set(s.index) for _, s in series_by_market])))
    probs = pd.DataFrame(index=idx)
    strikes = []
    for i, (strike, s) in enumerate(series_by_market):
        cname = f"m{i}"
        probs[cname] = s.reindex(idx).ffill()
        strikes.append(float(strike))
    probs = probs.dropna(how="all").sort_index()

    pvals = probs.to_numpy(dtype=float)
    row_sum = np.nansum(pvals, axis=1)
    norm = np.where(row_sum > 1e-12, row_sum, np.nan)
    w = pvals / norm[:, None]
    strike_arr = np.array(strikes, dtype=float)

    weighted_target = np.nansum(w * strike_arr[None, :], axis=1)
    mode_prob = np.nanmax(pvals, axis=1)
    dispersion = np.nanstd(pvals, axis=1)
    entropy = np.array(
        [
            float(-np.nansum([pp * math.log(max(pp, 1e-12)) for pp in row if np.isfinite(pp) and pp > 0.0]))
            for row in w
        ],
        dtype=float,
    )

    dp = probs.diff()
    breadth = dp.apply(np.sign).mean(axis=1).fillna(0.0)

    feat = pd.DataFrame(
        {
            "ts": probs.index,
            "mode_prob": mode_prob,
            "weighted_target": weighted_target,
            "dispersion": dispersion,
            "entropy": entropy,
            "breadth": breadth.to_numpy(dtype=float),
        }
    ).dropna(subset=["mode_prob", "weighted_target"])
    feat["ts"] = pd.to_datetime(feat["ts"], utc=True).astype("datetime64[ns, UTC]")
    feat = feat.sort_values("ts").reset_index(drop=True)

    px = _fetch_binance_1m(feat["ts"].min() - pd.Timedelta(minutes=60), feat["ts"].max() + pd.Timedelta(minutes=60))
    if len(px) == 0:
        return pd.DataFrame()
    merged = pd.merge_asof(feat.sort_values("ts"), px.sort_values("ts"), on="ts", direction="backward")
    merged = merged.dropna(subset=["close"]).reset_index(drop=True)

    merged["delta_1m"] = merged["mode_prob"].diff(1)
    merged["delta_3m"] = merged["mode_prob"].diff(3)
    merged["target_gap_pct"] = (merged["weighted_target"] - merged["close"]) / merged["close"].replace(0.0, np.nan)
    merged["target_gap_chg_1m"] = merged["target_gap_pct"].diff(1)
    merged["entropy_chg_1m"] = merged["entropy"].diff(1)
    merged["disp_chg_1m"] = merged["dispersion"].diff(1)

    for c in ["delta_1m", "delta_3m", "target_gap_pct", "target_gap_chg_1m", "entropy_chg_1m", "disp_chg_1m"]:
        merged[f"z_{c}"] = _rolling_z(merged[c].fillna(0.0), z_window)

    merged["ret_fut_5m"] = merged["close"].shift(-5) / merged["close"] - 1.0
    merged["ret_fut_15m"] = merged["close"].shift(-15) / merged["close"] - 1.0
    merged["ret_fut_30m"] = merged["close"].shift(-30) / merged["close"] - 1.0
    merged = merged.dropna(subset=["delta_1m", "target_gap_pct", "ret_fut_5m", "ret_fut_15m", "ret_fut_30m"]).reset_index(drop=True)
    return merged


def _acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    a = np.sign(y_true)
    b = np.sign(y_pred)
    m = a != 0
    if np.sum(m) == 0:
        return 0.0
    return float(np.mean(a[m] == b[m]))


def _evaluate_rule(df: pd.DataFrame, shock_th: float, gap_th: float, mode_floor: float, min_breadth: float, score_w: tuple[float, float, float, float]) -> dict:
    w1, w2, w3, w4 = score_w
    score = (
        w1 * df["z_delta_1m"].to_numpy(dtype=float)
        + w2 * df["z_target_gap_pct"].to_numpy(dtype=float)
        + w3 * df["breadth"].to_numpy(dtype=float)
        + w4 * (-df["z_entropy_chg_1m"].to_numpy(dtype=float))
    )
    signal = np.sign(score)
    trig = (
        (np.abs(df["delta_1m"].to_numpy(dtype=float)) >= shock_th)
        & (np.abs(df["target_gap_pct"].to_numpy(dtype=float)) >= gap_th)
        & (df["mode_prob"].to_numpy(dtype=float) >= mode_floor)
        & (np.abs(df["breadth"].to_numpy(dtype=float)) >= min_breadth)
        & (signal != 0)
    )
    if np.sum(trig) == 0:
        return {}
    s = signal[trig]
    r5 = df["ret_fut_5m"].to_numpy(dtype=float)[trig]
    r15 = df["ret_fut_15m"].to_numpy(dtype=float)[trig]
    r30 = df["ret_fut_30m"].to_numpy(dtype=float)[trig]

    acc5 = _acc(r5, s)
    acc15 = _acc(r15, s)
    acc30 = _acc(r30, s)
    edge5_bps = float(np.mean(s * r5) * 10000.0)
    edge15_bps = float(np.mean(s * r15) * 10000.0)
    cov = float(np.mean(trig))
    n = int(np.sum(trig))
    # 선행성 점수: 정확도(5/15/30) + signed return edge, 저커버리지 패널티
    lead_score = (
        (acc5 - 0.5) * 1.4
        + (acc15 - 0.5) * 1.0
        + (acc30 - 0.5) * 0.6
        + np.clip(edge5_bps / 20.0, -0.3, 0.3)
        + np.clip(edge15_bps / 25.0, -0.25, 0.25)
    )
    if cov < 0.01:
        lead_score -= 0.2
    if n < 20:
        lead_score -= 0.15
    return {
        "lead_score": float(lead_score),
        "n_signals": n,
        "coverage_pct": cov * 100.0,
        "acc5": acc5 * 100.0,
        "acc15": acc15 * 100.0,
        "acc30": acc30 * 100.0,
        "edge5_bps": edge5_bps,
        "edge15_bps": edge15_bps,
        "shock_th_pctp": shock_th * 100.0,
        "gap_th_pct": gap_th * 100.0,
        "mode_floor_pct": mode_floor * 100.0,
        "min_breadth": min_breadth,
        "weights": f"{w1:.2f},{w2:.2f},{w3:.2f},{w4:.2f}",
    }


def main():
    ap = argparse.ArgumentParser(description="Optimize polymarket leading-indicator thresholds/conditions.")
    ap.add_argument("--slug", default="ethereum-price-on-april-19")
    ap.add_argument("--tz", default="Asia/Seoul")
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--fidelity", type=int, default=1)
    ap.add_argument("--z-window", type=int, default=120)
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--out-csv", default="data/live/polymarket_lead_opt_results.csv")
    args = ap.parse_args()

    df = _build_feature_frame(args.slug, args.tz, days=int(max(1, args.days)), fidelity=int(max(1, args.fidelity)), z_window=int(max(30, args.z_window)))
    if len(df) < 100:
        print("insufficient feature rows")
        return

    shock_grid = [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025, 0.03]
    gap_grid = [0.0003, 0.0005, 0.0008, 0.001, 0.0015, 0.002, 0.003]
    mode_floor_grid = [0.15, 0.20, 0.25, 0.30]
    breadth_grid = [0.00, 0.05, 0.10, 0.15]
    weight_grid = [
        (1.00, 0.00, 0.00, 0.00),
        (0.70, 0.30, 0.00, 0.00),
        (0.60, 0.25, 0.15, 0.00),
        (0.55, 0.25, 0.10, 0.10),
        (0.50, 0.20, 0.20, 0.10),
        (0.45, 0.30, 0.15, 0.10),
    ]

    results = []
    for shock_th, gap_th, mode_floor, min_breadth, w in product(shock_grid, gap_grid, mode_floor_grid, breadth_grid, weight_grid):
        m = _evaluate_rule(df, shock_th=shock_th, gap_th=gap_th, mode_floor=mode_floor, min_breadth=min_breadth, score_w=w)
        if not m:
            continue
        if m["n_signals"] < 12:
            continue
        results.append(m)

    if not results:
        print("no candidate rule passed minimum constraints")
        return

    out = pd.DataFrame(results).sort_values(["lead_score", "acc15", "edge15_bps"], ascending=[False, False, False]).reset_index(drop=True)
    out.to_csv(args.out_csv, index=False)

    print("=== Optimized Polymarket Leading Indicator ===")
    print(f"slug={args.slug} samples={len(df)} window_days={args.days} fidelity={args.fidelity}m z_window={args.z_window}")
    print(f"searched={len(results)} candidates  saved={args.out_csv}")
    topk = out.head(int(max(1, args.top_k)))
    cols = [
        "lead_score",
        "n_signals",
        "coverage_pct",
        "acc5",
        "acc15",
        "acc30",
        "edge5_bps",
        "edge15_bps",
        "shock_th_pctp",
        "gap_th_pct",
        "mode_floor_pct",
        "min_breadth",
        "weights",
    ]
    with pd.option_context("display.max_rows", None, "display.width", 200, "display.float_format", "{:,.3f}".format):
        print(topk[cols].to_string(index=False))

    best = topk.iloc[0].to_dict()
    baseline = _evaluate_rule(
        df,
        shock_th=0.03,   # 기존 3%p 충격 기준
        gap_th=0.0,
        mode_floor=0.0,
        min_breadth=0.0,
        score_w=(1.0, 0.0, 0.0, 0.0),
    )
    if baseline:
        print("\n--- Baseline (delta_1m only, 3%p) ---")
        print(
            f"signals={baseline['n_signals']} cov={baseline['coverage_pct']:.3f}% "
            f"acc5={baseline['acc5']:.2f}% acc15={baseline['acc15']:.2f}% acc30={baseline['acc30']:.2f}% "
            f"edge5={baseline['edge5_bps']:+.3f}bps edge15={baseline['edge15_bps']:+.3f}bps "
            f"lead_score={baseline['lead_score']:.3f}"
        )
        print(
            f"best delta: acc15 {best['acc15']-baseline['acc15']:+.2f}%p, "
            f"edge15 {best['edge15_bps']-baseline['edge15_bps']:+.3f}bps, "
            f"coverage {best['coverage_pct']-baseline['coverage_pct']:+.3f}%p"
        )

    print("\n--- Recommended Condition (best) ---")
    print(
        "trigger: "
        f"|delta_1m|>={best['shock_th_pctp']:.2f}%p AND "
        f"|target_gap|>={best['gap_th_pct']:.3f}% AND "
        f"mode_prob>={best['mode_floor_pct']:.1f}% AND "
        f"|breadth|>={best['min_breadth']:.2f}"
    )
    print(f"score: sign({best['weights']}) over [z_delta_1m, z_target_gap, breadth, -z_entropy_chg]")


if __name__ == "__main__":
    main()
