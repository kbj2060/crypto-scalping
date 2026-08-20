#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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


def _parse_ts_kst(v) -> pd.Timestamp | None:
    try:
        ts = pd.Timestamp(v)
        if ts.tzinfo is None:
            ts = ts.tz_localize("Asia/Seoul")
        return ts.tz_convert("UTC")
    except Exception:
        return None


def _fetch_polymarket_mode_series(slug: str, tz: str, fidelity: int = 1) -> pd.DataFrame:
    now_local = pd.Timestamp.now(tz=tz)
    start_local = (now_local - pd.Timedelta(days=1)).normalize()
    end_local = now_local.normalize() + pd.Timedelta(days=1)
    start_ts = int(start_local.tz_convert("UTC").timestamp())
    end_ts = int(end_local.tz_convert("UTC").timestamp())

    ev = _http_json(f"https://gamma-api.polymarket.com/events?{urlencode({'slug': slug})}")
    if isinstance(ev, list):
        event = dict(ev[0] or {}) if ev else {}
    elif isinstance(ev, dict):
        arr = ev.get("events", ev.get("data", []))
        event = dict(arr[0] or {}) if isinstance(arr, list) and arr else dict(ev)
    else:
        event = {}
    markets = list((event or {}).get("markets", []) or [])

    series_map: dict[str, pd.Series] = {}
    for m in markets:
        mm = dict(m or {})
        token_ids = mm.get("clobTokenIds", [])
        if isinstance(token_ids, str):
            try:
                token_ids = json.loads(token_ids)
            except Exception:
                token_ids = [x.strip().strip("\"").strip("'") for x in token_ids.split(",") if str(x).strip()]
        if not isinstance(token_ids, list) or not token_ids:
            continue
        token_id = str(token_ids[0]).strip()  # outcomes[0] => Yes
        if not token_id:
            continue
        q = urlencode(
            {
                "market": token_id,
                "startTs": start_ts,
                "endTs": end_ts,
                "fidelity": int(max(1, fidelity)),
            }
        )
        raw = _http_json(f"https://clob.polymarket.com/prices-history?{q}")
        hist = list((raw or {}).get("history", []) or [])
        if not hist:
            continue
        rows = []
        for x in hist:
            try:
                t = pd.Timestamp(int(x.get("t")), unit="s", tz="UTC")
                p = float(x.get("p"))
            except Exception:
                continue
            if not np.isfinite(p):
                continue
            rows.append((t, float(np.clip(p, 0.0, 1.0))))
        if not rows:
            continue
        series_map[token_id] = pd.Series({t: p for t, p in rows}, dtype=float).sort_index()

    if not series_map:
        return pd.DataFrame(columns=["ts", "mode_prob"])

    idx = pd.Index(sorted(set().union(*[set(s.index) for s in series_map.values()])))
    wide = pd.DataFrame(index=idx)
    for k, s in series_map.items():
        wide[k] = s.reindex(idx).ffill()
    mode_prob = wide.max(axis=1, skipna=True)
    out = pd.DataFrame({"ts": mode_prob.index, "mode_prob": mode_prob.values})
    out = out.dropna(subset=["mode_prob"]).sort_values("ts").reset_index(drop=True)
    return out


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
        last_open = int(raw[-1][0])
        nxt = last_open + 60_000
        if nxt <= cursor:
            break
        cursor = nxt
    if not rows:
        return pd.DataFrame(columns=["ts", "close"])
    df = pd.DataFrame(rows).iloc[:, [0, 4]]
    df.columns = ["open_time", "close"]
    df["ts"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["ts", "close"]).sort_values("ts").drop_duplicates(subset=["ts"]).reset_index(drop=True)
    return df[["ts", "close"]]


def _rolling_z(arr: np.ndarray, win: int) -> np.ndarray:
    out = np.zeros(len(arr), dtype=float)
    for i in range(len(arr)):
        lo = max(0, i - win + 1)
        x = arr[lo : i + 1]
        mu = float(np.mean(x))
        sd = float(np.std(x))
        if sd <= 1e-12 or not np.isfinite(sd):
            out[i] = 0.0
        else:
            out[i] = float((arr[i] - mu) / sd)
    return out


def _trigger_table(pm: pd.DataFrame, z_window: int, z_slope_th: float, z_accel_th: float, confirm_ticks: int) -> pd.DataFrame:
    if len(pm) < 5:
        return pd.DataFrame(columns=["ts", "mode_prob", "z_slope", "z_accel", "slope_only", "accel_only", "both", "either"])
    out = pm.copy()
    p = out["mode_prob"].to_numpy(dtype=float)
    slope = np.diff(p, prepend=p[0])
    accel = np.diff(slope, prepend=slope[0])
    z_slope = _rolling_z(slope, z_window)
    z_accel = _rolling_z(accel, z_window)
    raw_slope = np.abs(z_slope) >= z_slope_th
    raw_accel = np.abs(z_accel) >= z_accel_th
    raw_both = raw_slope & raw_accel
    raw_either = raw_slope | raw_accel

    def _confirm(raw: np.ndarray) -> np.ndarray:
        outf = np.zeros_like(raw, dtype=bool)
        st = 0
        for i, v in enumerate(raw):
            st = st + 1 if v else 0
            outf[i] = st >= confirm_ticks
        return outf

    out["z_slope"] = z_slope
    out["z_accel"] = z_accel
    out["slope_only"] = _confirm(raw_slope)
    out["accel_only"] = _confirm(raw_accel)
    out["both"] = _confirm(raw_both)
    out["either"] = _confirm(raw_either)
    return out


@dataclass
class Trade:
    side: str
    open_ts: pd.Timestamp
    close_ts: pd.Timestamp
    open_price: float
    close_price: float
    realized_pct: float


def _load_trades_from_events(path: str, tz: str) -> list[Trade]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not isinstance(r, dict):
                continue
            rows.append(r)
    rows.sort(key=lambda x: str(x.get("ts", "")))

    trades: list[Trade] = []
    open_pos: dict | None = None
    for r in rows:
        ts = _parse_ts_kst(r.get("ts"))
        if ts is None:
            continue
        px = float(r.get("price", 0.0) or 0.0)
        frm = str(r.get("from", "") or "").upper()
        to = str(r.get("to", "") or "").upper()
        pnl_pct = float(r.get("pnl_pct", 0.0) or 0.0)

        if frm in {"LONG", "SHORT"} and open_pos and open_pos.get("side") == frm and px > 0.0:
            trades.append(
                Trade(
                    side=frm,
                    open_ts=open_pos["ts"],
                    close_ts=ts,
                    open_price=float(open_pos["price"]),
                    close_price=float(px),
                    realized_pct=float(pnl_pct),
                )
            )
            open_pos = None

        if to in {"LONG", "SHORT"} and px > 0.0:
            open_pos = {"side": to, "ts": ts, "price": float(px)}

    return trades


def _net_frac(side: str, entry: float, exitp: float, lev: float, fee: float, slip: float) -> float:
    if side == "LONG":
        entry_exec = entry * (1.0 + slip)
        exit_exec = exitp * (1.0 - slip)
        gross = (exit_exec - entry_exec) / max(entry_exec, 1e-12)
    else:
        entry_exec = entry * (1.0 - slip)
        exit_exec = exitp * (1.0 + slip)
        gross = (entry_exec - exit_exec) / max(abs(entry_exec), 1e-12)
    return float((gross * lev) - (2.0 * fee * lev))


def _estimate_leverage(tr: Trade, fee: float, slip: float) -> float:
    if tr.open_price <= 0.0 or tr.close_price <= 0.0:
        return 0.0
    if tr.side == "LONG":
        e = tr.open_price * (1.0 + slip)
        x = tr.close_price * (1.0 - slip)
        g = (x - e) / max(e, 1e-12)
    else:
        e = tr.open_price * (1.0 - slip)
        x = tr.close_price * (1.0 + slip)
        g = (e - x) / max(abs(e), 1e-12)
    denom = g - (2.0 * fee)
    actual = float(tr.realized_pct / 100.0)
    if abs(denom) <= 1e-10 or not np.isfinite(denom):
        return 0.0
    lev = actual / denom
    if not np.isfinite(lev):
        return 0.0
    return float(np.clip(lev, 0.0, 1.0))


def _price_asof(px: pd.DataFrame, ts: pd.Timestamp) -> float | None:
    if len(px) == 0:
        return None
    s = px.set_index("ts")["close"]
    try:
        v = s.asof(ts)
        if v is None or not np.isfinite(float(v)):
            return None
        return float(v)
    except Exception:
        return None


def _run_backtest(trades: list[Trade], trig_df: pd.DataFrame, rule_col: str, px: pd.DataFrame, fee: float, slip: float) -> dict:
    if len(trades) == 0:
        return {"trades": 0, "affected": 0, "base_sum_pct": 0.0, "new_sum_pct": 0.0, "delta_pct": 0.0, "base_wr": 0.0, "new_wr": 0.0}
    tset = trig_df[trig_df[rule_col]]["ts"]
    base = []
    new = []
    affected = 0
    for tr in trades:
        base_pct = float(tr.realized_pct)
        lev = _estimate_leverage(tr, fee=fee, slip=slip)
        trig_in = tset[(tset > tr.open_ts) & (tset <= tr.close_ts)]
        if len(trig_in) == 0:
            new_pct = base_pct
        else:
            t_exit = trig_in.iloc[0]
            px_exit = _price_asof(px, t_exit)
            if px_exit is None:
                new_pct = base_pct
            else:
                new_pct = float(_net_frac(tr.side, tr.open_price, px_exit, lev, fee, slip) * 100.0)
                affected += 1
        base.append(base_pct)
        new.append(new_pct)
    base_arr = np.array(base, dtype=float)
    new_arr = np.array(new, dtype=float)
    return {
        "trades": int(len(trades)),
        "affected": int(affected),
        "base_sum_pct": float(np.sum(base_arr)),
        "new_sum_pct": float(np.sum(new_arr)),
        "delta_pct": float(np.sum(new_arr) - np.sum(base_arr)),
        "base_wr": float(np.mean(base_arr > 0) * 100.0),
        "new_wr": float(np.mean(new_arr > 0) * 100.0),
    }


def main():
    ap = argparse.ArgumentParser(description="Backtest polymarket emergency-exit modes (exit only; no reduce).")
    ap.add_argument("--slug", default="ethereum-price-on-april-19")
    ap.add_argument("--tz", default="Asia/Seoul")
    ap.add_argument("--fidelity", type=int, default=1)
    ap.add_argument("--z-window", type=int, default=120)
    ap.add_argument("--z-slope-th", type=float, default=2.5)
    ap.add_argument("--z-accel-th", type=float, default=2.2)
    ap.add_argument("--confirm-ticks", type=int, default=2)
    ap.add_argument("--events-path", default="data/live/dashboard_events.jsonl")
    ap.add_argument("--fee", type=float, default=0.0005)
    ap.add_argument("--slip", type=float, default=0.0002)
    args = ap.parse_args()

    pm = _fetch_polymarket_mode_series(args.slug, args.tz, fidelity=args.fidelity)
    if len(pm) < 10:
        print("insufficient polymarket samples")
        return
    trig = _trigger_table(pm, z_window=int(args.z_window), z_slope_th=float(args.z_slope_th), z_accel_th=float(args.z_accel_th), confirm_ticks=int(args.confirm_ticks))

    trades = _load_trades_from_events(args.events_path, args.tz)
    if not trades:
        print("no trades parsed from events")
        return

    tmin = min(t.open_ts for t in trades) - pd.Timedelta(minutes=5)
    tmax = max(t.close_ts for t in trades) + pd.Timedelta(minutes=5)
    px = _fetch_binance_1m(tmin, tmax)

    # yesterday + today only
    now_local = pd.Timestamp.now(tz=args.tz)
    yday = (now_local - pd.Timedelta(days=1)).date()
    tday = now_local.date()
    selected = []
    for tr in trades:
        d = tr.close_ts.tz_convert(args.tz).date()
        if d in {yday, tday}:
            selected.append(tr)

    print("=== Backtest: Polymarket Exit Modes (Exit only) ===")
    print(f"slug={args.slug} samples={len(pm)} trades_selected={len(selected)} (close date in {yday}, {tday})")
    print(f"params: z_window={args.z_window} z_slope_th={args.z_slope_th} z_accel_th={args.z_accel_th} confirm={args.confirm_ticks} fidelity={args.fidelity}m")

    rules = ["slope_only", "accel_only", "both", "either"]
    for r in rules:
        m = _run_backtest(selected, trig, r, px, fee=float(args.fee), slip=float(args.slip))
        print(
            f"[{r:10s}] trades={m['trades']:3d} affected={m['affected']:3d} "
            f"base={m['base_sum_pct']:+8.3f}% -> new={m['new_sum_pct']:+8.3f}% "
            f"delta={m['delta_pct']:+7.3f}%p | WR {m['base_wr']:.1f}% -> {m['new_wr']:.1f}%"
        )


if __name__ == "__main__":
    main()

