#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
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


def _fetch_polymarket_features(slug: str, tz: str, fidelity: int = 1) -> pd.DataFrame:
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
    if not markets:
        return pd.DataFrame(columns=["ts", "mode_prob", "weighted_target", "delta_1m"])

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
        token_id = str(token_ids[0]).strip()  # outcomes[0] = Yes
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
            continue
        s = pd.Series({t: p for t, p in vals}, dtype=float).sort_index()
        series_by_market.append((float(strike), s))

    if not series_by_market:
        return pd.DataFrame(columns=["ts", "mode_prob", "weighted_target", "delta_1m"])

    idx = pd.Index(sorted(set().union(*[set(s.index) for _, s in series_by_market])))
    probs = pd.DataFrame(index=idx)
    strikes = []
    for i, (strike, s) in enumerate(series_by_market):
        cname = f"m{i}"
        probs[cname] = s.reindex(idx).ffill()
        strikes.append(float(strike))
    pvals = probs.to_numpy(dtype=float)
    strike_arr = np.array(strikes, dtype=float)
    row_sum = np.nansum(pvals, axis=1)
    norm = np.where(row_sum > 1e-12, row_sum, np.nan)
    w = pvals / norm[:, None]
    weighted_target = np.nansum(w * strike_arr[None, :], axis=1)
    mode_prob = np.nanmax(pvals, axis=1)
    out = pd.DataFrame({"ts": idx, "mode_prob": mode_prob, "weighted_target": weighted_target})
    out = out.dropna(subset=["mode_prob", "weighted_target"]).sort_values("ts").reset_index(drop=True)
    # 1분 변화량(%p): mode_prob(t) - mode_prob(t-1m)
    out = out.set_index("ts")
    out["mode_prob_1m_prev"] = out["mode_prob"].shift(1)
    out["delta_1m"] = out["mode_prob"] - out["mode_prob_1m_prev"]
    out = out.dropna(subset=["delta_1m"]).reset_index()
    return out[["ts", "mode_prob", "weighted_target", "delta_1m"]]


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
    df["ts"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["ts", "close"]).drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return df[["ts", "close"]]


@dataclass
class Trade:
    side: str
    open_ts: pd.Timestamp
    close_ts: pd.Timestamp
    open_price: float
    close_price: float
    realized_pct: float


def _load_trades(events_path: str) -> list[Trade]:
    rows: list[dict] = []
    with open(events_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if isinstance(r, dict):
                rows.append(r)
    rows.sort(key=lambda x: str(x.get("ts", "")))

    out: list[Trade] = []
    open_pos: dict | None = None
    for r in rows:
        ts = _parse_ts_kst(r.get("ts"))
        if ts is None:
            continue
        px = float(r.get("price", 0.0) or 0.0)
        frm = str(r.get("from", "") or "").upper()
        to = str(r.get("to", "") or "").upper()
        pnl = float(r.get("pnl_pct", 0.0) or 0.0)
        if frm in {"LONG", "SHORT"} and open_pos and open_pos.get("side") == frm and px > 0:
            out.append(
                Trade(
                    side=frm,
                    open_ts=open_pos["ts"],
                    close_ts=ts,
                    open_price=float(open_pos["price"]),
                    close_price=px,
                    realized_pct=pnl,
                )
            )
            open_pos = None
        if to in {"LONG", "SHORT"} and px > 0:
            open_pos = {"side": to, "ts": ts, "price": px}
    return out


def _net_frac(side: str, entry: float, exitp: float, lev: float, fee: float, slip: float) -> float:
    if side == "LONG":
        en = entry * (1.0 + slip)
        ex = exitp * (1.0 - slip)
        gross = (ex - en) / max(en, 1e-12)
    else:
        en = entry * (1.0 - slip)
        ex = exitp * (1.0 + slip)
        gross = (en - ex) / max(abs(en), 1e-12)
    return float((gross * lev) - (2.0 * fee * lev))


def _est_lev(tr: Trade, fee: float, slip: float) -> float:
    if tr.side == "LONG":
        en = tr.open_price * (1.0 + slip)
        ex = tr.close_price * (1.0 - slip)
        gross = (ex - en) / max(en, 1e-12)
    else:
        en = tr.open_price * (1.0 - slip)
        ex = tr.close_price * (1.0 + slip)
        gross = (en - ex) / max(abs(en), 1e-12)
    denom = gross - (2.0 * fee)
    if abs(denom) <= 1e-10:
        return 0.0
    lev = (tr.realized_pct / 100.0) / denom
    if not np.isfinite(lev):
        return 0.0
    return float(np.clip(lev, 0.0, 1.0))


def _asof_price(px: pd.DataFrame, ts: pd.Timestamp) -> float | None:
    if len(px) == 0:
        return None
    s = px.set_index("ts")["close"]
    try:
        v = s.asof(ts)
        if v is None:
            return None
        fv = float(v)
        if not np.isfinite(fv):
            return None
        return fv
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Backtest polymarket 1m shock exit based on entry-vs-target direction.")
    ap.add_argument("--slug", default="ethereum-price-on-april-19")
    ap.add_argument("--events-path", default="data/live/dashboard_events.jsonl")
    ap.add_argument("--tz", default="Asia/Seoul")
    ap.add_argument("--fidelity", type=int, default=1)
    ap.add_argument("--shock-th", type=float, default=0.03, help="absolute 1m mode_prob change threshold, e.g. 0.03=3%%p")
    ap.add_argument("--fee", type=float, default=0.0005)
    ap.add_argument("--slip", type=float, default=0.0002)
    ap.add_argument("--print-events", action="store_true", help="Print all triggered shock events and decisions.")
    args = ap.parse_args()

    pm = _fetch_polymarket_features(args.slug, args.tz, fidelity=int(max(1, args.fidelity)))
    if len(pm) == 0:
        print("No polymarket feature samples.")
        return
    trades = _load_trades(args.events_path)
    if len(trades) == 0:
        print("No trades loaded from events.")
        return

    now_local = pd.Timestamp.now(tz=args.tz)
    d0 = (now_local - pd.Timedelta(days=1)).date()
    d1 = now_local.date()
    trades = [t for t in trades if t.close_ts.tz_convert(args.tz).date() in {d0, d1}]
    if len(trades) == 0:
        print("No trades in yesterday/today window.")
        return

    tmin = min(t.open_ts for t in trades) - pd.Timedelta(minutes=5)
    tmax = max(t.close_ts for t in trades) + pd.Timedelta(minutes=5)
    px = _fetch_binance_1m(tmin, tmax)

    pm = pm.sort_values("ts").reset_index(drop=True)
    pm_t = pm.set_index("ts")
    shock = pm[np.abs(pm["delta_1m"]) >= float(max(0.0, args.shock_th))].copy()

    base_sum = 0.0
    new_sum = 0.0
    base_wins = 0
    new_wins = 0
    affected = 0
    keep_on_shock = 0
    exit_on_shock = 0
    event_rows: list[dict] = []
    for tr in trades:
        base_pct = float(tr.realized_pct)
        lev = _est_lev(tr, fee=float(args.fee), slip=float(args.slip))
        cand = shock[(shock["ts"] > tr.open_ts) & (shock["ts"] <= tr.close_ts)]
        new_pct = base_pct
        if len(cand) > 0:
            for _, row in cand.iterrows():
                tgt = float(row["weighted_target"])
                favorable = (tgt > tr.open_price) if tr.side == "LONG" else (tgt < tr.open_price)
                if favorable:
                    keep_on_shock += 1
                    event_rows.append(
                        {
                            "decision": "HOLD",
                            "ts": row["ts"],
                            "side": tr.side,
                            "entry_price": tr.open_price,
                            "target": tgt,
                            "delta_1m_pctp": float(row["delta_1m"]) * 100.0,
                            "trade_open_ts": tr.open_ts,
                            "trade_close_ts": tr.close_ts,
                        }
                    )
                    continue
                exit_ts = row["ts"]
                exit_px = _asof_price(px, exit_ts)
                if exit_px is not None:
                    new_pct = _net_frac(tr.side, tr.open_price, exit_px, lev, fee=float(args.fee), slip=float(args.slip)) * 100.0
                    affected += 1
                    exit_on_shock += 1
                    event_rows.append(
                        {
                            "decision": "EXIT",
                            "ts": exit_ts,
                            "side": tr.side,
                            "entry_price": tr.open_price,
                            "target": tgt,
                            "delta_1m_pctp": float(row["delta_1m"]) * 100.0,
                            "exit_price": exit_px,
                            "trade_open_ts": tr.open_ts,
                            "trade_close_ts": tr.close_ts,
                            "new_trade_pnl_pct": float(new_pct),
                        }
                    )
                break
        base_sum += base_pct
        new_sum += new_pct
        base_wins += 1 if base_pct > 0 else 0
        new_wins += 1 if new_pct > 0 else 0

    n = len(trades)
    print("=== Backtest: Polymarket 1m Shock Exit (Entry-vs-Target) ===")
    print(f"slug={args.slug} shock_th={args.shock_th*100:.2f}%p fidelity={args.fidelity}m trades={n} window=({d0}, {d1})")
    print(f"shock_samples={len(shock)} feature_samples={len(pm)}")
    print(f"base_sum={base_sum:+.3f}% -> new_sum={new_sum:+.3f}%  delta={new_sum-base_sum:+.3f}%p")
    print(f"win_rate={100.0*base_wins/n:.1f}% -> {100.0*new_wins/n:.1f}%")
    print(f"affected_trades={affected}/{n}  keep_on_shock_checks={keep_on_shock}  exit_on_shock={exit_on_shock}")
    if args.print_events:
        print("\n--- Triggered Events ---")
        if not event_rows:
            print("(none)")
        else:
            for ev in sorted(event_rows, key=lambda x: x["ts"]):
                ts_kst = pd.Timestamp(ev["ts"]).tz_convert(args.tz).strftime("%Y-%m-%d %H:%M:%S %Z")
                open_kst = pd.Timestamp(ev["trade_open_ts"]).tz_convert(args.tz).strftime("%Y-%m-%d %H:%M:%S")
                close_kst = pd.Timestamp(ev["trade_close_ts"]).tz_convert(args.tz).strftime("%Y-%m-%d %H:%M:%S")
                msg = (
                    f"{ts_kst} | {ev['decision']:<4} | side={ev['side']:<5} "
                    f"| d1m={ev['delta_1m_pctp']:+.2f}%p | entry={ev['entry_price']:.2f} | target={ev['target']:.2f} "
                    f"| trade=[{open_kst} -> {close_kst}]"
                )
                if ev["decision"] == "EXIT":
                    msg += f" | exit_px={ev['exit_price']:.2f} | new_pnl={ev['new_trade_pnl_pct']:+.3f}%"
                print(msg)


if __name__ == "__main__":
    main()
