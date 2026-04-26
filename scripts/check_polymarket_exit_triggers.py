#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


@dataclass
class TriggerConfig:
    reduce_z_slope: float = 2.5
    reduce_z_accel: float = 2.5
    exit_z_slope: float = 3.5
    exit_z_accel: float = 3.0
    confirm_ticks: int = 3
    z_window: int = 180  # 10s bars => 30 min


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


def _read_jsonl(path: str) -> list[dict]:
    out: list[dict] = []
    if not path or not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    out.append(row)
            except Exception:
                continue
    return out


def _parse_ts(v) -> pd.Timestamp | None:
    try:
        ts = pd.Timestamp(v)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert("UTC")
    except Exception:
        return None


def _fetch_series_from_polymarket_api(
    slug: str,
    tz: str,
    gamma_url: str = "https://gamma-api.polymarket.com/events",
    clob_history_url: str = "https://clob.polymarket.com/prices-history",
    fidelity_min: int = 1,
) -> pd.DataFrame:
    now_local = pd.Timestamp.now(tz=tz)
    start_local = (now_local - pd.Timedelta(days=1)).normalize()
    end_local = (now_local.normalize() + pd.Timedelta(days=1))
    start_ts = int(start_local.tz_convert("UTC").timestamp())
    end_ts = int(end_local.tz_convert("UTC").timestamp())

    ev_url = f"{gamma_url}?{urlencode({'slug': slug})}"
    ev_raw = _http_json(ev_url)
    if isinstance(ev_raw, list):
        ev = dict(ev_raw[0] or {}) if ev_raw else {}
    elif isinstance(ev_raw, dict):
        arr = ev_raw.get("events", ev_raw.get("data", []))
        ev = dict(arr[0] or {}) if isinstance(arr, list) and arr else dict(ev_raw)
    else:
        ev = {}
    markets = list((ev or {}).get("markets", []) or [])
    if not markets:
        return pd.DataFrame(columns=["ts", "mode_prob"])

    market_series: dict[str, pd.Series] = {}
    for m in markets:
        mk = dict(m or {})
        raw_ids = mk.get("clobTokenIds", [])
        if isinstance(raw_ids, str):
            try:
                raw_ids = json.loads(raw_ids)
            except Exception:
                raw_ids = [x.strip().strip("\"").strip("'") for x in raw_ids.split(",") if str(x).strip()]
        if not isinstance(raw_ids, list) or not raw_ids:
            continue
        token_id = str(raw_ids[0]).strip()  # outcomes[0] = Yes
        if not token_id:
            continue

        q = urlencode({
            "market": token_id,
            "startTs": start_ts,
            "endTs": end_ts,
            "fidelity": int(max(1, fidelity_min)),
        })
        h_url = f"{clob_history_url}?{q}"
        try:
            hist_raw = _http_json(h_url)
            hist = list((hist_raw or {}).get("history", []) or [])
        except Exception:
            continue
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
        s = pd.Series({t: p for t, p in rows}, dtype=float).sort_index()
        market_series[token_id] = s

    if not market_series:
        return pd.DataFrame(columns=["ts", "mode_prob"])

    idx = pd.Index(sorted(set().union(*[set(s.index) for s in market_series.values()])))
    wide = pd.DataFrame(index=idx)
    for k, s in market_series.items():
        wide[k] = s.reindex(idx).ffill()
    mode_prob = wide.max(axis=1, skipna=True)
    out = pd.DataFrame({"ts": mode_prob.index, "mode_prob": mode_prob.values})
    out = out.dropna(subset=["mode_prob"]).sort_values("ts").reset_index(drop=True)
    return out


def _build_series(history_rows: Iterable[dict], include_state_path: str | None = None) -> pd.DataFrame:
    rows: list[dict] = []
    for r in history_rows:
        ts = _parse_ts(r.get("updated_at") or r.get("ts"))
        p = r.get("mode_prob")
        if ts is None:
            continue
        try:
            p = float(p)
        except Exception:
            continue
        if not np.isfinite(p):
            continue
        rows.append({"ts": ts, "mode_prob": float(np.clip(p, 0.0, 1.0))})

    # fallback: current dashboard state polymarket snapshot
    if include_state_path and os.path.exists(include_state_path):
        try:
            st = json.load(open(include_state_path, "r", encoding="utf-8"))
            pm = dict((st or {}).get("polymarket", {}) or {})
            ts = _parse_ts(pm.get("updated_at") or (st or {}).get("shadow_updated_at") or (st or {}).get("updated_at"))
            p = pm.get("mode_prob")
            if ts is not None and p is not None:
                p = float(p)
                if np.isfinite(p):
                    rows.append({"ts": ts, "mode_prob": float(np.clip(p, 0.0, 1.0))})
        except Exception:
            pass

    if not rows:
        return pd.DataFrame(columns=["ts", "mode_prob"])
    df = pd.DataFrame(rows).drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return df


def _rolling_z(arr: np.ndarray, win: int) -> np.ndarray:
    n = len(arr)
    out = np.zeros(n, dtype=float)
    if n == 0:
        return out
    for i in range(n):
        lo = max(0, i - win + 1)
        x = arr[lo:i + 1]
        mu = float(np.mean(x))
        sd = float(np.std(x))
        if sd <= 1e-12 or not np.isfinite(sd):
            out[i] = 0.0
        else:
            out[i] = float((arr[i] - mu) / sd)
    return out


def _scan_triggers(df: pd.DataFrame, cfg: TriggerConfig) -> pd.DataFrame:
    if len(df) < 5:
        return pd.DataFrame(columns=["ts", "mode_prob", "slope", "accel", "z_slope", "z_accel", "reduce", "exit"])

    p = df["mode_prob"].to_numpy(dtype=float)
    slope = np.diff(p, prepend=p[0])
    accel = np.diff(slope, prepend=slope[0])
    z_slope = _rolling_z(slope, cfg.z_window)
    z_accel = _rolling_z(accel, cfg.z_window)

    reduce_raw = (np.abs(z_slope) >= cfg.reduce_z_slope) | (np.abs(z_accel) >= cfg.reduce_z_accel)
    exit_raw = (np.abs(z_slope) >= cfg.exit_z_slope) & (np.abs(z_accel) >= cfg.exit_z_accel)

    reduce = np.zeros_like(reduce_raw, dtype=bool)
    exit_ = np.zeros_like(exit_raw, dtype=bool)
    streak_r = 0
    streak_e = 0
    for i in range(len(p)):
        streak_r = (streak_r + 1) if reduce_raw[i] else 0
        streak_e = (streak_e + 1) if exit_raw[i] else 0
        reduce[i] = streak_r >= cfg.confirm_ticks
        exit_[i] = streak_e >= cfg.confirm_ticks

    out = df.copy()
    out["slope"] = slope
    out["accel"] = accel
    out["z_slope"] = z_slope
    out["z_accel"] = z_accel
    out["reduce"] = reduce
    out["exit"] = exit_
    return out


def main():
    parser = argparse.ArgumentParser(description="Check polymarket slope/accel emergency-exit trigger timestamps.")
    parser.add_argument("--history-jsonl", default="data/live/polymarket_history.jsonl")
    parser.add_argument("--state-json", default="data/live/dashboard_state.json")
    parser.add_argument("--from-api", action="store_true", help="Fetch yesterday/today time-series directly from Polymarket API.")
    parser.add_argument("--slug", default="ethereum-price-on-april-19")
    parser.add_argument("--gamma-url", default="https://gamma-api.polymarket.com/events")
    parser.add_argument("--clob-history-url", default="https://clob.polymarket.com/prices-history")
    parser.add_argument("--fidelity", type=int, default=1)
    parser.add_argument("--tz", default="Asia/Seoul")
    parser.add_argument("--show-reduce", action="store_true", help="Print REDUCE trigger timestamps.")
    parser.add_argument("--max-print", type=int, default=20)
    parser.add_argument("--confirm-ticks", type=int, default=3)
    parser.add_argument("--z-window", type=int, default=180)
    parser.add_argument("--reduce-z-slope", type=float, default=2.5)
    parser.add_argument("--reduce-z-accel", type=float, default=2.5)
    parser.add_argument("--exit-z-slope", type=float, default=3.5)
    parser.add_argument("--exit-z-accel", type=float, default=3.0)
    args = parser.parse_args()

    cfg = TriggerConfig(
        reduce_z_slope=float(args.reduce_z_slope),
        reduce_z_accel=float(args.reduce_z_accel),
        exit_z_slope=float(args.exit_z_slope),
        exit_z_accel=float(args.exit_z_accel),
        confirm_ticks=int(max(1, args.confirm_ticks)),
        z_window=int(max(30, args.z_window)),
    )

    hist = _read_jsonl(args.history_jsonl)
    series = _build_series(hist, include_state_path=args.state_json)
    source = "local_jsonl+state"
    if args.from_api:
        try:
            api_df = _fetch_series_from_polymarket_api(
                slug=str(args.slug),
                tz=str(args.tz),
                gamma_url=str(args.gamma_url),
                clob_history_url=str(args.clob_history_url),
                fidelity_min=int(max(1, args.fidelity)),
            )
            if len(api_df) > 0:
                series = api_df
                source = "polymarket_api"
        except Exception as e:
            print(f"api_fetch_error: {e}")
    analyzed = _scan_triggers(series, cfg)

    now_local = pd.Timestamp.now(tz=args.tz)
    today = now_local.date()
    yesterday = (now_local - pd.Timedelta(days=1)).date()

    print("=== Polymarket Exit Trigger Audit ===")
    print(f"source: {source}")
    print(f"history_jsonl: {args.history_jsonl} exists={os.path.exists(args.history_jsonl)} rows={len(hist)}")
    print(f"state_json: {args.state_json} exists={os.path.exists(args.state_json)}")
    print(f"samples_total: {len(series)}")
    print(f"config: reduce(|z_slope|>={cfg.reduce_z_slope} or |z_accel|>={cfg.reduce_z_accel}), "
          f"exit(|z_slope|>={cfg.exit_z_slope} and |z_accel|>={cfg.exit_z_accel}), "
          f"confirm_ticks={cfg.confirm_ticks}, z_window={cfg.z_window}")

    if len(analyzed) == 0:
        print("result: insufficient samples (need >=5).")
        return

    loc = analyzed.copy()
    loc["ts_local"] = loc["ts"].dt.tz_convert(args.tz)
    loc["date_local"] = loc["ts_local"].dt.date

    for d in [yesterday, today]:
        sub = loc[loc["date_local"] == d]
        sub_exit = sub[sub["exit"]]
        sub_reduce = sub[sub["reduce"]]
        print(f"\n[{d}] total={len(sub)} reduce_hits={len(sub_reduce)} exit_hits={len(sub_exit)}")
        if args.show_reduce:
            if len(sub_reduce) == 0:
                print("  reduce_timestamps: (none)")
            else:
                for _, r in sub_reduce.head(max(1, int(args.max_print))).iterrows():
                    ts = r["ts_local"].strftime("%Y-%m-%d %H:%M:%S %Z")
                    print(f"  REDUCE {ts} | p={r['mode_prob']:.4f} slope={r['slope']:+.5f} accel={r['accel']:+.5f} "
                          f"z_slope={r['z_slope']:+.2f} z_accel={r['z_accel']:+.2f}")
        if len(sub_exit) == 0:
            print("  exit_timestamps: (none)")
        else:
            for _, r in sub_exit.iterrows():
                ts = r["ts_local"].strftime("%Y-%m-%d %H:%M:%S %Z")
                print(f"  {ts} | p={r['mode_prob']:.4f} slope={r['slope']:+.5f} accel={r['accel']:+.5f} "
                      f"z_slope={r['z_slope']:+.2f} z_accel={r['z_accel']:+.2f}")


if __name__ == "__main__":
    main()
