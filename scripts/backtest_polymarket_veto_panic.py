#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

try:
    from scripts.backtest_polymarket_shock_entry_exit import (
        Trade,
        _asof_price,
        _est_lev,
        _fetch_binance_1m,
        _load_trades,
        _net_frac,
    )
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from scripts.backtest_polymarket_shock_entry_exit import (
        Trade,
        _asof_price,
        _est_lev,
        _fetch_binance_1m,
        _load_trades,
        _net_frac,
    )


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


@dataclass
class LabelInfo:
    label: str
    lo: float | None
    hi: float | None
    center: float | None


def _parse_label_range(label: str) -> LabelInfo:
    s = str(label or "")
    m_between = re.search(
        r"between\s*\$?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?)\s*and\s*\$?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?)",
        s,
        flags=re.IGNORECASE,
    )
    if m_between:
        a = float(m_between.group(1).replace(",", ""))
        b = float(m_between.group(2).replace(",", ""))
        lo, hi = sorted((a, b))
        return LabelInfo(label=s, lo=lo, hi=hi, center=(lo + hi) * 0.5)

    m_less = re.search(
        r"(less than|below|under|at most)\s*\$?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?)",
        s,
        flags=re.IGNORECASE,
    )
    if m_less:
        x = float(m_less.group(2).replace(",", ""))
        return LabelInfo(label=s, lo=None, hi=x, center=x - 100.0)

    m_more = re.search(
        r"(greater than|above|over|at least)\s*\$?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?)",
        s,
        flags=re.IGNORECASE,
    )
    if m_more:
        x = float(m_more.group(2).replace(",", ""))
        return LabelInfo(label=s, lo=x, hi=None, center=x + 100.0)

    s1 = _parse_strike(s)
    if s1 is not None:
        return LabelInfo(label=s, lo=None, hi=None, center=s1)
    return LabelInfo(label=s, lo=None, hi=None, center=None)


def _parse_slug_date(slug: str) -> date | None:
    m = re.search(r"-on-([a-z]+)-([0-9]{1,2})$", str(slug))
    if not m:
        return None
    mon_s = m.group(1).lower()
    day = int(m.group(2))
    mons = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }
    month = mons.get(mon_s)
    if month is None:
        return None
    return date(2026, month, day)


def _fetch_slug_markets(slug: str, tz: str, fidelity: int = 1) -> tuple[pd.DataFrame, dict[str, LabelInfo]]:
    slug_day = _parse_slug_date(slug)
    if slug_day is not None:
        start_local = pd.Timestamp(slug_day, tz=tz).normalize()
        end_local = start_local + pd.Timedelta(days=1)
    else:
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

    mkts = list((event or {}).get("markets", []) or [])
    if not mkts:
        return pd.DataFrame(), {}

    parts: list[pd.DataFrame] = []
    meta: dict[str, LabelInfo] = {}
    for m in mkts:
        mm = dict(m or {})
        label = str(mm.get("question", mm.get("title", mm.get("name", ""))) or "")
        strike = _parse_strike(label)
        if strike is None:
            continue
        token_ids = mm.get("clobTokenIds", [])
        if isinstance(token_ids, str):
            try:
                token_ids = json.loads(token_ids)
            except Exception:
                token_ids = [x.strip().strip('"').strip("'") for x in token_ids.split(",") if str(x).strip()]
        if not isinstance(token_ids, list) or not token_ids:
            continue
        token_id = str(token_ids[0]).strip()
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
        vals: list[tuple[pd.Timestamp, float]] = []
        for x in hist:
            try:
                t = pd.Timestamp(int(x.get("t")), unit="s", tz="UTC")
                p = float(x.get("p"))
            except Exception:
                continue
            if np.isfinite(p):
                vals.append((t, float(np.clip(p, 0.0, 1.0))))
        if not vals:
            continue
        df = pd.DataFrame(vals, columns=["ts", "prob"]).drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
        df["label"] = label
        parts.append(df)
        meta[label] = _parse_label_range(label)

    if not parts:
        return pd.DataFrame(), {}
    all_df = pd.concat(parts, ignore_index=True)
    return all_df, meta


def _build_slug_features(slug: str, tz: str) -> tuple[pd.DataFrame, dict[str, LabelInfo]]:
    long_df, meta = _fetch_slug_markets(slug, tz=tz, fidelity=1)
    if len(long_df) == 0:
        return pd.DataFrame(), {}

    idx = pd.Index(sorted(set(long_df["ts"].tolist())))
    labels = sorted(long_df["label"].unique().tolist())

    wide = pd.DataFrame(index=idx)
    for lb in labels:
        s = (
            long_df.loc[long_df["label"] == lb, ["ts", "prob"]]
            .drop_duplicates(subset=["ts"])
            .set_index("ts")["prob"]
            .reindex(idx)
            .ffill()
            .fillna(0.0)
        )
        wide[lb] = s

    wide = wide.sort_index()
    mode_label = wide.idxmax(axis=1)
    mode_prob = wide.max(axis=1)

    centers = np.array([meta.get(lb, LabelInfo(lb, None, None, None)).center or np.nan for lb in labels], dtype=float)
    arr = wide.to_numpy(dtype=float)
    row_sum = np.clip(arr.sum(axis=1), 1e-12, None)
    w = arr / row_sum[:, None]
    weighted = np.nansum(w * centers[None, :], axis=1)

    # 1m/5m change per label
    chg1 = wide - wide.shift(1)
    chg5 = wide - wide.shift(5)
    # proxy liquidity: changed-label count vs rolling 1h mean
    changed = (wide.diff().abs() > 1e-12).sum(axis=1).astype(float)
    liq_mean_1h = changed.rolling(60, min_periods=10).mean()
    liq_void = changed <= (liq_mean_1h * 0.30)

    out = pd.DataFrame(
        {
            "ts": wide.index,
            "slug": slug,
            "mode_label": mode_label.to_numpy(),
            "mode_prob": mode_prob.to_numpy(dtype=float),
            "weighted_target": weighted,
            "liq_changed_labels": changed.to_numpy(dtype=float),
            "liq_mean_1h": liq_mean_1h.to_numpy(dtype=float),
            "liq_void": liq_void.fillna(False).to_numpy(dtype=bool),
        }
    )
    # attach dict-like columns for label probs and changes
    out["prob_map"] = [dict(zip(labels, row)) for row in wide.to_numpy(dtype=float)]
    out["d1_map"] = [dict(zip(labels, row)) for row in chg1.fillna(0.0).to_numpy(dtype=float)]
    out["d5_map"] = [dict(zip(labels, row)) for row in chg5.fillna(0.0).to_numpy(dtype=float)]
    return out, meta


def _pick_labels(side: str, entry: float, labels_meta: dict[str, LabelInfo]) -> tuple[str | None, str | None]:
    # favorable label: LONG -> near/above entry, SHORT -> near/below entry
    # opposite label: reverse side
    candidates = []
    for lb, info in labels_meta.items():
        c = info.center
        if c is None or not np.isfinite(c):
            continue
        candidates.append((lb, float(c), info.lo, info.hi))
    if not candidates:
        return None, None

    # prefer containing interval
    contain = []
    for lb, c, lo, hi in candidates:
        if lo is not None and hi is not None and lo <= entry <= hi:
            contain.append((lb, c))
    if contain:
        contain = sorted(contain, key=lambda x: abs(x[1] - entry))
        neutral = contain[0][0]
    else:
        neutral = sorted(candidates, key=lambda x: abs(x[1] - entry))[0][0]

    if side.upper() == "LONG":
        fav_pool = sorted(candidates, key=lambda x: (0 if x[1] >= entry else 1, abs(x[1] - entry)))
        opp_pool = sorted(candidates, key=lambda x: (0 if x[1] < entry else 1, abs(x[1] - entry)))
    else:
        fav_pool = sorted(candidates, key=lambda x: (0 if x[1] <= entry else 1, abs(x[1] - entry)))
        opp_pool = sorted(candidates, key=lambda x: (0 if x[1] > entry else 1, abs(x[1] - entry)))

    fav = fav_pool[0][0] if fav_pool else neutral
    opp = opp_pool[0][0] if opp_pool else neutral
    return fav, opp


def _required_prob(alpha: float, k_decay: float, hours_to_expiry: float) -> float:
    h = max(0.0, float(hours_to_expiry))
    return float(alpha + (1.0 - alpha) * math.exp(-k_decay * h))


def _choose_slug_for_ts(ts_utc: pd.Timestamp, slug_feats: dict[str, pd.DataFrame], tz: str) -> str | None:
    kst_day = ts_utc.tz_convert(tz).date()
    for slug in slug_feats.keys():
        d = _parse_slug_date(slug)
        if d == kst_day:
            return slug
    # fallback nearest covered slug
    best = None
    best_dist = None
    for slug, df in slug_feats.items():
        if len(df) == 0:
            continue
        lo = df["ts"].min()
        hi = df["ts"].max()
        if lo <= ts_utc <= hi:
            return slug
        dist = min(abs((ts_utc - lo).total_seconds()), abs((ts_utc - hi).total_seconds()))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = slug
    return best


def run_backtest(args):
    slug_feats: dict[str, pd.DataFrame] = {}
    slug_meta: dict[str, dict[str, LabelInfo]] = {}
    for d in pd.date_range(args.start_date, args.end_date, freq="D", tz=args.tz):
        slug = f"ethereum-price-on-{d.strftime('%B').lower()}-{d.day}"
        try:
            feat, meta = _build_slug_features(slug, tz=args.tz)
        except Exception:
            feat, meta = pd.DataFrame(), {}
        if len(feat):
            slug_feats[slug] = feat
            slug_meta[slug] = meta

    trades = _load_trades(args.events_path)
    start_utc = pd.Timestamp(args.start_date, tz=args.tz).tz_convert("UTC")
    end_utc = (pd.Timestamp(args.end_date, tz=args.tz) + pd.Timedelta(hours=23, minutes=59, seconds=59)).tz_convert("UTC")
    trades = [t for t in trades if start_utc <= t.close_ts <= end_utc]
    if len(trades) == 0:
        print("No trades in date window.")
        return

    tmin = min(t.open_ts for t in trades) - pd.Timedelta(minutes=5)
    tmax = max(t.close_ts for t in trades) + pd.Timedelta(minutes=5)
    px = _fetch_binance_1m(tmin, tmax)

    base_sum = 0.0
    new_sum = 0.0
    base_wins = 0
    new_wins = 0

    vetoed = 0
    panic_exits = 0
    events = []

    for tr in trades:
        base_pct = float(tr.realized_pct)
        lev = _est_lev(tr, fee=float(args.fee), slip=float(args.slip))
        base_sum += base_pct
        base_wins += 1 if base_pct > 0 else 0

        slug = _choose_slug_for_ts(tr.open_ts, slug_feats, args.tz)
        if slug is None:
            new_sum += base_pct
            new_wins += 1 if base_pct > 0 else 0
            continue

        feat = slug_feats[slug]
        meta = slug_meta[slug]
        if len(feat) == 0:
            new_sum += base_pct
            new_wins += 1 if base_pct > 0 else 0
            continue

        # pick snapshot at entry
        snap0 = feat[feat["ts"] <= tr.open_ts]
        if len(snap0) == 0:
            new_sum += base_pct
            new_wins += 1 if base_pct > 0 else 0
            continue
        row0 = snap0.iloc[-1]
        pmap0 = dict(row0["prob_map"])
        d5map0 = dict(row0["d5_map"])

        fav_label, opp_label = _pick_labels(tr.side, tr.open_price, meta)
        if not fav_label or not opp_label:
            new_sum += base_pct
            new_wins += 1 if base_pct > 0 else 0
            continue

        p_fav = float(pmap0.get(fav_label, 0.0))
        d5_fav = float(d5map0.get(fav_label, 0.0))

        # expiry assumption: local date end 23:59:59 for slug day
        slug_day = _parse_slug_date(slug)
        if slug_day is None:
            expiry_local = tr.open_ts.tz_convert(args.tz).normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
        else:
            expiry_local = pd.Timestamp(slug_day, tz=args.tz) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        rem_h = max(0.0, (expiry_local.tz_convert("UTC") - tr.open_ts).total_seconds() / 3600.0)
        p_req = _required_prob(args.alpha, args.k_decay, rem_h)

        veto_reason = None
        # Filter 1: static floor
        if p_fav < args.alpha:
            veto_reason = f"VETO_F1 floor p={p_fav:.3f} < alpha={args.alpha:.3f}"
        # Filter 2: theta dynamic threshold
        elif p_fav < p_req:
            veto_reason = f"VETO_F2 theta p={p_fav:.3f} < preq={p_req:.3f} (rem_h={rem_h:.2f})"
        # Filter 3: divergence momentum (against agent side)
        else:
            if tr.side == "LONG" and d5_fav <= -abs(args.divergence_5m):
                veto_reason = f"VETO_F3 div d5={d5_fav*100:+.2f}%p"
            if tr.side == "SHORT" and d5_fav >= abs(args.divergence_5m):
                veto_reason = f"VETO_F3 div d5={d5_fav*100:+.2f}%p"

        if veto_reason:
            vetoed += 1
            # treat as no-trade: 0%
            new_pct = 0.0
            new_sum += new_pct
            new_wins += 1 if new_pct > 0 else 0
            events.append(
                {
                    "ts": tr.open_ts,
                    "kind": "VETO",
                    "side": tr.side,
                    "slug": slug,
                    "fav_label": fav_label,
                    "opp_label": opp_label,
                    "entry": tr.open_price,
                    "p_fav": p_fav,
                    "p_req": p_req,
                    "d5_fav": d5_fav,
                    "reason": veto_reason,
                }
            )
            continue

        # active trade with panic circuit
        new_pct = base_pct
        path = feat[(feat["ts"] > tr.open_ts) & (feat["ts"] <= tr.close_ts)].copy()
        if len(path):
            for _, rr in path.iterrows():
                pmap = dict(rr["prob_map"])
                d1map = dict(rr["d1_map"])
                d5map = dict(rr["d5_map"])
                p_f = float(pmap.get(fav_label, 0.0))
                p_o = float(pmap.get(opp_label, 0.0))
                d1_f = float(d1map.get(fav_label, 0.0))
                d5_f = float(d5map.get(fav_label, 0.0))

                # Trigger 1: flash crash waterfall
                cond_t1 = (d1_f <= -abs(args.flash_1m)) or (d5_f <= -abs(args.flash_5m))
                # Trigger 2: probability death cross
                cond_t2 = (p_f + abs(args.death_margin)) < p_o
                # Trigger 3: liquidity void proxy
                cond_t3 = bool(rr.get("liq_void", False))

                if cond_t1 or cond_t2 or cond_t3:
                    exit_px = _asof_price(px, rr["ts"])
                    if exit_px is not None:
                        new_pct = _net_frac(tr.side, tr.open_price, exit_px, lev, fee=float(args.fee), slip=float(args.slip)) * 100.0
                        panic_exits += 1
                        reason_bits = []
                        if cond_t1:
                            reason_bits.append(f"T1(d1={d1_f*100:+.2f}%p,d5={d5_f*100:+.2f}%p)")
                        if cond_t2:
                            reason_bits.append(f"T2(p_f={p_f:.3f},p_o={p_o:.3f})")
                        if cond_t3:
                            reason_bits.append("T3(liq_void_proxy)")
                        events.append(
                            {
                                "ts": rr["ts"],
                                "kind": "PANIC_EXIT",
                                "side": tr.side,
                                "slug": slug,
                                "fav_label": fav_label,
                                "opp_label": opp_label,
                                "entry": tr.open_price,
                                "exit": float(exit_px),
                                "new_pnl": float(new_pct),
                                "reason": " | ".join(reason_bits),
                            }
                        )
                    break

        new_sum += float(new_pct)
        new_wins += 1 if new_pct > 0 else 0

    n = len(trades)
    print("=== Backtest: Polymarket Veto + Panic Circuit ===")
    print(f"window=({args.start_date}~{args.end_date}) tz={args.tz}")
    print(f"trades={n} slugs_loaded={len(slug_feats)}")
    print(
        f"base_sum={base_sum:+.3f}% -> new_sum={new_sum:+.3f}%  delta={new_sum-base_sum:+.3f}%p | "
        f"win={100.0*base_wins/n:.1f}% -> {100.0*new_wins/n:.1f}%"
    )
    print(f"vetoed={vetoed} panic_exits={panic_exits}")

    if events:
        out = pd.DataFrame(events).sort_values("ts").reset_index(drop=True)
        out["ts_kst"] = pd.to_datetime(out["ts"], utc=True).dt.tz_convert(args.tz)
        out["hour"] = out["ts_kst"].dt.floor("h")
        # one per hour: strongest information event (panic first, then veto)
        out["prio"] = out["kind"].map({"PANIC_EXIT": 0, "VETO": 1}).fillna(9)
        out = out.sort_values(["hour", "prio"]).drop_duplicates(subset=["hour"], keep="first").sort_values("ts_kst")
        print("\\n--- Hourly Events (1 per hour) ---")
        for _, r in out.iterrows():
            ts = r["ts_kst"].strftime("%Y-%m-%d %H:%M:%S %Z")
            if r["kind"] == "PANIC_EXIT":
                print(
                    f"{ts} | PANIC_EXIT | {r['side']} | {r['slug']} | fav={r['fav_label']} | opp={r['opp_label']} | "
                    f"entry={r['entry']:.2f} exit={r['exit']:.2f} new_pnl={r['new_pnl']:+.3f}% | {r['reason']}"
                )
            else:
                print(
                    f"{ts} | VETO | {r['side']} | {r['slug']} | fav={r['fav_label']} | opp={r['opp_label']} | "
                    f"entry={r['entry']:.2f} p_f={r['p_fav']:.3f} p_req={r['p_req']:.3f} d5={r['d5_fav']*100:+.2f}%p | {r['reason']}"
                )
    else:
        print("\\n(no events)")


def main():
    ap = argparse.ArgumentParser(description="Backtest pre-trade veto + forced liquidation panic circuit with Polymarket API.")
    ap.add_argument("--events-path", default="data/live/dashboard_events.jsonl")
    ap.add_argument("--tz", default="Asia/Seoul")
    ap.add_argument("--start-date", default="2026-04-15")
    ap.add_argument("--end-date", default="2026-04-20")
    ap.add_argument("--fee", type=float, default=0.0005)
    ap.add_argument("--slip", type=float, default=0.0002)
    ap.add_argument("--alpha", type=float, default=0.15, help="Static floor probability")
    ap.add_argument("--k-decay", type=float, default=0.12, help="Theta decay coefficient")
    ap.add_argument("--divergence-5m", type=float, default=0.08, help="Divergence veto threshold (abs, e.g. 0.08=8%%p)")
    ap.add_argument("--flash-1m", type=float, default=0.10, help="Trigger1 1m crash threshold (abs)")
    ap.add_argument("--flash-5m", type=float, default=0.20, help="Trigger1 5m crash threshold (abs)")
    ap.add_argument("--death-margin", type=float, default=0.05, help="Trigger2 death cross margin")
    args = ap.parse_args()
    run_backtest(args)


if __name__ == "__main__":
    main()
