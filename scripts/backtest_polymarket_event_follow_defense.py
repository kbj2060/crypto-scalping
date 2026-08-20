#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.backtest_polymarket_shock_entry_exit import (
        _asof_price,
        _est_lev,
        _fetch_binance_1m,
        _load_trades,
        _net_frac,
    )
    from scripts.backtest_polymarket_veto_panic import (
        _build_slug_features,
        _parse_slug_date,
        _pick_labels,
    )
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from scripts.backtest_polymarket_shock_entry_exit import (
        _asof_price,
        _est_lev,
        _fetch_binance_1m,
        _load_trades,
        _net_frac,
    )
    from scripts.backtest_polymarket_veto_panic import (
        _build_slug_features,
        _parse_slug_date,
        _pick_labels,
    )


@dataclass
class Params:
    # shock score S = w1*d1 + w2*d5 + w3*breadth + w4*z
    w1: float = 0.45
    w2: float = 0.25
    w3: float = 0.15
    w4: float = 0.15
    th_reduce: float = 0.060
    th_exit: float = 0.095
    exit_confirm_ticks: int = 2
    reduce_to_size: float = 0.40
    # unfavorable filters
    opp_margin_reduce: float = 0.005
    opp_margin_exit: float = 0.010
    # cooldown / reentry
    cooldown_min: int = 30
    relax_th: float = 0.045
    reentry_confirm: int = 2


def _choose_slug(ts_utc: pd.Timestamp, slug_feats: dict[str, pd.DataFrame], tz: str) -> str | None:
    day = ts_utc.tz_convert(tz).date()
    for slug in slug_feats.keys():
        if _parse_slug_date(slug) == day:
            return slug
    return None


def _top3_from_prob_map(prob_map: dict[str, float]) -> list[str]:
    return [k for k, _ in sorted(prob_map.items(), key=lambda kv: float(kv[1]), reverse=True)[:3]]


def _build_day_features(start_date: str, end_date: str, tz: str) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    slug_feats: dict[str, pd.DataFrame] = {}
    slug_meta: dict[str, dict] = {}

    for d in pd.date_range(start_date, end_date, freq="D", tz=tz):
        slug = f"ethereum-price-on-{d.strftime('%B').lower()}-{d.day}"
        try:
            feat, meta = _build_slug_features(slug, tz=tz)
        except Exception:
            feat, meta = pd.DataFrame(), {}
        if len(feat):
            f = feat.copy().sort_values("ts").reset_index(drop=True)
            # per-row top3 breadth and z-score for d1(max abs among top3)
            d1_abs = []
            breadth = []
            for _, r in f.iterrows():
                pmap = dict(r["prob_map"])
                d1map = dict(r["d1_map"])
                top3 = _top3_from_prob_map(pmap)
                vals = [abs(float(d1map.get(lb, 0.0) or 0.0)) for lb in top3]
                d1_abs.append(max(vals) if vals else 0.0)
                breadth.append(float(sum(1 for v in vals if v >= 0.03)) / 3.0)
            f["d1_abs_top3"] = np.asarray(d1_abs, dtype=float)
            # d5_map is used as slower momentum proxy (API fidelity minute)
            d5_abs = []
            for _, r in f.iterrows():
                pmap = dict(r["prob_map"])
                d5map = dict(r["d5_map"])
                top3 = _top3_from_prob_map(pmap)
                vals = [abs(float(d5map.get(lb, 0.0) or 0.0)) for lb in top3]
                d5_abs.append(max(vals) if vals else 0.0)
            f["d5_abs_top3"] = np.asarray(d5_abs, dtype=float)
            f["breadth_top3"] = np.asarray(breadth, dtype=float)

            roll = f["d1_abs_top3"].rolling(60, min_periods=10)
            mu = roll.mean()
            sd = roll.std().replace(0.0, np.nan)
            z = (f["d1_abs_top3"] - mu) / sd
            f["z_d1"] = z.fillna(0.0).clip(-10.0, 10.0)

            slug_feats[slug] = f
            slug_meta[slug] = meta

    return slug_feats, slug_meta


def _shock_score(row: pd.Series, p: Params) -> float:
    d1 = float(row.get("d1_abs_top3", 0.0) or 0.0)
    d5 = float(row.get("d5_abs_top3", 0.0) or 0.0)
    br = float(row.get("breadth_top3", 0.0) or 0.0)
    z1 = abs(float(row.get("z_d1", 0.0) or 0.0))
    return p.w1 * d1 + p.w2 * d5 + p.w3 * br + p.w4 * (z1 / 4.0)


def _prepare_backtest_data(start_date: str, end_date: str, tz: str) -> dict:
    slug_feats, slug_meta = _build_day_features(start_date, end_date, tz)
    if not slug_feats:
        return {"ok": False, "reason": "no_slug_features"}

    trades = _load_trades("data/live/dashboard_events.jsonl")
    s_utc = pd.Timestamp(start_date, tz=tz).tz_convert("UTC")
    e_utc = (pd.Timestamp(end_date, tz=tz) + pd.Timedelta(hours=23, minutes=59, seconds=59)).tz_convert("UTC")
    trades = [t for t in trades if s_utc <= t.close_ts <= e_utc]
    trades = sorted(trades, key=lambda x: x.open_ts)
    if not trades:
        return {"ok": False, "reason": "no_trades"}

    px = _fetch_binance_1m(
        min(t.open_ts for t in trades) - pd.Timedelta(minutes=10),
        max(t.close_ts for t in trades) + pd.Timedelta(minutes=10),
    )

    return {
        "ok": True,
        "slug_feats": slug_feats,
        "slug_meta": slug_meta,
        "trades": trades,
        "px": px,
        "tz": tz,
    }


def run_backtest_prepared(data: dict, p: Params) -> dict:
    if not bool(data.get("ok", False)):
        return {"ok": False, "reason": str(data.get("reason", "prepare_failed"))}
    slug_feats: dict[str, pd.DataFrame] = data["slug_feats"]
    slug_meta: dict[str, dict] = data["slug_meta"]
    trades = data["trades"]
    px = data["px"]
    tz = data["tz"]

    n = len(trades)
    base_sum = float(sum(float(t.realized_pct) for t in trades))
    base_wr = 100.0 * float(sum(1 for t in trades if float(t.realized_pct) > 0)) / n

    sum_new = 0.0
    wins_new = 0
    reduce_events = 0
    hard_exit_events = 0
    skipped_by_cooldown = 0

    cooldown_until: pd.Timestamp | None = None
    rearm_wait = False
    rearm_side: str | None = None
    rearm_streak = 0

    for tr in trades:
        lev = _est_lev(tr, fee=0.0005, slip=0.0002)

        # cooldown and reentry control
        if cooldown_until is not None and tr.open_ts < cooldown_until:
            sum_new += 0.0
            wins_new += 0
            skipped_by_cooldown += 1
            continue

        slug = _choose_slug(tr.open_ts, slug_feats, tz)
        if slug is None:
            pnl = _net_frac(tr.side, tr.open_price, tr.close_price, lev, fee=0.0005, slip=0.0002) * 100.0
            sum_new += pnl
            wins_new += int(pnl > 0)
            continue

        feat = slug_feats.get(slug)
        meta = slug_meta.get(slug, {})
        if feat is None or len(feat) == 0:
            pnl = _net_frac(tr.side, tr.open_price, tr.close_price, lev, fee=0.0005, slip=0.0002) * 100.0
            sum_new += pnl
            wins_new += int(pnl > 0)
            continue

        # re-entry gate after hard exit
        if rearm_wait:
            row_open_df = feat[feat["ts"] <= tr.open_ts]
            if len(row_open_df):
                row_open = row_open_df.iloc[-1]
                s_open = _shock_score(row_open, p)
                if tr.side == rearm_side and s_open < p.relax_th:
                    rearm_streak += 1
                else:
                    rearm_streak = 0
                if rearm_streak < p.reentry_confirm:
                    sum_new += 0.0
                    wins_new += 0
                    skipped_by_cooldown += 1
                    continue
                rearm_wait = False
                rearm_side = None
                rearm_streak = 0

        # path simulation with reduce -> hard exit
        path = feat[(feat["ts"] > tr.open_ts) & (feat["ts"] <= tr.close_ts)].reset_index(drop=True)
        if len(path) == 0:
            pnl = _net_frac(tr.side, tr.open_price, tr.close_price, lev, fee=0.0005, slip=0.0002) * 100.0
            sum_new += pnl
            wins_new += int(pnl > 0)
            continue

        fav_label, opp_label = _pick_labels(tr.side, tr.open_price, meta)

        pos_size = 1.0
        realized = 0.0
        reduced = False
        adverse_exit_streak = 0
        trade_closed = False

        for _, row in path.iterrows():
            s = _shock_score(row, p)
            pmap = dict(row["prob_map"])
            pf = float(pmap.get(fav_label, 0.0) if fav_label else 0.0)
            po = float(pmap.get(opp_label, 0.0) if opp_label else 0.0)
            unfavorable = (po - pf) >= p.opp_margin_reduce

            if unfavorable and s >= p.th_reduce and (not reduced):
                # realize closed fraction now
                ex = _asof_price(px, row["ts"])
                if ex is not None and pos_size > p.reduce_to_size:
                    frac = pos_size - p.reduce_to_size
                    leg = _net_frac(tr.side, tr.open_price, ex, lev, fee=0.0005, slip=0.0002)
                    realized += leg * frac * 100.0
                    pos_size = p.reduce_to_size
                    reduced = True
                    reduce_events += 1

            unfavorable_exit = (po - pf) >= p.opp_margin_exit
            if unfavorable_exit and s >= p.th_exit:
                adverse_exit_streak += 1
            else:
                adverse_exit_streak = 0

            if adverse_exit_streak >= p.exit_confirm_ticks and pos_size > 0.0:
                ex = _asof_price(px, row["ts"])
                if ex is not None:
                    leg = _net_frac(tr.side, tr.open_price, ex, lev, fee=0.0005, slip=0.0002)
                    realized += leg * pos_size * 100.0
                    pos_size = 0.0
                    trade_closed = True
                    hard_exit_events += 1
                    cooldown_until = row["ts"] + pd.Timedelta(minutes=p.cooldown_min)
                    rearm_wait = True
                    rearm_side = tr.side
                    rearm_streak = 0
                break

        if not trade_closed and pos_size > 0.0:
            leg = _net_frac(tr.side, tr.open_price, tr.close_price, lev, fee=0.0005, slip=0.0002)
            realized += leg * pos_size * 100.0

        sum_new += float(realized)
        wins_new += int(realized > 0.0)

    return {
        "ok": True,
        "trades": n,
        "base_sum": base_sum,
        "base_wr": base_wr,
        "new_sum": float(sum_new),
        "new_wr": 100.0 * float(wins_new) / n,
        "delta": float(sum_new - base_sum),
        "reduce_events": int(reduce_events),
        "hard_exit_events": int(hard_exit_events),
        "skipped_by_cooldown": int(skipped_by_cooldown),
        "params": p,
    }


def run_backtest(start_date: str, end_date: str, tz: str, p: Params) -> dict:
    data = _prepare_backtest_data(start_date, end_date, tz)
    if not bool(data.get("ok", False)):
        return data
    return run_backtest_prepared(data, p)


def main():
    ap = argparse.ArgumentParser(description="Backtest polymarket event-follow defense (reduce/hard-exit/cooldown).")
    ap.add_argument("--start-date", default="2026-04-06")
    ap.add_argument("--end-date", default="2026-04-20")
    ap.add_argument("--tz", default="Asia/Seoul")
    ap.add_argument("--search", action="store_true")
    args = ap.parse_args()

    data = _prepare_backtest_data(args.start_date, args.end_date, args.tz)
    if not bool(data.get("ok", False)):
        print(data)
        return

    if not args.search:
        r = run_backtest_prepared(data, Params())
        print(r)
        return

    # randomized search over a bounded budget
    rng = np.random.default_rng(42)
    max_trials = 900
    w_sets = np.array(
        [
            [0.45, 0.25, 0.15, 0.15],
            [0.50, 0.20, 0.15, 0.15],
            [0.40, 0.30, 0.15, 0.15],
            [0.55, 0.20, 0.10, 0.15],
        ],
        dtype=float,
    )
    rows = []
    for _ in range(max_trials):
        ws = w_sets[int(rng.integers(0, len(w_sets)))]
        tr = float(rng.choice([0.050, 0.055, 0.060, 0.065, 0.070]))
        te = float(rng.choice([0.085, 0.095, 0.105, 0.115, 0.125]))
        ec = int(rng.choice([2, 3]))
        rz = float(rng.choice([0.30, 0.40, 0.50, 0.60]))
        mr = float(rng.choice([0.000, 0.005, 0.010]))
        me = float(rng.choice([0.005, 0.010, 0.015, 0.020, 0.030]))
        cd = int(rng.choice([20, 30]))
        relax = float(rng.choice([0.040, 0.045, 0.050, 0.055]))
        rc = int(rng.choice([2, 3]))
        if te <= tr:
            continue
        p = Params(
            w1=ws[0], w2=ws[1], w3=ws[2], w4=ws[3],
            th_reduce=tr,
            th_exit=te,
            exit_confirm_ticks=ec,
            reduce_to_size=rz,
            opp_margin_reduce=mr,
            opp_margin_exit=me,
            cooldown_min=cd,
            relax_th=relax,
            reentry_confirm=rc,
        )
        r = run_backtest_prepared(data, p)
        if not r.get("ok"):
            continue
        rows.append(
            {
                "new_sum": r["new_sum"],
                "new_wr": r["new_wr"],
                "delta": r["delta"],
                "hard_exit": r["hard_exit_events"],
                "reduce": r["reduce_events"],
                "skip": r["skipped_by_cooldown"],
                "th_reduce": tr,
                "th_exit": te,
                "exit_confirm": ec,
                "reduce_to": rz,
                "m_reduce": mr,
                "m_exit": me,
                "cooldown": cd,
                "relax": relax,
                "reentry_confirm": rc,
                "w1": ws[0], "w2": ws[1], "w3": ws[2], "w4": ws[3],
                "base_sum": r["base_sum"],
                "base_wr": r["base_wr"],
                "trades": r["trades"],
            }
        )

    if not rows:
        print("no_result")
        return

    df = pd.DataFrame(rows)
    # prefer best return, then fewer interventions
    df = df.sort_values(["new_sum", "new_wr", "hard_exit", "reduce"], ascending=[False, False, True, True]).reset_index(drop=True)
    print("=== TOP 20 ===")
    print(df.head(20).to_string(index=False))
    print("=== BEST ===")
    print(df.iloc[0].to_dict())


if __name__ == "__main__":
    main()
