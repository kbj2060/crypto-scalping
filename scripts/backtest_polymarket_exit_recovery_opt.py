#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
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
    shock_d1_th: float = 0.03
    shock_z_th: float = 1.5
    shock_d3_th: float = 0.005
    cooldown_min: int = 30
    recover_dp: float = 0.03
    recover_confirm: int = 2
    recover_min_wait: int = 10
    recover_max_wait: int = 90


def choose_slug(ts_utc: pd.Timestamp, slug_feats: dict[str, pd.DataFrame], tz: str) -> str | None:
    day = ts_utc.tz_convert(tz).date()
    for slug in slug_feats.keys():
        if _parse_slug_date(slug) == day:
            return slug
    return None


def prepare_data(start_date: str, end_date: str, tz: str) -> dict:
    slug_feats: dict[str, pd.DataFrame] = {}
    slug_meta: dict[str, dict] = {}

    for d in pd.date_range(start_date, end_date, freq="D", tz=tz):
        slug = f"ethereum-price-on-{d.strftime('%B').lower()}-{d.day}"
        try:
            feat, meta = _build_slug_features(slug, tz=tz)
        except Exception:
            feat, meta = pd.DataFrame(), {}
        if len(feat):
            f = feat.sort_values("ts").reset_index(drop=True).copy()

            # Build d3_map (3-minute delta) and z1_map (rolling z-score of d1 per label)
            labels = sorted({k for m in f["prob_map"] for k in dict(m).keys()})
            probs = np.zeros((len(f), len(labels)), dtype=float)
            d1 = np.zeros_like(probs)
            for i, r in f.iterrows():
                pmap = dict(r["prob_map"])
                d1map = dict(r["d1_map"])
                for j, lb in enumerate(labels):
                    probs[i, j] = float(pmap.get(lb, 0.0) or 0.0)
                    d1[i, j] = float(d1map.get(lb, 0.0) or 0.0)
            d3 = probs - np.vstack([np.zeros((3, len(labels))), probs[:-3]])

            z1 = np.zeros_like(d1)
            win = 120
            for j in range(len(labels)):
                arr = d1[:, j]
                for i in range(len(arr)):
                    lo = max(0, i - win + 1)
                    w = arr[lo : i + 1]
                    if len(w) >= 10:
                        mu = float(np.mean(w))
                        sd = float(np.std(w))
                        z1[i, j] = 0.0 if sd <= 1e-12 else (arr[i] - mu) / sd
                    else:
                        z1[i, j] = 0.0

            d3_maps = []
            z1_maps = []
            for i in range(len(f)):
                d3_maps.append({labels[j]: float(d3[i, j]) for j in range(len(labels))})
                z1_maps.append({labels[j]: float(z1[i, j]) for j in range(len(labels))})
            f["d3_map"] = d3_maps
            f["z1_map"] = z1_maps

            slug_feats[slug] = f
            slug_meta[slug] = meta

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

    cache = []
    for tr in trades:
        slug = choose_slug(tr.open_ts, slug_feats, tz)
        feat = slug_feats.get(slug)
        meta = slug_meta.get(slug, {})
        lev = _est_lev(tr, fee=0.0005, slip=0.0002)
        if feat is None or len(feat) == 0:
            cache.append((tr, slug, meta, lev, pd.DataFrame()))
            continue
        path = feat[(feat["ts"] > tr.open_ts) & (feat["ts"] <= tr.close_ts)].reset_index(drop=True)
        cache.append((tr, slug, meta, lev, path))

    return {
        "ok": True,
        "tz": tz,
        "slug_feats": slug_feats,
        "slug_meta": slug_meta,
        "trades": trades,
        "px": px,
        "cache": cache,
    }


def row_trigger(row: pd.Series, p: Params) -> tuple[bool, str, float, float, float]:
    pmap = dict(row["prob_map"])
    d1 = dict(row["d1_map"])
    d3 = dict(row["d3_map"])
    z1 = dict(row["z1_map"])

    top3 = [k for k, _ in sorted(pmap.items(), key=lambda kv: float(kv[1]), reverse=True)[:3]]
    candidates = []
    for lb in top3:
        v1 = float(d1.get(lb, 0.0) or 0.0)
        v3 = float(d3.get(lb, 0.0) or 0.0)
        vz = float(z1.get(lb, 0.0) or 0.0)
        cond = (abs(v1) >= p.shock_d1_th) and (abs(v3) >= p.shock_d3_th) and (abs(vz) >= p.shock_z_th)
        candidates.append((lb, v1, v3, vz, cond))

    trg = [x for x in candidates if x[4]]
    if not trg:
        return False, "", 0.0, 0.0, 0.0
    lb, v1, v3, vz, _ = max(trg, key=lambda x: abs(x[1]))
    return True, lb, float(v1), float(v3), float(vz)


def get_side_prob(row: pd.Series, side: str, ref_price: float, meta: dict) -> float:
    pmap = dict(row["prob_map"])
    fav, _ = _pick_labels(side, float(ref_price), meta)
    if not fav:
        return 0.0
    return float(pmap.get(fav, 0.0) or 0.0)


def run_prepared(data: dict, p: Params) -> dict:
    if not data.get("ok"):
        return data
    tz = data["tz"]
    slug_feats = data["slug_feats"]
    px = data["px"]
    cache = data["cache"]

    n = len(cache)
    base_sum = float(sum(float(x[0].realized_pct) for x in cache))
    base_wr = 100.0 * float(sum(1 for x in cache if float(x[0].realized_pct) > 0)) / max(1, n)

    sum_new = 0.0
    wins_new = 0
    n_exit = 0
    n_skip = 0
    n_reentry_ok = 0

    recovery = None  # dict with exit_ts, cooldown_until, baseline_probs, streak

    for tr, slug, meta, lev, path in cache:
        dsac_side = tr.side

        # Recovery gate state
        if recovery is not None:
            now = tr.open_ts
            elapsed_min = max(0.0, (now - recovery["exit_ts"]).total_seconds() / 60.0)

            if now < recovery["cooldown_until"] or elapsed_min < p.recover_min_wait:
                sum_new += 0.0
                wins_new += 0
                n_skip += 1
                continue

            if elapsed_min > p.recover_max_wait:
                recovery = None
            else:
                feat = slug_feats.get(slug)
                if feat is None or len(feat) == 0:
                    sum_new += 0.0
                    wins_new += 0
                    n_skip += 1
                    continue
                snap = feat[feat["ts"] <= tr.open_ts]
                if len(snap) == 0:
                    sum_new += 0.0
                    wins_new += 0
                    n_skip += 1
                    continue
                row_open = snap.iloc[-1]
                p_now = get_side_prob(row_open, dsac_side, tr.open_price, meta)
                p_base = float(recovery["baseline_probs"].get(dsac_side, 0.0))
                cond = p_now >= (p_base + p.recover_dp)
                if cond:
                    recovery["streak"] = int(recovery.get("streak", 0)) + 1
                else:
                    recovery["streak"] = 0

                if recovery["streak"] < p.recover_confirm:
                    sum_new += 0.0
                    wins_new += 0
                    n_skip += 1
                    continue
                recovery = None
                n_reentry_ok += 1

        # If no path data, keep original trade
        if path is None or len(path) == 0:
            pnl = float(tr.realized_pct)
            sum_new += pnl
            wins_new += int(pnl > 0)
            continue

        # evaluate first trigger in trade path
        acted = False
        for _, row in path.iterrows():
            trig, lb, v1, v3, vz = row_trigger(row, p)
            if not trig:
                continue
            tgt = float(row.get("weighted_target", 0.0) or 0.0)
            favorable = (tgt > tr.open_price) if dsac_side == "LONG" else (tgt < tr.open_price)
            if favorable:
                # emergency hold - keep position
                break

            ex = _asof_price(px, row["ts"])
            if ex is None:
                break

            # emergency exit
            pnl = _net_frac(dsac_side, tr.open_price, ex, lev, fee=0.0005, slip=0.0002) * 100.0
            sum_new += pnl
            wins_new += int(pnl > 0)
            n_exit += 1
            acted = True

            # setup recovery baselines from exit snapshot
            p_long = get_side_prob(row, "LONG", ex, meta)
            p_short = get_side_prob(row, "SHORT", ex, meta)
            recovery = {
                "exit_ts": pd.Timestamp(row["ts"]),
                "cooldown_until": pd.Timestamp(row["ts"]) + pd.Timedelta(minutes=p.cooldown_min),
                "baseline_probs": {"LONG": p_long, "SHORT": p_short},
                "streak": 0,
            }
            break

        if not acted:
            pnl = float(tr.realized_pct)
            sum_new += pnl
            wins_new += int(pnl > 0)

    return {
        "ok": True,
        "trades": n,
        "base_sum": base_sum,
        "base_wr": base_wr,
        "new_sum": float(sum_new),
        "new_wr": 100.0 * float(wins_new) / max(1, n),
        "delta": float(sum_new - base_sum),
        "n_exit": int(n_exit),
        "n_skip": int(n_skip),
        "n_reentry_ok": int(n_reentry_ok),
    }


def main():
    ap = argparse.ArgumentParser(description="Optimize Polymarket emergency-exit + probability-recovery reentry gate")
    ap.add_argument("--start-date", default="2026-04-06")
    ap.add_argument("--end-date", default="2026-04-20")
    ap.add_argument("--tz", default="Asia/Seoul")
    ap.add_argument("--trials", type=int, default=800)
    args = ap.parse_args()

    data = prepare_data(args.start_date, args.end_date, args.tz)
    if not data.get("ok"):
        print(data)
        return

    rng = np.random.default_rng(20260420)
    rows = []
    for _ in range(int(max(10, args.trials))):
        p = Params(
            shock_d1_th=float(rng.choice([0.02, 0.025, 0.03, 0.035, 0.04, 0.05])),
            shock_z_th=float(rng.choice([1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0])),
            shock_d3_th=float(rng.choice([0.003, 0.005, 0.0075, 0.01, 0.015, 0.02])),
            cooldown_min=int(rng.choice([10, 15, 20, 30, 45, 60])),
            recover_dp=float(rng.choice([0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06])),
            recover_confirm=int(rng.choice([1, 2, 3])),
            recover_min_wait=int(rng.choice([5, 10, 15, 20])),
            recover_max_wait=int(rng.choice([30, 45, 60, 90, 120])),
        )
        if p.recover_max_wait <= p.recover_min_wait:
            p.recover_max_wait = p.recover_min_wait + 30

        r = run_prepared(data, p)
        if not r.get("ok"):
            continue
        rows.append(
            {
                "new_sum": r["new_sum"],
                "new_wr": r["new_wr"],
                "delta": r["delta"],
                "n_exit": r["n_exit"],
                "n_skip": r["n_skip"],
                "n_reentry_ok": r["n_reentry_ok"],
                "base_sum": r["base_sum"],
                "base_wr": r["base_wr"],
                "trades": r["trades"],
                "shock_d1_th": p.shock_d1_th,
                "shock_z_th": p.shock_z_th,
                "shock_d3_th": p.shock_d3_th,
                "cooldown_min": p.cooldown_min,
                "recover_dp": p.recover_dp,
                "recover_confirm": p.recover_confirm,
                "recover_min_wait": p.recover_min_wait,
                "recover_max_wait": p.recover_max_wait,
            }
        )

    if not rows:
        print("no_results")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(["new_sum", "new_wr", "n_skip"], ascending=[False, False, True]).reset_index(drop=True)
    print("=== TOP 20 ===")
    print(df.head(20).to_string(index=False))
    print("=== BEST ===")
    print(df.iloc[0].to_dict())


if __name__ == "__main__":
    main()
