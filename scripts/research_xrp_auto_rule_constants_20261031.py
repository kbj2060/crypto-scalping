"""XRP 자동 규칙 상수 검정 -- 사전등록 docs/prereg/xrp_auto_rule_constants_20261031.md (aacb814b).

ETH 상수 C_E(L 2·첫 1/3·∓0.75% 물타기·∓5% 손절)의 위험(손절률·갭 손실·1년 −20% 낙폭)이 XRP 에서 같은가,
변동성 비 r 로 환산한 C_X(d·s ×r, L ÷r)가 ETH 수준으로 돌려놓는가. 1분봉 경로, 실력 0(E1)·ETH 원장 이식(E2).
실행: python scripts/research_xrp_auto_rule_constants_20261031.py [--selftest]
"""
import io, json, sys, urllib.request, zipfile
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd

ROOT = Path("/home/kbj20/crypto-scalping/data/binance_vision")
WT = Path(__file__).resolve().parents[1]
CACHE = WT / "data/klines1m_daily_cache"
OUT = WT / "tmp/xrp_auto_rule_20261031"
END = pd.Timestamp("2026-10-27 23:59")   # 수정 1: vision 10-28 없음
MK, TK = 2e-4, 5e-4
C_E = (2.0, 0.0075, 0.05)
SPLITS = {"stress": ("2022-01-01", "2023-12-31"), "train": ("2024-01-01", "2025-12-31"),
          "val": ("2026-01-01", "2026-05-31"), "test": ("2026-06-01", "2026-10-27")}
N_E1, TRIPS_DAY, NPATH, NBOOT, BLOCK = 20_000, 2.37, 4_000, 2_000, 5


def _daily(sym, day):
    f = CACHE / f"{sym}-1m-{day}.parquet"
    if not f.exists():
        url = f"https://data.binance.vision/data/futures/um/daily/klines/{sym}/1m/{sym}-1m-{day}.zip"
        with urllib.request.urlopen(url, timeout=60) as r:
            z = zipfile.ZipFile(io.BytesIO(r.read()))
        d = pd.read_csv(z.open(z.namelist()[0]), header=None)
        if not str(d.iat[0, 0]).isdigit():
            d = d.iloc[1:]
        d = d.iloc[:, [0, 2, 3, 4]].astype(float)
        d.columns = ["t", "h", "l", "c"]
        CACHE.mkdir(parents=True, exist_ok=True)
        d.to_parquet(f)
    return pd.read_parquet(f)


def load(sym):
    parts = [pd.read_parquet(p, columns=["t", "h", "l", "c"]) for p in sorted((ROOT / "klines1m").glob(f"{sym}-1m-20*.parquet"))]
    parts += [_daily(sym, d.date()) for d in pd.date_range("2026-10-01", END.normalize())]
    d = pd.concat(parts).drop_duplicates("t").sort_values("t")
    d.index = pd.to_datetime(d.t.astype("int64"), unit="ms")
    return d[["h", "l", "c"]].astype(float)


def aligned():
    e, x = load("ETHUSDT"), load("XRPUSDT")
    idx = pd.date_range(max(e.index[0], x.index[0]), END, freq="1min")
    out = {}
    for k, d in (("eth", e), ("xrp", x)):
        d = d.reindex(idx)
        c = d.c.ffill()
        out[k] = (d.h.fillna(c).values, d.l.fillna(c).values, c.values)
    return idx, out


@nb.njit(cache=True)
def sim(h, l, c, starts, sides, holds, L, d, s):
    """왕복마다 [손익 기본, 손익 스트레스, 손절?, 물타기?, 손절 스트레스 손실]. 순자산 단위."""
    n = len(starts)
    out = np.zeros((n, 5))
    T = len(c)
    for i in range(n):
        t = starts[i]
        if t < 1 or t >= T:
            out[i, :] = np.nan
            continue
        sg = sides[i]
        p0 = c[t - 1]
        a = p0 * (1 - sg * d)
        st = p0 * (1 - sg * s)
        n1, n2 = L / 3.0, 2.0 * L / 3.0
        added, stopped = 0.0, 0.0
        xb, xs = 0.0, 0.0
        end = min(t + holds[i], T) - 1
        for k in range(t, end + 1):
            if added == 0.0 and ((sg > 0 and l[k] < a) or (sg < 0 and h[k] > a)):
                added = 1.0
            if (sg > 0 and l[k] <= st) or (sg < 0 and h[k] >= st):
                stopped = 1.0
                if sg > 0:
                    xb = min(st, h[k])
                    xs = l[k]
                else:
                    xb = max(st, l[k])
                    xs = h[k]
                break
        if stopped == 0.0:
            xb = c[end]
            xs = c[end]
        fee = n1 * MK + added * n2 * MK + (n1 + added * n2) * TK
        rb = n1 * sg * (xb / p0 - 1) + added * n2 * sg * (xb / a - 1) - fee
        rs = n1 * sg * (xs / p0 - 1) + added * n2 * sg * (xs / a - 1) - fee
        out[i, 0], out[i, 1], out[i, 2], out[i, 3] = rb, rs, stopped, added
        out[i, 4] = -rs if stopped == 1.0 else 0.0
    return out


def sigma_ratio(idx, D, lo, hi):
    r = {}
    for k, (_, _, c) in D.items():
        lr = pd.Series(np.diff(np.log(c), prepend=np.nan), index=idx)
        r[k] = lr.rolling(1440).std()[lr.index.hour.isin([0]) & (lr.index.minute == 0)]
    q = (r["xrp"] / r["eth"]).loc[lo:hi].dropna()
    return float(q.median()), len(q)


def e1_trips(idx, lo, hi, hold_min, rng):
    pos = np.flatnonzero((idx >= pd.Timestamp(lo)) & (idx <= pd.Timestamp(hi) + pd.Timedelta("23:59:00")))
    t = np.sort(rng.choice(pos[1:], N_E1, replace=False))
    sides = np.where(rng.random(N_E1) < 0.6, 1.0, -1.0)
    holds = rng.choice(hold_min, N_E1).astype(np.int64)
    return t.astype(np.int64), sides, holds


def block_ci(day, vals_x, vals_e, rng):
    """5일 블록 부트스트랩, (XRP − ETH) 평균 차의 점추정·양측 95%."""
    days = np.unique(day)
    pos = {dd: i for i, dd in enumerate(days)}
    di = np.array([pos[dd] for dd in day])
    nd = len(days)
    sx = np.bincount(di, vals_x, nd)
    se = np.bincount(di, vals_e, nd)
    cnt = np.bincount(di, minlength=nd).astype(float)
    nb_ = int(np.ceil(nd / BLOCK))
    bs = []
    for _ in range(NBOOT):
        st = rng.integers(0, nd, nb_)
        ii = (st[:, None] + np.arange(BLOCK)[None]).ravel() % nd
        bs.append((sx[ii].sum() - se[ii].sum()) / cnt[ii].sum())
    pt = (vals_x.mean() - vals_e.mean())
    return pt, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def dd_prob(day, rx, re_, rng):
    """하루 Poisson(2.37)건, 같은 날 왕복에서 뽑아 365일 복리 → MDD ≤ −20% 비율(두 코인 같은 추첨)."""
    days = np.unique(day)
    groups = [np.flatnonzero(day == dd) for dd in days]
    hit = np.zeros((NPATH, 2), bool)
    for p in range(NPATH):
        dpick = rng.integers(0, len(days), 365)
        k = rng.poisson(TRIPS_DAY, 365)
        ii = np.concatenate([rng.choice(groups[g], m) for g, m in zip(dpick, k) if m > 0])
        for j, r in enumerate((rx, re_)):
            eq = np.cumprod(1 + np.maximum(r[ii], -0.999))
            peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
            hit[p, j] = (eq / peak - 1).min() <= -0.20
    px, pe = hit[:, 0].mean(), hit[:, 1].mean()
    dif = hit[:, 0].astype(float) - hit[:, 1]
    se = dif.std(ddof=1) / np.sqrt(NPATH)
    return float(px), float(pe), (float(px - pe), float(px - pe - 1.96 * se), float(px - pe + 1.96 * se))


def metrics(day, rx, re_, rng, tag):
    """rx/re_: sim 출력. M1·M2(전체 대비)·M3·M4."""
    m = {}
    for name, col, f in (("M1_stop_rate", 2, lambda o: o[:, 2]),
                         ("M2_stress_loss_gt10pct_all", 4, lambda o: (o[:, 4] > 0.10).astype(float))):
        pt, lo, hi = block_ci(day, f(rx), f(re_), rng)
        m[name] = {"xrp": float(f(rx).mean()), "eth": float(f(re_).mean()), "diff": pt, "ci95": [lo, hi]}
    sx, se_ = rx[:, 2] == 1, re_[:, 2] == 1
    m["M2_given_stop"] = {"xrp": f"{int((rx[sx, 4] > .1).sum())}/{int(sx.sum())}", "eth": f"{int((re_[se_, 4] > .1).sum())}/{int(se_.sum())}"}
    px, pe, (d0, lo, hi) = dd_prob(day, rx[:, 0], re_[:, 0], rng)
    m["M3_dd20_1y_base"] = {"xrp": px, "eth": pe, "diff": d0, "ci95": [lo, hi]}
    px, pe, (d0, lo, hi) = dd_prob(day, rx[:, 1], re_[:, 1], rng)
    m["M3_dd20_1y_stress"] = {"xrp": px, "eth": pe, "diff": d0, "ci95": [lo, hi]}
    m["M4_worst_stress"] = {"xrp": float(rx[:, 1].min()), "eth": float(re_[:, 1].min())}
    m["add_fill"] = {"xrp": float(rx[:, 3].mean()), "eth": float(re_[:, 3].mean())}
    m["mean_trip_bp"] = {"xrp": float(rx[:, 0].mean() * 1e4), "eth": float(re_[:, 0].mean() * 1e4)}
    return m


def strict_ok(m):
    return (m["M1_stop_rate"]["ci95"][1] <= 0.01 and m["M2_stress_loss_gt10pct_all"]["ci95"][1] <= 0.005
            and m["M3_dd20_1y_base"]["ci95"][1] <= 0.05)


def cand_ok(m):
    return (m["M1_stop_rate"]["diff"] <= 0.01 and m["M2_stress_loss_gt10pct_all"]["diff"] <= 0.005
            and m["M3_dd20_1y_base"]["diff"] <= 0.05)


def selftest():
    # S1: 손절·물타기 없음 → 1/3 명목 단순 수익
    c = np.array([100, 101, 99, 102, 103.0])
    o = sim(c + .5, c - .5, c, np.array([1]), np.array([1.0]), np.array([3]), 2.0, 0.9, 0.99)
    want = 2 / 3 * (102 / 100 - 1) - (2 / 3 * MK + 2 / 3 * TK)
    assert abs(o[0, 0] - want) < 1e-12, (o, want)
    # S2: 롱 p0 100, 손절 95. 분1 저가 94(고가 99) → 기본 95·스트레스 94 / 갭 분(고가 93 저가 90) → 기본 93·스트레스 90
    h = np.array([100, 99, 100.]); l = np.array([100, 94, 100.]); c = np.array([100, 96, 100.])
    o = sim(h, l, c, np.array([1]), np.array([1.0]), np.array([2]), 2.0, 0.0075, 0.05)
    a = 100 * (1 - .0075)
    fee = 2 / 3 * MK + 4 / 3 * MK + 2 * TK
    assert o[0, 2] == 1 and o[0, 3] == 1
    assert abs(o[0, 0] - (2 / 3 * (95 / 100 - 1) + 4 / 3 * (95 / a - 1) - fee)) < 1e-12
    assert abs(o[0, 1] - (2 / 3 * (94 / 100 - 1) + 4 / 3 * (94 / a - 1) - fee)) < 1e-12
    h = np.array([100, 93, 100.]); l = np.array([100, 90, 100.])
    o = sim(h, l, c, np.array([1]), np.array([1.0]), np.array([2]), 2.0, 0.0075, 0.05)
    assert abs(o[0, 0] - (2 / 3 * (93 / 100 - 1) + 4 / 3 * (93 / a - 1) - fee)) < 1e-12
    # 숏 대칭: p0 100, 손절 105, 분1 고가 106 저가 101 → 기본 105
    h = np.array([100, 106, 100.]); l = np.array([100, 101, 100.])
    o = sim(h, l, c, np.array([1]), np.array([-1.0]), np.array([2]), 2.0, 0.0075, 0.05)
    a = 100 * 1.0075
    assert o[0, 2] == 1 and o[0, 3] == 1 and abs(o[0, 0] - (-2 / 3 * (105 / 100 - 1) - 4 / 3 * (105 / a - 1) - fee)) < 1e-12
    # S3: 갭 없는 손절 손실 ≤ L×s + 비용
    assert -sim(np.array([100, 99, 100.]), np.array([100, 94, 100.]), c, np.array([1]), np.array([1.0]), np.array([2]), 2.0, .0075, .05)[0, 0] <= 2 * .05 + 2 * (MK + TK) + 1e-12
    print("selftest ok")


def main():
    selftest()
    if "--selftest" in sys.argv:
        return
    rng = np.random.default_rng(20261031)
    idx, D = aligned()
    led = pd.DataFrame([json.loads(x) for x in open(WT / "data/live/account_round_trips.jsonl")])
    led = led[led.symbol.str.startswith("ETH") & (led.closed == True)]
    hold_min_all = np.maximum(1, np.round((led.exit_time - led.entry_time) / 6e4)).astype(int).values
    led = led[pd.to_datetime(led.exit_time, unit="ms") <= END]   # 수정 1: 데이터 끝 밖 왕복은 E2 에서 제외
    hold_min = np.maximum(1, np.round((led.exit_time - led.entry_time) / 6e4)).astype(int).values
    hold_min, hold_e2 = hold_min_all, hold_min
    r, nr = sigma_ratio(idx, D, *SPLITS["train"])
    C_X = (2.0 / r, 0.0075 * r, 0.05 * r)
    rep = {"r": r, "r_days": nr, "C_E": C_E, "C_X": C_X, "n_ledger": len(led), "splits": {}}
    print(f"r={r:.4f} ({nr}일) C_X={C_X}")
    for sp, (lo, hi) in SPLITS.items():
        if sp == "train":
            continue
        t, sides, holds = e1_trips(idx, lo, hi, hold_min, rng)
        day = idx[t].normalize().values
        ee = sim(*D["eth"], t, sides, holds, *C_E)
        xe = sim(*D["xrp"], t, sides, holds, *C_E)
        xx = sim(*D["xrp"], t, sides, holds, *C_X)
        res = {"n": int(len(t)),
               "Q1_xrpCE_vs_ethCE": metrics(day, xe, ee, rng, "q1"),
               "Q2_xrpCX_vs_ethCE": metrics(day, xx, ee, rng, "q2")}
        # 위약: ETH 끼리, 시각 7일 원형 이동
        sh = (t - pd.Timedelta("7D").total_seconds() // 60).astype(np.int64)
        ok = sh > 1
        es = sim(*D["eth"], sh[ok], sides[ok], holds[ok], *C_E)
        res["placebo_eth_shift7d"] = {k: metrics(day[ok], es, ee[ok], rng, "pl")[k] for k in ("M1_stop_rate", "M2_stress_loss_gt10pct_all")}
        nost = (C_E[0], C_E[1], 0.99)
        res["control_nostop"] = {"xrp_worst": float(sim(*D["xrp"], t, sides, holds, *nost)[:, 1].min()),
                                 "eth_worst": float(sim(*D["eth"], t, sides, holds, *nost)[:, 1].min())}
        rep["splits"][sp] = res
        q1, q2 = res["Q1_xrpCE_vs_ethCE"], res["Q2_xrpCX_vs_ethCE"]
        print(sp, "Q1", {k: (round(v["diff"], 4), [round(z, 4) for z in v["ci95"]]) for k, v in q1.items() if "ci95" in v},
              "| Q2", {k: (round(v["diff"], 4), [round(z, 4) for z in v["ci95"]]) for k, v in q2.items() if "ci95" in v})
    # E2 원장 이식(보조)
    ts = pd.to_datetime(led.entry_time, unit="ms").dt.floor("1min")
    t2 = idx.get_indexer(ts).astype(np.int64)
    h2 = hold_e2.astype(np.int64)
    s2 = np.where(led.side.values == "LONG", 1.0, -1.0)
    e2 = {}
    for nm, (k, C) in {"eth_CE": ("eth", C_E), "xrp_CE": ("xrp", C_E), "xrp_CX": ("xrp", C_X)}.items():
        o = sim(*D[k], t2, s2, h2, *C)
        e2[nm] = {"n": int(np.isfinite(o[:, 0]).sum()), "stops": int(np.nansum(o[:, 2])), "worst_stress": float(np.nanmin(o[:, 1])),
                  "sum_base": float(np.nansum(o[:, 0])), "compound_base": float(np.prod(1 + np.nan_to_num(o[:, 0])))}
    rep["E2_ledger"] = e2
    print("E2", e2)
    # 2025-10-10 사건(롱이 21:00 UTC 열려 있었다면)
    k0 = idx.get_indexer([pd.Timestamp("2025-10-10 20:00")])[0]
    ev = {}
    for nm, (k, C) in {"eth_CE": ("eth", C_E), "xrp_CE": ("xrp", C_E), "xrp_CX": ("xrp", C_X)}.items():
        o = sim(*D[k], np.array([k0], np.int64), np.array([1.0]), np.array([180], np.int64), *C)
        ev[nm] = {"base": float(o[0, 0]), "stress": float(o[0, 1]), "stopped": bool(o[0, 2])}
    rep["event_20251010_long_20utc"] = ev
    print("20251010", ev)
    v, te = rep["splits"]["val"], rep["splits"]["test"]
    q1_worse = {sp: any(rep["splits"][sp]["Q1_xrpCE_vs_ethCE"][k]["ci95"][0] > 0 for k in ("M1_stop_rate", "M2_stress_loss_gt10pct_all", "M3_dd20_1y_base")) for sp in ("val", "test")}
    rep["verdict"] = {"Q1_xrp_riskier_with_CE": q1_worse,
                      "Q2_strict": strict_ok(v["Q2_xrpCX_vs_ethCE"]) and strict_ok(te["Q2_xrpCX_vs_ethCE"]),
                      "Q2_candidate": cand_ok(v["Q2_xrpCX_vs_ethCE"]) and cand_ok(te["Q2_xrpCX_vs_ethCE"]),
                      "Q1_CE_strict_equiv": strict_ok(v["Q1_xrpCE_vs_ethCE"]) and strict_ok(te["Q1_xrpCE_vs_ethCE"])}
    print("verdict", rep["verdict"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
