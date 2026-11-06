#!/usr/bin/env python3
"""미개봉 원천 3종 (사전등록 docs/prereg/new_sources_news_hl_onchain_20261107.md).
    python scripts/research_new_sources_20261107.py --selftest
    python scripts/research_new_sources_20261107.py extract   # 서버에서 읽기 전용(tmp/new_sources_20261107 에 csv)
    python scripts/research_new_sources_20261107.py fetch_cm  # 로컬, 커뮤니티 API
    python scripts/research_new_sources_20261107.py n1|n2|n3  # 로컬, 각 1회
"""
import hashlib, json, sqlite3, sys, urllib.request
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path("tmp/new_sources_20261107")
BOOT, SEED = 2000, 20261107


def write_once(name, rep):
    f = OUT / f"{name}_verdict.json"
    rep["input_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(OUT.glob("*.csv"))}
    if f.exists():
        sys.exit(f"판정은 1회만 -- {f}")
    f.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    print(json.dumps(rep, ensure_ascii=False, indent=1, default=float))


def extract():
    OUT.mkdir(parents=True, exist_ok=True)
    n = sqlite3.connect("file:data/hot/news_rss.sqlite?mode=ro", uri=True, timeout=10)
    pd.read_sql("select source,first_seen_ms,backfill from items", n).to_csv(OUT / "news.csv", index=False)
    pd.read_sql("select judged_ms,impact,relevant from judgments", n).to_csv(OUT / "judg.csv", index=False)
    h = sqlite3.connect("file:data/hot/hl_ctx.sqlite?mode=ro", uri=True, timeout=10)
    t = sqlite3.connect("file:data/hot/binance_tape.sqlite?mode=ro", uri=True, timeout=10)
    t0 = pd.read_sql("select min(recv_ms) m from hl_asset_ctx where coin='ETH'", h).m[0] // 1000
    for coin, bucket in (("ETH", 0.1), ("BTC", 1.0)):
        pd.read_sql(f"select recv_ms,mid_px from hl_asset_ctx where coin='{coin}'", h).to_csv(OUT / f"hl_{coin}.csv", index=False)
        pd.read_sql(f"""select ts_sec, sum(price_bin*(buy_qty+sell_qty))/sum(buy_qty+sell_qty)*{bucket} px from trade_tape_1s
                        where symbol='{coin.lower()}usdt' and ts_sec>={t0} and buy_qty+sell_qty>0 group by ts_sec order by ts_sec""", t).to_csv(OUT / f"bn_{coin}.csv", index=False)
    print({p.name: p.stat().st_size for p in OUT.glob("*.csv")})


def fetch_cm():
    OUT.mkdir(parents=True, exist_ok=True)
    for a in ("btc", "eth"):
        rows, url = [], ("https://community-api.coinmetrics.io/v4/timeseries/asset-metrics?assets=" + a
                         + "&metrics=FlowInExNtv,FlowOutExNtv,SplyCur,PriceUSD&frequency=1d&page_size=10000&start_time=2020-01-01")
        while url:
            j = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research-client"}), timeout=60))
            rows += j.get("data", [])
            url = j.get("next_page_url")
        d = pd.DataFrame(rows)
        d["date"] = pd.to_datetime(d.pop("time")).dt.tz_localize(None).dt.normalize()
        d = d.drop(columns="asset").set_index("date")[["FlowInExNtv", "FlowOutExNtv", "SplyCur", "PriceUSD"]].apply(pd.to_numeric, errors="coerce").sort_index()
        d.to_csv(OUT / f"cm_{a}.csv")
        print(a, len(d), d.index.min().date(), d.index.max().date(), d.isna().sum().to_dict())


# ───────── N1 ─────────
def hourly_rv(panel_close):
    r = np.log(panel_close).diff()
    return (r ** 2).resample("1h", label="left", closed="left").sum(min_count=6)


def news_frame(rv, x):
    """행 H: H 시간 끝에 결정. x = H 시간 안 값. 표적 = log RV(H+1)."""
    f = pd.DataFrame({"rv": rv}).dropna()
    f = f[f.rv > 0]
    lr = np.log(f.rv)
    f["a1"], f["a24"], f["a168"] = lr, np.log(f.rv.rolling(24).mean()), np.log(f.rv.rolling(168, min_periods=120).mean())
    # 수정 1: 표적 시각(H+1)의 직전 14일 평균. H−23 = 어제 같은 표적 시각이 행 H 에서 쓸 수 있는 가장 최근 값
    f["seas"] = lr.shift(23).groupby((f.index + pd.Timedelta("1h")).hour).transform(lambda s: s.rolling(14, min_periods=7).mean())
    f["wknd"] = ((f.index + pd.Timedelta("1h")).dayofweek >= 5).astype(float)   # 수정 1: 뉴스 건수가 요일 대리가 되지 않게
    f["y"] = lr.shift(-1)
    f["x"] = x.reindex(f.index)
    return f


def walk_mse(f, cols, min_fit=14 * 24):
    g = f.dropna(subset=cols + ["y"])
    X = np.column_stack([np.ones(len(g))] + [g[c].to_numpy() for c in cols])
    y = g.y.to_numpy()
    out = {}
    for i in range(min_fit, len(g)):
        b = np.linalg.lstsq(X[: i - 1], y[: i - 1], rcond=None)[0]   # 표적이 다음 시간이라 i−1 행까지 끝났다
        out[g.index[i]] = (y[i] - X[i] @ b) ** 2
    return pd.Series(out)


BASE = ["a1", "a24", "a168", "seas", "wknd"]


def n1_stat(rv, x):
    f = news_frame(rv, x)
    a, b = walk_mse(f.dropna(subset=["x"]), BASE), walk_mse(f, BASE + ["x"])
    d = (b - a).dropna()
    return d, a, b


def boot_days(d):
    w = [g.to_numpy() for _, g in d.groupby(d.index.floor("D"))]
    rng = np.random.default_rng(SEED)
    return np.array([np.concatenate([w[j] for j in rng.integers(0, len(w), len(w))]).mean() for _ in range(BOOT)])


def n1():
    px = pd.read_parquet("data/live/dirshort_panel/ETHUSDT.parquet", columns=["timestamp", "close"]).set_index("timestamp").close.sort_index()
    rv = hourly_rv(px[px.index >= "2026-09-01"])
    news = pd.read_csv(OUT / "news.csv")
    news = news[news["backfill"] == 0]
    t = pd.to_datetime(news.first_seen_ms, unit="ms").dt.floor("h")
    cnt = t.value_counts().sort_index().reindex(pd.date_range("2026-10-05", rv.index.max(), freq="1h")).fillna(0)
    x = np.log1p(cnt)
    d, a, b = n1_stat(rv, x)
    q = boot_days(d)
    up = float(np.quantile(q, .90))
    # 수정 1: k 마다 실제 판도 같은 시작점으로 잘라 비교(평가·적합 행 일치). k=7 은 요일 위상이 같은 판이라 반드시 이겨야 한다
    pl, wins = [], {}
    for k in range(1, 10):
        xk = x[x.index >= x.index[24 * k]]
        real_k = float(n1_stat(rv, xk)[0].mean())
        plk = float(n1_stat(rv, np.log1p(cnt.shift(24 * k).dropna()))[0].mean())
        pl.append({"k": k, "real": real_k, "placebo": plk})
        wins[k] = real_k < plk
    beat = sum(wins.values())
    half = len(d) // 2
    rep = {"n_hours": len(d), "n_days": int(d.index.floor("D").nunique()), "news_items": int(len(news)), "mse_A": float(a.reindex(d.index).mean()), "mse_B": float(b.reindex(d.index).mean()), "dmse": float(d.mean()), "upper90": up,
           "p": float((q >= 0).mean()), "placebo": pl, "placebo_beaten": f"{beat}/9", "k7_win": wins[7], "halves": [float(d.iloc[:half].mean()), float(d.iloc[half:].mean())], "verdict": "통과" if up < 0 and beat >= 8 and wins[7] else "불통과"}
    j = pd.read_csv(OUT / "judg.csv")
    jt = pd.to_datetime(j.judged_ms, unit="ms").dt.floor("h")
    imp = pd.to_numeric(j.impact, errors="coerce").groupby(jt).sum().reindex(cnt.index).fillna(0)
    imp = imp[imp.index <= jt.max()]      # 수정 1: 판정 정지(10-27) 뒤를 가짜 0 으로 채우지 않는다
    try:
        d2, _, _ = n1_stat(rv, np.log1p(imp.clip(lower=0)))
        rep["aux_llm_impact"] = {"dmse": float(d2.mean()), "n_hours": len(d2), "note": "판정 정지 전까지"}
    except ValueError:
        rep["aux_llm_impact"] = {"note": "적합 최소 행 미달로 표본 없음"}
    pb = pd.read_parquet("data/live/dirshort_panel/BTCUSDT.parquet", columns=["timestamp", "close"]).set_index("timestamp").close.sort_index()
    d3, _, _ = n1_stat(hourly_rv(pb[pb.index >= "2026-09-01"]), x)
    rep["aux_btc"] = {"dmse": float(d3.mean()), "upper90": float(np.quantile(boot_days(d3), .90))}
    write_once("n1", rep)


# ───────── N2 ─────────
def events(b, q=.995, gap=60):
    """b: 초 인덱스 가격. 5초 로그 수익 절대값이 전날까지 q 분위 이상. 반환 (사건 초, 부호)."""
    s = b.reindex(range(int(b.index[0]), int(b.index[-1]) + 1)).ffill(limit=2)
    r = np.log(s / s.shift(5)).dropna()
    day = r.index.to_numpy() // 86400
    thr = {}
    for d in np.unique(day)[1:]:
        thr[d] = float(np.quantile(np.abs(r[day < d]), q))
    out, last = [], -10**12
    for t, v, dd in zip(r.index, r.to_numpy(), day):
        th = thr.get(dd)
        if th is not None and abs(v) >= th and t - last >= gap:
            out.append((t, np.sign(v)))
            last = t
    return out


def n2_trades(ev, hl_ms, hl_px, delay=2, hold=30, cost=9.0):
    rows = []
    for t, s in ev:
        i = np.searchsorted(hl_ms, (t + 1 + delay) * 1000)    # t 초가 끝난 시각 = t+1
        j = np.searchsorted(hl_ms, (t + 1 + delay + hold) * 1000)
        if i >= len(hl_ms) or j >= len(hl_ms) or hl_ms[i] - (t + 1 + delay) * 1000 > 5000 or hl_ms[j] - (t + 1 + delay + hold) * 1000 > 5000:
            continue
        g = s * (hl_px[j] / hl_px[i] - 1) * 1e4
        rows.append((t // 86400, g, g - cost))
    return np.array(rows).reshape(-1, 3)


def cluster_ci(day, x, B=4000):
    u, inv = np.unique(day, return_inverse=True)
    s, m = np.bincount(inv, x), np.bincount(inv)
    idx = np.random.default_rng(SEED).integers(0, len(u), (B, len(u)))
    d = s[idx].sum(1) / m[idx].sum(1)
    return [float(np.quantile(d, .025)), float(np.quantile(d, .975))]


def n2_one(coin):
    b = pd.read_csv(OUT / f"bn_{coin}.csv").set_index("ts_sec").px
    h = pd.read_csv(OUT / f"hl_{coin}.csv").sort_values("recv_ms")
    ms, px = h.recv_ms.to_numpy(), h.mid_px.to_numpy()
    ev = events(b)
    tr = n2_trades(ev, ms, px)
    rep = {"events": len(ev), "n": len(tr), "days": int(len(np.unique(tr[:, 0]))) if len(tr) else 0, "gross_bp": float(tr[:, 1].mean()) if len(tr) else None, "net_bp": float(tr[:, 2].mean()) if len(tr) else None,
           "ci95_net": cluster_ci(tr[:, 0], tr[:, 2]) if len(tr) >= 10 else None}
    if len(tr):
        day_means = pd.Series(tr[:, 2]).groupby(tr[:, 0]).mean()
        rep["days_positive_net"] = f"{int((day_means > 0).sum())}/{len(day_means)}"
    if len(tr) < 10:
        return rep | {"note": "표본 없음"}
    rep["aux"] = {f"delay{dl}_hold{hd}": float(n2_trades(ev, ms, px, dl, hd)[:, 1].mean()) for dl, hd in ((0, 30), (1, 30), (5, 30), (2, 10), (2, 60))}
    rep["pos_control_gross_before_event"] = float(n2_trades(ev, ms, px, -7, 30)[:, 1].mean())   # 사건 전 mid 로 진입(미래를 앎)
    assert rep["pos_control_gross_before_event"] > rep["gross_bp"] + 1, ("양성 대조 실패", rep)
    return rep


def n2():
    rep = {"ETH": n2_one("ETH"), "aux_BTC": n2_one("BTC")}
    e = rep["ETH"]
    rep["verdict"] = "통과" if e["ci95_net"] and e["ci95_net"][0] > 0 else "불통과"
    write_once("n2", rep)


# ───────── N3 ─────────
def n3_signal(d):
    x = ((d.FlowInExNtv - d.FlowOutExNtv) / d.SplyCur).rolling(7).sum()
    z = (x - x.rolling(365, min_periods=300).mean()) / x.rolling(365, min_periods=300).std()
    return -z.shift(2)     # 결정일 d 에 d−2 행까지


def n3_ic(sig, fwd, a, b):
    g = pd.DataFrame({"s": sig, "f": fwd}).loc[a:b].dropna()
    return g, float(g.s.rank().corr(g.f.rank()))


def n3_one(d, a="2024-01-01", b=None, sig=None):
    fwd = np.log(d.PriceUSD.shift(-7) / d.PriceUSD)
    sig = n3_signal(d) if sig is None else sig
    g, ic = n3_ic(sig, fwd, a, b)
    n = len(g)
    rng = np.random.default_rng(SEED)
    st = rng.integers(0, n - 6, size=(BOOT, n // 7 + 1))
    S, F = g.s.to_numpy(), g.f.to_numpy()
    bs = []
    for row in st:
        ii = (row[:, None] + np.arange(7)).ravel()[:n]
        bs.append(pd.Series(S[ii]).rank().corr(pd.Series(F[ii]).rank()))
    lo, hi = np.quantile(bs, [.025, .975])
    nl = [pd.Series(np.roll(S, int(rng.integers(30, n - 30)))).rank().corr(g.f.reset_index(drop=True).rank()) for _ in range(1000)]
    up = g.f > 0
    return {"n_days": n, "ic": ic, "ci95": [float(lo), float(hi)], "null_pct": float((np.array(nl) < ic).mean() * 100), "base_up": f"{int(up.sum())}/{n}",
            "z_pos_hit": f"{int(up[g.s >= 1].sum())}/{int((g.s >= 1).sum())}", "z_neg_hit": f"{int((~up)[g.s <= -1].sum())}/{int((g.s <= -1).sum())}"}


def n3():
    rep = {}
    for a in ("btc", "eth"):
        d = pd.read_csv(OUT / f"cm_{a}.csv", parse_dates=["date"]).set_index("date")
        rep[a] = n3_one(d) | {"data_end": str(d.index.max().date())}
        rep[a + "_2022_23_aux"] = n3_one(d, "2022-01-01", "2023-12-31")
        rep[a + "_control_positive"] = n3_one(d, sig=np.log(d.PriceUSD.shift(-7) / d.PriceUSD))["ci95"]
        rep[a + "_control_negative_ic"] = n3_one(d, sig=pd.Series(np.random.default_rng(1).normal(size=len(d)), index=d.index))["ic"]
    ok = all(rep[a]["ci95"][0] > 0 and rep[a]["null_pct"] > 95 for a in ("btc", "eth"))
    rep["verdict"] = "통과" if ok else "불통과"
    assert all(rep[a + "_control_positive"][0] > .5 for a in ("btc", "eth")), "양성 대조 실패"
    write_once("n3", rep)


def selftest():
    # N1: 표적은 다음 시간, 적합은 끝난 행만 -- 마지막 표적을 망가뜨려도 그 이전 오차는 불변
    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-01-01", periods=24 * 40, freq="1h")
    rv = pd.Series(np.exp(rng.normal(-12, .5, len(idx))), index=idx)
    x = pd.Series(rng.normal(size=len(idx)), index=idx)
    f = news_frame(rv, x)
    assert f.y.iloc[10] == np.log(rv.iloc[11]) and f.a1.iloc[10] == np.log(rv.iloc[10])
    e0 = walk_mse(f, BASE)
    f2 = f.copy(); f2.iloc[-2, f2.columns.get_loc("y")] = 99.0
    e1 = walk_mse(f2, BASE)
    assert np.allclose(e0.iloc[:-2], e1.iloc[:-2]), "미래 표적이 적합에 들어감"
    # 수정 1: 중간 행 표적을 망가뜨려도 그 행 앞뒤 오차는 그대로여야(그 뒤 첫 행 적합은 j−1 까지 -- X[:i+1] 변이를 잡는다)
    g_ = f.dropna(subset=BASE + ["y"])
    j_ = 24 * 20
    f3 = g_.copy(); f3.iloc[j_, f3.columns.get_loc("y")] = 99.0
    a0, a3 = walk_mse(g_, BASE), walk_mse(f3, BASE)
    pos = a0.index.get_loc(g_.index[j_])
    assert abs(a0.iloc[pos - 1] - a3.iloc[pos - 1]) < 1e-12 and abs(a0.iloc[pos + 1] - a3.iloc[pos + 1]) < 1e-12, "적합 경계 오류"
    # 수정 1: seas 는 표적 시각의 값만·행 H 의 값은 안 쓴다
    r2 = rv.copy(); r2.iloc[500] *= 1000
    fs, fs2 = news_frame(rv, x), news_frame(r2, x)
    assert fs.seas.iloc[500] == fs2.seas.iloc[500] and fs.seas.iloc[522] == fs2.seas.iloc[522] and fs.seas.iloc[523] != fs2.seas.iloc[523], "seas 시각 오류"
    # 양성·음성: 표적을 담은 x 는 크게 개선, 난수는 ~0 이상
    ff = np.log(rv).shift(-1)
    assert n1_stat(rv, ff)[0].mean() < -.1 and n1_stat(rv, x)[0].mean() > -.01
    # N2: 전날까지 문턱·방향·지연
    t = np.arange(86400 * 3)
    p = 100 + np.cumsum(rng.normal(0, .001, len(t)))
    p[86400 * 2 + 100:] += 1.0                         # 3일째 큰 상승
    b = pd.Series(p, index=t)
    ev = events(b)
    assert any(86400 * 2 + 100 <= tt <= 86400 * 2 + 106 and s > 0 for tt, s in ev), "사건 미검출"
    assert all(tt >= 86400 for tt, _ in ev), "첫날은 문턱이 없어야 한다"
    # 손계산: t=1000 상승 사건, HL 은 진입 시각에 100, 30초 뒤 101 -> gross 100bp, net 91bp
    ms = np.arange(0, 2_000_000, 1000); hp = np.full(len(ms), 100.0); hp[ms >= 1000 * 1033] = 101.0
    tr = n2_trades([(1000, 1.0)], ms, hp)
    assert abs(tr[0, 1] - 100) < 1e-9 and abs(tr[0, 2] - 91) < 1e-9, tr
    # N3: d−2 시차
    d = pd.DataFrame({"FlowInExNtv": rng.normal(10, 1, 900), "FlowOutExNtv": rng.normal(10, 1, 900), "SplyCur": 1e3, "PriceUSD": np.exp(np.cumsum(rng.normal(0, .02, 900)))}, index=pd.date_range("2023-01-01", periods=900))
    s0 = n3_signal(d); d2 = d.copy(); d2.iloc[800, 0] += 500
    s1 = n3_signal(d2)
    assert s0.iloc[801] == s1.iloc[801] and s0.iloc[802] != s1.iloc[802]
    print("selftest OK")


if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else ""
    if c == "--selftest":
        selftest()
    elif c in ("extract", "fetch_cm", "n1", "n2", "n3"):
        globals()[c]()
    else:
        sys.exit(__doc__)
