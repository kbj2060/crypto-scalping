#!/usr/bin/env python3
"""상위 10% 롱 깨뜨리기 검증 (사전등록 docs/prereg/dir_top_decile_stress_20261107.md).
    python scripts/research_dir_top_decile_stress_20261107.py --selftest
    python scripts/research_dir_top_decile_stress_20261107.py --fetch   # 정적 아카이브에서 펀딩 받기
    python scripts/research_dir_top_decile_stress_20261107.py --judge   # 1회만
"""
import io, json, sys, urllib.error, urllib.request, zipfile
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_dir_top_decile_20261107 as T  # noqa: E402

OUT = Path("/home/kbj20/crypto-scalping/tmp/dir_top_decile_stress_20261107")
PREDS = Path("/home/kbj20/crypto-scalping/.claude/worktrees/agent-ab3571256711ceb43/tmp/btc_sol_xrp_dir4h_20261031/preds.npz")
HOLD, COINS, SEED = T.HOLD, T.COINS, 20261107
ARM = {"ETH": "ETH_S", "SOL": "SOL_E", "XRP": "XRP_J"}


def vect(d, sig, delay=0, pen_bp=0.0, mk=2.0, fb=5.0):
    """하네스 없는 재구현. sig[t]≠0 → 지정가 c[t+delay], 봉 t+delay+1 관통, 청산 지정가 c[t+delay+48], 다음 봉 관통 못 하면 그다음 종가 시장가."""
    c, h, l = d.close.to_numpy(), d.high.to_numpy(), d.low.to_numpy()
    t = np.flatnonzero(sig) + delay
    t = t[t + HOLD + 2 < len(c)]
    px = c[t]
    filled = l[t + 1] < px * (1 - pen_bp / 1e4)
    t, px = t[filled], px[filled]
    ex = t + HOLD
    xp = c[ex]
    ok = h[ex + 1] > xp * (1 + pen_bp / 1e4)
    xf = np.where(ok, xp, c[ex + 1])
    ex_i = np.where(ok, ex, ex + 1)
    gross = (xf / px - 1) * 1e4
    return pd.DataFrame({"i": t, "ex": ex_i, "t": d.index[t], "gross": gross, "net": gross - mk - np.where(ok, mk, fb)})


def block_ci(day, x, L, B=4000):
    """L 일 블록 부트(거래당 평균). 일 합·건수를 블록으로 뽑는다."""
    s = pd.DataFrame({"d": day, "x": x}).groupby("d").x.agg(["sum", "count"])
    full = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(full).fillna(0.0)
    su, cn = s["sum"].to_numpy(), s["count"].to_numpy()
    n = len(su)
    rng = np.random.default_rng(SEED)
    k = n // L + 1
    st = rng.integers(0, n - L + 1, size=(B, k))
    ii = (st[..., None] + np.arange(L)).reshape(B, -1)[:, :n]
    v = su[ii].sum(1) / np.maximum(cn[ii].sum(1), 1)
    return [float(np.quantile(v, .025)), float(np.quantile(v, .975))]


def fetch():
    OUT.mkdir(parents=True, exist_ok=True)
    for c in COINS:
        rows = []
        for ym in pd.period_range("2022-12", "2026-10", freq="M"):
            n = f"{c}USDT-fundingRate-{ym}"
            u = f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{c}USDT/{n}.zip"
            try:
                z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "research-client"}), timeout=60).read()))
            except urllib.error.HTTPError as e:
                print(c, ym, e.code)
                continue
            rows.append(pd.read_csv(io.BytesIO(z.read(n + ".csv"))))
        d = pd.concat(rows)
        d.to_csv(OUT / f"funding_{c}.csv", index=False)
        print(c, len(d), list(d.columns), pd.to_datetime(d.calc_time, unit="ms").min(), pd.to_datetime(d.calc_time, unit="ms").max(), float(d.last_funding_rate.mean()))


def add_funding(tr, d, coin):
    """롱이 치른 펀딩 bp: (진입 봉 끝, 청산 봉 끝] 안의 정산 합. 양수 펀딩 = 롱 지불."""
    f = pd.read_csv(OUT / f"funding_{coin}.csv")
    ts = pd.to_datetime(f.calc_time, unit="ms").to_numpy()
    o = np.argsort(ts)
    ts, r = ts[o], f.last_funding_rate.to_numpy()[o]
    cum = np.r_[0.0, np.cumsum(r)]
    bar_end = (d.index + pd.Timedelta("5min")).to_numpy()
    a = np.searchsorted(ts, bar_end[tr.i.to_numpy()], side="right")
    b = np.searchsorted(ts, bar_end[tr.ex.to_numpy()], side="right")
    return tr.assign(funding=(cum[b] - cum[a]) * 1e4)


def pooled(tr):
    return {"n": len(tr), "mean": float(tr.net.mean()), "gross": float(tr.gross.mean()), "ci5d": block_ci(tr.t.dt.floor("D"), tr.net.to_numpy(), 5), "ci20d": block_ci(tr.t.dt.floor("D"), tr.net.to_numpy(), 20)}


def daily_book(per):
    """코인별 자본 1/3, 위상 96개에 1/96씩 → 일 손익(%). 진입일 귀속(서술용)."""
    out = {c: (v.groupby(v.t.dt.floor("D")).net.sum() / T.PHASES / 1e4).rename(c) for c, v in per.items()}
    return pd.concat(out.values(), axis=1).fillna(0.0)


def judge():
    vp = OUT / "verdict.json"
    if vp.exists():
        sys.exit(f"판정은 1회만 -- {vp}")
    T.set_cost(T.STRICT)
    D, S = T.prepare()
    v0 = json.load(open(T.OUT / "verdict.json"))
    per = {c: vect(D[c], S[c]) for c in COINS}
    all_ = pd.concat([v.assign(coin=c) for c, v in per.items()])
    rep = {"prereg": "docs/prereg/dir_top_decile_stress_20261107.md"}
    # K0 재구성
    want = v0["strict"]["pooled"]
    rep["K0"] = {"n": len(all_), "want_n": want["n"], "mean": float(all_.net.mean()), "want_mean": want["mean_net"]}
    assert rep["K0"]["n"] == want["n"] and abs(rep["K0"]["mean"] - want["mean_net"]) < 1e-6, ("재구성 불일치", rep["K0"])
    # K1
    rep["K1"] = pooled(all_)
    # K2
    fu = pd.concat([add_funding(per[c], D[c], c).assign(coin=c) for c in COINS])
    fu["net"] = fu.net - fu.funding
    rep["K2"] = pooled(fu) | {"funding_bp_per_trade": float(fu.funding.mean()), "by_coin": {c: float(g.funding.mean()) for c, g in fu.groupby("coin")}}
    # K3
    p2 = pd.concat([vect(D[c], S[c], pen_bp=2.0) for c in COINS])
    p5 = pd.concat([vect(D[c], S[c], pen_bp=5.0) for c in COINS])
    rep["K3"] = pooled(p2) | {"fill_share_vs_base": len(p2) / len(all_)}
    rep["aux_pen5"] = pooled(p5) | {"fill_share_vs_base": len(p5) / len(all_)}
    # K4
    d1 = pd.concat([vect(D[c], S[c], delay=1) for c in COINS])
    rep["K4"] = pooled(d1) | {"ratio_to_base": float(d1.net.mean() / all_.net.mean())}
    # K5
    rep["K5"] = {"ex_year": {str(y): float(all_[all_.t.dt.year != y].net.mean()) for y in sorted(all_.t.dt.year.unique())}, "ex_coin": {c: float(all_[all_.coin != c].net.mean()) for c in COINS}}
    # K6
    P = np.load(PREDS, allow_pickle=True)
    ts = pd.to_datetime(P["ts"])
    rp, rho = [], {}
    for c in COINS:
        s = pd.Series(P[ARM[c]], index=ts)
        p = s.reindex(D[c].index)
        rho[c] = float(p.corr(D[c].p))
        sig = T.top_signal(p)
        sig[-(HOLD + 80):] = 0
        rp.append(vect(D[c], sig))
    rep["K6"] = pooled(pd.concat(rp)) | {"corr_with_seed": rho}
    # K7 시장 조정 알파
    book = daily_book(per)
    full = pd.date_range(book.index.min(), book.index.max(), freq="D")
    book = book.reindex(full).fillna(0.0)
    mkt = pd.concat([D[c].close.resample("1D").last().pct_change().rename(c) for c in COINS], axis=1).reindex(full).mean(axis=1).fillna(0.0)
    port = book.mean(axis=1)
    X_ = np.column_stack([np.ones(len(port)), mkt.to_numpy()])
    b = np.linalg.lstsq(X_, port.to_numpy(), rcond=None)[0]
    res = port.to_numpy() - b[1] * mkt.to_numpy()   # 베타를 고정한 알파 시계열
    n, L = len(res), 20
    rng = np.random.default_rng(SEED)
    st = rng.integers(0, n - L + 1, size=(4000, n // L + 1))
    al = res[(st[..., None] + np.arange(L)).reshape(4000, -1)[:, :n]].mean(1)
    rep["K7"] = {"beta": float(b[1]), "alpha_bp_day": float(res.mean() * 1e4), "ci20d": [float(np.quantile(al, .025) * 1e4), float(np.quantile(al, .975) * 1e4)]}
    # 보고: 포트폴리오
    ann = np.sqrt(365)
    eq = (1 + port).cumprod()
    hold = (1 + mkt).cumprod()
    mdd = lambda e: float((e / e.cummax() - 1).min())
    rep["portfolio"] = {"sharpe": float(port.mean() / port.std() * ann), "ann_return": float(port.mean() * 365), "mdd": mdd(eq), "hold_sharpe": float(mkt.mean() / mkt.std() * ann), "hold_mdd": mdd(hold), "hold_ann": float(mkt.mean() * 365), "corr_mkt": float(np.corrcoef(port, mkt)[0, 1]),
                        "exposure": float(np.mean([(S[c] != 0).mean() for c in COINS])), "years": {str(y): {"sharpe": float(g.mean() / g.std() * ann), "ret": float(g.sum()), "hold_ret": float(mkt[g.index].sum())} for y, g in port.groupby(port.index.year)}}
    # 보고: 단순 규칙
    naive = {}
    for nm, h in (("past4h_low10", 48), ("past24h_low10", 288)):
        trs = []
        for c in COINS:
            r = (D[c].close / D[c].close.shift(h) - 1).where(D[c].p.notna())
            sig = T.top_signal(-r)
            sig[-(HOLD + 80):] = 0
            trs.append(vect(D[c], sig))
        naive[nm] = pooled(pd.concat(trs))
    rep["aux_naive"] = naive
    # 보고: DSR
    sys.path.insert(0, "scripts")
    from scipy.stats import norm
    sd_d = port.std()
    sr = port.mean() / sd_d
    g3 = float(((port - port.mean()) ** 3).mean() / sd_d ** 3)
    g4 = float(((port - port.mean()) ** 4).mean() / sd_d ** 4)
    T_ = len(port)
    em = .5772156649
    v = (.8 / ann) ** 2
    dsr = {}
    for N in (50, 400):
        sr0 = np.sqrt(v) * ((1 - em) * norm.ppf(1 - 1 / N) + em * norm.ppf(1 - 1 / (N * np.e)))
        dsr[str(N)] = float(norm.cdf((sr - sr0) * np.sqrt(T_ - 1) / np.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))))
    rep["aux_dsr"] = dsr
    # 보고: 2026
    a26 = fu[fu.t.dt.year == 2026]
    rep["aux_2026_with_funding"] = pooled(a26)
    # 판정
    k = {"K1": rep["K1"]["ci5d"][0] > 0 and rep["K1"]["ci20d"][0] > 0, "K2": rep["K2"]["ci5d"][0] > 0, "K3": rep["K3"]["mean"] > 0 and rep["K3"]["ci5d"][0] > 0,
         "K4": rep["K4"]["mean"] > 0 and rep["K4"]["ratio_to_base"] >= .5, "K5": all(x > 0 for x in rep["K5"]["ex_year"].values()) and all(x > 0 for x in rep["K5"]["ex_coin"].values()), "K6": rep["K6"]["ci5d"][0] > 0, "K7": rep["K7"]["ci20d"][0] > 0}
    rep["gates"] = k
    rep["verdict"] = "전진 후보" if all(k.values()) else "후보 탈락"
    json.dump(rep, open(vp, "w"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(rep, ensure_ascii=False, indent=1, default=float))


def selftest():
    n = 60
    idx = pd.date_range("2026-01-01", periods=n, freq="5min")
    c = np.full(n, 100.0); c[50:] = 101.0
    d = pd.DataFrame({"o": c, "high": c + 2, "low": c - 2, "close": c}, index=idx)
    s = np.zeros(n, int); s[2] = 1
    tr = vect(d, s)
    # 지정가 100, 청산 봉 50 종가 101 지정가, 봉 51 고가 103 관통 → gross 100, net 96
    assert len(tr) == 1 and tr.ex[0] == 50 and abs(tr.gross[0] - 100) < 1e-9 and abs(tr.net[0] - 96) < 1e-9, tr
    assert len(vect(d, s, pen_bp=500.0)) == 0                # 저가가 충분히 안 내려오면 미체결
    assert vect(d, s, delay=1).i[0] == 3
    # 펀딩: (봉 끝, 청산 봉 끝] 안 정산만, 양수 = 롱 지불
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "funding_TST.csv"
    pd.DataFrame({"calc_time": [(idx[1] + pd.Timedelta("5min")).value // 10**6, (idx[10]).value // 10**6, (idx[51] + pd.Timedelta("5min")).value // 10**6], "last_funding_rate": [.001, .0002, .001]}).to_csv(tmp, index=False)
    f = add_funding(tr, d, "TST")
    assert abs(f.funding[0] - 2.0) < 1e-9, f       # 가운데 하나만(.0002 = 2bp)
    tmp.unlink()
    # 블록 CI: 평균을 감싼다
    day = pd.Series(pd.date_range("2026-01-01", periods=200, freq="D"))
    x = np.random.default_rng(0).normal(5, 1, 200)
    lo, hi = block_ci(day, x, 5)
    assert lo < 5.1 and hi > 4.9 and hi - lo < 1
    print("selftest OK")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--selftest" in a:
        selftest()
    elif "--fetch" in a:
        fetch()
    elif "--judge" in a:
        judge()
    else:
        sys.exit(__doc__)
