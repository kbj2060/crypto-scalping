#!/usr/bin/env python3
"""«추세 방향 + 2배·변동성 크기 + −5% 손절 + 지정가» 한 시스템 — 사전 등록 docs/prereg/system_trend_ruleb_20261025.md (042e0067).

    python3 scripts/research_system_trend_ruleb_20261025.py --selfcheck   # S1 보유 일치 · S2 누수/지연 · S3 원장 재현
    python3 scripts/research_system_trend_ruleb_20261025.py               # 4코인 G1~G6 · 실원장 L1·L2

데이터: panel 5분(~2026-09-14) + 바이낸스 공개 REST 5분·펀딩(캐시 tmp/system_trend_ruleb_20261025/rest_*.parquet).
실원장 1분 경로: tmp/system_trend_ruleb_20261025/eth1m_ledger.parquet (10-24 손실 장치 재생과 같은 파일).
"""
import argparse, json, math, pathlib, sys, time, urllib.request
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = pathlib.Path("/home/kbj20/crypto-scalping/data")
OUT = ROOT / "tmp" / "system_trend_ruleb_20261025"
COINS = ("ETH", "BTC", "SOL", "XRP")
START, END = "2022-02-01", "2026-10-23"
LMAX, ADD, STOP, TAKER, THRU, WIN = 2.0, 0.0075, 0.05, 4e-4, 1e-4, 12


def rest(path, **q):
    url = "https://fapi.binance.com" + path + "?" + "&".join(f"{k}={v}" for k, v in q.items())
    for i in range(5):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                return json.loads(r.read())
        except Exception:
            time.sleep(2 * (i + 1))
    raise RuntimeError(url)


def rest_klines(sym, t0, t1):
    f = OUT / f"rest_k5_{sym}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    rows, t = [], t0
    while t < t1:
        part = rest("/fapi/v1/klines", symbol=f"{sym}USDT", interval="5m", startTime=t, limit=1500)
        if not part:
            break
        rows += part; t = part[-1][0] + 300000; time.sleep(0.3)
    x = pd.DataFrame([(r[0], float(r[2]), float(r[3]), float(r[4]), r[6]) for r in rows], columns=["t", "h", "l", "c", "ct"]).drop_duplicates("t")
    x = x[x.t < t1].copy(); x.to_parquet(f); return x


def rest_funding(sym, t0, t1):
    f = OUT / f"rest_fund_{sym}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    rows, t = [], t0
    while t < t1:
        part = rest("/fapi/v1/fundingRate", symbol=f"{sym}USDT", startTime=t, limit=1000)
        if not part:
            break
        rows += part; t = part[-1]["fundingTime"] + 1; time.sleep(0.3)
    x = pd.DataFrame([(r["fundingTime"], float(r["fundingRate"])) for r in rows], columns=["t", "rate"]).drop_duplicates("t")
    x = x[x.t < t1]; x.to_parquet(f); return x


def bars5(sym):
    """5분봉 h,l,c (봉 시작 시각 색인), 2022-01-01~END. 빈 봉은 직전 종가로 채운다(개수 보고)."""
    p = pd.read_parquet(D / f"binance_vision/panel/{sym}USDT.parquet", columns=["timestamp", "high", "low", "close"])
    p = p.set_index("timestamp").rename(columns={"high": "h", "low": "l", "close": "c"})
    t0 = int(p.index[-1].value // 10**6) + 300000; t1 = int(pd.Timestamp(END).value // 10**6) + 86400000
    r = rest_klines(sym, t0, t1)
    r = r[r.ct < time.time() * 1000]   # 완결봉만
    r = r.set_index(pd.to_datetime(r.t, unit="ms"))[["h", "l", "c"]]
    x = pd.concat([p, r]); x = x[~x.index.duplicated(keep="first")].sort_index()
    idx = pd.date_range("2022-01-01", pd.Timestamp(END) + pd.Timedelta("23:55:00"), freq="5min")
    miss = int((~idx.isin(x.index)).sum()); x = x.reindex(idx)
    x["c"] = x.c.ffill(); x["h"] = x.h.fillna(x.c); x["l"] = x.l.fillna(x.c)
    return x, miss


def daily_close(x):
    c = x.c.values.reshape(-1, 288)[:, -1]
    return pd.Series(c, pd.date_range(x.index[0], periods=len(c), freq="D"))   # 일 d 종가(d 23:55 봉)


def daily_inputs(x):
    """일 d 결정 입력(d 00:00 직전까지): P0 = 전날 종가 · 추세 D · lvl."""
    close = daily_close(x)
    D = np.sign(sum(np.sign(close / close.shift(k) - 1) for k in (7, 14, 21, 28)) / 4)
    lr = np.diff(np.log(x.c.values), prepend=np.log(x.c.values[0]))
    rv = pd.Series(np.sqrt(pd.Series(lr ** 2).rolling(288).sum().values.reshape(-1, 288)[:, -1]), close.index)   # 일 d 끝까지 24h
    lvl = rv / rv.shift(1).rolling(30).median()                       # 직전 30일 같은 값의 중앙
    return pd.DataFrame({"P0": close, "D": D, "lvl": lvl}).shift(1)    # 일 d 의 결정 = 일 d−1 종가 기준


def simulate(data, inputs, fund, days, ncoin, Dover=None, ladder=True, stop=True, fees=True, funding=True, L_fixed=None):
    """일 바깥·코인 안쪽 순차. 순자산 공유. 반환 = 일별 포트폴리오 수익률(일 d 시작 순자산 대비)."""
    st = {s: dict(q=0.0, C=0.0, dir=0, addq=0.0, addpx=0.0, stp=0.0, V=0.0) for s in data}
    E = 1.0; rets = []
    for d in days:
        E0 = E; dp = 0.0
        for s, x in data.items():
            a = st[s]; row = inputs[s].loc[d]
            k0 = x.index.get_loc(d); h = x.h.values[k0:k0 + 288]; l = x.l.values[k0:k0 + 288]; c = x.c.values[k0:k0 + 288]
            P0 = row.P0; Dn = int(Dover[s].loc[d]) if Dover is not None else (0 if np.isnan(row.D) else int(row.D))
            L = L_fixed if L_fixed is not None else (LMAX if np.isnan(row.lvl) else min(LMAX, LMAX / row.lvl))
            cap = L * E0 / ncoin
            fz = fund[s].get(d, ()) if funding else ()
            exiting = a["dir"] != 0 and Dn != a["dir"]
            entering = Dn != 0 and a["dir"] != Dn
            act = -1   # 첫 진입 체결 봉(그 뒤 봉부터 물타기·손절)
            for k in range(288):
                o = c[k - 1] if k else P0
                for kk, rate in fz:
                    if kk == k:
                        a["C"] -= a["q"] * o * rate
                if a["dir"] != 0 and k > act:
                    s_ = a["dir"]
                    if ladder and not exiting and a["addq"] and ((s_ > 0 and l[k] <= a["addpx"] * (1 - THRU)) or (s_ < 0 and h[k] >= a["addpx"] * (1 + THRU))):
                        a["q"] += a["addq"]; a["C"] -= a["addq"] * a["addpx"]; a["addq"] = 0.0
                    if stop and ((s_ > 0 and l[k] <= a["stp"]) or (s_ < 0 and h[k] >= a["stp"])):
                        px = min(o, a["stp"]) if s_ > 0 else max(o, a["stp"])
                        a["C"] += a["q"] * px - (TAKER * abs(a["q"]) * px if fees else 0)
                        a.update(q=0.0, dir=0, addq=0.0); entering = exiting = False   # 손절 뒤 다음 결정까지 현금
                        continue
                if exiting and k < WIN:
                    sell = a["q"] > 0
                    hit = (h[k] >= P0 * (1 + THRU)) if sell else (l[k] <= P0 * (1 - THRU))
                    if hit or k == WIN - 1:
                        px = P0 if hit else c[k]
                        a["C"] += a["q"] * px - (0 if hit or not fees else TAKER * abs(a["q"]) * px)
                        a.update(q=0.0, dir=0, addq=0.0); exiting = False
                if entering and not exiting and a["dir"] == 0 and k < WIN:
                    hit = (l[k] <= P0 * (1 - THRU)) if Dn > 0 else (h[k] >= P0 * (1 + THRU))
                    if hit or k == WIN - 1:
                        px = P0 if hit else c[k]
                        q1 = Dn * (cap / 3 if ladder else cap) / px
                        a["C"] -= q1 * px + (0 if hit or not fees else TAKER * abs(q1) * px)
                        a.update(q=q1, dir=Dn, stp=px * (1 - Dn * STOP), addpx=px * (1 - Dn * ADD))
                        a["addq"] = Dn * (2 * cap / 3) / a["addpx"] if ladder else 0.0
                        entering = False; act = k
            V = a["C"] + a["q"] * c[-1]; dp += V - a["V"]; a["V"] = V
        E = E0 + dp; rets.append(dp / E0)
    return pd.Series(rets, days)


sharpe = lambda r: float(r.mean() / r.std() * math.sqrt(365)) if r.std() > 0 else 0.0


def mdd(r):
    eq = (1 + r).cumprod(); return float((eq / eq.cummax() - 1).min())


def block_ci(r, b=20, n=2000, seed=7):
    rng = np.random.default_rng(seed); v = r.values; T = len(v); ms = []
    for _ in range(n):
        st = rng.integers(0, T - b, T // b + 1); ms.append(np.concatenate([v[s:s + b] for s in st])[:T].mean())
    return [float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))]


def markov_dirs(Dser, days, seed):
    """실제 D(4코인 풀) 전이 행렬로 무작위 방향 경로 — 같은 상태 비율·보유 길이, 시점만 무작위."""
    T = np.zeros((3, 3))
    for s in Dser.values():
        v = s.loc[days].fillna(0).astype(int).values
        for a_, b_ in zip(v[:-1], v[1:]):
            T[a_ + 1, b_ + 1] += 1
    T = T / T.sum(1, keepdims=True); rng = np.random.default_rng(seed); out = {}
    for s in Dser:
        cur = rng.choice(3); v = []
        for _ in days:
            v.append(cur - 1); cur = rng.choice(3, p=T[cur])
        out[s] = pd.Series(v, days)
    return out


# ---------- 실원장 (10-24 손실 장치 재생과 같은 근사) ----------
def ledger_trips():
    L = pd.DataFrame([json.loads(l) for l in open(D / "live/account_round_trips.jsonl")]); H = json.load(open(D / "live/account_trip_habits.json"))
    L = L[L.symbol.str.startswith("ETH") & L.closed.astype(bool)].copy()
    key = L.symbol + "|" + L.side + "|" + L.entry_time.astype(str)
    L["E"] = [H[k]["equity_entry"] if k in H and H[k].get("equity_entry") else (r.equity_at_record - r.net_pnl if pd.notna(r.equity_at_record) else np.nan) for k, r in zip(key, L.itertuples())]
    L["peak"] = [H[k]["peak_notional"] if k in H and H[k].get("peak_notional") else (r.peak_notional if pd.notna(r.peak_notional) else r.max_qty * r.entry_price) for k, r in zip(key, L.itertuples())]
    L = L[(L.E > 50) & (L.entry_time >= int(pd.Timestamp("2026-08-01").value // 10**6))].sort_values("entry_time").reset_index(drop=True)
    L["s"] = np.where(L.side == "LONG", 1, -1); L["R"] = L.net_pnl / L.E; L["Nin"] = L.qty_in * L.entry_price
    return L


def ledger_replay(L, K, Ls=None):
    """Ls=None → 장치 없음(원장 그대로). 아니면 크기 = Ls×순자산, 손절 = 첫 진입가(원장 평균가) ∓5%."""
    T = K.t.values; Hh, Ll, Cc = K.h.values, K.l.values, K.c.values; mi = lambda ms: int(np.searchsorted(T, ms // 60000 * 60000))
    out = []
    for i, r in enumerate(L.itertuples()):
        if Ls is None:
            out.append(r.R); continue
        scale = Ls[i] * r.E / r.peak; s, e = r.s, r.entry_price; i0, i1 = mi(r.entry_time), mi(r.exit_time)
        kb = r.exit_price / Cc[min(i1, len(Cc) - 1)]; px = None   # USDC 왕복 = USDT 경로 × 청산 시점 비율
        for j in range(i0 + 1, i1 + 1):
            if s > 0 and Ll[j] * kb <= e * (1 - STOP): px = e * (1 - STOP); break
            if s < 0 and Hh[j] * kb >= e * (1 + STOP): px = e * (1 + STOP); break
        out.append(r.R * scale if px is None else scale * (s * (px / e - 1) * r.Nin - 0.5 * r.commission - TAKER * r.Nin) / r.E)
    return np.array(out)


def ledger_lvl(L, K):
    """원장 진입 시각의 lvl: 직전 24h 5분 실현변동 ÷ 직전 30개 00:00 UTC 같은 값 중앙."""
    k5 = K.assign(b=K.t // 300000 * 300000).groupby("b").c.last(); lr = np.log(k5).diff()
    rv = np.sqrt((lr ** 2).rolling(288).sum()); rv.index = pd.to_datetime(rv.index, unit="ms") + pd.Timedelta("5min")   # 봉 끝 시각
    mid = rv[(rv.index.hour == 0) & (rv.index.minute == 0)]
    return np.array([rv[:t].iloc[-1] / mid[mid.index <= t].tail(30).median() for t in pd.to_datetime(L.entry_time, unit="ms")])


def growth_mdd(R):
    eq = np.cumprod(1 + R); return float(eq[-1]), float((eq / np.maximum.accumulate(eq) - 1).min())


def load_all():
    data, miss, inputs, fund = {}, {}, {}, {}
    t0 = int(pd.Timestamp("2022-01-01").value // 10**6); t1 = int(pd.Timestamp(END).value // 10**6) + 86400000
    for s in COINS:
        x, m = bars5(s); data[s], miss[s] = x, m; inputs[s] = daily_inputs(x)
        f = rest_funding(s, t0, t1); ft = pd.to_datetime(f.t, unit="ms").dt.floor("5min")   # 정산 시각에 시작하는 봉 직전에 낸다
        g = {}
        for tt, rate in zip(ft, f.rate):
            g.setdefault(tt.normalize(), []).append((int((tt - tt.normalize()) / pd.Timedelta("5min")), rate))
        fund[s] = g
    return data, miss, inputs, fund


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--selfcheck", action="store_true"); a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    data, miss, inputs, fund = load_all()
    for s, x in data.items():
        print(s, x.index[0], "~", x.index[-1], "빈 봉", miss[s], "펀딩 일수", len(fund[s]))
    days = pd.date_range(START, END, freq="D")
    if a.selfcheck:
        r = simulate({"ETH": data["ETH"]}, {"ETH": inputs["ETH"]}, fund, days, 1, Dover={"ETH": pd.Series(1, days)},
                     ladder=False, stop=False, fees=False, funding=False, L_fixed=1.0)
        cl = daily_close(data["ETH"]); hold = cl[END] / cl[pd.Timestamp(START) - pd.Timedelta("1D")]
        print(f"S1 항상 롱 1배 {float((1 + r).prod()):.12f} vs 보유 {hold:.12f}"); assert abs((1 + r).prod() - hold) < 1e-9
        Dl = {s: np.sign(daily_close(data[s]).pct_change()).fillna(0) for s in COINS}   # 일 d 자신의 수익 부호(미래)
        Dlag = {s: inputs[s].D.shift(1).fillna(0) for s in COINS}
        base = sharpe(simulate(data, inputs, fund, days, 4)); leak = sharpe(simulate(data, inputs, fund, days, 4, Dover=Dl))
        lag = sharpe(simulate(data, inputs, fund, days, 4, Dover=Dlag))
        print(f"S2 샤프 실제 {base:.2f} · 누수판 {leak:.2f} · 하루 지연 {lag:.2f}"); assert leak > base + 2 and lag < base + 0.5
        L = ledger_trips(); K = pd.read_parquet(OUT / "eth1m_ledger.parquet")
        assert np.allclose(ledger_replay(L, K), L.R.values); print(f"S3 원장 {len(L)}건 장치 없음 = 원장 R 일치")
        print("SELFCHECK PASS"); return
    r = simulate(data, inputs, fund, days, 4)
    hold = pd.concat([daily_close(data[s]).pct_change() for s in COINS], axis=1).mean(1).loc[days]
    nulls = [sharpe(simulate(data, inputs, fund, days, 4, Dover=markov_dirs({s: inputs[s].D for s in COINS}, days, sd))) for sd in range(200)]
    ci = block_ci(r); yr = {str(y): float((1 + v).prod() - 1) for y, v in r.groupby(r.index.year)}
    r26 = r["2026-01-01":]
    rep = {"prereg": "docs/prereg/system_trend_ruleb_20261025.md@042e0067", "period": [START, END], "days": len(r), "missing_5m_bars": miss,
           "sharpe": sharpe(r), "ann_ret": float((1 + r).prod() ** (365 / len(r)) - 1), "mdd": mdd(r), "mean_daily_ci": ci,
           "hold_sharpe": sharpe(hold), "hold_ann_ret": float((1 + hold).prod() ** (365 / len(hold)) - 1), "hold_mdd": mdd(hold),
           "null_sharpe_p95": float(np.percentile(nulls, 95)), "null_sharpe_median": float(np.median(nulls)),
           "y2026_sharpe": sharpe(r26), "y2026_ret": float((1 + r26).prod() - 1), "yearly_ret": yr,
           "yearly_hold": {str(y): float((1 + v).prod() - 1) for y, v in hold.groupby(hold.index.year)}}
    rep["gates"] = {"G1_ret_ci": rep["ann_ret"] > 0 and ci[0] > 0, "G2_mdd": rep["mdd"] >= -0.20, "G3_vs_hold": rep["sharpe"] > rep["hold_sharpe"],
                    "G4_null": rep["sharpe"] > rep["null_sharpe_p95"], "G5_2026": rep["y2026_sharpe"] > 0, "G6_years": sum(v > 0 for v in yr.values()) >= 3}
    rep["pass_4coin"] = all(rep["gates"].values())
    L = ledger_trips(); K = pd.read_parquet(OUT / "eth1m_ledger.parquet"); Ls = np.minimum(LMAX, LMAX / ledger_lvl(L, K))
    g0, m0 = growth_mdd(L.R.values); g1, m1 = growth_mdd(ledger_replay(L, K, Ls))
    r2 = simulate({"ETH": data["ETH"]}, {"ETH": inputs["ETH"]}, fund, pd.date_range("2026-08-01", "2026-10-21", freq="D"), 1)
    rep["ledger"] = {"trips": len(L), "actual_growth": g0, "actual_mdd": m0, "actual_mult_median": float(np.median(L.peak / L.E)),
                     "L1_growth": g1, "L1_mdd": m1, "L1_mult_median": float(np.median(Ls)), "L1_pass": g1 >= g0 and m1 > m0,
                     "L2_system_eth_growth": float((1 + r2).prod()), "L2_system_eth_mdd": mdd(r2)}
    (OUT / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False)); r.to_csv(OUT / "system_daily.csv")
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
