"""청산벽 «뚫는» 조건 -- 남은 후보 2개 확인 검정 (2026-10-13). 사용자 «6개월 기다리지 말고 지금 데이터로 우선 테스트».
발견 = research_eth_liq_wall_break_conditions_20261013.py(ETH). 여기선 **후보를 고를 때 안 쓴 데이터**만 판정에 쓴다.

    python3 scripts/research_liq_wall_break_confirm_20261013.py --edges   # BTC·SOL·XRP 띠(대시보드 밀도 층 재현, 10-12 V1) → 캐시
    python3 scripts/research_liq_wall_break_confirm_20261013.py --check   # ETH 2026-03 띠를 같은 경로로 다시 계산 → edges_long 캐시와 대조
    python3 scripts/research_liq_wall_break_confirm_20261013.py --test

사전등록(결과 보기 전 고정)
- 사건·라벨·피쳐 = 발견 스크립트 함수 그대로(detect·label·base_frame·finish·join_book). 다른 코인은 1분봉이 없어 **5분 경로**(48칸=4h)로 판정.
- H_호가(벽 쪽 ±1% 호가가 반대쪽보다 얇을수록 더 뚫림, book_opp_wall1): BTC·SOL bookDepth.
  주 판정 2025-01-01~2026-08-22. 5분위 경계 = 그 코인 2024 실제 사건. 통과 = 두 코인 합친 Q5−Q1 잔차(y − p0) 일 블록 95% CI > 0
  ∧ 두 코인 각각 > 0 ∧ 합친 (실제 − 위약) CI > 0. 참고(판정 아님): 2024 · ETH 08-23~10-10(새로 받은 bookDepth, 1분 라벨).
- H_청산(닿은 봉에서 밀리는 쪽 청산 ≥ 직전 7일 90분위 → 더 뚫림, 거래량 통제): BTC·SOL·XRP 실측 청산
  (lake tail_risk_1m 08-17~09-18 + lake liquidations 09-26~10-10). 로지스틱 y ~ 오프셋 logit(p0) + hot + rvol + rng_sig + dlt + r15 + r1h + h_sig
  + 코인 더미, 일 군집 표준오차. 통과 = hot 계수 > 0 ∧ p < .05 ∧ 코인별 계수 모두 > 0. ETH(발견 데이터)는 참고.
플래그: fresh_forward_bar_by_bar=true · trade_ledgers_used_as_input=false · saved_parent_exit_timestamps_used=false · future_rows_used_for_entry=false
"""
import glob, json, os, sys, zipfile
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import research_eth_liq_wall_break_conditions_20261013 as R  # noqa: E402
import research_eth_liq_band_sweep_pingpong_20261012 as V1  # noqa: E402

BV, OUT, M5 = R.BV, R.OUT, R.M5
DLV = ("/tmp/claude-1000/-home-kbj20-crypto-scalping--claude-worktrees-liquidation-wall-breakout-analysis-fad604/"
       "a76f7c5d-6448-4654-a51d-9ab396fca982/scratchpad/dl_vision")              # vision 5분봉 09-15~10-10 · ETH bookDepth 08-23~10-10
LAKE = os.path.expanduser("~/backups/crypto-scalping-server/data/lake/binance")
START = {"BTC": "2023-12-20", "SOL": "2023-12-20", "XRP": "2026-07-25"}
END = "2026-10-11"
BOOK_MAIN, BOOK_END = "2025-01-01", "2026-08-23"
CTRL = ["rvol", "rng_sig", "dlt", "r15", "r1h", "h_sig"]


def load_5m(coin):
    p = pd.read_parquet(f"{BV}/panel/{coin}USDT.parquet", columns=["timestamp", "high", "low", "close", "volume", "taker_buy_base"])
    p = p.rename(columns={"taker_buy_base": "tb"})
    p["t"] = pd.to_datetime(p.timestamp).astype("int64") // 10**6
    parts = [p[["t", "high", "low", "close", "volume", "tb"]]]
    for z in sorted(glob.glob(f"{DLV}/k5/{coin}USDT-5m-*.zip")):
        with zipfile.ZipFile(z) as zf:
            d = pd.read_csv(zf.open(zf.namelist()[0]))
        parts.append(pd.DataFrame({"t": d.open_time.astype("int64"), "high": d.high, "low": d.low, "close": d.close, "volume": d.volume, "tb": d.taker_buy_volume}))
    b = pd.concat(parts).drop_duplicates("t").sort_values("t")
    b = b[(b.t >= pd.Timestamp(START[coin]).value // 10**6) & (b.t < pd.Timestamp(END).value // 10**6)]
    full = pd.DataFrame({"t": np.arange(b.t.iloc[0], b.t.iloc[-1] + 1, M5)}).merge(b, on="t", how="left")
    full["close"] = full.close.ffill()
    for c in ("high", "low"):
        full[c] = full[c].fillna(full.close)
    full[["volume", "tb"]] = full[["volume", "tb"]].fillna(0.0)
    full["timestamp"] = pd.to_datetime(full.t, unit="ms", utc=True)
    return full.reset_index(drop=True)


def hourly(b5):
    g = b5.groupby(b5.t // 3_600_000 * 3_600_000)
    h = pd.DataFrame({"high": g.high.max(), "low": g.low.min(), "close": g.close.last(), "volume": g.volume.sum(), "n": g.size()})
    h = h[h.n == 12].drop(columns="n").reset_index(names="t")
    h["timestamp"] = pd.to_datetime(h.t, unit="ms", utc=True)
    return h


def edges_for(b5):
    sn = V1.snapshots(hourly(b5), b5)
    E = pd.DataFrame(V1.lit_edges(sn, b5.close.to_numpy()), columns=["dn_edge", "dn_bottom", "up_edge", "up_top"])
    E["t"] = b5.t.to_numpy()
    return E


def edges():
    for coin in ("BTC", "SOL", "XRP"):
        path = f"{OUT}/edges_{coin}.parquet"
        if not os.path.exists(path):
            edges_for(load_5m(coin)).to_parquet(path)
            print(coin, "edges saved", flush=True)


def check():
    """ETH 2026-02-20~03-31 을 5분 패널로 다시 계산 → 10-12 캐시(1분봉에서 만든 5분봉)와 대조. 3월만 비교(앞은 예열)."""
    p = pd.read_parquet(f"{BV}/panel/ETHUSDT.parquet", columns=["timestamp", "high", "low", "close", "volume"])
    p = p[(p.timestamp >= "2026-02-20") & (p.timestamp < "2026-04-01")].reset_index(drop=True)
    p["t"] = pd.to_datetime(p.timestamp).astype("int64") // 10**6
    p["timestamp"] = pd.to_datetime(p.t, unit="ms", utc=True)
    m = edges_for(p).merge(pd.read_parquet(f"{OUT}/edges_long.parquet"), on="t", suffixes=("", "_ref"))
    m = m[m.t >= pd.Timestamp("2026-03-01").value // 10**6].dropna()
    for c in ("dn_edge", "up_edge", "dn_bottom", "up_top"):
        print(c, "같음(1e-6 상대)", np.mean(np.isclose(m[c], m[c + "_ref"], rtol=1e-6)).round(4), "n", len(m))


def make_events(coin, b5, E, book=None):
    b5 = b5.merge(E, on="t", how="left")
    f = R.base_frame(b5)
    cl = b5.close.to_numpy()
    sh = lambda a, n=1: np.concatenate([np.full(n, np.nan), a[:-n]])
    real = {c: sh(b5[c].to_numpy()) for c in ("up_edge", "up_top", "dn_edge", "dn_bottom")}
    plac = {c: sh(b5[c].to_numpy() / cl, R.PL_LAG + 1) * sh(cl) for c in real}
    k5 = b5.rename(columns={"high": "h", "low": "l"})
    out = []
    for kind, ed in (("real", real), ("placebo", plac)):
        ev = R.detect(b5, ed["up_edge"], ed["up_top"], ed["dn_edge"], ed["dn_bottom"])
        ev = ev[(ev.k > R.PL_LAG + 300) & ev.E.notna() & ev.F.notna()]
        ev = R.finish(R.label(ev, b5, k5, step=M5, hold=48), b5, f)
        if book is not None:
            ev = R.join_book(b5, ev, book)
        out.append(ev.assign(kind=kind, coin=coin))
    ev = pd.concat(out, ignore_index=True)
    ev = ev[ev.y.notna()].copy()
    ev["r"] = ev.y - ev.p0
    return ev


def book_frame_vision():
    rows = []
    for z in sorted(glob.glob(f"{DLV}/book/ETHUSDT-bookDepth-*.zip")):
        with zipfile.ZipFile(z) as zf:
            rows.append(pd.read_csv(zf.open(zf.namelist()[0])))
    d = pd.concat(rows)
    d["col"] = "n" + d.percentage.astype(float).astype(int).astype(str)
    w = d.pivot_table(index="timestamp", columns="col", values="notional", aggfunc="last")
    w.index = pd.to_datetime(w.index)
    return w


def liq_5m(coin):
    """5분 청산 합(L = 롱 청산 · S = 숏 청산), 유효 칸만. tail_risk(1분 합) 08-17~09-18 + liquidations(건별) 09-26~."""
    fs = sorted(glob.glob(f"{LAKE}/tail_risk_1m/coin={coin}/date=*/*.parquet"))
    tr = pd.concat([pd.read_parquet(f, columns=["ts", "long_usd_1m", "short_usd_1m", "valid_liq_stream", "ws_stale"]) for f in fs])
    tr = tr[tr.valid_liq_stream & ~tr.ws_stale.astype(bool)].copy()
    tr["ts"] = (pd.to_datetime(tr.ts, utc=True) - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(milliseconds=1)   # lake 는 datetime(분 시작, 단위 섞임)
    a = tr.groupby(tr.ts // M5 * M5).agg(L=("long_usd_1m", "sum"), S=("short_usd_1m", "sum"), nmin=("ts", "size"))
    a = a[a.nmin >= 4].drop(columns="nmin")                                 # 5분 중 4분 이상 유효한 칸만
    fs = sorted(glob.glob(f"{LAKE}/liquidations/coin={coin}/date=*/*.parquet"))
    lq = pd.concat([pd.read_parquet(f) for f in fs])
    days = sorted({os.path.basename(os.path.dirname(f))[5:] for f in fs})
    grid = np.concatenate([np.arange(pd.Timestamp(d).value // 10**6, pd.Timestamp(d).value // 10**6 + 86_400_000, M5) for d in days])
    lq["b"] = lq.ts_ms // M5 * M5
    g = lq.pivot_table(index="b", columns="side", values="usd", aggfunc="sum").reindex(grid).fillna(0.0)
    x = pd.DataFrame({"L": g["long"], "S": g["short"]}, index=grid)
    x = pd.concat([a, x[~x.index.isin(a.index)]]).sort_index()
    for c in ("L", "S"):                                                     # 유효 칸만 이은 직전 2016칸 90분위
        x[c + "q"] = x[c] / x[c].shift(1).rolling(2016, min_periods=500).quantile(0.9).replace(0, np.nan)
    return x.rename_axis("t").reset_index()


def side_check(b5, lq):
    m = lq.merge(pd.DataFrame({"t": b5.t, "r": np.log(b5.close).diff()}), on="t")
    return [float(np.corrcoef(m.r, np.log1p(m.L))[0, 1]), float(np.corrcoef(m.r, np.log1p(m.S))[0, 1])]


def glm_hot(d):
    import statsmodels.api as sm
    d = d.dropna(subset=CTRL + ["hot"])
    X = d[["hot"] + CTRL].copy()
    if d.coin.nunique() > 1:
        X = X.join(pd.get_dummies(d.coin, prefix="c", drop_first=True, dtype=float))
    off = np.log(d.p0.clip(.02, .98) / (1 - d.p0.clip(.02, .98)))
    m = sm.GLM(d.y, sm.add_constant(X), family=sm.families.Binomial(), offset=off).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.day)[0]})
    return dict(n=len(d), n_hot=int(d.hot.sum()), days=int(d.day.nunique()), coef=float(m.params.hot), se=float(m.bse.hot), p=float(m.pvalues.hot),
                raw_diff=float(d[d.hot == 1].r.mean() - d[d.hot == 0].r.mean()))


def pooled_q(evs, col, cuts_by_coin, kinds=("real",), nb=1000):
    """코인별 경계로 Q1/Q5 표시 → 합친 Q5−Q1 잔차 차, 같은 날 묶음 부트. kinds 가 둘이면 (첫째 − 둘째)."""
    tabs = [pd.concat([R._qtab(d[d.kind == kd][[col, "r", "day"]].dropna(), col, cuts_by_coin[c]) for c, d in evs.items()]).groupby(level=0).sum()
            for kd in kinds]
    days = sorted(set().union(*[t.index for t in tabs]))
    A = [t.reindex(days, fill_value=0).to_numpy() for t in tabs]
    f = lambda S: S[..., 2] / np.maximum(S[..., 3], 1) - S[..., 0] / np.maximum(S[..., 1], 1)
    idx = np.random.default_rng(7).integers(0, len(days), (nb, len(days)))
    est = f(A[0].sum(0)) - (f(A[1].sum(0)) if len(A) > 1 else 0)
    bs = f(A[0][idx].sum(1)) - (f(A[1][idx].sum(1)) if len(A) > 1 else 0)
    return dict(eff=float(est), lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975)), days=len(days))


def test():
    rep = {}
    win = lambda e, a, b: e[(e.t >= pd.Timestamp(a).value // 10**6) & (e.t < pd.Timestamp(b).value // 10**6)]
    # ── H_호가 ──
    evs, cuts = {}, {}
    for coin in ("BTC", "SOL"):
        ev = make_events(coin, load_5m(coin), pd.read_parquet(f"{OUT}/edges_{coin}.parquet"), R.daily_parquets(f"{BV}/{coin}USDT_bookDepth_*.parquet"))
        evs[coin] = ev
        cuts[coin] = np.quantile(win(ev[ev.kind == "real"], "2024-01-01", "2025-01-01").book_opp_wall1.dropna(), [.2, .4, .6, .8])
        print(coin, "events", ev.groupby("kind").size().to_dict(), "resid", ev.groupby("kind").r.mean().round(4).to_dict(), flush=True)
    hb = {}
    for nm, (a, b) in (("main_2025", (BOOK_MAIN, BOOK_END)), ("ref_2024", ("2024-01-01", "2025-01-01"))):
        sub = {c: win(e, a, b) for c, e in evs.items()}
        hb[nm] = {"pooled": pooled_q(sub, "book_opp_wall1", cuts), "placebo": pooled_q(sub, "book_opp_wall1", cuts, kinds=("placebo",)),
                  "wall_specific": pooled_q(sub, "book_opp_wall1", cuts, kinds=("real", "placebo")),
                  **{c: R.q_effect(sub[c][sub[c].kind == "real"], "book_opp_wall1", cuts[c]) for c in evs}}
    m = hb["main_2025"]
    hb["pass"] = bool(m["pooled"]["lo"] > 0 and all(m[c]["eff"] > 0 for c in evs) and m["wall_specific"]["lo"] > 0)
    # ETH 08-23~10-10 새 호가(참고) -- 발견 사건(1분 라벨)에 새로 받은 bookDepth 를 붙임
    allev = pd.read_parquet(f"{OUT}/events.parquet")
    eth = allev[allev.y.notna() & (allev.t >= pd.Timestamp("2026-08-23").value // 10**6)].copy()
    eth = R.join_book(R.bars(R.load_1m(), M5), eth.drop(columns=[c for c in eth.columns if c.startswith("book_")]), book_frame_vision())
    eth["r"] = eth.y - eth.p0
    ecut = np.quantile(win(allev[allev.kind == "real"], "2024-01-01", "2025-01-01").book_opp_wall1.dropna(), [.2, .4, .6, .8])
    hb["ETH_fresh"] = {kd: R.q_effect(eth[eth.kind == kd], "book_opp_wall1", ecut) for kd in ("real", "placebo")}
    rep["H_book"] = hb
    # ── H_청산 ──
    lev, sides = [], {}
    for coin in ("BTC", "SOL", "XRP"):
        b5, lq = load_5m(coin), liq_5m(coin)
        sides[coin] = side_check(b5, lq)
        ev = make_events(coin, b5, pd.read_parquet(f"{OUT}/edges_{coin}.parquet")).merge(lq, on="t", how="inner")   # 봉 k 5분 합(봉 끝에 앎) · 유효 칸만
        q = np.where(ev.s > 0, ev.Sq, ev.Lq)                                 # 밀리는 쪽: 위 = 숏 청산
        ev["hot"] = np.where(np.isnan(q), np.nan, (q >= 1) * 1.0)
        lev.append(ev)
    lev = pd.concat(lev, ignore_index=True)
    hl = {"side_corr(r~L, r~S)": sides}
    for kd in ("real", "placebo"):
        d = lev[lev.kind == kd]
        hl[kd] = {"pooled": glm_hot(d), **{c: glm_hot(d[d.coin == c]) for c in ("BTC", "SOL", "XRP")}, "hot_share": float(d.hot.mean())}
    p = hl["real"]
    hl["pass"] = bool(p["pooled"]["coef"] > 0 and p["pooled"]["p"] < .05 and all(p[c]["coef"] > 0 for c in ("BTC", "SOL", "XRP")))
    rep["H_liq"] = hl
    rep["flags"] = dict(fresh_forward_bar_by_bar=True, trade_ledgers_used_as_input=False, saved_parent_exit_timestamps_used=False, future_rows_used_for_entry=False)
    json.dump(rep, open(f"{OUT}/report_confirm.json", "w"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(rep, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    {"--edges": edges, "--check": check, "--test": test}[sys.argv[1]]()
