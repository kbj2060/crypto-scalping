"""고래↔리테일 맞대결(고래 추종 60분) — SOL·XRP 재현 검정. ETH 원 연구 tmp/whale/{build,disagree}.py 와 같은 정의.
사전등록: docs/experiments/whale_duel_solxrp_20261004.md

  python scripts/research_whale_duel_solxrp_20261004.py build SOLUSDT [2023-01]   # 월 zip 받기 → 1분 집계 → zip 삭제
  python scripts/research_whale_duel_solxrp_20261004.py analyze                   # verdict.json
  python scripts/research_whale_duel_solxrp_20261004.py selftest

데이터 = data.binance.vision 정적 아카이브(API 아님). fapi/api REST 는 호출하지 않는다.
"""
import io, json, os, sys, time, glob, zipfile, urllib.request
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'tmp/whale_solxrp')
URL = 'https://data.binance.vision/data/futures/um/monthly/aggTrades/{s}/{s}-aggTrades-{m}.zip'
ETH_TAPE = '/home/kbj20/crypto-scalping/tmp/worktree_salvage_20260930/scalping-entry-analysis-c0c47c/files/tmp/whale/tape/ETHUSDT'
# (리테일 상한, 고래 하한) — a = 코인 경계(live_trade_tape_collector SIZE_BANDS_USD), b = ETH 경계
BANDS = {'SOLUSDT': {'a': (7_500, 55_000), 'b': (10_000, 100_000)},
         'XRPUSDT': {'a': (3_400, 23_000), 'b': (10_000, 100_000)}}
COLS = ['agg_trade_id', 'price', 'quantity', 'first_trade_id', 'last_trade_id', 'transact_time', 'is_buyer_maker']
GRP = ['ret', 'mid', 'whl']


def _add(acc, key, mi, side, nt, edges, M):
    cls = np.searchsorted(edges, nt, side='right')                       # 0 리테일 1 중형 2 고래 (build.py 와 같다)
    for k, g in enumerate(GRP):
        m = cls == k
        acc[f'{key}_{g}_sn'] += np.bincount(mi[m], side[m] * nt[m], M)
        acc[f'{key}_{g}_nt'] += np.bincount(mi[m], nt[m], M)


def aggregate(chunks, t0_ms, M, bands):
    """aggTrade 청크들 → 분(M개) × {row_a,row_b,ord_a}×{ret,mid,whl}×{sn,nt} + 종가 c.
    ord = 같은 (ms·방향) 묶음 — 청크가 ms 를 쪼개지 않게 하는 건 aggregate_stream 몫."""
    acc = {f'{k}_{g}_{x}': np.zeros(M) for k in ('row_a', 'row_b', 'ord_a') for g in GRP for x in ('sn', 'nt')}
    c = np.full(M, np.nan)
    for df in chunks:
        t = df.transact_time.to_numpy(np.int64)
        if len(t) and t.max() > 1e14: t = t // 1000                       # 마이크로초 아카이브 방어
        if len(t) > 1 and not (np.diff(t) >= 0).all():
            o = np.argsort(t, kind='stable'); df = df.iloc[o]; t = t[o]
        px = df.price.to_numpy(float); q = df.quantity.to_numpy(float)
        side = np.where(df.is_buyer_maker.to_numpy(bool), -1.0, 1.0); nt = px * q
        mi = (t - t0_ms) // 60000
        assert len(mi) == 0 or (mi.min() >= 0 and mi.max() < M), 'month range'
        _add(acc, 'row_a', mi, side, nt, bands['a'], M)
        _add(acc, 'row_b', mi, side, nt, bands['b'], M)
        last = np.r_[np.flatnonzero(np.diff(mi)), len(mi) - 1] if len(mi) else np.array([], int)
        c[mi[last]] = px[last]
        o = pd.DataFrame({'t': t, 's': side, 'nt': nt}).groupby(['t', 's'], sort=False).nt.sum().reset_index()
        _add(acc, 'ord_a', (o.t.to_numpy() - t0_ms) // 60000, o.s.to_numpy(), o.nt.to_numpy(), bands['a'], M)
    out = pd.DataFrame(acc); out.insert(0, 'c', c)
    out.index = pd.to_datetime(t0_ms + np.arange(M) * 60000, unit='ms', utc=True)
    return out


def aggregate_stream(chunks, t0_ms, M, bands):
    """청크 경계의 같은 ms 를 다음 청크로 넘겨 ord 묶음이 쪼개지지 않게 한다."""
    def regroup():
        carry = None
        for df in chunks:
            if carry is not None: df = pd.concat([carry, df])
            t = df.transact_time.to_numpy(np.int64)
            cut = np.searchsorted(t, t[-1]) if (np.diff(t) >= 0).all() else len(t)
            carry = df.iloc[cut:]; yield df.iloc[:cut]
        if carry is not None and len(carry): yield carry
    return aggregate(regroup(), t0_ms, M, bands)


def read_zip(path):
    z = zipfile.ZipFile(path); f = z.open(z.namelist()[0])
    head = f.readline().decode(); f.close(); f = z.open(z.namelist()[0])
    hdr = 0 if head.startswith('agg_trade_id') else None
    return pd.read_csv(f, header=hdr, names=None if hdr == 0 else COLS, chunksize=4_000_000,
                       usecols=['price', 'quantity', 'transact_time', 'is_buyer_maker'])


def build(sym, start='2023-01', end='2026-09'):
    d = f'{OUT}/tape/{sym}'; os.makedirs(d, exist_ok=True)
    for m in pd.period_range(start, end, freq='M'):
        dst = f'{d}/{m}.parquet'
        if os.path.exists(dst): continue
        zp = f'{OUT}/_dl_{sym}_{m}.zip'; t1 = time.time()
        for k in range(4):
            try: urllib.request.urlretrieve(URL.format(s=sym, m=m), zp); break
            except Exception as e: print(f'[{sym} {m}] retry {k}: {e}', flush=True); time.sleep(10 * (k + 1))
        else: print(f'[{sym} {m}] MISSING', flush=True); continue
        try:
            t0 = int(m.start_time.tz_localize('UTC').value // 10**6); M = int(m.days_in_month * 1440)
            aggregate_stream(read_zip(zp), t0, M, BANDS[sym]).to_parquet(dst + '.part'); os.replace(dst + '.part', dst)
        finally:
            os.remove(zp)                                                  # 디스크: zip 은 바로 지운다
        print(f'[{sym} {m}] {time.time() - t1:.0f}s', flush=True)


# ---------------- 분석 (disagree.py 와 같은 규약) ----------------
rng = np.random.default_rng(0)


def bci(v, d, B=1000):
    if len(v) < 30: return dict(mean=float(np.mean(v)) if len(v) else None, ci=None, n=int(len(v)))
    g = pd.DataFrame({'v': v, 'd': d}).groupby('d').v.agg(['sum', 'count'])
    W = rng.multinomial(len(g), np.ones(len(g)) / len(g), size=B); est = (W @ g['sum'].values) / (W @ g['count'].values)
    return dict(mean=round(float(np.mean(v)), 2), ci=[round(float(x), 2) for x in np.percentile(est, [2.5, 97.5])], n=int(len(v)))


def load(paths):
    T = pd.concat([pd.read_parquet(f) for f in sorted(paths)]).sort_index()
    return T[~T.index.duplicated()].asfreq('1min')


def duel(T, key, Wm=60, th=0.5, other='ret'):
    c = T.c.ffill().values; n = len(c)
    t = np.arange(Wm + 30 * 1440, n - Wm, Wm); fwd = np.log(c[t + Wm] / c[t]) * 1e4; past = np.log(c[t] / c[t - Wm]) * 1e4
    ts = T.index[t]; Z = {}
    for g in GRP:
        s = T[f'{key}_{g}_sn'].fillna(0).rolling(Wm).sum()
        sd = s.iloc[::Wm].rolling(30 * 1440 // Wm, min_periods=7 * 1440 // Wm).std().reindex(s.index).ffill().shift(Wm)
        Z[g] = (s / sd).values[t]
    ok = np.isfinite(Z['ret']) & np.isfinite(Z['whl']) & np.isfinite(Z['mid']) & np.isfinite(fwd)
    dis = ok & (np.sign(Z['whl']) != np.sign(Z[other])) & (np.abs(Z['whl']) >= th) & (np.abs(Z[other]) >= th)
    return dict(ts=ts, day=ts.floor('D'), yr=ts.year, ok=ok, dis=dis, v=np.sign(Z['whl']) * fwd, past=past, fwd=fwd, zw=Z['whl'])


def report(T, key, main_from='2025-01-01', full=True):
    D = duel(T, key); main = D['ts'] >= pd.Timestamp(main_from, tz='UTC')
    v, day, dis, ok, yr = D['v'], D['day'], D['dis'], D['ok'], D['yr']
    r = dict(all=bci(v[dis], day[dis]), main=bci(v[dis & main], day[dis & main]), aux=bci(v[dis & ~main], day[dis & ~main]),
             years={int(y): bci(v[dis & (yr == y)], day[dis & (yr == y)])['mean'] for y in sorted(set(yr[dis]))},
             events_per_day=round(float((dis & main).sum() / ((ok & main).sum() * 60 / 1440)), 2),
             span=[str(D['ts'][ok][0]), str(D['ts'][ok][-1])])
    if not full: return r
    m = dis & main; vm = v[m]
    rnd = np.array([np.mean(rng.choice([-1.0, 1.0], len(vm)) * D['fwd'][m]) for _ in range(2000)])
    r['control_random_dir'] = dict(p95=round(float(np.percentile(rnd, 95)), 2), p_perm=round(float((rnd >= vm.mean()).mean()), 4))
    wm = np.sign(D['zw']) == np.sign(D['past'])
    r['whale_vs_price'] = dict(with_price=bci(v[m & wm], day[m & wm]), against_price=bci(v[m & ~wm], day[m & ~wm]))
    pf = np.sign(D['past']) * D['fwd']; base = ok & ~dis & main
    r['control_follow_price'] = dict(non_duel=bci(pf[base], day[base]), in_duel=bci(pf[m], day[m]))
    r['threshold'] = {}
    for th in (0.25, 1.0, 1.5):
        E = duel(T, key, th=th); mm = E['dis'] & (E['ts'] >= pd.Timestamp(main_from, tz='UTC'))
        r['threshold'][str(th)] = bci(E['v'][mm], E['day'][mm])
    r['net_main'] = dict(taker=round(r['main']['mean'] - 8, 2), maker=round(r['main']['mean'] - 2.2, 2))
    return r


def integrity(sym, T):
    """로컬 5분 klines 와 대조: 종가 · 총 테이커 순매수(= 세 구간 합)."""
    fs = sorted(glob.glob(f'/home/kbj20/crypto-scalping/data/{sym[:3].lower()}_5m_20*.csv'))
    k = pd.concat([pd.read_csv(f, usecols=['ts', 'close', 'quote_volume', 'taker_buy_quote']) for f in fs]).drop_duplicates('ts')
    k.index = pd.to_datetime(k.ts.astype('int64'), unit='ms', utc=True)
    a = pd.DataFrame({'c': T.c.resample('5min').last(), 'net': sum(T[f'row_a_{g}_sn'] for g in GRP).resample('5min').sum()})
    j = a.join(k, how='inner').dropna()
    net_k = 2 * j.taker_buy_quote - j.quote_volume
    sh = {g: T[f'row_a_{g}_nt'] for g in GRP}; tot = sum(sh.values())
    by_yr = {int(y): {g: round(float(sh[g][tot.index.year == y].sum() / tot[tot.index.year == y].sum()), 3) for g in GRP}
             for y in sorted(set(tot.index.year))}
    return dict(bars=int(len(j)), close_match=round(float((np.abs(j.c / j.close - 1) < 1e-9).mean()), 5),
                net_corr=round(float(np.corrcoef(j.net, net_k)[0, 1]), 5), net_ratio=round(float(j.net.abs().sum() / net_k.abs().sum()), 4),
                no_trade_minutes=int(T.c.isna().sum()), volume_share_a=by_yr)


def analyze():
    res = {}
    E = load(glob.glob(f'{ETH_TAPE}/*.parquet'))
    eth = report(E, 'row', full=False)
    assert abs(eth['all']['mean'] - 6.17) < 0.01, f"양성대조 실패: ETH {eth['all']['mean']} != +6.17"
    res['eth_control'] = eth
    for sym in ('SOLUSDT', 'XRPUSDT'):
        T = load(glob.glob(f'{OUT}/tape/{sym}/*.parquet'))
        a = report(T, 'row_a'); b = report(T, 'row_b', full=False); o = report(T, 'ord_a', full=False)
        y = a['years']
        res[sym[:3].lower()] = dict(pass_=bool(a['main']['ci'] and a['main']['ci'][0] > 0 and y.get(2025, -1) > 0 and y.get(2026, -1) > 0),
                                    main_mean_bp=a['main']['mean'], ci=a['main']['ci'], years=y, events_per_day=a['events_per_day'],
                                    detail=a, eth_bands=b, ord_unit=o, integrity=integrity(sym, T))
        print(sym, json.dumps({k: res[sym[:3].lower()][k] for k in ('pass_', 'main_mean_bp', 'ci', 'years', 'events_per_day')}), flush=True)
    s = json.dumps(res, indent=1, default=str).replace('"pass_"', '"pass"')
    open(f'{OUT}/verdict.json', 'w').write(s); print(s)


def selftest():
    r = np.random.default_rng(1); N = 5000; t0 = 1_700_000_000_000
    t = np.sort(t0 + r.integers(0, 10 * 60000, N)); t[100:110] = t[100]          # 같은 ms 묶음
    df = pd.DataFrame({'price': r.uniform(90, 110, N), 'quantity': r.exponential(300, N), 'transact_time': t,
                       'is_buyer_maker': r.random(N) < 0.5})
    bands = {'a': (7_500, 55_000), 'b': (10_000, 100_000)}
    got = aggregate_stream((df.iloc[i:i + 777] for i in range(0, N, 777)), t0, 10, bands)   # 청크 경계 강제
    nt = df.price * df.quantity; side = np.where(df.is_buyer_maker, -1, 1); mi = (t - t0) // 60000
    for key, (lo, hi) in (('row_a', bands['a']), ('row_b', bands['b'])):
        cls = np.where(nt >= hi, 'whl', np.where(nt < lo, 'ret', 'mid'))
        for g in GRP:
            want = pd.Series(side * nt * (cls == g)).groupby(mi).sum().reindex(range(10), fill_value=0).values
            assert np.allclose(got[f'{key}_{g}_sn'].values, want), (key, g)
    o = pd.DataFrame({'t': t, 's': side, 'nt': nt}).groupby(['t', 's']).nt.sum().reset_index()
    assert np.isclose(got[[f'ord_a_{g}_nt' for g in GRP]].values.sum(), nt.sum())
    assert np.isclose(got['ord_a_whl_nt'].sum(), o.nt[o.nt >= 55_000].sum())    # ord 묶음이 청크에 안 쪼개짐
    assert np.allclose(got.c.values, df.groupby(mi).price.last().values)
    print('selftest ok')


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'build': build(sys.argv[2], *(sys.argv[3:4] or ['2023-01']))
    elif cmd == 'analyze': analyze()
    else: selftest()
