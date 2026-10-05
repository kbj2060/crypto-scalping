"""고래↔리테일 맞대결(고래 추종 60분) — 바이낸스 단독 vs +OKX vs +OKX+바이비트 합산 검정 (ETH).
사전등록: docs/experiments/whale_duel_multivenue_20261006.md · 정의는 research_whale_duel_solxrp_20261004.duel 을 그대로 쓴다.

  python scripts/research_whale_duel_multivenue_20261006.py build      # 일 덤프 받기 → 1분 집계 → 원본 삭제 + 1일 캔들(품질 점검용)
  python scripts/research_whale_duel_multivenue_20261006.py analyze    # 품질 점검 + 세 팔 → verdict.json
  python scripts/research_whale_duel_multivenue_20261006.py selftest   # 네트워크 없음

데이터: 바이낸스 = data.binance.vision 정적 아카이브(🔴fapi/api REST 호출 없음) · OKX = static.okx.com 일 덤프(UTC+8 일 경계) ·
바이비트 = public.bybit.com. OKX·바이비트 캔들 REST 는 품질 점검용으로 몇 번만 부른다.
"""
import io, json, os, sys, glob, time, zipfile, urllib.request, urllib.error
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import research_whale_duel_solxrp_20261004 as W                       # duel·bci·load·read_zip (원 연구 정의)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'tmp/whale_multivenue_20261006')
EDGES = (10_000, 100_000)                                                # 리테일 < $10k · 고래 ≥ $100k (ETH 경계)
GRP = W.GRP
OKX_CT = 0.1                                                             # ETH-USDT-SWAP ctVal (size = 계약 수)
START, END, MAIN = '2024-12-01', '2026-09-30', '2025-01-01'
BIN_OLD_END = '2026-09-23'                                               # 원 테이프는 여기까지 쓰고 09-24~ 는 새로 만든다(09-24 = 겹침 대조)
UA = {'User-Agent': 'Mozilla/5.0'}
SRC = {'binance': 'https://data.binance.vision/data/futures/um/daily/aggTrades/ETHUSDT/ETHUSDT-aggTrades-{d}.zip',
       'okx': 'https://static.okx.com/cdn/okex/traderecords/trades/daily/{dc}/ETH-USDT-SWAP-trades-{d}.zip',
       'bybit': 'https://public.bybit.com/trading/ETHUSDT/ETHUSDT{d}.csv.gz'}
ARMS = {'b': ('binance',), 'bo': ('binance', 'okx'), 'boy': ('binance', 'okx', 'bybit')}


def minutes(t, side, px, q, group, tick=1):
    """체결(거래소 순서) → 1분 × {ret,mid,whl}×{sn,nt} + c(마지막 체결가)·qty·fills·lines. t = 정수 시각(ms × tick).
    group=True 면 연속된 (t, side, px) 를 한 줄(테이커 주문 하나의 한 가격)로 묶는다 — 바이낸스 aggTrade 와 같은 단위."""
    new = np.r_[True, (t[1:] != t[:-1]) | (side[1:] != side[:-1]) | (px[1:] != px[:-1])] if group else np.ones(len(t), bool)
    gid = np.cumsum(new) - 1
    nt = np.bincount(gid, px * q)
    cls = np.searchsorted(EDGES, nt, side='right')                       # 0 리테일 1 중형 2 고래 (build.py 와 같다)
    L = pd.DataFrame({'m': t[new] // (60000 * tick), 'g': cls, 'sn': side[new] * nt, 'nt': nt})
    a = L.groupby(['m', 'g'])[['sn', 'nt']].sum().unstack('g', fill_value=0.0)
    out = pd.DataFrame({f'{g}_{x}': a[x][k] if k in a[x] else 0.0 for k, g in enumerate(GRP) for x in ('sn', 'nt')}, index=a.index)
    F = pd.DataFrame({'m': t // (60000 * tick), 'px': px, 'q': q}).groupby('m').agg(c=('px', 'last'), qty=('q', 'sum'), fills=('q', 'size'))
    out = F.join(out).join(L.groupby('m').size().rename('lines')).fillna(0.0)
    out.index = pd.to_datetime(out.index.to_numpy(np.int64) * 60000, unit='ms', utc=True)
    return out


def read_venue(venue, blob):
    if venue == 'binance':
        df = pd.concat(W.read_zip(io.BytesIO(blob)))
        t = df.transact_time.to_numpy(np.int64)
        return minutes(t, np.where(df.is_buyer_maker.to_numpy(bool), -1.0, 1.0), df.price.to_numpy(float), df.quantity.to_numpy(float), False)
    if venue == 'okx':
        z = zipfile.ZipFile(io.BytesIO(blob))
        df = pd.read_csv(z.open(z.namelist()[0]), usecols=['trade_id', 'side', 'price', 'size', 'created_time']).sort_values('trade_id', kind='stable')
        return minutes(df.created_time.to_numpy(np.int64), np.where(df.side.to_numpy() == 'buy', 1.0, -1.0),
                       df.price.to_numpy(float), df['size'].to_numpy(float) * OKX_CT, True)
    df = pd.read_csv(io.BytesIO(blob), compression='gzip', usecols=['timestamp', 'side', 'size', 'price'])
    t = np.round(df.timestamp.to_numpy(float) * 1e4).astype(np.int64)   # 초 소수 4자리 → 0.1ms 틱
    if not (np.diff(t) >= 0).all():
        o = np.argsort(t, kind='stable'); df = df.iloc[o]; t = t[o]
    return minutes(t, np.where(df.side.to_numpy() == 'Buy', 1.0, -1.0), df.price.to_numpy(float), df['size'].to_numpy(float), True, tick=10)


def fetch(url):
    for k in range(5):
        try: return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300).read()
        except urllib.error.HTTPError as e:
            if e.code == 404: return None
        except Exception: pass
        time.sleep(5 * (k + 1))
    return None


def job(venue, d):
    dst = f'{OUT}/tape/{venue}/{d}.parquet'
    if os.path.exists(dst): return venue, d, 'have'
    blob = fetch(SRC[venue].format(d=d, dc=d.replace('-', '')))         # 받은 원본은 메모리에서만 → 집계 뒤 버린다(디스크 안 씀)
    if blob is None: return venue, d, 'MISSING'
    read_venue(venue, blob).to_parquet(dst + '.part'); os.replace(dst + '.part', dst)
    return venue, d, 'ok'


def klines():
    """품질 점검용 1일 캔들(UTC 일) → klines_1d.csv: venue, day, quote, base."""
    rows = []
    for m in pd.period_range(START[:7], END[:7], freq='M'):              # 바이낸스 = 정적 아카이브(REST 아님)
        blob = fetch(f'https://data.binance.vision/data/futures/um/monthly/klines/ETHUSDT/1d/ETHUSDT-1d-{m}.zip')
        if blob is None: continue
        z = zipfile.ZipFile(io.BytesIO(blob)); k = pd.read_csv(z.open(z.namelist()[0]), header=None)
        k = k[pd.to_numeric(k[0], errors='coerce').notna()]
        rows += [('binance', pd.Timestamp(int(r[0]), unit='ms').strftime('%Y-%m-%d'), float(r[7]), float(r[5])) for r in k.itertuples(index=False)]
    after = int(pd.Timestamp(END).value // 10**6) + 86_400_000 + 1        # OKX: 최신부터 100개씩 뒤로
    while True:
        b = json.loads(fetch(f'https://www.okx.com/api/v5/market/history-candles?instId=ETH-USDT-SWAP&bar=1Dutc&limit=100&after={after}'))['data']
        if not b: break
        rows += [('okx', pd.Timestamp(int(r[0]), unit='ms').strftime('%Y-%m-%d'), float(r[7]), float(r[6])) for r in b]
        after = min(int(r[0]) for r in b)
        if after < pd.Timestamp(START).value // 10**6: break
        time.sleep(1)
    s, e = (int(pd.Timestamp(x).value // 10**6) for x in (START, END))
    b = json.loads(fetch(f'https://api.bybit.com/v5/market/kline?category=linear&symbol=ETHUSDT&interval=D&start={s}&end={e}&limit=1000'))['result']['list']
    rows += [('bybit', pd.Timestamp(int(r[0]), unit='ms').strftime('%Y-%m-%d'), float(r[6]), float(r[5])) for r in b]
    k = pd.DataFrame(rows, columns=['venue', 'day', 'quote', 'base'])
    k[(k.day >= START) & (k.day <= END)].drop_duplicates(['venue', 'day']).to_csv(f'{OUT}/klines_1d.csv', index=False)


def build():
    for v in SRC: os.makedirs(f'{OUT}/tape/{v}', exist_ok=True)
    days = [d.strftime('%Y-%m-%d') for d in pd.date_range(START, END)]
    okx_files = [d.strftime('%Y-%m-%d') for d in pd.date_range(START, pd.Timestamp(END) + pd.Timedelta('1D'))]  # UTC+8 → 다음 날 파일까지
    todo = [('binance', d) for d in days if d > BIN_OLD_END] + [('okx', d) for d in okx_files] + [('bybit', d) for d in days]
    t0 = time.time(); miss = []
    with ProcessPoolExecutor(5) as ex:
        for i, (v, d, st) in enumerate(ex.map(job, *zip(*todo))):
            if st == 'MISSING': miss.append((v, d))
            if i % 50 == 0 or st == 'MISSING': print(f'{i}/{len(todo)} {v} {d} {st} {time.time() - t0:.0f}s', flush=True)
    json.dump(miss, open(f'{OUT}/missing.json', 'w'))
    if not os.path.exists(f'{OUT}/klines_1d.csv'): klines()
    print('build done, missing', miss, flush=True)


# ---------------- 분석 ----------------
def load_venue(v):
    T = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{OUT}/tape/{v}/*.parquet'))])
    agg = {c: 'sum' for c in T.columns}; agg['c'] = 'last'                 # OKX 파일 경계(16:00 UTC) 같은 분 방어
    T = T.groupby(level=0).agg(agg).sort_index()
    return T[(T.index >= START) & (T.index < pd.Timestamp(END, tz='UTC') + pd.Timedelta('1D'))]


def load_binance():
    old = [f for f in glob.glob(f'{W.ETH_TAPE}/*.parquet') if START <= os.path.basename(f)[:10] <= BIN_OLD_END]
    O = pd.concat([pd.read_parquet(f) for f in sorted(old)]).rename(columns=lambda c: c[4:] if c.startswith('row_') else c)
    N = load_venue('binance')
    # 겹침 대조: 원 테이프 09-24 vs 새 집계 09-24 — 같은 정의인지
    o = pd.read_parquet(f'{W.ETH_TAPE}/2026-09-24.parquet').rename(columns=lambda c: c[4:] if c.startswith('row_') else c)
    n = N[N.index.strftime('%Y-%m-%d') == '2026-09-24']
    cols = [f'{g}_{x}' for g in GRP for x in ('sn', 'nt')] + ['c']
    gap = float(np.max(np.abs(o[cols].values - n.reindex(o.index)[cols].values) / (np.abs(o[cols].values) + 1)))
    return pd.concat([O[cols], N[cols]]).sort_index(), gap


def zlive(s):
    """라이브 rl_1s_agent.whale_z 정의: 분마다 60분 합의 30일(43,200분) 평균 차감 z."""
    n60 = s.fillna(0).rolling(60).sum(); r = n60.rolling(43200, min_periods=20000)
    return (n60 - r.mean()) / r.std()


def passes(arm, base):
    """판정 규칙: 합산판 건당 평균 ≥ 바이낸스판 AND 합산판 CI 하한 > 0."""
    return bool(arm['ci'] is not None and arm['mean'] >= base['mean'] and arm['ci'][0] > 0)


def agreement(da, db):
    """da·db = 발동 방향(+1/−1, 미발동 0) 같은 격자. 바이낸스 사건 중 합산판에서도 같은 방향 비중 · 합산 사건 중 바이낸스와 같은 비중."""
    both = (da != 0) & (da == db)
    return dict(of_binance=round(float(both.sum() / max((db != 0).sum(), 1)), 3), of_arm=round(float(both.sum() / max((da != 0).sum(), 1)), 3),
                opposite=int(((da != 0) & (db != 0) & (da != db)).sum()))


def diff_ci(va, da, vb, db, B=1000, seed=0):
    """두 팔 건당 평균 차(a − b)의 같은-날 블록 부트스트랩 CI(같은 날 가중을 두 팔에 공유)."""
    days = sorted(set(da) | set(db)); ix = {d: i for i, d in enumerate(days)}
    def per(v, d):
        s, c = np.zeros(len(days)), np.zeros(len(days))
        np.add.at(s, [ix[x] for x in d], v); np.add.at(c, [ix[x] for x in d], 1); return s, c
    (sa, ca), (sb, cb) = per(va, da), per(vb, db)
    Wt = np.random.default_rng(seed).multinomial(len(days), np.ones(len(days)) / len(days), size=B)
    est = (Wt @ sa) / (Wt @ ca) - (Wt @ sb) / (Wt @ cb)
    return dict(diff=round(float(va.mean() - vb.mean()), 2), ci=[round(float(x), 2) for x in np.percentile(est, [2.5, 97.5])])


def quality(V, K):
    q = {}
    for v, T in V.items():
        day = T.index.strftime('%Y-%m-%d')
        nt = sum(T[f'{g}_nt'] for g in GRP).groupby(day).sum()
        k = K[K.venue == v].set_index('day')
        r = (nt / k.quote).dropna()
        allday = pd.date_range(START, END).strftime('%Y-%m-%d')
        q[v] = dict(days_with_data=int(nt.size), missing_days=sorted(set(allday) - set(nt.index[nt > 0])),
                    thin_days=list(T.groupby(day).size().loc[lambda x: x < 1300].index),   # 하루 1,440분 중 체결 있는 분 < 1,300
                    kline_days=int(len(r)), vol_ratio_median=round(float(r.median()), 4),
                    vol_ratio_minmax=[round(float(r.min()), 4), round(float(r.max()), 4)], days_off_1pct=int((np.abs(r - 1) > 0.01).sum()),
                    worst_days={d: round(float(x), 4) for d, x in (r - 1).abs().sort_values().tail(3).items()})
        if v == 'okx':
            qty = T.qty.groupby(day).sum(); rb = (qty / k.base).dropna()
            q[v]['base_ratio_ctval_0.1_median'] = round(float(rb.median()), 4)   # 1 이면 size×0.1 = ETH 가 맞다
        share = {g: T[f'{g}_nt'].groupby(T.index.year).sum() for g in GRP}; tot = sum(share.values())
        q[v]['notional_share_by_year'] = {int(y): {g: round(float(share[g][y] / tot[y]), 3) for g in GRP} for y in tot.index}
        q[v]['notional_total_usd_bn'] = round(float(tot.sum() / 1e9), 1)
    c = pd.DataFrame({v: T.c for v, T in V.items()}).dropna()
    r = np.log(c).diff().dropna()
    q['align'] = dict(minutes=int(len(c)), level_corr={v: round(float(c.binance.corr(c[v])), 6) for v in ('okx', 'bybit')},
                      ret_corr_by_lag={v: {L: round(float(r.binance.corr(r[v].shift(L))), 4) for L in (-2, -1, 0, 1, 2)} for v in ('okx', 'bybit')},
                      median_abs_bp_gap={v: round(float((np.abs(c[v] / c.binance - 1) * 1e4).median()), 2) for v in ('okx', 'bybit')})
    return q


def analyze():
    res = {}
    E = W.load(glob.glob(f'{W.ETH_TAPE}/*.parquet'))                     # 양성대조: 원 테이프 전 기간
    ctl = W.report(E, 'row', full=False)
    assert abs(ctl['all']['mean'] - 6.17) < 0.01, f"양성대조 실패: {ctl['all']['mean']} != +6.17"
    res['positive_control_original_tape'] = dict(all=ctl['all'], main_2025_to_0924=ctl['main'], years=ctl['years'])
    del E

    b, gap = load_binance()
    V = {'binance': b, 'okx': load_venue('okx'), 'bybit': load_venue('bybit')}
    K = pd.read_csv(f'{OUT}/klines_1d.csv')
    res['quality'] = quality(V, K); res['quality']['binance_0924_overlap_max_rel_gap'] = gap
    print(json.dumps(res['quality'], indent=1, default=str), flush=True)

    idx = pd.date_range(START, pd.Timestamp(END) + pd.Timedelta('1D'), freq='1min', tz='UTC', inclusive='left')
    miss = pd.Series(False, idx)                                          # 어느 거래소든 결측일이면 그 날 사건을 세 팔 모두에서 뺀다
    for v in V:
        for d in res['quality'][v]['missing_days']: miss.loc[d] = True
    T = pd.DataFrame({'c': V['binance'].c.reindex(idx)})                  # 수익은 세 팔 모두 바이낸스 종가
    for k, vs in ARMS.items():
        for g in GRP:
            T[f'{k}_{g}_sn'] = sum(V[v][f'{g}_sn'].reindex(idx).fillna(0.0) for v in vs)

    D = {k: W.duel(T, k) for k in ARMS}
    ts = D['b']['ts']; mainm = (ts >= pd.Timestamp(MAIN, tz='UTC'))
    bad = miss.rolling(120).max().shift(-60).reindex(ts).fillna(1).astype(bool).values   # 결정 전 60분·후 60분 창이 결측일에 걸치면 제외
    q = ts.year.astype(str) + 'Q' + ts.quarter.astype(str)
    rng = np.random.default_rng(1)
    pos = T.index.get_indexer(ts)
    for k in ARMS:
        d = D[k]; m = d['dis'] & mainm & ~bad; v = d['v']
        rnd = np.array([np.mean(rng.choice([-1.0, 1.0], m.sum()) * d['fwd'][m]) for _ in range(2000)])
        thr = {}
        for th in (0.25, 1.0, 1.5):
            e = W.duel(T, k, th=th); mm = e['dis'] & mainm & ~bad; thr[str(th)] = W.bci(e['v'][mm], e['day'][mm])
        zw, zr = zlive(T[f'{k}_whl_sn']).values[pos], zlive(T[f'{k}_ret_sn']).values[pos]
        ml = (zw * zr < 0) & (np.abs(zw) >= 0.5) & (np.abs(zr) >= 0.5) & np.isfinite(d['fwd']) & mainm & ~bad
        main = W.bci(v[m], d['day'][m])
        res[k] = dict(venues=ARMS[k], main=main, events=int(m.sum()),
                      events_per_day=round(float(m.sum() / ((d['ok'] & mainm & ~bad).sum() / 24)), 2),
                      years={int(y): W.bci(v[m & (d['yr'] == y)], d['day'][m & (d['yr'] == y)]) for y in (2025, 2026)},
                      quarters={x: round(float(v[m & (q == x)].mean()), 2) for x in sorted(set(q[m]))},
                      net_maker=round(main['mean'] - 2.2, 2), net_taker=round(main['mean'] - 8, 2),
                      control_random_dir=dict(p95=round(float(np.percentile(rnd, 95)), 2), p_perm=round(float((rnd >= v[m].mean()).mean()), 4)),
                      threshold=thr, live_z_def=W.bci((np.sign(zw) * d['fwd'])[ml], d['day'][ml]))
        res[k]['_dir'] = np.where(m, np.sign(d['zw']), 0.0); res[k]['_m'] = m
    base = res['b']['main']
    for k in ('bo', 'boy'):
        mb, mk = res['b']['_m'], res[k]['_m']
        res[k]['agreement_vs_binance'] = agreement(res[k]["_dir"], res["b"]["_dir"])
        res[k]['diff_vs_binance'] = diff_ci(D[k]['v'][mk], D[k]['day'][mk], D['b']['v'][mb], D['b']['day'][mb])
        res[k]['pass'] = passes(res[k]['main'], base)
    for k in ARMS: del res[k]['_dir'], res[k]['_m']
    win = [k for k in ('bo', 'boy') if res[k]['pass']]
    res['verdict'] = 'binance_keep' if not win else 'replace_with_' + max(win, key=lambda k: res[k]['main']['mean'])
    res['flags'] = dict(fresh_forward_bar_by_bar=False, note='연구 검정(시간 격자 이벤트) — 원 연구와 같은 방식, 승격 아님',
                        trade_ledgers_used_as_input=False, future_rows_used_for_entry=False)
    s = json.dumps(res, indent=1, default=str)
    open(f'{OUT}/verdict.json', 'w').write(s); print(s)


def selftest():
    # 1) 묶기: 연속 같은 (t,side,px) 만 한 줄, 떨어진 같은 키는 다른 줄 · 경계 $10k=중형 · $100k=고래
    t = np.array([0, 0, 0, 0, 5, 60000, 60000], np.int64); s = np.array([1, 1, -1, 1, 1, -1, -1.0])
    px = np.array([100, 100, 100, 100, 101, 102, 102.0]); q = np.array([60, 40, 50, 1, 2, 500, 500.0])
    o = minutes(t, s, px, q, True)
    # 줄: (0,+,100)=10,000 중형 · (0,−,100)=5,000 리테일 · (0,+,100)=100 리테일(떨어진 같은 키) · (5,+,101)=202 · (60000,−,102)=102,000 고래
    assert o.lines.tolist() == [4, 1] and o.fills.tolist() == [5, 2]
    assert np.isclose(o.mid_sn.iloc[0], 10_000) and np.isclose(o.ret_sn.iloc[0], -5000 + 100 + 202) and o.whl_nt.iloc[0] == 0
    assert np.isclose(o.whl_sn.iloc[1], -102_000) and o.c.tolist() == [101, 102]
    assert o.index[1] == pd.Timestamp(60000, unit='ms', tz='UTC')
    u = minutes(t, s, px, q, False)                                       # 바이낸스 = 줄마다(묶지 않음)
    assert u.lines.iloc[0] == 5 and np.isclose(u.ret_sn.iloc[0], 6000 + 4000 - 5000 + 100 + 202)
    # 2) 0.1ms 틱(바이비트): 분 경계
    o2 = minutes(np.array([599_999, 600_000], np.int64), np.array([1, 1.0]), np.array([1, 1.0]), np.array([1, 1.0]), True, tick=10)
    assert len(o2) == 2 and o2.index[1] == pd.Timestamp(60000, unit='ms', tz='UTC')
    # 3) 라이브 z: 상수 흐름 뒤 스파이크 → 양수 큰 z
    sr = pd.Series(np.r_[np.random.default_rng(0).normal(0, 1, 30000), np.full(60, 10.0)]); z = zlive(sr)
    assert np.isnan(z.iloc[100]) and z.iloc[-1] > 5
    # 4) 판정·일치율
    assert passes(dict(mean=8.1, ci=[1.0, 15]), dict(mean=8.0)) and not passes(dict(mean=7.9, ci=[1.0, 15]), dict(mean=8.0))
    assert not passes(dict(mean=9.0, ci=[-0.1, 15]), dict(mean=8.0)) and not passes(dict(mean=9.0, ci=None), dict(mean=8.0))
    a = agreement(np.array([1, -1, 0, 1, 1.0]), np.array([1, 1, 1, 0, 1.0]))
    assert a == dict(of_binance=0.5, of_arm=0.5, opposite=1), a
    dc = diff_ci(np.array([2.0, 2, 2]), ['a', 'b', 'c'], np.array([1.0, 1, 1]), ['a', 'b', 'c'])
    assert dc == dict(diff=1.0, ci=[1.0, 1.0])
    print('selftest ok')


if __name__ == '__main__':
    {'build': build, 'analyze': analyze, 'selftest': selftest}[sys.argv[1]]()
