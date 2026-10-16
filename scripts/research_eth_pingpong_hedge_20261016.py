"""핑퐁장 롱·숏 동시 진입(헷지) 검정 (2026-10-16 사용자 «매번 진입할 때 핑퐁장에서 롱과 숏을 10%씩 들어가는건 어때?»).

진입 시점  대시보드 청산맵 벽(_spliced_raw, 닫힌 1h×24, 칸 ≥0.25) 첫 찍음 = 핑퐁 진입 시점(10-14 생존 실험실과 같은 기계,
           하루 ~3.6건 = 사용자 원장 빈도). 진입가 P0 = 찍은 5분봉 종가. 경로는 1분봉(binance vision), 최대 72h.
헷지      P0 에서 롱 1·숏 1(각 명목 1 = 10% 증거금 × 10배). 같은 종목 같은 수량이라 둘 다 열린 동안 순노출 0.
           ±x 중 먼저 닿는 쪽이 «이긴 다리».
  FADE    이긴 다리 지정가 익절(+x) · 진 다리 보유 → P0 복귀 시 본전 청산(지정가) | P0 기준 stop% 손절(테이커) | 72h 테이커.
  BRK     진 다리 −x 손절(테이커) · 이긴 다리 보유 → P0 기준 +(1+k)x 익절 | P0(본전) 손절 | 72h.
  72h 안에 ±x 를 못 닿으면 두 다리 다 72h 에 테이커 청산(손익 0, 수수료만).
등가 단일  (헷지와 별도 코드) FADE = P0±x 양쪽 지정가 OCO 역추세 진입, BRK = P0±x 양쪽 스탑 OCO 추세 진입, 청산 가격 같음.
           모든 사건에서 헷지 다리별 회계 합 == 단일 손익(수량 기준 정확 일치)을 assert → 헷지 = «나중에 한쪽으로 진입»이다.
DIR       같은 시점 벽 방향 단일 페이드(아래 벽 롱) · 익절 +x · 손절 stop · 72h — 사용자가 평소 하는 «방향 고르기».
비용      열기 c_open ∈ {0 (양쪽 지정가), 4bp (시장가)} 다리당 · 지정가 청산 0 · 손절·시간 청산 4bp. 펀딩: 두 다리가 열린 동안은
           상쇄, 한 다리만 남은 구간은 미반영(8h 당 ~1bp, 보고서에 한계로 적음).
위약      같은 기간 무작위 5분봉 시각에 같은 헷지(3시드) — 핑퐁 벽 맥락이 도움 되는지.
판정(고정) 주 2025-01~ · 보조 2022~24. 통과 = 주 기간 c_open=4 순 평균 > 0 · 일 블록 부트스트랩 CI 하한 > 0 · 무작위 대비 차 CI 하한 > 0 ·
           보조 기간 같은 부호. 격자 25칸이라 1칸 우연 통과는 통과로 안 친다(본페로니 대신 보조 기간 재현을 필수로).
원장      사용자 실원장 ETH 173 진입(10-14 ledger_replay_rows.csv) 시각에 헷지 FADE vs 사용자 방향 단일(같은 익절·손절).
"""
import glob, json, sys, zipfile
from pathlib import Path
import numpy as np, pandas as pd

WT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WT / 'scripts'))
ROOT = Path('/home/kbj20/crypto-scalping')
BV = ROOT / 'data/binance_vision'
OUT = WT / 'tmp/pingpong_hedge_20261016'
LEDGER = ROOT / '.claude/worktrees/orderflow-trading-dashboard-review-d5d836/tmp/pingpong_survival_20261014/ledger_replay_rows.csv'
TAKER, H, BIG = 4.0, 4320, 10 ** 9
XS = (0.003, 0.005, 0.0075, 0.01, 0.015)
STOPS = (0.02, 0.03, None)
TPK = (1, 2)
log = lambda m: print(m, flush=True)


def first(m):
    i = np.flatnonzero(m)
    return int(i[0]) if len(i) else BIG


def hedge(hh, ll, cc, p0, x, mode, par, coin):
    """롱·숏 각 수량 1/p0. 다리별로 체결가를 적고 손익을 따로 셈. 반환 (총손익 명목비, 청산 수수료 bp, 종류, 최저 평가, 갇힌 분, 보유 분).
    10-16 리뷰 수정: ① 보유 분 = 진입부터(옛 순차 복리는 ±x 닿은 뒤부터 센 «갇힌 분»을 써 거래 수가 부풀었다)
    ② 닿은 1분봉 k 안에서 손절선(FADE)·목표선(BRK)까지 간 것도 인정 -- 같은 방향 연속 경로라 ±x 를 먼저 지난다. P0 복귀는 순서 불명이라 k+1 부터."""
    up, dn = first(hh >= p0 * (1 + x)), first(ll <= p0 * (1 - x))
    if up == dn == BIG:
        return 0.0, 2 * TAKER, 'flat', 0.0, 0, len(hh)
    w = 1 if (up < dn or (up == dn and coin)) else -1             # 같은 1분봉에 둘 다 닿으면 동전
    k = min(up, dn); px = p0 * (1 + w * x)
    h2, l2 = hh[k + 1:], ll[k + 1:]
    if mode == 'fade':                                           # 이긴 다리 익절, 진 다리(-w) 본전까지 보유
        stop = par
        be = first(l2 <= p0) if w == 1 else first(h2 >= p0)
        st = BIG if stop is None else (first(h2 >= p0 * (1 + stop)) if w == 1 else first(l2 <= p0 * (1 - stop)))
        if stop is not None and (hh[k] >= p0 * (1 + stop) if w == 1 else ll[k] <= p0 * (1 - stop)):
            st, be = -1, BIG                                     # 닿은 봉 안에서 이미 손절선
        if st < BIG and st <= be:
            ex, fee, kind, e = p0 * (1 + w * stop), TAKER, 'stop', st
        elif be < BIG:
            ex, fee, kind, e = p0, 0.0, 'be', be
        else:
            ex, fee, kind, e = cc[-1], TAKER, 'time', len(h2) - 1
        win_leg = w * (px - p0) / p0                             # 이긴 다리: P0 → px
        lose_leg = -w * (ex - p0) / p0                           # 진 다리: P0 → ex
        worst = x - stop if (kind == 'stop' and e < 0) else (x - w * ((h2[:e + 1].max() if w == 1 else l2[:e + 1].min()) - p0) / p0 if len(h2) else x)   # e<0 = 닿은 봉 손절(시간 청산의 빈 h2 도 −1 이라 kind 로 가른다)
        return win_leg + lose_leg, fee, kind, min(worst, 0.0), e + 1, k + 2 + e
    tpk = par                                                    # BRK: 진 다리 −x 손절, 이긴 다리(w) 보유
    tp = p0 * (1 + w * x * (1 + tpk))
    hit = first(h2 >= tp) if w == 1 else first(l2 <= tp)
    st = first(l2 <= p0) if w == 1 else first(h2 >= p0)
    if hh[k] >= tp if w == 1 else ll[k] <= tp:
        hit, st = -1, BIG                                        # 닿은 봉 안에서 이미 목표선
    if st < BIG and st <= hit:
        ex, fee, kind, e = p0, TAKER, 'stop', st
    elif hit < BIG:
        ex, fee, kind, e = tp, 0.0, 'tp', hit
    else:
        ex, fee, kind, e = cc[-1], TAKER, 'time', len(h2) - 1
    lose_leg = -w * (px - p0) / p0
    win_leg = w * (ex - p0) / p0
    worst = 0.0 if (kind == 'tp' and e < 0) or not len(h2) else w * ((l2[:e + 1].min() if w == 1 else h2[:e + 1].max()) - px) / p0
    return win_leg + lose_leg, TAKER + fee, kind, min(worst, 0.0), e + 1, k + 2 + e


def single(hh, ll, cc, p0, x, mode, par, coin):
    """등가 단일: P0±x 양쪽 대기 주문(OCO). FADE = 지정가 역추세, BRK = 스탑 추세. 반환 (손익, 수수료 bp, 진입했나)."""
    sell_at, buy_at = p0 * (1 + x), p0 * (1 - x)
    t_up, t_dn = first(hh >= sell_at), first(ll <= buy_at)
    if t_up == t_dn == BIG:
        return 0.0, 0.0, False
    went_up = t_up < t_dn or (t_up == t_dn and coin)
    k = min(t_up, t_dn); ent = sell_at if went_up else buy_at
    d = (-1 if went_up else 1) if mode == 'fade' else (1 if went_up else -1)
    h2, l2 = hh[k + 1:], ll[k + 1:]
    if mode == 'fade':
        tgt, stp, efee = p0, (None if par is None else p0 * (1 - d * par)), 0.0
    else:
        tgt, stp, efee = p0 * (1 + d * x * (1 + par)), p0, TAKER
    a = first(h2 >= tgt) if d == 1 else first(l2 <= tgt)
    b = BIG if stp is None else (first(l2 <= stp) if d == 1 else first(h2 >= stp))
    if mode == 'fade' and stp is not None and (ll[k] <= stp if d == 1 else hh[k] >= stp):
        a, b = BIG, -1                                           # 체결 봉 안에서 이미 손절선(진입가를 지나 같은 방향으로 계속)
    if mode == 'brk' and (hh[k] >= tgt if d == 1 else ll[k] <= tgt):
        a, b = -1, BIG                                           # 체결 봉 안에서 이미 목표선
    if b < BIG and b <= a:
        ex, fee = stp, TAKER
    elif a < BIG:
        ex, fee = tgt, 0.0
    else:
        ex, fee = cc[-1], TAKER
    return d * (ex - ent) / p0, efee + fee, True


def direct(hh, ll, cc, p0, s, x, stop):
    """DIR: 벽 방향 s 로 P0 단일 진입 · +x 익절(지정가) · stop 손절 · 72h."""
    a = first(hh >= p0 * (1 + x)) if s == 1 else first(ll <= p0 * (1 - x))
    b = BIG if stop is None else (first(ll <= p0 * (1 - stop)) if s == 1 else first(hh >= p0 * (1 + stop)))
    if b < BIG and b <= a:
        return -stop, TAKER, 'stop', b + 1
    if a < BIG:
        return x, 0.0, 'tp', a + 1
    return s * (cc[-1] - p0) / p0, TAKER, 'time', len(cc)


def variants():
    for x in XS:
        for st in STOPS:
            yield f'FADE x{x*100:g}% stop{"없음" if st is None else f"{st*100:g}%"}', 'fade', x, st
        for k in TPK:
            yield f'BRK x{x*100:g}% tp{(1+k)*x*100:g}%', 'brk', x, k


def selftest():
    p0 = 100.0
    # 위로 1% → 롱 익절, 숏 남음 → 다시 100 → 본전. 헷지 +1% == 등가 단일(101 숏 → 100) +1%.
    hh = np.array([100.5, 101.2, 100.8, 100.1, 100.2]); ll = np.array([99.8, 100.6, 100.2, 99.9, 99.9]); cc = ll
    g, f, kind, *_ = hedge(hh, ll, cc, p0, 0.01, 'fade', 0.03, False)
    assert kind == 'be' and abs(g - 0.01) < 1e-12 and f == 0, (g, f, kind)
    assert abs(single(hh, ll, cc, p0, 0.01, 'fade', 0.03, False)[0] - g) < 1e-12
    # 위로 1% 뒤 계속 올라 103 → 숏 손절: +1% − 3% = −2%
    hh = np.array([101.1, 102, 103.1]); ll = np.array([100.5, 101, 102]); cc = ll
    g, f, kind, *_ = hedge(hh, ll, cc, p0, 0.01, 'fade', 0.03, False)
    assert kind == 'stop' and abs(g + 0.02) < 1e-12 and f == TAKER
    g2, *_ = hedge(hh, ll, cc, p0, 0.01, 'brk', 1, False)       # BRK: 숏 −1% 컷, 롱 102 익절 → 합 +1%
    assert abs(g2 - 0.01) < 1e-12 and abs(single(hh, ll, cc, p0, 0.01, 'brk', 1, False)[0] - g2) < 1e-12
    # 닿은 봉(0) 안에서 +1% 를 지나 +3% 까지 → 진 다리(숏) 그 봉에서 손절: +1% − 3% = −2% · 보유 1분 · 단일도 같음
    hh = np.array([103.2, 102.0]); ll = np.array([100.1, 101.0]); cc = ll
    g, f, kind, w_, trap, hold = hedge(hh, ll, cc, p0, 0.01, 'fade', 0.03, False)
    assert kind == 'stop' and abs(g + 0.02) < 1e-12 and hold == 1, (g, kind, hold)
    assert abs(single(hh, ll, cc, p0, 0.01, 'fade', 0.03, False)[0] - g) < 1e-12
    g, *_ = hedge(hh, ll, cc, p0, 0.01, 'brk', 1, False)        # BRK: 같은 봉에서 목표 102 → 숏 −1% + 롱 +2% = +1%
    assert abs(g - 0.01) < 1e-12 and abs(single(hh, ll, cc, p0, 0.01, 'brk', 1, False)[0] - g) < 1e-12
    # 마지막 봉에서 ±x 에 닿고 손절 없음 → 빈 h2 시간 청산(10-16 재실행이 여기서 죽었다)
    g, f, kind, *_ = hedge(np.array([100.2, 101.5]), np.array([99.8, 100.4]), np.array([100.0, 101.2]), p0, 0.01, 'fade', None, False)
    assert kind == 'time' and abs(g - (0.01 - 0.012)) < 1e-12, (g, kind)
    # 아무것도 안 닿음 → 0, 두 다리 시간 청산
    g, f, kind, *_ = hedge(np.full(5, 100.2), np.full(5, 99.8), np.full(5, 100.0), p0, 0.01, 'fade', 0.03, False)
    assert g == 0 and f == 2 * TAKER and kind == 'flat' and _[-1] == 5
    log('selftest OK')


# ---------------------------------------------------------------- 데이터
def load_1m():
    parts = [pd.read_parquet(f) for f in sorted(glob.glob(f'{BV}/klines1m/ETHUSDT-1m-*.parquet')) +
             sorted(glob.glob(f'{BV}/klines1m_daily/ETHUSDT-1m-*.parquet'))]
    for z in sorted(glob.glob(str(OUT / 'vision_1m/*.zip'))):
        with zipfile.ZipFile(z) as zf:
            x = pd.read_csv(zf.open(zf.namelist()[0]))
        parts.append(pd.DataFrame({'t': x.open_time, 'h': x.high, 'l': x.low, 'c': x.close}))
    K = pd.concat([p[['t', 'h', 'l', 'c']] for p in parts]).drop_duplicates('t').sort_values('t').reset_index(drop=True)
    log(f'[1분봉] {len(K):,}개 {pd.to_datetime(K.t.iloc[0], unit="ms")} ~ {pd.to_datetime(K.t.iloc[-1], unit="ms")} · 빈칸 {(np.diff(K.t) != 60000).sum()}곳')
    return K.t.to_numpy(), K.h.to_numpy(float), K.l.to_numpy(float), K.c.to_numpy(float)


def wall_candidates():
    """10-14 survival_lab.py 의 벽 첫 찍음 후보 그대로(복사). (5분봉 시각, 벽 방향 s) 목록."""
    cache = OUT / 'cand.parquet'
    if cache.exists():
        return pd.read_parquet(cache)
    import live_liquidation_map_20260824 as LM
    d = pd.read_parquet(BV / 'panel/ETHUSDT.parquet').sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
    hi, lo = d.high.to_numpy(float), d.low.to_numpy(float); ts = d.timestamp; n = len(d)
    h1 = d.set_index('timestamp')[['high', 'low', 'close', 'volume']].resample('1h').agg(
        {'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
    hs = np.flatnonzero(ts.dt.minute.to_numpy() == 0)
    dn = np.full(n, np.nan); up = np.full(n, np.nan)
    for j, k in enumerate(hs):
        h0 = ts.iloc[k]
        win = h1.loc[h0 - pd.Timedelta(hours=24):h0 - pd.Timedelta(hours=1)].reset_index()
        if len(win) < 20:
            continue
        E = win.close.iloc[-1]; raw = LM._spliced_raw(win, E)
        if raw is None:
            continue
        sup, res, bw = raw; ed = []
        for bins, below in ((sup, True), (res, False)):
            pr = np.array([b * bw for b in bins]); w = np.array(list(bins.values()))
            m = np.abs(pr - E) / E <= 0.05
            if not m.any():
                break
            pr, w = pr[m], w[m]; pr = pr[w / w.max() >= 0.25]
            ed.append(pr.max() + bw / 2 if below else pr.min() - bw / 2)
        if len(ed) == 2:
            e_ = hs[j + 1] if j + 1 < len(hs) else n
            dn[k:e_], up[k:e_] = ed
    hid = ts.dt.floor('1h').to_numpy()
    t_dn, t_up = np.isfinite(dn) & (lo <= dn), np.isfinite(up) & (hi >= up)
    f_dn = t_dn & (pd.Series(t_dn.astype(int)).groupby(hid).cumsum().to_numpy() == 1)
    f_up = t_up & (pd.Series(t_up.astype(int)).groupby(hid).cumsum().to_numpy() == 1)
    c = pd.DataFrame({'ts': np.r_[ts.to_numpy()[f_dn], ts.to_numpy()[f_up]],
                      's': np.r_[np.ones(f_dn.sum(), int), -np.ones(f_up.sum(), int)]}).sort_values('ts').reset_index(drop=True)
    c.to_parquet(cache)
    return c


def boot_mean(gs, B=2000, seed=1):
    """일 블록 부트스트랩: gs = [(일별 합, 일별 건수)...] 같은 날짜축. 반환 각 그룹 평균의 표본 (B, len(gs))."""
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(gs[0][0]), (B, len(gs[0][0])))
    return [s[idx].sum(1) / np.maximum(c[idx].sum(1), 1) for s, c in gs]


def main():
    selftest()
    OUT.mkdir(parents=True, exist_ok=True)
    t1, h1m, l1m, c1m = load_1m()
    cand = wall_candidates()
    log(f'[벽 후보] {len(cand):,}건')

    def at(t_ms):                                                # 진입 1분봉 인덱스(그 분봉 종가가 P0)
        i = np.searchsorted(t1, t_ms)
        ok = (i < len(t1)) & (t1[np.minimum(i, len(t1) - 1)] == t_ms) & (i + H < len(t1))
        return np.where(ok, i, -1)

    cand['i'] = at((cand.ts + pd.Timedelta(minutes=4)).astype('int64') // 10 ** 6)   # 5분봉 τ 종가 = 1분봉 τ+4분 종가
    cand = cand[cand.i >= 0].reset_index(drop=True)
    cand['per'] = np.where(cand.ts >= pd.Timestamp('2025-01-01'), 'main', 'aux')
    rng = np.random.default_rng(7)
    rows = []

    def run_events(ev, tag):
        for r in ev.itertuples():
            i = r.i; hh, ll, cc = h1m[i + 1:i + 1 + H], l1m[i + 1:i + 1 + H], c1m[i + 1:i + 1 + H]
            p0 = c1m[i]; coin = bool(rng.random() < 0.5)
            for name, mode, x, par in variants():
                g, f, kind, worst, trap, hold = hedge(hh, ll, cc, p0, x, mode, par, coin)
                gs, fs, entered = single(hh, ll, cc, p0, x, mode, par, coin)
                assert abs(g - gs) < 1e-12, (name, r.ts, g, gs)      # 헷지 = 등가 단일
                rows.append((tag, r.per, r.ts, name, g * 1e4, f, fs, kind, worst * 1e4, trap, hold))
            if tag == 'wall':
                for x in XS:
                    for st in STOPS:
                        g, f, kind, hold = direct(hh, ll, cc, p0, r.s, x, st)
                        rows.append((tag, r.per, r.ts, f'DIR x{x*100:g}% stop{"없음" if st is None else f"{st*100:g}%"}',
                                     g * 1e4, f, f, kind, np.nan, 0, hold))

    run_events(cand, 'wall')
    log('[벽] 완료')
    for seed in range(3):
        rr = np.random.default_rng(100 + seed)
        for p in ('main', 'aux'):
            ci = cand.i[cand.per == p]; m = len(ci)
            ii = rr.integers(ci.min(), ci.max(), m * 8)
            ii = np.sort(ii[(t1[ii] // 60000) % 5 == 4][:m])          # 5분봉 종가 분만
            run_events(pd.DataFrame({'i': ii, 'per': p, 'ts': pd.to_datetime(t1[ii], unit='ms'), 's': 0}), f'rand{seed}')
    log('[무작위] 완료')
    R = pd.DataFrame(rows, columns=['tag', 'per', 'ts', 'name', 'gross', 'fee_close', 'fee_single', 'kind', 'worst', 'trap', 'hold'])
    R.to_parquet(OUT / 'events.parquet')
    summarize(R)
    ledger(t1, h1m, l1m, c1m)


def summarize(R):
    R = R.copy(); R['day'] = R.ts.dt.floor('1D')
    hed = ~R.name.str.startswith('DIR')
    for c_open in (4.0, 0.0):                                    # 헷지: 두 다리 열기 2×c_open · DIR: 한 다리
        R[f'net{c_open:g}'] = R.gross - R.fee_close - np.where(hed, 2 * c_open, c_open)
    R['net_single'] = R.gross - R.fee_single
    W = R[R.tag == 'wall']; Z = R[R.tag.str.startswith('rand')]
    out, lines = {}, []
    for name in W.name.unique():
        rec = {}
        for p in ('main', 'aux'):
            w = W[(W.name == name) & (W.per == p)]; z = Z[(Z.name == name) & (Z.per == p)]
            days = pd.Index(sorted(set(w.day) | set(z.day)))
            gw = w.groupby('day').net4.agg(['sum', 'count']).reindex(days, fill_value=0)
            gz = z.groupby('day').net4.agg(['sum', 'count']).reindex(days, fill_value=0)
            bw, bz = boot_mean([(gw['sum'].to_numpy(), gw['count'].to_numpy()), (gz['sum'].to_numpy(), gz['count'].to_numpy())])
            rec[p] = dict(n=len(w), per_day=len(w) / w.day.nunique(), net4=float(w.net4.mean()),
                          ci4=(float(np.quantile(bw, .025)), float(np.quantile(bw, .975))),
                          net0=float(w.net0.mean()), single=float(w.net_single.mean()), gross=float(w.gross.mean()),
                          win=float((w.net4 > 0).mean()), kinds=w.kind.value_counts(normalize=True).round(3).to_dict(),
                          p5=float(w.net4.quantile(0.05)), worst_trade=float(w.net4.min()),
                          worst_float=float(w.worst.min()) if w.worst.notna().any() else None,
                          trap_h_med=float(w.trap[w.trap > 0].median() / 60) if (w.trap > 0).any() else 0.0)
            if len(z):
                d_ = bw - bz
                rec[p].update(rand=float(z.net4.mean()), diff_ci=(float(np.quantile(d_, .025)), float(np.quantile(d_, .975))))
        mn, ax = rec['main'], rec['aux']
        is_h = not name.startswith('DIR')
        rec['pass'] = bool(mn['net4'] > 0 and mn['ci4'][0] > 0 and (not is_h or mn['diff_ci'][0] > 0) and ax['net4'] > 0)
        out[name] = rec
        lines.append(f"{name:24s} | 2025~ n{mn['n']:5d} 순{mn['net4']:+6.1f}bp[{mn['ci4'][0]:+.1f},{mn['ci4'][1]:+.1f}] "
                     f"지정가열기{mn['net0']:+6.1f} 단일{mn['single']:+6.1f} 승{mn['win']*100:3.0f}% p5{mn['p5']:+5.0f} 최악{mn['worst_trade']:+5.0f}"
                     + (f" 무작위{mn['rand']:+6.1f} 차[{mn['diff_ci'][0]:+.1f},{mn['diff_ci'][1]:+.1f}]" if is_h else '')
                     + f" | 22-24 순{ax['net4']:+6.1f}" + (f" 무작위{ax['rand']:+6.1f}" if is_h else '') + f" | {'통과' if rec['pass'] else '-'}")
    log('\n'.join(lines))
    log('\n■ 순차 복리 (다리당 명목 1.0×순자산 = 10% 증거금 × 10배, 시장가 열기)')
    for name in W.name.unique():
        res = {}
        for p in ('main', 'aux'):
            w = W[(W.name == name) & (W.per == p)].sort_values('ts')
            eq, peak, mdd, busy, cnt = 1.0, 1.0, 0.0, pd.Timestamp(0), 0
            for r in w.itertuples():
                if r.ts < busy:
                    continue
                eq *= 1 + r.net4 / 1e4; peak = max(peak, eq); mdd = min(mdd, eq / peak - 1); cnt += 1
                busy = r.ts + pd.Timedelta(minutes=5 + r.hold)   # 진입부터 청산까지(flat = 72h)
            yrs = (w.ts.max() - w.ts.min()).days / 365.25
            res[p] = (eq ** (1 / yrs) - 1, mdd, cnt)
        log(f"{name:24s} 2025~ 연{res['main'][0]*100:+7.1f}% MDD{res['main'][1]*100:6.1f}% ({res['main'][2]}건) | "
            f"22-24 연{res['aux'][0]*100:+7.1f}% MDD{res['aux'][1]*100:6.1f}%")
        out[name]['seq'] = res
    (OUT / 'report.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))


def ledger(t1, h1m, l1m, c1m):
    if not LEDGER.exists():
        log('[원장] 파일 없음 — 건너뜀'); return
    L = pd.read_csv(LEDGER, parse_dates=['et'])
    k = (L.et.dt.floor('1min').astype('int64') // 10 ** 6).to_numpy()
    i = np.searchsorted(t1, k)
    ok = (i < len(t1)) & (t1[np.minimum(i, len(t1) - 1)] == k)
    log(f'\n■ 원장 {len(L)}건 중 1분봉 있음 {ok.sum()} (72h 못 채운 끝 구간은 있는 데까지)')
    rows = []
    rng = np.random.default_rng(3)
    for r, ii in zip(L[ok].itertuples(), i[ok]):
        e = min(ii + 1 + H, len(t1)); hh, ll, cc = h1m[ii + 1:e], l1m[ii + 1:e], c1m[ii + 1:e]
        p0 = c1m[ii]; coin = bool(rng.random() < 0.5)
        for x in (0.005, 0.0075, 0.01):
            g, f, kind, *_ = hedge(hh, ll, cc, p0, x, 'fade', 0.03, coin)
            gd, fd, kd, _ = direct(hh, ll, cc, p0, r.side, x, 0.03)
            rows.append((x, g * 1e4 - f - 8, gd * 1e4 - fd - 4, kind, kd))
    D = pd.DataFrame(rows, columns=['x', 'hedge', 'dir', 'hk', 'dk'])
    for x, g in D.groupby('x'):
        dif = (g.dir - g.hedge).to_numpy()
        bs = np.random.default_rng(0).choice(dif, (2000, len(dif))).mean(1)
        log(f'x{x*100:g}% stop3%: 헷지 FADE {g.hedge.mean():+6.1f}bp · 사용자 방향 단일 {g.dir.mean():+6.1f}bp · '
            f'차 {dif.mean():+6.1f}[{np.quantile(bs, .025):+.1f},{np.quantile(bs, .975):+.1f}] · '
            f'헷지 {g.hk.value_counts(normalize=True).round(2).to_dict()} · 방향 {g.dk.value_counts(normalize=True).round(2).to_dict()}')
    D.to_csv(OUT / 'ledger_hedge.csv', index=False)


if __name__ == '__main__':
    selftest() if '--selftest' in sys.argv else main()
