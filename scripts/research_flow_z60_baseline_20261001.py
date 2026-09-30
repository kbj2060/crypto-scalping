"""대시보드 «② 누가 밀고 있나 · 60분» z60 의 기준선(분모·중심) 비교 — 2026-10-01.

현행(dashboard/server.py _flow_read_ctx): z = (직전 12개 마감 5분봉 그룹 순매수 합) / std(링 안 과거 60분 롤링합들),
  링 = 풋프린트 봉(재기동 직후 144봉=12h, 이후 최대 288봉=24h). 분자 중심 차감 없음.
데이터: binance.vision aggTrades 재구축 1분 테이프(«row» = aggTrade 한 줄 = 대시보드 경계 고래 ≥$100k · 리테일 <$10k),
  USD 순매수 ÷ 그 분 종가 = ETH(2026-09-20 하루를 aggTrades 로 정확히 다시 세 대조: 60분합 상관 1.000000, 최대 오차 σ의 0.6%).

후보(사전 고정, 8×2×2×2 = 64):
  창 h12 · h24(후행 롤링합 144·288봉) · d7 · d30 · hod7 · hod14 · hod30(이전 N일의 같은 UTC 시) · dow4(이전 4주 같은 요일·같은 UTC 시)
  × 척도 std · mad(1.4826·MAD) × 분자 net(ETH) · ratio(순매수÷같은 60분 총 테이커 거래량) × 중심 없음 · 차감(평균/중앙값).
  d7·d30 의 MAD 는 매시 정각에 직전 봉까지로 갱신(인과). hod·dow 는 이전 날짜만.
평가(결정 = 매시 정각, 비겹침 60분): 피쳐는 정각 이전 마감 5분봉만(봉 k = HH:55 봉, 종가 c[k]), 라벨 = log(c[k+12]/c[k]) bp.
  1 방향성: 그룹별 순위 IC · |z| 상위 10% · |z|≥1 조건부 적중률과 sign(z)×라벨 bp(단독은 되돌림 예상 — 부호 그대로 보고)
  2 맞대결: 고래·리테일 부호 반대 & 둘 다 |z|≥0.5 → sign(z_고래)×라벨 bp, 하루 빈도 (고래·중형도 보고)
  3 표시 공정성(전 5분봉): |z|≥1 비율의 UTC 시간별 표준편차(pp) · 한산 3시간/붐빔 3시간 비율(TRAIN 평균 거래량으로 시간 사전 고정)
  4 가격 모멘텀(직전 60분 수익) 순위 잔차 편IC
  CI = UTC 일 블록 부트스트랩(B=1000, 모든 후보가 같은 가중치를 공유 → 현행과의 짝지은 차이 CI).
선택 규칙(실행 전 고정):
  주 = TRAIN 맞대결(고래↔리테일) bp. 최고 후보와 짝지은 차이 CI 가 0 을 포함하면 «동률».
  동률 집합 안에서 공정성(시간별 std 최소) → 단순성(필요 이력 짧은 순) 으로 고른다.
  TEST 확인 = 승자 TEST 맞대결 bp CI 하한 > 0, 그리고 현행 대비 짝지은 차이 보고.
사후 추가(1차 결과를 본 뒤 — 선택 규칙에 안 씀): 빈도 맞춘 맞대결. |z|≥0.5 는 척도마다 발동 빈도가 1.3~3.9회/일로 달라
  «기준선이 낫다»와 «더 까다롭다»가 섞인다. 후보마다 TRAIN 에서 현행과 같은 하루 빈도가 되는 임계 θ 를 찾아 TEST 에 그대로 쓴다.

사용: python scripts/research_flow_z60_baseline_20261001.py <tape_dir> <SYM> <train_start> <out.json>
"""
import glob
import json
import sys

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv
from scipy.stats import rankdata

TAPE, SYM, T0, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
SPLIT = pd.Timestamp('2025-01-01', tz='UTC')
G = {'whale': 'whl', 'mid': 'mid', 'retail': 'ret'}
B = 1000
rng = np.random.default_rng(0)

# ── 5분봉 ────────────────────────────────────────────────────────────────
T = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{TAPE}/{SYM}/*.parquet'))]).sort_index()
T = T[~T.index.duplicated()].asfreq('1min')
T['c'] = T.c.ffill()
flow = T.filter(like='row_').fillna(0.0)
eth = pd.DataFrame({g: flow[f'row_{k}_sn'] / T.c for g, k in G.items()})
eth['vol'] = sum(flow[f'row_{k}_nt'] for k in G.values()) / T.c
F = eth.resample('5min').sum()
C = T.c.resample('5min').last()
ts = F.index; n = len(ts)
r60 = F.rolling(12).sum()                           # 봉 k 까지(포함) 60분 합
num = {'net': {g: r60[g].to_numpy() for g in G},
       'ratio': {g: (r60[g] / r60['vol']).to_numpy() for g in G}}
c = C.to_numpy()
day = ts.asi8 // 86_400_000_000_000
hour = ts.hour.to_numpy()
d0 = day[0]; ND = day[-1] - d0 + 1


def mad(a, med=np.median):
    """(중앙값, 1.4826·MAD). 후행창은 NaN 이 맨 앞 11봉뿐이라 np.median(빠름), 날짜창은 빈 날이 있어 nanmedian."""
    m = med(a, axis=-1, keepdims=True)
    return m[..., 0], 1.4826 * med(np.abs(a - m), axis=-1)


def trailing(x, L, est):
    """봉 k 의 (중심, 척도): x[k-L .. k-1]."""
    s = pd.Series(x).shift(1)
    if est == 'std':
        return s.rolling(L, min_periods=L // 2).mean().to_numpy(), s.rolling(L, min_periods=L // 2).std().to_numpy()
    cen = np.full(n, np.nan); sc = np.full(n, np.nan)
    if L <= 288:                                       # 매 봉 정확히
        W = swv(np.r_[np.full(L, np.nan), x[:-1]], L)  # 행 k = x[k-L..k-1]
        for i in range(0, n, 20000):
            cen[i:i + 20000], sc[i:i + 20000] = mad(W[i:i + 20000])
        return cen, sc
    hs = np.where(ts.minute.to_numpy() == 0)[0]        # 매시 정각에 직전 봉까지로 갱신
    for h in hs:
        if h >= L // 2:
            cen[h], sc[h] = mad(x[max(0, h - L):h])
    return pd.Series(cen).ffill().to_numpy(), pd.Series(sc).ffill().to_numpy()


def by_hour_days(x, lags, est):
    """이전 날짜들(lags)의 같은 UTC 시 값들로 (중심, 척도)."""
    A = np.full((ND, 24, 12), np.nan)
    A[day - d0, hour, ts.minute.to_numpy() // 5] = x
    S = np.stack([np.roll(A, L, axis=0) for L in lags], axis=-1)   # (ND,24,12,len)
    for j, L in enumerate(lags):
        S[:L, :, :, j] = np.nan
    S = S.reshape(ND, 24, -1)
    ok = np.isfinite(S).sum(-1) >= S.shape[-1] // 2
    if est == 'std':
        cen, sc = np.nanmean(S, -1), np.nanstd(S, -1, ddof=1)
    else:
        cen, sc = mad(S, np.nanmedian)
    cen[~ok] = np.nan; sc[~ok] = np.nan
    return cen[day - d0, hour], sc[day - d0, hour]


WINS = {'h12': ('t', 144, 0.5), 'h24': ('t', 288, 1), 'd7': ('t', 7 * 288, 7), 'd30': ('t', 30 * 288, 30),
        'hod7': ('h', list(range(1, 8)), 7), 'hod14': ('h', list(range(1, 15)), 14),
        'hod30': ('h', list(range(1, 31)), 30), 'dow4': ('h', [7, 14, 21, 28], 28)}
Z = {}
for wn, (kind, arg, _) in WINS.items():
    for est in ['std', 'mad']:
        for nn, X in num.items():
            for g, x in X.items():
                cen, sc = trailing(x, arg, est) if kind == 't' else by_hour_days(x, arg, est)
                sc = np.where(sc > 0, sc, np.nan)
                Z[(wn, est, nn, 'none', g)] = x / sc
                Z[(wn, est, nn, 'center', g)] = (x - cen) / sc
    print('done', wn, flush=True)
CANDS = sorted({k[:4] for k in Z})
BASE = ('h12', 'std', 'net', 'none')

# ── 결정·라벨 ───────────────────────────────────────────────────────────
k = np.where((ts.minute.to_numpy() == 55) & (np.arange(n) >= 12) & (np.arange(n) < n - 12))[0]
fwd = np.log(c[k + 12] / c[k]) * 1e4
past = np.log(c[k] / c[k - 12]) * 1e4
okall = np.isfinite(fwd) & np.isfinite(past) & (ts[k] >= pd.Timestamp(T0, tz='UTC'))
for key in Z:                                          # 모든 후보가 같은 표본(가장 긴 이력 기준)
    okall &= np.isfinite(Z[key][k])
per = {'TRAIN': okall & (ts[k] < SPLIT), 'TEST': okall & (ts[k] >= SPLIT)}
okbar = np.ones(n, bool)
for key in Z:
    okbar &= np.isfinite(Z[key])
okbar &= ts >= pd.Timestamp(T0, tz='UTC')
perbar = {'TRAIN': okbar & (ts < SPLIT), 'TEST': okbar & (ts >= SPLIT)}
vol_h = F.vol[perbar['TRAIN']].groupby(hour[perbar['TRAIN']]).mean()
QUIET, BUSY = list(vol_h.nsmallest(3).index), list(vol_h.nlargest(3).index)


class Boot:
    """UTC 일 블록 가중치 공유 부트스트랩."""
    def __init__(self, m):
        self.d = day[k][m] - d0
        u, self.inv = np.unique(self.d, return_inverse=True)
        self.W = rng.multinomial(len(u), np.ones(len(u)) / len(u), size=B).astype(float)
        self.nd = len(u)

    def mean(self, v, sel):
        s = np.bincount(self.inv[sel], v, self.nd); cnt = np.bincount(self.inv[sel], None, self.nd)
        return v.mean() if len(v) else np.nan, (self.W @ s) / np.maximum(self.W @ cnt, 1)

    def corr(self, x, y):
        cols = [np.bincount(self.inv, w, self.nd) for w in (np.ones_like(x), x, y, x * y, x * x, y * y)]
        n_, sx, sy, sxy, sxx, syy = (self.W @ col for col in cols)
        est = (n_ * sxy - sx * sy) / np.sqrt((n_ * sxx - sx ** 2) * (n_ * syy - sy ** 2))
        return np.corrcoef(x, y)[0, 1], est


def ci(pt, est):
    return [round(float(pt), 4), round(float(np.nanpercentile(est, 2.5)), 4), round(float(np.nanpercentile(est, 97.5)), 4)]


THETA = {}
res = {'sym': SYM, 'quiet_hours_utc': QUIET, 'busy_hours_utc': BUSY, 'n_candidates': len(CANDS), 'rows': []}
boots = {}
for pn, m in per.items():
    bt = boots[pn] = Boot(m)
    f = fwd[m]; ndays = bt.nd
    rp = rankdata(past[m]); rf = rankdata(f)
    ef = rf - np.polyval(np.polyfit(rp, rf, 1), rp)
    raw = {}
    for cand in CANDS:
        z = {g: Z[cand + (g,)][k][m] for g in G}
        row = {'cand': '|'.join(cand), 'per': pn, 'n': int(m.sum())}
        for g in G:
            rz = rankdata(z[g])
            row[f'IC_{g}'] = ci(*bt.corr(rz, rf))
            ez = rz - np.polyval(np.polyfit(rp, rz, 1), rp)
            row[f'pIC_{g}'] = round(float(np.corrcoef(ez, ef)[0, 1]), 4)
            top = np.abs(z[g]) >= np.quantile(np.abs(z[g]), 0.9)
            v = np.sign(z[g][top]) * f[top]
            row[f'top10_{g}'] = ci(*bt.mean(v, top)) + [round(float((v > 0).mean()), 4)]
            one = np.abs(z[g]) >= 1
            v = np.sign(z[g][one]) * f[one]
            row[f'z1_{g}'] = ci(*bt.mean(v, one)) + [round(float((v > 0).mean()), 4), round(float(one.mean()), 4)]
        for other in ['retail', 'mid']:
            dis = (np.sign(z['whale']) != np.sign(z[other])) & (np.abs(z['whale']) >= 0.5) & (np.abs(z[other]) >= 0.5)
            v = np.sign(z['whale'][dis]) * f[dis]
            pt, est = bt.mean(v, dis)
            raw[(cand, other)] = (pt, est)
            row[f'duel_{other}'] = ci(pt, est) + [round(float((v > 0).mean()), 4), round(float(dis.sum() / ndays), 3)]
        # 공정성(전 5분봉)
        mb = perbar[pn]
        fr = np.mean([pd.Series(np.abs(Z[cand + (g,)][mb]) >= 1).groupby(hour[mb]).mean().to_numpy() for g in G], axis=0)
        row['z1_frac'] = round(float(fr.mean()), 4)
        row['hour_std_pp'] = round(float(fr.std() * 100), 2)
        row['quiet_busy'] = round(float(fr[QUIET].mean() / fr[BUSY].mean()), 3)
        row['frac_by_hour'] = [round(float(x), 3) for x in fr]
        res['rows'].append(row)
    for cand in CANDS:                                   # 사후: 현행과 같은 빈도로 맞춘 맞대결(θ 는 TRAIN 에서만 고른다)
        zw, zr = Z[cand + ('whale',)][k][m], Z[cand + ('retail',)][k][m]
        opp = np.sign(zw) != np.sign(zr); lo_ = np.minimum(np.abs(zw), np.abs(zr))
        if pn == 'TRAIN':
            zb_w, zb_r = Z[BASE + ('whale',)][k][m], Z[BASE + ('retail',)][k][m]
            target = ((np.sign(zb_w) != np.sign(zb_r)) & (np.minimum(np.abs(zb_w), np.abs(zb_r)) >= 0.5)).sum()
            THETA[cand] = float(np.sort(lo_[opp])[::-1][min(int(target), opp.sum()) - 1])
        dis = opp & (lo_ >= THETA[cand])
        v = np.sign(zw[dis]) * f[dis]
        raw[(cand, 'matched')] = bt.mean(v, dis)
        r = next(x for x in res['rows'] if x['cand'] == '|'.join(cand) and x['per'] == pn)
        r['duel_matched'] = ci(*raw[(cand, 'matched')]) + [round(THETA[cand], 3), round(float(dis.sum() / ndays), 3)]
    for cand in CANDS:
        r = next(x for x in res['rows'] if x['cand'] == '|'.join(cand) and x['per'] == pn)
        a, b = raw[(cand, 'matched')], raw[(BASE, 'matched')]
        r['duel_matched_vs_base'] = ci(a[0] - b[0], a[1] - b[1])
    for cand in CANDS:                                   # 현행 대비 짝지은 차이
        r = next(x for x in res['rows'] if x['cand'] == '|'.join(cand) and x['per'] == pn)
        for other in ['retail', 'mid']:
            a, b = raw[(cand, other)], raw[(BASE, other)]
            r[f'duel_{other}_vs_base'] = ci(a[0] - b[0], a[1] - b[1])
    res[f'_raw_{pn}'] = {('|'.join(c_)): raw[(c_, 'retail')][1].tolist() for c_ in CANDS}
    print('eval', pn, flush=True)

# ── 선택 ────────────────────────────────────────────────────────────────
tr = {r['cand']: r for r in res['rows'] if r['per'] == 'TRAIN'}
te = {r['cand']: r for r in res['rows'] if r['per'] == 'TEST'}
best = max(tr, key=lambda c_: tr[c_]['duel_retail'][0])
rawtr = res.pop('_raw_TRAIN'); rawte = res.pop('_raw_TEST')
tied = []
for c_ in tr:
    d = np.array(rawtr[best]) - np.array(rawtr[c_])
    lo = np.nanpercentile(d, 2.5)
    if lo <= 0:
        tied.append(c_)
HIST = {w: WINS[w][2] for w in WINS}
tied.sort(key=lambda c_: (tr[c_]['hour_std_pp'], HIST[c_.split('|')[0]]))
win = tied[0]
res['selection'] = {'best_train_duel': best, 'n_tied_with_best': len(tied), 'tied': tied, 'winner': win,
                    'winner_test_duel': te[win]['duel_retail'], 'winner_test_vs_base': te[win]['duel_retail_vs_base'],
                    'base': '|'.join(BASE), 'base_train_duel': tr['|'.join(BASE)]['duel_retail'],
                    'base_test_duel': te['|'.join(BASE)]['duel_retail']}
json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1)
print(json.dumps(res['selection'], ensure_ascii=False, indent=1))
