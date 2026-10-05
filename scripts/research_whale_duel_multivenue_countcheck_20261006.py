"""맞대결 +OKX 판이 «사건 수가 많아서» 약해졌나 -- 사건 분해 + 사건 수 맞춘 비교(2026-10-06 사용자 «okx 합산은 사건 수가 많아서 떨어진 걸 수도 있는건지»).
데이터 = research_whale_duel_multivenue_20261006 build 결과(tmp/whale_multivenue_20261006/tape). python scripts/research_whale_duel_multivenue_countcheck_20261006.py"""
import sys, numpy as np, pandas as pd
A = __import__('os').path.dirname(__import__('os').path.abspath(__file__))
sys.path.insert(0, A)
import research_whale_duel_multivenue_20261006 as M
W = M.W
b, _ = M.load_binance()
V = {'binance': b, 'okx': M.load_venue('okx')}
idx = pd.date_range(M.START, pd.Timestamp(M.END) + pd.Timedelta('1D'), freq='1min', tz='UTC', inclusive='left')
T = pd.DataFrame({'c': V['binance'].c.reindex(idx)})
for k, vs in (('b', ('binance',)), ('bo', ('binance', 'okx'))):
    for g in M.GRP:
        T[f'{k}_{g}_sn'] = sum(V[v][f'{g}_sn'].reindex(idx).fillna(0.0) for v in vs)
main = lambda d: d['ts'] >= pd.Timestamp(M.MAIN, tz='UTC')
D = {k: W.duel(T, k) for k in ('b', 'bo')}
mb, mo = D['b']['dis'] & main(D['b']), D['bo']['dis'] & main(D['bo'])
sb, so = np.sign(D['b']['zw']), np.sign(D['bo']['zw'])
fwd, day = D['b']['fwd'], D['b']['day']
def show(name, m, sign):
    r = W.bci((sign * fwd)[m], day[m]); print(f"  {name:34s} n={int(m.sum()):5d}  {r['mean']:+6.2f} {r['ci']}")
print("① 사건 분해 (2025-01~2026-09, 건당 bp [95% CI])")
show("바이낸스 판 전체", mb, sb); show("+OKX 판 전체", mo, so)
same = mb & mo & (sb == so)
show("둘 다 발동·같은 방향", same, sb)
show("둘 다 발동·반대 방향(바이낸스 방향)", mb & mo & (sb != so), sb)
show("바이낸스만 발동", mb & ~mo, sb)
show("+OKX 만 발동", mo & ~mb, so)
print("② 사건 수 맞춤: 문턱(|z|)을 바꿔 가며 두 판의 사건 수·건당 bp")
rows = []
for th in (0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.0):
    for k in ('b', 'bo'):
        e = W.duel(T, k, th=th); m = e['dis'] & main(e); r = W.bci(e['v'][m], e['day'][m])
        rows.append((k, th, int(m.sum()), r['mean'], r['ci']))
for th in sorted({r[1] for r in rows}):
    a = [r for r in rows if r[1] == th]
    print(f"  th={th:<4}  바이낸스 n={a[0][2]:5d} {a[0][3]:+6.2f} {a[0][4]}   +OKX n={a[1][2]:5d} {a[1][3]:+6.2f} {a[1][4]}")
print("③ 같은 사건 수(바이낸스 th0.5 의 n)로 +OKX 를 «가장 강한 사건»만 남겨 자르기")
def zpair(key, Wm=60):   # W.duel 113~119행과 같은 z(고래·리테일)
    out = {}
    for g in ('whl', 'ret'):
        s = T[f'{key}_{g}_sn'].fillna(0).rolling(Wm).sum()
        sd = s.iloc[::Wm].rolling(30 * 1440 // Wm, min_periods=7 * 1440 // Wm).std().reindex(s.index).ffill().shift(Wm)
        out[g] = (s / sd).values[np.arange(Wm + 30 * 1440, len(T) - Wm, Wm)]
    return out
for k, base_n in (('bo', int(mb.sum())), ('b', int(mo.sum()))):
    z = zpair(k); st = np.minimum(np.abs(z['whl']), np.abs(z['ret']))
    cand = D[k]['ok'] & main(D[k]) & (np.sign(z['whl']) != np.sign(z['ret'])) & np.isfinite(st)
    cut = np.sort(st[cand])[::-1][base_n - 1]; m = cand & (st >= cut)
    r = W.bci((np.sign(z['whl']) * fwd)[m], day[m])
    print(f"  {'+OKX' if k == 'bo' else '바이낸스'} 상위 {int(m.sum())}건(강도 ≥ {cut:.3f}): {r['mean']:+6.2f} {r['ci']}")
