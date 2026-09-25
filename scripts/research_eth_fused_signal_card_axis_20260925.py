# 🔴위 스크립트의 z 창은 30일(사전 정의)이다. 서버 정의(24h)로 재려면 거기서 rolling(8640…) 을 rolling(288, min_periods=84).std().shift(1) 로 바꾼다(tmp/fusion/build_eval_24h.py).
"""융합 신호를 카드의 축(Y_sym: ±0.5×30분 창폭, 다음 30분 1분봉 선착)으로 잰다. 서버 정의(24h z · 고래 한 표).
추세(dir≠0): P(이동 방향 먼저 = B). 횡보(dir=0): P(위 먼저). 'none'·'amb' 는 방향 질문의 답이 아니라 뺀다(카드 규약)."""
import sys, numpy as np, pandas as pd
src = open('scripts/research_eth_fused_signal_votes_gate_20260925.py').read().split("for H in (")[0]
sys.argv = ['x', 'ETHUSDT']; exec(src)
m1h, m1l = T.h.values, T.l.values; t1 = T.index
pos = np.searchsorted(t1.values, b5.index.values + np.timedelta64(5, 'm'))   # 5분봉 i 종료 = 1분봉 인덱스
whale1 = np.where(V['wr'] != 0, V['wr'], V['wm']); S1 = whale1 + V['oi'] + V['al'] + V['rj']
fused = np.where((np.abs(S1) >= 2) & G, np.sign(S1), 0)
idx = np.arange(0, n - 7, 6); idx = idx[warm[idx]]
lab = np.full(len(idx), '', object)
for j, i in enumerate(idx):
    up, dn = c[i] + 0.5 * rg30[i] * c[i] / 1e4, c[i] - 0.5 * rg30[i] * c[i] / 1e4
    a = pos[i]; hh, ll = m1h[a:a + 30], m1l[a:a + 30]
    hu = np.flatnonzero(hh >= up); hd = np.flatnonzero(ll <= dn)
    fu = hu[0] if len(hu) else 99; fd = hd[0] if len(hd) else 99
    lab[j] = 'none' if fu == fd == 99 else 'amb' if fu == fd else ('up' if fu < fd else 'dn')
d = dir30[idx]; f = fused[idx]; te = test[idx]; dd = day[idx]
ok = (lab == 'up') | (lab == 'dn'); upv = (lab == 'up').astype(float)
def rate(sel):
    u, di = np.unique(dd[sel], return_inverse=True); s = np.bincount(di, y[sel], len(u)); k = np.bincount(di, None, len(u))
    W = rng.multinomial(len(u), np.ones(len(u)) / len(u), size=1000); e = (W @ s) / (W @ k)
    return f"{100 * y[sel].mean():5.1f}% [{100 * np.percentile(e, 2.5):.1f},{100 * np.percentile(e, 97.5):.1f}] n{sel.sum()}"
print("추세 구간 — P(B: 이동 방향 먼저)")
y = np.where(d > 0, upv, 1 - upv)
for per, m in [('TR', ~te), ('TE', te)]:
    base = ok & (d != 0) & m
    print(f"  {per} 기저 {rate(base)} | 융합=이동방향 {rate(base & (f == d))} | 융합=반대 {rate(base & (f == -d))} | 융합 없음 {rate(base & (f == 0))}")
print("횡보 구간 — P(위로 먼저)")
y = upv
for per, m in [('TR', ~te), ('TE', te)]:
    base = ok & (d == 0) & m
    print(f"  {per} 기저 {rate(base)} | 융합 롱 {rate(base & (f > 0))} | 융합 숏 {rate(base & (f < 0))}")
print("미도달·동시 비율:", pd.Series(lab).value_counts(normalize=True).round(3).to_dict())
print("융합 발동의 카드 레짐 분포:", pd.Series(np.where(f != 0, np.where(d == 0, '횡보', np.where(f == d, '추세·같은쪽', '추세·반대')), '-'))[f != 0].value_counts().to_dict())
