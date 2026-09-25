"""현재 상황 18칸 중 융합 표에 없는 두 후보를 먹일 가치가 있나 — 카드 3결과(TEST 로그손실)와 방향 기울기로 잰다.
  act  활동 분위: 마지막 1분 거래대금의 «같은 UTC 시각» 과거 7일 분위(카드 act_pct 재구성). hot = ≥0.8 (카드 ACT_HOT)
  bo   전환 탐지 재구성: 5분 거래대금 z288 과 5분 체결건수 z288 이 둘 다 과거 2016봉 q90 이상(탐지기 AND 규칙)
기준 = 현행 융합 표(레짐 × a × g). 후보를 셀에 한 축 더 얹어 TRAIN 표 → TEST 로그손실 차(일 블록 CI)."""
import sys, numpy as np, pandas as pd
src = open('scripts/research_eth_fused_card_table_20260925.py').read().split("# 셀 표(전체 표본)")[0]
sys.argv = ['x', 'ETHUSDT']; exec(src)
# ── act: 1분 거래대금의 같은 시각 7일 분위 ──
v1 = sum(T[f'row_{g}_nt'].fillna(0) for g in ['ret', 'mid', 'whl'])
hr = T.index.hour
act1 = pd.Series(np.nan, index=T.index)
for h in range(24):
    s = v1[hr == h]
    act1[s.index] = s.rolling(7 * 60, min_periods=3 * 60).rank(pct=True).values
act5 = act1.reindex(b5.index + pd.Timedelta('4min')).values        # 5분봉 i 의 마지막 1분(봉 종가 직전 1분)
# ── bo: 탐지기 AND (거래대금 z288, 체결건수 z288, 각자 과거 2016봉 q90) ──
qv5 = v1.resample('5min').sum().reindex(b5.index).values
nt5 = sum(T[f'row_{g}_n'].fillna(0) for g in ['ret', 'mid', 'whl']).resample('5min').sum().reindex(b5.index).values
def z288(x):
    s = pd.Series(x); return ((s - s.rolling(288, min_periods=144).mean()) / s.rolling(288, min_periods=144).std()).values
zq, zn = z288(qv5), z288(nt5)
q90 = lambda z: pd.Series(z).rolling(2016, min_periods=500).quantile(0.9).shift(1).values   # noqa: E731
bo = (zq >= q90(zq)) & (zn >= q90(zn))
df['hot'] = (act5[idx] >= 0.8).astype(int)
df['bo'] = bo[idx].astype(int)
df = df[np.isfinite(act5[idx])]
tr, ts = df[~df.te], df[df.te]
print(f"결정 {len(df)} · hot {df.hot.mean():.3f} · bo {df.bo.mean():.3f} · corr(hot,g2) {np.corrcoef(df.hot, df.g == 2)[0,1]:.2f} · corr(bo,g2) {np.corrcoef(df.bo, df.g == 2)[0,1]:.2f}")
base = ll_rows(tr, ts, ['reg', 'a', 'g'])
for extra in (['hot'], ['bo'], ['hot', 'bo']):
    l2 = ll_rows(tr, ts, ['reg', 'a', 'g'] + extra); diff = base - l2
    g_ = pd.DataFrame({'v': diff, 'd': ts.d.values}).groupby('d').v.agg(['sum', 'count'])
    Wt = rng2.multinomial(len(g_), np.ones(len(g_)) / len(g_), size=1000); est = (Wt @ g_['sum'].values) / (Wt @ g_['count'].values)
    print(f"  +{'+'.join(extra):7s} TEST 로그손실 {base.mean():.5f} → {l2.mean():.5f}  감소 {diff.mean():+.5f} [{np.percentile(est, 2.5):+.5f},{np.percentile(est, 97.5):+.5f}]")
# 방향: 추세 구간에서 with/(with+against) — hot/bo 켜짐 vs 꺼짐, TRAIN/TEST
t = df[(df.reg == 'trend') & (df.out != 'none')]
for col in ('hot', 'bo'):
    for nm, m in (('TR', ~t.te), ('TE', t.te)):
        on, off = t[m & (t[col] == 1)], t[m & (t[col] == 0)]
        print(f"  추세 지속률 {col} {nm}: 켜짐 {100 * (on.out == 'with').mean():.1f}% n{len(on)} · 꺼짐 {100 * (off.out == 'with').mean():.1f}% n{len(off)}")
    for nm, m in (('TR', ~df.te), ('TE', df.te)):
        s = df[m]; print(f"  미도달률 {col} {nm}: 켜짐 {100 * (s[s[col] == 1].out == 'none').mean():.1f}% · 꺼짐 {100 * (s[s[col] == 0].out == 'none').mean():.1f}%")
# 같은 융합 셀(레짐×a×g) 안에서 켜짐−꺼짐 지속률 차(셀 가중 평균) — 방향 축에 증분이 있나
for col in ('hot', 'bo'):
    for nm, m in (('TR', ~t.te), ('TE', t.te)):
        s = t[m]; diffs = []; w = []
        for key, grp in s.groupby(['a', 'g']):
            on, off = grp[grp[col] == 1], grp[grp[col] == 0]
            if len(on) >= 100 and len(off) >= 100:
                diffs.append((on.out == 'with').mean() - (off.out == 'with').mean()); w.append(len(on))
        print(f"  셀 안 지속률 차 {col} {nm}: {100 * np.average(diffs, weights=w):+.2f}pp (셀 {len(diffs)}개)")
