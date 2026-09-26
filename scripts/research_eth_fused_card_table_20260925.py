# 입력: scripts/research_eth_fused_signal_votes_gate_20260925.py 를 FUSED_Z=24h(서버 정의)로 읽는다. 테이프는 scripts/build_aggtrades_size3_tape_1m_20260925.py.
"""융합 카드의 3결과 확률표 — 카드 축(±0.5×30분 창폭, 다음 30분 1분봉 선착)에서 잰 조건부 빈도.
셀 = 레짐(추세/횡보) × 정렬 점수(추세: S1×dir, 횡보: S1) 구간 × 창폭 24h 분위 삼분위. 결과 = 추세 {with, against, none} / 횡보 {up, dn, none}.
결정 = 모든 5분봉(라벨 겹침 → CI 는 일 블록). TRAIN 표 → TEST 로그손실을 «표 없음(레짐×창폭)» 기준과 비교."""
import sys, numpy as np, pandas as pd, json
import os; os.environ['FUSED_Z'] = '24h'
src = open('scripts/research_eth_fused_signal_votes_gate_20260925.py').read().split("for H in (")[0]
sys.argv = ['x', 'ETHUSDT']; exec(src)
m1h, m1l = T.h.values.astype(np.float32), T.l.values.astype(np.float32)
pos = np.searchsorted(T.index.values, b5.index.values + np.timedelta64(5, 'm'))
whale1 = np.where(V['wr'] != 0, V['wr'], V['wm']); S1 = whale1 + V['oi'] + V['al'] + V['rj']
idx = np.where(warm & (np.arange(n) < n - 8) & np.isfinite(rg30) & (rg30 > 0) & np.isfinite(rgq))[0]
idx = idx[pos[idx] + 30 <= len(m1h)]
W = np.lib.stride_tricks.sliding_window_view
H30 = W(m1h, 30)[pos[idx]]; L30 = W(m1l, 30)[pos[idx]]
up = (c[idx] * (1 + 0.5 * rg30[idx] / 1e4)).astype(np.float32); dn = (c[idx] * (1 - 0.5 * rg30[idx] / 1e4)).astype(np.float32)
hu = H30 >= up[:, None]; hd = L30 <= dn[:, None]
fu = np.where(hu.any(1), hu.argmax(1), 99); fd = np.where(hd.any(1), hd.argmax(1), 99)
res = np.where((fu == 99) & (fd == 99), 'none', np.where(fu == fd, 'amb', np.where(fu < fd, 'up', 'dn')))
keep = res != 'amb'; idx, res = idx[keep], res[keep]
d = dir30[idx]; trend = d != 0
a = np.where(trend, S1[idx] * d, S1[idx]); abin = np.clip(a, -1, 2)           # ≤−1 · 0 · +1 · ≥+2
gbin = np.digitize(rgq[idx], [1 / 3, 2 / 3])                                   # 0 1 2
out = np.where(res == 'none', 'none', np.where(trend, np.where((res == 'up') == (d > 0), 'with', 'against'),
                                               np.where(res == 'up', 'up', 'dn')))
te = test[idx]; dd = day[idx]
df = pd.DataFrame({'reg': np.where(trend, 'trend', 'range'), 'a': abin, 'g': gbin, 'out': out, 'te': te, 'd': dd})
K1 = {'trend': ['with', 'against', 'none'], 'range': ['up', 'dn', 'none']}

def table(sub, keys):
    t = sub.groupby(keys + ['out']).size().unstack(fill_value=0)
    return t

def probs(t, reg):
    cols = K1[reg]; x = t.reindex(columns=cols, fill_value=0).astype(float) + 1.0   # 라플라스 +1
    return x.div(x.sum(1), axis=0)

def logloss(train, test_, keys):
    ll = []
    for reg in ('trend', 'range'):
        tr = train[train.reg == reg]; ts = test_[test_.reg == reg]
        P = probs(table(tr, keys), reg)
        k = ts[keys].apply(tuple, axis=1) if len(keys) > 1 else ts[keys[0]]
        pm = P.reindex(k.values if len(keys) > 1 else ts[keys[0]].values)
        colidx = pd.Index(K1[reg]).get_indexer(ts.out)
        p = pm.values[np.arange(len(ts)), colidx]
        ll.append(-np.log(np.where(np.isfinite(p), p, 1 / 3)))
    return np.concatenate(ll).mean()

tr, ts = df[~df.te], df[df.te]
print(f"결정 {len(df)} · TRAIN {len(tr)} · TEST {len(ts)} · 결과 비율 {df.out.value_counts(normalize=True).round(3).to_dict()}")
for keys, nm in [(['reg'], '레짐만'), (['reg', 'g'], '레짐×창폭'), (['reg', 'a'], '레짐×표'), (['reg', 'a', 'g'], '레짐×표×창폭(융합)')]:
    print(f"  TEST 로그손실 {nm:14s} {logloss(tr, ts, keys):.5f}")
# 일 블록 부트스트랩: 융합 − 레짐×창폭 로그손실 차 (TEST)
rng2 = np.random.default_rng(1)
def ll_rows(train, test_, keys):
    out_ = np.zeros(len(test_))
    for reg in ('trend', 'range'):
        m = (test_.reg == reg).values; tsr = test_[m]; P = probs(table(train[train.reg == reg], keys), reg)
        pm = P.reindex(pd.MultiIndex.from_frame(tsr[keys]) if len(keys) > 1 else tsr[keys[0]].values)
        p = pm.values[np.arange(len(tsr)), pd.Index(K1[reg]).get_indexer(tsr.out)]
        out_[m] = -np.log(np.where(np.isfinite(p), p, 1 / 3))
    return out_
l_f = ll_rows(tr, ts, ['reg', 'a', 'g']); l_b = ll_rows(tr, ts, ['reg', 'g']); diff = l_b - l_f
g = pd.DataFrame({'v': diff, 'd': ts.d.values}).groupby('d').v.agg(['sum', 'count'])
Wt = rng2.multinomial(len(g), np.ones(len(g)) / len(g), size=1000); est = (Wt @ g['sum'].values) / (Wt @ g['count'].values)
print(f"  융합이 줄인 로그손실(TEST, 일 블록) {diff.mean():+.5f} [{np.percentile(est, 2.5):+.5f},{np.percentile(est, 97.5):+.5f}]")
# 셀 표(전체 표본) + TRAIN/TEST 비교
full = {}
for reg in ('trend', 'range'):
    t_all = table(df[df.reg == reg], ['a', 'g']); P_all = probs(t_all, reg)
    P_tr = probs(table(tr[tr.reg == reg], ['a', 'g']), reg); P_te = probs(table(ts[ts.reg == reg], ['a', 'g']), reg)
    print(f"\n[{reg}] 셀(a,g): n · 전체 {K1[reg]} · TRAIN · TEST (첫 칸 %)")
    for key in P_all.index:
        nn = int(t_all.loc[key].sum()); k0 = K1[reg][0]
        print(f"  a={key[0]:+d} g={key[1]}  n{nn:6d}  " + " ".join(f"{100 * P_all.loc[key, k]:5.1f}" for k in K1[reg])
              + f"  | {k0} TR {100 * P_tr.loc[key, k0] if key in P_tr.index else float('nan'):5.1f} TE {100 * P_te.loc[key, k0] if key in P_te.index else float('nan'):5.1f}")
        full[f"{reg}|{key[0]}|{key[1]}"] = {"n": nn, **{k: round(float(P_all.loc[key, k]), 4) for k in K1[reg]}}
os.makedirs('tmp/fusion', exist_ok=True)
json.dump(full, open('tmp/fusion/card_table.json', 'w'), ensure_ascii=False, indent=0)
