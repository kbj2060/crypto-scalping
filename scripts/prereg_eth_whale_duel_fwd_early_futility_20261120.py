"""조기 점검 0(기각 전용): ETH 고래↔리테일 맞대결 전진 검정 — 설계 원문 docs/prereg/eth_whale_duel_fwd_early_futility_20261120.md

  --selftest      판정 규칙·검정력 표·잠금
  --fetch         일별 zip 받아 1분 집계만(통계 안 봄) · 누락·체결 분 수 출력
  --look0         1회 채점. 결과 파일이 있으면 거부.
정의(집계·맞대결·z·부트스트랩)는 prereg_eth_whale_duel_fwd_20261007.py 를 import 해서 그대로 쓴다. 원 점검 1·2 는 건드리지 않는다.
"""
import json, math, sys
from pathlib import Path

import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prereg_eth_whale_duel_fwd_20261007 as F  # noqa: E402

W = F.W
END = pd.Timestamp('2026-11-14', tz='UTC')                 # 받는 마지막 날. 11-15~ 는 다른 연구의 봉인 구간이라 안 받는다
LAST_DECISION = pd.Timestamp('2026-11-14 22:00', tz='UTC')
Z_REJECT, Z_CAND = -1.96, -1.28
MIN_N, MIN_DAYS = 20, 10
OUT = F.ROOT / 'docs/experiments/eth_whale_duel_fwd_early_look0.json'
PERM_SEED, N_PERM = 20261120, 2000


def decide0(z, n, days):
    """기각 전용. 확정은 없다(z 가 아무리 커도 계속)."""
    if z is None or n < MIN_N or days < MIN_DAYS: return '판정 불가 — 계속'
    return '기각' if z <= Z_REJECT else '기각 후보' if z <= Z_CAND else '계속'


def load():
    days = pd.date_range(F.WARM, END)
    ps = [F.fetch_day(d) for d in days]
    miss = [f'{d:%Y-%m-%d}' for d, p in zip(days, ps) if p is None]
    T = W.load([p for p in ps if p])
    mins = T.c.notna().groupby(T.index.floor('D')).sum()
    thin = {f'{d:%Y-%m-%d}': int(v) for d, v in mins.items() if v < 1300}
    return T, miss, thin


def look0():
    if OUT.exists(): print(f'이미 채점함: {OUT} — 다시 돌리지 않는다.'); return
    T, miss, thin = load()
    assert T.index[-1] == END + pd.Timedelta(minutes=1439), T.index[-1]
    E = F.events(T)
    D = W.duel(T, F.KEY, Wm=F.WM, th=F.TH); m = D['dis'] & (D['ts'] >= F.FREEZE)
    assert len(E) == int(m.sum()) and np.allclose(E.v.to_numpy(), D['v'][m]) and E.ts.max() <= LAST_DECISION
    v, fwd, side = D['v'][m], D['fwd'][m], np.sign(D['zw'][m]); day = E.ts.dt.floor('D').to_numpy()
    r = F.zstat(v, day)
    res = decide0(r.get('z'), r['n'], r.get('event_days', 0))
    perm = np.array([np.mean(s * fwd) for s in np.random.default_rng(PERM_SEED).choice([-1.0, 1.0], (N_PERM, len(fwd)))]) if len(fwd) else np.array([])
    out = dict(look=0, kind='기각 전용', window=[str(F.FREEZE.date()), str(LAST_DECISION)], missing_days=miss, thin_days=thin, **r,
               z_reject=Z_REJECT, z_candidate=Z_CAND, result=res,
               report_only=dict(maker_net=None if 'mean_bp' not in r else round(r['mean_bp'] - 2.2, 3),
                                taker_net=None if 'mean_bp' not in r else round(r['mean_bp'] - 8, 3),
                                always_long_bp=round(float(np.mean(fwd)), 3) if len(fwd) else None,
                                always_short_bp=round(float(-np.mean(fwd)), 3) if len(fwd) else None,
                                whale_buy_events=int((side > 0).sum()), whale_sell_events=int((side < 0).sum()),
                                perm_pctile=round(float((perm < np.mean(v)).mean() * 100), 1) if len(perm) else None))
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    if res == '기각':                                        # 전진 검정 종료 — 원 스크립트의 gate 가 점검 1·2 를 잠근다
        F.LOOKS.write_text(json.dumps([dict(look=0, result='기각', z=r['z'], mean_bp=r['mean_bp'], n=r['n'], source=OUT.name)], ensure_ascii=False, indent=1))
    line = (f"- 조기 점검 0 — 기각 전용 ({pd.Timestamp.now(tz='UTC').date()}, 창 {out['window'][0]} ~ {out['window'][1]}, "
            f"docs/prereg/eth_whale_duel_fwd_early_futility_20261120.md): z {r.get('z')} / 기각 ≤ {Z_REJECT} · 후보 ≤ {Z_CAND} → **{res}**"
            + (f" · 평균 {r['mean_bp']:+.2f}bp {r['ci95']} · 사건 {r['n']}건({r['event_days']}일)" if 'mean_bp' in r else f" · 사건 {r['n']}건") + '\n')
    F.DOC.write_text(F.DOC.read_text().rstrip('\n') + '\n' + line)
    print(json.dumps(out, ensure_ascii=False, indent=1))


def fetch():
    T, miss, thin = load()
    print(f'집계 {T.index[0]} ~ {T.index[-1]} · {len(T)}분 · 일별 zip 누락 {len(miss)}일 {miss} · 체결 분 1,300 미만인 날 {thin}')


def selftest():
    assert [decide0(z, 58, 30) for z in (-1.96, -1.95, -1.28, -1.27, 5.0)] == ['기각', '기각 후보', '기각 후보', '계속', '계속']   # 확정 없음
    assert decide0(-3.0, 19, 30) == decide0(-3.0, 58, 9) == decide0(None, 58, 30) == '판정 불가 — 계속'
    phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))  # noqa: E731
    se = 77.9 / math.sqrt(1.53 * 38)                         # 사전등록 검정력 표의 산수
    tab = {mu: (round(phi(Z_REJECT - mu / se), 3), round(phi(Z_CAND - mu / se), 3)) for mu in (6.17, 0, -6.17, -11)}
    assert tab[6.17][0] < 0.006 and 0.02 < tab[6.17][1] < 0.04 and tab[0] == (0.025, 0.1) and 0.08 < tab[-6.17][0] < 0.1 and 0.17 < tab[-11][0] < 0.2, tab
    assert 0.26 < phi(-6.17 / se) < 0.28                     # 원 규칙(z<0 중단)을 38일에 쓰면 참 효과도 27% 기각
    assert F.gate(1, [dict(look=0, result='기각')], pd.Timestamp('2030-01-01', tz='UTC'))   # 기각 기록이 점검 1·2 를 잠근다
    assert F.gate(1, [], pd.Timestamp('2027-11-26', tz='UTC')) is None                       # 기각이 아니면 원 일정 그대로
    assert LAST_DECISION + pd.Timedelta(minutes=60) <= END + pd.Timedelta(minutes=1439)      # 마지막 결정이 받은 데이터 안에서 해소
    print('selftest OK', tab)


if __name__ == '__main__':
    a = sys.argv
    if '--selftest' in a: selftest()
    elif '--fetch' in a: fetch()
    elif '--look0' in a: look0()
    else: print(__doc__)
