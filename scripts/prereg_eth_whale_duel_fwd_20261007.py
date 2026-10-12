"""사전등록 전진 검정: ETH 고래↔리테일 맞대결, 고래 추종 60분 (동결 2026-10-07 · 판정 데이터 2026-10-08 00:00 UTC~).

설계 원문: docs/experiments/eth_whale_duel_fwd_prereg_20261007.md (이 스크립트와 같은 커밋 = 동결).
  --status  판정 창 경과 일수·사건 수만(평균 안 봄)          --look N  N번째 점검일 이후에만 판정
  --parity  일별 zip = 월 zip = 원 테이프 집계 대조 · ETH 원 테이프로 +6.17 재현 · 워밍업 충분성 · 검정력   --selftest
정의(집계·맞대결)는 scripts/research_whale_duel_solxrp_20261004.py 를 import 해서 그대로 쓴다.
데이터 = data.binance.vision 정적 아카이브(API 아님). fapi/api REST 는 호출하지 않는다.
"""
import glob, json, math, os, sys, time, urllib.error, urllib.request
from pathlib import Path

import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_whale_duel_solxrp_20261004 as W  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'tmp/whale_duel_fwd'
DOC = ROOT / 'docs/experiments/eth_whale_duel_fwd_prereg_20261007.md'
LOOKS = ROOT / 'docs/experiments/eth_whale_duel_fwd_looks.json'
URL = 'https://data.binance.vision/data/futures/um/daily/aggTrades/ETHUSDT/ETHUSDT-aggTrades-{d}.zip'
BANDS = {'a': (10_000, 100_000), 'b': (10_000, 100_000)}      # 리테일 < $10k · 고래 ≥ $100k (ETH 원 연구 경계)
KEY, WM, TH = 'row_b', 60, 0.5
WARM = pd.Timestamp('2026-09-01', tz='UTC')                    # 37일 워밍업: z 분모 = 직전 30일 60분 합 표준편차
FREEZE = pd.Timestamp('2026-10-08', tz='UTC')
LOOK_DAYS, Z_CRIT = (414, 828), (2.80, 1.98)                  # O'Brien-Fleming 2회(정보 50%·100%), 단측 α=.025 · 828일 = 참 +6.17bp 에서 검정력 80%(사용자 10-07 선택)
SEED, N_BOOT = 20261008, 5000


def fetch_day(d):
    """일별 zip 받기 → 1분 집계(W.aggregate_stream) → zip 삭제. 아직 공개 안 된 날 = None."""
    p = CACHE / f'ETHUSDT/{d:%Y-%m-%d}.parquet'
    if p.exists():
        return p
    p.parent.mkdir(parents=True, exist_ok=True); zp = CACHE / f'_dl_{d:%Y-%m-%d}.zip'
    try:
        for k in range(3):
            try:
                urllib.request.urlretrieve(URL.format(d=f'{d:%Y-%m-%d}'), zp); break
            except urllib.error.HTTPError as e:
                if e.code == 404: return None
                time.sleep(5 * (k + 1))
            except OSError:
                time.sleep(5 * (k + 1))
        else:
            return None
        W.aggregate_stream(W.read_zip(zp), d.value // 10**6, 1440, BANDS).to_parquet(f'{p}.part'); os.replace(f'{p}.part', p)
        return p
    finally:
        zp.unlink(missing_ok=True)                                 # 디스크: zip 은 바로 지운다


def load_fwd():
    days = pd.date_range(WARM, pd.Timestamp.now(tz='UTC').normalize() - pd.Timedelta(days=1))
    ps = [fetch_day(d) for d in days]
    last = max(d for d, p in zip(days, ps) if p)
    return W.load([p for p in ps if p]), sum(1 for d, p in zip(days, ps) if p is None and d < last)   # 중간 누락 = asfreq NaN(원 연구와 같음)


def events(T, wm=WM):
    D = W.duel(T, KEY, Wm=wm, th=TH)                               # 비겹침 격자 · 고래 부호 × 다음 wm분 log 수익 bp
    m = D['dis'] & (D['ts'] >= FREEZE)
    return pd.DataFrame({'ts': D['ts'][m], 'v': D['v'][m]})


def zstat(v, day):
    """평균 bp / SE. SE = UTC 일 블록 부트스트랩(다항 가중 = W.bci 와 같은 방식, 5,000회, 고정 시드)의 표준편차."""
    g = pd.DataFrame({'v': v, 'd': day}).groupby('d').v.agg(['sum', 'count'])
    if len(g) < 2: return dict(n=int(len(v)), z=None)                # 보고 전용 하위표본이 빌 때만
    w =np.random.default_rng(SEED).multinomial(len(g), np.full(len(g), 1 / len(g)), size=N_BOOT)
    est = (w @ g['sum'].to_numpy()) / (w @ g['count'].to_numpy())
    m, se = float(np.mean(v)), float(est.std())
    return dict(n=int(len(v)), event_days=int(len(g)), mean_bp=round(m, 3), se=round(se, 3), z=round(m / se, 3),
                ci95=[round(float(x), 2) for x in np.percentile(est, [2.5, 97.5])])


def decide(z, li):
    return '확정' if z >= Z_CRIT[li] else '기각' if z < 0 or li == len(LOOK_DAYS) - 1 else '계속'


def gate(n, looks, now):
    """--look n 을 지금 해도 되는가. 거부 사유 문자열, 되면 None."""
    if any(r['result'] != '계속' for r in looks): return f"판정 완료: {looks[-1]['result']}"
    if n != len(looks) + 1 or n > len(LOOK_DAYS): return f'다음 점검은 {len(looks) + 1} 이다.'
    due = FREEZE + pd.Timedelta(days=LOOK_DAYS[n - 1])
    if now < due: return f'점검 {n} 은 {due.date()} 이후에만 — 지금 {now.date()}. 결과를 보지 않는다.'
    return None


def look(n):
    looks = json.loads(LOOKS.read_text()) if LOOKS.exists() else []
    why = gate(n, looks, pd.Timestamp.now(tz='UTC'))
    if why: print(why); return
    li = n - 1; due = FREEZE + pd.Timedelta(days=LOOK_DAYS[li])
    T, miss = load_fwd()
    if T.index[-1] < due:                                          # 창 마지막 23:00 결정은 due 00:00 분봉으로 해소
        print(f'점검 {n}: {due.date()} 일별 zip 아직 — 데이터 끝 {T.index[-1]}. 결과를 보지 않는다.'); return
    E = events(T); E = E[E.ts < due]; day = E.ts.dt.floor('D').to_numpy(); we = (E.ts.dt.dayofweek >= 5).to_numpy()
    E15 = events(T, 15); E15 = E15[E15.ts < due]
    r = zstat(E.v.to_numpy(), day)
    r = dict(look=n, window=[str(FREEZE.date()), str((due - pd.Timedelta(days=1)).date())], missing_days=miss, **r,
             z_crit=Z_CRIT[li], result=decide(r['z'], li),
             report_only=dict(weekday=zstat(E.v.to_numpy()[~we], day[~we]), weekend=zstat(E.v.to_numpy()[we], day[we]),
                              m15=zstat(E15.v.to_numpy(), E15.ts.dt.floor('D').to_numpy()), taker_8bp_net=round(r['mean_bp'] - 8, 3)))
    looks.append(r); LOOKS.write_text(json.dumps(looks, ensure_ascii=False, indent=1))
    DOC.write_text(DOC.read_text().replace('(없음)\n', '') +
                   f"- 점검 {n} ({pd.Timestamp.now(tz='UTC').date()}, 창 {r['window'][0]}~{r['window'][1]}): z {r['z']:.2f} / 문턱 {r['z_crit']}"
                   f" → **{r['result']}** · 평균 {r['mean_bp']:+.2f}bp {r['ci95']} · 사건 {r['n']}\n")
    print(json.dumps(r, ensure_ascii=False, indent=1))


def status():
    T, miss = load_fwd(); E = events(T)
    days = max((T.index[-1].normalize() - FREEZE).days + 1, 0)
    print(f'판정 창 경과 {days}일 · 사건 {len(E)}건' + (f' · 일별 zip 중간 누락 {miss}일' if miss else ''))


def parity():
    d = WARM; cols = [f'{KEY}_{g}_{x}' for g in W.GRP for x in ('sn', 'nt')]
    a = pd.concat([pd.read_parquet(fetch_day(x)) for x in pd.date_range(d, periods=30)])
    mp = CACHE / '_parity_monthly_2026-09.parquet'
    if not mp.exists():                                            # 월 zip 484MB — 한 번만 받고 집계만 남긴다
        zp = CACHE / '_dl_2026-09.zip'
        try:
            urllib.request.urlretrieve(W.URL.format(s='ETHUSDT', m='2026-09'), zp)
            W.aggregate_stream(W.read_zip(zp), d.value // 10**6, 30 * 1440, BANDS).to_parquet(mp)
        finally:
            zp.unlink(missing_ok=True)
    b = pd.read_parquet(mp)
    assert (a.index == b.index).all() and np.array_equal(a.c.to_numpy(), b.c.to_numpy(), equal_nan=True)
    dm = float(np.abs(a[cols].to_numpy() - b[cols].to_numpy()).max())
    assert np.allclose(a[cols], b[cols], rtol=1e-12, atol=1e-6)
    print(f'(a) 2026-09 일별 zip 30개 vs 월 zip(같은 집계 코드): 종가 {len(a)}분 일치 · {KEY} 6열 최대 차 {dm:.2e} USD')
    a = a.iloc[:1440]
    if not os.path.isdir(W.ETH_TAPE):
        print('(b) ETH 원 테이프 없음 — +6.17 재현 생략'); return
    o = pd.read_parquet(f'{W.ETH_TAPE}/2026-09-01.parquet').reindex(a.index)
    oc = [f'row_{g}_{x}' for g in W.GRP for x in ('sn', 'nt')]
    assert np.array_equal(a.c.to_numpy(), o.c.to_numpy(), equal_nan=True) and np.allclose(a[cols], o[oc].fillna(0), rtol=1e-9, atol=1e-6)
    print(f"(a') 같은 날 원 연구 테이프(tmp/whale/build.py, 일별 zip): 종가 일치 · row 6열 최대 차 {float(np.abs(a[cols].to_numpy() - o[oc].fillna(0).to_numpy()).max()):.2e} USD")
    E = W.load(glob.glob(f'{W.ETH_TAPE}/*.parquet'))
    r = W.report(E, 'row', full=False)
    assert abs(r['all']['mean'] - 6.17) < 0.01, r['all']
    print(f"(b) ETH 원 테이프 {r['span'][0][:10]}~{r['span'][1][:10]}: {r['all']['mean']:+.2f}bp n={r['all']['n']} · 연도 {r['years']} (+6.17 재현)")
    F = W.duel(E, 'row'); s0 = pd.Timestamp('2026-07-01', tz='UTC'); S = W.duel(E.loc[s0:], 'row'); cut = s0 + (FREEZE - WARM)
    f, s = F['ts'] >= cut, S['ts'] >= cut
    assert (F['ts'][f] == S['ts'][s]).all() and (F['dis'][f] == S['dis'][s]).all() and np.allclose(F['v'][f], S['v'][s])
    print(f'(c) 워밍업 {(FREEZE - WARM).days}일로 자른 테이프 = 전체 이력: {cut.date()}~ 결정 {int(f.sum())}개 · 사건 {int(F["dis"][f].sum())}개 동일')
    m = F['dis']; v = F['v'][m]; h = zstat(v, F['day'][m]); rate = m.sum() / (F['ok'].sum() / 24)
    sd = h['se'] * math.sqrt(h['n']); phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))  # noqa: E731
    ez = lambda days: 6.17 / (sd / math.sqrt(days * rate))  # noqa: E731
    print(f"(d) 검정력: 건당 SD {float(np.std(v)):.1f}bp · 일 블록 유효 SD {sd:.1f}bp · {rate:.2f}건/일 · "
          f"E[z] 414일 {ez(414):.2f}(점검1 통과확률 {phi(ez(414) - 2.80):.0%}) · 828일 {ez(828):.2f}({phi(ez(828) - 1.98):.0%}) · "
          f"E[z]=1.98 에 {(1.98 * sd / 6.17) ** 2 / rate:.0f}일 · 80% 검정력에 {((1.98 + 0.84) * sd / 6.17) ** 2 / rate:.0f}일")


def selftest():
    r = np.random.default_rng(1); day = np.repeat(np.arange(400), 2); v = r.normal(6, 60, len(day))
    z = zstat(v, day); naive = v.mean() / (v.std() / np.sqrt(len(v)))
    assert abs(z['z'] / naive - 1) < 0.1 and zstat(v, day) == z, (z, naive)       # 독립 자료면 블록 SE ≈ 단순 SE · 고정 시드
    assert zstat(np.r_[v, v], np.r_[day, day])['se'] > z['se'] * 0.95              # 같은 날 복제 = 정보 안 늘어남(SE ≈ 그대로)
    assert [decide(x, 0) for x in (2.80, 2.79, -0.01)] == ['확정', '계속', '기각'] and [decide(x, 1) for x in (1.98, 1.97)] == ['확정', '기각']
    ts = lambda s: pd.Timestamp(s, tz='UTC')  # noqa: E731
    assert FREEZE + pd.Timedelta(days=414) == ts('2027-11-26') and FREEZE + pd.Timedelta(days=828) == ts('2029-01-13')
    assert gate(1, [], ts('2027-11-25 23:59')) and gate(1, [], ts('2027-11-26')) is None and gate(2, [], ts('2030-01-01'))
    assert gate(2, [{'result': '계속'}], ts('2029-01-12')) and gate(2, [{'result': '계속'}], ts('2029-01-13')) is None
    assert gate(2, [{'result': '기각'}], ts('2030-01-01'))
    print('selftest OK')


if __name__ == '__main__':
    a = sys.argv
    if '--selftest' in a: selftest()
    elif '--parity' in a: parity()
    elif '--look' in a: look(int(a[a.index('--look') + 1]))
    else: status()
