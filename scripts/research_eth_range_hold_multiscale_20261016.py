"""박스 «유지 vs 이탈» 재검증 — 30분·1h·4h·24h 창 (2026-10-16 사용자 «①부터 재검증하고 시안 · 짧은 지평 실험을 긴 지평으로»).

원 연구(09-22, situation_within_regime_feature_screen): 30분 창 횡보에서 range_bp(창 고저폭) AUC .66 · 유지 7.6→27.7%.
그 결과는 09-22 엔진에 들어갔다가 09-25 상황 카드 백엔드 제거 때 같이 빠졌다(지금 화면에 없음).

정의(독립 재구현, 1분봉 경로)
  결정  5분봉 종가 분(분 % 5 == 4). 창 W분 = 그 분까지 W개 1분봉. 30분·1h 는 5분마다, 4h·24h 는 1시간마다 결정.
  박스  상단 = 종가 + 0.5×창폭, 하단 = 종가 − 0.5×창폭(원 연구 Y_sym 과 같은 등거리 배리어).
  유지  다음 H=W분(i+1..i+H) 1분봉 고가 < 상단 그리고 저가 > 하단. 순서가 필요 없어 미래 굴림 최대·최소로 정확.
  횡보  |창 이동| / 창폭 슈미트(.45 들어감 / .25 나감, 원 연구 상수) = 0. 전체 상태도 같이 낸다.
통제    🔴창이 넓으면 배리어가 멀다(기하). 기하 = d_σ = 0.5×창폭/종가 ÷ (σ24 × √H), σ24 = 직전 24h 1분 로그수익 RMS.
특징    A = log range_bp · G = log d_σ · F = G + log(σ1h/σ24) + log range_bp + UTC 시각 sin/cos + 주말.
학습    2022-01~2024-12 로지스틱. 채점 2025 · 2026(~10-14) 따로. AUC(일 블록 부트스트랩) · Brier 기술점수(학습 기저율 대비) · 10분위 보정.
판정    «기하를 넘는 정보» = 2025·2026 둘 다 AUC(F) − AUC(G) CI 하한 > 0. «쓸모(목적 정확도)» = F 보정 오차 10분위 최대 ≤ 5pp & BSS > 0.
"""
import glob, json, sys, zipfile
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

WT = Path(__file__).resolve().parents[1]
BV = Path('/home/kbj20/crypto-scalping/data/binance_vision')
EXTRA = WT / 'tmp/pingpong_hedge_20261016/vision_1m'
OUT = WT / 'tmp/range_hold_20261016'
SCALES = {'30m': (30, 5), '1h': (60, 5), '4h': (240, 60), '24h': (1440, 60)}   # (W=H 분, 결정 간격 분)
ENTER, EXIT = 0.45, 0.25
log = lambda m: print(m, flush=True)


def load_1m():
    parts = [pd.read_parquet(f)[['t', 'h', 'l', 'c']] for f in sorted(glob.glob(f'{BV}/klines1m/ETHUSDT-1m-*.parquet')) +
             sorted(glob.glob(f'{BV}/klines1m_daily/ETHUSDT-1m-*.parquet'))]
    for z in sorted(glob.glob(str(EXTRA / '*.zip'))):
        with zipfile.ZipFile(z) as zf:
            x = pd.read_csv(zf.open(zf.namelist()[0]))
        parts.append(pd.DataFrame({'t': x.open_time, 'h': x.high, 'l': x.low, 'c': x.close}))
    K = pd.concat(parts).drop_duplicates('t').sort_values('t').reset_index(drop=True)
    assert (np.diff(K.t) == 60000).all(), '1분봉 빈칸'
    return K


def fut_ext(x, H, fn):
    """out[i] = fn(x[i+1..i+H]) (끝 H개는 NaN)."""
    r = getattr(pd.Series(x[::-1]).rolling(H), fn)().to_numpy()[::-1]     # r[i] = fn(x[i..i+H-1])
    return np.r_[r[1:], np.nan]


def schmitt(ratio, sgn):
    d = np.zeros(len(ratio), np.int8); prev = 0
    for i in range(len(ratio)):
        r, s = ratio[i], int(sgn[i])
        if not np.isfinite(r):
            d[i] = prev = 0; continue
        if prev == 0:
            cur = s if r > ENTER else 0
        elif s == prev:
            cur = 0 if r < EXIT else prev
        else:
            cur = s if r > ENTER else (0 if r < EXIT else prev)
        d[i] = prev = cur
    return d


def build(K):
    t = K.t.to_numpy(); h, l, c = (K[k].to_numpy(float) for k in 'hlc')
    r = np.r_[0, np.diff(np.log(c))]
    s24 = np.sqrt(pd.Series(r ** 2).rolling(1440).mean().to_numpy())
    s1h = np.sqrt(pd.Series(r ** 2).rolling(60).mean().to_numpy())
    ts = pd.to_datetime(t, unit='ms')
    hr = (ts.hour + ts.minute / 60).to_numpy()
    rows = []
    for name, (W, step) in SCALES.items():
        whi = pd.Series(h).rolling(W).max().to_numpy(); wlo = pd.Series(l).rolling(W).min().to_numpy()
        rng = whi - wlo
        move = c - np.r_[np.full(W, np.nan), c[:-W]]
        fmax, fmin = fut_ext(h, W, 'max'), fut_ext(l, W, 'min')
        hold = (fmax < c + 0.5 * rng) & (fmin > c - 0.5 * rng)
        m = ((t // 60000) % step == step - 1) & np.isfinite(rng) & np.isfinite(fmax) & np.isfinite(s24) & (rng > 0)
        idx = np.flatnonzero(m)
        ratio = np.abs(move[idx]) / rng[idx]; sgn = np.nan_to_num(np.sign(move[idx]))
        reg = schmitt(ratio, sgn)
        rows.append(pd.DataFrame({
            'scale': name, 'ts': ts[idx], 'hold': hold[idx].astype(int), 'reg': reg,
            'log_rng': np.log(rng[idx] / c[idx] * 1e4), 'log_ds': np.log(0.5 * rng[idx] / c[idx] / (s24[idx] * np.sqrt(W))),
            'log_vr': np.log(np.maximum(s1h[idx], 1e-6) / s24[idx]),
            'hs': np.sin(2 * np.pi * hr[idx] / 24), 'hc': np.cos(2 * np.pi * hr[idx] / 24),
            'wkd': (ts[idx].dayofweek >= 5).astype(int)}))
        log(f'[{name}] 결정 {len(idx):,} · 유지 {hold[idx].mean():.3f} · 횡보 비중 {(reg == 0).mean():.2f}')
    return pd.concat(rows, ignore_index=True)


FEATS = {'A range_bp': ['log_rng'], 'G 기하 d_σ': ['log_ds'],
         'F 기하+α': ['log_ds', 'log_vr', 'log_rng', 'hs', 'hc', 'wkd']}


def block_auc(y, p, day, B=500, seed=0):
    """일(또는 주) 블록 부트스트랩 AUC 표본."""
    codes, uniq = pd.factorize(day)
    order = np.argsort(codes, kind='stable'); starts = np.searchsorted(codes[order], np.arange(len(uniq)))
    ends = np.r_[starts[1:], len(codes)]
    rng = np.random.default_rng(seed); out = []
    for _ in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        ii = np.concatenate([order[starts[k]:ends[k]] for k in pick])
        out.append(roc_auc_score(y[ii], p[ii]) if len(np.unique(y[ii])) == 2 else np.nan)
    return np.array(out)


def evaluate(D):
    rep = {}
    for scale in SCALES:
        for sub in ('횡보', 'all'):
            g0 = D[(D.scale == scale) & ((D.reg == 0) if sub == '횡보' else True)]
            tr = g0[g0.ts < '2025-01-01']; key = f'{scale}·{sub}'; rep[key] = {}
            base = tr.hold.mean()
            models = {n: LogisticRegression(C=1.0, max_iter=1000).fit(tr[f], tr.hold) for n, f in FEATS.items()}
            for yr in (2025, 2026):
                te = g0[g0.ts.dt.year == yr]
                if te.hold.nunique() < 2:
                    continue
                y = te.hold.to_numpy()
                P = {n: models[n].predict_proba(te[FEATS[n]])[:, 1] for n in FEATS}
                day = te.ts.dt.floor('7D' if scale == '24h' else '1D').to_numpy()
                bG, bF = block_auc(y, P['G 기하 d_σ'], day), block_auc(y, P['F 기하+α'], day)
                dF = bF - bG
                pf = P['F 기하+α']
                cal = pd.DataFrame({'p': pf, 'y': y, 'd': pd.qcut(pf, 10, labels=False, duplicates='drop')}).groupby('d').agg(p=('p', 'mean'), y=('y', 'mean'))
                brier = lambda p: float(np.mean((p - y) ** 2))
                rec = dict(n=len(te), hold=float(y.mean()), train_base=float(base),
                           auc={n: float(roc_auc_score(y, P[n])) for n in FEATS},
                           auc_G_ci=(float(np.nanquantile(bG, .025)), float(np.nanquantile(bG, .975))),
                           dauc_F_G=(float(np.nanquantile(dF, .025)), float(np.nanquantile(dF, .975))),
                           bss=1 - brier(pf) / brier(np.full(len(y), base)),
                           cal_maxerr=float((cal.p - cal.y).abs().max()), cal=cal.round(3).values.tolist(),
                           q_lo_hi=(float(cal.y.iloc[0]), float(cal.y.iloc[-1])))
                rep[key][str(yr)] = rec
                log(f"{key:10s} {yr} n{rec['n']:6d} 유지 {rec['hold']:.3f}(학습 {base:.3f}) AUC A {rec['auc']['A range_bp']:.3f} "
                    f"G {rec['auc']['G 기하 d_σ']:.3f} F {rec['auc']['F 기하+α']:.3f} ΔF−G [{rec['dauc_F_G'][0]:+.3f},{rec['dauc_F_G'][1]:+.3f}] "
                    f"BSS {rec['bss']:+.3f} 보정 최대오차 {rec['cal_maxerr']*100:.1f}pp 하위·상위 10% 유지 {rec['q_lo_hi'][0]*100:.0f}→{rec['q_lo_hi'][1]*100:.0f}%")
            rep[key]['coef_F'] = dict(zip(FEATS['F 기하+α'], np.round(models['F 기하+α'].coef_[0], 3).tolist()))
    return rep


def selftest():
    rng = np.random.default_rng(0)
    x = rng.normal(size=50).cumsum()
    for H in (1, 3, 7):
        mx = fut_ext(x, H, 'max')
        for i in range(len(x) - H):
            assert abs(mx[i] - x[i + 1:i + 1 + H].max()) < 1e-12
        assert np.isnan(mx[-1])
    d = schmitt(np.array([.5, .3, .2, .5, .5]), np.array([1, 1, 1, -1, -1]))
    assert d.tolist() == [1, 1, 0, -1, -1], d
    log('selftest OK')


def fit_coef(D, path):
    """서빙 계수(dashboard/box_hold_coef_20261016.json) -- «all» 상태 · F 특징 · 2022~24 학습 + 학습 예측 확률 20·80분위(화면 «깨질/머물 쪽» 문턱).
    10-16 리뷰: 처음엔 일회성 코드로 맞췄고, «평소 3/4 밑» 문턱이 시간의 46~48% 켜져 경보 구실을 못 했다 → 분위 문턱."""
    out = {}
    for s in ('30m', '1h', '4h'):
        tr = D[(D.scale == s) & (D.ts < '2025-01-01')]
        m = LogisticRegression(C=1.0, max_iter=1000).fit(tr[FEATS['F 기하+α']], tr.hold)
        p = m.predict_proba(tr[FEATS['F 기하+α']])[:, 1]
        out[s] = {'intercept': float(m.intercept_[0]), 'coef': dict(zip(FEATS['F 기하+α'], map(float, m.coef_[0]))),
                  'train_base': float(tr.hold.mean()), 'q20': float(np.quantile(p, .2)), 'q80': float(np.quantile(p, .8)),
                  'train': ['2022-01-01', '2024-12-31'], 'n': int(len(tr))}
    Path(path).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    log(json.dumps({s: {k: round(v, 4) for k, v in o.items() if k in ('train_base', 'q20', 'q80')} for s, o in out.items()}))


if __name__ == '__main__':
    selftest()
    if '--selftest' in sys.argv:
        sys.exit()
    if '--fit-coef' in sys.argv:          # decisions.parquet(앞 실행 산출) → 서빙 계수
        fit_coef(pd.read_parquet(OUT / 'decisions.parquet'), WT / 'dashboard/box_hold_coef_20261016.json')
        sys.exit()
    OUT.mkdir(parents=True, exist_ok=True)
    D = build(load_1m())
    D.to_parquet(OUT / 'decisions.parquet')
    rep = evaluate(D)
    (OUT / 'report.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1))
