"""방향 표 융합 + 크기 관문 — 사전 고정 (2026-09-25). ETH 5분봉, 2023-01 ~ 2026-09-14(패널 OI 끝).
결정 = 5분봉 i 가 닫힌 순간(종가). 피쳐는 봉 ≤ i(OI 는 한 봉 더 지연). 라벨 = c[i+H]/c[i] (H = 6봉 30분 / 12봉 60분).
표(±1/0):
  wr   고래↔리테일 60분 z(자기 과거 30일) 둘 다 |z|≥0.5 로 반대 → sign(고래)
  wm   고래↔중형 같은 규칙 → sign(고래)
  oi   1시간 |이동| ≥ 과거 24h p75 이고 하락 & OI↑ → −1 / 하락 & OI↓ → +1 (상승 쪽 0 — 안 쟀다)
  al   12h 추세(SMA144±ATR 히스테리시스) == 30분 방향(카드 슈미트) → 30분 방향
  rj   거부 봉(마지막 봉 델타가 30분 방향 반대 & |델타| ≥ 0.5×창 최대) → −30분 방향
관문 G: 30분 창 고저폭의 과거 24h 분위 ≥ 2/3 (카드 크기 축 최선 단일피쳐 range_bp).
평가: 표 각각 · 합 S · S×G. TRAIN < 2025 ≤ TEST. 비겹침(H 간격) 결정만. 일 블록 CI."""
import glob, numpy as np, pandas as pd, sys
rng = np.random.default_rng(20260925)
SYM = sys.argv[1] if len(sys.argv) > 1 else 'ETHUSDT'
T = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'tmp/whale/tape/{SYM}/*.parquet'))]).sort_index()
T = T[~T.index.duplicated()].asfreq('1min')
b5 = pd.DataFrame({'h': T.h.resample('5min').max(), 'l': T.l.resample('5min').min(), 'c': T.c.resample('5min').last(),
                   **{g: T[f'row_{g}_sn'].fillna(0).resample('5min').sum() for g in ['ret', 'mid', 'whl']}})
b5['c'] = b5.c.ffill(); b5['h'] = b5.h.fillna(b5.c); b5['l'] = b5.l.fillna(b5.c)
P = pd.read_parquet(f'/home/kbj20/crypto-scalping/data/binance_vision/panel/{SYM}.parquet')
P['timestamp'] = pd.to_datetime(P.timestamp, utc=True); P = P.set_index('timestamp').sort_index(); P = P[~P.index.duplicated()]
b5['oi'] = P.sum_open_interest.reindex(b5.index).ffill().shift(1)   # 한 봉 더 지연(스탬프 규약)
b5 = b5[b5.index <= P.index.max()]
c, h, l = b5.c.values, b5.h.values, b5.l.values; n = len(c)
delta = (b5.ret + b5.mid + b5.whl).values
# ── 카드 30분 방향(슈미트 .45/.25) ──
mv30 = np.r_[np.full(6, np.nan), (c[6:] / c[:-6] - 1) * 1e4]
hi6 = pd.Series(h).rolling(6).max().values; lo6 = pd.Series(l).rolling(6).min().values
rg30 = (hi6 - lo6) / c * 1e4
dir30 = np.zeros(n, int); prev = 0
for i in range(n):
    if not np.isfinite(mv30[i]) or not rg30[i] > 0: dir30[i] = prev; continue
    ratio = abs(mv30[i]) / rg30[i]; sg = int(np.sign(mv30[i]))
    if prev == 0: d = sg if ratio > .45 else 0
    elif sg == prev: d = 0 if ratio < .25 else prev
    else: d = sg if ratio > .45 else (0 if ratio < .25 else prev)
    dir30[i] = prev = d
# ── 12h 추세 veto: SMA144 ± 1·ATR144 히스테리시스 ──
pc = np.r_[c[0], c[:-1]]; tr = np.maximum(h, pc) - np.minimum(l, pc)
atr = pd.Series(tr).rolling(144).mean().values; sma = pd.Series(c).rolling(144).mean().values
veto = pd.Series(np.where(c > sma + atr, 1.0, np.where(c < sma - atr, -1.0, np.nan))).ffill().fillna(0).values.astype(int)
# ── 크기별 60분 z (자기 과거 30일 = 8640봉, 12봉 합의 표준편차, 자기 창 제외) ──
Z = {}
for g in ['ret', 'mid', 'whl']:
    s = b5[g].rolling(12).sum()
    Z[g] = (s / s.rolling(8640, min_periods=2016).std().shift(12)).values
# ── 1시간 가격↔OI ──
mv60 = np.r_[np.full(12, np.nan), (c[12:] / c[:-12] - 1) * 1e4]
p75 = pd.Series(np.abs(mv60)).rolling(288, min_periods=144).quantile(0.75).shift(1).values
doi = b5.oi.values - np.r_[np.full(12, np.nan), b5.oi.values[:-12]]
# ── 거부 봉 ──
maxd = pd.Series(np.abs(delta)).rolling(6).max().values
rj_on = (dir30 != 0) & (np.sign(delta) == -dir30) & (np.abs(delta) >= 0.5 * maxd)
# ── 표 ──
V = {}
big = lambda z: np.abs(z) >= 0.5   # noqa: E731
V['wr'] = np.where(big(Z['whl']) & big(Z['ret']) & (np.sign(Z['whl']) != np.sign(Z['ret'])), np.sign(Z['whl']), 0)
V['wm'] = np.where(big(Z['whl']) & big(Z['mid']) & (np.sign(Z['whl']) != np.sign(Z['mid'])), np.sign(Z['whl']), 0)
bigmove = np.abs(mv60) >= p75
V['oi'] = np.where(bigmove & (mv60 < 0), np.where(doi > 0, -1, np.where(doi < 0, 1, 0)), 0)
V['al'] = np.where((veto != 0) & (veto == dir30), dir30, 0)
V['rj'] = np.where(rj_on, -dir30, 0)
for k in V: V[k] = np.nan_to_num(V[k]).astype(int)
S = sum(V.values())
rgq = pd.Series(rg30).rolling(288, min_periods=144).rank(pct=True).values   # 자기 포함 24h 분위(봉 i 까지 앎)
G = rgq >= 2 / 3
ts = b5.index; day = ts.floor('D').values; test = ts >= pd.Timestamp('2025-01-01', tz='UTC'); yr = ts.year.values
warm = np.arange(n) > 8640 + 300

def bci(v, d, B=1000):
    if len(v) < 30: return f"{np.mean(v) if len(v) else np.nan:+6.2f} n{len(v)}"
    u, di = np.unique(d, return_inverse=True); s = np.bincount(di, v, len(u)); k = np.bincount(di, None, len(u))
    W = rng.multinomial(len(u), np.ones(len(u)) / len(u), size=B); est = (W @ s) / (W @ k)
    return f"{np.mean(v):+6.2f} [{np.percentile(est, 2.5):+6.2f},{np.percentile(est, 97.5):+6.2f}] n{len(v)}"

for H in ([6, 12] if __name__ == '__main__' else []):
    idx = np.arange(0, n - H, H); idx = idx[warm[idx]]
    fwd = (c[idx + H] / c[idx] - 1) * 1e4
    tr_, te_ = ~test[idx], test[idx]
    print(f"\n==================== H = {H * 5}분  (결정 {len(idx)}개, |r| 평균 {np.abs(fwd).mean():.1f}bp · 관문 안 {np.abs(fwd[G[idx]]).mean():.1f}bp)")
    print("표 발동률·상관:", {k: round(float((V[k][idx] != 0).mean()), 3) for k in V})
    M = np.column_stack([V[k][idx] for k in V]); nz = (M != 0)
    cor = pd.DataFrame(M, columns=list(V)).replace(0, np.nan).corr(min_periods=50).round(2)
    print(cor.to_string())
    def show(lab, sel, dirv):
        v = dirv[sel] * fwd[sel]
        line = f"  {lab:28s}"
        for nm, m in [('TR', tr_), ('TE', te_)]:
            mm = sel & m
            line += f" | {nm} {bci(dirv[mm] * fwd[mm], day[idx][mm])} hit {np.mean(dirv[mm] * fwd[mm] > 0) if mm.sum() else np.nan:.3f}"
        yrs = pd.Series(v).groupby(yr[idx][sel]).mean().round(1).to_dict()
        print(line + f" | 연도 {yrs} | {sel.sum() / (len(idx) * H * 5 / 1440):.2f}/일")
    for k in V:
        s_ = V[k][idx]; show(f"표 {k}", s_ != 0, s_)
    Sx = S[idx]; Gx = G[idx]
    for kk in [1, 2, 3]:
        show(f"|S|>={kk}", np.abs(Sx) >= kk, np.sign(Sx))
        show(f"|S|>={kk} & 관문", (np.abs(Sx) >= kk) & Gx, np.sign(Sx))
    show("관문만(대조: 30분 방향 따라)", Gx & (dir30[idx] != 0), dir30[idx])
