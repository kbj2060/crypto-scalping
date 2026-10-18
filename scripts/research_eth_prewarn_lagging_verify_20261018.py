"""전환 예고(경보기) «조용할 때 미리 켜지나» 재검증 (2026-10-18, 사용자 «이미 급등락 이후에 떠서 쓸 이유가 없다»).
원천 = 09-30 재생(서버 아티팩트 HGB 5시드 그대로, 마감 봉) replay.npz + eth5m_full.parquet. 평가 2025-09-01~2026-09-28(모델 학습 ≤2025-08).
Q1 켜지는 첫 봉의 직전 1h 상태(가격폭·|수익|·거래대금 z 의 후행 분위) · Q2 조용한 봉(직전 1h 폭 후행 분위 <50%)에서 예고의 앞 1h 큰 움직임 예측력
Q3 큰 움직임(앞 1h 최대이탈 상위 5%) 시작 봉에서 예고가 «조용할 때 이미» 켜져 있던 비율. 대조 = 같은 발동률의 무모델 규칙.
"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
B = "/home/kbj20/crypto-scalping/.claude/worktrees/options-sentiment-analysis-a1b431/tmp/bd_verify_20260930/"
z = np.load(B + "replay.npz"); d = pd.read_parquet(B + "eth5m_full.parquet").reset_index(drop=True)
ts = d.timestamp; c, h, l = d.close.to_numpy(float), d.high.to_numpy(float), d.low.to_numpy(float)
pw, sc, exc12, fire = z["pw"], z["sc"], z["exc12"], z["fire"]
rq = lambda x, w=2016: pd.Series(x).rolling(w, min_periods=500).rank(pct=True).to_numpy()   # 후행 분위(현재 봉 포함, 인과)
rng1h = (pd.Series(h).rolling(12).max() - pd.Series(l).rolling(12).min()).to_numpy() / c * 1e4
ret1h = np.abs(np.log(c / np.r_[np.full(12, np.nan), c[:-12]])) * 1e4
qv = d.quote_volume.astype(float); zq = ((qv - qv.rolling(288).mean()) / qv.rolling(288).std()).to_numpy()
p_rng, p_ret, p_zq = rq(rng1h), rq(ret1h), rq(zq)
ev = (ts >= "2025-09-01").to_numpy() & np.isfinite(exc12) & np.isfinite(sc) & np.isfinite(p_rng)
first = pw & ~pd.Series(pw).shift(1, fill_value=False).rolling(6, min_periods=1).max().astype(bool).to_numpy()   # 차트 «켜진 첫 봉»
act = pd.Series(pw).rolling(6, min_periods=1).max().astype(bool).to_numpy()                                     # 켜짐 유지 30분
big = exc12 >= np.nanquantile(exc12[ev], .95)
print(f"평가 봉 {ev.sum():,} · 예고 켜짐 {pw[ev].mean():.1%} · 첫 봉 {first[ev].sum():,}회({first[ev].sum() / (ev.sum() / 288):.1f}/일) · 큰 움직임 기준 앞1h 최대이탈 ≥{np.nanquantile(exc12[ev], .95):.0f}bp")

print("\nQ1 켜지는 첫 봉의 직전 1시간 상태(후행 분위, 0.5 = 평소)")
f = ev & first
for nm, p in (("직전1h 가격폭", p_rng), ("직전1h |수익|", p_ret), ("거래대금 z", p_zq)):
    print(f"  {nm:12s} 중앙 분위 {np.nanmedian(p[f]):.2f} · 상위20%(이미 움직임) {np.mean(p[f] >= .8):.0%} · 하위50%(조용) {np.mean(p[f] < .5):.0%}")
print(f"  직전 30분 안에 탐지(거래 급증) 이미 발동: {np.mean([fire[max(i - 5, 0):i + 1].any() for i in np.flatnonzero(f)]):.0%}")

print("\nQ2 조용한 봉(직전1h 폭 하위 50%)에서 앞 1h 큰 움직임")
q = ev & (p_rng < .5)
rate = act[q].mean()
print(f"  조용한 봉 {q.sum():,} · 그중 예고 켜짐 {rate:.1%} · 큰 움직임 기저 {big[q].mean():.2%}")
print(f"  예고 켜짐 {big[q & act].mean():.2%} vs 꺼짐 {big[q & ~act].mean():.2%} (배수 {big[q & act].mean() / big[q].mean():.2f}x)")
for nm, s in (("예고 점수", sc), ("거래대금 z", zq), ("직전1h 가격폭", rng1h)):
    g = q & np.isfinite(s)
    cut = np.nanquantile(s[g], 1 - rate)
    print(f"  같은 발동률 {rate:.1%} 상위 «{nm}»: 큰 움직임 {big[g & (s >= cut)].mean():.2%} · AUC {roc_auc_score(big[g], s[g]):.3f}")
g = ev & np.isfinite(zq)
print(f"  (전체 봉 AUC: 예고 {roc_auc_score(big[g], sc[g]):.3f} · 직전1h 폭 {roc_auc_score(big[g], rng1h[g]):.3f} · 거래대금 z {roc_auc_score(big[g], zq[g]):.3f})")

print("\nQ3 큰 움직임 «시작 봉»(앞 1h 상위5%, 1h 안 중복 제거)에서")
st = np.flatnonzero(ev & big & ~pd.Series(big & ev).shift(1, fill_value=False).rolling(12, min_periods=1).max().astype(bool).to_numpy())
qs = p_rng[st] < .5
print(f"  시작 {len(st)}회 · 그중 직전 1h 조용 {qs.mean():.0%}")
print(f"  예고 켜져 있음: 전체 시작 {act[st].mean():.0%} · 조용했던 시작 {act[st][qs].mean():.0%}  (아무 봉 기준 {act[ev].mean():.0%} · 조용한 봉 기준 {act[q].mean():.0%})")
