#!/usr/bin/env python3
"""청산 피드 방향 피쳐 — **사전등록 결정 게이트 실행** (2026-09-16).

판정: B1 통과 · B2 실패 => §6 규칙대로 **청산 방향 피쳐 축 종료(registry)**.
결과 기록: 위 설계 문서의 §15.

설계 원문: docs/experiments/eth_candidate_liquidation_feed_features_cheap_gate_20260817.md
  §4 피쳐 10종 · §5 벤치마크 4종 · §6 게이트 P/B1/B2/C-lite · §12 개정(primary 지평 h=3,
  B2-상호작용 변형 추가). 실행 예정일이 "≈2026-09-15 이후"였고 데이터가 도착했다.

🔴인과 규약 정정: `tail_risk_interceptor.py:516` 이 `bucket_ts=(now-1분).floor(분)` 로 쓰므로
   ts=τ 행은 구간 [τ,τ+1분) 이고 τ+1분에 확정된다. 따라서 5분봉 종가 T 의 피쳐는 **ts ≤ T-1분**
   까지만 쓴다. 문서의 "as-of" 를 문자 그대로 ts ≤ T 로 구현하면 미래 1분을 본다(정보 손실은 0).
"""
import sys, json, numpy as np, pandas as pd, duckdb
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
ROOT = Path(__file__).resolve().parent.parent

SEED = 20260817           # §6 B2 사전등록 시드
B_BOOT = 2000
# 🔴split 재등록 (§10 의 지시: "유효 epoch 8주 도달 시점에 §4~§7 구조 그대로 split만 재등록해 실행").
# §3 의 원 split(DEV 05-04~06-30)은 §9 의 forceOrder 엔드포인트 결함(05-03~07-18 전량 0, 커밋
# 7fbfd30 에서 수리) 이전 구간이라 실행 불가다 -- 실제로 돌려보니 DEV 전 피쳐 0행이었다.
SPLITS = {"FIT":    ("2026-07-20", "2026-08-11"),   # 08-17 탐색 스캔이 본 구간 = 적합 전용
          "DECIDE": ("2026-08-12", "2026-09-16")}   # 어떤 분석에도 안 쓴 구간 = 여기서만 판정
FIT, DEC = "FIT", "DECIDE"
HS = [1, 3, 12]           # 5분봉 개수
PRIMARY_H = 3             # §12 item2
FEATS = ["liq_long_12","liq_short_12","liq_net_z_12","liq_total_z_48","liq_event_rate_z_48",
         "large_long_recent","large_short_recent","mins_since_large","liq_asym_48","aftershock_prob"]
BENCH = ["lag1_ret","ret_12","abs_ret_12","taker_imbalance"]

def _pres(s, w):   # 창 존재율 ≥80% 아니면 NaN (갭 관통 금지)
    return s.notna().rolling(w, min_periods=1).mean() >= 0.80

# ---------- 1분 원천 ----------
c = duckdb.connect(str(ROOT/"data/live/tail_risk.duckdb"), read_only=True)
try: tr = c.execute("select ts, long_usd_1m, short_usd_1m, liq_event_count_1m, shadow_aftershock_prob from tail_risk_1m").df()
finally: c.close()
tr["ts"] = pd.to_datetime(tr.ts, utc=True).astype("datetime64[ns, UTC]")
tr = tr.sort_values("ts").drop_duplicates("ts", keep="last").set_index("ts")
tr = tr.reindex(pd.date_range(tr.index.min(), tr.index.max(), freq="1min", tz="UTC"))  # 완전 분 격자
L, S = tr.long_usd_1m, tr.short_usd_1m
C, A = tr.liq_event_count_1m, tr.shadow_aftershock_prob
tot = L.fillna(0) + S.fillna(0)

F = pd.DataFrame(index=tr.index)
F["liq_long_12"]  = np.log1p(L.rolling(12).sum()).where(_pres(L,12))
F["liq_short_12"] = np.log1p(S.rolling(12).sum()).where(_pres(S,12))
m2880 = tot.rolling(2880).mean()
F["liq_net_z_12"] = ((L.rolling(12).sum() - S.rolling(12).sum()) / (m2880 + 0.01*m2880)).where(_pres(L,12) & _pres(L,2880))
x48 = tot.rolling(48).sum()
F["liq_total_z_48"] = ((x48 - x48.rolling(2880).mean()) / x48.rolling(2880).std()).where(_pres(L,2880))
# 🔴갭 허용(2026-09-16 수정): C 를 fillna(0) 없이 두면 rolling(2880) 이 기본 min_periods=2880 이라
# 창 안의 NaN 하나로 통째 죽는다 -- DECIDE 에서 c48 NaN 9.9% 인데 그 std 는 NaN 91.4% 였다.
# 존재율 방어는 _pres(C,2880)>=80% 가 이미 한다(liq_net_z_12 가 tot=fillna(0) 로 받는 것과 같은 처리).
c48 = C.fillna(0).rolling(48).sum()
F["liq_event_rate_z_48"] = ((c48 - c48.rolling(2880).mean()) / c48.rolling(2880).std()).where(_pres(C,2880))
p99L = L.rolling(2880).quantile(0.99); p99S = S.rolling(2880).quantile(0.99)
bigL = (L > p99L).fillna(False); bigS = (S > p99S).fillna(False)
F["large_long_recent"]  = bigL.rolling(12).max().where(_pres(L,2880))
F["large_short_recent"] = bigS.rolling(12).max().where(_pres(S,2880))
big = (bigL | bigS).to_numpy()
since = np.full(len(big), np.nan); last = -1
for i, b in enumerate(big):
    if b: last = i
    if last >= 0: since[i] = min(i - last, 288)
F["mins_since_large"] = np.log1p(since)
den = L.rolling(48).sum() + S.rolling(48).sum()
F["liq_asym_48"] = ((L.rolling(48).sum() - S.rolling(48).sum()) / den.replace(0, np.nan)).where(_pres(L,48))
F["aftershock_prob"] = A

# ---------- 5분봉 + 벤치마크 + 라벨 ----------
bv = pd.read_parquet(ROOT/"data/binance_vision/panel/ETHUSDT.parquet")
bv["timestamp"] = pd.to_datetime(bv["timestamp"], utc=True).astype("datetime64[ns, UTC]")
bv = bv[(bv.timestamp >= "2026-05-01")].drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
lc = np.log(bv.close.to_numpy(float))
P = pd.DataFrame({"timestamp": bv.timestamp})
P["lag1_ret"] = np.concatenate([[np.nan], np.diff(lc)])
P["ret_12"] = pd.Series(lc).diff(12).to_numpy()
P["abs_ret_12"] = P.ret_12.abs()
P["taker_imbalance"] = ((2*bv.taker_buy_base - bv.volume)/bv.volume.replace(0,np.nan)).to_numpy()
for h in HS:
    f = np.full(len(lc), np.nan); f[:-h] = lc[h:] - lc[:-h]
    P[f"fwd{h}"] = f
P["close"] = bv.close.to_numpy(float)

# 인과 as-of: ts ≤ 봉종가 − 1분
P["asof"] = P.timestamp - pd.Timedelta(minutes=1)
Fr = F.reset_index().rename(columns={"index":"ts"})
P = pd.merge_asof(P.sort_values("asof"), Fr.sort_values("ts"), left_on="asof", right_on="ts",
                  direction="backward", tolerance=pd.Timedelta(minutes=10)).drop(columns=["ts"])
P = P.sort_values("timestamp").reset_index(drop=True)
P["day"] = P.timestamp.dt.floor("D")

def seg(name):
    a, b = SPLITS[name]
    return P[(P.timestamp >= a) & (P.timestamp <= pd.Timestamp(b, tz="UTC") + pd.Timedelta(days=1))]

print("=== 표본 ===")
for k in SPLITS:
    s = seg(k); ok = s[FEATS].notna().all(axis=1).sum()
    print(f"  {k:22s} {SPLITS[k][0]}..{SPLITS[k][1]}  봉={len(s):6d}  전피쳐유효={ok:6d}  독립일={s.day.nunique():4d}")

# ---------- 게이트 P ----------
c48d = c48.rolling(2880).std()
print(f"\n[진단] liq_event_rate_z_48 결측 원인 -- 48분 이벤트합의 2880분 std 가 0 인 비율="
      f"{(c48d.fillna(0)==0).mean()*100:.1f}%  · std NaN 비율={c48d.isna().mean()*100:.1f}%"
      f"  · liq_event_count_1m 분격자 존재율={C.notna().mean()*100:.1f}%")
print("\n=== 게이트 P (오염: |spearman(feature, close)| < 0.5) ===")
bad = []
for f in FEATS:
    m = P[f].notna()
    r = spearmanr(P.loc[m, f], P.loc[m, "close"]).statistic if m.sum() > 100 else np.nan
    if abs(r) >= 0.5: bad.append((f, r))
    print(f"  {f:22s} rho_close={r:+.3f}{'   ⛔' if abs(r)>=0.5 else ''}")
print(f"  => {'통과' if not bad else '위반 '+str(bad)}")

# ---------- IC 매트릭스 ----------
print("\n=== IC 매트릭스 (spearman, 피쳐 × 지평 × 구간) ===")
rows = []
for f in FEATS:
    r = {"feat": f}
    for k in SPLITS:
        s = seg(k)
        for h in HS:
            m = s[f].notna() & s[f"fwd{h}"].notna()
            r[f"{k}_h{h}"] = spearmanr(s.loc[m,f], s.loc[m,f"fwd{h}"]).statistic if m.sum()>200 else np.nan
    rows.append(r)
IC = pd.DataFrame(rows)
print(IC.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

# ---------- B1 ----------
print(f"\n=== B1 (존재) — primary liq_net_z_12, h={PRIMARY_H} (§12 item2 로 h=1→3 개정) ===")
d_ic = IC.loc[IC.feat=="liq_net_z_12", f"FIT_h{PRIMARY_H}"].iloc[0]
c_ic = IC.loc[IC.feat=="liq_net_z_12", f"DECIDE_h{PRIMARY_H}"].iloc[0]
sign_ok = np.sign(d_ic) == np.sign(c_ic); mag_ok = abs(c_ic) >= 0.025
print(f"  DEV IC={d_ic:+.4f}  CONFIRM IC={c_ic:+.4f}  부호일치={sign_ok}  |IC_CONFIRM|>=0.025={mag_ok}")
B1 = bool(sign_ok and mag_ok); print(f"  => B1 {'통과' if B1 else '실패'}")

# ---------- B2 ----------
# 🔴커버리지 진단으로 정정: liq_event_rate_z_48 은 DECIDE 에서 유효 8.8%/4독립일뿐이라
# 10종 동시 요구가 평가창을 35일→4일로 무너뜨린다(첫 실행의 "B2 실패"는 검정 불능이었다).
# 사전등록 §4 의 10종 중 **커버리지가 성립하는 9종**으로 B2 를 돌리고, 제외 사실을 명시한다.
DROP_COV = []          # 버그 수정 후 10종 전부 커버리지 성립 -- 제외 없음
FEATS_B2 = [f for f in FEATS if f not in DROP_COV]
ALL = BENCH + FEATS_B2 + ["ix_long", "ix_short"]

def fit_rho(cols, h, trn, te):
    """🔴학습·평가 표본을 세 모델이 **공유**한다. 모델마다 dropna 를 따로 하면 벤치(4열)가
    풀(14열)보다 많은 행을 얻어 Δrho 가 다른 표본끼리의 비교가 된다(첫 실행의 실제 버그)."""
    sc = StandardScaler().fit(trn[cols])
    mdl = Ridge(alpha=1.0).fit(sc.transform(trn[cols]), trn[f"fwd{h}"])
    p = mdl.predict(sc.transform(te[cols]))
    return spearmanr(p, te[f"fwd{h}"]).statistic, p

def boot_drho(pf, pb, te, h, rng):
    days = te.day.to_numpy(); uq = np.unique(days); y = te[f"fwd{h}"].to_numpy(); out = []
    pos = {d: np.flatnonzero(days == d) for d in uq}
    for _ in range(B_BOOT):
        idx = np.concatenate([pos[d] for d in rng.choice(uq, len(uq), replace=True)])
        out.append(spearmanr(pf[idx], y[idx]).statistic - spearmanr(pb[idx], y[idx]).statistic)
    return np.percentile(out, [2.5, 97.5])

print(f"\n=== B2 (증분) — ridge(bench) vs ridge(bench+청산{len(FEATS_B2)}), 학습 FIT / 평가 DECIDE, h={PRIMARY_H} ===")
print(f"  커버리지로 제외: {DROP_COV}")
dev, con = seg(FIT).copy(), seg(DEC).copy()
for d in (dev, con):
    d["ix_long"]  = d.lag1_ret * d.large_long_recent
    d["ix_short"] = d.lag1_ret * d.large_short_recent
trn = dev.dropna(subset=ALL + [f"fwd{PRIMARY_H}"])
te  = con.dropna(subset=ALL + [f"fwd{PRIMARY_H}"])
print(f"  공유 표본: 학습 {len(trn)}행({trn.day.nunique()}일) / 평가 {len(te)}행({te.day.nunique()}일)")
rb, pb = fit_rho(BENCH, PRIMARY_H, trn, te)
rf, pf = fit_rho(BENCH+FEATS_B2, PRIMARY_H, trn, te)
rng = np.random.default_rng(SEED)
lo, hi = boot_drho(pf, pb, te, PRIMARY_H, rng)
print(f"  rho_bench={rb:+.4f}  rho_full={rf:+.4f}  Δrho={rf-rb:+.4f}  95%CI=[{lo:+.4f},{hi:+.4f}]  (일블록 {B_BOOT}회, seed {SEED})")
B2 = bool(lo > 0) if np.isfinite(lo) else False; print(f"  => B2 {'통과' if B2 else '실패'}")

# ---------- B2-상호작용 (§12 item1) ----------
ri, pi = fit_rho(ALL, PRIMARY_H, trn, te)
print(f"\n=== B2-상호작용 (§12 item1) ===")
print(f"  bench={rb:+.4f}  +liq={rf:+.4f}  +liq+interaction={ri:+.4f}   Δ(inter vs bench)={ri-rb:+.4f}")

# ---------- C-lite (관찰, 비결정) ----------
print("\n=== 경제성 (결정) — DECIDE 구간, 날짜블록 CI ===")
sys.path.insert(0, str(ROOT/"scripts"))
from research_direction_event_expansion_20260915 import dateblock_ci
MAKER = 5.52
yq = te[f"fwd{PRIMARY_H}"].to_numpy()*1e4
dq = te.day.to_numpy()
qq = pd.Series(pf).rank(pct=True).to_numpy()
hi_m, lo_m = qq >= 0.9, qq <= 0.1
e_model = np.concatenate([yq[hi_m], -yq[lo_m]]) - MAKER
d_model = np.concatenate([dq[hi_m], dq[lo_m]])
e_long  = np.concatenate([yq[hi_m],  yq[lo_m]]) - MAKER      # 같은 봉에서 무조건 롱
l1, h1_ = dateblock_ci(e_model, d_model, np.random.default_rng(SEED), B=2000)
l2, h2_ = dateblock_ci(e_long,  d_model, np.random.default_rng(SEED), B=2000)
inc = e_model - e_long
l3, h3_ = dateblock_ci(inc, d_model, np.random.default_rng(SEED), B=2000)
print(f"  거래 {len(e_model)}건 · 독립일 {len(np.unique(d_model))}")
print(f"  모델   건당net {e_model.mean():+7.2f}bp  (gross {e_model.mean()+MAKER:+7.2f})  CI[{l1:+.2f},{h1_:+.2f}]")
print(f"  무조건롱 건당net {e_long.mean():+7.2f}bp  (gross {e_long.mean()+MAKER:+7.2f})  CI[{l2:+.2f},{h2_:+.2f}]")
print(f"  모델−롱  증분  {inc.mean():+7.2f}bp  CI[{l3:+.2f},{h3_:+.2f}]   <= 이 저장소의 진짜 관문")

print("\n=== C-lite (관찰·비결정) ===")
q = pd.Series(pf).rank(pct=True).to_numpy()
y = te[f"fwd{PRIMARY_H}"].to_numpy()*1e4
gross = np.concatenate([y[q>=0.9], -y[q<=0.1]]).mean()
print(f"  (a) composite 상/하위 decile 롱숏 gross = {gross:+.2f}bp/거래   vs 왕복비용 11bp(호메로스)·5.52bp(메이커)")
ev = con.dropna(subset=["large_long_recent","large_short_recent",f"fwd{PRIMARY_H}"])
fire = (ev.large_long_recent > 0) | (ev.large_short_recent > 0)
print(f"  (b) 이벤트-lift  발동 {int(fire.sum())} / 비발동 {int((~fire).sum())}")
print(f"      방향  fwd 평균  발동 {ev.loc[fire,f'fwd{PRIMARY_H}'].mean()*1e4:+.2f}bp  vs 비발동 {ev.loc[~fire,f'fwd{PRIMARY_H}'].mean()*1e4:+.2f}bp")
vf, vn = ev.loc[fire,f'fwd{PRIMARY_H}'].abs().mean()*1e4, ev.loc[~fire,f'fwd{PRIMARY_H}'].abs().mean()*1e4
print(f"      변동  |fwd| 평균 발동 {vf:.2f}bp vs 비발동 {vn:.2f}bp   lift={vf/vn:.2f}x")

print(f"\n{'='*70}\n판정: B1 {'통과' if B1 else '실패'} · B2 {'통과' if B2 else '실패'}  =>  "
      f"{'Phase 2 진행' if (B1 and B2) else '⛔청산 방향 피쳐 축 종료(registry)'}\n{'='*70}")
json.dump({"B1":B1,"B2":B2,"dev_ic":float(d_ic),"confirm_ic":float(c_ic),
           "rho_bench":float(rb),"rho_full":float(rf),"drho_ci":[float(lo),float(hi)],
           "rho_interaction":float(ri),"clite_gross_bp":float(gross),"vol_lift":float(vf/vn)},
          open("/tmp/liqgate_result.json","w"), indent=1)
