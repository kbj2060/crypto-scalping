"""극점 탐지기 vs V자 급등락 — **배포 아티팩트 정면 비교** (2026-09-21).

저장소 규칙(homer §7): 라벨이 다르면 AUC 로 비교하지 않는다. 그래서
  ① **같은 봉**(공통 표본외 창) ② **같은 콜 건수**(빈도가 4.5배 다르다)
  ③ **중립 성과**(둘 중 누구의 라벨도 아닌 앞으로 12봉 경로)로 잰다.
피쳐/스코어링은 두 라이브 모듈의 자기 함수를 그대로 부른다.
"""
import json, sys, numpy as np, pandas as pd, joblib
ROOT = "/home/kbj20/crypto-scalping"
WT = ROOT + "/.claude/worktrees/extremum-detector-signal-direction-6fc663"
for _p in (ROOT, ROOT + "/scripts", WT, WT + "/scripts"): sys.path.insert(0, _p)
import live_eth_extreme_detector_20260909 as LD
import research_eth_v_rebound_hgb_vs_tabpfn_20260919 as X
import research_eth_v_rebound_close_anchor_retrain_20260912 as R12

W, OOS0, OOS1 = 12, pd.Timestamp("2026-04-01"), pd.Timestamp("2026-09-08 22:30")

# ---------- 1) 극점: 저장된 프레임 + 배포 아티팩트 ----------
A = pd.read_parquet(f"{ROOT}/tmp/eth_extreme_frame_allbars_20260916/extreme_frame.parquet")
em = json.load(open(f"{ROOT}/data/live/eth_extreme_detector_artifact/meta.json"))
assert em["features"] == LD.FEATS, "아티팩트 형상 불일치"
mods = joblib.load(f"{ROOT}/data/live/eth_extreme_detector_artifact/model.joblib")
A = A[(A._ts >= OOS0) & (A._ts <= OOS1)].reset_index(drop=True)
A["p"] = np.mean([m.predict_proba(A[LD.FEATS])[:, 1] for m in mods], axis=0)
A["gated"] = LD.gated_of(A._tq.to_numpy(), A._long.to_numpy())
# 봉별 = 확률 높은 쪽 (라이브 by_ts 규약)
E = A.sort_values("p").groupby("_ts").tail(1)[["_ts", "p", "_long", "_y", "gated", "atr_pct", "atr_pctile"]]
E.columns = ["ts", "p_ext", "ext_long", "_drop", "ext_gated", "atr_pct", "atr_pctile"]; E = E.drop(columns="_drop")
E["ts"] = E.ts.dt.tz_localize("UTC")   # klcache 는 tz-naive UTC, CSV 경로는 tz-aware

# ---------- 2) V자: 라이브 빌더 + 배포 HGB 아티팩트 ----------
kl = X.load_klines_full()
cand = X.build_every_bar(kl)
vm = json.load(open(f"{ROOT}/data/live/eth_v_rebound_hgb_artifact/meta.json"))
vmods = joblib.load(f"{ROOT}/data/live/eth_v_rebound_hgb_artifact/model.joblib")
assert vm["features"] == list(X.LIVE.FEATURES), "V자 아티팩트 형상 불일치"
cand["p"] = np.mean([m.predict_proba(cand[X.LIVE.FEATURES])[:, 1] for m in vmods], axis=0)
V = cand.sort_values("p").groupby("timestamp").tail(1)[["timestamp", "p", "is_downside"]]
V.columns = ["ts", "p_vr", "vr_down"]

# ---------- 3) 라벨 + 중립 경로 ----------
klL = R12.load_klines(); lab = R12.build_labels(klL)
h, l, c, n = klL.high.to_numpy(), klL.low.to_numpy(), klL.close.to_numpy(), len(klL)
def fwd(x, k, how):
    r = pd.Series(x[::-1]).rolling(k, min_periods=k)
    v = (r.max() if how == "max" else r.min()).to_numpy()[::-1]
    o = np.full(n, np.nan); o[:n-k] = v[1:n-k+1]; return o
fmax_h, fmin_l = fwd(h, W, "max"), fwd(l, W, "min")
end12 = np.full(n, np.nan); end12[:n-W] = c[W:]
P = pd.DataFrame({"ts": klL.timestamp,
    "vr_y_down": lab.y_extreme_down.to_numpy(), "vr_y_up": lab.y_extreme_up.to_numpy(),
    "ext_y_down": np.where(np.isfinite(fmin_l), (fmin_l >= l), np.nan),   # 저점 안 깨짐
    "ext_y_up":   np.where(np.isfinite(fmax_h), (fmax_h <= h), np.nan),   # 고점 안 넘김
    # 중립 경로: 롱 기준 bp. 숏이면 부호를 뒤집는다
    "up_mfe": (fmax_h - c) / c * 1e4, "dn_mfe": (c - fmin_l) / c * 1e4,
    "ret12": (end12 - c) / c * 1e4})

d = E.merge(V, on="ts").merge(P, on="ts").dropna().reset_index(drop=True)
print(f"공통 표본외 {d.ts.min()} ~ {d.ts.max()} · {len(d):,}봉 "
      f"({(d.ts.max()-d.ts.min()).days}일)\n")

def outcomes(sub, long_col):
    lg = sub[long_col].to_numpy().astype(bool)
    mfe = np.where(lg, sub.up_mfe, sub.dn_mfe)          # 콜 방향의 최대 유리 이탈
    mae = np.where(lg, -sub.dn_mfe, -sub.up_mfe)        # 콜 방향의 최대 불리 이탈
    ret = np.where(lg, sub.ret12, -sub.ret12)           # 콜 방향 12봉 수익
    vy  = np.where(lg, sub.vr_y_down, sub.vr_y_up)
    ey  = np.where(lg, sub.ext_y_down, sub.ext_y_up)
    return dict(n=len(sub), ext=ey.mean(), vr=vy.mean(),
                mfe=mfe.mean(), mae=mae.mean(), ret=ret.mean(),
                se=ret.std(ddof=1)/np.sqrt(len(sub)))

days = (d.ts.max() - d.ts.min()).total_seconds()/86400
print(f"{'팔':<26}{'n':>6}{'/일':>7}{'극점라벨':>9}{'V자라벨':>9}"
      f"{'MFE':>8}{'MAE':>8}{'12봉수익':>10}{'±SE':>7}")
print("-"*92)
rows = {}
for rate in (1.46, 2.93, 13.25):
    N = int(round(rate*days))
    ext = d[~d.ext_gated].nlargest(N, "p_ext")                  # 게이트는 배포 규칙의 일부
    vr  = d.nlargest(N, "p_vr")
    for nm, sub, col in ((f"극점 상위{rate}/일", ext, "ext_long"),
                         (f"V자  상위{rate}/일", vr, "vr_down")):
        s = sub.copy()
        if col == "vr_down": s["_lg"] = s.vr_down.astype(bool); col = "_lg"
        o = outcomes(s, col); rows[nm] = (o, set(sub.ts))
        print(f"{nm:<26}{o['n']:>6}{o['n']/days:>7.2f}{o['ext']:>9.1%}{o['vr']:>9.1%}"
              f"{o['mfe']:>8.1f}{o['mae']:>8.1f}{o['ret']:>10.2f}{o['se']:>7.2f}")
    a, b = rows[f"극점 상위{rate}/일"][1], rows[f"V자  상위{rate}/일"][1]
    print(f"{'  └ 겹침(AND)':<26}{len(a&b):>6}  자카드 {len(a&b)/len(a|b):.1%}", end="")
    both = d[d.ts.isin(a & b)].copy()
    if len(both) > 20:
        both["_lg"] = both.ext_long.astype(bool)
        ob = outcomes(both, "_lg")
        print(f"{ob['ext']:>17.1%}{ob['vr']:>9.1%}{ob['mfe']:>8.1f}{ob['mae']:>8.1f}"
              f"{ob['ret']:>10.2f}{ob['se']:>7.2f}")
    else:
        print()
    print()
rng = np.random.default_rng(20260921)
def dayboot(sub, col, B=2000, seed=20260921):
    g = sub.assign(_d=sub.ts.dt.date).groupby("_d")
    per = np.array([outcomes(x, col)["ret"] for _, x in g]); w = np.array([len(x) for _, x in g])
    r = np.random.default_rng(seed); m = []
    for _ in range(B):
        i = r.integers(0, len(per), len(per)); m.append(np.average(per[i], weights=w[i]))
    return np.percentile(m, [2.5, 97.5])
print("\n[일군집 부트스트랩 95% CI — 12봉 수익 bp]  (겹치는 창·군집 발동이라 위 ±SE 는 과소추정)")
for rate in (1.46, 2.93, 13.25):
    N = int(round(rate*days))
    for nm, sub, col in ((f"극점 {rate}/일", d[~d.ext_gated].nlargest(N, "p_ext"), "ext_long"),
                         (f"V자  {rate}/일", d.nlargest(N, "p_vr"), "vr_down")):
        sub = sub.copy(); sub["_lg"] = sub[col].astype(bool)
        lo, hi = dayboot(sub, "_lg")
        print(f"  {nm:<14} [{lo:+7.2f}, {hi:+7.2f}]  {'0 배제' if lo*hi>0 else '0 포함'}")
rng = np.random.default_rng(20260921)
o = outcomes(d.assign(_lg=rng.random(len(d)) < 0.5), "_lg")
print(f"{'무작위(전 봉·무작위 측면)':<26}{o['n']:>6}{'':>7}{o['ext']:>9.1%}{o['vr']:>9.1%}"
      f"{o['mfe']:>8.1f}{o['mae']:>8.1f}{o['ret']:>10.2f}{o['se']:>7.2f}")


# ═══ 변동성 통제 ═══════════════════════════════════════════════════════════
d.to_parquet(f"{ROOT}/tmp/h2h_merged_20260921.parquet")
print("\n\n═══ 이 둘은 결국 «변동성»인가 ═══\n")
rng2 = np.random.default_rng(7)
d["vdec"] = pd.qcut(d.atr_pctile, 10, labels=False, duplicates="drop")

print("[A] 모델 없이 **변동성만**으로 같은 건수를 고르면 (측면은 무작위)")
print(f"{'팔':<26}{'n':>6}{'극점라벨':>9}{'V자라벨':>9}{'MFE':>8}{'MAE':>8}{'MFE/MAE':>9}")
for rate in (1.46, 2.93):
    N = int(round(rate*days))
    sub = d.nlargest(N, "atr_pctile").copy(); sub["_lg"] = rng2.random(len(sub)) < 0.5
    o = outcomes(sub, "_lg")
    print(f"{'ATR분위 상위'+str(rate)+'/일':<26}{o['n']:>6}{o['ext']:>9.1%}{o['vr']:>9.1%}"
          f"{o['mfe']:>8.1f}{o['mae']:>8.1f}{o['mfe']/-o['mae']:>9.2f}")
sub = d.copy(); sub["_lg"] = rng2.random(len(sub)) < 0.5
o = outcomes(sub, "_lg")
print(f"{'무작위 전체':<26}{o['n']:>6}{o['ext']:>9.1%}{o['vr']:>9.1%}{o['mfe']:>8.1f}"
      f"{o['mae']:>8.1f}{o['mfe']/-o['mae']:>9.2f}")

print("\n[B] **ATR 십분위 안에서** 모델이 고른 봉 vs 그 십분위 기저 (2.93/일)")
N = int(round(2.93*days))
arms = {"극점": (d[~d.ext_gated].nlargest(N, "p_ext"), "ext_long"),
        "V자": (d.nlargest(N, "p_vr"), "vr_down")}
print(f"{'십분위':>6}{'ATR분위':>9}{'기저ext':>9}{'극점콜':>8}{'n':>5}{'리프트':>7}"
      f"{'  |':>3}{'기저vr':>8}{'V자콜':>8}{'n':>5}{'리프트':>7}")
for q in range(10):
    dq = d[d.vdec == q]
    if len(dq) < 50: continue
    bl = rng2.random(len(dq)) < 0.5
    be = np.where(bl, dq.ext_y_down, dq.ext_y_up).mean()
    bv = np.where(bl, dq.vr_y_down, dq.vr_y_up).mean()
    row = f"{q:>6}{dq.atr_pctile.mean():>9.2f}{be:>9.1%}"
    for nm, (sub, col) in arms.items():
        sq = sub[sub.ts.isin(dq.ts)]
        if len(sq) < 5:
            row += f"{'-':>8}{len(sq):>5}{'-':>7}" + ("   |" if nm == "극점" else "")
            continue
        lg = sq[col].to_numpy().astype(bool)
        y = np.where(lg, sq.ext_y_down, sq.ext_y_up).mean() if nm == "극점" else \
            np.where(lg, sq.vr_y_down, sq.vr_y_up).mean()
        b = be if nm == "극점" else bv
        row += f"{y:>8.1%}{len(sq):>5}{y/b:>7.2f}"
        if nm == "극점": row += f"{'  |':>3}{bv:>8.1%}"
    print(row)
