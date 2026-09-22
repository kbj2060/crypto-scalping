"""레짐 «안»에서 답이 갈리는 피쳐 찾기 — 4.7년 5분 패널 (2026-09-22).

왜: 09-22 원장 감사 결과 엔진은 레짐 분류기일 뿐이고 레짐 안에서는 1순위가 사실상 상수였다
   (추세 A 406/420 · 횡보 B 292/332). 보정은 맞는데 변별이 0이다. 원장은 독립 30분창이 48개뿐이라
   여기서 피쳐를 고르면 그 자체가 과적합이다 -- 긴 패널에서 찾고 원장은 부호 확인에만 쓴다.

무엇을: 타깃은 **Y_sym** = mid ± 0.5×창고저폭 선착(등거리 배리어). 거리 편향이 없는 유일한 축이고
   라이브 `situation.resolve_sym` 과 같은 정의다. 레짐(dir=-1/0/+1)을 **조건으로 고정**하고
   그 안에서 P(up) 을 가르는 피쳐를 찾는다.

경계 계약: 피쳐는 봉 i 까지(행 τ 는 봉 τ 자신의 종가를 담는다), 배리어 탐색은 봉 i+1 부터. 같은 봉 공유 없음.

🔴함정 방어:
  - **가격 수준 인공물**(이 저장소 세 번째 재발): open/high/low/close/close_btc 원시 수준을 뺀다.
  - **다중검정**: 회전 귀무(하루 단위 순환이동)로 피쳐별 p 와 **전체 최대통계**를 같이 낸다.
    「최대통계보다 작으면 그 피쳐는 152개를 뒤진 결과일 뿐」이다.
  - **겹치는 창**: 5분 결정 × 30분 지평이라 6배 겹친다. CI 는 전부 **일 블록**으로 낸다.
  - **선택은 TRAIN 에서만**. VAL 은 부호 확인, OOS 는 마지막에 한 번 읽는다.
"""
import sys
import numpy as np
import pyarrow.parquet as pq
from scipy.stats import rankdata

PANEL = "data/frozen/zeus_20260918/features_with_regime_2022_2026_realfunding.parquet"
WINDOW, HOR = 6, 6                      # 30분 창 · 30분 지평 (5분봉)
ENTER, EXIT, SYM_K = 0.45, 0.25, 0.5    # situation.py 와 같은 상수
BARS_PER_DAY = 288
DROP = {"timestamp", "open", "high", "low", "close", "close_btc"}   # 가격 수준 인공물
SPLIT_VAL, SPLIT_OOS = "2025-09-01", "2026-04-01"
ROTS = 400
SEED = 20260922


def load():
    t = pq.read_table(PANEL)
    ts = t.column("timestamp").to_numpy().astype("datetime64[s]").astype(np.int64)
    cols = [n for n in t.schema.names if n not in DROP]
    X = np.column_stack([t.column(n).to_numpy().astype(np.float64) for n in cols])
    ohlc = {k: t.column(k).to_numpy().astype(np.float64) for k in ("high", "low", "close")}
    return ts, cols, X, ohlc


def regime(hi, lo, cl):
    """situation.classify 의 이동·슈미트 트리거를 그대로. 반환 d[i] (i 는 결정봉 index)."""
    n = len(cl)
    w_hi = np.full(n, np.nan); w_lo = np.full(n, np.nan)
    for k in range(WINDOW):
        w_hi[WINDOW:] = np.fmax(w_hi[WINDOW:], hi[WINDOW - k: n - k])
        w_lo[WINDOW:] = np.fmin(w_lo[WINDOW:], lo[WINDOW - k: n - k])
    rng = w_hi - w_lo
    move = np.full(n, np.nan)
    move[WINDOW:] = (cl[WINDOW:] - cl[:-WINDOW]) / cl[:-WINDOW] * 1e4
    rng_bp = rng / cl * 1e4
    ratio = np.where(rng_bp > 0, np.abs(move) / rng_bp, 0.0)
    sgn = np.sign(move)
    d = np.zeros(n, dtype=np.int8)
    prev = 0
    for i in range(WINDOW, n):                    # 슈미트는 상태를 이어받으므로 순차
        r, s = ratio[i], int(sgn[i]) if sgn[i] == sgn[i] else 0
        if prev == 0:
            cur = s if r > ENTER else 0
        elif s == prev:
            cur = 0 if r < EXIT else prev
        else:
            cur = s if r > ENTER else (0 if r < EXIT else prev)
        d[i] = prev = cur
    return d, rng, np.isfinite(rng) & np.isfinite(move)


def y_sym(hi, lo, cl, rng):
    """봉 i+1..i+HOR 에서 mid±0.5×창폭 선착. 1=up 0=down -1=none -2=amb(같은 봉 양쪽)."""
    n = len(cl)
    up_px = cl + SYM_K * rng
    dn_px = cl - SYM_K * rng
    y = np.full(n, -1, dtype=np.int8)
    done = np.zeros(n, dtype=bool)
    idx = np.arange(n)
    for j in range(1, HOR + 1):
        k = idx + j
        ok = (k < n) & ~done
        kk = k[ok]
        u = hi[kk] >= up_px[ok]
        dn = lo[kk] <= dn_px[ok]
        pos = idx[ok]
        y[pos[u & dn]] = -2
        y[pos[u & ~dn]] = 1
        y[pos[dn & ~u]] = 0
        done[pos[u | dn]] = True
    y[idx + HOR >= n] = -1
    return y


def auc_mat(R, y, n1, n0):
    """rank 행렬 R(n×F) 과 라벨 y(bool) 로 피쳐별 AUC. y 가 여러 벌이면 y 는 (n×K)."""
    s = R.T @ y
    return (s - n1 * (n1 + 1) / 2) / (n1 * n0)


def main() -> None:
    ts, cols, X, o = load()
    hi, lo, cl = o["high"], o["low"], o["close"]
    d, rng, okw = regime(hi, lo, cl)
    y = y_sym(hi, lo, cl, rng)
    day = (ts // 86400)
    val0 = np.datetime64(SPLIT_VAL).astype("datetime64[s]").astype(np.int64)
    oos0 = np.datetime64(SPLIT_OOS).astype("datetime64[s]").astype(np.int64)

    base = okw & np.isfinite(X).all(1) & (y >= 0)
    print(f"패널 {len(cl):,} 봉 · 사용 가능 {base.sum():,} · 피쳐 {len(cols)}")
    print(f"미도달 {100*np.mean(y[okw]==-1):.1f}% · 판정불가(같은 봉 양쪽) {100*np.mean(y[okw]==-2):.1f}%")
    print(f"분할  TRAIN <{SPLIT_VAL}  VAL {SPLIT_VAL}~  OOS {SPLIT_OOS}~")

    per = {"TRAIN": ts < val0, "VAL": (ts >= val0) & (ts < oos0), "OOS": ts >= oos0}
    rng_r = np.random.default_rng(SEED)

    survivors = []
    for cell, nm in ((0, "횡보 dir=0"), (1, "추세 dir=+1"), (-1, "추세 dir=-1")):
        m_all = base & (d == cell)
        print("\n" + "=" * 78)
        print(f"{nm}   전체 {m_all.sum():,}행 · up 비율 {100*y[m_all].mean():.1f}%")
        print("=" * 78)
        res = {}
        for p in ("TRAIN", "VAL", "OOS"):
            m = m_all & per[p]
            if m.sum() < 2000:
                print(f"  {p}: {m.sum()}행 — 건너뜀"); continue
            Xi, yi = X[m], y[m].astype(np.float64)
            R = rankdata(Xi, axis=0)
            n1, n0 = yi.sum(), len(yi) - yi.sum()
            a = auc_mat(R, yi, n1, n0)
            res[p] = a
            if p == "TRAIN":
                # 회전 귀무: **셀 안에서** y 를 하루치 행 수의 배수로 순환이동한다. y 자신의 자기상관은
                # 그대로 두고 X-y 정렬만 깬다. 전체 패널을 돌리면 미해결(-1/-2) 이 섞여 못 쓴다.
                nd = max(len(np.unique(day[m])), 2)
                rpd = max(int(round(len(yi) / nd)), 1)
                shifts = rng_r.integers(1, nd, ROTS) * rpd
                Yr = np.column_stack([np.roll(yi, int(s)) for s in shifts])
                nn1 = Yr.sum(0); nn0 = len(yi) - nn1
                A0 = (R.T @ Yr - nn1 * (nn1 + 1) / 2) / (nn1 * nn0)
                dev0 = np.abs(A0 - 0.5)
                res["null"] = dev0
                res["nullmax"] = np.quantile(dev0.max(0), 0.95)
            print(f"  {p}: {m.sum():,}행 ({len(np.unique(day[m]))}일) · up {100*yi.mean():.1f}%"
                  + (f" · 회전귀무 최대통계 95% = |AUC-.5| {res['nullmax']:.4f}" if p == "TRAIN" else ""))
        if "TRAIN" not in res:
            continue
        aT = res["TRAIN"]; dev = np.abs(aT - 0.5)
        pv = (res["null"] >= dev[:, None]).mean(1)
        order = np.argsort(-dev)
        print(f"\n  TRAIN 상위 10 (p 는 회전 귀무 · 전체 최대통계 문턱 {res['nullmax']:.4f})")
        print(f"  {'피쳐':34} {'TRAIN':>7} {'p':>7} {'VAL':>7} {'OOS':>7}")
        for i in order[:10]:
            v = res.get("VAL"); w = res.get("OOS")
            print(f"  {cols[i]:34} {aT[i]:7.4f} {pv[i]:7.3f} "
                  f"{(v[i] if v is not None else float('nan')):7.4f} {(w[i] if w is not None else float('nan')):7.4f}")
        # 사전등록 생존 기준
        for i in order:
            if dev[i] < max(0.02, res["nullmax"]) or pv[i] >= 0.01:
                break
            v, w = res.get("VAL"), res.get("OOS")
            if v is None or w is None:
                continue
            same = np.sign(v[i] - 0.5) == np.sign(aT[i] - 0.5) and abs(v[i] - 0.5) >= 0.01
            survivors.append((nm, cols[i], aT[i], pv[i], v[i], w[i], same))
        n_ok = sum(1 for s in survivors if s[0] == nm and s[6])
        print(f"  ⇒ TRAIN 통과 {sum(1 for s in survivors if s[0]==nm)}개 · VAL 부호까지 통과 {n_ok}개")

    print("\n" + "=" * 78)
    print("생존자 (TRAIN 최대통계 초과 + p<0.01 + VAL 부호 일치)")
    print("=" * 78)
    ok = [s for s in survivors if s[6]]
    if not ok:
        print("  없음 — 152개를 뒤져서 회전 귀무의 최대통계를 넘고 VAL 까지 따라온 피쳐가 하나도 없다.")
    for nm, c, a, p, v, w, _ in ok:
        print(f"  {nm:12} {c:34} TRAIN {a:.4f} (p{p:.3f}) · VAL {v:.4f} · OOS {w:.4f}")
    print(f"\n  ⚠️OOS 는 여기서 **처음** 읽었다. 위 표의 OOS 열은 선택에 안 썼다.")


# ══════════════════════════════════════════════════════════════════════════
# 2차 — 생존자를 «쓸 수 있는 모양»으로. `python ... detail`
#   1) 레짐 «안에서만» 되는가, 아니면 무조건부로도 되는가 (진짜 상호작용인지)
#   2) OOS 분위별 P(up) 과 Q5-Q1 (일 블록 부트스트랩)
#   3) 생존자끼리 한 인자인가 (상관)
#   4) 적합 없이 z 합산했을 때 (게으른 결합)
# ══════════════════════════════════════════════════════════════════════════
KEEP = ["mtf_trend_4h", "mtf_trend_1h", "state7_trend_score", "rsi", "volume_profile_signal",
        "anchored_vwap_session_dist", "fvg_dist", "kalman_velocity", "garch_vol_z", "atr_pct_rank_288"]


def _auc1(x, yb):
    r = rankdata(x)
    n1 = yb.sum(); n0 = len(yb) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (r[yb.astype(bool)].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def detail() -> None:
    ts, cols, X, o = load()
    hi, lo, cl = o["high"], o["low"], o["close"]
    d, rng, okw = regime(hi, lo, cl)
    y = y_sym(hi, lo, cl, rng)
    day = ts // 86400
    base = okw & np.isfinite(X).all(1) & (y >= 0)
    val0 = np.datetime64(SPLIT_VAL).astype("datetime64[s]").astype(np.int64)
    oos0 = np.datetime64(SPLIT_OOS).astype("datetime64[s]").astype(np.int64)
    per = {"TRAIN": ts < val0, "VAL": (ts >= val0) & (ts < oos0), "OOS": ts >= oos0}
    ix = {c: i for i, c in enumerate(cols)}
    keep = [c for c in KEEP if c in ix]

    print("[1] 레짐 안에서만 되는가 — AUC(P(up)) · 무조건부와 비교")
    print(f"  {'피쳐':30} {'무조건부':>9} {'dir=+1':>9} {'dir=0':>9} {'dir=-1':>9}   (OOS)")
    for p in ("TRAIN", "OOS"):
        print(f"  --- {p}")
        for c in keep:
            x = X[:, ix[c]]
            row = []
            for m in (base & per[p],
                      base & per[p] & (d == 1), base & per[p] & (d == 0), base & per[p] & (d == -1)):
                row.append(_auc1(x[m], y[m]) if m.sum() > 500 else np.nan)
            print(f"  {c:30} " + " ".join(f"{v:9.4f}" for v in row))

    print("\n[2] OOS 분위별 P(up) — 추세 셀. Q5-Q1 은 일 블록 부트스트랩 95%")
    rs = np.random.default_rng(SEED)
    for cell, nm in ((1, "dir=+1"), (-1, "dir=-1")):
        m = base & per["OOS"] & (d == cell)
        print(f"\n  {nm}  n={m.sum():,} · up {100*y[m].mean():.1f}%")
        yy, dd = y[m].astype(float), day[m]
        for c in keep:
            x = X[m, ix[c]]
            q = np.searchsorted(np.quantile(x, [.2, .4, .6, .8]), x, side="right")
            pu = [100 * yy[q == k].mean() if (q == k).sum() else np.nan for k in range(5)]
            days = np.unique(dd)
            bs = []
            for _ in range(2000):
                pick = rs.choice(days, len(days))
                sel = np.concatenate([np.flatnonzero(dd == t) for t in pick])
                qq, ys = q[sel], yy[sel]
                if (qq == 4).sum() and (qq == 0).sum():
                    bs.append(ys[qq == 4].mean() - ys[qq == 0].mean())
            bs = np.sort(bs) * 100
            lo_, hi_ = bs[int(.025 * len(bs))], bs[int(.975 * len(bs))]
            star = "  ← 0 배제" if lo_ > 0 or hi_ < 0 else ""
            print(f"    {c:28} " + " ".join(f"{v:5.1f}" for v in pu) +
                  f"   Q5-Q1 {pu[4]-pu[0]:+6.1f}pp [{lo_:+6.1f},{hi_:+6.1f}]{star}")

    print("\n[3] 생존자끼리 한 인자인가 (OOS 추세 셀 상관)")
    m = base & per["OOS"] & (d != 0)
    C = np.corrcoef(np.column_stack([X[m, ix[c]] for c in keep]).T)
    print("     " + " ".join(f"{c[:8]:>8}" for c in keep))
    for i, c in enumerate(keep):
        print(f"  {c[:14]:14} " + " ".join(f"{C[i,j]:8.2f}" for j in range(len(keep))))

    print("\n[4] 적합 없이 z 합산 (부호는 TRAIN 에서만 결정)")
    tr = base & per["TRAIN"] & (d != 0)
    sign = {c: (1.0 if _auc1(X[tr, ix[c]], y[tr]) > 0.5 else -1.0) for c in keep}
    mu = {c: X[tr, ix[c]].mean() for c in keep}
    sd = {c: X[tr, ix[c]].std() or 1.0 for c in keep}
    for lab, sub in (("추세 전부", d != 0), ("dir=+1", d == 1), ("dir=-1", d == -1)):
        for p in ("TRAIN", "OOS"):
            m = base & per[p] & sub
            z = sum(sign[c] * (X[m, ix[c]] - mu[c]) / sd[c] for c in keep)
            print(f"  {lab:8} {p:5}  AUC {_auc1(z, y[m]):.4f}  n={m.sum():,}")





# ══════════════════════════════════════════════════════════════════════════
# 3차 — «지속 vs 되돌림»으로 접는다. `python ... fold`
#   1차/2차로 두 가지가 보였다: (a) 상위 시간대 추세 정렬 = **무조건부** 방향 피쳐(레짐과 무관),
#   (b) 변동성 = **레짐 안에서만** 부호가 갈리는 진짜 상호작용(상승추세 高변동→위, 하락추세 高변동→아래).
#   (b) 는 접어야 정체가 보인다. 타깃을 y_cont(=이동 방향으로 배리어 선착)로 바꾸고 두 축을 같이 잰다.
#   축은 상관행렬이 말해준다: 추세군끼리 |r| 0.46~0.93 = 한 인자 · 변동성군과는 r≈0.03 = 다른 인자.
# ══════════════════════════════════════════════════════════════════════════
AXES = {                       # (피쳐, 이동 방향으로 접나) -- 사전등록. TRAIN 을 보고 고르지 않는다
    "추세정렬 mtf_trend_4h": ("mtf_trend_4h", True),
    "추세정렬 state7_trend": ("state7_trend_score", True),
    "추세정렬 rsi-50": ("rsi", True),
    "추세정렬 kalman_vel": ("kalman_velocity", True),
    "변동성 garch_vol_z": ("garch_vol_z", False),
    "변동성 atr_pct_rank": ("atr_pct_rank_288", False),
    "변동성 volatility_z": ("volatility_z", False),
    "이동강도 move_ratio*": ("chop_index", False),
}


def fold() -> None:
    ts, cols, X, o = load()
    hi, lo, cl = o["high"], o["low"], o["close"]
    d, rng, okw = regime(hi, lo, cl)
    y = y_sym(hi, lo, cl, rng)
    day = ts // 86400
    base = okw & np.isfinite(X).all(1) & (y >= 0) & (d != 0)
    cont = ((y == 1) & (d == 1)) | ((y == 0) & (d == -1))      # 이동 방향으로 선착 = 지속
    val0 = np.datetime64(SPLIT_VAL).astype("datetime64[s]").astype(np.int64)
    oos0 = np.datetime64(SPLIT_OOS).astype("datetime64[s]").astype(np.int64)
    per = {"TRAIN": ts < val0, "VAL": (ts >= val0) & (ts < oos0), "OOS": ts >= oos0}
    ix = {c: i for i, c in enumerate(cols)}
    rs = np.random.default_rng(SEED)

    for p in ("TRAIN", "VAL", "OOS"):
        m = base & per[p]
        print(f"  {p:5} 추세 {m.sum():,}행 · **지속** {100*cont[m].mean():.1f}%  (되돌림 {100-100*cont[m].mean():.1f}%)")
    print("\n[A] AUC(P(지속)) — 접은 뒤. 0.5 = 정보 없음")
    print(f"  {'축':26} {'TRAIN':>8} {'VAL':>8} {'OOS':>8}")
    built = {}
    for nm, (c, fold_it) in AXES.items():
        if c not in ix:
            print(f"  {nm:26} 없음"); continue
        x = X[:, ix[c]] * (d if fold_it else 1)
        built[nm] = x
        print(f"  {nm:26} " + " ".join(f"{_auc1(x[base & per[p]], cont[base & per[p]].astype(float)):8.4f}"
                                       for p in ("TRAIN", "VAL", "OOS")))

    print("\n[B] OOS 분위별 P(지속) · Q5-Q1 일 블록 부트스트랩 95%")
    m = base & per["OOS"]
    yy, dd = cont[m].astype(float), day[m]
    days = np.unique(dd)
    for nm, x0 in built.items():
        x = x0[m]
        q = np.searchsorted(np.quantile(x, [.2, .4, .6, .8]), x, side="right")
        pu = [100 * yy[q == k].mean() if (q == k).sum() else np.nan for k in range(5)]
        bs = []
        for _ in range(2000):
            sel = np.concatenate([np.flatnonzero(dd == t) for t in rs.choice(days, len(days))])
            qq, ys = q[sel], yy[sel]
            if (qq == 4).sum() and (qq == 0).sum():
                bs.append(ys[qq == 4].mean() - ys[qq == 0].mean())
        bs = np.sort(bs) * 100
        lo_, hi_ = bs[int(.025 * len(bs))], bs[int(.975 * len(bs))]
        print(f"  {nm:26} " + " ".join(f"{v:5.1f}" for v in pu) +
              f"   Q5-Q1 {pu[4]-pu[0]:+6.1f}pp [{lo_:+6.1f},{hi_:+6.1f}]"
              + ("  ← 0 배제" if lo_ > 0 or hi_ < 0 else ""))

    print("\n[C] 두 축을 합치면 (적합 없이 z 합 · 부호는 TRAIN)")
    tr = base & per["TRAIN"]
    for lab, keys in (("추세정렬만", [k for k in built if k.startswith("추세")]),
                      ("변동성만", [k for k in built if k.startswith("변동")]),
                      ("둘 다", [k for k in built if k.startswith(("추세", "변동"))])):
        sgn = {k: (1.0 if _auc1(built[k][tr], cont[tr].astype(float)) > 0.5 else -1.0) for k in keys}
        z = {p: sum(sgn[k] * (built[k][base & per[p]] - built[k][tr].mean()) / (built[k][tr].std() or 1)
                    for k in keys) for p in ("TRAIN", "VAL", "OOS")}
        print(f"  {lab:10} " + " ".join(
            f"{p} {_auc1(z[p], cont[base & per[p]].astype(float)):.4f}" for p in ("TRAIN", "VAL", "OOS")))
    keys = list(built)
    sgn = {k: (1.0 if _auc1(built[k][tr], cont[tr].astype(float)) > 0.5 else -1.0) for k in keys}
    zo = sum(sgn[k] * (built[k][m] - built[k][tr].mean()) / (built[k][tr].std() or 1) for k in keys)
    q = np.searchsorted(np.quantile(zo, [.1, .3, .7, .9]), zo, side="right")
    print("\n  OOS 합산 점수 분위별 P(지속): " +
          " · ".join(f"{lab} {100*yy[q==k].mean():.1f}% (n={(q==k).sum()})"
                     for k, lab in enumerate(("하위10%", "10~30", "30~70", "70~90", "상위10%"))))
    bs = []
    for _ in range(2000):
        sel = np.concatenate([np.flatnonzero(dd == t) for t in rs.choice(days, len(days))])
        qq, ys = q[sel], yy[sel]
        if (qq == 4).sum() and (qq == 0).sum():
            bs.append(ys[qq == 4].mean() - ys[qq == 0].mean())
    bs = np.sort(bs) * 100
    print(f"  상위10% - 하위10% = {100*yy[q==4].mean()-100*yy[q==0].mean():+.1f}pp "
          f"[{bs[int(.025*len(bs))]:+.1f}, {bs[int(.975*len(bs))]:+.1f}]")





# ══════════════════════════════════════════════════════════════════════════
# 4차 — 공짜로 있는 축 두 개. `python ... free`
#   (a) 추세: 엔진은 move_ratio(|이동|/창폭) 를 **임계값으로만** 쓰고 버린다. 연속값으로 쓰면?
#   (b) 횡보: 방향은 152개 중 하나도 못 갈랐다(1차). 그러나 엔진의 횡보 시나리오는 유지/상단/하단이고
#       **«유지 vs 이탈»은 방향 문제가 아니다.** 이쪽은 갈리나?
# ══════════════════════════════════════════════════════════════════════════
def free() -> None:
    ts, cols, X, o = load()
    hi, lo, cl = o["high"], o["low"], o["close"]
    n = len(cl)
    d, rng, okw = regime(hi, lo, cl)
    y = y_sym(hi, lo, cl, rng)
    day = ts // 86400
    w_hi, w_lo = np.full(n, np.nan), np.full(n, np.nan)
    for k in range(WINDOW):
        w_hi[WINDOW:] = np.fmax(w_hi[WINDOW:], hi[WINDOW - k: n - k])
        w_lo[WINDOW:] = np.fmin(w_lo[WINDOW:], lo[WINDOW - k: n - k])
    move = np.full(n, np.nan); move[WINDOW:] = (cl[WINDOW:] - cl[:-WINDOW]) / cl[:-WINDOW] * 1e4
    rng_bp = rng / cl * 1e4
    ratio = np.where(rng_bp > 0, np.abs(move) / rng_bp, 0.0)
    val0 = np.datetime64(SPLIT_VAL).astype("datetime64[s]").astype(np.int64)
    oos0 = np.datetime64(SPLIT_OOS).astype("datetime64[s]").astype(np.int64)
    per = {"TRAIN": ts < val0, "VAL": (ts >= val0) & (ts < oos0), "OOS": ts >= oos0}
    ix = {c: i for i, c in enumerate(cols)}
    rs = np.random.default_rng(SEED)

    def qtab(x, m, yb, lab):
        xx, yy, dd = x[m], yb[m].astype(float), day[m]
        q = np.searchsorted(np.quantile(xx, [.2, .4, .6, .8]), xx, side="right")
        pu = [100 * yy[q == k].mean() for k in range(5)]
        days = np.unique(dd); bs = []
        for _ in range(2000):
            sel = np.concatenate([np.flatnonzero(dd == t) for t in rs.choice(days, len(days))])
            qq, ys = q[sel], yy[sel]
            if (qq == 4).sum() and (qq == 0).sum():
                bs.append(ys[qq == 4].mean() - ys[qq == 0].mean())
        bs = np.sort(bs) * 100
        lo_, hi_ = bs[int(.025 * len(bs))], bs[int(.975 * len(bs))]
        print(f"  {lab:26} " + " ".join(f"{v:5.1f}" for v in pu) +
              f"   Q5-Q1 {pu[4]-pu[0]:+6.1f}pp [{lo_:+6.1f},{hi_:+6.1f}]"
              + ("  ← 0 배제" if lo_ > 0 or hi_ < 0 else ""))

    print("[A] 추세 — 엔진이 이미 계산해 놓고 버리는 두 값 (지속=1)")
    b = okw & np.isfinite(X).all(1) & (y >= 0) & (d != 0)
    cont = ((y == 1) & (d == 1)) | ((y == 0) & (d == -1))
    for lab, x in (("move_ratio |이동|/창폭", ratio), ("창폭 range_bp", rng_bp)):
        print(f"  {lab:26} AUC " + " ".join(
            f"{p} {_auc1(x[b & per[p]], cont[b & per[p]].astype(float)):.4f}" for p in ("TRAIN", "VAL", "OOS")))
    print("  OOS 분위별 P(지속):")
    for lab, x in (("move_ratio", ratio), ("range_bp", rng_bp)):
        qtab(x, b & per["OOS"], cont, lab)
    print("  ⊕ 상위시간대 정렬(state7×dir) 과 같이:")
    z = X[:, ix["state7_trend_score"]] * d
    tr = b & per["TRAIN"]
    comb = ((z - z[tr].mean()) / z[tr].std() + (ratio - ratio[tr].mean()) / ratio[tr].std()
            + (X[:, ix["garch_vol_z"]] - X[tr, ix["garch_vol_z"]].mean()) / X[tr, ix["garch_vol_z"]].std())
    # move_ratio 는 TRAIN 에서 지속과 **음**이므로 부호를 뒤집어 넣는다
    comb = ((z - z[tr].mean()) / z[tr].std() - (ratio - ratio[tr].mean()) / ratio[tr].std()
            + (X[:, ix["garch_vol_z"]] - X[tr, ix["garch_vol_z"]].mean()) / X[tr, ix["garch_vol_z"]].std())
    print("    3축 z합 AUC " + " ".join(
        f"{p} {_auc1(comb[b & per[p]], cont[b & per[p]].astype(float)):.4f}" for p in ("TRAIN", "VAL", "OOS")))
    qtab(comb, b & per["OOS"], cont, "3축 z합")

    print("\n[B] 횡보 — «유지 vs 이탈» (창 고가/저가 선착 = 이탈). 유지=1")
    up_ex = np.maximum.accumulate(np.stack([(hi[np.minimum(np.arange(n) + j, n - 1)] - cl) for j in range(1, HOR + 1)], 1), 1)
    dn_ex = np.maximum.accumulate(np.stack([(cl - lo[np.minimum(np.arange(n) + j, n - 1)]) for j in range(1, HOR + 1)], 1), 1)
    br = (up_ex[:, -1] >= (w_hi - cl)) | (dn_ex[:, -1] >= (cl - w_lo))
    hold = ~br
    bz = okw & np.isfinite(X).all(1) & (d == 0) & (np.arange(n) + HOR < n)
    for p in ("TRAIN", "VAL", "OOS"):
        m = bz & per[p]
        print(f"  {p:5} n={m.sum():,} · 유지 {100*hold[m].mean():.1f}%")
    cand = ["garch_vol_z", "atr_pct_rank_288", "volatility_z", "bb_width_z", "chop_index",
            "compression_score", "realized_vol_ratio", "trade_intensity", "hurst_48", "amihud_illiquidity_z"]
    print(f"  {'피쳐':26} {'TRAIN':>8} {'VAL':>8} {'OOS':>8}")
    for c in cand:
        if c not in ix:
            continue
        x = X[:, ix[c]]
        print(f"  {c:26} " + " ".join(f"{_auc1(x[bz & per[p]], hold[bz & per[p]].astype(float)):8.4f}"
                                      for p in ("TRAIN", "VAL", "OOS")))
    print("  또한 기하 자체:")
    for lab, x in (("창폭 range_bp", rng_bp), ("move_ratio", ratio)):
        print(f"  {lab:26} " + " ".join(f"{_auc1(x[bz & per[p]], hold[bz & per[p]].astype(float)):8.4f}"
                                        for p in ("TRAIN", "VAL", "OOS")))
    print("  OOS 분위별 P(유지):")
    for lab, x in (("range_bp", rng_bp), ("garch_vol_z", X[:, ix["garch_vol_z"]]),
                   ("bb_width_z", X[:, ix["bb_width_z"]]), ("chop_index", X[:, ix["chop_index"]])):
        qtab(x, bz & per["OOS"], hold, lab)




# ══════════════════════════════════════════════════════════════════════════
# 5차 — 횡보 «유지 vs 이탈»의 공짜 축 통제. `python ... ctrl`
#   4차에서 range_bp 가 AUC 0.66 으로 가장 셌다. 그런데 **이탈 배리어가 창 고저 그 자체**라
#   「창이 넓으면 배리어가 멀다」는 기하다. 정보인지 자로 잰 거리인지 가른다:
#   range_bp 분위를 고정한 뒤에도 변동성 축이 남는가.
# ══════════════════════════════════════════════════════════════════════════
def ctrl() -> None:
    ts, cols, X, o = load()
    hi, lo, cl = o["high"], o["low"], o["close"]
    n = len(cl)
    d, rng, okw = regime(hi, lo, cl)
    day = ts // 86400
    w_hi, w_lo = np.full(n, np.nan), np.full(n, np.nan)
    for k in range(WINDOW):
        w_hi[WINDOW:] = np.fmax(w_hi[WINDOW:], hi[WINDOW - k: n - k])
        w_lo[WINDOW:] = np.fmin(w_lo[WINDOW:], lo[WINDOW - k: n - k])
    rng_bp = rng / cl * 1e4
    up_ex = np.maximum.accumulate(np.stack([hi[np.minimum(np.arange(n) + j, n - 1)] - cl for j in range(1, HOR + 1)], 1), 1)
    dn_ex = np.maximum.accumulate(np.stack([cl - lo[np.minimum(np.arange(n) + j, n - 1)] for j in range(1, HOR + 1)], 1), 1)
    hold = ~((up_ex[:, -1] >= (w_hi - cl)) | (dn_ex[:, -1] >= (cl - w_lo)))
    oos0 = np.datetime64(SPLIT_OOS).astype("datetime64[s]").astype(np.int64)
    ix = {c: i for i, c in enumerate(cols)}
    m = okw & np.isfinite(X).all(1) & (d == 0) & (np.arange(n) + HOR < n) & (ts >= oos0)
    rs = np.random.default_rng(SEED)
    R, G, Y, D = rng_bp[m], X[m, ix["garch_vol_z"]], hold[m].astype(float), day[m]
    qr = np.searchsorted(np.quantile(R, [.2, .4, .6, .8]), R, side="right")
    print(f"OOS 횡보 n={m.sum():,} · 유지 {100*Y.mean():.1f}%")
    print("\n[1] range_bp 분위를 고정한 뒤 garch_vol_z 의 P(유지) 5분위 (안에서 다시 5분위)")
    days = np.unique(D)
    for k in range(5):
        s = qr == k
        g = G[s]; yy = Y[s]; dd = D[s]
        qg = np.searchsorted(np.quantile(g, [.2, .4, .6, .8]), g, side="right")
        pu = [100 * yy[qg == j].mean() for j in range(5)]
        bs = []
        for _ in range(1500):
            sel = np.concatenate([np.flatnonzero(dd == t) for t in rs.choice(days, len(days)) if (dd == t).any()])
            q2, y2 = qg[sel], yy[sel]
            if (q2 == 4).sum() and (q2 == 0).sum():
                bs.append(y2[q2 == 4].mean() - y2[q2 == 0].mean())
        bs = np.sort(bs) * 100
        print(f"  range_bp Q{k+1} (중앙 {np.median(R[s]):5.0f}bp, n={s.sum():,}) 유지 "
              + " ".join(f"{v:5.1f}" for v in pu)
              + f"   Q5-Q1 {pu[4]-pu[0]:+6.1f}pp [{bs[int(.025*len(bs))]:+6.1f},{bs[int(.975*len(bs))]:+6.1f}]"
              + ("  ← 0 배제" if bs[int(.025*len(bs))] > 0 or bs[int(.975*len(bs))] < 0 else ""))
    print("\n[2] 반대로 — garch_vol_z 분위를 고정한 뒤 range_bp")
    qg = np.searchsorted(np.quantile(G, [.2, .4, .6, .8]), G, side="right")
    for k in range(5):
        s = qg == k
        r = R[s]; yy = Y[s]
        q2 = np.searchsorted(np.quantile(r, [.2, .4, .6, .8]), r, side="right")
        pu = [100 * yy[q2 == j].mean() for j in range(5)]
        print(f"  garch Q{k+1} (n={s.sum():,}) 유지 " + " ".join(f"{v:5.1f}" for v in pu)
              + f"   Q5-Q1 {pu[4]-pu[0]:+6.1f}pp")
    print("\n[3] 참고 — 두 축의 상관 " + f"{np.corrcoef(R, G)[0,1]:+.3f}")


if __name__ == "__main__":
    {"detail": detail, "fold": fold, "free": free, "ctrl": ctrl}.get(
        sys.argv[1] if len(sys.argv) > 1 else "", main)()
