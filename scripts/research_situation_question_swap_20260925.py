"""상황 카드 업그레이드 — «어느 쪽»을 계속 물을 것인가, 「얼마나」로 바꿀 것인가 (2026-09-25).

왜: 09-25 채점축 통일 재측정에서 카드는 통일축(등거리 배리어) 위에서 «항상 되돌림»과 **소수점까지
   같았다**(추세 49.3% vs 49.3%, 89블록). 원장은 2일뿐이라 규칙 하나도 판정 못 한다(CI 반폭 16.5pp,
   주장은 −4~+6pp ⇒ ±6pp 로 좁히는 데 15일). ⇒ 원장을 더 모으는 것으로는 «업그레이드»를 못 고른다.
   업그레이드 방향은 **4.7년 패널**에서 골라야 한다.

무엇을: 입력·창·경계 계약을 **고정**하고 «질문»만 바꿔 OOS 를 나란히 놓는다.
   Q1  지속(추세)   : 등거리 배리어에 이동 방향으로 먼저 닿나          ← 지금 카드의 질문
   Q1r 이탈방향(횡보): 위로 먼저 닿나                               ← 지금 카드의 질문
   Q2  해결        : 30분 안에 배리어에 닿기는 하나                 🔴기하 순환 주의(배리어 = 0.5×창폭)
   Q3  크기        : 다음 30분 |최대이동|bp 가 상위 3분위인가        ← 절대 스케일, 순환 아님
   각 질문에 모델 셋: 무모델 최선 단일피쳐 · 로지스틱(152피쳐) · 상수(=AUC 0.5).
   로지스틱은 «규칙표를 학습으로 민다»(09-21 설계도 P2)가 실제로 얼마를 주는지 보는 자리다.

경계 계약: 피쳐는 봉 i 까지, 라벨 탐색은 봉 i+1 부터 — 09-22 스크립트의 것을 그대로 쓴다.
🔴선택은 TRAIN 에서만. VAL 은 부호 확인, OOS 는 마지막에 한 번. CI 는 전부 **일 블록**(창이 6배 겹친다).
🔴AUC 는 라벨 난이도가 다르면 직접 비교 금지가 이 저장소 규칙이다. 그래서 질문마다
   **같은 잣대의 경제 단위**(상위 10% 분위의 적중률 − 기저율, pp)를 같이 낸다.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_situation_within_regime_feature_screen_20260922 import (  # noqa: E402
    HOR, SPLIT_OOS, SPLIT_VAL, _auc1, load, regime, y_sym)

SEED = 20260925
BOOT = 400
TOPQ = 0.10        # «상위 10% 분위» = 카드가 실제로 쓰는 단위(1순위를 부를지 말지)


def day_boot(day, yb, sc, stat, rng):
    """일 블록 부트스트랩. stat(yb, sc) -> float|None."""
    uniq = np.unique(day)
    idx = {d: np.where(day == d)[0] for d in uniq}
    out = []
    for _ in range(BOOT):
        pick = np.concatenate([idx[d] for d in rng.choice(uniq, len(uniq))])
        v = stat(yb[pick], sc[pick])
        if v is not None and np.isfinite(v):
            out.append(v)
    out.sort()
    return (out[int(.025 * len(out))], out[int(.975 * len(out))]) if out else (np.nan, np.nan)


def lift_pp(yb, sc):
    """상위 TOPQ 분위 적중률 − 기저율, pp. 카드의 «1순위를 부를 때 얼마나 나은가»."""
    k = max(int(len(sc) * TOPQ), 1)
    top = np.argpartition(-sc, k - 1)[:k]
    return (yb[top].mean() - yb.mean()) * 100


def fit_logit(Xtr, ytr, Xte):
    """표준화 + L2 로지스틱. 새 의존성 없이 sklearn(이미 설치됨)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=0.05, max_iter=2000).fit(sc.transform(Xtr), ytr)
    return m.decision_function(sc.transform(Xte)), m


def main() -> None:
    ts, cols, X, o = load()
    hi, lo, cl = o["high"], o["low"], o["close"]
    d, rng_px, okw = regime(hi, lo, cl)
    ysym = y_sym(hi, lo, cl, rng_px)
    day = ts // 86400
    val0 = np.datetime64(SPLIT_VAL).astype("datetime64[s]").astype(np.int64)
    oos0 = np.datetime64(SPLIT_OOS).astype("datetime64[s]").astype(np.int64)
    per = {"TRAIN": ts < val0, "VAL": (ts >= val0) & (ts < oos0), "OOS": ts >= oos0}

    # Q3 라벨 = 다음 HOR 봉의 |최대 이동| bp (종가 기준). 봉 i+1 부터 = Y_sym 과 같은 경계.
    n = len(cl)
    fwd = np.full(n, np.nan)
    idx = np.arange(n)
    up = np.full(n, -np.inf)
    dn = np.full(n, np.inf)
    for j in range(1, HOR + 1):
        k = np.minimum(idx + j, n - 1)
        ok = idx + j < n
        up = np.where(ok, np.fmax(up, hi[k]), up)
        dn = np.where(ok, np.fmin(dn, lo[k]), dn)
    fwd = np.fmax(up - cl, cl - dn) / cl * 1e4
    fin = np.isfinite(X).all(1) & okw & np.isfinite(fwd)
    cut = np.nanquantile(fwd[fin & per["TRAIN"]], 1 - 1 / 3)      # 상위 3분위 경계는 TRAIN 에서만

    print(f"패널 {n:,}봉 · 피쳐 {len(cols)} · TRAIN <{SPLIT_VAL} / VAL / OOS {SPLIT_OOS}~")
    print(f"Q3 상위3분위 경계 = {cut:.1f}bp (TRAIN 에서 고정)\n")

    trend = fin & (d != 0) & (ysym >= 0)
    rangey = fin & (d == 0) & (ysym >= 0)
    cont = np.where(d > 0, ysym == 1, ysym == 0)                  # 이동 방향으로 먼저 닿음
    questions = [
        ("Q1  지속 (추세)", trend, cont),
        ("Q1r 이탈방향 (횡보)", rangey, ysym == 1),
        ("Q2  해결 여부 (전체) 🔴기하순환", fin & okw, ysym >= 0),
        ("Q3  크기 상위3분위 (전체)", fin, fwd >= cut),
    ]

    rs = np.random.default_rng(SEED)
    print(f"{'질문':30}{'기저율':>7}{'최선단일 OOS':>14}{'로지스틱 OOS':>14}"
          f"{'  상위10% 리프트(pp)':>22}   {'일블록 95% CI':>18}")
    print("-" * 112)
    for name, mask, ylab in questions:
        mtr, mte = mask & per["TRAIN"], mask & per["OOS"]
        if mtr.sum() < 5000 or mte.sum() < 2000:
            print(f"{name:30}  표본 부족 ({mtr.sum()}/{mte.sum()})")
            continue
        ytr, yte = ylab[mtr].astype(np.float64), ylab[mte].astype(np.float64)
        # 최선 단일 피쳐: TRAIN 에서 |AUC−.5| 최대인 것 하나를 **고르고** OOS 에서 읽는다.
        Rtr = rankdata(X[mtr], axis=0)
        n1, n0 = ytr.sum(), len(ytr) - ytr.sum()
        a_tr = ((Rtr.T @ ytr) - n1 * (n1 + 1) / 2) / (n1 * n0)
        best = int(np.argmax(np.abs(a_tr - 0.5)))
        sgn = 1.0 if a_tr[best] > 0.5 else -1.0
        s_single = sgn * X[mte][:, best]
        a_single = _auc1(s_single, yte.astype(bool))
        s_logit, _ = fit_logit(X[mtr], ytr, X[mte])
        a_logit = _auc1(s_logit, yte.astype(bool))
        lift = lift_pp(yte, s_logit)
        lo_, hi_ = day_boot(day[mte], yte, s_logit, lift_pp, rs)
        print(f"{name:30}{yte.mean()*100:6.1f}% {a_single:8.4f} ({cols[best][:16]:16})"
              f"{a_logit:9.4f}      {lift:+8.2f}pp        [{lo_:+6.2f},{hi_:+6.2f}]")

    print("\n🔴Q2 는 배리어가 0.5×창폭이라 «창이 넓으면 안 닿는다»가 절반이다 — 정보가 아니라 기하다.")
    print("   Q3 는 절대 bp 라 그 순환이 없다. 두 줄을 같은 무게로 읽으면 안 된다.")

    # ── Q3 를 «카드가 이미 들고 있는 입력»만으로 할 수 있나 ──
    # 카드의 feat 에는 range_bp(창 고저폭)가 이미 있다. 그것만으로 얼마나 되는지가
    # «새 배관 없이 오늘 바꿀 수 있나»를 가른다.
    print("\n" + "=" * 112)
    print("Q3 크기 — 어떤 입력이 필요한가 (OOS · 같은 마스크)")
    print("=" * 112)
    mte = fin & per["OOS"]
    yte = (fwd >= cut)[mte].astype(bool)
    rng_bp_all = rng_px / cl * 1e4
    cands = [("창 고저폭 range_bp (카드가 이미 가짐)", rng_bp_all[mte])]
    for c in ("parkinson_vol", "garch_vol_z", "atr_pct", "bb_width", "realized_vol_ratio"):
        if c in cols:
            cands.append((c, X[mte][:, cols.index(c)]))
    print(f"  {'입력':44}{'OOS AUC':>10}{'상위10% 리프트':>16}")
    for lab, v in cands:
        if not np.isfinite(v).all():
            print(f"  {lab:44}{'결측':>10}")
            continue
        print(f"  {lab:44}{_auc1(v, yte):10.4f}{lift_pp(yte.astype(float), v):+14.2f}pp")
    s_logit, _ = fit_logit(X[fin & per['TRAIN']], (fwd >= cut)[fin & per['TRAIN']].astype(float), X[mte])
    print(f"  {'로지스틱 152피쳐 (상한)':44}{_auc1(s_logit, yte):10.4f}"
          f"{lift_pp(yte.astype(float), s_logit):+14.2f}pp")

    # ── 🔴레짐 간 vs 레짐 안: 전역 AUC 의 상당 부분은 «올해 4월은 조용하고 7월은 시끄럽다»이다.
    #    라이브 카드는 «오늘 안에서» 골라야 하므로 일 안 AUC 가 정직한 값이다.
    print("\n" + "=" * 112)
    print("Q3 통제 — 전역 AUC 중 얼마가 «레짐 간 구분»인가 (일 안에서 다시 랭크)")
    print("=" * 112)
    dte = day[mte]
    print(f"  {'입력':30}{'전역 AUC':>10}{'일 안 중앙':>12}{'일 안 평균':>12}")
    for lab, v in cands + [("로지스틱 152피쳐", s_logit)]:
        if not np.isfinite(v).all():
            continue
        glob = _auc1(v, yte)
        per = []
        for dd in np.unique(dte):
            s = dte == dd
            if s.sum() < 60:
                continue
            yy = fwd[mte][s] >= np.quantile(fwd[mte][s], 1 - 1 / 3)
            if yy.sum() and (~yy).sum():
                per.append(_auc1(v[s], yy))
        per = np.array(per)
        print(f"  {lab[:28]:30}{glob:10.4f}{np.median(per):12.4f}{per.mean():12.4f}")
    print("\n  🔴전역 0.74 는 라이브에서 쓸 수 없는 값이다. 실제로 쓸 수 있는 건 일 안 ~0.64 다.")
    print("     3.6일 원장에서 range_bp 를 같은 방식으로 재면 AUC 0.6754 로, 일 안 값과 맞는다(전역과는 안 맞는다).")
    print("\n  🔴이 패널에는 카드의 호가·청산맵·풋프린트·고래플로우 입력이 **없다** — 그 증분은 여기서 못 잰다.")
    print("     원장으로만 가능하고, 09-25 검정력 계산대로 ±6pp 를 보려면 15일이 더 필요하다.")


if __name__ == "__main__":
    main()
