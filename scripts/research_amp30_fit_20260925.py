"""«다음 30분 고저폭» 예보 상수 적합 — 변동성 카드에 붙일 30분 줄 (2026-09-25).

09-25 질문 교체 실험의 결론(방향 천장 +3.5pp vs 크기 +43pp)을 화면에 올리기 위한 적합.
사용자 결정: 새 카드를 만들지 않고 **변동성 수준(4시간) 카드에 30분 줄로 통합**.

정의 — 결정 시점 t 의 완결 5분봉 기준. 피쳐는 봉 t 까지, 라벨은 t+1..t+6 (같은 봉 공유 없음).
  R_t    직전 30분 고저폭 bp = (max hi[t-5..t] − min lo[t-5..t]) / close[t] × 1e4
         🔴카드의 `feat.range_bp` 와 **같은 값**이다 — 새 입력이 아니다.
  F_t    다음 30분 고저폭 bp = (max hi[t+1..t+6] − min lo[t+1..t+6]) / close[t] × 1e4
  base_t 최근 24시간(288봉) R 의 중앙값        ← «평소»
  thr_t  최근 24시간 R 의 2/3 분위              ← «큰 쪽» 임계
  mult_t R_t / base_t                          ← 화면의 «배수»
  y_t    F_t ≥ thr_t                            ← 화면의 «큰 쪽 확률»

🔴임계를 **후행 분위로 선언한다**. 전역 분위로 재면 「2026-04 는 조용하고 07 은 시끄럽다」를
   세게 되고, 적응형 임계가 그 드리프트를 이미 흡수한다는 걸 09-23 에 한 번 놓쳤다
   ([[feedback_gate_change_must_be_measured_with_live_threshold_convention_20260923]]).
   같은 이유로 성능은 **일 안 AUC** 로도 같이 낸다 — 라이브 카드는 오늘 안에서 고른다.

적합은 TRAIN(<2025-09) 에서만. VAL 은 부호·보정 확인, OOS 는 마지막 한 번.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_situation_within_regime_feature_screen_20260922 import _auc1  # noqa: E402

PANEL = "data/binance_vision/panel/ETHUSDT.parquet"
WIN, HOR, LOOK = 6, 6, 288          # 30분 창 · 30분 지평 · 24시간 기준창
BIG_Q = 2 / 3                       # «큰 쪽» = 최근 24시간 상위 3분위
SPLIT_VAL, SPLIT_OOS = "2025-09-01", "2026-04-01"
BOOT, SEED = 400, 20260925


def build():
    t = pq.read_table(PANEL, columns=["timestamp", "high", "low", "close"]).to_pandas()
    t = t.sort_values("timestamp").reset_index(drop=True)
    hi, lo, cl = t["high"].to_numpy(float), t["low"].to_numpy(float), t["close"].to_numpy(float)
    # R: 봉 t 포함 직전 WIN 봉
    r_hi = pd.Series(hi).rolling(WIN).max().to_numpy()
    r_lo = pd.Series(lo).rolling(WIN).min().to_numpy()
    R = (r_hi - r_lo) / cl * 1e4
    # F: 봉 t+1..t+HOR  (뒤에서 앞으로 밀어 같은 봉 공유를 막는다)
    f_hi = pd.Series(hi).rolling(HOR).max().shift(-HOR).to_numpy()
    f_lo = pd.Series(lo).rolling(HOR).min().shift(-HOR).to_numpy()
    F = (f_hi - f_lo) / cl * 1e4
    sR = pd.Series(R)
    base = sR.rolling(LOOK, min_periods=LOOK).median().to_numpy()
    thr = sR.rolling(LOOK, min_periods=LOOK).quantile(BIG_Q).to_numpy()
    # 🔴R=0(30분 내내 한 가격) 봉이 있다 — ln(배수)가 −inf 가 된다. 분모로 쓰는 값은 전부 양수여야 한다.
    ok = (np.isfinite(R) & np.isfinite(F) & np.isfinite(base) & np.isfinite(thr)
          & (base > 0) & (R > 0) & (F > 0))
    return t["timestamp"].to_numpy(), R, F, base, thr, ok


def day_ci(day, y, p, stat, rng):
    u = np.unique(day)
    idx = {d: np.where(day == d)[0] for d in u}
    v = []
    for _ in range(BOOT):
        pick = np.concatenate([idx[d] for d in rng.choice(u, len(u))])
        x = stat(y[pick], p[pick])
        if x is not None and np.isfinite(x):
            v.append(x)
    v.sort()
    return (v[int(.025 * len(v))], v[int(.975 * len(v))]) if v else (np.nan, np.nan)


def main() -> None:
    ts, R, F, base, thr, ok = build()
    mult = R / base
    y = (F >= thr) & ok
    day = ts.astype("datetime64[D]").astype(np.int64)
    val0 = np.datetime64(SPLIT_VAL)
    oos0 = np.datetime64(SPLIT_OOS)
    tsd = ts.astype("datetime64[s]")
    per = {"TRAIN": ok & (tsd < val0), "VAL": ok & (tsd >= val0) & (tsd < oos0), "OOS": ok & (tsd >= oos0)}
    print(f"패널 {len(R):,}봉 · 사용 {ok.sum():,} · TRAIN {per['TRAIN'].sum():,} / VAL {per['VAL'].sum():,} / OOS {per['OOS'].sum():,}")
    print(f"기저율(큰 쪽) TRAIN {y[per['TRAIN']].mean()*100:.1f}% · VAL {y[per['VAL']].mean()*100:.1f}% · OOS {y[per['OOS']].mean()*100:.1f}%"
          "   🔴후행 분위라 33% 근처에서 스스로 안정된다(전역 분위였다면 창마다 떠돈다)")

    # ── ① 진폭 배수: F/R 의 중앙 비율 하나 (상수 1개, TRAIN)
    tr = per["TRAIN"]
    a = float(np.median(F[tr] / R[tr]))
    print(f"\n① 진폭 예보  pred_bp = {a:.4f} × R   (TRAIN F/R 중앙)")
    for p in ("TRAIN", "VAL", "OOS"):
        m = per[p]
        rel = (a * R[m] - F[m]) / F[m]
        print(f"   {p:5} 중앙 상대오차 {np.median(rel)*100:+5.1f}%  ·  |오차|<30% 비율 {np.mean(np.abs(rel) < .3)*100:4.1f}%"
              f"  ·  실측 중앙 {np.median(F[m]):5.1f}bp / 예보 중앙 {np.median(a*R[m]):5.1f}bp")

    # ── ② 큰 쪽 확률: sigmoid(α + β·ln mult), TRAIN 적합
    from sklearn.linear_model import LogisticRegression
    x_tr = np.log(mult[tr]).reshape(-1, 1)
    lr = LogisticRegression().fit(x_tr, y[tr])
    A, B = float(lr.intercept_[0]), float(lr.coef_[0][0])
    print(f"\n② 큰 쪽 확률  p = sigmoid({A:+.4f} {B:+.4f}·ln(배수))")
    rs = np.random.default_rng(SEED)
    for p in ("TRAIN", "VAL", "OOS"):
        m = per[p]
        pr = 1 / (1 + np.exp(-(A + B * np.log(mult[m]))))
        auc = _auc1(mult[m], y[m])
        # 일 안 AUC — 라이브가 실제로 하는 일
        inday = []
        for d in np.unique(day[m]):
            s = day[m] == d
            if s.sum() < 60 or y[m][s].sum() == 0 or (~y[m][s]).sum() == 0:
                continue
            inday.append(_auc1(mult[m][s], y[m][s]))
        lo_, hi_ = day_ci(day[m], y[m].astype(float), mult[m], lambda yy, pp: _auc1(pp, yy.astype(bool)), rs)
        # 보정 기울기: 관측 로짓 ~ 예측 로짓
        z = A + B * np.log(mult[m])
        slope = float(np.polyfit(z, y[m].astype(float), 1)[0] / np.polyfit(z, 1 / (1 + np.exp(-z)), 1)[0])
        print(f"   {p:5} AUC {auc:.4f} [{lo_:.4f},{hi_:.4f}]  ·  **일 안** 중앙 {np.median(inday):.4f}"
              f"  ·  말한 {pr.mean()*100:4.1f}% / 실제 {y[m].mean()*100:4.1f}%  ·  보정기울기 {slope:.3f}")

    # ── ③ 화면이 실제로 부를 구간 — 배수 5분위별
    m = per["OOS"]
    q = np.quantile(mult[m], [.2, .4, .6, .8])
    b = np.digitize(mult[m], q)
    print("\n③ OOS 배수 5분위 — 화면이 «1.4배»라고 말할 때 실제로 무엇이 일어나나")
    print(f"   {'분위':6}{'배수 중앙':>10}{'큰 쪽 실제':>11}{'말한 확률':>10}{'다음30분 고저폭 중앙':>20}")
    for i in range(5):
        s = b == i
        pr = 1 / (1 + np.exp(-(A + B * np.log(mult[m][s]))))
        print(f"   Q{i+1:<5}{np.median(mult[m][s]):9.2f}x{y[m][s].mean()*100:10.1f}%{pr.mean()*100:9.1f}%"
              f"{np.median(F[m][s]):19.1f}bp")
    print("\n🔴«큰 쪽»은 **최근 24시간 대비** 상위 3분위다 — 절대 크기가 아니라 «오늘 기준으로 크냐»다.")
    print("   전역 기준으로 바꾸면 조용한 주에는 0%, 시끄러운 주에는 100% 가 되어 화면이 죽는다.")


if __name__ == "__main__":
    main()
