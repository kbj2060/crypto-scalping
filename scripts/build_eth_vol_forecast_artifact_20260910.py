#!/usr/bin/env python3
"""**새 신호** — 24시간 변동성 확장 전망 아티팩트 (2026-09-10, 사용자 지시).

## 무엇이 새로운가
1. **새 정보원**: Deribit DVOL(30일 내재변동성). 바이낸스 perp klines 에서 파생 불가능 --
   화면의 다른 신호는 전부 같은 klines 파생이다.
2. **새 예측 대상**: **앞으로의** 변동성. 화면의 변동성 지표(ATR·atr_percentile·실현변동성)는
   전부 **과거만** 본다. 미래지향 입력이 지금 하나도 없다.

## 기각된 형태 (같은 세션, 전부 기록)
· VRP 방향 규칙 -- OOS +71.9bp 였으나 **일군집 95%CI 가 0 포함**([-17.4,+103.2], 독립일 82),
  이득이 **6월 한 달**에 집중(04 +38/05 -16/06 +156/07 -6/08 +66). TRAIN 727일 p=0.30. 기각.
· 변동성 확장 **분류에서 DVOL 증분** -- 건수를 맞추면 사라진다(OOS .600 vs .591). 처음 본
  +10.4pp 는 TRAIN 고정 임계가 만든 커버리지 이동 인공물이었다.
· VRP 를 극점 탐지기 **조건화** -- 순bp 갈림의 부호가 두 창에서 뒤집힌다(+6.4 → -1.6). 기각.

## 살아남은 것
· **회귀 증분**: DVOL 이 24시간 log RV 예측 잔차를 줄인다. 세 창 부호 일치, 증분 단조 증가
  (TRAIN 1.75% → OOS 5.35% → 🔒HOLDOUT 7.26%). 일군집 DM t = 2.46 / 1.67 / 1.37 --
  TRAIN(독립일 818)만 0 을 배제한다. t 가 주는 건 효과 축소가 아니라 **표본 축소**다
  (818 → 127 → 36일). 효과 크기는 오히려 커진다(+0.0018 → +0.0049 → +0.0128).
· **기준선 자체가 화면엔 없다**: HAR-RV 의 24시간 확장 분류 AUC .765/.809/.836.

## 지평 선택
6h AUC .687/.686/.715 · 12h .715/.721/.711 · **24h .765/.809/.836** -- 24시간 압승.

## 정직한 한계
DVOL 의 증분은 **회귀에서만** 유의하고 분류(화면이 쓰는 형태)에서는 AUC +0.0006 이다.
즉 화면에 새로 올라가는 정보의 대부분은 **HAR-RV(과거 변동성의 올바른 결합)**이고
DVOL 은 그 위에 얇게 얹힌다. 그래도 둘 다 이 대시보드엔 없던 것이다.

⚠️배포 모델은 **TRAIN(≤2026-03-31)에만 적합**한다. 그래야 위에 보고한 OOS·홀드아웃 숫자가
   그대로 배포본의 표본외 성적이 된다. 재적합해서 배포하면 그 숫자는 근거가 아니게 된다.
"""
from __future__ import annotations
import argparse, json, os, sys
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path: sys.path.insert(0, str(_p))
import research_eth_dvol_volexpansion_20260910 as V  # noqa: E402

LIVE = ROOT / "data/live/eth_vol_forecast_artifact"
TRAIN_END = pd.Timestamp("2026-03-31")
OOS_END = pd.Timestamp("2026-08-04 10:00")
FEATS = ["l_rv1", "l_rv24", "l_rv168", "l_dvol", "vrp"]
TIERS = (("위험", 0.10), ("주의", 0.25))     # 화면 「위험도」 어휘: 안정/주의/위험


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(LIVE))
    a = ap.parse_args()
    import joblib
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.metrics import roc_auc_score
    OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)

    d = V.build()
    ts = pd.to_datetime(d["timestamp"])
    tr = (ts <= TRAIN_END).to_numpy()
    oos = ((ts > TRAIN_END) & (ts <= OOS_END)).to_numpy()
    hold = (ts > OOS_END).to_numpy()
    X = d[FEATS].to_numpy(float); yb = d["expand"].to_numpy(); y = d["l_rv_fwd"].to_numpy()
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    clf = LogisticRegression(max_iter=2000).fit((X[tr] - mu) / sd, yb[tr])
    reg = LinearRegression().fit(X[tr], y[tr])
    p = clf.predict_proba((X - mu) / sd)[:, 1]

    # 컷은 OOS 에서, 정밀도는 🔒홀드아웃에서 (순환 방지 -- 컷과 평가 창을 분리)
    cuts = {g: float(np.quantile(p[oos], 1 - q)) for g, q in TIERS}
    prec, per_day, auc = {}, {}, {}
    days_h = (ts[hold].max() - ts[hold].min()).total_seconds() / 86400
    prev = None
    for g, _ in TIERS:
        sel = hold & (p >= cuts[g]) & ((p < cuts[prev]) if prev else True)
        prec[g] = round(float(yb[sel].mean()), 4) if sel.sum() >= 10 else None
        per_day[g] = round(sel.sum() / days_h, 2); prev = g
    for w, m in (("TRAIN", tr), ("OOS", oos), ("HOLDOUT", hold)):
        auc[w] = round(float(roc_auc_score(yb[m], p[m])), 4)
    print(f"학습 {tr.sum():,}시간(≤{TRAIN_END:%Y-%m-%d}) · OOS {oos.sum():,} · 🔒홀드아웃 {hold.sum():,}")
    print(f"AUC  TRAIN {auc['TRAIN']} · OOS {auc['OOS']} · 🔒HOLDOUT {auc['HOLDOUT']}")
    print(f"기저율 홀드아웃 {yb[hold].mean():.3f}")
    for g, _ in TIERS:
        print(f"  {g}  컷 {cuts[g]:.4f}  홀드아웃 정밀도 {prec[g]}  {per_day[g]}건/일")

    joblib.dump({"clf": clf, "reg": reg, "mu": mu, "sd": sd}, OUT / "model.joblib")
    meta = {
        "rule_id": "eth_vol_expansion_24h_har_dvol_20260910",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target": f"다음 {V.H}시간 실현변동성이 현재 24시간 실현변동성의 {V.EXPAND_K}배 이상",
        "features": FEATS, "horizon_hours": V.H, "expand_k": V.EXPAND_K,
        "model": "logistic(HAR-RV + DVOL + VRP)", "n_train": int(tr.sum()),
        "train_span": [str(ts[tr].min()), str(ts[tr].max())],
        "oos_span": [str(ts[oos].min()), str(ts[oos].max())],
        "holdout_span": [str(ts[hold].min()), str(ts[hold].max())],
        "auc": auc, "cuts": cuts, "precision_holdout": prec, "per_day_holdout": per_day,
        "base_rate_holdout": round(float(yb[hold].mean()), 4),
        "dvol_increment": {"resid_reduction_pct": {"TRAIN": 1.75, "OOS": 5.35, "HOLDOUT": 7.26},
                           "dm_t_dayclustered": {"TRAIN": 2.46, "OOS": 1.67, "HOLDOUT": 1.37},
                           "independent_days": {"TRAIN": 818, "OOS": 127, "HOLDOUT": 36},
                           "note": "회귀에서만 유의. 분류 AUC 증분은 홀드아웃 +0.0006 으로 사실상 0."},
        "rejected_forms": ["VRP 방향 규칙(일군집 CI 0 포함·6월 한 달 집중)",
                           "분류에서의 DVOL 증분(건수 맞추면 소멸)",
                           "극점 탐지기 조건화(순bp 부호 뒤집힘)"],
        "note": ("변동성 전망이다. **방향도 수익도 예측하지 않는다.** 사람이 크기·손절폭·관망을 "
                 "정할 때 쓰는 맥락이다. 컷은 OOS 에서 잡고 정밀도는 홀드아웃에서 쟀다. "
                 "모델은 TRAIN 에만 적합했으므로 위 OOS·홀드아웃 숫자가 그대로 배포본 성적이다."),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"\n저장 {OUT} ({sum(f.stat().st_size for f in OUT.iterdir())/1e3:.0f}KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
