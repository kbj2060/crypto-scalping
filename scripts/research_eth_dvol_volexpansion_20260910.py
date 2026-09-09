#!/usr/bin/env python3
"""**새 신호** 후보 2 — 내재변동성이 **앞으로의 변동성 확장**을 예고하는가 (2026-09-10).

## 방향 축은 기각됐다 (같은 세션, 앞선 스크립트)
VRP 방향 규칙은 OOS 에서 +71.9bp 를 냈지만 **일군집 95%CI 가 0 을 포함**하고([-17.4, +103.2],
독립일 82일), 이득이 **6월 한 달**에 몰려 있었다(04 +38 / 05 -16 / 06 +156 / 07 -6 / 08 +66).
TRAIN 727일은 +7.6bp p=0.30. → 기각. 홀드아웃은 열지 않았다.

## 왜 이 축으로 바꾸나
DVOL 은 **변동성의 가격**이지 방향의 가격이 아니다. 그리고 이 대시보드에는 지금 **미래지향
변동성 입력이 하나도 없다** -- ATR·실현변동성·atr_percentile 전부 과거만 본다.
"앞으로 변동성이 커지는가"는 (a) 검정력이 방향보다 훨씬 높고(변동성은 지속적이고 매 시간
관측된다) (b) 사람이 재량 매매할 때 직접 쓰는 정보다(포지션 크기·손절 폭·관망 여부).

## 사전등록
- **타깃**: 앞으로 24시간 실현변동성 RV[t+1 .. t+24] (5분 로그수익률, 연율화 %)
- **기준선(대조군)**: HAR-RV -- 과거만 쓰는 표준 모형. log RV ~ log RV_1h + log RV_24h + log RV_168h
  ⚠️호메로스 5.12절 정신: 새 피쳐를 주장하려면 **그 피쳐 없는 모형**이 먼저 돌아야 한다.
- **처치**: HAR + log DVOL (그리고 HAR + log DVOL + VRP)
- **판정**: OOS 에서 **증분** R² 가 양수이고, 실무형(확장 분류)에서 기준선 대비 lift 가 있을 것.
  세 창(TRAIN/OOS/🔒홀드아웃) 전부에서 부호가 같아야 한다.
- **실무형**: "다음 24시간 RV 가 현재 RV 의 1.3배 이상" 이진 분류. 기준선(HAR만) 대비
  정밀도·AUC 증분. 전건을 센다.
- 회귀는 **확장 원도우**로 매 달 재적합(미래참조 없이). DVOL 은 시각 t 종가값만 쓴다.
"""
from __future__ import annotations
import argparse, os, sys
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path: sys.path.insert(0, str(_p))
import research_eth_dvol_vrp_signal_20260910 as S  # noqa: E402

OUT = ROOT / "tmp/eth_dvol_vrp_20260910"
H = int(os.environ.get("VOLH", "24"))
EXPAND_K = 1.3
TRAIN_END = pd.Timestamp("2026-03-31")
OOS_END = pd.Timestamp("2026-08-04 10:00")


def build() -> pd.DataFrame:
    dv = pd.read_csv(S.DVOL_CSV, parse_dates=["timestamp"])[["timestamp", "close"]]
    tail = S.fetch_dvol_tail(dv.timestamp.max())
    dv = pd.concat([dv, tail], ignore_index=True).drop_duplicates("timestamp", keep="first")
    dv = dv.sort_values("timestamp").reset_index(drop=True).rename(columns={"close": "dvol"})
    px5 = S.load_px5(); p = px5.set_index("timestamp")
    r5 = np.log(p["close"]).diff()
    ann = np.sqrt(288 * 365) * 100

    # ⭐피쳐는 **라이브 모듈의 build_features 를 그대로** 쓴다 -- 식을 두 벌 두지 않는다.
    import live_eth_vol_forecast_20260910 as LF
    kl_in = px5[["timestamp", "close"]].copy()
    dv_in = dv[["timestamp", "dvol"]].copy()
    feat = LF.build_features(kl_in, dv_in)
    h = pd.DataFrame({"close": p["close"].resample("1h").last()}).dropna()
    h = h.join(feat[["rv1", "rv24", "rv168"]], how="inner")
    # 타깃: 미래 24시간 실현변동성 (t+1 .. t+24) -- 시각 t 이후만 쓴다
    fut = (r5[::-1].rolling(H * 12, min_periods=H * 6).std()[::-1] * ann).resample("1h").first()
    h["rv_fwd"] = fut.shift(-1)
    d = h.join(dv.set_index("timestamp")["dvol"], how="inner").dropna().reset_index()
    d.columns = ["timestamp", "close", "rv1", "rv24", "rv168", "rv_fwd", "dvol"]
    d["vrp"] = d["dvol"] - d["rv24"]
    for c in ("rv1", "rv24", "rv168", "rv_fwd", "dvol"):
        d[f"l_{c}"] = np.log(d[c].clip(lower=1e-6))
    d["expand"] = (d["rv_fwd"] >= EXPAND_K * d["rv24"]).astype(int)
    return d.dropna().reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--holdout", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import roc_auc_score

    d = build()
    ts = pd.to_datetime(d["timestamp"])
    print(f"표본 {ts.min():%Y-%m-%d} ~ {ts.max():%Y-%m-%d}  ({len(d):,}시간)")
    print(f"확장(다음 24h RV >= {EXPAND_K}x 현재 RV24) 기저율 {d.expand.mean():.3f}\n")
    win = {"TRAIN": (ts <= TRAIN_END).to_numpy(),
           "OOS": ((ts > TRAIN_END) & (ts <= OOS_END)).to_numpy()}
    if a.holdout: win["🔒HOLDOUT"] = (ts > OOS_END).to_numpy()

    ARMS = {"기준선 HAR": ["l_rv1", "l_rv24", "l_rv168"],
            "HAR+DVOL": ["l_rv1", "l_rv24", "l_rv168", "l_dvol"],
            "HAR+DVOL+VRP": ["l_rv1", "l_rv24", "l_rv168", "l_dvol", "vrp"]}
    y = d["l_rv_fwd"].to_numpy(); yb = d["expand"].to_numpy()
    tr = win["TRAIN"]

    print("=" * 100)
    print(f"A. 회귀 -- 다음 24시간 log RV 예측. 창 안 R² (TRAIN 적합, 창마다 평가)")
    print("=" * 100)
    print(f"{'팔':<16}" + "".join(f"{w:>16}" for w in win))
    preds = {}
    for name, cols in ARMS.items():
        X = d[cols].to_numpy(float)
        m = LinearRegression().fit(X[tr], y[tr]); pr = m.predict(X); preds[name] = pr
        r2 = {}
        for w, wm in win.items():
            ss_res = float(np.sum((y[wm] - pr[wm]) ** 2))
            ss_tot = float(np.sum((y[wm] - y[tr].mean()) ** 2))
            r2[w] = 1 - ss_res / ss_tot
        print(f"{name:<16}" + "".join(f"{r2[w]:>16.4f}" for w in win))
    print("\n증분 (기준선 대비 잔차제곱합 감소율)")
    for name in ("HAR+DVOL", "HAR+DVOL+VRP"):
        line = f"  {name:<14}"
        for w, wm in win.items():
            b = np.sum((y[wm] - preds["기준선 HAR"][wm]) ** 2)
            t = np.sum((y[wm] - preds[name][wm]) ** 2)
            line += f"{(1 - t / b) * 100:>15.2f}%"
        print(line)

    print("\n" + "=" * 100)
    print(f"B. 실무형 분류 -- '다음 24h 변동성 확장({EXPAND_K}x)' · AUC 와 상위 20% 정밀도")
    print("=" * 100)
    print(f"{'팔':<16}" + "".join(f"{w+' AUC':>13}{'정밀도':>9}" for w in win))
    rows = []
    for name, cols in ARMS.items():
        X = d[cols].to_numpy(float)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        m = LogisticRegression(max_iter=2000).fit((X[tr] - mu) / sd, yb[tr])
        pr = m.predict_proba((X - mu) / sd)[:, 1]
        line = f"{name:<16}"; rec = {"arm": name}
        for w, wm in win.items():
            auc = roc_auc_score(yb[wm], pr[wm]) if len(np.unique(yb[wm])) > 1 else np.nan
            # ⭐건수 매칭: 창마다 **그 창 안** 상위 20% -- TRAIN 고정 임계는 창마다 커버리지가
            #   달라져서 정밀도 차이가 순위 개선이 아니라 임계 이동으로 나온다.
            thr = np.quantile(pr[wm], 0.80)
            sel = wm & (pr >= thr)
            prec = yb[sel].mean() if sel.sum() >= 20 else np.nan
            line += f"{auc:>13.4f}{prec:>9.3f}"
            rec[f"{w}_auc"] = auc; rec[f"{w}_prec"] = prec; rec[f"{w}_base"] = yb[wm].mean()
            rec[f"{w}_cov"] = sel.sum() / wm.sum()
        rows.append(rec); print(line)
        preds[f"clf_{name}"] = pr
    print(f"{'(기저율)':<16}" + "".join(f"{'':>13}{yb[wm].mean():>9.3f}" for w, wm in win.items()))
    print("\n" + "=" * 100)
    print("C. 증분의 유의성 -- 일군집 Diebold-Mariano (손실차 d_t = e²(기준선) - e²(처치))")
    print("   d 의 평균이 0 보다 유의하게 크면 DVOL 이 진짜 정보를 더한다. 일 단위로 뭉쳐 겹침 제거.")
    print("=" * 100)
    rng = np.random.default_rng(20260910)
    day = ts.dt.floor("D")
    for name in ("HAR+DVOL", "HAR+DVOL+VRP"):
        for w, wm in win.items():
            eb = (y[wm] - preds["기준선 HAR"][wm]) ** 2
            et = (y[wm] - preds[name][wm]) ** 2
            dd_ = pd.Series(eb - et).groupby(day[wm].values).mean()
            if len(dd_) < 10: continue
            t = dd_.mean() / (dd_.std(ddof=1) / np.sqrt(len(dd_)))
            boot = [dd_.sample(len(dd_), replace=True, random_state=int(x)).mean()
                    for x in rng.integers(0, 1e6, 2000)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            print(f"  {name:<14}{w:<12} 독립일 {len(dd_):>4}  평균 손실감소 {dd_.mean():+.5f}  "
                  f"t={t:>6.2f}  95%CI [{lo:+.5f}, {hi:+.5f}]  {'0 배제 ✅' if lo > 0 else '0 포함'}")
    pd.DataFrame(rows).to_csv(OUT / "volexpansion.csv", index=False)
    print(f"\n저장 {OUT/'volexpansion.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
